#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""make_tiny_inkling2.py - a tiny Inkling 2 (looped) checkpoint in the released layout, plus a torch
oracle (all-position logits and greedy continuations).

The checkpoint mirrors thinkingmachines/Inkling-2-Mini: a top-level config with model_type
inkling2_loop and a nested text_config, `model.llm.`-prefixed names, BF16 matrices and norms, F32
attention-residual and cross-pass queries and router biases, and the experts quantised as the release
has them: NVFP4 in the first layers (two E2M1 codes a byte, an e4m3 scale per 16 values, amax /
outer_scale_denominator per expert), MXFP8 in the last two (e4m3 values, a power-of-two exponent byte
per 32). The oracle (inkling2_ref.py) is computed from exactly the values the file holds.

Usage: make_tiny_inkling2.py <out_dir> [--seed N]
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inkling2_ref import (  # noqa: E402
    E2M1, Cfg, Inkling2, P, dequant_mxfp8, dequant_nvfp4, random_weights, tiny_config,
)

PROMPTS = [[3, 7, 11, 5, 9, 200, 17, 4, 99, 150, 23, 61, 42, 8, 250, 1, 77, 13], [42, 8, 250, 1, 77, 13, 101]]
STEPS = 16
MXFP8_LAYERS = 2  # the last layers' experts (the release: layers 20-23 of 24)
F32 = ("attn_res.", "middle_cycle_cross_pass", "mlp.gate.bias")


def e4m3_table() -> np.ndarray:
    """the value of every e4m3 byte (NaN for the two NaN codes)"""
    t = np.zeros(256, dtype=np.float64)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 0xF, b & 7
        if e == 0xF and m == 7:
            v = np.nan
        elif e == 0:
            v = m / 8.0 * 2.0 ** -6
        else:
            v = (1 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = -v if s else v
    return t


E4M3 = e4m3_table()
_E4M3_POS = np.array([E4M3[b] for b in range(127)])  # 0 .. 448 ascending


def to_e4m3(x: np.ndarray) -> np.ndarray:
    """nearest e4m3 byte (saturating)"""
    a = np.abs(x).astype(np.float64)
    i = np.clip(np.searchsorted(_E4M3_POS, a), 1, 126)
    lo, hi = _E4M3_POS[i - 1], _E4M3_POS[i]
    b = np.where(a - lo <= hi - a, i - 1, i).astype(np.uint8)
    return np.where(x < 0, b | 0x80, b).astype(np.uint8)


def quant_nvfp4(w: torch.Tensor, denom: float = 448.0 * 6.0):
    """w [E, R, C] -> codes [E, R, C/2] u8, scale [E, R, C/16] e4m3 bytes, amax [E] f32, denom [1] f32"""
    E, R, C = w.shape
    x = w.double().numpy()
    amax = np.abs(x).reshape(E, -1).max(1).astype(np.float32)
    factor = (amax.astype(np.float64) / denom)[:, None, None, None]
    blk = x.reshape(E, R, C // 16, 16)
    sb = to_e4m3(np.abs(blk).max(-1) / 6.0 / factor[..., 0])
    s = E4M3[sb][..., None] * factor
    lut = np.array(E2M1, dtype=np.float64)
    q = blk / np.where(s == 0, 1.0, s)
    codes = np.abs(q[..., None] - lut[None, None, None, None, :]).argmin(-1).astype(np.uint8)  # 0.0 before -0.0
    codes = codes.reshape(E, R, C // 2, 2)
    return (codes[..., 0] | (codes[..., 1] << 4)).astype(np.uint8), sb, amax, np.array([denom], dtype=np.float32)


def quant_mxfp8(w: torch.Tensor):
    """w [E, R, C] -> e4m3 bytes [E, R, C], exponent bytes (+127) [E, R, C/32]"""
    E, R, C = w.shape
    blk = w.double().numpy().reshape(E, R, C // 32, 32)
    amax = np.maximum(np.abs(blk).max(-1), 1e-30)
    e = np.ceil(np.log2(amax / 448.0)).astype(np.int64)
    vals = to_e4m3(blk / 2.0 ** e[..., None])
    return vals.reshape(E, R, C), (e + 127).astype(np.uint8)


def bf16_bits(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float().numpy().view(np.uint32)
    return ((x + 0x7FFF + ((x >> 16) & 1)) >> 16).astype(np.uint16)  # round to nearest even


def from_bf16(b: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((b.astype(np.uint32) << 16).view(np.float32).copy())


def write_safetensors(path, tensors):
    header, blobs, off = {}, [], 0
    for name, (dt, arr) in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


def build(seed: int):
    """the file's tensors (released names and dtypes) and the float values they hold"""
    tc = tiny_config()
    c = Cfg(tc)
    w = random_weights(c, seed)
    tensors, held = {}, {}
    for name, t in w.items():
        full = P + name
        if ".mlp.experts." in name:
            layer = int(name.split(".")[1])
            if layer >= c.L - MXFP8_LAYERS:
                vals, ex = quant_mxfp8(t)
                tensors[full] = ("F8_E4M3", vals)
                tensors[full + "_scale"] = ("U8", ex)
                held[name] = dequant_mxfp8(torch.from_numpy(E4M3[vals]), torch.from_numpy(ex.astype(np.float32)))
            else:
                codes, sb, amax, den = quant_nvfp4(t)
                tensors[full] = ("U8", codes)
                tensors[full + "_scale"] = ("F8_E4M3", sb)
                tensors[full + "_amax"] = ("F32", amax)
                tensors[full + "_outer_scale_denominator"] = ("F32", den)
                held[name] = dequant_nvfp4(torch.from_numpy(codes), torch.from_numpy(E4M3[sb]),
                                           torch.from_numpy(amax), torch.from_numpy(den))
            held[name] = held[name].float()
        elif any(k in name for k in F32):
            tensors[full] = ("F32", t.float().numpy())
            held[name] = t.float()
        else:
            bits = bf16_bits(t)
            tensors[full] = ("BF16", bits)
            held[name] = from_bf16(bits).view(t.shape)
    config = {
        "architectures": ["Inkling2LoopForConditionalGeneration"], "model_type": "inkling2_loop", "dtype": "bfloat16",
        "eos_token_id": 2,
        "text_config": {"model_type": "inkling2_loop_text", "model_max_length": 4096, "torch_dtype": "bfloat16",
                        "q_bias": False, "o_bias": False, "use_value_norm_no_gamma": True, "n_shared_experts": 0,
                        "use_gate_bias": True, **tc, "loop_config": {**tc["loop_config"], "is_varloop": False}},
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    }
    return c, tensors, held, config


MTP = {  # a draft head (the release's mtp.safetensors: a DSpark draft), random: it only has to run
    "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
    "intermediate_size": 64, "rope_theta": 10000.0, "local_layer_ids": [0], "sliding_window_size": 5, "block_size": 4,
    "markov_rank": 8, "markov_head_type": "rnn_hidden", "rms_norm_eps": 1e-6,
    "target_layer_ids": [1, 7, 12, 16],  # visits in running order (8 layers, 4 of them three times), then the final norm
}


def mtp_tensors(vocab: int, seed: int):
    """the draft head's tensors under the release's names, bf16"""
    g = np.random.RandomState(seed + 1000)
    m, H = MTP, MTP["hidden_size"]
    Cq, Ck, I, R = m["num_attention_heads"] * m["head_dim"], m["num_key_value_heads"] * m["head_dim"], m["intermediate_size"], m["markov_rank"]
    shapes = {"fuse_weight": (m["num_hidden_layers"] + 1, 4), "mask_embed": (H,), "norm.weight": (H,),
              "markov_head.dflare_norm.weight": (H,), "markov_head.hidden_out.weight": (H, R), "markov_head.hidden_out.bias": (H,),
              "markov_head.markov_w1.weight": (vocab, R), "markov_head.rnn_proj.weight": (3 * R, 2 * R + 2 * H),
              "markov_head.rnn_proj.bias": (3 * R,)}
    for i in range(m["num_hidden_layers"]):
        p = f"layers.{i}."
        shapes.update({p + "dflare_norm.weight": (H,), p + "input_layernorm.weight": (H,), p + "post_attention_layernorm.weight": (H,),
                       p + "self_attn.q_proj.weight": (Cq, H), p + "self_attn.k_proj.weight": (Ck, H), p + "self_attn.v_proj.weight": (Ck, H),
                       p + "self_attn.k_proj_ctx.weight": (Ck, H), p + "self_attn.v_proj_ctx.weight": (Ck, H),
                       p + "self_attn.o_proj.weight": (H, Cq), p + "self_attn.q_norm.weight": (m["head_dim"],),
                       p + "self_attn.k_norm.weight": (m["head_dim"],), p + "self_attn.k_norm_ctx.weight": (m["head_dim"],),
                       p + "mlp.gate_proj.weight": (I, H), p + "mlp.up_proj.weight": (I, H), p + "mlp.down_proj.weight": (H, I)})
    out = {}
    for name, shape in sorted(shapes.items()):
        x = g.randn(*shape).astype(np.float32) * (0.3 if len(shape) == 2 else 0.2)
        if "norm" in name:
            x = 1.0 + x
        out[name] = ("BF16", bf16_bits(torch.from_numpy(x)).reshape(shape))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    c, tensors, held, config = build(a.seed)
    write_safetensors(os.path.join(a.out_dir, "model.safetensors"), dict(sorted(tensors.items())))
    config["mtp_config"] = MTP
    write_safetensors(os.path.join(a.out_dir, "mtp.safetensors"), mtp_tensors(config["text_config"]["vocab_size"], a.seed))
    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    # in float64: the tiny model's logits move ~4e-4 between float32 and float64 arithmetic
    m = Inkling2(c, {k: v.double() for k, v in held.items()})
    cases = []
    with torch.no_grad():
        for ids in PROMPTS:
            logits = m.forward(ids)
            st, seq, greedy = m.new_state(), list(ids), []
            nxt = int(m.forward(ids, st)[-1].argmax())
            for _ in range(STEPS):
                greedy.append(nxt)
                seq.append(nxt)
                nxt = int(m.forward([nxt], st)[-1].argmax())
            # the continuation decoded token by token equals teacher forcing over the whole text
            again = m.forward(seq)[len(ids) - 1: -1].argmax(-1).tolist()
            assert again == greedy, (again, greedy)
            cases.append({"prompt": ids, "tf_logits": logits.float().numpy().tolist(), "greedy": greedy})
            print("prompt len %2d -> greedy %s" % (len(ids), greedy))
    with open(os.path.join(a.out_dir, "oracle.json"), "w") as f:
        json.dump({"vocab": c.out_vocab, "greedy_steps": STEPS, "cases": cases}, f)
    print("wrote", a.out_dir, "(%d tensors)" % len(tensors))


if __name__ == "__main__":
    main()
