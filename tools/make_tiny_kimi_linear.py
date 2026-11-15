#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""make_tiny_kimi_linear.py - a tiny Kimi Linear checkpoint in the released layout, plus a
torch oracle (all-position logits and greedy continuations for two prompts).

The checkpoint mirrors moonshotai/Kimi-Linear-48B-A3B-Instruct: a top-level config with
model_type kimi_linear and a nested linear_attn_config, `model.`-prefixed names, BF16
matrices (experts included), BF16 norms, F32 A_log ([1,1,H,1]) and dt_bias. The oracle is
computed from exactly those (rounded) values.

Usage: make_tiny_kimi_linear.py <out_dir> [--seed N]
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
from kimi_linear_ref import KimiLinear, hf_name, tiny_config  # noqa: E402

PROMPTS = [[3, 7, 11, 5, 9, 200, 17, 4, 99, 150, 23, 61], [42, 8, 250, 1, 77, 13, 101]]
STEPS = 16
F32_KEEP = ("A_log", "dt_bias")  # fp32 parameters in the released model


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


def bf16_bits(t: torch.Tensor) -> np.ndarray:
    return (t.detach().float().numpy().view(np.uint32) >> 16).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    c = tiny_config()
    m = KimiLinear(c).float().eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() >= 2 and "A_log" not in name and "conv1d" not in name:
                torch.nn.init.normal_(p, std=0.05)
            elif "weight" in name and "conv1d" not in name:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))      # norms near 1, not exactly 1
            else:
                p.zero_()
        for i, layer in enumerate(m.layers):
            at = layer.self_attn
            if hasattr(at, "A_log"):
                at.A_log.copy_(torch.log(torch.linspace(1.0, 16.0, c.kda_num_heads)).view(1, 1, -1, 1))
                torch.nn.init.normal_(at.dt_bias, std=0.5)
                for conv in (at.q_conv1d, at.k_conv1d, at.v_conv1d):
                    torch.nn.init.normal_(conv.weight, std=0.4)
            mlp = layer.mlp
            if hasattr(mlp, "e_score_correction_bias"):
                mlp.e_score_correction_bias.copy_(torch.linspace(-0.1, 0.1, c.num_experts))
                torch.nn.init.normal_(mlp.gate.weight, std=0.2)
                for e in list(mlp.experts) + [mlp.shared]:
                    for lin in (e.w1, e.w3, e.w2):
                        torch.nn.init.normal_(lin.weight, std=0.3)
            else:
                for lin in (mlp.w1, mlp.w3, mlp.w2):
                    torch.nn.init.normal_(lin.weight, std=0.3)
        # round to what the checkpoint will hold
        for name, p in m.named_parameters():
            if not any(k in name for k in F32_KEEP):
                p.copy_(torch.from_numpy((bf16_bits(p).astype(np.uint32) << 16).view(np.float32)).view_as(p))

    tensors = {}
    for name, p in m.state_dict().items():
        hn = hf_name(name)
        if any(k in name for k in F32_KEEP):
            tensors[hn] = ("F32", p.float().numpy())
        else:
            tensors[hn] = ("BF16", bf16_bits(p))
    write_safetensors(os.path.join(a.out_dir, "model.safetensors"), dict(sorted(tensors.items())))

    kda_layers = [i + 1 for i in range(c.num_hidden_layers) if not c.is_mla(i)]
    config = {
        "architectures": ["KimiLinearForCausalLM"], "model_type": "kimi_linear",
        "hidden_size": c.hidden_size, "num_hidden_layers": c.num_hidden_layers, "vocab_size": c.vocab_size,
        "rms_norm_eps": c.rms_norm_eps, "hidden_act": "silu", "tie_word_embeddings": False,
        "num_attention_heads": c.num_attention_heads, "num_key_value_heads": c.num_attention_heads,
        "q_lora_rank": None, "kv_lora_rank": c.kv_lora_rank, "qk_nope_head_dim": c.qk_nope_head_dim,
        "qk_rope_head_dim": c.qk_rope_head_dim, "v_head_dim": c.v_head_dim, "mla_use_nope": True,
        "head_dim": c.qk_rope_head_dim + 8,
        "num_experts": c.num_experts, "num_experts_per_token": c.num_experts_per_token,
        "num_shared_experts": c.num_shared_experts, "moe_intermediate_size": c.moe_intermediate_size,
        "routed_scaling_factor": c.routed_scaling_factor, "moe_renormalize": c.moe_renormalize,
        "moe_router_activation_func": "sigmoid", "num_expert_group": 1, "topk_group": 1,
        "use_grouped_topk": True, "moe_layer_freq": 1,
        "first_k_dense_replace": c.first_k_dense_replace, "intermediate_size": c.intermediate_size,
        "linear_attn_config": {"full_attn_layers": c.full_attn_layers, "kda_layers": kda_layers,
                               "head_dim": c.kda_head_dim, "num_heads": c.kda_num_heads,
                               "short_conv_kernel_size": c.short_conv_kernel_size},
        "bos_token_id": 1, "eos_token_id": 2, "dtype": "bfloat16",
    }
    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    cases = []
    with torch.no_grad():
        for ids in PROMPTS:
            logits = m.forward(torch.tensor([ids]))[0].float().numpy()
            greedy = m.greedy(torch.tensor([ids]), STEPS)[0, len(ids):].tolist()
            cases.append({"prompt": ids, "tf_logits": logits.tolist(), "greedy": greedy})
            print("prompt len %2d -> greedy %s" % (len(ids), greedy))
    with open(os.path.join(a.out_dir, "oracle.json"), "w") as f:
        json.dump({"vocab": c.vocab_size, "greedy_steps": STEPS, "cases": cases}, f)
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
