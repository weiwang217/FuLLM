#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""check_inkling2_real.py - tools/inkling2_ref.py on the released Inkling 2 Mini checkpoint against
the transformers implementation: the same prompt through both, logits compared.

The tiny check (check_inkling2_ref.py) covers the arithmetic; this one covers the reference's
reading of the real file: every tensor name and shape, the NVFP4 and MXFP8 experts, the real
config. The reference runs in float32 on the CPU (it holds the dequantised experts: ~110 GB of
host memory), transformers in bf16 on a GPU, so the logits agree to bf16's rounding, not exactly.

Usage: check_inkling2_real.py <model_dir> [--tokens N] [--save FILE]
Needs: torch (CUDA), safetensors, a transformers with models/inkling2_loop.
"""
import argparse
import json
import os
import sys
import time

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inkling2_ref import Cfg, Inkling2, P, dequant_mxfp8, dequant_nvfp4, tensor_shapes  # noqa: E402

TEXT = ("The quick brown fox jumps over the lazy dog. In 1969, Apollo 11 landed on the Moon, and Neil "
        "Armstrong became the first person to walk on its surface. def fibonacci(n):\n    if n < 2:\n"
        "        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n")


def load_ref(model_dir: str, c: Cfg) -> dict:
    files = [f for f in sorted(os.listdir(model_dir)) if f.startswith("model") and f.endswith(".safetensors")]
    where = {}
    handles = [safe_open(os.path.join(model_dir, f), "pt") for f in files]
    for h in handles:
        for k in h.keys():
            where[k] = h
    get = lambda k: where[k].get_tensor(k)  # noqa: E731
    w = {}
    for name, shape in tensor_shapes(c).items():
        full = P + name
        if ".mlp.experts." in name:
            raw = get(full)
            if full + "_amax" in where:
                t = dequant_nvfp4(raw, get(full + "_scale").float(), get(full + "_amax"),
                                  get(full + "_outer_scale_denominator"))
            else:
                t = dequant_mxfp8(raw.float(), get(full + "_scale").float())
        else:
            t = get(full).float()
        assert tuple(t.shape) == tuple(shape), (name, tuple(t.shape), shape)
        w[name] = t
    text = {k for k in where if k.startswith(P)}
    used = {P + n for n in w} | {k for k in text if k.endswith(("_scale", "_amax", "_outer_scale_denominator"))}
    assert text == used, ("tensors the reference does not read", sorted(text - used)[:8])
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--save", help="write the prompt ids and the reference's logits here (.pt)")
    a = ap.parse_args()
    from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    ids = tok(TEXT, add_special_tokens=False)["input_ids"][: a.tokens]
    print(f"{len(ids)} tokens: {ids[:12]} ...")

    t0 = time.time()
    model = Inkling2LoopForConditionalGeneration.from_pretrained(a.model_dir, dtype=torch.bfloat16).to("cuda").eval()
    with torch.no_grad():
        hf = model(input_ids=torch.tensor([ids], device="cuda")).logits[0].float().cpu()
    print(f"transformers: loaded and ran in {time.time() - t0:.0f} s, logits {tuple(hf.shape)}")
    del model
    torch.cuda.empty_cache()

    c = Cfg(json.load(open(os.path.join(a.model_dir, "config.json"))))
    t0 = time.time()
    w = load_ref(a.model_dir, c)
    print(f"reference: {len(w)} tensors read and dequantised in {time.time() - t0:.0f} s")
    t0 = time.time()
    with torch.no_grad():
        ref = Inkling2(c, w).forward(ids)
    print(f"reference: forward in {time.time() - t0:.0f} s")
    if a.save:
        torch.save({"ids": ids, "ref_logits": ref, "hf_logits": hf}, a.save)
    d = (ref - hf).abs()
    agree = (ref.argmax(-1) == hf.argmax(-1)).float().mean().item()
    top = [tok.decode([int(t)]) for t in ref.argmax(-1)[-8:]]
    print(f"max |logit diff| {d.max().item():.3f}, mean {d.mean().item():.4f}; logits span {hf.min().item():.1f}..{hf.max().item():.1f}")
    print(f"argmax agreement {agree:.3f}; the reference's last picks: {top}")
    ok = agree >= 0.95 and d.mean().item() < 0.1
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
