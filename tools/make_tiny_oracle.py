#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""make_tiny_oracle.py - the tiny Kimi K3 checkpoint plus a multi-position torch oracle.

The checkpoint itself comes from tools/make_tiny_checkpoint.py of
https://github.com/FareedKhan-dev/kimi-k3-in-c (Apache-2.0; clone it beside this repository or
pass --k3-tools), so its bytes are laid out exactly like the released model: HF dotted names, a
BF16 trunk, MXFP4 routed experts (low nibble = even element, E8M0 per 32), fp32 small vectors.

That script records only the final-position logits of one prompt. A serving engine
needs more than that to be checked: every position of a prefill (teacher forcing),
a greedy continuation (decode), and two prompts of different lengths so that a
batched step, a chunked prefill and a single-sequence run can all be compared
against the same answer. This file adds those, from the same weights.

Usage: make_tiny_oracle.py <out_dir> [--k3-tools PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

DEFAULT_K3_TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "kimi-k3-in-c", "tools")

PROMPTS = [
    [3, 7, 11, 5, 9, 200, 17, 4, 99, 150, 23, 61],
    [42, 8, 250, 1, 77, 13, 101],
]
GREEDY_STEPS = 16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--k3-tools", default=DEFAULT_K3_TOOLS)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.k3_tools))
    import make_tiny_checkpoint as mtc  # noqa: E402

    # 1. the checkpoint, byte-for-byte what make_tiny_checkpoint.py writes
    argv = sys.argv
    sys.argv = ["make_tiny_checkpoint.py", a.out_dir, "--seed", str(a.seed),
                "--prompt-ids", ",".join(map(str, PROMPTS[0]))]
    try:
        mtc.main()
    finally:
        sys.argv = argv

    # 2. the same model, with weights transformed to exactly what the shards hold
    cfg, model = mtc.build(a.seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            w = p.detach().float().cpu().numpy()
            eng = mtc.engine_name(name)
            if mtc.is_expert(name):
                packed, scales = mtc.mxfp4_quant(w)
                p.copy_(torch.from_numpy(mtc.mxfp4_dequant(packed, scales)))
            elif mtc.is_reqn(eng, name):
                p.copy_(torch.from_numpy(mtc.bf16_roundtrip(w)))

    cases = []
    with torch.no_grad():
        for ids in PROMPTS:
            logits = model.forward(torch.tensor([ids]))[0][0].float().cpu().numpy()
            greedy = model.greedy(torch.tensor([ids]), GREEDY_STEPS)[0, len(ids):].tolist()
            cases.append({
                "prompt": ids,
                "tf_logits": logits.astype(np.float32).tolist(),  # [T][vocab]
                "greedy": greedy,
            })
            print("prompt len %2d -> greedy %s" % (len(ids), greedy))

    with open(os.path.join(a.out_dir, "oracle.json"), "w") as f:
        json.dump({"vocab": cfg.vocab_size, "greedy_steps": GREEDY_STEPS, "cases": cases}, f)
    print("wrote %s/oracle.json" % a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
