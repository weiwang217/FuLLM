#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""check_inkling2_ref.py - the torch reference against the transformers implementation of Inkling 2
(looped), on the tiny checkpoint.

tools/inkling2_ref.py restates the model in plain torch over the released tensor names. This script
loads the same tiny checkpoint (make_tiny_inkling2.py: released names, NVFP4 and MXFP8 experts) with
transformers' Inkling2LoopForCausalLM, whose loader renames the tensors and dequantises the experts,
and compares its logits with the oracle: whole prompts, and the same prompts fed in pieces through
its cache. A match checks the reference's arithmetic, its reading of the names and of the experts'
formats, and the tiny checkpoint's layout at once.

Usage: check_inkling2_ref.py <tiny_dir>
Needs: torch, a transformers with models/inkling2_loop (5.24 or its main branch).
"""
import json
import os
import shutil
import sys
import tempfile

import torch
from transformers import Inkling2LoopForCausalLM


def main() -> int:
    tiny = sys.argv[1]
    work = tempfile.mkdtemp(prefix="ink2_check_")
    shutil.copy(os.path.join(tiny, "model.safetensors"), work)
    cfg = json.load(open(os.path.join(tiny, "config.json")))["text_config"]  # the text decoder alone
    # the release lists the sliding layers; transformers takes each layer's kind
    local = cfg.pop("local_layer_ids")
    cfg["layer_types"] = ["hybrid_sliding" if i in local else "hybrid" for i in range(cfg["num_hidden_layers"])]
    json.dump(cfg, open(os.path.join(work, "config.json"), "w"))
    model, info = Inkling2LoopForCausalLM.from_pretrained(work, dtype=torch.float32, output_loading_info=True)
    model = model.eval()
    bad = {k: v for k, v in info.items() if v}
    if bad:
        print("loading info:", bad)
        return 1
    oracle = json.load(open(os.path.join(tiny, "oracle.json")))
    worst = 0.0
    with torch.no_grad():
        for case in oracle["cases"]:
            ids = torch.tensor([case["prompt"]])
            want = torch.tensor(case["tf_logits"])
            got = model(input_ids=ids).logits[0].float()
            d = (got - want).abs().max().item()
            # the same prompt in pieces through the cache
            cache, parts = None, []
            for a, b in ((0, 1), (1, 2), (2, 6), (6, len(case["prompt"]))):
                if a >= b:
                    continue
                out = model(input_ids=ids[:, a:b], cache=cache, use_cache=True)
                cache = out.cache
                parts.append(out.logits[0].float())
            dc = (torch.cat(parts) - want).abs().max().item()
            agree = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
            print(f"prompt of {ids.shape[1]:2d}: max |logit diff| {d:.3e} whole, {dc:.3e} in pieces; argmax agree {agree:.3f}")
            worst = max(worst, d, dc)
    shutil.rmtree(work, ignore_errors=True)
    # float32 against the oracle's float64: the reference itself moves ~4e-4 between the two
    ok = worst < 2e-3
    print("PASS" if ok else "FAIL", f"(worst {worst:.3e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
