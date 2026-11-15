#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""check_kimi_linear_ref.py - the torch reference against the OFFICIAL Kimi Linear code.

tools/kimi_linear_ref.py restates modeling_kimi.py in plain torch so the engine can be
checked anywhere. This script closes the loop on a GPU box: it loads the tiny checkpoint
with the released modeling_kimi.py / configuration_kimi.py (whose KDA, ShortConvolution,
gated norm and decay gate are fla Triton kernels) and compares its logits with the oracle
that kimi_linear_ref.py produced.

Usage: check_kimi_linear_ref.py <tiny_dir> <official_dir>
  official_dir holds modeling_kimi.py and configuration_kimi.py (e.g. the HF download).
Needs: torch (CUDA), transformers, flash-linear-attention, einops.
"""
import json
import os
import shutil
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM


def main() -> int:
    tiny, official = sys.argv[1], sys.argv[2]
    work = tempfile.mkdtemp(prefix="kl_check_")
    for f in ("modeling_kimi.py", "configuration_kimi.py"):
        src = open(os.path.join(official, f)).read()
        # @auto_docstring only builds documentation, and on some transformers/Python pairs it
        # chokes on `X | None` annotations; dropping it changes no computation.
        src = "\n".join(line for line in src.splitlines() if line.strip() != "@auto_docstring")
        open(os.path.join(work, f), "w").write(src + "\n")
    shutil.copy(os.path.join(tiny, "model.safetensors"), work)
    cfg = json.load(open(os.path.join(tiny, "config.json")))
    cfg["auto_map"] = {"AutoConfig": "configuration_kimi.KimiLinearConfig",
                       "AutoModel": "modeling_kimi.KimiLinearModel",
                       "AutoModelForCausalLM": "modeling_kimi.KimiLinearForCausalLM"}
    cfg.setdefault("rope_theta", 10000.0)
    cfg.setdefault("num_nextn_predict_layers", 0)
    json.dump(cfg, open(os.path.join(work, "config.json"), "w"))

    # our reference on the same weights, both router conventions
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from safetensors.torch import load_file
    from kimi_linear_ref import KimiLinear, hf_name, tiny_config
    state = load_file(os.path.join(tiny, "model.safetensors"))
    refs = {}
    for biased in (False, True):
        c = tiny_config()
        c.router_weights_biased = biased
        m = KimiLinear(c).float().eval()
        with torch.no_grad():
            for name, p in m.named_parameters():
                p.copy_(state[hf_name(name)].float().view_as(p))
        refs[biased] = m

    oracle = json.load(open(os.path.join(tiny, "oracle.json")))
    model = AutoModelForCausalLM.from_pretrained(work, trust_remote_code=True, torch_dtype=torch.float32,
                                                 attn_implementation="eager").cuda().eval()
    # the MLA layers read config._attn_implementation at call time; pin the plain softmax path
    model.config._attn_implementation = "eager"
    worst = {"oracle": 0.0, False: 0.0, True: 0.0}
    for case in oracle["cases"]:
        ids = torch.tensor([case["prompt"]])
        with torch.no_grad():
            off = model(input_ids=ids.cuda(), use_cache=False).logits[0].float().cpu()
            got = {b: refs[b](ids)[0].float() for b in (False, True)}
        rel = lambda a, b: ((a - b).norm(dim=-1) / b.norm(dim=-1)).max().item()
        worst["oracle"] = max(worst["oracle"], rel(got[False], torch.tensor(case["tf_logits"])))
        for b in (False, True):
            worst[b] = max(worst[b], rel(off, got[b]))
    print("kimi_linear_ref (unbiased) vs oracle.json:      worst rel-L2 %.3e" % worst["oracle"])
    print("official vs ref, router weights UNBIASED:       worst rel-L2 %.3e" % worst[False])
    print("official vs ref, router weights BIASED (in-place): worst rel-L2 %.3e" % worst[True])
    match = min(worst[False], worst[True])
    print("best agreement %.3e -> %s" % (match, "MATCH" if match < 1e-3 else "MISMATCH"))
    return 0 if match < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
