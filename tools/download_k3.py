#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""download_k3.py - fetch Kimi K3, or just the shards the first N layers need.

The full checkpoint is ~1.56 TB. `--layers N` downloads only the shards holding layers
0..N-1 plus the model-level tensors (embedding, final norm, AttnRes output, lm_head), which
is enough for `fullm compare --layers N` to check every kernel at the real dimensions.
`--tokenizer-only` fetches the four small files the tokenizer and chat template need.

Usage:
  python3 tools/download_k3.py DEST [--layers N | --tokenizer-only] [--repo moonshotai/Kimi-K3]
Needs: pip install -U huggingface_hub
"""
import argparse
import json
import os
import re
import sys

SMALL = ["config.json", "tokenizer_config.json", "tiktoken.model", "tokenization_kimi.py",
         "model.safetensors.index.json", "generation_config.json"]
LAYER = re.compile(r"^language_model\.model\.layers\.(\d+)\.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--repo", default="moonshotai/Kimi-K3")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--tokenizer-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list the shards and their size only")
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    os.makedirs(a.dest, exist_ok=True)
    for f in SMALL:
        try:
            hf_hub_download(a.repo, f, local_dir=a.dest)
            print("fetched", f)
        except Exception as e:  # generation_config.json etc. may not exist
            print("skipped %s (%s)" % (f, type(e).__name__))
    if a.tokenizer_only:
        return 0

    idx = json.load(open(os.path.join(a.dest, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    want = set()
    for name, shard in wm.items():
        if not name.startswith("language_model."):
            continue  # vision tower: not served yet
        m = LAYER.match(name)
        if m is None or a.layers is None or int(m.group(1)) < a.layers:
            want.add(shard)
    sizes = {}
    try:
        info = HfApi().model_info(a.repo, files_metadata=True)
        sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
    except Exception as e:
        print("could not read file sizes:", e)
    total = sum(sizes.get(s, 0) for s in want)
    print("%d shard(s), %.1f GB%s" % (len(want), total / 1e9, "" if a.layers is None else " for layers 0..%d" % (a.layers - 1)))
    if a.dry_run:
        for s in sorted(want):
            print("  %s %.2f GB" % (s, sizes.get(s, 0) / 1e9))
        return 0
    st = os.statvfs(a.dest)
    if total and st.f_bavail * st.f_frsize < total * 1.02:
        print("not enough free space in %s: need %.1f GB" % (a.dest, total / 1e9))
        return 1
    for i, s in enumerate(sorted(want)):
        print("[%d/%d] %s" % (i + 1, len(want), s), flush=True)
        hf_hub_download(a.repo, s, local_dir=a.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
