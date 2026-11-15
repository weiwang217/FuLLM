#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# k3_partial.sh DEST [N] - real Kimi K3 weights, first N layers (default 4): download the
# shards they need, then check the CUDA executor against the Rust reference at the real
# dimensions (7168 hidden, 96 heads, 896 experts) and time it.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:?usage: k3_partial.sh DEST [N]}
N=${2:-4}
B=./target/release/fullm
python3 tools/download_k3.py "$DEST" --layers "$N"
echo "== reference vs cuda, exact (bf16 trunk, f32 state and cache)"
$B compare --model "$DEST" --layers "$N" --a reference --b cuda --trunk bf16
echo "== reference vs cuda, int8 trunk on both"
$B compare --model "$DEST" --layers "$N" --a reference --b cuda --trunk int8
echo "== timing, int8 trunk, bf16 state (production defaults)"
$B bench --model "$DEST" --layers "$N" --skip-hw --prompt 256 --decode 16 --trunk int8
