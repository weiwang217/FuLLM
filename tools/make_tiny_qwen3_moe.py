#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""make_tiny_qwen3_moe.py - a tiny random Qwen3-MoE checkpoint and its transformers oracle.

Writes OUT/config.json, OUT/model.safetensors (bf16) and OUT/oracle.json: for a few prompts
over the 256-token byte vocabulary, transformers' fp32 logits at every position (computed
from the bf16-rounded weights) and 16 greedy tokens.

Usage: make_tiny_qwen3_moe.py OUT   (needs torch, transformers with Qwen3-MoE)
"""
import json
import sys

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM


def main():
    out = sys.argv[1]
    torch.manual_seed(1234)
    cfg = Qwen3MoeConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, num_experts=8, num_experts_per_tok=2,
        norm_topk_prob=True, rope_theta=1e6, max_position_embeddings=512, tie_word_embeddings=False,
        rms_norm_eps=1e-6, decoder_sparse_step=1, mlp_only_layers=[], initializer_range=0.08)
    model = Qwen3MoeForCausalLM(cfg).eval()
    with torch.no_grad():  # norms away from 1 so a wrong norm shows
        for n, p in model.named_parameters():
            if n.endswith("norm.weight"):
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
    model = model.to(torch.bfloat16)
    model.save_pretrained(out, safe_serialization=True)
    # reload in fp32 rather than model.float(): .to(bfloat16) also rounded the RoPE frequency
    # buffer, which .float() would keep
    model = Qwen3MoeForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    cases = []
    for prompt in ([3, 7, 11, 5, 9, 200, 31], [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33] * 3, [1]):
        ids = torch.tensor([prompt])
        with torch.no_grad():
            logits = model(ids).logits[0].tolist()
            greedy = model.generate(ids, max_new_tokens=16, do_sample=False, pad_token_id=0)[0][len(prompt):].tolist()
        cases.append({"prompt": prompt, "logits": logits, "greedy": greedy})
    json.dump({"cases": cases}, open(f"{out}/oracle.json", "w"))
    print("wrote", out, [len(c["prompt"]) for c in cases])


if __name__ == "__main__":
    main()
