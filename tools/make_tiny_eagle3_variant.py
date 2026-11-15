#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""make_tiny_eagle3_variant.py - a second tiny EAGLE-3 head for tests/fixtures/tiny_qwen3_moe, in the other layouts
heads come in: a draft vocabulary that is the whole vocabulary (no d2t; the head's own lm_head and embed_tokens are
ignored, as vLLM shares the target's when the shapes match), fp16 tensors, "layers.0." names (vLLM's own), a
transformers-5 config (rope_parameters, no draft_vocab_size, no eagle_config: the default taps 2, L/2, L-3).

Usage: make_tiny_eagle3_variant.py TARGET OUT   (then make_eagle3_scenarios.py TARGET OUT OUT/scenarios.json)
"""
import json
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoConfig


def main():
    tdir, out = sys.argv[1], sys.argv[2]
    tc = AutoConfig.from_pretrained(tdir)
    E, V = tc.hidden_size, tc.vocab_size
    Q, KV, I = tc.num_attention_heads * tc.head_dim, tc.num_key_value_heads * tc.head_dim, 80
    g = torch.Generator().manual_seed(77)
    r = lambda *s, sd=0.08: torch.randn(*s, generator=g) * sd  # noqa: E731
    w = {
        "fc.weight": r(E, 3 * E),
        "layers.0.self_attn.q_proj.weight": r(Q, 2 * E),
        "layers.0.self_attn.k_proj.weight": r(KV, 2 * E),
        "layers.0.self_attn.v_proj.weight": r(KV, 2 * E),
        "layers.0.self_attn.o_proj.weight": r(E, Q, sd=0.3),
        "layers.0.mlp.gate_proj.weight": r(I, E),
        "layers.0.mlp.up_proj.weight": r(I, E),
        "layers.0.mlp.down_proj.weight": r(E, I, sd=0.3),
        # both ignored: the target's are used
        "lm_head.weight": r(V, E, sd=1.0),
        "embed_tokens.weight": r(V, E, sd=1.0),
    }
    for n in ["layers.0.input_layernorm", "layers.0.hidden_norm", "layers.0.post_attention_layernorm", "norm"]:
        w[n + ".weight"] = 1.0 + 0.3 * torch.randn(E, generator=g)
    save_file({n: t.to(torch.float16).contiguous() for n, t in w.items()}, f"{out}/model.safetensors")
    cfg = {
        "architectures": ["LlamaForCausalLMEagle3"], "model_type": "llama", "hidden_size": E, "intermediate_size": I,
        "num_attention_heads": tc.num_attention_heads, "num_key_value_heads": tc.num_key_value_heads,
        "head_dim": tc.head_dim, "num_hidden_layers": 1, "vocab_size": V, "rms_norm_eps": tc.rms_norm_eps,
        "rope_parameters": {"rope_theta": tc.rope_parameters["rope_theta"], "rope_type": "default"},
        "hidden_act": "silu", "tie_word_embeddings": False, "dtype": "float16",
    }
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
