#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""make_tiny_eagle3.py - a tiny random EAGLE-3 draft head for tests/fixtures/tiny_qwen3_moe, and
its oracle.

Writes OUT/config.json and OUT/model.safetensors (bf16) in the layout of the published EAGLE-3
heads (LlamaForCausalLMEagle3: fc, midlayer.*, norm, lm_head, d2t, t2d), with a draft vocabulary
of half the target's, and OUT/oracle.json: the head's greedy drafts, computed here in fp32 from
the bf16-rounded weights following vLLM's Eagle3LlamaForCausalLM (llama_eagle3.py):

  taps      the residual stream entering the target layers eagle_aux_hidden_state_layer_ids
            (sorted), concatenated: transformers' hidden_states[l]
  position  j consumes token t(j+1) with the target's taps at j:
              g = fc(taps_j); x = [input_layernorm(embed(t(j+1))) | hidden_norm(g)]
              a = o(attention(x)) over the head's own cache (RoPE at j, causal, GQA)
              r = g + a; r = r + mlp(post_attention_layernorm(r))
              logits = lm_head(norm(r)); draft = argmax + d2t[argmax]
  drafting  each draft goes in at the next position with the last r as its hidden state.

Cases: each prompt consumed whole with the target's greedy token after it, then 5 drafts; and a
verify step after prompt 0 (its next token and 4 drafts, 2 accepted (or as many as leave a clear token), the target's token after
them) then 5 drafts. The head's seed is the first whose drafts vary (3+ distinct
of 5 per case) and whose every argmax wins by a margin, so fp32/bf16 rounding cannot flip it (a prompt
whose next token is unclear gets a token more).

Usage: make_tiny_eagle3.py TARGET OUT   (needs torch, transformers with Qwen3-MoE, safetensors)
then:  make_eagle3_scenarios.py TARGET OUT OUT/scenarios.json   (the random scenarios tests/eagle3.rs replays)
"""
import json
import math
import sys

import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeForCausalLM

TAPS = [2, 0, 1]  # written unsorted: the loader sorts them as vLLM does
DRAFT_VOCAB = 128
INTER = 96
MARGIN = 0.05


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, pos, inv):  # x [T][H][D], rotate_half
    ang = pos[:, None].float() * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None, :]
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], -1)
    return x * cos + rot * sin


class Head:
    def __init__(self, w, tcfg, embed):
        self.w, self.embed = w, embed
        self.E = tcfg.hidden_size
        self.H, self.Hk, self.D = tcfg.num_attention_heads, tcfg.num_key_value_heads, tcfg.head_dim
        self.eps = tcfg.rms_norm_eps
        theta = tcfg.rope_parameters["rope_theta"] if getattr(tcfg, "rope_parameters", None) else tcfg.rope_theta
        self.inv = 1.0 / (theta ** (torch.arange(0, self.D, 2, dtype=torch.int64).float() / self.D))
        self.k, self.v = [], []  # the head's cache, by position

    def step(self, g, tokens, pos):
        """Rows at positions pos (consecutive, appended to the cache): fused hidden g [T][E]."""
        w, T = self.w, g.shape[0]
        x = torch.cat([rms(self.embed[tokens], w["midlayer.input_layernorm.weight"], self.eps),
                       rms(g, w["midlayer.hidden_norm.weight"], self.eps)], -1)
        q = (x @ w["midlayer.self_attn.q_proj.weight"].T).view(T, self.H, self.D)
        k = (x @ w["midlayer.self_attn.k_proj.weight"].T).view(T, self.Hk, self.D)
        v = (x @ w["midlayer.self_attn.v_proj.weight"].T).view(T, self.Hk, self.D)
        q, k = rope(q, pos, self.inv), rope(k, pos, self.inv)
        assert len(self.k) == int(pos[0])
        self.k += list(k)
        self.v += list(v)
        K, V = torch.stack(self.k), torch.stack(self.v)  # [P][Hk][D]
        rep = self.H // self.Hk
        K, V = K.repeat_interleave(rep, 1), V.repeat_interleave(rep, 1)
        s = torch.einsum("thd,phd->thp", q, K) / math.sqrt(self.D)
        P = K.shape[0]
        mask = torch.arange(P)[None, :] > pos[:, None]
        s = s.masked_fill(mask[:, None, :], float("-inf"))
        a = torch.einsum("thp,phd->thd", s.softmax(-1), V).reshape(T, self.H * self.D)
        r = g + a @ w["midlayer.self_attn.o_proj.weight"].T
        hn = rms(r, w["midlayer.post_attention_layernorm.weight"], self.eps)
        m = (torch.nn.functional.silu(hn @ w["midlayer.mlp.gate_proj.weight"].T) * (hn @ w["midlayer.mlp.up_proj.weight"].T)) \
            @ w["midlayer.mlp.down_proj.weight"].T
        r = r + m
        logits = rms(r, w["norm.weight"], self.eps) @ w["lm_head.weight"].T
        return r, logits

    def pick(self, logits):
        top = logits.topk(2)
        assert float(top.values[0] - top.values[1]) > MARGIN, f"draft argmax margin {float(top.values[0] - top.values[1])}"
        d = int(top.indices[0])
        return d + int(self.w["d2t"][d])


def taps_of(target, ids):
    with torch.no_grad():
        out = target(torch.tensor([ids]), output_hidden_states=True)
    hs = out.hidden_states  # [l]: the residual stream entering layer l
    return torch.cat([hs[l][0] for l in sorted(TAPS)], -1), out.logits[0]


def greedy(logits):
    """The target's argmax, or None when it wins by too little to compare."""
    top = logits.topk(2)
    return int(top.indices[0]) if float(top.values[0] - top.values[1]) > MARGIN else None


def settled(target, p):
    """p, extended by a token at a time until the target's next token is clear; and that token."""
    for extra in range(256):
        q = p + ([extra] if extra else [])
        nxt = greedy(taps_of(target, q)[1][-1])
        if nxt is not None:
            return q, nxt
    raise SystemExit("no prompt with a clear next token")


def drafts(head, target, seq, k, consumed=0):
    """seq = tokens 0..m (m positions consumed, t(j+1) = seq[j+1]); positions before `consumed`
    are already in the head's cache. Returns k drafts."""
    m = len(seq) - 1
    taps, _ = taps_of(target, seq[:m])
    g = taps[consumed:] @ head.w["fc.weight"].T
    r, logits = head.step(g, torch.tensor(seq[consumed + 1:]), torch.arange(consumed, m))
    out, h = [head.pick(logits[-1])], r[-1:]
    for i in range(1, k):
        h, logits = head.step(h, torch.tensor([out[-1]]), torch.tensor([m - 1 + i]))
        out.append(head.pick(logits[-1]))
    # the drafted positions' cache is written over by the next consumption
    del head.k[m:], head.v[m:]
    return out


def weights(tc, seed):
    E, Q, KV = tc.hidden_size, tc.num_attention_heads * tc.head_dim, tc.num_key_value_heads * tc.head_dim
    torch.manual_seed(seed)
    sd = 0.08  # o, down and lm_head larger: each draft's token then moves the next (random heads repeat)
    w = {
        "fc.weight": torch.randn(E, 3 * E) * sd,
        "midlayer.self_attn.q_proj.weight": torch.randn(Q, 2 * E) * sd,
        "midlayer.self_attn.k_proj.weight": torch.randn(KV, 2 * E) * sd,
        "midlayer.self_attn.v_proj.weight": torch.randn(KV, 2 * E) * sd,
        "midlayer.self_attn.o_proj.weight": torch.randn(E, Q) * 0.3,
        "midlayer.mlp.gate_proj.weight": torch.randn(INTER, E) * sd,
        "midlayer.mlp.up_proj.weight": torch.randn(INTER, E) * sd,
        "midlayer.mlp.down_proj.weight": torch.randn(E, INTER) * 0.3,
        "lm_head.weight": torch.randn(DRAFT_VOCAB, E) * 1.0,
    }
    for n in ["midlayer.input_layernorm", "midlayer.hidden_norm", "midlayer.post_attention_layernorm", "norm"]:
        w[n + ".weight"] = 1.0 + 0.3 * torch.randn(E)  # away from 1 so a wrong norm shows
    w = {n: t.to(torch.bfloat16) for n, t in w.items()}
    # the draft vocabulary: a random half of the target's, ascending, as offsets (vLLM's d2t)
    keep = torch.randperm(tc.vocab_size)[:DRAFT_VOCAB].sort().values
    w["d2t"] = (keep - torch.arange(DRAFT_VOCAB)).to(torch.int64)
    t2d = torch.zeros(tc.vocab_size, dtype=torch.bool)
    t2d[keep] = True
    w["t2d"] = t2d
    return w


def oracle(target, w):
    """The cases (module docstring); AssertionError when an argmax is too close to call."""
    tc = target.config
    wf = {n: (t.float() if t.is_floating_point() else t) for n, t in w.items()}
    embed = target.model.embed_tokens.weight.detach()
    cases = []
    prompts = ([3, 7, 11, 5, 9, 200, 31, 64, 18, 90, 2], [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33] * 2,
               [1, 2])
    for p in prompts:
        p, nxt = settled(target, list(p))
        head = Head(wf, tc, embed)
        cases.append({"prompt": p, "next": nxt, "drafts": drafts(head, target, p + [nxt], 5)})
    # verify after prompt 0: [next, d1..d4] in one step, 2 drafts accepted, the target's token after them
    c = cases[0]
    p = c["prompt"]
    step = [c["next"]] + c["drafts"][:4]
    _, logits = taps_of(target, p + step)
    accepted = next(a for a in (2, 1, 3, 0) if greedy(logits[len(p) + a]) is not None)
    bonus = greedy(logits[len(p) + accepted])
    head = Head(wf, tc, embed)
    drafts(head, target, p + [c["next"]], 5)
    seq = p + step[:accepted + 1] + [bonus]
    verify = {"prompt": p, "step": step, "accepted": accepted, "bonus": bonus,
              "drafts": drafts(head, target, seq, 5, consumed=len(p))}
    # a head that repeats one token would hide a drafting position gone wrong
    for d in [c["drafts"] for c in cases] + [verify["drafts"]]:
        assert len(set(d)) >= 3, f"drafts {d} repeat"
    return {"cases": cases, "verify": verify}


def main():
    tdir, out = sys.argv[1], sys.argv[2]
    target = Qwen3MoeForCausalLM.from_pretrained(tdir, dtype=torch.float32).eval()
    tc = target.config
    for seed in range(4321, 4821):  # the first head whose every draft is clear and varied
        w = weights(tc, seed)
        try:
            orc = oracle(target, w)
            break
        except AssertionError as e:
            print(f"seed {seed}: {e}")
    else:
        raise SystemExit("no seed gave clear, varied drafts")
    save_file({n: t.contiguous() for n, t in w.items()}, f"{out}/model.safetensors")
    cfg = {
        "architectures": ["LlamaForCausalLMEagle3"], "model_type": "llama", "hidden_size": tc.hidden_size,
        "intermediate_size": INTER, "num_attention_heads": tc.num_attention_heads,
        "num_key_value_heads": tc.num_key_value_heads, "head_dim": tc.head_dim, "num_hidden_layers": 1,
        "vocab_size": tc.vocab_size, "draft_vocab_size": DRAFT_VOCAB, "rms_norm_eps": tc.rms_norm_eps,
        "rope_theta": tc.rope_parameters["rope_theta"], "hidden_act": "silu", "tie_word_embeddings": False,
        "max_position_embeddings": 512, "eagle_config": {"eagle_aux_hidden_state_layer_ids": TAPS},
        "torch_dtype": "bfloat16",
    }
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=2)
    json.dump(orc, open(f"{out}/oracle.json", "w"))
    print("wrote", out, "seed", seed, [c["drafts"] for c in orc["cases"]], orc["verify"])


if __name__ == "__main__":
    main()
