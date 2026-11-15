# SPDX-License-Identifier: Apache-2.0
"""eagle3_ref.py - EAGLE-3 drafts in torch, as vLLM computes them (Eagle3LlamaForCausalLM), for any head checkpoint
and transformers target: the reference fullm's draft head is compared with.

Ref(target_dir, head_dir).drafts(ctx, next, k) -> (ids, margins): the head reads positions 0..len(ctx)-1, position j
with token t(j+1) (ctx[j+1], then `next` at the last) and the target's taps at j, then drafts k tokens greedily, each
at the next position with the previous step's pre-norm output as its hidden state. Margins are the top-1 minus top-2
head logits of each draft: where they are small, rounding may pick the other token. Pure in its arguments: what a
head that has read the context step by step drafts (its cache is causal).

Rules (vLLM's): taps = eagle_config.eagle_aux_hidden_state_layer_ids (or top level) sorted and deduplicated, else
(2, L // 2, L - 3); transformers' hidden_states[l] is the residual stream entering layer l. A head over the whole
vocabulary uses the target's lm_head (and ids as they are); else its own lm_head and target id = id + d2t[id]. The
target's embedding. Head weights rounded to bf16 (vLLM casts to the model dtype), computed in fp32.

kv_bf16: every layer's keys and values (the target's, after RoPE, and the head's) rounded to bf16 before attention
reads them, as a bf16 KV cache stores them (fullm's default cache): the reference for that configuration.
"""
import json
import math

import torch
from safetensors.torch import load_file
from transformers import AttentionInterface, AutoModelForCausalLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward


def _bf16(x):
    return x.to(torch.bfloat16).to(x.dtype)


def _bf16_kv_attention(module, query, key, value, attention_mask, **kw):
    return sdpa_attention_forward(module, query, _bf16(key), _bf16(value), attention_mask, **kw)


AttentionInterface.register("bf16_kv", _bf16_kv_attention)


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, pos, inv):  # x [T][H][D], rotate_half (Llama / Qwen)
    ang = pos[:, None].float() * inv[None, :]  # f32, as transformers, vLLM and fullm compute it
    cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None, :]
    h = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., h:], x[..., :h]], -1) * sin


class Ref:
    def __init__(self, target_dir, head_dir, kv_bf16=False):
        self.kv_bf16 = kv_bf16
        self.target = AutoModelForCausalLM.from_pretrained(
            target_dir, dtype=torch.float32, attn_implementation="bf16_kv" if kv_bf16 else "sdpa").eval()
        tc = self.target.config
        c = json.load(open(f"{head_dir}/config.json"))
        raw = load_file(f"{head_dir}/model.safetensors")
        L = tc.num_hidden_layers
        ec = c.get("eagle_config") or {}
        ids = ec.get("eagle_aux_hidden_state_layer_ids") or c.get("eagle_aux_hidden_state_layer_ids")
        self.taps = sorted(set(ids)) if ids else sorted({2, L // 2, L - 3})
        assert len(self.taps) == 3
        self.E = c["hidden_size"]
        self.H, self.Hk = c["num_attention_heads"], c.get("num_key_value_heads", c["num_attention_heads"])
        self.D = c.get("head_dim", self.E // self.H)
        self.eps = c.get("rms_norm_eps", 1e-6)
        rp = c.get("rope_parameters") or {}
        theta = rp.get("rope_theta", c.get("rope_theta", 10000.0))
        assert (c.get("rope_scaling") or {}).get("rope_type", "default") == "default"
        self.inv = 1.0 / (torch.tensor(float(theta)) ** (torch.arange(0, self.D, 2, dtype=torch.int64).float() / self.D))
        vocab = c["vocab_size"]
        self.dv = c.get("draft_vocab_size") or vocab

        def find(name, layer):
            pre = ["midlayer.", "layers.0.", "model.layers.0.", "model.midlayer."] if layer else ["", "model."]
            for p in pre:
                if p + name in raw:
                    return raw[p + name].to(torch.bfloat16).float()
            raise KeyError(name)

        self.w = {n: find(n, True) for n in [
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight",
            "hidden_norm.weight", "post_attention_layernorm.weight"]}
        self.w["fc"] = find("fc.weight", False)
        self.w["norm"] = find("norm.weight", False)
        if self.dv == vocab:
            self.w["lm_head"] = self.target.lm_head.weight.detach().float()
            self.d2t = torch.arange(vocab)
        else:
            self.w["lm_head"] = raw["lm_head.weight"].to(torch.bfloat16).float()
            self.d2t = torch.arange(self.dv) + raw["d2t"].to(torch.int64)
        self.embed = self.target.get_input_embeddings().weight.detach().float()

    def target_pass(self, ctx):
        """The target over ctx: the taps [len][3E] and the logits [len][vocab]."""
        with torch.no_grad():
            out = self.target(torch.tensor([ctx]), output_hidden_states=True)
        return torch.cat([out.hidden_states[l][0] for l in self.taps], -1), out.logits[0]

    def _layer(self, g, tokens, pos, cache):
        w, T = self.w, g.shape[0]
        x = torch.cat([rms(self.embed[tokens], w["input_layernorm.weight"], self.eps),
                       rms(g, w["hidden_norm.weight"], self.eps)], -1)
        q = (x @ w["self_attn.q_proj.weight"].T).view(T, self.H, self.D)
        k = (x @ w["self_attn.k_proj.weight"].T).view(T, self.Hk, self.D)
        v = (x @ w["self_attn.v_proj.weight"].T).view(T, self.Hk, self.D)
        q, k = rope(q, pos, self.inv), rope(k, pos, self.inv)
        if self.kv_bf16:
            k, v = _bf16(k), _bf16(v)
        assert len(cache[0]) == int(pos[0])
        cache[0].extend(k)
        cache[1].extend(v)
        rep = self.H // self.Hk
        K = torch.stack(cache[0]).repeat_interleave(rep, 1)
        V = torch.stack(cache[1]).repeat_interleave(rep, 1)
        s = torch.einsum("thd,phd->thp", q, K) / math.sqrt(self.D)
        s = s.masked_fill((torch.arange(K.shape[0])[None, :] > pos[:, None])[:, None, :], float("-inf"))
        a = torch.einsum("thp,phd->thd", s.softmax(-1), V).reshape(T, self.H * self.D)
        r = g + a @ w["self_attn.o_proj.weight"].T
        hn = rms(r, w["post_attention_layernorm.weight"], self.eps)
        m = (torch.nn.functional.silu(hn @ w["mlp.gate_proj.weight"].T) * (hn @ w["mlp.up_proj.weight"].T)) \
            @ w["mlp.down_proj.weight"].T
        r = r + m
        return r, rms(r, w["norm"], self.eps) @ w["lm_head"].T

    def _pick(self, logits):
        top = logits.topk(2)
        return int(self.d2t[int(top.indices[0])]), float(top.values[0] - top.values[1])

    def drafts(self, ctx, nxt, k, taps=None):
        """(k draft target ids, their margins) after ctx and nxt; taps: target_pass(ctx)[0] if already computed."""
        if k == 0:
            return [], []
        m = len(ctx)
        if taps is None:
            taps = self.target_pass(ctx)[0]
        with torch.no_grad():
            cache = ([], [])
            r, logits = self._layer(taps @ self.w["fc"].T, torch.tensor(ctx[1:] + [nxt]), torch.arange(m), cache)
            d, mg = self._pick(logits[-1])
            ids, margins, h = [d], [mg], r[-1:]
            for i in range(1, k):
                h, logits = self._layer(h, torch.tensor([ids[-1]]), torch.tensor([m - 1 + i]), cache)
                d, mg = self._pick(logits[-1])
                ids.append(d)
                margins.append(mg)
        return ids, margins
