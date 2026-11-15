#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""kimi_linear_ref.py - a pure-PyTorch, CPU-runnable reference for Kimi Linear.

The released modeling_kimi.py (moonshotai/Kimi-Linear-48B-A3B-Instruct) imports its KDA
recurrence, ShortConvolution, gated RMSNorm and decay gate from `fla`, which is Triton and
needs a GPU. This file restates the same mathematics in plain torch so the engine can be
checked on any machine; tools/check_kimi_linear_ref.py checks THIS file against the
official code on a GPU box.

Differences from Kimi K3, each verified in modeling_kimi.py:
  - KDA decay has no lower bound: g = -exp(A_log[h]) * softplus(f_b(f_a(x)) + dt_bias)
    (fla fused_kda_gate without lower_bound). A_log is per head.
  - KDA output gate is LOW RANK: g_b(g_a(x)); applied as rmsnorm(o) * w * sigmoid(gate)
    (fla FusedRMSNormGated, activation='sigmoid').
  - MLA has no q LoRA (one q_proj) and no output gate. kv_a_layernorm is KimiRMSNorm with
    its DEFAULT eps 1e-6, not config.rms_norm_eps. NoPE, scale (qk_nope+qk_rope)^-0.5.
  - MoE is DeepSeek-V3 style at full width: sigmoid scores, bias for selection only,
    weights from unbiased scores, renormalised, times routed_scaling_factor; SiLU-GLU
    experts; shared experts added unweighted. No latent projection.
  - Plain pre-norm residuals (no Attention Residuals). SiLU everywhere (no SiTU).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KLConfig:
    hidden_size: int = 2304
    num_hidden_layers: int = 27
    vocab_size: int = 163840
    rms_norm_eps: float = 1e-5
    kda_num_heads: int = 32
    kda_head_dim: int = 128
    short_conv_kernel_size: int = 4
    num_attention_heads: int = 32
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    num_experts: int = 256
    num_experts_per_token: int = 8
    num_shared_experts: int = 1
    moe_intermediate_size: int = 1024
    routed_scaling_factor: float = 2.446
    moe_renormalize: bool = True
    first_k_dense_replace: int = 1
    intermediate_size: int = 9216
    full_attn_layers: list[int] = field(default_factory=lambda: [4, 8, 12, 16, 20, 24, 27])  # ONE-based
    # The released modeling_kimi.py adds the correction bias IN PLACE to a view of the scores
    # (`scores_for_choice = scores.view(...); scores_for_choice += bias`), so its combining
    # weights come from the BIASED scores. DeepSeek-V3, Kimi K3 and vLLM take them from the
    # unbiased scores. False reproduces the latter; True reproduces the HF code exactly.
    router_weights_biased: bool = False

    def is_mla(self, i: int) -> bool:
        return (i + 1) in self.full_attn_layers

    def is_dense(self, i: int) -> bool:
        return i < self.first_k_dense_replace


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        d = x.float()
        d = d * torch.rsqrt(d.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * d.to(x.dtype)


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt(x.float().pow(2).sum(-1, keepdim=True) + eps).to(x.dtype)


class ShortConv(nn.Module):
    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.weight = nn.Parameter(torch.zeros(channels, 1, kernel))

    def forward(self, x, state=None):
        B, _T, C = x.shape
        xt = x.transpose(1, 2)
        pad = state if state is not None else xt.new_zeros(B, C, self.kernel - 1)
        xp = torch.cat([pad, xt], dim=-1)
        y = F.conv1d(xp, self.weight, groups=C)
        return F.silu(y.transpose(1, 2)), xp[..., -(self.kernel - 1):]


class KDA(nn.Module):
    def __init__(self, c: KLConfig):
        super().__init__()
        H, D, E = c.kda_num_heads, c.kda_head_dim, c.hidden_size
        P = H * D
        self.c, self.H, self.D, self.P = c, H, D, P
        self.q_proj = nn.Linear(E, P, bias=False)
        self.k_proj = nn.Linear(E, P, bias=False)
        self.v_proj = nn.Linear(E, P, bias=False)
        self.q_conv1d = ShortConv(P, c.short_conv_kernel_size)
        self.k_conv1d = ShortConv(P, c.short_conv_kernel_size)
        self.v_conv1d = ShortConv(P, c.short_conv_kernel_size)
        self.A_log = nn.Parameter(torch.zeros(1, 1, H, 1))
        self.f_a_proj = nn.Linear(E, D, bias=False)
        self.f_b_proj = nn.Linear(D, P, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(P))
        self.b_proj = nn.Linear(E, H, bias=False)
        self.g_a_proj = nn.Linear(E, D, bias=False)
        self.g_b_proj = nn.Linear(D, P, bias=False)
        self.o_norm = RMSNorm(D, c.rms_norm_eps)  # FusedRMSNormGated: norm * w, then * sigmoid(g)
        self.o_proj = nn.Linear(P, E, bias=False)

    def forward(self, x, state=None):
        B, T, _ = x.shape
        H, D = self.H, self.D
        S = state[0] if state else x.new_zeros(B, H, D, D, dtype=torch.float32)
        cq, ck, cv = (state[1], state[2], state[3]) if state else (None, None, None)
        q, cq = self.q_conv1d(self.q_proj(x), cq)
        k, ck = self.k_conv1d(self.k_proj(x), ck)
        v, cv = self.v_conv1d(self.v_proj(x), cv)
        z = self.f_b_proj(self.f_a_proj(x)).float() + self.dt_bias.float()
        z = z.view(B, T, H, D)
        a = self.A_log.float().exp().view(1, 1, H, 1)
        g = -a * F.softplus(z)                                   # no lower bound
        alpha = g.exp()
        beta = torch.sigmoid(self.b_proj(x).float())
        q = l2norm(q.view(B, T, H, D)).float() * (D ** -0.5)
        k = l2norm(k.view(B, T, H, D)).float()
        v = v.view(B, T, H, D).float()
        o = x.new_zeros(B, T, H, D, dtype=torch.float32)
        for t in range(T):
            S = S * alpha[:, t].unsqueeze(-1)
            u = torch.einsum('bhk,bhkv->bhv', k[:, t], S)
            w = beta[:, t].unsqueeze(-1) * (v[:, t] - u)
            S = S + torch.einsum('bhk,bhv->bhkv', k[:, t], w)
            o[:, t] = torch.einsum('bhk,bhkv->bhv', q[:, t], S)
        gate = self.g_b_proj(self.g_a_proj(x)).view(B, T, H, D)
        y = self.o_norm(o) * torch.sigmoid(gate.float())
        return self.o_proj(y.view(B, T, self.P).to(x.dtype)), (S, cq, ck, cv)


class MLA(nn.Module):
    def __init__(self, c: KLConfig):
        super().__init__()
        E, H = c.hidden_size, c.num_attention_heads
        self.c, self.H = c, H
        self.qn, self.qr, self.vh = c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
        self.qh = self.qn + self.qr
        self.q_proj = nn.Linear(E, H * self.qh, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(E, c.kv_lora_rank + self.qr, bias=False)
        self.kv_a_layernorm = RMSNorm(c.kv_lora_rank, 1e-6)        # KimiRMSNorm default eps
        self.kv_b_proj = nn.Linear(c.kv_lora_rank, H * (self.qn + self.vh), bias=False)
        self.o_proj = nn.Linear(H * self.vh, E, bias=False)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        H = self.H
        q = self.q_proj(x).view(B, T, H, self.qh)
        q_nope, q_rope = q.split([self.qn, self.qr], dim=-1)
        latent, k_rope = self.kv_a_proj_with_mqa(x).split([self.c.kv_lora_rank, self.qr], dim=-1)
        latent = self.kv_a_layernorm(latent)
        if cache is not None and cache[0] is not None:
            latent = torch.cat([cache[0], latent], dim=1)
            k_rope = torch.cat([cache[1], k_rope], dim=1)
        new_cache = (latent, k_rope)
        kv = self.kv_b_proj(latent).view(B, -1, H, self.qn + self.vh)
        k_nope, v = kv.split([self.qn, self.vh], dim=-1)
        k_rope_b = k_rope.unsqueeze(2).expand(-1, -1, H, -1)
        qs = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2).float()
        ks = torch.cat([k_nope, k_rope_b], dim=-1).transpose(1, 2).float()
        vs = v.transpose(1, 2).float()
        att = torch.matmul(qs, ks.transpose(-1, -2)) * (self.qh ** -0.5)
        Tq, Tk = att.shape[-2], att.shape[-1]
        causal = torch.ones(Tq, Tk, dtype=torch.bool).tril(Tk - Tq)
        att = att.masked_fill(~causal, float('-inf')).softmax(-1)
        y = torch.matmul(att, vs).transpose(1, 2).reshape(B, T, H * self.vh)
        return self.o_proj(y.to(x.dtype)), new_cache


class SwiGLU(nn.Module):
    """Both KimiMLP (gate/up/down_proj) and KimiBlockSparseMLP (w1/w3/w2)."""

    def __init__(self, din: int, dinter: int):
        super().__init__()
        self.w1 = nn.Linear(din, dinter, bias=False)
        self.w3 = nn.Linear(din, dinter, bias=False)
        self.w2 = nn.Linear(dinter, din, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, c: KLConfig):
        super().__init__()
        self.c = c
        E = c.hidden_size
        self.gate = nn.Linear(E, c.num_experts, bias=False)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(c.num_experts))
        self.experts = nn.ModuleList([SwiGLU(E, c.moe_intermediate_size) for _ in range(c.num_experts)])
        self.shared = SwiGLU(E, c.moe_intermediate_size * c.num_shared_experts)

    def forward(self, x):
        B, T, E = x.shape
        flat = x.view(-1, E)
        scores = F.linear(flat.float(), self.gate.weight.float()).sigmoid()
        choice = scores + self.e_score_correction_bias.float()
        idx = torch.topk(choice, self.c.num_experts_per_token, dim=-1, sorted=False).indices
        w = (choice if self.c.router_weights_biased else scores).gather(1, idx)
        if self.c.num_experts_per_token > 1 and self.c.moe_renormalize:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        w = w * self.c.routed_scaling_factor
        out = torch.zeros_like(flat, dtype=torch.float32)
        for n in range(flat.shape[0]):
            for j in range(idx.shape[1]):
                out[n] += w[n, j] * self.experts[idx[n, j]](flat[n:n + 1])[0].float()
        return out.view(B, T, E).to(x.dtype) + self.shared(x)


class Layer(nn.Module):
    def __init__(self, c: KLConfig, i: int):
        super().__init__()
        self.self_attn = MLA(c) if c.is_mla(i) else KDA(c)
        self.mlp = SwiGLU(c.hidden_size, c.intermediate_size) if c.is_dense(i) else MoE(c)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)

    def forward(self, h, state=None):
        a, state = self.self_attn(self.input_layernorm(h), state)
        h = h + a
        return h + self.mlp(self.post_attention_layernorm(h)), state


class KimiLinear(nn.Module):
    def __init__(self, c: KLConfig):
        super().__init__()
        self.c = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([Layer(c, i) for i in range(c.num_hidden_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)

    def forward(self, ids):
        h = self.embed_tokens(ids)
        for layer in self.layers:
            h, _ = layer(h)
        return self.lm_head(self.norm(h))

    @torch.no_grad()
    def greedy(self, ids, n_new: int):
        out = ids
        for _ in range(n_new):
            out = torch.cat([out, self.forward(out)[:, -1].argmax(-1, keepdim=True)], dim=1)
        return out


def tiny_config() -> KLConfig:
    """9 layers: KDA x3, MLA, KDA x3, MLA, MLA (the last layer is MLA, as in the release)."""
    return KLConfig(hidden_size=128, num_hidden_layers=9, vocab_size=256, kda_num_heads=4, kda_head_dim=16,
                    num_attention_heads=4, kv_lora_rank=32, qk_nope_head_dim=24, qk_rope_head_dim=8,
                    v_head_dim=16, num_experts=8, num_experts_per_token=2, num_shared_experts=1,
                    moe_intermediate_size=48, intermediate_size=96, full_attn_layers=[4, 8, 9])


def hf_name(name: str) -> str:
    """Reference parameter name -> the released checkpoint's name."""
    n = "model." + name if not name.startswith("lm_head") else name
    n = n.replace(".mlp.e_score_correction_bias", ".block_sparse_moe.gate.e_score_correction_bias")
    n = n.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight")
    n = n.replace(".mlp.experts.", ".block_sparse_moe.experts.")
    n = n.replace(".mlp.shared.w1.", ".block_sparse_moe.shared_experts.gate_proj.")
    n = n.replace(".mlp.shared.w3.", ".block_sparse_moe.shared_experts.up_proj.")
    n = n.replace(".mlp.shared.w2.", ".block_sparse_moe.shared_experts.down_proj.")
    # the dense layer's MLP uses gate/up/down_proj names
    for a, b in ((".mlp.w1.", ".mlp.gate_proj."), (".mlp.w3.", ".mlp.up_proj."), (".mlp.w2.", ".mlp.down_proj.")):
        n = n.replace(a, b)
    return n
