#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""inkling2_ref.py - a plain torch reference of Inkling 2 Mini's text decoder (model_type inkling2_loop:
thinkingmachines/Inkling-2-Mini), over the released checkpoint's own tensor names.

One sequence at a time, no batch dimension, every step written out: the reference fullm's forward is
checked against (make_tiny_inkling2.py), itself checked against the transformers implementation
(check_inkling2_ref.py). The vision tower and the drafter (mtp.safetensors) are not here.

The model: a 24-layer decoder with sparse MoE MLPs (128 experts, 8 per token) whose layers 6-17 run
three times with shared weights (48 layer visits a token; every visit has its own attention cache and
convolution states). Residuals are Block Attention Residuals: the stream is cut into blocks of 6
layers, and every sub-layer input is an attention, by a learned query, over the finished blocks and
the running sum of the current one. Attention has no RoPE: a bias per (query, head, backward
distance < 1024) mixed from 16 trained profiles by the query token's hidden state, a per-query sink
logit, keys clipped to a maximum L2 norm, values RMS-normalised, and a short causal convolution
(4 taps) over keys, values, and each sub-layer's output. Layers 5, 11, 17, 23 attend to the whole
context ("full"), the others to the last 1024 positions ("sliding").

Passes 2 and 3 over layers 6-17 restart from the stream as it entered layer 6 and receive
"messages" from the pass before: attention, by four learned queries per layer (for the query, key,
value and MLP inputs), over the previous pass's block states, mixed in by low-rank gates.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
P = "model.llm."


def tiny_config() -> dict:
    """A small text_config with every structural case: sliding and full layers, a loop of two blocks
    run three times, block boundaries inside and outside the loop, a window and a bias extent shorter
    than the test prompts, log scaling that starts inside them."""
    return {
        "hidden_size": 32,
        "num_hidden_layers": 8,
        "vocab_size": 300,
        "unpadded_vocab_size": 290,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 8,
        "d_rel": 4,
        "rel_extent": 6,
        "sliding_window_size": 5,
        "log_scaling_n_floor": 4,
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-6,
        "qk_l2_clip_threshold": 2.0,
        "rel_l2_clip_threshold": 1.5,
        "local_layer_ids": [0, 2, 3, 4, 6, 7],
        "sconv_kernel_size": 4,
        "final_logit_softcapping": 30.0,
        "n_layers_per_attn_res_block": 2,
        "n_routed_experts": 8,
        "num_experts_per_tok": 3,
        "intermediate_size": 32,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "route_scale": 8.0,
        "loop_config": {"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 8},
    }


class Cfg:
    def __init__(self, c: dict):
        c = c.get("text_config", c)
        self.H = c["hidden_size"]
        self.L = c["num_hidden_layers"]
        self.vocab = c["vocab_size"]
        self.out_vocab = c.get("unpadded_vocab_size", self.vocab)
        self.eps = c.get("rms_norm_eps", 1e-6)
        self.d_rel = c["d_rel"]
        self.rel_extent = c["rel_extent"]
        self.window = c["sliding_window_size"]
        self.n_floor = c["log_scaling_n_floor"]
        self.alpha = c["log_scaling_alpha"]
        self.qk_clip = c["qk_l2_clip_threshold"]
        self.rel_clip = c["rel_l2_clip_threshold"]
        self.K = c.get("sconv_kernel_size", c.get("conv_kernel_size", 4))
        self.block = c["n_layers_per_attn_res_block"]
        self.E = c["n_routed_experts"]
        self.topk = c["num_experts_per_tok"]
        self.I = c.get("intermediate_size", c.get("moe_intermediate_size"))
        self.gate_cap = c["mlp_gate_softcap"]
        self.up_cap = c["mlp_up_softcap"]
        self.route_scale = c["route_scale"]
        self.logit_cap = c["final_logit_softcapping"]
        lp = c.get("loop_config") or {}
        self.loop_start = lp.get("core_start", c.get("loop_layer_index_start", 6))
        self.loop_end = lp.get("core_end", c.get("loop_layer_index_end", 18))
        self.passes = lp.get("num_passes", c.get("loop_num_passes", 3))
        self.gate_rank = lp.get("gate_rank", c.get("loop_gate_rank", 64))
        if "local_layer_ids" in c:
            local = set(c["local_layer_ids"])
        else:  # the transformers default
            local = {i for i in range(self.L) if (i + 1) % 6 != 0}
        self.sliding = [i in local for i in range(self.L)]
        # heads per layer kind: (query heads, kv heads, head dim)
        self.full_heads = (c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"])
        self.swa_heads = (
            c.get("swa_num_attention_heads", self.full_heads[0]),
            c.get("swa_num_key_value_heads", self.full_heads[1]),
            c.get("swa_head_dim", self.full_heads[2]),
        )
        assert self.loop_start > 0 and self.loop_start % self.block == 0 and self.loop_end % self.block == 0
        self.n_loop = self.loop_end - self.loop_start
        self.visits = self.L + (self.passes - 1) * self.n_loop

    def heads(self, l: int):
        return self.swa_heads if self.sliding[l] else self.full_heads

    def visit(self, l: int, p: int) -> int:
        """The cache a layer's pass uses: its own index in the first pass (and outside the loop), then
        the loop's layers again after the model's, pass by pass."""
        return l if p == 0 else self.L + (p - 1) * self.n_loop + (l - self.loop_start)


def tensor_shapes(c: Cfg) -> dict[str, tuple]:
    """Every tensor of the text decoder by its checkpoint name, with its shape (experts: dequantised)."""
    H, s = c.H, {}
    s["embed.weight"] = (c.vocab, H)
    s["embed_norm.weight"] = (H,)
    s["unembed.weight"] = (c.vocab, H)
    s["norm.weight"] = (H,)
    s["attn_res.output_query"] = (H,)
    s["middle_cycle_cross_pass_queries"] = (2 * c.n_loop, H)
    s["middle_cycle_cross_pass_kv_queries"] = (2 * c.n_loop, H)
    for l in range(c.L):
        Hq, Hk, D = c.heads(l)
        p = f"layers.{l}."
        if l > 0:  # layer l's attention-input queries are stored one index down
            s[f"attn_res.attn_queries.{l - 1}"] = (H,) if c.sliding[l] else (3, H)
        s[f"attn_res.mlp_queries.{l}"] = (H,)
        s[p + "attn.wq_du.weight"] = (Hq * D, H)
        s[p + "attn.wk_dv.weight"] = (Hk * D, H)
        s[p + "attn.wv_dv.weight"] = (Hk * D, H)
        s[p + "attn.wo_ud.weight"] = (H, Hq * D)
        s[p + "attn.gate_attn.weight"] = (Hq * D, H)
        s[p + "attn.attn_sink_proj.weight"] = (Hq, H)
        s[p + "attn.wr_du.weight"] = (Hq * c.d_rel, H)
        s[p + "attn.q_l2_clip.scale"] = (Hq, D)
        s[p + "attn.rel_l2_clip.scale"] = (Hq, c.d_rel)
        s[p + "attn.rel_logits_proj.proj"] = (c.d_rel, c.window if c.sliding[l] else c.rel_extent)
        if c.sliding[l]:
            s[p + "attn.k_sconv.weight"] = (Hk * D, 1, c.K)
            s[p + "attn.v_sconv.weight"] = (Hk * D, 1, c.K)
            s[p + "attn_norm.weight"] = (H,)
        else:
            for kv in "kv":
                s[p + f"attn.{kv}_sconv.static_w"] = (c.K, Hk * D)
                s[p + f"attn.{kv}_sconv.weight_proj"] = (Hk * c.K, H)
            for n in "qkv":
                s[p + f"{n}_input_norm.weight"] = (H,)
        s[p + "attn_sconv.weight"] = (H, 1, c.K)
        s[p + "mlp_sconv.weight"] = (H, 1, c.K)
        s[p + "mlp_norm.weight"] = (H,)
        s[p + "mlp.gate.weight"] = (c.E, H)
        s[p + "mlp.gate.bias"] = (c.E,)
        s[p + "mlp.experts.w13_weight"] = (c.E, 2 * c.I, H)  # rows: gate 0, up 0, gate 1, up 1, ...
        s[p + "mlp.experts.w2_weight"] = (c.E, H, c.I)
        if c.loop_start <= l < c.loop_end:
            for g in ("attn", "k", "v", "mlp"):
                s[p + f"controllers.{g}.down_weight"] = (c.gate_rank, 2 * H)
                s[p + f"controllers.{g}.forget_weight"] = (H, c.gate_rank)
                s[p + f"controllers.{g}.gain_weight"] = (H, c.gate_rank)
    return s


# ------------------------------------------------------------------ the experts' formats
def dequant_nvfp4(codes: torch.Tensor, scale: torch.Tensor, amax: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """NVFP4: two E2M1 codes a byte (the even element in the low nibble), an e4m3 scale per 16 values,
    and per expert amax / denominator. codes [E, R, C/2] uint8, scale [E, R, C/16] -> [E, R, C]."""
    lut = torch.tensor(E2M1)
    b = codes.long()
    v = torch.stack((lut[b & 0xF], lut[b >> 4]), dim=-1)  # [E, R, C/2, 2]
    v = v.view(*scale.shape, 16) * scale.float()[..., None]
    v = v * (amax.float() / denom.float())[:, None, None, None]
    return v.reshape(codes.shape[0], codes.shape[1], -1)


def dequant_mxfp8(vals: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """MXFP8: e4m3 values (given as floats), a power of two 2^(scale - 127) per 32 values."""
    v = vals.float().view(*scale.shape, 32) * torch.exp2(scale.float() - 127)[..., None]
    return v.reshape(vals.shape)


# ------------------------------------------------------------------ pieces
def unorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return w * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps))


def l2_clip(x: torch.Tensor, thr: float) -> torch.Tensor:
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (thr / (n + 1e-6)).clamp(max=1.0)


def attn_res(query: torch.Tensor, blocks: list[torch.Tensor], partial: torch.Tensor) -> list[torch.Tensor]:
    """Block Attention Residuals: each query ([H] or [n, H]) attends over the finished blocks and the
    running sum of the current one (all RMS-normalised, no gain), at every position alike."""
    st = unorm(torch.stack(blocks + [partial], dim=1))  # [S, n, H]
    q = query.view(-1, st.shape[-1])  # [nq, H]
    p = torch.softmax(torch.einsum("snh,qh->sqn", st, q) * st.shape[-1] ** -0.5, dim=-1)
    out = torch.einsum("sqn,snh->sqh", p, st)
    return list(out.unbind(1))


def sconv_static(x: torch.Tensor, w: torch.Tensor, state: torch.Tensor | None):
    """Short causal convolution per channel: out[t] = sum_j w[c, j] x[t - (K-1) + j] (the last tap is
    the current token), then (out + x) / sqrt(K + 1). x [S, C], w [C, 1, K], state: the K - 1 inputs
    before x (zeros at a sequence's start). Returns the output and the next state."""
    K = w.shape[-1]
    if state is None:
        state = x.new_zeros(K - 1, x.shape[1])
    ctx = torch.cat([state, x], dim=0)  # [S + K - 1, C]
    win = ctx.unfold(0, K, 1)  # [S, C, K]: win[t, c, j] = ctx[t + j, c]
    out = (win * w[:, 0, :]).sum(-1)
    return (out + x) / math.sqrt(K + 1), ctx[-(K - 1):].clone()


def sconv_dynamic(x, dyn_in, static_w, weight_proj, n_heads, state):
    """The full layers' key / value convolution: tap `lag` (0 = the current token) weighs x[t - lag] by
    static_w[lag, c] + a per-token, per-head weight projected from the layer input; then
    (out + x) / sqrt(2K + 1). x [S, C], dyn_in [S, H], static_w [K, C], weight_proj [n_heads * K, H]."""
    K, C = static_w.shape
    S = x.shape[0]
    if state is None:
        state = x.new_zeros(K - 1, C)
    ctx = torch.cat([state, x], dim=0)
    win = ctx.unfold(0, K, 1).flip(-1)  # [S, C, K]: win[t, c, lag] = x[t - lag, c]
    dyn = (dyn_in @ weight_proj.T).view(S, n_heads, 1, K)
    wts = static_w.T.reshape(n_heads, C // n_heads, K) + dyn  # [S, heads, C / heads, K]
    out = (win.view(S, n_heads, C // n_heads, K) * wts).sum(-1).reshape(S, C)
    return (out + x) / math.sqrt(2 * K + 1), ctx[-(K - 1):].clone()


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.tanh(x / cap) * cap


class Inkling2:
    """w: the text decoder's tensors by checkpoint name without the `model.llm.` prefix (float; the
    experts dequantised: w13 [E, 2I, H] with rows gate 0, up 0, gate 1, ..., w2 [E, H, I])."""

    def __init__(self, cfg: Cfg, w: dict[str, torch.Tensor]):
        self.c, self.w = cfg, w

    # one sequence's state: per layer visit the keys and values so far and the four convolutions'
    # last K - 1 inputs; the positions seen
    def new_state(self) -> dict:
        return {"n": 0, "k": [None] * self.c.visits, "v": [None] * self.c.visits,
                "conv": [[None] * 4 for _ in range(self.c.visits)]}

    def gate(self, l: int, name: str, h: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        """(1 - forget) * h + gain * message, both gates tanh of a low-rank map of [norm(h) | norm(m)]"""
        p = f"layers.{l}.controllers.{name}."
        g = torch.cat([unorm(h, self.c.eps), unorm(msg, self.c.eps)], dim=-1) @ self.w[p + "down_weight"].T
        return (1 - torch.tanh(g @ self.w[p + "forget_weight"].T)) * h + torch.tanh(g @ self.w[p + "gain_weight"].T) * msg

    def attention(self, l, vis, xq, xk, xv, pos, st):
        c, w, p = self.c, self.w, f"layers.{l}.attn."
        Hq, Hk, D = c.heads(l)
        S, full = xq.shape[0], not c.sliding[l]
        tau = 1.0 + c.alpha * torch.log(((pos + 1).float() / c.n_floor).clamp(min=1.0))  # [S]
        # queries: scaled, clipped to the L2 threshold, and in the full layers scaled up with position
        q = l2_clip((xq @ w[p + "wq_du.weight"].T).view(S, Hq, D) * w[p + "q_l2_clip.scale"], c.qk_clip)
        # the relative states, the same way, mix the trained bias-by-distance profiles
        rel = l2_clip((xq @ w[p + "wr_du.weight"].T).view(S, Hq, c.d_rel) * w[p + "rel_l2_clip.scale"], c.rel_clip)
        if full:
            q, rel = q * tau[:, None, None], rel * tau[:, None, None]
        rel_logits = rel @ w[p + "rel_logits_proj.proj"]  # [S, Hq, extent]
        extent = rel_logits.shape[-1]
        # keys and values: projection, short convolution, then keys clipped and values normalised
        k, v = xk @ w[p + "wk_dv.weight"].T, xv @ w[p + "wv_dv.weight"].T
        cs = st["conv"][vis]
        if full:
            k, cs[0] = sconv_dynamic(k, xk, w[p + "k_sconv.static_w"], w[p + "k_sconv.weight_proj"], Hk, cs[0])
            v, cs[1] = sconv_dynamic(v, xv, w[p + "v_sconv.static_w"], w[p + "v_sconv.weight_proj"], Hk, cs[1])
        else:
            k, cs[0] = sconv_static(k, w[p + "k_sconv.weight"], cs[0])
            v, cs[1] = sconv_static(v, w[p + "v_sconv.weight"], cs[1])
        k = l2_clip(k.view(S, Hk, D), c.qk_clip)
        v = unorm(v.view(S, Hk, D), c.eps)
        st["k"][vis] = k if st["k"][vis] is None else torch.cat([st["k"][vis], k], dim=0)
        st["v"][vis] = v if st["v"][vis] is None else torch.cat([st["v"][vis], v], dim=0)
        K, V = st["k"][vis], st["v"][vis]  # [T, Hk, D], positions 0..T-1
        T, G = K.shape[0], Hq // Hk
        dist = pos[:, None] - torch.arange(T)[None, :]  # [S, T]
        scores = torch.einsum("shd,thd->hst", q, K.repeat_interleave(G, dim=1)) * D ** -0.5
        bias = rel_logits.permute(1, 0, 2).gather(-1, dist.clamp(0, extent - 1).expand(Hq, -1, -1))
        scores = scores + bias.masked_fill((dist < 0) | (dist >= extent), 0.0)
        hidden = dist < 0
        if not full:
            hidden = hidden | (dist >= c.window)
        scores = scores.masked_fill(hidden, float("-inf"))
        # a sink logit per (query, head): it takes probability mass and adds nothing
        sink = (xq @ w[p + "attn_sink_proj.weight"].T).T[:, :, None]  # [Hq, S, 1]
        prob = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
        out = torch.einsum("hst,thd->shd", prob, V.repeat_interleave(G, dim=1)).reshape(S, Hq * D)
        out = out * torch.sigmoid(xq @ w[p + "gate_attn.weight"].T)
        return out @ w[p + "wo_ud.weight"].T

    def moe(self, l, x):
        c, w, p = self.c, self.w, f"layers.{l}.mlp."
        scores = torch.sigmoid(x @ w[p + "gate.weight"].T)
        idx = torch.topk(scores + w[p + "gate.bias"], c.topk, dim=-1).indices
        wt = scores.gather(1, idx)
        wt = wt / (wt.sum(-1, keepdim=True) + 1e-20) * c.route_scale
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for j in range(c.topk):
                e = idx[t, j]
                gu = w[p + "experts.w13_weight"][e] @ x[t]
                g, u = gu[0::2], gu[1::2]
                a = softcap(g, c.gate_cap) * torch.sigmoid(g) * softcap(u, c.up_cap)
                out[t] += wt[t, j] * (w[p + "experts.w2_weight"][e] @ a)
        return out

    def layer(self, l, p, h, blocks, msgs, pos, st):
        """Layer l's pass p on rows h [S, H]; blocks: the finished blocks' states; msgs: the four
        messages [S, H] (query, key, value, mlp) in passes after the first."""
        c, w, vis = self.c, self.w, self.c.visit(l, p)
        pre = f"layers.{l}."
        block_residual = h
        ins = attn_res(w[f"attn_res.attn_queries.{l - 1}"], blocks, h) if l > 0 else [h]
        if c.sliding[l]:
            if msgs is None:
                xq = xk = xv = rms(ins[0], w[pre + "attn_norm.weight"], c.eps)
            else:
                xq, xk, xv = (rms(self.gate(l, n, ins[0], m), w[pre + "attn_norm.weight"], c.eps)
                              for n, m in zip(("attn", "k", "v"), msgs[:3]))
        else:
            if len(ins) == 1:
                ins = ins * 3
            if msgs is not None:
                ins = [self.gate(l, n, x, m) for n, x, m in zip(("attn", "k", "v"), ins, msgs[:3])]
            xq, xk, xv = (rms(x, w[pre + f"{n}_input_norm.weight"], c.eps) for n, x in zip("qkv", ins))
        a = self.attention(l, vis, xq, xk, xv, pos, st)
        a, st["conv"][vis][2] = sconv_static(a, w[pre + "attn_sconv.weight"], st["conv"][vis][2])
        if l % c.block == 0:  # a block starts: the stream so far becomes a block state
            blocks, resid = blocks + [block_residual], a
        else:
            resid = block_residual + a
        m = attn_res(w[f"attn_res.mlp_queries.{l}"], blocks, resid)[0]
        if msgs is not None:
            m = self.gate(l, "mlp", m, msgs[3])
        m = self.moe(l, rms(m, w[pre + "mlp_norm.weight"], c.eps))
        m, st["conv"][vis][3] = sconv_static(m, w[pre + "mlp_sconv.weight"], st["conv"][vis][3])
        return resid + m

    def forward(self, tokens, st: dict | None = None) -> torch.Tensor:
        """Logits [S, out_vocab] for `tokens` appended to the sequence in `st` (a new one when None)."""
        c, w = self.c, self.w
        st = self.new_state() if st is None else st
        tok = torch.as_tensor(tokens, dtype=torch.long)
        pos = st["n"] + torch.arange(tok.shape[0])
        h = rms(w["embed.weight"][tok], w["embed_norm.weight"], c.eps)
        blocks: list[torch.Tensor] = []

        def run(layers, p, h, blocks, prev=None):
            for l in layers:
                msgs = None
                if prev is not None:  # messages from the pass before, by this layer's four queries
                    i = l - c.loop_start
                    qm = w["middle_cycle_cross_pass_queries"][2 * i: 2 * i + 2]  # query, mlp
                    kv = w["middle_cycle_cross_pass_kv_queries"][2 * i: 2 * i + 2]  # key, value
                    msgs = attn_res(torch.stack([qm[0], kv[0], kv[1], qm[1]]), prev[0], prev[1])
                nh = self.layer(l, p, h, blocks, msgs, pos, st)
                if l % c.block == 0:
                    blocks = blocks + [h]
                h = nh
            return h, blocks

        h, blocks = run(range(c.loop_start), 0, h, blocks)
        pre_h, pre_blocks = h, blocks
        h, blocks = run(range(c.loop_start, c.loop_end), 0, h, blocks)
        for p in range(1, c.passes):
            # the previous pass's block states past the one it started with, and its output
            prev = (blocks[len(pre_blocks) + 1:], h)
            h, blocks = run(range(c.loop_start, c.loop_end), p, pre_h, pre_blocks, prev)
        h, blocks = run(range(c.loop_end, c.L), 0, h, blocks)
        h = rms(attn_res(w["attn_res.output_query"], blocks, h)[0], w["norm.weight"], c.eps)
        st["n"] += tok.shape[0]
        return softcap(h @ w["unembed.weight"][: c.out_vocab].T, c.logit_cap)


def random_weights(c: Cfg, seed: int = 7) -> dict[str, torch.Tensor]:
    """Random tensors exercising every path (queries and static taps away from their zero init, scales
    and norms away from one, vectors long enough for the clips to act on some)."""
    g = torch.Generator().manual_seed(seed)
    w = {}
    for name, shape in tensor_shapes(c).items():
        if name.endswith("norm.weight") or name.endswith("l2_clip.scale"):
            t = 1.0 + 0.2 * torch.randn(shape, generator=g)
        elif "queries" in name or name.endswith("output_query"):
            t = torch.randn(shape, generator=g)
        elif "sconv" in name:
            t = 0.4 * torch.randn(shape, generator=g)
        elif name.endswith("gate.bias"):
            t = 0.1 * torch.randn(shape, generator=g)
        elif name.endswith("embed.weight"):
            t = torch.randn(shape, generator=g)
        else:
            t = torch.randn(shape, generator=g) / math.sqrt(shape[-1])
        w[name] = t
    return w
