"""Lean static-cache decode engine for Looped-CED and the dense control (one sequence, batch 1).

The reference path (`LoopedCED.extend`) is the ground truth but is not a speed proxy: it runs the
aux head at every step, grows every cache with torch.cat, and builds dense masks. This engine
reuses the model's modules and weights (nothing is copied) and does what a real decoder does:

- Preallocated buffers for max_len tokens, written in place; reads are views of the filled prefix.
    encoder global-attention K/V        per (encoder loop, attention layer)
    memory K/V (context positions)      one shared buffer, or one per memory layer
    own K/V (decoder positions)         per (decoder memory-attention layer, loop)
    GDN recurrent state S and conv      per (encoder loop, encoder GDN layer) and
                                        per (decoder GDN layer, loop)
    Canon layer history                 per encoder global layer that has one (EncoderConfig.canon_kernel = K;
                                        ced/model/canon.py): its last K - 1 inputs, zeros at a document's start
- `step(token, is_decoder_position)`: one token through the encoder (single-step GDN with the
  module's own projections, gates and conv; attention over the KV buffer). A context position
  appends its memory K/V and returns nothing (aux logits only on request). A decoder position runs
  the T decoder loops (HC read, injection, loop embedding, layers, HC write); memory-attention
  layers attend over the memory prefix and their own loop-k prefix as two partial attentions merged
  by log-sum-exp (no concatenation). The final readout goes once through the tied head.
- LoopSpec pipeline at T = 2 (arXiv 2609.17184): `spec_step` is ONE decoder pass whose matmuls are
  shared by two rows, row 0 = loop 1 at position n+1 (fresh draft token) and row 1 = loop 2 at
  position n (from its saved loop-1 state). Each row uses its own per-loop parameters, own-KV
  buffer and GDN state; both rows read the memory K/V in the same kernel call. Then the head on
  both rows (final readout for row 1, draft readout for row 0), the greedy accept check, and the
  encoder for the next draft token. A rejection restores the snapshot of the small states (GDN
  states, conv windows) and the buffer lengths, and runs a bubble: encoder + loop 1 + draft head for
  the corrected token.

Numerics: projections and the head run in the storage dtype (bf16 GEMV accumulates in fp32 on CPU
but rounds its output to bf16); norms, softmax and the attention merge run in at least fp32. In
float64 every result equals the reference to ~1e-15 (tests/test_engine.py). The reference's head
upcasts the whole embedding matrix to fp32 at every call, which a decoder would never do.

Scope: one document starting at position 0, batch 1. Dense FFN only; routed FFN and depth
attention raise NotImplementedError.

The control's per-layer residual scalars (EncoderConfig.layer_scalars, every form): `_encoder_stack` ends every layer
with the model's own apply_layer_scalars, the function EncoderLayer.forward ends with, on the layer's input and its two
residual adds, with x0 the embedding output of the token being read and a - 1 and b of every layer from the model's own
layer_scalar_terms (the one place that says what a form's stored tensors mean; a - 1 is None in "x0", b in "skip").
The sum is formed in at least float32 whatever the engine's storage dtype.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ced.model.attention import rope_cap
from ced.model.control import HybridLM
from ced.model.attn_res import refuse_attn_res
from ced.model.encoder import apply_layer_scalars, refuse_recur_loop
from ced.model.gdn import HEAD_ZERO_REFUSAL, l2_normalize
from ced.model.gdn_arms import refuse_gdn_arms
from ced.model.gdn_rules import refuse_model as refuse_gdn_rules
from ced.model.looped_ced import LoopedCED
from ced.model.mlp import ReLU2, SwiGLU
from ced.model.norms import compute_dtype
from ced.model.rope import apply_rope


def _flash_kernel():
    """The CPU flash-attention kernel (returns the output AND the log-sum-exp), if this build has it."""
    try:
        return torch.ops.aten._scaled_dot_product_flash_attention_for_cpu
    except (AttributeError, RuntimeError):
        return None


_FLASH = _flash_kernel()


def attend_partial(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax attention of q (1, H, Q, hd) over k, v (1, H, S, hd), every query seeing every key.

    Returns the normalised output (1, H, Q, hd) and the log-sum-exp of the scaled scores (1, H, Q),
    both in the compute dtype, so that partial attentions over disjoint key sets can be merged.
    """
    cd = compute_dtype(q)
    if backend == "flash_cpu":
        o, lse = _FLASH(q, k, v, 0.0, False)
        return o.to(cd), lse.to(cd)
    s = torch.matmul(q.to(cd), k.transpose(-1, -2).to(cd)) * q.shape[-1] ** -0.5
    lse = torch.logsumexp(s, dim=-1)
    return torch.matmul(torch.exp(s - lse[..., None]), v.to(cd)), lse


def merge_partials(parts: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Combine partial attentions (o_i, lse_i) over disjoint key sets into the attention over their union."""
    if len(parts) == 1:
        return parts[0][0]
    lse = torch.logsumexp(torch.stack([l for _, l in parts]), dim=0)
    return sum(torch.exp(l - lse)[..., None] * o for o, l in parts)


@dataclass
class _Pending:
    """LoopSpec pipeline state between steps: loop 1 has run at `pos`, loop 2 has not; the encoder
    has read the draft token at pos + 1."""

    Y1: torch.Tensor  # loop-boundary state after loop 1 at pos: HC streams (n, d) or (d,)
    e_cur: torch.Tensor  # (d,) injection input at pos
    pos: int
    e_next: torch.Tensor  # (d,) injection input of the draft token at pos + 1
    draft: int  # drafted token at pos + 1


@dataclass
class SpecStats:
    steps: int = 0  # pipeline steps (one verified draft each)
    accepted: int = 0
    accepted_flags: list = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(1, self.steps)


def buffer_bytes(model_cfg, max_len: int, max_context: int, max_decoder: int, dtype: torch.dtype) -> int:
    """Bytes of the engine's preallocated buffers for a config (for memory checks before allocating)."""
    cfg = model_cfg
    es = torch.empty((), dtype=dtype).element_size()
    kinds = cfg.encoder.layer_kinds()
    a, g = cfg.encoder.attention, cfg.encoder.gdn
    loops = cfg.encoder.loops
    gdn_state = lambda gc: gc.n_v_heads * gc.k_head_dim * gc.v_head_dim + (gc.conv_kernel - 1) * (  # noqa: E731
        2 * gc.n_k_heads * gc.k_head_dim + gc.n_v_heads * gc.v_head_dim)
    kv = 3 if rope_cap(a) is not None else 2  # under the rotary distance cap also the unrotated keys
    total = loops * kinds.count("attention") * kv * a.n_kv_heads * a.head_dim * max_len
    total += 2 * loops * kinds.count("gdn") * gdn_state(g)  # states + snapshot
    if cfg.encoder.canon_kernel:  # the Canon layers' histories (the control's global layers; one loop)
        total += kinds.count("attention") * (cfg.encoder.canon_kernel - 1) * cfg.d_model
    if cfg.arch == "looped_ced":
        T = cfg.loop.t_eval
        dk = cfg.decoder.layer_kinds()
        da = cfg.decoder.attention
        n_mem = dk.count("memory_attention")
        n_proj = n_mem if cfg.decoder.memory_kv_per_layer else 1
        total += n_proj * 2 * da.n_kv_heads * da.head_dim * max_context
        total += n_mem * T * 2 * da.n_kv_heads * da.head_dim * max_decoder
        total += (T + 1) * dk.count("gdn") * gdn_state(cfg.decoder.gdn)
    return total * es


class DecodeEngine:
    """Static-cache single-sequence decoder around a built LoopedCED or HybridLM (see module docstring).

    max_len: positions the encoder buffers hold; max_context / max_decoder: context / decoder
    positions the memory / own buffers hold (default max_len). attn_backend: "flash_cpu" (the CPU
    flash kernel, which returns the log-sum-exp), "matmul", or "auto". gemv_row_chunk: split
    projections over more rows than this into chunks (bf16 GEMM on some CPUs is much slower for 3+
    rows than for 1-2; see scripts/bench_decode.py).
    """

    def __init__(self, model, max_len: int, max_context: int | None = None, max_decoder: int | None = None,
                 attn_backend: str = "auto", gemv_row_chunk: int | None = None):
        self.model = model
        self.looped = isinstance(model, LoopedCED)
        assert self.looped or isinstance(model, HybridLM), type(model)
        cfg = self.cfg = model.cfg
        w = model.embed.weight
        self.dtype, self.device = w.dtype, w.device
        self.cdtype = compute_dtype(w)
        if attn_backend == "auto":
            attn_backend = "flash_cpu" if (_FLASH is not None and self.device.type == "cpu") else "matmul"
        assert attn_backend in ("flash_cpu", "matmul"), attn_backend
        assert attn_backend != "flash_cpu" or _FLASH is not None, "this torch build has no CPU flash kernel"
        self.attn_backend = attn_backend
        self.gemv_row_chunk = gemv_row_chunk
        self._check_supported()
        self.max_len = max_len
        self.max_context = max_len if max_context is None else max_context
        self.max_decoder = max_len if max_decoder is None else max_decoder
        self.T = cfg.loop.t_eval if self.looped else 0
        empty = lambda *shape: torch.empty(*shape, dtype=self.dtype, device=self.device)  # noqa: E731
        zeros = lambda *shape: torch.zeros(*shape, dtype=self.dtype, device=self.device)  # noqa: E731

        # ---- encoder
        refuse_recur_loop(cfg, "the decode engine")
        refuse_attn_res(cfg, "the decode engine")
        refuse_gdn_arms(cfg, "the decode engine")
        # _gdn_one writes one GDN step a token from gates(): negative eigenvalues read right (beta comes from gates()); KDA, the
        # erase and Gated DeltaProduct refused by name (ced/model/gdn_rules.py; DeltaProduct was computed as GDN before, silently)
        refuse_gdn_rules(cfg, "the decode engine (its GDN step, _gdn_one)", reads=("negeig",))
        enc = model.encoder
        loops = cfg.encoder.loops
        self.enc_slot = {}  # layer index -> index among layers of its kind
        n_attn = n_gdn = 0
        for i, layer in enumerate(enc.layers):
            if layer.kind == "attention":
                self.enc_slot[i], n_attn = n_attn, n_attn + 1
            else:
                self.enc_slot[i], n_gdn = n_gdn, n_gdn + 1
        ea, eg = cfg.encoder.attention, cfg.encoder.gdn
        self.enc_k = empty(loops, n_attn, ea.n_kv_heads, max_len, ea.head_dim)
        self.enc_v = empty(loops, n_attn, ea.n_kv_heads, max_len, ea.head_dim)
        # The rotary distance cap of the encoder's global layers (AttentionConfig.rope_max_distance; attention.py,
        # "Rotary distance cap"): the keys before rotation, which the keys beyond the cap are scored with.
        # The by-layer mix (EncoderConfig.nope_global_layers): every layer is read under its own config
        # (_encoder_stack: rope_cap(layer.mixer.cfg)); a NoPE layer has no cap, rotates nothing and leaves its slice of
        # enc_kf unused (the buffer is allocated for every global layer, as buffer_bytes counts it).
        self.enc_cap = rope_cap(ea)
        self.enc_kf = empty(loops, n_attn, ea.n_kv_heads, max_len, ea.head_dim) if self.enc_cap is not None else None
        conv_ch = 2 * eg.n_k_heads * eg.k_head_dim + eg.n_v_heads * eg.v_head_dim
        self.enc_S = zeros(loops * n_gdn, eg.n_v_heads, eg.k_head_dim, eg.v_head_dim)
        self.enc_conv = zeros(loops * n_gdn, eg.conv_kernel - 1, conv_ch)
        self.n_enc_gdn = n_gdn
        # Canon layers (EncoderConfig.canon_kernel = K; ced/model/canon.py): per global layer the last K - 1 inputs of
        # its Canon layer, newest last (zeros: a document's start). The control only (one encoder loop).
        self.enc_canon = zeros(n_attn, cfg.encoder.canon_kernel - 1, cfg.d_model) if cfg.encoder.canon_kernel else None

        # ---- decoder
        if self.looped:
            kinds = cfg.decoder.layer_kinds()
            da, dg = cfg.decoder.attention, cfg.decoder.gdn
            self.dec_slot = {}
            n_mem = n_dgdn = 0
            for l, kind in enumerate(kinds):
                if kind == "memory_attention":
                    self.dec_slot[l], n_mem = n_mem, n_mem + 1
                else:
                    self.dec_slot[l], n_dgdn = n_dgdn, n_dgdn + 1
            T = self.T
            n_proj = len(model.memory_kv)
            self.mem_k = empty(n_proj, da.n_kv_heads, self.max_context, da.head_dim)
            self.mem_v = empty(n_proj, da.n_kv_heads, self.max_context, da.head_dim)
            self.own_k = empty(n_mem, T, da.n_kv_heads, self.max_decoder, da.head_dim)
            self.own_v = empty(n_mem, T, da.n_kv_heads, self.max_decoder, da.head_dim)
            dconv = 2 * dg.n_k_heads * dg.k_head_dim + dg.n_v_heads * dg.v_head_dim
            self.dec_S = zeros(n_dgdn, T, dg.n_v_heads, dg.k_head_dim, dg.v_head_dim)
            self.dec_conv = zeros(n_dgdn, T, dg.conv_kernel - 1, dconv)
        # Snapshot of the small states at the accept point (LoopSpec rollback): every encoder GDN
        # state and the loop-1 decoder GDN states, plus the buffer lengths.
        self.spec_ok = self.looped and self.T == 2
        if self.spec_ok:
            self.snap_enc_S, self.snap_enc_conv = torch.empty_like(self.enc_S), torch.empty_like(self.enc_conv)
            self.snap_dec_S = torch.empty_like(self.dec_S[:, 0])
            self.snap_dec_conv = torch.empty_like(self.dec_conv[:, 0])
        self.reset()

    # ------------------------------------------------------------------------------------------
    # bookkeeping

    def _check_supported(self) -> None:
        cfg = self.cfg
        if not self.looped and cfg.loop.pass2_gate != "none":
            # the control with a gated second pass (ced/model/control.py, "Two passes"): the engine's step is the
            # single pass, so it would decode with pass 1 alone under a two-pass label
            raise NotImplementedError("the control with a second pass (loop.pass2_gate) is not supported by the engine: "
                                      "its step runs one pass; use the model's prefill / extend")
        attn_cfgs = [cfg.encoder.attention]
        if cfg.encoder.loops > 1:
            assert cfg.arch == "looped_ced", "a looped encoder exists only in Looped-CED"
            assert cfg.loop.hyper_connections in ("none", "static", "dynamic_diag"), cfg.loop.hyper_connections
        if self.looped:
            if cfg.decoder.ffn_experts > 0:
                raise NotImplementedError("routed decoder FFN: the engine supports dense FFN only")
            if cfg.loop.depth_attention != "none":
                raise NotImplementedError("depth attention is not supported by the engine")
            assert cfg.loop.hyper_connections in ("none", "static", "dynamic_diag"), cfg.loop.hyper_connections
            assert cfg.loop.injection in ("parcae", "none"), cfg.loop.injection
            assert cfg.decoder.untie_mixers in ("none", "attention", "all"), cfg.decoder.untie_mixers
            assert 1 <= cfg.loop.t_eval <= cfg.loop.t_max
            # The engine's decoder merges memory and own keys without positions: the cap is built for the
            # encoder's global layers only (the control's, _encoder_stack).
            assert rope_cap(cfg.decoder.attention) is None, \
                "decoder.attention.rope_max_distance: the engine implements the rotary distance cap in the encoder's " \
                "global layers only"
            attn_cfgs.append(cfg.decoder.attention)
            assert all(isinstance(layer.mlp, (SwiGLU, ReLU2)) for layer in self.model.decoder_layers)
        for a in attn_cfgs:
            # grid KV sharing (AttentionConfig.kv_sharing; ced/model/attention.py) keeps components in its cache and assembles per-head keys
            # at read time: the engine's buffers and projections are the gqa layer's (use the model's prefill / extend)
            assert a.kv_sharing == "gqa", \
                f"attention kv_sharing {a.kv_sharing!r}: the engine supports 'gqa' only (use the model's prefill / extend)"
            assert a.gate in ("elementwise", "headwise", "none"), a.gate
            assert a.n_q_heads % a.n_kv_heads == 0
            # The engine forms the queries itself (_attn_project_one), without GatedAttention.query_factor.
            assert a.temperature == "fixed", f"attention temperature {a.temperature!r}: the engine supports 'fixed' only"
            # The engine's own attention is a softmax over its buffers (attend_partial / merge by log-sum-exp).
            assert a.normalizer == "softmax", \
                f"attention normalizer {a.normalizer!r}: the engine supports softmax only (use the model's prefill / extend)"
        gdn_cfgs = [cfg.encoder.gdn] + ([cfg.decoder.gdn] if self.looped else [])
        assert all(g.conv_kernel >= 2 for g in gdn_cfgs), "the single-step GDN conv assumes a kernel of at least 2"

    @torch.no_grad()
    def reset(self) -> None:
        """Start a new sequence (one document starting at position 0)."""
        self.enc_S.zero_()
        self.enc_conv.zero_()
        if self.enc_canon is not None:
            self.enc_canon.zero_()
        self.n_enc = 0  # positions the encoder has read (= next position)
        if self.looped:
            self.dec_S.zero_()
            self.dec_conv.zero_()
            self.mem_len = 0
            self.own_len = [0] * self.T
        self.pending: _Pending | None = None
        self.stats = SpecStats()

    def _snapshot(self) -> None:
        self.snap_enc_S.copy_(self.enc_S)
        self.snap_enc_conv.copy_(self.enc_conv)
        self.snap_dec_S.copy_(self.dec_S[:, 0])
        self.snap_dec_conv.copy_(self.dec_conv[:, 0])
        self.snap_counts = (self.n_enc, self.mem_len, self.own_len[0])

    def _restore(self) -> None:
        self.enc_S.copy_(self.snap_enc_S)
        self.enc_conv.copy_(self.snap_enc_conv)
        self.dec_S[:, 0].copy_(self.snap_dec_S)
        self.dec_conv[:, 0].copy_(self.snap_dec_conv)
        self.n_enc, self.mem_len, self.own_len[0] = self.snap_counts

    @torch.no_grad()
    def fill_random(self, context_len: int, n_generated: int, generator: torch.Generator | None = None) -> None:
        """Benchmarking: fill caches and states with random values as if context_len tokens had been
        read, n_generated of them decoder positions. Cost depends only on structure, so no prefill."""
        assert context_len <= self.max_len
        self.reset()
        g = generator
        self.enc_k[:, :, :, :context_len].normal_(generator=g)
        self.enc_v[:, :, :, :context_len].normal_(generator=g)
        if self.enc_kf is not None:
            self.enc_kf[:, :, :, :context_len].normal_(generator=g)
        self.enc_S.normal_(0.0, 0.05, generator=g)
        self.enc_conv.normal_(generator=g)
        if self.enc_canon is not None:
            self.enc_canon.normal_(generator=g)
        self.n_enc = context_len
        if self.looped:
            n_ctx = context_len - n_generated
            assert n_ctx <= self.max_context and n_generated <= self.max_decoder
            self.mem_k[:, :, :n_ctx].normal_(generator=g)
            self.mem_v[:, :, :n_ctx].normal_(generator=g)
            self.own_k[:, :, :, :n_generated].normal_(generator=g)
            self.own_v[:, :, :, :n_generated].normal_(generator=g)
            self.dec_S.normal_(0.0, 0.05, generator=g)
            self.dec_conv.normal_(generator=g)
            self.mem_len = n_ctx
            self.own_len = [n_generated] * self.T

    # ------------------------------------------------------------------------------------------
    # building blocks

    def _lin(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        c = self.gemv_row_chunk
        if c and x.shape[0] > c:
            return torch.cat([F.linear(part, w) for part in x.split(c)])
        return F.linear(x, w)

    def _ffn(self, mlp: SwiGLU | ReLU2, x: torch.Tensor) -> torch.Tensor:
        if isinstance(mlp, ReLU2):
            return self._lin(F.relu(self._lin(x, mlp.w_up.weight)).square(), mlp.w_down.weight)
        return self._lin(F.silu(self._lin(x, mlp.w_gate.weight)) * self._lin(x, mlp.w_up.weight), mlp.w_down.weight)

    @staticmethod
    def _rope(x: torch.Tensor, pos: torch.Tensor, a) -> torch.Tensor:
        """x (R, H, hd), pos (R,)."""
        return apply_rope(x[None], pos[None], a.rope_fraction, a.rope_theta)[0]

    def _attn_project(self, mixers: list, h: torch.Tensor, pos: torch.Tensor):
        """Queries, gate, keys, values for rows h (R, d); row r uses mixers[r] (rows sharing one
        module share the matmuls)."""
        if all(m is mixers[0] for m in mixers):
            return self._attn_project_one(mixers[0], h, pos)
        outs = [self._attn_project_one(m, h[r : r + 1], pos[r : r + 1]) for r, m in enumerate(mixers)]
        q, gate, k, v = (torch.cat(t) if t[0] is not None else None for t in zip(*outs))
        return q, gate, k, v

    def _attn_project_one(self, mixer, h, pos):
        a = mixer.cfg
        R, hq, hkv, hd = h.shape[0], a.n_q_heads, a.n_kv_heads, a.head_dim
        qp = self._lin(h, mixer.q_proj.weight)
        gate = None
        if a.gate == "elementwise":
            q, gate = qp.view(R, hq, 2 * hd).chunk(2, dim=-1)
        else:
            q = qp.view(R, hq, hd)
            if a.gate == "headwise":
                gate = self._lin(h, mixer.g_proj.weight).view(R, hq, 1)
        q = self._rope(mixer.q_norm(q), pos, a)
        k = self._rope(mixer.k_norm(self._lin(h, mixer.k_proj.weight).view(R, hkv, hd)), pos, a)
        v = self._lin(h, mixer.v_proj.weight).view(R, hkv, hd)
        return q, gate, k, v

    def _attn_project_capped(self, mixer, h, pos, cap: int):
        """_attn_project_one under the rotary distance cap: (q, q_far, gate, k, k_far, v). q_far is the normed query
        rotated by the cap instead of its position, k_far the normed key before rotation (GatedAttention.capped_queries
        / capped_keys_values; the engine supports the fixed temperature only, so no factor on q)."""
        a = mixer.cfg
        R, hq, hkv, hd = h.shape[0], a.n_q_heads, a.n_kv_heads, a.head_dim
        qp = self._lin(h, mixer.q_proj.weight)
        gate = None
        if a.gate == "elementwise":
            q, gate = qp.view(R, hq, 2 * hd).chunk(2, dim=-1)
        else:
            q = qp.view(R, hq, hd)
            if a.gate == "headwise":
                gate = self._lin(h, mixer.g_proj.weight).view(R, hq, 1)
        qn = mixer.q_norm(q)
        k_far = mixer.k_norm(self._lin(h, mixer.k_proj.weight).view(R, hkv, hd))
        v = self._lin(h, mixer.v_proj.weight).view(R, hkv, hd)
        return self._rope(qn, pos, a), self._rope(qn, torch.full_like(pos, cap), a), gate, self._rope(k_far, pos, a), k_far, v

    def _attention_capped(self, q: torch.Tensor, q_far: torch.Tensor, near: tuple, far: tuple | None) -> torch.Tensor:
        """One softmax over two disjoint key sets scored with different queries, merged by log-sum-exp: q (1, Hq, hd)
        over near = (k, v), the rotated keys within the cap, and q_far over far = (k_far, v), the unrotated keys
        beyond it (None when there are none). Views (1, Hkv, S, hd). Returns (1, Hq, hd) in the storage dtype."""
        R, hq, hd = q.shape
        hkv = near[0].shape[1]
        grp = hq // hkv
        parts = []
        for qq, part in ((q, near), (q_far, far)):
            if part is None:
                continue
            qs = qq.view(R, hkv, grp, hd).transpose(0, 1).reshape(1, hkv, R * grp, hd)
            o, lse = attend_partial(qs, part[0], part[1], self.attn_backend)
            parts.append((o.view(hkv, R, grp, hd).transpose(0, 1), lse.view(hkv, R, grp).transpose(0, 1)))
        return merge_partials(parts).reshape(R, hq, hd).to(q.dtype)

    def _attn_output(self, mixers: list, o: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
        if gate is not None:
            o = o * torch.sigmoid(gate)
        o = o.flatten(1)
        if all(m is mixers[0] for m in mixers):
            return self._lin(o, mixers[0].o_proj.weight)
        return torch.cat([self._lin(o[r : r + 1], m.o_proj.weight) for r, m in enumerate(mixers)])

    def _attention(self, q: torch.Tensor, shared: list, own: list | None) -> torch.Tensor:
        """q (R, Hq, hd). shared: [(k, v)] views (1, Hkv, S, hd) every row attends to (one kernel
        call for all rows); own: per row, one more (k, v) view. One softmax over the union of the
        key sets, merged by log-sum-exp. Returns (R, Hq, hd) in the storage dtype."""
        R, hq, hd = q.shape
        hkv = (shared[0][0] if shared else own[0][0]).shape[1]
        grp = hq // hkv
        parts = []
        if shared:
            # GQA: query head j * grp + i reads KV head j; all rows' queries go in the length axis.
            qs = q.view(R, hkv, grp, hd).transpose(0, 1).reshape(1, hkv, R * grp, hd)
            for k, v in shared:
                o, lse = attend_partial(qs, k, v, self.attn_backend)
                parts.append((o.view(hkv, R, grp, hd).transpose(0, 1), lse.view(hkv, R, grp).transpose(0, 1)))
        if own is not None:
            outs = [attend_partial(q[r].view(1, hkv, grp, hd), k, v, self.attn_backend) for r, (k, v) in enumerate(own)]
            parts.append((torch.stack([o.view(hkv, grp, hd) for o, _ in outs]),
                          torch.stack([lse.view(hkv, grp) for _, lse in outs])))
        return merge_partials(parts).reshape(R, hq, hd).to(q.dtype)

    def _gdn(self, mixers: list, x: torch.Tensor, S: torch.Tensor, conv: torch.Tensor) -> torch.Tensor:
        """Single-step Gated DeltaNet for rows x (R, d); row r updates S[r], conv[r] in place."""
        if all(m is mixers[0] for m in mixers):
            return self._gdn_one(mixers[0], x, S, conv)
        return torch.cat([self._gdn_one(m, x[r : r + 1], S[r : r + 1], conv[r : r + 1]) for r, m in enumerate(mixers)])

    def _gdn_one(self, mod, x: torch.Tensor, S: torch.Tensor, conv: torch.Tensor) -> torch.Tensor:
        if getattr(mod, "head_keep", None) is not None:  # the evaluation-only head zero is not read here: refuse by name
            raise ValueError(HEAD_ZERO_REFUSAL)
        c = mod.cfg
        R, hk, hv, dk, dv, K = x.shape[0], c.n_k_heads, c.n_v_heads, c.k_head_dim, c.v_head_dim, c.conv_kernel
        u_new = self._lin(x, mod.in_proj_qkv.weight)  # (R, C)
        w = mod.conv_weight  # (C, K): w[:, K-1] multiplies the current input, w[:, K-1-j] j steps back
        u = u_new * w[:, K - 1] + (conv * w[:, : K - 1].t()).sum(dim=1)
        for i in range(K - 2):  # shift the window in place; conv[:, K-2] is the newest input
            conv[:, i].copy_(conv[:, i + 1])
        conv[:, K - 2].copy_(u_new)
        u = F.silu(u)
        q, k, v = torch.split(u, [hk * dk, hk * dk, hv * dv], dim=-1)
        q = l2_normalize(q.view(R, hk, dk)) * dk**-0.5
        k = l2_normalize(k.view(R, hk, dk))
        v = v.view(R, hv, dv)
        if hv != hk:
            q = q.repeat_interleave(hv // hk, dim=1)
            k = k.repeat_interleave(hv // hk, dim=1)
        g, beta = mod.gates(x)  # (R, hv)
        S.mul_(torch.exp(g)[:, :, None, None])
        predicted = torch.matmul(k[:, :, None, :], S)[:, :, 0]  # (R, hv, dv)
        delta = beta[:, :, None] * (v - predicted)
        S.add_(k[:, :, :, None] * delta[:, :, None, :])
        o = torch.matmul(q[:, :, None, :], S)[:, :, 0]
        z = self._lin(x, mod.in_proj_z.weight).view(R, hv, dv)
        return self._lin(mod.norm(o, z).flatten(1), mod.out_proj.weight)

    @staticmethod
    def _canon(mod, x: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        """Single-step Canon layer (ced/model/canon.py CanonLayer) for rows x (R, d), the normed input of a global
        attention layer: x + the causal depthwise convolution over [hist ; x]. hist (R, K-1, d), the last K - 1 inputs
        (newest last), is updated in place. The engine reads one document from position 0, so the history holds
        zeros at the start and only that document's inputs afterwards: the reference's document rule."""
        w = mod.conv_weight  # (d, K): w[:, K-1] multiplies the current input, w[:, K-1-j] the input j steps back
        K = w.shape[1]
        out = x + x * w[:, K - 1] + (hist * w[:, : K - 1].t()).sum(dim=1)
        for i in range(K - 2):  # shift the window in place; hist[:, K-2] is the newest input
            hist[:, i].copy_(hist[:, i + 1])
        hist[:, K - 2].copy_(x)
        return out.to(x.dtype)

    # ------------------------------------------------------------------------------------------
    # encoder, memory, decoder loops, head

    def _encoder_stack(self, h: torch.Tensor, loop: int, pos: int) -> torch.Tensor:
        pos_t = torch.tensor([pos], device=self.device)
        # The per-layer residual scalars (EncoderConfig.layer_scalars; the control only): x0 is the stack's input at
        # this position, the embedding output of this token, so nothing of it is kept between steps.
        enc = self.model.encoder
        scalars = getattr(enc, "layer_scalars", False)
        gain, b = enc.layer_scalar_terms() if scalars else (None, None)  # a - 1 (None: a is 1) and b of every layer
        x0 = h
        for i, layer in enumerate(enc.layers):
            h_in = h
            x = layer.mixer_in_norm(h)
            j = self.enc_slot[i]
            # a global layer's Canon layer: the attention below reads x' (queries, keys, values). getattr: a layer
            # class that predates the key has no Canon layer
            canon = getattr(layer, "canon", None)
            if canon is not None:
                x = self._canon(canon, x, self.enc_canon[j : j + 1])
            if layer.kind == "gdn":
                s = loop * self.n_enc_gdn + j
                mixed = self._gdn([layer.mixer], x, self.enc_S[s : s + 1], self.enc_conv[s : s + 1])
            elif (cap := rope_cap(layer.mixer.cfg)) is not None:
                # the rotary distance cap (this layer's own: the model's, or none in a NoPE layer of the by-layer mix,
                # which takes the plain branch below): keys lo .. pos are within it (the plain read), keys before lo
                # beyond it (the query rotated by the cap against the unrotated keys); one softmax over both
                q, q_far, gate, k, k_far, v = self._attn_project_capped(layer.mixer, x, pos_t, cap)
                kb, kfb, vb = self.enc_k[loop, j], self.enc_kf[loop, j], self.enc_v[loop, j]
                kb[:, pos] = k[0]
                kfb[:, pos] = k_far[0]
                vb[:, pos] = v[0]
                lo = max(0, pos - cap)
                o = self._attention_capped(q, q_far, (kb[None, :, lo : pos + 1], vb[None, :, lo : pos + 1]),
                                           (kfb[None, :, :lo], vb[None, :, :lo]) if lo else None)
                mixed = self._attn_output([layer.mixer], o, gate)
            else:
                q, gate, k, v = self._attn_project_one(layer.mixer, x, pos_t)
                kb, vb = self.enc_k[loop, j], self.enc_v[loop, j]
                kb[:, pos] = k[0]
                vb[:, pos] = v[0]
                o = self._attention(q, [(kb[None, :, : pos + 1], vb[None, :, : pos + 1])], None)
                mixed = self._attn_output([layer.mixer], o, gate)
            h = h + layer.residual_scale * layer.mixer_out_norm(mixed)
            h = h + layer.residual_scale * layer.mlp_out_norm(self._ffn(layer.mlp, layer.mlp_in_norm(h)))
            if scalars:  # x <- a x + b x0 + f(x): the function the model's layers end with (EncoderLayer.forward)
                h = apply_layer_scalars(h, h_in, x0, None if gain is None else gain[i], None if b is None else b[i])
        return h

    def _encode(self, token: int) -> torch.Tensor:
        """The encoder at the next position for one token; returns h_enc (1, d)."""
        m, pos = self.model, self.n_enc
        assert pos < self.max_len, "encoder buffers are full"
        h = m.embed_norm(m.embed.weight[token][None])
        enc = m.encoder
        if enc.loops == 1:
            h = self._encoder_stack(h, 0, pos)
        else:  # the YOCO-U arm: the same loop boundary as the decoder's, per-loop caches
            hc = enc.hyper_connections
            e = enc.inject_norm(h)
            Y = hc.start(e[0])
            for k in range(enc.loops):
                z_in = hc.read(Y, k)
                u = enc.injection(z_in[None], e)
                if enc.loop_embedding is not None:
                    u = u + enc.loop_embedding[k]
                u = self._encoder_stack(u, k, pos)
                Y = hc.write(Y, z_in, u[0], k)
            h = hc.finish(Y)[None]
        self.n_enc += 1
        return h

    def _append_memory(self, h: torch.Tensor, pos: int) -> None:
        """A context position offers its memory K/V (one per memory projection)."""
        pos_t = torch.tensor([pos], device=self.device)
        i = self.mem_len
        assert i < self.max_context, "memory buffer is full"
        for j, mem in enumerate(self.model.memory_kv):
            a = mem.attn_cfg
            k, v = self._lin(mem.in_norm(h), mem.proj.weight).view(1, 2, a.n_kv_heads, a.head_dim).unbind(1)
            self.mem_k[j, :, i] = self._rope(mem.k_norm(k), pos_t, a)[0]
            self.mem_v[j, :, i] = v[0]
        self.mem_len += 1

    def _decoder_pass(self, k0: int, Ys: list, e: torch.Tensor, positions: list[int]) -> list:
        """One pass of the shared decoder block over R rows; row r runs loop k0 + r at positions[r]
        from its loop-boundary state Ys[r] with injection input e[r]. Matmuls of rows that share
        weights are batched. Returns the new loop-boundary states."""
        m = self.model
        hc = m.hyper_connections
        R = len(Ys)
        loops = list(range(k0, k0 + R))
        z_in = [hc.read(Y, k) for Y, k in zip(Ys, loops)]
        u = m.injection(torch.stack(z_in), e)
        if m.loop_embedding is not None:
            u = u + m.loop_embedding[k0 : k0 + R]
        pos_t = torch.tensor(positions, device=self.device)
        idx = [self.own_len[k] for k in loops]
        assert max(idx) < self.max_decoder, "own-KV buffers are full"
        for l, layer in enumerate(m.decoder_layers):
            mixers = [layer.mixers[k if len(layer.mixers) > 1 else 0] for k in loops]
            h = layer.attn_in_norm(u)
            j = self.dec_slot[l]
            if layer.kind == "memory_attention":
                q, gate, kk, vv = self._attn_project(mixers, h, pos_t)
                own = []
                for r, k in enumerate(loops):
                    kb, vb = self.own_k[j, k], self.own_v[j, k]
                    kb[:, idx[r]] = kk[r]
                    vb[:, idx[r]] = vv[r]
                    own.append((kb[None, :, : idx[r] + 1], vb[None, :, : idx[r] + 1]))
                s = m.memory_slot[l]
                shared = [(self.mem_k[s, None, :, : self.mem_len], self.mem_v[s, None, :, : self.mem_len])] if self.mem_len else []
                mixed = self._attn_output(mixers, self._attention(q, shared, own), gate)
            else:
                mixed = self._gdn(mixers, h, self.dec_S[j, k0 : k0 + R], self.dec_conv[j, k0 : k0 + R])
            u = u + layer.residual_scale * layer.attn_out_norm(mixed)
            u = u + layer.residual_scale * layer.mlp_out_norm(self._ffn(layer.mlp, layer.mlp_in_norm(u)))
        for k in loops:
            self.own_len[k] += 1
        return [hc.write(Y, z, u[r], k) for r, (Y, z, k) in enumerate(zip(Ys, z_in, loops))]

    def _head(self, items: list) -> torch.Tensor:
        """Logits (R, V) for loop-boundary states [(Y, loops_done)]: the final readout norm after
        T loops, the intermediate (draft) readout norm otherwise; one tied-head matmul for all rows."""
        m = self.model
        zs = []
        for Y, done in items:
            norm = m.decoder_norm if done == self.T else m.readout_norms[done - 1]
            zs.append(norm(m.hyper_connections.finish(Y)))
        return self._lin(torch.stack(zs), m.head_weight()).to(self.cdtype)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        return self._lin(x, self.model.head_weight())[0].to(self.cdtype)

    # ------------------------------------------------------------------------------------------
    # public API

    @torch.no_grad()
    def step(self, token: int, is_decoder_position: bool = True, return_aux: bool = False) -> torch.Tensor | None:
        """Read one token at the next position. Returns the logits (V,) for the next token: the
        final readout at a decoder position (and at every position of the control), the aux head at
        a context position if return_aux, else None."""
        pos = self.n_enc
        h = self._encode(token)
        if not self.looped:
            return self._logits(self.model.final_norm(h))
        if not is_decoder_position:
            self._append_memory(h, pos)
            return self._logits(self.model.aux_norm(h)) if return_aux else None
        e = self.model.inject_norm(h)
        Y = self.model.hyper_connections.start(e[0])
        for k in range(self.T):
            Y = self._decoder_pass(k, [Y], e, [pos])[0]
        return self._head([(Y, self.T)])[0]

    @torch.no_grad()
    def prefill(self, tokens, dec_mask, return_aux: bool = False) -> list:
        """Step through a prompt; returns the per-position outputs of `step`."""
        return [self.step(int(t), bool(d), return_aux) for t, d in zip(tokens, dec_mask)]

    @torch.no_grad()
    def generate(self, prompt, prompt_dec_mask, max_new_tokens: int) -> list[int]:
        """Plain greedy decoding (T loops per generated token). The prompt must end at a decoder position."""
        assert bool(prompt_dec_mask[-1]) or not self.looped, "the prompt must end at a decoder position"
        logits = self.prefill(prompt, prompt_dec_mask)[-1]
        out = []
        for i in range(max_new_tokens):
            out.append(int(logits.argmax()))
            if i + 1 < max_new_tokens:
                logits = self.step(out[-1], True)
        return out

    # ---- LoopSpec pipeline (T = 2)

    def _bubble(self, token: int, pos: int):
        """Encoder + loop 1 + draft readout for one row at pos (after a rejection, or to start)."""
        assert self.n_enc == pos
        e = self.model.inject_norm(self._encode(token))
        Y1 = self._decoder_pass(0, [self.model.hyper_connections.start(e[0])], e, [pos])[0]
        draft = int(self._head([(Y1, 1)])[0].argmax())
        return Y1, e[0], draft

    def _encode_draft(self, Y1, e_cur, pos, draft) -> None:
        """Snapshot at the accept point, then the encoder for the draft token at pos + 1."""
        self._snapshot()
        e_next = self.model.inject_norm(self._encode(draft))[0]
        self.pending = _Pending(Y1=Y1, e_cur=e_cur, pos=pos, e_next=e_next, draft=draft)

    @torch.no_grad()
    def spec_begin(self, token: int) -> None:
        """Start the pipeline at the next position with a committed token (no draft to verify yet)."""
        assert self.spec_ok, "the LoopSpec pipeline is implemented for Looped-CED with T = 2"
        pos = self.n_enc
        Y1, e, draft = self._bubble(token, pos)
        self._encode_draft(Y1, e, pos, draft)

    @torch.no_grad()
    def spec_step(self, force: bool | None = None) -> tuple[int, bool]:
        """One pipeline step: the batched 2-row decoder pass, both heads, the greedy accept check,
        then the encoder for the next draft (accept) or rollback + bubble + encoder (reject).
        Commits exactly one token, the target model's greedy token, and returns (token, accepted).
        force=True/False overrides the accept decision (benchmarking only: output is then not exact)."""
        p = self.pending
        assert p is not None, "call spec_begin first"
        hc = self.model.hyper_connections
        Ys = self._decoder_pass(0, [hc.start(p.e_next), p.Y1], torch.stack([p.e_next, p.e_cur]), [p.pos + 1, p.pos])
        logits = self._head([(Ys[0], 1), (Ys[1], 2)])
        target, next_draft = int(logits[1].argmax()), int(logits[0].argmax())
        accept = (target == p.draft) if force is None else force
        if accept:
            self._encode_draft(Ys[0], p.e_next, p.pos + 1, next_draft)
        else:
            self._restore()  # undo the encoder at pos+1 and loop 1 at pos+1; loop 2 at pos stands
            Y1, e, draft = self._bubble(target, p.pos + 1)
            self._encode_draft(Y1, e, p.pos + 1, draft)
        self.stats.steps += 1
        self.stats.accepted += int(accept)
        self.stats.accepted_flags.append(bool(accept))
        return (p.draft if accept else target), bool(accept)

    @torch.no_grad()
    def spec_bubble(self, token: int) -> int:
        """Benchmarking: the extra work of a rejection (restore the snapshot, then encoder + loop 1 +
        draft head for the corrected token). Leaves the pipeline state unchanged for a repeat call."""
        self._restore()
        return self._bubble(token, self.snap_counts[0])[2]

    @torch.no_grad()
    def speculative_generate(self, prompt, prompt_dec_mask, max_new_tokens: int) -> tuple[list[int], SpecStats]:
        """Greedy LoopSpec decoding: emits exactly the tokens of `generate`."""
        assert self.spec_ok, "the LoopSpec pipeline is implemented for Looped-CED with T = 2"
        assert bool(prompt_dec_mask[-1]), "the prompt must end at a decoder position"
        logits = self.prefill(prompt, prompt_dec_mask)[-1]
        out = [int(logits.argmax())]  # the first token comes straight from the target
        if max_new_tokens > 1:
            self.spec_begin(out[0])
        while len(out) < max_new_tokens:
            out.append(self.spec_step()[0])
        return out[:max_new_tokens], self.stats

    @torch.no_grad()
    def pipeline_pass_structural(self, n_rows: int, n_head_rows: int, generator=None) -> None:
        """Benchmarking at any T: one pipeline pass with rows at loops 1..n_rows from random inputs,
        the head on n_head_rows rows, then the encoder for one token. Advances the buffers like a real
        step; computes nothing meaningful."""
        assert self.looped and n_rows <= self.T
        m, d = self.model, self.cfg.d_model
        e = m.inject_norm(torch.randn(n_rows, d, generator=generator).to(self.dtype))
        pos = self.n_enc - 1
        Ys = self._decoder_pass(0, [m.hyper_connections.start(e[r]) for r in range(n_rows)], e,
                                [max(0, pos - r) for r in range(n_rows)])
        self._head([(Ys[r], r + 1) for r in range(n_head_rows)])
        self._encode(int(torch.randint(0, self.cfg.vocab_size, (1,), generator=generator)))
