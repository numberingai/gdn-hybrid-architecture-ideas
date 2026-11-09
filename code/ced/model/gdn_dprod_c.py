"""Gated DeltaProduct with a second step made from the first (GDNConfig.dprod.key lowrank, prev, perm, rot): the variant of the lighter forms of the
Gated DeltaProduct form.

THE RULE. Arm 6's recursion (ced/model/gdn_dprod.py, THE RULE; n_h = 2, o_t = S_t^T q_t, S in R^{d_k x d_v}), with step 2's (k2, v2, beta2) built
cheaply instead of from the full projections dprod.in_proj_kv and the short conv:

    S_t = R(k2_t, v2_t, beta2_t) R(k1_t, v1_t, beta1_t) (a_t S_{t-1}),    R(k, v, beta) S = (I - beta k k^T) S + beta k v^T

    C1  key lowrank   k2_t = L2norm(k1_t + U_h W_h x_t)    W_h: d -> r per key head (dprod.k_down, one GEMM d -> r H_k), U_h: r -> d_k (dprod.k_up,
                                                           (H_k d_k, r) applied block by block, as EDA's e_up), r = dprod.rank (16)
    C2  key prev      k2_t = k1_{t-1}                      the previous token's key after conv, SiLU and norm; ZERO at a document start and at
                                                           the first token of a sequence (R(0, v, beta) is the identity: no second write there)
    C3p key perm      k2_t = P k1_t                        P a fixed permutation of a head's d_k channels: roll (a cyclic shift by d_k / 2, the
                                                           design's choice) or random (a seeded draw, a persistent buffer)
    C3r key rot       k2_t = Q k1_t                        Q a fixed seeded d_k x d_k ROTATION (Haar by a signed QR, det +1), one for every head of
                                                           a layer, a persistent buffer: no parameter, no gradient

Step 1 is the layer's own with beta1 = 2 sigmoid(b), as the Gated DeltaProduct form's layer (gdn_dprod.BETA_MAX). The decay once a token before
step 1, the read once after step 2, GQA kept (value head j reads key head j // (H_v / H_k)), the gates, the output gate and norm are the layer's,
unchanged. Causality: C1, C3 read x_t or k1_t only; C2 reads k1_{t-1}, so token t's second write is built from tokens up to t.

WHAT THE DESIGN CHOSE FOR C1'S START, and why: the correction is NOT zero-initialised. W (k_down) ~ N(0, std^2) as every linear of the trunk and U
(k_up) ~ N(0, s^2) with s^2 = 1 / (d_k r std^2 d_model), so that E |U W x|^2 is about 1 for an input of unit RMS: k2 is k1 plus a correction of the
size of k1, cos(k1, k2) about 1 / sqrt(2) at the start, and the second write is a second, different address from the first step on. A zero-initialised
U would start the arm at k2 = k1 (two writes of one value under one key: a single write of strength beta1 + beta2 - beta1 beta2, a different layer
from the one the design prices) and give W no gradient until U moved. There is no zero-initialised gate here, so optim.zero_gate_decay moves no tensor
of this arm (the arms' runs set it to none as every arm does). The tests measure the cosine at init.

THE PATHS (each the same function): the stacks k (B, L, 2, H_k, d_k), v (B, L, 2, H_v, d_v), beta
(B, L, 2, H_v) go through ced/model/gdn_dprod_place.py _steps_rule, which is the Gated DeltaProduct form's routes exactly: the reference loop
(gdn_dprod.dprod_recurrence), the chunked rule on the 2T interleave, and the fused path (CUDA, kernel "ced" or "fla" from an empty state with varlen
boundaries: our kernel on the interleave AS IS, boundaries x 2, the output read on each token's second sub-step). The input preparation is the plain
layer's one fused kernel (qkv_prep over [q | k | v]); nothing of step 2 is in it. K2 (kda.kernel "k2") is not wired to this layer (refused by name).
THE ROUTE: GDNConfig.dprod.route "one-readout" (main's key, ced/kernels/gdn_1r.py) is read here. On the fused path the stacks above go to
gdn_dprod._one_readout_call as the layer's own layout (one branch in derived_forward before the interleave's _steps_rule): the readout once a token, q
and g at the token rows, no interleave copy of them and no readout on the sub-step with q = 0; the kernel assumes nothing about k_2, v_2, beta_2, so
every form here (a second key from the first, v_2 = v_1, beta_2 projected) is served as it is. The reference paths (the loop, the chunked rule, the
CPU, a carried state, cached generation) ignore the key. The stack v = [v1, v1] stays a materialised copy (about T H_v d_v elements a layer written
and read again): the design's one-readout bar assumes the kernel reads v once, which needs a second pointer in the kernels (kernel work, not done).
"interleave" (the default): the lines as they were, bit for bit. Fixed shapes throughout (stack, roll, a fixed index buffer, one batched GEMM; no
boolean mask, no nonzero).

THE CACHE. C2 carries one tensor more, DerivedState.k_prev (B, H_k, d_k), the unit key of the last token of the previous call; the next call's first
token reads it, masked to zero at a document start (ctx.is_doc_start, which sequence_context forms with the previous call's last document id).
DerivedState is GDNState with that one field (a subclass here, so that ced/model/gdn.py stays as it was); every other form needs no k_prev. The conv
history is the plain layer's (B, K-1, 2 H_k d_k + H_v d_v): no channel of step 2 passes the conv.

INITIALISATION. The extra tensors (in_proj_b, k_down, k_up, then the seeded buffer) are drawn from a generator of their own (init_own_parameters,
called by the model's build after init_weights: seed torch.initial_seed() + SEED_OFFSET), so every tensor R1-entry has starts as in R1-entry of the
same seed and the global RNG is not touched (gdn_dprod.py's, gdn_erase.py's and ldm.py's rule). A layer built on its own draws them at construction
from a fixed generator (STANDALONE_SEED). The seeded buffers (rot, a random perm) are persistent: they are in the state dict and come back with a
checkpoint, so a resume never redraws them (the model built for a resume is built under another global seed, and loading overwrites them).

OPTIMISER GROUPS (ced/train/optim.py): dprod.in_proj_b.weight is a gate projection (ADAMW_MATRIX_SUFFIXES "in_proj_b.weight", as); dprod.k_down.weight
and dprod.k_up.weight are named there too (the low-rank address pair, EDA's e_down / e_up): AdamW, never Muon, in the group WITH the run's weight
decay. The design says "hidden matrices"; the repo's rule for a thin projection that makes an address or a gate is AdamW (EDA's identical pair, KDA's
f_down / f_up), and a Muon update of a 16 x 512 matrix would orthogonalise a matrix whose singular values carry the correction's size.

PARAMETERS (exact): d H_v [+ d r H_k + r H_k d_k for C1] a layer; at 60M (d 512, 2 / 4 heads of 128) C2, C3: 2,048 a layer, 30,720 over the 15 GDN
layers; C1 r = 16: 22,528 a layer, 337,920 over 15; at 4B (d 2,560, 10 / 20 heads of 128) 51,200 and 481,280 a layer (1,536,000 and 14,438,400 over
30): the design's numbers. The seeded buffers (rot: d_k^2 floats; a random perm: d_k integers) are not parameters.

NOT BUILT (refused by name; check_second, check_model, derived_forward): a placement (dprod.layers / heads / heads_by_layer) and a GDN shape table
with a derived key (lighter forms of are whole-layer arms here; placement's partial layers run second step), n_h other than 2, K2 (kda.kernel "k2"),
every key of dprod that this module does not read (beta_max, beta_init and the like of other builds; `route` is read: a key that would be dropped
silently is refused), the value and beta alternatives the design prices (v2 = 0, a low-rank or permuted value, a tied or constant beta), the chunk
summaries and the stride (ced/model/gdn_dprod_sparse.py: not built together with a derived key), and everything the Gated DeltaProduct form refuses
(gdn_dprod.py NOT BUILT: the decoder, arches other than "control", the GDN-side arms, keep, padding, TST bag phases, CUDA graphs' metadata, the edges,
FlashQLA, the MTP drafter, the decode engine, FP8 / NVFP4). Python 3.11 compatible.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ced.config import DeltaProductConfig, GDNConfig
from ced.model.gdn import GDNState
from ced.model.gdn_dprod import BETA_MAX, _one_readout_call, _refuse_call, n_steps, route_of

SEED_OFFSET = 0xC2_0D_2026_1025  # the derived steps' own stream: seed = torch.initial_seed() + this
STANDALONE_SEED = 0xC2_0D  # a layer built outside a model (benches, kernel checks): a fixed stream, not the global one
DERIVED = ("lowrank", "prev", "perm", "rot")  # the keys of C1, C2, C3 permutation, C3 rotation
KEYS = ("proj",) + DERIVED  # "proj" is the Gated DeltaProduct form's (conv false and a stride or a summary: gdn_dprod_sparse.py)
VALUES = ("proj", "v1")
BETAS = ("proj",)
PERMS = ("roll", "random")
SECOND_FIELDS = ("key", "value", "beta", "rank", "perm", "conv")  # the keys of this module
READ_FIELDS = ("n_h",) + SECOND_FIELDS + ("route",)  # the DeltaProductConfig keys derived_forward reads (route: the fused path's readout, gdn_dprod.ROUTES); any other key off its default is refused


# --------------------------------------------------------------------------------------------------------------------------
# the keys


def is_derived(cfg: GDNConfig) -> bool:
    """True when step 2's key is made from the first step (GDNConfig.dprod.key in DERIVED). Does not raise on a bad value (check_second does)."""
    return getattr(getattr(cfg, "dprod", None), "key", "proj") in DERIVED


def second_keys_off(d: DeltaProductConfig) -> list:
    """The second-step keys of `d` that are off their defaults."""
    dflt = DeltaProductConfig()
    return [f for f in SECOND_FIELDS if getattr(d, f) != getattr(dflt, f)]


def check_second(cfg: GDNConfig, where: str = "GDNConfig.dprod") -> None:
    """The second-step keys of a layer's GDNConfig, checked (ValueError naming the key). n_h 1: every one at its default (they are read only with n_h
    2). key proj: the Gated DeltaProduct form's form (value, beta proj, conv true, rank and perm at their defaults). A derived key: n_h 2, value
    v1, beta proj, conv false, rank only with lowrank (1 .. d_k), perm only with key perm, no placement and no key this module does not read, no
    K2."""
    d = cfg.dprod
    n = n_steps(cfg)
    dflt = DeltaProductConfig()
    if d.key not in KEYS:
        raise ValueError(f"{where}.key {d.key!r}: one of {KEYS} ('summary' and 'pool' are not values of this key: arm D's chunk summaries are dprod.summary, "
                         "built in ced/model/gdn_dprod_sparse.py)")
    if d.value not in VALUES:
        raise ValueError(f"{where}.value {d.value!r}: one of {VALUES} (the design's other values, v2 = 0 and the low-rank or permuted value, are priced "
                         "in the design and not built)")
    if d.beta not in BETAS:
        raise ValueError(f"{where}.beta {d.beta!r}: {BETAS[0]!r} (beta2 = 2 sigmoid of its own N = H_v projection; the design's tied, constant and "
                         "summary betas are not built)")
    if d.perm not in PERMS:
        raise ValueError(f"{where}.perm {d.perm!r}: one of {PERMS}")
    if type(d.rank) is not int or d.rank < 1 or (d.key == "lowrank" and d.rank > cfg.k_head_dim):
        raise ValueError(f"{where}.rank {d.rank!r}: an integer >= 1 (with key 'lowrank' at most k_head_dim = {cfg.k_head_dim})")
    if type(d.conv) is not bool:
        raise ValueError(f"{where}.conv {d.conv!r}: true or false")
    off = second_keys_off(d)
    if n == 1:
        if off:
            raise ValueError(f"{where}.{', '.join(off)} set with n_h 1: the second step's keys are read only with n_h 2 (at n_h 1 they would "
                             "change nothing)")
        return
    if d.key == "proj":
        # conv is not held here: conv false with key proj is the strided or summarised second step of the variants (ced/model/gdn_dprod_sparse.py),
        # whose check_keys judges it (conv true with a stride or a summary, a stride or a summary with conv true: refused there)
        want = {"value": "proj", "beta": "proj", "rank": dflt.rank, "perm": dflt.perm}
        wrong = [f"{f} {getattr(d, f)!r} (the Gated DeltaProduct form wants {w!r})" for f, w in want.items() if getattr(d, f) != w]
        if wrong:
            raise ValueError(f"{where}.key 'proj' (the Gated DeltaProduct second step: full projections and the short conv; with conv false and a stride or a "
                             f"summary, the sparse variants') with {'; '.join(wrong)}; value v1 is the derived keys' (lowrank, prev, perm, rot)")
        return
    # a derived key
    if n != 2:
        raise ValueError(f"{where}.key {d.key!r} with n_h {n}: the derived second step is built for n_h 2 (one extra step made from the first)")
    wrong = [(f, getattr(d, f), w) for f, w in (("value", "v1"), ("beta", "proj"), ("conv", False)) if getattr(d, f) != w]
    if wrong:
        raise ValueError(f"{where}.key {d.key!r}: " + "; ".join(f"{f} {v!r} (a derived key wants {w!r}: the second step has no value or conv channels "
                                                               "of its own, and its beta is its own projection)" for f, v, w in wrong))
    if d.key != "lowrank" and d.rank != dflt.rank:
        raise ValueError(f"{where}.rank {d.rank!r} is read with key 'lowrank'; this layer's key is {d.key!r}")
    if d.key != "perm" and d.perm != dflt.perm:
        raise ValueError(f"{where}.perm {d.perm!r} is read with key 'perm'; this layer's key is {d.key!r}")
    own = type(d)()  # the section's own defaults (a build that adds keys to DeltaProductConfig adds them to the class this reads)
    unread = [f.name for f in dataclasses.fields(d) if f.name not in READ_FIELDS and getattr(d, f.name) != getattr(own, f.name)]
    if unread:
        raise ValueError(f"{where}.{', '.join(unread)} with key {d.key!r}: the derived second step (ced/model/gdn_dprod_c.py) reads none of these keys "
                         "(placement, beta_max, beta_init and the like), so it would drop them silently; not built together")
    if cfg.kda.kernel == "k2":
        raise ValueError(f"{where}.key {d.key!r} with GDNConfig.kda.kernel 'k2': K2's steps mode is not wired to the derived second step "
                         "(use kda.kernel 'chunk_kda', the default)")


def check_model(cfg) -> None:
    """The model-level checks (gdn_dprod.check_config's first hook): the second-step keys of both GDN sections, a derived key only in arch
    "control"'s encoder (check_config's own list follows), with the default placement and no GDN shape table."""
    check_second(cfg.encoder.gdn, "model.encoder.gdn.dprod")
    check_second(cfg.decoder.gdn, "model.decoder.gdn.dprod")
    if is_derived(cfg.decoder.gdn):
        raise ValueError("model.decoder.gdn.dprod.key: the derived second step is built for the encoder's GDN layers only")
    if is_derived(cfg.encoder.gdn) and cfg.encoder.gdn_shape_by_layer:
        raise ValueError("model.encoder.gdn_shape_by_layer with model.encoder.gdn.dprod.key "
                         f"{cfg.encoder.gdn.dprod.key!r}: a GDN shape of each layer's own is not built with the derived second step")


def params_per_layer(d_model: int, cfg: GDNConfig) -> int:
    """The tensors one GDN layer gains with a derived second step (exact): d H_v (the beta projection) [+ d r H_k + r H_k d_k for key lowrank].
    The seeded buffers (rot, a random perm) are not parameters."""
    d = cfg.dprod
    n = d_model * cfg.n_v_heads
    if d.key == "lowrank":
        n += d_model * d.rank * cfg.n_k_heads + d.rank * cfg.n_k_heads * cfg.k_head_dim
    return n


# --------------------------------------------------------------------------------------------------------------------------
# the layer's tensors


def random_rotation(dk: int, g: torch.Generator) -> torch.Tensor:
    """A Haar-random rotation (det +1) of R^dk from the CPU generator g: the Q of a Gaussian matrix's QR with R's diagonal made positive (so that Q is
    Haar), and the first column negated when the determinant is -1. float32 (a buffer), orthogonal to its rounding."""
    a = torch.randn(dk, dk, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))[None, :]
    if float(torch.linalg.det(q)) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(torch.float32)


class DerivedSteps(nn.Module):
    """The tensors of a derived second step (GatedDeltaNet.dprod; module docstring): in_proj_b (d -> H_v, the second step's beta), for key lowrank
    k_down (d -> r H_k) and k_up ((H_k d_k, r)), and the seeded buffer of key rot (rot, (d_k, d_k)) or of key perm with perm 'random' (perm, (d_k,),
    integers). Built on the meta device and moved to `device` without a draw, then drawn from a fixed generator (reset_own); the model's build draws
    them again from the run's own stream (gdn_dprod.init_own_parameters). `partial` False and `carry` None: every head carries; `n_extra` 1: one
    extra step. GatedDeltaNet.forward calls forward_fn in place of dprod_forward."""

    partial = False
    carry = None
    n_extra = 1

    def __init__(self, d_model: int, cfg: GDNConfig, device=None, dtype=None):
        super().__init__()
        check_second(cfg)
        assert is_derived(cfg) and n_steps(cfg) == 2
        d = cfg.dprod
        self.key, self.perm_kind, self.rank = d.key, d.perm, d.rank
        self.d_model, self.hk, self.hv, self.dk = d_model, cfg.n_k_heads, cfg.n_v_heads, cfg.k_head_dim
        dev = torch.device("cpu") if device is None else torch.device(device)

        def linear(n_in: int, n_out: int) -> nn.Linear:
            lin = nn.Linear(n_in, n_out, bias=False, device="meta", dtype=dtype).to_empty(device=dev)
            lin.own_init = True  # init_weights (ced/model/looped_ced.py) draws every other linear from the global stream
            return lin

        self.in_proj_b = linear(d_model, self.hv)
        if self.key == "lowrank":
            self.k_down = linear(d_model, self.rank * self.hk)
            self.k_up = linear(self.rank, self.hk * self.dk)  # (H_k d_k, r), applied block by block (second_key)
        if self.key == "rot":
            self.register_buffer("rot", torch.empty(self.dk, self.dk, device=dev, dtype=torch.float32))
        if self.key == "perm" and self.perm_kind == "random":
            self.register_buffer("perm", torch.empty(self.dk, device=dev, dtype=torch.long))
        self.reset_own(torch.Generator().manual_seed(STANDALONE_SEED), 0.02)

    @staticmethod
    def forward_fn(layer, *args, **kw):
        return derived_forward(layer, *args, **kw)

    @torch.no_grad()
    def reset_own(self, g: torch.Generator, std: float) -> None:
        """in_proj_b, then k_down, k_up, then the seeded buffer: from the CPU generator g in float32 (the rotation in float64, then rounded), copied in.
        in_proj_b and k_down N(0, std); k_up N(0, s) with s = 1 / (sqrt(d_k r) std sqrt(d_model)) (module docstring: E |U W x|^2 about 1 for an input
        of unit RMS). Tensors on the meta device are skipped (nothing to draw into)."""
        draws = [(self.in_proj_b.weight, std)]
        if self.key == "lowrank":
            draws += [(self.k_down.weight, std), (self.k_up.weight, 1.0 / (math.sqrt(self.dk * self.rank) * std * math.sqrt(self.d_model)))]
        for p, s in draws:
            if not p.is_meta:
                p.copy_((torch.randn(p.shape, generator=g, dtype=torch.float32) * s).to(device=p.device, dtype=p.dtype))
        if self.key == "rot" and not self.rot.is_meta:
            self.rot.copy_(random_rotation(self.dk, g).to(device=self.rot.device, dtype=self.rot.dtype))
        if self.key == "perm" and self.perm_kind == "random" and not self.perm.is_meta:
            self.perm.copy_(torch.randperm(self.dk, generator=g).to(self.perm.device))

    def second_key(self, x: torch.Tensor, k1: torch.Tensor, is_doc_start: torch.Tensor, k_prev: torch.Tensor | None) -> torch.Tensor:
        """k2 (B, L, H_k, d_k) in k1's dtype from the layer's input x (B, L, d), the unit keys k1 (B, L, H_k, d_k), the document starts (B, L) and, for
        key prev, the last key of the previous call (B, H_k, d_k) or None (the first call of a sequence). Fixed shapes: a roll or a fixed index
        buffer, one batched GEMM, a shift and a 0 / 1 factor."""
        from ced.model.gdn import l2_normalize

        wide = torch.promote_types(k1.dtype, torch.float32)
        if self.key == "lowrank":
            z = self.k_down(x).unflatten(-1, (self.hk, self.rank))
            # one batched GEMM over the key heads (H_k, tokens, r) x (H_k, r, d_k), as EDA's direction: not z[..., None, :] @ W^T, which would
            # broadcast U to every token and save that copy for the backward
            w = self.k_up.weight.to(z.dtype).view(self.hk, self.dk, self.rank)
            delta = torch.einsum("...hr,hdr->...hd", z, w)
            return l2_normalize(k1.to(wide) + delta.to(wide)).to(k1.dtype)
        if self.key == "prev":
            first = torch.zeros_like(k1[:, :1]) if k_prev is None else k_prev[:, None].to(k1.dtype)
            prev = torch.cat([first, k1[:, :-1]], dim=1)  # k1_{t-1}; the first row from the previous call (zeros at the start of a sequence)
            return prev * (~is_doc_start).to(prev.dtype)[..., None, None]  # a document start reads no previous key: R(0, v, beta) = identity
        if self.key == "perm":
            return torch.roll(k1, shifts=self.dk // 2, dims=-1) if self.perm_kind == "roll" else k1[..., self.perm]
        assert self.key == "rot"
        # Q k1 on each head. NOT "float32, rounded once" under autocast: the casts to `wide` are undone there (autocast runs an einsum, a bmm, in its own dtype),
        # so with bf16 autocast both operands are rounded to bf16 before the product (accumulated in float32, the output bf16): Q' = bf16(Q) has |Q' Q'^T - I| up
        # to 1.3e-3 and |k2| within 1.4e-3 of 1 at d_k 128 (measured), the spread of a bf16 unit key. That is the op the cost tool prices (a
        # bf16 GEMM at the qkv GEMM's rate); a float32 product here (autocast disabled) would be a different, dearer op. In float32 and float64 runs it is exact.
        return torch.einsum("...hd,ed->...he", k1.to(wide), self.rot.to(wide)).to(k1.dtype)

    def extra_repr(self) -> str:
        return f"key={self.key!r}" + (f", rank={self.rank}" if self.key == "lowrank" else "") + (
            f", perm={self.perm_kind!r}" if self.key == "perm" else "")


def steps_of(d_model: int, cfg: GDNConfig, weight: torch.Tensor, heads) -> DerivedSteps:
    """gdn_dprod.build's last statement for a derived key: every head carries (heads None, or the layer's every value head as the encoder hands
    them); some heads only is refused. `weight`: the layer's in_proj_qkv.weight (its device and dtype)."""
    check_second(cfg)
    if heads is not None and tuple(heads) != tuple(range(cfg.n_v_heads)):
        raise ValueError(f"GDNConfig.dprod.key {cfg.dprod.key!r} with the carrying value heads {tuple(heads)!r} of {cfg.n_v_heads}: the derived second "
                         "step is built for every head of a layer")
    return DerivedSteps(d_model, cfg, device=weight.device, dtype=weight.dtype)


# --------------------------------------------------------------------------------------------------------------------------
# the layer's forward


@dataclasses.dataclass
class DerivedState(GDNState):
    """GDNState of a layer with a derived second step: k_prev (B, H_k, d_k), the unit key k_1 of the last token processed (key prev; None for the
    other keys, and in the training forward, which builds no cache)."""

    k_prev: torch.Tensor | None = None


def derived_forward(layer, x: torch.Tensor, ctx, state=None, valid=None, keep: bool = False, arms=None, x_write=None, ccr=None,
                    x_read=None, lerped: bool = False):
    """GatedDeltaNet.forward for a layer whose `dprod` is a DerivedSteps: the same inputs and result, (y, GDNState). The plain layer's prologue
    (one conv over [q | k_1 | v_1], the fused prep on the fused path), then step 2's key from DerivedSteps.second_key, v2 = v1 and beta2 from
    dprod.in_proj_b stacked with step 1's, and the Gated DeltaProduct form's routes (gdn_dprod_place._steps_rule) on the stacks, or on the fused
    path with dprod.route "one-readout" the one-readout route (gdn_dprod._one_readout_call) on the same stacks."""
    from ced.model import gdn as G
    from ced.model.gdn_dprod_place import _steps_rule

    _refuse_call(layer, ctx, valid, keep, arms, x_write, ccr, x_read)
    B, L, _ = x.shape
    cfg, steps = layer.cfg, layer.dprod
    hk, hv, dk, dv, K = cfg.n_k_heads, cfg.n_v_heads, cfg.k_head_dim, cfg.v_head_dim, cfg.conv_kernel
    u = layer.in_proj_qkv(x)  # [q | k_1 | v_1]: the second step has no channel of its own
    if state is not None and state.conv_input.shape[-1] != u.shape[-1]:
        raise ValueError(f"this GDN cache holds {state.conv_input.shape[-1]} conv channels; a layer with a derived second step carries {u.shape[-1]}")
    if steps.key == "prev" and state is not None and getattr(state, "k_prev", None) is None:
        raise ValueError("this GDN cache holds no previous key (DerivedState.k_prev): it was not written by a layer with GDNConfig.dprod.key 'prev'")
    use_fused = cfg.kernel in ("fla", "ced") and state is None and ctx.cu_seqlens is not None and G.fla_supported(x)
    kernel_dtype = (torch.bfloat16 if use_fused and cfg.input_dtype == "bfloat16" and u.dtype == torch.bfloat16 else None)
    if layer._fused_prep and state is None and ctx.cu_seqlens is not None and G.fla_supported(x) and L >= K - 1:
        q, k1, v1 = G.qkv_prep(u, ctx.doc, layer.conv_weight, hk, hv, dk, out_dtype=kernel_dtype)  # q scaled and unit, k_1 unit
        conv_input, conv_doc = u[:, L - (K - 1):], ctx.doc[:, L - (K - 1):]
    else:
        hist, hist_doc = (None, None) if state is None else (state.conv_input, state.conv_doc)
        uc, conv_input, conv_doc = G.causal_doc_conv(u, ctx.doc, layer.conv_weight, hist, hist_doc)
        uc = F.silu(uc)
        q, k1, v1 = torch.split(uc, [hk * dk, hk * dk, hv * dv], dim=-1)
        q = G.l2_normalize(q.reshape(B, L, hk, dk)) * dk**-0.5
        k1 = G.l2_normalize(k1.reshape(B, L, hk, dk))
        v1 = v1.reshape(B, L, hv, dv)
    k2 = steps.second_key(x, k1, ctx.is_doc_start, None if state is None else getattr(state, "k_prev", None))
    # beta = 2 sigmoid(b) on both steps (gdn_dprod.BETA_MAX): step 1's from gates(), step 2's from dprod.in_proj_b
    g, beta1 = layer.gates(x)
    beta2 = BETA_MAX * torch.sigmoid(steps.in_proj_b(x))
    k = torch.stack([k1, k2], dim=2)  # (B, L, 2, H_k, d_k)
    v = torch.stack([v1, v1], dim=2)  # (B, L, 2, H_v, d_v): one value, two addresses (dprod.value 'v1')
    beta = torch.stack([BETA_MAX * beta1, beta2], dim=2)  # (B, L, 2, H_v)
    if kernel_dtype is not None:
        q, k, v, beta = q.to(kernel_dtype), k.to(kernel_dtype), v.to(kernel_dtype), beta.to(kernel_dtype)
    S0 = None if use_fused else (x.new_zeros(B, hv, dk, dv) if state is None else state.S)
    if use_fused and route_of(cfg) == "one-readout":
        # GDNConfig.dprod.route (ced/kernels/gdn_1r.py): the stacks above as they are, the readout once a token, q and g at the token rows (module docstring, THE ROUTE)
        o, S = _one_readout_call(q, k, v, g, beta, ctx.cu_seqlens, ctx.cu_seqlens_cpu, **layer._ced_kw), None
    else:
        o, S = _steps_rule(layer, q, k, v, g, beta, ctx, S0, use_fused)
    z = layer.in_proj_z(x).view(B, L, hv, dv)
    y = layer.out_proj(layer.norm(o, z).flatten(2))
    k_prev = k1[:, -1].clone() if steps.key == "prev" and not use_fused else None  # the training forward builds no cache (S is None)
    return y, DerivedState(conv_input=conv_input, conv_doc=conv_doc, S=S, k_prev=k_prev)
