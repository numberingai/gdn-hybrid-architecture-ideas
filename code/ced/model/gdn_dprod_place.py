"""Gated DeltaProduct per layer and per head: which GDN layers and which value heads of a layer carry the extra steps (PLACEMENT), and a GDN shape
of each layer's own (EncoderConfig.gdn_shape_by_layer).

WHY A MODULE OF ITS OWN. Arm 6's code (ced/model/gdn_dprod.py) is the code the screen's arm trained and the GPU values check passed on, and the
design and split notes cite its lines and statements (tests/test_gdn_dprod_split.py, tests/test_dprod_lite_cost.py hold them). This module
leaves its statements as they were and adds the placement beside it: gdn_dprod.py calls three thin hooks at its end (steps_of, no_steps,
place_check), and GatedDeltaNet.forward calls the function a layer's `dprod` names (`forward_fn`, partial_forward here) in place of
dprod_forward when the layer carries only some of its heads. A layer that carries every head is the Gated DeltaProduct form's layer, built and
run by gdn_dprod.py exactly as before (DeltaProductSteps, dprod_forward).

THE KEYS (ced/config.py DeltaProductConfig and EncoderConfig; resolved per GDN layer by EncoderConfig.gdn_shapes / dprod_carry / gdn_layers,
all off by default): dprod.layers (the GDN ordinals that carry; () = every layer), dprod.heads (the first m value heads of a carrying layer;
0 = all), dprod.heads_by_layer ({ordinal: value heads}); gdn_shape_by_layer ({ordinal: (n_k_heads, n_v_heads, head_dim)}). A layer that does
not carry has n_h 1 in its own config (no `dprod` module, nothing drawn: R1-entry's layer, tensors and outputs). A layer that carries some value
heads (a union of whole key groups, which the kernels take) is built here:

- PartialSteps: DeltaProductSteps for the carrying heads only (a copy of the layer's config with the carrying heads' counts; so in_proj_kv is
  (n_h - 1)(H_k' d_k + H_v' d_v) x d, in_proj_b (n_h - 1) H_v' x d, conv_weight likewise, drawn from the same generator in module order), with
  `carry` (the carrying value heads), `partial` True and the static runs of heads the forward slices on.
- partial_forward: the layer's forward. One [in_proj_qkv | in_proj_kv] projection and one conv over every channel (the fused prep on the fused
  path), as the Gated DeltaProduct form's; the carrying heads' q, k_1, v_1, g, beta_1 are sliced out (take: static slices and at most one
  concatenation: fixed shapes, no index tensor, no boolean mask, no nonzero), the extra steps' keys and values stacked and step 1's beta made 2
  sigmoid(b) as the Gated DeltaProduct form's, and the n_h rows run on the n_h T interleave (one fused kernel call, boundaries x n_h) or off the
  fused path through the chunked rule / the loop; the OTHER heads run the plain GDN step the layer makes for them (beta = sigmoid(b), one rank-one
  write, one fused kernel call on the token rows, or the chunked rule / the loop): a head that does not carry is Gated DeltaNet's head exactly, not
  an n_h head with beta 0. The outputs are joined along the head axis (merge) before the gated output norm; off the fused path the state is split and
  joined the same way, so the cache holds the layer's own and the carrying heads' conv channels and the state of every head.
- The one-readout kernel K2 (KDAConfig.kernel "k2") takes whole layers that all carry; a placement that leaves any layer or head without the
  extra steps is refused with it by name (head-mask mode is the design's kernel, not built).

Refused by name here: placement keys with n_h 1 (EncoderConfig.dprod_carry), the decoder's placement keys, a placement key in a layer's own
config (a layer has no ordinal), carrying heads that cut a key group, and, with a shape table, the models and tools that read ONE GDN shape for
the whole stack (model.fused_projections, loop.recur_layers, arch other than "control", the GDN-side arms, the second pass, the MTP drafter,
FP8 / NVFP4; the decode engine and the GDN state tools through ced/model/gdn_rules.py). Python 3.11 compatible.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from ced.config import GDNConfig
from ced.model.gdn_dprod import (BETA_MAX, DeltaProductSteps, _fused_call, _refuse_call, _unit, dprod_recurrence, interleave, last_substep,
                                 n_steps)


PLACEMENT_FIELDS = ("n_h", "layers", "heads", "heads_by_layer")  # the DeltaProductConfig keys partial_forward reads; any other key is not read


def carry_counts(cfg: GDNConfig, carry: tuple | None = None) -> tuple:
    """(H_k', H_v') of the heads that carry the extra steps: the whole layer's (carry None) or those of the value heads `carry` (a union of whole
    key groups: H_v' = len(carry), H_k' = H_v' H_k / H_v)."""
    if carry is None:
        return cfg.n_k_heads, cfg.n_v_heads
    return len(carry) * cfg.n_k_heads // cfg.n_v_heads, len(carry)


def step_channels(cfg: GDNConfig, carry: tuple | None = None) -> int:
    """The pre-conv channels of one extra step: [k_j (H_k' d_k) | v_j (H_v' d_v)], the carrying heads' (the layer's all, carry None)."""
    hk, hv = carry_counts(cfg, carry)
    return hk * cfg.k_head_dim + hv * cfg.v_head_dim


def params_per_layer(d_model: int, cfg: GDNConfig, carry: tuple | None = None) -> int:
    """The tensors one GDN layer adds (exact): (n_h - 1) [d c + d H_v' + K c], c = step_channels, H_v' the carrying value heads."""
    n, c = n_steps(cfg) - 1, step_channels(cfg, carry)
    return n * (d_model * c + d_model * carry_counts(cfg, carry)[1] + cfg.conv_kernel * c)


def runs_of(heads) -> tuple:
    """((start, length), ...): the maximal runs of consecutive indices in `heads` (sorted, distinct)."""
    out: list = []
    for h in heads:
        if out and out[-1][0] + out[-1][1] == h:
            out[-1] = (out[-1][0], out[-1][1] + 1)
        else:
            out.append((h, 1))
    return tuple(out)


def take(t: torch.Tensor, runs: tuple, dim: int) -> torch.Tensor:
    """The heads of `runs` (runs_of) along `dim`, in order: slices (views) and one concatenation. Static Python integers throughout: the shapes are
    fixed, there is no index tensor, no boolean mask and no nonzero."""
    pieces = [t.narrow(dim, s, n) for s, n in runs]
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim)


def merge(a: torch.Tensor, runs_a: tuple, b: torch.Tensor, runs_b: tuple, dim: int) -> torch.Tensor:
    """The inverse of take for a partition of the heads: a holds the heads of runs_a and b those of runs_b (each in order); the result has them in
    head order along `dim`. Slices and one concatenation."""
    pieces, off_a, off_b = [], 0, 0
    for s, n in runs_a:
        pieces.append((s, a.narrow(dim, off_a, n)))
        off_a += n
    for s, n in runs_b:
        pieces.append((s, b.narrow(dim, off_b, n)))
        off_b += n
    pieces.sort(key=lambda p: p[0])
    return torch.cat([p for _, p in pieces], dim)


def placement_keys(cfg: GDNConfig) -> list:
    """The PLACEMENT keys of this GDNConfig.dprod that are off their defaults (EncoderConfig.gdn_layers resets them in the config each layer is
    built from: a layer has no ordinal of its own)."""
    d = cfg.dprod
    return [k for k, on in (("layers", d.layers != ()), ("heads", d.heads != 0), ("heads_by_layer", d.heads_by_layer != {})) if on]


class PartialSteps(DeltaProductSteps):
    """The extra steps' tensors of a layer that carries SOME of its value heads (module docstring): DeltaProductSteps built on a copy of the
    layer's config with the carrying heads' counts, so every tensor, its name, its group and its draw are DeltaProductSteps' at that size. `carry`:
    the carrying value heads (a sorted tuple, whole key groups); `partial` True; `forward_fn` the layer's forward (GatedDeltaNet.forward calls it in
    place of dprod_forward); hk_c, hv_c the carrying heads' counts; runs_* the static runs of heads the forward slices on."""

    partial = True

    def __init__(self, d_model: int, cfg: GDNConfig, carry: tuple, device=None, dtype=None):
        hk, hv = cfg.n_k_heads, cfg.n_v_heads
        g = hv // hk
        keys = tuple(sorted({h // g for h in carry}))
        assert 0 < len(carry) < hv and len(keys) * g == len(carry), f"carry {carry!r}: some whole key groups of {g} value heads, not all"
        hk_c, hv_c = carry_counts(cfg, tuple(carry))
        super().__init__(d_model, dataclasses.replace(cfg, n_k_heads=hk_c, n_v_heads=hv_c), device=device, dtype=dtype)
        self.hk_c, self.hv_c = hk_c, hv_c
        self.carry = tuple(carry)
        self.runs_v = runs_of(self.carry)
        self.runs_k = runs_of(keys)
        self.runs_v_plain = runs_of([h for h in range(hv) if h not in set(self.carry)])
        self.runs_k_plain = runs_of([h for h in range(hk) if h not in set(keys)])

    @staticmethod
    def forward_fn(layer, *args, **kw):
        return partial_forward(layer, *args, **kw)


def check_heads(cfg: GDNConfig, heads) -> tuple:
    """The carrying heads a layer is given (EncoderConfig.dprod_carry's entry), checked: distinct, ascending, whole key groups."""
    heads = tuple(heads)
    hk, hv = cfg.n_k_heads, cfg.n_v_heads
    g = hv // hk
    good = (len(heads) > 0 and list(heads) == sorted(set(heads)) and all(type(h) is int and 0 <= h < hv for h in heads)
            and all(all(j in heads for j in range(k * g, (k + 1) * g)) for k in {h // g for h in heads}))
    if not good:
        raise ValueError(f"the carrying value heads {heads!r}: distinct, ascending, among 0 .. {hv - 1}, whole key groups of {g} "
                         f"value heads (a layer with {hk} key and {hv} value heads)")
    return heads


def steps_of(d_model: int, cfg: GDNConfig, weight: torch.Tensor, heads) -> DeltaProductSteps:
    """gdn_dprod.build's last statement: the layer's extra steps for the value heads `heads` (some heads: PartialSteps). `weight`: the
    layer's in_proj_qkv.weight (its device and dtype)."""
    if placement_keys(cfg):
        raise ValueError(f"GDNConfig.dprod.{', '.join(placement_keys(cfg))} on a GDN layer built on its own: the placement keys are "
                         "model.encoder.gdn.dprod's, resolved per layer by EncoderConfig.gdn_layers (the layer is given its heads)")
    if heads is None:
        return DeltaProductSteps(d_model, cfg, device=weight.device, dtype=weight.dtype)
    heads = check_heads(cfg, heads)
    if len(heads) == cfg.n_v_heads:
        return DeltaProductSteps(d_model, cfg, device=weight.device, dtype=weight.dtype)
    if cfg.kda.kernel == "k2":
        raise ValueError(f"GDNConfig.kda.kernel 'k2' (K2, one-readout) with a layer that carries only value heads {heads!r} of {cfg.n_v_heads}: "
                         "K2 takes whole layers (a head-mask mode is not built); use the interleave route "
                         "(kda.kernel 'chunk_kda', the default)")
    return PartialSteps(d_model, cfg, heads, device=weight.device, dtype=weight.dtype)


def no_steps(heads) -> None:
    """gdn_dprod.build's n_h 1 return: a layer that does not carry has no carrying heads."""
    if heads:
        raise ValueError(f"GDNConfig.dprod.n_h 1 with carrying heads {tuple(heads)!r}: a layer that does not carry has none")
    return None


def place_check(cfg, enc: int) -> bool:
    """gdn_dprod.check_config's early return, extended: validates the placement keys and the shape table by name whatever n_h is (EncoderConfig.
    dprod_carry / gdn_layers), refuses the decoder's placement keys, and with a shape table or a placement refuses what reads one GDN shape or does
    not take it. Returns True when nothing more is to be checked (n_h 1 and no shape table): gdn_dprod.check_config then returns."""
    if placement_keys(cfg.decoder.gdn):
        raise ValueError(f"model.decoder.gdn.dprod.{', '.join(placement_keys(cfg.decoder.gdn))}: the placement keys are the encoder's "
                         "(Gated DeltaProduct is built for the encoder's GDN layers only)")
    carry = cfg.encoder.dprod_carry()
    cfg.encoder.gdn_layers()
    if enc == 1 and not cfg.encoder.gdn_shape_by_layer:
        return True
    if cfg.encoder.gdn_shape_by_layer:  # tools and ops that read ONE GDN shape for the whole stack (ced/model/gdn_rules.py lists them)
        refused = [(cfg.arch != "control", f"arch {cfg.arch!r} (built for 'control')"),
                   (cfg.encoder.gdn_write_mix != "none", f"model.encoder.gdn_write_mix {cfg.encoder.gdn_write_mix!r} (the GDN-side arm A)"),
                   (bool(cfg.encoder.gdn_ccr_spacing), f"model.encoder.gdn_ccr_spacing {cfg.encoder.gdn_ccr_spacing!r} (the GDN-side arm CCR)"),
                   (cfg.loop.pass2_gate != "none", f"model.loop.pass2_gate {cfg.loop.pass2_gate!r} (the zero-write second pass reads "
                                                   "GDN's record)"),
                   (bool(cfg.mtp.enabled), "model.mtp.enabled (the drafter writes the GDN step itself, ced/infer/mtp_spec.py)"),
                   (cfg.fp8.mode != "none", f"model.fp8.mode {cfg.fp8.mode!r}"),
                   (cfg.nvfp4.mode != "none", f"model.nvfp4.mode {cfg.nvfp4.mode!r}"),
                   (bool(cfg.fused_projections), f"model.fused_projections {list(cfg.fused_projections)!r} (the fused GEMMs are sized from "
                                                 "one GDN shape)"),
                   (bool(cfg.loop.recur_layers), f"model.loop.recur_layers {list(cfg.loop.recur_layers)!r} (the looped middle's recurrence "
                                                 "FLOPs read one GDN shape)")]
        bad = [why for cond, why in refused if cond]
        if bad:
            raise ValueError(f"model.encoder.gdn_shape_by_layer (a GDN shape of each layer's own, ced/config.py) is not built with: "
                             f"{'; '.join(bad)}")
    d = cfg.encoder.gdn.dprod  # every other key of dprod (the keys of other builds: beta_max, beta_init) is not read by partial_forward
    partial = [i for i, (c, s) in enumerate(zip(carry, cfg.encoder.gdn_shapes())) if c is not None and 0 < len(c) < s.n_v_heads]
    other = [f.name for f in dataclasses.fields(d) if f.name not in PLACEMENT_FIELDS and getattr(d, f.name) != getattr(type(d)(), f.name)]
    if partial and other:
        raise ValueError(f"model.encoder.gdn.dprod.{', '.join(other)} with GDN layer(s) {partial} that carry only some of their value heads: "
                         "partial_forward (ced/model/gdn_dprod_place.py) writes the Gated DeltaProduct betas (2 sigmoid(b) on the carrying heads, sigmoid(b) "
                         "on the others) and reads none of these keys, so it would drop them silently; a placement of whole layers (every "
                         "value head of the layers in dprod.layers) runs dprod_forward and reads them")
    if enc != 1 and cfg.encoder.gdn.kda.kernel == "k2" and any(c is not None and len(c) != s.n_v_heads for c, s in
                                                                zip(carry, cfg.encoder.gdn_shapes())):
        raise ValueError("model.encoder.gdn.kda.kernel 'k2' (K2, one readout a token) with a placement that leaves some GDN layer or head "
                         "without the extra steps (model.encoder.gdn.dprod.layers / heads / heads_by_layer): K2 takes whole layers that "
                         "all carry (a head-mask mode is not built); use kda.kernel 'chunk_kda'")
    return False


# --------------------------------------------------------------------------------------------------------------------------
# the layer's forward when it carries some of its heads


def partial_forward(layer, x: torch.Tensor, ctx, state=None, valid=None, keep: bool = False, arms=None, x_write=None, ccr=None,
                    x_read=None, lerped: bool = False):
    """GatedDeltaNet.forward for a layer whose `dprod` is a PartialSteps: the same inputs and result, (y, GDNState). Module docstring; the
    prologue is dprod_forward's (one projection and one conv over [q | k_1 | v_1 | the carrying heads' k_j, v_j]), the rest is split by head."""
    from ced.model import gdn as G

    _refuse_call(layer, ctx, valid, keep, arms, x_write, ccr, x_read)
    B, L, _ = x.shape
    cfg, steps = layer.cfg, layer.dprod
    hk, hv, dk, dv, K = cfg.n_k_heads, cfg.n_v_heads, cfg.k_head_dim, cfg.v_head_dim, cfg.conv_kernel
    hk_c, hv_c = steps.hk_c, steps.hv_c
    n_x = steps.n_extra
    n_h = n_x + 1
    u_all = torch.cat([layer.in_proj_qkv(x), steps.in_proj_kv(x)], dim=-1)  # [q | k_1 | v_1 | k_2 | v_2 | ...] (the extras: carrying heads)
    w_all = torch.cat([layer.conv_weight, steps.conv_weight], dim=0)
    if state is not None and state.conv_input.shape[-1] != u_all.shape[-1]:
        raise ValueError(f"this GDN cache holds {state.conv_input.shape[-1]} conv channels; a layer with Gated DeltaProduct "
                         f"n_h {n_h} on {hv_c} of its {hv} value heads carries {u_all.shape[-1]} (its own and the extra steps')")
    use_fused = cfg.kernel in ("fla", "ced") and state is None and ctx.cu_seqlens is not None and G.fla_supported(x)
    kernel_dtype = (torch.bfloat16 if use_fused and cfg.input_dtype == "bfloat16" and u_all.dtype == torch.bfloat16 else None)
    rest_heads = hv + n_x * (hk_c + hv_c)  # the prep's value slots: v_1 (every head), then k_j, v_j of every extra step (carrying heads)
    if layer._fused_prep and state is None and ctx.cu_seqlens is not None and G.fla_supported(x) and L >= K - 1:
        q, k1, rest = G.qkv_prep(u_all, ctx.doc, w_all, hk, rest_heads, dk, out_dtype=kernel_dtype)
        rest = rest.reshape(B, L, rest_heads * dk)
        conv_input, conv_doc = u_all[:, L - (K - 1):], ctx.doc[:, L - (K - 1):]
        unit = _unit  # the extra keys leave the prep as SiLU(conv) in its output dtype: their norm in float32
    else:
        hist, hist_doc = (None, None) if state is None else (state.conv_input, state.conv_doc)
        u, conv_input, conv_doc = G.causal_doc_conv(u_all, ctx.doc, w_all, hist, hist_doc)
        u = F.silu(u)
        q, k1, rest = torch.split(u, [hk * dk, hk * dk, u.shape[-1] - 2 * hk * dk], dim=-1)
        q = G.l2_normalize(q.reshape(B, L, hk, dk)) * dk**-0.5
        k1 = G.l2_normalize(k1.reshape(B, L, hk, dk))
        unit = G.l2_normalize
    parts = torch.split(rest, [hv * dv] + [hk_c * dk, hv_c * dv] * n_x, dim=-1)
    v1 = parts[0].reshape(B, L, hv, dv)
    xk = [unit(parts[1 + 2 * j].reshape(B, L, hk_c, dk)) for j in range(n_x)]  # the extra steps' keys (carrying heads)
    xv = [parts[2 + 2 * j].reshape(B, L, hv_c, dv) for j in range(n_x)]  # and values
    g, beta1 = layer.gates(x)
    xb = list((BETA_MAX * torch.sigmoid(steps.in_proj_b(x))).reshape(B, L, n_x, hv_c).unbind(2))
    rv, rk, rvp, rkp = steps.runs_v, steps.runs_k, steps.runs_v_plain, steps.runs_k_plain
    qc, kc = take(q, rk, 2), torch.stack([take(k1, rk, 2)] + xk, dim=2)  # the carrying heads: (B, L, n_h, H_k', d_k) ...
    vc = torch.stack([take(v1, rv, 2)] + xv, dim=2)
    bc = torch.stack([BETA_MAX * take(beta1, rv, 2)] + xb, dim=2)  # step 1 of a carrying head is 2 sigmoid(b), as the Gated DeltaProduct form's
    gc = take(g, rv, 2)
    qp, kp, vp, gp, bp = take(q, rkp, 2), take(k1, rkp, 2), take(v1, rvp, 2), take(g, rvp, 2), take(beta1, rvp, 2)  # the others: GDN's
    if kernel_dtype is not None:
        qc, kc, vc, bc = qc.to(kernel_dtype), kc.to(kernel_dtype), vc.to(kernel_dtype), bc.to(kernel_dtype)
        qp, kp, vp, bp = qp.to(kernel_dtype), kp.to(kernel_dtype), vp.to(kernel_dtype), bp.to(kernel_dtype)
    S0c = S0p = None
    if not use_fused:
        S_all = x.new_zeros(B, hv, dk, dv) if state is None else state.S
        S0c, S0p = take(S_all, rv, 1), take(S_all, rvp, 1)
    o_c, S_c = _steps_rule(layer, qc, kc, vc, gc, bc, ctx, S0c, use_fused)
    o_p, S_p = _plain_rule(layer, qp, kp, vp, gp, bp, ctx, S0p, use_fused)
    o = merge(o_c, rv, o_p, rvp, 2)
    S = None if S_c is None else merge(S_c, rv, S_p, rvp, 1)
    z = layer.in_proj_z(x).view(B, L, hv, dv)
    y = layer.out_proj(layer.norm(o, z).flatten(2))
    return y, G.GDNState(conv_input=conv_input, conv_doc=conv_doc, S=S)


def _steps_rule(layer, q, k, v, g, beta, ctx, S0, use_fused):
    """The n_h-step recurrence over the carrying heads (q (B, L, H_k', d_k), k (B, L, n_h, H_k', d_k), v (B, L, n_h, H_v', d_v), g (B, L, H_v'),
    beta (B, L, n_h, H_v')): the fused n_h T interleave (one kernel call, boundaries x n_h) or the reference (the chunked rule on the interleave,
    or the loop); dprod_forward's statements for those routes. S0: the carrying heads' state for the reference routes, None on the fused one.
    Returns (o (B, L, H_v', d_v), the final state or None)."""
    from ced.model import gdn as G
    from ced.model.gdn_chunked import chunked_delta_rule

    cfg = layer.cfg
    n_h = k.shape[2]
    if use_fused:
        qi, ki, vi, gi, bi, _ = interleave(q, k, v, g, beta)
        fused = G.fla_delta_rule if cfg.kernel == "fla" else G.ced_delta_rule
        o = last_substep(_fused_call(fused, qi, ki, vi, gi, bi, ctx.cu_seqlens, ctx.cu_seqlens_cpu, n_h, **layer._ced_kw), n_h)
        return o, None  # training / scoring forward: the final state is not needed
    rep = v.shape[3] // k.shape[3]
    q_r, k_r = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=3)
    if cfg.kernel in ("chunked", "fla", "ced") and q.shape[1] > 1:
        qi, ki, vi, gi, bi, si = interleave(q_r, k_r, v, g, beta, ctx.is_doc_start)
        oi, S = chunked_delta_rule(qi, ki, vi, gi, bi, si, S0, cfg.chunk_size)
        return last_substep(oi, n_h), S
    return dprod_recurrence(q_r, k_r, v, g, beta, ctx.is_doc_start, S0)


def _plain_rule(layer, q, k, v, g, beta, ctx, S0, use_fused):
    """The plain GDN step over the heads that do NOT carry the extra steps: the calls GatedDeltaNet.forward makes for them, one rank-one write a
    token with beta = sigmoid(b), q (B, L, H_k'', d_k), k likewise, v (B, L, H_v'', d_v), g and beta (B, L, H_v''). S0: those heads' state for the
    reference routes, None on the fused one."""
    from ced.model import gdn as G
    from ced.model.gdn_chunked import chunked_delta_rule

    cfg = layer.cfg
    if use_fused:  # fla (and ours, which computes fla's function) handles grouped value heads itself; contiguous inputs, as the layer's
        fused = G.fla_delta_rule if cfg.kernel == "fla" else G.ced_delta_rule
        o = fused(q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(), ctx.cu_seqlens, ctx.cu_seqlens_cpu,
                  **layer._ced_kw)
        return o, None
    rep = v.shape[2] // k.shape[2]
    q_r, k_r = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
    if cfg.kernel in ("chunked", "fla", "ced") and q.shape[1] > 1:
        return chunked_delta_rule(q_r, k_r, v, g, beta, ctx.is_doc_start, S0, cfg.chunk_size)
    return G.delta_rule_recurrence(q_r, k_r, v, g, beta, ctx.is_doc_start, S0)
