"""The causal encoder: a 3:1 hybrid of Gated DeltaNet and global attention layers with Peri-LN.

Every token passes through the encoder exactly once. The non-looped control is the same stack
with a different depth and no decoder. The second-pass control (ced/model/second_pass.py) runs
the same layers twice: once normally (keeping what a second pass reads) and once in "zero-write"
mode at chosen positions.
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from ced.config import AttentionConfig, ModelConfig
from ced.model import act_checkpoint, attn_res, gdn_arms, recur_depth
from ced.model.attention import (CAPPED_MASKS_KEY, CAPPED_ROTATION_KEY, FLASH_BACKENDS, FLASH_VARLEN_KEY, ROPE_POS_KEY,
                                 GatedAttention, KeySet, capped_rotation, mask_cap, rope_cap)
from ced.model.canon import CANON_INITS, CanonLayer, init_canon_taps
from ced.model.gdn import GatedDeltaNet, GDNPass1, GDNState
from ced.model.mlp import dense_ffn
from ced.model.norms import RMSNorm, compute_dtype, round_stream
from ced.model.sequence import SeqContext


# Per-layer residual scalars with x0 re-injection (EncoderConfig.layer_scalars). Every form computes x <- a_i x + b_i
# x0 + f_i(x) in every layer; the forms differ in how a and b follow from the stored tensors (layer_scalar_terms
# below, the one place that says so). The sum is part of the layer's own residual add (EncoderLayer.forward calls
# apply_layer_scalars).
LAYER_SCALARS = ("none", "skip_x0", "skip_x0_span", "x0", "skip")
# Every tensor a form may store on the Encoder: (n_layers,) each, zeros at the start. layer_scalar_parameters(form)
# says which of them a form has.
LAYER_SCALAR_PARAMETERS = ("skip_gain", "x0_gain")
# "skip_x0_span": the stored u is read clamped to +-SPAN_CLAMP (tanh(3) = 0.99505), and the trainer puts the stored
# value back into that range after every optimizer step (Encoder.project_layer_scalars).
SPAN_CLAMP = 3.0


def layer_scalar_parameters(mode: str) -> tuple[str, ...]:
    """The names of the tensors the form `mode` stores on the Encoder, in state-dict order: () without the scalars;
    both vectors for "skip_x0" (skip_gain holds a - 1) and "skip_x0_span" (skip_gain holds u_1 .. u_L); x0_gain alone
    for "x0", where a is 1 and nothing is stored for it; skip_gain alone (a - 1) for "skip", where b is 0 and nothing
    is stored for it."""
    assert mode in LAYER_SCALARS, f"encoder.layer_scalars {mode!r}: one of {LAYER_SCALARS}"
    return {"none": (), "x0": ("x0_gain",), "skip": ("skip_gain",)}.get(mode, LAYER_SCALAR_PARAMETERS)


def layer_scalar_counts(mode: str, n_layers: int) -> dict:
    """What a form adds and what can leave its start, for the checks that count them: "parameters" (stored values),
    "a_free" and "b_free" (how many of the n_layers a and b a run can move away from 1 and 0). In the two bounded forms
    b_0 is fixed at 0; in "x0" every a is fixed at 1; in "skip" every b is fixed at 0."""
    names = layer_scalar_parameters(mode)
    fixed_b0 = mode in ("skip_x0_span", "x0")
    return {"parameters": len(names) * n_layers, "a_free": n_layers if "skip_gain" in names else 0,
            "b_free": (n_layers - fixed_b0) if "x0_gain" in names else 0}


def layer_scalar_terms(mode: str, skip_gain: torch.Tensor | None, x0_gain: torch.Tensor | None, span: float = 4.0
                       ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """(a - 1, b) of every layer from the stored tensors, per form: two (n_layers,) tensors, bottom first, in the
    stored dtype, with autograd's gradients to the stored tensors. The ONE place that says what a form's a and b are:
    Encoder._stack, the decode engine, the GDN diagnostic and the logs all read it. a - 1 is None where a is 1 exactly
    and no product with x is made (the form "x0"); b is None where b is 0 exactly, no product with x0 is made and x0
    is not read at all (the form "skip").

        "skip_x0" a_i - 1 = skip_gain[i], b_i = x0_gain[i]: the stored tensors themselves (the same
                        objects, so the arm as run is the arm as run, operation for operation). "skip_x0_span" skip_gain
                        stores u_1 .. u_L (L = n_layers). S_0 = 0, S_k = (ln(span) / 2) tanh(clamp(u_k, +-3)), a_i =
                        exp(S_{i+1} - S_i), so a_i - 1 = expm1(S_{i+1} - S_i). The product of a over layers i .. k is
                        exp(S_{k+1} - S_i): every |S| is at most 0.99505 ln(span) / 2, so every such product is inside
                        (span^-0.99505, span^0.99505), strictly inside (1 / span, span), whatever the stored values and
                        the depth. b_0 = 0, b_i = x0_gain[i] for i >= 1. "x0" a_i = 1 (None here), b_0 = 0, b_i =
                        x0_gain[i] for i >= 1. "skip" a_i - 1 = skip_gain[i] (the stored tensor itself, as in
                        "skip_x0"), b_i = 0 (None here): the arm "a only". There is no x0_gain. a_0 is free: with b_0 =
                        0 it is one parameter.

    At the start (zeros) every form gives a - 1 = 0 and b = 0 exactly: tanh(0) = 0 and expm1(0) = 0 as floats.
    b_0 = 0 in the two bounded forms: layer 0's input is x0 itself, so a_0 and b_0 would be one parameter and a bound
    on a_0 alone would bound nothing. The stored x0_gain keeps its shape (n_layers,); its entry 0 is not read (a
    constant zero takes its place), so it gets a gradient of exactly 0 and AdamW without weight decay never moves it.
    The clamp of u is on the VALUE only: u is read as clamp(u) with the gradient passed straight through (d/du = 1,
    also at and beyond +-3). torch.clamp's own gradient is 0 at the bound itself (measured, torch 2.14: at u = 3.0
    exactly), and the trainer's projection leaves a pushed u exactly there, where a zero gradient would hold it for
    good. With the pass-through a u on the bound has the tanh's gradient there (0.0099 of the upstream one, and AdamW
    normalises its size) and moves back as soon as the gradient's sign says so."""
    if mode == "skip_x0":
        return skip_gain, x0_gain
    if mode == "skip":
        return skip_gain, None
    assert mode in ("skip_x0_span", "x0"), f"encoder.layer_scalars {mode!r} has no scalars (one of {LAYER_SCALARS[1:]})"
    b = torch.cat([x0_gain.new_zeros(1), x0_gain[1:]])
    if mode == "x0":
        return None, b
    assert span > 1.0 and math.isfinite(span), f"encoder.layer_scalars_span {span!r}: a finite width above 1"
    u = skip_gain.detach().clamp(-SPAN_CLAMP, SPAN_CLAMP) + (skip_gain - skip_gain.detach())  # clamp(u), gradient 1
    cum = (0.5 * math.log(span)) * torch.tanh(u)  # S_1 .. S_L
    return torch.expm1(cum - torch.cat([cum.new_zeros(1), cum[:-1]])), b  # S_{i+1} - S_i with S_0 = 0


def apply_layer_scalars(out: torch.Tensor, x: torch.Tensor, x0: torch.Tensor | None, skip_gain: torch.Tensor | None,
                        x0_gain: torch.Tensor | None) -> torch.Tensor:
    """One layer's residual scalars (EncoderConfig.layer_scalars), the last term of the layer's own residual add: from
    the layer's input x, the sum out = x + m + p of its two residual adds (m the mixer branch computed from x, p the
    MLP branch computed from x + m, so out - x = f(x)) and the stack's input x0 to

        a x + b x0 + f(x)  =  out + ((a - 1) x + b x0),      a - 1 = skip_gain, b = x0_gain (0-dim tensors).

    skip_gain and x0_gain are one layer's entries of layer_scalar_terms: for "skip_x0" the stored values, for the other
    forms what that function makes of them. skip_gain None: a is 1 exactly (the form "x0") and the sum is
    out + b x0, with no product on x. x0_gain None: b is 0 exactly (the form "skip"), the sum is out + (a - 1) x and
    x0 is not read (it may be None).

    This is the definition, the only one. EncoderLayer.forward calls it as the last operation of the layer, so the
    reference and the fast paths, training, the periodic eval, scoring and cached generation all run it INSIDE the
    layer call: in a compiled layer (compile "all") it is part of the layer's graph, fused with the layer's last
    residual add, and there is no pass over the stream after the layer. The decode engine, which computes a layer's
    arithmetic itself, calls it on its own two residual adds. Until 2026-10-13 it was called by Encoder._stack on the
    layer's returned stream, outside the layer and its compiled graph: the same operations in the same order, so in
    eager code the values and the gradients are those of the runs trained that way, bit for bit
    (tests/test_resid_fuse.py).

    In at least float32 (norms.compute_dtype): the scalars are float32 parameters and are read as they are stored; a
    stream in a narrower dtype (a bfloat16 engine) is widened for the sum and the result is cast back, so a - 1 and b
    are never rounded to bfloat16. Under bf16 autocast the stream is float32 already and nothing is cast. A float64
    stream stays float64 (exact tests).

    At skip_gain = x0_gain = 0 the products are zeros and the sum is `out` as it is (for finite x and x0): the model
    without the key, bit for bit. Plain tensor operations: the gradients are autograd's (d/d skip_gain = <g, x>,
    d/d x0_gain = <g, x0>, g to out unchanged, and g (a - 1) and g b more to x and x0)."""
    dtype = compute_dtype(out)
    extra = None
    if skip_gain is not None:
        extra = x.to(dtype) * skip_gain.to(dtype)
    if x0_gain is not None:
        injected = x0.to(dtype) * x0_gain.to(dtype)
        extra = injected if extra is None else extra + injected
    return out if extra is None else (out.to(dtype) + extra).to(out.dtype)


def recur_input_mix(e: torch.Tensor, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The scalar input mix of a looped middle (LoopConfig.recur_layers with injection "mix"; Encoder.forward_recur):
    what a pass after the first reads,

        z = w e + (1 - w) h

    e the first pass's input to the middle (the prelude's output), h the output of the pass before, w ONE learned
    scalar (0-dim; Encoder.recur_mix, LoopConfig.recur_mix_init = 0.99 at the start).

    This is the definition, the only one: the reference path and the fast paths, training, scoring and cached
    generation all call it. It is ONE elementwise kernel over the stream, torch.lerp(h, e, w) = h + w (e - h), which
    is the line above (one read of e and of h, one write of z, no temporary; the residual scalars measured what a
    separate multiply and add over the stream cost, 1.12 x a step when it ran in every layer: ced/train/trainer.py,
    "MEASURED"). torch.lerp evaluates e - (e - h) (1 - w) for w >= 0.5 and h + w (e - h) below, so:
    - at w = 1 it returns e exactly, for finite e and h (e - (e - h) * 0): a later pass then repeats the first pass
      operation for operation, and the model at T passes is the one-pass model bit for bit;
    - at w = 0 it returns h exactly: the loop without injection (injection "none");
    - elsewhere it is w e + (1 - w) h to the rounding of one multiply and two sums (tests/test_loop_grow_model.py
      compares it with that line written out).
    In at least float32 and outside autocast (norms.compute_dtype): under the runs' bfloat16 autocast the layers'
    stream is float32 already, w is a float32 parameter, and nothing is cast: the one kernel. A stream in a narrower
    dtype (a bfloat16 model) is widened for the sum and the result cast back, so w and 1 - w are never rounded to
    bfloat16. A float64 stream stays float64 (exact tests). Plain autograd: d z / d w = e - h, and the gradient
    reaches e with weight w and h with weight 1 - w (at w = 1 the pass before gets exactly zero through here)."""
    dtype = compute_dtype(h)
    with torch.autocast(device_type=h.device.type, enabled=False):
        if h.dtype == dtype and e.dtype == dtype:
            return torch.lerp(h, e, w.to(dtype))
        return torch.lerp(h.to(dtype), e.to(dtype), w.to(dtype)).to(h.dtype)


def refuse_layer_scalars(cfg: ModelConfig, what: str) -> None:
    """For code that walks the encoder's layers itself (a diagnostic, a cost harness): with the residual scalars set
    it would compute another model under this model's name. ValueError, by name."""
    if cfg.encoder.layer_scalars != "none":
        raise ValueError(f"{what} walks the encoder's layers itself and does not apply model.encoder.layer_scalars "
                         f"({cfg.encoder.layer_scalars!r}: the per-layer residual scalars): use the model's own forward, "
                         "prefill / extend, or the decode engine")


def refuse_recur_loop(cfg: ModelConfig, what: str) -> None:
    """For code that walks the encoder's layers itself (a diagnostic, a cost harness, the decode engine): with a looped
    middle (LoopConfig.recur_layers) it would run every layer once and compute another model under this model's name.
    ValueError, by name."""
    if cfg.arch == "control" and cfg.loop.recur_layers:
        raise ValueError(f"{what} walks the encoder's layers itself and does not run the looped middle "
                         f"(model.loop.recur_layers {list(cfg.loop.recur_layers)!r}): use the model's own forward or "
                         "prefill / extend")


class EncoderLayer(nn.Module):
    """Peri-LN layer: x = x + eps * Norm(mixer(Norm(x))); x = x + eps * Norm(MLP(Norm(x)))."""

    def __init__(self, kind: str, cfg: ModelConfig, residual_scale: float, attention: AttentionConfig | None = None, gdn: tuple | None = None):
        """attention: this global layer's own attention config (EncoderConfig.global_attention: the by-layer mix of
        position schemes, nope_global_layers); None: cfg.encoder.attention. gdn: (GDNConfig, carrying heads) of this GDN layer (gdn_layers / dprod_carry)."""
        super().__init__()
        d, eps = cfg.d_model, cfg.norm_eps
        self.kind = kind
        self.residual_scale = residual_scale
        # EncoderConfig.residual_dtype "bf16": the stream this layer returns (and the one between its two sublayers) is held at bfloat16 precision
        # (round_stream). A plain attribute (no tensor; never in the state dict): False for every model before the key, and the layer is the layer it was.
        self.stream_bf16 = cfg.encoder.residual_dtype == "bf16"
        # Every norm inside the layer takes EncoderConfig.block_norm_gain: "channel" (the default, a per-channel gain:
        # the layer it always was) or "scalar" (one gain per norm, shape (1,)): the four Peri-LN norms here and the GDN
        # mixer's gated output norm. The attention's QK-norm keeps per-channel gains (2026-10-15 ~).
        gain = cfg.encoder.block_norm_gain
        self.mixer_in_norm = RMSNorm(d, eps, gain=gain)
        self.mixer_out_norm = RMSNorm(d, eps, gain=gain)
        self.mlp_in_norm = RMSNorm(d, eps, gain=gain)
        self.mlp_out_norm = RMSNorm(d, eps, gain=gain)
        if kind == "gdn":
            self.mixer = GatedDeltaNet(d, gdn[0] if gdn else cfg.encoder.gdn, eps, norm_gain=gain, dprod_heads=gdn[1] if gdn else None)
        elif kind == "attention":
            self.mixer = GatedAttention(d, cfg.encoder.attention if attention is None else attention, eps)
        else:
            raise ValueError(kind)
        self.mlp = dense_ffn(d, cfg.ffn_hidden, cfg.ffn_activation)
        # A Canon layer on the input of a global attention layer (EncoderConfig.canon_kernel; ced/model/canon.py).
        # None without the key (the default) and in every GDN layer: no module, no parameter, no random number, and
        # the layer's code path is the one it always was. With it, the last module of the layer (the names, shapes
        # and order of every other parameter are unchanged), zeros as built here; the encoder gives the taps another
        # start when EncoderConfig.canon_init asks for one (Encoder.__init__: init_canon_taps).
        self.canon = CanonLayer(d, cfg.encoder.canon_kernel) if kind == "attention" and cfg.encoder.canon_kernel else None

    def forward(
        self, x: torch.Tensor, ctx: SeqContext, cache: GDNState | KeySet | None, keep: bool = False,
        scalars: tuple | None = None, depth: list | None = None, write: list | None = None, ccr=None,
        stream: torch.Tensor | None = None, entry: tuple | None = None, x_write: torch.Tensor | None = None,
        x_read: torch.Tensor | None = None,
    ):
        """Returns (x, new_cache), or with keep (x, new_cache, record): what a zero-write second pass reads from this
        layer, a GDNPass1 (GDN) or the layer's KeySet over the cached and current positions (global attention; the same
        object as new_cache). With a Canon layer (self.canon) the attention reads h' = canon(h), and new_cache also
        carries the Canon layer's history for the next chunk (KeySet.canon_input / canon_doc). scalars (the residual
        scalars, EncoderConfig.layer_scalars; Encoder.layer_scalar_args): this layer's (x0, a - 1, b), and the returned
        stream is a x + b x0 + f(x), the scalar terms added by apply_layer_scalars as the last part of the layer's
        residual add (inside a compiled layer's graph). None: every model without the key, and the layer is the layer
        it was. depth (Block AttnRes, EncoderConfig.attn_res "block"; ced/model/attn_res.py layer_forward): the
        finished blocks before x; x is then the layer's block's partial sum, and so is the returned x. None: every
        model without the key, and nothing below changes. Under "block_entry" (R1-entry) only a layer that opens or
        closes a block is called with `depth` (attn_res.entry_layer_forward says what x, depth and the return then
        are); a layer inside a block is called without it and runs the statements below, as in the model without the
        key. The two keys are never set together (attn_res.check_attn_res refuses the pair by name): a call takes at
        most one of scalars and depth. With depth the whole call is attn_res.layer_forward, the first statement below,
        and none of the statements after it runs; without it the statements after it are every other model's layer, the
        scalar forms' included, and nothing of the depth attention is read. write, ccr (the GDN-side arms,
        ced/model/gdn_arms.py; a GDN layer with self.gdn_arms only): write, the write mix's sources (R1-entry's
        finished blocks b_0 ..), from which the layer forms its write mix; ccr, arm CCR's record of the previous GDN
        layer. A layer with self.gdn_arms runs its mixer through gdn_arms_mixer; every other layer is the layer it was
        (nothing of either is read). stream (R1-entry with token-dependent queries, EncoderConfig.attn_res_query
        "token"; a block's opening layer only): the stream where the block before ended, from which the entry reader's
        query is formed. Under "ldm"
        (Linear Depth Memory, ced/model/ldm.py) every layer is called with depth [M] (the memory; [] in layer 0) and
        returns the stream and the memory. None / unused: every model before these keys. The GDN-side arms and these
        options are never set together (attn_res.check_attn_res refuses the pairs by name): a call takes at most one of
        ccr and stream.
        entry, x_write (the fused write mix: model.encoder.attn_res_fused with gdn_write_mix "layer", a forward with
        gradients; attn_res.EntryWalk._run_arms, ced/kernels/write_mix.py): a block's opening GDN layer takes `entry`, its
        (block input, normed input, write mix m') from the block's group read; an inner GDN layer takes its m' as `x_write`
        in place of `write`. None in every other call, and the layer is the layer it was. x_read (the wsl variant qmix on
        the fused write mix): an inner GDN layer's query mix m'_q from the same group read, with its x_write (an opening
        layer's comes as the fourth element of `entry`)."""
        assert x_read is None or x_write is not None, "m'_q from the group read comes with m' (the fused write mix under qmix)"
        if depth is not None:  # the same sublayers; each reads a softmax mix over depth instead of the running sum
            assert scalars is None, "a layer call takes the residual scalars or the finished blocks, not both (attn_res.check_attn_res)"
            assert x_write is None, "a block's opening layer takes its write mix in `entry`"
            if entry is not None:  # (fused write mix: an R1-entry block's opening GDN layer under)
                assert stream is None
                return attn_res.layer_forward(self, x, ctx, cache, depth, keep, ccr=ccr, entry=entry)
            if ccr is not None:  # (an R1-entry block's opening GDN layer under arm CCR; its write mix reads `depth` + x)
                assert stream is None, "the GDN-side arms are not built with token-dependent queries (attn_res.check_attn_res)"
                return attn_res.layer_forward(self, x, ctx, cache, depth, keep, ccr=ccr)
            return attn_res.layer_forward(self, x, ctx, cache, depth, keep, stream)
        assert entry is None, "`entry` is a block's opening layer's (it is called with `depth`)"
        if getattr(self, "mx_wire", None) is not None and self.training:  # model.fp8.chains in_norm / exit_entry
            # (a GDN layer with the arms' "layer" form wired as the producer form takes its sources or its m': mx_wire.mix_mixer)
            assert scalars is None and not keep and self.canon is None and ccr is None and (
                self.mx_wire.mix or (write is None and x_write is None and x_read is None)), \
                "the chains are wired in the plain layer call (ced/model/mx_wire.py check_chains)"
            from ced.model import mx_wire

            return mx_wire.layer_forward(self, x, ctx, cache, write, x_write, x_read)
        h = self.mixer_in_norm(x)
        if self.canon is not None:
            assert not keep, "a Canon layer is not built together with a second pass (control.check_canon)"
            assert cache is None or cache.canon_input is not None, \
                "this cache was written without a Canon layer (KeySet.canon_input is None): it cannot be continued with one"
            h, canon_input, canon_doc = self.canon(h, ctx.doc, None if cache is None else cache.canon_input,
                                                   None if cache is None else cache.canon_doc)
        if keep and self.kind == "gdn":
            mixed, new_cache, record = self.mixer(h, ctx, cache, keep=True)
        elif getattr(self, "gdn_arms", None) is not None:  # the GDN-side arms (none in every model without them)
            if x_write is None:
                mixed, new_cache = self.gdn_arms_mixer(h, ctx, cache, write, ccr)
            else:  # the fused write mix: m' from the block's group read
                assert write is None, "an inner GDN layer takes its sources or its m' from the group read, not both"
                if x_read is None:
                    mixed, new_cache = self.gdn_arms_mixer(h, ctx, cache, None, ccr, x_write=x_write)
                else:  # qmix: m'_q from the same group read
                    mixed, new_cache = self.gdn_arms_mixer(h, ctx, cache, None, ccr, x_write=x_write, x_read=x_read)
            record = new_cache
        else:
            mixed, new_cache = self.mixer(h, ctx, cache)
            record = new_cache
        if self.canon is not None:
            new_cache = new_cache.with_canon(canon_input, canon_doc)
        y = x + self.residual_scale * self.mixer_out_norm(mixed)
        if self.stream_bf16:  # residual_dtype "bf16": the stream between the two sublayers is stored at bfloat16
            y = round_stream(y)
        y = y + self.residual_scale * self.mlp_out_norm(self.mlp(self.mlp_in_norm(y)))
        if self.stream_bf16:  # (and the stream the layer returns)
            y = round_stream(y)
        if scalars is not None:  # a x + b x0 + f(x): the scalar terms of this residual add, on the layer's input x
            y = apply_layer_scalars(y, x, *scalars)
        return (y, new_cache, record) if keep else (y, new_cache)

    def gdn_arms_mixer(self, h: torch.Tensor, ctx: SeqContext, cache, write: list | None, ccr,
                       x_write: torch.Tensor | None = None, x_read: torch.Tensor | None = None, lerped: bool = False):
        """The mixer call of a GDN layer with the GDN-side arms (self.gdn_arms; ced/model/gdn_arms.py), on its normed
        input h. Arm A: m = the layer's softmax depth mix over `write` (b_0 alone in block 1), m' = mixer_in_norm(m),
        the layer's own input norm, handed to the mixer as x_write; under the wsl variant "qmix" (gdn_write_variant) also
        m'_q, the q rows' own normed mix over the same sources, as x_read (gdn_arms.write_mixes). Arm CCR: the previous
        GDN layer's record `ccr`.
        x_write given (the fused write mix; attn_res.EntryWalk._run_arms): m' as the block's group read formed it, in the
        GEMMs' dtype; `write` is then None. Under qmix x_read, m'_q from the same group read, comes with it (and only
        then). lerped (qmix's block-opening layer on the fused write mix's second round): x_write and x_read are that layer's
        x_w and x_q, formed by the group read (gdn_arms.write_inputs's `lerped`)."""
        arms = self.gdn_arms
        if x_write is not None:
            assert arms.write_mix != "none" and write is None, "a precomputed m' is arm A's, in place of its sources"
            assert arms.variant in gdn_arms.FUSED_VARIANTS, \
                "the fused write mix is built with wsl and the variants qshare, qshare_kv, qshare_bd, qmix only (gdn_arms.check_write_variant)"
            assert (x_read is not None) == (arms.variant == "qmix"), "qmix's fused layers take m'_q from the group read, and only they"
        elif arms.write_mix != "none":
            assert x_read is None, "a layer that forms its own m' forms its own m'_q"
            assert write is not None, "a GDN layer with the write mix (arm A) is called with its sources"
            # m' (and, under the wsl variant "qmix", m'_q, the q rows' own normed mix over the same sources)
            x_write, x_read = gdn_arms.write_mixes(self.mixer_in_norm, arms, write)
        if lerped:
            assert x_read is not None, "the group read forms x_w and x_q together (qmix)"
            return self.mixer(h, ctx, cache, arms=arms, x_write=x_write, ccr=ccr, x_read=x_read, lerped=True)
        if x_read is not None:
            return self.mixer(h, ctx, cache, arms=arms, x_write=x_write, ccr=ccr, x_read=x_read)
        return self.mixer(h, ctx, cache, arms=arms, x_write=x_write, ccr=ccr)

    def zero_write(
        self, x: torch.Tensor, slots, slot_index: torch.Tensor, record: GDNPass1 | KeySet, block_mask=None,
        allow: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The same layer (same weights, norms and residual scale) at C slots, reading pass 1's record
        of this layer and writing nothing (section 2, control (c)):

            GDN:              o2_t = S1_t^T q2_t       (GatedDeltaNet.zero_write_read)
            global attention: q2_t over pass-1 K/V at s <= t of t's document (GatedAttention.read)
            FFN:              per position, as usual

        x: (B, C, d) pass-2 residual stream; slots: a SlotContext (idx, doc, pos, valid);
        slot_index: (B, C) chunk-relative positions (L for empty slots).
        block_mask / allow (a global-attention layer only; the segment pass schedule, ced/model/control.py
        scheduled_passes): the mask of this read built once by the caller and shared by the block's attention layers
        of a pass, the flex BlockMask or the dense (B, C, S) mask of the reference and sdpa backends
        (GatedAttention.read). None (the default): the read builds its own, as before."""
        h = self.mixer_in_norm(x)
        if self.kind == "gdn":
            mixed = self.mixer.zero_write_read(h, slot_index, slots.doc, slots.valid, record, slot_inverse=slots.inverse)
        elif block_mask is not None or allow is not None:
            mixed = self.mixer.read(h, slots.pos, slots.idx, slots.doc, record, block_mask=block_mask, allow=allow)
        else:
            mixed = self.mixer.read(h, slots.pos, slots.idx, slots.doc, record)
        x = x + self.residual_scale * self.mixer_out_norm(mixed)
        x = x + self.residual_scale * self.mlp_out_norm(self.mlp(self.mlp_in_norm(x)))
        return x

    def zero_write_all(self, x: torch.Tensor, ctx: SeqContext, record: GDNPass1 | KeySet, block_mask=None,
                       allow: torch.Tensor | None = None) -> torch.Tensor:
        """zero_write at EVERY position of the chunk, in row order (the looped control's second pass in training and
        scoring; ced/model/control.py, "Two passes"): the same layer, the same reads of pass 1's record, nothing
        written, and nothing gathered or scattered, since the slots are the positions themselves:

            GDN:              o2_t = S1_t^T q2_t       (GatedDeltaNet.zero_write_read_all)
            global attention: q2_t over pass-1 K/V at s <= t of t's document (GatedAttention.read: the slot read
                              with every position a slot, so the same causal and document mask as pass 1, and
                              Scalable-Softmax's key count pos_t + 1)
            FFN:              per position, as usual

        x: (B, L, d) pass 2's residual stream; ctx: the chunk's sequence context (pass 1's).
        block_mask / allow: as zero_write's (a caller-built mask of the read, shared by the block's attention layers
        of a pass); None: the read builds its own, as before."""
        h = self.mixer_in_norm(x)
        if self.kind == "gdn":
            mixed = self.mixer.zero_write_read_all(h, ctx.doc, record)
        elif block_mask is not None or allow is not None:
            mixed = self.mixer.read(h, ctx.pos, ctx.idx, ctx.doc, record, block_mask=block_mask, allow=allow)
        else:
            mixed = self.mixer.read(h, ctx.pos, ctx.idx, ctx.doc, record)
        x = x + self.residual_scale * self.mixer_out_norm(mixed)
        x = x + self.residual_scale * self.mlp_out_norm(self.mlp(self.mlp_in_norm(x)))
        return x


class Encoder(nn.Module):
    """The encoder stack, run once per token, or `loops` times (the YOCO-U arm).

    Looped: the loop boundary is the same machinery as the decoder's (injection of the normalised
    embedding, hyper-connections, per-loop embedding; ced/model/loop.py), and every loop keeps its
    own per-layer caches (GDN states, global-attention KV), so loop k at position t only needs loop-k
    states of earlier positions and teacher forcing equals cached decoding.
    """

    def __init__(self, cfg: ModelConfig, residual_scale: float):
        super().__init__()
        # Every global layer's own attention config, bottom first: cfg.encoder.attention itself for all of them unless
        # the by-layer mix is set (EncoderConfig.nope_global_layers: those layers get a copy without a rotary part and
        # run the plain read, as a layer of the NoPE arm does). The mix adds no parameter and draws no random number.
        # Not with the control's second pass either (loop.pass2_gate: forward_keep below and the layers' zero_write
        # reads; control.check_nope_global_layers says why).
        assert not cfg.encoder.nope_global_layers or (cfg.arch == "control" and rope_cap(cfg.encoder.attention) is not None
                                                      and cfg.loop.pass2_gate == "none"), \
            f"encoder.nope_global_layers is built for arch 'control' with a rotary distance cap on encoder.attention and " \
            f"no second pass, not for arch {cfg.arch!r} with cap {rope_cap(cfg.encoder.attention)!r} and loop.pass2_gate " \
            f"{cfg.loop.pass2_gate!r} (control.check_nope_global_layers)"
        # Canon layers (EncoderConfig.canon_kernel; ced/model/canon.py): on the control's global layers, one pass. Not
        # with a second pass (forward_keep and the layers' zero_write reads do not run a Canon layer), and not in the
        # other arches' encoders (control.check_canon says why).
        assert not cfg.encoder.canon_kernel or (cfg.arch == "control" and cfg.loop.pass2_gate == "none"), \
            f"encoder.canon_kernel is built for arch 'control' without a second pass, not for arch {cfg.arch!r} with " \
            f"loop.pass2_gate {cfg.loop.pass2_gate!r} (control.check_canon)"
        attention, gdn_plan = iter(cfg.encoder.global_attention()), iter(zip(cfg.encoder.gdn_layers(), cfg.encoder.dprod_carry()))
        self.layers = nn.ModuleList(
            EncoderLayer(kind, cfg, residual_scale, next(attention) if kind == "attention" else None, next(gdn_plan) if kind == "gdn" else None)
            for kind in cfg.encoder.layer_kinds()
        )
        # The start of the Canon taps (EncoderConfig.canon_init; ced/model/canon.py, "The source's start"). "zeros"
        # (the default): nothing is called, read or drawn. "uniform": each Canon layer's taps from a generator of its
        # own, seeded from the run's seed and the layer's index here; the global generator is not advanced, so every
        # other parameter (drawn later, in the model's init_weights) starts at the value it has without this.
        assert cfg.encoder.canon_init in CANON_INITS and (cfg.encoder.canon_kernel or cfg.encoder.canon_init == "zeros"), \
            f"encoder.canon_init {cfg.encoder.canon_init!r} with encoder.canon_kernel {cfg.encoder.canon_kernel!r}: one of " \
            f"{CANON_INITS}, and 'zeros' without a kernel (control.check_canon)"
        if cfg.encoder.canon_kernel and cfg.encoder.canon_init != "zeros":
            init_canon_taps(self.layers, cfg.encoder.canon_init)
        self.loops = cfg.encoder.loops
        # Per-layer residual scalars with x0 re-injection (EncoderConfig.layer_scalars; apply_layer_scalars above).
        # "none": no tensor and no operation, so the state dict, the parameter order and the RNG use are the encoder's
        # as before. "skip_x0": two (n_layers,) tensors of zeros after the layers' own (no random number is drawn, and
        # init_weights touches Linear and Embedding weights only), where the stack is the stack without them bit for
        # bit. Only where it is built (control.check_layer_scalars says why): the plain control, one pass.
        # The other forms ("skip_x0_span", "x0", "skip"; layer_scalar_terms above) store the same named tensors, zeros
        # at the start too: both for the span-bounded form, x0_gain alone for "x0", skip_gain alone for "skip"
        # (layer_scalar_parameters). scalar_form is the config's value and scalar_span the span bound's width;
        # layer_scalars stays the yes / no it was.
        assert cfg.encoder.layer_scalars in LAYER_SCALARS, \
            f"encoder.layer_scalars {cfg.encoder.layer_scalars!r}: one of {LAYER_SCALARS}"
        self.scalar_form = cfg.encoder.layer_scalars
        self.scalar_span = float(cfg.encoder.layer_scalars_span)
        self.layer_scalars = cfg.encoder.layer_scalars != "none"
        if self.layer_scalars:
            assert cfg.arch == "control" and cfg.loop.pass2_gate == "none" and cfg.encoder.loops == 1, \
                f"encoder.layer_scalars is built for arch 'control' with one pass, not for arch {cfg.arch!r} with " \
                f"loop.pass2_gate {cfg.loop.pass2_gate!r} and encoder.loops {cfg.encoder.loops} (control.check_layer_scalars)"
            assert self.scalar_span > 1.0 and math.isfinite(self.scalar_span), \
                f"encoder.layer_scalars_span {cfg.encoder.layer_scalars_span!r}: a finite width above 1 (control.check_layer_scalars)"
            for name in layer_scalar_parameters(self.scalar_form):
                setattr(self, name, nn.Parameter(torch.zeros(cfg.encoder.n_layers)))
        # The loop in the published form (LoopConfig.recur_layers on arch "control"; forward_recur below). () (the
        # default): `recur` is None, no tensor and no module is added, and forward_recur is never called: the state
        # dict, the parameter order and the RNG use are the encoder's as before. With it: the looped middle is layers
        # first .. end - 1; with injection "parcae" the injection's three tensors and (recur_prelude_norm) one norm
        # gain are added AFTER the layers' own and draw no random number (init_weights touches Linear and Embedding
        # weights only), so with injection "none" the model has exactly the plain control's tensors and initial values.
        # With injection "mix": those tensors and values plus the one scalar recur_mix.
        # With injection "depth_mix" (R3; ced/model/recur_depth.py): those tensors and values plus one reader,
        # recur_depth (a pseudo-query and a key-norm gain, zeros), added AFTER the layers' own.
        self.recur = None
        if cfg.arch == "control" and cfg.loop.recur_layers:
            assert cfg.encoder.loops == 1 and not self.layer_scalars and cfg.loop.pass2_gate == "none", \
                "loop.recur_layers is not built with a looped encoder, encoder.layer_scalars or loop.pass2_gate " \
                "(control.check_recur_loop)"
            first, end = cfg.loop.recur_layers
            assert 0 <= first < end <= cfg.encoder.n_layers, f"loop.recur_layers {cfg.loop.recur_layers!r} (control.check_recur_loop)"
            self.recur = (first, end)
            self.recur_state_init = cfg.loop.recur_state_init
            self.recur_norm = self.recur_injection = self.recur_mix = self.recur_depth = None
            if cfg.loop.injection == "parcae":
                from ced.model.loop import RecurInjection

                if cfg.loop.recur_prelude_norm:
                    self.recur_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
                self.recur_injection = RecurInjection(cfg.d_model, cfg.loop.parcae_init_decay, cfg.loop.recur_inject_b)
            elif cfg.loop.injection == "mix":
                # The scalar input mix (recur_input_mix above): ONE 0-dim tensor, w, a parameter of the encoder
                # itself (so in the state dict and in the parameter order it comes before the layers' tensors, as a
                # module's own parameters do). It draws no random number, so every other tensor has the plain
                # control's name, relative order and initial value, and a plain control's state loads into this model
                # with exactly this name missing.
                self.recur_mix = nn.Parameter(torch.tensor(float(cfg.loop.recur_mix_init)))
            elif cfg.loop.injection == recur_depth.INJECTION:
                # R3, the depth mix at the loop boundary (ced/model/recur_depth.py): ONE reader for every pass, R1's
                # DepthQuery: a (d,) pseudo-query and the (d,) gain of its key norm, zeros both, so every candidate
                # starts with equal weight. A submodule registered after the layers: its two tensors come after the
                # layers' own in the state dict, and every other tensor has the plain control's name, relative order
                # and initial value (zeros only: no random number is drawn, and init_weights touches Linear and
                # Embedding weights only).
                self.recur_depth = attn_res.DepthQuery(cfg.d_model, cfg.norm_eps)
                # "fast" (every forward) or "reference" (recur_depth.reference_recur): a plain attribute, never in
                # the state dict or the config; tests and the GPU check set it.
                self.recur_depth_impl = "fast"
        # Block Attention Residuals (EncoderConfig.attn_res; ced/model/attn_res.py). "none" (the default): None, and
        # nothing is built or touched: the state dict, the parameter order and the RNG use are the encoder's as
        # before. "block": every layer gets its depth attention (layer.attn_res: a query and a key norm per sublayer,
        # zeros; the last layer also the output's); forward then runs the stack through attn_res.stack. Refused
        # wherever it is not built (attn_res.check_attn_res: the scalars above, the loops, any other arch).
        # "block_entry" (R1-entry): one reader per block, on the first layer of blocks 2 .. N and on the last layer;
        # the layers inside a block get nothing and keep the base's call.
        self.attn_res = attn_res.build(cfg, self.layers)
        # EncoderConfig.residual_dtype "bf16": the embedding stage's stream copy is held at bfloat16 precision at the stack's entry (Encoder.forward for
        # attn_res "none"; EntryWalk and entry_reference_stack for R1-entry, whose source b_0 stays float32). A plain attribute, False by default.
        self.stream_bf16 = cfg.encoder.residual_dtype == "bf16"
        # The GDN-side arms (EncoderConfig.gdn_write_mix, gdn_ccr_spacing; ced/model/gdn_arms.py). Off (the default):
        # False, and nothing is built or touched. On: each GDN layer that takes part gets layer.gdn_arms (zeros only, its
        # last submodule, after attn_res's), and the stack hands CCR's records from GDN layer to GDN layer.
        self.gdn_arms_on = gdn_arms.build(cfg, self.layers)
        # Activation checkpointing of the layers in a training forward (ced/model/act_checkpoint.py; set by the
        # trainer from TrainConfig.activation_checkpoint). A plain attribute, "none" by default: nothing changes.
        self.activation_checkpoint = "none"
        # The layers of one checkpointed block under activation_checkpoint "block": one hybrid group,
        # full_attn_interval layers (_stack_blocks). A plain int, read only in that mode.
        self.ckpt_block = cfg.encoder.full_attn_interval
        # Monitoring of the stream's size by layer (TrainConfig.log_stream_rms_every; record_stream_rms below). A plain
        # attribute, None by default: nothing is read. A list: the next call of _stack appends the RMS of the stream
        # entering each layer and of the stream after the last layer (n_layers + 1 values), under no_grad. The trainer
        # sets it for one forward and takes it back.
        self.stream_rms: list | None = None
        # Monitoring of qmix's query-mix weights (EncoderConfig.gdn_qmix_select_step; attn_res._monitor, gdn_arms.record_query_weights):
        # a plain attribute, None by default: nothing is read. A dict: the next BlockAttnRes.stack records, per GDN layer with a query
        # mix, the token-sums of its weights (under no_grad, on a monitored forward only). The trainer sets it for one forward and takes
        # it back.
        self.qmix_record: dict | None = None
        if self.loops > 1:
            from ced.model.loop import build_hyper_connections, build_injection

            loop_cfg = dataclasses.replace(cfg.loop, t_max=self.loops)
            self.inject_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.injection = build_injection(cfg.d_model, loop_cfg)
            self.loop_embedding = nn.Parameter(torch.zeros(self.loops, cfg.d_model)) if cfg.loop.loop_embedding else None
            self.hyper_connections = build_hyper_connections(cfg.d_model, loop_cfg)

    def share_capped_masks(self, ctx: SeqContext, caches: list | None) -> None:
        """Under the rotary distance cap on the flex backend, build the two BlockMasks of the global layers' own
        read once for this forward and put them in the context's per-forward memo
        (SeqContext.memo[CAPPED_MASKS_KEY]), where every global layer finds them (GatedAttention.forward) instead
        of building its own two.

        Why one pair serves every capped global layer of the stack (and every loop of a looped encoder): without a
        cache a global layer's keys are this chunk's positions (KeySet idx = ctx.idx, doc = ctx.doc, ok all True) and
        its queries are the same positions (ctx.idx, ctx.doc, ctx.pos); all capped layers are built from one
        AttentionConfig (the cap, the backend). So the masks are a function of this context and the cap alone. Checked
        here on the configs, and at every read on shapes and settings, on values too when the read runs eagerly
        (fast_attention.CappedMasks.check).

        The by-layer mix (EncoderConfig.nope_global_layers): the masks are built for the capped layers and read by
        them alone. A NoPE layer has no cap (attention.rope_cap is None without rotary dimensions): it never looks in
        the memo and runs the plain read with a mask of its own, exactly as a layer of the NoPE arm does. With every
        global layer NoPE nothing is built.

        Nothing is shared, and every layer builds its own masks as before, when: there is a cache (a later chunk: the
        key set is longer than the context); the backend is not flex; the row is no longer than cap + 1 (no pair
        beyond the cap: the layers run the plain read). The plain read (no cap) is not touched.
        The context is new in every forward (sequence_context), so the masks never outlive their batch; a layer that
        activation checkpointing runs a second time in the backward gets the same context, hence the same masks.
        Compiled layers (compile "all") take the masks' tensors as graph inputs; the build then runs through the
        compiled create_block_mask (fast_attention.block_mask_builder), as it ran compiled inside each layer."""
        ctx.memo.pop(CAPPED_MASKS_KEY, None)
        mixers = self._capped_mixers()
        if caches is not None or not mixers:
            return
        cap = rope_cap(mixers[0].cfg)
        if ctx.rope_scale != 1:
            # TST's token-scale rotary (TSTConfig.token_scale_rope): the near / far split at cap // rope_scale positions
            # (attention.mask_cap, GatedAttention._forward_token_scale); with rope_scale 1 the cap itself, as before
            assert all(rope_cap(m.cfg) == cap for m in mixers), "the capped global layers differ in their cap"
            cap = mask_cap(cap, ctx.rope_scale)
        if mixers[0].cfg.backend != "flex" or ctx.idx.shape[1] <= cap + 1:
            return
        assert all(rope_cap(m.cfg) == rope_cap(mixers[0].cfg) and m.cfg.backend == "flex" for m in mixers), \
            "the encoder's capped global layers differ in their rotary distance cap or backend: no shared masks"
        from ced.model.fast_attention import capped_block_masks

        # the metadata of a global layer's own keys without a cache (GatedAttention.forward): this chunk's positions
        own_keys = SimpleNamespace(ok=torch.ones_like(ctx.doc, dtype=torch.bool), idx=ctx.idx, doc=ctx.doc)
        tiles = mixers[0].cfg.flex_tiles
        assert all(m.cfg.flex_tiles == tiles for m in mixers), "the encoder's capped global layers differ in flex_tiles: no shared masks"
        ctx.memo[CAPPED_MASKS_KEY] = capped_block_masks(own_keys, ctx.idx, ctx.doc, ctx.pos, cap, compiled=self._layers_compiled(),
                                                        **({} if tiles == "classified" else {"tiles": tiles}))

    def share_flash_varlen(self, ctx: SeqContext, caches: list | None) -> None:
        """On attention backend "fa3" (ced/kernels/flash_capped.py) or "fa4" (ced/kernels/flash_fa4.py, the same
        metadata), build the varlen metadata of this forward's rows once and put it in the context's per-forward memo
        (SeqContext.memo[FLASH_VARLEN_KEY]), where every global layer reads it (GatedAttention._fa3): the segment
        boundaries at a fixed capacity (AttentionConfig.fa3_docs_per_row a row) and, under the rotary distance cap, the
        far call's key counts and the rows with a far key, for the cap in positions of the row (TST's token-scale rotary:
        cap // rope_scale, attention.mask_cap). Like the capped masks, a function of this context and the cap alone, so
        one build serves every global layer (the NoPE layers of the by-layer mix read the boundaries only). On any other
        backend, and with a cache (which backends "fa3" and "fa4" refuse in the layer), nothing is built and no tensor
        operation runs: the forward is the forward it was."""
        ctx.memo.pop(FLASH_VARLEN_KEY, None)
        mixers = [layer.mixer for layer in self.layers if layer.kind == "attention"]
        if not mixers or mixers[0].cfg.backend not in FLASH_BACKENDS or caches is not None:
            return
        capped = self._capped_mixers()  # also checks that every global layer is on one backend
        cap = None
        if capped:
            cap = rope_cap(capped[0].cfg)
            assert all(rope_cap(m.cfg) == cap for m in capped), "the capped global layers differ in their cap"
            if ctx.rope_scale != 1:
                cap = mask_cap(cap, ctx.rope_scale)
        from ced.kernels.flash_capped import flash_varlen

        ctx.memo[FLASH_VARLEN_KEY] = flash_varlen(ctx.is_doc_start, mixers[0].cfg.fa3_docs_per_row, cap, ctx.cu_seqlens_cpu,
                                                  backend=mixers[0].cfg.backend)

    def _capped_mixers(self) -> list:
        """The attention modules of the global layers that read under a rotary distance cap, bottom first: all of them
        in a capped model, none in a model without the cap, and in the by-layer mix (EncoderConfig.nope_global_layers)
        the layers that keep their rotary part. Read from each layer's own config: also right for a mixer class that
        predates the cap (it has none). The global layers must agree on the backend: a NoPE layer's config is a copy
        made when the model was built, so a backend set on the model config afterwards would reach only some layers."""
        mixers = [layer.mixer for layer in self.layers if layer.kind == "attention"]
        assert len({m.cfg.backend for m in mixers}) <= 1, \
            "the encoder's global layers differ in their attention backend (set it on the model config before the " \
            "model is built: the by-layer mix gives its NoPE layers their own copy of the attention config)"
        return [m for m in mixers if rope_cap(m.cfg) is not None]

    def set_gdn_alpha_floor(self, heads=(), floor: float = 0.5) -> dict:
        """EVALUATION ONLY: a floor on the forget gate of chosen GDN heads of this (loaded) encoder, alpha_t =
        max(alpha_t, floor) before the state recurrence (GatedDeltaNet.set_alpha_floor has the definition and the
        paths it reaches).

        heads: (layer, head) pairs: `layer` the index among this encoder's layers (0 .. n_layers - 1, as the GDN
        diagnostics number them: ced/eval/gdn_diag.py, gdn_wipe.py), which must be a GDN layer; `head` the value head.
        Every call sets the whole encoder: a GDN layer without a named head has its floor taken off, so
        set_gdn_alpha_floor() with no head is the model as trained (every layer's g_floor None).
        A runtime setting: no parameter, no buffer, not in the config, not in a state dict. Under a looped stack the
        layers are the same modules at every pass, so the floor holds at every pass.
        Returns what is set: {"floor": floor or None, "heads": [[layer, head], ...] sorted}."""
        by_layer: dict[int, list[int]] = {}
        for layer, head in heads:
            by_layer.setdefault(int(layer), []).append(int(head))
        for i in by_layer:
            if not 0 <= i < len(self.layers) or self.layers[i].kind != "gdn":
                kinds = [j for j, layer in enumerate(self.layers) if layer.kind == "gdn"]
                raise ValueError(f"layer {i} is not a GDN layer of this encoder (its GDN layers: {kinds})")
        done = []
        for i, layer in enumerate(self.layers):
            if layer.kind == "gdn":
                done += [[i, h] for h in layer.mixer.set_alpha_floor(by_layer.get(i, ()), floor)]
        return {"floor": float(floor) if done else None, "heads": done}

    def set_gdn_alpha_cap(self, caps=None) -> dict:
        """EVALUATION ONLY: a cap on the forget gate of chosen GDN heads of this (loaded) encoder, alpha_t = min(alpha_t,
        cap) before the state recurrence (GatedDeltaNet.set_alpha_cap has the definition and the paths it reaches).

        caps: {(layer, head): cap} (or an iterable of (layer, head, cap)): `layer` the index among this encoder's layers
        (as set_gdn_alpha_floor and the GDN diagnostics number them), which must be a GDN layer; `head` the value head;
        cap in (0, 1]. Every call sets the whole encoder: a GDN layer without a named head has its cap taken off, so
        set_gdn_alpha_cap() with no head is the model as trained (every layer's g_cap None). A runtime setting: no
        parameter, no buffer, not in the config, not in a state dict. Under a looped stack the layers are the same
        modules at every pass, so the cap holds at every pass.
        Returns what is set: {"heads": [[layer, head, cap], ...] sorted}."""
        items = list(caps.items()) if isinstance(caps, dict) else [((lh[0], lh[1]), lh[2]) for lh in (caps or ())]
        by_layer: dict[int, dict[int, float]] = {}
        for (layer, head), cap in items:
            per = by_layer.setdefault(int(layer), {})
            if int(head) in per:
                raise ValueError(f"head {layer}.{head} is named twice")
            per[int(head)] = cap
        for i in by_layer:
            if not 0 <= i < len(self.layers) or self.layers[i].kind != "gdn":
                kinds = [j for j, layer in enumerate(self.layers) if layer.kind == "gdn"]
                raise ValueError(f"layer {i} is not a GDN layer of this encoder (its GDN layers: {kinds})")
        # validate every layer's caps before any layer is changed, so a refused call leaves the encoder as it was
        for i, per in by_layer.items():
            hv = self.layers[i].mixer.cfg.n_v_heads
            for h, c in per.items():
                if not 0 <= h < hv:
                    raise ValueError(f"value head {h} of layer {i}: this layer has heads 0 .. {hv - 1}")
                if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0.0 < c <= 1.0:
                    raise ValueError(f"alpha cap {c!r} for head {i}.{h}: a number in (0, 1]")
        done = []
        for i, layer in enumerate(self.layers):
            if layer.kind == "gdn":
                done += [[i, h, c] for h, c in layer.mixer.set_alpha_cap(by_layer.get(i, {})).items()]
        return {"heads": done}

    def gdn_alpha_cap(self) -> dict:
        """The evaluation-only gate cap as it is set now: {"heads": [[layer, head, cap], ...]} (empty: the model as
        trained). The cap is read back from ln cap (exp(ln cap), rounded to 12 digits)."""
        heads = []
        for i, layer in enumerate(self.layers):
            g = getattr(layer.mixer, "g_cap", None) if layer.kind == "gdn" else None
            if g is not None:
                for h in range(g.numel()):
                    if math.isfinite(float(g[h])):
                        heads.append([i, h, round(math.exp(float(g[h])), 12)])
        return {"heads": heads}

    def gdn_alpha_floor(self) -> dict:
        """The evaluation-only gate floor as it is set now: {"heads": [[layer, head], ...], "floor": the floor, or None
        without one (the model as trained)}. A floor of 0 on a head reads back as no floor on it (it clamps nothing)."""
        heads, floors = [], set()
        for i, layer in enumerate(self.layers):
            g = getattr(layer.mixer, "g_floor", None) if layer.kind == "gdn" else None
            if g is not None:
                for h in torch.nonzero(torch.isfinite(g)).flatten().tolist():
                    heads.append([i, h])
                    floors.add(round(math.exp(float(g[h])), 12))
        assert len(floors) <= 1, f"the heads were given different floors: {sorted(floors)}"
        return {"heads": heads, "floor": floors.pop() if floors else None}

    def set_gdn_head_zero(self, heads=()) -> dict:
        """EVALUATION ONLY: zero the readout of chosen GDN heads of this (loaded) encoder, o[::, h:] = 0 before the gated output
        norm and out_proj (GatedDeltaNet.set_head_zero has the definition and the paths it reaches).

        heads: (layer, head) pairs: `layer` the index among this encoder's layers (0 .. n_layers - 1, as set_gdn_alpha_floor and the
        GDN diagnostics number them), which must be a GDN layer; `head` the value head. Every call sets the whole encoder: a GDN layer
        without a named head has its zero taken off, so set_gdn_head_zero() with no head is the model as trained (every layer's
        head_keep None). A runtime setting: no parameter, no buffer, not in the config, not in a state dict. Under a looped stack the
        layers are the same modules at every pass, so the zero holds at every pass.
        Returns what is set: {"heads": [[layer, head], ...] sorted}."""
        by_layer: dict[int, list[int]] = {}
        for layer, head in heads:
            by_layer.setdefault(int(layer), []).append(int(head))
        for i in by_layer:
            if not 0 <= i < len(self.layers) or self.layers[i].kind != "gdn":
                kinds = [j for j, layer in enumerate(self.layers) if layer.kind == "gdn"]
                raise ValueError(f"layer {i} is not a GDN layer of this encoder (its GDN layers: {kinds})")
        # validate every layer's heads before any layer is changed, so a refused call leaves the encoder as it was
        for i, hs in by_layer.items():
            hv = self.layers[i].mixer.cfg.n_v_heads
            bad = [h for h in hs if not 0 <= h < hv]
            if bad:
                raise ValueError(f"value heads {bad} of layer {i}: this layer has heads 0 .. {hv - 1}")
        done = []
        for i, layer in enumerate(self.layers):
            if layer.kind == "gdn":
                done += [[i, h] for h in layer.mixer.set_head_zero(by_layer.get(i, ()))]
        return {"heads": done}

    def gdn_head_zero(self) -> dict:
        """The evaluation-only head zero as it is set now: {"heads": [[layer, head], ...]} (empty: the model as trained)."""
        heads = []
        for i, layer in enumerate(self.layers):
            keep = getattr(layer.mixer, "head_keep", None) if layer.kind == "gdn" else None
            if keep is not None:
                heads += [[i, h] for h in range(keep.numel()) if float(keep[h]) == 0.0]
        return {"heads": heads}

    def _layers_compiled(self) -> bool:
        """Whether the layers run compiled (nn.Module.compile: the trainer's compile "all", the scoring scripts)."""
        return any(getattr(layer, "_compiled_call_impl", None) is not None for layer in self.layers)

    def share_capped_rotation(self, h: torch.Tensor, ctx: SeqContext) -> None:
        """Under the rotary distance cap, build the rotation tables of this context's positions once for this forward
        (attention.CappedRotation: cos and sin at ctx.pos, and at the one position C) and put them in the context's
        per-forward memo, where every global layer finds them for its queries and its keys (GatedAttention.forward)
        instead of computing them again, twice per layer.

        The tables depend on the positions, the cap and the attention config's rotary settings alone, and all capped
        global layers are built from one AttentionConfig: one set serves them all, with or without a cache (a chunk's
        queries and its new keys sit at ctx.pos either way) and on every backend. In the by-layer mix
        (EncoderConfig.nope_global_layers) they are built for the capped layers and read by them alone: a NoPE layer
        rotates nothing (rope.apply_rope with no rotary dimension returns its input). In the dtype the layers rotate in:
        compute_dtype of the stream h (float32 under autocast). Compiled layers take them as graph inputs; they are
        then built by a compiled function. The plain read (no cap) is not touched: its layers call rope.apply_rope.
        The context is new in every forward, so the tables never outlive their batch."""
        ctx.memo.pop(CAPPED_ROTATION_KEY, None)
        ctx.memo.pop(ROPE_POS_KEY, None)
        mixers = self._capped_mixers()
        if not mixers:
            return
        cfg = mixers[0].cfg
        same = (rope_cap(cfg), cfg.head_dim, cfg.rope_fraction, cfg.rope_theta)
        assert all((rope_cap(m.cfg), m.cfg.head_dim, m.cfg.rope_fraction, m.cfg.rope_theta) == same for m in mixers), \
            "the encoder's capped global layers differ in their rotary settings: no shared rotation tables"
        pos = ctx.pos
        if ctx.rope_scale != 1:  # TST's token-scale rotary: the tables at rope_scale x pos, the far one at the cap (tokens)
            pos = ctx.memo[ROPE_POS_KEY] = ctx.pos * ctx.rope_scale
        ctx.memo[CAPPED_ROTATION_KEY] = capped_rotation(pos, cfg, compute_dtype(h), compiled=self._layers_compiled())

    def _stack(self, h, ctx, caches, stats=None):
        if self.gdn_arms_on:  # the GDN-side arms on the plain stack (arm CCR): the same layer calls, with the records
            return self._stack_arms(h, ctx, caches, stats)
        if caches is None and act_checkpoint.block_active(self.activation_checkpoint, h):
            return self._stack_blocks(h, ctx, stats)  # one checkpoint per block: the same calls in the same order
        caches = caches if caches is not None else [None] * len(self.layers)
        new_caches = []
        # The residual scalars (layer_scalars): x0 is this call's input, the embedding output of the chunk's own
        # positions, so a cached chunk carries nothing for it. Each layer gets its (x0, a - 1, b) and adds the scalar
        # terms in its own residual add (EncoderLayer.forward: inside the layer call, its checkpoint and its compiled
        # graph). Without them no tensor is made and the layer call is the call it was.
        args = self.layer_scalar_args(h) if self.layer_scalars else None
        rms = self.stream_rms  # monitoring (None: nothing is read)
        for i, (layer, cache) in enumerate(zip(self.layers, caches)):
            if stats is not None and layer.kind == "attention":
                record_activation_max(stats, f"enc{i}", h)
            if rms is not None:
                record_stream_rms(rms, h)
            kwargs = {} if args is None else {"scalars": args[i]}  # x <- a x + b x0 + f(x), in the layer's own add
            if cache is None and act_checkpoint.active(self.activation_checkpoint, h):
                h, new_cache = act_checkpoint.run_layer(layer, h, ctx, cache, **kwargs)  # the same call, checkpointed
            else:
                h, new_cache = layer(h, ctx, cache, **kwargs)
            new_caches.append(new_cache)
        if rms is not None:
            record_stream_rms(rms, h)
        return h, new_caches

    def _stack_arms(self, h, ctx, caches, stats=None):
        """_stack with the GDN-side arms (attn_res "none": arm CCR alone; ced/model/gdn_arms.py): every layer called as
        _stack calls it, a GDN layer with layer.gdn_arms also with the previous GDN layer's CCR record (None before the
        first), whose own record is taken out of its new cache entry for the next GDN layer (CCR v2: the walk holds the last
        record written and hands it only to a reader, gdn_arms.record_in / hand_record; under "previous" that is v1's
        chain). No residual scalars
        (refused by name). Activation checkpointing (gdn_arms' docstring): "layer" runs each layer call under its own
        checkpoint (the stream kept, the record held by reference; a layer's own record is in its call's output and
        is taken out after the call), "block" each block of ckpt_block layers as one checkpointed call
        (_stack_arms_blocks); the host modes are refused with CCR (gdn_arms.check_checkpoint_mode)."""
        assert self.activation_checkpoint in gdn_arms.CCR_CHECKPOINT_MODES and not self.layer_scalars, \
            f"the GDN-side arms' CCR runs with activation checkpointing {gdn_arms.CCR_CHECKPOINT_MODES} and without the " \
            "residual scalars (gdn_arms.check_checkpoint_mode, check_gdn_arms, the trainer)"
        if caches is None and act_checkpoint.block_active(self.activation_checkpoint, h):
            return self._stack_arms_blocks(h, ctx, stats)
        caches = caches if caches is not None else [None] * len(self.layers)
        new_caches, rec = [], None
        rms = self.stream_rms
        for i, (layer, cache) in enumerate(zip(self.layers, caches)):
            if stats is not None and layer.kind == "attention":
                record_activation_max(stats, f"enc{i}", h)
            if rms is not None:
                record_stream_rms(rms, h)
            ckpt = cache is None and act_checkpoint.active(self.activation_checkpoint, h)
            if getattr(layer, "gdn_arms", None) is not None:
                rec_in = gdn_arms.record_in(layer.gdn_arms, rec)  # v2: a layer that reads none is called without it
                if ckpt:  # the same call, checkpointed: the record is an output of the call, taken out below
                    h, new_cache = act_checkpoint.run_layer(layer, h, ctx, cache, ccr=rec_in)
                else:
                    h, new_cache = layer(h, ctx, cache, ccr=rec_in)
                rec, new_cache = gdn_arms.hand_record(layer.gdn_arms, rec, new_cache)
            elif ckpt:
                h, new_cache = act_checkpoint.run_layer(layer, h, ctx, cache)
            else:
                h, new_cache = layer(h, ctx, cache)
            new_caches.append(new_cache)
        if rms is not None:
            record_stream_rms(rms, h)
        return h, new_caches

    def _stack_arms_blocks(self, h, ctx, stats=None):
        """_stack_arms under activation_checkpoint "block" in a training forward: each block of ckpt_block layers one
        checkpointed call of _block_arms, which takes the stream and the previous GDN layer's CCR record where the
        block begins and returns them where it ends (the record of the block's last GDN layer is an output of the
        block's call and the input of the next block's). The layer calls, their order and the monitoring reads (in a
        block's first forward only) are _stack_arms', so the outputs and the gradients are its own bit for bit."""
        S, n = self.ckpt_block, len(self.layers)
        assert S > 0 and n % S == 0, f"activation_checkpoint 'block': {n} layers are not a whole number of blocks of {S}"
        rms = self.stream_rms
        new_caches, rec = [], None
        for i0 in range(0, n, S):
            first = act_checkpoint.FirstForward()
            h, rec, block_caches = act_checkpoint.run_block(self._block_arms, h, rec, ctx, i0, stats, rms, first)
            first.on = False  # the recomputation in the backward records nothing
            new_caches += block_caches
        if rms is not None:
            record_stream_rms(rms, h)
        return h, new_caches

    def _block_arms(self, h, rec, ctx, i0: int, stats, rms, first):
        """Layers i0 .. i0 + ckpt_block - 1 as _stack_arms calls them (no cache), from the stream h and the CCR record
        rec; returns (the stream after them, the record of their last GDN layer (rec itself if none), their new cache
        entries). The function a checkpointed block runs, in the forward and again in the backward (which changes
        nothing outside it); the monitoring reads only while first.on."""
        caches = []
        for i in range(i0, i0 + self.ckpt_block):
            layer = self.layers[i]
            if first.on:
                if stats is not None and layer.kind == "attention":
                    record_activation_max(stats, f"enc{i}", h)
                if rms is not None:
                    record_stream_rms(rms, h)
            if getattr(layer, "gdn_arms", None) is not None:
                h, new_cache = layer(h, ctx, None, ccr=gdn_arms.record_in(layer.gdn_arms, rec))
                rec, new_cache = gdn_arms.hand_record(layer.gdn_arms, rec, new_cache)
            else:
                h, new_cache = layer(h, ctx, None)
            caches.append(new_cache)
        return h, rec, caches

    def _stack_blocks(self, h, ctx, stats=None):
        """_stack under activation_checkpoint "block" (ced/model/act_checkpoint.py) in a training forward: the layers
        in blocks of ckpt_block (one hybrid group), each block one checkpointed call of _block that keeps the stream
        where the block begins. The layer calls, their order and the monitoring reads (in a block's first forward
        only) are _stack's, so the outputs, the new caches and the gradients are _stack's bit for bit."""
        S, n = self.ckpt_block, len(self.layers)
        assert S > 0 and n % S == 0, f"activation_checkpoint 'block': {n} layers are not a whole number of blocks of {S}"
        assert not self.layer_scalars, "activation_checkpoint 'block' is not built with encoder.layer_scalars (the trainer refuses it)"
        rms = self.stream_rms
        new_caches = []
        for i0 in range(0, n, S):
            first = act_checkpoint.FirstForward()
            h, block_caches = act_checkpoint.run_block(self._block, h, ctx, i0, stats, rms, first)
            first.on = False  # the recomputation in the backward records nothing
            new_caches += block_caches
        if rms is not None:
            record_stream_rms(rms, h)
        return h, new_caches

    def _block(self, h, ctx, i0: int, stats, rms, first):
        """Layers i0 .. i0 + ckpt_block - 1 on the stream h, as _stack calls them (no cache); returns (the stream after
        them, their new cache entries). The function a checkpointed block runs, in the forward and again in the
        backward; the monitoring reads only while first.on."""
        caches = []
        for i in range(i0, i0 + self.ckpt_block):
            layer = self.layers[i]
            if first.on:
                if stats is not None and layer.kind == "attention":
                    record_activation_max(stats, f"enc{i}", h)
                if rms is not None:
                    record_stream_rms(rms, h)
            h, new_cache = layer(h, ctx, None)
            caches.append(new_cache)
        return h, caches

    def layer_scalar_terms(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """(a - 1, b) of every layer of this encoder's form from its stored tensors (the module's layer_scalar_terms):
        what the layers, the decode engine and the GDN diagnostic hand to apply_layer_scalars, entry by entry. a - 1
        is None in the form "x0", b is None in the form "skip"."""
        assert self.layer_scalars, "this encoder has no layer scalars (encoder.layer_scalars 'none')"
        return layer_scalar_terms(self.scalar_form, getattr(self, "skip_gain", None), getattr(self, "x0_gain", None),
                                  self.scalar_span)

    def layer_scalar_args(self, x0: torch.Tensor) -> list[tuple]:
        """Every layer's `scalars` argument (EncoderLayer.forward) for a stack whose input is x0, bottom first:
        (x0, a_i - 1, b_i), the last three arguments of apply_layer_scalars. a_i - 1 and b_i are 0-dim views of
        layer_scalar_terms' two vectors, taken with ONE unbind each per call (one autograd node per vector, not one
        per layer); None where the form has no such term, and x0 is None too where no layer reads it (the form
        "skip"), so nothing of it enters a layer's graph."""
        gain, b = self.layer_scalar_terms()
        n = len(self.layers)
        gains = [None] * n if gain is None else gain.unbind(0)
        bs = [None] * n if b is None else b.unbind(0)
        x0 = None if b is None else x0
        return [(x0, gains[i], bs[i]) for i in range(n)]

    def layer_scalar_values(self) -> dict | None:
        """The residual scalars as they are now, for the logs: {"a": [a_i], "b": [b_i]} per layer, bottom first: the
        EFFECTIVE a and b of x <- a x + b x0 + f(x), whatever the form stores (Python floats; one host read). For
        "skip_x0", a_i = 1 + skip_gain[i] and b_i = x0_gain[i]; for the other forms layer_scalar_terms of the stored
        values, computed here in float64 (a_i = 1.0 in every layer of "x0", b_0 = 0.0 in both). None without them.
        The span form's stored u is not logged: it follows from a (S_k = sum of ln a_i over i < k,
        u_k = atanh(2 S_k / ln span))."""
        if not self.layer_scalars:
            return None
        w = self.skip_gain.detach().double().cpu() if hasattr(self, "skip_gain") else None
        v = self.x0_gain.detach().double().cpu() if hasattr(self, "x0_gain") else None
        gain, b = layer_scalar_terms(self.scalar_form, w, v, self.scalar_span)
        n = len(self.layers)  # a is 1.0 in every layer of "x0", b is 0.0 in every layer of "skip"
        return {"a": [1.0] * n if gain is None else (1.0 + gain).tolist(), "b": [0.0] * n if b is None else b.tolist()}

    @torch.no_grad()
    def project_layer_scalars(self) -> None:
        """After an optimizer step (the trainer calls it): put the stored values back where the form reads them. Only
        "skip_x0_span" has such a range: its u is clamped in place to +-SPAN_CLAMP, so a u pushed against the bound
        does not travel into the flat part of the tanh, from where it would have to travel back before a moves again
        (the proposal's review; AdamW moves a stored value by about the LR a step whatever the gradient's size, and
        the run's summed LR is 16.6). Every other form: nothing is touched, and the step is the step it was."""
        if self.scalar_form == "skip_x0_span":
            self.skip_gain.clamp_(-SPAN_CLAMP, SPAN_CLAMP)

    def forward_keep(
        self, h: torch.Tensor, ctx: SeqContext, caches: list | None, stats: dict | None, keep_from: int
    ) -> tuple[torch.Tensor, list, list]:
        """forward for a stack whose layers keep_from .. n - 1 are read again by a zero-write second pass (the
        looped control; ced/model/control.py, "Two passes"): the same layer calls in the same order, with keep=True
        from layer keep_from on, so that those layers also return what a second pass reads. Returns (h, the new
        caches, the records of layers keep_from .. n - 1). The values of h and of the caches are forward's (keep
        changes nothing a layer computes). Activation checkpointing wraps every call, as in forward; a kept record
        is an output of its checkpointed call and stays alive until the second pass's call of that layer has gone
        backward. Under the rotary distance cap the layers' own reads take the block masks and rotation tables built
        once for this forward, as in forward (share_capped_masks, share_capped_rotation); the later passes' reads
        (EncoderLayer.zero_write, zero_write_all: GatedAttention.read) are slot reads and build their own."""
        assert self.loops == 1, "a zero-write second pass is not built for a looped encoder"
        assert not self.layer_scalars, "a zero-write second pass is not built with encoder.layer_scalars"
        assert self.activation_checkpoint != "block", "activation_checkpoint 'block' is not built for a second pass (the trainer refuses it)"
        assert self.recur is None, "a zero-write second pass is not built with loop.recur_layers"
        self.share_capped_masks(ctx, caches)
        self.share_capped_rotation(h, ctx)
        caches = caches if caches is not None else [None] * len(self.layers)
        new_caches, records = [], []
        for i, (layer, cache) in enumerate(zip(self.layers, caches)):
            if stats is not None and layer.kind == "attention":
                record_activation_max(stats, f"enc{i}", h)
            ckpt = cache is None and act_checkpoint.active(self.activation_checkpoint, h)
            if i < keep_from:
                h, new_cache = act_checkpoint.run_layer(layer, h, ctx, cache) if ckpt else layer(h, ctx, cache)
            else:
                h, new_cache, record = act_checkpoint.run_layer(layer, h, ctx, cache, keep=True) if ckpt else \
                    layer(h, ctx, cache, keep=True)
                records.append(record)
            new_caches.append(new_cache)
        return h, new_caches, records

    # -- the loop in the published form (LoopConfig.recur_layers; ced/model/control.py has the model's description) --

    def recur_cache_length(self, T: int) -> int:
        """The number of per-layer cache entries of a forward at T passes: one per layer APPLICATION, in the order
        the layers run (the prelude's, pass 1's middle, ..., pass T's middle, the coda's)."""
        first, end = self.recur
        return first + T * (end - first) + (len(self.layers) - end)

    def _segment(self, h, ctx, caches, new_caches, layers, stats, tag: str | None):
        """Run `layers` ((stack index, layer) pairs) on the stream h: the plain layer call of _stack, each with the
        next entry of `caches` (None: no cache), appending each layer's new cache entry to new_caches. tag: the
        prefix of this segment's monitoring names."""
        for i, layer in layers:
            cache = None if caches is None else caches[len(new_caches)]
            if stats is not None and layer.kind == "attention":
                record_activation_max(stats, f"{tag}enc{i}", h)
            if cache is None and act_checkpoint.active(self.activation_checkpoint, h):
                h, new_cache = act_checkpoint.run_layer(layer, h, ctx, cache)  # the same call, checkpointed
            else:
                h, new_cache = layer(h, ctx, cache)
            new_caches.append(new_cache)
        return h

    def forward_recur(self, h: torch.Tensor, ctx: SeqContext, caches: list | None, stats: dict | None, T: int
                      ) -> tuple[torch.Tensor, list]:
        """forward for a stack with a looped middle (LoopConfig.recur_layers = [first, end]) at T >= 1 passes:

            p   = Prelude(h)                       layers 0 .. first - 1, once
            e   = RMSNorm(p)                       with injection and recur_prelude_norm; else e = p
            z_0 = 0 ("zeros") or e ("prelude")     with injection; without it z_0 = p and no u is formed
            u_k = decay * z_{k-1} + dt * (B e)     with injection (loop.RecurInjection), k = 1 .. T
            z_k = Middle(u_k)                      layers first .. end - 1, the same weights at every pass
            out = Coda(z_T)                        layers end .. n - 1, once

        With the scalar input mix (injection "mix"; recur_input_mix above): u_1 = p, as without injection, and
        u_k = w p + (1 - w) z_{k-1} for k = 2 .. T (the same w and the same p at every later pass); nothing else of
        the forward without injection changes, and at T = 1 w is not read.

        With the depth mix (injection "depth_mix": R3; ced/model/recur_depth.py has the form): x0 is this call's
        input h (the embedding's output), and EVERY pass reads a softmax mix over its candidates, per position,

            u_k = sum over V_k of alpha_v v,   alpha = softmax over V_k of q^T RMSNorm(v),
            V_1 = [x0, p],   V_k = [x0, p, z_1, ..., z_{k-1}],

        with one pseudo-query q and one key norm for every pass (self.recur_depth); the coda reads z_T. The mix is
        attn_res.depth_mix, called between the layer calls (in at least float32, autocast off). With
        recur_depth_impl "reference" the whole forward is recur_depth.reference_recur. Monitoring adds every pass's
        weights, "depth_mix/pass<k>" (B, L, k + 1), formed a second time under no_grad (recur_depth.mix_weights).

        Every layer call is the plain call of forward (EncoderLayer.forward): each pass computes its own keys and
        values in its global layers and its own GDN states from ITS input over the whole chunk, and reads nothing of
        another pass; the only thing a pass takes from the one before is the stream z_{k-1}, position by position.
        So a position's output depends on earlier positions only through that pass's own K/V and states, and the
        cache of a chunk is one entry per layer application: `caches` and the returned list have
        recur_cache_length(T) entries in running order (a cached sequence keeps its T: ControlCache.T).
        The rotary cap's masks and rotation tables are built once for the forward and serve every pass (they depend
        on the positions alone). Activation checkpointing wraps every layer call, as in forward. Monitoring: the
        activation maxima of the prelude, the LAST pass and the coda under forward's names ("enc<i>"), of an earlier
        pass k under "pass<k>.enc<i>", and at T >= 2 "loop_usage", the mean of ||z_T - z_{T-1}|| / ||z_T||."""
        assert self.recur is not None, "forward_recur is for a stack with loop.recur_layers"
        assert type(T) is int and T >= 1, f"T={T!r}: a pass count >= 1"
        assert self.activation_checkpoint != "block", "activation_checkpoint 'block' is not built for loop.recur_layers (the trainer refuses it)"
        first, end = self.recur
        n = self.recur_cache_length(T)
        assert caches is None or len(caches) == n, \
            f"this cache holds {len(caches)} layer entries; a forward at T = {T} needs {n} (a cached sequence keeps its T)"
        self.share_capped_masks(ctx, caches)
        self.share_capped_rotation(h, ctx)
        depth = self.recur_depth  # the depth mix's reader (injection "depth_mix": R3), or None
        if depth is not None and self.recur_depth_impl != "fast":
            assert self.recur_depth_impl == "reference", self.recur_depth_impl
            record = None if stats is None else []  # monitoring: the reference's own weights, under the same names
            out = recur_depth.reference_recur(self, h, ctx, caches, T, record)
            for r in record or ():
                stats[f"{recur_depth.STATS_PREFIX}pass{r['pass']}"] = r["alpha"].detach()
            return out
        layers = list(enumerate(self.layers))
        new_caches: list = []
        x0 = h  # the embedding's output: a candidate of the depth mix at every pass (not read by any other form)
        h = self._segment(h, ctx, caches, new_caches, layers[:first], stats, "")
        if depth is not None:
            # R3: every pass reads the softmax mix over [x0, p, z_1, ..., z_{k-1}]; the coda reads z_T.
            sources = [x0, h]  # V_1 = [x0, p]
            z = z_prev = None
            for k in range(T):
                u = attn_res.depth_mix(sources, depth)  # u_k: in float32, between the layer calls
                if stats is not None:  # the pass's weights, per position (a second, eager copy under no_grad)
                    stats[f"{recur_depth.STATS_PREFIX}pass{k + 1}"] = recur_depth.mix_weights(sources, depth)
                z_prev = z
                z = self._segment(u, ctx, caches, new_caches, layers[first:end], stats, "" if k == T - 1 else f"pass{k + 1}.")
                sources = sources + [z]  # V_{k+1}: the pass's output is a candidate of every later pass
            if stats is not None and T >= 2:
                with torch.no_grad():
                    d = (z - z_prev).float().norm(dim=-1) / z.float().norm(dim=-1).clamp_min(1e-12)
                    stats["loop_usage"] = d.mean()
            h = self._segment(z, ctx, caches, new_caches, layers[end:], stats, "")
            return h, new_caches
        inject = self.recur_injection
        mix = self.recur_mix  # the scalar input mix's w (injection "mix"), or None
        if inject is None:
            e, z = (None if mix is None else h), h  # (the mix reads the prelude's output itself: no norm)
        else:
            e = h if self.recur_norm is None else self.recur_norm(h)
            z = e if self.recur_state_init == "prelude" else None  # None: zeros (RecurInjection.forward)
        z_prev = None
        for k in range(T):
            if inject is not None:
                u = inject(z, e)
            elif mix is not None and k > 0:
                u = recur_input_mix(e, z, mix)  # w e + (1 - w) z_{k-1}: one kernel, in float32
            else:
                u = z
            z_prev = z
            z = self._segment(u, ctx, caches, new_caches, layers[first:end], stats, "" if k == T - 1 else f"pass{k + 1}.")
        if stats is not None and T >= 2:
            with torch.no_grad():
                d = (z - z_prev).float().norm(dim=-1) / z.float().norm(dim=-1).clamp_min(1e-12)
                stats["loop_usage"] = d.mean()
        h = self._segment(z, ctx, caches, new_caches, layers[end:], stats, "")
        return h, new_caches

    def forward(
        self, h: torch.Tensor, ctx: SeqContext, caches: list | None = None, stats: dict | None = None
    ) -> tuple[torch.Tensor, list]:
        """Run all layers on a chunk. caches[i] is layer i's state from earlier chunks (or None);
        for a looped encoder, caches[k][i] is loop k's. stats (monitoring, optional): a dict that
        receives the max |activation| at the input of every global-attention layer (last loop)."""
        assert self.recur is None, "a stack with loop.recur_layers runs through forward_recur (it takes the pass count)"
        self.share_capped_masks(ctx, caches)
        self.share_capped_rotation(h, ctx)
        self.share_flash_varlen(ctx, caches)
        if self.attn_res is not None:  # Block AttnRes: the same layers, the stream formed by attention over depth
            return self.attn_res.stack(self, h, ctx, caches, stats)
        if self.stream_bf16:  # residual_dtype "bf16", the plain stack: the embedding output is the stream's first stored tensor
            h = round_stream(h)
        if self.loops == 1:
            return self._stack(h, ctx, caches, stats)
        assert self.activation_checkpoint != "block", "activation_checkpoint 'block' is not built for a looped encoder (the trainer refuses it)"
        e = self.inject_norm(h)
        hc = self.hyper_connections
        Y = hc.start(e)
        new_caches = []
        for k in range(self.loops):
            z_in = hc.read(Y, k)
            u = self.injection(z_in, e)
            if self.loop_embedding is not None:
                u = u + self.loop_embedding[k]
            u, loop_caches = self._stack(u, ctx, caches[k] if caches is not None else None, stats)
            new_caches.append(loop_caches)
            Y = hc.write(Y, z_in, u, k)
        return hc.finish(Y), new_caches


@torch.no_grad()
def record_stream_rms(out: list, h: torch.Tensor) -> None:
    """Monitoring (TrainConfig.log_stream_rms_every): the root mean square of the residual stream h over tokens and
    dimensions, sqrt(mean(h^2)) (the "rms" of scripts/resid_stream_rms.py), appended to `out` as a 0-dim tensor (no
    host sync). Called between layers, outside any compiled layer and outside any checkpointed call, on the detached
    stream: it is never part of the autograd graph, writes to no tensor of the model and draws no random number. The
    norm is taken without autocast, in float32 (the stream's own dtype under bf16 autocast, where the norms return
    float32; a float64 stream keeps float64): one reduction, no copy of the stream."""
    v = h.detach()
    if v.dtype not in (torch.float32, torch.float64):
        v = v.float()
    with torch.autocast(v.device.type, enabled=False):
        out.append(torch.linalg.vector_norm(v) / math.sqrt(v.numel()))


@torch.no_grad()
def record_activation_max(stats: dict, name: str, h: torch.Tensor, valid: torch.Tensor | None = None) -> None:
    """Monitoring: the max |activation| of the residual stream h at the input of a global
    (softmax-attention) layer, stored as stats["act_max/<name>"], a 0-dim float32 tensor (no host
    sync). Called between layers, outside any compiled layer, and never part of the autograd graph.
    Names: "enc<i>" (encoder layer i), "dec<i>.loop<k>" (CED v2 decoder layer i in loop k = 1..T:
    every decoder layer attends over the memory), "pass2.enc<i>" (control (c)'s pass 2 through
    encoder layer i).

    valid (B, C) bool, optional: only these positions count (decoder slots: empty slots excluded)."""
    a = h.detach().abs()
    if valid is not None:
        a = torch.where(valid[..., None], a, torch.zeros((), dtype=a.dtype, device=a.device))
    stats[f"act_max/{name}"] = a.amax().float()
