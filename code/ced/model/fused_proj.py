"""One GEMM a shared input: the projections of a layer that read the same input run as ONE GEMM in a training forward
(ModelConfig.fused_projections; item other.concat, rank 1 of the B200 step audit).

THE GROUPS (fused_projections names a subset; () by default: nothing here runs and no module is touched):

    "gate_up"  SwiGLU     w_gate | w_up                 N = 2 x ffn_hidden        (4B: 14,336)
    "qkv"      attention  q_proj | k_proj | v_proj      N = q (with the elementwise gate's rows) + 2 x KV width (4B: 10,240)
    "qkv_z"    GDN        in_proj_qkv | in_proj_z       N = conv channels + value width (4B: 7,680)
    "ab"       GDN        in_proj_b | in_proj_a         N = 2 x value heads (4B: 40; kept in high precision: one bf16 GEMM)

The parts are listed in model order (named_parameters), which is the order of their rows in the concatenated weight and
of their columns in the GEMM's output. The output is ONE tensor y (tokens x N); each projection's output is a view of it
(torch.split along the last dimension: no copy), consumed where the separate projection's output was.

THE PARAMETERS ARE TODAY'S. Every part keeps its own nn.Parameter (its name, shape, state-dict key, optimizer group, Muon's
orthogonalisation of that one matrix, NorMuon's rows, ZeRO-1's ownership of that one matrix, the checkpoint): nothing about
them changes, so a checkpoint of either form loads into the other and the optimizer's update is the separate forms' update
of the same gradients. The alternative, one concatenated nn.Parameter a group with the load and save mapping today's names,
would put Muon on the concatenation as ONE matrix (another update) unless every matrix optimizer, ZeRO-1's layout, the
weight-decay and AdamW name rules, the initialisation and every reader of the names (the second pass, the GDN-side arms,
the probes, the inference engine) learned the split: refused for its reach.

The concatenated weight [W_1; W_2; ...] (N x K) without a copy a step:
- SEATED (seat): the group's parameters are moved once into one buffer, adjacent rows in model order (Tensor.set_, as
  ZeRO-1 seats every parameter in its flat buffer), so the concatenation IS a view of that storage (cat_view: as_strided
  on the first part, after checking that the parts lie back to back in one storage). build_model seats on the CPU; the
  trainer seats again after the model is moved (moving gives each parameter a storage of its own); ZeRO-1 keeps a
  group's parts adjacent in its own flat buffer (ced/train/zero1.py assign_owners: one owner a group, model order, every
  part a whole number of 64-element blocks). The optimizers update in place, so the view always holds the current
  weights. Cost: one copy of the group's weights at seating, none a step.
- Transformer Engine on our operands (model.fp8 backend "te", te_operands "ours", the B200's path; eager, a graph break
  as OursTELinear): the GEMMs read the seated view; a group that is not seated is refused by name (NotSeated), never
  copied silently. One quantiser pass of x (both orientations, ced/kernels/mx_chains.py quantize), one of the
  concatenated weight, one of dy; one Fprop, one Dgrad and one Wgrad GEMM (OursTEConcatLinear below).
- every other path (bf16 under autocast, float32 / float64 without it, the emulated MXFP8 linears) concatenates with
  torch.cat inside the autograd graph: under bf16 autocast each part is cast first and the casts concatenated
  (torch.cat([p.to(bf16) ...])), which Inductor writes straight into the concatenated bf16 buffer (one kernel, the bytes
  of today's per-part casts: measured on the CPU's Inductor); a float32 torch.cat feeding a reduction (the emulated
  quantisers) is realised (one float32 copy of the group's weights a call: the emulation measures quality, not speed). In
  eager code each torch.cat is a copy.

THE BACKWARD. The gradient of y arrives as one tensor: under torch.compile the backward of the split is a cat that Inductor
writes in place from the pointwise kernels that make its pieces (the SwiGLU backward, the norm backwards; measured on the
CPU's Inductor); a piece made by an opaque kernel (the GDN prep backward's du, FA4's dv) is copied into it. Dgrad: ONE GEMM,
dy (tokens x N) times the concatenated weight, the sum over all N in one accumulator; with TE it writes x's dtype (the
float32 stream) directly (OursTELinear writes bf16 and casts, and autograd adds the parts' float32 gradients). Wgrad: ONE
GEMM, dW (N x K) in float32, each part's gradient a view of it (rows), accumulated into that parameter's .grad.

NUMERICS. Fprop and Wgrad: each output element is the same dot product of the same operands as the separate GEMM's (an
output column reads one weight row; MXFP8's blocks of 32 run along K, and every part's N is a multiple of 32, so a
column-wise block never straddles two parts: the quantised bytes of the concatenation are the parts' bytes); a GEMM
kernel chosen for another N may sum K in another order. Dgrad: the sum over N in one accumulator instead of per-part
sums (rounded to bf16 in the separate TE path) added in float32: not bit for bit. Judged against a float64 truth under
the 4x criterion (ced/utils/criterion.py), then the 60M band.

Training mode only, as model.fp8 (ced/model/fp8.py): an eval-mode forward (in-training evals, the re-score, the probes,
cached generation) runs the separate projections, the model as it was, bit for bit.

Fixed shapes: every shape follows from the layer's (cat, split, as_strided); no boolean-mask indexing or nonzero.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# name -> (the module kind it applies to, its parts in model order)
GROUPS = {
    "gate_up": ("mlp", ("w_gate", "w_up")),
    "qkv": ("attention", ("q_proj", "k_proj", "v_proj")),
    "qkv_z": ("gdn", ("in_proj_qkv", "in_proj_z")),
    "ab": ("gdn", ("in_proj_b", "in_proj_a")),
}
SEAT_ALIGN = 64  # ZeRO-1's alignment (ced/train/zero1.py ALIGN): a part of a whole number of blocks keeps its group adjacent


class NotSeated(ValueError):
    """A fused group's parameters are not one buffer where the path needs them to be (the message names the group)."""


# ------------------------------------------------------------------------------------------------------------------
# the switch


def names_of(cfg) -> tuple:
    """ModelConfig.fused_projections as a tuple (a config built from a list keeps a list until set again)."""
    return tuple(getattr(cfg, "fused_projections", ()) or ())


def check_config(cfg) -> None:
    """Refuse ModelConfig.fused_projections by name wherever it is not built (cfg: a ModelConfig). () accepts anything."""
    names = names_of(cfg)
    if not names:
        return
    bad = [n for n in names if n not in GROUPS]
    if bad or len(set(names)) != len(names):
        raise ValueError(f"model.fused_projections {list(names)!r}: distinct names of {list(GROUPS)}")
    e, g, q = cfg.encoder, cfg.encoder.gdn, cfg.fp8
    gdn = {"qkv_z", "ab"} & set(names)
    refused = [
        (cfg.arch != "control", f"arch {cfg.arch!r} (built for arch 'control': its encoder layers)"),
        (cfg.loop.pass2_gate != "none", "model.loop.pass2_gate (the second pass applies part of in_proj_qkv outside the module)"),
        (bool(cfg.loop.recur_layers), "model.loop.recur_layers (a looped middle)"),
        (e.loops != 1, f"model.encoder.loops {e.loops}"),
        ("gate_up" in names and cfg.ffn_activation != "swiglu",
         f"'gate_up' with ffn_activation {cfg.ffn_activation!r} (it is SwiGLU's w_gate | w_up)"),
        ("qkv" in names and e.attention.gate == "headwise",
         "'qkv' with the headwise attention gate (g_proj, one output a head, sits between q_proj and k_proj)"),
        ("qkv" in names and e.attention.normalizer != "softmax",
         f"'qkv' with the {e.attention.normalizer!r} normaliser (nothing tests the pair)"),
        (bool(gdn) and (e.gdn_write_mix != "none" or bool(e.gdn_ccr_spacing)),
         "the GDN groups with the GDN-side arms (gdn_write_mix, gdn_ccr_spacing: their projections read x_w)"),
        (bool(gdn) and g.kda.decay != "scalar", "the GDN groups with per-channel decay (gdn.kda: its own forward)"),
        (bool(gdn) and g.erase.mode != "none", "the GDN groups with the erase direction (gdn.erase: its own forward)"),
        (bool(gdn) and g.dprod.n_h != 1, "the GDN groups with Gated DeltaProduct (gdn.dprod: its own forward)"),
        (cfg.nvfp4.mode != "none", "model.nvfp4 (its linears are NVFP4Linear)"),
        # D1a's recipes take a scale per tensor ("tensorwise") or per row ("rowwise": per token of dy and per input
        # feature of W in the Dgrad, each spanning every output column): over the concatenation that scale is the group's,
        # not the part's, so the quantised operands would not be the separate linears' (review, 2026-10-23: tensorwise
        # changes Fprop by 5.8e-3 relative on parts of different magnitude). MXFP8's blocks of 32 never straddle a part.
        (q.mode != "none" and q.scaling != "mxfp8",
         f"model.fp8.scaling {q.scaling!r} (a scale over the whole concatenation is not the part's; built for 'mxfp8')"),
        (q.mode != "none" and bool(tuple(getattr(q, "chains", ()))),
         "model.fp8.chains (the chains make the in-projections' operands one linear at a time; in_norm and exit_entry on "
         "Transformer Engine too: their linears read the chain's operand, not a concatenation)"),
        # the G0 folds take the SwiGLU's w_gate and w_up GEMMs themselves, norm_mlp defers the MLP's input norm into them:
        # SwiGLU.forward runs one of the two paths. g0_wire.check_folds refuses the same pair from the folds' side
        # (check_fp8 runs first in build_model); the other three groups do not meet the folds
        ("gate_up" in names and bool({"swiglu_bwd", "norm_mlp"} & set(getattr(q, "folds", ()))),
         f"'gate_up' with model.fp8.folds {list(getattr(q, 'folds', ()))!r} (the folds take the SwiGLU's w_gate and w_up "
         "GEMMs; norm_mlp defers the MLP's input norm into them)"),
        (q.mode != "none" and getattr(q, "backend", "emulated") == "te" and getattr(q, "te_operands", "te") != "ours",
         "model.fp8.backend 'te' with te_operands 'te' (TE's own te.Linear; the fused GEMMs are built on our operands, "
         "te_operands 'ours')"),
    ]
    why = [w for c, w in refused if c]
    if why:
        raise ValueError(f"model.fused_projections {list(names)!r} is not built for: {'; '.join(why)}. Leave it []")


def check_train_config(cfg) -> None:
    """The trainer-level refusals (cfg: a TrainConfig)."""
    if not names_of(cfg.model):
        return
    if getattr(cfg, "wgrad_accum", "autograd") != "autograd":
        raise ValueError(f"model.fused_projections with wgrad_accum {cfg.wgrad_accum!r}: build F accumulates each "
                         "separate linear's Wgrad in place, which the fused GEMM bypasses (not built together). Set one")


def _kind(module) -> str | None:
    from ced.model.attention import GatedAttention
    from ced.model.gdn import GatedDeltaNet
    from ced.model.mlp import SwiGLU

    if isinstance(module, SwiGLU):
        return "mlp"
    if isinstance(module, GatedAttention):
        return "attention"
    if isinstance(module, GatedDeltaNet):
        return "gdn"
    return None


def _fp8_key(m) -> tuple:
    return tuple(getattr(m, a, None) for a in ("fp8_active", "fp8_scaling", "fp8_backend", "fp8_quantiser",
                                                 "fp8_te_operands", "fp8_flag", "fp8_flag_form"))


def apply(model: nn.Module, names) -> dict:
    """Mark every encoder layer's mixer and FFN that has a requested group (module.fused_proj: a tuple of part-name
    tuples, a plain attribute, never in the state dict), after the model is built, initialised and (with model.fp8)
    quantised, and seat each group (seat). Returns the summary (also model.fused_summary)."""
    names = tuple(names)
    if not names:
        return {}
    summary = {"groups": list(names), "fused": [], "seated": 0}
    for i, layer in enumerate(model.encoder.layers):
        for part in ("mixer", "mlp"):
            mod = getattr(layer, part, None)
            kind = _kind(mod)
            groups = tuple(GROUPS[n][1] for n in names if GROUPS[n][0] == kind)
            if not groups:
                continue
            for parts in groups:
                lins = [getattr(mod, n, None) for n in parts]
                where = f"encoder.layers.{i}.{part}"
                if any(not isinstance(lin, nn.Linear) or lin.bias is not None for lin in lins):
                    raise ValueError(f"model.fused_projections: {where} {parts} are not all bias-free linears")
                # the pairs check_config refuses, behind it (a model wired by apply_fp8 and this function alone, merge review)
                if any(getattr(lin, "mx_hold", None) is not None for lin in lins):
                    raise ValueError(f"model.fused_projections: {where} {parts} hold a norm chain's operand "
                                     "(model.fp8.chains): not built together")
                if kind == "mlp" and getattr(mod, "g0_folds", None) is not None:
                    raise ValueError(f"model.fused_projections: {where} is wired with model.fp8.folds (swiglu_bwd / "
                                     "norm_mlp): 'gate_up' is not built together with them")
                if len({lin.in_features for lin in lins}) != 1 or len({_fp8_key(lin) for lin in lins}) != 1:
                    raise ValueError(f"model.fused_projections: {where} {parts} differ in input size or precision")
                if getattr(lins[0], "fp8_active", False) and getattr(lins[0], "fp8_scaling", "") == "mxfp8" and any(
                        lin.out_features % 32 for lin in lins):
                    raise ValueError(f"model.fused_projections: {where} {parts}: an MXFP8 part's outputs are not whole "
                                     f"blocks of 32 ({[lin.out_features for lin in lins]}): a column-wise block would "
                                     "straddle two parts")
                summary["fused"].append(f"{where}.{'|'.join(parts)}")
            mod.fused_proj = groups
    summary["seated"] = seat(model)
    model.fused_summary = summary
    return summary


def modules(model: nn.Module):
    """(module, groups) for every module apply() marked."""
    for m in model.modules():
        groups = getattr(m, "fused_proj", None)
        if groups:
            yield m, groups


# ------------------------------------------------------------------------------------------------------------------
# the seating


def is_seated(parts) -> bool:
    """The parts lie back to back in one storage, each contiguous, in order (their concatenation is a view of it)."""
    p0 = parts[0]
    st = p0.untyped_storage().data_ptr()
    off = p0.storage_offset()
    for p in parts:
        if (p.device, p.dtype) != (p0.device, p0.dtype) or not p.is_contiguous() or p.dim() != 2 or \
                p.shape[1] != p0.shape[1] or p.untyped_storage().data_ptr() != st or p.storage_offset() != off:
            return False
        off += p.numel()
    return True


@torch.no_grad()
def seat_group(parts) -> bool:
    """Move the parts into one new buffer, back to back in order (Tensor.set_; their values copied first). False when
    they already were, or are on the meta device (a model built for its shapes: nothing is done)."""
    if parts[0].device.type == "meta" or is_seated(parts):
        return False
    buf = torch.cat([p.detach().reshape(-1) for p in parts])
    off = 0
    for p in parts:
        n = p.numel()
        p.set_(buf.untyped_storage(), off, p.shape, torch.empty(p.shape, device="meta").stride())
        off += n
    return True


def seat(model: nn.Module) -> int:
    """Seat every marked group (seat_group); returns how many were moved. Idempotent: run again after anything that
    gives the parameters storages of their own (Module.to onto another device or dtype)."""
    n = 0
    for m, groups in modules(model):
        for parts in groups:
            n += seat_group([getattr(m, p).weight for p in parts])
    return n


def unseated(model: nn.Module) -> list[str]:
    """The marked groups whose parameters are not one buffer, by module and parts ([] when every one is)."""
    names = {id(m): n for n, m in model.named_modules()}
    return [f"{names.get(id(m), '?')}.{'|'.join(parts)}" for m, groups in modules(model) for parts in groups
            if not is_seated([getattr(m, p).weight for p in parts])]


def index_groups(model: nn.Module, params: list) -> list[list[int]]:
    """The marked groups as lists of indices into `params` (ZeRO-1's parameter list), each in model order."""
    index = {id(p): i for i, p in enumerate(params)}
    out = []
    for m, groups in modules(model):
        for parts in groups:
            idx = [index.get(id(getattr(m, p).weight)) for p in parts]
            if None not in idx:
                out.append(idx)
    return out


def cat_view(parts) -> torch.Tensor:
    """The concatenated weight (sum of rows x K) as a view of the seated parts' storage, detached: no copy. NotSeated
    when the parts are not one buffer."""
    if not is_seated(parts):
        raise NotSeated(
            f"model.fused_projections: a group of {len(parts)} parameters ({[tuple(p.shape) for p in parts]}) is not one "
            "buffer, so its concatenated weight would be a copy every call (Transformer Engine's fused path never "
            "copies): seat it (ced/model/fused_proj.py seat) after anything that moved the parameters")
    p0 = parts[0]
    n = sum(p.shape[0] for p in parts)
    return p0.detach().as_strided((n, p0.shape[1]), (p0.shape[1], 1), p0.storage_offset())


# ------------------------------------------------------------------------------------------------------------------
# the forward


def outputs(module, x: torch.Tensor) -> dict:
    """{part name: its output}, each a view of its group's one GEMM output, for every group of `module` in a training
    forward; {} in eval mode or without groups (the caller then runs the separate projections)."""
    groups = getattr(module, "fused_proj", None)
    if not groups or not module.training:
        return {}
    out = {}
    for parts in groups:
        lins = [getattr(module, p) for p in parts]
        y = fused_linear(lins, x)
        out.update(zip(parts, y.split([lin.out_features for lin in lins], -1)))
    return out


def fused_linear(lins, x: torch.Tensor) -> torch.Tensor:
    """x [W_1; W_2; ...]^T for the group's linears (bias-free, one input size, one precision): one GEMM by the path the
    separate linears take (module docstring)."""
    from ced.model.fp8 import FLAG_FORM_DEFAULT, FP8Linear, FP8LinearFunction, MXTritonLinearFunction

    l0 = lins[0]
    parts = [lin.weight for lin in lins]
    dev = x.device.type
    autocast = torch.is_autocast_enabled(dev)
    dtype = torch.get_autocast_dtype(dev) if autocast else x.dtype
    # A linear that holds a norm chain's operand (model.fp8.chains, mx_wire.MXHold) reads it, not x: on the TE form x is a
    # placeholder nothing wrote, and a quantised, a high-precision (GDN's a | b) or a TE-backend group would read garbage.
    # Refused at the config too (check_config); checked here first, before any path, since the TE form of the chains would otherwise read the placeholder
    if any(getattr(lin, "mx_hold", None) is not None for lin in lins):
        raise ValueError("model.fused_projections with model.fp8.chains 'in_norm' / 'exit_entry': not built together (these "
                         "linears read the chain's operand, not their input)")
    if isinstance(l0, FP8Linear) and l0.training and l0.fp8_active:
        if getattr(l0, "fp8_backend", "emulated") == "te":
            if getattr(l0, "fp8_te_operands", "te") != "ours":  # refused at the config too (check_config)
                raise ValueError("model.fused_projections with model.fp8.backend 'te' is built on our operands "
                                 "(te_operands 'ours')")
            return te_concat_linear(lins, x)
        if getattr(l0, "fp8_scaling", None) != "mxfp8":  # refused at the config too (check_config)
            raise ValueError(f"model.fused_projections with model.fp8.scaling {getattr(l0, 'fp8_scaling', None)!r}: a scale "
                             "over the whole concatenation is not the part's (built for 'mxfp8')")
        w = torch.cat(parts)  # the emulation: a float32 copy of the group's weights a call (module docstring)
        if getattr(l0, "fp8_quantiser", "torch") == "triton":
            flag = bool(getattr(l0, "fp8_flag", False))
            if flag and getattr(l0, "fp8_flag_form", FLAG_FORM_DEFAULT) == "pure":
                flag = "pure"
            return MXTritonLinearFunction.apply(x, w, dtype, flag)
        return FP8LinearFunction.apply(x, w, l0.fp8_scaling, dtype)
    if autocast:  # each part cast as autocast casts it, the casts concatenated (Inductor: one kernel, no float32 copy)
        return F.linear(x, torch.cat([p.to(dtype) for p in parts]))
    return F.linear(x, torch.cat(parts))


# ------------------------------------------------------------------------------------------------------------------
# Transformer Engine on our operands (model.fp8 backend "te", te_operands "ours")


class OursTEConcatLinear(torch.autograd.Function):
    """y = x [W_1; W_2; ...]^T with Transformer Engine's MXFP8 GEMMs on our operands only, the weight the seated
    group's view: OursTELinear (ced/model/fp8_te.py) on the concatenated weight, statement for statement, but for the
    Dgrad's output dtype (x's: the float32 stream is written by the GEMM, no bf16 copy and no cast) and the Wgrad's
    float32 dW returned to each part as a view of its rows."""

    @staticmethod
    def forward(ctx, x, out_dtype, flag_on, *weights):
        from ced.kernels import mx_chains as MC
        from ced.kernels import mx_ops
        from ced.model.fp8_te import ours_operand, te_gemm

        weight = cat_view(weights)
        k, n = x.shape[-1], weight.shape[0]
        x2 = x.reshape(-1, k).contiguous()
        if x2.shape[0] % 32 or k % 32 or n % 32:
            raise ValueError(f"model.fp8.te_operands 'ours': {x2.shape[0]} tokens x {k} -> {n}: TE's MXFP8 GEMMs take "
                             "whole blocks of 32 along every dot product (tokens for Wgrad, K for Fprop, N for Dgrad)")
        flag = mx_ops.FLAG if flag_on else None
        if flag_on and flag is None:
            raise RuntimeError("model.fp8.nonfinite_flag: the flag was never made (ced.kernels.mx_ops.flag_for)")
        ox = MC.quantize(x2, flag=flag, bit=mx_ops.ACTIVATION)
        ow = MC.quantize(weight, flag=flag, bit=mx_ops.WEIGHT)
        y = te_gemm("fprop", ours_operand(ow.q_row, ow.s_row, None, None, "W (Fprop)"),
                    ours_operand(ox.q_row, ox.s_row, None, None, "x (Fprop)"), out_dtype)
        ctx.save_for_backward(ox.q_col, ox.s_col, ow.q_col, ow.s_col)
        ctx.out_dtype, ctx.x_shape, ctx.x_dtype, ctx.w_dtype, ctx.n, ctx.flag_on = (
            out_dtype, x.shape, x.dtype, weight.dtype, n, flag_on)
        ctx.rows = [w.shape[0] for w in weights]
        return y.reshape(*x.shape[:-1], n)

    @staticmethod
    def backward(ctx, dy):
        from ced.kernels import mx_chains as MC
        from ced.kernels import mx_ops
        from ced.model.fp8_te import ours_operand, te_gemm

        xq, xs, wq, ws = ctx.saved_tensors
        od = MC.quantize(dy.reshape(-1, ctx.n).to(ctx.out_dtype).contiguous(),
                         flag=mx_ops.FLAG if ctx.flag_on else None, bit=mx_ops.GRADIENT)
        dx = None
        dws = [None] * len(ctx.rows)
        if ctx.needs_input_grad[0]:
            dx = te_gemm("dgrad", ours_operand(None, None, wq, ws, "W (Dgrad)"),
                         ours_operand(od.q_row, od.s_row, None, None, "dy (Dgrad)"), ctx.x_dtype)
            dx = dx.reshape(ctx.x_shape)
        if any(ctx.needs_input_grad[3:]):
            dw = te_gemm("wgrad", ours_operand(None, None, xq, xs, "x (Wgrad)"),
                         ours_operand(None, None, od.q_col, od.s_col, "dy (Wgrad)"), torch.float32).to(ctx.w_dtype)
            dws = list(dw.split(ctx.rows, 0))
        return (dx, None, None, *dws)


@torch.compiler.disable
def te_concat_linear(lins, x: torch.Tensor) -> torch.Tensor:
    """y = x [W_1; W_2; ...]^T through OursTEConcatLinear (training mode only), the output in the autocast dtype (bf16
    in training) or x's dtype without autocast; a graph break, as fp8_te.ours_te_linear."""
    from ced.model.fp8_te import import_te

    import_te()
    dev = x.device.type
    dtype = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else x.dtype
    return OursTEConcatLinear.apply(x, dtype, bool(getattr(lins[0], "fp8_flag", False)), *[lin.weight for lin in lins])
