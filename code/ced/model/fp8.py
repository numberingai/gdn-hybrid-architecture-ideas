"""FP8 linear layers: the reference (a fake quantisation) and its wiring (opt-in, off by default: ModelConfig.fp8,
FP8Config).

What it is. Every GEMM of a quantised linear layer (Fprop y = x W^T, Dgrad dx = dy W, Wgrad dW = dy^T x) runs on
operands rounded to an FP8 format, with the GEMM computed on the FP8 values and the operands' scales applied to its
float32 accumulator, as torch._scaled_mm (the kernel torchao calls) and Transformer Engine's FP8 GEMMs define it. It is
the reference a fast path must match ("within GEMM tolerance"), and the emulation the 60M A/B trains with. It measures
quality only; it says nothing about speed.

The two recipes (FP8Config.scaling). Both use CURRENT scaling: a tensor's scale is computed from the tensor itself at
the moment it is quantised, so there is no amax history, no state, nothing in the checkpoint, and no random draw.

    rowwise     torchao's Float8LinearRecipeName.ROWWISE (torchao/float8/config.py, read 2026-10-14): E4M3 for the
                activation, the weight and the gradient; "axiswise" scales, one per row of each GEMM operand along the
                GEMM's dot-product dimension; scales rounded down to a power of two (round_scales_to_power_of_2). Each
                GEMM quantises its two operands afresh from the high-precision tensors:
                  Fprop  x (M, K) per token over K;           W (N, K) per output feature over K
                  Dgrad  dy (M, N) per token over N;          W per input feature over N
                  Wgrad  dy per output feature over tokens;   x per input feature over tokens
    tensorwise  Transformer Engine's Float8CurrentScaling with the HYBRID format (E4M3 forward, E5M2 gradients); also
                torchao's TENSORWISE recipe up to the all-zero case below: one scale per tensor, not rounded. A per-tensor
                scale does not change with a transpose, so each tensor is quantised once: Fprop's x and W are the ones
                Wgrad and Dgrad read, and one quantised dy serves Dgrad and Wgrad.

    mxfp8 OCP Microscaling MXFP8 as Transformer Engine's MXFP8BlockScaling recipe runs it: E4M3 for the
                activation, the weight and the gradient (TE's default fp8_format Format.E4M3, and NVIDIA's MXFP8
                pre-training recipe, arXiv 2506.08027: with a scale per 32 elements E4M3's range suffices for gradients,
                and its extra mantissa bit is kept); one E8M0 power-of-two scale per block of 32 consecutive elements
                along each GEMM's dot product, from the block's amax by TE's rounding (ced/model/mxfp8.py states it and
                the OCP spec's, which differs). Each operand is quantised from the high-precision tensor in the
                orientation its GEMM needs, as TE does ("both versions are computed from the high precision input to
                avoid double quantization errors", TE's recipe docstring): Fprop x (M, K) blocks of 32 along K; W (N, K)
                blocks along K Dgrad dy (M, N) blocks along N; W blocks along N (its column-wise copy) Wgrad dy blocks
                of 32 tokens; x blocks of 32 tokens (its column-wise copy) The scales sit inside the dot product, so the
                GEMM runs on the dequantised operands (_mx path below). Non-finite values: a block holding an inf or NaN
                is NaN in every element with a NaN scale (mxfp8.py), so every output of Fprop, Dgrad or Wgrad that the
                block reaches is NaN and the trainer's finite check sees it; finite blocks are unchanged bit for bit.
                Backends (FP8Config.backend): "emulated" (this module, any device) or "te" (TE's MXFP8 te.Linear on
                sm100; ced/model/fp8_te.py, refused by name on sm90 and without TE; its te_nonfinite_check). The tuning
                pass: the emulated linear's operands from our Triton quantiser (FP8Config.quantiser "triton": the same
                bytes; MXTritonLinearFunction), TE's GEMMs fed only our operands (FP8Config.te_operands "ours"), and the
                global non-finite flag (FP8Config.nonfinite_flag). The chains (FP8Config.chains): the fused chains make
                the operands of the linears that read their outputs, in the producer's pass (ced/model/mx_wire.py).

The scale (torchao's amax_to_scale, the same for rowwise and tensorwise): scale = float32(fmax / max(amax, 1e-12)) with the division in
float64, fmax = 448 (E4M3) or 57344 (E5M2); rowwise then takes 2^floor(log2 scale), here exactly, from the exponent bits
(pow2_floor). The values: q = round_to_nearest_even(clamp(float32(t) * scale, -fmax, fmax)) in the FP8 format, by
torch's cast (tests/test_fp8.py checks it against every representable value enumerated in float64, subnormals and ties
included). The dequantisation scale is float32(1 / scale), as torchao hands _scaled_mm. An all-zero tensor or row
quantises to zeros (Transformer Engine sets the scale to 1 there, torchao to fmax / 1e-12: the values are zeros either
way). A power-of-two scale changes no mantissa: under rowwise an element's rounding is that of the element itself, and
only the range (saturation, subnormals) depends on its row.

The GEMM (_gemm). FP8 values are exact in bfloat16 (at most 4 significant bits, magnitudes 2^-16 .. 57344), so:
- with power-of-two scales (rowwise) the GEMM runs in the autocast dtype (bf16, float32 accumulation, one rounding of
  the output), and the scales, multiplied in afterwards, are exact: the same bits as accumulate, scale, round once;
- otherwise (tensorwise) the GEMM runs in float32 with autocast off (exact operands; on CUDA with TF32 too, whose 10
  mantissa bits hold them), the product of the two dequantisation scales multiplies the accumulator, and the result is
  rounded once to the output dtype;
- without autocast (float32 or float64 models, the CPU tests) the GEMM runs in the model's dtype;
- mxfp8: the GEMM runs on the dequantised operands (exact in float32; in bf16 exact unless a block's scale is below
  2^-124, ced/model/mxfp8.py), in the autocast dtype (float32 accumulation, one rounding of the output) or, without
  autocast, in the model's dtype; the hardware applies the block scales to each block's partial sums inside the
  tensor-core MMA (tcgen05 block-scaled MMA on sm100), which is the same sum of exact products.
Up to the order of accumulation this is torch._scaled_mm's result. Not emulated: H100's "fast accumulation" of FP8
products (torchao's Fprop default, use_fast_accum; Transformer Engine's Fprop without split accumulation), the order of
accumulation inside a kernel.

Where the operands come from:
- Fprop quantises the activation as it arrives (float32 from the norms, bf16 from a linear under autocast) and the
  float32 master weight. torchao's eager Float8Linear first casts the activation to the autocast dtype; a bf16 copy of
  a float32 activation lands on an E4M3 rounding midpoint for about 1/16 of the elements, where round-half-to-even may
  go the other way (measured on a Gaussian activation: 4.1% of the elements one E4M3 step apart, 1.6% of its norm, and
  both quantisations equally far from x; tests/test_fp8.py). Quantising the tensor as it arrives is one rounding, and
  torch.compile cannot drop an intermediate cast that is not there, so eager and compiled code quantise the same values
  (the NVFP4 emulation's rule, ced/model/nvfp4.py). A fast path is compared with the reference on the same inputs; one
  of our own keeps this quantiser and replaces only the GEMM.
- rowwise: Wgrad re-quantises x along the tokens from the copy saved for the backward in the GEMM dtype (what
  nn.Linear saves under autocast, the same memory; torchao saves the same bf16 input), and Dgrad re-quantises the
  float32 master weight along its other axis.
- tensorwise: the backward reads the forward's quantised x and W, saved as FP8 tensors (one byte an element; what
  Transformer Engine saves).
- mxfp8: Fprop also quantises x column-wise (blocks of 32 tokens) from the same input and saves that copy, FP8 values
  and E8M0 scales (1 + 1/32 bytes an element: TE's column-wise usage of the input quantiser); Dgrad quantises the
  float32 master weight column-wise; dy is quantised row-wise for Dgrad and column-wise for Wgrad, both from the
  gradient as it arrives (the GEMM dtype).
- gradients dy arrive in the GEMM dtype; dx returns in x's dtype and dW in the weight's (accumulated into the float32
  .grad as nn.Linear's under autocast). Muon reads that .grad and its update is unchanged: master weights and optimizer
  state stay float32.
Compiled: Inductor does not round fused fp32 -> bf16 -> fp32 round trips (emulate_precision_casts off by default), so in
compiled code the rowwise Wgrad may quantise the float32 activation instead of its bf16 copy, and a bf16 GEMM output may
skip a rounding. With emulate_precision_casts on, and with aot_eager, the compiled layer under bf16 autocast is
bitwise the eager one (tests/test_fp8.py).

Kept in high precision: the token embedding, the output head, norms, the GDN delta rule, its short conv, gates and
state, attention's softmax and its QK^T / PV products, the depth mix of Block AttnRes (float32, autocast off:
ced/model/attn_res.py depth_mix), and two kinds of linear: GDN's in_proj_a and in_proj_b (one output per value head, 4
at 60M and 20 or 24 at 3.8B; they train under AdamW, ced/train/optim.py ADAMW_MATRIX_SUFFIXES) and any linear whose
input or output size is not a multiple of 16 (the alignment an FP8 GEMM needs: cuBLASLt and torch._scaled_mm), e.g. the
headwise attention gate (one output per query head). The output head stays BF16 (the fused head is not an nn.Linear).

Training mode only. The quantised path runs while the module is in training mode; every eval-mode forward (in-training
evals, the re-score, the probes, cached generation) is the BF16 model bit for bit, as the NVFP4 screen read its arms.

Positions. Rowwise and mxfp8 quantise a training forward's activation per token (mxfp8 in blocks along the features of
one token) and the weights do not depend on the input, so a training forward is exactly causal, bit for bit. mxfp8's
column-wise copies (blocks of 32 tokens) enter only the weight gradient. Tensorwise couples positions through one number per tensor (its largest magnitude over
the micro-batch): causal given the tensor scales, and exactly causal in the gradient (no gradient flows through a scale).

Activation checkpointing and the depth mix (NVFP4 is refused with both; the reasons do not apply here). Nothing here
draws from a random generator and every scale is a function of the tensor quantised, so the recomputed forward of a
checkpointed layer or block quantises exactly what the first forward did, and the backward is the same function of
the same saved tensors: 'layer', 'host' and 'block' are bit for bit 'none' with FP8 on (tests/test_fp8.py). Block AttnRes
(R1, "block") and R1-entry ("block_entry") call the same mixer and FFN modules; the depth mix has no nn.Linear and stays
float32 with autocast off.

Fixed shapes: every tensor has a shape that follows from its input's (amax over a fixed dimension, clamp, cast,
transpose); no boolean-mask indexing, nonzero or masked_select.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ced.model import mxfp8 as mx

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
AMAX_EPS = 1e-12  # torchao's EPS: an all-zero row or tensor gets the scale fmax / 1e-12 and quantises to zeros
MODES = ("none", "linears")
SCALINGS = ("rowwise", "tensorwise")  # D1a's recipes: a scale per row or per tensor (torch._scaled_mm on Hopper)
MX_SCALINGS = ("mxfp8",)  # a scale per block of 32 inside the dot product (ced/model/mxfp8.py)
SCALING_CHOICES = SCALINGS + MX_SCALINGS  # what model.fp8.scaling may be
BACKENDS = ("emulated", "te")  # FP8Config.backend: this module's emulation, or Transformer Engine (mxfp8 only)
QUANTISERS = ("torch", "triton")  # FP8Config.quantiser: the emulated mxfp8 linear's quantiser
TE_OPERANDS = ("te", "ours")  # FP8Config.te_operands: who quantises Transformer Engine's GEMM operands
FLAG_FORMS = ("ordered", "pure")  # FP8Config.nonfinite_flag_form: the flagged operator's form (ced/kernels/mx_ops.py)
# FP8Config.nonfinite_flag_form's default (ced/config.py): "pure" since; "ordered" before. A config written before the
# key existed trained "ordered", the only form then (FLAG_FORM_BEFORE_KEY; ced/config.py checkpoint_config).
FLAG_FORM_DEFAULT = "pure"
FLAG_FORM_BEFORE_KEY = "ordered"
HIGH_PRECISION = ("in_proj_a", "in_proj_b")  # GDN's per-head projections: kept in high precision by name
ALIGN = 16  # an FP8 GEMM's alignment: a linear with another input or output size stays in high precision
_EXPONENT_BITS = 0x7F800000


def fp8_max(fmt: torch.dtype) -> float:
    return float(torch.finfo(fmt).max)


def pow2_floor(s: torch.Tensor) -> torch.Tensor:
    """2^floor(log2 s) for positive normal float32 s, exactly: the exponent bits of s with the mantissa cleared
    (torchao computes exp2(floor(log2 s)), the same number wherever log2 rounds correctly)."""
    return (s.view(torch.int32) & _EXPONENT_BITS).view(torch.float32)


def fp8_scale(amax: torch.Tensor, fmt: torch.dtype, pow2: bool) -> torch.Tensor:
    """The quantisation scale of a tensor or row whose largest magnitude is amax (torchao's amax_to_scale): float32 of
    fmax / max(amax, 1e-12) computed in float64, rounded down to a power of two with pow2."""
    s = (fp8_max(fmt) / amax.double().clamp(min=AMAX_EPS)).float()
    return pow2_floor(s) if pow2 else s


def quantize(t: torch.Tensor, fmt: torch.dtype, dim: int | None, pow2: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """t quantised to fmt with current scaling: (q, s_dec). q has t's shape and dtype fmt; s_dec is the float32
    dequantisation scale 1 / scale, 0-dim (dim None: one scale for the tensor) or with t's shape and size 1 along dim
    (one scale per slice along dim: per row with dim -1). The dequantised tensor is q * s_dec."""
    t32 = t.float()
    amax = t32.abs().amax() if dim is None else t32.abs().amax(dim=dim, keepdim=True)
    scale = fp8_scale(amax, fmt, pow2)
    fmax = fp8_max(fmt)
    q = (t32 * scale).clamp(-fmax, fmax).to(fmt)
    return q, scale.reciprocal()


def fake_quantize(t: torch.Tensor, fmt: torch.dtype = E4M3, dim: int | None = -1, pow2: bool = True) -> torch.Tensor:
    """The dequantised tensor (float32): what an FP8 GEMM multiplies."""
    q, s = quantize(t, fmt, dim, pow2)
    return q.float() * s


def _gemm(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, out_dtype: torch.dtype,
          pow2: bool) -> torch.Tensor:
    """(a @ b^T) * sa * sb for FP8 operands a (M, k) and b (N, k) and dequantisation scales sa (0-dim or (M, 1)) and sb
    (0-dim or (1, N)): float32 accumulation, the scales applied to the accumulator, one rounding to out_dtype (the module
    docstring, "The GEMM"). With pow2 the scales are powers of two and multiplying by them is exact; otherwise (0-dim
    scales) their product multiplies the accumulator, as cuBLASLt's alpha."""
    if out_dtype in (torch.float32, torch.float64):  # no autocast: the model's own precision
        with torch.autocast(a.device.type, enabled=False):
            acc = torch.matmul(a.to(out_dtype), b.to(out_dtype).t())
            return acc * sa.to(out_dtype) * sb.to(out_dtype) if pow2 else acc * (sa * sb).to(out_dtype)
    if pow2:  # a GEMM in the low-precision dtype (one rounding of its float32 accumulator), then the exact scales
        out = torch.matmul(a.to(out_dtype), b.to(out_dtype).t())
        return out * sa.to(out_dtype) * sb.to(out_dtype)
    with torch.autocast(a.device.type, enabled=False):  # float32 GEMM on the exact operands, scaled, rounded once
        return (torch.matmul(a.float(), b.float().t()) * (sa * sb)).to(out_dtype)


def gemm_tolerance(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, ref: torch.Tensor,
                   other: torch.Tensor) -> torch.Tensor:
    """The GEMM tolerance a fast path's FP8 GEMM `other` is held to against the reference `ref` (both a @ b^T scaled by
    sa, sb, in ref's dtype): elementwise, how far apart two GEMMs on the same FP8 operands may land when each sums the
    exact products in float32 in its own order and rounds once to the output dtype:

        2 * k * 2^-24 * (|a| @ |b|^T) * |sa| * |sb|   +   one unit in the last place of the output dtype

    (twice Higham's gamma_k bound for float32 summation in any order, and the two roundings of the output, at the
    larger of the two magnitudes). Returns |other - ref| divided by that bound: a fast path is within tolerance where
    every element is at most 1. H100's fast accumulation (fewer accumulator bits) is outside this bound by design."""
    k = a.shape[-1]
    with torch.autocast(a.device.type, enabled=False):
        mag = torch.matmul(a.double().abs(), b.double().abs().t()) * sa.double().abs() * sb.double().abs()
    r, o = ref.double(), other.double()
    big = torch.maximum(r.abs(), o.abs())
    eps = torch.finfo(ref.dtype).eps  # 2^-(mantissa bits): the ulp of 1
    ulp = torch.where(big > 0, torch.exp2(torch.floor(torch.log2(big))) * eps, torch.zeros_like(big))
    bound = 2 * k * 2.0**-24 * mag + ulp
    return torch.where(bound > 0, (o - r).abs() / bound.clamp(min=torch.finfo(torch.float64).tiny), torch.zeros_like(r))


class FP8LinearFunction(torch.autograd.Function):
    """y = x W^T with FP8 operands in Fprop, Dgrad and Wgrad (the module docstring)."""

    @staticmethod
    def forward(ctx, x, weight, scaling, gemm_dtype):
        k = x.shape[-1]
        n = weight.shape[0]
        x2 = x.reshape(-1, k)
        rowwise = scaling == "rowwise"
        if scaling == "mxfp8":
            # Fprop: x and W in blocks of 32 along K, from x as it arrives and the float32 master weight; the GEMM on the
            # dequantised operands. Saved: x's column-wise copy (blocks of 32 tokens, from the same x) as FP8 values and
            # E8M0 exponents, and the master weight (a parameter: no copy).
            xd, wd = mx.fake_quantize(x2, -1, gemm_dtype), mx.fake_quantize(weight, -1, gemm_dtype)
            y = mx.gemm(xd, wd.t(), gemm_dtype)
            xq, xb = mx.quantize(x2, 0)
            ctx.save_for_backward(xq, xb, weight)
        elif rowwise:
            xq, xs = quantize(x2, E4M3, -1, True)  # (M, 1): one scale per token
            wq, ws = quantize(weight, E4M3, -1, True)  # (N, 1): one scale per output feature
            y = _gemm(xq, wq, xs, ws.t(), gemm_dtype, True)
            # Saved: the activation in the GEMM dtype (as nn.Linear under autocast) and the master weight (a parameter:
            # no copy); the backward quantises them along the other axis.
            ctx.save_for_backward(x2.to(gemm_dtype), weight)
        else:
            xq, xs = quantize(x2, E4M3, None, False)
            wq, ws = quantize(weight, E4M3, None, False)
            y = _gemm(xq, wq, xs, ws, gemm_dtype, False)
            ctx.save_for_backward(xq, xs, wq, ws)  # the FP8 tensors themselves: Dgrad and Wgrad read them
        ctx.scaling, ctx.gemm_dtype, ctx.x_shape = scaling, gemm_dtype, x.shape
        ctx.x_dtype, ctx.w_dtype = x.dtype, weight.dtype
        return y.reshape(*x.shape[:-1], n)

    @staticmethod
    def backward(ctx, dy):
        gd = ctx.gemm_dtype
        dx = dw = None
        if ctx.scaling == "mxfp8":
            xq, xb, weight = ctx.saved_tensors
            n, k = weight.shape
            dy2 = dy.to(gd).reshape(-1, n)
            if ctx.needs_input_grad[0]:
                # Dgrad: dx = dy W, the dot product over the output features: dy in blocks along N, W column-wise
                dx = mx.gemm(mx.fake_quantize(dy2, -1, gd), mx.fake_quantize(weight, 0, gd), gd)
                dx = dx.reshape(ctx.x_shape).to(ctx.x_dtype)
            if ctx.needs_input_grad[1]:
                # Wgrad: dW = dy^T x, the dot product over the tokens: both in blocks of 32 tokens
                dw = mx.gemm(mx.fake_quantize(dy2, 0, gd).t(), mx.dequantize(xq, xb, 0, gd), gd).to(ctx.w_dtype)
        elif ctx.scaling == "rowwise":
            x_saved, weight = ctx.saved_tensors
            n, k = weight.shape
            dy2 = dy.to(gd).reshape(-1, n)
            if ctx.needs_input_grad[0]:
                # Dgrad: dx = dy W, the dot product over the output features n: dy per token, W per input feature.
                gq, gs = quantize(dy2, E4M3, -1, True)  # (M, 1)
                wq, ws = quantize(weight.t(), E4M3, -1, True)  # (k, n) quantised per row: (k, 1)
                dx = _gemm(gq, wq, gs, ws.t(), gd, True).reshape(ctx.x_shape).to(ctx.x_dtype)
            if ctx.needs_input_grad[1]:
                # Wgrad: dW = dy^T x, the dot product over the tokens: both operands blocked along the tokens.
                gq, gs = quantize(dy2.t(), E4M3, -1, True)  # (n, M): one scale per output feature
                xq, xs = quantize(x_saved.t(), E4M3, -1, True)  # (k, M): one scale per input feature
                dw = _gemm(gq, xq, gs, xs.t(), gd, True).to(ctx.w_dtype)
        else:
            xq, xs, wq, ws = ctx.saved_tensors
            n, k = wq.shape
            gq, gs = quantize(dy.to(gd).reshape(-1, n), E5M2, None, False)  # one quantised gradient for both GEMMs
            if ctx.needs_input_grad[0]:
                dx = _gemm(gq, wq.t(), gs, ws, gd, False).reshape(ctx.x_shape).to(ctx.x_dtype)
            if ctx.needs_input_grad[1]:
                dw = _gemm(gq.t(), xq.t(), gs, xs, gd, False).to(ctx.w_dtype)
        return dx, dw, None, None


class MXTritonLinearFunction(torch.autograd.Function):
    """FP8LinearFunction's mxfp8 branch with every operand from our Triton quantiser (FP8Config.quantiser "triton"): x, W and
    dy each quantised ONCE, both orientations from one read (ced/kernels/mx_ops.py), where the reference quantises each
    orientation from the tensor. The operands are the same bytes (the CPU path is the reference itself; on CUDA the GPU
    checks judge the kernel's bytes against it), so every GEMM below is the reference branch's GEMM on the same dequantised
    values, statement for statement: Fprop y = x W^T: x row-wise, W row-wise (reference: fake_quantize(x2, -1), (weight,
    -1)) Dgrad dx = dy W: dy row-wise, W column-wise (fake_quantize(dy2, -1), (weight, 0)) Wgrad dW = dy^T x: dy
    column-wise, x column-wise (saved) (fake_quantize(dy2, 0), quantize(x2, 0) saved) Saved for the backward: x's
    column-wise copy and exponents and the master weight, as the reference saves. With flag_on, each quantisation ORs its
    bit into the process's non-finite flag (mx_ops.FLAG; FP8Config.nonfinite_flag) through the effectful operator
    ced::mx_quantize_flagged, which finds the flag when it runs: no compiled graph holds the flag, so none saves it for a
    backward. flag_on "pure" (FP8Config.nonfinite_flag_form "pure") calls ced::mx_quantize_flagged_pure instead: the same
    body, declared pure, so a compiled graph is the flag off's (ced/kernels/mx_ops.py, "The pure form"); it is the default
    form since (FLAG_FORM_DEFAULT). Every quantisation feeds a GEMM, so no call is dead for a compiler to drop with its bit
    (tests/test_mxfp8_flag_operands.py)."""

    @staticmethod
    def forward(ctx, x, weight, gemm_dtype, flag_on):
        from ced.kernels import mx_ops

        k = x.shape[-1]
        n = weight.shape[0]
        x2 = x.reshape(-1, k)
        ox = mx_ops.quantize(x2, flag_on, mx_ops.ACTIVATION)
        ow = mx_ops.quantize(weight, flag_on, mx_ops.WEIGHT)
        xd = mx.dequantize(ox.q_row, ox.s_row, -1, gemm_dtype)
        wd = mx.dequantize(ow.q_row, ow.s_row, -1, gemm_dtype)
        y = mx.gemm(xd, wd.t(), gemm_dtype)
        ctx.save_for_backward(ox.q_col, ox.s_col, weight)
        ctx.gemm_dtype, ctx.x_shape, ctx.flag_on = gemm_dtype, x.shape, flag_on
        ctx.x_dtype, ctx.w_dtype = x.dtype, weight.dtype
        return y.reshape(*x.shape[:-1], n)

    @staticmethod
    def backward(ctx, dy):
        from ced.kernels import mx_ops

        gd = ctx.gemm_dtype
        xq, xb, weight = ctx.saved_tensors
        n, k = weight.shape
        od = mx_ops.quantize(dy.to(gd).reshape(-1, n), ctx.flag_on, mx_ops.GRADIENT)
        dx = dw = None
        if ctx.needs_input_grad[0]:
            ow = mx_ops.quantize(weight, ctx.flag_on, mx_ops.WEIGHT)  # its column-wise copy: blocks along N
            dx = mx.gemm(mx.dequantize(od.q_row, od.s_row, -1, gd), mx.dequantize(ow.q_col, ow.s_col, 0, gd), gd)
            dx = dx.reshape(ctx.x_shape).to(ctx.x_dtype)
        if ctx.needs_input_grad[1]:
            dw = mx.gemm(mx.dequantize(od.q_col, od.s_col, 0, gd).t(), mx.dequantize(xq, xb, 0, gd), gd).to(ctx.w_dtype)
        return dx, dw, None, None


class FP8Linear(nn.Linear):
    """nn.Linear with the same parameters and state dict whose GEMMs run on FP8 operands while `fp8_active` and the
    module is in training mode. Made by apply_fp8, which turns every linear layer of the layers' mixers and FFNs into
    this class (the ones kept in high precision compute F.linear), so every layer of a kind has the same structure for
    torch.compile. In eval mode it is F.linear: the plain layer, bit for bit.

    Attributes: fp8_active (bool), fp8_scaling ("rowwise", "tensorwise" or "mxfp8"), fp8_backend ("emulated" or "te":
    Transformer Engine's MXFP8 linear, ced/model/fp8_te.py), fp8_te_check (FP8Config.te_nonfinite_check), and the
    tuning pass's fp8_quantiser (FP8Config.quantiser), fp8_te_operands (FP8Config.te_operands) and fp8_flag
    (FP8Config.nonfinite_flag), and fp8_flag_form (FP8Config.nonfinite_flag_form); plain attributes, never in the
    state dict. A module made before the tuning pass, or by hand, without the last four, runs as at their defaults
    (fp8_flag_form: FLAG_FORM_DEFAULT, "pure" since)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.fp8_active:
            hold = getattr(self, "mx_hold", None) if self.training else None  # the chains' TE form (mx_wire): GDN's a / b
            if hold is not None and hold.low is not None:  # h's copy in the GEMM's dtype, made by the norm chain
                from ced.model import mx_wire

                return F.linear(mx_wire.LowRead.apply(x, hold.low), self.weight, self.bias)
            return F.linear(x, self.weight, self.bias)
        return self.quantised(x, self.weight)

    def quantised_now(self) -> bool:
        """Whether a call runs the quantised path now: training mode and a quantised layer (fp8_active). The GDN-side arms
        (ced/model/gdn_arms.py write_inputs) ask it before they send a row range of this layer's weight through `quantised`."""
        return bool(self.training and self.fp8_active)

    def quantised(self, x: torch.Tensor, weight: torch.Tensor, own: bool = True, op: tuple | None = None) -> torch.Tensor:
        """The quantised path (forward's, after its eval / kept-layer branch) on `weight`: self.weight, or a range of its rows (the
        GDN-side arms' two projections of in_proj_qkv, a row range each: wsl reads x for the q rows and x_w for the k / v rows,
        ced/model/gdn_arms.py "On MXFP8"). The weight's rows are quantised as the whole weight's are when the range starts and ends on
        a block of 32 (the column-wise copy's blocks run along the rows, MX's N); the operands, scales and GEMMs are those of a layer
        whose weight is `weight`. The norm chains' operand (mx_hold) is read only for the layer's own weight: the chains are refused
        with the arms (mx_wire.check_chains) but for the producer form. own: weight is self.weight (forward's call); False for a
        row range, which reads no hold. op (q_row, s_row, q_col, s_col): x's operand, given by the caller (the producer form of the
        arms' mixes: the layer's input-norm pass made the operand of x_w or x_q, and the row range of the weight is multiplied by
        it, gdn_arms.write_inputs); x is then the autograd edge only. Built for the emulated linear on our Triton quantiser and TE on
        our operands (model.encoder.mix_fold 'producer' is refused with any other)."""
        if getattr(self, "fp8_backend", "emulated") == "te":
            from ced.model import fp8_te

            if getattr(self, "fp8_te_operands", "te") == "ours":  # TE's GEMMs on our operands only
                hold = getattr(self, "mx_hold", None) if own else None  # model.fp8.chains on this path (mx_wire, the TE form)
                if op is not None:  # the operand the caller's norm pass made, multiplied by this row range
                    y = fp8_te.ours_te_operand_linear(self, x, op, weight)
                elif hold is not None and hold.op is not None:  # the operand the layer's norm chain made
                    y = fp8_te.ours_te_operand_linear(self, x, hold.op)
                elif own:  # (the layer's own weight: the call as it was, two arguments)
                    y = fp8_te.ours_te_linear(self, x)
                else:  # a row range of the weight (the GDN-side arms)
                    y = fp8_te.ours_te_linear(self, x, weight)
                return y if self.bias is None else y + self.bias.to(y.dtype)
            assert own and op is None, "a row range of the weight or a given operand is for the emulated path and TE on our operands (fp8.py quantised)"
            # with model.fp8.te_nonfinite_check: infinities of x and of dy become NaN before TE quantises them
            run = fp8_te.checked_te_linear if getattr(self, "fp8_te_check", False) else fp8_te.te_linear_forward
            y = run(self, x)
            return y if self.bias is None else y + self.bias.to(y.dtype)
        dev = x.device.type
        dtype = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else x.dtype
        if getattr(self, "fp8_quantiser", "torch") == "triton":  # our Triton quantiser, the same operands
            hold = getattr(self, "mx_hold", None) if own else None  # model.fp8.chains "in_norm": set on the mixer's in-projections only
            if op is not None:  # the operand the caller's norm pass made, multiplied by this row range of the weight
                from ced.model import mx_wire

                y = mx_wire.MXOperandLinearFunction.apply(x, weight, dtype, *op)
            elif hold is not None and hold.op is not None:  # the operand the layer's input norm made (ced/model/mx_wire.py)
                from ced.model import mx_wire

                y = mx_wire.MXOperandLinearFunction.apply(x, weight, dtype, *hold.op)
            else:
                flag = bool(getattr(self, "fp8_flag", False))
                if flag and getattr(self, "fp8_flag_form", FLAG_FORM_DEFAULT) == "pure":  # nonfinite_flag_form
                    flag = "pure"
                y = MXTritonLinearFunction.apply(x, weight, dtype, flag)
        else:
            assert op is None, "a given operand is for our Triton quantiser's linear and TE on our operands (fp8.py quantised)"
            y = FP8LinearFunction.apply(x, weight, self.fp8_scaling, dtype)
        return y if self.bias is None else y + self.bias.to(y.dtype)


# ----------------------------------------------------------------------------------------------
# which layers


def kept_reason(name: str, m: nn.Linear) -> str | None:
    """Why a layer's linear stays in high precision, or None when it is quantised: GDN's in_proj_a / in_proj_b by
    name, any other linear whose input or output size is not a multiple of 16."""
    if name.split(".")[-1] in HIGH_PRECISION:
        return "per-head projection (one output per value head; AdamW)"
    if m.in_features % ALIGN or m.out_features % ALIGN:
        return f"{m.in_features} -> {m.out_features}: not a multiple of {ALIGN}"
    return None


def check_fp8(cfg, q=None) -> None:
    """Refuse the switch (cfg: a ModelConfig; q: the FP8Config, default cfg.fp8, with a mode other than "none")
    wherever it is not built, so that no model ignores or half-applies one it was given. ValueError, by name.
    Built and tested: arch "control" with one pass through the stack, the softmax normaliser, model.encoder.attn_res
    "none", "block" (R1) or "block_entry" (R1-entry). Refused:
    - any arch but the control (CED v2's decoder and memory projections, v1, the second-pass arch: never wired);
    - the control's gated second pass and the loop in the published form (pass 2 applies part of a layer's
      in-projection with F.linear on its weight, ced/model/gdn.py; a looped middle calls its layers more than once per
      forward: neither is tested), and a looped encoder;
    - the asentmax normaliser (its scale heads are applied with F.linear on their weights, ced/model/attention.py);
    - the residual scalars (encoder.layer_scalars) and Canon layers (encoder.canon_kernel): nothing tests them with it;
    - model.nvfp4 at the same time (both replace the same linears);
    - backend "te" with a scaling other than "mxfp8" (TE's other recipes are not wired);
    - te_nonfinite_check without backend "te";
    - the tuning pass's: quantiser "triton" but with scaling "mxfp8" and backend "emulated"; te_operands "ours" but
      with backend "te", and not with te_nonfinite_check; nonfinite_flag but with quantiser "triton" or te_operands
      "ours" (our kernels write the flag);
    - nonfinite_flag_form outside FLAG_FORMS. Either form is accepted everywhere else: the form is read only with
      nonfinite_flag and quantiser "triton" (flag_form_of), and is inert elsewhere (the flag off; te_operands "ours",
      whose linears run eagerly, a graph break, with no operator form to choose). An earlier version refused "pure" there; with
      "pure" the default that refusal would refuse every config;
    - the chains: ced/model/mx_wire.py check_chains says where each is refused;
    - the G0 folds: ced/model/g0_wire.py check_folds says where they are refused;
    - values outside their range.
    Backend "te" is also refused by name where it cannot run (no Transformer Engine, a device below sm100): that needs
    the device, so the trainer calls ced.model.fp8_te.check_te_backend once the model is on it."""
    q = cfg.fp8 if q is None else q
    if q.mode not in MODES:
        raise ValueError(f"model.fp8.mode {q.mode!r}: one of {MODES}")
    if q.scaling not in SCALING_CHOICES:
        raise ValueError(f"model.fp8.scaling {q.scaling!r}: one of {SCALING_CHOICES}")
    backend = getattr(q, "backend", "emulated")
    if backend not in BACKENDS:
        raise ValueError(f"model.fp8.backend {backend!r}: one of {BACKENDS}")
    if backend == "te" and q.scaling != "mxfp8":
        raise ValueError(f"model.fp8.backend 'te' is built for model.fp8.scaling 'mxfp8' only (Transformer Engine's "
                         f"MXFP8BlockScaling), not {q.scaling!r}: use backend 'emulated'")
    if getattr(q, "te_nonfinite_check", False) and backend != "te":
        raise ValueError(f"model.fp8.te_nonfinite_check is a check on Transformer Engine's operands (backend 'te'), not "
                         f"backend {backend!r}: the emulated MXFP8 quantiser makes a non-finite block NaN itself "
                         "(ced/model/mxfp8.py)")
    # the tuning pass's options: each refused by name where it is not built
    quantiser = getattr(q, "quantiser", "torch")
    te_operands = getattr(q, "te_operands", "te")
    if quantiser not in QUANTISERS:
        raise ValueError(f"model.fp8.quantiser {quantiser!r}: one of {QUANTISERS}")
    if te_operands not in TE_OPERANDS:
        raise ValueError(f"model.fp8.te_operands {te_operands!r}: one of {TE_OPERANDS}")
    if quantiser == "triton" and (q.scaling != "mxfp8" or backend != "emulated"):
        raise ValueError(f"model.fp8.quantiser 'triton' is our MXFP8 kernels' quantiser for the emulated linear (scaling "
                         f"'mxfp8', backend 'emulated'), not scaling {q.scaling!r} with backend {backend!r}; with backend "
                         "'te' model.fp8.te_operands says who quantises")
    if te_operands == "ours" and backend != "te":
        raise ValueError(f"model.fp8.te_operands 'ours' feeds Transformer Engine's GEMMs (backend 'te'), not backend "
                         f"{backend!r}")
    if te_operands == "ours" and getattr(q, "te_nonfinite_check", False):
        raise ValueError("model.fp8.te_nonfinite_check checks the tensors Transformer Engine quantises itself; with "
                         "model.fp8.te_operands 'ours' TE quantises nothing (our kernels make a non-finite block NaN): "
                         "set one")
    if getattr(q, "nonfinite_flag", False) and quantiser != "triton" and te_operands != "ours":
        raise ValueError("model.fp8.nonfinite_flag is written by our MXFP8 kernels: it needs model.fp8.quantiser "
                         "'triton' (backend 'emulated') or model.fp8.te_operands 'ours' (backend 'te'); the PyTorch "
                         "reference quantiser and Transformer Engine's own write no flag")
    form = getattr(q, "nonfinite_flag_form", FLAG_FORM_DEFAULT)
    if form not in FLAG_FORMS:  # read only where flag_form_of says; inert (and either value accepted) elsewhere
        raise ValueError(f"model.fp8.nonfinite_flag_form {form!r}: one of {FLAG_FORMS}")
    chains = tuple(getattr(q, "chains", ()))
    if chains:  # the fused chains in training (ced/model/mx_wire.py)
        from ced.model import mx_wire

        mx_wire.check_chains(cfg, q, chains)
    folds = tuple(getattr(q, "folds", ()))
    if folds:  # the G0 folds (ced/model/g0_wire.py)
        from ced.model import g0_wire

        g0_wire.check_folds(cfg, q, folds)
    if q.mode == "none":
        return
    what = f"model.fp8 (mode {q.mode!r})"
    if cfg.arch != "control":
        raise ValueError(f"{what} is built for arch 'control', not {cfg.arch!r}")
    if cfg.loop.pass2_gate != "none":
        raise ValueError(f"{what} is not built for the control's second pass (model.loop.pass2_gate "
                         f"{cfg.loop.pass2_gate!r}): pass 2 applies part of a layer's in-projection outside the module")
    if cfg.loop.recur_layers:
        raise ValueError(f"{what} is not built for the loop in the published form (model.loop.recur_layers "
                         f"{list(cfg.loop.recur_layers)!r}): nothing tests it")
    if cfg.encoder.loops != 1:
        raise ValueError(f"{what} is not built for a looped encoder (model.encoder.loops {cfg.encoder.loops})")
    if cfg.encoder.attention.normalizer != "softmax":
        raise ValueError(f"{what} is built for the softmax normaliser, not {cfg.encoder.attention.normalizer!r} (its "
                         "scale heads are applied outside their modules)")
    if cfg.encoder.layer_scalars != "none":
        raise ValueError(f"{what} is not built together with model.encoder.layer_scalars "
                         f"({cfg.encoder.layer_scalars!r}): nothing tests the pair")
    if cfg.encoder.canon_kernel:
        raise ValueError(f"{what} is not built together with model.encoder.canon_kernel ({cfg.encoder.canon_kernel!r}): "
                         "nothing tests the pair")
    if cfg.nvfp4.mode != "none":
        raise ValueError(f"{what} and model.nvfp4 (mode {cfg.nvfp4.mode!r}) both replace the layers' linears: set one")


def apply_fp8(model: nn.Module, q) -> dict:
    """Turn the linear layers of the model's layers' mixers and FFNs into FP8Linear per q (an FP8Config), in place,
    after the model is built and initialised and before the optimizer is built. Parameters, their names, the state dict
    and the RNG use do not change. Returns a summary (also stored as model.fp8_summary): the mode, the scaling, the
    quantised linears, the ones kept in high precision with the reason, and the share of the layers' linear weights
    that is quantised."""
    if q.mode == "none":
        return {}
    check_fp8(model.cfg, q)
    backend = getattr(q, "backend", "emulated")
    summary = {"mode": q.mode, "scaling": q.scaling, "backend": backend, "quantised": [], "high_precision": {},
               "quantised_params": 0, "linear_params": 0}
    for i, layer in enumerate(model.encoder.layers):
        for part in ("mixer", "mlp"):
            for name, m in getattr(layer, part).named_modules():
                if type(m) is not nn.Linear:
                    continue
                full = f"encoder.layers.{i}.{part}.{name}"
                why = kept_reason(name, m)
                m.__class__ = FP8Linear
                m.fp8_active = why is None
                m.fp8_scaling = q.scaling
                m.fp8_backend = backend
                m.fp8_te_check = bool(getattr(q, "te_nonfinite_check", False))
                m.fp8_quantiser = getattr(q, "quantiser", "torch")
                m.fp8_te_operands = getattr(q, "te_operands", "te")
                m.fp8_flag = bool(getattr(q, "nonfinite_flag", False))
                m.fp8_flag_form = getattr(q, "nonfinite_flag_form", FLAG_FORM_DEFAULT)
                if why is None:
                    summary["quantised"].append(full)
                    summary["quantised_params"] += m.weight.numel()
                else:
                    summary["high_precision"][full] = why
                summary["linear_params"] += m.weight.numel()
    summary["quantised_param_share"] = summary["quantised_params"] / max(1, summary["linear_params"])
    if getattr(q, "chains", ()):  # the fused chains: absent at the default ()
        from ced.model import mx_wire

        summary["chains"] = mx_wire.apply_chains(model, tuple(q.chains), q)
    if getattr(q, "folds", ()):  # the G0 folds: absent at the default ()
        from ced.model import g0_wire

        summary["folds"] = g0_wire.apply_folds(model, tuple(q.folds))
    if getattr(q, "te_nonfinite_check", False):  # absent at the default (B-fixes)
        summary["te_nonfinite_check"] = True
    for key, default in (("quantiser", "torch"), ("te_operands", "te"), ("nonfinite_flag", False)):  # the tuning pass's
        if getattr(q, key, default) != default:  # absent at their defaults
            summary[key] = getattr(q, key)
    # The flag's form: recorded wherever it is read (flag_form_of: the flag on with quantiser "triton"), whichever
    # form, and absent elsewhere. Before it was recorded only when "pure" (the default then "ordered"): a record with
    # the flag on, quantiser "triton" and no form is the ordered form.
    form = flag_form_of(q)
    if form is not None:
        summary["nonfinite_flag_form"] = form
    model.fp8_summary = summary
    return summary


def flag_form_of(q) -> str | None:
    """The form the MXFP8 non-finite flag's quantiser runs under q (an FP8Config, or its dict as in a checkpoint's
    config["model"]["fp8"]): None where the form is read by nothing (model.fp8 mode "none", nonfinite_flag off, or a
    quantiser other than "triton": with te_operands "ours" the linears run our kernels eagerly, with no operator form),
    else nonfinite_flag_form: "ordered" or "pure". A dict without the key was written before, when the ordered form was
    the only one (FLAG_FORM_BEFORE_KEY). The trainer compares checkpoints and runs by this, so the inert key (every
    flag-off checkpoint written before "pure" became the default carries "ordered", every later one "pure") never makes two configs
    differ."""
    if isinstance(q, dict):
        get = q.get
    else:
        def get(key, default=None):
            return getattr(q, key, default)
    if get("mode", "none") == "none" or not get("nonfinite_flag", False) or get("quantiser", "torch") != "triton":
        return None
    return get("nonfinite_flag_form", FLAG_FORM_BEFORE_KEY)
