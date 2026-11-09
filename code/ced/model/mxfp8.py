"""MXFP8: the reference quantiser (OCP Microscaling with E4M3 elements and E8M0 block scales; FP8Config.scaling "mxfp8").

The format (OCP Microscaling Formats (MX) v1.0, section 5; Transformer Engine's MXFP8BlockScaling recipe): a tensor is
cut into blocks of 32 consecutive elements along one dimension (for a GEMM operand, along the GEMM's dot product); each
block carries one shared scale X = 2^(b - 127) stored as an 8-bit biased exponent b (E8M0: no sign, no mantissa; b = 255
is NaN) and 32 elements P_i in FP8 E4M3 (largest finite value 448, no infinities, NaN 0x7F), with V_i ~ X P_i.

    te   b = the biased exponent of the power of two at or above amax / 448, from the float32 product
         v = amax * float32(1/448): Transformer Engine 2.20.2 (transformer_engine/common/util/ptx_arch_spec.cuh
         float_to_e8m0: on sm100 "cvt.rp.satfinite.ue8m0x2.f32", elsewhere the same rounding up from the bits;
         the callers in common/cast/mxfp8/quantize_mxfp8.cuh pass thread_amax * Quantized_Limits<OType>::max_norm_rcp,
         with max_norm_rcp = float(1.0 / 448)). Rounding up means amax / X <= 448: no element of a finite block
         saturates (tests/test_mxfp8.py sweeps it). v = 0 gives b = 0 (2^-127); v = inf gives 254 (satfinite); a NaN
         v gives 255 (it cannot occur: amax ignores NaN, below). NVIDIA's MXFP8 pre-training recipe (arXiv
         2506.08027) chose this rounding because the OCP rule below saturates.
    ocp  the spec's section 6.3: X = 2^(floor(log2 amax) - 8) (8 = the largest exponent of E4M3), clamped to
         [2^-127, 2^127], an infinite amax giving NaN (Microsoft's microxcaling library, mx_ops._shared_exponents and
         _quantize_mx). For a block whose amax has a mantissa above 1.75 (448 = 1.75 x 2^8) the largest elements exceed
         448 and saturate; elsewhere the two rules agree. TE's b is OCP's b or OCP's b + 1 (tests/test_mxfp8.py).

The elements: P_i = E4M3(clamp(float32(V_i) * float32(2^(127 - b)), -448, 448)), round to nearest, ties to even, by
torch's cast (whose own overflow goes to NaN, hence the clamp first; TE's cast is "satfinite", the same values). The
multiplier 2^(127 - b) is TE's exp2f_rcp (b = 254 gives 2^-127, a float32 subnormal; b = 255 gives NaN); multiplying by a
power of two is exact unless the product is below float32's normal range, where E4M3 holds zero anyway.

NaN and infinities: OUR policy (nonfinite="nan", the default). So a block (32 elements along the dot product) holding
any +-inf or NaN comes out as NaN in every element, E4M3 0x7F (one canonical NaN, positive: torch's cast of the
positive NaN this module writes; the Triton kernels store the byte itself), with the scale byte 0xFF (E8M0's NaN, X =
NaN: in OCP MX v1.0 a NaN shared scale makes every element of its block NaN). Either half alone makes every product of
the block NaN, so a GEMM consumer that reads only the scales, or only the elements, still turns every output the block
touches into NaN (q x X = NaN x NaN; the emulated GEMM, ced/model/fp8.py, has a NaN in every dot product the block
enters: tests/test_mxfp8_nonfinite.py). The flag is a per-block isfinite reduction and two wheres (fixed shapes); a
finite block's bytes are the TE rule's below, bit for bit (tests/golden/mxfp8_finite_553147e5edc9.json). Inputs are
read as float32 first, so a float64 value beyond float32's range is an infinity here.

Transformer Engine's policy (nonfinite="te": the record of what TE's own kernels do, kept for the TE backend's stand-in
tests; no model path uses it): amax is a max that ignores NaN (CUDA's fmaxf / __hmax, which TE's amax reductions use),
so a NaN element leaves its block's scale to the other elements and quantises to E4M3 NaN, which dequantises to NaN:
NaN propagates element by element. An infinite element makes amax and v infinite, b = 254 (2^127): the infinite
element saturates to +-448 (dequantised +-448 x 2^127, which overflows float32 to +-inf; inside a block-scaled MMA the
scale is applied to partial sums and the product with a small operand can stay finite) and the block's finite elements
are multiplied by 2^-127: each of at most 2^117 in magnitude becomes 0, a larger one (near float32's largest) keeps an
E4M3 value (3e38 gives 1.75). A block of NaNs only: amax 0, b = 0, NaN elements. A finite float32 block never reaches b
= 254 (float32's largest gives 247), so under TE's policy b = 254 marks exactly the blocks that held an infinity.

Inputs are read as float32 (a bf16 or float32 tensor is exact in float32; a float64 tensor of the CPU tests is rounded to
float32 first, as TE's cast kernels take float32, bf16 or fp16). Partial blocks (a dimension that is not a multiple of
32) are padded with zeros, which change no amax, and the padding is dropped: the hardware GEMM needs multiples of 32,
the reference does not.

Layouts: quantize(t, dim) returns q with t's shape (E4M3) and the biased exponents b (uint8) with t's shape but
ceil(n / 32) along dim. For a 2-D GEMM operand (M, K): dim -1 is TE's row-wise usage, b of shape (M, K / 32); dim 0 is
TE's column-wise usage, q still (M, K) and b of shape (M / 32, K): TE 2.20.2's compact (unswizzled) layout
(transformer_engine/pytorch/tensor/mxfp8_tensor.py MXFP8Quantizer.get_scale_shape, before its padding to multiples of
128 x 4 and 4 x 128, which the 4B shapes need none of).

The dequantised value q * 2^(b - 127) is exact in float32 for every b (an E4M3 value has at most 4 significant bits,
the smallest 2^-9), and exact in bf16 whenever b >= 3 or the block is all zeros (bf16's smallest subnormal is 2^-133):
a block with 0 < amax < 448 x 2^-125 (about 2^-116) is the only case a bf16 GEMM operand rounds (tests/test_mxfp8.py).

Fixed shapes: every shape follows from the input's (pad by a static amount, reshape, an amax and an isfinite-all over
the block axis, bit operations, a multiply, a clamp, wheres, a cast); no boolean-mask indexing or nonzero.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

BLOCK = 32
E4M3 = torch.float8_e4m3fn
FMAX = 448.0
EMAX_ELEM = 8  # the largest exponent of an E4M3 normal (448 = 1.75 x 2^8): OCP's emax_elem
RCP_BITS = 0x3B124925  # float32(1 / 448): Transformer Engine's Quantized_Limits<fp8e4m3>::max_norm_rcp
# The same number as a Python float: a float32 tensor times a Python scalar computes in float32 with the scalar rounded
# to float32 (eager PyTorch's opmath and Inductor's constants alike), so amax * RCP448 is TE's float32 product, with no
# tensor constant (no host-to-device copy a call; tests/test_mxfp8.py checks the bits and the product).
RCP448 = 1.0 / 448.0
RULES = ("te", "ocp")
NONFINITE = ("nan", "te")  # a block with an inf or NaN: all NaN (ours, the default), or TE's saturation (the record)
NAN_E4M3 = 0x7F  # the E4M3 byte of every element of a non-finite block (positive NaN)
NAN_E8M0 = 0xFF  # the scale byte of a non-finite block (E8M0 NaN)
_MANT = 0x7FFFFF


def block_amax(xb: torch.Tensor) -> torch.Tensor:
    """The largest magnitude over the last axis, NaN ignored (fmaxf semantics; an all-NaN block gives 0)."""
    a = xb.abs()
    return torch.where(torch.isnan(a), torch.zeros_like(a), a).amax(-1)


def e8m0(amax: torch.Tensor, rule: str = "te") -> torch.Tensor:
    """The biased E8M0 exponent b (int32 in 0..255) of the blocks whose largest magnitudes are amax (float32, >= 0 or
    inf), by Transformer Engine's rule ("te") or the OCP spec's ("ocp"); the module docstring states both."""
    amax = amax.float()
    if rule == "te":
        v = amax * RCP448
        bits = v.view(torch.int32)
        e = (bits >> 23) & 0xFF
        m = bits & _MANT
        up = (m > 0) & (e != 254) & ((e != 0) | (m > 0x400000))
        b = e + up.to(torch.int32)
        b = torch.where((e == 255) & (m == 0), torch.full_like(b, 254), b)  # inf: satfinite
        return torch.where((e == 255) & (m != 0), torch.full_like(b, 255), b)  # NaN
    if rule == "ocp":
        bits = amax.view(torch.int32)
        e = (bits >> 23) & 0xFF  # amax's biased exponent: floor(log2 amax) + 127 for a normal amax
        shared = torch.where(e > 0, e - 127 - EMAX_ELEM, torch.full_like(e, -127))  # zero or subnormal amax: 2^-127
        b = shared.clamp(-127, 127) + 127
        return torch.where(torch.isinf(amax) | torch.isnan(amax), torch.full_like(b, 255), b)
    raise ValueError(f"MXFP8 scale rule {rule!r}: one of {RULES}")


def scale_inverse(b: torch.Tensor) -> torch.Tensor:
    """float32 2^(127 - b), the multiplier applied before the cast (TE's exp2f_rcp): b = 254 gives 2^-127 (a float32
    subnormal), b = 255 NaN."""
    b = b.to(torch.int32)
    bits = torch.where(b == 254, torch.full_like(b, 0x00400000), (254 - b) << 23)
    bits = torch.where(b == 255, torch.full_like(b, 0x7FFFFFFF), bits)
    return bits.view(torch.float32)


def scale_value(b: torch.Tensor) -> torch.Tensor:
    """float32 2^(b - 127), the dequantisation scale X: b = 0 gives 2^-127 (a float32 subnormal), b = 255 NaN."""
    b = b.to(torch.int32)
    bits = torch.where(b == 0, torch.full_like(b, 0x00400000), b << 23)
    bits = torch.where(b == 255, torch.full_like(b, 0x7FFFFFFF), bits)
    return bits.view(torch.float32)


def _blocked(t: torch.Tensor, dim: int) -> tuple[torch.Tensor, int]:
    """t as float32 with dim moved last, zero-padded to a multiple of 32 and cut into blocks: (..., n_blocks, 32)."""
    x = t.float().movedim(dim, -1)
    n = x.shape[-1]
    pad = (-n) % BLOCK
    if pad:
        x = F.pad(x, (0, pad))
    return x.reshape(*x.shape[:-1], x.shape[-1] // BLOCK, BLOCK), n


def block_nonfinite(xb: torch.Tensor) -> torch.Tensor:
    """Per block (the last axis): whether any element is +-inf or NaN (bool, the block axis reduced)."""
    return ~torch.isfinite(xb).all(-1)


def quantize(t: torch.Tensor, dim: int = -1, rule: str = "te", nonfinite: str = "nan") -> tuple[torch.Tensor, torch.Tensor]:
    """t in MXFP8 with blocks of 32 along dim: (q, b). q has t's shape and dtype float8_e4m3fn (contiguous); b is the
    uint8 biased exponent of each block, t's shape with ceil(n / 32) along dim. The dequantised tensor is
    dequantize(q, b, dim). nonfinite: "nan" (a block with any inf or NaN: every element NaN, b = 0xFF; the module
    docstring) or "te" (Transformer Engine's own policy, the record)."""
    if nonfinite not in NONFINITE:
        raise ValueError(f"MXFP8 non-finite policy {nonfinite!r}: one of {NONFINITE}")
    xb, n = _blocked(t, dim)
    b = e8m0(block_amax(xb), rule)
    y = (xb * scale_inverse(b)[..., None]).clamp(-FMAX, FMAX)  # clamp keeps NaN
    if nonfinite == "nan":  # a finite block selects y and b unchanged: its bytes are TE's rule's, bit for bit
        bad = block_nonfinite(xb)
        y = torch.where(bad[..., None], float("nan"), y)  # the positive NaN: torch's cast gives 0x7F (NAN_E4M3)
        b = torch.where(bad, NAN_E8M0, b)
    q = y.to(E4M3).reshape(*xb.shape[:-2], -1)[..., :n].movedim(-1, dim).contiguous()
    return q, b.to(torch.uint8).movedim(-1, dim).contiguous()


def dequantize(q: torch.Tensor, b: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """q * 2^(b - 127) block by block along dim, in dtype (exact in float32; in bf16 see the module docstring). b =
    0xFF (a non-finite block) gives NaN for every element whatever q holds."""
    n = q.shape[dim]
    s = scale_value(b).movedim(dim, -1)
    s = s.unsqueeze(-1).expand(*s.shape, BLOCK).reshape(*s.shape[:-1], -1)[..., :n].movedim(-1, dim)
    return (q.float() * s).to(dtype)


def fake_quantize(t: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32, rule: str = "te",
                  nonfinite: str = "nan") -> torch.Tensor:
    """The dequantised MXFP8 tensor in dtype: what an MXFP8 GEMM multiplies."""
    q, b = quantize(t, dim, rule, nonfinite)
    return dequantize(q, b, dim, dtype)


def gemm(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """a @ b for dequantised MXFP8 operands (float32, exact): a (M, k), b (k, N). In float32 or float64 (no autocast:
    the model's own precision) with autocast off; otherwise (bf16 under autocast) a GEMM on the operands in out_dtype:
    the products exact, float32 accumulation, one rounding of the output, as the hardware's block-scaled GEMM (up to the
    order of accumulation; module docstring for when a bf16 operand is not exact)."""
    if out_dtype in (torch.float32, torch.float64):
        with torch.autocast(a.device.type, enabled=False):
            return torch.matmul(a.to(out_dtype), b.to(out_dtype))
    return torch.matmul(a.to(out_dtype), b.to(out_dtype))
