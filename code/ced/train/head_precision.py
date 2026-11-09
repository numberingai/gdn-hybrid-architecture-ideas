"""The precision of the LM head's three matmuls in the training loss (TrainConfig.head_precision).

Item 17: such a change "that is not bit-for-bit" is allowed "if a 60M short run
(~500 steps) stays within the two-seed rounding band, as F3 was checked" (scripts/band_check_60m.py).

    "fp32" (the default, every run before the key): the head's matmuls take float32 operands (float64 for float64
           inputs), as the code always did; on CUDA cuBLAS uses TF32 for them when TrainConfig.tf32 is on (the global
           torch setting the trainer makes). Nothing here runs: the callers keep their own code for it.
    "bf16": each operand of the three matmuls is rounded once to bfloat16 (round to nearest even):
               logits = bf16(h) @ bf16(W)^T,  dh = bf16(G) @ bf16(W),  dW += bf16(G)^T @ bf16(h),
           and the products are accumulated in the compute dtype (float32; float64 for float64 inputs, the tests'
           exact form of the same function) into outputs of that dtype: the logits, dh and dW (dW also across the
           fused head's chunks). The logits' softmax, log-sum-exp, CE, z-loss and the logit gradient G are computed in
           the compute dtype from those logits; G is rounded to bf16 only as an operand of the two backward matmuls.

A product of two bfloat16 numbers (8-bit significands) is exact in float32 (24 bits), so on every device the "bf16"
matmul is: exact products of the rounded operands, summed in float32. On CUDA it is one cuBLAS call with bf16 inputs
and a float32 output (torch.mm / torch.addmm with out_dtype=torch.float32, torch >= 2.8); elsewhere the rounded
operands are upcast and multiplied in the compute dtype (the same products; only the summation order can differ).

This module is the readable reference (`head_linear`, an autograd function) and the two matmul helpers the fused head
uses (`mm_acc`, `addmm_acc_`). ced/train/chunked_loss.py's chunked head (the trainer's head_impl "chunked") is built on
`head_linear`; ced/train/fused_head.py's fused head (head_impl "fused") on the helpers.
"""

from __future__ import annotations

import torch

PRECISIONS = ("fp32", "bf16")
OPERAND = torch.bfloat16


def check_head_precision(precision: str) -> str:
    """ValueError unless `precision` is one of PRECISIONS; returns it."""
    if precision not in PRECISIONS:
        raise ValueError(f"head_precision {precision!r}: one of {PRECISIONS} ('fp32': float32 operands, TF32 where "
                         "TrainConfig.tf32 allows; 'bf16': bf16 operands, float32 accumulation and float32 softmax/CE)")
    return precision


def _cuda_out_dtype(a: torch.Tensor, acc: torch.dtype) -> bool:
    """True where the matmul is one cuBLAS call with bf16 operands and a float32 output."""
    return a.is_cuda and acc == torch.float32


def mm_acc(a: torch.Tensor, b: torch.Tensor, acc: torch.dtype, out: torch.Tensor | None = None) -> torch.Tensor:
    """a @ b for bf16 a (m, k) and b (k, n): exact products of the operands summed in `acc` (float32 or float64), an
    (m, n) result of dtype acc (written into `out` when given)."""
    assert a.dtype == OPERAND and b.dtype == OPERAND, (a.dtype, b.dtype)
    if _cuda_out_dtype(a, acc):
        return torch.mm(a, b, out_dtype=torch.float32) if out is None else torch.mm(a, b, out_dtype=torch.float32, out=out)
    return torch.mm(a.to(acc), b.to(acc)) if out is None else torch.mm(a.to(acc), b.to(acc), out=out)


# On CUDA, `addmm_acc_` accumulates in the GEMM's epilogue (beta = 1 on the float32 accumulator, torch.addmm with
# out_dtype and out=acc). If this torch refuses that call, the product is formed in float32 and added (one more pass
# over acc; the same sum, acc + fl32(a @ b)). Probed once a process, on the first call: None (not probed), True, False.
ADDMM_OUT_DTYPE: bool | None = None


def _probe_addmm(device) -> bool:
    """Whether torch.addmm(acc, a, b, out_dtype=torch.float32, out=acc) runs here and equals acc + a @ b."""
    try:
        g = torch.Generator(device="cpu").manual_seed(0)
        a = torch.randn(64, 48, generator=g).to(device=device, dtype=OPERAND)
        b = torch.randn(48, 32, generator=g).to(device=device, dtype=OPERAND)
        acc = torch.randn(64, 32, generator=g).to(device=device)
        want = acc + torch.mm(a, b, out_dtype=torch.float32)
        got = acc.clone()
        torch.addmm(got, a, b, out_dtype=torch.float32, out=got)
        return bool(torch.allclose(got, want, rtol=1e-5, atol=1e-5))
    except (RuntimeError, NotImplementedError, TypeError):
        return False


def addmm_acc_(acc: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """acc += a @ b in place: bf16 a (m, k), b (k, n), acc (m, n) float32 or float64; the products exact, summed in acc's
    dtype."""
    global ADDMM_OUT_DTYPE
    assert a.dtype == OPERAND and b.dtype == OPERAND, (a.dtype, b.dtype)
    if _cuda_out_dtype(a, acc.dtype):
        if ADDMM_OUT_DTYPE is None:
            ADDMM_OUT_DTYPE = _probe_addmm(acc.device)
        if ADDMM_OUT_DTYPE:
            return torch.addmm(acc, a, b, out_dtype=torch.float32, out=acc)
        return acc.add_(torch.mm(a, b, out_dtype=torch.float32))
    return acc.addmm_(a.to(acc.dtype), b.to(acc.dtype))


class _Bf16Linear(torch.autograd.Function):
    """logits = bf16(h) @ bf16(W)^T accumulated in h's dtype (float32 or float64); backward: G rounded to bf16, then
    dh = bf16(G) @ bf16(W) and dW = bf16(G)^T @ bf16(h), accumulated in the same dtype. h (n, d), W (V, d)."""

    @staticmethod
    def forward(ctx, h, weight):
        acc = h.dtype
        hb, wb = h.to(OPERAND), weight.to(acc).to(OPERAND)
        ctx.save_for_backward(hb, wb)
        ctx.dtypes = (h.dtype, weight.dtype)
        return mm_acc(hb, wb.t(), acc)

    @staticmethod
    def backward(ctx, g):
        hb, wb = ctx.saved_tensors
        h_dtype, w_dtype = ctx.dtypes
        gb = g.to(OPERAND)
        dh = mm_acc(gb, wb, g.dtype).to(h_dtype) if ctx.needs_input_grad[0] else None
        dw = mm_acc(gb.t(), hb, g.dtype).to(w_dtype) if ctx.needs_input_grad[1] else None
        return dh, dw


def head_linear(h: torch.Tensor, weight: torch.Tensor, precision: str) -> torch.Tensor:
    """The reference head matmul: logits (..., V) of h (..., d) and weight (V, d), with autograd. h must already be in
    the compute dtype (float32 or float64); the caller turns autocast off. "fp32": F.linear(h, weight in h's dtype), the
    code of every run before the key; "bf16": _Bf16Linear (the module docstring)."""
    if precision == "fp32":
        return torch.nn.functional.linear(h, weight.to(h.dtype))
    check_head_precision(precision)
    shape = h.shape
    return _Bf16Linear.apply(h.reshape(-1, shape[-1]), weight).reshape(*shape[:-1], weight.shape[0])
