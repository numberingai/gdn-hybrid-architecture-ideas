"""Normalization layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_dtype(x: torch.Tensor) -> torch.dtype:
    """Norms and softmaxes run in at least float32 (float64 inputs stay float64 for exact tests)."""
    return torch.promote_types(x.dtype, torch.float32)


def _round_bf16_float32(x: torch.Tensor) -> torch.Tensor:
    """A float32 tensor rounded to the nearest bfloat16 value (ties to even), as float32, by integer arithmetic on its bits: add 0x7FFF plus the lowest kept
    bit to the 32-bit pattern and clear the low 16 bits (the rule of c10::BFloat16's conversion). Equal bit for bit to `x.to(bfloat16).to(float32)` for every
    finite and infinite input (the overflow of the largest finite values to infinity included); a NaN stays a NaN. WHY NOT THE CAST PAIR: torch.compile's
    Inductor removes `x.to(bfloat16).to(float32)` as a "pointless convert" (a joint-graph pattern for AMP's chains; measured on the CPU backend, 2026-10-27: the
    compiled output equals the unrounded one), so a stream rounded by the cast pair would be float32 under compile "all" and the run would train the control. The
    arithmetic is traced as written."""
    b = x.view(torch.int32)
    r = ((b + (((b >> 16) & 1) + 0x7FFF)) & -65536).view(torch.float32)
    return torch.where(torch.isnan(x), x, r)


def _round_bf16(x: torch.Tensor) -> torch.Tensor:
    return _round_bf16_float32(x) if x.dtype == torch.float32 else x.to(torch.bfloat16).to(x.dtype)


class _RoundStream(torch.autograd.Function):
    """The value rounded to bfloat16 and its gradient rounded to bfloat16 (the stream's gradient is held at the stream's precision wherever its value is)."""

    @staticmethod
    def forward(ctx, x):
        return _round_bf16(x)

    @staticmethod
    def backward(ctx, grad):
        return _round_bf16(grad)


def round_stream(x: torch.Tensor) -> torch.Tensor:
    """x held at bfloat16 precision (EncoderConfig.residual_dtype "bf16"): the value a bfloat16 tensor would store, in x's own dtype, and the gradient too.
    Equal to `x.to(bfloat16).to(x.dtype)` (an eager bfloat16 add computes in float32 and rounds once to nearest-even, which is this rounding of the float32
    sum), forward and backward; written as integer arithmetic for float32 so that compiled layers keep it (_round_bf16_float32). Other dtypes (the float64
    tests): the cast pair. Called only where the key is on; off, no statement of any forward changes."""
    return _RoundStream.apply(x)


class RMSNorm(nn.Module):
    """RMSNorm with a zero-centred gain: y = x / rms(x) * (1 + w), with w initialised to 0.

    Weight decay on w pulls the gain towards 1 instead of towards 0. This is the Qwen3-Next
    convention (zero-centred RMSNorm with weight decay on norm weights), used here for every norm.

    gain: "channel" (the default: w has shape (dim,), one gain per channel) or "scalar" (w has shape (1,), ONE gain
    for the whole norm, broadcast over the channels by the same multiply; EncoderConfig.block_norm_gain). The forward
    is the same code for both, so "channel" is the norm it always was, bit for bit.
    """

    GAINS = ("channel", "scalar")

    def __init__(self, dim: int, eps: float = 1e-6, gain: str = "channel"):
        super().__init__()
        assert gain in self.GAINS, f"RMSNorm gain {gain!r}: one of {self.GAINS}"
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim if gain == "channel" else 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = compute_dtype(x)
        xf = x.to(dtype)
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.to(dtype))).to(x.dtype)


class GatedRMSNorm(RMSNorm):
    """The GDN output norm: RMSNorm over the value-head dimension, then multiplied by act(z).

    act = SiLU (gate "silu", v1 and the Hugging Face Qwen3.5 code) or sigmoid (gate "sigmoid",
    architecture spec v2). Qwen's code initialises this gain to ones and uses a plain multiply.
    Here it is zero-centred like every other norm, so a single "weight decay on norm gains" rule is
    safe everywhere.
    """

    def __init__(self, dim: int, eps: float = 1e-6, gate: str = "silu", gain: str = "channel"):
        """gain: as RMSNorm's ("scalar": one gain for the whole norm; EncoderConfig.block_norm_gain)."""
        super().__init__(dim, eps, gain=gain)
        assert gate in ("silu", "sigmoid"), f"unknown GDN output gate {gate!r}"
        self.gate = gate

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        act = F.silu(z) if self.gate == "silu" else torch.sigmoid(z)
        return super().forward(x) * act
