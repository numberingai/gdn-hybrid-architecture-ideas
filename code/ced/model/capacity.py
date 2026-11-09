"""Fixed-capacity selection of decoder positions, with gather and scatter.

The decoder runs only at decoder positions. To keep every tensor shape fixed during training (no
boolean-mask indexing, no nonzero), each packed row of length L reserves exactly C decoder slots:

    C = min(L, round_up(ceil(p * L), multiple))

Slot selection keeps the FIRST C decoder positions of each row, in order. "First C" is
prefix-stable: whether position t is kept depends only on dec_mask[:t+1], so a later position can
never cause an earlier one to be dropped. That keeps the selection causal.

- Empty slots (a row with fewer than C decoder positions) point at a sentinel row L. Gathers read
  a zero row there; scatters write into a trash row that is then cut off.
- Positions beyond the first C are dropped: they are treated as context positions (they offer a
  memory KV and get the aux-head loss). The pretraining span sampler emits exactly C decoder
  positions per row, so drops only happen as a fallback; the count is returned for logging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def decoder_capacity(length: int, fraction: float, multiple: int) -> int:
    return min(length, math.ceil(math.ceil(fraction * length) / multiple) * multiple)


@dataclass
class Slots:
    index: torch.Tensor  # (B, C) position of each slot, or L for an empty slot
    valid: torch.Tensor  # (B, C) bool, False for empty slots
    kept: torch.Tensor  # (B, L) bool, True at positions that run the decoder
    n_dropped: torch.Tensor  # (B,) decoder positions dropped because the row exceeded C


def select_slots(dec_mask: torch.Tensor, capacity: int) -> Slots:
    """Assign the first `capacity` decoder positions of each row to slots, in order."""
    B, L = dec_mask.shape
    positions = torch.arange(L, device=dec_mask.device).expand(B, L)
    rank = torch.cumsum(dec_mask.long(), dim=1) - 1  # rank among this row's decoder positions
    keep = dec_mask & (rank < capacity)
    slot = torch.where(keep, rank, torch.full_like(rank, capacity))  # slot `capacity` is a trash slot
    index = torch.full((B, capacity + 1), L, dtype=torch.long, device=dec_mask.device)
    index = index.scatter(1, slot, positions)[:, :capacity]
    valid = index < L
    kept = torch.zeros(B, L + 1, dtype=torch.bool, device=dec_mask.device)
    kept = kept.scatter(1, index, valid)[:, :L]
    n_dropped = dec_mask.long().sum(1) - kept.long().sum(1)
    return Slots(index=index, valid=valid, kept=kept, n_dropped=n_dropped)


def _expand_index(index: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return index.view(*index.shape, *([1] * (x.dim() - 2))).expand(*index.shape, *x.shape[2:])


def gather_slots(x: torch.Tensor, index: torch.Tensor, fill=0) -> torch.Tensor:
    """x (B, L, ...) -> (B, C, ...); empty slots (index L) read `fill`."""
    pad = torch.full_like(x[:, :1], fill)
    padded = torch.cat([x, pad], dim=1)
    return torch.gather(padded, 1, _expand_index(index, padded))


def scatter_slots(base: torch.Tensor, index: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Write values (B, C, ...) into a copy of base (B, L, ...) at the slot positions.

    Empty slots write into a trash row that is cut off, so they never touch base.
    """
    padded = torch.cat([base, torch.zeros_like(base[:, :1])], dim=1)
    padded = padded.scatter(1, _expand_index(index, values), values)
    return padded[:, :-1]


# ------------------------------------------------------------------------------------------------ Gathers
# with an inverse map (control (c)'s pass 2, ced/model/second_pass.py). torch.gather's backward is a
# scatter-add, and a scatter is an index_put; under torch.use_deterministic_algorithms (training) both run
# PyTorch's sort-based deterministic kernel over every element of the expanded index, which measured about
# 0.8 s of (c)'s 3.06 s step at 7 layers. Slot positions are distinct, so every gather here has an exact
# inverse, and its backward is another gather: the same values (each gradient row has at most one source,
# so the scatter-add's sum is that one value), with no sort and no accumulation.


def slot_inverse(index: torch.Tensor, length: int) -> torch.Tensor:
    """index (B, C): distinct positions < length per row, or `length` for an empty slot (any number of
    them). Returns (B, length) long: the slot whose index is each position, or C where there is none."""
    B, C = index.shape
    inv = torch.full((B, length + 1), C, dtype=torch.long, device=index.device)
    # empty slots all write column `length`, which is cut off: only distinct indices survive
    inv = inv.scatter(1, index, torch.arange(C, device=index.device).expand(B, C))
    return inv[:, :length]


class _InverseGather(torch.autograd.Function):
    """out[b, m] = cat(x, fill row)[b, fwd[b, m]]; the backward is dx[b, n] = cat(g, zero row)[b, bwd[b, n]].
    Exact when bwd is fwd's inverse: for every n < N, bwd[b, n] is the one m with fwd[b, m] = n, or M if
    no m reads row n (x (B, N, ...), fwd (B, M) in [0, N], bwd (B, N) in [0, M]); rows m with fwd = N read
    the fill row and send no gradient."""

    @staticmethod
    def forward(ctx, x, fwd, bwd, fill):
        padded = torch.cat([x, torch.full_like(x[:, :1], fill)], dim=1)
        ctx.save_for_backward(bwd)
        return torch.gather(padded, 1, _expand_index(fwd, padded))

    @staticmethod
    def backward(ctx, g):
        (bwd,) = ctx.saved_tensors
        padded = torch.cat([g, torch.zeros_like(g[:, :1])], dim=1)
        return torch.gather(padded, 1, _expand_index(bwd, padded)), None, None, None


def gather_rows(x: torch.Tensor, fwd: torch.Tensor, bwd: torch.Tensor, fill=0) -> torch.Tensor:
    """x (B, N, ...) -> (B, M, ...) rows fwd (N: the fill row), with the exact inverse map bwd (B, N) for the
    backward (see _InverseGather). Same values and gradients as torch.gather over cat(x, fill row)."""
    if not x.requires_grad:
        padded = torch.cat([x, torch.full_like(x[:, :1], fill)], dim=1)
        return torch.gather(padded, 1, _expand_index(fwd, padded))
    return _InverseGather.apply(x, fwd, bwd, fill)


def gather_slots_inv(x: torch.Tensor, index: torch.Tensor, inverse: torch.Tensor, fill=0) -> torch.Tensor:
    """gather_slots with the slots' inverse map (slot_inverse(index, L)): same values and gradients."""
    return gather_rows(x, index, inverse, fill)


class _SlotGather(torch.autograd.Function):
    """gather_slots(x, index) (fill 0) whose backward is a gather through the inverse map instead of
    torch.gather's scatter-add (decoder optimisation r1: CED v2's slot gather of h_enc; TrainConfig.decoder_opt).

    torch.gather's backward scatter-adds the gradient into zeros. Under torch.use_deterministic_algorithms that
    is the sort-based index_put over every element of the expanded index (B x C x d keys): about 46 ms per v2
    step at C = 2,048 and four times that at C = L. Slot positions are distinct, so every position receives at
    most one slot's gradient: the scatter-add's value there is 0 + g, and 0 where no slot reads the position.
    The backward here gathers g through the inverse map and adds 0.0, which is that same sum bit for bit (it
    turns a -0.0 into +0.0, as 0 + g does). Empty slots (index L) read the zero row and send nothing."""

    @staticmethod
    def forward(ctx, x, index, inverse):
        padded = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
        ctx.save_for_backward(inverse)
        return torch.gather(padded, 1, _expand_index(index, padded))

    @staticmethod
    def backward(ctx, g):
        (inverse,) = ctx.saved_tensors
        padded = torch.cat([g, torch.zeros_like(g[:, :1])], dim=1)
        return torch.gather(padded, 1, _expand_index(inverse, padded)) + 0.0, None, None


def gather_slots_exact(x: torch.Tensor, index: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    """gather_slots(x, index) (empty slots read 0) with the inverse map inverse = slot_inverse(index, L): the same
    values and the same gradient bits, with a gather as the backward (_SlotGather)."""
    if not (torch.is_grad_enabled() and x.requires_grad):
        return gather_slots(x, index)
    return _SlotGather.apply(x, index, inverse)


def scatter_slots_inv(values: torch.Tensor, index: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    """scatter_slots into a zero base: values (B, C, ...) at the slot positions of a (B, L, ...) tensor of
    zeros, written as the inverse gather (row L of the result's inverse is the zero row): same values and
    gradients."""
    return gather_rows(values, inverse, index, 0)
