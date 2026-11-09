"""Partial rotary position embedding (NeoX layout), applied on document-relative positions."""

from __future__ import annotations

import torch


def rotary_dims(head_dim: int, fraction: float) -> int:
    """Number of leading head dimensions that rotate (even). 0 means NoPE."""
    r = int(head_dim * fraction)
    return r - (r % 2)


def apply_rope(x: torch.Tensor, pos: torch.Tensor, fraction: float, theta: float) -> torch.Tensor:
    """Rotate the first `rotary_dims` dimensions of every head of x.

    x:   (B, L, H, head_dim)
    pos: (B, L) integer positions. Positions restart at 0 for every document, so a packed row
         and the same documents run one by one see the same angles.

    Dimension i is paired with i + r/2 inside the rotating block (NeoX "rotate_half"), as in the
    Qwen3-Next / Qwen3.5 code. The remaining dimensions pass through unchanged.
    """
    r = rotary_dims(x.shape[-1], fraction)
    if r == 0:
        return x
    inv_freq = theta ** (-torch.arange(0, r, 2, dtype=torch.float64, device=x.device) / r)
    angles = pos.to(torch.float64)[..., None] * inv_freq  # (B, L, r/2)
    angles = torch.cat([angles, angles], dim=-1)[:, :, None, :]  # (B, L, 1, r)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)

    x_rot, x_pass = x[..., :r], x[..., r:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    rotated_half = torch.cat([-x2, x1], dim=-1)
    return torch.cat([x_rot * cos + rotated_half * sin, x_pass], dim=-1)


def rope_tables(pos: torch.Tensor, head_dim: int, fraction: float, theta: float, dtype: torch.dtype
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin), each (B, L, 1, r) in dtype: the tables apply_rope computes for positions pos (B, L), by its own
    operations in its own order (float64 angles, then the cast). r = rotary_dims(head_dim, fraction) > 0."""
    r = rotary_dims(head_dim, fraction)
    assert r > 0, "no rotary dimensions: nothing to tabulate"
    inv_freq = theta ** (-torch.arange(0, r, 2, dtype=torch.float64, device=pos.device) / r)
    angles = pos.to(torch.float64)[..., None] * inv_freq  # (B, L, r/2)
    angles = torch.cat([angles, angles], dim=-1)[:, :, None, :]  # (B, L, 1, r)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate_with_tables(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """apply_rope's rotation of x (B, L, H, head_dim) with given tables (B or 1, L or 1, 1, r): its last four lines.
    With tables in x's dtype this is apply_rope bit for bit (float32, float64). With float32 tables and a bfloat16 x
    (the capped read under autocast) the rotated block is computed in float32 and rounded once."""
    r = cos.shape[-1]
    x_rot, x_pass = x[..., :r], x[..., r:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    rotated_half = torch.cat([-x2, x1], dim=-1)
    return torch.cat([(x_rot * cos + rotated_half * sin).to(x.dtype), x_pass], dim=-1)
