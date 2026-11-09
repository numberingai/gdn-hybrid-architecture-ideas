"""Training loss: decoder-head and aux-head cross-entropy, plus z-loss.

For Looped-CED, `split` marks the positions scored by the decoder head (the kept decoder
positions); every other position is scored by the aux encoder head with weight `ctx_weight`.
For the control there is one head, `split` is the data's decoder mask (only used to report CE at
the same positions as Looped-CED), and ctx_weight is 1, which makes the loss a plain token mean.

    global_token_mean: (sum_dec obj + w * sum_ctx obj) / (n_dec + n_ctx)
    per_head_mean:     mean_dec obj + w * mean_ctx obj
    obj = CE + z_coef * logsumexp(logits)^2

Targets that cross a document boundary are invalid and get weight 0 in both heads.
"""

from __future__ import annotations

import torch


def lm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    split: torch.Tensor,
    ctx_weight: float,
    z_coef: float,
    normalization: str = "global_token_mean",
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    lse = torch.logsumexp(logits, dim=-1)
    ce = lse - logits.gather(-1, targets[..., None]).squeeze(-1)
    objective = ce + z_coef * lse.pow(2)

    on_dec = valid & split
    on_ctx = valid & ~split
    zero = torch.zeros((), dtype=ce.dtype, device=ce.device)
    sum_dec = torch.where(on_dec, objective, zero).sum()
    sum_ctx = torch.where(on_ctx, objective, zero).sum()
    n_dec = on_dec.sum()
    n_ctx = on_ctx.sum()

    if normalization == "global_token_mean":
        loss = (sum_dec + ctx_weight * sum_ctx) / (n_dec + n_ctx).clamp(min=1)
    elif normalization == "per_head_mean":
        loss = sum_dec / n_dec.clamp(min=1) + ctx_weight * sum_ctx / n_ctx.clamp(min=1)
    else:
        raise ValueError(f"unknown normalization {normalization!r}")

    with torch.no_grad():
        metrics = {
            "ce_dec": (torch.where(on_dec, ce, zero).sum() / n_dec.clamp(min=1)).item(),
            "ce_ctx": (torch.where(on_ctx, ce, zero).sum() / n_ctx.clamp(min=1)).item(),
            "ce_all": (torch.where(valid, ce, zero).sum() / valid.sum().clamp(min=1)).item(),
            "lse_sq": (torch.where(valid, lse.pow(2), zero).sum() / valid.sum().clamp(min=1)).item(),
            "n_dec": int(n_dec),
            "n_ctx": int(n_ctx),
        }
    return loss, metrics
