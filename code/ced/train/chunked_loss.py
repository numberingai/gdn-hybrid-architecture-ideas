"""The training loss computed from pre-head hidden states, without materialising full logits.

At vocab 128k and rows of 8k positions, full logits are 4 GB per row in float32 (the head is FP32).
`token_ce` computes per-position cross-entropy and logsumexp chunk by chunk under activation
checkpointing: each chunk's logits exist only while that chunk's forward or backward runs. The math
is exactly `ced.train.loss.lm_loss` on the full logits (tests/test_chunked_loss.py checks values
and gradients in float64).

The head matmul runs with autocast disabled, in float32 (or wider) inputs. Whether a CUDA float32
matmul uses TF32 tensor cores is the global torch setting the trainer records in its config.

precision (TrainConfig.head_precision; ced/train/head_precision.py): "fp32" (the default) is the code above, unchanged;
"bf16" computes each chunk's logits with head_precision.head_linear (bf16 operands, float32 accumulation, the logit
gradient rounded to bf16 for the two backward matmuls) and everything after the logits as before, in float32. It is the
reference of the fused head's "bf16" form (ced/train/fused_head.py) and is never compiled (the trainer's compiled chunk
function is the "fp32" one).

ce_form (TrainConfig.loss.ce_form; ced/train/ce_fix.py): "standard" (the default of every function below: the code above, unchanged)
or "fixed" (Barron et al.'s hard-target cross-entropy, softplus(LSE_{j != t}(z_j) - z_t); the chunk functions with the suffix
_fixed). Only the trainer's training-loss calls pass "fixed"; the eval scorers call these functions at the default.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from ced.model.norms import compute_dtype
from ced.train.ce_fix import check_ce_form, fixed_ce_lse, fixed_multi_ce_lse
from ced.train.head_precision import check_head_precision, head_linear


def _chunk_ce(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = compute_dtype(h)
    with torch.autocast(device_type=h.device.type, enabled=False):
        logits = torch.nn.functional.linear(h.to(dtype), weight.to(dtype))
        lse = torch.logsumexp(logits, dim=-1)
        ce = lse - logits.gather(-1, targets[..., None]).squeeze(-1)
    return ce, lse


def _chunk_ce_bf16(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """_chunk_ce with the logits from bf16 operands (head_precision "bf16")."""
    dtype = compute_dtype(h)
    with torch.autocast(device_type=h.device.type, enabled=False):
        logits = head_linear(h.to(dtype), weight, "bf16")
        lse = torch.logsumexp(logits, dim=-1)
        ce = lse - logits.gather(-1, targets[..., None]).squeeze(-1)
    return ce, lse


def _chunk_ce_fixed(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """_chunk_ce with the fixed cross-entropy (ce_form "fixed"): ce = softplus(LSE_{j != t}(z_j) - z_t), lse the row's
    logaddexp(LSE_{j != t}(z_j), z_t); ced/train/ce_fix.py fixed_ce_lse."""
    dtype = compute_dtype(h)
    with torch.autocast(device_type=h.device.type, enabled=False):
        logits = torch.nn.functional.linear(h.to(dtype), weight.to(dtype))
        ce, lse = fixed_ce_lse(logits, targets)
    return ce, lse


def _chunk_ce_bf16_fixed(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """_chunk_ce_bf16 with the fixed cross-entropy."""
    dtype = compute_dtype(h)
    with torch.autocast(device_type=h.device.type, enabled=False):
        logits = head_linear(h.to(dtype), weight, "bf16")
        ce, lse = fixed_ce_lse(logits, targets)
    return ce, lse


_CHUNK_CE_FIXED_EAGER = _chunk_ce_fixed  # the plain function; Trainer(compile=...) may swap a compiled one in

HEAD_CHUNK = 1024


def head_positions(mask: torch.Tensor, multiple: int = HEAD_CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """Host side: (indices, real) for the True entries of a flat bool mask, padded with index 0 to a
    multiple of `multiple` so the head sees few distinct shapes. real marks the true entries."""
    idx = torch.nonzero(mask.reshape(-1)).squeeze(1)
    n = idx.numel()
    padded = max(multiple, -(-n // multiple) * multiple)
    out = torch.zeros(padded, dtype=torch.long)
    out[:n] = idx
    real = torch.zeros(padded, dtype=torch.bool)
    real[:n] = True
    return out, real


def selected_ce(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, idx: torch.Tensor,
                real: torch.Tensor, precision: str = "fp32", ce_form: str = "standard") -> tuple[torch.Tensor, torch.Tensor]:
    """(ce, lse) at the selected flat positions only, zero at padding entries. precision, ce_form: as token_ce's."""
    h = hidden.reshape(-1, hidden.shape[-1]).index_select(0, idx)
    t = targets.reshape(-1).index_select(0, idx)
    ce, lse = token_ce(h, weight, t, precision=precision, ce_form=ce_form)
    zero = torch.zeros((), dtype=ce.dtype, device=ce.device)
    return torch.where(real, ce, zero), torch.where(real, lse, zero)


_CHUNK_CE_EAGER = _chunk_ce  # the plain function; Trainer(compile=...) may swap a compiled one in


def token_ce(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk: int = 1024, precision: str = "fp32",
    ce_form: str = "standard",
) -> tuple[torch.Tensor, torch.Tensor]:
    """hidden (..., d), weight (V, d), targets (...) -> (ce, lse), each (...), float32 or wider.

    Equal to logsumexp(logits) - logits[target] and logsumexp(logits) with logits = hidden @ weight.T,
    computed `chunk` positions at a time. With grad enabled each chunk is checkpointed, so backward
    recomputes its logits instead of storing them. precision: "fp32" (this module's _chunk_ce, as always) or
    "bf16" (_chunk_ce_bf16; the module docstring). ce_form: "standard" (as always) or "fixed" (loss.ce_form; the same
    chunks with _chunk_ce_fixed / _chunk_ce_bf16_fixed: the value differs from the standard one only at the float32
    cliff; the gradient's target element is the page's).
    """
    if check_ce_form(ce_form) == "fixed":
        fn = _chunk_ce_fixed if check_head_precision(precision) == "fp32" else _chunk_ce_bf16_fixed
    else:
        fn = _chunk_ce if check_head_precision(precision) == "fp32" else _chunk_ce_bf16
    shape = targets.shape
    h = hidden.reshape(-1, hidden.shape[-1])
    t = targets.reshape(-1)
    ces, lses = [], []
    for start in range(0, h.shape[0], chunk):
        hs, ts = h[start : start + chunk], t[start : start + chunk]
        if torch.is_grad_enabled() and (hs.requires_grad or weight.requires_grad):
            ce, lse = checkpoint(fn, hs, weight, ts, use_reentrant=False)
        else:
            ce, lse = fn(hs, weight, ts)
        ces.append(ce)
        lses.append(lse)
    return torch.cat(ces).reshape(shape), torch.cat(lses).reshape(shape)


def _chunk_lse_zy(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """_chunk_ce's numbers as (logsumexp, target logit) instead of (ce, logsumexp): the Copy-Paste head's mixture
    (ced/model/cp_head.py) reads LSE_V and z_y separately."""
    dtype = compute_dtype(h)
    with torch.autocast(device_type=h.device.type, enabled=False):
        logits = torch.nn.functional.linear(h.to(dtype), weight.to(dtype))
        lse = torch.logsumexp(logits, dim=-1)
        zy = logits.gather(-1, targets[..., None]).squeeze(-1)
    return lse, zy


def _chunk_lse_zy_bf16(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """_chunk_lse_zy with the logits from bf16 operands (head_precision "bf16")."""
    dtype = compute_dtype(h)
    with torch.autocast(device_type=h.device.type, enabled=False):
        logits = head_linear(h.to(dtype), weight, "bf16")
        lse = torch.logsumexp(logits, dim=-1)
        zy = logits.gather(-1, targets[..., None]).squeeze(-1)
    return lse, zy


def token_lse_zy(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk: int = 1024,
                 precision: str = "fp32") -> tuple[torch.Tensor, torch.Tensor]:
    """token_ce's chunked head with (LSE_V, z_y) as its outputs: hidden (..., d), weight (V, d), targets (...) ->
    (logsumexp(logits), logits[target]), each (...), float32 or wider; each chunk checkpointed when a gradient is
    needed. The Copy-Paste head's mixture reads these two (ced/model/cp_head.py); nothing else calls it. precision: as
    token_ce's."""
    fn = _chunk_lse_zy if check_head_precision(precision) == "fp32" else _chunk_lse_zy_bf16
    shape = targets.shape
    h = hidden.reshape(-1, hidden.shape[-1])
    t = targets.reshape(-1)
    lses, zys = [], []
    for start in range(0, h.shape[0], chunk):
        hs, ts = h[start : start + chunk], t[start : start + chunk]
        if torch.is_grad_enabled() and (hs.requires_grad or weight.requires_grad):
            lse, zy = checkpoint(fn, hs, weight, ts, use_reentrant=False)
        else:
            lse, zy = fn(hs, weight, ts)
        lses.append(lse)
        zys.append(zy)
    return torch.cat(lses).reshape(shape), torch.cat(zys).reshape(shape)


def combine_loss(
    ce: torch.Tensor,
    lse: torch.Tensor,
    valid: torch.Tensor,
    split: torch.Tensor,
    ctx_weight: float,
    z_coef: float,
    normalization: str = "global_token_mean",
    ce_split: torch.Tensor | None = None,
    lse_split: torch.Tensor | None = None,
    valid_split: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """`lm_loss` from per-position (ce, lse).

    One-head form (control): ce, lse, valid, split are (B, L); positions with split are the "dec"
    group, the rest the "ctx" group, all from the same head.

    Two-head form (Looped-CED): (ce, lse) are the aux head at every position (B, L); split marks the
    positions that ran the decoder; (ce_split, lse_split, valid_split) are the decoder head at the
    slots (B, C). The dec group is the valid slots, the ctx group the valid non-decoder positions.
    This is lm_loss(scatter(aux_logits, slots, dec_logits), ...) without the scatter.

    Metrics are returned as 0-dim tensors (no host sync here); the caller converts them.
    """
    zero = torch.zeros((), dtype=ce.dtype, device=ce.device)
    obj = ce + z_coef * lse.pow(2)
    on_ctx = valid & ~split
    if ce_split is None:
        on_dec = valid & split
        dec_ce, dec_obj, dec_lse = ce, obj, lse
    else:
        on_dec = valid_split
        dec_ce, dec_lse = ce_split, lse_split
        dec_obj = ce_split + z_coef * lse_split.pow(2)
    sum_dec = torch.where(on_dec, dec_obj, zero).sum()
    sum_ctx = torch.where(on_ctx, obj, zero).sum()
    n_dec = on_dec.sum()
    n_ctx = on_ctx.sum()
    if normalization == "global_token_mean":
        loss = (sum_dec + ctx_weight * sum_ctx) / (n_dec + n_ctx).clamp(min=1)
    elif normalization == "per_head_mean":
        loss = sum_dec / n_dec.clamp(min=1) + ctx_weight * sum_ctx / n_ctx.clamp(min=1)
    else:
        raise ValueError(f"unknown normalization {normalization!r}")
    with torch.no_grad():
        sum_ce_dec = torch.where(on_dec, dec_ce, zero).sum()
        sum_ce_ctx = torch.where(on_ctx, ce, zero).sum()
        lse_sq = torch.where(on_dec, dec_lse.pow(2), zero).sum() + torch.where(on_ctx, lse.pow(2), zero).sum()
        n_all = (n_dec + n_ctx).clamp(min=1)
        metrics = {
            "ce_dec": sum_ce_dec / n_dec.clamp(min=1),
            "ce_ctx": sum_ce_ctx / n_ctx.clamp(min=1),
            "ce_all": (sum_ce_dec + sum_ce_ctx) / n_all,
            "lse_sq": lse_sq / n_all,
            "n_dec": n_dec,
            "n_ctx": n_ctx,
        }
    return loss, metrics


def _chunk_multi_ce(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """_chunk_ce with K targets per row: targets (n, K) -> ce (n, K), lse (n,)."""
    dtype = compute_dtype(h)
    with torch.autocast(device_type=h.device.type, enabled=False):
        logits = torch.nn.functional.linear(h.to(dtype), weight.to(dtype))
        lse = torch.logsumexp(logits, dim=-1)
        ce = lse[:, None] - logits.gather(-1, targets)
    return ce, lse


def _chunk_multi_ce_bf16(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """_chunk_multi_ce with the logits from bf16 operands (head_precision "bf16")."""
    dtype = compute_dtype(h)
    with torch.autocast(device_type=h.device.type, enabled=False):
        logits = head_linear(h.to(dtype), weight, "bf16")
        lse = torch.logsumexp(logits, dim=-1)
        ce = lse[:, None] - logits.gather(-1, targets)
    return ce, lse


def _chunk_multi_ce_fixed(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """_chunk_multi_ce with the fixed cross-entropy on each of the K terms (ce_form "fixed"; ce_fix.fixed_multi_ce_lse)."""
    dtype = compute_dtype(h)
    with torch.autocast(device_type=h.device.type, enabled=False):
        logits = torch.nn.functional.linear(h.to(dtype), weight.to(dtype))
        ce, lse = fixed_multi_ce_lse(logits, targets)
    return ce, lse


def _chunk_multi_ce_bf16_fixed(h: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """_chunk_multi_ce_bf16 with the fixed cross-entropy on each of the K terms."""
    dtype = compute_dtype(h)
    with torch.autocast(device_type=h.device.type, enabled=False):
        logits = head_linear(h.to(dtype), weight, "bf16")
        ce, lse = fixed_multi_ce_lse(logits, targets)
    return ce, lse


def token_multi_ce(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk: int = 1024,
                   precision: str = "fp32", ce_form: str = "standard") -> tuple[torch.Tensor, torch.Tensor]:
    """token_ce with K targets per position (Token Superposition Training, ced/train/tst.py): hidden (n, d), weight
    (V, d), targets (n, K) -> (ce (n, K), lse (n,)), ce[:, k] = logsumexp(logits) - logits[targets[:, k]], one logit row
    per position, `chunk` positions at a time, each chunk checkpointed when a gradient is needed. precision, ce_form: as
    token_ce's (the fixed form is applied to each of the K hard-target terms)."""
    if check_ce_form(ce_form) == "fixed":
        fn = _chunk_multi_ce_fixed if check_head_precision(precision) == "fp32" else _chunk_multi_ce_bf16_fixed
    else:
        fn = _chunk_multi_ce if check_head_precision(precision) == "fp32" else _chunk_multi_ce_bf16
    ces, lses = [], []
    for start in range(0, hidden.shape[0], chunk):
        hs, ts = hidden[start : start + chunk], targets[start : start + chunk]
        if torch.is_grad_enabled() and (hs.requires_grad or weight.requires_grad):
            ce, lse = checkpoint(fn, hs, weight, ts, use_reentrant=False)
        else:
            ce, lse = fn(hs, weight, ts)
        ces.append(ce)
        lses.append(lse)
    return torch.cat(ces), torch.cat(lses)
