"""The fixed hard-target cross-entropy (TrainConfig.loss.ce_form "fixed"): the pure-PyTorch forms, the config check, the routing
of the Triton kernels (ced/kernels/ce_fix_triton.py).

THE PAGE'S FORM (as read from https://jonbarron.info/softmaxfix/ and the arXiv listing, 2026-10-24; the page wins over this
text). Softmax collapse: when the target logit is far above the others, the standard target gradient p_t - 1 rounds to 0
and the row's gradient no longer sums to zero (the page's cliffs: a margin of about 5.5 in bf16, 16.6 in float32, 36.7 in
float64). The fixed hard-target cross-entropy, with z the logits and t the target:

    m = -LSE_{j != t}(z_j - z_t)          L(z, t) = softplus(-m) = log(1 + e^{-m})
    g_j = e^{z_j - z_t} / (1 + e^{-m})    for j != t       (= the usual softmax_j)
    g_t = -e^{-m} / (1 + e^{-m}) = -sigmoid(-m) = -sum_{j != t} g_j

So the fix changes ONE element of a row's gradient, the target's: -sigmoid(LSE_{j != t}(z_j - z_t)) in place of
softmax_t - 1. Our L' = LSE_{j != t}(z_j) - z_t = LSE_{j != t}(z_j - z_t) = -m.

WHAT IT COVERS (check_config is the code of this list; a path it does not name raises, never falls back):
  covered  head_impl "chunked" (reference: ced/train/chunked_loss.py token_ce, selected_ce, token_multi_ce) and head_impl
           "fused" (ced/train/fused_head.py fused_linear_ce, fused_linear_multi_ce), each at head_precision "fp32" and
           "bf16"; the aux, decoder and draft heads (all hard targets); Token Superposition's multi-target heads (the
           bag phase and the multi-hot aux: the loss is a weighted sum of K hard-target cross-entropies of one logit row,
           so each term gets the fix: ce_k = softplus(L'_k)).
  raises   head_impl "quack_lce", "liger_flce", "cce" (third-party kernels: their gradient is not ours to change);
           model.cp_head.enabled (the mixture's CE is not a softmax cross-entropy on the vocabulary: its gradient on the V
           logits is g_lse softmax - g_zy onehot with g_lse != g_zy, the row does not sum to zero, and a cancellation-free
           form needs its own derivation, not the page's); model.mtp.enabled (the drafter's trainer has its own head loss).
  soft targets  none of our training losses has them (TST's bags are weighted sums of hard-target terms; the CP mixture is
           above): the page's soft-target form (distillation) is not built.
  not changed  the eval scorers (ced/eval/*: token_ce at its default ce_form "standard"), the periodic eval, the fused head's
           eval check. They score the same loss; the standard form's float32 value is off by up to half an ulp of lse
           (6e-8 |lse|, below 1e-6 nats) and is exactly 0 from a margin of about 16.6, where the fixed form keeps the true tiny value
           (measured:, section 4). The z-loss term b lse^2 is the base's: its gradient 2 b lse softmax_j has no p_t - 1 difference to
           cancel, so it is formed as before from the (row-consistent) lse of the same pass.

THE KERNELS (the fusion of 2026-11-02): where Triton is routed (use_triton: CUDA, or TRITON_INTERPRET=1) the fixed form is a variant of
the standard kernel itself, with the same two passes, launches, block and traffic: one-target heads ce_grad_fixed (fp32 in place) and
ce_grad_out_fixed (bf16 store); K-target heads (Token Superposition's bag phase) ce_grad_multi_fixed, one gather, one launch and one
scatter a chunk. The PyTorch forms in this file stay the reference and the CPU route.
"""

from __future__ import annotations

import importlib
import os

import torch

CE_FORMS = ("standard", "fixed")
UNCOVERED_HEADS = ("quack_lce", "liger_flce", "cce")
KERNEL_LOGITS_DTYPES = (torch.float32, torch.float64)  # = fused_head.KERNEL_LOGITS_DTYPES (the kernels accumulate in the logits' dtype)
_NEG_INF = float("-inf")
_HAVE_TRITON: list = []
RAN = {"ce_grad_fixed": 0, "ce_grad_out_fixed": 0, "ce_grad_multi_fixed": 0}


def check_ce_form(form: str) -> str:
    """ValueError unless `form` is one of CE_FORMS; returns it."""
    if form not in CE_FORMS:
        raise ValueError(f"loss.ce_form {form!r}: one of {CE_FORMS} ('standard': lse - z_t as always; 'fixed': Barron et al.'s "
                         "cross-entropy, softplus(LSE_{j!=t}(z_j) - z_t), ced/train/ce_fix.py)")
    return form


def check_config(cfg) -> str:
    """The trainer's check (a TrainConfig): ValueError, by name, for a path loss.ce_form "fixed" does not cover (module
    docstring). Returns the form. "standard" checks nothing else."""
    form = check_ce_form(cfg.loss.ce_form)
    if form == "standard":
        return form
    bad = []
    if cfg.head_impl in UNCOVERED_HEADS:
        bad.append(f"head_impl {cfg.head_impl!r} (a third-party fused linear-CE: its gradient is not built with the fix; use "
                   "head_impl 'fused' or 'chunked')")
    if cfg.model.cp_head.enabled:
        bad.append("model.cp_head.enabled (the Copy-Paste mixture's cross-entropy is not the page's softmax cross-entropy: "
                   "its gradient on the vocabulary logits is g_lse softmax - g_zy onehot with g_lse != g_zy; not built)")
    if cfg.model.mtp.enabled:
        bad.append("model.mtp.enabled (the drafter's trainer has its own float32 head loss; not built)")
    if bad:
        raise ValueError("loss.ce_form 'fixed' is not built for: " + "; ".join(bad))
    return form


# ----------------------------------------------------------------------------------------------------------------------
# the fixed per-row numbers (autograd forms: the reference)


def _masked_lse_and_target(logits: torch.Tensor, targets: torch.Tensor):
    """(LSE_{j != t}(z_j), z_t) for logits (n, V) and targets (n,), differentiable: the target's column is replaced by -inf in a
    copy (its gradient is exactly 0 there), so d/dz_t of the pair's first number is 0 and the target's logit enters only
    through the second."""
    t = targets[:, None]
    zt = logits.gather(-1, t).squeeze(-1)
    return torch.logsumexp(logits.scatter(-1, t, _NEG_INF), dim=-1), zt


def fixed_ce_lse(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference form: logits (..., V), targets (...) -> (ce, lse), each (...), in the logits' dtype, with autograd.
    ce = softplus(L') with L' = LSE_{j != t}(z_j) - z_t; lse = logaddexp(LSE_{j != t}(z_j), z_t). The gradient of ce through
    autograd is the page's: dce/dz_t = -sigmoid(L') (exactly: the masked log-sum-exp does not see z_t, and logaddexp's
    backward is exp(L' - ce)), dce/dz_j = softmax_j for j != t."""
    shape = targets.shape
    lse_nt, zt = _masked_lse_and_target(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    lp = lse_nt - zt
    ce = torch.logaddexp(lp, torch.zeros_like(lp))
    lse = torch.logaddexp(lse_nt, zt)
    return ce.reshape(shape), lse.reshape(shape)


def fixed_multi_ce_lse(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fixed_ce_lse with K targets per row: logits (n, V), targets (n, K) -> (ce (n, K), lse (n,)); ce[:, k] is the fixed
    cross-entropy of the row against target k alone (one masked log-sum-exp a term); lse is the row's, from term 0."""
    ces = []
    lse = None
    for k in range(targets.shape[1]):
        lse_nt, zt = _masked_lse_and_target(logits, targets[:, k])
        lp = lse_nt - zt
        ces.append(torch.logaddexp(lp, torch.zeros_like(lp)))
        if lse is None:
            lse = torch.logaddexp(lse_nt, zt)
    return torch.stack(ces, dim=1), lse


def sigmoid_stable(x: torch.Tensor) -> torch.Tensor:
    """sigmoid(x) as e/(1 + e) for x < 0 and 1/(1 + e) otherwise, e = exp(-|x|): no overflow, and the tiny values keep their
    relative accuracy (the form of the Triton kernels)."""
    e = torch.exp(-x.abs())
    return torch.where(x < 0, e, torch.ones_like(e)) / (1.0 + e)


# ----------------------------------------------------------------------------------------------------------------------
# the in-place gradient forms (the fused head's reference and its CPU route)


def ce_grad_fixed_torch(logits, targets, a, b, ce, lse) -> None:
    """In place: logits (n, V) -> the gradient of a ce + b lse^2 with the fixed cross-entropy; fills ce and lse (n,). The
    reference version of ce_grad_fixed_kernel (any device): the same numbers, not the same bits."""
    t = targets[:, None]
    zt = logits.gather(1, t).squeeze(1)
    logits.scatter_(1, t, _NEG_INF)
    lse_nt = torch.logsumexp(logits, dim=-1)
    lp = lse_nt - zt
    l = torch.logaddexp(lse_nt, zt)
    ce.copy_(torch.logaddexp(lp, torch.zeros_like(lp)))
    lse.copy_(l)
    # the others: exp(z_j - lse) (a + 2 b lse); the target's column (still -inf) is overwritten below
    logits.sub_(l[:, None]).exp_().mul_((a + 2 * b * l)[:, None])
    gt = 2 * b * l * torch.exp(zt - l) - a * sigmoid_stable(lp)
    logits.scatter_(1, t, gt[:, None])


def multi_prep(logits: torch.Tensor, targets: torch.Tensor):
    """Before the softmax pass overwrites a row (K targets a row): (z_t, L') for each term, both (n, K), L'_k = LSE_{j != t_k}
    (z_j) - z_{t_k}; the logits are restored. Only the row's argmax can have a softmax within rounding of 1, so only its
    column needs the log-sum-exp without it (one argmax and one masked log-sum-exp over the chunk, whatever K is): for the
    argmax term L' = LSE_{j != argmax} - z_argmax; for every other term p_k <= 1/2 and L'_k = (lse - z_k) + log1p(-p_k),
    p_k = exp(z_k - lse), with no cancellation. (The autograd reference, fixed_multi_ce_lse, masks each term's own column:
    the two are the tests' cross-check.)"""
    xt = logits.gather(1, targets)
    am = logits.argmax(dim=-1, keepdim=True)
    zam = logits.gather(1, am)
    logits.scatter_(1, am, _NEG_INF)
    lse_am = torch.logsumexp(logits, dim=-1)
    logits.scatter_(1, am, zam)
    zam = zam.squeeze(1)
    lse = torch.logaddexp(lse_am, zam)
    lp_arg = lse_am - zam
    lp_other = (lse[:, None] - xt) + torch.log1p(-torch.exp(xt - lse[:, None]))
    return xt, torch.where(targets == am, lp_arg[:, None], lp_other)


def multi_ce(lp: torch.Tensor) -> torch.Tensor:
    """The K terms' fixed cross-entropies softplus(L'_k), (n, K), from multi_prep's L'."""
    return torch.logaddexp(lp, torch.zeros_like(lp))


def multi_fix_targets(logits, targets, a, b, lse, xt, lp) -> None:
    """After the softmax pass has written G = (A + 2 b lse) softmax (A = sum_k a_k) over the logits: the K target columns
    rewritten with the fixed terms' gradient. Column j gets (A - A_j + 2 b lse) softmax_j - A_j sigmoid(L'_j), A_j the sum
    of a_k over the terms whose target is j (a token twice in a bag counts both; every term with the same column writes the
    same number). For K = 1 that is the one-target element 2 b lse softmax_t - a sigmoid(L')."""
    big_a = a.sum(1)
    same = targets[:, :, None] == targets[:, None, :]
    a_j = (a[:, None, :] * same).sum(-1)  # (n, K)
    p = torch.exp(xt - lse[:, None])
    val = (big_a[:, None] - a_j + 2 * b[:, None] * lse[:, None]) * p - a_j * sigmoid_stable(lp)
    for k in range(targets.shape[1]):
        logits.scatter_(1, targets[:, k : k + 1], val[:, k : k + 1])


# ----------------------------------------------------------------------------------------------------------------------
# routing: the Triton kernels, or the PyTorch forms


def have_triton() -> bool:
    if not _HAVE_TRITON:
        try:
            importlib.import_module("triton")
            _HAVE_TRITON.append(True)
        except ImportError:
            _HAVE_TRITON.append(False)
    return _HAVE_TRITON[0]


def use_triton(x: torch.Tensor) -> bool:
    """The Triton kernels for x: CUDA, or any device under TRITON_INTERPRET=1 (the interpreter, or the tests' emulation), when
    Triton imports and x is float32 or float64."""
    return (x.is_cuda or os.environ.get("TRITON_INTERPRET") == "1") and have_triton() and x.dtype in KERNEL_LOGITS_DTYPES


_KERNELS: list = []


def kernels():
    """The kernel module (ced.kernels.ce_fix_triton), imported by the first call of the process and kept in _KERNELS: Dynamo marks
    importlib.import_module as a function it does not trace, so a call inside a torch.compile(fullgraph=True) region (the speed
    check's compiled cell, scripts/ce_fix_check.py) would be an error on every CUDA call if it imported each time (the review of
    the build found it with a probe; tests/test_ce_fix.py::test_the_kernel_lookup_is_traceable_once_warm). The first call is always
    eager (the check's first call of a cell is outside the trace). reset_routing() forgets it: the tests' emulation swaps the module
    in sys.modules."""
    if not _KERNELS:
        _KERNELS.append(importlib.import_module("ced.kernels.ce_fix_triton"))
    return _KERNELS[0]


def reset_routing() -> None:
    """Forget what use_triton and kernels() learned (tests/tl_emu.py installs and removes a Triton emulation under the same names)."""
    _HAVE_TRITON.clear()
    _KERNELS.clear()


def _grid(logits: torch.Tensor, cfg: dict) -> int:
    n = logits.shape[0]
    if cfg["programs_per_sm"] and logits.is_cuda:
        return min(n, cfg["programs_per_sm"] * torch.cuda.get_device_properties(logits.device).multi_processor_count)
    return n


def ce_grad_fixed(logits, targets, a, b, ce, lse, cfg: dict) -> None:
    """In place: logits -> their gradient with the fixed cross-entropy; fills ce, lse. The Triton kernel
    (ce_grad_fixed_kernel) where use_triton, else ce_grad_fixed_torch. cfg: fused_head.CE_CONFIG's keys."""
    if use_triton(logits):
        RAN["ce_grad_fixed"] += 1
        n, V = logits.shape
        kernels().ce_grad_fixed_kernel[(_grid(logits, cfg),)](logits, logits.stride(0), targets, a, b, ce, lse, n, V,
                                                              BLOCK=cfg["block"], num_warps=cfg["num_warps"])
    else:
        ce_grad_fixed_torch(logits, targets, a, b, ce, lse)


def ce_grad_out_fixed(logits, out, targets, a, b, ce, lse, cfg: dict) -> None:
    """The gradient with the fixed cross-entropy rounded into out (bfloat16), the logits only read; fills ce, lse. The Triton
    kernel (ce_grad_out_fixed_kernel) on float32 logits where use_triton; else the PyTorch form in place over the logits, then
    one cast into out."""
    if use_triton(logits) and logits.dtype == torch.float32:
        RAN["ce_grad_out_fixed"] += 1
        n, V = logits.shape
        kernels().ce_grad_out_fixed_kernel[(_grid(logits, cfg),)](logits, logits.stride(0), out, out.stride(0), targets, a, b,
                                                                  ce, lse, n, V, BLOCK=cfg["block"],
                                                                  num_warps=cfg["num_warps"])
    else:
        ce_grad_fixed_torch(logits, targets, a, b, ce, lse)
        out.copy_(logits)


def ce_grad_multi_fixed(logits, targets, a, b, ce, lse, cfg: dict) -> None:
    """The K-target kernel route (ced/kernels/ce_fix_triton.ce_grad_multi_fixed_kernel; the module docstring of that file), for
    logits (n, V) in the kernels' dtypes, targets (n, K), a (n, K), b (n,), ce (n, K), lse (n,): one gather of the K target logits
    (as the standard multi form does), one launch (the softmax part in place over the logits, ce, lse and the K target columns'
    new values), one scatter of those values into the row. Only called where use_triton(logits)."""
    n, V = logits.shape
    K = targets.shape[1]
    targets, a = targets.contiguous(), a.contiguous()
    xt = logits.gather(1, targets)  # the target logits, before the kernel overwrites the row
    val = torch.empty_like(xt)
    RAN["ce_grad_multi_fixed"] += 1
    kernels().ce_grad_multi_fixed_kernel[(_grid(logits, cfg),)](logits, logits.stride(0), targets, xt, a, b, ce, lse, val, n, V,
                                                               K=K, BLOCK=cfg["block"], num_warps=cfg["num_warps"])
    logits.scatter_(1, targets, val)  # a token twice in a bag: the same value written twice
