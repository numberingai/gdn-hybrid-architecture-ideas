"""Negative eigenvalues, the negative-eigenvalue form of the GDN memory-rule screen: the write strength's range [0,
beta_max] (GDNConfig.negeig).

THE RULE. A GDN step is S' = alpha (I - beta k k^T) S + beta k v^T with |k| = 1 (ced/model/gdn.py delta_rule_recurrence). Its
transition has the eigenvalue alpha (1 - beta) along k and alpha across it. With beta = sigmoid(b) in (0, 1) the eigenvalue along
k is in (0, alpha); with beta = 2 sigmoid(b) in (0, 2) (Grazzi et al., arXiv 2411.12537, Eq. 2) it is in (-alpha, alpha): at
beta = 2 the step reflects the state along k. The products of transitions stay contractions (every eigenvalue in [-1, 1]).

WHERE IT ACTS. In GatedDeltaNet.gates and gates_from, after the sigmoid, so every caller of the gates follows (the training
forward, the second pass, the decode engine, the MTP drafter, the GDN diagnostics) and every kernel takes the scaled beta as an
input. beta_max 1 (the default): the gate lines are main's, not executed differently (scale_beta is not called).

THE KERNELS (checked by this build): our Triton chunk kernels use beta only as a multiplier (the solve's rows, w = A (k beta e^g), u
= A (v beta), the WY backward's dbeta), the reference kernels too (gdn_chunked's solve is unitriangular for any beta); none clamps
beta or assumes beta <= 1. The solve's inverse (I + A)^-1 has entries up to about 2 instead of 1; its bf16 store rounds them at twice
the absolute error: the GPU values check measures kernel "ced" against
the float64 reference at beta drawn in [0, 2] against the same draws at [0, 1], under err <= 4 x err_ref + 1e-6.

REFUSED (check, at the layer's construction): kernel "fla", chunk_backend "flashqla" and edges other than "plain" with beta_max 2
(they take beta as an input with no clamp too, but no check has run them at beta > 1); and, at the call, a TST bag phase with
the write-strength power (tst.gdn_resolution_gates "decay_beta"): its 1 - (1 - beta)^s is written for beta = sigmoid(b) (the
identity (I - beta k k^T)^s = I - (1 - (1 - beta)^s) k k^T holds for any beta, but that formula is not built for this arm).
"""

from __future__ import annotations

import torch

BETA_MAX = (1, 2)  # the values built: 1 (the layer as it was) and 2 (negative eigenvalues)
CHECKED_KERNELS = ("ced", "chunked", "recurrence")  # kernel "ced" on chunk_backend "triton", edges "plain"; the two references


def check(cfg) -> int:
    """The layer's beta_max from its GDNConfig (cfg.negeig.beta_max), checked: an int in BETA_MAX (not a bool); with 2, the
    kernel, chunk backend and edges this build checked. Returns it."""
    m = cfg.negeig.beta_max
    if isinstance(m, bool) or not isinstance(m, int) or m not in BETA_MAX:
        raise ValueError(f"GDNConfig.negeig.beta_max {m!r}: 1 (beta = sigmoid(b), the layer as it was) or 2 (beta = 2 sigmoid(b), "
                         "negative eigenvalues)")
    if m != 1:
        bad = [why for cond, why in ((cfg.kernel not in CHECKED_KERNELS, f"kernel {cfg.kernel!r}"),
                                     (cfg.chunk_backend != "triton", f"chunk_backend {cfg.chunk_backend!r}"),
                                     (cfg.edges != "plain", f"edges {cfg.edges!r}")) if cond]
        if bad:
            raise ValueError(f"GDNConfig.negeig.beta_max {m} (beta in (0, {m})) is built and checked at beta > 1 for kernel 'ced' "
                             "with chunk_backend 'triton' and edges 'plain' (the GPU values check gdnrule-values-neg) and the "
                             f"reference kernels 'chunked' and 'recurrence'; not for: {'; '.join(bad)}")
    return m


def refuse_beta_power(beta_max: int) -> None:
    """A TST bag phase with the write-strength power (gates' scale != 1 and beta_power) on a layer with beta_max != 1."""
    if beta_max != 1:
        raise ValueError(f"GDNConfig.negeig.beta_max {beta_max} with TST's write-strength power (tst.gdn_resolution_gates "
                         "'decay_beta'): its bag write strength 1 - (1 - beta)^s is written for beta = sigmoid(b) in (0, 1); "
                         "not built for beta in (0, 2)")


def scale_beta(beta: torch.Tensor, beta_max: int) -> torch.Tensor:
    """beta_max x beta (beta = sigmoid(b)); called only with beta_max != 1. A power of two: exact in every float format."""
    return beta * beta_max
