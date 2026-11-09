"""Muon for hidden weight matrices, AdamW for every other parameter, in one optimizer.

The default: Muon (Moonlight settings, matched update RMS) with Polar Express orthogonalization for hidden
matrices; AdamW for embeddings, LM head, norms, and GDN scalars, convolutions, and gates.
The parameter split is in ced/train/optim.py; this file is the update rule.

Muon step for a matrix W (m x n) with gradient G (Moonlight, arXiv 2502.16982, with Nesterov momentum):
    B <- mu * B + G;    U = G + mu * B  (Nesterov)  or  U = B
    O = PolarExpress(U)                     approximately U (U^T U)^(-1/2), computed in bfloat16
    W <- W - lr * (weight_decay * W + 0.2 * sqrt(max(m, n)) * O)
The factor 0.2 * sqrt(max(m, n)) gives the update the RMS of a typical AdamW update (Moonlight's
"matched update RMS"), so Muon and AdamW parameters share one learning rate.

Polar Express (arXiv 2505.16932) replaces the fixed Newton-Schulz quintic with a sequence of
per-iteration quintic polynomials X <- a X + b (X X^T) X + c (X X^T)^2 X, optimal for each step.
The coefficients are copied from the cited source in POLAR_EXPRESS; they are not derived here.
Iteration i uses tuple i; past the end of the list the last tuple repeats (the listing's rule).

AdamW groups: the standard decoupled update (torch.optim.AdamW semantics, bias-corrected).
Groups are marked with group["kind"] in {"muon", "adamw"}; state_dict() is the standard format.
"""

from __future__ import annotations

import math

import torch

# Polar Express, arXiv 2505.16932v5, appendix listing `alg:polar-express` ("Code for Polar Express"),
# the code the paper used for its GPT experiments (recommended d = 5, T = 5, l = 1e-3, bfloat16).
# Verified 2026-10-05 against the e-print TeX and regenerated from the paper's offline-stage code to
# ~1e-14. The raw tuples are pasted verbatim; the safety factor is applied below exactly as the
# listing does: p(x) -> p(x / 1.01) for all but the last tuple.
_POLAR_EXPRESS_RAW = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]
POLAR_EXPRESS: dict = {
    "source": "https://arxiv.org/e-print/2505.16932 (v5), appendix listing alg:polar-express",
    "coefficients": [(a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in _POLAR_EXPRESS_RAW[:-1]] + [_POLAR_EXPRESS_RAW[-1]],
    "normalize_safety": 1.01,  # X0 = U / (||U||_F * 1.01 + 1e-7), as in the listing
    "eps": 1e-7,
}


def polar_express(U: torch.Tensor, steps: int) -> torch.Tensor:
    """Approximate the orthogonal polar factor of U (m x n) with Polar Express, in bfloat16."""
    return polar_express_bf16(U, steps).to(U.dtype)


def polar_express_bf16(U: torch.Tensor, steps: int) -> torch.Tensor:
    """polar_express before its last cast: the bfloat16 factor in U's orientation (a transposed view when U is tall).
    CUDA graphs' "step" mode (ced/train/cuda_graphs.py) keeps it as the graph's output and casts it outside."""
    coeffs = POLAR_EXPRESS["coefficients"]
    if not coeffs:
        raise RuntimeError("Polar Express coefficients are not filled in (see POLAR_EXPRESS in ced/train/muon.py)")
    X = U.to(torch.bfloat16)
    transpose = X.shape[0] > X.shape[1]
    if transpose:  # iterate on the wide orientation: X X^T is the smaller Gram matrix
        X = X.T
    X = X / (X.norm() * POLAR_EXPRESS["normalize_safety"] + POLAR_EXPRESS["eps"])
    for i in range(steps):
        a, b, c = coeffs[min(i, len(coeffs) - 1)]
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X


class MuonWithAdamW(torch.optim.Optimizer):
    def __init__(self, groups: list[dict], lr: float, betas=(0.9, 0.95), eps: float = 1e-8, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        for g in groups:
            assert g.get("kind") in ("muon", "adamw"), g.get("kind")
            if g["kind"] == "muon":
                assert all(p.ndim == 2 for p in g["params"]), "Muon takes 2-D weight matrices only"
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=0.0)
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            if group["kind"] == "muon":
                mu = group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(mu).add_(p.grad)
                    update = p.grad.add(buf, alpha=mu) if group["nesterov"] else buf
                    O = polar_express(update, group["ns_steps"])
                    scale = 0.2 * math.sqrt(max(p.shape))
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    p.add_(O, alpha=-lr * scale)
            else:
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = torch.zeros((), dtype=torch.float32)
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    t = float(state["step"])
                    state["exp_avg"].lerp_(p.grad, 1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                    denom = (state["exp_avg_sq"] / (1 - beta2**t)).sqrt_().add_(group["eps"])
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    p.addcdiv_(state["exp_avg"], denom, value=-lr / (1 - beta1**t))

    # --- step() in three parts, for CUDA graphs' "step" mode (TrainConfig.cuda_graphs; ced/train/cuda_graphs.py) ----------
    # muon_orthogonalise is the part of step() that reads no LR and no step count (it is captured in a CUDA graph);
    # muon_apply and adamw_step read the step's LR and AdamW's step count on the host and run eagerly after it. Each
    # parameter's update reads only its own gradient and state, so the three in this order make, op for op on every
    # parameter, the update step() makes (tests/test_n1_cuda_graphs.py compares them bit for bit).
    @torch.no_grad()
    def muon_orthogonalise(self) -> list:
        """For every Muon parameter with a gradient: step()'s momentum update and Polar Express. Returns
        [(group, p, X, dtype)], X the bfloat16 factor before polar_express's cast to dtype (the update's)."""
        out = []
        for group in self.param_groups:
            if group["kind"] != "muon":
                continue
            mu = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(p.grad)
                update = p.grad.add(buf, alpha=mu) if group["nesterov"] else buf
                out.append((group, p, polar_express_bf16(update, group["ns_steps"]), update.dtype))
        return out

    @torch.no_grad()
    def muon_apply(self, orthogonalised: list) -> None:
        """step()'s Muon update from muon_orthogonalise's factors, at the groups' LR now."""
        for group, p, X, dtype in orthogonalised:
            lr, wd = group["lr"], group["weight_decay"]
            O = X.to(dtype)
            scale = 0.2 * math.sqrt(max(p.shape))
            if wd:
                p.mul_(1.0 - lr * wd)
            p.add_(O, alpha=-lr * scale)

    @torch.no_grad()
    def adamw_step(self) -> None:
        """step()'s AdamW groups, as step() runs them."""
        for group in self.param_groups:
            if group["kind"] == "muon":
                continue
            lr, wd = group["lr"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = torch.zeros((), dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = float(state["step"])
                state["exp_avg"].lerp_(p.grad, 1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                denom = (state["exp_avg_sq"] / (1 - beta2**t)).sqrt_().add_(group["eps"])
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.addcdiv_(state["exp_avg"], denom, value=-lr / (1 - beta1**t))

    @torch.no_grad()
    def reset_moments(self) -> int:
        """Token Superposition Training's switch (ced/train/optim.py reset_moments): every Muon momentum buffer and every
        AdamW exp_avg / exp_avg_sq / step zeroed in place. All this optimizer holds is moments, so the next step is the
        one after self.state.clear(), bit for bit (tests/test_matrix_optim.py). Returns the number of tensors reset."""
        n = 0
        for st in self.state.values():
            if not st:
                continue
            n += 1
            for v in st.values():
                v.zero_()
        return n
