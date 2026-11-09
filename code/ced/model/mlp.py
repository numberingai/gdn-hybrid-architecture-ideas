"""Dense feed-forward blocks (SwiGLU, ReLU squared) and the routed SwiGLU of the v1 decoder arm."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FFN_ACTIVATIONS = ("swiglu", "relu2")


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "mx_mlp", None) is not None and self.training:  # model.fp8.chains swiglu / swiglu_bwd
            from ced.model import mx_wire

            return mx_wire.mlp_forward(self, x)
        folds = getattr(self, "g0_folds", None)
        if folds is not None and (self.training or folds.norm_mlp):  # model.fp8.folds swiglu_bwd, norm_mlp (in eval
            from ced.model import g0_wire  # too: g0_wire.mlp_forward checks that the norm deferred to it is in its mode)

            return g0_wire.mlp_forward(self, x)
        if getattr(self, "fused_proj", None) and self.training:  # model.fused_projections "gate_up": one GEMM
            from ced.model import fused_proj

            out = fused_proj.outputs(self, x)
            return self.w_down(F.silu(out["w_gate"]) * out["w_up"])
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ReLU2(nn.Module):
    """ReLU-squared FFN (Primer, arXiv 2109.08668): W_down relu(W_up x)^2. Two matrices, so at
    hidden 1.5 h it has the parameters and matmul FLOPs of a SwiGLU of hidden h. The linears are
    plain nn.Linear like SwiGLU's (the models' init_weights draws them alike, and Muon takes both)."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.relu(self.w_up(x)).square())


def dense_ffn(d_model: int, hidden: int, activation: str) -> nn.Module:
    """The dense FFN of ModelConfig.ffn_activation. "swiglu" builds exactly the module (and draws
    exactly the random numbers) it always did."""
    if activation == "swiglu":
        return SwiGLU(d_model, hidden)
    if activation == "relu2":
        return ReLU2(d_model, hidden)
    raise ValueError(f"ffn_activation must be one of {FFN_ACTIVATIONS}, got {activation!r}")


class RoutedSwiGLU(nn.Module):
    """Mixture of SwiGLU experts, top-k routing per token (the "decoder FFN routed" ladder arm).

    Reference implementation with fixed shapes: every expert runs on every token, and the router's
    top-k gates (renormalised softmax over the chosen experts, zero elsewhere) weight the outputs.
    A fast version would dispatch tokens instead; it must match this one.

    Routing statistics accumulate over every call since `reset_stats()`. In a looped decoder the
    same FFN runs once per loop, and balance is measured over all of those calls (as SUT pools
    over L*T gate calls), not only the last loop's:
      balance_loss = n_experts * sum_e (fraction of routed slots to e) * (mean router prob of e),
      which is 1 for uniform routing (Switch Transformer);
      z_loss = mean over tokens of logsumexp(router logits)^2.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int, expert_hidden: int):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d_model, expert_hidden) for _ in range(n_experts))
        self.reset_stats()

    def reset_stats(self) -> None:
        self._chosen, self._prob, self._n, self._z = 0.0, 0.0, 0.0, 0.0

    def balance_loss(self) -> torch.Tensor:
        frac = self._chosen / (self._n * self.top_k)
        return self.n_experts * (frac * (self._prob / self._n)).sum()

    def z_loss(self) -> torch.Tensor:
        return self._z / self._n

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.router(x)
        probs = torch.softmax(logits.to(torch.promote_types(logits.dtype, torch.float32)), dim=-1)  # (..., E)
        top_p, top_i = probs.topk(self.top_k, dim=-1)
        gates = torch.zeros_like(probs).scatter(-1, top_i, top_p / top_p.sum(-1, keepdim=True))
        out = sum(gates[..., e : e + 1].to(x.dtype) * expert(x) for e, expert in enumerate(self.experts))

        # Load balance over real tokens only (empty decoder slots carry no tokens).
        w = torch.ones(x.shape[:-1], dtype=probs.dtype, device=x.device) if valid is None else valid.to(probs.dtype)
        token_dims = tuple(range(probs.dim() - 1))
        chosen = (gates > 0).to(probs.dtype)
        self._chosen = self._chosen + (chosen * w[..., None]).sum(dim=token_dims)
        self._prob = self._prob + (probs * w[..., None]).sum(dim=token_dims)
        self._n = self._n + w.sum().clamp(min=1)
        lse = torch.logsumexp(logits.to(probs.dtype), dim=-1)
        self._z = self._z + (lse.pow(2) * w).sum()
        return out
