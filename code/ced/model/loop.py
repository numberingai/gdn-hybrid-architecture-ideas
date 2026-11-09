"""What happens at the loop boundary of the decoder: input injection, loop-index conditioning, and
(optionally) hyper-connections. All of it is per position, so none of it affects causality.

One loop k = 1..T, at the gathered decoder slots:

    z_in = HC.read(Y, k)                      # Y = z when hyper-connections are off
    u    = inject(z_in, e) + loop_emb[k]      # e = RMSNorm(h_enc), computed once
    s    = DecBlock(u)                        # the 4 shared decoder layers
    Y    = HC.write(Y, z_in, s, k)            # Y = s when hyper-connections are off

Injection and hyper-connections are switched independently, so the Phase 2 arms
injection-only / HC-only / both are config values.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ced.config import LoopConfig
from ced.model.pass_schedule import built_passes


class ParcaeInjection(nn.Module):
    """Stable input injection from Parcae (arXiv 2604.12946, its diagonal injection):

        inject(z, e) = exp(-dt * exp(A_log)) * z + dt * (B e),   dt = softplus(dt_bias)

    The decay exp(-dt * exp(A_log)) lies in (0, 1) for any parameter values, so the linear part of
    the recurrence is a contraction. A_log and dt_bias are per channel; B is a dense d x d matrix.
    Init: A_log = 0, dt_bias such that the decay is init_decay
    (sqrt(1/5) = 0.447, giving dt = 0.805), and B = identity. One adapter is shared by all loops.
    """

    def __init__(self, d_model: int, init_decay: float):
        super().__init__()
        dt0 = -math.log(init_decay)
        self.A_log = nn.Parameter(torch.zeros(d_model))
        self.dt_bias = nn.Parameter(torch.full((d_model,), dt0 + math.log(-math.expm1(-dt0))))
        self.B = nn.Parameter(torch.eye(d_model))

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        dt = F.softplus(self.dt_bias)
        return torch.exp(-dt * torch.exp(self.A_log)), dt  # (decay, dt)

    def forward(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        decay, dt = self.coefficients()
        return decay * z + dt * F.linear(e, self.B)


RECUR_B_FORMS = ("dense", "diag")


class RecurInjection(nn.Module):
    """The input injection of the loop in the published form on arch "control" (LoopConfig.recur_layers with injection
    "parcae"; ced/model/encoder.py Encoder.forward_recur), the diagonal injection of Parcae (arXiv 2604.12946):

        u = decay * h + dt * (B e),     dt = softplus(dt_bias),  decay = exp(-dt * exp(A_log))

    h is the state the previous pass left (B, L, d), e the prelude's output, the same at every pass. A = -exp(A_log)
    is a negative diagonal whatever the parameters, so decay = exp(dt A) lies in (0, 1) in exact arithmetic: the
    linear part of the recurrence is a contraction by construction. In floating point exp underflows to 0 or rounds
    to 1 for extreme parameters, so the product dt * exp(A_log) is clamped to [DECAY_MIN_RATE, DECAY_MAX_RATE] before
    the exp: decay stays inside (0, 1) as a float32 too (inside the range, which holds decays from 0.9999999 down
    to 1e-13, the clamped and the unclamped form are the same function).
    A_log and dt_bias are per channel (d each). B: "dense" a d x d matrix (the published form), "diag" a (d,) vector.
    Init: A_log = 0, dt_bias the inverse softplus
    of -ln(init_decay) (so decay = init_decay, sqrt(1/5) = 0.447 and dt = 0.805 by default), B the identity (ones).
    No random number is drawn. None of the three gets weight decay (ced/train/optim.py
    NO_DECAY_SUFFIXES) and B is on AdamW, not Muon (optim.is_muon_param: "injection"), unless
    OptimConfig.recur_b says otherwise: "muon" puts the dense B in the Muon group of the layers' matrices, with their
    weight decay; "muon_no_decay" gives it the Muon update without decay (ced/train/optim.py, module docstring).

    The sum is formed in the stream's own dtype, outside autocast: under the runs' bfloat16 autocast the layers'
    stream is float32, and a bfloat16 linear here would round the injected e, which is most of the next pass's
    input, to 8 bits of mantissa at every pass."""

    DECAY_MIN_RATE, DECAY_MAX_RATE = 1e-7, 30.0

    def __init__(self, d_model: int, init_decay: float, b_form: str = "dense"):
        super().__init__()
        assert b_form in RECUR_B_FORMS, f"loop.recur_inject_b {b_form!r}: one of {RECUR_B_FORMS}"
        assert 0.0 < init_decay < 1.0, f"loop.parcae_init_decay {init_decay!r}: a decay in (0, 1)"
        dt0 = -math.log(init_decay)
        self.b_form = b_form
        self.A_log = nn.Parameter(torch.zeros(d_model))
        self.dt_bias = nn.Parameter(torch.full((d_model,), dt0 + math.log(-math.expm1(-dt0))))
        self.B = nn.Parameter(torch.eye(d_model) if b_form == "dense" else torch.ones(d_model))

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(decay, dt), each (d,), in the parameters' dtype."""
        dt = F.softplus(self.dt_bias)
        rate = (dt * torch.exp(self.A_log)).clamp(self.DECAY_MIN_RATE, self.DECAY_MAX_RATE)
        return torch.exp(-rate), dt

    def forward(self, h: torch.Tensor | None, e: torch.Tensor) -> torch.Tensor:
        """u for the next pass. h None: the state is zeros (the first pass under recur_state_init "zeros"), and the
        decay term, exactly zero, is not formed."""
        with torch.autocast(device_type=e.device.type, enabled=False):
            decay, dt = self.coefficients()
            decay, dt, B = decay.to(e.dtype), dt.to(e.dtype), self.B.to(e.dtype)
            driven = dt * (F.linear(e, B) if self.b_form == "dense" else e * B)
            return driven if h is None else decay * h + driven

    @torch.no_grad()
    def summary(self) -> dict:
        """For the logs (one host read): the decay's and dt's min / mean / max over the channels, how far B is from
        its start ("b_drift": the Frobenius norm of B - I over that of I; for "diag", of B - 1 over that of 1), and
        B's own size over its start's ("b_size": 1 at the start; under 1 when B has shrunk, as weight decay on B
        would shrink it. The drift alone does not say whether B shrank or turned)."""
        decay, dt = (t.double().cpu() for t in self.coefficients())
        B = self.B.detach().double().cpu()
        start = torch.eye(B.shape[0], dtype=B.dtype) if self.b_form == "dense" else torch.ones_like(B)
        three = lambda t: [round(float(t.min()), 6), round(float(t.mean()), 6), round(float(t.max()), 6)]  # noqa: E731
        return {"decay": three(decay), "dt": three(dt), "b_drift": round(float((B - start).norm() / start.norm()), 6),
                "b_size": round(float(B.norm() / start.norm()), 6)}


class NoInjection(nn.Module):
    """Passthrough: the encoder output enters only through z_0 (or the hyper-connection streams)."""

    def forward(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return z


class NoHyperConnections(nn.Module):
    """A single residual state: read returns it, write replaces it with the block output."""

    def start(self, e: torch.Tensor) -> torch.Tensor:
        return e

    def read(self, Y: torch.Tensor, k: int) -> torch.Tensor:
        return Y

    def write(self, Y: torch.Tensor, z_in: torch.Tensor, s: torch.Tensor, k: int) -> torch.Tensor:
        return s

    def finish(self, Y: torch.Tensor) -> torch.Tensor:
        return Y


class StaticHyperConnections(nn.Module):
    """Static hyper-connections at the loop boundary: the Zhu et al. form used in arXiv 2604.21106,
    App. G.1, where it replaces the input injection (there phi rose from ~0.45 to ~0.65).

    K lanes start as copies of e. Loop k (0-based) has its own scalars a_k (K), M_k (K x K), b_k (K):

        read:   r_k = sum_j a_k[j] Y[j]
        write:  Y[i] <- sum_j M_k[i, j] Y[j] + b_k[i] * w_k
        w_k = s_k - r_k  ("delta", default)   or   s_k  ("full")

    s_k is the decoder block's full output stream. The paper does not say whether s is the full
    stream or its increment; only the delta form makes its stated init (a_k = one_hot(k mod K),
    M_k = I, b_k = 1) "reduce to a plain looped forward pass", so delta is the default.
    Collapse: mean over lanes (the paper sums; under the final RMSNorm the two differ only through
    the norm's eps, and the mean makes HC-at-init equal HC-off).
    """

    def __init__(self, d_model: int, cfg: LoopConfig):
        super().__init__()
        K, t_max = cfg.hc_streams, cfg.t_max
        self.n_streams = K
        self.write_mode = cfg.hc_write
        read = torch.zeros(t_max, K)
        read[torch.arange(t_max), torch.arange(t_max) % K] = 1.0  # cyclic one-hot reads
        self.read_weights = nn.Parameter(read)
        self.mix_weights = nn.Parameter(torch.eye(K).repeat(t_max, 1, 1))
        self.write_weights = nn.Parameter(torch.ones(t_max, K))

    def start(self, e: torch.Tensor) -> torch.Tensor:
        return e.unsqueeze(-2).expand(*e.shape[:-1], self.n_streams, e.shape[-1]).clone()

    def read(self, Y: torch.Tensor, k: int) -> torch.Tensor:
        return torch.einsum("n,...nd->...d", self.read_weights[k].to(Y.dtype), Y)

    def write(self, Y: torch.Tensor, z_in: torch.Tensor, s: torch.Tensor, k: int) -> torch.Tensor:
        w = s - z_in if self.write_mode == "delta" else s
        mixed = torch.einsum("mn,...nd->...md", self.mix_weights[k].to(Y.dtype), Y)
        return mixed + self.write_weights[k].to(Y.dtype)[:, None] * w.unsqueeze(-2)

    def finish(self, Y: torch.Tensor) -> torch.Tensor:
        return Y.mean(dim=-2)


# The tensors a gated second pass adds to the single-pass model (LoopConfig.pass2_gate), by their names in CED v2:
# pass 2's hyper-connection scalars or its gate (GatedSecondPass below) and its loop-index embedding. The trainer
# gives exactly these their own parameter groups (ced/train/optim.py) and accepts exactly these as new when a T = 1
# checkpoint grows to two passes (ced/train/trainer.py). The looped control (ced/model/control.py) adds two of them
# under the same names: hyper_connections.pass2_gate (ScalarSecondPassGate below) and loop_embedding_pass2 (and, with a
# third pass, hyper_connections.pass3_gate and loop_embedding_pass3 of PASS3_PARAMETERS).
PASS2_PARAMETERS = ("loop_embedding_pass2", "hyper_connections.pass2_read", "hyper_connections.pass2_mix",
                    "hyper_connections.pass2_write", "hyper_connections.pass2_gate")
# A third pass exists ONLY through LoopConfig.segment_passes with a 3 in the map: its own tensors, the same forms as pass
# 2's. LATER_PASS_PARAMETERS is what the trainer treats as "pass 2's own tensors": their own optimizer groups, new when a
# T = 1 checkpoint grows. A model without a third pass has none of the pass-3 names.
PASS3_PARAMETERS = ("loop_embedding_pass3", "hyper_connections.pass3_read", "hyper_connections.pass3_mix",
                    "hyper_connections.pass3_write", "hyper_connections.pass3_gate")
LATER_PASS_PARAMETERS = PASS2_PARAMETERS + PASS3_PARAMETERS


class GatedSecondPass(StaticHyperConnections):
    """Static hyper-connections for T = 2 with a second pass that starts as the identity
    (LoopConfig.pass2_gate). The design ("Decoder computation"):

        "z_2 = z_1 + alpha * (Block(z_1) - z_1), with alpha initialised to 0 when the loop is switched on
         (pretraining phase B, section 7). The model at switch-on is exactly the single-pass model."
        "Loop-index embedding; hyper-connection at the boundary."

    The spec names no form of the hyper-connection; this is the repo's static module (the 60M models' setting).

    Pass 1 (k = 0) is StaticHyperConnections with one loop: the T = 1 model's tensors (read_weights (1, K),
    mix_weights (1, K, K), write_weights (1, K)), the same names, shapes, init and ops, so a T = 1 checkpoint loads
    into them and a forward at T = 1 is the T = 1 model's bit for bit. Pass 2 (k = 1) has its own tensors, in one of
    two forms, with w = s_2 - r_2 (the delta write) and z_1 = mean_i Y[i] (finish after pass 1):

    "hc"      pass2_read a (K), pass2_mix M (K, K), pass2_write b (K), all trained:
                  r_2 = sum_j a[j] Y[j];     Y[i] <- sum_j M[i, j] Y[j] + b[i] * w
              init a = 1/K (the lane mean), M = I, b = 0.
    "scalar"  pass2_gate alpha (one trained scalar, init 0); the read and the mix are fixed:
                  r_2 = mean_j Y[j];         Y[i] <- Y[i] + alpha * w

    With a at the lane mean and M = I, r_2 = z_1 and finish gives z_2 = z_1 + alpha * (s_2 - z_1) with
    alpha = mean(b) ("hc") or the scalar: the spec's equation, s_2 = Block(z_1 + loop embedding of pass 2) (that
    embedding starts at zero). At alpha = 0 (b = 0) pass 2 leaves both lanes as they are (x * 1 + y * 0 + 0 * w is
    exact for finite values), so the two-pass output equals the one-pass output bit for bit, whatever pass 1's
    trained scalars are (tests/test_loop_t2.py), also under bfloat16 autocast (pass 2's read, mix and write are
    multiplies and sums in the lanes' dtype, no einsum: see write; tests/test_loop_t2_review.py). alpha gets a
    gradient at once (d z_2 / d alpha = s_2 - z_1); the block's pass-2 path gets none until alpha moves. In the "hc"
    form pass 2's mix M also gets a gradient at once (d z_2 / d M[i, j] does not pass through alpha): it re-weights
    pass 1's lanes, which pass 1's own scalars can already express.

    A third pass (n_passes 3; only through LoopConfig.segment_passes with a 3 in the map: PASS3_PARAMETERS above) is
    defined like the second: its own tensors pass3_read / pass3_mix / pass3_write ("hc") or pass3_gate ("scalar"), the
    same init, the same ops with k = 2. It reads the lanes pass 2 left at the positions it runs at, so with its gate
    at 0 the three-pass output is the two-pass output bit for bit, and z_3 = z_2 + alpha_3 * (s_3 - z_2).
    """

    def __init__(self, d_model: int, cfg: LoopConfig, n_passes: int = 2):
        assert cfg.pass2_gate in ("hc", "scalar"), cfg.pass2_gate
        assert cfg.t_max == 2, "loop.pass2_gate is built for T = 2 (t_max = 2)"
        assert cfg.hc_write == "delta", "loop.pass2_gate needs hc_write 'delta' (the gate multiplies Block(z_1) - z_1)"
        assert n_passes in (2, 3), f"a gated loop has two passes, or three through loop.segment_passes; not {n_passes}"
        super().__init__(d_model, dataclasses.replace(cfg, t_max=1))  # pass 1: the T = 1 module's tensors
        K = cfg.hc_streams
        self.form = cfg.pass2_gate
        self.n_passes = n_passes
        if self.form == "hc":
            self.pass2_read = nn.Parameter(torch.full((K,), 1.0 / K))
            self.pass2_mix = nn.Parameter(torch.eye(K))
            self.pass2_write = nn.Parameter(torch.zeros(K))
        else:
            self.pass2_gate = nn.Parameter(torch.zeros(()))
        if n_passes == 3:
            if self.form == "hc":
                self.pass3_read = nn.Parameter(torch.full((K,), 1.0 / K))
                self.pass3_mix = nn.Parameter(torch.eye(K))
                self.pass3_write = nn.Parameter(torch.zeros(K))
            else:
                self.pass3_gate = nn.Parameter(torch.zeros(()))

    def _own(self, kind: str, k: int) -> torch.Tensor:
        """The tensor `kind` ("read", "mix", "write", "gate") of the pass with 0-based index k >= 1."""
        assert 1 <= k < self.n_passes, f"pass {k + 1} of a {self.n_passes}-pass gated loop"
        return getattr(self, f"pass{k + 1}_{kind}")

    def alpha(self, k: int = 1) -> torch.Tensor:
        """The gate of the pass with 0-based index k (default pass 2): the mean write weight ("hc") or the scalar. 0-dim."""
        return self._own("write", k).mean() if self.form == "hc" else self._own("gate", k)

    @torch.no_grad()
    def is_identity(self) -> bool:
        """Whether every later pass leaves every lane exactly as it is, so that the passes give the one-pass output bit
        for bit: every write weight 0 (not only their mean) and the mix the identity ("hc"), or the gate 0 ("scalar")."""
        for k in range(1, self.n_passes):
            if self.form == "hc":
                mix, write = self._own("mix", k), self._own("write", k)
                eye = torch.eye(self.n_streams, dtype=mix.dtype, device=mix.device)
                if not (bool((write == 0).all()) and bool((mix == eye).all())):
                    return False
            elif not bool(self._own("gate", k) == 0):
                return False
        return True

    def read(self, Y: torch.Tensor, k: int) -> torch.Tensor:
        if k == 0:
            return super().read(Y, 0)
        if self.form == "hc":
            # sum_j a[j] Y[j] as a multiply and a sum over the lanes, in Y's dtype (see write)
            return (self._own("read", k).to(Y.dtype)[:, None] * Y).sum(dim=-2)
        return Y.mean(dim=-2)

    def write(self, Y: torch.Tensor, z_in: torch.Tensor, s: torch.Tensor, k: int) -> torch.Tensor:
        if k == 0:
            return super().write(Y, z_in, s, 0)
        w = (s - z_in).unsqueeze(-2)
        if self.form == "hc":
            # sum_j M[i, j] Y[j] as a multiply and a sum over the lanes, in Y's dtype, NOT pass 1's einsum: an einsum
            # contracts through bmm, which autocast runs in bfloat16, and the lanes are float32 under the runs'
            # autocast (the block's stream is). The einsum would round pass 1's whole output to bfloat16 here
            # (measured on the CPU: 1.7e-3 relative), so the two-pass model at alpha = 0 would not be the single-pass
            # model under autocast, and the "hc" form would differ from the "scalar" form by that rounding at every
            # step. With the multiply and sum, M = I gives Y[i] * 1 + Y[j] * 0 exactly in any dtype.
            mixed = (self._own("mix", k).to(Y.dtype)[:, :, None] * Y.unsqueeze(-3)).sum(dim=-2)
            return mixed + self._own("write", k).to(Y.dtype)[:, None] * w
        return Y + self._own("gate", k).to(Y.dtype) * w


class ScalarSecondPassGate(nn.Module):
    """The gate of a second pass on a stack without hyper-connection lanes: the looped control (ced/model/control.py,
    "Two passes"; section 3). GatedSecondPass's "scalar" form with one lane and no pass-1 tensors:

        z_2 = z_1 + alpha * (s_2 - z_1),    s_2 = Block(z_1 + loop embedding of pass 2)

    alpha is one trained scalar, initialised 0, under the stage 2 build's name (as a module named hyper_connections:
    hyper_connections.pass2_gate, one of PASS2_PARAMETERS), so the optimizer groups, the growth checks and the
    metrics of the trainer apply to it unchanged. At alpha = 0, z_2 = z_1 + 0 * (s_2 - z_1) is z_1 exactly for finite
    values, in float32, float64 and under bfloat16 autocast: a multiply and two sums in the stream's own dtype, no
    einsum and no matmul (an einsum contracts through bmm, which autocast runs in bfloat16: GatedSecondPass.write).
    alpha gets a gradient at once (d z_2 / d alpha = s_2 - z_1); the block's pass-2 path gets exactly zero until
    alpha moves.

    A third pass (n_passes 3; only through LoopConfig.segment_passes with a 3 in the map, as in GatedSecondPass) is
    defined like the second: its own scalar pass3_gate (hyper_connections.pass3_gate, one of PASS3_PARAMETERS), 0 at
    init, and z_3 = z_2 + alpha_3 * (s_3 - z_2) at the positions that run it, s_3 = Block(z_2 + loop embedding of
    pass 3). With n_passes 2 (the default) the module has the one tensor and the one op it always had."""

    def __init__(self, n_passes: int = 2):
        super().__init__()
        assert n_passes in (2, 3), f"a gated loop has two passes, or three through loop.segment_passes; not {n_passes}"
        self.form = "scalar"
        self.n_passes = n_passes
        self.pass2_gate = nn.Parameter(torch.zeros(()))
        if n_passes == 3:
            self.pass3_gate = nn.Parameter(torch.zeros(()))

    def _gate(self, k: int) -> torch.Tensor:
        """The gate of the pass with 0-based index k >= 1."""
        assert 1 <= k < self.n_passes, f"pass {k + 1} of a {self.n_passes}-pass gated loop"
        return self.pass2_gate if k == 1 else self.pass3_gate

    def alpha(self, k: int = 1) -> torch.Tensor:
        """The gate of the pass with 0-based index k (default pass 2). 0-dim."""
        return self._gate(k)

    @torch.no_grad()
    def is_identity(self) -> bool:
        """Whether every later pass gives the one-pass output bit for bit: every gate is 0."""
        return all(bool(self._gate(k) == 0) for k in range(1, self.n_passes))

    def forward(self, z1: torch.Tensor, s2: torch.Tensor, k: int = 1) -> torch.Tensor:
        """The state after the pass with 0-based index k (default pass 2), from the state before it z1 and the
        block's output of this pass s2 (same shape): z_2 from z_1 and s_2; z_3 from z_2 and s_3."""
        return z1 + self._gate(k).to(z1.dtype) * (s2 - z1)


class DynamicDiagHyperConnections(nn.Module):
    """Hyperloop's loop-level hyper-connections (arXiv 2604.21254, section 2.2), with a dynamic
    diagonal residual matrix instead of mHC's Sinkhorn-projected one.

    n streams start as copies of e. Loop k has its own parameters. From the pre-block streams Y:

        z      = RMSNorm(flatten(Y))                              (no gain; it is absorbed in W)
        H_pre  = sigmoid(a_pre  * (z W_pre)  + b_pre)            (n)
        H_post = 2 * sigmoid(a_post * (z W_post) + b_post)       (n)
        H_res  = sigmoid(a_res  * (z W_res)  + b_res)            (n, the diagonal)
        read:  r = sum_j H_pre[j] Y[j]
        write: Y[i] <- H_res[i] Y[i] + H_post[i] (s + e_k)

    s is the block's full output stream (its layers keep their own skips) and e_k is Hyperloop's
    per-loop output embedding. Collapse: mean over streams.

    The paper used PyTorch default init. Ours (a choice, recorded here): W ~ N(0, 1/(n d)),
    a = 0.01 (mHC), b_pre = logit(1/n) so the read is about the stream mean, b_post = 0 so
    H_post is about 1, sigmoid(b_res) = hc_res_init, e_k = 0.

    W must be random. The streams start as identical copies of e and the model is symmetric under
    permuting streams, so with W = 0 every stream gets the same gradient forever and the n-stream
    HC trains as a single stream (found in review, 2026-10-02). The random W makes the coefficients
    differ slightly per stream from the first step. With W = 0 and hc_res_init -> 0 the module is
    exactly the plain loop, which the tests use.
    """

    def __init__(self, d_model: int, cfg: LoopConfig):
        super().__init__()
        n, t_max = cfg.hc_streams, cfg.t_max
        self.n_streams = n
        std = (n * d_model) ** -0.5
        self.proj_pre = nn.Parameter(torch.randn(t_max, n * d_model, n) * std)
        self.proj_post = nn.Parameter(torch.randn(t_max, n * d_model, n) * std)
        self.proj_res = nn.Parameter(torch.randn(t_max, n * d_model, n) * std)
        self.scale_pre = nn.Parameter(torch.full((t_max,), 0.01))
        self.scale_post = nn.Parameter(torch.full((t_max,), 0.01))
        self.scale_res = nn.Parameter(torch.full((t_max,), 0.01))
        self.bias_pre = nn.Parameter(torch.full((t_max, n), -math.log(n - 1) if n > 1 else 10.0))
        self.bias_post = nn.Parameter(torch.zeros(t_max, n))
        res = cfg.hc_res_init
        self.bias_res = nn.Parameter(torch.full((t_max, n), math.log(res / (1 - res))))
        self.output_embedding = nn.Parameter(torch.zeros(t_max, d_model))

    def start(self, e: torch.Tensor) -> torch.Tensor:
        return e.unsqueeze(-2).expand(*e.shape[:-1], self.n_streams, e.shape[-1]).clone()

    def coefficients(self, Y: torch.Tensor, k: int):
        """H_pre, H_post, H_res of loop k, all from the same pre-block streams Y."""
        flat = Y.flatten(-2)
        z = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + 1e-6)
        h_pre = torch.sigmoid(self.scale_pre[k] * (z @ self.proj_pre[k]) + self.bias_pre[k])
        h_post = 2 * torch.sigmoid(self.scale_post[k] * (z @ self.proj_post[k]) + self.bias_post[k])
        h_res = torch.sigmoid(self.scale_res[k] * (z @ self.proj_res[k]) + self.bias_res[k])
        return h_pre, h_post, h_res

    def read(self, Y: torch.Tensor, k: int) -> torch.Tensor:
        h_pre, _, _ = self.coefficients(Y, k)
        return (h_pre.unsqueeze(-1) * Y).sum(dim=-2)

    def write(self, Y: torch.Tensor, z_in: torch.Tensor, s: torch.Tensor, k: int) -> torch.Tensor:
        # Y is still the pre-block streams, so this recomputes exactly the coefficients read() used.
        _, h_post, h_res = self.coefficients(Y, k)
        w = (s + self.output_embedding[k]).unsqueeze(-2)
        return h_res.unsqueeze(-1) * Y + h_post.unsqueeze(-1) * w

    def finish(self, Y: torch.Tensor) -> torch.Tensor:
        return Y.mean(dim=-2)


def build_injection(d_model: int, cfg: LoopConfig) -> nn.Module:
    if cfg.injection == "parcae":
        return ParcaeInjection(d_model, cfg.parcae_init_decay)
    if cfg.injection == "none":
        return NoInjection()
    raise ValueError(f"unknown injection {cfg.injection!r}")


def build_hyper_connections(d_model: int, cfg: LoopConfig) -> nn.Module:
    if cfg.pass2_gate != "none":  # a gated second pass (LoopConfig.pass2_gate): the static module with pass 2's own tensors
        if cfg.pass2_gate not in ("hc", "scalar"):
            raise ValueError(f"unknown loop.pass2_gate {cfg.pass2_gate!r}: 'none', 'hc' or 'scalar'")
        if cfg.hyper_connections != "static":
            raise ValueError(f"loop.pass2_gate {cfg.pass2_gate!r} is built on the static hyper-connections, not "
                             f"{cfg.hyper_connections!r}")
        # two passes, or three when loop.segment_passes holds a 3 (pass_schedule.built_passes)
        return GatedSecondPass(d_model, cfg, n_passes=built_passes(cfg))
    if cfg.hyper_connections == "none":
        return NoHyperConnections()
    if cfg.hyper_connections == "static":
        return StaticHyperConnections(d_model, cfg)
    if cfg.hyper_connections == "dynamic_diag":
        return DynamicDiagHyperConnections(d_model, cfg)
    # mHC (Sinkhorn-projected H_res, arXiv 2512.24880) is optional and low priority; not implemented.
    raise ValueError(f"unknown hyper_connections {cfg.hyper_connections!r}")
