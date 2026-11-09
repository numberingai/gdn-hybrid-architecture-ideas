"""Per-channel decay (KDA) and the horizon floor in a Gated DeltaNet layer: GDN memory-rule the KDA
forms.

The survey gives the mechanism, the kernel and the floor's proposal. Config: GDNConfig.kda (ced/config.py KDAConfig); off by default.

THE RULE (Kimi Linear, arXiv 2510.26692, Eq. 1; fla 0.5.2 fla/ops/kda/naive.py naive_recurrent_kda), per value head:

    S_t = (I - beta_t k_t k_t^T) Diag(a_t) S_{t-1} + beta_t k_t v_t^T,   o_t = S_t^T q_t,   a_t = exp(g_t) in (0, 1]^{d_k}

that is, per token: S <- Diag(exp(g_t)) S (row i of S, key channel i, decays by its own a_{t,i}), then GDN's delta write
S += k (beta (v - S^T k))^T and the read S^T q. With g_t the same in every channel this is GDN's rule (ced/model/gdn.py
delta_rule_recurrence) bit for bit (tests/test_gdn_kda.py).

THE GATE (GatedDeltaNet.kda, a KDAGate; g has shape (B, L, H_v, d_k)):

    u = W_up W_down x the low-rank decay projection (KDAConfig.proj) free g_{t,h,i} = -exp(A_log_h) softplus(u_{t,h,i} +
    dt_bias_{h,i}) (fla's KimiDeltaAttention gate) floor g_{t,h,i} = ln(1 - 1/H_i) sigmoid(u_{t,h,i} + dt_bias_{h,i}), i
    < floor_channels beta = sigmoid(in_proj_b(x)) (unchanged: R1-entry's)

- in_proj_a and the per-head dt_bias give way to W_down, W_up and a per-channel dt_bias, stored flat (H_v d_k) as fla's (a 2-D
  tensor would go to Muon: ced/train/optim.py is_muon_param), entry h d_k + i for head h, channel i; A_log stays one per value head
  (fla's layer: A_log (H_v,), dt_bias (H_v d_k)). The output gate, the conv, q, k, v and beta are R1-entry's.
- proj "shared" (the default; KDAConfig): W_down d -> rank, W_up rank -> H_v d_k (Kimi Linear / fla's f_proj; rank 128 = fla's
  head_v_dim). At 60M it adds 129,532 parameters a GDN layer, 1,942,980 in all (3.05 % of R1-entry's 63.74M). proj "per_head" (an
  option, built before the ruling): W_down d -> rank x H_v, W_up per value head rank -> d_k on its own rank slice (the EDA paper's
  "rank-16 (per-head)" gate generator, arXiv 2606.26560 Appendix A; rank 16): 39,420 a layer, 591,300 in all (0.93 %). W_up is
  stored as one (H_v d_k, rank) matrix in both forms.
- THE HORIZON FLOOR (KDAConfig.floor_channels > 0): the first n key channels of every value head. H_i log-spaced from floor_min to
  floor_max tokens (H_i = floor_min x 2^(log2(floor_max / floor_min) i / (n - 1)); the survey's 64 to 262,144 over 32 channels,
  x1.3078 a channel). The bound ln(1 - 1/H_i) is a non-persistent buffer (floor_log, float32, rounded toward zero so it is never
  below the exact value): never a weight, never in a state dict, recomputed from the config. sigmoid <= 1, so the floored channels'
  g >= floor_log exactly in floating point at every token (rounding is monotone and floor_log is representable): a >= 1 - 1/H_i and
  the census's horizon h = 1 / (1 - a) >= H_i (exact up to float64's rounding of ln(1 - 1/H)). b (those channels' entries of
  dt_bias) starts at 0: sigmoid 1/2, horizons about 2 H_i at step 0 while u is near 0. At 60M (k_head_dim 128, as at 4B) the
  survey's fraction 1/4 is 32 channels: the arms set floor_channels 32.
- Weight decay: dt_bias (and so b) and A_log are in the AdamW group without decay (".dt_bias", ".A_log" in
  ced/train/optim.py NO_DECAY_SUFFIXES); W_down and W_up are the gate's matrices and go where in_proj_a went, AdamW (not Muon) with
  the run's decay (ADAMW_MATRIX_SUFFIXES). No zero-initialised gate is added (b is a bias inside the gate's sigmoid, zero meaning
  "half", not "off"); the arms still set optim.zero_gate_decay=none, the standing rule for new arms.
- Initialisation: W_down, W_up N(0, init_std) by the model's init_weights (fla's models: N(0, 0.02) for every Linear); A_log per
  head as R1-entry's (GDNConfig.init); dt_bias per channel with GDN's law for the scheme ("fla": dt ~ exp(U(ln 1e-3, ln 1e-1)) a
  channel, dt_bias = softplus^-1(dt); "hf": 1), then 0 on the floored channels. The random numbers drawn do not depend on the floor,
  so the KDA forms of a seed start from the same tensors except the floored channels' b. The global random stream differs from
  R1-entry's of the same seed (in_proj_a is gone and the new tensors are drawn in its place), so layer.head of a KDA arm is not
  R1-entry's head of that seed.

THE PATHS (GDNConfig.kernel, as for the scalar layer):
- "recurrence": kda_recurrence, the per-token loop above: the ground truth.
- "chunked" (and "fla" / "ced" off the CUDA training path, e.g. cached chunks with a carried state or the CPU): kda_chunked, the
  WY/UT chunked form with per-channel decays (ced/model/gdn_chunked.py's algorithm with the pairwise decay exp(c_i - c_j) taken per
  channel inside the products), matched to the recurrence in float64 by the tests.
- "fla" / "ced" on the CUDA training path (no state, varlen boundaries): kda_fused, fla 0.5.2's chunk_kda (fla/ops/kda/chunk.py:
  forward and backward, varlen, GVA, g precomputed in log space, use_gate_in_kernel / safe_gate off, scale 1 since q is already
  scaled). fla is imported at the first call (the GPU runs' venv, not on the VM). With kernel "ced" the input preparation stays our
  fused qkv_prep (the same function as the PyTorch code). A fast path is accepted only when it matches the reference:
  scripts/gdn_kda_check.py on a GPU (float64 reference, the 4x criterion of ced/utils/criterion.py).

NOT BUILT, refused by name: F3 (decay_cumsum "compensated"), the fused edges (edges != "plain"), FlashQLA (chunk_backend), CUDA
graphs' fixed-capacity GDN metadata, a zero-write second pass (keep), decoder slots (valid), the GDN-side arms (A, CCR), TST's
resolution-explicit gates (a bag phase), and every tool that computes a GDN layer from GatedDeltaNet.gates (the decode engine, the
MTP drafter, the GDN state diagnostics, gdn_memory, gate_floor, gdn_force): gates() refuses a per-channel layer; the head census
reads channel_gates() (scripts/gdn_g_dist.py --census, scripts/gdn_channel_census.py).
"""

from __future__ import annotations

import math
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

DECAYS = ("scalar", "channel")
PROJ_FORMS = ("per_head", "shared")
# the fused path's kernel (KDAConfig.kernel): fla's chunk_kda (the default, today's route) or K1 (ced/kernels/gdn_k1.py)
FUSED_KERNELS = ("chunk_kda", "k1")
# with GDN's scalar decay (decay "scalar"): the default (today's route: the 2T or n_h T interleave into our scalar kernel) or K2
# (ced/kernels/gdn_k2.py), read with the erase or Gated DeltaProduct only
SCALAR_KERNELS = ("chunk_kda", "k2")
K2_MAX_STEPS = 3  # K2's rows a token (gdn_k2_ref.MAX_ROWS)
PER_HEAD_RANK = 16  # the EDA paper's "rank-16 (per-head)" gate generator (arXiv 2606.26560, Appendix A)
# Calls of the fused path (kda_fused; k1_fused): the GPU checks read it to see that the model ran fla's chunk_kda or K1 (never the
# reference) on CUDA.
RAN = {"chunk_kda": 0, "k1": 0, "k2": 0}


# ------------------------------------------------------------------------------------------------------------------------------
# the keys


def is_on(cfg) -> bool:
    """cfg: a GDNConfig. True when its decay is per channel (KDAConfig.decay "channel")."""
    return cfg.kda.decay == "channel"


def _number(x) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(float(x))


def check_config(cfg) -> None:
    """GDNConfig.kda, checked (ValueError naming the key). With decay "scalar" every other key must be at its default."""
    from ced.config import KDAConfig

    k = cfg.kda
    if not isinstance(k, KDAConfig):
        raise ValueError(f"GDNConfig.kda {k!r}: a KDAConfig")
    if k.decay not in DECAYS:
        raise ValueError(f"GDNConfig.kda.decay {k.decay!r}: one of {DECAYS}")
    if k.decay == "scalar":
        default = KDAConfig()
        moved = [f"kda.{name}" for name in ("proj", "rank", "floor_channels", "floor_min", "floor_max", "kernel")
                 if getattr(k, name) != getattr(default, name) and not (name == "kernel" and k.kernel == "k2")]
        if moved:
            raise ValueError(f"GDNConfig {', '.join(moved)} {'is' if len(moved) == 1 else 'are'} read with kda.decay 'channel' "
                             "(per-channel decay); this layer's decay is 'scalar'")
        if k.kernel == "k2":
            check_k2(cfg)
        return
    if k.proj not in PROJ_FORMS:
        raise ValueError(f"GDNConfig.kda.proj {k.proj!r}: one of {PROJ_FORMS}")
    if k.kernel not in FUSED_KERNELS:
        hint = " ('k2' is K2, GDN's scalar decay's: read with kda.decay 'scalar')" if k.kernel == "k2" else ""
        raise ValueError(f"GDNConfig.kda.kernel {k.kernel!r}: one of {FUSED_KERNELS}{hint}")
    if type(k.rank) is not int or k.rank < 0:
        raise ValueError(f"GDNConfig.kda.rank {k.rank!r}: an integer >= 0 (0: the form's own, per_head {PER_HEAD_RANK}, "
                         "shared v_head_dim)")
    if type(k.floor_channels) is not int or not 0 <= k.floor_channels < cfg.k_head_dim:
        raise ValueError(f"GDNConfig.kda.floor_channels {k.floor_channels!r}: an integer in [0, k_head_dim = {cfg.k_head_dim}) "
                         "(at least one free channel: the floor leaves the other channels free)")
    if not (_number(k.floor_min) and _number(k.floor_max) and 1.0 < float(k.floor_min) <= float(k.floor_max)):
        raise ValueError(f"GDNConfig.kda.floor_min / floor_max {k.floor_min!r} / {k.floor_max!r}: horizons in tokens, "
                         "1 < floor_min <= floor_max")
    refused = [(cfg.decay_cumsum != "plain", f"decay_cumsum {cfg.decay_cumsum!r} (F3 is an option of our scalar-decay kernel)"),
               (cfg.edges != "plain", f"edges {cfg.edges!r} (the fused edges are our scalar-decay kernels)"),
               (cfg.chunk_backend != "triton", f"chunk_backend {cfg.chunk_backend!r} (FlashQLA's solve is exact for a scalar decay "
                                               "only)")]
    bad = [why for cond, why in refused if cond]
    if bad:
        raise ValueError(f"GDNConfig.kda.decay 'channel' (ced/model/gdn_kda.py) is not built with: {'; '.join(bad)}")


def check_k2(cfg) -> None:
    """KDAConfig.kernel "k2" on a scalar-decay layer (cfg a GDNConfig): K2 (ced/kernels/gdn_k2.py) runs the erase or Gated
    DeltaProduct with 1 to 3 steps, one of the two, on the fused path; refused by name otherwise."""
    erase = getattr(getattr(cfg, "erase", None), "mode", "none") == "eda"
    n_h = getattr(getattr(cfg, "dprod", None), "n_h", 1)
    refused = [(not erase and n_h == 1, "neither the erase (erase.mode 'eda') nor Gated DeltaProduct (dprod.n_h >= 2): "
                                        "GDN alone runs our scalar kernel"),
               (erase and n_h != 1, "the erase and Gated DeltaProduct together (K2 runs one of the two)"),
               (not isinstance(n_h, int) or n_h > K2_MAX_STEPS, f"dprod.n_h {n_h!r} (K2 takes 1 to {K2_MAX_STEPS} rows a token)"),
               (cfg.decay_cumsum != "plain", f"decay_cumsum {cfg.decay_cumsum!r} (F3 is not ported to K2)")]
    bad = [why for cond, why in refused if cond]
    if bad:
        raise ValueError(f"GDNConfig.kda.kernel 'k2' (K2, ced/kernels/gdn_k2.py) is not built with: {'; '.join(bad)}")


def rank_of(cfg) -> int:
    """The decay projection's bottleneck: kda.rank, or the form's own when 0 (per_head 16, shared v_head_dim)."""
    k = cfg.kda
    if k.rank:
        return k.rank
    return PER_HEAD_RANK if k.proj == "per_head" else cfg.v_head_dim


def floor_horizons(cfg) -> list[float]:
    """H_i of the floored channels, i = 0 .. n - 1: log-spaced from floor_min to floor_max (survey's formula,
    tools/analysis/gdn_rules_cost.py floor_spectrum on); [floor_min] for one channel; [] without a floor."""
    k = cfg.kda
    n = k.floor_channels if is_on(cfg) else 0
    if n == 0:
        return []
    lo, hi = float(k.floor_min), float(k.floor_max)
    if n == 1:
        return [lo]
    r = math.log2(hi / lo)
    return [lo * 2.0 ** (r * i / (n - 1)) for i in range(n)]


def floor_log_values(horizons: list[float], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """ln(1 - 1/H_i) in `dtype`, rounded toward zero: never below the float64 value, so g >= this means a >= 1 - 1/H_i."""
    exact = torch.tensor([math.log1p(-1.0 / h) for h in horizons], dtype=torch.float64)
    out = exact.to(dtype)
    below = out.to(torch.float64) < exact  # rounded away from zero (more negative): step one ulp toward zero
    return torch.where(below, torch.nextafter(out, torch.zeros_like(out)), out)


def added_parameters(cfg, d_model: int) -> dict:
    """The tensors one GDN layer gains and loses with per-channel decay (exact, from the shapes)."""
    hv, dk = cfg.n_v_heads, cfg.k_head_dim
    r = rank_of(cfg)
    down = d_model * r * (hv if cfg.kda.proj == "per_head" else 1)
    up = r * hv * dk
    dt = hv * dk
    removed = d_model * hv + hv  # in_proj_a and the per-head dt_bias
    return {"f_down": down, "f_up": up, "dt_bias": dt, "removed": removed, "per_layer": down + up + dt - removed}


# ------------------------------------------------------------------------------------------------------------------------------
# the gate's tensors


class KDAGate(nn.Module):
    """A GDN layer's per-channel decay tensors (GatedDeltaNet.kda): f_down, f_up (the low-rank decay projection), dt_bias (H_v d_k,)
    and the floor's bound floor_log (a non-persistent buffer, (floor_channels,))."""

    def __init__(self, d_model: int, cfg):
        super().__init__()
        check_config(cfg)
        assert is_on(cfg)
        self.hv, self.dk, self.proj, self.rank = cfg.n_v_heads, cfg.k_head_dim, cfg.kda.proj, rank_of(cfg)
        self.n_floor = cfg.kda.floor_channels
        self.horizons = floor_horizons(cfg)  # plain floats
        self.f_down = nn.Linear(d_model, self.rank * (self.hv if self.proj == "per_head" else 1), bias=False)
        self.f_up = nn.Linear(self.rank, self.hv * self.dk, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.hv * self.dk))  # flat, as fla's: one dimension keeps it off Muon
        self.register_buffer("floor_log", floor_log_values(self.horizons), persistent=False)

    @torch.no_grad()
    def reset_dt_bias(self, scheme: str) -> None:
        """dt_bias per channel with the law GatedDeltaNet.reset_gate_parameters uses per head ("fla": dt ~ exp(U(ln 1e-3,
        ln 1e-1)), clamped to >= 1e-4, dt_bias = softplus^-1(dt); "hf": 1), then 0 on the floored channels (the floor's b)."""
        if scheme == "fla":
            dt = torch.exp(torch.rand(self.hv, self.dk) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            dt = dt.clamp(min=1e-4)
            value = dt + torch.log(-torch.expm1(-dt))
        elif scheme == "hf":
            value = torch.ones(self.hv, self.dk)
        else:
            raise ValueError(f"unknown GDN init scheme {scheme!r}")
        value[:, : self.n_floor] = 0.0
        self.dt_bias.copy_(value.flatten())

    def decay_input(self, x: torch.Tensor) -> torch.Tensor:
        """u + dt_bias, (..., H_v, d_k): the low-rank projection of x plus the per-channel bias (float32 at least: dt_bias is)."""
        z = self.f_down(x)
        if self.proj == "shared":
            u = self.f_up(z).unflatten(-1, (self.hv, self.dk))
        else:  # per value head: its own rank slice through its own (d_k, rank) block of W_up
            # one batched GEMM over the value heads (H_v, B L, rank) x (H_v, rank, d_k). Not z[..., None, :] @ w^T: matmul
            # broadcasts W_up to every token and saves that copy for its backward, (B L H_v, rank, d_k): 402.7 MB a layer at 3 x
            # 8,192 tokens in bf16 (review 2026-10-22, measured on the VM under bf16 autocast at 3 x 1,024 tokens: the decay
            # projection's saved tensors 53.9 MB that way, 3.6 MB this way; x 8 for 3 x 8,192, computed)
            w = self.f_up.weight.to(z.dtype).view(self.hv, self.dk, self.rank)
            u = torch.einsum("...hr,hdr->...hd", z.unflatten(-1, (self.hv, self.rank)), w)
        return u + self.dt_bias.view(self.hv, self.dk)

    def extra_repr(self) -> str:
        return f"proj={self.proj}, rank={self.rank}, floor_channels={self.n_floor}"


def build(d_model: int, cfg) -> KDAGate | None:
    """GatedDeltaNet's hook: the gate's tensors, or None (decay "scalar": nothing is built, no random number is drawn)."""
    check_config(cfg)
    return KDAGate(d_model, cfg) if is_on(cfg) else None


def gates(layer, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(g, beta) of a per-channel layer: g (..., H_v, d_k) <= 0, beta (..., H_v). The free channels take KDA's gate, the floored
    ones the floor's (module docstring). An evaluation-only floor or cap of GatedDeltaNet (set_alpha_floor / set_alpha_cap; None
    in every model as trained) acts per value head on every channel, after the gate, as on the scalar layer."""
    kda = layer.kda
    pre = kda.decay_input(x)
    nf = kda.n_floor
    free = -torch.exp(layer.A_log)[:, None] * F.softplus(pre[..., nf:])
    if nf:
        c = kda.floor_log
        if c.dtype not in (torch.float32, torch.float64) or pre.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"the horizon floor is formed in float32 or float64 (floor {c.dtype}, gate {pre.dtype}): a "
                             "narrower float would round the bound below ln(1 - 1/H)")
        g = torch.cat([c.to(pre.dtype) * torch.sigmoid(pre[..., :nf]), free], dim=-1)
    else:
        g = free
    if layer.g_floor is not None:  # evaluation only, per value head: alpha = max(alpha, floor) in every channel
        g = torch.maximum(g, layer.g_floor.to(device=g.device, dtype=g.dtype)[:, None])
    if layer.g_cap is not None:  # evaluation only, per value head: alpha = min(alpha, cap) in every channel (after the floor)
        g = torch.minimum(g, layer.g_cap.to(device=g.device, dtype=g.dtype)[:, None])
    return g, torch.sigmoid(layer.in_proj_b(x))


# ------------------------------------------------------------------------------------------------------------------------------
# the delta rule with per-channel decay: the reference loop, the chunked form, the fused kernel


def kda_recurrence(q, k, v, g, beta, is_doc_start, S):
    """The KDA rule one token at a time: the ground truth. q, k: (B, L, H, d_k) (already repeated to the value heads); v: (B, L,
    H, d_v); g: (B, L, H, d_k); beta: (B, L, H); is_doc_start: (B, L) bool; S: (B, H, d_k, d_v). Per token: reset at a document
    start, S <- Diag(exp(g_t)) S, write k (x) beta (v - S^T k), read S^T q. Returns (outputs (B, L, H, d_v), the final state).
    ced/model/gdn.py delta_rule_recurrence with the decay a row of S at a time."""
    outputs = []
    steps = zip(q.unbind(1), k.unbind(1), v.unbind(1), g.unbind(1), beta.unbind(1), is_doc_start.unbind(1))
    for q_t, k_t, v_t, g_t, beta_t, start_t in steps:
        keep = (~start_t).to(S.dtype)[:, None, None, None]
        S = S * keep  # a new document starts from an empty state
        S = S * torch.exp(g_t)[..., None]  # row i (key channel i) by its own decay
        predicted = torch.einsum("bhkv,bhk->bhv", S, k_t)
        delta = beta_t[..., None] * (v_t - predicted)
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, delta)
        outputs.append(torch.einsum("bhkv,bhk->bhv", S, q_t))
    return torch.stack(outputs, dim=1), S


def kda_chunked(q, k, v, g, beta, is_doc_start, S, chunk_size: int = 64, checkpoint: bool = False):
    """kda_recurrence's function in the WY / UT chunked form (ced/model/gdn_chunked.py's algorithm and conventions: fixed chunks,
    documents by masks, padding with g = 0 and beta = 0, the pairwise exponents summed directly, the double `where`), with the
    decay per channel. Inside a chunk with entry state S_0, positions i, j and the per-channel cumulative log decay c_i = g_0 + ...
    + g_i (a d_k vector):

        A[i, j] = beta_i sum_d k_{i,d} k_{j,d} exp(c_{i,d} - c_{j,d})          j < i, same document
        U       = (I + A)^-1 diag(beta) V,  W = (I + A)^-1 diag(beta) (exp(c) * K)   (rows; the S_0 term in the first segment only)
        w       = U - W S_0
        o_i     = (exp(c_i) * q_i)^T S_0 + sum_{j <= i} (sum_d q_{i,d} k_{j,d} exp(c_{i,d} - c_{j,d})) w_j
        S_C     = Diag(exp(c_last)) S_0 + sum_j (exp(c_last - c_j) * k_j) w_j^T

    The pairwise per-channel decays are a (C, C, d_k) tensor a chunk and head, formed inside the loop over chunks, so the forward's
    memory is one chunk's at any length. Every exponent kept is <= 0 when g <= 0. Returns (outputs (B, L, H, d_v), the final state),
    in the dtype the recurrence would return; the work is float32 or wider.

    checkpoint: each chunk's step under torch.utils.checkpoint (recomputed in the backward; the same values and gradients), so a
    backward keeps one chunk's pairwise tensors, not every chunk's: the GPU check's float64 ground truth at 8k-token rows
    (scripts/gdn_kda_check.py). False (the default, every model path): the step is called directly."""
    assert chunk_size >= 1
    B, L, H, dk = q.shape
    dv = v.shape[-1]
    assert g.shape == (B, L, H, dk) and beta.shape == (B, L, H), (g.shape, beta.shape)
    out_dtype = reduce(torch.promote_types, (t.dtype for t in (q, k, v, g, beta, S)))
    work = torch.promote_types(out_dtype, torch.float32)
    C = min(chunk_size, L)
    n = math.ceil(L / C)
    pad = n * C - L

    def per_head(x: torch.Tensor) -> torch.Tensor:
        """(B, L, H, ...) -> (B, H, n, C, ...), padded at the end of the row with zeros."""
        x = x.to(work).transpose(1, 2)
        trailing = x.dim() - 3
        x = F.pad(x, (0, 0) * trailing + (0, pad))
        return x.reshape(B, H, n, C, *x.shape[3:])

    q, k, v, g, beta = map(per_head, (q, k, v, g, beta))
    starts = F.pad(is_doc_start.to(torch.long), (0, pad)).reshape(B, 1, n, C)
    seg = starts.cumsum(-1)  # 0 = the document carried in from the previous chunk
    causal = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    strictly_lower = causal.tril(-1)
    eye = torch.eye(C, dtype=work, device=q.device)
    zero = torch.zeros((), dtype=work, device=q.device)

    def step(state, q_c, k_c, v_c, g_c, b_c, seg_c):
        """One chunk: (its outputs, the state after it). q_c, k_c, g_c: (B, H, C, dk); v_c: (B, H, C, dv); b_c: (B, H, C);
        seg_c: (B, 1, C)."""
        pair_ok = (seg_c[..., :, None] == seg_c[..., None, :]) & causal  # (B, 1, C, C): j <= i, the same document
        in_first = (seg_c == 0).to(work)[..., None]  # (B, 1, C, 1): sees the incoming state
        in_last = (seg_c == seg_c[..., -1:]).to(work)[..., None]  # writes into the outgoing state
        carry = (seg_c[..., -1] == 0).to(work)[..., None]  # (B, 1, 1): no start in the chunk, S_0 survives
        # pair_exp[..., i, j, :] = g_{j+1} + ... + g_i for j < i and 0 for j >= i, a channel at a time, summed directly
        pair_exp = torch.where(strictly_lower[:, :, None], g_c[..., :, None, :], zero).cumsum(-3)  # (B, H, C, C, dk)
        ok = pair_ok[..., None]
        pair_decay = torch.where(ok, torch.exp(torch.where(ok, pair_exp, zero)), zero)
        gamma = torch.exp(g_c.cumsum(-2))  # (B, H, C, dk): exp(c_i)
        to_end = torch.exp(pair_exp[..., -1, :, :])  # (B, H, C, dk): exp(c_last - c_j)
        k_beta = k_c * b_c[..., None]
        A = torch.einsum("bhid,bhjd,bhijd->bhij", k_beta, k_c, pair_decay) * strictly_lower
        rhs = torch.cat([v_c * b_c[..., None], k_beta * gamma * in_first], dim=-1)
        solved = torch.linalg.solve_triangular(eye + A, rhs, upper=False, unitriangular=True)
        U, W = solved.split([dv, dk], dim=-1)
        written = U - W @ state  # (B, H, C, dv): w = U - W S_0
        intra = torch.einsum("bhid,bhjd,bhijd->bhij", q_c, k_c, pair_decay)  # j <= i, the same document
        out = (q_c * gamma * in_first) @ state + intra @ written
        return out, state * (gamma[..., -1, :] * carry)[..., None] + (k_c * to_end * in_last).transpose(-1, -2) @ written

    state = S.to(work)
    outputs = []
    for chunk in zip(*(t.unbind(2) for t in (q, k, v, g, beta, seg))):
        if checkpoint:
            from torch.utils.checkpoint import checkpoint as recompute

            out, state = recompute(step, state, *chunk, use_reentrant=False)
        else:
            out, state = step(state, *chunk)
        outputs.append(out)
    o = torch.stack(outputs, dim=2).reshape(B, H, n * C, dv)[:, :, :L].transpose(1, 2)
    return o.to(out_dtype), state.to(out_dtype)


def load_chunk_kda():
    """fla 0.5.2's chunk_kda, imported at the first call (fla is in the GPU runs' venv, not on the VM). A
    function, so the CPU tests can route the fused path through an emulation."""
    from fla.ops.kda import chunk_kda

    return chunk_kda


@torch.compiler.disable  # never traced: an opaque fused op, varlen inputs that change length every batch (ced/train/varlen.py)
def kda_fused(q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu=None):
    """The KDA rule from an empty state with fla 0.5.2's chunk_kda (Triton forward and backward; the varlen and GVA path: our
    value heads are a multiple of the key heads, so its CUTLASS FlashKDA and TileLang backends, which refuse GVA, never take the
    call). q, k: (B, L, H_k, d_k) L2-normalised, q scaled by d_k^-0.5 (scale 1 here); v: (B, L, H_v, d_v); g: (B, L, H_v, d_k)
    log decay (passed as it is: use_gate_in_kernel and safe_gate off, so the kernel takes g without a bound); beta: (B, L, H_v)
    after its sigmoid. Rows are flattened into one sequence and the state restarts at every boundary of cu_seqlens (row and
    document starts), exactly the reference's reset at is_doc_start. g goes to the kernel in float32 (float64 with float64 inputs,
    for the CPU emulation), beta in q's dtype. Returns (B, L, H_v, d_v)."""
    chunk_kda = load_chunk_kda()
    B, L = q.shape[:2]
    flat = lambda t: t.reshape(1, B * L, *t.shape[2:])  # noqa: E731
    g_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    RAN["chunk_kda"] += 1
    o, _ = chunk_kda(flat(q), flat(k), flat(v), flat(g).to(g_dtype), flat(beta).to(q.dtype), scale=1.0, initial_state=None,
                     output_final_state=False, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                     use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, safe_gate=False, lower_bound=None,
                     cu_seqlens=cu_seqlens.long(), cu_seqlens_cpu=cu_seqlens_cpu.long() if cu_seqlens_cpu is not None else None)
    return o.reshape(B, L, *o.shape[2:])


def load_k1():
    """ced/kernels/gdn_k1.py's k1_delta_rule, imported at the first call (Triton is in the GPU runs' venv, not on the VM). A function,
    so the CPU tests can route the fused path through the kernels' emulation (tests/tl_emu.py) or the reference."""
    from ced.kernels.gdn_k1 import k1_delta_rule

    return k1_delta_rule


def k1_prologue(layer, x) -> dict | None:
    """K1's gate inputs (the gate in the kernel's prologue): {"z": W_down x (B, L, rank), "w_up", "dt_bias", "a_log", "floor_log"} for
    the shared projection (KDAConfig.proj "shared") as trained (no evaluation-only floor or cap); None otherwise (proj "per_head",
    or set_alpha_floor / set_alpha_cap): then K1 takes g from gates() as chunk_kda does."""
    kda = layer.kda
    if kda.proj != "shared" or layer.g_floor is not None or layer.g_cap is not None:
        return None
    return {"z": kda.f_down(x), "w_up": kda.f_up.weight, "dt_bias": kda.dt_bias, "a_log": layer.A_log, "floor_log": kda.floor_log}


@torch.compiler.disable  # never traced: an opaque fused op on varlen inputs, as kda_fused
def k1_fused(q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu=None, e=None, gamma=None, gate=None):
    """The fused path with K1 (KDAConfig.kernel "k1"; ced/kernels/gdn_k1.py): kda_fused's call with our kernel. Without e and gamma
    KDA alone (r = 1 row a token); with them (e (B, L, H_k, d_k) at the key heads, gamma (B, L, H_v)) EDA's erase and KDA's write of
    each token as two rows of one WY form (r = 2), no interleave. gate (k1_prologue's, g None): the gate formed in the kernel's
    prologue from W_down x and W_up (g never written). Rows flattened as kda_fused's; g in float32 (float64 with float64 inputs);
    beta, gamma and e in q's dtype. Returns (B, L, H_v, d_v)."""
    k1 = load_k1()
    B, L = q.shape[:2]
    flat = lambda t: t.reshape(1, B * L, *t.shape[2:])  # noqa: E731
    g_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    RAN["k1"] += 1
    kw = {} if e is None else {"e": flat(e).to(q.dtype), "gamma": flat(gamma).to(q.dtype)}
    if gate is not None:
        kw["gate"] = {**gate, "z": flat(gate["z"])}
    o = k1(flat(q), flat(k), flat(v), None if g is None else flat(g).to(g_dtype), flat(beta).to(q.dtype), cu_seqlens.long(),
           cu_seqlens_cpu.long() if cu_seqlens_cpu is not None else None, **kw)
    return o.reshape(B, L, *o.shape[2:])


def load_k2():
    """ced/kernels/gdn_k2.py's k2_delta_rule, imported at the first call (Triton is in the GPU runs' venv, not on the VM). A function,
    so the CPU tests can route the fused path through the kernels' emulation (tests/tl_emu.py) or the reference."""
    from ced.kernels.gdn_k2 import k2_delta_rule

    return k2_delta_rule


@torch.compiler.disable  # never traced: an opaque fused op on varlen inputs, as k1_fused
def k2_fused(q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu=None, e=None, gamma=None):
    """The fused path with K2 (KDAConfig.kernel "k2" on a scalar-decay layer; ced/kernels/gdn_k2.py), GDN's decay g (B, L, H_v). With e
    and gamma (e (B, L, H_k, d_k) at the key heads, gamma (B, L, H_v); k (B, L, H_k, d_k), v (B, L, H_v, d_v), beta (B, L, H_v)) EDA's
    erase and GDN's write of each token as two rows of one WY form; without them Gated DeltaProduct's n_h steps as n_h rows (k (B, L,
    n_h, H_k, d_k), v (B, L, n_h, H_v, d_v), beta (B, L, n_h, H_v), the layer's stacks). No interleave. Rows flattened as k1_fused's;
    g in float32 (float64 with float64 inputs); the others in q's dtype. Returns (B, L, H_v, d_v)."""
    k2 = load_k2()
    B, L = q.shape[:2]
    flat = lambda t: t.reshape(1, B * L, *t.shape[2:])  # noqa: E731
    g_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    RAN["k2"] += 1
    kw = {} if e is None else {"e": flat(e).to(q.dtype), "gamma": flat(gamma).to(q.dtype)}
    o = k2(flat(q), flat(k).to(q.dtype), flat(v).to(q.dtype), flat(g).to(g_dtype), flat(beta).to(q.dtype), cu_seqlens.long(),
           cu_seqlens_cpu.long() if cu_seqlens_cpu is not None else None, **kw)
    return o.reshape(B, L, *o.shape[2:])


# ------------------------------------------------------------------------------------------------------------------------------
# the layer's forward with per-channel decay


def forward(layer, x, ctx, state=None, valid=None, keep=False, arms=None, x_write=None, ccr=None, x_read=None, lerped=False):
    """GatedDeltaNet.forward of a per-channel layer (its first line hands the call here): R1-entry's input preparation, gates of
    this module, the paths above, R1-entry's output norm and projection. Returns (y, GDNState) as the scalar layer does."""
    from ced.model import gdn as G

    refused = [(keep, "keep (the zero-write second pass reads a scalar decay)"),
               (valid is not None, "valid (decoder slots)"),
               (arms is not None or x_write is not None or ccr is not None or x_read is not None or layer.arms_attached,
                "the GDN-side arms (model.encoder.gdn_write_mix, gdn_ccr_spacing)"),
               (ctx.gdn_scale != 1, f"TST's resolution-explicit gates (a bag phase of {ctx.gdn_scale} tokens)")]
    bad = [why for cond, why in refused if cond]
    if bad:
        raise ValueError(f"GDNConfig.kda.decay 'channel' (ced/model/gdn_kda.py) is not built with: {'; '.join(bad)}")
    cfg = layer.cfg
    B, L, _ = x.shape
    hk, hv, dk, dv = cfg.n_k_heads, cfg.n_v_heads, cfg.k_head_dim, cfg.v_head_dim
    K = cfg.conv_kernel
    u_pre = layer.in_proj_qkv(x)
    fused_ok = state is None and ctx.cu_seqlens is not None and G.fla_supported(x)
    use_fused = cfg.kernel in ("fla", "ced") and fused_ok
    # input_dtype "bfloat16": q, k, v and beta reach the fused kernel in bf16 under bf16 autocast (g stays float32), as on the
    # scalar layer (GDNConfig.input_dtype)
    kernel_dtype = (torch.bfloat16 if use_fused and cfg.input_dtype == "bfloat16" and u_pre.dtype == torch.bfloat16 else None)
    if layer._fused_prep and fused_ok and L >= K - 1:  # kernel "ced": our fused input preparation (the same function)
        q, k, v = G.qkv_prep(u_pre, ctx.doc, layer.conv_weight, hk, hv, dk, out_dtype=kernel_dtype)
        conv_input, conv_doc = u_pre[:, L - (K - 1):], ctx.doc[:, L - (K - 1):]
    else:
        u, conv_input, conv_doc = layer.short_conv(u_pre, ctx, state)
        u = F.silu(u)
        q, k, v = torch.split(u, [hk * dk, hk * dk, hv * dv], dim=-1)
        q = G.l2_normalize(q.view(B, L, hk, dk)) * dk**-0.5
        k = G.l2_normalize(k.view(B, L, hk, dk))
        v = v.view(B, L, hv, dv)
        if kernel_dtype is not None:
            q, k, v = q.to(kernel_dtype), k.to(kernel_dtype), v.to(kernel_dtype)
    prologue = k1_prologue(layer, x) if use_fused and cfg.kda.kernel == "k1" else None
    if prologue is None:
        g, beta = gates(layer, x)
    else:  # K1 forms g in its prologue: beta as gates() forms it
        g, beta = None, torch.sigmoid(layer.in_proj_b(x))
    if kernel_dtype is not None:
        beta = beta.to(kernel_dtype)
    if use_fused:
        if ctx.gdn_chunks is not None:
            raise ValueError("GDNConfig.kda.decay 'channel' with TrainConfig.cuda_graphs: not built (the fixed-capacity GDN "
                             "metadata is our scalar kernel's)")
        if cfg.kda.kernel == "k1":  # KDAConfig.kernel: K1 (r = 1), the gate in its prologue when k1_prologue gives it
            o = k1_fused(q, k, v.contiguous(), g, beta, ctx.cu_seqlens, ctx.cu_seqlens_cpu, gate=prologue)
        else:  # chunk_kda (the default: today's path)
            o = kda_fused(q, k, v.contiguous(), g, beta, ctx.cu_seqlens, ctx.cu_seqlens_cpu)
        S = None  # training forward: the final state is not needed (no cache is built)
    else:
        q = q.repeat_interleave(hv // hk, dim=2)
        k = k.repeat_interleave(hv // hk, dim=2)
        S0 = x.new_zeros(B, hv, dk, dv) if state is None else state.S
        if cfg.kernel in ("chunked", "fla", "ced") and L > 1:
            o, S = kda_chunked(q, k, v, g, beta, ctx.is_doc_start, S0, cfg.chunk_size)
        else:
            o, S = kda_recurrence(q, k, v, g, beta, ctx.is_doc_start, S0)
    if layer.head_keep is not None:  # evaluation only: the zeroed heads' readouts are 0 (GatedDeltaNet.set_head_zero)
        o = layer.keep_heads(o)
    z = layer.in_proj_z(x).view(B, L, hv, dv)
    if getattr(layer, "mx_out_norm", False) and layer.training:  # model.fp8.chains gdn_out_norm (ced/model/mx_wire.py)
        from ced.model import mx_wire

        y = mx_wire.gated_norm_proj(layer, o, z)
    else:
        y = layer.out_proj(layer.norm(o, z).flatten(2))
    return y, G.GDNState(conv_input=conv_input, conv_doc=conv_doc, S=S)
