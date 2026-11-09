"""ASEntmax attention: the exact, slow reference (the three-variant attention experiment; spec and our
implementation choices).

alpha = 1.5 entmax over the logits of the scaled queries:

    z_ts  = (c_t q_t) . k_s * head_dim^-0.5          s allowed for t (attention.allowed / slots_allowed)
    p_ts  = [z_ts / 2 - tau_t]_+^2                   tau_t such that sum_s p_ts = 1
    out_t = sum_s p_ts v_s

c_t (per head and position) is applied to q in GatedAttention.queries (ced/model/attention.py), after QK-norm
and RoPE, before this kernel. What is here does not know about c: it is the alpha = 1.5 entmax attention of
whatever q it is given.

The threshold tau is solved exactly (Peters, Niculae and Martins 2019, the sort-based closed form for
alpha = 1.5), in at least float32 (float64 inputs stay float64, for the exact tests):

- With x = z / 2, a support of the k largest entries x_(1) >= ... >= x_(k) gives
  sum_{i <= k} (x_(i) - tau)^2 = 1, so tau_k = mean_k - sqrt((1 - k (meansq_k - mean_k^2)) / k), the smaller root;
  the support size is k* = max{k : tau_k <= x_(k)}, and tau = tau_{k*}.
- Candidates. Every support entry satisfies x_i >= x_max - 1, that is z_i >= z_max - 2. Proof: the largest
  entry is in the support (tau < x_max, else every p would be 0 and the sum 1 impossible), and
  p_max = (x_max - tau)^2 <= sum_i p_i = 1, so tau >= x_max - 1; a support entry has x_i > tau >= x_max - 1.
  So the candidates {z_i >= z_max - 2} are a superset of the support, the sorted candidates are a prefix of
  the sorted row, and tau_k, k <= #candidates, are the same as over the whole row. No k beyond the
  candidates can satisfy the condition: if tau_k <= x_(k) < x_max - 1 with tau_k a root (delta_k >= 0), then
  1 = sum_{i <= k} (x_(i) - tau_k)^2 >= (x_max - tau_k)^2 > 1; and if delta_k < 0 then tau_k = mean_k (the
  clamp) and tau_k <= x_(k) would make every x_(i), i <= k, equal to the mean, a zero spread, so delta_k = 1 /
  k > 0, a contradiction. Hence k* (over the candidates) = k* (over the row).
- The solver sorts only the candidates: topk with k = the largest candidate count of the call (a full sort
  when that is more than a quarter of the row), entries past a row's own candidate count replaced by
  x_max - 2 (finite, never in the support by the argument above). The prefix sums that locate k* are blocked
  triangular matmuls (deterministic on every device, unlike a CUDA float cumsum outside deterministic mode).
  Once k* is known, tau is recomputed from the k* largest entries with two-pass sums (their mean, then the
  sum of squared deviations), the same closed form without the cancellation of meansq - mean^2 in float32
  over thousands of entries.
- A row with no allowed key (never produced by the models: a query always sees itself, an empty slot reads
  key 0) gets tau = +inf, p = 0 and output 0.

The autograd Function (`entmax_attention`) works per batch row and per chunk of query rows, so no (B, Lq, S)
mask or (B, H, Lq, S) score tensor exists: the mask of a chunk is built with attention.allowed /
slots_allowed on that chunk's queries, and the keys are cut to the chunk's range of allowed columns [lo, hi)
(every column outside is masked for every row and head, so this is exact). It saves q, k, v, tau (B, Hq, Lq)
and the (B, Lq) / (B, S) metadata only; the backward recomputes z and p from tau, chunk by chunk:

    s   = sqrt(p) = [z / 2 - tau]_+
    dp  = dout . v
    dz  = s * dp - s * (sum_s s * dp) / (sum_s s)          (the entmax Jacobian, diag(s) - s s^T / sum s)
    dv  = p^T dout;    dq = dz k * head_dim^-0.5;    dk = dz^T q * head_dim^-0.5 (summed over a KV head's query heads)

Precision: scores, tau, p, the outputs and every gradient in float32 (float64 for float64 inputs) with
autocast off and TF32 off for the call; the output is cast to q's dtype, the gradients to q's, k's and v's.
GQA / MQA: key/value head j serves query heads j * group ... (j + 1) * group - 1, as attention.attend.
Deterministic: matmuls, sorts, elementwise ops and reductions only (no atomics, no scatter-add); the same
inputs and chunking give the same bits run to run. The chunking changes results at rounding level only.

torch.compile does not trace into it (torch.compiler.disable): inside a compiled layer the call is a graph
break, the rest of the layer compiles as usual.

Training statistics (`EntmaxStats`, attached by the trainer to each asentmax attention module): with
stats.active the forward also records, per chunk, what it already has: the support size of every valid row
(the log step's snapshot), which keys are in some valid row's support ("alive") and which keys some valid
row may see ("reachable"), accumulated on the device; one host sync on the log step (EntmaxStats.summary).
"""

from __future__ import annotations

import contextlib
import math

import torch

from ced.model.norms import compute_dtype

# One (Hq, rows, S) float32 score tensor of a chunk holds at most this many elements when
# AttentionConfig.asentmax_chunk is 0 (auto): 128 MiB; 1,024 rows at S = 8,192 with 4 heads, 128 at 65,536.
AUTO_CHUNK_ELEMENTS = 1 << 25
_BLOCK = 64  # block size of the triangular-matmul prefix sums


# ---------------------------------------------------------------------------------------------------
# the threshold
# ---------------------------------------------------------------------------------------------------


def prefix_sum(x: torch.Tensor) -> torch.Tensor:
    """Inclusive prefix sums along the last dimension, from blocked triangular matmuls (exact summation order
    fixed by shapes: deterministic on CPU and CUDA). Run it with TF32 off (entmax_attention does)."""
    K = x.shape[-1]
    nb = -(-K // _BLOCK)
    pad = nb * _BLOCK - K
    xp = torch.nn.functional.pad(x, (0, pad)) if pad else x
    xb = xp.reshape(*x.shape[:-1], nb, _BLOCK)
    tri = torch.ones(_BLOCK, _BLOCK, dtype=x.dtype, device=x.device).triu()  # out[j] = sum_{i <= j} in[i]
    within = xb @ tri
    if nb > 1:
        totals = within[..., -1]  # (..., nb)
        excl = torch.ones(nb, nb, dtype=x.dtype, device=x.device).triu(1)  # offset[j] = sum_{i < j} total[i]
        within = within + (totals @ excl)[..., None]
    return within.reshape(*x.shape[:-1], nb * _BLOCK)[..., :K]


def entmax15_threshold(z: torch.Tensor) -> torch.Tensor:
    """tau (...) of alpha = 1.5 entmax over the last dimension of z (..., S): p = [z / 2 - tau]_+^2 sums to 1.
    Masked entries are -inf. Exact closed form over the candidates z >= z_max - 2 (module docstring), in z's
    dtype. A row without a finite entry gets +inf."""
    x = z * 0.5
    xmax = x.amax(-1, keepdim=True)
    finite = torch.isfinite(xmax)
    x0 = torch.where(finite, xmax, torch.zeros_like(xmax))
    xs = x - x0  # <= 0; -inf where masked
    ncand = (xs >= -1.0).sum(-1)  # the candidates: a superset of the support
    out_shape = z.shape[:-1]
    K = int(ncand.max()) if ncand.numel() else 0  # host sync: topk's k
    if K == 0:
        return torch.full(out_shape, math.inf, dtype=z.dtype, device=z.device)
    S = z.shape[-1]
    if 4 * K <= S:
        vals = torch.topk(xs, K, dim=-1, largest=True, sorted=True).values
    else:
        vals = torch.sort(xs, dim=-1, descending=True).values[..., :K]
    rank = torch.arange(K, device=z.device)
    in_cand = rank < ncand[..., None]
    vals = torch.where(in_cand, vals, torch.full_like(vals, -2.0))  # finite; never in the support
    rho = (rank + 1).to(z.dtype)
    mean = prefix_sum(vals) / rho
    mean_sq = prefix_sum(vals * vals) / rho
    delta = (1.0 - rho * (mean_sq - mean * mean)) / rho
    tau_k = mean - torch.sqrt(torch.clamp(delta, min=0.0))
    ok = (tau_k <= vals) & in_cand
    k_star = torch.where(ok, rank + 1, torch.zeros_like(rank)).amax(-1)  # >= 1 on a finite row (k = 1 always holds)
    # tau from the k* largest entries, two-pass (no cancellation): sum_{i <= k*} (x_(i) - tau)^2 = 1
    sup = rank < k_star[..., None]
    n = k_star.clamp_min(1).to(z.dtype)
    m = torch.where(sup, vals, torch.zeros_like(vals)).sum(-1) / n
    dev = torch.where(sup, vals - m[..., None], torch.zeros_like(vals))
    var_sum = (dev * dev).sum(-1)
    tau = m - torch.sqrt(torch.clamp((1.0 - var_sum) / n, min=0.0)) + x0[..., 0]
    return torch.where(finite[..., 0], tau, torch.full_like(tau, math.inf))


def entmax15(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(p, tau): alpha = 1.5 entmax of z (..., S) over the last dimension (masked entries -inf), p = [z/2 - tau]_+^2,
    support exactly {z > 2 tau}. Not differentiable (the attention Function has its own backward)."""
    tau = entmax15_threshold(z)
    s = torch.clamp(z * 0.5 - tau[..., None], min=0.0)
    return s * s, tau


def entmax_rows(scores: torch.Tensor, allow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The diagnostics' explicit rows (ced/eval/attn_diag.py): scores (B, H, P, S) (scale included), allow
    (B, P, S) -> (probabilities (B, H, P, S), tau (B, H, P)), the same solver as the attention."""
    return entmax15(scores.masked_fill(~allow[:, None], float("-inf")))


# ---------------------------------------------------------------------------------------------------
# training statistics
# ---------------------------------------------------------------------------------------------------


def _quantiles(x: torch.Tensor, qs: tuple) -> list[float]:
    """torch.quantile (linear interpolation), rounded to 5 significant digits."""
    return [float(f"{float(v):.5g}") for v in torch.quantile(x.double(), torch.tensor(qs, dtype=torch.float64, device=x.device))]


class EntmaxStats:
    """Training statistics of one asentmax attention module (collapsed heads, keys that never receive gradient
    over a window of steps), logged under metrics "entmax" on the log steps (ced/train/trainer.py). Nothing
    here changes the computation.

    The trainer sets `active` around every training micro-batch's forward (never during evals) and `snapshot`
    on the log step's first micro-batch; it calls end_microbatch() after each micro-batch's forward, end_step()
    after the step, and summary() on the log step (the one host sync; the window restarts).

    Rows are valid query rows (an empty decoder slot, document < 0, is not one). A key is reachable when some
    valid row may see it (the mask), alive when it is in some valid row's support for that head; a reachable
    key that is not alive gets no gradient from that head (p = 0 and dp/dz = 0 there). Within a micro-batch
    the calls of one module with the same key shape (CED v2's decoder layer in every loop) are OR-ed: a key
    counts as dead only if no call's support has it.

    - support (snapshot): per head, the support size of every valid row: mean, p10, p50, p90, p99, max, and the
      share of rows with support <= 2; collapsed: the share of this module's heads whose mean support <= 2.
    - dead_keys: per head, the share of reachable keys that are dead, per step (dead keys summed over the
      step's micro-batches over reachable keys summed); "step" is the log step's, "mean" and "max" the
      per-step share's mean and max over the window (the steps since the previous log step, this one
      included; after a resume the first window is shorter: "steps"); "all_steps": by key position s (the
      key's column in the row), the share of positions reachable at some step of the window that were dead
      at every step of it, for every row (None if the key count changed within the window).
      With GQA (n_kv_heads < n_heads) also "kv_heads": the same four figures per KV head, a key counting as
      dead only when it is dead for every query head of the group. That is the key that gets no gradient at
      all in this layer (its k and v are read only through those query heads: dk and dv are 0 there); a key
      dead for one query head alone still gets gradient through the others.
    - scale (snapshot): c = 1 + beta ln(pos + 2)^gamma over the valid rows and heads: mean, p10, p90; the means
      of beta = softplus(.) and gamma = 2 tanh(.); per head means of c."""

    def __init__(self, n_heads: int, n_kv_heads: int | None = None):
        self.n_heads = n_heads
        self.n_kv = n_kv_heads or n_heads
        assert n_heads % self.n_kv == 0, (n_heads, self.n_kv)
        self.active = False
        self.snapshot = False
        self._pending = None  # (alive (B, H, S) bool, reach (B, S) bool) of the current micro-batch
        self._support: list[torch.Tensor] = []  # snapshot: (H, n_rows) support sizes per call
        self._scale: list[tuple] = []  # snapshot: (c, beta, gamma) (n, H) at valid rows
        self._step_dead = None
        self._step_reach = None
        self._step_dead_kv = None
        self._reset_window()

    def _reset_window(self) -> None:
        self._win_sum = None  # (H,) sum of per-step dead shares
        self._win_max = None
        self._win_last = None
        self._win_steps = 0
        self._pos_alive = None  # (H, S) alive at some step of the window, any row
        self._pos_reach = None  # (S,)
        self._pos_ok = True
        self._kv_sum = self._kv_max = self._kv_last = None  # (Hkv,) the same per KV head (GQA)
        self._pos_alive_kv = None  # (Hkv, S)

    # -- recording (from the Function's forward, eager) --------------------------------------------

    def begin_call(self, B: int, Lq: int, S: int, device: torch.device) -> dict:
        """Per-call buffers: alive / reach (OR-ed into the micro-batch's when the shape matches) and, on a
        snapshot, the support size of every row."""
        p = self._pending
        if p is None or p[0].shape != (B, self.n_heads, S):
            self._flush()
            self._pending = (torch.zeros(B, self.n_heads, S, dtype=torch.bool, device=device),
                             torch.zeros(B, S, dtype=torch.bool, device=device))
        call = {"alive": self._pending[0], "reach": self._pending[1]}
        if self.snapshot:
            call["support"] = torch.zeros(B, self.n_heads, Lq, dtype=torch.int32, device=device)
            call["valid"] = torch.zeros(B, Lq, dtype=torch.bool, device=device)
        return call

    @staticmethod
    def record_chunk(call: dict, b: int, r0: int, r1: int, lo: int, hi: int, in_support: torch.Tensor,
                     allow: torch.Tensor, row_valid: torch.Tensor) -> None:
        """in_support (H, c, S') = p > 0 over the chunk's key range [lo, hi); allow (c, S'); row_valid (c,)."""
        sup = in_support & row_valid[None, :, None]
        call["alive"][b, :, lo:hi] |= sup.any(1)
        call["reach"][b, lo:hi] |= (allow & row_valid[:, None]).any(0)
        if "support" in call:
            call["support"][b, :, r0:r1] = in_support.sum(-1, dtype=torch.int32)
            call["valid"][b, r0:r1] = row_valid

    def end_call(self, call: dict) -> None:
        if "support" in call:
            v = call["valid"]  # (B, Lq)
            sup = call["support"].transpose(0, 1)  # (H, B, Lq)
            self._support.append(sup[:, v])  # boolean indexing: monitoring only, outside the training graph

    def record_scale(self, c: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, valid: torch.Tensor) -> None:
        """Snapshot of the scale: c, beta, gamma (B, L, H) at rows valid (B, L)."""
        self._scale.append(tuple(t.detach().float()[valid] for t in (c, beta, gamma)))

    # -- accumulation (the trainer) ------------------------------------------------------------------

    def _flush(self) -> None:
        if self._pending is None:
            return
        alive, reach = self._pending
        self._pending = None
        dead = (reach[:, None, :] & ~alive).sum((0, 2))  # (H,)
        n = reach.sum().expand(self.n_heads)
        self._step_dead = dead if self._step_dead is None else self._step_dead + dead
        self._step_reach = n if self._step_reach is None else self._step_reach + n
        B, S = reach.shape
        alive_kv = None
        if self.n_kv < self.n_heads:  # GQA: alive for some query head of the KV head's group (key head j: j * G ...)
            alive_kv = alive.view(B, self.n_kv, self.n_heads // self.n_kv, S).any(2)  # (B, Hkv, S)
            dk = (reach[:, None, :] & ~alive_kv).sum((0, 2))  # (Hkv,)
            self._step_dead_kv = dk if self._step_dead_kv is None else self._step_dead_kv + dk
        if self._pos_alive is not None and self._pos_alive.shape[1] != S:
            self._pos_ok = False
        if self._pos_ok:
            a, r = alive.any(0), reach.any(0)
            self._pos_alive = a if self._pos_alive is None else self._pos_alive | a
            self._pos_reach = r if self._pos_reach is None else self._pos_reach | r
            if alive_kv is not None:
                ak = alive_kv.any(0)
                self._pos_alive_kv = ak if self._pos_alive_kv is None else self._pos_alive_kv | ak

    def end_microbatch(self) -> None:
        self._flush()

    def end_step(self) -> None:
        self._flush()
        if self._step_dead is None:
            return
        reach = self._step_reach.clamp_min(1).double()
        share = self._step_dead.double() / reach
        self._win_sum = share if self._win_sum is None else self._win_sum + share
        self._win_max = share if self._win_max is None else torch.maximum(self._win_max, share)
        self._win_last = share
        if self._step_dead_kv is not None:
            sk = self._step_dead_kv.double() / reach[: self.n_kv]  # every head's reachable count is the same
            self._kv_sum = sk if self._kv_sum is None else self._kv_sum + sk
            self._kv_max = sk if self._kv_max is None else torch.maximum(self._kv_max, sk)
            self._kv_last = sk
        self._win_steps += 1
        self._step_dead = self._step_reach = self._step_dead_kv = None

    def summary(self) -> dict:
        """The window's statistics (host sync), then a new window. Lists are per head."""
        out: dict = {}
        if self._support:
            sup = torch.cat(self._support, dim=1).double()  # (H, n)
            if sup.shape[1]:
                q = [_quantiles(sup[h], (0.1, 0.5, 0.9, 0.99)) for h in range(self.n_heads)]
                mean = sup.mean(1)
                out["support"] = {"mean": [round(float(v), 4) for v in mean],
                                  "p10": [v[0] for v in q], "p50": [v[1] for v in q], "p90": [v[2] for v in q],
                                  "p99": [v[3] for v in q], "max": [int(v) for v in sup.amax(1)],
                                  "le2": [round(float(v), 6) for v in (sup <= 2).double().mean(1)], "rows": int(sup.shape[1])}
                out["collapsed"] = float((mean <= 2).double().mean())
        if self._win_steps:
            dead = {"step": [round(float(v), 6) for v in self._win_last],
                    "mean": [round(float(v), 6) for v in self._win_sum / self._win_steps],
                    "max": [round(float(v), 6) for v in self._win_max], "steps": self._win_steps, "all_steps": None}
            if self._pos_ok and self._pos_reach is not None:
                r = self._pos_reach
                never = (r[None, :] & ~self._pos_alive).sum(1).double() / r.sum().clamp_min(1).double()
                dead["all_steps"] = [round(float(v), 6) for v in never]
            if self._kv_sum is not None:
                kv = {"step": [round(float(v), 6) for v in self._kv_last],
                      "mean": [round(float(v), 6) for v in self._kv_sum / self._win_steps],
                      "max": [round(float(v), 6) for v in self._kv_max], "all_steps": None}
                if self._pos_ok and self._pos_reach is not None and self._pos_alive_kv is not None:
                    r = self._pos_reach
                    never = (r[None, :] & ~self._pos_alive_kv).sum(1).double() / r.sum().clamp_min(1).double()
                    kv["all_steps"] = [round(float(v), 6) for v in never]
                dead["kv_heads"] = kv
            out["dead_keys"] = dead
        if self._scale:
            c, beta, gamma = (torch.cat([s[i] for s in self._scale], 0) for i in range(3))  # (n, H)
            if c.shape[0]:
                flat = c.reshape(-1)
                if flat.numel() > (1 << 24):  # torch.quantile's limit: a deterministic subsample
                    flat = flat[:: -(-flat.numel() // (1 << 24))]
                p10, p90 = _quantiles(flat, (0.1, 0.9))
                out["scale"] = {"c_mean": float(c.mean()), "c_p10": p10, "c_p90": p90, "beta_mean": float(beta.mean()),
                                "gamma_mean": float(gamma.mean()), "c_mean_head": [round(float(v), 5) for v in c.mean(0)],
                                "beta_mean_head": [round(float(v), 5) for v in beta.mean(0)],
                                "gamma_mean_head": [round(float(v), 5) for v in gamma.mean(0)]}
        self._support, self._scale = [], []
        self._reset_window()
        return out


# ---------------------------------------------------------------------------------------------------
# the attention Function
# ---------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _exact_fp32(device: torch.device):
    """Autocast off, and on CUDA TF32 off (restored on exit): every matmul of the call in IEEE float32."""
    with torch.autocast(device.type, enabled=False):
        if device.type != "cuda":
            yield
            return
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old


def chunk_rows(chunk: int, Hq: int, S: int, Lq: int) -> int:
    """Query rows per chunk: AttentionConfig.asentmax_chunk, or (0) as many as keep one (Hq, rows, S) score
    tensor at AUTO_CHUNK_ELEMENTS."""
    rows = chunk if chunk > 0 else AUTO_CHUNK_ELEMENTS // max(1, Hq * S)
    return max(1, min(rows, Lq))


class _Meta:
    """The KeySet fields the mask functions read (attention.allowed / slots_allowed), for one batch row."""

    def __init__(self, idx, doc, ok):
        self.idx, self.doc, self.ok = idx, doc, ok

    def __len__(self) -> int:
        return self.idx.shape[1]


def _chunk_mask(k_idx, k_doc, k_ok, q_idx, q_doc, empty_reads_first: bool) -> torch.Tensor:
    """(c, S) the mask of one batch row's chunk of queries, by the models' own rule functions."""
    from ced.model.attention import allowed, slots_allowed

    keys = _Meta(k_idx[None], k_doc[None], k_ok[None])
    rule = slots_allowed if empty_reads_first else allowed
    return rule(keys, q_idx[None], q_doc[None])[0]


def _key_range(allow: torch.Tensor) -> tuple[int, int]:
    """[lo, hi): the smallest column range outside which every row of allow (c, S) is masked (one host sync)."""
    S = allow.shape[1]
    cols = allow.any(0)
    j = torch.arange(S, device=allow.device)
    lo_hi = torch.stack([torch.where(cols, j, S).amin(), torch.where(cols, j, -1).amax() + 1])
    lo, hi = lo_hi.tolist()
    return (lo, hi) if hi > lo else (0, 0)


def _scores(qc: torch.Tensor, kr: torch.Tensor, allow: torch.Tensor, scale: float) -> torch.Tensor:
    """qc (c, Hq, hd), kr (S', Hkv, hd), allow (c, S') -> z (Hq, c, S'), -inf where masked."""
    c, Hq, hd = qc.shape
    Hkv = kr.shape[1]
    z = torch.einsum("ckgd,skd->kgcs", qc.reshape(c, Hkv, Hq // Hkv, hd), kr).reshape(Hq, c, kr.shape[0]) * scale
    return z.masked_fill(~allow[None], float("-inf"))


def _rows(p: torch.Tensor, vr: torch.Tensor) -> torch.Tensor:
    """p (Hq, c, S'), vr (S', Hkv, hd) -> (c, Hq, hd)."""
    Hq, c, S = p.shape
    Hkv, hd = vr.shape[1], vr.shape[2]
    return torch.einsum("kgcs,skd->ckgd", p.reshape(Hkv, Hq // Hkv, c, S), vr).reshape(c, Hq, hd)


class _Entmax15Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k_idx, k_doc, k_ok, q_idx, q_doc, empty_reads_first, chunk, stats):
        B, Lq, Hq, hd = q.shape
        S = k.shape[1]
        dt = compute_dtype(q)
        scale = hd**-0.5
        rows = chunk_rows(chunk, Hq, S, Lq)
        record = stats is not None and stats.active
        out = torch.zeros(B, Lq, Hq, hd, dtype=dt, device=q.device)
        tau = torch.full((B, Hq, Lq), math.inf, dtype=dt, device=q.device)
        with torch.no_grad(), _exact_fp32(q.device):
            call = stats.begin_call(B, Lq, S, q.device) if record else None
            for b in range(B):
                kb, vb = k[b].to(dt), v[b].to(dt)
                for r0 in range(0, Lq, rows):
                    r1 = min(Lq, r0 + rows)
                    allow = _chunk_mask(k_idx[b], k_doc[b], k_ok[b], q_idx[b, r0:r1], q_doc[b, r0:r1], empty_reads_first)
                    lo, hi = _key_range(allow)
                    if hi == lo:
                        continue  # no allowed key in the chunk: tau +inf, output 0
                    allow = allow[:, lo:hi]
                    z = _scores(q[b, r0:r1].to(dt), kb[lo:hi], allow, scale)
                    t = entmax15_threshold(z)
                    s = torch.clamp(z * 0.5 - t[..., None], min=0.0)
                    out[b, r0:r1] = _rows(s * s, vb[lo:hi])
                    tau[b, :, r0:r1] = t
                    if call is not None:
                        EntmaxStats.record_chunk(call, b, r0, r1, lo, hi, s > 0, allow, q_doc[b, r0:r1] >= 0)
            if call is not None:
                stats.end_call(call)
        ctx.save_for_backward(q, k, v, tau, k_idx, k_doc, k_ok, q_idx, q_doc)
        ctx.empty_reads_first, ctx.rows = empty_reads_first, rows
        return out.to(q.dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout):
        q, k, v, tau, k_idx, k_doc, k_ok, q_idx, q_doc = ctx.saved_tensors
        B, Lq, Hq, hd = q.shape
        Hkv = k.shape[2]
        G = Hq // Hkv
        dt = compute_dtype(q)
        scale = hd**-0.5
        rows = ctx.rows
        dq = torch.zeros(B, Lq, Hq, hd, dtype=dt, device=q.device)
        dk = torch.zeros(k.shape, dtype=dt, device=q.device)
        dv = torch.zeros(v.shape, dtype=dt, device=q.device)
        tiny = torch.finfo(dt).tiny
        with _exact_fp32(q.device):
            for b in range(B):
                kb, vb = k[b].to(dt), v[b].to(dt)
                for r0 in range(0, Lq, rows):
                    r1 = min(Lq, r0 + rows)
                    allow = _chunk_mask(k_idx[b], k_doc[b], k_ok[b], q_idx[b, r0:r1], q_doc[b, r0:r1], ctx.empty_reads_first)
                    lo, hi = _key_range(allow)
                    if hi == lo:
                        continue
                    c = r1 - r0
                    qc, kr, vr = q[b, r0:r1].to(dt), kb[lo:hi], vb[lo:hi]
                    z = _scores(qc, kr, allow[:, lo:hi], scale)
                    s = torch.clamp(z * 0.5 - tau[b, :, r0:r1, None], min=0.0)  # sqrt(p), recomputed from tau
                    do = dout[b, r0:r1].to(dt).reshape(c, Hkv, G, hd)
                    dp = torch.einsum("ckgd,skd->kgcs", do, vr).reshape(Hq, c, hi - lo)
                    sdp = s * dp
                    dz = sdp - s * (sdp.sum(-1, keepdim=True) / s.sum(-1, keepdim=True).clamp_min(tiny))
                    dz = (dz * scale).reshape(Hkv, G, c, hi - lo)
                    dq[b, r0:r1] = torch.einsum("kgcs,skd->ckgd", dz, kr).reshape(c, Hq, hd)
                    dk[b, lo:hi] += torch.einsum("kgcs,ckgd->skd", dz, qc.reshape(c, Hkv, G, hd))
                    dv[b, lo:hi] += torch.einsum("kgcs,ckgd->skd", (s * s).reshape(Hkv, G, c, hi - lo), do)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None, None, None, None


@torch.compiler.disable
def entmax_attention(q: torch.Tensor, keys, q_idx: torch.Tensor, q_doc: torch.Tensor, empty_reads_first: bool = False,
                     chunk: int = 0, stats: EntmaxStats | None = None, scale_info: tuple | None = None) -> torch.Tensor:
    """alpha = 1.5 entmax attention of q (B, Lq, Hq, hd) over keys (an attention.KeySet), with the mask of
    attention.allowed (or, empty_reads_first, attention.slots_allowed: an empty slot, q_doc < 0, reads key 0).
    chunk: query rows per chunk (0: auto, chunk_rows). stats: an EntmaxStats, recording when active; scale_info:
    (c, beta, gamma) (B, Lq, Hq) from GatedAttention, kept on a stats snapshot. Not traced by torch.compile."""
    B, Lq, Hq, _ = q.shape
    assert Hq % keys.k.shape[2] == 0 and q_idx.shape == q_doc.shape == (B, Lq)
    assert getattr(keys, "k_far", None) is None, \
        "this key set carries the unrotated keys of the rotary distance cap (KeySet.k_far): the entmax attention does " \
        "not implement AttentionConfig.rope_max_distance"
    if stats is not None and stats.active and stats.snapshot and scale_info is not None:
        stats.record_scale(*scale_info, q_doc >= 0)
    return _Entmax15Attention.apply(q, keys.k, keys.v, keys.idx, keys.doc, keys.ok, q_idx, q_doc,
                                    bool(empty_reads_first), int(chunk), stats)
