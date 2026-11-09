"""R3: depth mixing at the loop boundary (LoopConfig.recur_layers with injection "depth_mix" on arch "control").


The form. The stack is cut as every looped middle is (ced/model/control.py, "The loop in the published form"): the
prelude (layers 0 .. first - 1), the middle (layers first .. end - 1, the same weights at every pass), the coda (layers
end .. N - 1). At T >= 1 passes:

    x0  = the embedding's output (after embed_norm)       always a candidate, at every pass
    p   = Prelude(x0)                                     once
    V_1 = [x0, p]
    V_k = [x0, p, z_1, ..., z_{k-1}]                      k >= 2: the earlier passes' outputs are candidates
    alpha_v = softmax over v in V_k of  q^T RMSNorm(v)    per position
    u_k = sum over V_k of alpha_v v                       what pass k reads
    z_k = Middle(u_k)                                     k = 1 .. T
    out = Coda(z_T)                                       the coda reads the last pass's output

The candidates are STREAMS: x0, p and the z_k as the layers hand them on. (R1's sources are block sums without the
stream below them; that form at the loop boundary needs Block AttnRes inside the trunk.) The layers keep their plain
residual add, so z_k = u_k + what the middle's layers add.

q: ONE pseudo-query of d values for every pass, ZEROS at the start, and ONE RMSNorm on the keys (gain 1 + g, g zeros
at the start); the values are the candidates themselves. It is R1's reader, attn_res.DepthQuery, held by the encoder
as `recur_depth`: two tensors of d values each, encoder.recur_depth.query and encoder.recur_depth.key_norm.weight (2 x
d = 1,024 new parameters at d = 512). With q at zero every logit is zero and every candidate has equal weight: pass 1
reads (x0 + p) / 2 and pass 2 reads (x0 + p + z_1) / 3. So at its start this model is NOT the plain control, also at
T = 1. One query for all passes: a forward at any pass count runs trained parameters only.

The mix is per position: nothing in it mixes positions. So causality and the cache of a chunk are the layers' own, as
in every looped middle: each pass has its own GDN states and K/V, a cached chunk carries nothing of the mix, and the
candidates of a chunk are that chunk's own x0, p and z_k.

Precision. The logits, the key norm, the softmax and the weighted sum run in at least float32 with autocast off,
whatever the autocast dtype around them (attn_res.depth_mix's own rule).

Two implementations of the same rule:
- reference_recur below: eager, line for line from the form above (reference_mix: the key norm of every candidate, the
  dot product with q, the softmax over the candidates, the weighted sum). It calls every layer's own `forward`, so it
  runs eagerly also on a model whose layers are compiled. It can record every pass's weights. The ground truth.
- the fast path, which every forward runs (training under compile "all", evaluation, scoring, prefill / extend):
  Encoder.forward_recur calls attn_res.depth_mix between the layer calls, where the scalar input mix of injection
  "mix" is called (the logit of a candidate without a normalised copy of it, no stacked copy of the candidates). The
  layers are the plain layer calls, compiled when the layers are. Accepted against the reference in
  tests/test_loop_r3.py.
Encoder.recur_depth_impl says which one a forward runs ("fast" by default; a plain attribute, never in the state dict).

Monitoring (mix_weights below): with the model's collect_stats on, a forward also keeps every pass's weights,
"depth_mix/pass<k>" (B, L, k + 1), formed from the same candidates a second time under no_grad. Nothing of the
forward's own arithmetic changes. (On the reference path the names hold the weights the reference itself used.)
"""

from __future__ import annotations

import torch

from ced.config import ModelConfig
from ced.model.attn_res import DepthQuery, _no_autocast
from ced.model.norms import compute_dtype

INJECTION = "depth_mix"
# the two tensors the injection adds to the control, in state-dict order
PARAMETERS = ("encoder.recur_depth.query", "encoder.recur_depth.key_norm.weight")
STATS_PREFIX = "depth_mix/"  # the monitoring names: "depth_mix/pass<k>", k = 1 .. T


def is_depth_mix(cfg: ModelConfig) -> bool:
    """Whether the config is the control with a looped middle whose every pass reads the depth mix (R3)."""
    return cfg.arch == "control" and bool(cfg.loop.recur_layers) and cfg.loop.injection == INJECTION


def parameter_names(cfg: ModelConfig) -> list[str]:
    """The names of the tensors the injection adds to arch "control", in state-dict order; [] for any other model."""
    return list(PARAMETERS) if is_depth_mix(cfg) else []


def source_labels(k: int) -> list[str]:
    """The candidates of pass k (1-based), in the order of the weights: "x0", "p", "z1", ..., "z<k-1>"."""
    assert k >= 1
    return ["x0", "p"] + [f"z{j}" for j in range(1, k)]


def reference_mix(sources: list, q: DepthQuery):
    """The form's two lines for one pass, as written: alpha_v = softmax over v of q^T RMSNorm(v), u = sum alpha_v v.
    sources: the candidates V_k, (B, L, d) each. In at least float32 with autocast off. Returns (u (B, L, d) in the
    candidates' dtype, alpha (B, L, n): the weight of each candidate at each position)."""
    v0 = sources[0]
    dtype = compute_dtype(v0)
    with _no_autocast(v0):
        values = [v.to(dtype) for v in sources]
        keys = [q.key_norm(v) for v in values]  # RMSNorm(v): the keys; the values are the candidates themselves
        logits = torch.stack([(k * q.query.to(dtype)).sum(-1) for k in keys], dim=-1)  # q^T RMSNorm(v): (B, L, n)
        alpha = torch.softmax(logits, dim=-1)  # over the candidates, per position
        u = alpha[..., 0:1] * values[0]
        for i in range(1, len(values)):
            u = u + alpha[..., i : i + 1] * values[i]
    return u.to(v0.dtype), alpha


def reference_recur(enc, h: torch.Tensor, ctx, caches: list | None, T: int, record: list | None = None):
    """The reference: Encoder.forward_recur under injection "depth_mix", written from the form in the module docstring,
    eager, one layer after the other. Returns (the stream after the coda, the layers' new cache entries in running
    order: one per layer application). Every layer is run through its own `forward` (not its compiled call).
    record (a list): receives one dict per pass, in order, {"pass": k, "sources": the candidates' labels, "alpha":
    (B, L, k + 1), "input": u_k, "output": z_k}."""
    assert enc.recur is not None and enc.recur_depth is not None, "reference_recur is for a looped middle with injection 'depth_mix'"
    assert type(T) is int and T >= 1, f"T={T!r}: a pass count >= 1"
    first, end = enc.recur
    layers = list(enc.layers)
    n = enc.recur_cache_length(T)
    assert caches is None or len(caches) == n, \
        f"this cache holds {len(caches)} layer entries; a forward at T = {T} needs {n} (a cached sequence keeps its T)"
    new_caches: list = []

    def run(x, segment):
        for layer in segment:
            x, new_cache = layer.forward(x, ctx, None if caches is None else caches[len(new_caches)])
            new_caches.append(new_cache)
        return x

    x0 = h  # the embedding's output: always a candidate
    p = run(x0, layers[:first])  # the prelude, once
    V = [x0, p]  # V_1
    q = enc.recur_depth
    z = None
    for k in range(1, T + 1):
        u, alpha = reference_mix(V, q)  # u_k = sum over V_k of alpha_v v
        z = run(u, layers[first:end])  # z_k = Middle(u_k)
        if record is not None:
            record.append({"pass": k, "sources": source_labels(k), "alpha": alpha, "input": u, "output": z})
        V = V + [z]  # V_{k+1} = [x0, p, z_1, ..., z_k]
    return run(z, layers[end:]), new_caches  # the coda reads z_T


@torch.no_grad()
def mix_weights(sources: list, q: DepthQuery) -> torch.Tensor:
    """Monitoring: the weights attn_res.depth_mix gives these candidates, alpha (B, L, n) in at least float32, formed
    a second time by depth_mix's own lines (the logit of a candidate without a normalised copy of it) on the detached
    candidates, under no_grad and with autocast off. It writes to no tensor of the model and draws no random number:
    the forward it is called from computes what it computes without it."""
    v0 = sources[0]
    dtype = compute_dtype(v0)
    with _no_autocast(v0):
        u = q.query.detach().to(dtype) * (1.0 + q.key_norm.weight.detach().to(dtype))
        eps = q.key_norm.eps
        logits = []
        for v in sources:
            vf = v.detach().to(dtype)
            logits.append((vf @ u) * torch.rsqrt(vf.pow(2).mean(-1) + eps))
        return torch.softmax(torch.stack(logits, dim=-1), dim=-1)


@torch.no_grad()
def summary(q: DepthQuery) -> dict:
    """The reader as it is now, for the logs: {"query_norm": ||q||, "key_gain_dev": max |g|} (Python floats). The
    query's norm is how far it has moved from zero, where every candidate has equal weight; the weights themselves
    need a forward (mix_weights)."""
    return {"query_norm": float(torch.linalg.vector_norm(q.query.double())), "key_gain_dev": float(q.key_norm.weight.double().abs().max())}


def stats_weights(stats: dict, where: torch.Tensor | None = None) -> dict:
    """The mean weight of each candidate at each pass from a forward's monitoring stats (the model's `stats` after a
    forward with collect_stats on): {"pass<k>": [the mean weight of x0, p, z1, ..., z<k-1>]} (Python floats, the passes
    in order; {} for a model without the depth mix). where (B, L) bool, optional: the positions that count (default:
    every position of the forward)."""
    out = {}
    for name in sorted((k for k in stats if k.startswith(STATS_PREFIX)), key=lambda s: int(s[len(STATS_PREFIX) + 4 :])):
        alpha = stats[name].double()  # (B, L, n)
        if where is None:
            mean = alpha.mean(dim=(0, 1))
        else:
            w = where.to(alpha.device, alpha.dtype)
            mean = (alpha * w[..., None]).sum(dim=(0, 1)) / w.sum().clamp_min(1.0)
        out[name[len(STATS_PREFIX) :]] = mean.cpu().tolist()
    return out
