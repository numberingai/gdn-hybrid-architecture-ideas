"""GDN state diagnostics by position: does the recurrent state of the Gated DeltaNet layers drift at lengths the
model never trained at?

For every GDN layer of the encoder, along ONE row (one document from position 0), the layer's recurrence

    S_t = alpha_t S_{t-1} + k_t (x) beta_t (v_t - (alpha_t S_{t-1})^T k_t),   alpha_t = exp(g_t),   o_t = S_t^T q_t

is walked again from the layer's own input with the model's own code (GatedDeltaNet.in_proj_qkv / short_conv / gates,
gdn.l2_normalize, gdn_chunked.chunked_delta_rule, gdn.delta_rule_recurrence: read-only, nothing in ced/model is changed), in float32 (float64
when the model is), and at sample positions p of every position bucket the state after p and after p - 1 give, per
value head:

    norm        ||S_p||_F
    erank       exp(H(sigma / sum sigma)): the effective rank of S_p (d_k x d_v) from its singular values' entropy
    srank       ||S_p||_F^2 / sigma_max^2 (stable rank);  sigma_max: the largest singular value
    delta       ||S_p - S_{p-1}||_F: the change of the state at one token;  rel_delta = delta / ||S_{p-1}||_F
    decay       (1 - alpha_p) ||S_{p-1}||_F: the part of the change that is forgetting
    write       ||beta_p (v_p - (alpha_p S_{p-1})^T k_p)||: the part that is the new write (||k|| = 1)
    read        ||o_p|| = ||S_p^T q_p||: what the layer reads out (before its output norm and gate)

and, over EVERY position of a bucket (not only the samples), the gate statistics per head: the mean log decay g, the
mean decay alpha = exp(g), the mean write gate beta, and the half-life ln 2 / mean(-g) in tokens (a head whose
half-life exceeds the training row length still holds, beyond 8k, state written further back than any row it trained
on).

Buckets (default 0 / 2k / 8k / 16k / 32k / 64k / 128k / 256k). The reference is the last bucket inside the training
length ("2k-8k"); `GDNStats.result` gives every statistic per layer and bucket and its ratio to the reference bucket
(the drift), with the layers whose ratio leaves [1 / threshold, threshold].

Two ways in:
- `encoder_gdn_diag(model, tokens, doc_ids, n_valid, stats, ...)`: runs the encoder layer by layer (compiled layers or
  not, the fused kernels or not), and before each GDN layer walks its recurrence from mixer_in_norm(h). The decoder is
  not run (CED v2's decoder has no GDN layer), so a 128k or 256k row costs the encoder only. Control (a) and CED v2.
- `GDNDiagHooks(model, stats, ...)`: forward pre-hooks on every GatedDeltaNet, for an eager (not compiled) model: any
  forward of the model over one row adds that row's statistics. (A compiled EncoderLayer does not run hooks added
  after its compilation, and would trace the hook's Python into its graph otherwise: use encoder_gdn_diag there.)
Both give the same numbers.

The walk differs from the model's own kernel only in arithmetic: the fused kernels get q / k / v / beta in bf16 under
autocast (GDNConfig.input_dtype) and keep the state in float32; here the inputs are float32 too.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ced.eval.long_context import BASE_LEN, bucket_index, bucket_names
from ced.eval.probes import model_device, unwrap
from ced.model.encoder import refuse_recur_loop
from ced.model.gdn import GatedDeltaNet, GDNState, delta_rule_recurrence, l2_normalize
from ced.model.gdn_chunked import chunked_delta_rule
from ced.model.gdn_rules import refuse as refuse_gdn_rules
from ced.model.sequence import SeqContext, sequence_context

STATS = ("norm", "erank", "srank", "sigma_max", "delta", "rel_delta", "decay", "write", "read")
DRIFT_STATS = ("norm", "erank", "rel_delta", "read", "write")
GATES = ("g", "alpha", "beta")
DRIFT_THRESHOLD = 1.5


def default_gdn_edges(base: int = BASE_LEN) -> tuple[int, ...]:
    """0, base / 4, base, 2 base, ..., 32 base: 0 / 2k / 8k / 16k / 32k / 64k / 128k / 256k at base 8,192."""
    return (0, base // 4, base, 2 * base, 4 * base, 8 * base, 16 * base, 32 * base)


def sample_positions(n_valid: int, edges, per_bucket: int = 16) -> torch.Tensor:
    """Sorted sample positions p, 1 <= p < n_valid: per bucket [lo, hi) the midpoints of per_bucket equal parts of its
    part below n_valid (position 0 has no state before it). Fewer where a bucket holds fewer positions."""
    out = set()
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo, hi = max(int(lo), 1), min(int(hi), n_valid)
        if hi <= lo:
            continue
        for j in range(per_bucket):
            out.add(lo + int((j + 0.5) * (hi - lo) / per_bucket))
    return torch.tensor(sorted(out), dtype=torch.long)


def state_measures(S: torch.Tensor) -> dict[str, torch.Tensor]:
    """norm, erank, srank and sigma_max of states S (H, d_k, d_v), each (H,) float64. A zero state has erank and srank
    NaN (no spectrum)."""
    sigma = torch.linalg.svdvals(S.double())  # (H, min(d_k, d_v)), descending
    total = sigma.sum(-1, keepdim=True)
    p = sigma / total.clamp_min(1e-300)
    entropy = -(torch.where(p > 0, p * torch.log(p.clamp_min(1e-300)), torch.zeros_like(p))).sum(-1)
    nan = torch.full_like(entropy, float("nan"))
    some = total.squeeze(-1) > 0
    fro2 = sigma.pow(2).sum(-1)
    return {"norm": fro2.sqrt(), "erank": torch.where(some, entropy.exp(), nan),
            "srank": torch.where(some, fro2 / sigma[:, 0].pow(2).clamp_min(1e-300), nan), "sigma_max": sigma[:, 0]}


class StateWalk:
    """One GDN layer's recurrent state along ONE row, fed block by block (any block sizes: the state is carried).

    positions: the sample positions (sample_positions); n_valid: the row's real positions (the gate statistics cover
    positions < n_valid; a sample must be below it); edges: the position buckets of the gate statistics.
    feed(q, k, v, g, beta, is_doc_start): the next block, shaped as delta_rule_recurrence takes them: q, k (1, n, H,
    d_k) (already repeated to the value heads, q scaled), v (1, n, H, d_v), g, beta (1, n, H), is_doc_start (1, n).
    result(): {"positions": (P,), <stat>: (P, H) float64 for STATS, "gate_count": (buckets,), "gate_<g|alpha|beta>":
    (buckets, H) sums over the bucket's positions, "final_state": (1, H, d_k, d_v)}."""

    def __init__(self, positions: torch.Tensor, n_valid: int, edges, chunk_size: int = 64):
        self.positions = [int(p) for p in positions]
        assert self.positions == sorted(set(self.positions)), "sample positions must be sorted and distinct"
        assert all(1 <= p < n_valid for p in self.positions), "a sample position needs 1 <= p < n_valid"
        self.n_valid, self.edges, self.chunk_size = n_valid, tuple(edges), chunk_size
        self.sample = set(self.positions)
        self.cuts = sorted(self.sample | {p - 1 for p in self.positions})  # the state is needed after each of these
        self.at = 0
        self.S = None
        self.before: dict[int, torch.Tensor] = {}
        self.rows: dict[str, list] = {name: [] for name in STATS}
        self.gate: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def feed(self, q, k, v, g, beta, is_doc_start) -> None:
        assert q.shape[0] == 1, "one row at a time"
        n, H = q.shape[1], q.shape[2]
        a, b = self.at, self.at + n
        work = torch.promote_types(q.dtype, torch.float32)
        q, k, v, g, beta = (t.to(work) for t in (q, k, v, g, beta))
        if self.S is None:
            self.S = q.new_zeros(1, H, k.shape[-1], v.shape[-1])
            nb = len(self.edges) - 1
            self.gate = {"count": torch.zeros(nb, dtype=torch.float64),
                         **{name: torch.zeros(nb, H, dtype=torch.float64) for name in GATES}}
        # gate statistics of every real position of the block
        pos = torch.arange(a, b)
        bucket = bucket_index(pos, self.edges)
        keep = (bucket >= 0) & (pos < self.n_valid)
        if bool(keep.any()):
            idx = bucket[keep]
            gd = g[0].double().cpu()[keep]
            self.gate["count"].index_add_(0, idx, torch.ones(len(idx), dtype=torch.float64))
            self.gate["g"].index_add_(0, idx, gd)
            self.gate["alpha"].index_add_(0, idx, gd.exp())
            self.gate["beta"].index_add_(0, idx, beta[0].double().cpu()[keep])

        def advance(lo: int, hi: int) -> None:
            """The state after position hi - 1 from the state after lo - 1: the chunked rule, or for one token (the
            step from p - 1 to a sample p) one step of the readable recurrence."""
            if hi <= lo:
                return
            s = slice(lo - a, hi - a)
            rule = delta_rule_recurrence if hi - lo == 1 else (lambda *x: chunked_delta_rule(*x, self.chunk_size))
            _, self.S = rule(q[:, s], k[:, s], v[:, s], g[:, s], beta[:, s], is_doc_start[:, s], self.S)

        prev = a
        for c in self.cuts:
            if not a <= c < b:
                continue
            advance(prev, c + 1)
            prev = c + 1
            if c in self.sample:
                self._record(c, self.before.pop(c), self.S, q[0, c - a], k[0, c - a], v[0, c - a], g[0, c - a], beta[0, c - a],
                             bool(is_doc_start[0, c - a]))
            if c + 1 in self.sample:
                self.before[c + 1] = self.S.clone()
        advance(prev, b)
        self.at = b

    def _record(self, p, S_prev, S, q, k, v, g, beta, is_start) -> None:
        S_prev, S = S_prev[0].double(), S[0].double()  # (H, d_k, d_v)
        if is_start:  # the state restarts at a document start: nothing carried into p
            S_prev = torch.zeros_like(S_prev)
        q, k, v, g, beta = (t.double() for t in (q, k, v, g, beta))
        m = state_measures(S)
        prev_norm = S_prev.flatten(1).norm(dim=-1)
        delta = (S - S_prev).flatten(1).norm(dim=-1)
        alpha = torch.exp(g)
        predicted = torch.einsum("hkv,hk->hv", S_prev * alpha[:, None, None], k)
        nan = torch.full_like(delta, float("nan"))
        m.update(delta=delta, rel_delta=torch.where(prev_norm > 0, delta / prev_norm.clamp_min(1e-300), nan),
                 decay=(1.0 - alpha) * prev_norm, write=(beta[:, None] * (v - predicted)).norm(dim=-1),
                 read=torch.einsum("hkv,hk->hv", S, q).norm(dim=-1))
        for name in STATS:
            self.rows[name].append(m[name].cpu())

    def result(self) -> dict:
        assert not self.before and self.at >= (self.positions[-1] + 1 if self.positions else 0), "the row was not fed to its last sample"
        H = self.S.shape[1] if self.S is not None else 0
        out = {"positions": torch.tensor(self.positions, dtype=torch.long), "final_state": self.S}
        for name in STATS:
            out[name] = torch.stack(self.rows[name]) if self.rows[name] else torch.zeros(0, H, dtype=torch.float64)
        for name, t in (self.gate or {}).items():
            out[f"gate_{name}"] = t
        return out


def gdn_inputs(mixer: GatedDeltaNet, x: torch.Tensor, ctx: SeqContext, block: int = 8192, x_write: torch.Tensor | None = None,
               arms=None, x_read: torch.Tensor | None = None):
    """Yield the inputs of one GDN layer's recurrence over one row, `block` positions at a time: (q, k, v, g, beta,
    is_doc_start) shaped as delta_rule_recurrence takes them (q, k (1, n, H_v, d_k) already repeated to the value heads,
    q scaled by d_k^-0.5; v (1, n, H_v, d_v); g, beta (1, n, H_v); is_doc_start (1, n)). Built as GatedDeltaNet.forward
    builds them on its reference path: in_proj_qkv, short_conv (history carried across blocks), SiLU, split, L2 norm,
    gates. x: (1, L, d), the layer's normed input (what EncoderLayer.mixer_in_norm gives the mixer).
    x_write, arms (the GDN-side write-mix arm, ced/model/gdn_arms.py; None in every other model): the layer's normed write mix
    m' (1, L, d) and its GDNArms; k, v, beta and the decay then read x_w, as the layer's forward builds them
    (gdn_arms.write_inputs, GatedDeltaNet.gates_from). x_read (the wsl variant "qmix", EncoderConfig.gdn_write_variant):
    the layer's normed read mix m'_q (1, L, d); its q rows then read x_q, as in the layer's forward."""
    assert x.shape[0] == 1, "one row at a time"
    # the walk is one decay a value head and one write a token (gates(), in_proj_qkv): negative eigenvalues read right; KDA, the erase
    # and Gated DeltaProduct refused by name (ced/model/gdn_rules.py; a DeltaProduct layer was walked as plain GDN before, silently)
    refuse_gdn_rules(mixer.cfg, "the GDN state walk (ced/eval/gdn_diag.py gdn_inputs: StateWalk, the lce- suite's \"gdn\" part, "
                                "the wipe look)", reads=("negeig",))
    assert (x_write is None) == (arms is None or arms.write_mix == "none"), "a layer with the write-mix arm needs its write input, and only it"
    from ced.model.gdn_arms import READ_MIX_VARIANTS

    assert (x_read is None) == (arms is None or arms.variant not in READ_MIX_VARIANTS), "a layer of the qmix or kqmix variant needs its read mix, and only it"
    cfg = mixer.cfg
    hk, hv, dk, dv = cfg.n_k_heads, cfg.n_v_heads, cfg.k_head_dim, cfg.v_head_dim
    L = x.shape[1]
    dtype = mixer.in_proj_qkv.weight.dtype
    conv = None
    with torch.autocast(x.device.type, enabled=False):
        for a in range(0, L, block):
            b = min(a + block, L)
            xb = x[:, a:b].to(dtype)
            part = SeqContext(idx=ctx.idx[:, a:b], doc=ctx.doc[:, a:b], pos=ctx.pos[:, a:b], is_doc_start=ctx.is_doc_start[:, a:b])
            if x_write is None:
                u_pre = mixer.in_proj_qkv(xb)
            else:  # write-mix arm: the k / v rows and the gates read x_w
                from ced.model.gdn_arms import write_inputs

                if x_read is None:
                    u_pre, a_lin, b_lin = write_inputs(mixer, arms, xb, x_write[:, a:b].to(dtype))
                else:  # the qmix variant: the q rows read x_q
                    u_pre, a_lin, b_lin = write_inputs(mixer, arms, xb, x_write[:, a:b].to(dtype), x_read[:, a:b].to(dtype))
            u, hist, hist_doc = mixer.short_conv(u_pre, part, conv)
            conv = GDNState(conv_input=hist, conv_doc=hist_doc, S=None)
            q, k, v = torch.split(F.silu(u), [hk * dk, hk * dk, hv * dv], dim=-1)
            n = b - a
            q = (l2_normalize(q.view(1, n, hk, dk)) * dk**-0.5).repeat_interleave(hv // hk, dim=2)
            k = l2_normalize(k.view(1, n, hk, dk)).repeat_interleave(hv // hk, dim=2)
            scale = getattr(ctx, "gdn_scale", 1)
            if scale == 1:
                g, beta = mixer.gates(xb) if x_write is None else mixer.gates_from(a_lin, b_lin)
            else:  # a TST bag read with E (SeqContext.gdn_scale): the gates of a position of gdn_scale tokens, as the layer's forward forms them
                g, beta = (mixer.gates(xb, scale, ctx.gdn_beta_power) if x_write is None
                           else mixer.gates_from(a_lin, b_lin, scale, ctx.gdn_beta_power))
            yield q, k, v.view(1, n, hv, dv), g, beta, part.is_doc_start


@torch.no_grad()
def gdn_layer_walk(mixer: GatedDeltaNet, x: torch.Tensor, ctx: SeqContext, positions: torch.Tensor, n_valid: int, edges,
                   block: int = 8192, x_write: torch.Tensor | None = None, arms=None, x_read: torch.Tensor | None = None) -> dict:
    """The StateWalk result of one GDN layer over one row from the layer's normed input x (1, L, d) (what
    EncoderLayer.mixer_in_norm gives the mixer) and the row's SeqContext, `block` positions at a time (the short conv's
    history and the state are carried, so the block size does not change the result). The inputs of the recurrence are
    built by `gdn_inputs` (x_write, arms: the GDN-side write-mix arm's, as there). The arm CCR adds a read to the layer's output
    and changes nothing of its state: its layers are walked as any other. x_read: the qmix variant's read mix, as there."""
    walk = StateWalk(positions, n_valid, edges, mixer.cfg.chunk_size)
    for inputs in gdn_inputs(mixer, x, ctx, block, x_write, arms, x_read):
        walk.feed(*inputs)
    return walk.result()


# --------------------------------------------------------------------------------------------
# aggregation over rows
# --------------------------------------------------------------------------------------------


class GDNStats:
    """Per GDN layer and position bucket: the statistics summed over sample positions and rows.

    add(layer, result): one row's StateWalk result for the layer at encoder index `layer`.
    result(): {"edges", "buckets", "train_len", "ref_bucket", "rows", "threshold", "layers": [{"layer", "heads",
    "samples": per bucket, "<stat>": {"mean": per bucket (head mean), "heads": per bucket per head, "ratio": per bucket
    mean / the ref bucket's mean}, "gate": {"positions": per bucket, "g" / "alpha" / "beta": per bucket per head,
    "half_life": per bucket per head (tokens; None where g = 0), "long_heads": heads whose half-life in the ref bucket
    exceeds train_len}}], "drift": {stat: {"median_ratio" / "min_ratio" / "max_ratio": per bucket over the layers,
    "flagged": per bucket the layers whose ratio is outside [1 / threshold, threshold]}}}. Values without samples are
    None."""

    def __init__(self, edges, train_len: int = BASE_LEN, threshold: float = DRIFT_THRESHOLD):
        self.edges = tuple(int(e) for e in edges)
        assert train_len in self.edges, f"the training length {train_len} must be a bucket edge of {self.edges}"
        self.train_len, self.threshold = train_len, threshold
        self.rows = 0
        self.layers: dict[int, dict] = {}

    def add(self, layer: int, result: dict) -> None:
        nb = len(self.edges) - 1
        H = result["norm"].shape[1]
        acc = self.layers.setdefault(layer, {
            "heads": H, "sum": {s: torch.zeros(nb, H, dtype=torch.float64) for s in STATS},
            "n": {s: torch.zeros(nb, H, dtype=torch.float64) for s in STATS},
            "gate": {name: torch.zeros(nb, H, dtype=torch.float64) for name in GATES}, "gate_count": torch.zeros(nb, dtype=torch.float64)})
        bucket = bucket_index(result["positions"], self.edges)
        keep = bucket >= 0
        for s in STATS:
            vals = result[s][keep]
            ok = torch.isfinite(vals)
            acc["sum"][s].index_add_(0, bucket[keep], torch.where(ok, vals, torch.zeros_like(vals)))
            acc["n"][s].index_add_(0, bucket[keep], ok.double())
        if "gate_count" in result:
            acc["gate_count"] += result["gate_count"]
            for name in GATES:
                acc["gate"][name] += result[f"gate_{name}"]

    def end_row(self) -> None:
        self.rows += 1

    def result(self) -> dict:
        names = bucket_names(self.edges)
        ref = self.edges.index(self.train_len) - 1  # the last bucket inside the training length
        layers = []
        for layer in sorted(self.layers):
            acc = self.layers[layer]
            rec = {"layer": layer, "heads": acc["heads"], "samples": [int(x) for x in acc["n"]["norm"][:, 0]]}
            for s in STATS:
                heads = acc["sum"][s] / acc["n"][s]  # NaN where a bucket has no sample
                mean = heads.nanmean(dim=1) if heads.numel() else heads.sum(1)
                rec[s] = {"mean": _clean(mean), "heads": [_clean(row) for row in heads],
                          "ratio": _clean(mean / mean[ref])}
            count = acc["gate_count"]
            gate = {name: acc["gate"][name] / count[:, None] for name in GATES}
            half = math.log(2.0) / gate["g"].abs()  # g <= 0; g = 0 (no decay): an infinite half-life
            rec["gate"] = {"positions": [int(x) for x in count], **{name: [_clean(row) for row in gate[name]] for name in GATES},
                           "half_life": [_clean(row) for row in half],
                           "long_heads": int((half[ref] > self.train_len).sum()) if count[ref] > 0 else None}
            layers.append(rec)
        drift = {}
        for s in DRIFT_STATS:
            ratios = torch.tensor([[float("nan") if v is None else v for v in rec[s]["ratio"]] for rec in layers], dtype=torch.float64)
            per = {"median_ratio": [], "min_ratio": [], "max_ratio": [], "flagged": []}
            for b in range(len(names)):
                col = ratios[:, b] if len(layers) else torch.zeros(0)
                ok = torch.isfinite(col)
                vals = col[ok]
                per["median_ratio"].append(float(vals.median()) if len(vals) else None)
                per["min_ratio"].append(float(vals.min()) if len(vals) else None)
                per["max_ratio"].append(float(vals.max()) if len(vals) else None)
                per["flagged"].append([layers[i]["layer"] for i in range(len(layers))
                                       if bool(ok[i]) and not (1.0 / self.threshold <= float(col[i]) <= self.threshold)])
            drift[s] = per
        return {"edges": list(self.edges), "buckets": names, "train_len": self.train_len, "ref_bucket": names[ref],
                "rows": self.rows, "threshold": self.threshold, "stats": list(STATS), "layers": layers, "drift": drift}


def _clean(t: torch.Tensor) -> list:
    """A tensor row as a JSON list: non-finite values as None."""
    return [float(x) if math.isfinite(float(x)) else None for x in t.reshape(-1)]


# --------------------------------------------------------------------------------------------
# the two ways in
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def encoder_gdn_diag(model: torch.nn.Module, tokens: torch.Tensor, doc_ids: torch.Tensor, n_valid: int, stats: GDNStats,
                     per_bucket: int = 16, autocast_dtype: torch.dtype | None = None, fused_varlen: bool = False,
                     block: int = 8192, layer_hook=None) -> None:
    """Add one row's GDN statistics to `stats`: run the encoder (only) over tokens / doc_ids (1, L) layer by layer, as
    Encoder._stack does, and walk every GDN layer's recurrence from its normed input first. n_valid: the row's real
    positions (document 0 from position 0; the rest is the padding document). Compiled layers are called as in
    training; the walk itself is eager. Control (a) and CED v2 (their encoder, one pass).

    layer_hook (default None): a function (layer index, mixer, normed input x (1, L, d), SeqContext) called for every GDN
    layer in place of the sample walk (stats then only needs `.edges` and `.end_row()`); a caller can read
    the state at every position this way."""
    inner = unwrap(model)
    if inner.cfg.arch not in ("control", "ced_v2") or inner.encoder.loops != 1:
        raise ValueError(f"GDN diagnostics run the encoder of control (a) / CED v2 (one pass), not arch {inner.cfg.arch!r}")
    refuse_recur_loop(inner.cfg, "the GDN diagnostics")
    assert tokens.shape[0] == 1, "one row at a time"
    device = model_device(model)
    ctx, _ = sequence_context(doc_ids.to(device), None)
    if fused_varlen:
        from ced.train.varlen import to_device, varlen_for_batch

        varlen = to_device(varlen_for_batch(doc_ids, None, None), device)
        ctx.cu_seqlens, ctx.cu_seqlens_cpu = varlen["enc"], varlen.get("enc_cpu")
    positions = sample_positions(n_valid, stats.edges, per_bucket)
    auto = lambda: torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None)  # noqa: E731
    with auto():
        h = inner.embed_norm(inner.embed(tokens.to(device)))
    # The control's per-layer residual scalars (EncoderConfig.layer_scalars): handed to every layer call, as
    # Encoder._stack hands them, so every layer here reads the stream it reads in the model's own forward.
    enc = inner.encoder
    scalars = getattr(enc, "layer_scalars", False)
    x0 = h if scalars else None
    gain, b = enc.layer_scalar_terms() if scalars else (None, None)  # a - 1 (None: a is 1) and b of every layer, per form
    # Block AttnRes (EncoderConfig.attn_res; ced/model/attn_res.py): a GDN layer's mixer reads its depth attention's
    # output, and the layers run through the model's own walk (Walk.run: the layer call of the model's forward).
    walk = enc.attn_res.walk(enc, h) if getattr(enc, "attn_res", None) is not None else None
    ccr_rec = None  # the GDN-side arm CCR on the plain stack: the last GDN layer's record (R1-entry's walk carries its own)
    for i, layer in enumerate(enc.layers):
        if layer.kind == "gdn":
            with auto():
                x = layer.mixer_in_norm(h if walk is None else walk.mixer_input(i))
            # the GDN-side write-mix arm (ced/model/gdn_arms.py): the layer's normed write mix over the walk's finished blocks, as
            # EncoderLayer.gdn_arms_mixer forms it (None in every other layer and model)
            arms = getattr(layer, "gdn_arms", None)
            x_write = x_read = None
            if arms is not None and arms.write_mix != "none":
                from ced.model.gdn_arms import write_mixes

                with auto():  # m' and, under the wsl variant "qmix", m'_q: as EncoderLayer.gdn_arms_mixer forms them
                    x_write, x_read = write_mixes(layer.mixer_in_norm, arms, [v.detach() for v in walk.done])
            if layer_hook is not None:
                assert x_write is None, "a layer hook reads the layer's x alone: not built for the GDN-side write-mix arm"
                layer_hook(i, layer.mixer, x, ctx)
            else:
                stats.add(i, gdn_layer_walk(layer.mixer, x, ctx, positions, n_valid, stats.edges, block, x_write,
                                            arms if x_write is not None else None, x_read))
            del x, x_write, x_read
        if walk is not None:
            with auto():
                walk.run(i, ctx, None)
            continue
        h_in = h if scalars else None
        with auto():
            if scalars:  # the layer's own residual add with its scalar terms (EncoderLayer.forward), as in the model
                h, _ = layer(h_in, ctx, None, scalars=(x0, None if gain is None else gain[i], None if b is None else b[i]))
            elif getattr(layer, "gdn_arms", None) is not None:  # the GDN-side arm CCR on the plain stack (Encoder._stack_arms)
                from ced.model.gdn_arms import hand_record, record_in

                h, c = layer(h, ctx, None, ccr=record_in(layer.gdn_arms, ccr_rec))
                ccr_rec, _ = hand_record(layer.gdn_arms, ccr_rec, c)
            else:
                h, _ = layer(h, ctx, None)
    stats.end_row()


class GDNDiagHooks:
    """Forward pre-hooks on every GatedDeltaNet of an eager model: inside the `with`, each model forward over ONE row
    (no cache) adds the row's statistics to `stats`. Call row(n_valid) before each forward; layers are numbered by
    their index among the encoder's layers when the module is one (else by the order of model.modules()).

        stats = GDNStats(edges)
        with GDNDiagHooks(model, stats) as hooks:
            hooks.row(n_valid)
            model(tokens, doc_ids, dec_mask)
    """

    def __init__(self, model: torch.nn.Module, stats: GDNStats, per_bucket: int = 16, block: int = 8192):
        self.model, self.stats, self.per_bucket, self.block = unwrap(model), stats, per_bucket, block
        self.n_valid: int | None = None
        self.handles: list = []
        self.seen = 0

    def row(self, n_valid: int) -> None:
        if self.seen:
            self.stats.end_row()
        self.n_valid, self.seen = n_valid, 0

    def __enter__(self) -> "GDNDiagHooks":
        index = {}
        encoder = getattr(self.model, "encoder", None)
        for i, layer in enumerate(getattr(encoder, "layers", []) or []):
            if isinstance(getattr(layer, "mixer", None), GatedDeltaNet):
                index[layer.mixer] = i
        for j, m in enumerate(mod for mod in self.model.modules() if isinstance(mod, GatedDeltaNet)):
            name = index.get(m, 10_000 + j)
            self.handles.append(m.register_forward_pre_hook(self._hook(name), with_kwargs=True))
        return self

    def _hook(self, name: int):
        def hook(module, args, kwargs):
            x = args[0] if args else kwargs["x"]
            ctx = args[1] if len(args) > 1 else kwargs["ctx"]
            state = args[2] if len(args) > 2 else kwargs.get("state")
            assert state is None, "GDN diagnostics need a forward without a cache"
            assert self.n_valid is not None, "call row(n_valid) before the forward"
            positions = sample_positions(self.n_valid, self.stats.edges, self.per_bucket)
            # the GDN-side write-mix arm (ced/model/gdn_arms.py): the layer's write input comes with the call (None otherwise)
            arms, x_write, x_read = kwargs.get("arms"), kwargs.get("x_write"), kwargs.get("x_read")
            self.stats.add(name, gdn_layer_walk(module, x, ctx, positions, self.n_valid, self.stats.edges, self.block,
                                                x_write, arms if x_write is not None else None, x_read))
            self.seen += 1

        return hook

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        if self.seen:
            self.stats.end_row()
            self.seen = 0
