"""TST mechanism diagnostics: the pieces that a GPU scorer, CPU analyses and their readers share. The purpose is to find
where TST's gain lives and what breaks at the switch in the GDN hybrid.

Everything here is a pure function of arrays and eval rows; nothing reads data files and nothing is trained.

DEFINITIONS:

FREQUENCY CLASSES. counts[i] = the number of times token id i occurs in the whole training shard set
(every document and position). Rank the ids by (count descending, id ascending); m_i = the training
tokens of the ids ranked before i, as a share of all training tokens. Four disjoint classes:
    frequent  m_i < 1/3          mid  1/3 <= m_i < 2/3          rare  2/3 <= m_i < 0.95          rarest  m_i >= 0.95
(ids with count 0 are in "rarest"). Each of the first three third-of-the-mass classes carries about a third of the training
tokens; "rare" in the three-bucket tables (rare / mid / frequent) is rare + rarest, and "rarest" is the
last 5% of the training mass, reported beside it. The comparison is exact integer arithmetic (no float edge).

POSITION BUCKETS. A scored position is a position j of a row of length L whose next token is scored by the
four-set bpb (target_valid and dec_mask: exactly the positions ced.eval.span_bpb.evaluate sums into dec_bpb).
    row position j (0-based; the model has seen j + 1 tokens of the row): [0,128) [128,512) [512,2048) [2048,L);
        "first 512" = j < 512, "later" = j >= 512;
    context in the document c = j minus the index of the first token of the position's document in the row (the tokens of
        the same document the model sees before this one): [0,512) [512,inf). A document that started in an earlier row
        has c counted from the row's start: the model has no earlier row in its context.

BUCKET SUMS. For a set, a class and a model: N = nats, B = UTF-8 bytes, T = tokens of the scored target positions in the
class. The set's dec_bpb = sum N / (ln 2 sum B). The class's own bpb = N / (ln 2 B); its CONTRIBUTION to the set's bpb =
N / (ln 2 B_total). A difference between two models on the same tokens has contributions that add up to the set's bpb
difference, so "where the gain is" is a sum, not a ratio of ratios. The classes of the TARGET token are the primary
split (what the head predicts); the classes of the INPUT token (the token at the position, whose embedding is read) are
the secondary split.

GRAFTS. The tensors that make a model's "embeddings and head": embed.weight (the input embedding) and
out_embed.weight (the untied output head); with norms, also embed_norm.weight (the gain of the norm right after the
embedding) and final_norm.weight (the gain of the norm right before the head). Everything else is the body. A graft is
the body of one model with the embedding and head tensors of another; a graft needs identical tensor shapes and an
untied head.

CONV SPAN. A GDN layer's short conv is depthwise with kernel K; conv_weight[:, K-1] multiplies the current token and
conv_weight[:, K-1-j] the token j steps back (lag j). Per channel, the energy at lag j is e_j = w_j^2 / sum_k w_k^2.
    centroid lag  = sum_j j e_j   (0: only the current token; K-1: only the oldest tap);
    span90        = the smallest m such that e_0 + ... + e_{m-1} >= 0.9 (taps counted from the current token);
    pass-through  = the channel has e_0 >= 0.9 (span90 = 1).
Reported per layer and per channel group (q, k, v) as means over channels; the weights alone, no data.

THE NORM AFTER THE BAG POOLING (if a norm already follows the pooling, the sum-pooling variant H is moot):
the model pools a bag as embed_norm(mean_k embed(tokens_k)) (ced/model/control.py embed_tokens) and embed_norm is
RMSNorm(eps): y = x / sqrt(mean(x^2) + eps) (1 + w). Replacing the mean m by sum / sqrt(s) = sqrt(s) m changes y only through
eps: y_H = m / sqrt(ms + eps / s) against y = m / sqrt(ms + eps), ms = mean(m^2). `bag_norm_eps_change` is the factor
sqrt((ms + eps) / (ms + eps / s)) - 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

LN2 = math.log(2.0)

CLASSES = ("frequent", "mid", "rare", "rarest")
N_CLASSES = len(CLASSES)
ROWPOS_EDGES = (0, 128, 512, 2048)          # the last bucket runs to the row's end
ROWPOS_NAMES = ("0-127", "128-511", "512-2047", "2048-end")
FIRST_512 = 512                              # "first 512" = row position < 512
CTX_EDGES = (0, 512)
CTX_NAMES = ("0-511", "512-end")
EMBED_KEYS = ("embed.weight", "out_embed.weight")
NORM_KEYS = ("embed_norm.weight", "final_norm.weight")
SPLITS = ("tclass", "iclass", "rowpos", "ctx", "tclass_pos2")   # the splits kept per row (the order of `aggregate_ce`'s output)


# --------------------------------------------------------------------------------------------------------------
# frequency classes


def frequency_classes(counts) -> tuple[np.ndarray, dict]:
    """(classes, info): classes (V,) uint8 in 0..3 (CLASSES) from the training counts of every token id (the module
    docstring), info = per class the number of ids, the training mass share and the smallest and largest count."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0 or counts.min() < 0 or counts.sum() <= 0:
        raise ValueError("counts: a 1-D array of non-negative counts with a positive sum")
    total = int(counts.sum())
    order = np.lexsort((np.arange(counts.size), -counts))
    before = np.cumsum(counts[order]) - counts[order]
    cls_sorted = np.full(counts.size, 3, dtype=np.uint8)
    cls_sorted[before * 20 < 19 * total] = 2
    cls_sorted[before * 3 < 2 * total] = 1
    cls_sorted[before * 3 < total] = 0
    classes = np.empty(counts.size, dtype=np.uint8)
    classes[order] = cls_sorted
    info = {"total_tokens": total, "vocab": int(counts.size), "zero_count_ids": int((counts == 0).sum()), "classes": {}}
    for k, name in enumerate(CLASSES):
        sel = classes == k
        c = counts[sel]
        info["classes"][name] = {"ids": int(sel.sum()), "mass": float(c.sum() / total),
                                 "min_count": int(c.min()) if c.size else None, "max_count": int(c.max()) if c.size else None}
    return classes, info


def class_name_three(k: int) -> str:
    """The three-bucket name of class k: rare absorbs rarest."""
    return "frequent" if k == 0 else ("mid" if k == 1 else "rare")


# --------------------------------------------------------------------------------------------------------------
# per-row sums of a set's per-position cross-entropy


def context_in_document(doc_ids: torch.Tensor) -> torch.Tensor:
    """(N, L) long: the number of earlier positions of the same document in the row (0 at the first position of each
    document in the row)."""
    N, L = doc_ids.shape
    idx = torch.arange(L).expand(N, L)
    start = torch.ones(N, L, dtype=torch.bool)
    start[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    first = torch.where(start, idx, torch.zeros_like(idx)).cummax(dim=1).values
    return idx - first


def _bucket(x: torch.Tensor, edges) -> torch.Tensor:
    """Bucket index of x for ascending lower edges `edges` (edges[0] = 0): the last i with edges[i] <= x."""
    return torch.bucketize(x.contiguous(), torch.tensor(edges[1:], dtype=x.dtype), right=True)


def _sum_by(key: torch.Tensor, K: int, scored: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """(N, K) float64: per row, the sum of `value` over the scored positions whose key is k."""
    N, L = key.shape
    flat = (torch.arange(N)[:, None] * K + key).reshape(-1)
    out = torch.zeros(N * K, dtype=torch.float64)
    out.index_add_(0, flat, torch.where(scored, value.double(), torch.zeros((), dtype=torch.float64)).reshape(-1))
    return out.view(N, K)


def aggregate_ce(ce: torch.Tensor, rows, classes: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row sums of nats, bytes and tokens of the scored positions of `rows` (an EvalRows) for the per-position
    cross-entropy `ce` (N, L) float64, split by the module docstring's buckets. Returns arrays shaped (N, ...):
      nats, bytes, tokens   (N,)  the row totals (their sums are the set's dec_bpb numerator and denominator);
      <split>_nats, <split>_bytes, <split>_tokens   for split in SPLITS:
          tclass (N, 4) target-token class; iclass (N, 4) input-token class; rowpos (N, 4); ctx (N, 2);
          tclass_pos2 (N, 4, 2) target class x (row position < 512, >= 512).
    Only positions with target_valid and dec_mask are summed (the positions evaluate() scores as dec)."""
    N, L = ce.shape
    assert tuple(rows.tokens.shape) == (N, L), f"ce {tuple(ce.shape)} against rows {tuple(rows.tokens.shape)}"
    cls = torch.from_numpy(np.asarray(classes, dtype=np.int64))
    scored = rows.target_valid & rows.dec_mask
    nbytes = rows.target_bytes
    tcls, icls = cls[rows.targets], cls[rows.tokens]
    pos = torch.arange(L).expand(N, L)
    rowpos = _bucket(pos, ROWPOS_EDGES)
    ctx = _bucket(context_in_document(rows.doc_ids), CTX_EDGES)
    pos2 = (pos >= FIRST_512).long()
    keys = {"tclass": (tcls, N_CLASSES), "iclass": (icls, N_CLASSES), "rowpos": (rowpos, len(ROWPOS_NAMES)),
            "ctx": (ctx, len(CTX_NAMES)), "tclass_pos2": (tcls * 2 + pos2, N_CLASSES * 2)}
    one = torch.ones(N, L, dtype=torch.float64)
    out = {"nats": torch.where(scored, ce.double(), torch.zeros((), dtype=torch.float64)).sum(1).numpy(),
           "bytes": torch.where(scored, nbytes, torch.zeros((), dtype=torch.long)).sum(1).double().numpy(),
           "tokens": scored.long().sum(1).double().numpy()}
    for name, (key, K) in keys.items():
        for what, val in (("nats", ce.double()), ("bytes", nbytes), ("tokens", one)):
            a = _sum_by(key, K, scored, val).numpy()
            out[f"{name}_{what}"] = a.reshape(N, N_CLASSES, 2) if name == "tclass_pos2" else a
    return out


def check_aggregate(agg: dict[str, np.ndarray], row_nats: np.ndarray, row_bytes: np.ndarray, row_tokens: np.ndarray,
                    rtol: float = 1e-9) -> None:
    """The aggregate must be the evaluation's own sums: the row totals equal evaluate()'s row_dec_nats, row_dec_bytes and
    row_dec_tokens, and every split adds up to the row totals. AssertionError otherwise."""
    # equal_nan=False: two NaN are a NaN difference, which fails; the default would pass them
    np.testing.assert_allclose(agg["nats"], row_nats, rtol=rtol, atol=1e-9, equal_nan=False)
    np.testing.assert_array_equal(agg["bytes"], row_bytes)
    np.testing.assert_array_equal(agg["tokens"], row_tokens)
    for name in SPLITS:
        for what in ("nats", "bytes", "tokens"):
            a = agg[f"{name}_{what}"]
            np.testing.assert_allclose(a.reshape(a.shape[0], -1).sum(1), agg[what], rtol=rtol, atol=1e-9, equal_nan=False,
                                       err_msg=f"{name}_{what}")


# --------------------------------------------------------------------------------------------------------------
# capturing the per-position cross-entropy of ced.eval.span_bpb.evaluate


class CECapture:
    """Inside the `with`, every call of span_bpb._scored_ce (the one place evaluate() takes a control model's per-position
    cross-entropy from) also keeps its result: the numbers evaluate() sums are the numbers kept, not a second computation.
    `ce()` is the (N, L) float64 tensor of all batches in order. Read-only: the wrapped function's return value is
    returned unchanged. Not thread-safe (it patches a module attribute for the duration)."""

    def __enter__(self):
        from ced.eval import span_bpb as sb

        self._sb, self._orig, self.parts = sb, sb._scored_ce, []
        outer = self

        def wrapper(*args, **kwargs):
            ce = outer._orig(*args, **kwargs)
            outer.parts.append(ce.detach().double().cpu())
            return ce

        sb._scored_ce = wrapper
        return self

    def __exit__(self, *exc):
        self._sb._scored_ce = self._orig
        return False

    def ce(self) -> torch.Tensor:
        if not self.parts:
            raise RuntimeError("no cross-entropy was captured: evaluate() did not reach _scored_ce (a control model with chunked_head=True does)")
        return torch.cat(self.parts, dim=0)


# --------------------------------------------------------------------------------------------------------------
# grafts


@dataclass(frozen=True)
class Condition:
    """One scoring pass: the body of `body`, with the embedding and head (and, with norms, their norm gains) of `donor`."""

    name: str
    body: str
    donor: str | None = None
    norms: bool = False

    @property
    def grafted(self) -> bool:
        return self.donor is not None

    def describe(self) -> str:
        if not self.grafted:
            return f"{self.name}: {self.body} as trained"
        return f"{self.name}: body of {self.body} with the embedding and head{' and their norm gains' if self.norms else ''} of {self.donor}"


def parse_condition(text: str) -> Condition:
    """`name=body` (a model as trained) or `name=body<-donor` or `name=body<-donor+norms`."""
    name, sep, spec = text.partition("=")
    if not sep or not name or not spec:
        raise ValueError(f"condition {text!r}: name=body or name=body<-donor[+norms]")
    norms = spec.endswith("+norms")
    spec = spec[: -len("+norms")] if norms else spec
    body, arrow, donor = spec.partition("<-")
    if not body or (arrow and not donor) or (norms and not arrow):
        raise ValueError(f"condition {text!r}: name=body or name=body<-donor[+norms]")
    return Condition(name, body, donor or None, norms)


def graft_state(body: dict, donor: dict, norms: bool = False) -> dict:
    """A new state dict: `body`'s tensors, with EMBED_KEYS (and NORM_KEYS if norms) taken from `donor`. Neither input is
    changed and the donor's tensors are copied. ValueError for a missing key, a shape that differs, or a tied head."""
    keys = EMBED_KEYS + (NORM_KEYS if norms else ())
    for k in keys:
        for who, sd in (("body", body), ("donor", donor)):
            if k not in sd:
                raise ValueError(f"{who} state has no {k}" + (" (a tied head has no out_embed.weight)" if k == "out_embed.weight" else ""))
        if tuple(body[k].shape) != tuple(donor[k].shape):
            raise ValueError(f"{k}: shape {tuple(body[k].shape)} against {tuple(donor[k].shape)}")
    out = dict(body)
    for k in keys:
        out[k] = donor[k].detach().clone()
    return out


# --------------------------------------------------------------------------------------------------------------
# the short conv's span and the bag norm


def conv_span_stats(weight: torch.Tensor, groups: dict[str, tuple[int, int]]) -> dict:
    """The module docstring's conv span for conv_weight (channels, K), per channel group {name: (first, last + 1)} and
    for all channels: {"all": {...}, <group>: {"channels", "lag_energy": [K], "centroid_lag", "span90", "pass_through"}},
    each a mean over the group's channels (lag_energy[j] = mean of e_j; pass_through = the share of channels with e_0 >= 0.9)."""
    w = weight.detach().double()
    K = w.shape[1]
    e = w.flip(1).pow(2)                     # column j = lag j (conv_weight[:, K-1] is the current token)
    tot = e.sum(1, keepdim=True)
    e = torch.where(tot > 0, e / tot.clamp_min(1e-300), torch.full_like(e, 1.0 / K))
    cum = e.cumsum(1)
    span90 = (cum < 0.9 - 1e-12).sum(1) + 1   # the smallest m with cum[m-1] >= 0.9
    lag = torch.arange(K, dtype=torch.float64)
    out = {}
    for name, (a, b) in {"all": (0, w.shape[0]), **groups}.items():
        sl = slice(a, b)
        out[name] = {"channels": int(b - a), "lag_energy": e[sl].mean(0).tolist(), "centroid_lag": float((e[sl] * lag).sum(1).mean()),
                     "span90": float(span90[sl].double().mean()), "pass_through": float((e[sl, 0] >= 0.9).double().mean())}
    return out


def bag_norm_eps_change(mean_sq: np.ndarray, s: int, eps: float) -> np.ndarray:
    """The relative change of embed_norm's output when the bag's mean m is replaced by sum / sqrt(s) = sqrt(s) m, as a
    function of ms = mean(m^2) (per bag): sqrt((ms + eps) / (ms + eps / s)) - 1 (the module docstring)."""
    ms = np.asarray(mean_sq, dtype=np.float64)
    return np.sqrt((ms + eps) / (ms + eps / s)) - 1.0
