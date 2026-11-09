"""Evals beyond bpb, part 1: loss by token position within 8k, copy loss on a repeated random token sequence, and
retrieval inside the training length (single and multi-key needle-in-a-haystack, variable tracking, up to 8k; the
extrapolation cells stay separate). Pure functions of arrays and probe results: nothing here reads data files, loads
a model or trains. The retrieval items are in ced/eval/report_pcr_gen.py and their builder is tools/build_report_pcr.py.

DEFINITIONS.

(1) LOSS BY TOKEN POSITION. Input: the per-position cross-entropy ce (N, L) of a model on one of the four held-out sets (web, code,
agent, general_web; EvalRows of ced/eval/span_bpb.py, rows of 8,192 positions), as a per-position scorer writes it
(ce-<set>.npy: the numbers evaluate() sums, float32). A position j has a target (the next token) that is "valid" (target_valid: in the
same document) and "dec" (valid and in the row's decoder mask, the positions the four-set bpb scores; the control scores them with its one
head, which runs at every position, so ce is a real cross-entropy at every valid position too).
    POSITION. Two measures, both bucketed by the same lower edges POS_EDGES = (0, 64, 256, 1024, 2048, 4096), the last bucket running to the
    row's end (8,191): "doc" = the number of earlier positions of the same document in the row (the tokens the model can read: the
    models isolate documents, so this is the true context length; a document that began in an earlier row counts from the row's start,
    as in ced/eval/tst_diag.py context_in_document), and "row" = the index in the row (the packed row's own position).
    MASK. "dec" = the positions the four-set bpb scores (its sums over buckets are the set's dec_bpb, checked); "valid" = every position that
    has a target in its document (about 4x more tokens; the spans of the dec mask are fixed-seed and independent of the tokens, so the
    two are unbiased for the same quantity).
    VALUE. Per set, bucket, position measure and mask: N = nats, B = UTF-8 bytes of the targets, T = tokens; bpb = N / (ln 2 B). The
    FOUR-SET value of a bucket is the mean over the four sets of the set's bucket bpb (the four-set bpb is the mean of the sets'
    dec_bpb, so its bucket form is the same mean); it is defined only when all four sets hold bytes in the bucket. A bucket's share of
    a set's bpb (N_bucket / (ln 2 B_set)) adds up to the set's bpb across buckets: both are kept, so a reader can say where the loss is.
    The unit of a seed comparison is the run: the per-row sums are kept (position-rows.npz) for a row-clustered standard error.

(2) COPY LOSS ON A REPEATED RANDOM SEQUENCE. Item: L token ids drawn uniformly (with replacement) from the vocabulary's non-special ids, from
a fixed seed (COPY_SEED; one stream per (variant, L, item index)); the item is s s (variant pcr_copy2), s s s (pcr_copy3: the copies after
the first are scored), or s t with t an independent draw (pcr_fresh: the same first half, no repeat: the reference for what the model
pays on random ids it cannot copy). Scored: every token of the copies after the first, teacher-forced, in nats (the full-vocabulary
log-probability of the gold token; eval_probes' tok_logp), bucketed by the token's position in its copy, COPY_EDGES = (0, 1, 4, 16, 64, 256,
1024) (a bucket at or above L is empty). Bucket 0, the copy's first token, is a fresh draw to the model for the SECOND copy only (the context holds no
earlier copy); for the third copy of pcr_copy3 it is predictable by induction (the end of the first copy was followed by the start of the second), so it is
its own cell there, and `nats_after_first` leaves bucket 0 out of both copies. Reported in nats per token; the argmax-correct share (tok_correct) beside it. The text is not natural: bytes per token are the table's, answer_bytes is the sum.

(3) RETRIEVAL INSIDE THE TRAINING LENGTH: ced/eval/report_pcr_gen.py (its docstring has the definitions); scored by scripts/eval_probes.py
as every probe; this module only reads the result.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from ced.eval import tst_diag as TD

LN2 = math.log(2.0)

# --------------------------------------------------------------------------------------------------------------
# (1) loss by token position

FOUR = ("web", "code", "agent", "general_web")  # the four sets of the four-set bpb
POS_EDGES = (0, 64, 256, 1024, 2048, 4096)  # lower edges; the last bucket runs to the row's end
POS_NAMES = ("0-63", "64-255", "256-1023", "1024-2047", "2048-4095", "4096-end")
POS_KINDS = ("row", "doc")  # the position measure: the index in the row, the context in the document
MASKS = ("dec", "valid")  # the scored positions of the four-set bpb (decoder mask), every position that has a target
WHATS = ("nats", "bytes", "tokens")
CHECK_RTOL = 1e-5  # a float32 dump against evaluate()'s float64 row sums: 24 bits per position, thousands summed per row


def bucket_sums(ce, rows, edges=POS_EDGES) -> dict[str, np.ndarray]:
    """Per-row sums of nats, bytes and tokens of the positions of `rows` (an EvalRows), split by the module docstring's buckets:
    {"<kind>_<mask>_<what>": (N, K) float64} for kind in POS_KINDS, mask in MASKS, what in WHATS. ce: (N, L), any float dtype, one
    row per row of `rows` (a different shape is an AssertionError)."""
    ce = (ce if torch.is_tensor(ce) else torch.as_tensor(np.array(ce))).double()  # np.array: a writable copy of a read-only mmap
    N, L = ce.shape
    assert tuple(rows.tokens.shape) == (N, L), f"ce {tuple(ce.shape)} against rows {tuple(rows.tokens.shape)}"
    K = len(edges)
    valid = rows.target_valid.bool()
    masks = {"dec": valid & rows.dec_mask.bool(), "valid": valid}
    pos = torch.arange(L).expand(N, L)
    measures = {"row": pos, "doc": TD.context_in_document(rows.doc_ids)}
    values = {"nats": ce, "bytes": rows.target_bytes, "tokens": torch.ones(N, L, dtype=torch.float64)}
    out = {}
    for kind, x in measures.items():
        key = TD._bucket(x, edges)
        for mname, m in masks.items():
            for what, val in values.items():
                out[f"{kind}_{mname}_{what}"] = TD._sum_by(key, K, m, val).numpy()
    return out


def token_counts(rows, edges=POS_EDGES) -> dict:
    """What a set holds in each position bucket, from the rows alone (no model, no dump): {"rows", "seq_len", "tokens": {kind: {mask: [K]}}, "bytes": {kind: {mask:
    [K]}}, "documents": {mask: [K]}} with the scored targets and their UTF-8 bytes per bucket for both position measures and both masks, and the number of distinct
    documents (a document of a row; one that continues into the next row counts in each) that hold at least one scored target in the bucket, on the document measure.
    The decoder token counts of the position tables: a bucket that holds few of them, or few documents, is thin."""
    N, L = rows.tokens.shape
    sums = bucket_sums(torch.zeros(N, L, dtype=torch.float64), rows, edges)
    valid = rows.target_valid.bool()
    masks = {"dec": valid & rows.dec_mask.bool(), "valid": valid}
    key = TD._bucket(TD.context_in_document(rows.doc_ids), edges)
    pair = torch.arange(N)[:, None] * (1 << 40) + rows.doc_ids.long()  # one id per (row, document)
    out: dict = {"rows": int(N), "seq_len": int(L), "tokens": {}, "bytes": {}, "documents": {}}
    for kind in POS_KINDS:
        out["tokens"][kind] = {m: [int(x) for x in sums[f"{kind}_{m}_tokens"].sum(0)] for m in MASKS}
        out["bytes"][kind] = {m: [int(x) for x in sums[f"{kind}_{m}_bytes"].sum(0)] for m in MASKS}
    for m, scored in masks.items():
        out["documents"][m] = [int(torch.unique(pair[scored & (key == k)]).numel()) for k in range(len(edges))]
    return out


def check_sums(sums: dict[str, np.ndarray], row_nats, row_bytes, row_tokens, rtol: float = CHECK_RTOL) -> None:
    """The evaluation's own sums are the buckets' sums: for both position measures, the dec buckets of a row add up to evaluate()'s
    row_dec_nats, row_dec_bytes and row_dec_tokens (nats to rtol; bytes and tokens exactly), and the valid buckets hold at least as much.
    AssertionError otherwise (NaN fails)."""
    for kind in POS_KINDS:
        np.testing.assert_allclose(sums[f"{kind}_dec_nats"].sum(1), np.asarray(row_nats, dtype=np.float64), rtol=rtol, atol=1e-6,
                                   equal_nan=False, err_msg=f"{kind}_dec_nats against row_dec_nats")
        np.testing.assert_array_equal(sums[f"{kind}_dec_bytes"].sum(1), np.asarray(row_bytes, dtype=np.float64), err_msg=f"{kind}_dec_bytes")
        np.testing.assert_array_equal(sums[f"{kind}_dec_tokens"].sum(1), np.asarray(row_tokens, dtype=np.float64), err_msg=f"{kind}_dec_tokens")
        for what in ("bytes", "tokens"):
            assert (sums[f"{kind}_valid_{what}"].sum(1) >= sums[f"{kind}_dec_{what}"].sum(1)).all(), f"{kind}_valid_{what} below dec"


def bpb(nats, nbytes) -> float:
    """nats / (ln 2 bytes); NaN for a bucket with no bytes."""
    return float(nats) / (LN2 * float(nbytes)) if float(nbytes) > 0 else float("nan")


def set_record(sums: dict[str, np.ndarray], edges=POS_EDGES) -> dict:
    """The totals over rows of one set: {kind: {mask: {"nats": [K], "bytes": [K], "tokens": [K], "bpb": [K], "share": [K],
    "bpb_total": float, "tokens_total": int}}}. share[k] = nats[k] / (ln 2 bytes_total): the bucket's part of the set's bpb."""
    out: dict = {}
    for kind in POS_KINDS:
        out[kind] = {}
        for mname in MASKS:
            n, b, t = (sums[f"{kind}_{mname}_{w}"].sum(0) for w in WHATS)
            bt = float(b.sum())
            out[kind][mname] = {
                "nats": n.tolist(), "bytes": b.tolist(), "tokens": t.tolist(),
                "bpb": [bpb(x, y) for x, y in zip(n, b)],
                "share": [float(x) / (LN2 * bt) if bt > 0 else float("nan") for x in n],
                "bpb_total": bpb(n.sum(), bt), "tokens_total": int(t.sum())}
    return out


def four_set(records: dict[str, dict]) -> dict:
    """The mean over FOUR of the sets' bucket bpb, per position measure and mask: {kind: {mask: {"bpb": [K], "bpb_total": float}}}.
    A bucket is NaN unless every one of the four sets has bytes in it; `records` maps a set name to its set_record."""
    missing = [s for s in FOUR if s not in records]
    if missing:
        raise ValueError(f"the four-set value needs the sets {list(FOUR)}; missing {missing}")
    out: dict = {}
    for kind in POS_KINDS:
        out[kind] = {}
        for mname in MASKS:
            per = np.array([records[s][kind][mname]["bpb"] for s in FOUR], dtype=np.float64)  # (4, K); NaN where a set has no bytes
            tot = np.array([records[s][kind][mname]["bpb_total"] for s in FOUR], dtype=np.float64)
            out[kind][mname] = {"bpb": per.mean(0).tolist(), "bpb_total": float(tot.mean())}
    return out


# --------------------------------------------------------------------------------------------------------------
# (2) copy loss on a repeated random token sequence

COPY_SEED = 20261029
COPY_LENGTHS = (64, 256, 1024, 2048)
COPY_ITEMS = {64: 128, 256: 64, 1024: 24, 2048: 12}  # items per length and variant (8k to 25k scored tokens a variant and length; 295k positions in all three variants:
# eval_probes keeps the full-vocabulary log-probabilities of every scored position, so the scored positions, not the tokens read, are what these rows cost)
COPY_EDGES = (0, 1, 4, 16, 64, 256, 1024)  # lower edges of the position in the copy; buckets at or above L are empty
COPY_VARIANTS = ("pcr_copy2", "pcr_copy3", "pcr_fresh")
_VARIANT_CODE = {v: i for i, v in enumerate(COPY_VARIANTS)}


def copy_bucket_names(edges=COPY_EDGES) -> list[str]:
    """"0", "1-3", "4-15", "16-63", "64-255", "256-1023", "1024-end"."""
    names = []
    for i, a in enumerate(edges):
        b = edges[i + 1] if i + 1 < len(edges) else None
        names.append("%d-end" % a if b is None else (str(a) if b == a + 1 else "%d-%d" % (a, b - 1)))
    return names


def allowed_ids(vocab_size: int, special_ids) -> np.ndarray:
    """The ids a random sequence is drawn from: every id of the vocabulary but the special tokens (control tokens would reset or
    end a document for the model; the draw stays uniform over what is left)."""
    special = {int(i) for i in special_ids}
    ids = np.array([i for i in range(int(vocab_size)) if i not in special], dtype=np.int32)
    if len(ids) < 2:
        raise ValueError("fewer than two non-special ids")
    return ids


def _copy_rng(variant: str, L: int, index: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([COPY_SEED, _VARIANT_CODE[variant] if stream else 0, int(L), int(index), int(stream)]))


def copy_item(allowed: np.ndarray, variant: str, L: int, index: int) -> tuple[np.ndarray, int]:
    """(ids, n_answer) of one copy item. The first sequence s of an item index is the same in all three variants (stream 0 is shared;
    only the second draw t of pcr_fresh has a stream of its own), so a fresh item and a repeat item are paired by their first half."""
    if variant not in _VARIANT_CODE:
        raise ValueError(f"variant {variant!r}: one of {COPY_VARIANTS}")
    if L < 2:
        raise ValueError("L >= 2")
    s = allowed[_copy_rng(variant, L, index, 0).integers(0, len(allowed), size=L)]
    if variant == "pcr_copy2":
        return np.concatenate([s, s]).astype(np.int32), L
    if variant == "pcr_copy3":
        return np.concatenate([s, s, s]).astype(np.int32), 2 * L
    t = allowed[_copy_rng(variant, L, index, 1).integers(0, len(allowed), size=L)]
    return np.concatenate([s, t]).astype(np.int32), L


def copy_item_id(variant: str, L: int, index: int) -> str:
    return f"{variant}/L={L}/{index:04d}"


def copy_table(results: list[dict], edges=COPY_EDGES) -> dict:
    """The copy loss by position in the copy, from eval_probes' per-item results (record["items"]) of the three copy probes:
    {probe: {L: {copy_index: {"items", "tokens", "nats": [K], "n": [K], "acc": [K], "nats_all", "acc_all", "nats_after_first"}}}}
    where copy_index is 1 for the second copy, 2 for the third (pcr_copy3 has both); nats[k] is the mean nats per token of the bucket k
    (None when the bucket is empty), acc[k] the share of its tokens whose argmax is the gold token, nats_after_first the mean over the
    positions 1 and up (the copy's first token is a fresh draw to the model for the second copy only; in the third copy it is predictable by induction and
    is its own bucket, left out of nats_after_first of both copies)."""
    K = len(edges)
    acc: dict = defaultdict(lambda: {"items": 0, "sum": np.zeros(K), "n": np.zeros(K), "ok": np.zeros(K)})
    for r in results:
        if r["probe"] not in COPY_VARIANTS:
            continue
        L = int(r["cell"]["L"])
        n = int(r["n_answer"])
        if n % L:
            raise ValueError(f"{r['id']}: {n} scored tokens are not a whole number of copies of {L}")
        lp = np.asarray(r["tok_logp"], dtype=np.float64)
        ok = np.frombuffer(r["tok_correct"].encode(), dtype=np.uint8) == ord("1")
        if len(lp) != n or len(ok) != n:
            raise ValueError(f"{r['id']}: tok_logp / tok_correct hold {len(lp)} / {len(ok)} tokens, n_answer is {n}")
        bucket = np.searchsorted(np.asarray(edges), np.arange(L), side="right") - 1
        for c in range(n // L):
            a = acc[(r["probe"], L, c + 1)]
            seg = slice(c * L, (c + 1) * L)
            a["items"] += 1
            np.add.at(a["sum"], bucket, -lp[seg])
            np.add.at(a["n"], bucket, 1.0)
            np.add.at(a["ok"], bucket, ok[seg].astype(np.float64))
    out: dict = {}
    for (probe, L, c), a in sorted(acc.items()):
        n = a["n"]
        tot, tok = a["sum"].sum(), n.sum()
        after = n.copy()
        after[0] = 0.0
        sum_after = a["sum"].copy()
        sum_after[0] = 0.0
        out.setdefault(probe, {}).setdefault(L, {})[c] = {
            "items": a["items"], "tokens": int(tok),
            "nats": [float(s / m) if m else None for s, m in zip(a["sum"], n)],
            "n": [int(m) for m in n], "acc": [float(o / m) if m else None for o, m in zip(a["ok"], n)],
            "nats_all": float(tot / tok) if tok else None, "acc_all": float(a["ok"].sum() / tok) if tok else None,
            "nats_after_first": float(sum_after.sum() / after.sum()) if after.sum() else None}
    return out
