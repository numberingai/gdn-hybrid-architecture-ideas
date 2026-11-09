"""Analysis of the copying probes' per-item results (ced/eval/probe_gen_copy.py).

Judged metrics (one pooled number per probe, with its item-sampling standard error; the form of the other
probes: an item mean and std / sqrt(n)):

copy_span  JUDGED = nats per token of the answer span, item mean pooled over the plain cells WITHIN the training
           length (dist 1024, 4096, 8192; types x lengths x sources, equal item counts per cell), SE = std / sqrt(n).
           Companion `bpb` (bits per byte of the span; tokenizer-neutral: use it to compare t64k with t128k).
           Beyond-length cells (16384, 32768) and the near_dup family are reported, not pooled into the judged number.
edit_old   JUDGED = nats per token of the old text, pooled over the four MAIN renderings (json_num, json_plain,
           dec_num, dec_plain) and all (lines, span) cells. The four renderings of one item share its file and its
           line range, so the item is the cluster: the value of a cluster is the mean of its four renderings and the SE is
           std over clusters / sqrt(clusters). Companion `bpb` (bits per byte of the unescaped old text; the tokens of the
           renderings differ, the text does not). Per-rendering means with SE are diagnostics, and the paired
           differences between renderings (same cluster) carry their own paired SE; the diagnostic renderings raw_*
           separate escaping (json vs raw: same frame, text escaped or not) from the novelty of v2's delimiters (raw vs
           dec: unescaped text, JSON frame vs control tokens).
Companions on answer tokens 2..n (COMPANION_KEYS): nats_per_token_rest, token_acc_rest, exact_rest,
and the first token alone (nats_first, first_acc), next to the judged pools and per rendering; an item with one answer
token has no rest metrics and is left out of their means.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ced.eval.probes import REST_KEYS, item_metrics, mean_se
from ced.eval.probe_gen_copy import BEYOND_DISTS, COPY_PROBE, EDIT_PROBE, MAIN_RENDERS, WITHIN_DISTS

LN2 = math.log(2.0)
COMPANION_KEYS = REST_KEYS + ("nats_first", "first_acc")  # answer tokens 2..n, and the first token alone
TABLE_METRICS = ("exact", "token_acc", "nats_per_token", "floor_exact", "floor_nats_per_token", "exp_exact_t1", "exp_exact_t07",
                 "nats_per_token_t07", "first_err_frac", "err", "bpb") + COMPANION_KEYS


def _rows(results: list[dict], probe: str) -> list[dict]:
    return [r for r in results if r["probe"] == probe]


def copy_judged_items(results: list[dict]) -> list[dict]:
    return [r for r in _rows(results, COPY_PROBE) if r["cell"]["variant"] == "plain" and r["cell"]["dist"] in WITHIN_DISTS]


def judged_copy(results: list[dict]) -> dict | None:
    rs = copy_judged_items(results)
    if not rs:
        return None
    out = {"n": len(rs)}
    for k in ("nats_per_token", "bpb", "exact", "exp_exact_t1", "exp_exact_t07", "first_err_frac") + COMPANION_KEYS:
        vals = [item_metrics(r).get(k) for r in rs]
        vals = [v for v in vals if v is not None]
        if vals:
            out[k], out[k + "_se"] = mean_se(vals)
    out["floor_nats_per_token"], out["floor_nats_per_token_se"] = mean_se(item_metrics(r)["floor_nats_per_token"] for r in rs)
    return out


def edit_clusters(results: list[dict], renders=MAIN_RENDERS) -> dict[str, dict[str, dict]]:
    """{base: {render: item_metrics}} for the given renderings (clusters with all of them only)."""
    by: dict = defaultdict(dict)
    for r in _rows(results, EDIT_PROBE):
        if r["cell"]["render"] in renders:
            by[r["base"]][r["cell"]["render"]] = item_metrics(r)
    return {b: d for b, d in by.items() if all(x in d for x in renders)}


def judged_edit(results: list[dict]) -> dict | None:
    cl = edit_clusters(results)
    if not cl:
        return None
    out = {"n_clusters": len(cl), "n_rows": len(cl) * len(MAIN_RENDERS)}
    for k in ("nats_per_token", "bpb", "exact", "exp_exact_t1", "exp_exact_t07", "first_err_frac") + COMPANION_KEYS:
        # a companion key is pooled over the clusters whose four renderings all have it (rest: two answer tokens or more)
        vals = [float(np.mean([d[r][k] for r in MAIN_RENDERS])) for d in cl.values() if all(k in d[r] for r in MAIN_RENDERS)]
        if vals:
            out[k], out[k + "_se"] = mean_se(vals)
    out["per_render"] = {r: {k: dict(zip(("mean", "se"), mean_se(d[r][k] for d in cl.values() if k in d[r])))
                             for k in ("nats_per_token", "bpb", "exact", "exp_exact_t07", "first_err_frac") + COMPANION_KEYS}
                         for r in MAIN_RENDERS}
    return out


def edit_render_differences(results: list[dict]) -> dict:
    """Paired (same cluster) differences a - b of nats per token and exact between renderings, with the paired SE,
    over the clusters that have both. Includes the diagnostic raw_* renderings when scored."""
    allr = MAIN_RENDERS + ("raw_num", "raw_plain")
    by = defaultdict(dict)
    for r in _rows(results, EDIT_PROBE):
        by[r["base"]][r["cell"]["render"]] = item_metrics(r)
    pairs = [("json_num", "dec_num"), ("json_plain", "dec_plain"),            # format at a fixed view
             ("json_num", "json_plain"), ("dec_num", "dec_plain"),              # line numbers at a fixed format
             ("json_num", "raw_num"), ("json_plain", "raw_plain"),              # escaping (same frame)
             ("raw_num", "dec_num"), ("raw_plain", "dec_plain")]                # delimiters (text unescaped)
    out = {}
    for a, b in pairs:
        if a not in allr or b not in allr:
            continue
        rows = [(d[a], d[b]) for d in by.values() if a in d and b in d]
        if not rows:
            continue
        rec = {"n": len(rows)}
        for k in ("nats_per_token", "bpb", "exact"):
            rec[k], rec[k + "_se"] = mean_se(x[k] - y[k] for x, y in rows)
        out[f"{a} - {b}"] = rec
    return out


def judged(results: list[dict]) -> dict:
    return {k: v for k, v in (("copy_span", judged_copy(results)), ("edit_old", judged_edit(results))) if v}


def paired_judged_copy(a: list[dict], b: list[dict]) -> dict | None:
    """Mean over items of (a - b) of nats per token on the judged copy items, with the paired SE (same items by id)."""
    ib = {r["id"]: r for r in copy_judged_items(b)}
    pairs = [(item_metrics(r), item_metrics(ib[r["id"]])) for r in copy_judged_items(a) if r["id"] in ib]
    if not pairs:
        return None
    out = {"n": len(pairs)}
    for k in ("nats_per_token", "bpb", "exact"):
        out[k], out[k + "_se"] = mean_se(x[k] - y[k] for x, y in pairs)
    return out


def paired_judged_edit(a: list[dict], b: list[dict]) -> dict | None:
    ca, cb = edit_clusters(a), edit_clusters(b)
    common = [x for x in ca if x in cb]
    if not common:
        return None
    out = {"n_clusters": len(common)}
    for k in ("nats_per_token", "bpb", "exact"):
        out[k], out[k + "_se"] = mean_se(np.mean([ca[x][r][k] for r in MAIN_RENDERS]) - np.mean([cb[x][r][k] for r in MAIN_RENDERS])
                                         for x in common)
    return out


def marginal_table(results: list[dict], probe: str, by: tuple[str, ...], metrics=TABLE_METRICS, where=None) -> dict:
    """Item means (+ SE) of `metrics` grouped by the cell keys `by` (a marginal table: type, len, dist, src, variant or
    lines, span, render). `where`: a predicate on the cell dict."""
    groups: dict = defaultdict(list)
    for r in _rows(results, probe):
        if where is None or where(r["cell"]):
            groups[tuple(r["cell"][k] for k in by)].append(item_metrics(r))
    out = {}
    for key, ms in sorted(groups.items(), key=lambda kv: tuple(str(x) if not isinstance(x, (int, float)) else x for x in kv[0])):
        rec = {"n": len(ms)}
        for k in metrics:
            vals = [m[k] for m in ms if k in m]
            if vals:
                rec[k], rec[k + "_se"] = mean_se(vals)
        out[",".join(f"{a}={b}" for a, b in zip(by, key))] = rec
    return out


def first_error_hist(results: list[dict], probe: str, where=None, bins=(0, 0.25, 0.5, 0.75, 1.0)) -> dict:
    """Where the first greedy error falls: the share of items with no error, and of the items with an error the share per
    quartile of the span (first_err_tok / n_answer in [0, .25), ...), plus the median first-error token."""
    fr, tok = [], []
    n = 0
    for r in _rows(results, probe):
        if where is not None and not where(r["cell"]):
            continue
        n += 1
        m = item_metrics(r)
        if m["err"]:
            fr.append(m["first_err_frac"])
            tok.append(m["first_err_tok"])
    h = np.histogram(fr, bins=[*bins[:-1], 1.0000001])[0] if fr else np.zeros(len(bins) - 1)
    return {"n": n, "no_error_share": 1 - len(fr) / n if n else None,
            "quartile_share_of_errors": [float(x) / max(1, len(fr)) for x in h], "median_first_error_token": float(np.median(tok)) if tok else None}


def beyond_vs_within(results: list[dict]) -> dict:
    """Plain copy cells beyond the training length against within, per metric: item means with SE."""
    def pooled(dists):
        ms = [item_metrics(r) for r in _rows(results, COPY_PROBE) if r["cell"]["variant"] == "plain" and r["cell"]["dist"] in dists]
        return {k: dict(zip(("mean", "se"), mean_se(m[k] for m in ms))) for k in ("nats_per_token", "exact", "exp_exact_t07")} | {"n": len(ms)} if ms else None
    return {"within": pooled(WITHIN_DISTS), "beyond": pooled(BEYOND_DISTS)}


def judged_cells(results: list[dict]) -> dict:
    """The judged pools as pseudo-cells ({unit id: (fingerprint, metrics)} per probe) for a noise analysis: {(probe, "JUDGED"): {unit id: (fingerprint, metrics)}}.
    copy_span: the plain within-length items (unit = item). edit_old: one unit per cluster (the item's four main renderings),
    its metrics the mean over the four renderings, its fingerprint the four fingerprints joined."""
    out: dict = {}
    cp = copy_judged_items(results)
    if cp:
        out[(COPY_PROBE, "JUDGED")] = {r["id"]: (r["fingerprint"], item_metrics(r)) for r in cp}
    by: dict = defaultdict(dict)
    for r in _rows(results, EDIT_PROBE):
        if r["cell"]["render"] in MAIN_RENDERS:
            by[r["base"]][r["cell"]["render"]] = r
    units = {}
    for b, d in by.items():
        if all(x in d for x in MAIN_RENDERS):
            ms = [item_metrics(d[x]) for x in MAIN_RENDERS]
            keys = [k for k in ms[0] if all(k in m for m in ms)]
            units[b] = ("|".join(d[x]["fingerprint"] for x in MAIN_RENDERS), {k: float(np.mean([m[k] for m in ms])) for k in keys})
    if units:
        out[(EDIT_PROBE, "JUDGED")] = units
    return out
