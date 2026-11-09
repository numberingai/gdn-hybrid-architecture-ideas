"""Analysis of the transcoding probe's per-item results (ced/eval/probe_gen_transcode.py).

JUDGED (one pooled number, item SE): nats per token of the answer, pooled over both directions and the 3-line and 8-line
cells. The two directions of a base (same content) share their text, so the unit is the BASE: its value is the mean of its
two directions, SE = std over bases / sqrt(bases). Companion `bpb` (bits per byte of the raw text; tokenizer-neutral).
Why 3 and 8 lines: 1 line is 6-23 answer tokens, dominated by the first token and the frame; 20 lines is 180-410 tokens and a
single slip dominates exact match, while nats per token is smooth for it, but a 60M model is expected near the copy-only level
there, which adds noise and little separation; 3 and 8 lines carry several escapes per item and still leave range
(this is a proposal; the per-length tables are reported so another pooling can be read off).
Companions on answer tokens 2..n (COMPANION_KEYS: nats_per_token_rest, token_acc_rest, exact_rest) and on the first token
alone (nats_first, first_acc), in the judged pool, per direction and in the cell table; an item
with one scored answer token has no rest metrics and is left out of their means.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ced.eval.probe_gen_transcode import DIRS_, LINES, PROBE
from ced.eval.probes import REST_KEYS, item_metrics, mean_se

JUDGED_LINES = (3, 8)
KEYS = ("nats_per_token", "bpb", "exact", "exp_exact_t1", "exp_exact_t07", "first_err_frac", "token_acc")
COMPANION_KEYS = REST_KEYS + ("nats_first", "first_acc")  # answer tokens 2..n, and the first token alone


def _rows(results):
    return [r for r in results if r["probe"] == PROBE]


def clusters(results, lines=JUDGED_LINES) -> dict[str, dict[str, dict]]:
    """{base: {direction: item_metrics}} for the bases with both directions and a line count in `lines`."""
    by: dict = defaultdict(dict)
    for r in _rows(results):
        if r["cell"]["lines"] in lines:
            by[r["base"]][r["cell"]["dir"]] = item_metrics(r)
    return {b: d for b, d in by.items() if all(x in d for x in DIRS_)}


def judged_transcode(results, lines=JUDGED_LINES) -> dict | None:
    cl = clusters(results, lines)
    if not cl:
        return None
    out = {"n_bases": len(cl), "n_items": 2 * len(cl), "lines": list(lines)}
    for k in KEYS:
        out[k], out[k + "_se"] = mean_se(float(np.mean([d[x][k] for x in DIRS_])) for d in cl.values())
    for k in COMPANION_KEYS:  # over the bases whose two directions both have the key (rest: two scored tokens or more)
        vals = [float(np.mean([d[x][k] for x in DIRS_])) for d in cl.values() if all(k in d[x] for x in DIRS_)]
        if vals:
            out[k], out[k + "_se"] = mean_se(vals)
    for k in ("floor_token_acc", "floor_match_share", "floor_nats_per_token"):  # copy-only reference
        vals = [float(np.mean([d[x][k] for x in DIRS_])) for d in cl.values() if all(k in d[x] for x in DIRS_)]
        if vals:
            out[k], out[k + "_se"] = mean_se(vals)
    if "floor_nats_per_token" in out:  # model below the copy-with-alignment floor, paired by base (positive gap = better than the floor)
        gaps = [float(np.mean([d[x]["floor_nats_per_token"] - d[x]["nats_per_token"] for x in DIRS_])) for d in cl.values()]
        g, gse = mean_se(gaps)
        out["floor_gap"] = {"mean": g, "se": gse, "z": g / gse if gse > 0 else float("inf")}
    out["per_direction"] = {x: {k: dict(zip(("mean", "se"), mean_se(d[x][k] for d in cl.values() if k in d[x]))) for k in KEYS + COMPANION_KEYS}
                            for x in DIRS_}
    a, b = DIRS_
    out["direction_difference"] = {k: dict(zip(("mean", "se"), mean_se(d[a][k] - d[b][k] for d in cl.values())))
                                   for k in ("nats_per_token", "exact", "first_err_frac")}
    out["all_lengths"] = _pool(results, LINES)
    return out


def _pool(results, lines) -> dict | None:
    cl = clusters(results, lines)
    if not cl:
        return None
    out = {"n_bases": len(cl)}
    for k in ("nats_per_token", "bpb", "exact"):
        out[k], out[k + "_se"] = mean_se(float(np.mean([d[x][k] for x in DIRS_])) for d in cl.values())
    return out


def cell_table(results, metrics=("nats_per_token", "bpb", "exact", "exp_exact_t07", "first_err_frac", "token_acc", "floor_token_acc",
                                 "floor_match_share", "floor_nats_per_token") + COMPANION_KEYS) -> dict:
    """{dir, lines, kind: {n, metric, metric_se}} item means with SE (kind = the item's group)."""
    by: dict = defaultdict(list)
    for r in _rows(results):
        m = item_metrics(r)
        by[(r["cell"]["dir"], r["cell"]["lines"], "all")].append(m)
        by[(r["cell"]["dir"], r["cell"]["lines"], r["group"].get("kind", "?"))].append(m)
    out = {}
    for key, ms in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        rec = {"n": len(ms)}
        for k in metrics:
            vals = [m[k] for m in ms if k in m]
            if vals:
                rec[k], rec[k + "_se"] = mean_se(vals)
        out[f"{key[0]},lines={key[1]},{key[2]}"] = rec
    return out


def judged_cells(results) -> dict:
    """The judged pool as the pseudo-cell (transcode, JUDGED) for a noise analysis: unit = base."""
    by: dict = defaultdict(dict)
    for r in _rows(results):
        if r["cell"]["lines"] in JUDGED_LINES:
            by[r["base"]][r["cell"]["dir"]] = r
    units = {}
    for b, d in by.items():
        if all(x in d for x in DIRS_):
            ms = [item_metrics(d[x]) for x in DIRS_]
            keys = [k for k in ms[0] if all(k in m for m in ms)]
            units[b] = ("|".join(d[x]["fingerprint"] for x in DIRS_), {k: float(np.mean([m[k] for m in ms])) for k in keys})
    return {(PROBE, "JUDGED"): units} if units else {}


# Range rule, fixed before any 60M result. The probe HAS RANGE at 60M for an arm when, over its
# seeds, ALL of: (a) transcoding: pooled judged exact >= EXACT_MIN (a copy-only model has exact 0); (b) below the floor: the
# paired gap floor - model in nats per token is >= GAP_Z_MIN SE in every seed; (c) not saturated: exact <= EXACT_MAX;
# (d) resolution: the SD of the judged nats per token across the arm's seeds <= SEED_SD_MAX_FRAC of the mean gap, i.e. the
# floor-to-model range spans >= 4 seed SDs. The ceiling (copy probe, within-8k level of the same checkpoint) is reported
# next to the result but is not part of the rule: it is natural text, not this content.
EXACT_MIN, EXACT_MAX, GAP_Z_MIN, SEED_SD_MAX_FRAC = 0.01, 0.95, 3.0, 0.25


def range_verdict(judged_by_seed: list[dict]) -> dict:
    """`judged_by_seed`: the `judged.transcode` dicts (judged_transcode output) of one arm's seeds."""
    npt = [j["nats_per_token"] for j in judged_by_seed]
    gap = [j["floor_gap"]["mean"] for j in judged_by_seed]
    sd = float(np.std(npt, ddof=1)) if len(npt) > 1 else float("nan")
    v = {"seeds": len(npt), "exact": [j["exact"] for j in judged_by_seed], "gap": gap, "gap_z": [j["floor_gap"]["z"] for j in judged_by_seed],
         "seed_sd": sd, "mean_gap": float(np.mean(gap))}
    v["transcodes"] = all(j["exact"] >= EXACT_MIN for j in judged_by_seed)
    v["below_floor"] = all(j["floor_gap"]["z"] >= GAP_Z_MIN for j in judged_by_seed)
    v["not_saturated"] = all(j["exact"] <= EXACT_MAX for j in judged_by_seed)
    v["resolved"] = bool(sd <= SEED_SD_MAX_FRAC * v["mean_gap"])
    v["has_range"] = bool(v["transcodes"] and v["below_floor"] and v["not_saturated"] and v["resolved"])
    return v
