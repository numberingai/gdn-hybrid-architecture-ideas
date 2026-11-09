"""The long-context evaluation's per-checkpoint summary and the degradation rule:
what scripts/eval_long.py writes as summary.json / summary.md, which says
what breaks first as length grows.

`compact(parts, settings)` reduces the four parts' outputs (ced/eval/long_context.py, probe_gen_multifact.py,
gdn_diag.py) to one record keyed by length. `degradations(summary, thresholds)` lists, per metric and length beyond
the training length, how far the metric has moved from ITS OWN value at <= 8k and whether that passes the stated
threshold:

    metric                     value at length L                                            threshold (default)
    bpb.<set>                  bpb(full context) - bpb(the same targets with 4k-8k of       > +0.010 bpb
                               context), in the position bucket ending at L
    agent.tail                 assistant bpb of a trajectory's last 4,096 targets with a    > +0.010 bpb
                               window of L - the same targets with a window of 8,192
                               (trajectories of at least L + 1 tokens)
    mf.<probe>.k<k>.nats       nats per fact token at length L - at 8k                      > +0.10 nats
    mf.<probe>.k<k>.exact      exact (all k facts) at 8k - at length L                      > 0.15
    gdn.<rows>.<stat>          the worst GDN layer's factor max(r, 1 / r), r = the layer's   > 1.5
                               statistic in the bucket ending at L over its 2k-8k value
                               (state norm, effective rank, relative change per token)

A metric is "degraded" at L when its value exceeds the threshold, and "clear" when it does so by more than twice its
standard error (the GDN factors have no standard error and count as clear; a difference whose standard error is
unknown, one cluster, never does). The bpb and agent values are differences
on the same target tokens, so they do not include the ordinary drop of bpb with position.
"""

from __future__ import annotations

import math

from ced.eval.long_context import length_label

THRESHOLDS = {"bpb": 0.010, "agent": 0.010, "mf_nats": 0.10, "mf_exact": 0.15, "gdn": 1.5}
GDN_FLAG_STATS = ("norm", "erank", "rel_delta")
MF_SHOWN = ("exact", "fact_acc", "nats", "first_nats", "rest_nats", "exact_all")


# --------------------------------------------------------------------------------------------
# compact summary
# --------------------------------------------------------------------------------------------


def _gdn_compact(result: dict) -> dict:
    """GDNStats.result() by bucket end: per statistic the median / min / max ratio over the layers and the flagged
    layers; the samples and positions per bucket; the layers' long heads."""
    by = {}
    for b, (name, hi) in enumerate(zip(result["buckets"], result["edges"][1:])):
        rec = {"bucket": name, "samples": sum(layer["samples"][b] for layer in result["layers"])}
        for stat, d in result["drift"].items():
            rec[stat] = {"median_ratio": d["median_ratio"][b], "min_ratio": d["min_ratio"][b], "max_ratio": d["max_ratio"][b],
                         "flagged": d["flagged"][b]}
        by[str(hi)] = rec
    return {"rows": result["rows"], "ref_bucket": result["ref_bucket"], "train_len": result["train_len"], "layers": len(result["layers"]),
            "heads": sum(layer["heads"] for layer in result["layers"]),
            "long_heads": sum(layer["gate"]["long_heads"] or 0 for layer in result["layers"]), "by_length": by}


def compact(parts: dict, settings: dict) -> dict:
    """The per-checkpoint summary from the parts' outputs ({"bpb": {set: score_position_bpb's}, "agent":
    score_agent_windows', "multifact": {"summary": probe_gen_multifact.summarize's}, "gdn": {"natural": {set: GDNStats
    result}, "joined": {set: ...}}}; any part may be missing)."""
    base = int(settings.get("base_len", 8192))
    out = {"run": settings.get("run"), "settings": settings, "base_len": base}
    if parts.get("bpb"):
        out["bpb"] = {}
        for name, r in parts["bpb"].items():
            scored = [d for d in r["docs"] if d["n_scored"] > 0]
            out["bpb"][name] = {"docs": len(scored), "tokens": int(sum(d["n_scored"] for d in scored)), "ref": r["ref"],
                                "by_length": {str(row["hi"]): {k: row[k] for k in ("name", "docs", "tokens", "bpb", "bpb_se", "ref_bpb", "delta", "delta_se")}
                                              for row in r["all"]},
                                "cohorts": {c: {str(row["hi"]): {k: row[k] for k in ("name", "docs", "tokens", "bpb", "bpb_se", "delta", "delta_se")}
                                                for row in rows} for c, rows in r["cohorts"].items()}}
    if parts.get("agent"):
        r = parts["agent"]
        tail = {}
        for C, rec in r["tail_by_cohort"].items():
            ref = rec["by_window"].get(str(r["ref_window"]), {})
            tail[C] = {"trajectories": rec["trajectories"], "clusters": rec["clusters"], "tokens": rec["tokens"],
                       "ref_bpb": ref.get("bpb"), **{k: rec["by_window"][C][k] for k in ("bpb", "bpb_se", "delta", "delta_se")},
                       "by_window": rec["by_window"]}
        out["agent"] = {"trajectories": len(r["docs"]), "skipped_short": r["skipped_short"], "ref_window": r["ref_window"],
                        "tail_targets": r["tail"], "window": r["window"], "tail": tail}
    if parts.get("multifact") and parts["multifact"].get("summary"):
        out["multifact"] = {probe: {k: {L: {m: cell.get(m) for m in ("n", *MF_SHOWN, *[x + "_se" for x in MF_SHOWN], "hops", "layouts")}
                                        for L, cell in cells.items()} for k, cells in ks.items()}
                            for probe, ks in parts["multifact"]["summary"].items()}
        for ks in out["multifact"].values():  # layouts: only the headline metrics
            for cells in ks.values():
                for cell in cells.values():
                    cell["layouts"] = {name: {m: v.get(m) for m in ("n", "exact", "fact_acc", "nats")}
                                       for name, v in (cell.get("layouts") or {}).items()}
    if parts.get("gdn"):
        out["gdn"] = {}
        for kind in ("natural", "joined"):
            for name, r in (parts["gdn"].get(kind) or {}).items():
                if r and r.get("layers"):
                    out["gdn"][f"{kind}-{name}"] = _gdn_compact(r)
    return out


# --------------------------------------------------------------------------------------------
# the degradation rule
# --------------------------------------------------------------------------------------------


def _entry(metric: str, length: int, value, se, threshold: float, detail: str = "", kind: str = "difference") -> dict | None:
    """kind: "difference" (from the metric's <= 8k value) or "factor" (a ratio >= 1 to it: the GDN statistics)."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    degraded = value > threshold
    known = se is not None and se == se
    # clear: past the threshold by more than two standard errors; a difference without a standard error (one cluster)
    # is never clear, a factor has none and always is
    clear = degraded and (kind == "factor" or (known and value - 2 * se > threshold))
    return {"metric": metric, "length": int(length), "value": float(value), "se": float(se) if known else None,
            "threshold": threshold, "degraded": bool(degraded), "clear": bool(clear), "margin": float(value) / threshold,
            "kind": kind, "detail": detail}


def _root_sum(*ses) -> float | None:
    vals = [s for s in ses if s is not None and s == s]
    return math.sqrt(sum(s * s for s in vals)) if len(vals) == len(ses) and vals else None


def degradations(summary: dict, thresholds: dict | None = None) -> list[dict]:
    """Every (metric, length beyond the training length) of a checkpoint's summary with its value against its own
    <= 8k value and the threshold (module docstring), sorted by length then metric."""
    th = {**THRESHOLDS, **(thresholds or {})}
    base = int(summary.get("base_len", 8192))
    out = []
    for name, r in (summary.get("bpb") or {}).items():
        for hi, row in r["by_length"].items():
            if int(hi) > base and row["tokens"]:
                out.append(_entry(f"bpb.{name}", int(hi), row["delta"], row["delta_se"], th["bpb"],
                                  f"{row['docs']} docs, {row['tokens']} tokens"))
    for W, row in ((summary.get("agent") or {}).get("tail") or {}).items():
        if int(W) > base and row["trajectories"]:
            out.append(_entry("agent.tail", int(W), row["delta"], row["delta_se"], th["agent"], f"{row['trajectories']} trajectories"))
    for probe, ks in (summary.get("multifact") or {}).items():
        for k, cells in ks.items():
            lengths = sorted(int(L) for L in cells)
            if not lengths:
                continue
            # the reference: the items at the training length (the longest length inside it; the shortest length
            # scored when the set has none inside it)
            ref_len = max((L for L in lengths if L <= base), default=lengths[0])
            ref, at = cells[str(ref_len)], length_label(ref_len)
            for L in lengths:
                if L <= ref_len:
                    continue
                cell = cells[str(L)]
                tag = f"mf.{probe.removeprefix('mf_')}.{k.replace('=', '')}"
                if cell.get("nats") is not None and ref.get("nats") is not None:
                    out.append(_entry(f"{tag}.nats", L, cell["nats"] - ref["nats"], _root_sum(cell.get("nats_se"), ref.get("nats_se")),
                                      th["mf_nats"], f"{at}: {ref['nats']:.3f}"))
                if cell.get("exact") is not None and ref.get("exact") is not None:
                    out.append(_entry(f"{tag}.exact", L, ref["exact"] - cell["exact"], _root_sum(cell.get("exact_se"), ref.get("exact_se")),
                                      th["mf_exact"], f"{at}: {ref['exact']:.3f}"))
    for rows, r in (summary.get("gdn") or {}).items():
        for hi, rec in r["by_length"].items():
            if int(hi) <= base or not rec["samples"]:
                continue
            for stat in GDN_FLAG_STATS:
                lo, up = rec[stat]["min_ratio"], rec[stat]["max_ratio"]
                if lo is None or up is None or lo <= 0:
                    continue
                out.append(_entry(f"gdn.{rows}.{stat}", int(hi), max(up, 1.0 / lo), None, th["gdn"],
                                  f"ratio {lo:.2f}-{up:.2f} over the layers, {len(rec[stat]['flagged'])} outside", kind="factor"))
    return sorted((e for e in out if e is not None), key=lambda e: (e["length"], e["metric"]))


def breaks_first(entries: list[dict], clear_only: bool = False) -> dict:
    """{"length": the smallest length with a degraded metric (None: nothing degrades), "metrics": the metrics degraded
    there, largest margin (value / threshold) first, "by_length": {length: degraded metrics}}. clear_only: only the
    metrics that pass the threshold by more than two standard errors."""
    by: dict[int, list] = {}
    for e in entries:
        if e["degraded"] and (e["clear"] or not clear_only):
            by.setdefault(e["length"], []).append(e)
    for v in by.values():
        v.sort(key=lambda e: -e["margin"])
    first = min(by) if by else None
    return {"length": first, "metrics": [e["metric"] for e in by[first]] if by else [],
            "by_length": {str(L): [e["metric"] for e in v] for L, v in sorted(by.items())}}


# --------------------------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------------------------


def num(x, nd: int = 4, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def pm(x, se, nd: int = 4, sign: bool = False) -> str:
    return num(x, nd, sign) if se is None or se != se or x is None else f"{num(x, nd, sign)}+-{num(se, nd)}"


def table(header: list[str], rows: list[list]) -> str:
    """A markdown table."""
    line = lambda r: "| " + " | ".join(str(c) for c in r) + " |"  # noqa: E731
    return "\n".join([line(header), line(["---"] * len(header))] + [line(r) for r in rows])


def entry_value(e: dict, nd: int = 3) -> str:
    """A degradation entry's value as text: a signed difference +- SE, or a factor "x1.23"."""
    return f"x{e['value']:.2f}" if e.get("kind") == "factor" else pm(e["value"], e["se"], nd, sign=True)


def describe_settings(s: dict) -> str:
    scale = f"inference length scale on (L = {s.get('length_scale_ref')})" if s.get("length_scale") else "no inference length scale"
    dec = f", decoder at {s['decoder_at']}" if s.get("decoder_at") else ""
    if s.get("replay_window") is not None:  # scored_sinks_replay: the window, or the whole last turn and its cap
        dec += (f" (the last turn, at most {s.get('replay_turn_max')} positions)" if s["replay_window"] == "turn"
                else f" (window {s['replay_window']})")
    # --rope-clamp (scripts/eval_long.py): said only when set, so every summary written before it reads as it did
    clamp = f", rotary distance clamped at {s['rope_clamp']:,}" if s.get("rope_clamp") is not None else ""
    return (f"`{s.get('run')}`: {s.get('arch')}, attention temperature {s.get('temperature')}, {scale}{clamp}{dec}; checkpoint step "
            f"{s.get('step')}")


def render(summary: dict, entries: list[dict] | None = None, brief: bool = False) -> str:
    """summary.md of one checkpoint. brief (summary-tail.md, inside a 16 KiB log tail): without the cohort lines, the
    per-layout rows and the read / write columns, and with only the degraded rows of the last table."""
    s = summary["settings"]
    base = summary["base_len"]
    at = length_label(base)
    entries = degradations(summary) if entries is None else entries
    lines = ["# Long-context evaluation", "", describe_settings(s), ""]
    first = breaks_first(entries)
    clear = breaks_first(entries, clear_only=True)
    if first["length"] is None:
        lines += ["**Nothing passes its threshold** at the lengths scored.", ""]
    else:
        lines += [f"**Breaks first at {length_label(first['length'])}:** {', '.join(first['metrics'])}"
                  + (f" (clear of noise from {length_label(clear['length'])}: {', '.join(clear['metrics'])})" if clear["length"] is not None
                     else " (none clear of noise)"), ""]
    for name, r in (summary.get("bpb") or {}).items():
        ref = r.get("ref") or {}
        short = (f"the same targets scored with only the preceding {length_label(ref['window'] - ref['stride'])}-"
                 f"{length_label(ref['window'])} tokens") if ref else "no reference pass"
        lines += [f"## bpb by position: {name} ({r['docs']} documents, {r['tokens']} scored tokens)", "",
                  f"Whole documents, one per row. short context: {short}. delta = bpb - short-context bpb (negative: the "
                  "longer context helps).", ""]
        rows = [[row["name"], row["docs"], row["tokens"], pm(row["bpb"], row["bpb_se"]), num(row["ref_bpb"]),
                 pm(row["delta"], row["delta_se"], sign=True) if int(hi) > base else "-"] for hi, row in r["by_length"].items()]
        lines += [table(["positions", "docs", "tokens", "bpb", "short-context bpb", "delta"], rows), ""]
        if not brief:
            for c, rows_c in r["cohorts"].items():
                if not rows_c or not any(row["docs"] for row in rows_c.values()):
                    continue
                docs = max(row["docs"] for row in rows_c.values())
                lines += [f"Documents that reach {c} ({docs}): "
                          + "; ".join(f"{row['name']} {num(row['bpb'])}" + (f" ({num(row['delta'], sign=True)})" if int(hi) > base else "")
                                      for hi, row in rows_c.items()), ""]
    if summary.get("agent"):
        a = summary["agent"]
        lines += [f"## Assistant bpb on whole agent trajectories ({a['trajectories']} trajectories longer than {length_label(base)})", "",
                  "Window W: the row holds the trajectory's last W tokens before its final target (the whole trajectory when "
                  f"shorter). Tail: the last {a['tail_targets']} targets only, in the trajectories of at least W + 1 tokens.", ""]
        rows = []
        for W, w in a["window"].items():
            t = a["tail"].get(W, {})
            rows.append([length_label(int(W)), f"{w['trajectories']} ({w['filled']} fill it)", w["tokens"], pm(w["bpb"], w["bpb_se"]),
                         t.get("trajectories", 0), num(t.get("ref_bpb")), pm(t.get("bpb"), t.get("bpb_se")),
                         pm(t.get("delta"), t.get("delta_se"), sign=True) if int(W) > a["ref_window"] else "-"])
        lines += [table(["window", "trajectories", "assistant tokens", "window bpb", "tail: trajectories",
                         f"tail bpb at {length_label(a['ref_window'])}", "tail bpb at W", "delta"], rows), ""]
    for probe, ks in (summary.get("multifact") or {}).items():
        lines += [f"## Multi-fact: {probe}", ""]
        for k, cells in ks.items():
            lengths = sorted(cells, key=int)
            n = cells[lengths[0]]["n"] if lengths else 0
            rows = [[m] + [pm(cells[L].get(m), cells[L].get(m + "_se"), 3) for L in lengths] for m in MF_SHOWN]
            lines += [f"{k} ({n} items per length)", "", table(["metric"] + [length_label(int(L)) for L in lengths], rows), ""]
            if not brief:
                for layout in sorted({x for L in lengths for x in (cells[L].get("layouts") or {})}):
                    lines += [f"- {layout}: exact " + " / ".join(num((cells[L]["layouts"].get(layout) or {}).get("exact"), 3) for L in lengths)
                              + "; nats " + " / ".join(num((cells[L]["layouts"].get(layout) or {}).get("nats"), 3) for L in lengths)]
                lines += [""]
    for rows_name, r in (summary.get("gdn") or {}).items():
        lines += [f"## GDN state by position: {rows_name} (rows: {r['rows']}, GDN layers: {r['layers']})", "",
                  f"Ratio to the {r['ref_bucket']} bucket, median over the layers (min-max); the number of layers outside "
                  f"[1/1.5, 1.5] in brackets. {r['long_heads']} of {r['heads']} heads have a half-life beyond "
                  f"{length_label(r['train_len'])}.", ""]
        stats = ("norm", "erank", "rel_delta") if brief else ("norm", "erank", "rel_delta", "read", "write")
        rows = []
        for hi, rec in r["by_length"].items():
            if not rec["samples"]:
                continue
            rows.append([rec["bucket"]] + [f"{num(rec[st]['median_ratio'], 2)} ({num(rec[st]['min_ratio'], 2)}-{num(rec[st]['max_ratio'], 2)})"
                                           + (f" [{len(rec[st]['flagged'])}]" if rec[st]["flagged"] else "") for st in stats])
        lines += [table(["positions", *stats], rows), ""]
    lines += [f"## Against the <= {at} value", ""]
    shown = [e for e in entries if e["degraded"]] if brief else entries
    rows = [[e["metric"], length_label(e["length"]), entry_value(e), f"> {e['threshold']}",
             ("DEGRADED" + ("" if e["clear"] else " (within 2 SE)")) if e["degraded"] else "ok", e["detail"]] for e in shown]
    if not entries:
        lines += ["No metric was scored beyond the training length.", ""]
    else:
        if rows:
            lines += [table(["metric", "length", "value", "threshold", "", "detail"], rows), ""]
        ok = sum(not e["degraded"] for e in entries)
        lines += [f"{ok} of {len(entries)} (metric, length) pairs are within their thresholds"
                  + (" (not listed here: summary.md has them)." if brief else "."), ""]
    t = summary.get("timing")
    if t:
        lines += ["## Time", "", "; ".join(f"{k} {v:.0f} s" for k, v in t.items() if isinstance(v, (int, float))), ""]
    return "\n".join(lines).rstrip() + "\n"
