#!/usr/bin/env python3
"""The values file of the larger multi-key item sets, in the common format of the recall builds (family "items"), from the per-group outputs of scripts/eval_probes.py.

    python tools/analysis/recall_items_values.py run --plan configs/eval/recall-items-plan.json --run f-60m-s1 --probes-dir DIR --out DIR2/f-60m-s1.values.json
        [--code-sha SHA] [--exclude-file FILE] [--allow-partial] [--first-block-only]
    python tools/analysis/recall_items_values.py brief --values FILE
    python tools/analysis/recall_items_values.py target-check --values-dir DIR --pairs run-s1:run-s2,other-s1:other-s2 [--k 4] [--out DIR2]

DIR holds <group>.json for every group of the plan, each the output of `scripts/eval_probes.py --probes <items>/groups/<group>/t64k` (see EVAL.md). Python 3.11, standard library and
ced/eval/recall_items.py (no torch). Written: the values file {"run", "family": "items", "code_sha", "cells": {"items|<probe>|<cell>|exact": accuracy}, "n_items": {...}, "excluded": [...]}
and, beside it, <out stem>.detail.json (per cell: the first block of old items and the new block apart, items missing).

--exclude-file: a JSON list of "<probe>:<item id>" strings; those items are left out of every cell and listed in "excluded"
(canonical spelling "multikey_n4:multikey/len=8192/n=4/0012"; "multikey:<id>", "pcr_niahk:<id>" and the bare item id name the same item, since the item id is what identifies it; a string that names
no item of the plan is an error, exit 3, not a silent no-op).
--first-block-only: the cells over the first block of items only (index below n_old, the items of the earlier, smaller sets), recomputed on the same scoring pass.
Exit: 0 written; 2 bad input; 3 a group file missing, an item of the plan not scored (without --allow-partial), or an exclusion that names no item of the plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ced.eval import recall_items as RI  # noqa: E402


def load_group_items(probes_dir: Path, plan: dict, allow_partial: bool) -> dict[str, list[dict]]:
    out = {}
    for g in plan["groups"]:
        path = probes_dir / f"{g['name']}.json"
        if not path.exists():
            if allow_partial:
                continue
            raise FileNotFoundError(f"{path}: group {g['name']} has no output")
        out[g["name"]] = json.loads(path.read_text())["items"]
    return out


def cmd_run(a) -> int:
    plan = RI.load_plan(a.plan)
    exclude = set(json.loads(Path(a.exclude_file).read_text())) if a.exclude_file else set()
    try:
        groups = load_group_items(Path(a.probes_dir), plan, a.allow_partial)
        values, detail = RI.values_for_run(a.run, plan, groups, code_sha=a.code_sha, exclude=exclude, allow_partial=a.allow_partial,
                                           first_block_only=a.first_block_only)
    except (FileNotFoundError, ValueError) as e:
        print(f"recall_items_values: {e}", file=sys.stderr)
        return 3
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(values, indent=1, sort_keys=True) + "\n")
    stem = out.name[: -len(".values.json")] if out.name.endswith(".values.json") else out.stem
    (out.parent / f"{stem}.detail.json").write_text(json.dumps({"run": a.run, "plan_rule": plan.get("rule", ""), "cells": detail}, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(values['cells'])} cells, {len(values['excluded'])} items excluded)")
    return 0


def cmd_brief(a) -> int:
    v = json.loads(Path(a.values).read_text())
    print(f"== {v['run']} (family {v['family']}, code {v.get('code_sha', '')})")
    for key in sorted(v["cells"], key=lambda k: (k.split("|")[1], int(k.split("|")[2]))):
        n = v["n_items"][key]
        p = v["cells"][key]
        print(f"  {key:44s} {p:.4f}  n={n:5d}  se={RI.binomial_se(p, n):.4f}")
    return 0


def target_check(values: dict[str, dict], pairs: list[tuple[str, str]], k: int = 4) -> list[dict]:
    """The target of the rerun, read on the new values: per cell, the observed between-seed SD of the pairs (the sample SD of two seeds is |x1 - x2| / sqrt 2, pooled over the pairs: df = the
    number of pairs; it contains the cell's sampling variance), the binomial sampling SE sqrt(v / n) of a run's value (v = the mean of p (1 - p) over the runs of the pairs, n = the cell's items),
    their ratio, and whether SE <= SD / k. Information only: nothing is ruled or adopted by this table."""
    rows = []
    keys = sorted(next(iter(values.values()))["cells"], key=lambda c: (c.split("|")[1], int(c.split("|")[2])))
    for key in keys:
        d2 = [(values[a]["cells"][key] - values[b]["cells"][key]) ** 2 / 2 for a, b in pairs if key in values[a]["cells"] and key in values[b]["cells"]]
        if not d2:
            continue
        runs = {r for pr in pairs for r in pr if key in values[r]["cells"]}
        v = sum(values[r]["cells"][key] * (1 - values[r]["cells"][key]) for r in runs) / len(runs)
        n = min(values[r]["n_items"][key] for r in runs)
        sd = (sum(d2) / len(d2)) ** 0.5
        se = RI.binomial_se_from_v(v, n)
        rows.append({"cell": key, "n": n, "df": len(d2), "seed_sd_observed": sd, "se": se, "ratio": (se / sd) if sd > 0 else float("inf"), "ok": se <= sd / k})
    return rows


def values_table(values: dict[str, dict]) -> str:
    runs = sorted(values)
    keys = sorted({k for v in values.values() for k in v["cells"]}, key=lambda c: (c.split("|")[1], int(c.split("|")[2])))
    out = ["| cell | " + " | ".join(runs) + " |\n", "|---|" + "---|" * len(runs) + "\n"]
    for k in keys:
        out.append(f"| {k} | " + " | ".join((f"{v['cells'][k]:.4f} (n={v['n_items'][k]})" if k in v["cells"] else "-") for v in (values[r] for r in runs)) + " |\n")
    return "".join(out)


def render_target_check(values: dict[str, dict], pairs: list[tuple[str, str]], k: int) -> str:
    out = [f"target check, SE <= observed seed SD / {k}; pairs {pairs} (df {len(pairs)}); information only\n", f"{'cell':44s} {'n':>5s} {'seed SD':>8s} {'SE':>7s} {'SE/SD':>6s}  ok\n"]
    for r in target_check(values, pairs, k):
        out.append(f"{r['cell']:44s} {r['n']:5d} {r['seed_sd_observed']:8.4f} {r['se']:7.4f} {r['ratio']:6.2f}  {'yes' if r['ok'] else 'no'}\n")
    return "".join(out)


def read_values_dir(values_dir: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    values, details = {}, {}
    for path in sorted(values_dir.glob("*.values.json")):
        v = json.loads(path.read_text())
        values[v["run"]] = v
        detail = path.with_name(path.name[: -len(".values.json")] + ".detail.json")
        if detail.exists():
            details[v["run"]] = json.loads(detail.read_text())["cells"]
    return values, details


def cmd_target_check(a) -> int:
    values, _ = read_values_dir(Path(a.values_dir))
    pairs = [tuple(x.split(":")) for x in a.pairs.split(",")]
    missing = [r for pr in pairs for r in pr if r not in values]
    if missing:
        print(f"recall_items_values: no values file for {missing}", file=sys.stderr)
        return 3
    text = render_target_check(values, pairs, a.k)
    text += "\nthe values of every run (accuracy, items):\n\n" + values_table(values)
    print(text)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.md").write_text(text)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("target-check")
    t.add_argument("--values-dir", required=True)
    t.add_argument("--pairs", required=True, help="seed pairs, run1:run2,run3:run4")
    t.add_argument("--k", type=int, default=4)
    t.add_argument("--out", default=None, help="also write OUT/summary.md")
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True)
    r.add_argument("--run", required=True)
    r.add_argument("--probes-dir", required=True)
    r.add_argument("--code-sha", default="")
    r.add_argument("--out", required=True)
    r.add_argument("--exclude-file", default=None)
    r.add_argument("--allow-partial", action="store_true")
    r.add_argument("--first-block-only", action="store_true")
    b = sub.add_parser("brief")
    b.add_argument("--values", required=True)
    a = p.parse_args(argv)
    return {"run": cmd_run, "brief": cmd_brief, "target-check": cmd_target_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
