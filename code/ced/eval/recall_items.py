"""Larger multi-key item sets for the 200M models: the multi-key cells are rerun with enough items that the sampling error is small next
to the seed spread.

Pure functions, no torch, no data files: the cells, the plan (how many items a cell gets and which scoring group scores which items), the
time model, and the conversion of scripts/eval_probes.py outputs into the common values file of the recall builds. The builder is
tools/build_recall_items.py and the values converter is tools/analysis/recall_items_values.py.

THE CELLS (16). The multi-key needle in a haystack, N keys (4 and 16), one key asked, exact match of the answer (every answer token the argmax under teacher
forcing; scripts/eval_probes.py):
  the 10 multi-key cells, probe "multikey" (ced/eval/probe_gen.py multikey_item) at 8k, 16k, 32k, 64k, 128k (the ten cells of the retrieval comparison): 300 items a cell at 8k to
      64k (the probes-v1 set) and 100 at 128k (the 128k set);
  the pcr_niahk cells, probe "pcr_niahk" (ced/eval/report_pcr_gen.py niahk_item) at 2,048, 4,096 and 8,192 tokens: 200 items a cell
      (the report-pcr set).
An item is a pure function of (seed, probe, cell, index) and the haystack pool, so a larger set is the old set with items appended: the cell's items
0 .. n_old - 1 are the ones already scored, n_old .. n - 1 are new (tools/build_recall_items.py checks the first block token for token against the old files).

THE COMMON FORMAT: one file <run>.values.json = {"run", "family": "items", "code_sha", "cells": {"items|<probe>|<cell>|exact": accuracy},
"n_items": {the same keys: count}, "excluded": ["<probe>:<item id>", ...]}; <probe> is multikey_n4, multikey_n16, pcr_niahk_n4 or pcr_niahk_n16 and <cell> the length in tokens.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

FAMILY = "items"
LEG = "multikey"
PCRK = "pcr_niahk"
PROBES = (LEG, PCRK)
N_KEYS = (4, 16)
LEG_LENGTHS = (8192, 16384, 32768, 65536, 131072)
PCRK_LENGTHS = (2048, 4096, 8192)
TRAIN_LEN = 8192  # the 200M configs' seq_len
OLD_N_LEG = 300  # items a cell of the probes-v1 multikey.pt (8k to 64k)
OLD_N_LEG_128K = 100  # items a cell of the 128k set's multikey.pt
OLD_N_PCRK = 200  # items a cell of the report-pcr set's pcr_niahk.pt
SOURCES = ("mk", "mk128", "pcrk")  # the three builds: multikey at 8k to 64k, multikey at 128k, the pcr_niahk cells
ID_RE = re.compile(r"^(multikey|pcr_niahk)/len=(\d+)/n=(\d+)/(\d{4,})$")

# ---- the time model: seconds a 200M model needs per item on one GPU slice of about 24 GB, forward only, scripts/eval_probes.py (a model, not a measurement of the run itself)
# 8k to 64k: per-row seconds of the multi-fact rows of the long-context suite (the same forward on single rows of that length), taken from the slowest of the 200M
# models: 0.135, 0.248, 0.549, 1.390 s a row at 8k, 16k, 32k, 64k (another 200M model: 0.102, 0.211, 0.495, 1.312).
# 128k: 660 rows in 3,085.9 s = 4.68 s a row (28,105 tokens/s after the first); other 200M models: 4.34 to 4.60.
SECONDS_PER_ITEM = {8192: 0.135, 16384: 0.248, 32768: 0.549, 65536: 1.390, 131072: 4.70}
# pcr_niahk: the items of at most 4,096 tokens are packed whole into 8,192-token rows (eval_probes --pack-max-item 4096): 4 items of 2,048 or 2 of 4,096 a row; a packed row
# of a 200M model took 8,192 / 33,386 = 0.245 s (copy rows included, so this is an upper bound for needle rows); an 8,192-token item is a single row
PACKED_ROW_SECONDS = 0.25
PACKED_ITEMS_PER_ROW = {2048: 4, 4096: 2}
START_SECONDS = 90.0  # a scoring process's start: model load, compiled layers (load 5 to 30 s, first batch up to 39 s)
SHAPE_SECONDS = 30.0  # the first batch of a row shape (compile): 8k single, 8k packed, 16k, 32k, 64k, 128k
GROUP_BUDGET_S = 3000.0  # a group's modelled seconds, start and shapes included, so that its deadline of at most 3,600 s has 20 % headroom
DEADLINE_MAX_S = 3600
DEADLINE_STEP_S = 300
DEADLINE_FACTOR = 1.2
SPLIT_STEP = 50  # a cell is cut between groups at a multiple of this many items


@dataclass(frozen=True)
class Cell:
    probe: str  # multikey | pcr_niahk
    n_keys: int
    length: int

    @property
    def key(self) -> str:
        """"multikey_n4|8192": the cell as the plan names it (the middle of the common format's cell key)."""
        return f"{self.probe}_n{self.n_keys}|{self.length}"

    @property
    def value_key(self) -> str:
        return f"{FAMILY}|{self.probe}_n{self.n_keys}|{self.length}|exact"

    @property
    def source(self) -> str:
        if self.probe == PCRK:
            return "pcrk"
        return "mk128" if self.length >= 131072 else "mk"

    @property
    def old_n(self) -> int:
        if self.probe == PCRK:
            return OLD_N_PCRK
        return OLD_N_LEG_128K if self.length >= 131072 else OLD_N_LEG

    @property
    def item_prefix(self) -> str:
        return f"{self.probe}/len={self.length}/n={self.n_keys}/"


def all_cells() -> list[Cell]:
    """The 16 cells in a fixed order: multikey by n_keys then length, then pcr_niahk by n_keys then length."""
    return [Cell(LEG, n, L) for n in N_KEYS for L in LEG_LENGTHS] + [Cell(PCRK, n, L) for n in N_KEYS for L in PCRK_LENGTHS]


CELLS = {c.key: c for c in all_cells()}


def parse_item_id(item_id: str) -> tuple[Cell, int]:
    """("multikey/len=8192/n=4/0012") -> (Cell(multikey, 4, 8192), 12). ValueError for any other id or a cell outside the 16."""
    m = ID_RE.match(item_id)
    if not m:
        raise ValueError(f"not a multi-key item id: {item_id!r}")
    cell = Cell(m.group(1), int(m.group(3)), int(m.group(2)))
    if cell.key not in CELLS:
        raise ValueError(f"{item_id!r}: {cell.key} is not one of the 16 cells")
    return cell, int(m.group(4))


# ------------------------------------------------------------------------------------------------------------------------------
# the time model


def item_seconds(cell: Cell) -> float:
    if cell.probe == PCRK and cell.length in PACKED_ITEMS_PER_ROW:
        return PACKED_ROW_SECONDS / PACKED_ITEMS_PER_ROW[cell.length]
    return SECONDS_PER_ITEM[cell.length]


def row_shape(cell: Cell) -> str:
    """The row shape a cell's items run in: packed 8k rows for pcr_niahk at 2k and 4k, else one row of the cell's length."""
    return "8192p" if (cell.probe == PCRK and cell.length in PACKED_ITEMS_PER_ROW) else str(cell.length)


def parts_seconds(parts: list["Part"]) -> float:
    """A group's modelled seconds: the start, one compile per distinct row shape, and the items."""
    shapes = {row_shape(CELLS[p.cell]) for p in parts if p.stop > p.start}
    # math.fsum: the same sum under Python 3.11 and 3.12 (whose sum() compensates), so that the plan file is the same under both
    return START_SECONDS + SHAPE_SECONDS * len(shapes) + math.fsum((p.stop - p.start) * item_seconds(CELLS[p.cell]) for p in parts)


def deadline_for(modelled_s: float) -> int:
    """A group's deadline: 1.2 x the model rounded up to 300 s, at most 3,600 s (the slice-hour bound is deadline / 3,600)."""
    d = int(math.ceil(DEADLINE_FACTOR * modelled_s / DEADLINE_STEP_S) * DEADLINE_STEP_S)
    return min(DEADLINE_MAX_S, max(DEADLINE_STEP_S, d))


# ------------------------------------------------------------------------------------------------------------------------------
# the plan: items a cell gets, and which group scores which items


@dataclass(frozen=True)
class Part:
    cell: str  # Cell.key
    start: int  # item indices [start, stop) of the cell
    stop: int

    def to_json(self) -> dict:
        return {"cell": self.cell, "start": self.start, "stop": self.stop}


@dataclass
class Group:
    name: str
    parts: list[Part]

    @property
    def modelled_s(self) -> float:
        return parts_seconds(self.parts)

    @property
    def deadline_s(self) -> int:
        return deadline_for(self.modelled_s)

    @property
    def items(self) -> int:
        return sum(p.stop - p.start for p in self.parts)

    def to_json(self) -> dict:
        return {"name": self.name, "modelled_s": round(self.modelled_s, 1), "deadline_s": self.deadline_s, "items": self.items,
                "parts": [p.to_json() for p in self.parts]}


def assign_groups(cell_n: dict[str, int], budget_s: float = GROUP_BUDGET_S, prefix: str = "g") -> list[Group]:
    """Cut the work (every cell's items 0 .. n - 1) into groups of at most `budget_s` modelled seconds each, the dearest cells first, a cell cut at a multiple of
    SPLIT_STEP items when it does not fit. Deterministic: the same plan always gives the same groups, which the builder and the scoring runs both rely on.
    ValueError when a single item cannot fit (budget below start + shape + one item)."""
    order = sorted((CELLS[k] for k in cell_n), key=lambda c: (-item_seconds(c), c.probe, c.n_keys, c.length))
    todo = [(c, 0, cell_n[c.key]) for c in order if cell_n[c.key] > 0]
    groups: list[Group] = []
    cur: list[Part] = []
    while todo:
        c, a, b = todo.pop(0)
        trial = cur + [Part(c.key, a, b)]
        if parts_seconds(trial) <= budget_s:
            cur = trial
            continue
        # the largest prefix of [a, b) that fits, cut at a multiple of SPLIT_STEP from the cell's start
        fixed = parts_seconds(cur + [Part(c.key, a, a + 1)]) - item_seconds(c)  # the start, the shapes (this cell's included) and the items already in the group
        k = int((budget_s - fixed) // item_seconds(c))
        stop = a + (k // SPLIT_STEP) * SPLIT_STEP if k >= SPLIT_STEP else a
        stop = min(stop, b)
        if stop > a:
            cur.append(Part(c.key, a, stop))
            todo.insert(0, (c, stop, b))
            groups.append(Group(f"{prefix}{len(groups) + 1}", cur))
            cur = []
        elif cur:
            groups.append(Group(f"{prefix}{len(groups) + 1}", cur))
            cur = []
            todo.insert(0, (c, a, b))
        else:
            raise ValueError(f"cell {c.key}: not even {SPLIT_STEP} items fit a group of {budget_s:.0f} s")
    if cur:
        groups.append(Group(f"{prefix}{len(groups) + 1}", cur))
    return groups


def make_plan(cell_n: dict[str, int], rule: str = "", extra: dict | None = None, smoke: bool = False) -> dict:
    """The plan file's content: {"version", "rule", "cells": [{key, probe, n_keys, length, source, n_old, n}], "groups": [...], "per_final": {...}}.
    smoke=True: a plan of a few items a cell (a smoke test's), exempt from the rule that a cell keeps its old items as the first block; "smoke": true is in the file."""
    for k, n in cell_n.items():
        if k not in CELLS:
            raise ValueError(f"unknown cell {k}")
        if n < CELLS[k].old_n and not smoke:
            raise ValueError(f"{k}: {n} items is fewer than the {CELLS[k].old_n} already scored (the old items are the first block)")
    groups = assign_groups(cell_n)
    plan = {
        "version": 1, "rule": rule, **({"smoke": True} if smoke else {}),
        "cells": [{"key": c.key, "probe": c.probe, "n_keys": c.n_keys, "length": c.length, "source": c.source, "n_old": c.old_n, "n": cell_n[c.key]}
                  for c in all_cells() if c.key in cell_n],
        "groups": [g.to_json() for g in groups],
        "per_final": {"modelled_s": round(math.fsum(g.modelled_s for g in groups), 1), "groups": len(groups),
                      "slice_hours_modelled": round(math.fsum(g.modelled_s for g in groups) / 3600, 3),
                      "slice_hours_bound": round(math.fsum(g.deadline_s for g in groups) / 3600, 3)},
    }
    if extra:
        plan.update(extra)
    return plan


def load_plan(path: str | Path) -> dict:
    plan = json.loads(Path(path).read_text())
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> None:
    """AssertionError unless: the cells are known and keep the old items as the first block; every item index of every cell is in exactly one group part;
    a group is within the budget and its deadline is the model's."""
    n_of = {}
    for c in plan["cells"]:
        cell = CELLS[c["key"]]
        assert (c["probe"], c["n_keys"], c["length"], c["source"], c["n_old"]) == (cell.probe, cell.n_keys, cell.length, cell.source, cell.old_n), c["key"]
        assert c["n"] >= c["n_old"] or plan.get("smoke"), f"{c['key']}: n {c['n']} below the first block {c['n_old']}"
        n_of[c["key"]] = c["n"]
    seen: dict[str, list[tuple[int, int]]] = {}
    for g in plan["groups"]:
        parts = [Part(p["cell"], p["start"], p["stop"]) for p in g["parts"]]
        assert 0 < len(parts), g["name"]
        assert parts_seconds(parts) <= GROUP_BUDGET_S + 1e-6, f"{g['name']}: {parts_seconds(parts):.0f} s is over the budget"
        assert g["deadline_s"] == deadline_for(parts_seconds(parts)) and g["deadline_s"] <= DEADLINE_MAX_S, g["name"]
        for p in parts:
            assert p.cell in n_of and 0 <= p.start < p.stop <= n_of[p.cell], (g["name"], p)
            seen.setdefault(p.cell, []).append((p.start, p.stop))
    for key, n in n_of.items():
        spans = sorted(seen.get(key, []))
        pos = 0
        for a, b in spans:
            assert a == pos, f"{key}: items {pos}..{a} are in no group (or two)"
            pos = b
        assert pos == n, f"{key}: groups cover {pos} of {n} items"


# ------------------------------------------------------------------------------------------------------------------------------
# results -> the common values file


def result_exact(r: dict) -> bool:
    """scripts/eval_probes.py's per-item exact: every answer token is the argmax (ced.eval.probes.item_metrics)."""
    return int(r["n_correct"]) == int(r["n_answer"])


def exclusion_name(item_id: str) -> str:
    """"multikey_n4:multikey/len=8192/n=4/0012": the 'task:id' string of an exclude file (the task is the cell's probe and key count, as in the value keys)."""
    cell, _ = parse_item_id(item_id)
    return f"{cell.probe}_n{cell.n_keys}:{item_id}"


def canonical_exclusion(name: str) -> str:
    """An exclude-file string in any of its plausible spellings -> the canonical 'multikey_n4:multikey/len=8192/n=4/0012'. Accepted: the canonical string, 'multikey:<id>' (the probe
    name the item texts carry, which is how a scan of the texts would write it), 'pcr_niahk:<id>', or the bare item id. The item id is what identifies the item; the part before the first
    ':' is not trusted. ValueError for a string that holds no item id of the 16 cells: an exclusion that matched nothing would be a silent no-op."""
    item_id = name.split(":", 1)[1] if ":" in name else name
    return exclusion_name(item_id)


def values_for_run(run: str, plan: dict, group_items: dict[str, list[dict]], code_sha: str = "", exclude: set[str] | None = None,
                   allow_partial: bool = False, first_block_only: bool = False) -> tuple[dict, dict]:
    """(the values file, the detail) of one run. group_items: {group name: eval_probes "items" list}. Every item of the plan must be scored exactly once
    (ValueError otherwise, unless allow_partial: then the counts are what was scored, and the detail says which items are missing). exclude: 'task:id'
    strings left out of every cell and listed in "excluded". first_block_only: the cells over the old items only (index < n_old), the subset the earlier
    readings had. The detail: per cell the counts and accuracies of the first block, the new block and all, and the items that are missing or twice."""
    want: dict[str, int] = {c["key"]: c["n"] for c in plan["cells"]}
    exclude = {canonical_exclusion(s) for s in (exclude or set())}
    if exclude:  # a flagged string that names no item of this plan is an error, not a silent no-op
        known = {exclusion_name(f"{CELLS[k].item_prefix}{i:04d}") for k, n in want.items() for i in range(n)}
        unknown = sorted(exclude - known)
        if unknown:
            raise ValueError(f"{run}: exclusions that name no item of the plan: {unknown[:5]} ({len(unknown)} in all)")
    got: dict[str, dict[int, bool]] = {k: {} for k in want}
    twice: list[str] = []
    stray: list[str] = []
    for gname, items in group_items.items():
        for r in items:
            try:
                cell, idx = parse_item_id(r["id"])
            except ValueError:
                stray.append(r.get("id", "?"))
                continue
            if cell.key not in want or idx >= want[cell.key]:
                stray.append(r["id"])
                continue
            if idx in got[cell.key]:
                twice.append(r["id"])
            got[cell.key][idx] = result_exact(r)
    if stray or twice:
        raise ValueError(f"{run}: items outside the plan {stray[:5]} / scored twice {twice[:5]}")
    missing = {k: [i for i in range(n) if i not in got[k]] for k, n in want.items()}
    if any(missing.values()) and not allow_partial:
        short = {k: len(v) for k, v in missing.items() if v}
        raise ValueError(f"{run}: items of the plan are not scored: {short}")
    cells, counts, excluded, detail = {}, {}, [], {}
    for key, n in want.items():
        cell = CELLS[key]
        old = cell.old_n
        keep = []
        for i, ok in sorted(got[key].items()):
            name = exclusion_name(f"{cell.item_prefix}{i:04d}")
            if name in exclude:
                excluded.append(name)
                continue
            if first_block_only and i >= old:
                continue
            keep.append((i, ok))
        if not keep:
            continue
        cells[cell.value_key] = sum(ok for _, ok in keep) / len(keep)
        counts[cell.value_key] = len(keep)
        first = [ok for i, ok in keep if i < old]
        new = [ok for i, ok in keep if i >= old]
        detail[key] = {"n": len(keep), "exact": cells[cell.value_key], "n_first_block": len(first),
                       "exact_first_block": (sum(first) / len(first)) if first else None, "n_new_block": len(new),
                       "exact_new_block": (sum(new) / len(new)) if new else None, "missing": len(missing[key])}
    values = {"run": run, "family": FAMILY, "code_sha": code_sha, "cells": cells, "n_items": counts, "excluded": sorted(set(excluded))}
    return values, detail


def binomial_se(p: float, n: int) -> float:
    return math.sqrt(p * (1.0 - p) / n)


def binomial_se_from_v(v: float, n: int) -> float:
    """sqrt(v / n) with v the mean of p (1 - p) over runs."""
    return math.sqrt(v / n)
