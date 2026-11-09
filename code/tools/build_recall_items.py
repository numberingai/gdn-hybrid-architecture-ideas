#!/usr/bin/env python3
"""Build the larger multi-key item sets of the 200M retrieval rerun: the multi-key cells with enough items that the sampling error is small next to the seed spread.

    python tools/build_recall_items.py build --tokenizer-t64k <tokenizer.json> --docs <held-out docs>/web.docs.jsonl.gz --docs <held-out docs>/general_web.docs.jsonl.gz \\
        --out <items> [--unit mk|mk128|pcrk] [--plan configs/eval/recall-items-plan.json] [--tokenizer-t128k <other tokenizer.json>] [--old-root <dir>] [--workers 4]
    python tools/build_recall_items.py split --out <items> [--plan ...]        the groups' directories, from the raw builds
    python tools/build_recall_items.py check-first-block --old OLD.pt --new NEW.pt
    (--dry-run on build prints the commands and writes nothing)

CPU only; reads the held-out haystack documents (the web and general_web eval sets' docs files: held out, never in a training shard) and the tokenizers.
Three raw builds, each with the existing builder and the arguments the old set was built with, so that item i of a cell is the item it always was:
  mk     tools/build_probes.py --probes multikey, 8k to 64k, with BOTH length tokenizers of probes-v1 (t128k and t64k: the length measure of the old set is the larger of the two
         counts), --multikey-per-cell the plan's largest n of these cells; raw/mk/t64k/multikey.pt. Without --tokenizer-t128k the build uses t64k alone, and its items then
         differ from those of a two-tokenizer build.
  mk128  the same at 131,072 tokens with t64k alone (the 128k set); raw/mk128/t64k/multikey.pt
  pcrk   tools/build_report_pcr.py --probes pcr_niahk at 2,048, 4,096 and 8,192 tokens (t64k alone); raw/pcrk/t64k/pcr_niahk.pt
THE FIRST BLOCK. The generators are pure functions of (probe, cell, item index) and the pool, so the old items are the first n_old items of a cell. With --old-root <dir>
(holding probes-v1/t64k/multikey.pt, probes-lc-128k/t64k/multikey.pt, report-pcr/t64k/pcr_niahk.pt and the manifests probes-v1/manifest.json, report-pcr/manifest.json) the build does
not trust that: every old item that the new set holds must have the same id and the same token ids (EncodedSet.fingerprint); a difference fails the build (exit 4) and no group is
written. The haystack files' and tokenizers' sha256 must equal the old manifests' (probes-v1's manifest.json). Without --old-root (or with --no-check) these checks are skipped.
`split` then cuts the raw sets into the plan's groups, each a directory <out>/groups/<g>/t64k/ that scripts/eval_probes.py scores as it is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ced.eval import recall_items as RI  # noqa: E402

EX_CHECK = 4
DEFAULT_PLAN = ROOT / "configs" / "eval" / "recall-items-plan.json"
# the old sets, relative to --old-root
OLD_REL = {"mk": "probes-v1/t64k/multikey.pt", "mk128": "probes-lc-128k/t64k/multikey.pt", "pcrk": "report-pcr/t64k/pcr_niahk.pt"}
OLD_MANIFEST_REL = {"mk": "probes-v1/manifest.json", "pcrk": "report-pcr/manifest.json"}
RAW_FILE = {"mk": ("multikey", "multikey.pt"), "mk128": ("multikey", "multikey.pt"), "pcrk": ("pcr_niahk", "pcr_niahk.pt")}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def unit_cells(plan: dict, unit: str) -> list[dict]:
    return [c for c in plan["cells"] if c["source"] == unit]


def unit_n(plan: dict, unit: str) -> int:
    """Items a cell of the raw build needs: the largest n of its cells in the plan (the groups then take each cell's prefix)."""
    cells = unit_cells(plan, unit)
    return max((c["n"] for c in cells), default=0)


def unit_command(unit: str, plan: dict, a) -> list[str]:
    """The command line of a raw build (a list; run with cwd = the repository root)."""
    out = Path(a.out) / "raw" / unit
    docs_args = [x for d in a.docs for x in ("--docs", d)]
    t64k = a.tokenizer_t64k
    n = unit_n(plan, unit)
    if n <= 0:
        raise ValueError(f"the plan has no cell for the build unit {unit}")
    py = a.python
    if unit == "mk":
        lengths = sorted({c["length"] for c in unit_cells(plan, "mk")})
        tokenizers = (["--tokenizer", f"t128k={a.tokenizer_t128k}"] if a.tokenizer_t128k else []) + ["--tokenizer", f"t64k={t64k}"]
        return [py, "tools/build_probes.py", "--out", str(out), *tokenizers, *docs_args,
                "--probes", "multikey", "--lengths", ",".join(map(str, lengths)), "--multikey-per-cell", str(n), "--workers", str(a.workers)]
    if unit == "mk128":
        return [py, "tools/build_probes.py", "--out", str(out), "--tokenizer", f"t64k={t64k}", *docs_args,
                "--probes", "multikey", "--lengths", "131072", "--multikey-per-cell", str(n), "--workers", str(a.workers)]
    lengths = sorted({c["length"] for c in unit_cells(plan, "pcrk")})
    cmd = [py, "tools/build_report_pcr.py", "--out", str(out), "--tokenizer", f"t64k={t64k}", *docs_args,
           "--probes", "pcr_niahk", "--lengths", ",".join(map(str, lengths)), "--per-cell", str(n), "--workers", str(a.workers)]
    if a.old_root and ((Path(a.old_root) / OLD_MANIFEST_REL["mk"]).exists() or a.dry_run):
        cmd += ["--same-pool-as", str(Path(a.old_root) / OLD_MANIFEST_REL["mk"])]
    return cmd


def check_pool_and_tokenizers(unit: str, raw_dir: Path, root: Path) -> list[str]:
    """Problems (strings) between the raw build's manifest and probes-v1's: the haystack files (the pool of every old set), and for the 8k-64k build the tokenizers and the length
    measure. Empty when the old manifest says the same."""
    new_path = raw_dir / "manifest.json"
    if not new_path.exists():
        return [f"{new_path}: no manifest"]
    new = json.loads(new_path.read_text())
    ref_path = root / OLD_MANIFEST_REL["mk"]
    if not ref_path.exists():
        return [f"{ref_path}: the old manifest is not there, the pool cannot be compared"]
    ref = json.loads(ref_path.read_text())
    problems = []
    if ref.get("haystack_pool", {}).get("files") != new.get("haystack_pool", {}).get("files"):
        problems.append(f"haystack files differ: old {ref.get('haystack_pool', {}).get('files')} new {new.get('haystack_pool', {}).get('files')}")
    if unit == "mk":
        old_tok = {n: t.get("sha256") for n, t in ref.get("tokenizers", {}).items()}
        new_tok = {n: t.get("sha256") for n, t in new.get("tokenizers", {}).items()}
        if old_tok != new_tok:
            problems.append(f"tokenizers differ: old {old_tok} new {new_tok}")
        if ref.get("length_measure") != new.get("length_measure"):
            problems.append(f"length measure differs: old {ref.get('length_measure')!r} new {new.get('length_measure')!r}")
    return problems


def check_first_block(old_pt: str | Path, new_pt: str | Path) -> dict:
    """Every item of the old EncodedSet that the new one holds has the same id and the same token ids. Returns {"old", "new", "compared", "different": [ids], "missing": [ids]}.
    An old item the new set does not hold counts as missing (a smoke build holds fewer items on purpose: the caller decides)."""
    from ced.eval.probes import EncodedSet  # torch

    old, new = EncodedSet.load(old_pt), EncodedSet.load(new_pt)
    pos = {it["id"]: i for i, it in enumerate(new.items)}
    different, missing, compared = [], [], 0
    for i, it in enumerate(old.items):
        j = pos.get(it["id"])
        if j is None:
            missing.append(it["id"])
            continue
        compared += 1
        if old.fingerprint(i) != new.fingerprint(j):
            different.append(it["id"])
    return {"old": len(old), "new": len(new), "compared": compared, "different": different, "missing": missing}


def run(cmd: list[str], dry: bool) -> int:
    print("+ " + " ".join(cmd), flush=True)
    if dry:
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


def cmd_build(a) -> int:
    plan = json.loads(Path(a.plan).read_text())
    RI.validate_plan(plan)
    units = [a.unit] if a.unit else [u for u in RI.SOURCES if unit_cells(plan, u)]
    out = Path(a.out)
    root = Path(a.old_root) if a.old_root else None
    check = root is not None and not a.no_check  # the first-block and pool checks need the old sets
    for unit in units:
        raw = out / "raw" / unit
        done = raw / "build.done"
        if done.exists() and not a.force:
            print(f"[recall-items] {unit}: built ({done})", flush=True)
            continue
        t0 = time.time()
        if run(unit_command(unit, plan, a), a.dry_run) != 0:
            print(f"[recall-items] {unit}: the builder failed", flush=True)
            return 1
        if a.dry_run:
            continue
        problems = check_pool_and_tokenizers(unit, raw, root) if check else []
        old_pt = (root / OLD_REL[unit]) if check else None
        if check and not problems and old_pt.exists():
            res = check_first_block(old_pt, raw / "t64k" / RAW_FILE[unit][1])
            print(f"[recall-items] {unit}: first block {res['compared']} of {res['old']} old items compared, {len(res['different'])} different, {len(res['missing'])} not in the new set", flush=True)
            if res["different"]:
                problems.append(f"{len(res['different'])} old items differ from the new set's (first: {res['different'][:3]})")
            if not a.allow_missing and res["missing"]:
                problems.append(f"{len(res['missing'])} old items are not in the new set (first: {res['missing'][:3]})")
        elif check and not old_pt.exists():
            problems.append(f"{old_pt}: the old set is not there, the first block cannot be checked (--no-check)")
        if problems and check:
            for p in problems:
                print(f"[recall-items] {unit}: CHECK FAILED: {p}", flush=True)
            (raw / "check.failed").write_text("\n".join(problems) + "\n")
            return EX_CHECK
        done.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {time.time() - t0:.0f}s\n")
    if not a.unit and not a.dry_run:
        return cmd_split(a)
    return 0


def concat_sets(parts):
    """One EncodedSet from several of the same probe (the 8k-64k and the 128k builds of one group): items in the given order."""
    import torch
    from ced.eval.probes import EncodedSet

    if len(parts) == 1:
        return parts[0]
    assert len({p.probe for p in parts}) == 1 and all(p.cand_ids is None for p in parts)
    lengths = torch.cat([(p.offsets[1:] - p.offsets[:-1]) for p in parts])
    return EncodedSet(probe=parts[0].probe, ids=torch.cat([p.ids for p in parts]),
                      offsets=torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)]),
                      n_answer=torch.cat([p.n_answer for p in parts]), first_info=torch.cat([p.first_info for p in parts]),
                      answer_bytes=torch.cat([p.answer_bytes for p in parts]), items=[it for p in parts for it in p.items], cand_ids=None,
                      info=dict(parts[0].info))


def cell_digests(plan: dict, raw_sets: dict) -> dict:
    """Per cell of the plan, a sha256 over the lines 'item id TAB text_sha' of its items 0 .. n - 1 (all of them) and of its first block (the old items, 0 .. n_old - 1): the item-text hashes of
    the contamination record at the granularity a summary can hold (the per-item list stays in the raw sets' EncodedSets, `text_sha`, and the text files). Two builds that make the same
    items give the same digests; the first-block digest is the one of the old sets' items."""
    out = {}
    for c in plan["cells"]:
        cell = RI.CELLS[c["key"]]
        es, pos = raw_sets[cell.source]
        h_all, h_first = hashlib.sha256(), hashlib.sha256()
        for i in range(c["n"]):
            it = es.items[pos[f"{cell.item_prefix}{i:04d}"]]
            line = f"{it['id']}\t{it.get('text_sha', '')}\n".encode()
            h_all.update(line)
            if i < c["n_old"]:
                h_first.update(line)
        out[c["key"]] = {"n": c["n"], "sha256_all": h_all.hexdigest(), "n_first_block": min(c["n"], c["n_old"]), "sha256_first_block": h_first.hexdigest()}
    return out


def cmd_split(a) -> int:
    """The plan's groups from the raw builds. A group's directory <out>/groups/<g>/t64k holds multikey.pt and/or pcr_niahk.pt (the group's items, in cell then index order)."""
    from ced.eval.probes import EncodedSet  # torch

    plan = json.loads(Path(a.plan).read_text())
    RI.validate_plan(plan)
    out = Path(a.out)
    raw_sets: dict[str, tuple[EncodedSet, dict]] = {}
    for unit in RI.SOURCES:
        if not unit_cells(plan, unit):
            continue
        pt = out / "raw" / unit / "t64k" / RAW_FILE[unit][1]
        if not (out / "raw" / unit / "build.done").exists() and not a.no_check:
            print(f"[recall-items] {unit}: not built (no {out / 'raw' / unit / 'build.done'})")
            return 1
        es = EncodedSet.load(pt)
        raw_sets[unit] = (es, {it["id"]: i for i, it in enumerate(es.items)})
    manifest_groups = {}
    for g in plan["groups"]:
        by_probe: dict[str, list[int]] = {}
        for part in g["parts"]:
            cell = RI.CELLS[part["cell"]]
            es, pos = raw_sets[cell.source]
            for idx in range(part["start"], part["stop"]):
                item_id = f"{cell.item_prefix}{idx:04d}"
                if item_id not in pos:
                    print(f"[recall-items] {g['name']}: {item_id} is not in the raw {cell.source} build")
                    return 1
                by_probe.setdefault(cell.source, []).append(pos[item_id])
        gdir = out / "groups" / g["name"] / "t64k"
        gdir.mkdir(parents=True, exist_ok=True)
        subs: dict[str, list] = {}
        for unit, keep in by_probe.items():
            es, _ = raw_sets[unit]
            subs.setdefault(RAW_FILE[unit][1], []).append(es.subset(keep))
        files = {}
        for fname, parts_ in subs.items():
            sub = concat_sets(parts_)
            path = gdir / fname
            sub.save(path)
            files[fname] = {"items": len(sub), "sha256": sha256(path)}
        (gdir.parent / "group.json").write_text(json.dumps({"name": g["name"], "parts": g["parts"], "files": files}, indent=1))
        manifest_groups[g["name"]] = files
        print(f"[recall-items] {g['name']}: {sum(f['items'] for f in files.values())} items in {len(files)} file(s)", flush=True)
    (out / "manifest.json").write_text(json.dumps({"build": "recall-items", "plan": plan, "groups": manifest_groups,
                                                   "raw": {u: sha256(out / "raw" / u / "t64k" / RAW_FILE[u][1]) for u in raw_sets},
                                                   "cell_digests": cell_digests(plan, raw_sets)}, indent=1))
    (out / "build.done").write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("build", "split"):
        sp = sub.add_parser(name)
        sp.add_argument("--plan", default=str(DEFAULT_PLAN))
        sp.add_argument("--out", required=True)
        sp.add_argument("--old-root", default=None, help="the directory of the old sets for the first-block and pool checks (see the module docstring); without it they are skipped")
        sp.add_argument("--tokenizer-t64k", default=None, help="tokenizer.json of the 64k tokenizer (build)")
        sp.add_argument("--tokenizer-t128k", default=None, help="tokenizer.json of the second length tokenizer of the mk build (optional)")
        sp.add_argument("--docs", action="append", default=[], help="<set>.docs.jsonl.gz of a held-out eval set, the open haystack (build; twice: web and general_web)")
        sp.add_argument("--workers", type=int, default=4)
        sp.add_argument("--python", default=sys.executable)
        sp.add_argument("--unit", choices=RI.SOURCES, default=None, help="build one raw unit only (no split)")
        sp.add_argument("--dry-run", action="store_true")
        sp.add_argument("--force", action="store_true", help="build a unit again although its build.done exists")
        sp.add_argument("--no-check", action="store_true", help="skip the first-block and pool checks (a smoke build of items the old set does not hold)")
        sp.add_argument("--allow-missing", action="store_true", help="old items the new set does not hold are not an error (a smoke build holds fewer)")
    cp = sub.add_parser("check-first-block")
    cp.add_argument("--old", required=True)
    cp.add_argument("--new", required=True)
    a = p.parse_args(argv)
    if a.cmd == "build":
        if not a.tokenizer_t64k or not a.docs:
            raise SystemExit("build: --tokenizer-t64k and --docs (twice) are required")
        return cmd_build(a)
    if a.cmd == "split":
        return cmd_split(a)
    res = check_first_block(a.old, a.new)
    print(json.dumps({k: (v if not isinstance(v, list) else v[:10]) for k, v in res.items()}))
    return EX_CHECK if res["different"] else 0


if __name__ == "__main__":
    sys.exit(main())
