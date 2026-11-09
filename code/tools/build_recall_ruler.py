"""Build the RULER subset's items (ced/eval/recall_ruler.py).

    python tools/build_recall_ruler.py --out <items> \\
        --tokenizer t64k=<tokenizer.json> \\
        --docs <held-out docs>/web.docs.jsonl.gz --docs <held-out docs>/general_web.docs.jsonl.gz --per-cell 500 --workers 4

CPU only. It reads the held-out documents (the pool of the pcr evals: the web and general_web eval sets' docs files, never in a training shard) and the
tokenizer, and writes under --out:
    items/<task>.jsonl.gz       the items of the task, one JSON per line (ced.eval.recall_ruler.RulerItem), by length then index
    manifest.json               seeds, counts per cell, token lengths, the redraws, the sha256 of every file, the pool and the tokenizer; the checks
    haystack-docs-used.json     the ids of the documents the open-haystack items were cut from, per task and in all
    item-hashes.json            {"<task>:len=<T>/<index>": sha256 of the prompt}: the item-text hashes of the contamination manifest
    contamination-manifest.json this build's ids and hashes
Deterministic: the same inputs give the same files (the manifest holds the sha256). An item that fails a structural rule (check_item) raises and ends the build with no
manifest; a build in which the prompt's token count under the tokenizer differs from the builder's writes the manifest and exits 1. The items of an index below the per-cell count are the same whatever the
count: a build with 500 items a cell holds, as its first 250 indices, the build with 250.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import build_probes as bp  # noqa: E402
from ced.eval import probe_gen as g  # noqa: E402
from ced.eval import recall_ruler as RR  # noqa: E402

_W: dict = {}


def _init(tok_path: str, pool_state: dict) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _W["tok"], _ = bp.load_tok(tok_path)
    _W["pool"] = g.HaystackPool(texts=pool_state["texts"], doc_ids=pool_state["doc_ids"], counts=np.asarray(pool_state["counts"]))


def count_text(text: str) -> int:
    return len(_W["tok"].encode(text, add_special_tokens=False).ids)


def make_one(job: tuple) -> dict:
    task, length, index = job
    it = RR.make_item(task, length, index, count_text, _W["pool"] if task in RR.OPEN_TASKS else None)
    RR.check_item(it)
    n_tok = count_text(it.prompt)  # a second, independent count of the finished prompt
    return {"item": it.to_json(), "tokens_ok": n_tok == it.n_prompt_tokens, "n_tokens": n_tok}


def sha256_file(path: Path) -> str:
    return bp.sha256(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", required=True, help="t64k=<tokenizer.json>")
    p.add_argument("--docs", action="append", required=True, help="<set>.docs.jsonl.gz of a held-out eval set (the open haystack)")
    p.add_argument("--per-cell", type=int, default=RR.PER_CELL, help="items per (task, length); RULER's default is 500")
    p.add_argument("--tasks", default=",".join(RR.TASKS))
    p.add_argument("--lengths", default=",".join(str(x) for x in RR.LENGTHS))
    p.add_argument("--same-pool-as", default=None, help="the pcr build's manifest.json: record whether the docs files are the pool it was built from")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    tasks = a.tasks.split(",")
    lengths = [int(x) for x in a.lengths.split(",")]
    bad = [t for t in tasks if t not in RR.TASKS]
    if bad:
        raise SystemExit(f"--tasks: {bad} are not tasks of the subset {RR.TASKS}")
    name, _, tok_path = a.tokenizer.partition("=")
    if not tok_path or name != "t64k":
        raise SystemExit("--tokenizer t64k=<path>")
    out = Path(a.out)
    (out / "items").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, _ = bp.load_tok(tok_path)
    tok_sha = bp.sha256(Path(tok_path))

    docs = bp.read_docs(a.docs)
    keep = [(i, t.strip()) for _, i, t in docs if t.strip() and RR.usable_doc(t)]
    enc = tok.encode_batch([t for _, t in keep], add_special_tokens=False)
    counts = np.array([len(e.ids) for e in enc], dtype=np.int64)
    pool_state = {"texts": [t for _, t in keep], "doc_ids": [i for i, _ in keep], "counts": counts.tolist()}
    pool = {"files": {Path(d).name: bp.sha256(Path(d)) for d in a.docs}, "docs": len(docs), "usable": len(keep), "dropped_lookalikes": len(docs) - len(keep),
            "tokens": int(counts.sum()), "median_doc_tokens": float(np.median(counts)) if len(counts) else 0.0}
    if a.same_pool_as:
        ref = json.loads(Path(a.same_pool_as).read_text()).get("haystack_pool", {}).get("files", {})
        pool["same_pool_as"] = {"manifest": a.same_pool_as, "same_files": ref == pool["files"]}
    print(f"pool: {len(keep)} of {len(docs)} docs, {int(counts.sum())} tokens ({time.time() - t0:.0f}s)", flush=True)

    manifest = {"build": "recall-ruler", "family": "ruler", "variant": "RULER subset, open haystack", "base_seed": RR.BASE_SEED, "lengths": lengths, "tasks": tasks,
                "per_cell": a.per_cell, "length_tolerance": RR.LENGTH_TOL, "length_measure": "prompt tokens plus tokens to generate, under t64k",
                "tokens_to_generate": {"niah": RR.NIAH_NEW, "vt": RR.VT_NEW},
                "tokenizer": {"name": name, "path": tok_path, "sha256": tok_sha, "vocab": tok.get_vocab_size()},
                "haystack_pool": pool, "code": os.environ.get("CED_CODE_SHA", "local"), "numpy": np.__version__, "tasks_detail": {}, "files": {}}
    jobs = [(t, L, i) for t in tasks for L in lengths for i in range(a.per_cell)]
    ctx = get_context("fork")
    hashes: dict = {}
    docs_used: dict = {}
    bad_items: list = []
    with ctx.Pool(a.workers, initializer=_init, initargs=(tok_path, pool_state)) as workers:
        for task in tasks:
            ts = time.time()
            tj = [j for j in jobs if j[0] == task]
            results = workers.map(make_one, tj, chunksize=max(1, min(8, len(tj) // (4 * a.workers) or 1)))
            path = out / "items" / f"{task}.jsonl.gz"
            with gzip.GzipFile(path, "wb", mtime=0) as f:
                for r in results:
                    f.write((json.dumps(r["item"], ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            manifest["files"][f"items/{task}.jsonl.gz"] = sha256_file(path)
            cells: dict = defaultdict(list)
            redraws = {}
            used: set = set()
            for r in results:
                it = r["item"]
                if not r["tokens_ok"]:
                    bad_items.append(it["id"])
                cells[it["length"]].append(it["n_prompt_tokens"] + it["max_new"])
                hashes[RR.item_key(it["task"], it["length"], it["index"])] = RR.text_sha(it["prompt"])
                if it["meta"].get("redraws"):
                    redraws[it["id"]] = it["meta"]["redraws"]
                used.update(it["meta"].get("haystack_docs", []))
            docs_used[task] = sorted(used)
            manifest["tasks_detail"][task] = {
                "haystack": RR.SPEC[task]["haystack"], "ruler_haystack": RR.SPEC[task]["ruler_haystack"], "items": len(results),
                "cells": {str(L): {"items": len(v), "length_min": int(min(v)), "length_mean": round(float(np.mean(v)), 1), "length_max": int(max(v))}
                          for L, v in sorted(cells.items())},
                "redraws": {"items": len(redraws), "failed_draws": sum(v["count"] for v in redraws.values()),
                            "max_per_item": max((v["count"] for v in redraws.values()), default=0)}}
            print(f"{task}: {len(results)} items in {time.time() - ts:.0f}s; lengths {json.dumps({L: [min(v), max(v)] for L, v in sorted(cells.items())})}; "
                  f"redraws {manifest['tasks_detail'][task]['redraws']}", flush=True)
            del results

    (out / "item-hashes.json").write_text(json.dumps(hashes, sort_keys=True))
    manifest["files"]["item-hashes.json"] = sha256_file(out / "item-hashes.json")
    union = sorted(set(d for v in docs_used.values() for d in v))
    (out / "haystack-docs-used.json").write_text(json.dumps({"per_task": docs_used, "union": union, "union_count": len(union), "pool_usable": len(keep)}, sort_keys=True))
    manifest["files"]["haystack-docs-used.json"] = sha256_file(out / "haystack-docs-used.json")
    manifest["checks"] = {"prompt_token_count_mismatches": len(bad_items), "mismatched": bad_items[:20]}
    manifest["item_hashes_digest"] = hashlib.sha256(json.dumps(sorted(hashes.items())).encode()).hexdigest()
    manifest["haystack_docs_used"] = {"union_count": len(union), "pool_usable": len(keep)}
    manifest["build_seconds"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    c = {"family": "ruler"}  # the contamination manifest: this build's ids and hashes
    c["build"] = {"code": manifest["code"], "per_cell": a.per_cell, "tokenizer_sha256": tok_sha, "items_manifest_sha256": hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest(),
                  "pool_files_sha256": pool["files"], "pool_usable": len(keep), "item_hashes_digest": manifest["item_hashes_digest"],
                  "item_count": len(hashes), "haystack_docs_used_union_count": len(union), "haystack_docs_used_file": "haystack-docs-used.json",
                  "item_hashes_file": "item-hashes.json"}
    (out / "contamination-manifest.json").write_text(json.dumps(c, indent=1, sort_keys=True))
    print(json.dumps({"checks": manifest["checks"]}), flush=True)
    if bad_items:
        print(f"FAILED: {len(bad_items)} items whose prompt token count differs from the builder's", flush=True)
        return 1
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
