"""Build the retrieval-inside-the-training-length items and the copy-loss sequences (ced/eval/report_pcr_gen.py, ced/eval/report_pcr.py).

    python tools/build_report_pcr.py --out <pcr> \\
        --tokenizer t64k=<tokenizer.json> \\
        --docs <held-out docs>/web.docs.jsonl.gz --docs <held-out docs>/general_web.docs.jsonl.gz --workers 4
    (optional: --verify-rows <web.pt> --verify-rows <general_web.pt> --verify-tokenizer <tokenizer.json>)

CPU only; it reads the haystack documents (the web and general_web eval sets' docs files: held-out, never in a training shard) and the
tokenizer, and writes under --out (the same layout as tools/build_probes.py, so scripts/eval_probes.py --probes <out>/t64k scores it):
    <out>/text/<probe>.jsonl.gz     the retrieval text items (tokenizer-independent), one JSON per line
    <out>/<tok>/<probe>.pt          ced.eval.probes.EncodedSet per probe: pcr_niah1, pcr_niahk, pcr_vt, pcr_copy2, pcr_copy3, pcr_fresh
    <out>/<tok>/manifest.json       per probe: items, cells, token lengths, per-item check failures, sha256 of files
    <out>/manifest.json             seeds, counts, the haystack pool, all sha256, the token totals
With --verify-rows the documents are checked to be exactly what the EvalRows files hold (tools/build_probes.py verify_rows). A build whose
per-item checks fail (a context/answer boundary that tokenizes differently apart than together, an answer that is not in the context exactly
once, a variable that tokenizes differently in two places) asserts and exits 1 after writing the manifest. It is deterministic: the same inputs
give the same files (checked by sha256 in the manifest). A draw whose haystack has no whitespace cut in the length window is redrawn (the item's own
seeded next draw, ced/eval/report_pcr_gen.py "NO CUT, AND THE REDRAW"): the redraws are printed per probe and counted in manifest["probes"][probe]["redraws"]. The extrapolation probes (probes-v1 above 8k, the 128k set) are not regenerated or touched.
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
from ced.eval import report_pcr as R  # noqa: E402
from ced.eval import report_pcr_gen as G  # noqa: E402

_W: dict = {}


def _init(tok_path: str, pool_state: dict) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _W["tok"], _W["table"] = bp.load_tok(tok_path)
    _W["pool"] = g.HaystackPool(texts=pool_state["texts"], doc_ids=pool_state["doc_ids"], counts=np.asarray(pool_state["counts"]))


def count_item(context: str, answer: str) -> int:
    tok = _W["tok"]
    return len(tok.encode(context, add_special_tokens=False).ids) + len(tok.encode(answer, add_special_tokens=False).ids)


def make_item(job: tuple) -> dict:
    probe, args = job
    pool = _W["pool"]
    if probe == "pcr_niah1":
        it = G.niah1_item(pool, count_item, *args)
    elif probe == "pcr_niahk":
        it = G.niahk_item(pool, count_item, *args)
    else:
        it = G.vt_item(pool, count_item, *args)
    G.check_item(it)
    tok, table = _W["tok"], _W["table"]
    ce = tok.encode(it.context, add_special_tokens=False)
    ctx, ans = ce.ids, tok.encode(it.answer, add_special_tokens=False).ids
    joint = tok.encode(it.context + it.answer, add_special_tokens=False).ids
    rec = {"ids": np.asarray(ctx + ans, dtype=np.int32), "n_answer": len(ans), "answer_byte_lens": [int(table[i]) for i in ans],
           "boundary_ok": joint == ctx + ans}
    if probe in ("pcr_niah1", "pcr_niahk"):  # the value's tokens occur once in the context: the answer is not guessable from a second copy
        rec["answer_once"] = bp._count_sub(np.asarray(ctx), np.asarray(ans)) == 1
    else:  # every variable is one token sequence wherever it occurs, and the answer is those sequences in the chain's order
        chains = it.meta["chains"]
        ok = True
        for ch in chains:
            for k, nm in enumerate(ch):
                ok &= bp.var_tokens_same(ce, it.context, nm, 1 if k == len(ch) - 1 else 2)
        listed = [t for nm in chains[it.meta["asked"]] for t in tok.encode(" " + nm, add_special_tokens=False).ids]
        rec["var_tokens_ok"] = bool(ok and listed == ans)
    return {"item": it.to_json(), "tok": {"t64k": rec}, "redraws": it.meta.get("redraws")}


def redraw_summary(results: list[dict]) -> dict:
    """The build's count of the draws that found no cut in the length window (ced/eval/report_pcr_gen.py, NO CUT, AND THE REDRAW): items
    redrawn, failed draws in all, the most one item needed, and each redrawn item's id with its failed draws' texts."""
    redrawn = {r["item"]["id"]: r["redraws"] for r in results if r.get("redraws")}
    return {"items": len(redrawn), "failed_draws": sum(v["count"] for v in redrawn.values()),
            "max_per_item": max((v["count"] for v in redrawn.values()), default=0), "redrawn": redrawn}


def jobs_for(probe: str, a) -> list[tuple]:
    if probe == "pcr_niah1":
        return [(probe, (L, d, i)) for L in a.lengths for d in G.NIAH_DEPTHS for i in range(a.niah1_per_depth)]
    if probe == "pcr_niahk":
        return [(probe, (L, n, i)) for L in a.lengths for n in a.multi_n for i in range(a.per_cell)]
    return [(probe, (L, i)) for L in a.lengths for i in range(a.per_cell)]


def encode_copy_set(variant: str, allowed: np.ndarray, table: np.ndarray, lengths, items: dict, tok_name: str, tok_sha: str):
    """The EncodedSet of one copy variant: for each length L the first items[L] sequences of ced.eval.report_pcr.copy_item."""
    import torch

    from ced.eval.probes import EncodedSet

    ids, lens, n_answer, answer_bytes, meta = [], [], [], [], []
    for L in lengths:
        for i in range(items[L]):
            x, m = R.copy_item(allowed, variant, L, i)
            ids.append(x)
            lens.append(len(x))
            n_answer.append(m)
            answer_bytes.append(int(table[x[-m:]].sum()))
            meta.append({"id": R.copy_item_id(variant, L, i), "cell": {"L": int(L)}, "group": {},
                         "text_sha": hashlib.blake2b(x.tobytes(), digest_size=8).hexdigest()})
    return EncodedSet(
        probe=variant, ids=torch.from_numpy(np.concatenate(ids)),
        offsets=torch.from_numpy(np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)),
        n_answer=torch.tensor(n_answer, dtype=torch.int32), first_info=torch.zeros(len(ids), dtype=torch.int32),
        answer_bytes=torch.tensor(answer_bytes, dtype=torch.int32), items=meta, cand_ids=None,
        info={"tokenizer": tok_name, "tokenizer_sha256": tok_sha, "copy_seed": R.COPY_SEED, "vocab_size": int(len(table))})


def describe_copy(es) -> dict:
    cells = defaultdict(list)
    for i, it in enumerate(es.items):
        cells[g.cell_key(g.TextItem(id=it["id"], probe=es.probe, cell=it["cell"], context="", answer=""))].append((es.length(i), int(es.n_answer[i])))
    return {"cells": {c: {"items": len(v), "tokens_mean": round(float(np.mean([a for a, _ in v])), 1), "answer_tokens_mean": round(float(np.mean([b for _, b in v])), 2)}
                      for c, v in sorted(cells.items())},
            "tokens_total": int(es.ids.numel()), "checks": {"boundary_failures": 0}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", required=True, help="t64k=<tokenizer.json> (one tokenizer; the name is the probe directory)")
    p.add_argument("--docs", action="append", required=True, help="<set>.docs.jsonl.gz of a held-out eval set")
    p.add_argument("--verify-rows", action="append", default=[], help="the EvalRows .pt of each --docs file, same order")
    p.add_argument("--verify-tokenizer", default=None, help="the tokenizer the EvalRows were built with")
    p.add_argument("--lengths", default=",".join(str(x) for x in G.LENGTHS))
    p.add_argument("--niah1-per-depth", type=int, default=G.PER_DEPTH)
    p.add_argument("--per-cell", type=int, default=G.PER_CELL, help="pcr_niahk items per (length, n_keys); pcr_vt items per length")
    p.add_argument("--multi-n", default=",".join(str(x) for x in G.MULTI_N))
    p.add_argument("--copy-lengths", default=",".join(str(x) for x in R.COPY_LENGTHS))
    p.add_argument("--copy-items", type=int, default=None, help="copy items per length and variant (default: ced.eval.report_pcr.COPY_ITEMS)")
    p.add_argument("--probes", default=",".join((*G.PROBES, *R.COPY_VARIANTS)))
    p.add_argument("--same-pool-as", default=None, help="a probes manifest.json (probes-v1's): record whether the haystack documents are the files it was built from")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    a.lengths = [int(x) for x in a.lengths.split(",")]
    a.multi_n = [int(x) for x in a.multi_n.split(",")]
    copy_lengths = [int(x) for x in a.copy_lengths.split(",")]
    copy_items = {L: a.copy_items or R.COPY_ITEMS.get(L, 64) for L in copy_lengths}
    probes = a.probes.split(",")
    unknown = [x for x in probes if x not in (*G.PROBES, *R.COPY_VARIANTS)]
    if unknown:
        raise SystemExit(f"--probes: {unknown} are not report probes")
    name, _, tok_path = a.tokenizer.partition("=")
    if not tok_path:
        raise SystemExit("--tokenizer name=path")
    if name != "t64k":
        raise SystemExit("scripts/eval_probes.py maps the tokenizer name to the vocabulary size (t64k: 64,000): name it t64k")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, table = bp.load_tok(tok_path)
    tok_sha = bp.sha256(Path(tok_path))
    vocab = int(len(table))
    special = [int(i) for i in tok.get_added_tokens_decoder()]
    allowed = R.allowed_ids(vocab, special)

    manifest = {"build": "report-pcr", "base_seed": G.BASE_SEED, "copy_seed": R.COPY_SEED, "lengths": a.lengths, "length_tolerance": G.LENGTH_TOL,
                "length_measure": f"token count under {name}", "tokenizer": {"name": name, "path": tok_path, "sha256": tok_sha, "vocab": vocab,
                                                                          "special_ids": len(special), "copy_ids": int(len(allowed))},
                "redraw_rule": {"max_redraws": G.REDRAW_MAX, "first_stream": G.REDRAW_STREAM + 1,
                                "note": "a draw with no whitespace cut in the length window takes the next draw of the item's own seeded sequence; the window is not widened"},
                "counts_args": {"niah1_per_depth": a.niah1_per_depth, "per_cell": a.per_cell, "multi_n": a.multi_n,
                                "copy_lengths": copy_lengths, "copy_items": copy_items},
                "code": os.environ.get("CED_CODE_SHA", "local"), "numpy": np.__version__, "probes": {}, "files": {}}
    tok_manifest = {"tokenizer": manifest["tokenizer"], "files": {}, "probes": {}}

    retrieval = [x for x in probes if x in G.PROBES]
    if retrieval:
        docs = bp.read_docs(a.docs)
        verification = []
        if a.verify_rows:
            assert len(a.verify_rows) == len(a.docs) and a.verify_tokenizer, "--verify-rows needs one file per --docs and --verify-tokenizer"
            for rows_path, docs_path in zip(a.verify_rows, a.docs):
                res = bp.verify_rows(rows_path, bp.read_docs([docs_path]), a.verify_tokenizer)
                print(json.dumps(res), flush=True)
                assert res["docs_matching"] == res["docs"], f"{docs_path}: documents differ from {rows_path}"
                verification.append(res)
        keep = [(i, t.strip()) for _, i, t in docs if t.strip() and g.usable_haystack_doc(t)]
        enc = tok.encode_batch([t for _, t in keep], add_special_tokens=False)
        counts = np.array([len(e.ids) for e in enc], dtype=np.int64)
        pool_state = {"texts": [t for _, t in keep], "doc_ids": [i for i, _ in keep], "counts": counts.tolist()}
        manifest["haystack_pool"] = {"files": {Path(d).name: bp.sha256(Path(d)) for d in a.docs}, "docs": len(docs), "usable": len(keep),
                                     "dropped_lookalikes": len(docs) - len(keep), "tokens": int(counts.sum()),
                                     "median_doc_tokens": float(np.median(counts)) if len(counts) else 0.0, "verified_against_rows": verification}
        if a.same_pool_as:
            ref = json.loads(Path(a.same_pool_as).read_text()).get("haystack_pool", {}).get("files", {})
            manifest["haystack_pool"]["same_pool_as"] = {"manifest": a.same_pool_as, "same_files": ref == manifest["haystack_pool"]["files"]}
        print(f"pool: {len(keep)} of {len(docs)} docs, {int(counts.sum())} tokens ({time.time() - t0:.0f}s)", flush=True)
        ctx = get_context("fork")
        with ctx.Pool(a.workers, initializer=_init, initargs=(tok_path, pool_state)) as workers:
            for probe in retrieval:
                jobs = jobs_for(probe, a)
                ts = time.time()
                results = workers.map(make_item, jobs, chunksize=max(1, min(8, len(jobs) // (4 * a.workers) or 1)))
                text_path = out / "text" / f"{probe}.jsonl.gz"
                text_path.parent.mkdir(exist_ok=True)
                with gzip.GzipFile(text_path, "wb", mtime=0) as f:
                    for r in results:
                        f.write((json.dumps(r["item"], ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
                manifest["files"][f"text/{probe}.jsonl.gz"] = bp.sha256(text_path)
                es = bp.encode_set(probe, [{"item": r["item"], "tok": {name: r["tok"]["t64k"]}} for r in results], name, tok, tok_sha)
                # one decode makes the records' shared strings the same whatever the chunking of the worker pool was: the pickle of a record
                # list depends on which equal strings are one object, so without this the file's bytes would depend on --workers
                es.items = json.loads(json.dumps(es.items))
                path = out / name / f"{probe}.pt"
                path.parent.mkdir(exist_ok=True)
                es.save(path)
                digest = bp.sha256(path)
                tok_manifest["files"][probe] = manifest["files"][f"{name}/{probe}.pt"] = digest
                recs = [{"item": r["item"], "tok": {name: r["tok"]["t64k"]}} for r in results]
                tok_manifest["probes"][probe] = bp.describe(recs, name)
                redraws = redraw_summary(results)
                manifest["probes"][probe] = {"items": len(results),
                                             "cells": {c: v["items"] for c, v in tok_manifest["probes"][probe]["cells"].items()},
                                             "redraws": redraws}
                print(f"{probe}: {len(results)} items in {time.time() - ts:.0f}s; {json.dumps(tok_manifest['probes'][probe]['checks'])}; "
                      f"redraws {json.dumps({k: v for k, v in redraws.items() if k != 'redrawn'})}", flush=True)
                for item_id, rd in redraws["redrawn"].items():
                    print(f"  redrawn {item_id}: {rd['count']} failed draw(s): {'; '.join(rd['failed_draws'])}", flush=True)
                del results

    for variant in (x for x in probes if x in R.COPY_VARIANTS):
        es = encode_copy_set(variant, allowed, table, copy_lengths, copy_items, name, tok_sha)
        path = out / name / f"{variant}.pt"
        path.parent.mkdir(exist_ok=True)
        es.save(path)
        digest = bp.sha256(path)
        tok_manifest["files"][variant] = manifest["files"][f"{name}/{variant}.pt"] = digest
        tok_manifest["probes"][variant] = describe_copy(es)
        manifest["probes"][variant] = {"items": len(es), "cells": {c: v["items"] for c, v in tok_manifest["probes"][variant]["cells"].items()}}
        print(f"{variant}: {len(es)} items, {es.ids.numel()} tokens", flush=True)

    (out / name).mkdir(exist_ok=True)
    (out / name / "manifest.json").write_text(json.dumps(tok_manifest, indent=1))
    manifest["files"][f"{name}/manifest.json"] = bp.sha256(out / name / "manifest.json")
    manifest["tokens_total"] = {p_: int(v["tokens_total"]) for p_, v in tok_manifest["probes"].items()}
    manifest["build_seconds"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    failed = {p_: d["checks"] for p_, d in tok_manifest["probes"].items() if any(v for k, v in d["checks"].items() if k.endswith("failures"))}
    print(json.dumps({"failed_checks": failed}), flush=True)
    assert not failed, "per-item checks failed"
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
