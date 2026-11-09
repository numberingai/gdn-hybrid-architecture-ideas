"""Build the multi-fact probe sets (mf_chain, mf_concat; ced/eval/probe_gen_multifact.py) and tokenize them (CPU only):

    python tools/build_probes_multifact.py --out <multifact> \\
        --tokenizer t64k=<tokenizer.json> \\
        --docs <held-out docs>/web.docs.jsonl.gz --docs <held-out docs>/general_web.docs.jsonl.gz \\
        --per-cell 60 --workers 4

Writes, as tools/build_probes.py does
    <out>/text/<probe>.jsonl.gz       the text items (tokenizer-independent), one JSON per line
    <out>/<tok>/<probe>.pt            ced.eval.probes.EncodedSet per tokenizer; every item record also holds fact_spans
                                      (the answer-token range of each fact), fact_depths (the planned depth fractions),
                                      fact_depth_tokens (the realised depth of each fact in tokens, 0..1), layout, k
                                      and question_tokens (the context tokens the question covers: the item's last
                                      turn for the replay window, scripts/eval_long.py --replay-window turn)
    <out>/<tok>/manifest.json         counts, token lengths per cell, per-item checks, sha256 of files
    <out>/manifest.json               seeds, item counts, length measure, haystack pool, all sha256

Haystack: held-out documents only, the documents of the web and general_web eval sets (<set>.docs.jsonl.gz, a JSON line {"id", "text"} per document;
never in a training shard), as the uuid needle's haystacks, without the
documents that hold a UUID-shaped string, "access code" or "supervisor of".

Cells: probe x length (--lengths, default 8k / 16k / 32k / 64k) x k (--ks, default 2 / 4 / 8), --per-cell items each;
the layouts (spread / front / back) cycle with the item index, so a cell holds per-cell / 3 items of each.
Length measure: the larger of an item's token counts under the --length-tokenizer tokenizers (default: all given).

Per-item checks under each tokenizer (counted in the manifest; the build fails on any miss):
- boundary: tokenize(context + answer) equals tokenize(context) + tokenize(answer);
- pieces: the answer's fact and scaffold pieces, tokenized one by one, give the tokens of the whole answer (so the
  fact spans are exact);
- question: the question tokenized on its own gives the context's last tokens (so question_tokens is exact).
Reported, not a failure: items in which two facts share their first token (first_token_shared).
"""

from __future__ import annotations

import argparse
import gzip
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

from build_probes import encode_set, load_tok, read_docs, sha256  # noqa: E402
from ced.eval import probe_gen as g  # noqa: E402
from ced.eval import probe_gen_multifact as mf  # noqa: E402
from ced.eval.tokenized import TokenizedDoc  # noqa: E402

_W: dict = {}


def _init(tok_paths: dict, length_names: list[str], pool_state: dict) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _W["toks"] = {name: load_tok(p) for name, p in tok_paths.items()}
    _W["length"] = length_names
    _W["pool"] = g.HaystackPool(texts=pool_state["texts"], doc_ids=pool_state["doc_ids"], counts=np.asarray(pool_state["counts"]))


def _n(name: str, text: str) -> int:
    return len(_W["toks"][name][0].encode(text, add_special_tokens=False).ids)


def count_item(context: str, answer: str) -> int:
    return max(_n(t, context) + _n(t, answer) for t in _W["length"])


def make_item(job: tuple) -> dict:
    probe, length, k, index = job
    it = mf.make_item(probe, _W["pool"], count_item, length, k, index)
    mf.check_item(it)
    assert g.check_boundary_free(it.context, it.answer), it.id
    q = mf.query_start(it)
    per_tok = {}
    for name, (tok, table) in _W["toks"].items():
        def encode(text: str) -> TokenizedDoc:
            ids = tok.encode(text, add_special_tokens=False).ids
            return TokenizedDoc(ids=ids, byte_lens=[int(table[i]) for i in ids])

        ce = tok.encode(it.context, add_special_tokens=False)
        ctx, ans = ce.ids, tok.encode(it.answer, add_special_tokens=False).ids
        joint = tok.encode(it.context + it.answer, add_special_tokens=False).ids
        rec = {"ids": np.asarray(ctx + ans, dtype=np.int32), "n_answer": len(ans),
               "answer_byte_lens": [int(table[i]) for i in ans], "boundary_ok": joint == ctx + ans}
        try:
            spans = mf.fact_spans(it, encode)["fact_spans"]
            rec["pieces_ok"] = True
        except AssertionError:
            spans, rec["pieces_ok"] = [], False
        rec["fact_spans"] = spans
        rec["first_token_shared"] = len({ans[a] for a, _ in spans}) < len(spans)
        q_ids = tok.encode(mf.question_text(it.probe, it.context), add_special_tokens=False).ids
        rec["question_tokens"] = len(q_ids)
        rec["question_ok"] = bool(q_ids) and ctx[len(ctx) - len(q_ids):] == q_ids
        starts = np.array([a for a, _ in ce.offsets])
        q_tok = max(1, int(np.searchsorted(starts, q)))  # the tokens before the query
        rec["fact_depth_tokens"] = [round(float(np.searchsorted(starts, off) / q_tok), 4) for off in it.meta["char_offsets"]]
        per_tok[name] = rec
    return {"item": it.to_json(), "tok": per_tok}


def describe(results: list[dict], name: str) -> dict:
    """Per cell: items, token lengths (min / mean / max), answer and fact tokens (mean), layouts; and check failures."""
    cells = defaultdict(list)
    for r in results:
        cells[g.cell_key(r["item"]["cell"])].append((r["item"], r["tok"][name]))
    out = {"cells": {}}
    for c, pairs in sorted(cells.items()):
        n = np.array([len(t["ids"]) for _, t in pairs])
        layouts = defaultdict(int)
        for it, _ in pairs:
            layouts[it["group"]["layout"]] += 1
        out["cells"][c] = {"items": len(pairs), "tokens_min": int(n.min()), "tokens_mean": round(float(n.mean()), 1), "tokens_max": int(n.max()),
                           "answer_tokens_mean": round(float(np.mean([t["n_answer"] for _, t in pairs])), 2),
                           "fact_tokens_mean": round(float(np.mean([sum(b - a for a, b in t["fact_spans"]) for _, t in pairs])), 2),
                           "layouts": dict(layouts)}
    all_t = [r["tok"][name] for r in results]
    out["checks"] = {"boundary_failures": sum(1 for t in all_t if not t["boundary_ok"]),
                     "piece_failures": sum(1 for t in all_t if not t["pieces_ok"]),
                     "question_failures": sum(1 for t in all_t if not t["question_ok"]),
                     "first_token_shared": sum(1 for t in all_t if t["first_token_shared"])}
    out["tokens_total"] = int(sum(len(t["ids"]) for t in all_t))
    return out


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", action="append", required=True, help="name=path (e.g. t64k=<tokenizer.json>)")
    p.add_argument("--length-tokenizer", action="append", default=None, help="names that measure length (default: all)")
    p.add_argument("--docs", action="append", required=True, help="<set>.docs.jsonl.gz of a held-out eval set")
    p.add_argument("--probes", default=",".join(mf.PROBES))
    p.add_argument("--lengths", default=",".join(str(x) for x in mf.LENGTHS))
    p.add_argument("--ks", default=",".join(str(x) for x in mf.KS))
    p.add_argument("--per-cell", type=int, default=60, help="items per (probe, length, k); a multiple of 3 keeps the layouts equal")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    lengths, ks, probes = [int(x) for x in a.lengths.split(",")], [int(x) for x in a.ks.split(",")], a.probes.split(",")
    assert all(pr in mf.PROBES for pr in probes), probes
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok_paths = dict(kv.split("=", 1) for kv in a.tokenizer)
    length_names = a.length_tokenizer or list(tok_paths)
    t0 = time.time()

    docs = read_docs(a.docs)
    all_toks = {n: load_tok(pth) for n, pth in tok_paths.items()}
    keep = [(i, t.strip()) for _, i, t in docs if t.strip() and mf.usable_haystack_doc(t)]
    counts = np.zeros(len(keep), dtype=np.int64)
    for n in length_names:
        enc = all_toks[n][0].encode_batch([t for _, t in keep], add_special_tokens=False)
        counts = np.maximum(counts, np.array([len(e.ids) for e in enc], dtype=np.int64))
    pool_state = {"texts": [t for _, t in keep], "doc_ids": [i for i, _ in keep], "counts": counts.tolist()}
    pool_info = {"files": {Path(d).name: sha256(Path(d)) for d in a.docs}, "docs": len(docs), "usable": len(keep),
                 "dropped_lookalikes": len(docs) - len(keep), "tokens_length_measure": int(counts.sum()),
                 "median_doc_tokens": float(np.median(counts)) if len(counts) else None}
    print(f"pool: {pool_info['usable']} of {len(docs)} docs, {pool_info['tokens_length_measure']} tokens ({time.time() - t0:.0f}s)", flush=True)
    assert pool_info["tokens_length_measure"] >= 2 * max(lengths), "the haystack pool is too small for the longest items"

    tok_info = {n: {"path": pth, "sha256": sha256(Path(pth))} for n, pth in tok_paths.items()}
    manifest = {"probe_version": "multifact-v1", "base_seed": mf.BASE_SEED, "length_tolerance": g.LENGTH_TOL,
                "length_measure": f"max token count over {length_names}", "tokenizers": tok_info, "haystack_pool": pool_info,
                "numpy": np.__version__, "args": {"lengths": lengths, "ks": ks, "per_cell": a.per_cell, "layouts": list(mf.LAYOUTS)},
                "probes": {}, "files": {}}
    tok_manifest = {n: {"tokenizer": tok_info[n], "files": {}, "probes": {}} for n in tok_paths}

    ctx = get_context("fork")
    with ctx.Pool(a.workers, initializer=_init, initargs=(tok_paths, length_names, pool_state)) as workers:
        for probe in probes:
            jobs = [(probe, L, k, i) for L in lengths for k in ks for i in range(a.per_cell)]
            ts = time.time()
            results = workers.map(make_item, jobs, chunksize=max(1, min(8, len(jobs) // (4 * a.workers) or 1)))
            text_path = out / "text" / f"{probe}.jsonl.gz"
            text_path.parent.mkdir(exist_ok=True)
            with gzip.GzipFile(text_path, "wb", mtime=0) as f:
                for r in results:
                    f.write((json.dumps(r["item"], ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            manifest["files"][f"text/{probe}.jsonl.gz"] = sha256(text_path)
            cells = defaultdict(int)
            for r in results:
                cells[g.cell_key(r["item"]["cell"])] += 1
            manifest["probes"][probe] = {"items": len(results), "cells": dict(cells)}
            for name in tok_paths:
                es = encode_set(probe, results, name, all_toks[name][0], tok_info[name]["sha256"])
                for rec, r in zip(es.items, results):
                    t, it = r["tok"][name], r["item"]
                    rec.update(fact_spans=t["fact_spans"], fact_depths=[round(float(d), 4) for d in it["meta"]["depths"]],
                               fact_depth_tokens=t["fact_depth_tokens"], layout=it["meta"]["layout"], k=it["cell"]["k"],
                               question_tokens=t["question_tokens"])
                path = out / name / f"{probe}.pt"
                path.parent.mkdir(exist_ok=True)
                es.save(path)
                digest = sha256(path)
                tok_manifest[name]["files"][probe] = digest
                manifest["files"][f"{name}/{probe}.pt"] = digest
                tok_manifest[name]["probes"][probe] = describe(results, name)
            print(f"{probe}: {len(results)} items in {time.time() - ts:.0f}s; "
                  + "; ".join(f"{n}: {json.dumps(tok_manifest[n]['probes'][probe]['checks'])}" for n in tok_paths), flush=True)
            del results
    for name in tok_paths:
        (out / name / "manifest.json").write_text(json.dumps(tok_manifest[name], indent=1))
        manifest["files"][f"{name}/manifest.json"] = sha256(out / name / "manifest.json")
    manifest["build_seconds"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    failed = {n: {pr: d["checks"] for pr, d in m["probes"].items() if any(v for k, v in d["checks"].items() if k.endswith("failures"))}
              for n, m in tok_manifest.items()}
    print(json.dumps({"failed_checks": failed}), flush=True)
    assert not any(failed.values()), "per-item checks failed"
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return manifest


if __name__ == "__main__":
    main()
