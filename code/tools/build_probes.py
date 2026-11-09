"""Build the probe sets (ced/eval/probe_gen.py) and tokenize them.

    python tools/build_probes.py --out <probes> \\
        --tokenizer t64k=<tokenizer.json> \\
        --docs <held-out docs>/web.docs.jsonl.gz --docs <held-out docs>/general_web.docs.jsonl.gz --workers 4

Two tokenizers can be given (--tokenizer t128k=<other tokenizer.json> --tokenizer t64k=<tokenizer.json>); the sets are then written once per tokenizer. To check the
documents against an EvalRows file add --verify-rows <web.pt> --verify-rows <general_web.pt> --verify-tokenizer <tokenizer.json> (the tokenizer the rows were built with).

Writes
    <out>/text/<probe>.jsonl.gz       the text items (tokenizer-independent), one JSON per line
    <out>/<tok>/<probe>.pt            ced.eval.probes.EncodedSet per tokenizer
    <out>/<tok>/manifest.json         counts, token lengths per cell, per-item checks, sha256 of files
    <out>/manifest.json               seeds, item counts, length measure, haystack pool, all sha256

Haystack: the documents of the web and general_web eval sets (held out: never in a training shard),
read from their <set>.docs.jsonl.gz (a JSON line {"id", "text"} per document). With
--verify-rows, every document's text is checked to be exactly what the EvalRows file holds: the
tokenization of the text plus <|endoftext|> equals that document's tokens in the rows (so the
text is the decoded eval-set document).

Length measure: an item's length is the larger of its token counts under the --length-tokenizer
tokenizers (default: every --tokenizer given), so an item of nominal length T is at most T tokens
under each of them. Lengths of 1x, 2x, 4x, 8x the 8,192-token training row.

Fixed-distance needles (--needle-distances; ced/eval/probe_gen_distance.py): uuid_needle and multikey items whose (asked) needle sits a requested number of tokens before the
question instead of at a depth fraction, for example

    python tools/build_probes.py --out <probes-dist> --tokenizer t64k=<tokenizer.json> \
        --docs <held-out docs>/web.docs.jsonl.gz --docs <held-out docs>/general_web.docs.jsonl.gz \
        --probes uuid_needle,multikey --lengths 32768,65536 --needle-distances 2048:61440:2048 \
        --distance-per-cell 40 --distance-keys 4 --workers 6

--needle-distances: a list (2048,4096) or start:stop:step with the stop included; every distance that fits a length
(distance + 512 <= 0.97 x length) is built at that length, --distance-per-cell items each, multikey with
--distance-keys keys. The distance is measured under the first length tokenizer (from the needle's first value token
to the context's last token) and must be within --distance-tol tokens of the request (distance_failures). An item's
depth is 1 - distance / length (group: depth or asked_depth, and distance); the measured distance is in the item
(needle_distance_tokens, per tokenizer). The documents, key and value of item i are the same at every distance of a
length; only the needle's place and the documents' order differ. Items are written by length, item index, distance.

Per-item checks under each tokenizer (all counted in the manifest; the build fails on any miss):
- the context and the answer tokenize at a clean boundary: tokenize(context + answer) equals
  tokenize(context) + tokenize(answer);
- uuid_needle, multikey, path_copy: the answer's tokens occur exactly once in the context's tokens;
- path_copy one-choice cells (K x 1 x 1): at the first choice token every listed path has its own
  token (share 1, options K), and that token is more than the "/" (choice_failures);
- depth: every candidate is a single token; for the "body" format, every occurrence of the asked
  variable (its assignment, the right-hand side that copies it, the query) is the same token
  sequence (var_token_failures).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ced.eval import probe_gen as g  # noqa: E402
from ced.eval import probe_gen_distance as gd  # noqa: E402

EOT = "<|endoftext|>"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load_tokenizer(path: str):
    """(tokenizer, byte-length table). The tokenizer encodes special-token strings as plain text (encode_special_tokens);
    the table holds the UTF-8 byte length of every token id."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(path)
    tok.encode_special_tokens = True
    added = {t.content for t in tok.get_added_tokens_decoder().values()}
    table = np.zeros(tok.get_vocab_size(), dtype=np.int64)
    for token, i in tok.get_vocab().items():
        table[i] = len(token.encode("utf-8")) if token in added else len(token)
    assert table.max() <= 255, f"a token covers {table.max()} bytes; the shard format stores uint8"
    return tok, table


def load_tok(path: str):
    return load_tokenizer(path)


def read_docs(paths: list[str]) -> list[tuple[str, str, str]]:
    """(source file stem, id, text) of every document, in file order."""
    out = []
    for p in paths:
        stem = Path(p).name.split(".")[0]
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                out.append((stem, d["id"], d["text"]))
    return out


def verify_rows(rows_path: str, docs: list[tuple[str, str, str]], tokenizer_path: str) -> dict:
    """Check that the documents of the set equal the decoded documents of its EvalRows file."""
    import torch

    tok, _ = load_tok(tokenizer_path)
    eot = tok.token_to_id(EOT)
    data = torch.load(rows_path, map_location="cpu", weights_only=False)
    tokens, targets, doc_ids = data["tokens"], data["targets"], data["doc_ids"]
    stream = torch.cat([tokens.reshape(-1), targets[-1, -1:]]).numpy()
    docs_stream = torch.cat([doc_ids.reshape(-1), doc_ids[-1, -1:]]).numpy()
    n_rows, L = tokens.shape
    assert torch.equal(tokens[1:, 0], targets[:-1, -1]), "rows do not overlap by one token (rows dropped?)"
    bounds = np.flatnonzero(np.diff(docs_stream)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(stream)]])
    by_doc = {int(docs_stream[a]): stream[a:b] for a, b in zip(starts, ends)}
    texts = [t for _, _, t in docs]
    enc = tok.encode_batch(texts, add_special_tokens=False)
    bad = []
    for k, e in enumerate(enc):
        want = np.asarray(e.ids + [eot])
        got = by_doc.get(k)
        if got is None or len(got) != len(want) or not np.array_equal(got, want):
            bad.append(k)
    return {"rows_file": rows_path, "rows_sha256": sha256(Path(rows_path)), "docs": len(texts),
            "docs_matching": len(texts) - len(bad), "mismatch_examples": bad[:10]}


# --------------------------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------------------------

_W: dict = {}


def _init(tok_paths: dict, length_names: list[str], pool_state: dict, formats: dict, distance_tol: int = gd.DIST_TOL) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _W["formats"] = formats
    _W["toks"] = {name: load_tok(p) for name, p in tok_paths.items()}
    _W["length"] = length_names
    _W["pool"] = g.HaystackPool(texts=pool_state["texts"], doc_ids=pool_state["doc_ids"],
                                counts=np.asarray(pool_state["counts"]))
    _W["tokens"] = gd.TokenIndex(_W["pool"], token_starts)  # fixed-distance items: per-document token offsets, on first use
    _W["distance_tol"] = distance_tol  # not in `formats`: that dict goes into every set's manifest as it is


def token_starts(text: str) -> np.ndarray:
    """The character offset where each token of text starts, under the placement tokenizer (the first length
    tokenizer): the measure of a fixed-distance item's distance."""
    tok = _W["toks"][_W["length"][0]][0]
    return np.array([a for a, _ in tok.encode(text, add_special_tokens=False).offsets], dtype=np.int64)


def _n(name: str, text: str) -> int:
    tok = _W["toks"][name][0]
    return len(tok.encode(text, add_special_tokens=False).ids)


def count_item(context: str, answer: str) -> int:
    return max(_n(t, context) + _n(t, answer) for t in _W["length"])


def make_item(job: tuple) -> dict:
    probe, args = job
    pool = _W["pool"]
    if probe in ("uuid_needle", "multikey") and args[0] == "dist":  # fixed distance: ("dist", length, distance, index, n_keys)
        _, L, D, i, n_keys = args
        it = gd.distance_item(pool, count_item, _W["tokens"], probe, L, D, i, n_keys=n_keys, tol=_W["distance_tol"])
    elif probe == "uuid_needle":
        it = g.uuid_needle_item(pool, count_item, *args)
    elif probe == "multikey":
        it = g.multikey_item(pool, count_item, *args)
    elif probe == "path_copy":
        k, prefix, i, grid = args
        it = g.path_copy_item(k, prefix, i, fmt=_W["formats"]["path_copy"], grid=grid)
    else:
        h, i, chains, statements, layout = args
        it = g.depth_item(h, i, fmt=_W["formats"]["depth"], chains=chains, statements=statements, layout=layout)
    g.check_item(it)
    assert g.check_boundary_free(it.context, it.answer), it.id
    per_tok = {}
    for name, (tok, table) in _W["toks"].items():
        ce = tok.encode(it.context, add_special_tokens=False)
        ctx, ans = ce.ids, tok.encode(it.answer, add_special_tokens=False).ids
        joint = tok.encode(it.context + it.answer, add_special_tokens=False).ids
        rec = {"ids": np.asarray(ctx + ans, dtype=np.int32), "n_answer": len(ans),
               "answer_byte_lens": [int(table[i]) for i in ans], "boundary_ok": joint == ctx + ans}
        if it.candidates is None:
            rec["answer_once"] = _count_sub(np.asarray(ctx), np.asarray(ans)) == 1
        else:
            cands = [tok.encode(c, add_special_tokens=False).ids for c in it.candidates]
            rec["cands_single"] = all(len(c) == 1 for c in cands)
        if probe == "path_copy":  # the first answer token that decides which listed path is copied
            from ced.eval.probes import first_choice_info

            listed = [tok.encode(" " + p, add_special_tokens=False).ids for p in it.meta["paths"]]
            k, options, share = first_choice_info(ans, listed)
            rec["choice"] = (k, options, share)
            G, S, D = g.parse_grid(it.cell["grid"])
            if S * D == 1:  # one choice: every listed path has its own token there, and it is more than "/"
                rec["choice_ok"] = share == 1 and options == it.cell["k"] and int(table[ans[k]]) >= 2
        if probe == "depth" and it.meta["format"] == "body":  # every variable: one token sequence everywhere
            asked = it.meta["asked_var"]  # (assigned once, read by the query: two occurrences at least)
            rec["var_tokens_ok"] = all(var_tokens_same(ce, it.context, v, 2 if v == asked else 1)
                                       for v, _ in g.depth_statements(it.context))
        if probe in ("uuid_needle", "multikey"):  # where the (asked) needle sits, in tokens
            off = it.meta["char_offset"] if probe == "uuid_needle" else it.meta["char_offsets"][it.meta["asked"]]
            q = g.query_start(it)
            starts = np.array([a for a, _ in ce.offsets])
            rec["needle_depth_tokens"] = float(np.searchsorted(starts, off) / max(1, np.searchsorted(starts, q)))
            if "distance" in it.group:  # fixed distance: the needle's first value token to the context's last token
                at = _find_sub(np.asarray(ctx), np.asarray(ans))
                rec["needle_distance_tokens"] = None if at is None else len(ctx) - 1 - at
                if name == _W["length"][0]:  # the request is met under the placement tokenizer
                    rec["distance_ok"] = at is not None and abs(len(ctx) - 1 - at - it.group["distance"]) <= _W["distance_tol"]
        per_tok[name] = rec
    return {"item": it.to_json(), "tok": per_tok}


def var_tokens_same(enc, context: str, var: str, at_least: int = 1) -> bool:
    """Every occurrence of variable `var` in `context` (at least `at_least`) is preceded by a space
    and covered by exactly the tokens of " <var>", the same ids each time (enc: the context's
    Encoding, with offsets)."""
    seqs = []
    for m in re.finditer(rf"(?<!\w){var}(?!\w)", context):
        a, b = m.start() - 1, m.end()
        if a < 0 or context[a] != " ":
            return False
        cover = [(t, o) for t, o in zip(enc.ids, enc.offsets) if o[0] < b and o[1] > a]
        if not cover or cover[0][1][0] != a or cover[-1][1][1] != b:
            return False
        seqs.append(tuple(t for t, _ in cover))
    return len(seqs) >= at_least and len(set(seqs)) == 1


def _find_sub(hay: np.ndarray, needle: np.ndarray) -> int | None:
    """The start of the first occurrence of needle in hay, or None."""
    m = len(needle)
    if m == 0 or len(hay) < m:
        return None
    for a in np.flatnonzero(hay[: len(hay) - m + 1] == needle[0]):
        if np.array_equal(hay[a : a + m], needle):
            return int(a)
    return None


def _count_sub(hay: np.ndarray, needle: np.ndarray) -> int:
    m = len(needle)
    if m == 0 or len(hay) < m:
        return 0
    cand = np.flatnonzero(hay[: len(hay) - m + 1] == needle[0])
    return int(sum(1 for a in cand if np.array_equal(hay[a : a + m], needle)))


# --------------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------------


def parse_distances(text: str) -> list[int]:
    """"2048,4096" or "2048:61440:2048" (start:stop:step, the stop included)."""
    if ":" in text:
        lo, hi, step = (int(x) for x in text.split(":"))
        return list(range(lo, hi + 1, step))
    return [int(x) for x in text.split(",") if x]


def distance_jobs(probe: str, a) -> list[tuple]:
    """Fixed-distance jobs, by length, item index, distance (a worker keeps the last items' documents: every distance
    of an item index reuses one fit)."""
    keys = [1] if probe == "uuid_needle" else a.distance_keys
    return [(probe, ("dist", L, D, i, n)) for L in a.lengths for n in keys for i in range(a.distance_per_cell)
            for D in gd.distances_for(L, a.needle_distances)]


def jobs_for(probe: str, a) -> list[tuple]:
    if a.needle_distances and probe in ("uuid_needle", "multikey"):
        return distance_jobs(probe, a)
    if probe == "uuid_needle":
        return [(probe, (L, d, i)) for L in a.lengths for d in g.NEEDLE_DEPTHS for i in range(a.uuid_per_depth)]
    if probe == "multikey":
        return [(probe, (L, n, i)) for L in a.lengths for n in g.MULTIKEY_N for i in range(a.multikey_per_cell)]
    if probe == "path_copy":
        return [(probe, (k, p, i, grid)) for k, grid in a.path_cells for p in a.path_prefixes for i in range(a.path_per_cell)]
    return [(probe, (h, i, a.depth_chains, a.depth_statements, a.depth_layout)) for h in a.depth_hops
            for i in range(a.depth_per_hop[h])]


def parse_hops(text: str) -> list[int]:
    """"1-8" or "0,1,2,4"."""
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def parse_per_hop(text: str, hops: list[int]) -> dict[int, int]:
    """"1000" (every hop) or "0-2:4000,3-8:1000" (hop ranges; every hop of --depth-hops needed)."""
    if ":" not in text:
        return {h: int(text) for h in hops}
    out = {}
    for part in text.split(","):
        rng, _, n = part.partition(":")
        for h in parse_hops(rng):
            out[h] = int(n)
    missing = [h for h in hops if h not in out]
    assert not missing, f"--depth-per-hop gives no count for hops {missing}"
    return {h: out[h] for h in hops}


def parse_path_cells(text: str) -> list[tuple[int, str]]:
    """"12,24" (each K with its default grid) or "8:8x1x1,12:2x3x4"."""
    out = []
    for part in text.split(","):
        k, _, grid = part.partition(":")
        k = int(k)
        out.append((k, grid or "x".join(str(x) for x in g.PATH_GRID[k])))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", action="append", required=True, help="name=path (e.g. t64k=<tokenizer.json>)")
    p.add_argument("--length-tokenizer", action="append", default=None, help="names that measure length (default: all)")
    p.add_argument("--docs", action="append", required=True, help="<set>.docs.jsonl.gz of a held-out eval set")
    p.add_argument("--verify-rows", action="append", default=[], help="the EvalRows .pt of each --docs file, same order")
    p.add_argument("--verify-tokenizer", default=None, help="the tokenizer the EvalRows were built with")
    p.add_argument("--probes", default=",".join(g.PROBES))
    p.add_argument("--lengths", default=",".join(str(x) for x in g.LENGTHS))
    p.add_argument("--uuid-per-depth", type=int, default=60)
    p.add_argument("--multikey-per-cell", type=int, default=300)
    p.add_argument("--path-per-cell", type=int, default=800)
    p.add_argument("--depth-per-hop", default="4000", help="items per hop: 4000, or per hop range: 0-2:4000,3-8:1000")
    p.add_argument("--path-cells", default=",".join(f"{k}:{'x'.join(map(str, grid))}" for k, grid in g.PATH_CELLS),
                   help="path_copy cells K:GxSxD (groups x stems x digits), comma-separated")
    p.add_argument("--path-prefixes", default=",".join(g.PATH_PREFIX))
    p.add_argument("--depth-hops", default=f"{min(g.HOPS)}-{max(g.HOPS)}", help="e.g. 0-8 or 0,1,2,4")
    p.add_argument("--depth-chains", type=int, default=g.DEPTH_CHAINS)
    p.add_argument("--depth-statements", type=int, default=None,
                   help=f"lines per item (default: {g.DEPTH_STATEMENTS} for the equal layout, {g.FREE_DEPTH_STATEMENTS} for free)")
    p.add_argument("--depth-layout", default=g.DEPTH_LAYOUT, choices=g.DEPTH_LAYOUTS)
    p.add_argument("--needle-distances", default=None,
                   help="uuid_needle / multikey with the needle a requested number of tokens before the question: "
                        "2048,4096 or start:stop:step (stop included); replaces the depth cells of both probes")
    p.add_argument("--distance-per-cell", type=int, default=40, help="items per (length, distance) and probe")
    p.add_argument("--distance-keys", default="4", help="multikey key counts of the fixed-distance items")
    p.add_argument("--distance-tol", type=int, default=gd.DIST_TOL, help="accepted |measured distance - request|, tokens")
    p.add_argument("--update", action="store_true",
                   help="rebuild only --probes inside an existing set: keep the other probes' files and manifest entries "
                        "(the tokenizers and the haystack pool must be the same)")
    p.add_argument("--path-format", default="arrow", choices=g.PATH_FORMATS)
    p.add_argument("--depth-format", default=g.DEPTH_FORMAT, choices=g.DEPTH_FORMATS)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    a.lengths = [int(x) for x in a.lengths.split(",")]
    a.needle_distances = parse_distances(a.needle_distances) if a.needle_distances else []
    a.distance_keys = [int(x) for x in a.distance_keys.split(",") if x]
    a.path_cells = parse_path_cells(a.path_cells)
    a.path_prefixes = a.path_prefixes.split(",")
    a.depth_hops = parse_hops(a.depth_hops)
    a.depth_per_hop = parse_per_hop(a.depth_per_hop, a.depth_hops)
    if a.depth_statements is None:
        a.depth_statements = g.DEPTH_STATEMENTS if a.depth_layout == "equal" else g.FREE_DEPTH_STATEMENTS
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok_paths = dict(kv.split("=", 1) for kv in a.tokenizer)
    formats = {"path_copy": a.path_format, "depth": a.depth_format}
    length_names = a.length_tokenizer or list(tok_paths)
    t0 = time.time()

    docs = read_docs(a.docs)
    verification = []
    if a.verify_rows:
        assert len(a.verify_rows) == len(a.docs) and a.verify_tokenizer, "--verify-rows needs one file per --docs and --verify-tokenizer"
        for rows_path, docs_path in zip(a.verify_rows, a.docs):
            res = verify_rows(rows_path, read_docs([docs_path]), a.verify_tokenizer)
            print(json.dumps(res), flush=True)
            assert res["docs_matching"] == res["docs"], f"{docs_path}: documents differ from {rows_path}"
            verification.append(res)

    # pool counts under the length measure (max over the length tokenizers)
    all_toks = {n: load_tok(pth) for n, pth in tok_paths.items()}
    toks = {n: all_toks[n] for n in length_names}
    keep = [(i, t.strip()) for _, i, t in docs if t.strip() and g.usable_haystack_doc(t)]
    counts = np.zeros(len(keep), dtype=np.int64)
    for n, (tok, _) in toks.items():
        enc = tok.encode_batch([t for _, t in keep], add_special_tokens=False)
        counts = np.maximum(counts, np.array([len(e.ids) for e in enc], dtype=np.int64))
    pool_state = {"texts": [t for _, t in keep], "doc_ids": [i for i, _ in keep], "counts": counts.tolist()}
    pool_info = {"files": {Path(d).name: sha256(Path(d)) for d in a.docs}, "docs": len(docs), "usable": len(keep),
                 "dropped_lookalikes": len(docs) - len(keep), "tokens_length_measure": int(counts.sum()),
                 "median_doc_tokens": float(np.median(counts)), "verified_against_rows": verification}
    print(f"pool: {pool_info['usable']} of {len(docs)} docs, {pool_info['tokens_length_measure']} tokens ({time.time() - t0:.0f}s)", flush=True)

    tok_info = {n: {"path": pth, "sha256": sha256(Path(pth))} for n, pth in tok_paths.items()}
    manifest = {"probe_version": "probes-v1", "base_seed": g.BASE_SEED, "train_len": g.TRAIN_LEN,
                "length_tolerance": g.LENGTH_TOL, "length_measure": f"max token count over {length_names}",
                "formats": formats,
                "tokenizers": tok_info, "haystack_pool": pool_info, "numpy": np.__version__,
                "counts_args": {"uuid_per_depth": a.uuid_per_depth, "multikey_per_cell": a.multikey_per_cell,
                                "path_per_cell": a.path_per_cell, "depth_per_hop": a.depth_per_hop, "lengths": a.lengths},
                "probe_args": {}, "probes": {}, "files": {}}
    tok_manifest = {n: {"tokenizer": tok_info[n], "files": {}, "probes": {}} for n in tok_paths}
    probe_args = {"path_copy": {"cells": [f"{k}:{grid}" for k, grid in a.path_cells], "prefixes": a.path_prefixes,
                                "per_cell": a.path_per_cell, "format": a.path_format},
                  "depth": {"hops": a.depth_hops, "chains": a.depth_chains, "statements": a.depth_statements,
                            "layout": a.depth_layout, "per_hop": a.depth_per_hop, "format": a.depth_format,
                            "final_segment": g.DEPTH_FINAL if a.depth_layout == "equal" else None},
                  "uuid_needle": {"lengths": a.lengths, "per_depth": a.uuid_per_depth},
                  "multikey": {"lengths": a.lengths, "per_cell": a.multikey_per_cell}}
    if a.needle_distances:  # fixed-distance needles instead of depth cells (probe_gen_distance.py)
        per_length = {str(L): gd.distances_for(L, a.needle_distances) for L in a.lengths}
        common = {"lengths": a.lengths, "needle_distances": per_length, "per_cell": a.distance_per_cell,
                  "tol_tokens": a.distance_tol, "distance_tokenizer": length_names[0], "min_head_tokens": gd.MIN_HEAD}
        probe_args["uuid_needle"] = dict(common)
        probe_args["multikey"] = {**common, "n_keys": a.distance_keys}
        manifest["variant"] = "needle-distances"
    if a.update:
        manifest, tok_manifest = merge_base(out, manifest, tok_manifest, a.probes.split(","))
    for probe in a.probes.split(","):
        manifest["probe_args"][probe] = probe_args[probe]

    ctx = get_context("fork")
    with ctx.Pool(a.workers, initializer=_init, initargs=(tok_paths, length_names, pool_state, formats, a.distance_tol)) as workers:
        for probe in a.probes.split(","):
            jobs = jobs_for(probe, a)
            ts = time.time()
            results = workers.map(make_item, jobs, chunksize=max(1, min(16, len(jobs) // (4 * a.workers) or 1)))
            text_path = out / "text" / f"{probe}.jsonl.gz"
            text_path.parent.mkdir(exist_ok=True)
            with gzip.GzipFile(text_path, "wb", mtime=0) as f:
                for r in results:
                    f.write((json.dumps(r["item"], ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            manifest["files"][f"text/{probe}.jsonl.gz"] = sha256(text_path)
            cells = defaultdict(int)
            for r in results:
                cells[g.cell_key(g.TextItem.from_json(r["item"]))] += 1
            manifest["probes"][probe] = {"items": len(results), "cells": dict(cells)}
            for name in tok_paths:
                es = encode_set(probe, results, name, all_toks[name][0], tok_info[name]["sha256"])
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
    failed = {n: {p: d["checks"] for p, d in m["probes"].items() if any(v for k, v in d["checks"].items() if k.endswith("failures"))}
              for n, m in tok_manifest.items()}
    print(json.dumps({"failed_checks": failed}), flush=True)
    assert not any(failed.values()), "per-item checks failed"
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def merge_base(out: Path, manifest: dict, tok_manifest: dict, rebuilt: list[str]) -> tuple[dict, dict]:
    """--update: start from the existing manifests of `out`, keeping every entry of the probes that
    are not rebuilt. The tokenizers and the haystack pool must be the ones the set was built with."""
    old = json.loads((out / "manifest.json").read_text())
    for key in ("base_seed", "train_len", "length_tolerance", "length_measure"):
        assert old[key] == manifest[key], f"--update: {key} differs from the existing set"
    assert {n: t["sha256"] for n, t in old["tokenizers"].items()} == {n: t["sha256"] for n, t in manifest["tokenizers"].items()}, \
        "--update: tokenizers differ from the existing set"
    assert old["haystack_pool"]["files"] == manifest["haystack_pool"]["files"], "--update: haystack documents differ"
    merged = dict(manifest)
    merged["haystack_pool"] = old["haystack_pool"]  # the same files (checked); keeps the EvalRows verification
    merged["probes"] = {p: v for p, v in old["probes"].items() if p not in rebuilt}
    merged["files"] = {f: h for f, h in old["files"].items()
                       if not any(f in (f"text/{p}.jsonl.gz", *[f"{n}/{p}.pt" for n in manifest["tokenizers"]]) for p in rebuilt)
                       and not f.endswith("manifest.json")}
    merged["probe_args"] = {p: v for p, v in old.get("probe_args", {}).items() if p not in rebuilt}
    count_key = {"uuid_needle": "uuid_per_depth", "multikey": "multikey_per_cell", "path_copy": "path_per_cell",
                 "depth": "depth_per_hop"}
    merged["counts_args"] = {**old["counts_args"], **{count_key[p]: manifest["counts_args"][count_key[p]] for p in rebuilt}}
    merged["formats"] = {**old.get("formats", {}), **{p: manifest["formats"][p] for p in rebuilt if p in manifest["formats"]}}
    merged["history"] = old.get("history", []) + [{"rebuilt": rebuilt, "previous_files": {
        f: h for f, h in old["files"].items() if f not in merged["files"] and not f.endswith("manifest.json")}}]
    toks = {}
    for n, tm in tok_manifest.items():
        prev = json.loads((out / n / "manifest.json").read_text())
        toks[n] = {"tokenizer": tm["tokenizer"],
                   "files": {p: h for p, h in prev["files"].items() if p not in rebuilt},
                   "probes": {p: v for p, v in prev["probes"].items() if p not in rebuilt}}
    return merged, toks


def encode_set(probe: str, results: list[dict], name: str, tok, tok_sha: str):
    """The EncodedSet of one probe under tokenizer `name` from the workers' per-item results."""
    import torch

    from ced.eval.probes import EncodedSet, text_sha

    ids, lengths, n_answer, first_info, answer_bytes, items, cands = [], [], [], [], [], [], []
    for r in results:
        it, t = r["item"], r["tok"][name]
        ids.append(t["ids"])
        lengths.append(len(t["ids"]))
        n_answer.append(t["n_answer"])
        answer_bytes.append(len(it["answer"].encode("utf-8")))
        # first answer token that covers a non-whitespace byte (path_copy: the first choice token)
        data, at, first = it["answer"].encode("utf-8"), 0, 0
        for k, nb in enumerate(t["answer_byte_lens"]):
            if data[at : at + nb].strip():
                first = k
                break
            at += nb
        rec = {"id": it["id"], "cell": it["cell"], "group": it.get("group", {}), "text_sha": text_sha(it["context"], it["answer"])}
        if "choice" in t:
            first, rec["choice_options"], rec["choice_share"] = t["choice"]
        first_info.append(first)
        if "needle_depth_tokens" in t:
            rec["needle_depth_tokens"] = round(t["needle_depth_tokens"], 4)
        if "needle_distance_tokens" in t:  # fixed-distance items only (probes-v1 items keep their records)
            rec["needle_distance_tokens"] = t["needle_distance_tokens"]
        if it.get("candidates") is not None:
            cands.append([tok.encode(c, add_special_tokens=False).ids[0] for c in it["candidates"]])
            rec["gold"] = it["gold"]
            rec["distractors"] = [it["candidates"].index(str(d)) for d in it["meta"].get("distractor_digits", [])]
            assert cands[-1][it["gold"]] == int(t["ids"][-1])
        items.append(rec)
    return EncodedSet(
        probe=probe, ids=torch.from_numpy(np.concatenate(ids)),
        offsets=torch.from_numpy(np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)),
        n_answer=torch.tensor(n_answer, dtype=torch.int32), first_info=torch.tensor(first_info, dtype=torch.int32),
        answer_bytes=torch.tensor(answer_bytes, dtype=torch.int32), items=items,
        cand_ids=torch.tensor(cands, dtype=torch.int32) if cands else None,
        info={"tokenizer": name, "tokenizer_sha256": tok_sha})


def describe(results: list[dict], name: str) -> dict:
    """Per cell: items, token lengths (min / mean / max), answer tokens (mean), and check failures."""
    cells = defaultdict(list)
    ks = {}
    for r in results:
        c = g.cell_key(g.TextItem.from_json(r["item"]))
        cells[c].append(r["tok"][name])
        ks[c] = r["item"]["cell"].get("k")
    out = {"cells": {}}
    for c, ts in sorted(cells.items()):
        n = np.array([len(t["ids"]) for t in ts])
        na = np.array([t["n_answer"] for t in ts])
        rec = {"items": len(ts), "tokens_min": int(n.min()), "tokens_mean": round(float(n.mean()), 1), "tokens_max": int(n.max()),
               "answer_tokens_mean": round(float(na.mean()), 2)}
        choices = [t["choice"] for t in ts if "choice" in t]
        if choices:  # path_copy: the first choice token; a copier without binding: share / K
            rec["choice_options_mean"] = round(float(np.mean([o for _, o, _ in choices])), 3)
            rec["choice_share_mean"] = round(float(np.mean([sh for _, _, sh in choices])), 3)
            rec["choice_floor"] = round(float(np.mean([sh / ks[c] for _, _, sh in choices])), 4)
            rec["choice_floor_1_over_options"] = round(float(np.mean([1.0 / o for _, o, _ in choices])), 4)
        depths = [t["needle_depth_tokens"] for t in ts if "needle_depth_tokens" in t]
        if depths:
            rec["needle_depth_tokens_mean"] = round(float(np.mean(depths)), 4)
        out["cells"][c] = rec
    by_dist = defaultdict(list)
    for r in results:
        t, grp = r["tok"][name], r["item"].get("group", {})
        if "distance" in grp and t.get("needle_distance_tokens") is not None:
            by_dist[(r["item"]["cell"]["length"], r["item"]["cell"].get("n_keys", 1), grp["distance"])].append(
                (t["needle_distance_tokens"] - grp["distance"], len(t["ids"])))
    if by_dist:  # fixed-distance items: measured minus requested distance per (length, keys, distance)
        out["distances"] = {f"length={L},n_keys={n},distance={D}": {
            "items": len(v), "error_mean": round(float(np.mean([e for e, _ in v])), 3),
            "error_max_abs": int(max(abs(e) for e, _ in v)), "tokens_mean": round(float(np.mean([m for _, m in v])), 1)}
            for (L, n, D), v in sorted(by_dist.items())}
    all_t = [r["tok"][name] for r in results]
    out["checks"] = {
        "boundary_failures": sum(1 for t in all_t if not t["boundary_ok"]),
        "answer_once_failures": sum(1 for t in all_t if "answer_once" in t and not t["answer_once"]),
        "candidate_failures": sum(1 for t in all_t if "cands_single" in t and not t["cands_single"]),
        "choice_failures": sum(1 for t in all_t if "choice_ok" in t and not t["choice_ok"]),
        "var_token_failures": sum(1 for t in all_t if "var_tokens_ok" in t and not t["var_tokens_ok"]),
        "choice_checked": sum(1 for t in all_t if "choice_ok" in t),
        "var_tokens_checked": sum(1 for t in all_t if "var_tokens_ok" in t),
    }
    if any("distance" in r["item"].get("group", {}) for r in results):
        # fixed-distance items only: a set built without --needle-distances keeps the manifest it always had, key for key
        out["checks"]["distance_failures"] = sum(1 for t in all_t if "distance_ok" in t and not t["distance_ok"])
        out["checks"]["distance_checked"] = sum(1 for t in all_t if "distance_ok" in t)
    out["tokens_total"] = int(sum(len(t["ids"]) for t in all_t))
    return out


if __name__ == "__main__":
    main()
