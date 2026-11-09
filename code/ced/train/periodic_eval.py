"""Evaluation during and after training, on the eval sets named in the config (cfg.eval.sets).

Every set is scored with `ced.eval.span_bpb.evaluate` in the training precision (the same autocast,
the chunked float32 head, the fused GDN kernel when configured). Results, including per-row sums
for paired comparisons, go to <run_dir>/eval/step_<step>[_periodic].json. Multiple-choice per-item
results go next to them, to step_<step>_<set>_items.jsonl.gz.
Under data parallelism with the evals sharded (E1, ced/train/sharded_eval.py; data_parallel.ENV_SHARD_EVALS, off by
default) every rank runs run_eval_sets and scores its block of each set's batches; rank 0 joins them, writes the same
files and returns the same results as one process does (the other ranks write nothing and return {}).

The in-loop lite long-context eval (EvalConfig.lc_every_steps, 0 = off). It gives a trace over steps of three of
the suite's metrics, on a fixed subset of the suite's eval set (EvalConfig.lc_dir), scored with the suite's own
code (ced/eval/long_context.py):

- bpb by position bucket up to L (EvalConfig.lc_length, the stage's row length) on lc_docs `web` and lc_docs
  `code_repo` documents, each cut to its first L + 1 tokens (score_position_bpb, no reference pass);
- assistant bpb at window L on lc_docs `agent` trajectories (score_agent_windows, the one window L);
- with EvalConfig.lc_mf_items > 0, the multi-fact task at length L: the first lc_mf_items items of every (probe,
  k) cell of the suite's multi-fact set (<lc_dir>/multifact/t64k: mf_chain and mf_concat, k = 2 / 4 / 8), scored as
  scripts/eval_long.py scores them (ced.eval.probes.score_sets, probe_gen_multifact.multifact_metrics): exact, fact
  accuracy, nats per fact token and first-token nats, per cell and over all the items ("lc/mf/..."). With 4 items a
  cell is 4 items and the whole part 24: the cells are there to be looked at, the pooled values are what a rule can
  read.

When: at step 0 of a new run, every lc_every_steps optimizer steps, and at the end of the run (step = cfg.steps).
Which documents: `lc_select`, a function of the set and L alone (no random draw); which multi-fact items:
`lc_mf_select`, the same. Every row is padded to L with the suite's inert padding document, and a multi-fact item of
length L is one row of L (checked at the start), so a compiled model sees one eval row shape, the stage's own.
Where the numbers go: one record per evaluation in <run_dir>/metrics.jsonl ({"event": "lc_eval", "step",
"tokens_seen", "lc/..."}; with cfg.metrics_jsonl), the same keys to W&B, and the per-document sums and per-item
multi-fact values in <run_dir>/eval/lc_step_<step>.json. A resumed run that repeats steps writes a step's record
again: the last one wins.

It does not change training (tests/test_lc_curves.py: losses, parameters, optimizer state and RNG states are equal
bit for bit with it on and off): forward only under no_grad, the model in eval mode (so activation checkpointing,
which acts on training forwards with grad, is never used) and back in its mode afterwards, no random draw (the RNG
states are compared and restored all the same), no use of the data stream (a batch is a function of the step). A
wrong configuration (no lc_dir, no document, a length that is not lc_base_len x {1, 2, 4, 8}) stops the run at its
start; an error while scoring (out of memory, say) is logged, recorded as {"event": "lc_eval_failed"} and switches
the lite eval off for the rest of the process: it never stops training. A stop request (SIGTERM) during an in-loop
evaluation abandons it, so the trainer saves at once; the resumed run evaluates that step first.

Memory. An evaluation runs between two training steps, when the step's graph is free, and keeps no tensor: the
selected documents and items stay on the CPU, every score comes back as a CPU tensor or a float. Its forwards are
no-grad rows of the stage's own length at batch 1, so on CUDA they are served from the pool the training step has
already reserved (measured training peaks: 8.2 GiB allocated / 9.8 reserved of 21.6 at 64k with
activation checkpointing; 19.5 / 20.5 at 32k without). The cache is not emptied before an evaluation: the allocator
reuses the free pool, and frees cached blocks by itself before it reports out of memory. After an evaluation that
made the reserved pool grow, the cache is emptied (torch.cuda.empty_cache): the one at step 0 always does, since
nothing has trained yet, and the optimizer state, first allocated by the step after it, must not pin eval-shaped
segments next to a training pool that leaves 1.1 to 1.7 GiB of the slice at 32k (measured). An evaluation that fits
in the pool, the expected case in the loop, releases nothing and costs the next step nothing. What an evaluation
did to the process's memory is recorded with it ("cuda_memory" in the per-step file: before, after, the peaks it
raised, whether the cache was released; the log line says so when it raised the reserved peak), so the first GPU
run measures this estimate.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import time
from pathlib import Path

import torch

from ced.eval.span_bpb import EvalRows, evaluate
from ced.train.fused_head import CHUNK as FUSED_CHUNK

LC_TEXT_SETS = ("web", "code_repo")  # bpb by position bucket up to L
LC_AGENT_SET = "agent"  # assistant bpb at window L
LC_EVENT = "lc_eval"  # the "event" of the trace's records in metrics.jsonl
LC_FAILED = "lc_eval_failed"
# EvalConfig.decoder_at (span_bpb.evaluate's names) -> RowScorer's regime: where CED v2's decoder runs in a row
LC_DECODER_AT = {None: "every", "every": "every", "rows": "scored", "rows_sinks": "scored_sinks"}
LC_MF = "multifact"  # the multi-fact part's name (in "missing", "ids")
LC_MF_SUBDIR = ("multifact", "t64k")  # the suite's multi-fact set under the eval set (scripts/eval_long.py's default)
LC_MF_METRICS = ("exact", "fact_acc", "nats", "first_nats")  # probe_gen_multifact.multifact_metrics' names
# RowScorer's regime -> score_sets' decoder_at for CED v2, as scripts/eval_long.py maps it (its PROBE_REGIME): None is
# score_sets' own choice (the decoder everywhere for a self-KV decoder, at the answers otherwise)
LC_MF_REGIME = {"every": None, "scored": "answers", "scored_sinks": "answers_sinks"}


def _load(path: str):
    """An EvalRows set, or a multiple-choice file (ced/eval/multiple_choice.py; has "items")."""
    data = torch.load(path, weights_only=False)
    if isinstance(data, dict) and "items" in data:
        from ced.eval.multiple_choice import Item
        from ced.eval.span_bpb import RoleDoc

        items = [Item(**it) for it in data["items"]]
        docs = [RoleDoc(**d) for d in data["docs"]]
        return ("mc", items, docs, [tuple(x) for x in data["index"]])
    return EvalRows.load(path)


def _sha256(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _write_items(path, per_item) -> None:
    """The per-item multiple-choice records of one set, one JSON line each."""
    with gzip.open(path, "wt") as f:
        for rec in per_item:
            f.write(json.dumps(rec) + "\n")


def run_eval_sets(trainer, periodic: bool) -> dict:
    ev = trainer.cfg.eval
    if not ev.sets:
        return {}
    # E1 (ced/train/sharded_eval.py): the evals sharded over the data-parallel ranks. Every rank is here, scores its block
    # of each set's batches and sends it to rank 0, which alone joins, writes and returns the results. None: one process,
    # or rank 0 alone while the others wait (Trainer._lead_duty), as before.
    dp = getattr(trainer, "dp", None)
    shard = dp if dp is not None and dp.shard_evals else None
    lead = shard is None or shard.lead
    cache = getattr(trainer, "_eval_rows", None)
    if cache is None:
        cache = trainer._eval_rows = {name: _load(path) for name, path in ev.sets.items()}
    results = {}
    out = trainer.run_dir / "eval"
    if lead:
        out.mkdir(parents=True, exist_ok=True)
    # A last-turn replay window ("turn", "turn128"; scripts/rescore.py --replay-window) reads the <|im_start|> tokens:
    # their id is the chat format's when the set records a tokenizer trained with it, as scripts/eval_long.py decides
    # it; a set without one is refused by evaluate (no guess).
    im_start = {}
    if ev.decoder_at == "rows_sinks_replay" and isinstance(ev.replay_window, str):
        from ced.data.tokenizers import TOKENIZER_VOCAB_BY_SHA256, eval_set_tokenizer
        from ced.eval.long_context import IM_START_ID

        im_start = {name: IM_START_ID for name, path in ev.sets.items() if eval_set_tokenizer(path) in TOKENIZER_VOCAB_BY_SHA256}
    # Resumable re-scoring (scripts/rescore.py --resume; ced/eval/units.py): one unit per eval set, kept under a key of
    # the set's file, the eval settings and the model (the store's base). None: every set is scored, as always.
    store = getattr(trainer, "unit_store", None)
    if shard is not None and store is not None:  # scripts/rescore.py (its only user) is one process
        raise ValueError("the sharded evals (CED_DP_SHARD_EVALS) take no unit store (resumable re-scoring)")
    for name, rows in cache.items():
        unit_key = None
        if store is not None and not periodic:
            unit_key = {"set": name, "sha256": _sha256(ev.sets[name]), "decoder_at": ev.decoder_at,
                        "replay_window": ev.replay_window, "batch_size": ev.batch_size, "head_impl": trainer.cfg.head_impl,
                        "seq_len": trainer.cfg.data.seq_len, "fused_varlen": bool(trainer.fused_varlen),
                        "autocast": str(trainer.autocast_dtype), "im_start": im_start.get(name)}
            cached = store.load(f"set-{name}", unit_key)
            if cached is not None:
                results[name] = cached["result"]
                if cached.get("items") is not None:
                    _write_items(out / f"step_{trainer.step:08d}_{name}_items.jsonl.gz", cached["items"])
                continue
        if isinstance(rows, tuple):  # multiple choice: only in the final eval
            if periodic:
                continue
            from ced.eval.multiple_choice import score_items

            _, items, docs, index = rows
            mc = dict(seq_len=trainer.cfg.data.seq_len, autocast_dtype=trainer.autocast_dtype,
                      fused_varlen=trainer.fused_varlen)
            if shard is None:
                agg, per_item = score_items(trainer.model, items, docs, index, **mc)
            else:
                from ced.train.sharded_eval import score_items_sharded

                got = score_items_sharded(trainer.model, items, docs, index, shard, **mc)
                if got is None:  # not rank 0: its part is with rank 0
                    continue
                agg, per_item = got
            results[name] = {"multiple_choice": agg}
            _write_items(out / f"step_{trainer.step:08d}_{name}_items.jsonl.gz", per_item)
            if unit_key is not None:
                store.save(f"set-{name}", unit_key, {"result": results[name], "items": per_item})
            continue
        if periodic:
            rows = rows[: ev.periodic_max_rows]
        # The reference head always gives the scores. The final eval of a run trained with the fused
        # head also checks that kernel against the reference head on the same hidden states.
        # ev.decoder_at (EvalConfig): None, the default, is evaluate's own choice per arch, as before.
        kw = dict(batch_size=ev.batch_size, chunked_head=True,
                  autocast_dtype=trainer.autocast_dtype, fused_varlen=trainer.fused_varlen,
                  fused_check=not periodic and trainer.cfg.head_impl == "fused", decoder_at=ev.decoder_at,
                  replay_window=ev.replay_window,  # read by decoder_at "rows_sinks_replay" only
                  im_start=im_start.get(name),
                  # the fused check of a run with a bf16 head checks that kernel (the scores do not read it)
                  **({} if trainer.cfg.head_precision == "fp32" else {"head_precision": trainer.cfg.head_precision}),
                  # and at the run's chunk when it is not the fused head's default (TrainConfig.head_chunk)
                  **({} if trainer.cfg.head_chunk == FUSED_CHUNK else {"head_chunk": trainer.cfg.head_chunk}))
        if shard is None:
            r = evaluate(trainer.model, rows, **kw)
        else:
            from ced.train.sharded_eval import evaluate_sharded

            r = evaluate_sharded(trainer.model, rows, shard, **kw)
            if r is None:  # not rank 0: its batches are with rank 0
                continue
        results[name] = r.to_dict()
        if unit_key is not None:
            store.save(f"set-{name}", unit_key, {"result": results[name], "items": None})
    if not lead:
        return {}
    suffix = "_periodic" if periodic else ""
    (out / f"step_{trainer.step:08d}{suffix}.json").write_text(json.dumps(results))
    return results


# --------------------------------------------------------------------------------------------
# the in-loop lite long-context eval (module docstring)
# --------------------------------------------------------------------------------------------


class LCStop(Exception):
    """A stop was requested while the lite eval ran: the evaluation is abandoned and nothing is recorded."""


def lc_length(cfg) -> int:
    """L: EvalConfig.lc_length, or the run's own row length."""
    return int(cfg.eval.lc_length or cfg.data.seq_len)


def lc_due(cfg, step: int) -> bool:
    """Whether the lite eval belongs to `step`: step 0, every lc_every_steps steps, and the end of the run."""
    every = cfg.eval.lc_every_steps
    return bool(every) and (step == 0 or step >= cfg.steps or step % every == 0)


def lc_file(run_dir, step: int) -> Path:
    """The per-document record of the evaluation at `step`; its existence is what "evaluated" means on a resume."""
    return Path(run_dir) / "eval" / f"lc_step_{step:08d}.json"


def lc_select(docs, length: int, n: int, min_tokens: int = 2) -> list[int]:
    """The fixed subset of a set, as indices in the set's order: the first n documents that fill a row of `length`
    (at least length + 1 tokens stored), and when the set has fewer, the longest of the others (ties: the set's
    order). A function of the set and `length` alone: no random draw. Left out: documents of fewer than min_tokens
    tokens, and trajectories the set stores cut shorter than the window needs."""
    whole = [int(m.get("n_tokens", docs.length(i))) for i, m in enumerate(docs.meta)]

    def usable(i: int) -> bool:
        stored = docs.length(i)
        if whole[i] < min_tokens or stored < 2:
            return False
        return docs.kind == "text" or stored == whole[i] or stored >= length + 1

    ok = [i for i in range(len(docs)) if usable(i)]
    fills = [i for i in ok if docs.length(i) >= length + 1][:n]
    rest = sorted((i for i in ok if docs.length(i) < length + 1), key=lambda i: (-docs.length(i), i))
    return sorted(fills + rest[: n - len(fills)])


def lc_mf_select(es, length: int, n: int) -> list[int]:
    """The fixed multi-fact subset of one probe's set (an EncodedSet of probe_gen_multifact items), as indices in the
    set's order: of the items whose cell is `length`, the first n of every k. A function of the set and `length`
    alone: no random draw. Left out: an item of more than `length` tokens (its row would be longer than the stage's)
    and an item without its fact spans (a set not built by tools/build_probes_multifact.py)."""
    taken: dict[int, int] = {}
    keep = []
    for i, it in enumerate(es.items):
        cell = it["cell"]
        if int(cell["length"]) != length or es.length(i) > length or not it.get("fact_spans"):
            continue
        k = int(cell["k"])
        if taken.get(k, 0) < n:
            taken[k] = taken.get(k, 0) + 1
            keep.append(i)
    return keep


class FixedRows:
    """A row scorer (ced.eval.long_context.RowScorer) called with every row padded to `length` positions by the suite's
    inert padding document (make_row's own tail: token 0, document id 1, not valid, not scored), so the model sees
    one shape. The scores of the row's own positions are returned; they do not depend on the padding (causal models:
    a position reads only what is before it). stop: a callable checked before every forward; when it returns true the
    evaluation is abandoned (LCStop)."""

    def __init__(self, scorer, length: int, stop=None):
        self.scorer, self.length, self.stop = scorer, length, stop

    def __call__(self, tokens, targets, doc_ids, valid, scored=None):
        if self.stop is not None and self.stop():
            raise LCStop
        n = tokens.shape[1]
        assert n <= self.length, f"a row of {n} positions in a lite eval at {self.length}"
        if n == self.length:
            return self.scorer(tokens, targets, doc_ids, valid, scored)
        pad = (0, self.length - n)
        f = torch.nn.functional.pad
        out = self.scorer(f(tokens, pad), f(targets, pad), f(doc_ids, pad, value=1), f(valid, pad),
                          None if scored is None else f(scored, pad))
        return out[:, :n]


def score_lc(model, sets: dict, length: int, base: int, autocast_dtype=None, fused_varlen: bool = False,
             decoder_at: str = "every", stop=None) -> dict:
    """The lite eval's numbers for `model` (in eval mode) on `sets` ({name: LongDocs}, already the selected documents):
    for a text set the suite's bucket table up to `length` over the documents cut to their first length + 1 tokens,
    for a chat set the suite's window-`length` record; with the per-document sums of both.

    {"length", "base", "edges", "text": {set: {"docs": per document, "all": per bucket}}, "agent": {"docs", "window",
    "tail"} or None, "row_timing"}. Bucket edges: the suite's (0, base / 4, base, 2 base, ...) up to `length`."""
    from ced.eval.long_context import RowScorer, default_edges, score_agent_windows, score_position_bpb

    edges = tuple(e for e in default_edges(base) if e <= length)
    scorer = RowScorer(model, autocast_dtype=autocast_dtype, fused_varlen=fused_varlen, decoder_at=decoder_at)
    rows = FixedRows(scorer, length, stop)
    out = {"length": length, "base": base, "edges": list(edges), "text": {}, "agent": None}
    for name, docs in sets.items():
        if docs.kind == "text":
            r = score_position_bpb(rows, docs, max_len=length, edges=edges, base=base, ref_window=0)
            out["text"][name] = {"docs": r["docs"], "all": r["all"]}
        else:
            r = score_agent_windows(rows, docs, windows=[length], ref_window=length, base=base)
            assert r["skipped_short"] == 0, "lc_select leaves out the trajectories the suite would skip"
            tail = r["tail_by_cohort"][str(length)]
            out["agent"] = {"docs": r["docs"], "window": r["window"][str(length)],
                            "tail": {"trajectories": tail["trajectories"], "targets": r["tail"],
                                     **{k: tail["by_window"][str(length)][k] for k in ("bpb", "bpb_se")}}}
    out["row_timing"] = scorer.timing_report()
    return out


def score_lc_multifact(model, sets: list, base: int, autocast_dtype=None, fused_varlen: bool = False,
                       decoder_at: str = "every", stop=None) -> dict:
    """The lite eval's multi-fact numbers for `model` on `sets` (EncodedSets, already the selected items), with the
    suite's own scoring and nothing else: ced.eval.probes.score_sets called as scripts/eval_long.py calls it (rows of
    base x {1, 2, 4, 8}, items over base / 2 alone in a row, no tempered scores, CED v2's decoder where the suite runs
    it for this regime), probe_gen_multifact.multifact_metrics per item and its summarize per cell.

    {"items": per item {id, probe, k, layout, exact, fact_acc, nats, first_nats}, "cells": {probe: {"k=<k>": {"n",
    <metric>, <metric>_se}}} (the suite's cell means and item-level SEs), "all": the same over every item (both probes
    and every k pooled: the curves tool's mf.exact and mf.nats), "decoder_at", "timing": score_sets' per row shape}.
    stop: a callable checked before the first forward and after every forward; true abandons the evaluation (LCStop)."""
    from ced.eval import probe_gen_multifact as mf
    from ced.eval.probes import mean_se, score_sets, unwrap

    def check(*_):
        if stop is not None and stop():
            raise LCStop

    check()
    regime = LC_MF_REGIME[decoder_at] if unwrap(model).cfg.arch == "ced_v2" else None
    results, timing = score_sets(model, sets, autocast_dtype=autocast_dtype, fused_varlen=fused_varlen, pack_len=base,
                                 pack_max_item=base // 2, progress=check, taus=(), decoder_at=regime)
    metas = [m for es in sets for m in es.items]
    per_item = [mf.multifact_metrics(r, m) for r, m in zip(results, metas)]
    summary = mf.summarize(results, metas)
    shown = ("n", *LC_MF_METRICS, *(x + "_se" for x in LC_MF_METRICS))
    cells: dict = {}
    for probe, ks in summary.items():
        for k, by_length in ks.items():
            (cell,) = by_length.values()  # one length: lc_mf_select keeps the items of the stage's length only
            cells.setdefault(probe, {})[k] = {m: cell.get(m) for m in shown}
    pooled = {"n": len(per_item)}
    for m in LC_MF_METRICS:
        pooled[m], pooled[m + "_se"] = mean_se(x[m] for x in per_item)
    items = [{"id": r["id"], "probe": r["probe"], "k": int(r["cell"]["k"]), "layout": meta.get("layout"),
              **{m: x[m] for m in LC_MF_METRICS}} for r, meta, x in zip(results, metas, per_item)]
    return {"items": items, "cells": cells, "all": pooled,
            "decoder_at": next(iter({t["decoder_at"] for t in timing.values()}), None), "timing": timing}


def lc_flat(result: dict) -> dict:
    """score_lc's result as the trace's flat keys:
    lc/length; per text set lc/<set>/bpb/<bucket> and lc/<set>/bpb_se/<bucket> (SE over documents) for every bucket
    with tokens, lc/<set>/bpb_last and lc/<set>/bpb_last_se (the last bucket, (L/2, L]: the suite's curve metric) and
    lc/<set>/docs; lc/agent/bpb and lc/agent/bpb_se (assistant bpb at window L, SE over environments),
    lc/agent/tail_bpb (the last base / 2 targets of the trajectories that fill the window), lc/agent/trajectories and
    lc/agent/filled; with the multi-fact part lc/mf/items, lc/mf/<metric> and lc/mf/<metric>_se over every item
    (metric: exact, fact_acc, nats, first_nats; SE over items) and per cell lc/mf/<chain | concat>/k<k>/<metric>;
    lc/seconds when the result carries it."""
    flat = {"lc/length": result["length"]}
    for name, r in result["text"].items():
        for row in r["all"]:
            if row["tokens"]:
                flat[f"lc/{name}/bpb/{row['name']}"] = row["bpb"]
                flat[f"lc/{name}/bpb_se/{row['name']}"] = row["bpb_se"]
        last = r["all"][-1]
        flat[f"lc/{name}/bpb_last"], flat[f"lc/{name}/bpb_last_se"] = last["bpb"], last["bpb_se"]
        flat[f"lc/{name}/docs"] = sum(1 for d in r["docs"] if d["n_scored"] > 0)
    a = result.get("agent")
    if a:
        w = a["window"]
        flat.update({"lc/agent/bpb": w["bpb"], "lc/agent/bpb_se": w["bpb_se"], "lc/agent/tail_bpb": a["tail"]["bpb"],
                     "lc/agent/trajectories": w["trajectories"], "lc/agent/filled": w["filled"]})
    m = result.get(LC_MF)
    if m:
        flat["lc/mf/items"] = m["all"]["n"]
        for key in LC_MF_METRICS:
            for name in (key, key + "_se"):
                if m["all"].get(name) is not None:
                    flat[f"lc/mf/{name}"] = m["all"][name]
        for probe, ks in m["cells"].items():
            for k, cell in ks.items():
                for key in LC_MF_METRICS:
                    if cell.get(key) is not None:
                        flat[f"lc/mf/{probe.removeprefix('mf_')}/{k.replace('=', '')}/{key}"] = cell[key]
    if "seconds" in result:
        flat["lc/seconds"] = result["seconds"]
    return flat


def lc_state(trainer) -> dict:
    """The lite eval's fixed inputs, loaded and checked once per trainer: {"length", "base", "sets": {name: the
    selected documents}, "mf": the selected multi-fact items (EncodedSets, one per probe; empty without
    lc_mf_items), "ids": {name: their ids; "multifact": the items' ids}, "missing": sets without a file or a usable
    document (and "multifact" when lc_mf_items asks for items the eval set does not have at this length), "notes"}.
    ValueError on a configuration that cannot be scored."""
    state = getattr(trainer, "_lc_state", None)
    if state is not None:
        return state
    from ced.data.tokenizers import check_tokenizer
    from ced.eval.long_context import SCORED_ARCHES, LongDocs

    cfg, ev = trainer.cfg, trainer.cfg.eval
    length, base = lc_length(cfg), int(ev.lc_base_len)
    if length not in (base, 2 * base, 4 * base, 8 * base):
        raise ValueError(f"eval.lc_length {length}: one of eval.lc_base_len ({base}) x 1, 2, 4, 8 (the suite's rows)")
    if cfg.model.arch not in SCORED_ARCHES:
        raise ValueError(f"eval.lc_every_steps: the long-context eval scores {SCORED_ARCHES}, not arch {cfg.model.arch!r}")
    if ev.decoder_at not in LC_DECODER_AT:
        raise ValueError(f"eval.decoder_at {ev.decoder_at!r}: one of {tuple(LC_DECODER_AT)}")
    if not ev.lc_dir or ev.lc_docs < 1:
        raise ValueError("eval.lc_every_steps needs eval.lc_dir (the long-context eval set) and eval.lc_docs >= 1")
    if ev.lc_mf_items < 0:
        raise ValueError(f"eval.lc_mf_items {ev.lc_mf_items}: 0 (no multi-fact part) or the items per (probe, k) cell")
    sets, ids, missing, notes = {}, {}, [], []
    for name in (*LC_TEXT_SETS, LC_AGENT_SET):
        f = Path(ev.lc_dir) / f"{name}.pt"
        if not f.exists():
            missing.append(name)
            continue
        docs = LongDocs.load(f)
        notes.append(check_tokenizer(docs.info.get("tokenizer_sha256"), cfg.model.vocab_size, f"lc eval set {name} ({f})"))
        keep = lc_select(docs, length, ev.lc_docs, min_tokens=base + 1 if docs.kind == "chat" else 2)
        if not keep:
            missing.append(name)
            continue
        sub = docs.subset(keep)
        top = int(sub.ids.max())
        if top >= cfg.model.vocab_size:
            raise ValueError(f"{f}: token id {top} is outside the model's vocabulary of {cfg.model.vocab_size}")
        sets[name], ids[name] = sub, [m.get("id") for m in sub.meta]
    if not sets:
        raise ValueError(f"eval.lc_dir {ev.lc_dir}: no document to score in {', '.join((*LC_TEXT_SETS, LC_AGENT_SET))}")
    mf_sets = []
    if ev.lc_mf_items:
        from ced.eval.probes import EncodedSet, plan_rows

        directory = Path(ev.lc_dir).joinpath(*LC_MF_SUBDIR)
        for f in sorted(directory.glob("mf_*.pt")) if directory.is_dir() else []:
            es = EncodedSet.load(f)
            notes.append(check_tokenizer(es.info.get("tokenizer_sha256"), cfg.model.vocab_size, f"lc multi-fact set {f}"))
            sub = es.subset(lc_mf_select(es, length, ev.lc_mf_items))
            if not len(sub):
                continue
            top = int(sub.ids.max())
            if top >= cfg.model.vocab_size:
                raise ValueError(f"{f}: token id {top} is outside the model's vocabulary of {cfg.model.vocab_size}")
            mf_sets.append(sub)
        if not mf_sets:
            missing.append(LC_MF)
        else:  # every item alone in a row of the stage's length, as the suite lays it out: one eval row shape
            odd = sorted({(r.length, r.kind) for r in plan_rows(mf_sets, base, base // 2)} - {(length, "single")})
            if odd:
                raise ValueError(f"eval.lc_mf_items: multi-fact items of length {length} in rows of {odd}, not one row of "
                                 f"{length} each (eval.lc_base_len {base})")
            ids[LC_MF] = [m["id"] for es in mf_sets for m in es.items]
    trainer._lc_state = {"length": length, "base": base, "sets": sets, "mf": mf_sets, "ids": ids, "missing": missing,
                         "notes": notes}
    return trainer._lc_state


def _rng_states(device) -> dict:
    """The RNG states a checkpoint keeps (Trainer.state_dict): torch's, CUDA's and Python's."""
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def _rng_restore(saved: dict, device) -> dict:
    """Put the saved states back; {name: whether the state had moved} (expected: every one False)."""
    now = _rng_states(device)
    touched = {"torch": not torch.equal(now["torch"], saved["torch"]), "python": now["python"] != saved["python"],
               "cuda": saved["cuda"] is not None and any(not torch.equal(a, b) for a, b in zip(now["cuda"], saved["cuda"]))}
    torch.set_rng_state(saved["torch"])
    random.setstate(saved["python"])
    if saved["cuda"] is not None:
        torch.cuda.set_rng_state_all(saved["cuda"])
    return touched


def _cuda_memory(device) -> dict | None:
    """The process's CUDA memory, in GiB: allocated and reserved now, and the peaks so far (read only: the peaks are
    not reset, the training metrics report them too). None off CUDA."""
    if device.type != "cuda":
        return None
    gib = lambda n: round(n / 2**30, 3)  # noqa: E731
    return {"allocated_gib": gib(torch.cuda.memory_allocated(device)), "reserved_gib": gib(torch.cuda.memory_reserved(device)),
            "peak_alloc_gib": gib(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device))}


def run_lc_eval(trainer, log=print, only_if_missing: bool = False, abortable: bool = False) -> dict | None:
    """The lite long-context eval of the trainer's current state, when it is on and due at trainer.step (lc_due).
    only_if_missing: not when this step's record exists (the start of a resumed run, the end of a run). abortable: a
    stop request abandons it (the in-loop calls, which have unsaved steps behind them). Returns the result that was
    recorded, or None (off, not due, already there, abandoned or failed)."""
    cfg = trainer.cfg
    if not cfg.eval.lc_every_steps or getattr(trainer, "_lc_failed", False):
        return None
    state = lc_state(trainer)  # the first call loads and checks: a wrong lc_dir or lc_length stops the run at its start
    step = trainer.step
    path = lc_file(trainer.run_dir, step)
    if not lc_due(cfg, step) or (only_if_missing and path.exists()):
        return None
    from ced.eval.probes import compiled_block_masks

    model, device = trainer.model, trainer.device
    was_training = model.training
    saved = _rng_states(device)
    result = error = None
    touched: dict = {}
    memory = _cuda_memory(device)
    t0 = time.perf_counter()
    try:
        model.eval()
        kw = dict(autocast_dtype=trainer.autocast_dtype, fused_varlen=trainer.fused_varlen,
                  decoder_at=LC_DECODER_AT[cfg.eval.decoder_at], stop=(lambda: trainer._stop_requested) if abortable else None)
        # Compiled layers trace the mask creation themselves, with the global they were compiled with; only an
        # eager flex model needs the compiled mask builder for long rows (ced.eval.probes.compiled_block_masks).
        with compiled_block_masks(device.type == "cuda" and cfg.compile != "all"):
            scored = score_lc(model, state["sets"], state["length"], state["base"], **kw)
            if state["mf"]:
                scored[LC_MF] = score_lc_multifact(model, state["mf"], state["base"], **kw)
        result = scored  # only a whole evaluation is recorded
    except LCStop:
        log(f"lc eval at step {step}: stop requested, abandoned (the resumed run evaluates this step first)")
    except Exception as e:  # noqa: BLE001 - a diagnostic must never stop training
        error = f"{type(e).__name__}: {e}"[:500]
    finally:
        touched = _rng_restore(saved, device)
        model.train(was_training)
        after = _cuda_memory(device)
        # CUDA: an evaluation (or its failure) that made the reserved pool grow gives the cache back, so that no
        # eval-shaped segment stays next to the training pool (module docstring, "Memory")
        released = memory is not None and (error is not None or after["reserved_gib"] > memory["reserved_gib"])
        if released:
            torch.cuda.empty_cache()
    tokens_seen = step * cfg.data.batch_size * cfg.data.seq_len
    if error is not None:
        trainer._lc_failed = True
        log(f"lc eval at step {step} FAILED ({error}); switched off for the rest of this process, training continues")
        if cfg.metrics_jsonl:
            trainer._write_jsonl("metrics.jsonl", {"event": LC_FAILED, "step": step, "tokens_seen": tokens_seen, "error": error})
        return None
    if result is None:
        return None
    result.update(step=step, tokens_seen=tokens_seen, seconds=time.perf_counter() - t0, ids=state["ids"],
                  missing=state["missing"], rng_touched=touched)
    if memory is not None:  # CUDA: what the evaluation did to the process's memory (module docstring, "Memory")
        result["cuda_memory"] = {"before": memory, "after": after, "cache_released": released,
                                 "peak_alloc_raised_gib": round(after["peak_alloc_gib"] - memory["peak_alloc_gib"], 3),
                                 "peak_reserved_raised_gib": round(after["peak_reserved_gib"] - memory["peak_reserved_gib"], 3)}
    flat = lc_flat(result)
    if cfg.metrics_jsonl:  # the trace's record first: were the process to die here, the resumed run scores this step again
        trainer._write_jsonl("metrics.jsonl", {"event": LC_EVENT, "step": step, "tokens_seen": tokens_seen, **flat})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result))
    os.replace(tmp, path)
    trainer._wandb_call("log", flat, step=step, log=log)
    num = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
    raised = (result.get("cuda_memory") or {}).get("peak_reserved_raised_gib")
    log(f"lc eval step {step} (L {state['length']}): "
        + "  ".join(f"{name} {num(flat.get(f'lc/{name}/bpb_last'))}" for name in result["text"])
        + (f"  agent {num(flat.get('lc/agent/bpb'))}" if result.get("agent") else "")
        + (f"  mf exact {num(flat.get('lc/mf/exact'))} nats {num(flat.get('lc/mf/nats'))} ({flat['lc/mf/items']} items)"
           if result.get(LC_MF) else "")
        + f"  {result['seconds']:.1f}s"
        + (f"  missing: {', '.join(state['missing'])}" if state["missing"] else "")
        + (f"  reserved peak +{raised:.2f} GiB" if raised else ""))
    return result
