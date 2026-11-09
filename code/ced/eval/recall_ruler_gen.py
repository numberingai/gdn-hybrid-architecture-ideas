"""Greedy generation for the RULER subset (ced/eval/recall_ruler.py): the loop of ced/eval/qa_probe.py greedy_continuation (left-padded prompts, the padding a
document of its own, prefill, then one token at a time through the model's cache: `prefill` / `extend` of the control arch, the reference code that is checked
against the full forward) with the changes that 8,192-token prompts and a 64,000-entry vocabulary need:

- the head is applied to the LAST position only (greedy_continuation's `prefill` returns the logits of every position: B x 8,192 x 64,000 float32 is 2.1 GB a row);
- the prompt is prefilled in chunks (`prefill_chunk` tokens, default 1,024) through the same cache the single-token steps use, so the reference attention's
  score tensor is chunk x cache length and not 8,192 x 8,192 (the cache is chunk-size invariant);
- a row stops as soon as it has produced the end-of-document token (it is cut before it, as a Hugging Face `generate` stops at the eos token) or, checked
  every `check_every` steps on the host, as soon as `is_done` says the score cannot change any more: RULER's recall is the share of the references that
  occur in the generated text, which only grows with the text, so a row whose references are all present has its final score (the text kept is the
  tokens up to that check, deterministic for a row; the rest of the batch goes on). Nothing else stops a row before `max_new` (RULER sets no stop words).

The tokens are those of greedy_continuation on the same prompts with the same number of steps (equal in float64 on tiny
models): the loop is its loop with a cheaper head and a bounded prefill.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import time
from collections import defaultdict
from typing import Callable

import torch

from ced.eval import recall_ruler as RR
from ced.eval.qa_probe import left_pad
from ced.model.looped_ced import lm_head

PREFILL_CHUNK = 1024
CHECK_EVERY = 8


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def generate_greedy(model, prompts: list[list[int]], max_new: int, *, is_done: Callable[[int, list[int]], bool] | None = None, eos_id: int | None = None,
                    prefill_chunk: int = PREFILL_CHUNK, check_every: int = CHECK_EVERY, pad_id: int = 0, autocast_dtype=None) -> tuple[list[list[int]], dict]:
    """Greedy continuation of `prompts` (lists of token ids; the model is a `control` arch, eval mode, on its device). Returns (generated token ids per prompt,
    stats): each list has at most max_new tokens and stops before the first `eos_id`; stats = {"rows", "prompt_tokens" (padded width x rows), "prefill_s",
    "decode_s", "decode_steps"}. is_done(row, tokens so far) is asked every check_every steps for the rows still going."""
    if getattr(model, "cp_head", None) is not None:
        raise ValueError("the loop applies the bare head: a Copy-Paste model's distribution is the mixture")
    device = next(model.parameters()).device
    tokens, docs = left_pad(prompts, pad_id, device)
    B, L = tokens.shape
    ctx = torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None)
    stats = {"rows": B, "prompt_tokens": B * L, "prefill_s": 0.0, "decode_s": 0.0, "decode_steps": 0}
    with ctx:
        _sync(device)
        t0 = time.time()
        cache, out = None, None
        for a in range(0, L, prefill_chunk):
            b = min(a + prefill_chunk, L)
            out = model(tokens[:, a:b], docs[:, a:b], None, cache=cache, return_cache=True, return_hidden=True)
            cache = out.cache
        logits = lm_head(out.hidden_aux[:, -1:], model.head_weight())[:, 0]
        tok = logits.argmax(-1)
        _sync(device)
        t1 = time.time()
        stats["prefill_s"] = t1 - t0
        gen = []
        stop: list[int | None] = [None] * B  # the number of tokens a row keeps once it has stopped
        for i in range(max_new):
            gen.append(tok)
            n = i + 1
            if n == max_new:
                break
            if n % check_every == 0:
                cols = torch.stack(gen, 1).tolist()
                for r in range(B):
                    if stop[r] is not None:
                        continue
                    row = cols[r]
                    if eos_id is not None and eos_id in row:
                        stop[r] = row.index(eos_id)
                    elif is_done is not None and is_done(r, row):
                        stop[r] = n
                if all(s is not None for s in stop):
                    break
            out = model.extend(tok[:, None], docs[:, -1:], None, cache)
            cache = out.cache
            tok = out.logits[:, -1].argmax(-1)
            stats["decode_steps"] += 1
        _sync(device)
        stats["decode_s"] = time.time() - t1
    cols = torch.stack(gen, 1).tolist()
    result = []
    for r in range(B):
        row = cols[r][: stop[r]] if stop[r] is not None else cols[r]
        if eos_id is not None and eos_id in row:
            row = row[: row.index(eos_id)]
        result.append(row)
    return result, stats


def _cat_leaf(objs: list):
    """The batch-dimension concatenation of one cache field across caches: tensors by torch.cat on dim 0, dataclasses and lists field by field, None stays None,
    anything else (the pass count T, flags) must be the same in every cache."""
    first = objs[0]
    if first is None:
        assert all(o is None for o in objs), "a cache field is set in one cache and not in another"
        return None
    if isinstance(first, torch.Tensor):
        return torch.cat(objs, 0)
    if dataclasses.is_dataclass(first) and not isinstance(first, type):
        return type(first)(**{f.name: _cat_leaf([getattr(o, f.name) for o in objs]) for f in dataclasses.fields(first)})
    if isinstance(first, (list, tuple)):
        assert all(len(o) == len(first) for o in objs), "caches of different depth"
        return type(first)(_cat_leaf(list(z)) for z in zip(*objs))
    assert all(o == first for o in objs), f"a non-tensor cache field differs between the caches: {objs}"
    return first


def cat_caches(caches: list):
    """One ControlCache (ced/model/control.py) of the rows of several caches, in order: every tensor concatenated on the batch dimension. The caches must have been
    built over the same width (left-padded prompts of one length) so that the key sets and the sequence indices line up; nothing in a cache is shared between rows, so
    the rows of the result decode exactly as they did in their own cache."""
    assert caches, "no caches"
    return caches[0] if len(caches) == 1 else _cat_leaf(caches)


@torch.no_grad()
def generate_greedy_batched(model, prompts: list[list[int]], max_new: int, *, is_done: Callable[[int, list[int]], bool] | None = None, eos_id: int | None = None,
                            prefill_chunk: int = PREFILL_CHUNK, check_every: int = CHECK_EVERY, pad_id: int = 0, autocast_dtype=None, prefill_rows: int | None = None,
                            margins: bool = False) -> tuple[list[list[int]], dict]:
    """generate_greedy (the reference loop, which stays as it is) with the two phases batched on their own terms. The prompts are left-padded to ONE width and the prefill
    runs `prefill_rows` rows at a time (the memory of a chunk's attention scores and states bounds it; default: all rows), each sub-batch's cache is kept, and the caches are
    concatenated on the batch dimension so that the decode steps run over ALL the rows at once: a decode step is a few hundred small kernels launched from Python whatever the
    number of rows (the time of a step was measured the same for 4 and 16 rows), so the step costs about the same for 64 rows as
    for 4. Rows are independent in every layer (a row's attention, state and conv read only its own cache), so the tokens are those of generate_greedy up to the rounding of
    the batch shape (bit for bit in float64). The early stops are the same rule, per row; the loop ends when every row has stopped.
    margins: also return, per row and generated token, the gap between the largest and the second largest logit at the position that chose it (stats["margins"], a list of
    float lists, collected on the device and copied once at the end: no host sync a step). Returns (tokens per prompt, stats) with stats as generate_greedy's plus
    "prefill_rows"."""
    if getattr(model, "cp_head", None) is not None:
        raise ValueError("the loop applies the bare head: a Copy-Paste model's distribution is the mixture")
    device = next(model.parameters()).device
    tokens, docs = left_pad(prompts, pad_id, device)
    B, L = tokens.shape
    pr = B if not prefill_rows else max(1, min(prefill_rows, B))
    ctx = torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None)
    stats = {"rows": B, "prompt_tokens": B * L, "prefill_s": 0.0, "decode_s": 0.0, "decode_steps": 0, "prefill_rows": pr}

    def gap_of(logits):
        top = logits.float().topk(2, dim=-1).values
        return top[:, 0] - top[:, 1]

    with ctx:
        _sync(device)
        t0 = time.time()
        caches, firsts, first_gaps = [], [], []
        for r0 in range(0, B, pr):
            cache, out = None, None
            for a in range(0, L, prefill_chunk):
                b = min(a + prefill_chunk, L)
                out = model(tokens[r0: r0 + pr, a:b], docs[r0: r0 + pr, a:b], None, cache=cache, return_cache=True, return_hidden=True)
                cache = out.cache
            logits = lm_head(out.hidden_aux[:, -1:], model.head_weight())[:, 0]
            firsts.append(logits.argmax(-1))
            if margins:
                first_gaps.append(gap_of(logits))
            caches.append(cache)
            del out
        cache = cat_caches(caches)
        del caches
        tok = torch.cat(firsts)
        gaps = [torch.cat(first_gaps)] if margins else []
        _sync(device)
        t1 = time.time()
        stats["prefill_s"] = t1 - t0
        gen = []
        stop: list[int | None] = [None] * B
        for i in range(max_new):
            gen.append(tok)
            n = i + 1
            if n == max_new:
                break
            if n % check_every == 0:
                cols = torch.stack(gen, 1).tolist()
                for r in range(B):
                    if stop[r] is not None:
                        continue
                    row = cols[r]
                    if eos_id is not None and eos_id in row:
                        stop[r] = row.index(eos_id)
                    elif is_done is not None and is_done(r, row):
                        stop[r] = n
                if all(s is not None for s in stop):
                    break
            out = model.extend(tok[:, None], docs[:, -1:], None, cache)
            cache = out.cache
            step_logits = out.logits[:, -1]
            tok = step_logits.argmax(-1)
            if margins:
                gaps.append(gap_of(step_logits))
            stats["decode_steps"] += 1
        _sync(device)
        stats["decode_s"] = time.time() - t1
    cols = torch.stack(gen, 1).tolist()
    if margins:
        stats["margins"] = torch.stack(gaps, 1).tolist()
    result = []
    for r in range(B):
        row = cols[r][: stop[r]] if stop[r] is not None else cols[r]
        if eos_id is not None and eos_id in row:
            row = row[: row.index(eos_id)]
        result.append(row)
    return result, stats


@torch.no_grad()
def full_forward_argmax(model, ids: list[int], positions: list[int], *, pad_id: int = 0, autocast_dtype=None) -> tuple[list[int], list[float]]:
    """The full forward over one row `ids` (a single document): for each position p of `positions`, the argmax of the next-token distribution at p and the
    gap between the largest and the second-largest logit there (the margin: a flip with a margin near the precision's own rounding is a tie, not a leak)."""
    device = next(model.parameters()).device
    toks, docs = left_pad([ids], pad_id, device)
    ctx = torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None)
    with ctx:
        out = model(toks, docs, None, return_hidden=True)
        logits = lm_head(out.hidden_aux[:, positions], model.head_weight())[0].float()
    top2 = logits.topk(2, dim=-1)
    return top2.indices[:, 0].tolist(), (top2.values[:, 0] - top2.values[:, 1]).tolist()


@torch.no_grad()
def explain_divergence(model, prompt: list[int], a: list[int], b: list[int], autocast_dtype=None, reference: bool = True, eos_id: int | None = None) -> dict | None:
    """Two greedy generations of ONE prompt (the reference loop's and the batched loop's): None when they are the same list; else the FIRST position j where they part and
    what separates the two tokens there, from one full forward over prompt + the common prefix (the position that chose token j): `gap` = |logit(a_j) - logit(b_j)| in the
    forward the scorer runs (autocast), `gap_ref` the same with autocast off, `top` the largest logit and `spacing` its bf16 spacing, `eps` the largest |autocast logit -
    reference logit| over the five largest reference logits (ced.eval.recall_based.check_cached_generation's noise, measured at this position). A parting at a near tie shows
    as gap <= BOUND_FACTOR x eps; the tokens after j are not compared (the sequences are different from there on). A generation that is a strict prefix of the other is reported with
    `prefix_only`: the generator cuts a row before its end-of-document token, so the shorter list ended at a position where its loop chose that token and the other loop
    another; with `eos_id` the gap between the two is computed at that position like any other part (without it the part has no gap and counts as unexplained)."""
    from ced.eval.recall_based import bf16_spacing, full_logits

    if a == b:
        return None
    j = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    prefix_only = j is None
    if prefix_only:
        # One list is a strict prefix of the other: the generator cuts a row before its end-of-document token, so the shorter list ended where the token that the other
        # list continues with was, in the shorter one's loop, the end-of-document token (a near tie between the two). With eos_id the gap is computed at that position.
        j = min(len(a), len(b))
        if eos_id is None or j >= max(len(a), len(b)):
            return {"first": j, "prefix_only": True, "len_a": len(a), "len_b": len(b)}
        a_tok, b_tok = (eos_id, b[j]) if len(a) < len(b) else (a[j], eos_id)
        a, b = list(a[:j]) + [a_tok], list(b[:j]) + [b_tok]
    ids, pos = list(prompt) + list(a[:j]), [len(prompt) + j - 1]
    lg = full_logits(model, ids, pos, autocast_dtype)[0]
    ref = lg
    if autocast_dtype is not None and not reference:  # the noise is measured on some positions only (summarize_divergences takes the largest): this one has the gap alone
        ref = None
    elif autocast_dtype is not None:
        try:
            ref = full_logits(model, ids, pos, None)[0]
        except torch.cuda.OutOfMemoryError:  # the float32 forward over 8k does not always fit beside the cache of the run: the noise of this position is then unknown (None)
            gc.collect()
            torch.cuda.empty_cache()
            ref = None
    top = float(lg.max())
    if ref is None:
        eps, gap_ref = None, None
    else:
        near = ref.topk(5).indices
        eps = float((lg[near] - ref[near]).abs().max()) if autocast_dtype is not None else 0.0
        gap_ref = abs(float(ref[a[j]] - ref[b[j]]))
    return {"first": j, "tok_a": a[j], "tok_b": b[j], "gap": abs(float(lg[a[j]] - lg[b[j]])), "gap_ref": gap_ref, "top": top, "spacing": bf16_spacing(top), "eps": eps,
            "prefix_only": prefix_only}


def summarize_divergences(divs: list) -> dict:
    """The explain_divergence records of the items that parted (None entries are the items that did not): how many, the noise (the largest eps), the bound
    (BOUND_FACTOR x noise, as check_cached_generation) and how many have a gap within it."""
    from ced.eval.recall_based import BOUND_FACTOR

    parted = [d for d in divs if d is not None]
    noise = max((d["eps"] for d in parted if d.get("eps") is not None), default=0.0)
    bound = BOUND_FACTOR * noise
    for d in parted:
        d["within_bound"] = "gap" in d and d["gap"] <= bound  # a prefix-only part without a gap (no end-of-document id given) is not explained
    return {"items": len(divs), "parted": len(parted), "noise": noise, "bound": bound, "within_bound": sum(1 for d in parted if d["within_bound"]),
            "beyond_bound": sum(1 for d in parted if not d["within_bound"]), "max_gap": max((d["gap"] for d in parted if "gap" in d), default=None), "parted_items": parted}


# --------------------------------------------------------------------------------------------
# scoring a list of items with resume (scripts/recall_ruler_score.py is the command line)
def load_results(path, fingerprint: dict) -> dict:
    """The records of an earlier (stopped) pass: {item key: record}. The file is JSON lines, the first the header {"fingerprint": ...}. A file with another
    fingerprint (another checkpoint, settings or code) is moved aside (<path>.stale-<n>) and nothing is read from it; a last line cut by a kill is ignored."""
    path = str(path)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    try:
        header = json.loads(lines[0])
    except (ValueError, IndexError):
        header = None
    if not isinstance(header, dict) or header.get("fingerprint") != fingerprint:
        n = 1
        while os.path.exists(f"{path}.stale-{n}"):
            n += 1
        os.replace(path, f"{path}.stale-{n}")
        return {}
    out = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # the line a kill cut
        out[rec["key"]] = rec
    if lines[-1].strip():  # the file does not end with a newline: the next record would be glued to the cut line
        _append(path, "\n")
    return out


def read_records(path) -> dict:
    """The records of a results file, whatever its fingerprint (the values-only reading needs no checkpoint): {item key: record}; a line a kill cut is ignored."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    out = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        out[rec["key"]] = rec
    return out


def _append(path, text: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


class BudgetExceeded(RuntimeError):
    """The soft time budget ended before the next batch: the records written so far stay, a rerun goes on from them."""


def split_batches(n_tokens: int, count: int, max_batch_tokens: int) -> int:
    """Rows a batch of prompts `n_tokens` wide may hold within the token budget (at least one)."""
    return max(1, min(count, max_batch_tokens // max(n_tokens, 1)))


def score_items(model, tok, items: list, *, results_path, fingerprint: dict, eos_id: int | None = None, max_batch_tokens: int = 32768,
                prefill_chunk: int = PREFILL_CHUNK, check_every: int = CHECK_EVERY, autocast_dtype=None, pad_id: int = 0, length_check: bool = True,
                generate: Callable = generate_greedy, log: Callable = print, oom_errors: tuple = (torch.cuda.OutOfMemoryError,),
                stop_at: float | None = None, decode_rows: int | None = None) -> tuple[list[dict], dict]:
    """Greedy-generate for every item (ced.eval.recall_ruler.RulerItem) and score it with RULER's string match. Items already in `results_path` (the same
    fingerprint) are not run again; each batch's records are appended and synced as it ends, so a killed pass loses at most the batch in flight. Batches are
    formed within a (task, length) cell, in index order, at most max_batch_tokens / widest prompt rows. A batch that runs out of memory is split in two
    (down to one row). decode_rows (the batched loop, generate_greedy_batched): a batch holds that many rows (the decode steps run over all of them), the prefill inside it
    `max_batch_tokens / widest prompt` rows at a time; None (the reference loop): the rows a batch is prefilled with are the rows it decodes with. Returns (records of every item in `items` order, timing per length of this process: items, prompt_tokens, prefill_s, decode_s,
    decode_steps, batches, oom_splits); BudgetExceeded when `stop_at` (an epoch time) passes between two batches. A record: key, task, length, index, recall, all_found, n_gen, n_prompt_tokens, pred, gen_ids."""
    path = str(results_path)
    done = load_results(path, fingerprint)
    if not os.path.exists(path):
        _append(path, json.dumps({"fingerprint": fingerprint}, sort_keys=True) + "\n")
    cells: dict = defaultdict(list)
    for it in items:
        if it.key not in done:
            cells[(it.length, it.task)].append(it)
    timing: dict = defaultdict(lambda: {"items": 0, "prompt_tokens": 0, "prefill_s": 0.0, "decode_s": 0.0, "decode_steps": 0, "batches": 0, "oom_splits": 0})

    def run(sub: list[tuple]) -> tuple[list[list[int]], dict]:
        its = [x[0] for x in sub]
        refs = [it.refs for it in its]
        extra = {} if decode_rows is None else {"prefill_rows": split_batches(max(len(x[1]) for x in sub), len(sub), max_batch_tokens)}
        try:
            return generate(model, [x[1] for x in sub], its[0].max_new, is_done=lambda r, ids: RR.recall(tok.decode(ids), refs[r]) == 1.0, eos_id=eos_id,
                            prefill_chunk=prefill_chunk, check_every=check_every, pad_id=pad_id, autocast_dtype=autocast_dtype, **extra)
        except oom_errors:
            if len(sub) == 1:
                raise
        # outside the handler: inside it the exception's traceback still holds the failed call's tensors (the cache), and the retry would not find the memory free
        gc.collect()
        torch.cuda.empty_cache()
        h = len(sub) // 2
        a, sa = run(sub[:h])
        b, sb = run(sub[h:])
        st = {k: sa.get(k, 0) + sb.get(k, 0) for k in set(sa) | set(sb)}
        st["oom_splits"] = sa.get("oom_splits", 0) + sb.get("oom_splits", 0) + 1
        return a + b, st

    for (length, task), cell in sorted(cells.items(), key=lambda kv: (-kv[0][0], kv[0][1])):  # the longest first: the costliest and least known cells
        enc = []
        for it in cell:
            ids = tok.encode(it.prompt, add_special_tokens=False).ids
            if length_check and len(ids) != it.n_prompt_tokens:
                raise RuntimeError(f"{it.key}: the prompt is {len(ids)} tokens under this tokenizer, the builder counted {it.n_prompt_tokens}")
            if len(ids) + it.max_new > it.length:
                raise RuntimeError(f"{it.key}: {len(ids)} + {it.max_new} tokens exceed the length {it.length}")
            enc.append((it, ids))
        width = max(len(ids) for _, ids in enc)
        B = split_batches(width, len(enc), max_batch_tokens) if decode_rows is None else max(1, min(decode_rows, len(enc)))
        for a in range(0, len(enc), B):
            if stop_at is not None and time.time() > stop_at:
                raise BudgetExceeded(f"the time budget ended before {task} len={length} item {a} of {len(enc)} still to do; {len(done)} records are kept")
            sub = enc[a: a + B]
            gens, st = run(sub)
            lines = []
            for (it, _), ids in zip(sub, gens):
                pred = tok.decode(ids)
                rec = {"key": it.key, "task": it.task, "length": it.length, "index": it.index, "recall": RR.recall(pred, it.refs),
                       "all_found": RR.all_found(pred, it.refs), "n_gen": len(ids), "n_prompt_tokens": it.n_prompt_tokens, "pred": pred, "gen_ids": ids}
                done[it.key] = rec
                lines.append(json.dumps(rec, sort_keys=True))
            _append(path, "\n".join(lines) + "\n")
            t = timing[length]
            t["items"] += len(sub)
            t["batches"] += 1
            t["oom_splits"] += st.get("oom_splits", 0)
            for k in ("prompt_tokens", "prefill_s", "decode_s", "decode_steps"):
                t[k] += st[k]
            n_have = [r["recall"] for r in done.values() if r["task"] == task and r["length"] == length]
            log(f"[recall-ruler] {task} len={length}: {len(n_have)} items, recall {sum(n_have) / len(n_have):.3f}  (batch {len(sub)} rows: prefill {st['prefill_s']:.1f}s, "
                f"{st['decode_steps']} steps {st['decode_s']:.1f}s)")
    return [done[it.key] for it in items], {int(k): dict(v) for k, v in timing.items()}
