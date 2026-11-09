"""The Based recall suite (SWDE, FDA, SQuAD completion) as an evaluation of the 200M models. EVALUATION ONLY: nothing here is ever trained on.

THE PROTOCOL IS THE REFERENCE'S (Arora et al. 2024, "Simple linear attention language models balance the recall-throughput tradeoff",
arXiv 2402.18668; the tasks of https://github.com/HazyResearch/based-evaluation-harness at commit 22c085628045, a clone of EleutherAI's
lm-evaluation-harness, MIT licence, Copyright (c) 2020 EleutherAI; the Gated DeltaNet paper and JRT report the same three sets with the same harness):

  prompt     doc["text"], as the dataset gives it (lm_eval/tasks/{swde,fda,squad_completion}/task.py, doc_to_text, line 36 / 32 / 32)
  target     doc["value"]
  request    generate_until, arguments=(ctx, {"until": ["\\n"], "max_gen_toks": 48})        (task.py line 58 / 54 / 54)
  decoding   greedy (HFLM._model_generate: temperature 0.0 gives do_sample=False; lm_eval/models/huggingface.py lines 741-749), at most 48 new
             tokens, stopped at the first "\\n"; the generation is cut at the first "\\n" afterwards too (huggingface.py line 1181)
  input      left-truncated to its last max_length - max_gen_toks tokens (huggingface.py lines 1143-1151; the wrapper's max_length is 2048,
             lm_eval/models/based_lm.py line 69), so a long document loses its beginning and keeps its end, where the question is
  metric     "contains": 1 when the gold value, regex-escaped, is found in the generation by a case-insensitive search (task.py contains_score,
             line 102 / 98 / 98), the mean over the items. Nothing else is normalised: no stripping, no punctuation or article removal.

WHAT DIFFERS HERE: the tokenizer (the 64k BPE of the 200M models, no BOS: add_special_tokens=False), the 48 new tokens and the
left truncation at this project's training length (8,192 tokens, so a prompt keeps its last 8,192 - 48 = 8,144 tokens; a longer one would lose its
beginning, the reference's rule at this length), and the generation path: ced.eval.qa_probe.greedy_continuation, the helper the long-context grounded-QA
probe uses (LoopedCED.generate for looped models; prefill and extend of the cache for the control, which is every 200M model), on the repo's
slow readable reference model (GDN kernel "chunked", attention backend "reference", the trained precision: scripts/recall_based_score.py
load_reference_model), whose cached generation is held equal to the full forward. The fixed 48 steps run for every batch: the
generation is cut at the first "\\n" afterwards, which is what the reference's stop does to the text.

ONE LABELLED CHANGE: the generation is also cut BEFORE the first <|endoftext|> (EOS_ID 0, the token that closes every
non-chat training document). The harness's HFLM generates with transformers' generate, which ends a row at its end-of-sequence token; the fixed 48 steps
here would go on after it, into a "new document" that still sees the passage and can copy the gold from it (a false hit the reference cannot make). Whether
the Based wrapper's own model class stops at EOS was not determinable from the four harness files read, so the cut is a choice, not a copy: every record
keeps `hit_no_eos_cut` (the hit with only the newline stop) and `eos_at`, and the task summary counts both, so the effect is a number and not an argument.

Three sets, one items file each (`<task>.items.jsonl.gz`, raw text: the tokenizer is applied at scoring), one manifest (`manifest.json`):

    task    source (split)                                    items   gold                       licence
    squad   hazyresearch/based-squad (validation)             2,984   the answer span            SQuAD 2.0: CC BY-SA 4.0
    swde    hazyresearch/based-swde (validation)              1,111   the attribute value        no licence stated
    fda     hazyresearch/based-fda (validation)               1,102   the attribute value        no licence stated

The items are downloaded for scoring and never redistributed; tools/data/build_recall_based.py fetches the pinned files (`SOURCES`).

THE ITEMS OF SWDE AND FDA: the prompt is the document (an HTML page as text for SWDE, a 2,000-character-step chunk of an FDA 510(k) summary for FDA) and ends
with the attribute name and a colon ("...\\nyear:"), so the question is the LAST line and a left truncation keeps it; SWDE prompts also carry a block
"Summary of information above..." with other attributes of the same page as "key: value" lines. The gold is `value`. The reference's metric is the same
"contains" on the first generated line.

Every item has a per-item sha256 and every task a set sha256 (`item_sha256`, `set_sha256`): a list for contamination checks and the integrity check of the
item file (`load_set` recomputes them).

OUTPUT (the common values format of the recall builds): per scored checkpoint one `<out>/<run>.values.json` =
{"run", "family": "based", "code_sha", "cells": {"based|<task>|all|acc": float}, "n_items": {<cell>: int}, "excluded": ["task:id", ...]}; per
(run, task) `<out>/<run>/<task>.items.jsonl.gz` (one line an item: id, hit, n_tokens, truncated, generation, eos_at, hit_no_eos_cut) and `<task>.json` (the key the
task was scored under, seconds, tokens). The scorer scores ALL items and the values take an exclusion list afterwards: `values_json` with an
`exclude` set of "task:id" strings, so a later list needs no new scoring pass (scripts/recall_based_score.py values).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

VERSION = "recall-based-v1"
FAMILY = "based"
TASKS = ("swde", "fda", "squad")
CELL_METRIC = "acc"                      # the cell "based|<task>|all|acc": the share of items whose generation contains the gold (the reference's "contains")
MAX_GEN_TOKS = 48                        # task.py: {"until": ["\n"], "max_gen_toks": 48}
STOP = ("\n",)                           # task.py: "until"
EOS_ID = 0                               # <|endoftext|> of the 64k tokenizer (id 0): the end of a document; the generation stops before it (module docstring, "ONE LABELLED CHANGE")
TRAIN_LENGTH = 8192                     # the 200M models' seq_len
REFERENCE_MAX_LENGTH = 2048              # lm_eval/models/based_lm.py line 69 (GPT-2 tokens)
FINAL_STEP_200M = 7996                   # the final step of every 200M run (steps: 7996)
HARNESS_REPO = "https://github.com/HazyResearch/based-evaluation-harness"
HARNESS_COMMIT = "22c085628045ed5898bdff33a8524286daecddbc"
HARNESS_FILES_SHA256 = {                 # the reference files this module follows, sha256 of the bytes at HARNESS_COMMIT
    "lm_eval/models/huggingface.py": "0b59a637376e9b767b2ce2fad1b511bbcd6c38039caf67a4dd9e1e5860553afd",
    "lm_eval/tasks/swde/task.py": "abcde9b4691e2185208f4e2d38a71cfc95131c6822868ec733ad5be55d98e0ad",
    "lm_eval/tasks/fda/task.py": "4a6590b4d9f4abac475e8bba4e5fb8204e64069b518e005888ea4104eccffd78",
    "lm_eval/tasks/squad_completion/task.py": "8c848fc1874ea24c9faf4f5e7b070e32066853f7b76ed53ea65a68fbf4791399",
}

# Pinned sources (the Hugging Face hub). sha256 = of the file at the revision (the hub's LFS oid); "rows" = the rows of the file.
# swde: the harness names "hazyresearch/based-swde-v2"; the hub redirects it to "hazyresearch/based-swde" (a renamed repository).
# The "licence" line is a note, not legal advice: SQuAD 2.0 is CC BY-SA 4.0; SWDE and FDA state no licence, so their items are downloaded for scoring and never redistributed.
SOURCES: dict[str, dict] = {
    "squad": {
        "repo": "hazyresearch/based-squad", "revision": "d152c14c2d7d8d5114db4cb1306eee547bdbb5a7", "file": "data/validation-00000-of-00001.parquet",
        "sha256": "a97003a58c17f401670ef1f86991a4b64da56ed13096425423d36641370d0248", "rows": 2984, "split": "validation",
        "licence": "SQuAD 2.0: CC BY-SA 4.0"},
    "swde": {
        "repo": "hazyresearch/based-swde", "revision": "27f0223028a5d1ce174011051d3d22a5fafda830", "file": "data/validation-00000-of-00001.parquet",
        "sha256": "5a23577a07a8318667d0f1d353a46dda7716c7d60af6a1c8e676b2c56397980a", "rows": 1111, "split": "validation",
        "licence": "no licence stated; items are downloaded for scoring and never redistributed"},
    "fda": {
        "repo": "hazyresearch/based-fda", "revision": "42569d301e12fbcf8d5a69e04e892aa013e20314", "file": "data/validation-00000-of-00001.parquet",
        "sha256": "8be771261b43c826e39525697ca6b2d84102e5034066781221f171504c1a2542", "rows": 1102, "split": "validation",
        "licence": "no licence stated; items are downloaded for scoring and never redistributed"},
}
assert set(SOURCES) == set(TASKS)


# ------------------------------------------------------------------------------------------------ items


@dataclass(frozen=True)
class Item:
    task: str
    id: str                  # stable: the dataset's own id (squad: the SQuAD question id; swde, fda: "<doc_id>#<key>", repeats suffixed ~2, ~3 in file order)
    prompt: str              # doc["text"] as the dataset gives it
    gold: str                # doc["value"]
    meta: dict = field(default_factory=dict, compare=False, hash=False)


def format_item(task: str, row: dict, index: int) -> Item:
    """The Item of one dataset row (a dict as pyarrow's to_pylist gives it); `index` is the row's position in the pinned file."""
    if task == "squad":
        prompt, context = str(row["text"]), str(row["context"])
        # the completion is what follows the passage in the prompt (the prompt is left-padded with spaces and has no separator before the completion)
        core = context.strip()
        pos = prompt.find(core) if core else -1
        completion = prompt[pos + len(core):] if pos >= 0 else ""
        return Item(task, str(row["doc_id"]), prompt, str(row["value"]),
                    {"title": str(row["title"]), "question": str(row["question"]), "completion": completion, "row": index})
    if task in ("swde", "fda"):
        prompt = str(row["text"])
        tail = prompt.rstrip().rsplit("\n", 1)[-1]
        return Item(task, f"{row['doc_id']}#{row['key']}", prompt, str(row["value"]),
                    {"file_name": str(row.get("file_name", "")), "key": str(row["key"]), "completion": tail, "row": index})
    raise ValueError(f"unknown task {task!r}")


def build_items(rows_by_task: dict[str, list[dict]]) -> tuple[dict[str, list[Item]], dict]:
    """(items by task, report). Ids are made unique (a repeat gets ~2, ~3 in file order); NOTHING is dropped: the reference scores every row,
    so exact (prompt, gold) repeats stay (they are counted in the report)."""
    items: dict[str, list[Item]] = {}
    report: dict = {"tasks": {}}
    for task, rows in rows_by_task.items():
        seen: dict[str, int] = {}
        out = []
        for i, row in enumerate(rows):
            it = format_item(task, row, i)
            n = seen.get(it.id, 0) + 1
            seen[it.id] = n
            if n > 1:
                it = Item(it.task, f"{it.id}~{n}", it.prompt, it.gold, it.meta)
            out.append(it)
        pair_count: dict[tuple, int] = {}
        for it in out:
            pair_count[(it.prompt, it.gold)] = pair_count.get((it.prompt, it.gold), 0) + 1
        items[task] = out
        report["tasks"][task] = {
            "rows": len(rows), "items": len(out), "ids_made_unique": sum(1 for v in seen.values() if v > 1),
            "exact_prompt_gold_repeats": sum(v - 1 for v in pair_count.values() if v > 1),
            "empty_gold": sum(1 for it in out if not it.gold.strip()),
            "gold_not_in_prompt": sum(1 for it in out if it.gold.lower() not in it.prompt.lower()),
        }
    return items, report


def item_sha256(it: Item) -> str:
    """sha256 of the item's text as a canonical JSON line: [task, id, prompt, gold], UTF-8, no ASCII escaping."""
    payload = json.dumps([it.task, it.id, it.prompt, it.gold], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def set_sha256(items: Sequence[Item]) -> str:
    """sha256 of a task's item hashes, one per line, in file order."""
    return hashlib.sha256("\n".join(item_sha256(it) for it in items).encode("ascii")).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def gzip_bytes(data: bytes) -> bytes:
    """A gzip stream of `data` with a fixed header (no file name, mtime 0), so the same items give the same file."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as g:
        g.write(data)
    return buf.getvalue()


def write_items(path, items: Sequence[Item]) -> str:
    """The items as one gzip JSON line each; returns the file's sha256."""
    lines = [json.dumps({"task": it.task, "id": it.id, "prompt": it.prompt, "gold": it.gold, "meta": it.meta}, ensure_ascii=False) for it in items]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(gzip_bytes(data))
    tmp.replace(path)
    return sha256_file(path)


def read_items(path) -> list[Item]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append(Item(r["task"], r["id"], r["prompt"], r["gold"], r.get("meta", {})))
    return out


MANIFEST_NAME = "manifest.json"                  # the default manifest of an items directory (scripts/recall_based_score.py --manifest picks another)


def load_set(directory, task: str, manifest: str = MANIFEST_NAME) -> tuple[list[Item], dict]:
    """(items, the manifest's record of the task). The set sha256 is recomputed from the items and must equal the manifest's: a damaged upload stops here."""
    directory = Path(directory)
    man = json.loads((directory / manifest).read_text())
    rec = man["tasks"][task]
    items = read_items(directory / rec["file"])
    if len(items) != rec["items"] or set_sha256(items) != rec["set_sha256"]:
        raise ValueError(f"{directory / rec['file']}: {len(items)} items, set sha256 {set_sha256(items)[:12]}, the manifest has {rec['items']} items and {rec['set_sha256'][:12]}")
    if len({it.id for it in items}) != len(items):
        raise ValueError(f"{directory / rec['file']}: item ids are not unique")
    return items, rec


def available_tasks(directory, manifest: str = MANIFEST_NAME) -> list[str]:
    """The tasks of TASKS the manifest in `directory` lists (a held set that was never built is not there)."""
    man = json.loads((Path(directory) / manifest).read_text())
    return [t for t in TASKS if t in man.get("tasks", {})]


# ------------------------------------------------------------------------------------------------ the reference's text rules


def contains_score(prediction: str, labels: Sequence[str]) -> int:
    """The reference's metric, verbatim (task.py contains_score): 1 when any label, regex-escaped, is found in the prediction ignoring case."""
    return max(int(bool(re.search(re.compile(re.escape(label), re.IGNORECASE), prediction))) for label in labels)


def cut_at_stop(text: str, until: Sequence[str] = STOP) -> str:
    """The generation up to the first stop string (huggingface.py line 1181: s = s.split(term)[0] for each term)."""
    for term in until:
        if len(term) > 0:
            text = text.split(term)[0]
    return text


def cut_at_eos(ids: Sequence[int], eos_id: int | None = EOS_ID) -> tuple[list[int], int]:
    """(the ids before the first `eos_id`, the index of that id or -1): a generation ends at the end of the document (module docstring, "ONE LABELLED CHANGE").
    `eos_id` None cuts nothing."""
    ids = list(ids)
    if eos_id is not None and eos_id in ids:
        at = ids.index(eos_id)
        return ids[:at], at
    return ids, -1


def left_truncate(ids: Sequence[int], max_length: int, max_gen: int) -> tuple[list[int], bool]:
    """(the last max_length - max_gen ids, whether anything was cut): the reference's `encoding[-left_truncate_len:]` with max_ctx_len = max_length - max_gen_toks
    (huggingface.py lines 1143-1151, 667-668). The reference's `if left_truncate_len:` guard is kept: a context length of 0 or less cuts nothing."""
    keep = max_length - max_gen
    if keep >= 1 and len(ids) > keep:
        return list(ids[-keep:]), True
    return list(ids), False


@dataclass
class Prompt:
    item: Item
    ids: list[int]
    n_full: int              # tokens before truncation
    truncated: bool


def encode_prompts(items: Sequence[Item], encode: Callable[[str], list[int]], max_length: int = TRAIN_LENGTH, max_gen: int = MAX_GEN_TOKS) -> list[Prompt]:
    out = []
    for it in items:
        full = encode(it.prompt)
        ids, cut = left_truncate(full, max_length, max_gen)
        out.append(Prompt(it, ids, len(full), cut))
    return out


# ------------------------------------------------------------------------------------------------ generation


@dataclass
class Codec:
    """The tokenizer as two functions: encode (no special tokens) and decode (special tokens kept, as the reference's tok_decode)."""
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]
    sha256: str = ""


def hf_codec(tokenizer_path) -> Codec:
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(tokenizer_path))
    return Codec(lambda text: tk.encode(text, add_special_tokens=False).ids,
                 lambda ids: tk.decode(list(ids), skip_special_tokens=False), sha256_file(tokenizer_path))


def batch_plan(lengths: Sequence[int], batch_rows: int, max_batch_tokens: int) -> list[list[int]]:
    """Indices grouped into batches of similar length, longest first: at most batch_rows rows and rows x longest <= max_batch_tokens (ced.eval.qa_probe.batches_by_length)."""
    from ced.eval.qa_probe import batches_by_length

    return batches_by_length(list(lengths), batch_rows, max_batch_tokens)


def greedy_tokens(model, id_lists: Sequence[Sequence[int]], max_gen: int, pad_id: int = 0):
    """(B, max_gen) greedy tokens after each prompt, by the existing helper ced.eval.qa_probe.greedy_continuation (prefill and extend of the cache; LoopedCED.generate for a
    looped model). The helper takes EncodedItems whose last n_answer ids stand for the answer: placeholders of `pad_id`, never read as input."""
    from ced.eval.qa_probe import EncodedItem, greedy_continuation

    enc = [EncodedItem(ids=list(ids) + [pad_id] * max_gen, n_answer=max_gen, answer_bytes=1) for ids in id_lists]
    return greedy_continuation(model, enc, pad_id=pad_id)


def generate_all(model, prompts: Sequence[Prompt], codec: Codec, *, max_gen: int = MAX_GEN_TOKS, batch_rows: int = 64, max_batch_tokens: int = 8192,
                 pad_id: int = 0, done: dict[str, dict] | None = None, on_batch: Callable[[list[dict], dict], None] | None = None,
                 sync: Callable[[], None] | None = None, eos_id: int | None = EOS_ID) -> dict[str, dict]:
    """Greedy generation for every prompt, in batches of similar length. Returns {item id: record}; `done` holds records of an earlier, stopped pass (their items are
    skipped); `on_batch(new records, timing)` is called after each batch (the scorer appends them to its part file). Record: id, hit, n_tokens (the prompt as fed),
    n_full, truncated, generation (the text before the first EOS and the first "\\n"), tokens (the 48 ids as generated), eos_at (index of the first `eos_id` in the
    48 ids, -1 if none), hit_no_eos_cut (the hit with the newline stop alone). timing: rows, longest, seconds of the call."""
    import torch

    records: dict[str, dict] = dict(done or {})
    todo = [p for p in prompts if p.item.id not in records]
    if not todo:
        return records
    for idx in batch_plan([len(p.ids) for p in todo], batch_rows, max_batch_tokens):
        batch = [todo[i] for i in idx]
        t0 = time.time()
        gen = greedy_tokens(model, [p.ids for p in batch], max_gen, pad_id)
        if sync is not None:
            sync()
        seconds = time.time() - t0
        gen = gen.cpu() if isinstance(gen, torch.Tensor) else gen
        new = []
        for row, p in enumerate(batch):
            toks = [int(t) for t in gen[row].tolist()]
            kept, eos_at = cut_at_eos(toks, eos_id)
            text = cut_at_stop(codec.decode(kept))
            text_no_eos_cut = text if eos_at < 0 else cut_at_stop(codec.decode(toks))
            rec = {"id": p.item.id, "hit": contains_score(text, [p.item.gold]), "n_tokens": len(p.ids), "n_full": p.n_full, "truncated": p.truncated,
                   "generation": text[:300], "tokens": toks, "eos_at": eos_at, "hit_no_eos_cut": contains_score(text_no_eos_cut, [p.item.gold])}
            records[p.item.id] = rec
            new.append(rec)
        if on_batch is not None:
            on_batch(new, {"rows": len(batch), "longest": max(len(p.ids) for p in batch), "seconds": round(seconds, 3)})
    return records


BOUND_FACTOR = 4.0       # a flip is within the bf16 bound when the cached token trails the full forward's argmax by at most BOUND_FACTOR x the measured logit noise (check_cached_generation: 2 + 2)
NOISE_TOP_K = 5          # the noise is measured on the NOISE_TOP_K largest logits of the float32 forward at each checked position (the tokens a flip is between)
DETAIL_LIMIT = 24        # flips kept with their numbers in the check's record


def bf16_spacing(x: float) -> float:
    """The spacing of bfloat16 numbers around |x| (8 significant bits: 2^(e - 7) for |x| in [2^e, 2^(e+1))): the resolution of a logit that the head returns in bf16."""
    import math

    return 2.0 ** (math.floor(math.log2(abs(x))) - 7) if x else 0.0


def full_logits(model, ids: Sequence[int], positions: Sequence[int], autocast_dtype=None):
    """(len(positions), V) logits of one row `ids` (a single document 1, no padding) at `positions`: the full forward with the head applied at those positions only,
    in at least float32 (a float64 model stays float64). `autocast_dtype` None runs the forward without autocast."""
    import torch

    from ced.model.looped_ced import lm_head

    device = next(model.parameters()).device
    seq = torch.tensor([list(ids)], dtype=torch.long, device=device)
    doc = torch.ones_like(seq)
    with torch.no_grad(), torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        out = model(seq, doc, None, return_hidden=True)
        logits = lm_head(out.hidden_aux[:, list(positions)], model.head_weight())[0]
    return logits.to(torch.promote_types(logits.dtype, torch.float32))


def check_cached_generation(model, prompts: Sequence[Prompt], records: dict[str, dict], n: int, autocast_dtype=None, pad_id: int = 0, detail_limit: int = DETAIL_LIMIT) -> dict:
    """Cached generation against the full forward on the same model, on `n` items spread over the set: the full forward of prompt + generated tokens (one row, document 1) must
    have, at the positions that predicted the generated tokens, the generated token as its argmax. A disagreement is a position where the two paths round differently into a
    different argmax (a near tie) or a defect of the cache or of the batching. The count is recorded, not judged, and each flip is recorded WITH ITS NUMBERS so that a
    reader can tell the two apart (a bare count such as "2 of 384 positions differ" gives nothing to judge them by; the equality holds in float64,
    where a padding leak or a cache defect shows as a difference of any size):

      gap        the full forward's largest logit minus its second largest at the position (0 is an exact tie in the logits' own resolution)
      shortfall  the full forward's largest logit minus its logit of the CACHED token: how much the cached path's choice trails the full forward's (<= gap when the cached
                 token is the runner-up; 0 when the two tie, and the argmax then keeps the lower index)
      top        the largest logit, and `spacing` the bf16 spacing there (under autocast the head returns bf16: a logit of 16 to 32 moves in steps of 0.125)
      eps        max over the NOISE_TOP_K largest logits of the float32 forward (autocast off) at that position of |the autocast logit - the float32 logit|

    `noise` = the largest eps over every checked position (None without autocast: the forward is its own reference), `bound` = BOUND_FACTOR x noise. WHY 4: let f be the
    full forward's argmax and c the cached token. The full forward has l_full[f] - l_full[c] = shortfall >= 0, and each of its two logits is within `noise` of the float32 one, so the
    float32 gap l32[f] - l32[c] >= shortfall - 2 noise. The cached path chose c, so l_cached[c] >= l_cached[f]; if its logits too are within `noise` of the float32 ones (the same
    arithmetic at another shape: an ASSUMPTION, the cached path's own logits are not read here), l32[f] - l32[c] <= 2 noise. Together shortfall <= 4 noise. A flip with a larger
    shortfall is not explained by rounding and is a failure to look into. Without autocast the bound is 0: only an exact tie passes."""
    import torch

    if n <= 0 or not prompts:
        return {"items": 0, "positions": 0, "mismatches": 0}
    step = max(1, len(prompts) // n)
    pick = [prompts[i] for i in range(0, len(prompts), step)][:n]
    positions = mismatches = 0
    first = None
    flips: list[dict] = []
    noise = None
    noise_skipped = 0
    for p in pick:
        toks = records[p.item.id]["tokens"]
        ids = p.ids + toks[:-1]
        where = list(range(len(p.ids) - 1, len(p.ids) - 1 + len(toks)))
        lg = full_logits(model, ids, where, autocast_dtype)
        top2 = lg.topk(2, dim=-1)
        pred = top2.indices[:, 0].cpu().tolist()
        bad = [j for j, (a, b) in enumerate(zip(pred, toks)) if a != b]
        positions += len(toks)
        mismatches += len(bad)
        eps = None
        if autocast_dtype is not None:
            try:
                ref = full_logits(model, ids, where, None)
                near = ref.topk(NOISE_TOP_K, dim=-1).indices
                eps = (lg.gather(1, near) - ref.gather(1, near)).abs().max(dim=1).values.cpu().tolist()
                noise = max(eps) if noise is None else max(noise, max(eps))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                noise_skipped += 1
        for j in bad:
            top = float(top2.values[j, 0])
            flips.append({"id": p.item.id, "step": j, "cached": toks[j], "full": pred[j], "gap": top - float(top2.values[j, 1]), "shortfall": top - float(lg[j, toks[j]]),
                          "top": top, "spacing": bf16_spacing(top), "eps": None if eps is None else eps[j]})
        if bad and first is None:
            first = {"id": p.item.id, "step": bad[0], "cached": toks[bad[0]], "full": pred[bad[0]]}
    bound = BOUND_FACTOR * (noise or 0.0)
    for f in flips:
        f["within_bound"] = f["shortfall"] <= bound
    return {"items": len(pick), "positions": positions, "mismatches": mismatches, "first": first, "noise": noise, "noise_skipped_items": noise_skipped, "bound": bound,
            "within_bound": sum(1 for f in flips if f["within_bound"]), "beyond_bound": sum(1 for f in flips if not f["within_bound"]),
            "max_shortfall": max((f["shortfall"] for f in flips), default=None), "max_gap": max((f["gap"] for f in flips), default=None), "flips": flips[:detail_limit]}


# ------------------------------------------------------------------------------------------------ values (the common output format)


def cell_key(task: str) -> str:
    return f"{FAMILY}|{task}|all|{CELL_METRIC}"


def values_json(run: str, code_sha: str, records_by_task: dict[str, Sequence[dict]], exclude: Iterable[str] = ()) -> dict:
    """The common-format values of one run: accuracy per task over the items not in `exclude` ("task:id" strings)."""
    exclude = set(exclude)
    cells: dict[str, float] = {}
    n_items: dict[str, int] = {}
    excluded: list[str] = []
    for task in TASKS:
        recs = records_by_task.get(task)
        if recs is None:
            continue
        kept = [r for r in recs if f"{task}:{r['id']}" not in exclude]
        excluded += [f"{task}:{r['id']}" for r in recs if f"{task}:{r['id']}" in exclude]
        if not kept:
            continue
        cells[cell_key(task)] = sum(int(r["hit"]) for r in kept) / len(kept)
        n_items[cell_key(task)] = len(kept)
    return {"run": run, "family": FAMILY, "code_sha": code_sha, "cells": cells, "n_items": n_items, "excluded": sorted(excluded)}


def write_json(path, obj) -> None:
    """Atomic: written under a name of its own (the process id), then renamed. A run's <run>.values.json can have two writers when two scoring
    passes of the same run overlap, and a shared temporary name would let one rename the other's half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    tmp.replace(path)


def write_records(path, records: Sequence[dict], keep_tokens: bool = False) -> str:
    """A task's per-item records as a gzip JSON-lines file (the 48 token ids are dropped unless keep_tokens); returns its sha256."""
    lines = []
    for r in records:
        r = dict(r)
        if not keep_tokens:
            r.pop("tokens", None)
        lines.append(json.dumps(r, ensure_ascii=False, sort_keys=True))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(gzip_bytes(("\n".join(lines) + "\n").encode("utf-8")))
    tmp.replace(path)
    return sha256_file(path)


def read_records(path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_run_records(out_dir, label: str) -> dict[str, list[dict]]:
    """{task: the per-item records} of every task with a finished items file under <out_dir>/<label>/ (a task still in progress has only its .part.jsonl and is not there).
    The values of a run are computed from ALL of these, whichever pass wrote which task: two passes of one run (one for SQuAD, one for SWDE and FDA) share the
    run's <label>.values.json, and the later one must add its cells to the earlier one's, not replace them."""
    out_dir = Path(out_dir)
    return {t: read_records(out_dir / label / f"{t}.items.jsonl.gz") for t in TASKS if (out_dir / label / f"{t}.items.jsonl.gz").exists()}
