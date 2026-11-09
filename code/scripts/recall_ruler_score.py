"""Score a trainer checkpoint on the RULER subset (open haystack) by greedy generation (ced/eval/recall_ruler.py, ced/eval/recall_ruler_gen.py).

    python scripts/recall_ruler_score.py --checkpoint <checkpoint>/step_00007996.pt --items <items> \\
        --tokenizer <tokenizer.json> --out <out> --run f-200m-s1 [--n-per-cell 500] [--tasks ...] [--lengths ...]

<items> is the output directory of tools/build_recall_ruler.py (see EVAL.md).

The model is rebuilt from the checkpoint's config as scripts/eval_probes.py does (the training dtype and autocast, TF32 as trained), with two changes that the cached
path needs: the attention backend "reference" and the GDN kernel "chunked" (the reference code, checked against the full forward with
--check-forward; --attn-backend / --gdn-kernel change them) and no layer compilation (the cache lengths change at every step).
Generation is ced/eval/recall_ruler_gen.py generate_greedy (greedy; RULER's tokens to generate, 128 for the needle tasks and 30 for variable tracking; no stop
words; a row stops at the end-of-document token or when all its references are in its text).

Outputs under --out:
    <run>.items.jsonl     one record per scored item (the prediction, the generated ids, the recall); the resume file: a rerun reads it and scores only
                          the items it lacks, if the checkpoint, the settings and the code files are the same (the first line holds that fingerprint)
    <run>.values.json     the common values file of the three recall builds: {"run", "family": "ruler", "code_sha", "cells", "n_items", "excluded"}
    <run>.meta.json       settings, timing per length, the forward check, memory
--exclude-file: a JSON list of "<task>:len=<T>/<index>" strings left out of every cell (none expected: nothing in the prompts has a
text source but the held-out haystack documents).
--check-forward K: for the first K scored items of each length, the generated tokens against the argmax of a full forward over the prompt and the generation
(cached generation equal to the full forward): the count of positions that differ and the margin of each; in the meta file and the log.
Exit: 0 all scored; 2 a bad request; 3 the soft time budget (--max-seconds) ended first: the records are kept, run it again.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from ced.config import TrainConfig, from_dict  # noqa: E402
from ced.eval import gqa_fold as GF  # noqa: E402
from ced.eval import recall_ruler as RR  # noqa: E402
from ced.eval import recall_ruler_gen as GEN  # noqa: E402
from ced.eval import units  # noqa: E402
from ced.model.control import build_model  # noqa: E402


def load_items(items_dir: Path, tasks: list[str], lengths: list[int], n_per_cell: int) -> list[RR.RulerItem]:
    """The first n_per_cell indices of every (task, length) cell from <items_dir>/items/<task>.jsonl.gz, ordered by length, task order, index."""
    out = []
    for task in tasks:
        path = items_dir / "items" / f"{task}.jsonl.gz"
        if not path.exists():
            raise SystemExit(f"no items for {task}: {path} (run the build first)")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                it = RR.RulerItem.from_json(json.loads(line))
                if it.length in lengths and it.index < n_per_cell:
                    out.append(it)
    order = {t: i for i, t in enumerate(tasks)}
    out.sort(key=lambda it: (it.length, order[it.task], it.index))
    have = {(t, L): 0 for t in tasks for L in lengths}  # a cell the build does not hold at all is short too (zero items), not silently absent
    for it in out:
        have[(it.task, it.length)] += 1
    short = {f"{t}/{L}": n for (t, L), n in have.items() if n < n_per_cell}
    if short:
        raise SystemExit(f"fewer than {n_per_cell} items built in {short}: build more or lower --n-per-cell")
    return out


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forward_check(model, tok, items_by_key: dict, records: list[dict], k: int, autocast_dtype) -> dict:
    """The generated tokens of the first k scored items of each length against a full forward over prompt + generation (module docstring)."""
    out: dict = {"k_per_length": k, "items": [], "positions": 0, "mismatches": 0, "skipped": []}
    seen: dict = {}
    for rec in records:
        L = rec["length"]
        if seen.get(L, 0) >= k or not rec["gen_ids"]:
            continue
        seen[L] = seen.get(L, 0) + 1
        it = items_by_key[rec["key"]]
        prompt = tok.encode(it.prompt, add_special_tokens=False).ids
        gen = rec["gen_ids"]
        P = len(prompt)
        try:
            arg, margin = GEN.full_forward_argmax(model, prompt + gen[:-1], list(range(P - 1, P - 1 + len(gen))), autocast_dtype=autocast_dtype)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out["skipped"].append({"key": rec["key"], "why": "out of memory in the full forward"})
            continue
        bad = [(j, round(float(m), 4)) for j, (a, g, m) in enumerate(zip(arg, gen, margin)) if a != g]
        out["items"].append({"key": rec["key"], "positions": len(gen), "mismatches": bad})
        out["positions"] += len(gen)
        out["mismatches"] += len(bad)
    return out


def write_values(out: Path, run: str, values: dict) -> None:
    """Write <run>.values.json. An earlier file of the run that holds cells the new one lacks (a pass on a subset of the tasks or lengths, or a reading that left a whole cell out)
    is kept as <run>.values.json.prev-<n> first, so that no cell the signal-to-noise reader would have read disappears silently: the reader takes only files that end in
    ".values.json"."""
    path = out / f"{run}.values.json"
    if path.exists():
        try:
            old = json.loads(path.read_text()).get("cells", {})
        except ValueError:
            old = {}
        lost = sorted(set(old) - set(values["cells"]))
        if lost:
            n = 1
            while (out / f"{run}.values.json.prev-{n}").exists():
                n += 1
            os.replace(path, out / f"{run}.values.json.prev-{n}")
            print(f"[recall-ruler] {run}: the new values file lacks {len(lost)} cells of the earlier one (e.g. {lost[0]}); the earlier file is kept as {run}.values.json.prev-{n}", flush=True)
    units.write_atomic(path, json.dumps(values, indent=1))


def values_only(a, tasks: list[str], lengths: list[int], exclude: list[str]) -> int:
    """--values-only: the values file of a run from its records, no model."""
    path = Path(a.out) / f"{a.run}.items.jsonl"
    if not path.exists():
        raise SystemExit(f"no records: {path}")
    recs = [r for r in GEN.read_records(path).values() if r["task"] in tasks and r["length"] in lengths and r["index"] < a.n_per_cell]
    have: dict = {}
    for r in recs:
        have[(r["task"], r["length"])] = have.get((r["task"], r["length"]), 0) + 1
    short = {f"{t}/{L}": n for (t, L), n in sorted(have.items()) if n < a.n_per_cell}
    missing = [f"{t}/{L}" for t in tasks for L in lengths if (t, L) not in have]
    if short or missing:
        raise SystemExit(f"{a.run}: fewer than {a.n_per_cell} records in {short or missing}: score the run first")
    values = RR.values_file(a.run, recs, os.environ.get("CED_CODE_SHA", "local"), exclude)
    write_values(Path(a.out), a.run, values)
    print(f"[recall-ruler] {a.run}: values from {len(recs)} records, {len(values['excluded'])} flagged ids listed, "
          f"{sum(1 for r in recs if r['key'] in set(exclude))} items left out", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", default=None, help="a checkpoint file, a run directory or its checkpoints/ directory")
    p.add_argument("--items", default=None, help="the build's output directory (tools/build_recall_ruler.py)")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--values-only", action="store_true", help="no model: rewrite <run>.values.json from the records of <run>.items.jsonl (the first --n-per-cell indices of the "
                   "--tasks and --lengths cells) with --exclude-file applied; for flags that arrive after a checkpoint was scored")
    p.add_argument("--out", required=True)
    p.add_argument("--run", required=True, help="the run name (the values file's \"run\")")
    p.add_argument("--n-per-cell", type=int, default=RR.PER_CELL)
    p.add_argument("--tasks", default=",".join(RR.TASKS))
    p.add_argument("--lengths", default=",".join(str(x) for x in RR.LENGTHS))
    p.add_argument("--max-batch-tokens", type=int, default=32768, help="rows of a prefill batch x the widest prompt; a batch that runs out of memory is split")
    p.add_argument("--scorer", default="fast", choices=["fast", "reference"], help="fast: the prefill in sub-batches of --max-batch-tokens / width rows, the caches joined, the decode over "
                   "--decode-rows rows (ced/eval/recall_ruler_gen.py generate_greedy_batched); reference: the loop of the first version (the rows prefilled are the rows decoded)")
    p.add_argument("--decode-rows", type=int, default=32, help="--scorer fast: rows decoded together (a decode step is launch-bound: about the same seconds for 4 rows as for 16; the "
                   "rows are bounded by the memory of the repeated key heads of the reference read, ced/eval/gqa_fold.py)")
    p.add_argument("--gqa-fold", action="store_true", help="--scorer fast: the reference attention reads on the unexpanded key heads (ced/eval/gqa_fold.py): the same values, no "
                   "repeated k, k_far and v in float32")
    p.add_argument("--prefill-chunk", type=int, default=GEN.PREFILL_CHUNK)
    p.add_argument("--check-every", type=int, default=GEN.CHECK_EVERY)
    p.add_argument("--attn-backend", default="reference")
    p.add_argument("--gdn-kernel", default="chunked")
    p.add_argument("--autocast", default="auto", choices=["auto", "none"])
    p.add_argument("--check-forward", type=int, default=2, help="items a length checked against the full forward (0: none)")
    p.add_argument("--exclude-file", default=None)
    p.add_argument("--max-seconds", type=float, default=None, help="soft budget: stop between two batches after this many seconds (exit 3)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    tasks = a.tasks.split(",")
    lengths = [int(x) for x in a.lengths.split(",")]
    if [t for t in tasks if t not in RR.TASKS]:
        raise SystemExit(f"--tasks: one of {RR.TASKS}")
    t_start = time.time()
    exclude = json.loads(Path(a.exclude_file).read_text()) if a.exclude_file else []
    if a.values_only:
        return values_only(a, tasks, lengths, exclude)
    if not (a.checkpoint and a.items and a.tokenizer):
        raise SystemExit("--checkpoint, --items and --tokenizer (or --values-only)")
    import eval_probes as EP  # scripts/eval_probes.py: the checkpoint resolver and the backend override

    path = EP.resolve_checkpoint(a.checkpoint)
    items = load_items(Path(a.items), tasks, lengths, a.n_per_cell)
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = from_dict(TrainConfig, state["config"])
    if cfg.model.arch != "control":
        raise SystemExit(f"arch {cfg.model.arch!r}: the generation loop is the control arch's prefill / extend")
    EP.set_backends(cfg, a.attn_backend, a.gdn_kernel)
    model = build_model(cfg.model).to(getattr(torch, cfg.dtype))
    model.load_state_dict(state["model"])
    device = torch.device(a.device)
    model = model.to(device).eval()
    cuda = device.type == "cuda"
    autocast = getattr(torch, cfg.autocast) if cuda and cfg.autocast != "none" and a.autocast == "auto" else None
    if cuda:
        from ced.train.trainer import configure_compile

        configure_compile()
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(a.tokenizer)
    eos_id = tok.token_to_id("<|endoftext|>")
    load_s = time.time() - t_start

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_sha = sha256_of(Path(a.items) / "manifest.json") if (Path(a.items) / "manifest.json").exists() else None
    fingerprint = {
        "script": "recall_ruler_score", "checkpoint": str(path), "step": state.get("step"), "checkpoint_file": units.file_fingerprint(path), "arch": cfg.model.arch,
        "torch": torch.__version__, "device": torch.cuda.get_device_name(device) if cuda else "cpu", "autocast": str(autocast), "attn_backend": a.attn_backend,
        "gdn_kernel": a.gdn_kernel, "prefill_chunk": a.prefill_chunk, "items_manifest_sha256": manifest_sha, "scorer": a.scorer, "gqa_fold": bool(a.gqa_fold and a.scorer == "fast"),  # not the batch budget: rows of a batch change padding only
        "code_files": {f: sha256_of(REPO / f) for f in ("ced/eval/recall_ruler.py", "ced/eval/recall_ruler_gen.py", "scripts/recall_ruler_score.py")}}
    results_path = out / f"{a.run}.items.jsonl"
    stop_at = t_start + a.max_seconds if a.max_seconds else None
    print(f"[recall-ruler] {a.run}: scorer {a.scorer}" + (f", decode rows {a.decode_rows}, gqa fold {bool(a.gqa_fold)}" if a.scorer == "fast" else "") +
          f", prefill chunk {a.prefill_chunk}, token budget a prefill {a.max_batch_tokens}, attention {a.attn_backend}, gdn {a.gdn_kernel}", flush=True)
    t0 = time.time()
    try:
        fast = a.scorer == "fast"
        with (GF.folded_reads() if fast and a.gqa_fold else contextlib.nullcontext()):  # the forward check below stays on the reference reads: it is the arbiter
            records, timing = GEN.score_items(model, tok, items, results_path=results_path, fingerprint=fingerprint, eos_id=eos_id, max_batch_tokens=a.max_batch_tokens,
                                              prefill_chunk=a.prefill_chunk, check_every=a.check_every, autocast_dtype=autocast, stop_at=stop_at, log=lambda s: print(s, flush=True),
                                              generate=GEN.generate_greedy_batched if fast else GEN.generate_greedy, decode_rows=a.decode_rows if fast else None)
    except GEN.BudgetExceeded as e:
        print(f"[recall-ruler] {a.run}: {e}", flush=True)
        return 3
    score_s = time.time() - t0

    by_key = {it.key: it for it in items}
    check = forward_check(model, tok, by_key, records, a.check_forward, autocast) if a.check_forward else None
    values = RR.values_file(a.run, records, os.environ.get("CED_CODE_SHA", "local"), exclude)  # every flagged id is listed: a reader wants the same list in every file
    write_values(out, a.run, values)
    meta = {"run": a.run, "checkpoint": str(path), "step": state.get("step"), "arch": cfg.model.arch, "n_per_cell": a.n_per_cell, "tasks": tasks, "lengths": lengths,
            "settings": {**{k: v for k, v in fingerprint.items() if k != "code_files"}, "max_batch_tokens": a.max_batch_tokens, "check_every": a.check_every,
                         "decode_rows": a.decode_rows if a.scorer == "fast" else None},  # (the fingerprint holds scorer and gqa_fold)
            "timing": {"load_s": round(load_s, 1), "score_s": round(score_s, 1), "per_length": timing},
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if cuda else None, "forward_check": check,
            "n_scored": len(records), "exclude_unknown": [k for k in exclude if k not in by_key]}
    units.write_atomic(out / f"{a.run}.meta.json", json.dumps(meta, indent=1, sort_keys=True))
    print(f"[recall-ruler] {a.run}: {len(records)} items in {score_s:.0f}s (load {load_s:.0f}s); cells: " +
          " ".join(f"{k.split('|', 1)[1]}={v:.3f}" for k, v in values["cells"].items() if k.endswith("|recall") and "|all|" not in k and "|mean|" not in k), flush=True)
    for L, t in sorted(timing.items()):
        print(f"[recall-ruler] timing len={L}: {t['items']} items in {t['batches']} batches, {t['prompt_tokens']} prompt tokens, prefill {t['prefill_s']:.1f}s "
              f"({t['prompt_tokens'] / max(t['prefill_s'], 1e-9):.0f} tok/s), decode {t['decode_steps']} steps {t['decode_s']:.1f}s "
              f"({1000 * t['decode_s'] / max(t['decode_steps'], 1):.1f} ms/step), oom splits {t['oom_splits']}", flush=True)
    if check is not None:
        print(f"[recall-ruler] forward check: {check['mismatches']} of {check['positions']} generated positions differ from the full forward's argmax "
              f"({len(check['items'])} items, {len(check['skipped'])} skipped); " + json.dumps([i for i in check["items"] if i["mismatches"]])[:600], flush=True)
    if cuda:
        print(f"[recall-ruler] peak allocated {meta['peak_allocated_gib']} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
