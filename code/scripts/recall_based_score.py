"""Score finished 200M checkpoints on the Based recall suite (SWDE, FDA, SQuAD completion): GPU, forward and greedy generation only. The definition (the
reference's prompt, 48 greedy tokens, the "contains" metric, the left truncation) and the choices are in the docstring of ced/eval/recall_based.py.

    python scripts/recall_based_score.py score --out <out> --items <items> --tokenizer <tokenizer.json> --final-step 7996 --run f-200m-s1=<run dir> ...
    python scripts/recall_based_score.py values --out <out> --exclude-file flagged.json [--run f-200m-s1 ...]

<items> is the output directory of tools/data/build_recall_based.py (see EVAL.md).

score. Each --run is LABEL=PATH, the label being the run's name (the `<label>.values.json` it writes), PATH a run directory (its checkpoints/latest) or a
checkpoint file. With --final-step N (default 7996, the last step of the 200M runs; 0 = any step) a checkpoint that is not at that step is skipped as "not final"
(a run still training has a rolling `latest`). The model is the repo's slow readable reference (GDN kernel "chunked", attention backend "reference", the trained
dtype under the trained autocast; load_reference_model below), and the generation is ced.eval.qa_probe.greedy_continuation (prefill and extend of the cache), so
nothing here is a decoder of its own. --check-cached N compares the cached generation of N items with the full forward of the same model (positions where the
argmax differs are counted and recorded with their logit gaps, and set against the bf16 noise of the logits measured against a float32 forward:
ced/eval/recall_based.py check_cached_generation; recorded, not judged).
--probe-lengths L1,L2 times one batch of random-token prompts of each length (written to <out>/<label>/synthetic_timing.json; nothing is scored): the cost of
the long prompts of SWDE and FDA without reading either set.

Several passes may score one run, each its own tasks (for example SQuAD from manifest.json, SWDE and FDA from another manifest file given by --manifest): the run's
<label>.values.json is recomputed from every finished task file on disk after each pass, so a later pass ADDS its cells to the earlier one's (written atomically, a
temporary name of its own); `values` does the same on the CPU.

Resumable: a finished (run, task) with the same key (set sha256, tokenizer, step, lengths) is read, not scored again (another key stops the script, --force
scores it again); inside a task the records of the batches done are appended to <task>.part.jsonl and read on a rerun. The tasks are those of --tasks, else
those the items manifest lists.

Output: <out>/<label>/<task>.items.jsonl.gz (id, hit, n_tokens, n_full, truncated, generation), <task>.json (the key, counts, seconds, the batch timings and
their fit, the cached check), <out>/<label>.values.json (the common format; the exclusion list from --exclude-file, none at score time), <out>/result-<tag>.md
(under 15,000 bytes). Exit: 0 done; 2 a refusal (arguments, items, tokenizer); 3 a checkpoint missing or not final and nothing failed.

values. Recompute <label>.values.json from the per-item files with an exclusion list (--exclude-file, repeatable: a JSON list of "task:id"): CPU, no model.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ced.eval import recall_based as RB  # noqa: E402

RESULT_LIMIT = 14000


def say(text: str) -> None:
    print(f"[recall-based] {text}", flush=True)


def parse_runs(items: list[str]) -> list[tuple[str, str]]:
    out = []
    for it in items:
        if "=" not in it or not it.split("=", 1)[0] or not it.split("=", 1)[1]:
            raise SystemExit(f"--run {it!r}: LABEL=PATH")
        label, path = it.split("=", 1)
        if not all(c.isalnum() or c in "._-" for c in label):
            raise SystemExit(f"--run label {label!r}: letters, digits and . _ - only")
        if label in [x[0] for x in out]:
            raise SystemExit(f"--run label {label!r} twice")
        out.append((label, path))
    return out


def resolve_checkpoint(path: str) -> Path | None:
    """The checkpoint file of a run directory (checkpoints/latest) or of a file path; None when it is not there."""
    p = Path(path)
    for pointer in (p / "checkpoints" / "latest", p / "latest"):
        if pointer.exists():
            q = pointer.parent / pointer.read_text().strip()
            return q if q.exists() else None
    return p if p.is_file() else None


def load_exclude(files: list[str]) -> set[str]:
    out: set[str] = set()
    for f in files:
        data = json.loads(Path(f).read_text())
        if not isinstance(data, list) or not all(isinstance(x, str) and ":" in x for x in data):
            raise SystemExit(f"--exclude-file {f}: a JSON list of \"task:id\" strings")
        out.update(data)
    return out


# ------------------------------------------------------------------------------------------------ model


def load_reference_model(path: Path, device: str):
    """(model, cfg, step, autocast dtype): the checkpoint's model rebuilt with the reference code (GDN 'chunked', attention 'reference'), in the trained dtype."""
    import torch
    from ced.config import TrainConfig, checkpoint_config, from_dict
    from ced.model.control import build_model

    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = from_dict(TrainConfig, checkpoint_config(state["config"]))   # as the trainer reads a checkpoint's config
    mc = copy.deepcopy(cfg.model)
    mc.encoder.gdn.kernel = "chunked"
    mc.encoder.attention.backend = "reference"
    model = build_model(mc).to(getattr(torch, cfg.dtype))
    model.load_state_dict(state["model"])
    model = model.to(device).eval()
    cuda = torch.device(device).type == "cuda"
    autocast = None if (not cuda or cfg.autocast == "none") else getattr(torch, cfg.autocast)
    return model, cfg, int(state.get("step", -1)), autocast


# ------------------------------------------------------------------------------------------------ the fit of the batch timings


def fit_timings(batches: list[dict], max_gen: int) -> dict:
    """seconds = prefill_padded_tokens / rate + per_batch, over the batches' (rows x longest, seconds): least squares. per_batch is max_gen decode steps."""
    import numpy as np

    if len(batches) < 3:
        return {}
    x = np.array([b["rows"] * b["longest"] for b in batches], dtype=float)
    y = np.array([b["seconds"] for b in batches], dtype=float)
    if np.ptp(x) == 0:
        return {"per_batch_s": float(y.mean()), "decode_step_ms": float(y.mean() / max_gen * 1000), "prefill_tokens_per_s": None}
    a = np.vstack([x, np.ones_like(x)]).T
    (slope, const), *_ = np.linalg.lstsq(a, y, rcond=None)
    return {"prefill_tokens_per_s": float(1.0 / slope) if slope > 0 else None, "per_batch_s": float(const), "decode_step_ms": float(const / max_gen * 1000),
            "batches": len(batches), "seconds_total": float(y.sum())}


def time_synthetic(model, autocast, device: str, lengths: list[int], args) -> list[dict]:
    """Seconds of one batch of random-token prompts of each length (rows = max_batch_tokens // length, at least 1) through the same generation call: the cost of the
    long prompts of SWDE and FDA, which are not scored here and need not be read to be timed. Nothing is scored; the token ids are a fixed pseudo-random draw, never an item."""
    import torch

    g = torch.Generator().manual_seed(20261102)
    vocab = int(model.cfg.vocab_size)
    cuda = torch.device(device).type == "cuda"
    out = []
    for length in lengths:
        rows = max(1, min(args.batch_rows, args.max_batch_tokens // length))
        ids = torch.randint(1, vocab, (rows, length), generator=g).tolist()
        t0 = time.time()
        with torch.autocast(torch.device(device).type, dtype=autocast, enabled=autocast is not None):
            RB.greedy_tokens(model, ids, args.max_gen)
        if cuda:
            torch.cuda.synchronize()
        out.append({"rows": rows, "longest": length, "seconds": round(time.time() - t0, 3)})
        say(f"synthetic prompts: {rows} rows x {length} tokens + {args.max_gen} new tokens in {out[-1]['seconds']:.2f} s")
    return out


# ------------------------------------------------------------------------------------------------ score


def score_task(model, autocast, device, label: str, task: str, items, codec, outdir: Path, key: dict, args) -> dict:
    import torch

    tdir = outdir / label
    tdir.mkdir(parents=True, exist_ok=True)
    done_json, done_items, part = tdir / f"{task}.json", tdir / f"{task}.items.jsonl.gz", tdir / f"{task}.part.jsonl"
    if done_json.exists() and done_items.exists():
        old = json.loads(done_json.read_text())
        if old["key"] == key:
            say(f"{label}/{task}: already there ({old['n_items']} items, acc {old['acc']:.4f})")
            return {**old, "reused": True, "records": RB.read_records(done_items)}
        if not args.force:
            raise SystemExit(f"{done_json} was scored under another key:\n  there {old['key']}\n  now   {key}\n(--force scores it again)")
    if args.max_items:
        items = items[: args.max_items]
    prompts = RB.encode_prompts(items, codec.encode, args.max_length, args.max_gen)
    n_cut = sum(1 for p in prompts if p.truncated)
    toks = [len(p.ids) for p in prompts]
    say(f"{label}/{task}: {len(prompts)} items, prompt tokens {sum(toks):,} (mean {sum(toks) / len(toks):.0f}, max {max(toks)}), {n_cut} cut to the last "
        f"{args.max_length - args.max_gen} tokens")
    already: dict[str, dict] = {}
    if part.exists() and not args.force:
        for line in part.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                already[r["id"]] = r
        say(f"{label}/{task}: {len(already)} records of an earlier pass read from {part.name}")
    elif part.exists():
        part.unlink()
    timings: list[dict] = []
    cuda = torch.device(device).type == "cuda"
    t0 = time.time()
    state = {"n": 0}

    def on_batch(new, timing):
        with open(part, "a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        timings.append(timing)
        state["n"] += 1
        if state["n"] % 10 == 1:
            say(f"{label}/{task}: batch {state['n']}: {timing['rows']} rows x {timing['longest']} tokens in {timing['seconds']:.2f} s, "
                f"{len(already) + sum(b['rows'] for b in timings)}/{len(prompts)} items, {time.time() - t0:.0f} s")

    with torch.autocast(torch.device(device).type, dtype=autocast, enabled=autocast is not None):
        records = RB.generate_all(model, prompts, codec, max_gen=args.max_gen, batch_rows=args.batch_rows, max_batch_tokens=args.max_batch_tokens,
                                  done=already, on_batch=on_batch, sync=(torch.cuda.synchronize if cuda else None))
    seconds = time.time() - t0
    check = RB.check_cached_generation(model, prompts, records, args.check_cached, autocast) if args.check_cached else None
    if check is not None:
        say(f"{label}/{task}: cached generation against the full forward: {check['mismatches']} of {check['positions']} positions differ in {check['items']} items"
            + (f" (first: {check['first']})" if check["first"] else ""))
        if check["mismatches"] or check.get("noise") is not None:
            noise = check.get("noise")
            say(f"{label}/{task}: bf16 noise of the logits (max |autocast - float32| over the {RB.NOISE_TOP_K} largest logits): "
                + ("none (no autocast)" if noise is None else f"{noise:.4f}") + f"; bound {check['bound']:.4f} = {RB.BOUND_FACTOR:g} x noise; "
                f"{check['within_bound']} flips within it, {check['beyond_bound']} beyond it" + (f"; {check['noise_skipped_items']} items without a float32 reference" if check["noise_skipped_items"] else ""))
        for f in check.get("flips", []):
            say(f"{label}/{task}: flip {f['id']} step {f['step']} cached {f['cached']} full {f['full']}: top-2 gap {f['gap']:.4f}, the cached token trails by {f['shortfall']:.4f}, "
                f"top logit {f['top']:.3f} (bf16 spacing {f['spacing']:.4f}), eps {'-' if f['eps'] is None else format(f['eps'], '.4f')}, "
                + ("within the bound" if f["within_bound"] else "BEYOND THE BOUND"))
    ordered = [records[p.item.id] for p in prompts]
    acc = sum(int(r["hit"]) for r in ordered) / len(ordered)
    RB.write_records(done_items, ordered)
    # code_sha: the snapshot that scored THIS task. A run's values file has one code_sha, the last writer's, and the passes of a run can be different snapshots:
    # the per-task file is where each cell's code is recorded
    summary = {"key": key, "task": task, "code_sha": os.environ.get("CED_CODE_SHA", "local"), "n_items": len(ordered), "acc": acc, "hits": sum(int(r["hit"]) for r in ordered),
               "prompt_tokens": sum(toks), "prompt_tokens_max": max(toks), "n_truncated": n_cut, "n_resumed": len(already), "seconds": round(seconds, 2),
               "batches": timings, "fit": fit_timings(timings, args.max_gen), "cached_check": check, "max_items": args.max_items or None,
               "empty_generations": sum(1 for r in ordered if not r["generation"].strip()), "items_file_sha256": RB.sha256_file(done_items),
               # the labelled EOS cut (ced/eval/recall_based.py docstring): how many generations met <|endoftext|> inside the 48 tokens, and the hits without the cut
               "n_eos": sum(1 for r in ordered if r.get("eos_at", -1) >= 0), "hits_no_eos_cut": sum(int(r.get("hit_no_eos_cut", r["hit"])) for r in ordered)}
    RB.write_json(done_json, summary)
    if part.exists():
        part.unlink()
    say(f"{label}/{task}: acc {acc:.4f} ({summary['hits']}/{len(ordered)}) in {seconds:.0f} s; {summary['n_eos']} generations met <|endoftext|> "
        f"(hits without that cut: {summary['hits_no_eos_cut']}); fit {summary['fit']}")
    return {**summary, "reused": False, "records": ordered}


def cmd_score(args) -> int:
    import torch

    runs = parse_runs(args.run)
    if not runs:
        raise SystemExit("--run LABEL=PATH is needed")
    out = Path(args.out)
    items_dir = Path(args.items)
    tasks = [t for t in args.tasks.split(",") if t] if args.tasks else RB.available_tasks(items_dir, args.manifest)
    for t in tasks:
        if t not in RB.TASKS:
            raise SystemExit(f"--tasks {t!r}: one of {RB.TASKS}")
    if not tasks:
        raise SystemExit(f"{items_dir}: the manifest lists no task")
    loaded = {}
    for t in tasks:
        try:
            loaded[t] = RB.load_set(items_dir, t, args.manifest)
        except (OSError, ValueError, KeyError) as e:
            raise SystemExit(f"--items {items_dir} --manifest {args.manifest} task {t}: {e}")
    codec = RB.hf_codec(args.tokenizer)
    exclude = load_exclude(args.exclude_file)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if torch.device(device).type == "cuda":
        say(f"device {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    code_sha = os.environ.get("CED_CODE_SHA", "local")
    say(f"plan {args.tag}: runs {[r[0] for r in runs]} tasks {tasks} code {code_sha} max_gen {args.max_gen} max_length {args.max_length} batch_rows {args.batch_rows} "
        f"max_batch_tokens {args.max_batch_tokens}" + (f" max_items {args.max_items}" if args.max_items else ""))
    out.mkdir(parents=True, exist_ok=True)
    rows, missing, not_final, failed = [], [], [], []
    for label, path in runs:
        ck = resolve_checkpoint(path)
        if ck is None:
            say(f"{label}: no checkpoint at {path}")
            missing.append(label)
            continue
        t0 = time.time()
        try:
            model, cfg, step, autocast = load_reference_model(ck, device)
        except Exception as e:  # a model that cannot be rebuilt is a failure of this run only
            say(f"{label}: cannot build the model from {ck}: {type(e).__name__}: {e}")
            failed.append(label)
            continue
        if args.final_step and step != args.final_step:
            say(f"{label}: {ck} is at step {step}, not the final {args.final_step}: skipped")
            not_final.append(label)
            del model
            continue
        say(f"{label}: {ck.name} step {step}, arch {cfg.model.arch}, dtype {cfg.dtype}, autocast {autocast}, built in {time.time() - t0:.0f} s")
        recs_by_task = {}
        for task in tasks:
            items, rec = loaded[task]
            key = {"version": RB.VERSION, "task": task, "set_sha256": rec["set_sha256"], "tokenizer_sha256": codec.sha256, "step": step,
                   "max_gen": args.max_gen, "max_length": args.max_length, "max_items": args.max_items or None, "dtype": cfg.dtype, "autocast": str(autocast)}
            try:
                res = score_task(model, autocast, device, label, task, items, codec, out, key, args)
            except SystemExit:
                raise
            except Exception as e:
                say(f"{label}/{task}: FAILED {type(e).__name__}: {e}")
                failed.append(f"{label}/{task}")
                continue
            recs_by_task[task] = res.pop("records")
            rows.append((label, task, res))
        if recs_by_task:
            # the run's values come from EVERY finished task on disk, this pass's and the earlier ones: the cells of a task this
            # pass did not score are kept, not dropped
            RB.write_json(out / f"{label}.values.json", RB.values_json(label, code_sha, {**RB.read_run_records(out, label), **recs_by_task}, exclude))
        if args.probe_lengths:
            try:
                RB.write_json(out / label / "synthetic_timing.json", {"max_gen": args.max_gen, "batches": time_synthetic(
                    model, autocast, device, [int(x) for x in args.probe_lengths.split(",") if x], args)})
            except Exception as e:  # a timing probe never costs a result (a long batch may not fit the slice's memory)
                say(f"{label}: the synthetic timing failed: {type(e).__name__}: {e}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_result(out, args.tag, rows, missing, not_final, failed, args)
    if failed:
        return 1
    return 3 if (missing or not_final) else 0


def write_result(out: Path, tag: str, rows, missing, not_final, failed, args) -> None:
    lines = [f"# recall-based {tag} ({socket.gethostname()}, code {os.environ.get('CED_CODE_SHA', 'local')})", "",
             "| run | task | items | acc | hits | prompt tokens | cut | seconds | prefill tok/s | decode ms/step | cached check |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, task, r in rows:
        fit = r.get("fit") or {}
        ck = r.get("cached_check")
        ck_text = "-" if not ck else (f"{ck['mismatches']}/{ck['positions']} ({ck['items']} items)"
                                      + (f", beyond the bf16 bound {ck['bound']:.3f}: {ck['beyond_bound']}" if ck.get("mismatches") and "beyond_bound" in ck else ""))
        pf = fit.get("prefill_tokens_per_s")
        lines.append(f"| {label} | {task} | {r['n_items']} | {r['acc']:.4f} | {r['hits']} | {r['prompt_tokens']} | {r['n_truncated']} | {r['seconds']} | "
                     f"{'-' if not pf else f'{pf:.0f}'} | {fit.get('decode_step_ms', float('nan')):.1f} | {ck_text} |")
    lines += ["", f"missing: {missing or 'none'}; not final: {not_final or 'none'}; failed: {failed or 'none'}"]
    text = "\n".join(lines)
    (out / f"result-{tag}.md").write_text(text[:RESULT_LIMIT])
    print("RESULT")
    print(text[:RESULT_LIMIT])


# ------------------------------------------------------------------------------------------------ values


def cmd_values(args) -> int:
    out = Path(args.out)
    exclude = load_exclude(args.exclude_file)
    labels = [r.split("=", 1)[0] for r in args.run] if args.run else sorted(d.name for d in out.iterdir() if d.is_dir() and any(d.glob("*.items.jsonl.gz")))
    if not labels:
        raise SystemExit(f"{out}: no scored run")
    code_sha = os.environ.get("CED_CODE_SHA", "local")
    for label in labels:
        v = RB.values_json(label, code_sha, RB.read_run_records(out, label), exclude)
        target = Path(args.values_out) / f"{label}.values.json" if args.values_out else out / f"{label}.values.json"
        RB.write_json(target, v)
        say(f"{label}: {v['cells']} n {v['n_items']} excluded {len(v['excluded'])} -> {target}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--items", required=True, help="the directory of <task>.items.jsonl.gz and the manifest")
    s.add_argument("--manifest", default=RB.MANIFEST_NAME, help="the manifest file in --items (default manifest.json; tools/data/build_recall_based.py --subset-manifest writes others)")
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--run", action="append", default=[], help="LABEL=PATH (the label is the run's name)")
    s.add_argument("--tasks", default="", help="comma list (default: the tasks the manifest lists)")
    s.add_argument("--tag", default="score")
    s.add_argument("--final-step", type=int, default=RB.FINAL_STEP_200M, help="skip a checkpoint at another step (0: any)")
    s.add_argument("--max-gen", type=int, default=RB.MAX_GEN_TOKS)
    s.add_argument("--max-length", type=int, default=RB.TRAIN_LENGTH)
    s.add_argument("--batch-rows", type=int, default=64)
    s.add_argument("--max-batch-tokens", type=int, default=8192, help="rows x the longest prompt of a batch (the prefill's logits are rows x length x vocabulary)")
    s.add_argument("--max-items", type=int, default=0, help="the first N items of each task (smoke)")
    s.add_argument("--check-cached", type=int, default=0, help="N items: cached generation against the full forward")
    s.add_argument("--probe-lengths", default="", help="comma list of prompt lengths: time one batch of random-token prompts of each (the cost of long prompts; nothing is scored)")
    s.add_argument("--exclude-file", action="append", default=[])
    s.add_argument("--device", default="")
    s.add_argument("--force", action="store_true")
    v = sub.add_parser("values")
    v.add_argument("--out", required=True)
    v.add_argument("--run", action="append", default=[])
    v.add_argument("--exclude-file", action="append", default=[])
    v.add_argument("--values-out", default="")
    args = p.parse_args(argv)
    if args.cmd == "score":
        if args.max_gen < 1 or args.max_length <= args.max_gen or args.batch_rows < 1 or args.max_batch_tokens < 1:
            raise SystemExit("--max-gen, --max-length, --batch-rows and --max-batch-tokens: positive, and max-length above max-gen")
        return cmd_score(args)
    return cmd_values(args)


if __name__ == "__main__":
    sys.exit(main())
