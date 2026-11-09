"""Build the Based recall suite's item files: SQuAD completion, SWDE and FDA. EVALUATION ONLY. The sources, the reference's definition and the choices are in the
docstring of ced/eval/recall_based.py.

    python tools/data/build_recall_based.py --raw <raw> --out <items> [--tasks squad,swde,fda] [--tokenizer <tokenizer.json>] \\
        [--reference-tokenizer <the reference's GPT-2 tokenizer.json>] [--texts] [--subset-manifest manifest-swde-fda.json=swde,fda]

Each file is fetched at its pinned revision from https://huggingface.co (a file already under --raw/<task>/ is reused) and checked against its pinned sha256 and row
count; a mismatch stops the build. Needs pyarrow. SQuAD 2.0 is CC BY-SA 4.0; SWDE and FDA state no licence, so the items are downloaded for scoring and never
redistributed: keep the downloaded files and the built items out of any repository. --subset-manifest writes a second manifest listing some of the tasks only.

Writes under --out:
    <task>.items.jsonl.gz            the items (raw text, the tokenizer is applied at scoring), a fixed gzip header so the same items give the same file
    manifest.json                    per task: the file, its sha256, the set sha256, the item count, the source record (the file load_set verifies)
    recall_based.contamination.json  per set: source, revision, file sha256, split, item count, item-text hashes, licence note (hashes, no text)
    report.json                      counts, repeats, and with --tokenizer the prompt token counts per task
    itemtexts/<task>.jsonl.gz        with --texts: one JSON line an item {id, context, completion, gold, text} for an n-gram contamination scan (text = completion + " " + gold)
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))

from ced.eval import recall_based as RB  # noqa: E402

FORMAT = "recall_based contamination list/1"


def say(text: str) -> None:
    print(f"[build-recall-based] {text}", flush=True)


def source_url(task: str) -> str:
    s = RB.SOURCES[task]
    return f"https://huggingface.co/datasets/{s['repo']}/resolve/{s['revision']}/{s['file']}"


def fetch(task: str, raw: Path, opener=urllib.request.urlopen) -> Path:
    """The pinned file of `task` under raw/<task>/, downloaded if it is not there; SystemExit if its sha256 is not the pinned one."""
    s = RB.SOURCES[task]
    path = raw / task / Path(s["file"]).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        with opener(source_url(task), timeout=300) as r, open(tmp, "wb") as f:
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
        tmp.replace(path)
    got = RB.sha256_file(path)
    if got != s["sha256"]:
        raise SystemExit(f"{task}: {path} has sha256 {got}, the pinned one is {s['sha256']} (revision {s['revision']}): not used")
    return path


def read_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


SUMMARY_MARKER = "Summary of information above...\n"      # SWDE prompts: the page, then this line, then "key: value" lines of other attributes of the page, then the asked "key:"


def item_text_record(it: RB.Item) -> dict:
    """What an n-gram contamination scan reads: the passage or document (`context`), the question part (`completion`: RQ), the gold, and completion + gold (`text`: RG).
    SWDE and FDA: `completion` is the prompt's last line (the attribute name and a colon, "year:"), `context` the prompt without it; SWDE items also carry `summary`, the
    "key: value" lines between the marker and that last line (the dataset's own structured view of the page, which the web page does not contain)."""
    if it.task == "squad":
        core = it.prompt.strip()
        completion = it.meta.get("completion", "")
        context = core[: len(core) - len(completion.strip())] if completion and core.endswith(completion.strip()) else core
        return {"id": it.id, "context": context, "completion": completion, "gold": it.gold, "text": (completion + " " + it.gold).strip()}
    completion = it.meta.get("completion", "")
    body = it.prompt.rstrip()
    context = body[: len(body) - len(completion)].rstrip() if completion and body.endswith(completion) else it.prompt
    rec = {"id": it.id, "context": context, "completion": completion, "gold": it.gold, "text": (completion + " " + it.gold).strip()}
    if it.task == "swde" and SUMMARY_MARKER in context:
        rec["summary"] = context.split(SUMMARY_MARKER, 1)[1].strip()
    return rec


def write_texts(path: Path, items: list[RB.Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "\n".join(json.dumps(item_text_record(it), ensure_ascii=False) for it in items) + "\n"
    path.write_bytes(RB.gzip_bytes(data.encode("utf-8")))


def token_summary(lens: list[int]) -> dict:
    lens = sorted(lens)
    n = len(lens)
    return {"total": sum(lens), "min": lens[0], "mean": round(sum(lens) / n, 1), "median": lens[n // 2], "p90": lens[int(n * 0.9)], "p99": lens[int(n * 0.99)], "max": lens[-1]}


def token_report(items_by_task: dict[str, list[RB.Item]], tokenizer: str, reference_tokenizer: str = "") -> dict:
    """Prompt token counts per task with the project tokenizer (`over_8144_cut_at_our_length`: the prompts the left truncation at 8,144 tokens would cut; `prompt_plus_48_over_8192`: prompt plus the 48
    new tokens above the 8,192 positions). With `reference_tokenizer` (the reference's GPT-2 tokenizer.json) the same prompts counted the reference's way: `reference` holds
    the counts and `over_reference_2000` is then the number of prompts the REFERENCE's left truncation (max_length 2,048 - 48 new tokens = 2,000) cuts, and
    `reference_tokens_cut` how many tokens it drops from them. Without it `over_reference_2000` counts the project's tokens above 2,000, which is not the reference's cut."""
    codec = RB.hf_codec(tokenizer)
    ref = RB.hf_codec(reference_tokenizer) if reference_tokenizer else None
    out = {"tokenizer_sha256": codec.sha256, "reference_tokenizer_sha256": ref.sha256 if ref else None, "tasks": {}}
    keep = RB.TRAIN_LENGTH - RB.MAX_GEN_TOKS
    room = RB.REFERENCE_MAX_LENGTH - RB.MAX_GEN_TOKS
    for task, items in items_by_task.items():
        lens = sorted(len(codec.encode(it.prompt)) for it in items)
        gold = [len(codec.encode(" " + it.gold)) for it in items]
        n = len(lens)
        out["tasks"][task] = {
            "items": n, "prompt_tokens_total": sum(lens), "min": lens[0], "mean": round(sum(lens) / n, 1), "median": lens[n // 2], "p90": lens[int(n * 0.9)],
            "p99": lens[int(n * 0.99)], "max": lens[-1],
            "over_reference_2000": sum(1 for x in lens if x > room),
            "over_8144_cut_at_our_length": sum(1 for x in lens if x > keep),
            "prompt_plus_48_over_8192": sum(1 for x in lens if x + RB.MAX_GEN_TOKS > RB.TRAIN_LENGTH),
            "gold_tokens_mean": round(statistics.mean(gold), 2), "gold_tokens_max": max(gold),
            # golds the 48 new tokens cannot hold, or that a newline stop cuts: a miss by construction, in the reference as here
            "gold_over_48_tokens": sum(1 for g in gold if g > RB.MAX_GEN_TOKS), "gold_with_newline": sum(1 for it in items if "\n" in it.gold),
            "gold_shorter_than_4_chars": sum(1 for it in items if len(it.gold) < 4),
            "lengths": lens,
        }
        if ref is not None:
            r = sorted(len(ref.encode(it.prompt)) for it in items)
            out["tasks"][task]["reference"] = token_summary(r)
            out["tasks"][task]["over_reference_2000"] = sum(1 for x in r if x > room)
            out["tasks"][task]["reference_tokens_cut"] = sum(x - room for x in r if x > room)
            out["tasks"][task]["gold_over_48_tokens_reference"] = sum(1 for it in items if len(ref.encode(" " + it.gold)) > RB.MAX_GEN_TOKS)
    return out


def contamination_record(task: str, built: dict | None) -> dict:
    """The record of a set (hashes and the licence note, never an item)."""
    s = RB.SOURCES[task]
    rec = {"repo": s["repo"], "revision": s["revision"], "file": s["file"], "file_sha256": s["sha256"], "split": s["split"], "rows_in_file": s["rows"],
           "licence": s["licence"]}
    if built is None:
        rec.update({"items": None, "item_texts_sha256": None, "note": "not downloaded and not read; rows_in_file is the hub card's num_examples and file_sha256 the hub's LFS oid"})
    else:
        rec.update({"items": built["items"], "item_texts_sha256": built["set_sha256"], "duplicates_dropped": 0,
                    "exact_prompt_gold_repeats": built["exact_prompt_gold_repeats"], "ids_made_unique": built["ids_made_unique"]})
    return rec


def parse_subset_manifests(specs: list[str]) -> dict[str, list[str]]:
    """--subset-manifest NAME=TASK,TASK : {file name: tasks}. A name is a plain file name ending in .json (never manifest.json itself: that is the full manifest)."""
    out: dict[str, list[str]] = {}
    for spec in specs:
        name, _, tasks = spec.partition("=")
        ts = [t for t in tasks.split(",") if t]
        if not name.endswith(".json") or "/" in name or name == RB.MANIFEST_NAME or not ts or any(t not in RB.TASKS for t in ts):
            raise SystemExit(f"--subset-manifest {spec!r}: NAME.json=TASK,TASK with the tasks among {RB.TASKS}, a plain file name other than {RB.MANIFEST_NAME}")
        out[name] = ts
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw", required=True, help="where the pinned files are kept (fetched if missing)")
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", default="squad,swde,fda", help="comma list of %s (default: all three)" % ",".join(RB.TASKS))
    p.add_argument("--tokenizer", default="", help="a tokenizer.json: prompt token counts per task in report.json")
    p.add_argument("--reference-tokenizer", default="", help="the reference's GPT-2 tokenizer.json: the same prompts counted the reference's way, and what its left truncation cuts")
    p.add_argument("--texts", action="store_true", help="also write the per-task item texts (itemtexts/)")
    p.add_argument("--subset-manifest", action="append", default=[], metavar="NAME.json=TASK,TASK",
                   help="also write a manifest of those tasks alone under that name (for example manifest-swde-fda.json=swde,fda; scripts/recall_based_score.py --manifest reads it)")
    a = p.parse_args(argv)
    tasks = [t for t in a.tasks.split(",") if t]
    subsets = parse_subset_manifests(a.subset_manifest)
    for name, ts in subsets.items():
        if any(t not in tasks for t in ts):
            raise SystemExit(f"--subset-manifest {name}: tasks {ts} are not all among --tasks {tasks}")
    for t in tasks:
        if t not in RB.TASKS:
            raise SystemExit(f"--tasks {t!r}: one of {RB.TASKS}")
        say(f"{t}: licence note: {RB.SOURCES[t]['licence']}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_by_task = {}
    for t in tasks:
        path = fetch(t, Path(a.raw))
        rows_by_task[t] = read_rows(path)
        if len(rows_by_task[t]) != RB.SOURCES[t]["rows"]:
            raise SystemExit(f"{t}: {len(rows_by_task[t])} rows, the pinned file has {RB.SOURCES[t]['rows']}")
        say(f"{t}: {len(rows_by_task[t])} rows, sha256 ok ({RB.SOURCES[t]['repo']} @ {RB.SOURCES[t]['revision'][:12]})")
    items_by_task, report = RB.build_items(rows_by_task)
    manifest = {"format": "recall_based items/1", "version": RB.VERSION, "built_from": "tools/data/build_recall_based.py", "tasks": {}}
    contamination = {"format": FORMAT, "built_from": "tools/data/build_recall_based.py",
                     "method": "item_sha256 = sha256 (hex) of the UTF-8 bytes of json.dumps([task, id, prompt, gold], ensure_ascii=False, separators=(',', ':')); "
                               "item_texts_sha256 = sha256 (hex) of the item_sha256 values of the set's items, in file order, joined by '\\n'. The texts are in "
                               "<task>.items.jsonl.gz (the items) and itemtexts/<task>.jsonl.gz (the texts for a scan) under --out (external sets, scores only, "
                               "items never redistributed)",
                     "reference": {"repo": RB.HARNESS_REPO, "commit": RB.HARNESS_COMMIT, "files_sha256": RB.HARNESS_FILES_SHA256}, "tasks": {}}
    for t in RB.TASKS:
        if t not in items_by_task:
            contamination["tasks"][t] = contamination_record(t, None)
            continue
        items = items_by_task[t]
        digest = RB.write_items(out / f"{t}.items.jsonl.gz", items)
        manifest["tasks"][t] = {"file": f"{t}.items.jsonl.gz", "file_sha256": digest, "set_sha256": RB.set_sha256(items), "items": len(items),
                                "source": {k: RB.SOURCES[t][k] for k in ("repo", "revision", "file", "sha256", "rows", "split", "licence")}}
        contamination["tasks"][t] = contamination_record(t, {"items": len(items), "set_sha256": manifest["tasks"][t]["set_sha256"], **report["tasks"][t]})
        if a.texts:
            write_texts(out / "itemtexts" / f"{t}.jsonl.gz", items)
            contamination["tasks"][t]["item_texts_file_sha256"] = RB.sha256_file(out / "itemtexts" / f"{t}.jsonl.gz")      # the file a scan reads (item_texts_sha256 is the set's)
        say(f"{t}: {len(items)} items, file sha256 {digest[:16]}, set sha256 {manifest['tasks'][t]['set_sha256'][:16]}; {report['tasks'][t]}")
    if a.tokenizer:
        report["tokens"] = token_report(items_by_task, a.tokenizer, a.reference_tokenizer)
        for t, r in report["tokens"]["tasks"].items():
            say(f"{t}: prompt tokens {r['prompt_tokens_total']:,} (mean {r['mean']}, p90 {r['p90']}, max {r['max']}); over 2000 "
                f"({'reference tokenizer: the reference cuts these' if a.reference_tokenizer else 'project tokenizer, not the reference cut'}): {r['over_reference_2000']}; "
                f"over 8,144 (cut at the training length): {r['over_8144_cut_at_our_length']}; prompt + 48 over 8,192: {r['prompt_plus_48_over_8192']}; "
                f"gold tokens mean {r['gold_tokens_mean']} max {r['gold_tokens_max']}, over 48: {r['gold_over_48_tokens']}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    for name, ts in subsets.items():
        sub = {**manifest, "tasks": {t: manifest["tasks"][t] for t in ts}}
        (out / name).write_text(json.dumps(sub, indent=1, sort_keys=True))
    (out / "recall_based.contamination.json").write_text(json.dumps(contamination, indent=1, sort_keys=True))
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    held = [t for t in RB.TASKS if t not in items_by_task]
    say(f"built {sorted(items_by_task)}; held (not downloaded): {held}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
