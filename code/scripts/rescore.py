"""Re-score a checkpoint on eval sets: the final eval (reference head; with the fused-kernel check
when the run trained with the fused head), as at the end of training, on sets that may have been
rebuilt since.

    python scripts/rescore.py --checkpoint <checkpoint>.pt \
        --set web=<eval sets>/web.pt --set code=<eval sets>/code.pt --set agent=<eval sets>/agent.pt --set general_web=<eval sets>/general_web.pt \
        --out <out>/rescore_final.json

The eval sets are EvalRows files (ced/eval/span_bpb.py); EVAL.md describes the format.

The trainer is rebuilt from the checkpoint's own config (so compile, autocast, the fused varlen path
and the head settings are the run's); only the model weights are loaded (no data, no optimizer
state). Every arch is scored as in training's final eval (ced/eval/span_bpb.evaluate): v1 Looped-CED
with the decoder at the rows' decoder positions, CED v2 and the second-pass control with it at every
scored position (plus the aux head and, for CED v2, the loop-1 readout), the controls by their head.
Results: one JSON like <run>/eval/step_<step>.json, plus the eval sets' sha256 and the checkpoint
path, and per-item multiple-choice results next to it.

--decoder-at rows / rows_sinks (CED v2 only; EvalConfig.decoder_at): the decoder at the rows' own decoder positions
only (rows_sinks: plus each document's first n_sinks positions, not scored), the restricted regime;
dec_bpb is then scored there. Without it (the default)
every arch is scored as above, whatever the checkpoint's own eval config says. The choice is recorded in the output
("decoder_at").
--decoder-at rows_sinks_replay [--replay-window N, default 128]: rows_sinks plus
the decoder at the N positions of the same document before each run of consecutive scored positions (each span, each
assistant turn; not scored): the restricted regime with the replay window, dec_bpb at the same positions as the other
regimes. Recorded in the output ("replay_window"; None in every other regime).
--replay-window turn / turn<N> (turn128 is the whole last turn, capped at 128 positions;
ced/eval/replay_window.py): the decoder at the whole last message before each assistant turn, read from the
<|im_start|> tokens, capped at its last 512 / N positions. A document without chat tokens (the web, code and
general_web sets) has no turn boundary: the replay starts at the document's first position in the row, so there
turn<N> is replay N, position for position; only the agent set's positions differ. The eval sets must record a known
tokenizer (tokenizer_sha256.txt next to them): the <|im_start|> id is the chat format's. Recorded as the name.

--report-memory (default off; used for the 32k pass): after the pass,
print the GPU memory the pass peaked at (torch.cuda.max_memory_allocated / max_memory_reserved, from the start of the pass)
and the longest eval row, so that the log of a 32k pass records what a row of 32,768 positions needs. It prints only: the scores
and the output file are the same with it or without.

--eval-batch-size N (default: the checkpoint's own eval.batch_size, as always): rows a forward. The 32k pass
uses 1, so that a row of 32,768 positions is the whole forward whatever the checkpoint's config says.

--resume (or CED_SCORE_RESUME=1; default off; ced/eval/units.py): each
eval set's result is kept in <out>.units/ when the set is done, and a rerun with the same checkpoint, set file,
settings and code reads it instead of scoring the set; --force (CED_SCORE_FORCE=1) scores every set again. The per-item
files are moved next to --out first and --out is written last, each under a temporary name and renamed, with or
without units: a stopped pass leaves no --out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch

from ced.config import TSTConfig, TrainConfig, from_dict
from ced.eval import units
from ced.eval.replay_window import is_turn, parse_replay_window, replay_window_name
from ced.eval.span_bpb import REPLAY_WINDOW
from ced.train.trainer import Trainer


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def window_arg(text: str):
    """--replay-window: an integer, or a last-turn name ("turn", "turn<N>") in its canonical form."""
    if is_turn(text):
        return replay_window_name(text)
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: an integer, 'turn' or 'turn<N>' (e.g. 'turn128')") from None


def scoring_tst_off(cfg: TrainConfig, step: int) -> None:
    """Scoring reads a checkpoint's model with next-token evals on synthetic data, so Token Superposition Training's
    training-time settings (TrainConfig.tst: the bag stream reads the packed shards, ced/train/tst.py refuses synthetic
    data) are turned off here. Allowed only for a checkpoint whose step is in a next-token phase (s = 1): every run's
    final checkpoint is (the schedule ends at s 1); a checkpoint saved during a bag phase is refused, because its model
    predicts bags. What the switch did to the weights (gdn_decay_rescale's A_log shift) is in the checkpoint already."""
    sched = list(cfg.tst.schedule or [])
    if not sched:
        return
    s = next(int(v) for start, v in reversed(sched) if step >= int(start))
    if s != 1:
        raise SystemExit(f"rescore: step {step} is in a TST bag phase (s = {s}); scoring reads next-token checkpoints only")
    cfg.tst = TSTConfig()


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--set", action="append", required=True, help="name=path of an eval set")
    from ced.eval import schedule_override

    schedule_override.add_argument(p)
    p.add_argument("--out", required=True)
    p.add_argument("--decoder-at", choices=["every", "rows", "rows_sinks", "rows_sinks_replay"], default=None,
                   help="CED v2: 'rows' = the decoder at the rows' decoder positions only, 'rows_sinks' = "
                        "plus each document's first n_sinks positions (not scored), 'rows_sinks_replay' = plus the "
                        "--replay-window positions before each run of scored positions (not scored); default: as trained")
    p.add_argument("--replay-window", type=window_arg, default=REPLAY_WINDOW,
                   help="--decoder-at rows_sinks_replay: the decoder also runs at the last N positions of the same "
                        "document before each span / assistant turn (default %(default)s), or with 'turn' / 'turn<N>' at "
                        "the whole last message before each assistant turn, capped at 512 / N positions (turn128)")
    p.add_argument("--resume", action="store_true",
                   help="resumable pass (also CED_SCORE_RESUME=1): each eval set's result is kept in <out>.units/ when it is "
                        "done; a rerun with the same checkpoint, set files and settings reads it instead of scoring the set. "
                        "The --out file and the per-item files are the same as without (ced/eval/units.py)")
    p.add_argument("--force", action="store_true", help="--resume: score every set again and overwrite its unit (also CED_SCORE_FORCE=1)")
    p.add_argument("--report-memory", action="store_true",
                   help="print the peak GPU memory of the pass and the longest eval row (prints only; default off)")
    p.add_argument("--eval-batch-size", type=int, default=None,
                   help="rows a forward, in place of the checkpoint's eval.batch_size (default: the checkpoint's)")
    p.add_argument("--units-dir", default=None, help="--resume: where the unit files are (default <out>.units)")
    a = p.parse_args(argv)
    replay = a.decoder_at == "rows_sinks_replay"
    if replay and not isinstance(a.replay_window, str) and a.replay_window < 1:
        p.error(f"--replay-window {a.replay_window}: at least 1 with --decoder-at rows_sinks_replay")
    if replay and isinstance(a.replay_window, str):
        parse_replay_window(a.replay_window)  # a last-turn name: checked here, before the checkpoint is read
    state = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    cfg = from_dict(TrainConfig, state["config"])
    cfg.data.kind = "synthetic"  # no training data is read
    # the per-token loss mask is a training setting (DataConfig.loss_mask; the trainer refuses it on synthetic data):
    # scoring never applies it, every eval set is scored as always ("all" is every checkpoint before the key)
    cfg.data.loss_mask = "all"
    # the sets of a union run (DataConfig.extra_shard_dirs, union_order; the scale checks' branches) are training data too: the trainer refuses them with synthetic data
    cfg.data.extra_shard_dirs, cfg.data.union_order = [], "permute"
    scoring_tst_off(cfg, state["step"])  # a TST run is scored as the next-token model it ends as (no bag stream here)
    cfg.eval.sets = dict(item.split("=", 1) for item in a.set)
    if a.eval_batch_size is not None:
        if a.eval_batch_size < 1:
            p.error("--eval-batch-size: at least 1")
        cfg.eval.batch_size = a.eval_batch_size
    cfg.eval.decoder_at = a.decoder_at
    cfg.eval.replay_window = a.replay_window
    work = Path(tempfile.mkdtemp())
    t = Trainer(cfg, work)  # refuses eval sets of another tokenizer (tokenizer_sha256.txt next to them)
    for note in t.tokenizer_checks:
        print(note, flush=True)
    t.model.load_state_dict(state["model"])
    from ced.eval import schedule_override

    segment_map = schedule_override.apply(t.model, a.segment_map)  # None: the pass schedule the checkpoint was trained with
    t.step = state["step"]
    store = None
    if a.resume or units.env_flag(units.ENV_RESUME):
        store = t.unit_store = units.UnitStore(a.units_dir or f"{a.out}.units", {
            "script": "rescore", **({} if a.segment_map == "trained" else {"segment_map": a.segment_map}), "checkpoint": a.checkpoint, "step": t.step, "checkpoint_file": units.file_fingerprint(a.checkpoint),
            "arch": cfg.model.arch, "torch": torch.__version__,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"},
            force=a.force or units.env_flag(units.ENV_FORCE), scripts=(__file__,))
    if a.report_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    results = t.run_evals(periodic=False)
    if a.report_memory:
        rows = [r for r in getattr(t, "_eval_rows", {}).values() if hasattr(r, "seq_len")]
        gib = 1024 ** 3
        peak = (f"peak GPU memory allocated {torch.cuda.max_memory_allocated() / gib:.2f} GiB, reserved "
                f"{torch.cuda.max_memory_reserved() / gib:.2f} GiB" if torch.cuda.is_available() else "no GPU: no memory peak")
        print(f"memory: {peak}; eval.batch_size {cfg.eval.batch_size}, longest eval row "
              f"{max((r.seq_len for r in rows), default=0)} positions, {len(rows)} eval sets", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {"checkpoint": a.checkpoint, "step": t.step, "decoder_at": a.decoder_at,
              "segment_map": a.segment_map, "segment_map_counts": segment_map,
              "replay_window": a.replay_window if replay else None, "eval_sets": {n: {"path": s, "sha256": sha256(s)} for n, s in cfg.eval.sets.items()},
              "tokenizer_checks": t.tokenizer_checks, "results": results}
    # The per-item files first, --out last, each under a temporary name and renamed: --out is the mark of a finished
    # pass, so a stop between the two never leaves an --out
    # whose per-item files are missing or cut short.
    for f in (work / "eval").glob("*_items.jsonl.gz"):  # per-item multiple-choice results
        units.move_atomic(f, out.parent / f"{out.stem}_{f.name}")  # across file systems
    units.write_atomic(out, json.dumps(record))
    if store is not None:
        for note in store.notes:
            print(note, flush=True)
        print(f"units: {store.reused} eval sets read from {store.dir}, {store.scored} scored", flush=True)
    # dec_bpb is the main head's (CED v2 and control (c): the decoder / pass 2 at every scored position);
    # aux_bpb and loop1_bpb are their diagnostics at the same positions, when the model has them.
    keys = ("dec_bpb", "all_bpb", "aux_bpb", "loop1_bpb")
    summary = {n: {k: r[k] for k in keys if r.get(k) is not None}
               | ({"fused_check_ok": r["fused_check"]["within_criteria"]} if r.get("fused_check") else {})
               for n, r in results.items() if isinstance(r, dict) and "dec_bpb" in r}
    print(json.dumps({"checkpoint": a.checkpoint, "step": t.step, "summary": summary}, indent=1), flush=True)


if __name__ == "__main__":
    main()
