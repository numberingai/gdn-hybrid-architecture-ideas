"""Train a model from a YAML config, resuming from the run directory's latest checkpoint if present.

    python scripts/train.py --config configs/tiny.yaml --run-dir runs/tiny
    python scripts/train.py --config configs/tiny.yaml --run-dir runs/tiny --wandb online

Run it again with the same --run-dir after an interruption and it continues where it stopped.
On resume the requested config (YAML plus overrides) must equal the checkpoint's config, apart
from W&B settings; otherwise the script stops and lists the differences. Pass
--allow-config-change to continue with the requested config anyway (for example to extend a run;
note that changing `steps` changes the LR schedule from the current step on).

A config with `init_from` (a checkpoint file) starts a NEW run from that checkpoint's model weights,
with a fresh optimizer, schedule and step counter (Trainer.init_from_checkpoint). Only when the run
directory has no checkpoint yet: otherwise the run resumes from its own latest checkpoint as above.

A config with `resume_from` (a checkpoint file of ANOTHER run) starts a new run that continues that run's
whole training state under this config (a nested budget's branch; Trainer.resume_from_checkpoint, strict).
Again only when the run directory has no checkpoint yet; `resume_from` takes precedence over `init_from`.

The loop experiment: a gated two-pass config (`model.loop.pass2_gate`) may take `init_from` from the checkpoint
of its single-pass model, and, with `loop_switch_step` S, `resume_from` from its T = 1 trunk's kept checkpoint
of step S; only pass 2's own tensors are new (Trainer.init_from_checkpoint / resume_from_checkpoint).

Exit status: 0 when the run reached its last step (or stopped on SIGTERM after its checkpoint); 86
(ced.train.trainer.EXIT_DIVERGED) when a step's loss or gradient norm was not finite, or a batch failed the loader's
validation (ced/train/batch_check.py): the trainer has written {"event": "diverged"} (or {"event": "bad_batch"}) to
metrics.jsonl and the marker <run-dir>/diverged.json and no checkpoint, and every later start in that run directory
exits with 86 at once, before it builds anything. Any other status is a crash.

Data parallelism (ced/train/data_parallel.py; D1): started by torchrun.

    torchrun --nproc_per_node 8 scripts/train.py --config ... --run-dir ...

each process reads RANK, LOCAL_RANK and WORLD_SIZE, takes its device (cuda:LOCAL_RANK) and joins the process groups
before anything is built (data_parallel.from_environment); the trainer then trains the rank's share of every batch.
CED_DP_BACKEND=gloo carries the tensor group on gloo (the one-slice check), CED_DP_DEVICES=<dev0>,<dev1>,... gives each
local rank its own device as its CUDA_VISIBLE_DEVICES, then cuda:0 (one MIG slice a rank: the two-slice probe),
CED_DP_CHECK_EVERY=N compares the ranks'
parameters every N steps (they are compared once before every process's first step anyway), CED_DP_WAIT_SECONDS sets
how long a rank waits for rank 0's evals and saves (default 7200). Rank 0 writes the run's files and logs (D4); every
rank resumes from the run directory's latest checkpoint, at any world size (D5): the world size is not part of the
config this script compares, and micro_batch_size, activation_checkpoint and optimizer_sharding
(ced.train.trainer.DP_RESUME_FREE) may differ from the checkpoint's without --allow-config-change when the run is
data-parallel or was written at another world size (the change is recorded in the run's "data_parallel" history). A run
with optimizer_sharding "zero1" (the sharded optimizer, ced/train/zero1.py) writes each checkpoint as rank 0's step file
and one optimizer file a rank beside it (step_<s>.optim/); any world size resumes from it, sharded or not. Without
torchrun's environment (or at WORLD_SIZE 1) nothing of it runs: the script is what it was. Note: torchrun itself exits 1
when its workers exit 86; the marker file says why.
"""

from __future__ import annotations

import argparse
import gc

import torch

from ced.config import load_yaml, to_dict
from ced.train import data_parallel
from ced.train.trainer import (DIVERGED_MARKER, DP_RESUME_FREE, EXIT_DIVERGED, Diverged, Trainer, diverged_marker,
                                fp8_comparable)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, default=None, help="override cfg.steps (schedule length)")
    parser.add_argument("--until-step", type=int, default=None, help="stop early at this step")
    parser.add_argument("--wandb", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--allow-config-change", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config field by dotted path, value parsed as YAML (e.g. data.batch_size=32)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_yaml(args.config)
    for item in args.set:
        apply_override(cfg, item)
    if args.steps is not None:
        cfg.steps = args.steps
    if args.wandb is not None:
        cfg.wandb.mode = args.wandb

    # A run that stopped on a non-finite loss (ced/train/trainer.py, "The stop on a non-finite loss") is not trained
    # again: its marker ends every later start here, before anything is built, with the same exit status.
    marker = diverged_marker(args.run_dir)
    if marker is not None:
        print(f"[train] {args.run_dir} holds {DIVERGED_MARKER}: {marker['message']}. Nothing is trained; exit status "
              f"{EXIT_DIVERGED}. (A run under a new name starts afresh.)", flush=True)
        raise SystemExit(EXIT_DIVERGED)
    # D1: under torchrun, this rank's device and process groups, before anything is built; None in one process
    dp = data_parallel.from_environment(cfg.device)
    try:
        trainer = prepare(cfg, args.run_dir, args.allow_config_change, dp=dp)
        try:
            trainer.fit(until_step=args.until_step)
        except Diverged as stop:
            # not a crash and not a finished run: the record is in metrics.jsonl and in the marker file
            if dp is None or dp.lead:
                print(f"[train] {stop}: the run has ended; exit status {EXIT_DIVERGED}", flush=True)
            raise SystemExit(EXIT_DIVERGED) from None
    finally:
        if dp is not None:
            dp.close()


def prepare(cfg, run_dir, allow_config_change: bool = False, log=print, dp=None) -> Trainer:
    """The trainer a run continues with: resumed from the run directory's latest checkpoint when there
    is one (its config must equal `cfg` apart from W&B unless allow_config_change); otherwise a new run:
    continuing cfg.resume_from's training state when set (Trainer.resume_from_checkpoint), else from
    cfg.init_from's model weights when set (Trainer.init_from_checkpoint), else from scratch. dp: the process's data
    parallelism (ced/train/data_parallel.py; None in one process): rank 0 logs, and a resume that is data-parallel or
    changes the world size may change DP_RESUME_FREE (micro_batch_size, activation_checkpoint, optimizer_sharding) without
    allow_config_change (the trainer records the change)."""
    if dp is not None and not dp.lead:
        log = _quiet
    trainer = Trainer(cfg, run_dir, dp=dp)
    for note in trainer.tokenizer_checks:  # the data's tokenizer vs the vocabulary (refused on a mismatch)
        log(note)
    latest = trainer.latest_checkpoint()
    if latest is not None:
        saved, world = Trainer.saved_config_and_world(latest)  # the checkpoint's config and world size (one read)
        differences = config_differences(to_dict(saved), to_dict(cfg))
        if differences and (dp is not None or world != 1):  # data parallelism (D5): how a step is held in memory
            free = [d for d in differences if d.split(":", 1)[0] in DP_RESUME_FREE]
            differences = [d for d in differences if d not in free]
            if free:
                log(f"resuming at world size {1 if dp is None else dp.world} (the checkpoint's: {world}); changed: "
                    + "; ".join(free))
        if differences and not allow_config_change:
            raise SystemExit(
                "the requested config differs from the checkpoint's config:\n  "
                + "\n  ".join(differences)
                + "\nresume with the original config, or pass --allow-config-change"
            )
        if differences:
            log("continuing with the requested config; changed: " + "; ".join(differences))
        # Built from the requested config (equal to the saved one apart from W&B, unless allowed).
        if dp is not None:  # data parallelism: one trainer at a time (each holds a flat gradient buffer: 13.74 GiB at 4B)
            del trainer
            gc.collect()
        trainer = Trainer.from_checkpoint(latest, run_dir, cfg=cfg, dp=dp)
        log(f"resumed from {latest} at step {trainer.step}")
    elif cfg.resume_from:
        info = trainer.resume_from_checkpoint(cfg.resume_from)
        stage = info.get("stage")
        on = ("" if stage is None else
              f"; stage 'continue' on {'the source data, its stream going on' if stage['data'] == 'same' else 'new data, its stream from its beginning'}"
              f" (data_start_step {stage['data_start_step']}), decay_fraction {stage['source_decay_fraction']} -> {stage['decay_fraction']}")
        shape = (stage or {}).get("shape")
        if shape:  # stage_shape "tokens": the rows a step and the row length, the source's and this run's
            on += (f"; rows {shape['source'][0]} x {shape['source'][1]} -> {shape['this_run'][0]} x {shape['this_run'][1]} "
                   f"({shape['tokens_per_step']} tokens a step), micro_batch_size and activation_checkpoint "
                   f"{shape['memory']['micro_batch_size'][0]}/{shape['memory']['activation_checkpoint'][0]} -> "
                   f"{shape['memory']['micro_batch_size'][1]}/{shape['memory']['activation_checkpoint'][1]}")
        prec = info.get("precision")  # TrainConfig.precision_switch: absent with it off (the line of before)
        if prec:
            on += (f"; precision nvfp4 {prec['source']['nvfp4']['mode']} / fp8 {prec['source']['fp8']['mode']} -> nvfp4 "
                   f"{prec['this']['nvfp4']['mode']} / fp8 {prec['this']['fp8']['mode']}")
        log(f"new run continued from {info['path']} (step {info['step']} of {info['source_steps']}, sha256 "
            f"{info['sha256']}): the whole training state; this run's schedule has {cfg.steps} steps{on}")
    elif cfg.init_from:
        info = trainer.init_from_checkpoint(cfg.init_from)
        log(f"new run initialised from {info['path']} (step {info['step']} of {info['source_steps']}, "
            f"sha256 {info['sha256']}): model weights only; fresh optimizer, LR schedule and step 0")
    return trainer


def _quiet(*args, **kwargs) -> None:
    """The log of every data-parallel rank but rank 0 (D4)."""


def apply_override(cfg, item: str) -> None:
    """Set one field: "a.b.c=value", the value parsed as YAML. Unknown fields are an error."""
    import dataclasses

    import yaml

    key, _, raw = item.partition("=")
    obj = cfg
    *path, last = key.split(".")
    for part in path:
        obj = getattr(obj, part)
    if not (dataclasses.is_dataclass(obj) and last in {f.name for f in dataclasses.fields(obj)}):
        raise SystemExit(f"--set: unknown config field {key!r}")
    setattr(obj, last, yaml.safe_load(raw))


def config_differences(saved: dict, requested: dict, prefix: str = "") -> list[str]:
    """Dotted keys whose values differ, ignoring W&B settings. model.fp8.nonfinite_flag_form is compared as the model
    reads it (ced/train/trainer.py fp8_comparable): only with the flag on and quantiser "triton". Its default changed
    from "ordered" to "pure", so every flag-off checkpoint written before then stores "ordered" and the same runs
    file now gives "pure": an inert key, not a difference."""
    out = []
    for key in sorted(set(saved) | set(requested)):
        name = f"{prefix}{key}"
        if name == "wandb":
            continue
        a, b = saved.get(key), requested.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            if name == "model.fp8":  # the flag's form as it is read
                a, b = fp8_comparable(a), fp8_comparable(b)
            out += config_differences(a, b, name + ".")
        elif a != b:
            out.append(f"{name}: checkpoint {a!r} -> requested {b!r}")
    return out


if __name__ == "__main__":
    main()
