"""Score a checkpoint under a different pass map than the one it was trained with.

A checkpoint trained with `model.loop.segment_passes` carries its map and tokenizer in its config, and every scorer that builds the
model from that config applies the schedule the checkpoint was trained with: the model reads the counts from the tokens itself
(ced/train/pass_positions.py PassScheduler) when a scorer gives no plan. This module changes only the counts, never the weights or
the built passes: the same checkpoint scored

    trained   the map it was trained with (the default; nothing is changed)
    all2      every segment at 2 passes   (a 3-pass model runs no third pass)
    all1      every segment at 1 pass     (the pass-1 readout everywhere)
    SEG:N,..  any other map, e.g. "reasoning:3,call_head:3,content:2,call_body:1,other:2" (a key left out counts 2)

so the same file of per-token losses can be had for one checkpoint at its own schedule and at all-2 and at all-1 for comparison.
A count above the model's built passes is refused (a 2-pass model cannot run a third). Scorers: scripts/eval_probes.py and scripts/rescore.py take `--segment-map SPEC` (default `trained`) and add the spec to the name of the file
they write when it is not `trained`, so a second scoring of the same checkpoint is never taken for the first.
"""

from __future__ import annotations

import argparse

SEGMENT_KEYS = ("reasoning", "call_head", "content", "call_body", "other")


def parse_spec(spec: str) -> dict | None:
    """None for `trained`; else a full map over SEGMENT_KEYS."""
    if spec in (None, "", "trained"):
        return None
    if spec in ("all2", "all1"):
        n = int(spec[-1])
        return {k: n for k in SEGMENT_KEYS}
    out = {k: 2 for k in SEGMENT_KEYS}
    for item in filter(None, spec.split(",")):
        k, _, v = item.partition(":")
        if k not in out or not v.isdigit() or int(v) not in (1, 2, 3):
            raise ValueError(f"--segment-map {spec!r}: trained | all2 | all1 | segment:passes,... over {SEGMENT_KEYS} with passes 1, 2 or 3")
        out[k] = int(v)
    return out


def tag(spec: str) -> str:
    """The part of a file name that tells a scoring under another map from the trained one ('' for trained)."""
    return "" if spec in (None, "", "trained") else "-map-" + spec.replace(":", "").replace(",", "_")


def apply(model, spec: str) -> dict | None:
    """Give `model` another counts map (its weights and built passes are untouched); returns the map applied or None for `trained`.
    The tokenizer is the one the checkpoint was trained with (model.cfg.loop.segment_tokenizer)."""
    from ced.eval.probes import unwrap
    from ced.model.pass_schedule import schedule_of
    from ced.train.pass_positions import PassScheduler

    counts = parse_spec(spec)
    if counts is None:
        return None
    inner = unwrap(model)
    sched = getattr(inner, "schedule", None)
    if sched is None:
        raise ValueError("--segment-map: this checkpoint has no pass schedule (model.loop.segment_passes / unscored_passes); "
                         "a two-pass model without one runs every position at its loop count")
    built = inner.n_passes
    if max(counts.values()) > built:
        raise ValueError(f"--segment-map {spec!r}: a count of {max(counts.values())}, the model was built for {built} passes")
    loop = inner.cfg.loop
    if sched.counts is None:  # unscored_passes only: no tokenizer is read, and a map needs one
        raise ValueError("--segment-map: this checkpoint's schedule has no segment map (no tokenizer to read the segments with)")
    import copy

    probe = copy.copy(loop)
    probe.segment_passes = dict(counts)
    inner.pass_scheduler = PassScheduler(schedule_of(probe), inner.scheduler().table)
    return counts


def add_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--segment-map", default="trained",
                        help="score under this pass map: trained (default) | all2 | all1 | segment:passes,... (ced/eval/schedule_override.py)")
