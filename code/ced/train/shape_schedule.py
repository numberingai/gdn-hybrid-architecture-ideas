"""A batch-shape schedule: the rows per optimizer step and the row length of a run, by phases of its tokens.

TrainConfig.shape_schedule (batch warmup and short-to-long rows). One mechanism for both screens: a list of the
phases BEFORE the config's own shape, each [until, rows, seq_len]; the last phase is always the config's own shape.
Off (the default, []): nothing here runs and the trainer is the trainer of before, bit for bit.

Units. One REFERENCE step is the config's own step: B = data.batch_size rows of L = data.seq_len tokens. cfg.steps is
the run's length in reference steps, so a run reads cfg.steps x B x L tokens with or without a schedule (equal tokens).

A phase p = [until_p, rows_p, len_p] ends at the reference step

    end_p = floor(until_p x cfg.steps + 0.5)       round half up: a phase boundary is a whole reference step

and starts where the phase before it ends (the first at 0). After the last listed phase, B rows of L run to
cfg.steps. In phase p one optimizer step reads rows_p rows of len_p tokens:

    per_ref_p = (B x L) / (rows_p x len_p)         optimizer steps per reference step; must be a whole number
    steps_p   = (end_p - start_p) x per_ref_p      optimizer steps of the phase
    micro_p   = micro x L / len_p                  rows per forward/backward: the config's TOKENS a micro-batch,
                                                   so the memory of a micro-batch stays the config's (rows_p is a
                                                   multiple of it)

The data. The unscheduled run reads the packed token stream (ced/train/shards.py) in rows of L: its step s reads
rows s x B .. s x B + B - 1, row r being the stream's tokens [r x L, r x L + L] (L inputs and the next token of the
last). A scheduled run reads the same stream in the same order. Optimizer step k, the j-th step of phase p, reads

    first_p + j x rows_p, ..., first_p + j x rows_p + rows_p - 1      rows of len_p,   first_p = start_p x B x L / len_p

row i of length len_p being the stream's tokens [i x len_p, i x len_p + len_p]. With len_p = L these are the
unscheduled run's rows, fewer to a step. With len_p < L each of its rows is cut into L / len_p consecutive pieces
(ShardPacker.rows says what a cut does to a document).

Tokens seen. Everything that must sit where the unscheduled run has it IN TOKENS is read at the clock

    clock(k) = start_p + j / per_ref_p             reference steps of tokens read before optimizer step k

an integer wherever per_ref_p = 1. The LR of step k is the config's schedule at clock(k) (ced.train.trainer.lr_schedule
takes a fraction): the warmup and the decay are where the unscheduled run's are in tokens, and with per_ref_p = 1 the
LR of every step is the unscheduled run's bit for bit. The periodic evals (eval.every_steps) fire when the tokens read
are a multiple of every_steps reference steps. What counts OPTIMIZER steps and is left counting them: the optimizer's
own state (AdamW's bias correction, the momenta), weight decay (applied per update), log_every, ckpt_every_steps,
checkpoint_steps and the names of the checkpoint files.

The LR by rows (TrainConfig.shape_lr). With shape_lr "clock" (the default) the paragraph above is the whole rule: in
a phase with per_ref_p optimizer steps to a reference step the run makes per_ref_p updates at the LR the unscheduled
run uses once, so the LR summed over the run is larger than the unscheduled run's.

    ref(k)  = start_p + floor(j / per_ref_p)                       the reference step whose tokens the step reads
    lr(k)   = schedule(ref(k)) / per_ref_p                         = schedule(ref(k)) x rows_p x len_p / (B x L)

the unscheduled run's LR of that reference step times the share of its tokens the step reads (the linear scaling rule).
The per_ref_p steps of a reference step share its LR equally, so their sum is the unscheduled run's LR of that step and
the LR summed over any phase, and over the run, is the unscheduled run's (to float rounding; exactly when per_ref_p is a
power of two and the sum is taken exactly). Nothing of the schedule moves: the warmup and the decay sit where they are in
tokens. Inside a reference step the LR is constant ("clock" reads the warmup at the fraction instead: a ramp of
per_ref_p values where "rows" has one; scaling that ramp would leave the sum above the unscheduled run's, by 0.23%
in the first phase of bw10). With per_ref_p = 1 the LR is "clock"'s, bit for bit. Every parameter group takes the scaled
LR times its own lr_scale, as it takes the unscaled one (the trainer sets group["lr"] = lr x lr_scale), and weight
decay, applied per update as (1 - lr x weight_decay), is scaled with it: the decay summed over a reference step is the
unscheduled run's to first order (tests/test_shape_schedule.py prints the sums). What "rows" does not change: the number
of updates (the momenta and AdamW's bias correction still count optimizer steps) and Muon's fixed-size update, which is
now per_ref_p updates of 1 / per_ref_p of the size.

`reference_walk` is the slow, readable definition (it walks the steps and counts tokens); `ShapeSchedule` is the
closed form the trainer uses, and tests/test_shape_schedule.py holds it to the walk.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass


# TrainConfig.shape_lr (module docstring, "The LR by rows"): the LR at tokens seen, or scaled with the step's tokens.
SHAPE_LR_MODES = ("clock", "rows")


@dataclass(frozen=True)
class Phase:
    index: int
    rows: int        # rows per optimizer step
    seq_len: int     # row length
    micro: int       # rows per forward/backward (the config's tokens a micro-batch)
    ref_start: int   # the phase's first reference step
    ref_end: int     # one past its last
    per_ref: int     # optimizer steps per reference step
    first_step: int  # the optimizer step index of its first step
    steps: int       # its optimizer steps
    first_row: int   # the index of its first row, in rows of seq_len


@dataclass(frozen=True)
class StepShape:
    """What optimizer step k reads (ShapeSchedule.at)."""

    phase: int
    rows: int
    seq_len: int
    micro: int
    first_row: int        # in rows of seq_len: the step reads first_row .. first_row + rows - 1
    clock: int | float    # reference steps of tokens read before this step (an int at a whole reference step)
    tokens_before: int    # tokens read before this step


def boundary(until: float, steps: int) -> int:
    """The reference step at which a phase with share `until` ends: round half up."""
    return int(math.floor(until * steps + 0.5))


def _entries(schedule) -> list[tuple[float, int, int]]:
    if not isinstance(schedule, (list, tuple)):
        raise ValueError(f"shape_schedule {schedule!r}: a list of [until, rows, seq_len]")
    out = []
    for e in schedule:
        if not isinstance(e, (list, tuple)) or len(e) != 3:
            raise ValueError(f"shape_schedule entry {e!r}: [until, rows, seq_len]")
        until, rows, seq_len = e
        if isinstance(until, bool) or not isinstance(until, (int, float)) or not 0 < until < 1:
            raise ValueError(f"shape_schedule entry {e!r}: until is the share of the run's tokens at which the phase ends, "
                             "0 < until < 1")
        for name, v in (("rows", rows), ("seq_len", seq_len)):
            if type(v) is not int or v < 1:
                raise ValueError(f"shape_schedule entry {e!r}: {name} is a whole number >= 1")
        out.append((float(until), rows, seq_len))
    return out


def reference_walk(schedule, steps: int, batch_size: int, seq_len: int, micro: int | None = None):
    """The definition, slowly: walk the optimizer steps of a run of `steps` reference steps and yield, per step,
    (phase index, rows, row length, micro-batch rows, tokens read before the step). The tokens read before a step are
    also the stream position of its first row. The phase of a step is the first one whose end (in tokens) lies beyond
    the tokens already read; the config's own shape after the last. No validation: ShapeSchedule validates."""
    micro = micro or batch_size
    per_ref = batch_size * seq_len
    ends = [boundary(until, steps) * per_ref for until, _, _ in schedule]
    shapes = [(rows, length) for _, rows, length in schedule] + [(batch_size, seq_len)]
    total, read = steps * per_ref, 0
    while read < total:
        p = next((i for i, end in enumerate(ends) if read < end), len(ends))
        rows, length = shapes[p]
        yield p, rows, length, micro * seq_len // length, read
        read += rows * length


class ShapeSchedule:
    """The phases of a run in closed form (module docstring). phases: the listed ones, then the config's own shape."""

    def __init__(self, schedule, steps: int, batch_size: int, seq_len: int, micro: int | None = None):
        entries = _entries(schedule)
        if not entries:
            raise ValueError("shape_schedule is empty: no schedule (ShapeSchedule.from_config gives None)")
        micro = micro or batch_size
        B, L = batch_size, seq_len
        self.steps, self.batch_size, self.seq_len, self.micro = steps, B, L, micro
        self.tokens_per_ref = B * L
        phases, start, first_step = [], 0, 0
        for i, (until, rows, length) in enumerate([*entries, (1.0, B, L)]):
            last = i == len(entries)
            end = steps if last else boundary(until, steps)
            what = "the config's own shape" if last else f"phase {i} {[until, rows, length]}"
            if L % length:
                raise ValueError(f"shape_schedule {what}: the row length {length} does not divide data.seq_len {L} (a shorter "
                                 "row is a cut of the config's row into equal pieces)")
            if (B * L) % (rows * length):
                raise ValueError(f"shape_schedule {what}: {rows} rows of {length} tokens do not divide the config's step of "
                                 f"{B} x {L} tokens (phase boundaries are whole reference steps)")
            m = micro * L // length
            if rows % m:
                raise ValueError(f"shape_schedule {what}: {rows} rows are not a multiple of its micro-batch of {m} rows "
                                 f"(micro_batch_size {micro} x {L} / {length}: the config's tokens a micro-batch)")
            if end <= start:
                raise ValueError(f"shape_schedule {what} ends at reference step {end}, not after its start {start}: every "
                                 f"phase needs at least one reference step of the run's {steps} (until must increase, and the "
                                 "config's own shape runs last)")
            per_ref = (B * L) // (rows * length)
            n = (end - start) * per_ref
            phases.append(Phase(i, rows, length, m, start, end, per_ref, first_step, n, start * B * (L // length)))
            start, first_step = end, first_step + n
        self.phases: tuple[Phase, ...] = tuple(phases)
        self.total_steps = first_step
        self._first_steps = [p.first_step for p in self.phases]

    @classmethod
    def from_config(cls, cfg) -> "ShapeSchedule | None":
        """The schedule of a TrainConfig, validated; None without one."""
        if not cfg.shape_schedule:
            if cfg.shape_schedule not in ([], ()):
                raise ValueError(f"shape_schedule {cfg.shape_schedule!r}: a list of [until, rows, seq_len] ([]: none)")
            return None
        return cls(cfg.shape_schedule, cfg.steps, cfg.data.batch_size, cfg.data.seq_len, cfg.micro_batch_size)

    def phase_of(self, step: int) -> Phase:
        """The phase of optimizer step `step` (0-based); steps past the run's end stay in the last phase."""
        return self.phases[max(0, bisect.bisect_right(self._first_steps, step) - 1)]

    def at(self, step: int) -> StepShape:
        p = self.phase_of(step)
        j = step - p.first_step
        clock = p.ref_start + j // p.per_ref if j % p.per_ref == 0 else p.ref_start + j / p.per_ref
        return StepShape(p.index, p.rows, p.seq_len, p.micro, p.first_row + j * p.rows, clock,
                         p.ref_start * self.tokens_per_ref + j * p.rows * p.seq_len)

    def lr_point(self, step: int) -> tuple[int, int]:
        """Where optimizer step `step` reads the LR under shape_lr "rows" (module docstring): (the reference step whose
        tokens the step reads, the optimizer steps to a reference step in its phase). The LR is the config's schedule at
        the first, divided by the second."""
        p = self.phase_of(step)
        return p.ref_start + (step - p.first_step) // p.per_ref, p.per_ref

    def tokens_seen(self, step: int) -> int:
        """Tokens read by the first `step` optimizer steps."""
        return self.at(step).tokens_before

    def ref_done(self, step: int) -> int | None:
        """The reference steps of tokens read by the first `step` optimizer steps when that is a whole number, else
        None (inside a reference step: a phase with more than one optimizer step to a reference step)."""
        clock = self.at(step).clock
        return clock if isinstance(clock, int) else None

    def step_of_ref(self, ref: int) -> int:
        """The number of optimizer steps after which `ref` reference steps of tokens have been read (0 <= ref <= steps)."""
        if not 0 <= ref <= self.steps:
            raise ValueError(f"reference step {ref} outside 0..{self.steps}")
        p = next((p for p in self.phases if ref < p.ref_end), self.phases[-1])
        return p.first_step + (ref - p.ref_start) * p.per_ref

    def describe(self) -> list[dict]:
        """One record per phase, for the metrics' start event and the logs."""
        return [{"phase": p.index, "rows": p.rows, "seq_len": p.seq_len, "micro": p.micro, "ref_steps": [p.ref_start, p.ref_end],
                 "first_step": p.first_step, "steps": p.steps, "tokens": p.steps * p.rows * p.seq_len} for p in self.phases]
