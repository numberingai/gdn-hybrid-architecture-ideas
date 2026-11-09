"""The bpb readout inside training: bits per byte at T = 1 (and at T = 2 for a looped model) on a fixed held-out sample of
eval rows, along a run.

What it is for. The 200M programme ("How the branches are scored") reads two curves against tokens since the
switch-on: the gap, T = 2 minus the branch's own T = 1, and the gain, T = 2 minus base seed 1's T = 1 at the same
step, both in bpb and paired over the same rows. Off by default (EvalConfig.bpb_readout_every 0): nothing here runs.

When. At every step s >= eval.bpb_readout_from with (s - from) % eval.bpb_readout_every == 0, and at the run's last step;
never at step 0. A due step is scored once: a step whose record is in metrics.jsonl already is not scored again (a
resume), and a due step without one is scored when fit() starts (a stage C or a loop branch at its first step, before
any update of its own; a resumed run whose readout a stop request abandoned).

The sample. eval.bpb_readout_sets {name: path of an EvalRows file} (the run's own eval sets: the four headline sets web,
code, agent and general_web), eval.bpb_readout_rows rows in all (256), split equally over the sets (64 each), chosen
once by a fixed seed: the rows of a set are the first k of a permutation drawn from a generator seeded by a hash of
(eval.bpb_readout_seed, the set's name, the set's row count), in ascending order. A function of those alone: every arm
that reads the same files scores the same rows, and the record names each subset's fingerprint
(ced.eval.span_bpb.EvalRows.fingerprint), so a report can check that two arms' rows are the same before it pairs them.
The files are read once, when the run starts; a missing file, another tokenizer's set or a set with too few rows stops
the run there, before it trains.

How it is scored: by the reference scorer's own function, ced.eval.span_bpb.evaluate, on the TRAINING model, with the
arguments the run's final eval and scripts/rescore.py give it (chunked_head, the run's autocast, the fused GDN varlen
path when the run uses it, eval.batch_size rows a forward). So under compile "all" the readout runs the trainer's
compiled layers, as rescore.py runs a model rebuilt from the run's config; it never runs an eager scoring twin. A looped
model (has_loop: the loop in the published form, model.loop.recur_layers) is scored at T = 1 and T = 2 through the
scoring runs' own switch, CED_LOOP_T (ced.model.control.ENV_LOOP_T), which evaluate's control path reads and which is set
for the call and restored after it; the training forward always passes its T explicitly and never reads it. The numbers
are those evaluate gives a model loaded from the step's checkpoint (tests/test_stage_continue.py, bit for bit on the CPU).

Training is the same bit for bit with it on or off (losses, gradient norms, weights, optimizer state): evaluate runs
under no_grad, with the model in eval mode and back in its mode after (activation checkpointing acts on training
forwards only), draws no random number (the RNG states are compared and put back all the same: "rng_touched"), reads no
training data (a batch is a function of the step) and writes nothing to the model. It runs in fit() between two steps,
outside train_step, so no step's "step_seconds" holds its time. A stop request (SIGTERM) is checked between sets and
abandons it, so the trainer saves at once; the resumed run scores that step first. An error while scoring is logged,
recorded as {"event": "bpb_readout_failed"} and switches the readout off for the rest of the process: it never stops
training.

The record, one line of <run_dir>/metrics.jsonl:

    {"event": "bpb_readout", "step": s, "passes": [1] or [1, 2], "rows": 256, "sample": "<hash>",
     "bpb": {"1": {"mean": .., "all_mean": .., "sets": {"web": {"dec": .., "all": ..}, ...}}, "2": {...}},
     "gap": {"dec": {"mean": .., "se": .., "se_clustered": .., "sets": {"web": {"diff", "se", "se_clustered"}, ...}},
             "all": {...}},                                           looped models only: T = 2 minus T = 1, paired
     "row_nats": {"1": {"web": {"dec": [per row], "ctx": [per row]}, ...}, "2": {...}},
     "row_bytes": {"web": {"dec": [per row], "ctx": [per row]}, ...},
     "seconds": .., "rng_touched": {...}, "cuda_memory": {...}}

"dec" is the scorer's dec_bpb, the number the run's final four-set bpb is the mean of (tools/analysis/auto/runinfo.py
load_rescore); "all" is every scored target. "mean" is the mean over the sets of each set's bpb, the headline's rule.
The per-row nats and bytes (float64 nats, integer bytes, rows in the order of the set's chosen indices) are what a
report pairs: T = 2 against the run's own T = 1 (the gap, given here), and against base seed 1's T = 1 at the same step
(the gain, across two runs' records). A gap's SE: paired_difference's per set, rows independent and clustered by
document; the mean's SE is the root of the sum of the sets' squared SEs over the number of sets.

Memory. A forward is eval.batch_size rows of the run's length under no_grad, served from the pool the training step has
reserved; after a readout that made the reserved pool grow the cache is emptied (as the lite long-context eval and the
synthetic readout do), and the record says what the readout did to the process's memory ("cuda_memory").
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import time
from pathlib import Path

import torch

EVENT = "bpb_readout"  # the "event" of the records in metrics.jsonl
FAILED = "bpb_readout_failed"
PASSES = (1, 2)  # a looped model's pass counts; any other model: (1,)
METRICS = ("dec", "all")


class ReadoutStop(Exception):
    """A stop was requested while the readout ran: it is abandoned and nothing is recorded."""


def check_config(cfg, total_steps: int) -> None:
    """The readout's keys and what it is built for (Trainer.__init__). Off (bpb_readout_every 0): only the key's type."""
    from ced.config import looped_control

    ev = cfg.eval
    every = ev.bpb_readout_every
    if type(every) is not int or every < 0:
        raise ValueError(f"eval.bpb_readout_every {every!r}: 0 (off) or a number of optimizer steps")
    if not every:
        return
    start = ev.bpb_readout_from
    if type(start) is not int or not 0 <= start <= total_steps:
        raise ValueError(f"eval.bpb_readout_from {start!r}: the first readout step, 0 .. the run's steps ({total_steps})")
    if cfg.model.arch != "control" or looped_control(cfg.model):
        raise ValueError("eval.bpb_readout_every is built for arch 'control' (the plain control, the depth mix, the loop in the "
                         f"published form), not arch {cfg.model.arch!r}"
                         + (" with a gated second pass (model.loop.pass2_gate)" if cfg.model.arch == "control" else "")
                         + ": leave it 0")
    if not cfg.metrics_jsonl:
        raise ValueError("eval.bpb_readout_every writes its records to metrics.jsonl: set metrics_jsonl")
    sets = ev.bpb_readout_sets
    if not isinstance(sets, dict) or not sets or not all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in sets.items()):
        raise ValueError(f"eval.bpb_readout_sets {sets!r}: a non-empty map of set name -> path of an EvalRows file")
    rows, seed = ev.bpb_readout_rows, ev.bpb_readout_seed
    if type(rows) is not int or rows < 1 or rows % len(sets):
        raise ValueError(f"eval.bpb_readout_rows {rows!r}: rows in all, a whole number above 0 that the {len(sets)} sets "
                         "share equally")
    if type(seed) is not int:
        raise ValueError(f"eval.bpb_readout_seed {seed!r}: an integer")


def readout_due(cfg, step: int, total_steps: int) -> bool:
    """Whether a readout belongs to `step` (the number of updates made): from bpb_readout_from on, every
    bpb_readout_every steps counted from it, and the run's last step. Never step 0."""
    ev = cfg.eval
    every, start = ev.bpb_readout_every, ev.bpb_readout_from
    if not every or step <= 0 or step < start:
        return False
    return (step - start) % every == 0 or step == total_steps


def choose_rows(n_rows: int, k: int, seed: int, name: str) -> list[int]:
    """The k rows of a set of n_rows rows the readout scores, ascending: the first k of a permutation drawn from a
    generator seeded by a hash of (seed, name, n_rows). ValueError when the set has fewer than k rows."""
    if k > n_rows:
        raise ValueError(f"eval.bpb_readout_sets: the set {name!r} has {n_rows} rows, fewer than the {k} the readout scores")
    digest = hashlib.blake2b(f"bpb-readout:{seed}:{name}:{n_rows}".encode(), digest_size=8).digest()
    g = torch.Generator()
    g.manual_seed(int.from_bytes(digest, "little") & (2**63 - 1))
    return sorted(torch.randperm(n_rows, generator=g)[:k].tolist())


def subset(rows, indices: list[int]):
    """The EvalRows of the given row indices, in that order (doc_group and special_ids kept: they index documents and
    the vocabulary, not rows; target_class is cut with the rows)."""
    from ced.eval.span_bpb import EvalRows

    idx = torch.tensor(indices, dtype=torch.long)
    return rows._with({f: getattr(rows, f).index_select(0, idx) for f in EvalRows._FIELDS},
                      None if rows.target_class is None else rows.target_class.index_select(0, idx))


def load_sample(cfg) -> list[dict]:
    """[{"name", "path", "rows" (the chosen EvalRows), "indices", "set_rows", "fingerprint", "tokenizer"}] in the
    config's order. ValueError by name: a missing file, a multiple-choice file, another tokenizer's set, a token outside
    the vocabulary, a set with too few rows."""
    from ced.data.tokenizers import check_tokenizer, eval_set_tokenizer
    from ced.eval.span_bpb import EvalRows

    ev, vocab = cfg.eval, cfg.model.vocab_size
    k = ev.bpb_readout_rows // len(ev.bpb_readout_sets)
    out = []
    for name, path in ev.bpb_readout_sets.items():
        if not Path(path).is_file():
            raise ValueError(f"eval.bpb_readout_sets: no file {path} (set {name!r})")
        note = check_tokenizer(eval_set_tokenizer(path), vocab, f"bpb readout set {name} ({path})")
        data = torch.load(path, weights_only=False)
        if not (isinstance(data, dict) and all(f in data for f in EvalRows._FIELDS)):
            raise ValueError(f"eval.bpb_readout_sets: {path} (set {name!r}) is not a span bpb set (ced.eval.span_bpb.EvalRows; "
                             "a multiple-choice file has no rows to score)")
        # as EvalRows.load, from the dict already read (one read of the file)
        rows = EvalRows(**{f: data[f] for f in EvalRows._FIELDS}, capacity=int(data["capacity"]), kind=data["kind"],
                        **{key: data.get(key) for key in EvalRows._OPTIONAL})
        assert rows.fingerprint() == data["fingerprint"], f"{path}: contents do not match the saved fingerprint"
        del data
        n_rows = len(rows)
        idx = choose_rows(n_rows, k, ev.bpb_readout_seed, name)
        sub = subset(rows, idx)
        del rows
        top = max(int(sub.tokens.max()), int(sub.targets.max()))
        if top >= vocab:
            raise ValueError(f"eval.bpb_readout_sets: {path} (set {name!r}) holds token id {top}, outside the vocabulary of {vocab}")
        out.append({"name": name, "path": str(path), "rows": sub, "indices": idx, "set_rows": n_rows,
                    "fingerprint": sub.fingerprint(), "tokenizer": note})
    return out


def sample_hash(sample: list[dict]) -> str:
    """A short hash of the sample: the sets' names and their subsets' fingerprints, in order."""
    h = hashlib.sha256()
    for s in sample:
        h.update(s["name"].encode())
        h.update(s["fingerprint"].encode())
    return h.hexdigest()[:16]


def recorded_steps(run_dir) -> set:
    """The steps whose bpb readout record is in <run_dir>/metrics.jsonl."""
    import json

    path = Path(run_dir) / "metrics.jsonl"
    steps = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                if f'"event": "{EVENT}"' in line:
                    try:
                        steps.add(int(json.loads(line)["step"]))
                    except (ValueError, KeyError, TypeError):
                        continue
    return steps


def readout_state(trainer) -> dict:
    """The readout's fixed state, built at the first call and kept on the trainer: the sample, the pass counts and the
    steps already recorded. A wrong set raises here."""
    state = getattr(trainer, "_bpb_readout", None)
    if state is not None:
        return state
    from ced.model.control import has_loop

    sample = load_sample(trainer.cfg)
    state = {"sample": sample, "passes": list(PASSES if has_loop(trainer.cfg.model) else PASSES[:1]),
             "hash": sample_hash(sample), "rows": sum(len(s["rows"]) for s in sample),
             "recorded": recorded_steps(trainer.run_dir)}
    trainer._bpb_readout = state
    return state


def describe(trainer) -> dict:
    """What the start record of metrics.jsonl says about the readout."""
    state, ev = readout_state(trainer), trainer.cfg.eval
    return {"every": ev.bpb_readout_every, "from": ev.bpb_readout_from, "passes": state["passes"], "rows": state["rows"],
            "seed": ev.bpb_readout_seed, "sample": state["hash"], "batch_size": ev.batch_size,
            "sets": {s["name"]: {"path": s["path"], "rows": len(s["rows"]), "set_rows": s["set_rows"],
                                 "fingerprint": s["fingerprint"], "indices": s["indices"], "tokenizer": s["tokenizer"]}
                     for s in state["sample"]}}


@contextlib.contextmanager
def pass_count(T: int | None):
    """CED_LOOP_T = T for the call (the scoring runs' switch, read by a looped control's forward when it is given no T);
    the variable as it was afterwards. None: nothing is set."""
    from ced.model.control import ENV_LOOP_T

    if T is None:
        yield
        return
    before = os.environ.get(ENV_LOOP_T)
    os.environ[ENV_LOOP_T] = str(T)
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(ENV_LOOP_T, None)
        else:
            os.environ[ENV_LOOP_T] = before


def score(trainer, state: dict, stop=None) -> dict:
    """{T: {set name: SpanEval}} of the trainer's model on the sample: ced.eval.span_bpb.evaluate with the arguments of
    the run's final eval (Trainer.run_evals). stop: a callable checked before every (T, set)."""
    from ced.eval.span_bpb import evaluate

    looped = len(state["passes"]) > 1
    out = {}
    for T in state["passes"]:
        out[T] = {}
        with pass_count(T if looped else None):
            for s in state["sample"]:
                if stop is not None and stop():
                    raise ReadoutStop()
                out[T][s["name"]] = evaluate(trainer.model, s["rows"], batch_size=trainer.cfg.eval.batch_size,
                                             chunked_head=True, autocast_dtype=trainer.autocast_dtype,
                                             fused_varlen=trainer.fused_varlen)
                if looped:
                    assert out[T][s["name"]].T == T, (out[T][s["name"]].T, T)
    return out


def _num(x: float):
    """A float for the record: as it is (full precision); None for a NaN (the SE of fewer than two rows)."""
    return None if x != x else float(x)


def summarize(state: dict, scored: dict) -> dict:
    """The record's numbers from the per-set evaluations (module docstring)."""
    from ced.eval.span_bpb import paired_difference

    Ts = state["passes"]
    names = [s["name"] for s in state["sample"]]
    bpb_rec, row_nats = {}, {}
    for T in Ts:
        sets = {n: {"dec": scored[T][n].dec_bpb, "all": scored[T][n].all_bpb} for n in names}
        bpb_rec[str(T)] = {"mean": sum(v["dec"] for v in sets.values()) / len(sets),
                           "all_mean": sum(v["all"] for v in sets.values()) / len(sets), "sets": sets}
        row_nats[str(T)] = {n: {"dec": scored[T][n].row_dec_nats.tolist(), "ctx": scored[T][n].row_ctx_nats.tolist()} for n in names}
    first = scored[Ts[0]]
    row_bytes = {n: {"dec": first[n].row_dec_bytes.tolist(), "ctx": first[n].row_ctx_bytes.tolist()} for n in names}
    out = {"passes": list(Ts), "rows": state["rows"], "sample": state["hash"], "bpb": bpb_rec}
    if len(Ts) > 1:
        lo, hi = Ts[0], Ts[-1]
        out["gap"] = {}
        for metric in METRICS:
            per = {}
            for n in names:
                d = paired_difference(scored[hi][n], scored[lo][n], metric=metric)
                per[n] = {"diff": _num(d.diff), "se": _num(d.se), "se_clustered": _num(d.se_clustered)}
            k = len(per)
            mean = None if any(v["diff"] is None for v in per.values()) else sum(v["diff"] for v in per.values()) / k
            se = lambda key: (None if any(v[key] is None for v in per.values())  # noqa: E731
                              else math.sqrt(sum(v[key] ** 2 for v in per.values())) / k)
            out["gap"][metric] = {"mean": mean, "se": se("se"), "se_clustered": se("se_clustered"), "sets": per}
    out["row_nats"] = row_nats
    out["row_bytes"] = row_bytes
    return out


def run(trainer, log=print, abortable: bool = False) -> dict | None:
    """The readout of the trainer's current weights, when it is on, due at trainer.step (readout_due) and not recorded
    yet. abortable: a stop request abandons it (the in-loop calls, which have unsaved steps behind them). Returns the
    record that was written, or None (off, not due, already there, abandoned or failed)."""
    cfg = trainer.cfg
    if not cfg.eval.bpb_readout_every or getattr(trainer, "_bpb_readout_failed", False):
        return None
    state = readout_state(trainer)  # the first call reads the sets: a wrong one stops the run at its start
    step = trainer.step
    if not readout_due(cfg, step, trainer.total_steps) or step in state["recorded"]:
        return None
    from ced.train import periodic_eval as pe

    device = trainer.device
    was_training = trainer.model.training
    saved = pe._rng_states(device)
    memory = pe._cuda_memory(device)
    scored = summary = error = None
    after, released = None, False
    t0 = time.perf_counter()
    try:
        scored = score(trainer, state, stop=(lambda: trainer._stop_requested) if abortable else None)
        # inside the try: an error while summarising (e.g. a non-finite row) is a failed readout too, never a stop
        summary = summarize(state, scored)
    except ReadoutStop:
        log(f"bpb readout at step {step}: stop requested, abandoned (the resumed run scores this step first)")
    except Exception as e:  # noqa: BLE001 - a diagnostic must never stop training
        error = f"{type(e).__name__}: {e}"[:500]
    finally:
        touched = pe._rng_restore(saved, device)
        after = pe._cuda_memory(device)
        # CUDA: a readout (or its failure) that made the reserved pool grow gives the cache back (module docstring)
        released = memory is not None and (error is not None or after["reserved_gib"] > memory["reserved_gib"])
        if released:
            torch.cuda.empty_cache()
    assert trainer.model.training == was_training, "the readout never changes the training model's mode"
    if error is not None:
        trainer._bpb_readout_failed = True
        log(f"bpb readout at step {step} FAILED ({error}); switched off for the rest of this process, training continues")
        trainer._write_jsonl("metrics.jsonl", {"event": FAILED, "step": step, "error": error})
        return None
    if summary is None:
        return None
    record = {"event": EVENT, "step": step, **summary, "seconds": round(time.perf_counter() - t0, 3),
              "rng_touched": touched}
    if memory is not None:  # CUDA: what the readout did to the process's memory
        record["cuda_memory"] = {"before": memory, "after": after, "cache_released": released,
                                 "peak_alloc_raised_gib": round(after["peak_alloc_gib"] - memory["peak_alloc_gib"], 3),
                                 "peak_reserved_raised_gib": round(after["peak_reserved_gib"] - memory["peak_reserved_gib"], 3)}
    trainer._write_jsonl("metrics.jsonl", record)
    state["recorded"].add(step)
    flat = {f"bpb_readout/T{T}": record["bpb"][str(T)]["mean"] for T in record["passes"]}
    if "gap" in record:
        flat["bpb_readout/gap"] = record["gap"]["dec"]["mean"]
    trainer._wandb_call("log", flat, step=step, log=log)
    gap = record.get("gap", {}).get("dec")
    fmt = lambda x, spec: "n/a" if x is None else format(x, spec)  # noqa: E731 - a gap of a non-finite row is None
    log(f"bpb readout step {step}: " + "  ".join(f"T={T} {record['bpb'][str(T)]['mean']:.4f}" for T in record["passes"])
        + (f"  gap {fmt(gap['mean'], '+.4f')}" + (f" (SE {gap['se']:.4f})" if gap["se"] is not None else "") if gap else "")
        + f"  {record['rows']} rows")
    return record
