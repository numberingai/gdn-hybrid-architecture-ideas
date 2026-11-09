"""The readout inside training: exact match at T = 1 against T = 2 on a fixed sample of dev items, along a run.

section 5.7, way A. Off by default (EvalConfig.readout_every 0): nothing here runs.

What it is. Every R optimizer steps (eval.readout_every), at the run's loop switch step and at its last step, the
sample is scored at one pass and at two, and one record goes to <run_dir>/metrics.jsonl:

    {"event": "readout", "step": s, "passes": [1, 2], "items": n, "seconds": ...,
     "exact": {"1": ..., "2": ...}, "logp": {"1": ..., "2": ...},      pooled over the rows
     "gap": {"mean": ..., "se": ...}, "gap_logp": {"mean": ..., "se": ...},   T = 2 minus T = 1, paired by item
     "rows": {"<set>/<task>": {"n", "exact", "logp", "gap", "gap_logp", "depths": {"<d>": {"n", "exact": [T1, T2],
              "logp": [T1, T2]}}}},
     "sample": "<hash of the items>", "rng_touched": {...}, "deterministic": [...], "cuda_memory": {...}}

A row's value is the mean over its depths of the depth's mean; the pooled value is the mean over the rows; a gap is
the mean of the per-item differences with the SE of that mean, combined over depths and rows by the scoring reports'
rule (ced/eval/loop_passes.py: paired_stats, macro, pooled). The pooled numbers are over ALL configured rows; a report
that pools some of them recomputes from "rows". A model without a loop is scored at T = 1 only: "passes" is [1] and
the gap keys are absent.

The sample (eval.readout_sets): a list of [eval directory, task, depths, items per depth]. For each entry the first
`items per depth` items of each listed depth of <eval directory>/<task>.dev.pt, in file order. No random draw: the rule
is the scorer's own `--max-per-cell`, so `scripts/eval_loop_passes.py --max-per-cell n` on a checkpoint scores the same
items in the same batches when an entry lists every depth of its file. Only the DEV split is read (SPLIT): the test
items stay with the final scoring. The files are read once, when the run starts; a missing file, a depth with too few
items or a tokenizer of another vocabulary stops the run there, before it trains.

How it is scored: by the scorer's own function, ced.eval.loop_passes.score_passes_indices, in units of UNIT_ITEMS
items (the scoring runs' unit), on a SCORING TWIN of the model. The twin is what scripts/eval_syn_reason.py's
load_model builds for a scoring run: the model of the run's config with the dense reference attention
(AttentionConfig.backend, an implementation choice of the same function), eager, in eval mode, never compiled. Its
parameters are not copies: they are the trainer's own parameter tensors (load_state_dict(assign=True): the same
storage, checked at every readout), so it always holds the weights of the step and costs no memory beyond its
activations. So the numbers are the scoring run's numbers on those items and weights (tests/test_loop_readout.py
compares them with the scorer on a model loaded from the step's checkpoint), and the training model is never called,
never put in eval mode and never traced:

- no compile and no recompile: the twin is eager and the trainer's compiled layers do not see the readout's shapes;
- training is the same bit for bit with the readout on or off (weights, optimizer state, RNG states, every step's
  loss): forward only under no_grad, no random draw (the RNG states are compared and put back all the same; building
  the twin draws initial weights that are then dropped, and the states are put back after it), nothing written to the
  model, no use of the data stream;
- the process-wide switch torch.use_deterministic_algorithms is turned OFF for the scoring and put back after it: the
  scoring path runs the pure-PyTorch chunked delta rule (no varlen boundaries are built for eval items), whose cumsum
  has no deterministic CUDA kernel, and the 60M configs train with the switch on (scripts/loop_grow_gpu.py does the same
  for its scoring part). Training steps run with the switch as the config set it.

Its time is not training time: it runs in fit() between two steps, outside train_step, so no step's "step_seconds"
holds it. A stop request (SIGTERM) is checked between units and abandons the readout, so the trainer saves at once;
the resumed run scores that step first. An error while scoring is logged, recorded as {"event": "readout_failed"} and
switches the readout off for the rest of the process: it never stops training. A step whose record is already in
metrics.jsonl is not scored again, so a resumed run that repeats steps writes no readout twice.

Memory (modelled; the GPU check of the build's note measures it). Between two steps the training graph is free: the
process holds the weights, the optimizer state and the last gradients, about 1.7 GiB at 60M (the 1.21 GiB checkpoint
plus 0.48 GiB of gradients), inside a pool the training step has reserved (17.71 GiB for the G form). A scoring forward
is 16,384 padded tokens under no_grad; its largest tensors are the float32 log-softmax of the answer positions, one
(P, 64,000) tensor per pass count and three more of that size inside the scorer's omega = 0 assertion (P is at most
768 for three-token answers in rows of 64: 0.2 GiB each), so a readout adds about 1.5 GiB at its peak and is served
from the free pool. If the reserved pool grew all the same, the cache is emptied after the readout, as the lite
long-context eval does, and the record says so ("cuda_memory").
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from pathlib import Path

import torch

SPLIT = "dev"  # the split the readout reads; never the test items
EVENT = "readout"  # the "event" of the records in metrics.jsonl
FAILED = "readout_failed"
PASSES = (1, 2)  # the pass counts of a looped model's readout; a model without a loop: (1,)
UNIT_ITEMS = 1024  # items a call of the scorer (scripts/eval_loop_passes.py --unit-items): the same batches as a scoring run
SCORING_BACKEND = "reference"  # the scoring runs' attention backend (--attn-backend reference)
TOKENIZER_VOCAB = {"t64k": 64000}  # an eval file's tokenizer name -> the vocabulary it needs (as the scoring scripts check)


class ReadoutStop(Exception):
    """A stop was requested while the readout ran: it is abandoned and nothing is recorded."""


def set_label(d: Path) -> str:
    """The name of an eval set, as scripts/eval_loop_passes.py names it: the directory's parent for a tokenizer
    directory (<data volume> -> syn-reason-deep-v1), else the directory's own name."""
    return d.parent.name if d.name.startswith("t") else d.name


def parse_sets(value) -> list[dict]:
    """eval.readout_sets, checked: [{"name", "dir", "task", "depths", "n"}] in the config's order. Refused by name: a
    value that is not a non-empty list of [eval directory, task, depths, items per depth], and two entries of one row."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"eval.readout_sets {value!r}: a non-empty list of [eval directory, task, depths, items per depth]")
    out, seen = [], set()
    for e in value:
        ok = (isinstance(e, (list, tuple)) and len(e) == 4 and isinstance(e[0], str) and e[0] and isinstance(e[1], str) and e[1]
              and isinstance(e[2], (list, tuple)) and len(e[2]) > 0 and all(type(d) is int for d in e[2])
              and len(set(e[2])) == len(e[2]) and type(e[3]) is int and e[3] > 0)
        if not ok:
            raise ValueError(f"eval.readout_sets entry {e!r}: [eval directory, task, depths (a list of different whole numbers), "
                             "items per depth (a whole number above 0)]")
        d = Path(e[0])
        name = f"{set_label(d)}/{e[1]}"
        if name in seen:
            raise ValueError(f"eval.readout_sets: two entries for the row {name}")
        seen.add(name)
        out.append({"name": name, "dir": d, "task": e[1], "depths": list(e[2]), "n": e[3]})
    return out


def check_readout_config(cfg) -> None:
    """The readout's keys and what it is built for (Trainer.__init__). Off (readout_every 0): only the key's type."""
    from ced.config import looped_control

    ev = cfg.eval
    every = ev.readout_every
    if type(every) is not int or every < 0:
        raise ValueError(f"eval.readout_every {every!r}: 0 (off) or a number of optimizer steps")
    if not every:
        return
    if cfg.model.arch != "control" or looped_control(cfg.model):
        raise ValueError("eval.readout_every is built for arch 'control' (the plain control and the loop in the published form, "
                         f"model.loop.recur_layers), not arch {cfg.model.arch!r}"
                         + (" with a gated second pass (model.loop.pass2_gate)" if cfg.model.arch == "control" else "")
                         + ": leave it 0")
    if cfg.model.nvfp4.mode != "none":
        raise ValueError("eval.readout_every is not built together with model.nvfp4 (the scoring twin would not carry the "
                         "quantisation switch): leave it 0")
    if not cfg.metrics_jsonl:
        raise ValueError("eval.readout_every writes its records to metrics.jsonl: set metrics_jsonl")
    bt = ev.readout_batch_tokens
    if type(bt) is not int or bt < 1:
        raise ValueError(f"eval.readout_batch_tokens {bt!r}: the padded tokens of a scoring forward, a whole number above 0")
    parse_sets(ev.readout_sets)


def readout_due(cfg, step: int, total_steps: int | None = None) -> bool:
    """Whether a readout belongs to `step` (the number of updates made): every readout_every steps, the loop switch
    step (the state a branch starts from, before its first two-pass update) and the last step. Never step 0."""
    every = cfg.eval.readout_every
    total = cfg.steps if total_steps is None else total_steps
    return bool(every) and step > 0 and (step % every == 0 or step == total or step == cfg.loop_switch_step)


def load_sample(sets: list[dict], vocab_size: int) -> list[dict]:
    """The sample's rows, read from the eval files: per entry {"name", "file", "depths", "n", "es" (the kept items as an
    EncodedSet, in file order), "depth" (numpy, per kept item), "fingerprints"}. ValueError by name: a missing file, a
    tokenizer of another vocabulary, a listed depth with fewer than n items."""
    import numpy as np

    from ced.eval.probes import EncodedSet

    rows = []
    for s in sets:
        f = s["dir"] / f"{s['task']}.{SPLIT}.pt"
        if not f.is_file():
            raise ValueError(f"eval.readout_sets: no file {f} (the {SPLIT} items of task {s['task']!r} in {s['dir']})")
        es = EncodedSet.load(f)
        expect = TOKENIZER_VOCAB.get(es.info.get("tokenizer"))
        if expect is not None and expect != vocab_size:
            raise ValueError(f"eval.readout_sets: {f} was built with tokenizer {es.info.get('tokenizer')} and the model's "
                             f"vocabulary is {vocab_size}")
        wanted, have, keep = set(s["depths"]), {}, []
        for i, it in enumerate(es.items):
            d = it["depth"]
            if d in wanted and have.get(d, 0) < s["n"]:
                have[d] = have.get(d, 0) + 1
                keep.append(i)
        short = {d: have.get(d, 0) for d in s["depths"] if have.get(d, 0) < s["n"]}
        if short:
            raise ValueError(f"eval.readout_sets: {f} has fewer than {s['n']} items at depth(s) {short} (depth: items found)")
        sub = es.subset(keep)
        rows.append({"name": s["name"], "file": str(f), "depths": list(s["depths"]), "n": s["n"], "es": sub,
                     "depth": np.asarray([it["depth"] for it in sub.items]),
                     "fingerprints": [sub.fingerprint(i) for i in range(len(sub))]})
    return rows


def sample_hash(rows: list[dict]) -> str:
    """A short hash of the sample: the rows' names and every item's tokens, in order."""
    h = hashlib.sha256()
    for r in rows:
        h.update(r["name"].encode())
        for fp in r["fingerprints"]:
            h.update(fp.encode())
    return h.hexdigest()[:16]


def padded_lengths(rows: list[dict]) -> dict:
    """{padded item length: items} over the sample (the scorer pads an item to a multiple of 64 tokens): what a readout
    costs is the sum of length x items, per pass count."""
    from ced.eval import syn_score as sc

    out: dict = {}
    for r in rows:
        for i in range(len(r["es"])):
            L = sc.padded_length(r["es"].length(i))
            out[L] = out.get(L, 0) + 1
    return {str(L): out[L] for L in sorted(out)}


def scoring_model_config(model_cfg):
    """The model config a scoring run scores with: a copy with every attention backend set to SCORING_BACKEND
    (scripts/eval_syn_reason.py set_attn_backend). An implementation choice of the same function."""
    cfg = copy.deepcopy(model_cfg)

    def walk(obj) -> None:
        for name in getattr(obj, "__dataclass_fields__", ()):
            v = getattr(obj, name)
            if name == "attention" and hasattr(v, "backend"):
                v.backend = SCORING_BACKEND
            walk(v)

    walk(cfg)
    return cfg


def scoring_twin(trainer) -> torch.nn.Module:
    """The scoring twin of the trainer's model (module docstring): the scoring config's model, eager, in eval mode,
    whose parameters and persistent buffers ARE the trainer's tensors (assign=True: no copy). build_model draws
    initial weights from the global RNG; they are dropped by the assignment and the RNG states are put back."""
    from ced.model.control import build_model

    saved = (torch.get_rng_state(), random.getstate())
    try:
        twin = build_model(scoring_model_config(trainer.cfg.model)).to(trainer.dtype)
    finally:
        torch.set_rng_state(saved[0])
        random.setstate(saved[1])
    twin.load_state_dict(trainer.model.state_dict(), assign=True)
    twin = twin.to(trainer.device).eval()  # the tensors the state dict does not hold; the assigned ones are there already
    twin.requires_grad_(False)  # the twin's own Parameter objects: the trainer's parameters are not touched
    check_shared(twin, trainer.model)
    return twin


def check_shared(twin: torch.nn.Module, model: torch.nn.Module) -> None:
    """The twin's parameters are the model's own storage (so the twin holds the weights of the step); RuntimeError
    naming the first that is not."""
    own = dict(twin.named_parameters())
    for name, p in model.named_parameters():
        q = own.get(name)
        if q is None or q.shape != p.shape or q.data_ptr() != p.data_ptr():
            raise RuntimeError(f"the scoring twin no longer shares the model's parameter {name}")
    if len(own) != sum(1 for _ in model.named_parameters()):
        raise RuntimeError("the scoring twin has parameters the model does not have")


def recorded_steps(run_dir) -> set:
    """The steps whose readout record is in <run_dir>/metrics.jsonl."""
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
    """The readout's fixed state, built at the first call and kept on the trainer: the sample (read from the files),
    the scoring twin, the pass counts and the steps already recorded. A wrong configuration raises here."""
    state = getattr(trainer, "_readout", None)
    if state is not None:
        return state
    from ced.model.control import has_loop

    cfg = trainer.cfg
    rows = load_sample(parse_sets(cfg.eval.readout_sets), cfg.model.vocab_size)
    looped = has_loop(cfg.model)
    state = {"rows": rows, "twin": scoring_twin(trainer), "passes": list(PASSES if looped else PASSES[:1]),
             "call_T_none": not looped, "sample": sample_hash(rows), "items": sum(len(r["es"]) for r in rows),
             "recorded": recorded_steps(trainer.run_dir)}
    trainer._readout = state
    return state


def describe(trainer) -> dict:
    """What the start record of metrics.jsonl says about the readout."""
    state, ev = readout_state(trainer), trainer.cfg.eval
    return {"every": ev.readout_every, "split": SPLIT, "passes": state["passes"], "items": state["items"], "sample": state["sample"],
            "batch_tokens": ev.readout_batch_tokens,
            "rows": {r["name"]: {"file": r["file"], "depths": r["depths"], "items_per_depth": r["n"]} for r in state["rows"]},
            "padded_lengths": padded_lengths(state["rows"])}


@torch.no_grad()
def score_sample(state: dict, device, autocast, batch_tokens: int, stop=None) -> dict:
    """{row name: {T: (exact (n,), logp (n,))}} (numpy, float64) for the sample on the twin: the scorer's own function,
    unit by unit, with the scoring runs' eager batching (fixed_rows off). stop: a callable checked before every unit."""
    import numpy as np

    from ced.eval import loop_passes as LP

    Ts = state["passes"]
    out = {}
    for row in state["rows"]:
        es, n = row["es"], len(row["es"])
        plain = {T: np.zeros((n, 2)) for T in Ts}
        for lo in range(0, n, UNIT_ITEMS):
            if stop is not None and stop():
                raise ReadoutStop()
            idx = list(range(lo, min(lo + UNIT_ITEMS, n)))
            r = LP.score_passes_indices(state["twin"], es, idx, Ts, [], device, autocast, batch_tokens, fixed_rows=False,
                                        call_T_none=state["call_T_none"])
            for T in Ts:
                for i in idx:
                    plain[T][i] = r["plain"][T][i][:2]
        out[row["name"]] = {T: (plain[T][:, 0], plain[T][:, 1]) for T in Ts}
    return out


def _num(x: float, digits: int = 6):
    """A float for the record: rounded; None for a NaN (the SE of a single item)."""
    return None if x != x else round(float(x), digits)


def summarize(state: dict, scored: dict) -> dict:
    """The record's numbers from the per-item scores (module docstring): pooled, per row and per depth."""
    from ced.eval import loop_passes as LP

    Ts = state["passes"]
    lo, hi = Ts[0], Ts[-1]
    rows, row_vals = {}, {"exact": {T: [] for T in Ts}, "logp": {T: [] for T in Ts}, "gap": [], "gap_logp": []}
    for row in state["rows"]:
        depth, sc = row["depth"], scored[row["name"]]
        cells = {}
        for d in row["depths"]:
            m = depth == d
            cell = {"n": int(m.sum()), "exact": {T: float(sc[T][0][m].mean()) for T in Ts}, "logp": {T: float(sc[T][1][m].mean()) for T in Ts}}
            if hi != lo:
                cell["gap"] = LP.paired_stats((sc[hi][0] - sc[lo][0])[m])
                cell["gap_logp"] = LP.paired_stats((sc[hi][1] - sc[lo][1])[m])
            cells[d] = cell
        rec = {"n": int(len(depth))}
        for key in ("exact", "logp"):
            rec[key] = {}
            for T in Ts:
                v = sum(c[key][T] for c in cells.values()) / len(cells)
                rec[key][str(T)] = _num(v)
                row_vals[key][T].append(v)
        if hi != lo:
            for key in ("gap", "gap_logp"):
                v, se = LP.macro(cells, lambda c, key=key: (c[key]["mean"], c[key]["se"]))
                rec[key] = {"mean": _num(v), "se": _num(se)}
                row_vals[key].append((v, se))
        rec["depths"] = {str(d): {"n": c["n"], "exact": [_num(c["exact"][T]) for T in Ts], "logp": [_num(c["logp"][T], 5) for T in Ts]}
                         for d, c in cells.items()}
        rows[row["name"]] = rec
    n_rows = len(rows)
    out = {"passes": list(Ts), "items": state["items"],
           "exact": {str(T): _num(sum(row_vals["exact"][T]) / n_rows) for T in Ts},
           "logp": {str(T): _num(sum(row_vals["logp"][T]) / n_rows) for T in Ts}}
    if hi != lo:
        for key in ("gap", "gap_logp"):
            v, se = LP.pooled(row_vals[key])
            out[key] = {"mean": _num(v), "se": _num(se)}
    out["rows"] = rows
    return out


def run_readout(trainer, log=print, abortable: bool = False) -> dict | None:
    """The readout of the trainer's current weights, when it is on, due at trainer.step (readout_due) and not recorded
    yet. abortable: a stop request abandons it (the in-loop calls, which have unsaved steps behind them). Returns the
    record that was written, or None (off, not due, already there, abandoned or failed)."""
    cfg = trainer.cfg
    if not cfg.eval.readout_every or getattr(trainer, "_readout_failed", False):
        return None
    state = readout_state(trainer)  # the first call reads the files and builds the twin: a wrong sample stops the run at its start
    step = trainer.step
    if not readout_due(cfg, step, trainer.total_steps) or step in state["recorded"]:
        return None
    from ced.train import periodic_eval as pe

    device = trainer.device
    autocast = trainer.autocast_dtype if device.type == "cuda" else None  # the scoring scripts' rule: autocast on CUDA only
    was_training = trainer.model.training
    deterministic = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    saved = pe._rng_states(device)
    memory = pe._cuda_memory(device)
    scored = error = None
    touched: dict = {}
    t0 = time.perf_counter()
    try:
        torch.use_deterministic_algorithms(False)  # the scoring path's cumsum has no deterministic CUDA kernel (module docstring)
        check_shared(state["twin"], trainer.model)
        scored = score_sample(state, device, autocast, cfg.eval.readout_batch_tokens,
                              stop=(lambda: trainer._stop_requested) if abortable else None)
    except ReadoutStop:
        log(f"readout at step {step}: stop requested, abandoned (the resumed run scores this step first)")
    except Exception as e:  # noqa: BLE001 - a diagnostic must never stop training
        error = f"{type(e).__name__}: {e}"[:500]
    finally:
        torch.use_deterministic_algorithms(deterministic[0], warn_only=deterministic[1])
        touched = pe._rng_restore(saved, device)
        after = pe._cuda_memory(device)
        # CUDA: a readout (or its failure) that made the reserved pool grow gives the cache back (module docstring, "Memory")
        released = memory is not None and (error is not None or after["reserved_gib"] > memory["reserved_gib"])
        if released:
            torch.cuda.empty_cache()
    assert trainer.model.training == was_training, "the readout never changes the training model's mode"
    if error is not None:
        trainer._readout_failed = True
        log(f"readout at step {step} FAILED ({error}); switched off for the rest of this process, training continues")
        trainer._write_jsonl("metrics.jsonl", {"event": FAILED, "step": step, "error": error})
        return None
    if scored is None:
        return None
    record = {"event": EVENT, "step": step, **summarize(state, scored), "seconds": round(time.perf_counter() - t0, 3),
              "sample": state["sample"], "rng_touched": touched, "deterministic": list(deterministic)}
    if memory is not None:  # CUDA: what the readout did to the process's memory
        record["cuda_memory"] = {"before": memory, "after": after, "cache_released": released,
                                 "peak_alloc_raised_gib": round(after["peak_alloc_gib"] - memory["peak_alloc_gib"], 3),
                                 "peak_reserved_raised_gib": round(after["peak_reserved_gib"] - memory["peak_reserved_gib"], 3)}
    trainer._write_jsonl("metrics.jsonl", record)
    state["recorded"].add(step)
    flat = {f"readout/exact_T{T}": record["exact"][str(T)] for T in record["passes"]}
    if "gap" in record:
        flat["readout/gap"] = record["gap"]["mean"]
    trainer._wandb_call("log", flat, step=step, log=log)
    gap = record.get("gap")
    log(f"readout step {step}: " + "  ".join(f"T={T} {record['exact'][str(T)]:.4f}" for T in record["passes"])
        + (f"  gap {gap['mean']:+.4f}" + (f" (SE {gap['se']:.4f})" if gap["se"] is not None else "") if gap else "")
        + f"  {record['items']} items  {record['seconds']:.1f}s")
    return record
