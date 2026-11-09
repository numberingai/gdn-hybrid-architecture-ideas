"""Score a trainer checkpoint on the probe sets (ced/eval/probes.py).

    python scripts/eval_probes.py --checkpoint <run>/checkpoints/step_00006104.pt \\
        --probes <probes>/t64k --out probes.json [--compare other.json]

--checkpoint: a checkpoint file, a run directory or its checkpoints/ directory (`latest` is
followed). --probes: a directory of EncodedSet files (one per probe) for the tokenizer the model was
trained with. The model is rebuilt from the checkpoint's config: on CUDA in its training precision
(autocast), with the fused varlen GDN path when the config uses fla, and with torch.compile
failing loudly instead of falling back to eager past the recompile limit (an eager flex attention
at 64k would not fit). Decoder arches are scored by their decoder at the answer positions,
controls by their single head (see ced/eval/probes.py).

Output JSON: settings, the probe files' sha256, timing per row shape (length, single / packed;
the scoring-cost measurement), the summary per probe / cell / group with item-level standard errors, and every
item's result (for paired comparisons across checkpoints and seeds: --compare, or
ced.eval.probes.paired_difference). When the directory holds the multi-fact sets (mf_chain, mf_concat), also
"multifact": that task's own metrics per probe, k and length (ced/eval/probe_gen_multifact.py summarize: exact,
fact_acc, nats per fact token, first_nats, ...), the summary scripts/eval_long.py --parts multifact writes; the
items then carry their fact spans. When it holds the latent two-hop set (twohop; ced/eval/probe_gen_twohop.py), also "twohop": per setting (all, each design level, each cell) every variant's first-token
accuracy and nats, exact match and nats per answer token with item SEs, the chance level, and the differences paired
per base item with their SE: hop_cost = twohop - onehop (the same answer asked directly by its key), link_use =
twohop - shuffled (the link pointing nowhere: the ceiling of guessing) and link_follow = twohop - swapped (every link
pointing to another chain's bridge: the share of answers that follow the link). --T 1 on a looped checkpoint scores
its pass-1 readout and --T 2 its final one, as for every probe; with --compare the record also holds
"compare"."twohop": this scoring minus that one, paired per base item.

--length-scale (default off): score with every full-attention read's query
multiplied by max(1, ln(pos + 1) / ln(L)), L = --length-scale-ref (8,192): the encoder's global layers and CED v2's
decoder full layers (n_t = pos + 1, the memory count), local decoder layers unscaled; with --decoder-at answers that is
the restricted regime with the length fix (ced/eval/probes.py length_scaled_reads). Recorded in settings
(length_scale, length_scale_ref, length_scale_layers).

--decoder-at answers_sinks: the restricted regime plus the decoder at each item's first
decoder.n_sinks positions, so the local decoder layers keep their sinks as in training (and the full
layers see those own keys too); the sink positions are not scored. Capacity: the answers plus the sinks, rounded up to
a multiple of 128 as for answers. Recorded in settings (decoder_at, decoder_sink_positions).

--decoder-at answers_sinks_replay [--replay-window N, default 128]: answers_sinks
plus the decoder at the min(N, available) input positions of each item immediately before its answer span (a bounded
replay of the decoder's sliding-window state before generation: those positions write decoder K/V, so the first answer
token's local window holds own keys of the question as in training; not scored, never another item's positions).
Capacity: the answers plus the sinks plus the replay positions, rounded up to a multiple of 128. --length-scale applies
to every full-attention read as in the other regimes, those of the replay and sink slots included. Recorded in settings
(decoder_at, decoder_sink_positions, decoder_replay_window; the latter None in every other regime).
--replay-window turn / turn<N> (turn128 is the whole last turn, capped at 128 positions;
ced/eval/replay_window.py): each item replays its own last turn, capped at 512 / N positions. For a needle item
(uuid_needle, multikey) and a multi-fact item the turn is the question, from its first word to the answer, so the
replay is the min(cap, question tokens - 1) positions before the first answer position; the decoder then runs from the
question's first token on and at no haystack position but the sinks. The question's token count is the item's
question_tokens; sets that do not carry it (every needle set) get it from their text items and tokenizer (--probes-text,
default <probes>/../text; --probes-tokenizer, default the path in <probes>/manifest.json), each count checked against
the item's stored tokens (ced/eval/probe_turn.py). Other probes have no turn defined and are refused: name the sets
with --probe-names. Recorded: decoder_replay_window (the name), decoder_replay_turn_max, question_tokens_from; and in
every replay regime each item's "replay" (positions replayed) and "run" (positions at which the decoder ran).

--rope-clamp C (default off): score with the
rotary part of every full-attention read computed with min(relative distance, C): the encoder's global layers (control
(a)'s, CED v2's) and CED v2's decoder memory reads, in every decoder regime; nothing else changes
(AttentionConfig.rope_max_distance, set on the checkpoint's config before the model is built: ced/eval/probes.py
set_rope_clamp; the diagnostic harness's rope_clamp_all through the model's own attention). Composes with --length-scale
(the factor multiplies both rotations of the query). A NoPE model and a model trained with a cap <= C are scored
unchanged; a clamp below a trained cap, CED v2 with decoder self-KV and ASEntmax are refused. Recorded in settings
(rope_clamp: None or C; rope_clamp_layers: the effective cap per role and what was left unchanged). Give a clamped
scoring its own --out (for example with a suffix -rc<C>).

--resume (or CED_SCORE_RESUME=1; default off; ced/eval/units.py): a stop never costs a whole pass again. The rows of each
row shape are scored in units, closed after --unit-tokens input tokens (4M) or --unit-seconds (300), whichever comes
first; each finished unit is kept in <out>.units/ and a rerun with the same checkpoint, sets, settings and code reads
it instead of scoring. --out is the uninterrupted pass's number for number; its timing block also holds "units"
(scored, reused, reused_seconds). --force (CED_SCORE_FORCE=1) scores every unit again. --out is written last, under
a temporary name and renamed, with or without units: a stopped pass leaves no --out.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ced.config import TrainConfig, from_dict  # noqa: E402
from ced.eval import probe_gen_multifact as mf  # noqa: E402
from ced.eval import probe_turn  # noqa: E402
from ced.eval import replay_window as rw  # noqa: E402
from ced.eval import probe_gen_twohop as th  # noqa: E402
from ced.eval.probe_copy_analysis import edit_render_differences, judged  # noqa: E402
from ced.eval import units  # noqa: E402
from ced.eval.probes import (DECODER_AT, LENGTH_SCALE_REF, REPLAY_WINDOW, UNIT_SECONDS, UNIT_TOKENS, EncodedSet, compile_layers,  # noqa: E402
                             decoder_sinks, has_decoder, length_scaled_reads, paired_difference, score_sets, set_rope_clamp, summarize)
from ced.model.control import build_model  # noqa: E402


TOKENIZER_VOCAB = {"t128k": 128000, "t64k": 64000}  # probe-set tokenizer name -> vocabulary size


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    for pointer in (path / "checkpoints" / "latest", path / "latest"):
        if pointer.exists():
            return pointer.parent / pointer.read_text().strip()
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def set_backends(cfg, attn: str | None, gdn: str | None) -> None:
    """Override the attention backend / GDN kernel of every attention and GDN config in the model."""
    def walk(obj):
        if hasattr(obj, "__dataclass_fields__"):
            for name in obj.__dataclass_fields__:
                v = getattr(obj, name)
                if attn and name == "attention" and hasattr(v, "backend"):
                    v.backend = attn
                if gdn and name == "gdn" and hasattr(v, "kernel"):
                    v.kernel = gdn
                walk(v)
    walk(cfg.model)


def uses_fla(cfg) -> bool:
    kernels = []

    def walk(obj):
        if hasattr(obj, "__dataclass_fields__"):
            for name in obj.__dataclass_fields__:
                v = getattr(obj, name)
                if name == "gdn" and hasattr(v, "kernel"):
                    kernels.append(v.kernel)
                walk(v)
    walk(cfg.model)
    return "fla" in kernels


def per_cell_limit(es: EncodedSet, n: int) -> EncodedSet:
    """The first n items of every (cell, group) in file order (smoke and timing runs)."""
    seen: dict = defaultdict(int)
    keep = []
    for i, it in enumerate(es.items):
        key = json.dumps([it["cell"], it.get("group", {})], sort_keys=True)
        if seen[key] < n:
            seen[key] += 1
            keep.append(i)
    return es.subset(keep)


def window_arg(text: str):
    """--replay-window: an integer, or a last-turn name ("turn", "turn<N>": ced/eval/replay_window.py)."""
    if rw.is_turn(text):
        return text.strip()
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: an integer, 'turn' or 'turn<N>' (e.g. {rw.TURN128!r})") from None


def decoder_positions(n_inputs: int, n_answer: int, regime: str, n_sinks: int, window: int) -> tuple[int, int]:
    """(decoder positions, replay positions) of one item under answers_sinks_replay: its answers, its first n_sinks
    positions and the min(window, available) positions before the first answer (sinks and replay overlap in a short
    context), as score_sets builds the mask (ced/eval/probes.py sink_mask, replay_mask)."""
    assert regime == "answers_sinks_replay", regime
    before = n_inputs - n_answer
    return n_answer + min(before, n_sinks + window), min(before, window)


def _pm(cell: dict | None, key: str, sign: bool = False) -> str:
    """"mean±SE" of one metric of a two-hop summary cell, or n/a."""
    if not cell or cell.get(key) is None:
        return "n/a"
    return (f"{cell[key]:+.3f}" if sign else f"{cell[key]:.3f}") + f"±{(cell.get(key + '_se') or 0):.3f}"


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--probes", required=True, help="directory of <probe>.pt EncodedSet files (one tokenizer)")
    p.add_argument("--probe-names", default=None, help="comma-separated subset (default: every set in the directory)")
    p.add_argument("--max-per-group", type=int, default=None, help="first N items of every (cell, group): smoke / timing runs")
    p.add_argument("--max-item-tokens", type=int, default=None, help="skip items longer than this (reference-path checks)")
    p.add_argument("--autocast", default="auto", choices=["auto", "none"], help="auto: the run's training precision on CUDA")
    p.add_argument("--allow-vocab-mismatch", action="store_true", help="score a set built for another vocabulary size")
    p.add_argument("--T", type=int, default=None, help="loop count (default: the config's t_eval)")
    p.add_argument("--batch-rows", type=int, default=1, help="rows per forward for packed 8k rows")
    p.add_argument("--pack-len", type=int, default=8192, help="packed row length and smallest bucket (tests: small)")
    p.add_argument("--pack-max-item", type=int, default=4096, help="largest item that is packed")
    p.add_argument("--attn-backend", default=None, help="override the attention backend (e.g. flex)")
    p.add_argument("--gdn-kernel", default=None, help="override the GDN kernel (e.g. fla, chunked)")
    p.add_argument("--compile", default="auto", choices=["auto", "none", "all"],
                   help="compile the layers (CUDA): auto = as the run was trained (its compile setting)")
    p.add_argument("--taus", default="0.7", help="comma-separated sampling temperatures besides 1.0 for the per-token tempered log-prob "
                   "(tok_logp_tau; the copying probes report exp_exact / nats per token at them); empty: none")
    p.add_argument("--decoder-at", default="auto", choices=["auto", *DECODER_AT],
                   help="decoder arches: run the decoder at the answer positions only (answers), at those plus each "
                        "item's first decoder.n_sinks positions (answers_sinks: the local layers' sinks, not scored), "
                        "at those plus the --replay-window positions before each answer span (answers_sinks_replay: "
                        "not scored), or at every item position (every); auto: every for CED v2 with decoder self_kv "
                        "(its training regime), answers otherwise (ced/eval/probes.py score_sets)")
    p.add_argument("--replay-window", type=window_arg, default=REPLAY_WINDOW,
                   help="--decoder-at answers_sinks_replay: the decoder also runs at the last N input positions of each "
                        "item before its answer span (fewer when the item has fewer; default 128, the local "
                        "window); or 'turn' / 'turn<N>': at each item's own last turn, its question, capped at 512 / N "
                        "positions (turn128: the last turn capped at 128 positions); ignored by the other regimes")
    p.add_argument("--probes-text", default=None, help="the sets' text items, <probe>.jsonl.gz (default <probes>/../text); read "
                   "only when --replay-window turn / turn<N> needs question_tokens the sets do not carry")
    p.add_argument("--probes-tokenizer", default=None, help="the sets' tokenizer.json for the same (default: the path in "
                   "<probes>/manifest.json)")
    p.add_argument("--length-scale", action="store_true",
                   help="inference-time length scale (default off): every full-attention read's query times "
                        "max(1, ln(pos + 1) / ln(L)) (the encoder's global layers, CED v2's decoder full layers; local "
                        "decoder layers unscaled; ced/eval/probes.py length_scaled_reads)")
    p.add_argument("--length-scale-ref", type=int, default=LENGTH_SCALE_REF,
                   help="L of --length-scale: the training row length (default 8192; tests: small)")
    p.add_argument("--rope-clamp", type=int, default=None, metavar="C",
                   help="inference-time rotary distance clamp (default off): every full-attention read's rotary part with "
                        "min(relative distance, C) (the encoder's global layers, CED v2's decoder memory reads; "
                        "ced/eval/probes.py set_rope_clamp)")
    p.add_argument("--resume", action="store_true",
                   help="resumable pass (also CED_SCORE_RESUME=1): finished units of rows (of one row shape; closed after "
                        "--unit-tokens input tokens or --unit-seconds, whichever comes first) are kept in <out>.units/ and a "
                        "rerun with the same checkpoint, sets, settings and code reads them instead of scoring; the --out "
                        "file is the same as without (timing apart). ced/eval/units.py")
    p.add_argument("--force", action="store_true", help="--resume: score every unit again and overwrite its file (also CED_SCORE_FORCE=1)")
    p.add_argument("--unit-tokens", type=int, default=UNIT_TOKENS, help="--resume: input tokens per unit (rows of one shape; default %(default)s)")
    p.add_argument("--unit-seconds", type=float, default=UNIT_SECONDS,
                   help="--resume: a unit is also closed after this many seconds (default %(default)s): what a stop costs at "
                        "most on a slow scorer, plus the batch in flight")
    p.add_argument("--units-dir", default=None, help="--resume: where the unit files are (default <out>.units)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    from ced.eval import schedule_override

    schedule_override.add_argument(p)
    p.add_argument("--out", required=True)
    p.add_argument("--compare", default=None, help="an earlier --out JSON: paired difference (this - that) per cell")
    a = p.parse_args(argv)
    replay = a.decoder_at == "answers_sinks_replay"
    turn = replay and rw.is_turn(a.replay_window)  # each item's own last turn (its question), capped
    if replay and not turn and a.replay_window < 1:
        p.error(f"--replay-window {a.replay_window}: at least 1 with --decoder-at answers_sinks_replay")
    turn_max = rw.parse_replay_window(a.replay_window)[1] if turn else None

    t_start = time.time()
    path = resolve_checkpoint(a.checkpoint)
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = from_dict(TrainConfig, state["config"])
    set_backends(cfg, a.attn_backend, a.gdn_kernel)
    clamped = None
    if a.rope_clamp is not None:  # on the config, before the model is built (no parameter: the checkpoint loads as it is)
        try:
            clamped = set_rope_clamp(cfg.model, a.rope_clamp)
        except ValueError as e:
            raise SystemExit(f"--rope-clamp: {e}")
    model = build_model(cfg.model).to(getattr(torch, cfg.dtype))
    model.load_state_dict(state["model"])
    from ced.eval import schedule_override

    segment_map = schedule_override.apply(model, a.segment_map)  # None: the pass schedule the checkpoint was trained with
    device = torch.device(a.device)
    model = model.to(device).eval()
    cuda = device.type == "cuda"
    autocast = getattr(torch, cfg.autocast) if cuda and cfg.autocast != "none" and a.autocast == "auto" else None
    fused_varlen = cuda and uses_fla(cfg)
    n_compiled = 0
    if cuda:
        from ced.train.trainer import configure_compile

        configure_compile()
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        if (cfg.compile if a.compile == "auto" else a.compile) == "all":
            n_compiled = compile_layers(model)

    directory = Path(a.probes)
    names = a.probe_names.split(",") if a.probe_names else sorted(q.stem for q in directory.glob("*.pt"))
    sets, files = [], {}
    for n in names:
        f = directory / f"{n}.pt"
        es = EncodedSet.load(f)
        files[n] = {"path": str(f), "sha256": sha256(f), "items": len(es), "tokenizer": es.info.get("tokenizer"),
                    "tokenizer_sha256": es.info.get("tokenizer_sha256")}
        expect = TOKENIZER_VOCAB.get(es.info.get("tokenizer"))
        if expect is not None and expect != cfg.model.vocab_size and not a.allow_vocab_mismatch:
            raise SystemExit(f"{f}: built with tokenizer {es.info.get('tokenizer')} ({expect} entries) but the model "
                             f"has a vocabulary of {cfg.model.vocab_size}; pick the probe directory of its tokenizer")
        if a.max_item_tokens:
            es = es.subset([i for i in range(len(es)) if es.length(i) <= a.max_item_tokens])
        if a.max_per_group:
            es = per_cell_limit(es, a.max_per_group)
        if a.max_item_tokens or a.max_per_group:
            files[n]["items_scored"] = len(es)
        sets.append(es)
    questions_from, item_windows = None, None
    if turn:  # after the filters: only the scored items need their question
        try:
            questions_from = probe_turn.attach_questions(sets, directory, a.probes_text, a.probes_tokenizer,
                                                         what=f"--replay-window {a.replay_window}")
        except ValueError as e:
            raise SystemExit(str(e))
        item_windows = probe_turn.turn_windows(sets, turn_max)
    load_s = time.time() - t_start
    resume = a.resume or units.env_flag(units.ENV_RESUME)
    store = None
    if resume:  # everything that changes a number goes into the key of every unit (ced/eval/units.py)
        store = units.UnitStore(a.units_dir or f"{a.out}.units", {
            "script": "eval_probes", "checkpoint": str(path), "step": state.get("step"), "checkpoint_file": units.file_fingerprint(path),
            "arch": cfg.model.arch, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(device) if cuda else "cpu", "autocast": str(autocast),
            "fused_varlen": fused_varlen, "compiled_layers": n_compiled, "T": a.T, "taus": a.taus,
            "decoder_at": a.decoder_at, "replay_window": a.replay_window if replay else None, "turn_max": turn_max,
            "questions_from": questions_from, "length_scale": a.length_scale_ref if a.length_scale else None,
            "rope_clamp": a.rope_clamp, **({} if a.segment_map == "trained" else {"segment_map": a.segment_map}), "attn_backend": a.attn_backend, "gdn_kernel": a.gdn_kernel, "compile": a.compile,
            "batch_rows": a.batch_rows, "pack_len": a.pack_len, "pack_max_item": a.pack_max_item,
            "max_per_group": a.max_per_group, "max_item_tokens": a.max_item_tokens,
            "names": names,  # in this order: a row's members are (set index, item index)
            "files": {n: {k: files[n].get(k) for k in ("sha256", "items", "items_scored")} for n in files}},
            force=a.force or units.env_flag(units.ENV_FORCE), scripts=(__file__,))

    def progress(i, n):
        if i % 50 == 0 or i == n:
            print(f"rows {i}/{n}  {time.time() - t_start:.0f}s", flush=True)

    t0 = time.time()
    scale = length_scaled_reads(model, a.length_scale_ref) if a.length_scale else contextlib.nullcontext(None)
    with scale as scaled:
        results, timing = score_sets(model, sets, T=a.T, autocast_dtype=autocast, fused_varlen=fused_varlen,
                                     batch_rows=a.batch_rows, pack_len=a.pack_len, pack_max_item=a.pack_max_item,
                                     progress=progress, taus=tuple(float(x) for x in a.taus.split(",") if x),
                                     decoder_at=None if a.decoder_at == "auto" else a.decoder_at,
                                     replay_window=REPLAY_WINDOW if turn else a.replay_window, replay_windows=item_windows,
                                     unit_store=store, unit_tokens=a.unit_tokens, unit_seconds=a.unit_seconds)
    score_s = time.time() - t0
    summary = summarize(results)
    record = {
        "settings": {"checkpoint": str(path), "step": state.get("step"), "arch": cfg.model.arch,
                     "decoder_scored": has_decoder(model),
                     "decoder_at": next(iter({t["decoder_at"] for t in timing.values()}), None),
                     # answers_sinks / answers_sinks_replay: the document positions < this that also run the decoder
                     # (decoder.n_sinks)
                     "decoder_sink_positions": decoder_sinks(model) if a.decoder_at in ("answers_sinks", "answers_sinks_replay") else None,
                     # answers_sinks_replay: the positions before each answer span that also run the decoder (at most):
                     # an integer, or the last-turn name ("turn", "turn128") with its cap and where the questions'
                     # token counts came from
                     "decoder_replay_window": a.replay_window if replay else None,
                     "decoder_replay_turn_max": turn_max, "question_tokens_from": questions_from,
                     "decoder_self_kv": bool(getattr(cfg.model.decoder, "self_kv", False)),
                     "decoder_read_pattern": getattr(cfg.model.decoder, "read_pattern", "F"), "T": a.T, "device": torch.cuda.get_device_name(device) if cuda else "cpu",
                     "autocast": str(autocast), "fused_varlen": fused_varlen, "attn_backend_override": a.attn_backend,
                     "gdn_kernel_override": a.gdn_kernel, "max_per_group": a.max_per_group, "batch_rows": a.batch_rows,
                     "compiled_layers": n_compiled,
                     # --length-scale: False, or True with L and the layers scaled (length_scaled_reads)
                     "length_scale": bool(a.length_scale), "length_scale_ref": a.length_scale_ref if a.length_scale else None,
                     "length_scale_layers": scaled,
                     # --rope-clamp: None, or C with the effective cap per role (set_rope_clamp)
                     "rope_clamp": a.rope_clamp, "rope_clamp_layers": clamped,
                     "segment_map": a.segment_map, "segment_map_counts": segment_map},
        "files": files,
        "filters": {"max_per_group": a.max_per_group, "max_item_tokens": a.max_item_tokens, "probe_names": a.probe_names},
        "timing": {"load_seconds": round(load_s, 1), "score_seconds": round(score_s, 1),
                   "per_row_shape": timing,
                   # the process's peak of allocated CUDA memory (None on the CPU): whether a row shape fits in GPU memory
                   "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if cuda else None},
        "summary": summary,
        "items": results,
    }
    if replay and has_decoder(model):  # the cost side: where the decoder ran for each item, and how much of it is replay
        n_sinks = decoder_sinks(model)
        flat = [w for ws in item_windows for w in ws] if turn else [a.replay_window] * len(results)
        metas = [m for es in sets for m in es.items]
        for r, m, w in zip(results, metas, flat):
            r["run"], r["replay"] = decoder_positions(r["n_tokens"] - 1, r["n_answer"], a.decoder_at, n_sinks, w)
            if turn:
                r["question_tokens"] = int(m["question_tokens"])
    mf_meta = {m["id"]: m for es in sets if es.probe in mf.PROBES for m in es.items}
    if mf_meta:  # the multi-fact task's own metrics (fact_acc, nats per fact token, ...), as scripts/eval_long.py reports them
        mf_results = [r for r in results if r["probe"] in mf.PROBES]
        record["multifact"] = mf.summarize(mf_results, [mf_meta[r["id"]] for r in mf_results])
        for r in mf_results:  # what those metrics read, kept with the item: they can be recomputed and paired without the set
            m = mf_meta[r["id"]]
            r.update(fact_spans=m["fact_spans"], fact_depths=m.get("fact_depths"), fact_depth_tokens=m.get("fact_depth_tokens"),
                     layout=m.get("layout"))
    if any(es.probe == th.PROBE for es in sets):  # latent two-hop: the levels per variant and the paired differences, per setting
        record["twohop"] = th.summarize(results)
    if any(es.probe in ("copy_span", "edit_old") for es in sets):  # judged metrics of the copying probes (probe_copy_analysis.py)
        record["judged"] = judged(results)
        record["edit_render_differences"] = edit_render_differences(results)
    if any(es.probe == "transcode" for es in sets):  # judged metric of the transcoding probe (probe_transcode_analysis.py)
        from ced.eval.probe_transcode_analysis import cell_table, judged_transcode

        record.setdefault("judged", {})["transcode"] = judged_transcode(results) or {}
        record["transcode_cells"] = cell_table(results)
    if a.compare:
        other = json.loads(Path(a.compare).read_text())
        record["compare"] = {"against": a.compare, "cells": paired_difference(results, other["items"])}
        if "twohop" in record:  # this scoring minus that one, paired per base item (e.g. --T 2 against --T 1)
            record["compare"]["twohop"] = th.compare(results, other["items"])
    # How much of this pass was read from unit files (the timing block only). score_seconds is this process's own
    # time; reused_seconds is what the units read from files took when they were scored (a cost figure needs both).
    # peak_allocated_gib: the largest peak of this process and of the processes that scored the units it read (a
    # rerun that read most of its rows would otherwise report a peak that says nothing about whether the rows fit).
    if store is not None:
        if cuda:
            record["timing"]["peak_allocated_gib"] = round(max(torch.cuda.max_memory_allocated(device), store.reused_peak_bytes) / 2**30, 2)
        record["timing"]["units"] = {"dir": str(store.dir), "scored": store.scored, "reused": store.reused,
                                     "reused_seconds": round(store.reused_seconds, 1),
                                     "unit_tokens": a.unit_tokens, "unit_seconds": a.unit_seconds, "code": store.code,
                                     "notes": store.notes}
        for note in store.notes:
            print(note, flush=True)
        print(f"units: {store.reused} read from {store.dir}, {store.scored} scored", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    units.write_atomic(a.out, json.dumps(record))  # temporary name, then rename: a stop never leaves a partial --out

    print(f"{cfg.model.arch}: {len(results)} items in {score_s:.0f}s (load {load_s:.0f}s)  decoder_at "
          f"{record['settings']['decoder_at']}{' W=%s' % a.replay_window if replay else ''}  "
          f"length_scale {'L=%d %s' % (a.length_scale_ref, scaled) if a.length_scale else 'off'}"
          + (f"  rope_clamp {clamped}" if clamped is not None else ""))
    for t in sorted(timing.values(), key=lambda t: (t["length"], t["kind"])):
        print(f"  L={t['length']:6d} {t['kind']:6s}: {t['rows']:5d} rows  capacity {t['capacity']:4d}  {t['seconds']:7.1f}s  "
              f"first batch {t['first_batch_seconds']:.1f}s  {t['tokens_per_s_after_first'] or 0:,.0f} tok/s after the first")
    if cuda:
        print(f"  peak allocated {record['timing']['peak_allocated_gib']:.2f} GiB")
    for probe, j in record.get("judged", {}).items():
        print(f"  JUDGED {probe}: " + "  ".join(f"{k}={v:.4f}" for k, v in j.items() if isinstance(v, float)))
    for probe, ks in record.get("multifact", {}).items():
        for k, cells in ks.items():
            for length, c in cells.items():
                print(f"  {probe:12s} {k:4s} L={length:>6s} n={c['n']:4d}  exact={c['exact']:.3f}  fact_acc={c['fact_acc']:.3f}  "
                      f"nats/fact tok={c['nats']:.3f}  first_nats={c['first_nats']:.3f}")
    for setting, c in record.get("twohop", {}).items():  # the levels by setting; the cells are in the JSON
        if th.is_cell(setting):
            continue
        v, pr = c["variants"], c["paired"]
        print(f"  TWOHOP {setting:16s} n={c['n']:4d}  first token right: 2-hop {_pm(v.get('twohop'), 'first_ok')}  "
              f"1-hop {_pm(v.get('onehop'), 'first_ok')}  shuffled {_pm(v.get('shuffled'), 'first_ok')}  "
              f"swapped {_pm(v.get('swapped'), 'first_ok')}  chance {c['chance']:.3f}  "
              f"2-hop minus 1-hop {_pm(pr.get('hop_cost'), 'first_ok', True)}  minus shuffled {_pm(pr.get('link_use'), 'first_ok', True)}  "
              f"minus swapped {_pm(pr.get('link_follow'), 'first_ok', True)}  "
              f"first nats: 2-hop {_pm(v.get('twohop'), 'first_nats')}  1-hop {_pm(v.get('onehop'), 'first_nats')}")
    for probe, cells in summary.items():
        if probe == th.PROBE:  # one cell per variant and setting (120 in the built set): in the JSON; the lines above carry the reading
            continue
        for c, s in cells.items():
            line = f"  {probe:12s} {c:24s} n={s['n']:4d}  nats/tok={s['nats_per_token']:.3f}  tok_acc={s['token_acc']:.3f}"
            line += f"  exact={s['exact']:.3f}±{(s['exact_se'] or 0):.3f}  first={s['first_acc']:.3f}"
            if "mc_acc" in s:
                line += f"  mc_acc={s['mc_acc']:.3f}±{(s['mc_acc_se'] or 0):.3f}  distractor={s['distractor_rate']:.3f}"
                if "literal_acc" in s:
                    line += f"  literal_acc={s['literal_acc']:.3f}"
            print(line)
    return record


if __name__ == "__main__":
    main()
