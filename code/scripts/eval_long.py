"""Long-context evaluation of one checkpoint: the four parts that are not needles.

    python scripts/eval_long.py --checkpoint <run> --data <long-context sets> --out <out>
    python scripts/eval_long.py --checkpoint <run> --data <long-context sets> --out <out>-ln \\
        --length-scale

--checkpoint: a checkpoint file, a run directory or its checkpoints/ directory (`latest` is followed). --data: the
long-context eval set (EvalRows files, see EVAL.md: web.pt, code.pt, code_repo.pt, agent.pt, gdn_joined.pt) with the
multi-fact probes in <data>/multifact/t64k (tools/build_probes_multifact.py); a file that is not there is reported as
missing and its part left out. The model is rebuilt from the checkpoint's config as scripts/eval_probes.py does: on
CUDA in its training precision, with the fused GDN path and its layers compiled when the run trained that way.

Parts (--parts, default all):
  bpb        bpb by position on whole held-out long documents, one document per row out to 64k, per set (web, code,
             code_repo), with the same targets scored again with a 4k-8k context (ced/eval/long_context.py)
  agent      assistant bpb on whole held-out agent trajectories longer than 8k, with the window cut to 16k / 32k / 64k
             (and 8k, the reference), on assistant tokens
  multifact  the multi-fact task (mf_chain, mf_concat; k = 2 / 4 / 8 facts; 8k / 16k / 32k / 64k)
  gdn        GDN state diagnostics by position: natural documents to 64k, and joined rows at the longest of
             --gdn-lengths (256k, then 128k, then 64k) whose encoder forward fits

Models: control (a) (one head at every position) and CED v2 (--decoder-at every, the default: the decoder at every
valid target, the regime these checkpoints trained in; "scored" / "scored_sinks": only at the scored targets, without /
with the document's first decoder.n_sinks positions, the restricted regime, which matters on the agent part).
--decoder-at scored_sinks_replay (the restricted regime with a replay window; the agent and multi-fact parts): scored_sinks plus the replay positions before each span start (a
scored position whose previous position in the same document is not scored), which run the decoder and write decoder
K/V and are not scored, so the scored positions and their targets are those of every other regime:
  --replay-window W      the min(W, available) positions immediately before the span start;
  --replay-window turn   the whole last turn before the span start, capped at its last --replay-turn-max (512)
                         positions. Agent trajectories: the last message before the assistant turn (a user message or
                         a tool result, with its header) and the assistant header's tokens before the span start,
                         read from the chat format's <|im_start|> tokens (--im-start-id; ced/eval/long_context.py
                         replay_positions); a message whose start the window cut off is replayed from the row's first
                         position. Multi-fact items: the question, from its first word to the answer (the item's
                         question_tokens less its last token, which is the first scored position; sets built before
                         the builder stored it get it from their text items and tokenizer: --multifact-text,
                         --multifact-tokenizer).
  --replay-window turn<N>  the same with the cap in the name: "turn128" is the whole last turn, at most the 128 positions
                         nearest the span start (ced/eval/replay_window.py has
                         the rule for every kind of scored unit). It is "--replay-window turn --replay-turn-max N": the
                         same settings and the same part keys, so either form resumes the other's output.
The capacity is what the restricted regimes had: the row length on the agent part, the answers plus the extra positions
rounded up to 128 on the multi-fact part. --reuse-from DIR: a part that DIR holds for the same checkpoint and settings
(the part's key, and the scoring settings outside it: REUSE_SETTINGS), complete for this code (the agent part needs the
first-token split), is copied instead of scored (an "every" pass can take the multi-fact
part of an earlier output this way). The code that scored it is not compared: the copy is as good as that pass.
Softmax and Scalable-Softmax checkpoints alike: an ssmax model's length scale is in its weights. --length-scale (softmax
and learned-temperature models only; refused for ssmax / ASEntmax) multiplies every full-attention read's query by
max(1, ln(pos + 1) / ln L), L = --length-scale-ref (default: the base length, 8,192), as scripts/eval_probes.py
--length-scale does (ced/eval/probes.length_scaled_reads: the encoder's global layers and CED v2's decoder full layers,
through GatedAttention.query_scale); score it into its own --out directory.
--rope-clamp C (default off): the rotary part of every full-attention read with min(relative
distance, C), as scripts/eval_probes.py --rope-clamp (ced/eval/probes.set_rope_clamp: the encoder's global layers and CED
v2's decoder memory reads; composes with --length-scale and with --decoder-at). All four parts run under it: bpb, agent
and multifact score the clamped model; gdn then reports the GDN states of the clamped model (the GDN layers themselves
have no rotary part: only their inputs change). Recorded in the settings (rope_clamp, rope_clamp_layers) and in every
part's key, so a clamped and an unclamped scoring never share an output; score it into its own --out directory too.

Memory: every row is one forward at batch 1 (the 8k reference rows: --ref-batch per forward); the head is applied to
hidden states in chunks of 1,024 positions (no full logits); the GDN walk goes 8,192 positions at a time; a joined row
that does not fit is tried at the next shorter length.

Output, in --out: bpb-<set>.json, agent.json, multifact.json, gdn.json (each part's full result with its settings; a part
whose file exists for the same checkpoint and settings is not scored again, --force rescores), summary.json (the
compact record), summary.md and summary-tail.md (brief, under 16 KiB).

--base-len (default 8,192): the training row length everything is scaled from (bucket edges 0 / L/4 / L / 2L / 4L /
8L, the largest row 8L, the reference window L and stride L/2, the agent windows 2L / 4L / 8L and tail L/2, the GDN
buckets up to 32L); tiny models can be run at a small base length.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from ced.config import TrainConfig, from_dict  # noqa: E402
from ced.eval import gdn_diag, long_report, probe_turn  # noqa: E402
from ced.eval import probe_gen_multifact as mf  # noqa: E402
from ced.eval import replay_window as rw  # noqa: E402
from ced.data.tokenizers import TOKENIZER_VOCAB_BY_SHA256, eval_set_tokenizer  # noqa: E402
from ced.eval.long_context import (DECODER_AT, IM_START_ID, REPLAY_TURN, REPLAY_TURN_MAX, LongDocs, RowScorer,  # noqa: E402
                                   default_edges, length_stats, make_row, row_length, score_agent_windows, score_position_bpb)
from ced.eval.probes import (EncodedSet, compile_layers, compiled_block_masks, length_scaled_reads, score_sets,  # noqa: E402
                             set_rope_clamp)
from ced.model.control import build_model  # noqa: E402
from eval_probes import per_cell_limit, resolve_checkpoint, set_backends, sha256  # noqa: E402

PARTS = ("bpb", "agent", "multifact", "gdn")
BPB_SETS = ("web", "code", "code_repo")
# score_sets' decoder_at
PROBE_REGIME = {"every": None, "scored": "answers", "scored_sinks": "answers_sinks", "scored_sinks_replay": "answers_sinks_replay"}
AGENT_SPLIT_KEYS = ("first_nats", "first_bytes", "first_tokens", "tail_first_nats", "scored_sha", "spans")
TAIL_LIMIT = 15_000  # bytes of summary-tail.md


def uses_fused_gdn(cfg) -> bool:
    """A GDN config of the model asks for a fused varlen kernel ("fla" or "ced"): pass the document boundaries."""
    found = []

    def walk(obj):
        if hasattr(obj, "__dataclass_fields__"):
            for name in obj.__dataclass_fields__:
                v = getattr(obj, name)
                if name == "gdn" and hasattr(v, "kernel"):
                    found.append(v.kernel)
                walk(v)

    walk(cfg.model)
    return any(k in ("fla", "ced") for k in found)


def ints(text: str | None) -> list[int] | None:
    return None if text is None else [int(x) for x in text.split(",") if x]


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def load_part(path: Path, key: dict, force: bool):
    """A part's stored result when it was scored for the same checkpoint and settings, else None."""
    if force or not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except ValueError:
        return None
    return data["result"] if data.get("key") == key else None


def save_part(path: Path, key: dict, settings: dict, result: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"key": key, "settings": settings, "result": result}))
    tmp.replace(path)


def part_complete(part: str, result: dict) -> bool:
    """A stored part has what this code writes and the replay comparison reads (a part scored by earlier code may not):
    agent: the first-token split, the span counts and the scored fingerprint of every (trajectory, window);
    multifact: the per-token record of every item."""
    if part == "agent":
        return bool(result.get("docs")) and all(all(k in w for k in AGENT_SPLIT_KEYS) for d in result["docs"] for w in d["windows"].values())
    if part == "multifact":
        return bool(result.get("items")) and all("tok_logp" in it and "tok_correct" in it for it in result["items"])
    return True


# what a part's numbers depend on besides its key: reused only when the pass that scored it had the same (--reuse-from)
REUSE_SETTINGS = ("arch", "temperature", "normalizer", "decoder_self_kv", "decoder_read_pattern", "autocast", "fused_varlen",
                  "compiled_layers", "attn_backend_override", "gdn_kernel_override", "length_scale_layers")


def reuse_part(directory: Path | None, name: str, part: str, key: dict, settings: dict):
    """--reuse-from: the stored result of `name` in `directory` when it was scored for the same key (checkpoint, step,
    decoder regime, T, the inference factor and its reference length, the data, the part's own arguments), with the
    same scoring settings the key does not hold (REUSE_SETTINGS: the body's precision, compiled or not, the kernel
    overrides, the layers the factor is on), and is complete (`part_complete`); else None."""
    if directory is None:
        return None
    got = load_part(directory / name, key, False)
    if got is None or not part_complete(part, got):
        return None
    stored = json.loads((directory / name).read_text()).get("settings") or {}
    return got if all(stored.get(k) == settings.get(k) for k in REUSE_SETTINGS) else None


def parse_replay_window(text: str | None, turn_max: int | None = None):
    """--replay-window with --replay-turn-max: (None, None) without it; (W, None) for an integer W >= 1; ("turn", cap)
    for "turn" (cap: --replay-turn-max, default 512) and for "turn<N>" (cap N: "turn128"; a --replay-turn-max given
    with it must be N). The names are ced/eval/replay_window.py's."""
    if text is None:
        return None, None
    try:
        return rw.parse_replay_window(text, turn_max)
    except ValueError as e:
        raise SystemExit(f"--replay-window {text!r}: an integer >= 1 or {REPLAY_TURN!r} (or 'turn<N>', the last turn capped "
                         f"at N positions, e.g. {rw.TURN128!r}): {e}")


def multifact_questions(sets: list, directory: Path, text_dir: Path | None, tokenizer: str | None) -> str:
    """Make sure every item of the multi-fact sets carries question_tokens (--replay-window turn / turn<N>). Sets built
    since the builder stores it have it. Others get it here: the question text from the set's text items (<text_dir>/
    <probe>.jsonl.gz, default <multifact>/../text), tokenized with the set's tokenizer (--multifact-tokenizer, default
    the path in <multifact>/manifest.json) and checked against each item's stored tokens (ced/eval/probe_turn.py
    attach_questions, which scripts/eval_probes.py uses for the needle sets too). Returns where it came from."""
    try:
        return probe_turn.attach_questions(sets, directory, text_dir, tokenizer, what="--replay-window turn",
                                           text_flag="--multifact-text", tokenizer_flag="--multifact-tokenizer")
    except ValueError as e:
        raise SystemExit(str(e))


def multifact_positions(n_inputs: int, n_answer: int, regime: str | None, n_sinks: int, window: int) -> tuple[int, int]:
    """(decoder positions, replay positions) of one multi-fact item: n_inputs inputs, the last n_answer scored. every:
    all inputs; scored: the answers; scored_sinks: plus the first n_sinks; scored_sinks_replay: plus the min(window,
    available) positions before the first scored one (sinks and replay overlap in a short context). None (the
    control): no decoder."""
    before = n_inputs - n_answer
    if regime is None:
        return 0, 0
    if regime == "every":
        return n_inputs, 0
    if regime == "scored":
        return n_answer, 0
    if regime == "scored_sinks":
        return n_answer + min(before, n_sinks), 0
    return n_answer + min(before, n_sinks + window), min(before, window)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="the long-context eval set directory (EvalRows files and multifact/t64k; see EVAL.md)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--parts", default=",".join(PARTS))
    p.add_argument("--force", action="store_true", help="score again the parts whose output exists")
    p.add_argument("--base-len", type=int, default=8192, help="the training row length (small for tiny test models)")
    p.add_argument("--decoder-at", default="every", choices=DECODER_AT, help="CED v2: where the decoder runs")
    p.add_argument("--replay-window", default=None, help="--decoder-at scored_sinks_replay: an integer W (the positions replayed "
                   "before each span start), 'turn' (the whole last turn before it) or 'turn<N>' (the whole last turn, at most "
                   "the N positions nearest the span start: turn128)")
    p.add_argument("--replay-turn-max", type=int, default=None, help=f"--replay-window turn: the cap, in positions (default "
                   f"{REPLAY_TURN_MAX}; with turn<N> the cap is N)")
    p.add_argument("--im-start-id", type=int, default=None, help="the <|im_start|> token id of the agent set (default: "
                   f"{IM_START_ID} when the set records a known tokenizer; else the turn replay and the turn lengths are refused / left out)")
    p.add_argument("--reuse-from", default=None, help="a directory of an earlier pass: a part it holds for the same checkpoint "
                   "and settings, complete for this code, is copied instead of scored")
    p.add_argument("--length-scale", action="store_true", help="inference-time ln n / ln L factor on every full-attention read "
                   "(softmax models; ced/eval/probes.length_scaled_reads)")
    p.add_argument("--length-scale-ref", type=int, default=None, help="L of --length-scale (default: --base-len)")
    p.add_argument("--rope-clamp", type=int, default=None, metavar="C", help="inference-time rotary distance clamp: every "
                   "full-attention read's rotary part with min(relative distance, C) (ced/eval/probes.set_rope_clamp)")
    p.add_argument("--T", type=int, default=None, help="CED v2 loop count (default: the config's t_eval)")
    # bpb
    p.add_argument("--bpb-sets", default=",".join(BPB_SETS))
    p.add_argument("--edges", default=None, help="position bucket edges (default 0, L/4, L, 2L, 4L, 8L)")
    p.add_argument("--max-len", type=int, default=None, help="scored targets per document (default 8L)")
    p.add_argument("--max-docs", type=int, default=None, help="first N documents of every set (smoke runs)")
    p.add_argument("--ref-window", type=int, default=None, help="the reference context window (default L); 0: no reference pass")
    p.add_argument("--ref-stride", type=int, default=None, help="default: half the reference window")
    p.add_argument("--ref-batch", type=int, default=4, help="reference rows per forward")
    # agent
    p.add_argument("--agent-windows", default=None, help="default 2L,4L,8L")
    p.add_argument("--agent-tail", type=int, default=None, help="the last targets every window is compared on (default L/2)")
    p.add_argument("--agent-min-tokens", type=int, default=None, help="default L + 1 (longer than the training row)")
    p.add_argument("--agent-max", type=int, default=None, help="first N trajectories (smoke runs)")
    # multifact
    p.add_argument("--multifact", default=None, help="directory of mf_*.pt EncodedSet files (default <data>/multifact/t64k)")
    p.add_argument("--multifact-lengths", default=None, help="only the items of these lengths")
    p.add_argument("--multifact-max-per-group", type=int, default=None, help="first N items of every (cell, layout)")
    p.add_argument("--pack-len", type=int, default=None, help="score_sets' packed row length and smallest bucket (default L)")
    p.add_argument("--multifact-text", default=None, help="the sets' text items, <probe>.jsonl.gz (default <multifact>/../text); "
                   "read only when --replay-window turn needs question_tokens the sets do not carry")
    p.add_argument("--multifact-tokenizer", default=None, help="the sets' tokenizer.json for the same (default: the path in "
                   "<multifact>/manifest.json)")
    # gdn
    p.add_argument("--gdn-edges", default=None, help="default 0, L/4, L, 2L, 4L, 8L, 16L, 32L")
    p.add_argument("--gdn-docs", type=int, default=4, help="natural documents per set (the longest)")
    p.add_argument("--gdn-lengths", default=None, help="joined row lengths to try, longest first (default 32L,16L,8L)")
    p.add_argument("--gdn-samples", type=int, default=8, help="sample positions per bucket and row")
    p.add_argument("--gdn-block", type=int, default=8192)
    # model
    p.add_argument("--autocast", default="auto", choices=["auto", "none"], help="auto: the run's training precision on CUDA")
    p.add_argument("--attn-backend", default=None, help="override the attention backend (e.g. reference, sdpa)")
    p.add_argument("--gdn-kernel", default=None, help="override the GDN kernel (e.g. chunked)")
    p.add_argument("--compile", default="auto", choices=["auto", "none", "all"], help="auto: as the run was trained")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)

    t_start = time.time()
    L0 = a.base_len
    parts_wanted = [x for x in a.parts.split(",") if x]
    assert set(parts_wanted) <= set(PARTS), f"--parts: a subset of {PARTS}"
    edges = tuple(ints(a.edges) or default_edges(L0))
    max_len = a.max_len or 8 * L0
    ref_window = L0 if a.ref_window is None else a.ref_window
    windows = ints(a.agent_windows) or [2 * L0, 4 * L0, 8 * L0]
    gdn_edges = tuple(ints(a.gdn_edges) or gdn_diag.default_gdn_edges(L0))
    gdn_lengths = sorted(ints(a.gdn_lengths) or [32 * L0, 16 * L0, 8 * L0], reverse=True)
    scale_ref = a.length_scale_ref or L0
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

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
    device = torch.device(a.device)
    model = model.to(device).eval()
    cuda = device.type == "cuda"
    autocast = getattr(torch, cfg.autocast) if cuda and cfg.autocast != "none" and a.autocast == "auto" else None
    fused_varlen = cuda and uses_fused_gdn(cfg)
    n_compiled = 0
    if cuda:
        from ced.train.trainer import configure_compile

        configure_compile()
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        if (cfg.compile if a.compile == "auto" else a.compile) == "all":
            n_compiled = compile_layers(model)
    arch = cfg.model.arch
    att = cfg.model.encoder.attention
    replay_window, turn_max = parse_replay_window(a.replay_window, a.replay_turn_max)  # turn_max: the turn's cap, else None
    replay = a.decoder_at == "scored_sinks_replay"
    if replay and replay_window is None:
        raise SystemExit("--decoder-at scored_sinks_replay needs --replay-window (an integer or 'turn')")
    if not replay and replay_window is not None:
        raise SystemExit(f"--replay-window is for --decoder-at scored_sinks_replay, not {a.decoder_at}")
    replay = replay and arch == "ced_v2"  # the control has no decoder: the regime is ignored, as the others are
    turn = replay and replay_window == REPLAY_TURN
    # the agent set's <|im_start|> id: given, or the chat format's when the set records a tokenizer trained with it
    im_start = a.im_start_id
    if im_start is None and eval_set_tokenizer(data / "agent.pt") in TOKENIZER_VOCAB_BY_SHA256:
        im_start = IM_START_ID
    n_sinks = int(cfg.model.decoder.n_sinks) if arch == "ced_v2" and a.decoder_at in ("scored_sinks", "scored_sinks_replay") else 0
    settings = {"run": out.name, "checkpoint": str(path), "step": state.get("step"), "arch": arch,
                "temperature": att.temperature, "normalizer": att.normalizer, "base_len": L0,
                "decoder_at": a.decoder_at if arch == "ced_v2" else None,
                "replay_window": replay_window if replay else None, "replay_turn_max": turn_max if turn else None,
                "decoder_sinks": n_sinks, "im_start_id": im_start,
                "decoder_self_kv": bool(getattr(cfg.model.decoder, "self_kv", False)),
                "decoder_read_pattern": getattr(cfg.model.decoder, "read_pattern", "F"), "T": a.T,
                "length_scale": bool(a.length_scale), "length_scale_ref": scale_ref if a.length_scale else None,
                # --rope-clamp: None, or C with the effective cap per role (set_rope_clamp)
                "rope_clamp": a.rope_clamp, "rope_clamp_layers": clamped,
                "device": torch.cuda.get_device_name(device) if cuda else "cpu", "autocast": str(autocast),
                "fused_varlen": fused_varlen, "compiled_layers": n_compiled, "data": str(data),
                "attn_backend_override": a.attn_backend, "gdn_kernel_override": a.gdn_kernel}
    manifest = data / "manifest.json"
    if manifest.exists():
        settings["data_manifest_sha256"] = sha256(manifest)
    base_key = {"checkpoint": str(path), "step": state.get("step"), "decoder_at": settings["decoder_at"], "T": a.T,
                "length_scale": settings["length_scale"], "length_scale_ref": settings["length_scale_ref"], "base_len": L0,
                "data_manifest_sha256": settings.get("data_manifest_sha256")}
    if a.rope_clamp is not None:  # only then: the keys of every part scored before the option existed stay valid
        base_key["rope_clamp"] = a.rope_clamp
    if replay:  # only then: the keys of every other regime stay what earlier outputs carry
        base_key.update(replay_window=replay_window, replay_turn_max=turn_max if turn else None)
    reuse = Path(a.reuse_from) if a.reuse_from else None
    print(f"[eval-long] {long_report.describe_settings(settings)}; parts {parts_wanted}; device {settings['device']}; "
          f"compiled layers {n_compiled}", flush=True)

    if turn and im_start is None and "agent" in parts_wanted:
        raise SystemExit("--replay-window turn on the agent part reads the <|im_start|> tokens: the set records no known "
                         "tokenizer, pass --im-start-id")
    # (the turn without an <|im_start|> id and without the agent part: only the multi-fact part replays, by its own
    # per-item windows; the row scorer then never sees a span to replay before and runs as scored_sinks)
    row_regime = "scored_sinks" if turn and im_start is None else a.decoder_at
    scorer = RowScorer(model, T=a.T, autocast_dtype=autocast, fused_varlen=fused_varlen, decoder_at=row_regime,
                       replay_window=replay_window if row_regime == "scored_sinks_replay" else None,
                       replay_turn_max=turn_max if row_regime == "scored_sinks_replay" else None, im_start=im_start)
    parts: dict = {}
    seconds: dict = {}
    missing: list[str] = []

    def progress(tag: str, every: int = 10):
        def f(i, n):
            if i % every == 0 or i == n:
                print(f"[eval-long] {tag} {i}/{n}  {time.time() - t_start:.0f}s", flush=True)
        return f

    def load_docs(name: str) -> LongDocs | None:
        f = data / f"{name}.pt"
        if not f.exists():
            missing.append(str(f))
            return None
        docs = LongDocs.load(f)
        assert len(docs.ids) == 0 or int(docs.ids.max()) < cfg.model.vocab_size, \
            f"{f}: token id {int(docs.ids.max())} outside the model's vocabulary of {cfg.model.vocab_size}"
        return docs

    try:
        scale = length_scaled_reads(model, scale_ref) if a.length_scale else contextlib.nullcontext(None)
        scale_ctx = scale.__enter__()
    except ValueError as e:
        raise SystemExit(f"--length-scale: {e}")
    settings["length_scale_layers"] = scale_ctx
    masks = compiled_block_masks(cuda)
    masks.__enter__()
    try:
        # ---- 1. bpb by position
        if "bpb" in parts_wanted:  # one output per set, so a stopped run loses one set at most
            parts["bpb"], seconds["bpb"] = {}, 0.0
            for name in filter(None, a.bpb_sets.split(",")):
                key = {**base_key, "part": "bpb", "set": name, "edges": list(edges), "max_len": max_len, "max_docs": a.max_docs,
                       "ref_window": ref_window, "ref_stride": a.ref_stride}
                got = load_part(out / f"bpb-{name}.json", key, a.force)
                if got is None:
                    docs = load_docs(name)
                    if docs is None or len(docs) == 0:
                        continue
                    got = score_position_bpb(scorer, docs, max_len=max_len, edges=edges, base=L0, ref_window=ref_window,
                                             ref_stride=a.ref_stride, ref_batch=a.ref_batch, max_docs=a.max_docs,
                                             progress=progress(f"bpb {name}"))
                    save_part(out / f"bpb-{name}.json", key, settings, got)
                else:
                    print(f"[eval-long] bpb {name}: exists, not rescored", flush=True)
                parts["bpb"][name] = got
                seconds["bpb"] += got["seconds"]
        # ---- 2. agent trajectories
        if "agent" in parts_wanted:
            key = {**base_key, "part": "agent", "windows": windows, "ref_window": L0, "tail": a.agent_tail,
                   "min_tokens": a.agent_min_tokens, "max": a.agent_max}
            got = load_part(out / "agent.json", key, a.force)
            if got is None and not a.force and (got := reuse_part(reuse, "agent.json", "agent", key, settings)) is not None:
                save_part(out / "agent.json", key, {**settings, "reused_from": str(reuse)}, got)
                print(f"[eval-long] agent: reused from {reuse}", flush=True)
            elif got is None:
                docs = load_docs("agent")
                if docs is not None and len(docs):
                    got = score_agent_windows(scorer, docs, windows=windows, ref_window=L0, tail=a.agent_tail,
                                              min_tokens=a.agent_min_tokens, base=L0, max_docs=a.agent_max,
                                              progress=progress("agent"), im_start=im_start)
                    save_part(out / "agent.json", key, settings, got)
            else:
                print("[eval-long] agent: exists, not rescored", flush=True)
            if got is not None:
                parts["agent"], seconds["agent"] = got, got["seconds"]
        # ---- 3. the multi-fact task
        if "multifact" in parts_wanted:
            directory = Path(a.multifact) if a.multifact else data / "multifact" / "t64k"
            files = sorted(directory.glob("mf_*.pt")) if directory.exists() else []
            key = {**base_key, "part": "multifact", "files": {f.name: sha256(f) for f in files},
                   "lengths": a.multifact_lengths, "max_per_group": a.multifact_max_per_group, "pack_len": a.pack_len}
            got = load_part(out / "multifact.json", key, a.force)
            if got is None and not a.force and (got := reuse_part(reuse, "multifact.json", "multifact", key, settings)) is not None:
                save_part(out / "multifact.json", key, {**settings, "reused_from": str(reuse)}, got)
                print(f"[eval-long] multifact: reused from {reuse}", flush=True)
            elif got is None and not files:
                missing.append(str(directory / "mf_*.pt"))
            elif got is None:
                t0 = time.time()
                sets = [EncodedSet.load(f) for f in files]
                questions = None
                if turn:
                    questions = multifact_questions(sets, directory, Path(a.multifact_text) if a.multifact_text else None,
                                                    a.multifact_tokenizer)
                loaded, sets = sets, []
                for es in loaded:
                    if a.multifact_lengths:
                        keep_len = set(ints(a.multifact_lengths))
                        es = es.subset([i for i in range(len(es)) if es.items[i]["cell"]["length"] in keep_len])
                    if a.multifact_max_per_group:
                        es = per_cell_limit(es, a.multifact_max_per_group)
                    sets.append(es)
                pack = a.pack_len or L0
                # the replay window per item: W, or for "turn" the question less its last token (the first scored
                # position), capped: the decoder then runs from the question's first token on
                kw = {}
                if replay and not turn:
                    kw = {"replay_window": replay_window}
                elif turn:
                    kw = {"replay_windows": probe_turn.turn_windows(sets, turn_max)}
                results, timing = score_sets(model, sets, T=a.T, autocast_dtype=autocast, fused_varlen=fused_varlen,
                                             pack_len=pack, pack_max_item=pack // 2, progress=progress("multifact rows", 50),
                                             taus=(), decoder_at=PROBE_REGIME[a.decoder_at] if arch == "ced_v2" else None, **kw)
                metas = [m for es in sets for m in es.items]
                item_windows = [w for ws in kw["replay_windows"] for w in ws] if turn else [replay_window if replay else 0] * len(metas)
                # where the decoder ran, from the regime score_sets reports (a CED v2 model without decoder self-KV is
                # scored at its answers under "every": the same scores, fewer positions)
                ran = next(iter({t["decoder_at"] for t in timing.values()}), None)
                ran = {v: k for k, v in PROBE_REGIME.items() if v is not None}.get(ran, ran)  # "every" and None stay
                items = []
                for r, m, w in zip(results, metas, item_windows):
                    n_run, n_replay = multifact_positions(r["n_tokens"] - 1, r["n_answer"], ran, n_sinks, w)
                    items.append({**{k: r[k] for k in ("probe", "id", "cell", "group", "n_tokens", "n_answer", "fingerprint",
                                                       "tok_logp", "tok_correct")},
                                  "fact_spans": m["fact_spans"], "fact_depths": m.get("fact_depths"),
                                  "fact_depth_tokens": m.get("fact_depth_tokens"), "question_tokens": m.get("question_tokens"),
                                  "run": n_run, "replay": n_replay})
                got = {"summary": mf.summarize(results, metas), "items": items, "timing": timing,
                       "decoder_at": next(iter({t["decoder_at"] for t in timing.values()}), None),
                       "question_tokens_from": questions,
                       "questions": {es.probe: length_stats([m["question_tokens"] for m in es.items if m.get("question_tokens") is not None])
                                     for es in sets},
                       "seconds": time.time() - t0}
                save_part(out / "multifact.json", key, settings, got)
            else:
                print("[eval-long] multifact: exists, not rescored", flush=True)
            if got is not None:
                parts["multifact"], seconds["multifact"] = got, got["seconds"]
        # ---- 4. GDN state diagnostics
        if "gdn" in parts_wanted:
            key = {**base_key, "part": "gdn", "edges": list(gdn_edges), "docs": a.gdn_docs, "lengths": gdn_lengths,
                   "samples": a.gdn_samples, "sets": a.bpb_sets, "max_len": max_len}
            got = load_part(out / "gdn.json", key, a.force)
            if got is None:
                t0 = time.time()
                got = {"natural": {}, "joined": {}, "joined_attempts": []}
                kw = dict(per_bucket=a.gdn_samples, autocast_dtype=autocast, fused_varlen=fused_varlen, block=a.gdn_block)
                natural_edges = tuple(e for e in gdn_edges if e <= max_len)
                for name in filter(None, a.bpb_sets.split(",")):
                    docs = load_docs(name) if (data / f"{name}.pt").exists() else None
                    if docs is None or len(docs) == 0:
                        continue
                    longest = sorted(range(len(docs)), key=lambda i: (-docs.length(i), i))[: a.gdn_docs]
                    stats = gdn_diag.GDNStats(natural_edges, train_len=L0)
                    for i in sorted(longest):
                        ids = docs.doc(i)[0][: max_len + 1]
                        if len(ids) < 3:
                            continue
                        m = len(ids) - 1
                        tokens, _, doc_ids, _ = make_row(ids, row_length(m, L0))
                        gdn_diag.encoder_gdn_diag(model, tokens[None], doc_ids[None], m, stats, **kw)
                    got["natural"][name] = stats.result()
                    print(f"[eval-long] gdn natural {name}: {stats.rows} rows  {time.time() - t_start:.0f}s", flush=True)
                joined = load_docs("gdn_joined")
                if joined is not None and len(joined):
                    by_source: dict = {}
                    for i in range(len(joined)):
                        by_source.setdefault(joined.meta[i].get("source", "joined"), []).append(i)
                    fit = None
                    for length in gdn_lengths:  # the longest length whose encoder forward fits
                        if length > joined.length(0) - 1:
                            got["joined_attempts"].append({"length": length, "fit": False, "error": "the joined rows are shorter"})
                            continue
                        trial: dict = {}
                        try:
                            for name, rows in by_source.items():
                                stats = gdn_diag.GDNStats(tuple(e for e in gdn_edges if e <= length), train_len=L0)
                                for i in rows:
                                    ids = joined.doc(i)[0][: length + 1]
                                    tokens, _, doc_ids, _ = make_row(ids, row_length(length, L0))
                                    gdn_diag.encoder_gdn_diag(model, tokens[None], doc_ids[None], length, stats, **kw)
                                trial[name] = stats.result()
                        except Exception as e:  # noqa: BLE001  (an out-of-memory error: try the next shorter length)
                            if not is_oom(e):
                                raise
                            got["joined_attempts"].append({"length": length, "fit": False, "error": str(e)[:300]})
                            del e
                            if cuda:
                                torch.cuda.empty_cache()
                            continue
                        got["joined_attempts"].append({"length": length, "fit": True})
                        got["joined"], fit = trial, length
                        break
                    got["joined_length"] = fit
                    print(f"[eval-long] gdn joined: {got['joined_attempts']}  {time.time() - t_start:.0f}s", flush=True)
                got["seconds"] = time.time() - t0
                save_part(out / "gdn.json", key, settings, got)
            else:
                print("[eval-long] gdn: exists, not rescored", flush=True)
            parts["gdn"], seconds["gdn"] = got, got["seconds"]
    finally:
        masks.__exit__(None, None, None)
        scale.__exit__(None, None, None)

    summary = long_report.compact(parts, settings)
    summary["timing"] = {**seconds, "total": time.time() - t_start}
    summary["row_timing"] = scorer.timing_report()
    # the process's peak of allocated CUDA memory (None on the CPU): whether the longest row shape fits in GPU memory
    summary["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if cuda else None
    summary["missing"] = missing
    if parts.get("gdn"):
        summary["gdn_joined_length"] = parts["gdn"].get("joined_length")
        summary["gdn_joined_attempts"] = parts["gdn"].get("joined_attempts")
    entries = long_report.degradations(summary)
    summary["degradations"] = entries
    summary["breaks_first"] = long_report.breaks_first(entries)
    summary["breaks_first_clear"] = long_report.breaks_first(entries, clear_only=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    text = long_report.render(summary, entries)
    if missing:
        text += "\nMissing inputs (their parts were not scored): " + ", ".join(missing) + "\n"
    (out / "summary.md").write_text(text)
    tail = long_report.render(summary, entries, brief=True)
    if len(tail.encode("utf-8")) > TAIL_LIMIT:
        tail = tail.encode("utf-8")[:TAIL_LIMIT].decode("utf-8", "ignore") + "\n[... cut: see summary.md]\n"
    (out / "summary-tail.md").write_text(tail)
    print(f"[eval-long] done in {time.time() - t_start:.0f}s: {out / 'summary.md'}"
          + (f"; peak allocated {summary['peak_allocated_gib']:.2f} GiB" if cuda else ""), flush=True)
    return summary


if __name__ == "__main__":
    main()
