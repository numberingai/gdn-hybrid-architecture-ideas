"""Long-context evaluation that is not only needles. Two of its four parts live here (the multi-fact task is
ced/eval/probe_gen_multifact.py, the GDN state diagnostics ced/eval/gdn_diag.py; the CLI is scripts/eval_long.py):

1. bpb by position (`score_position_bpb`): every held-out long document is scored whole, ONE document per row at its
   full length (no packing across documents; a document longer than max_len tokens is cut to its first max_len + 1
   tokens), and the scored targets are bucketed by their position in the document.
2. assistant bpb on whole agent trajectories (`score_agent_windows`): each held-out trajectory longer than the training
   row, unsplit, scored on its assistant tokens with the context window cut to W.

Conventions, used by both parts:

- Position. A document of n tokens gives m = n - 1 (input, target) pairs. The pair at input position t (0-based in the
  document) reads tokens 0..t and predicts token t + 1; its bucket is the one whose [lo, hi) holds t. Default edges
  0 / 2,048 / 8,192 / 16,384 / 32,768 / 65,536 ("0-2k", "2k-8k", "8k-16k", "16k-32k", "32k-64k").
- Rows. One document per row, as document 0 starting at row position 0; the row length is the smallest of
  base x {1, 2, 4, 8} (8k / 16k / 32k / 64k) that holds the m inputs, beyond that the next multiple of base, so a compiled
  kernel sees a handful of shapes; the rest of the row is an inert padding document (id 1, token 0, never scored).
- Models. The control is scored by its single head at every position. CED v2 runs its decoder at every valid target
  of the row (capacity = the row length), the regime these checkpoints trained in and the one
  ced.eval.span_bpb.evaluate uses (decoder_at "every"); "scored" runs it only at the scored targets (the restricted
  regime of stage 3: an agent row's assistant targets) and "scored_sinks" at those plus the document's first
  decoder.n_sinks positions. "scored_sinks_replay" adds the replay positions before each span start
  (`replay_positions`; a span start is a scored position whose previous position in the same document is not scored):
  with an integer window W the min(W, available) positions immediately before it; with "turn" the whole last turn
  before it, capped at its last turn_max positions ("turn<N>": capped at N; "turn128" is the candidate regime of
  2026-10-11; the names and the rule for every kind of scored unit are in ced/eval/replay_window.py). The turn of an
  agent trajectory is the last message before the assistant turn, read from the chat format's <|im_start|> tokens
  (`IM_START_ID`): the replay starts at the second-to-last <|im_start|> at or before the span start (the last one
  opens the assistant turn itself, whose header tokens before the span start are replayed too), or at the row's first
  position when the window cut that message's start off. Replay and sink positions run the decoder and write decoder
  K/V; their outputs are not scored, so the scored positions and their targets are those of every other regime. The
  head is the reference head in chunks (ced/train/chunked_loss.token_ce, float32 or wider, no full logits).
- The <= 8k value of a target ("ref"): the same target scored with a short context, as a fresh row. For the bpb part
  (`ref_window` W = 8,192, `ref_stride` S = 4,096): a target at position t >= W is scored again in a row of W inputs that
  starts at document position c - (W - S), c = S * floor(t / S), so it sees between W - S + 1 and W tokens of context
  (4,097 to 8,192); a target at t < W has that much context or less in the whole-document row already, so its ref
  value IS its whole-document value (no second forward). delta = bpb(full context) - bpb(ref) on the same targets:
  negative when the context beyond the last 4k-8k tokens helps, positive when the model is worse with the long context
  than without it. For the agent part the ref is the window W = 8,192 (`score_agent_windows`).
- Standard errors: the ratio estimator sum(nats) / sum(bytes) over clusters (a text document; an agent trajectory's
  environment), delta method, as ced/eval/span_bpb.py; differences are paired per cluster.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ced.eval.probes import bucket_length, decoder_sinks, head_weight, model_device, unwrap
from ced.eval.replay_window import REPLAY_TURN, REPLAY_TURN_MAX, is_turn, parse_replay_window  # noqa: F401  (re-exported)

LN2 = math.log(2.0)
BASE_LEN = 8192  # the 60M runs' training row length
SCORED_ARCHES = ("control", "ced_v2")
DECODER_AT = ("every", "scored", "scored_sinks", "scored_sinks_replay")
RESTRICTED = ("scored", "scored_sinks", "scored_sinks_replay")
# REPLAY_TURN ("turn": the whole last turn before a span start) and REPLAY_TURN_MAX (512, its default cap in positions)
# are ced/eval/replay_window.py's, imported above
# <|im_start|> of the chat format: the tokenizers are trained with tools/data/chat_format.py SPECIAL_TOKENS at ids
# 0..31 in that order (tools/data/train_tokenizer.py), and control tokens enter a tokenized conversation only from the
# template, never from message text (chat_format's docstring), so every <|im_start|> token opens a message
IM_START_ID = 1
TURN_BINS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)  # histogram edges of turn lengths (tokens)
TURN_QUANTILES = (10, 25, 50, 75, 90, 95, 99)


def default_edges(base: int = BASE_LEN) -> tuple[int, ...]:
    """0, base / 4, base, 2 base, 4 base, 8 base: 0 / 2k / 8k / 16k / 32k / 64k at base 8,192."""
    return (0, base // 4, base, 2 * base, 4 * base, 8 * base)


def length_label(n: int) -> str:
    """8192 -> "8k", 2048 -> "2k", 0 -> "0"; lengths that are not a multiple of 1,024 are printed as they are."""
    if n == 0:
        return "0"
    return f"{n // 1024}k" if n % 1024 == 0 else str(n)


def bucket_names(edges) -> list[str]:
    return [f"{length_label(a)}-{length_label(b)}" for a, b in zip(edges[:-1], edges[1:])]


def bucket_index(positions: torch.Tensor, edges) -> torch.Tensor:
    """The bucket of each position: i with edges[i] <= p < edges[i + 1]; -1 outside [edges[0], edges[-1])."""
    e = torch.as_tensor(list(edges), dtype=torch.long)
    assert bool((e[1:] > e[:-1]).all()), "bucket edges must increase"
    i = torch.bucketize(positions.long(), e, right=True) - 1
    return torch.where((i >= 0) & (i < len(e) - 1), i, torch.full_like(i, -1))


def bucket_sums(values: torch.Tensor, positions: torch.Tensor, edges, weight: torch.Tensor | None = None) -> torch.Tensor:
    """(len(edges) - 1,) float64: values summed per bucket of positions (positions outside the edges are left out);
    with weight (bool or 0/1), only where it is set."""
    i = bucket_index(positions, edges)
    keep = i >= 0
    if weight is not None:
        keep = keep & weight.bool()
    out = torch.zeros(len(edges) - 1, dtype=torch.float64)
    return out.index_add_(0, i[keep], values.double()[keep])


# --------------------------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------------------------


@dataclass
class LongDocs:
    """Whole tokenized documents, never packed. Document i: ids[offsets[i]:offsets[i + 1]].

    kind "text": documents ending in <|endoftext|> (web, code); kind "chat": agent trajectories with `assistant`
    (1 where the token is generated by the assistant: message content and its end-of-turn token, the `generated`
    flags of the training shards). meta, per document: "id", "source", "group" (int cluster id: the document itself
    for text, the environment for a trajectory), "n_tokens" (the WHOLE document's token count) and "stored": "all",
    or "head" / "tail" when the builder kept only the first / the last tokens (text: the first max_len + 1; chat:
    the last max window + 1). info: tokenizer, tokenizer_sha256, the builder's limits and report."""

    name: str
    kind: str
    ids: torch.Tensor  # (N,) int32
    byte_lens: torch.Tensor  # (N,) uint8
    offsets: torch.Tensor  # (n_docs + 1,) int64
    meta: list[dict]
    assistant: torch.Tensor | None = None  # (N,) uint8 (chat)
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.kind in ("text", "chat"), self.kind
        assert len(self.meta) == len(self.offsets) - 1 and int(self.offsets[-1]) == len(self.ids) == len(self.byte_lens)
        assert (self.assistant is not None) == (self.kind == "chat"), "chat sets carry the assistant flags, text sets none"
        assert self.assistant is None or len(self.assistant) == len(self.ids)

    def __len__(self) -> int:
        return len(self.meta)

    def length(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])

    def doc(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """(ids long, byte_lens long, assistant bool or None) of document i, as stored."""
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return (self.ids[a:b].long(), self.byte_lens[a:b].long(),
                None if self.assistant is None else self.assistant[a:b].bool())

    @classmethod
    def from_docs(cls, name: str, kind: str, docs: list[tuple], meta: list[dict], info: dict | None = None) -> "LongDocs":
        """docs: (ids, byte_lens) or (ids, byte_lens, assistant) per document, any integer sequences."""
        ids = [np.asarray(d[0], dtype=np.int32) for d in docs]
        lens = np.array([len(x) for x in ids], dtype=np.int64)
        cat = lambda parts, dtype: torch.from_numpy(np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype))  # noqa: E731
        return cls(name=name, kind=kind, ids=cat(ids, np.int32),
                   byte_lens=cat([np.asarray(d[1]) for d in docs], np.uint8),
                   offsets=torch.from_numpy(np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)), meta=list(meta),
                   assistant=cat([np.asarray(d[2]) for d in docs], np.uint8) if kind == "chat" else None, info=dict(info or {}))

    def subset(self, keep: list[int]) -> "LongDocs":
        docs = [self.doc(i) for i in keep]
        return LongDocs.from_docs(self.name, self.kind, [tuple(x.numpy() for x in d if x is not None) for d in docs],
                                  [self.meta[i] for i in keep], self.info)

    def save(self, path: str | Path) -> None:
        torch.save({"name": self.name, "kind": self.kind, "ids": self.ids, "byte_lens": self.byte_lens, "offsets": self.offsets,
                    "meta": self.meta, "assistant": self.assistant, "info": self.info}, path)

    @classmethod
    def load(cls, path: str | Path) -> "LongDocs":
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


def groups_of(docs: LongDocs) -> list[int]:
    """Cluster id per document ("group" in its meta; a document without one is its own cluster)."""
    return [int(m.get("group", -1 - i)) for i, m in enumerate(docs.meta)]


# --------------------------------------------------------------------------------------------
# rows and the model forward
# --------------------------------------------------------------------------------------------


def row_length(n_inputs: int, base: int = BASE_LEN) -> int:
    """The row length for n_inputs real positions: the smallest of base x {1, 2, 4, 8} that holds them, beyond that the
    next multiple of base (ced.eval.probes.bucket_length without its spare padding position: a full row is fine)."""
    assert n_inputs >= 1
    return bucket_length(n_inputs - 1, base)


def make_row(ids: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(tokens, targets, doc_ids, valid), each (length,), for a document piece ids (m + 1,): the m inputs ids[:-1] at
    row positions 0..m-1 as document 0, their targets ids[1:], and the padding document (id 1, token 0) behind them."""
    m = len(ids) - 1
    assert 1 <= m <= length, (m, length)
    tokens = torch.zeros(length, dtype=torch.long)
    targets = torch.zeros(length, dtype=torch.long)
    doc_ids = torch.ones(length, dtype=torch.long)
    valid = torch.zeros(length, dtype=torch.bool)
    tokens[:m], targets[:m], doc_ids[:m], valid[:m] = ids[:-1], ids[1:], 0, True
    return tokens, targets, doc_ids, valid


@torch.no_grad()
def position_ce(model: torch.nn.Module, tokens: torch.Tensor, targets: torch.Tensor, doc_ids: torch.Tensor,
                valid: torch.Tensor, run: torch.Tensor | None = None, T: int | None = None,
                autocast_dtype: torch.dtype | None = None, fused_varlen: bool = False) -> torch.Tensor:
    """Cross-entropy in nats of every position's target, (B, L) float64 on the CPU, from the model's hidden states and
    the reference head in chunks (ced/train/chunked_loss.token_ce; float32 or wider, autocast off in the head).

    control: its single head at every position. CED v2: the decoder (main head) at the positions `run` (B, L) bool
    (default: valid, the decoder at every valid target, as ced.eval.span_bpb.evaluate scores it), capacity L, so no
    position is dropped; positions that do not run the decoder hold NaN (the aux head is not scored: a caller must
    only read positions inside `run`). With decoder self-KV, `run` changes the scores (the own keys in the context).
    T: CED v2's loop count (default: the config's t_eval). autocast_dtype: the body's precision (the run's training
    precision on CUDA). fused_varlen: pass the host-side document boundaries so the fused GDN kernel runs."""
    from ced.model.capacity import gather_slots, scatter_slots
    from ced.train.chunked_loss import token_ce

    inner = unwrap(model)
    arch = inner.cfg.arch
    if arch not in SCORED_ARCHES:
        raise ValueError(f"the long-context evaluation scores control (a) and CED v2, not arch {arch!r}")
    device = model_device(model)
    B, L = tokens.shape
    varlen = None
    if fused_varlen:
        from ced.train.varlen import to_device, varlen_for_batch

        varlen = to_device(varlen_for_batch(doc_ids, None, None), device)
    from ced.model.cp_head import cp_head_of, head_token_ce

    cp = cp_head_of(inner) is not None  # a Copy-Paste head: its next-token distribution (cp_head.head_token_ce)
    weight = None if cp else head_weight(model)
    tok, tgt, doc = tokens.to(device), targets.to(device), doc_ids.to(device)
    with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        if arch == "control":
            out = model(tok, doc, None, return_hidden=True, varlen=varlen)
            if cp:
                ce = head_token_ce(inner, out.hidden_aux, tgt, tok, doc, pointer=out.cp_pointer)[0].double()
            else:
                ce = token_ce(out.hidden_aux, weight, tgt)[0].double()
        else:
            where = (valid if run is None else run).to(device)
            T = inner.cfg.loop.t_eval if T is None else T
            out = model(tok, doc, where, T=T, capacity=L, return_hidden=True, varlen=varlen, aux=False)
            assert int(out.n_dropped.sum()) == 0 and torch.equal(out.kept, where), "the decoder did not run at exactly the run positions"
            dec = token_ce(out.hidden_dec, weight, gather_slots(tgt, out.slot_index, fill=0))[0].double()
            ce = scatter_slots(torch.full((B, L), float("nan"), dtype=torch.float64, device=device), out.slot_index, dec)
    return ce.cpu()


def span_starts(scored: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
    """(B, L) bool: the span starts, the scored positions whose previous position in the same document is not scored
    (a scored position that is its document's first, or the row's first, is one: nothing precedes it)."""
    prev = torch.zeros_like(scored)
    prev[:, 1:] = scored[:, :-1] & (doc_ids[:, 1:] == doc_ids[:, :-1])
    return scored & ~prev


def document_positions(doc_ids: torch.Tensor) -> torch.Tensor:
    """(B, L) long: each position's index within its document (0 at a document's first position of the row)."""
    L = doc_ids.shape[1]
    j = torch.arange(L)[None].expand_as(doc_ids)
    first = torch.ones_like(doc_ids, dtype=torch.bool)
    first[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    return j - torch.where(first, j, torch.zeros_like(j)).cummax(dim=1).values


def replay_positions(scored: torch.Tensor, doc_ids: torch.Tensor, window, tokens: torch.Tensor | None = None,
                     im_start: int | None = None, turn_max: int | None = None) -> torch.Tensor:
    """(B, L) bool: the replay positions of decoder_at "scored_sinks_replay" (the module docstring), for (B, L) CPU
    tensors. With s the nearest span start after position p in p's document (`span_starts`), p is a replay position
    when

        window an integer W >= 1:  s - p <= W                       the min(W, available) positions before s
        window "turn":             s - p <= turn_max, and at most one <|im_start|> token (id im_start, in `tokens`)
                                   lies at the positions p + 1 .. s (turn_max None: REPLAY_TURN_MAX, 512)
        window "turn<N>":          the same with turn_max = N ("turn128": the whole last turn, at most the 128
                                   positions nearest s; ced/eval/replay_window.py)

    The turn rule in words: the last <|im_start|> at or before s opens the assistant turn that s belongs to (s is the
    last token of its header, "<|im_start|>assistant\\n", the position that predicts the turn's first token), and the
    one before it opens the last message before that turn (a user message or a tool result; any message). The replay
    runs from that message's <|im_start|> through the assistant header's tokens before s: the whole last message with
    its header, capped at the last turn_max positions before s. When the row holds fewer than two <|im_start|> at
    or before s (the window cut the message's start off, or the row starts inside the assistant turn), the replay
    starts at the document's first position of the row. `available` never leaves the document: positions of another
    document, and the padding document, are never replayed. A replay position may be a scored position of an
    earlier span (a short message between two assistant turns): it runs the decoder either way. Positions are not
    filtered by `scored` here; the caller's decoder mask is scored | sinks | this."""
    B, L = scored.shape
    starts = span_starts(scored, doc_ids)
    j = torch.arange(L)[None].expand(B, L)
    at_or_after = torch.where(starts, j, torch.full_like(j, L)).flip(1).cummin(dim=1).values.flip(1)
    nxt = torch.full_like(j, L)  # the nearest span start strictly after p; L: none
    nxt[:, :-1] = at_or_after[:, 1:]
    s = nxt.clamp(max=L - 1)
    # p and s in one document: compared by their stretch of the row (the count of document changes up to a position),
    # as span_starts and document_positions read documents, not by the ids
    stretch = torch.zeros_like(j)
    stretch[:, 1:] = (doc_ids[:, 1:] != doc_ids[:, :-1]).long().cumsum(dim=1)
    ok = (nxt < L) & (stretch.gather(1, s) == stretch)
    dist = nxt - j
    if is_turn(window):
        _, turn_max = parse_replay_window(window, turn_max)  # "turn": the given cap or 512; "turn<N>": N
        assert tokens is not None and im_start is not None, "the turn replay reads the <|im_start|> tokens"
        assert turn_max >= 1, turn_max
        count = (tokens == im_start).long().cumsum(dim=1)  # <|im_start|> tokens at positions <= p
        return ok & (dist <= turn_max) & (count.gather(1, s) - count <= 1)
    assert not isinstance(window, bool) and isinstance(window, int) and window >= 1, f"replay window {window!r}"
    return ok & (dist <= window)


def turn_spans(tokens: torch.Tensor, scored: torch.Tensor, im_start: int) -> dict:
    """The assistant turns of ONE document's row piece (tokens, scored: (m,), the row's real positions): for every span
    start s > 0, in order,

        "start"   s
        "turn"    s - t, t the position of the second-to-last <|im_start|> at or before s (0 when there are fewer
                  than two): the positions the "turn" replay runs before s, uncapped
        "cut"     True when there are fewer than two (the last message's start is outside the row)
        "header"  s - (the position of the last <|im_start|> at or before s), -1 without one: 2 in the chat format
                  under our tokenizers ("<|im_start|>", "assistant", "\\n": s is the newline)

    as lists. A span start at position 0 has nothing before it and is left out."""
    m = len(tokens)
    start = scored.clone()
    start[1:] &= ~scored[:-1]
    start[0] = False
    s = start.nonzero().squeeze(1)
    im_pos = (tokens == im_start).nonzero().squeeze(1)
    k = torch.searchsorted(im_pos, s, right=True)  # <|im_start|> tokens at positions <= s
    last = torch.where(k >= 1, im_pos[(k - 1).clamp(min=0)] if len(im_pos) else torch.zeros_like(s), torch.full_like(s, -1))
    before = torch.where(k >= 2, im_pos[(k - 2).clamp(min=0)] if len(im_pos) else torch.zeros_like(s), torch.zeros_like(s))
    assert m == len(scored)
    return {"start": s.tolist(), "turn": (s - before).tolist(), "cut": (k < 2).tolist(),
            "header": torch.where(k >= 1, s - last, torch.full_like(s, -1)).tolist()}


def length_stats(values) -> dict:
    """Counts of token lengths: n, mean, min, max, the TURN_QUANTILES percentiles, the histogram over TURN_BINS
    ("<=1", "<=2", ..., ">4096": the smallest edge that holds the value) and the share at or below 32 / 128 / 512."""
    x = np.asarray(list(values), dtype=np.int64)
    if len(x) == 0:
        return {"n": 0}
    edges = np.asarray(TURN_BINS)
    which = np.searchsorted(edges, x, side="left")
    hist = {(f"<={e}" if i < len(edges) else f">{edges[-1]}"): int((which == i).sum())
            for i, e in enumerate(list(edges) + [None])}
    return {"n": int(len(x)), "mean": float(x.mean()), "min": int(x.min()), "max": int(x.max()),
            "quantiles": {str(q): float(np.percentile(x, q)) for q in TURN_QUANTILES},
            "hist": hist, "share_le": {str(w): float((x <= w).mean()) for w in (32, 128, 512)}}


class RowScorer:
    """How one model scores rows: position_ce with the model's settings fixed, the decoder regime, and the time per row
    shape. Called as scorer(tokens, targets, doc_ids, valid, scored) with (B, L) CPU tensors; returns the (B, L) float64
    cross-entropy (NaN where a CED v2 position did not run the decoder).

    decoder_at (CED v2 only; the control ignores it): "every" runs the decoder at every valid target; "scored" at the
    scored targets only; "scored_sinks" at those plus each document's first n_sinks positions (decoder.n_sinks: the
    local layers' sinks, as scripts/eval_probes.py --decoder-at answers_sinks); "scored_sinks_replay" at those plus
    the replay positions before each span start (`replay_positions`): replay_window an integer W >= 1, "turn"
    (with replay_turn_max, default 512) or "turn<N>" (the cap in the name, e.g. "turn128"), the last two with
    im_start, the <|im_start|> token id. The decoder capacity is the row length in every
    regime (position_ce), so no position is ever dropped; Scalable-Softmax key counts in the full self-KV layers
    are the model's own (ced/model/ced_v2.py slot_key_counts counts the valid slots, whichever positions they are).
    `run_mask` gives the regime's decoder positions of a batch without scoring it."""

    def __init__(self, model: torch.nn.Module, T: int | None = None, autocast_dtype: torch.dtype | None = None,
                 fused_varlen: bool = False, decoder_at: str = "every", replay_window=None,
                 replay_turn_max: int | None = None, im_start: int | None = None):
        if decoder_at not in DECODER_AT:
            raise ValueError(f"decoder_at {decoder_at!r}: one of {DECODER_AT}")
        self.model, self.T, self.autocast_dtype, self.fused_varlen = model, T, autocast_dtype, fused_varlen
        self.decoder = unwrap(model).cfg.arch == "ced_v2"
        self.decoder_at = decoder_at if self.decoder else None
        self.n_sinks = decoder_sinks(model) if self.decoder_at in ("scored_sinks", "scored_sinks_replay") else 0
        replay = self.decoder and decoder_at == "scored_sinks_replay"  # the control ignores the regime and its window
        if replay:
            turn = is_turn(replay_window)
            if not turn and (isinstance(replay_window, bool) or not isinstance(replay_window, int) or replay_window < 1):
                raise ValueError(f"replay_window {replay_window!r}: decoder_at 'scored_sinks_replay' needs an integer >= 1, "
                                 f"{REPLAY_TURN!r} or 'turn<N>'")
            if turn:
                try:  # "turn": the given cap or 512; "turn<N>": N (a given cap must agree)
                    replay_window, replay_turn_max = parse_replay_window(replay_window, replay_turn_max)
                except ValueError as e:
                    raise ValueError(f"replay_window: {e}") from None
                if im_start is None:
                    raise ValueError("replay_window 'turn' needs im_start (the <|im_start|> token id) and replay_turn_max >= 1")
        elif replay_window is not None and decoder_at != "scored_sinks_replay":
            raise ValueError(f"replay_window is for decoder_at 'scored_sinks_replay', not {decoder_at!r}")
        self.replay_window = replay_window if replay else None  # an integer, or "turn" with its cap in replay_turn_max
        self.replay_turn_max, self.im_start = (REPLAY_TURN_MAX if replay_turn_max is None else replay_turn_max), im_start
        self.timing: dict[str, dict] = {}

    def run_mask(self, tokens, doc_ids, valid, scored=None) -> torch.Tensor | None:
        """(B, L) bool: the positions at which the decoder runs for these rows; None for the control (no decoder).
        "every": the valid positions."""
        if not self.decoder:
            return None
        if self.decoder_at == "every":
            return valid.clone()
        assert scored is not None, f"decoder_at {self.decoder_at} needs the scored positions"
        run = scored & valid
        if self.n_sinks:  # each document's first n_sinks positions (the padding document has none: it is not valid)
            run = run | (valid & (document_positions(doc_ids) < self.n_sinks))
        if self.replay_window is not None:
            run = run | (valid & replay_positions(scored & valid, doc_ids, self.replay_window, tokens=tokens,
                                                  im_start=self.im_start, turn_max=self.replay_turn_max))
        return run

    def __call__(self, tokens, targets, doc_ids, valid, scored=None) -> torch.Tensor:
        run = self.run_mask(tokens, doc_ids, valid, scored) if self.decoder_at in RESTRICTED else None
        device = model_device(self.model)
        t0 = time.perf_counter()
        ce = position_ce(self.model, tokens, targets, doc_ids, valid, run=run, T=self.T, autocast_dtype=self.autocast_dtype,
                         fused_varlen=self.fused_varlen)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        B, L = tokens.shape
        rec = self.timing.setdefault(f"{L}x{B}", {"length": L, "batch": B, "forwards": 0, "tokens": 0, "seconds": 0.0,
                                                  "first_seconds": None})
        if rec["first_seconds"] is None:
            rec["first_seconds"] = dt  # includes the compile of this shape
        rec["forwards"] += 1
        rec["tokens"] += B * L
        rec["seconds"] += dt
        return ce

    def timing_report(self) -> dict:
        out = {}
        for key, r in self.timing.items():
            rest = r["seconds"] - r["first_seconds"]
            n = r["forwards"] - 1
            out[key] = {**r, "tokens_per_s_after_first": (n * r["length"] * r["batch"] / rest) if n > 0 and rest > 0 else None}
        return out


# --------------------------------------------------------------------------------------------
# aggregates
# --------------------------------------------------------------------------------------------


def ratio_se(y, x, groups=None) -> tuple[float, float, int]:
    """(sum(y) / sum(x), its delta-method standard error, units) over units: the entries, or the clusters when groups
    is given (y and x summed per cluster first). Units with x = 0 are left out; NaN without units, SE NaN with one."""
    y, x = np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)
    if groups is not None:
        _, inv = np.unique(np.asarray(groups), return_inverse=True)
        y, x = np.bincount(inv, weights=y), np.bincount(inv, weights=x)
    keep = x > 0
    y, x = y[keep], x[keep]
    n = int(keep.sum())
    if n == 0:
        return float("nan"), float("nan"), 0
    est = float(y.sum() / x.sum())
    if n < 2:
        return est, float("nan"), n
    resid = y - est * x
    return est, math.sqrt(float((resid ** 2).sum()) / (n - 1) / n) / float(x.mean()), n


def _bpb(y, x, groups) -> tuple[float | None, float | None, int]:
    est, se, n = ratio_se(y, x, groups)
    f = lambda v: None if v != v else v / LN2  # noqa: E731
    return f(est), f(se), n


def bucket_table(records: list[dict], edges, groups: list[int], keep: list[bool] | None = None, has_ref: bool = True) -> list[dict]:
    """Per bucket over the documents (records: per-bucket lists "tokens", "bytes", "nats", "ref_nats"): documents and
    clusters with tokens there, tokens, bytes, bpb +- SE, the ref bpb and delta = bpb - ref bpb +- its paired SE. keep:
    only these documents (a cohort). has_ref False (no reference pass was run: the records' "ref_nats" are then the
    whole-document values): the ref bpb and the delta are None, never a measured-looking 0."""
    rows = []
    idx = [i for i in range(len(records)) if keep is None or keep[i]]
    for b, name in enumerate(bucket_names(edges)):
        sel = [i for i in idx if records[i]["tokens"][b] > 0]
        g = [groups[i] for i in sel]
        nats = [records[i]["nats"][b] for i in sel]
        ref = [records[i]["ref_nats"][b] for i in sel]
        nbytes = [records[i]["bytes"][b] for i in sel]
        bpb, se, n = _bpb(nats, nbytes, g)
        rbpb, rse, _ = _bpb(ref, nbytes, g)
        d, dse, _ = _bpb([a - r for a, r in zip(nats, ref)], nbytes, g)
        if not has_ref:
            rbpb = rse = d = dse = None
        rows.append({"name": name, "lo": int(edges[b]), "hi": int(edges[b + 1]), "docs": len(sel), "clusters": n,
                     "tokens": int(sum(records[i]["tokens"][b] for i in sel)), "bytes": int(sum(nbytes)),
                     "bpb": bpb, "bpb_se": se, "ref_bpb": rbpb, "ref_bpb_se": rse, "delta": d, "delta_se": dse})
    return rows


# --------------------------------------------------------------------------------------------
# part 1: bpb by position on whole long documents
# --------------------------------------------------------------------------------------------


def ref_chunks(m: int, window: int, stride: int) -> list[tuple[int, int, int]]:
    """The ref rows of a document with m inputs: (row start, scored start, scored end) in document positions, one per
    stride chunk [c, c + stride) with c >= window (targets below `window` need no ref row). The row holds the inputs
    [c - (window - stride), min(c + stride, m)) and scores the targets at [c, min(c + stride, m))."""
    assert 0 < stride <= window and window % stride == 0, (window, stride)
    return [(c - (window - stride), c, min(c + stride, m)) for c in range(window, m, stride)]


def score_position_bpb(scorer, docs: LongDocs, max_len: int | None = None, edges=None, base: int = BASE_LEN,
                       ref_window: int | None = None, ref_stride: int | None = None, ref_batch: int = 4,
                       max_docs: int | None = None, progress=None) -> dict:
    """bpb by position on whole documents (the module docstring). scorer(tokens, targets, doc_ids, valid, scored) -> the
    (B, L) cross-entropy (a RowScorer). Every document is cut to its first max_len + 1 tokens (max_len scored targets,
    positions 0..max_len - 1; default 8 x base) and scored in one row of its own. ref_window: the reference context
    (default base; ref_stride default half of it); 0: no ref pass (ref = the whole-document value everywhere, delta
    0). ref_batch: ref rows per forward (the last batch is filled with copies of its last row).

    Returns {"set", "edges", "buckets" (names), "max_len", "ref", "docs": per document {id, source, group, n_tokens,
    n_scored, row_len, tokens / bytes / nats / ref_nats per bucket}, "all": the bucket table over every document,
    "cohorts": {"<edge label>": the table over the documents that reach that edge (n_scored >= edge)}, "seconds"}."""
    assert docs.kind == "text", "position bpb is for text documents (chat trajectories: score_agent_windows)"
    edges = tuple(default_edges(base) if edges is None else edges)
    max_len = 8 * base if max_len is None else max_len
    ref_window = base if ref_window is None else ref_window
    ref_stride = (ref_window // 2 if ref_stride is None else ref_stride) if ref_window else None
    n_docs = len(docs) if max_docs is None else min(max_docs, len(docs))
    groups = groups_of(docs)[:n_docs]
    records = []
    t0 = time.perf_counter()
    for i in range(n_docs):
        ids, nbytes, _ = docs.doc(i)
        ids, nbytes = ids[: max_len + 1], nbytes[: max_len + 1]
        m = len(ids) - 1
        if m < 1:
            records.append({**_doc_head(docs, i, groups[i]), "n_scored": 0, "row_len": 0,
                            **{k: [0] * (len(edges) - 1) for k in ("tokens", "bytes", "nats", "ref_nats")}})
            continue
        L = row_length(m, base)
        tokens, targets, doc_ids, valid = make_row(ids, L)
        ce = scorer(tokens[None], targets[None], doc_ids[None], valid[None], valid[None])[0, :m]
        assert bool(torch.isfinite(ce).all()), f"{docs.meta[i].get('id')}: non-finite cross-entropy"
        ref = ce.clone()
        if ref_window:
            chunks = ref_chunks(m, ref_window, ref_stride)
            for a in range(0, len(chunks), ref_batch):
                part = chunks[a : a + ref_batch]
                rows = [make_row(ids[s : e + 1], ref_window) for s, _, e in part]
                rows += [rows[-1]] * (ref_batch - len(rows))  # one batch shape
                stack = [torch.stack([r[j] for r in rows]) for j in range(4)]
                out = scorer(*stack, stack[3])
                for b, (s, c, e) in enumerate(part):
                    ref[c:e] = out[b, c - s : e - s]
            assert bool(torch.isfinite(ref).all())
        pos = torch.arange(m)
        tb = nbytes[1:]
        records.append({**_doc_head(docs, i, groups[i]), "n_scored": m, "row_len": L,
                        "tokens": [int(x) for x in bucket_sums(torch.ones(m), pos, edges)],
                        "bytes": [int(x) for x in bucket_sums(tb, pos, edges)],
                        "nats": bucket_sums(ce, pos, edges).tolist(), "ref_nats": bucket_sums(ref, pos, edges).tolist()})
        if progress:
            progress(i + 1, n_docs)
    cohorts = {}
    for e in edges[1:]:
        if e > base:  # the documents that reach this edge: the same documents in every bucket up to it
            cohorts[length_label(e)] = bucket_table(records, edges, groups, keep=[r["n_scored"] >= e for r in records],
                                                    has_ref=bool(ref_window))
            cohorts[length_label(e)] = [row for row in cohorts[length_label(e)] if row["hi"] <= e]
    return {"set": docs.name, "edges": list(edges), "buckets": bucket_names(edges), "max_len": max_len, "base": base,
            "ref": {"window": ref_window, "stride": ref_stride} if ref_window else None, "docs": records,
            "all": bucket_table(records, edges, groups, has_ref=bool(ref_window)), "cohorts": cohorts,
            "seconds": time.perf_counter() - t0}


def _doc_head(docs: LongDocs, i: int, group: int) -> dict:
    m = docs.meta[i]
    return {"id": m.get("id", str(i)), "source": m.get("source"), "group": group, "n_tokens": int(m.get("n_tokens", docs.length(i)))}


# --------------------------------------------------------------------------------------------
# part 2: assistant bpb on whole agent trajectories, by context window
# --------------------------------------------------------------------------------------------


def scored_fingerprint(positions: torch.Tensor, targets: torch.Tensor) -> str:
    """A fingerprint of a row's scored positions and their target tokens: two passes that scored the same positions
    against the same targets (any decoder regime) carry the same one, so their sums are paired position for position."""
    h = hashlib.blake2b(digest_size=8)
    h.update(positions.numpy().astype(np.int64).tobytes())
    h.update(targets.numpy().astype(np.int64).tobytes())
    return h.hexdigest()


def score_agent_windows(scorer, docs: LongDocs, windows=None, ref_window: int | None = None,
                        tail: int | None = None, min_tokens: int | None = None, base: int = BASE_LEN,
                        max_docs: int | None = None, progress=None, im_start: int | None = None) -> dict:
    """Assistant bpb on whole, unsplit agent trajectories with the context window cut to W.

    For a trajectory of n tokens and a window W the row holds m = min(W, n - 1) inputs: the trajectory's tokens
    n - 1 - m .. n - 2, with targets n - m .. n - 1. So when n - 1 > W the model reads exactly the last W tokens before
    the final target (the cut is at a token, it may fall inside a message; the row starts as a fresh document: position
    0, empty recurrent state), and a shorter trajectory is scored whole. Scored: the targets that are assistant
    tokens (the set's assistant flags: message content and its end-of-turn token). The windows are `windows` (default
    2 / 4 / 8 x base: 16k / 32k / 64k) plus the ref window (default base, 8,192: the training row length); windows that
    give a trajectory the same inputs share one forward. Trajectories of fewer than min_tokens tokens (default base +
    1: "longer than 8k") are left out.

    Two numbers per window:
    - window bpb: every scored target of the row, over all trajectories ("filled": the trajectories with n - 1 >= W;
      the others are scored whole);
    - tail bpb: only the targets among the trajectory's last `tail` (default base / 2 = 4,096) targets, which every
      window holds, so the SAME assistant tokens are predicted with at most 8k / 16k / 32k / 64k of context. Reported
      per cohort (the trajectories with n - 1 >= C, for each window C) at every window W <= C, with delta = tail bpb(W)
      - tail bpb(ref window), paired per trajectory, SE over environments.

    The first token of each assistant turn against the rest: a scored position is "first" when it is a span start
    with a position before it in the row (its previous position is not scored: in the chat format, the last token
    of "<|im_start|>assistant\\n", which predicts the turn's first token); every other scored position is "rest", a
    scored position at the row's first position included (a turn the window cut: nothing precedes it in any
    regime). first_* / tail_first_* are the sums over the first positions; the rest is the difference to the
    totals.

    Decoder positions: "run" (the positions at which the scorer's regime runs the decoder: scorer.run_mask; None for
    a scorer without one and for the control) and "extra" (those of them that are not scored: sinks and replay).
    "scored_sha": `scored_fingerprint` of the row, the same in every regime. im_start (the <|im_start|> token id;
    None: not reported): "turns", per window, the lengths of the last turn before each assistant turn
    (`turn_spans`, `length_stats`), how many the window cut, and the header lengths found (the check that the chat
    tokens are read as expected).

    Returns {"set", "windows", "ref_window", "tail", "min_tokens", "skipped_short", "docs": per trajectory {id, source,
    group, n_tokens, windows: {W: {m, filled, tokens, bytes, nats, tail_tokens, tail_bytes, tail_nats, first_tokens,
    first_bytes, first_nats, tail_first_tokens, tail_first_bytes, tail_first_nats, spans, run, extra, scored_sha}}},
    "window": {W: {...}}, "tail_by_cohort": {C: {"trajectories", "clusters", "by_window": {W: {...}}}}, "positions":
    {W: {inputs, scored, spans, first, run, extra}}, "turns": {W: {...}} or None, "seconds"}."""
    assert docs.kind == "chat", "agent windows need a chat set (assistant flags)"
    ref_window = base if ref_window is None else ref_window
    windows = sorted({int(w) for w in (windows or (2 * base, 4 * base, 8 * base))} | {int(ref_window)})
    tail = base // 2 if tail is None else tail
    min_tokens = base + 1 if min_tokens is None else min_tokens
    n_docs = len(docs) if max_docs is None else min(max_docs, len(docs))
    all_groups = groups_of(docs)
    records, groups, skipped = [], [], 0
    run_of = getattr(scorer, "run_mask", None)
    turns: dict[int, dict] = {W: {"turn": [], "cut": 0, "header": {}} for W in windows}
    t0 = time.perf_counter()
    for i in range(n_docs):
        ids, nbytes, asst = docs.doc(i)
        n_stored = len(ids)
        n = int(docs.meta[i].get("n_tokens", n_stored))
        if n < min_tokens:
            skipped += 1
            continue
        assert n_stored == n or n_stored >= max(windows) + 1, \
            f"{docs.meta[i].get('id')}: the set keeps {n_stored} of {n} tokens, fewer than the largest window needs"
        rec = {**_doc_head(docs, i, all_groups[i]), "n_tokens": n, "windows": {}}
        cache: dict[int, dict] = {}
        for W in windows:
            m = min(W, n - 1)
            if m not in cache:
                seg = slice(n_stored - 1 - m, n_stored)
                L = row_length(m, base)
                tokens, targets, doc_ids, valid = make_row(ids[seg], L)
                scored = torch.zeros(L, dtype=torch.bool)
                scored[:m] = asst[seg][1:]
                ce = scorer(tokens[None], targets[None], doc_ids[None], valid[None], scored[None])[0, :m]
                on = scored[:m]
                assert bool(torch.isfinite(ce[on]).all()), f"{docs.meta[i].get('id')}: non-finite cross-entropy"
                tb = nbytes[seg][1:]
                in_tail = on & (torch.arange(m) >= m - tail)
                zero = torch.zeros((), dtype=torch.float64)
                start = on.clone()  # span starts: the row is one document from position 0
                start[1:] &= ~on[:-1]
                first = start.clone()
                first[0] = False  # nothing precedes the row's first position: not the first token of a turn
                first_tail = first & in_tail
                run = run_of(tokens[None], doc_ids[None], valid[None], scored[None]) if run_of is not None else None
                cache[m] = {"m": m, "row_len": L, "tokens": int(on.sum()), "bytes": int(tb[on].sum()),
                            "nats": float(torch.where(on, ce, zero).sum()), "tail_tokens": int(in_tail.sum()),
                            "tail_bytes": int(tb[in_tail].sum()), "tail_nats": float(torch.where(in_tail, ce, zero).sum()),
                            "first_tokens": int(first.sum()), "first_bytes": int(tb[first].sum()),
                            "first_nats": float(torch.where(first, ce, zero).sum()),
                            "tail_first_tokens": int(first_tail.sum()), "tail_first_bytes": int(tb[first_tail].sum()),
                            "tail_first_nats": float(torch.where(first_tail, ce, zero).sum()),
                            "spans": int(start.sum()), "run": None if run is None else int(run[0, :m].sum()),
                            "extra": None if run is None else int((run[0, :m] & ~on).sum()),
                            "scored_sha": scored_fingerprint(on.nonzero().squeeze(1), targets[:m][on])}
                if im_start is not None:
                    cache[m]["_turns"] = turn_spans(tokens[:m], on, im_start)
            rec["windows"][str(W)] = {**{k: v for k, v in cache[m].items() if k != "_turns"}, "filled": n - 1 >= W}
            if im_start is not None:
                t = cache[m]["_turns"]
                turns[W]["turn"] += [x for x, c in zip(t["turn"], t["cut"]) if not c]
                turns[W]["cut"] += sum(t["cut"])
                for h in t["header"]:
                    turns[W]["header"][h] = turns[W]["header"].get(h, 0) + 1
        records.append(rec)
        groups.append(all_groups[i])
        if progress:
            progress(i + 1, n_docs)

    def get(W, key, sel):
        return [records[i]["windows"][str(W)][key] for i in sel]

    every = list(range(len(records)))
    window = {}
    for W in windows:
        bpb, se, n = _bpb(get(W, "nats", every), get(W, "bytes", every), groups)
        window[str(W)] = {"bpb": bpb, "bpb_se": se, "clusters": n, "trajectories": len(every),
                          "filled": sum(get(W, "filled", every)), "tokens": int(sum(get(W, "tokens", every))),
                          "bytes": int(sum(get(W, "bytes", every)))}
        # the first token of each assistant turn (nats per token) against the rest (bpb)
        fn, fse, _ = ratio_se(get(W, "first_nats", every), get(W, "first_tokens", every), groups)
        rest_nats = [a - b for a, b in zip(get(W, "nats", every), get(W, "first_nats", every))]
        rest_bytes = [a - b for a, b in zip(get(W, "bytes", every), get(W, "first_bytes", every))]
        rb, rse, _ = _bpb(rest_nats, rest_bytes, groups)
        window[str(W)].update(first_tokens=int(sum(get(W, "first_tokens", every))), first_nats_per_token=None if fn != fn else fn,
                              first_nats_per_token_se=None if fse != fse else fse, rest_bpb=rb, rest_bpb_se=rse)
    tail_by_cohort = {}
    for C in windows:
        sel = [i for i in every if records[i]["n_tokens"] - 1 >= C]
        g = [groups[i] for i in sel]
        ref_nats, nbytes = get(ref_window, "tail_nats", sel), get(ref_window, "tail_bytes", sel)
        by = {}
        for W in windows:
            if W > C:
                continue
            nats = get(W, "tail_nats", sel)
            bpb, se, n = _bpb(nats, nbytes, g)
            d, dse, _ = _bpb([a - r for a, r in zip(nats, ref_nats)], nbytes, g)
            by[str(W)] = {"bpb": bpb, "bpb_se": se, "delta": d, "delta_se": dse, "clusters": n}
        tail_by_cohort[str(C)] = {"trajectories": len(sel), "clusters": len(set(g)), "tokens": int(sum(get(ref_window, "tail_tokens", sel))),
                                  "bytes": int(sum(nbytes)), "by_window": by}
    positions = {}
    for W in windows:
        runs, extra = get(W, "run", every), get(W, "extra", every)
        positions[str(W)] = {"inputs": int(sum(get(W, "m", every))), "scored": int(sum(get(W, "tokens", every))),
                             "spans": int(sum(get(W, "spans", every))), "first": int(sum(get(W, "first_tokens", every))),
                             "run": None if any(r is None for r in runs) else int(sum(runs)),
                             "extra": None if any(r is None for r in extra) else int(sum(extra))}
    turn_stats = None
    if im_start is not None:
        turn_stats = {str(W): {**length_stats(t["turn"]), "cut": int(t["cut"]),
                               "header": {str(h): c for h, c in sorted(t["header"].items(), key=lambda kv: -kv[1])[:6]}}
                      for W, t in turns.items()}
    return {"set": docs.name, "windows": windows, "ref_window": ref_window, "tail": tail, "min_tokens": min_tokens, "base": base,
            "skipped_short": skipped, "docs": records, "window": window, "tail_by_cohort": tail_by_cohort,
            "positions": positions, "turns": turn_stats, "im_start": im_start, "seconds": time.perf_counter() - t0}
