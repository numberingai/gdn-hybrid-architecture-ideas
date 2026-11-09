"""Random decoder and aux-head positions: DataConfig.decoder_positions = "random" and
LossConfig.aux_positions = "random".

Per training row b of step s, with V_b the row's valid-target positions (target_valid):

    dec_mask[b] = a uniform random subset of V_b of size min(C, |V_b|)                 C = capacity
    aux_mask[b] = a uniform random subset of V_b of size min(round(f * L), |V_b|)      f = aux_fraction

each drawn without replacement from its own counter-RNG generator of (data seed, s, stream + b)
(`ced.train.data.counter_generator`). The two streams are independent, so the sets overlap at
random. Everything is a pure function of (seed, step, row) and the batch's own valid mask: resume
needs only the step, and every process computes the same masks.

Valid targets only: a position whose target is the next document's first token is never selected.
Unlike v1's decoder spans, the decoder mask may depend on the document boundaries: in v2 (and in
control (c)) a decoder position offers the same memory KV as any other position and writes nothing
else, so which positions are slots changes no other position's output (the models' selection
independence), and nothing about the future can leak through the mask. The controls only report
ce_dec there. Looped-CED (v1) is refused: its decoder positions offer their own KV to later
positions, so its mask must not depend on the row's future (the span sampler's rule). CED v2 with
decoder self_kv offers own KV too, so it takes random positions only at capacity L, where the subset
is every valid target: the positions left out are each document's last, and no other query of that
document can see them.

Host side, per batch (CPU tensors, before anything moves to the device): the training step itself
sees fixed-shape bool masks and needs no nonzero.

Answer spans (DataConfig.decoder_positions = "answer_spans"; stage 3's adaptation). The decoder runs only at
"answer" positions, as it will after pretraining; every other position offers its memory entry only. With
decoder self-KV a slot reads the own K/V of the earlier slots of its document, so which positions are slots must
not depend on the row's future: a rule such as "the last 20% of each document" would tell the first answer slot
where its document ends. The rule here (answer_span_mask) decides position t from the RNG and the document
structure at positions <= t only:

    chat documents:  the answers are the data's role positions (the packer's decoder mask there: t + 1 is an
                     assistant token of the same document, v1's rule)
    other documents: walk the row; a position is eligible to START a span when its document already has
                     answer_min_context earlier positions in the row (document position >= m). Skip a geometric
                     number of eligible positions (mean answer_gap_mean), start a span there with a length from
                     the span_* distribution (the measured assistant-turn lengths), cut at its document's end;
                     continue from the span's end
    then:            AND the valid targets (a document's last position predicts the next document)

Rows of every step draw from their own counter-RNG generator of (data seed, step, ANSWER_SPAN_STREAM + b). Keep-
first-C in the model (capacity C from DataConfig.answer_capacity_fraction) turns answers past C into context (logged
as n_dropped); keep-first-C is prefix-stable, so that is causal too.

Sinks (DataConfig.answer_sinks): CED v2 also runs the decoder at each document's first decoder.n_sinks positions of
the row (document position < n_sinks, causal), so the local layers read their sinks' own K/V as in training. Those
positions are not scored: the batch's target_valid is cleared there unless they are answers. That is
scripts/eval_probes.py's "answers_sinks" regime.

Replay window (DataConfig.answer_replay_window = W > 0, the zero-shot condition being met, "make 'answers + sinks +
replay window' the restricted regime for the adaptation run"). CED v2 also runs the decoder at the min(W, available)
positions of the same document immediately before each answer span's first position (replay_positions): before each
assistant turn of a chat document, before each drawn span elsewhere. A span start is an answer whose previous position
is not an answer of the same document, so two drawn spans that touch are one span. Replay positions write decoder K/V
like the sinks and carry no loss (target_valid cleared); a replay position that is itself an answer or a sink stays
what it is. That is scripts/eval_probes.py's "answers_sinks_replay" regime. W = 0 (the default) adds nothing: the masks
are the ones above, bit for bit.

What is causal with the replay, and what is not. With M = answers | sinks | replay (the decoder slots):

    for every SCORED position t (an answer), M at positions <= t, and so everything the slots at positions <= t
    computed, is a function of the row up to t (the RNG and the document structure at positions <= t, and t having
    a target, which a scored position has).

Why: the answers and the sinks at positions <= t are decided by the row up to t (above). A replay slot p <= t exists
because of a span start s with p < s <= p + W in p's document. If s <= t, the row up to t decides it. If s > t, let
s' <= t be the start of t's own span; p, t, s and s' are in one document. Either p >= s', and p is an answer of t's
span, or p < s' <= t < s <= p + W, and p is a replay slot of s' already. So no slot at a position <= t is there
because of anything after t. A replay slot is never scored, so nothing is predicted where the mask looked ahead.

What does NOT hold any more: prefix stability of the mask itself. At an UNSCORED position p, whether the decoder
runs depends on whether a span starts within the next W positions (the mask at p can change with the row up to
p + W, and with whether position p + W has a target; never with anything beyond). That look-ahead is the inference
regime itself: the replay runs at generation start, when the position where generation starts is known, over the
last W positions of the context, which are known too. A replay slot can tell from its own keys how far into the
replay it is; it is asked to predict nothing. Keep-first-C stays prefix-stable over M, so the property above holds
for the kept slots; with a C below the row's slots the replay slots take capacity ahead of the answers they precede
(answer_capacity_fraction 1.0, the default, drops nothing).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch

from ced.config import TrainConfig
from ced.train.data import Batch, counter_generator, draw_span_length

# Counter-RNG streams (stream + row). Training uses 1 (T), 3 and 4 (shard packer), 1000 + b and
# 2000 + b (synthetic data); eval spans 5. Rows stay below the stride, so no two (stream, row)
# pairs share a generator.
DEC_POSITION_STREAM = 100_000
AUX_POSITION_STREAM = 200_000
ANSWER_SPAN_STREAM = 300_000
_STRIDE = 100_000
# Arches that take answer spans: CED v2 (the stage 3 model), and the controls, whose single head trains on every
# valid target anyway and only reports ce_dec at the answers.
ANSWER_SPAN_ARCHES = ("ced_v2", "control")


@dataclass(frozen=True)
class AnswerSpanRule:
    """The non-chat part of the answer-span rule (module docstring): eligible starts at document position
    >= min_context; geometric gaps of mean gap_mean eligible positions; span lengths from draw_span_length."""

    min_context: int
    gap_mean: float
    distribution: str
    median: float
    sigma: float
    span_min: int
    span_max: int

    def __post_init__(self):
        if self.min_context < 0:
            raise ValueError(f"data.answer_min_context {self.min_context} < 0")
        if not (self.gap_mean >= 0 and math.isfinite(self.gap_mean)):
            raise ValueError(f"data.answer_gap_mean {self.gap_mean}: a finite mean >= 0")
        if not 1 <= self.span_min <= self.span_max:
            raise ValueError(f"data.span_min / span_max {self.span_min} / {self.span_max}: need 1 <= min <= max")

    def gap(self, g: torch.Generator) -> int:
        """Eligible positions skipped before the next span: geometric on {0, 1, ...} with mean gap_mean (inverse CDF of
        one float64 uniform; gap_mean 0 gives 0 and draws nothing)."""
        if self.gap_mean == 0:
            return 0
        u = float(torch.rand((), generator=g, dtype=torch.float64))
        p = 1.0 / (self.gap_mean + 1.0)  # P(gap = k) = p (1 - p)^k, mean (1 - p) / p = gap_mean
        return int(math.floor(math.log1p(-u) / math.log1p(-p)))

    def length(self, g: torch.Generator) -> int:
        return draw_span_length(g, self.span_min, self.span_max, self.distribution, self.median, self.sigma)


def document_positions(doc_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per position of a row: its document position (positions of its document earlier in the row) and the end of its
    document's run in the row (the first later position of another document, or L)."""
    L = len(doc_ids)
    idx = np.arange(L)
    new = np.ones(L, dtype=bool)
    new[1:] = doc_ids[1:] != doc_ids[:-1]
    start = np.maximum.accumulate(np.where(new, idx, 0))
    ends = np.append(np.flatnonzero(new)[1:], L)
    return idx - start, ends[np.cumsum(new) - 1]


def sink_positions(doc_ids: torch.Tensor, n_sinks: int) -> torch.Tensor:
    """(B, L) bool: the first n_sinks positions of each document's run in each row (document position < n_sinks)."""
    return torch.stack([torch.from_numpy(document_positions(d.numpy())[0] < n_sinks) for d in doc_ids])


def span_starts(doc_ids: torch.Tensor, scored: torch.Tensor) -> torch.Tensor:
    """(B, L) bool: the first position of each run of consecutive `scored` positions within one document: a scored
    position whose previous position in the row is not a scored position of the same document. doc_ids, scored
    (B, L) host tensors."""
    prev = torch.zeros_like(scored)
    prev[:, 1:] = scored[:, :-1] & (doc_ids[:, 1:] == doc_ids[:, :-1])
    return scored & ~prev


def replay_positions(doc_ids: torch.Tensor, scored: torch.Tensor, window: int) -> torch.Tensor:
    """(B, L) bool: the replay window of each span of `scored` positions (module docstring): position p is True iff a
    span start s (span_starts) of p's document lies in (p, p + window], i.e. p is one of the min(window, available)
    positions of the same document immediately before a span's first position. Scored positions and sinks inside a
    window are returned True as well; the caller keeps what they are. window 0: all False."""
    assert window >= 0, window
    B, L = scored.shape
    idx = torch.arange(L).expand(B, L)
    far = L + window  # "no start": farther than any window
    nxt = torch.where(span_starts(doc_ids, scored), idx, torch.full_like(idx, far))
    nxt = torch.flip(torch.cummin(torch.flip(nxt, dims=[1]), dim=1).values, dims=[1])  # the first start at or after p
    after = torch.cat([nxt[:, 1:], torch.full((B, 1), far, dtype=nxt.dtype)], dim=1)  # the first start after p
    # The first start after p in another document: then no start of p's document follows p (a document is one
    # stretch of the row), so p is in no window. Documents are compared by their stretch of the row (the count of
    # document changes up to a position), as span_starts and sink_positions read them, not by their ids.
    stretch = torch.zeros_like(idx)
    stretch[:, 1:] = (doc_ids[:, 1:] != doc_ids[:, :-1]).long().cumsum(dim=1)
    same = torch.gather(stretch, 1, after.clamp(max=L - 1)) == stretch
    return (after - idx <= window) & (after < L) & same


def answer_span_mask(doc_ids: torch.Tensor, valid: torch.Tensor, rule: AnswerSpanRule, g: torch.Generator,
                     chat: torch.Tensor | None = None, roles: torch.Tensor | None = None) -> torch.Tensor:
    """One row's answer positions (module docstring): doc_ids, valid (target_valid), chat and roles (the packer's
    decoder mask, read at chat positions only) are (L,) host tensors; chat None means no chat documents. Returns (L,)
    bool. The draws, in order: a gap, then per span its length and the next gap; spans stop at the row's end."""
    d = doc_ids.numpy()
    L = len(d)
    docpos, seg_end = document_positions(d)
    is_chat = np.zeros(L, dtype=bool) if chat is None else chat.numpy().astype(bool)
    eligible = np.flatnonzero((docpos >= rule.min_context) & ~is_chat)
    spans = np.zeros(L, dtype=bool)
    j = rule.gap(g)
    while j < len(eligible):
        s = int(eligible[j])
        e = min(s + rule.length(g), int(seg_end[s]))  # cut at the document's end (decided at each position by its doc id)
        spans[s:e] = True
        j = int(np.searchsorted(eligible, e)) + rule.gap(g)  # the next eligible position at or after e, then the gap
    out = torch.from_numpy(spans)
    if roles is not None and chat is not None:
        out = out | (roles & chat)
    return out & valid


def random_positions(valid: torch.Tensor, count: int, g: torch.Generator) -> torch.Tensor:
    """valid (L,) bool -> (L,) bool with min(count, valid.sum()) True entries, a uniform random
    subset of the valid positions (without replacement)."""
    idx = torch.nonzero(valid).squeeze(1)
    take = idx[torch.randperm(idx.numel(), generator=g)[:count]]
    out = torch.zeros_like(valid)
    out[take] = True
    return out


@dataclass(frozen=True)
class PositionSampler:
    """Replaces a batch's decoder mask and adds its aux mask, as configured.

    dec_count: decoder positions per row (the capacity C), or None to keep the data's mask;
    aux_count: aux-head positions per row, or None for no aux mask (v1: the context positions); 0 with
               aux_positions "none" (an all-False aux mask);
    answer: the answer-span rule (decoder_positions "answer_spans"), which replaces the decoder mask; None otherwise;
    sinks: with answer, the decoder also at each document's first `sinks` positions, not scored (module docstring).
    replay: with answer, the decoder also at the `replay` positions before each answer span, not scored (module
            docstring, "Replay window"); 0: none.
    """

    seed: int
    dec_count: int | None
    aux_count: int | None
    answer: AnswerSpanRule | None = None
    sinks: int = 0
    replay: int = 0

    @classmethod
    def from_config(cls, cfg: TrainConfig, capacity: int) -> "PositionSampler":
        d, loss = cfg.data, cfg.loss
        if d.decoder_positions not in ("data", "random", "answer_spans"):
            raise ValueError(f"data.decoder_positions {d.decoder_positions!r}: 'data', 'random' or 'answer_spans'")
        if loss.aux_positions not in ("context", "random", "none"):
            raise ValueError(f"loss.aux_positions {loss.aux_positions!r}: 'context', 'random' or 'none'")
        if loss.aux_positions == "none" and loss.aux_weight != 0.0:
            raise ValueError(f"loss.aux_positions 'none' has no aux-head position: it needs loss.aux_weight 0, not {loss.aux_weight}")
        answer = None
        if d.decoder_positions == "answer_spans":
            if cfg.model.arch not in ANSWER_SPAN_ARCHES:
                raise ValueError(f"data.decoder_positions 'answer_spans' is for {ANSWER_SPAN_ARCHES}, not {cfg.model.arch!r}")
            answer = AnswerSpanRule(d.answer_min_context, d.answer_gap_mean, d.span_distribution, d.span_median,
                                    d.span_sigma, d.span_min, d.span_max)
        sinks = cfg.model.decoder.n_sinks if (answer is not None and d.answer_sinks and cfg.model.arch == "ced_v2") else 0
        window = d.answer_replay_window
        if isinstance(window, bool) or not isinstance(window, int) or window < 0:
            raise ValueError(f"data.answer_replay_window {window!r}: an integer >= 0")
        if window and answer is None:
            raise ValueError(f"data.answer_replay_window {window} is read with data.decoder_positions 'answer_spans' only, "
                             f"not {d.decoder_positions!r}")
        # CED v2 only, like the sinks: a control trains on every valid target and only reports ce_dec at the answers
        replay = window if (answer is not None and cfg.model.arch == "ced_v2") else 0
        if not 0.0 < loss.aux_fraction <= 1.0:
            raise ValueError(f"loss.aux_fraction {loss.aux_fraction} outside (0, 1]")
        if d.decoder_positions == "random" and cfg.model.arch == "looped_ced":
            raise ValueError("data.decoder_positions 'random' depends on document boundaries: not for Looped-CED (v1)")
        if d.decoder_positions == "random" and cfg.model.decoder.self_kv and capacity < d.seq_len:
            # With decoder self-KV a slot reads the other slots of its document, so a random subset
            # (which depends on the whole row's valid targets, future document ends included) would
            # leak the future. At capacity L the subset is every valid target, and the only positions
            # left out are each document's last one, which no other query of that document can see.
            raise ValueError("data.decoder_positions 'random' with decoder self_kv needs capacity = seq_len "
                             f"(every valid target), not {capacity} of {d.seq_len}")
        aux_count = None
        if loss.aux_positions == "random":
            aux_count = round(loss.aux_fraction * d.seq_len)
        elif loss.aux_positions == "none":
            aux_count = 0
        return cls(
            seed=d.seed,
            dec_count=capacity if d.decoder_positions == "random" else None,
            aux_count=aux_count,
            answer=answer,
            sinks=sinks,
            replay=replay,
        )

    @property
    def active(self) -> bool:
        return self.dec_count is not None or self.aux_count is not None or self.answer is not None

    def answer_masks(self, batch: Batch, step: int, row0: int = 0) -> torch.Tensor:
        """(B, L) answer positions of the batch of `step` (answer_span_mask per row, its own generator). row0: the batch's
        first row's index in the step's batch (apply)."""
        B = batch.target_valid.shape[0]
        assert row0 + B <= _STRIDE
        return torch.stack([
            answer_span_mask(batch.doc_ids[b], batch.target_valid[b], self.answer,
                             counter_generator(self.seed, step, stream=ANSWER_SPAN_STREAM + row0 + b),
                             chat=None if batch.chat is None else batch.chat[b], roles=batch.dec_mask[b])
            for b in range(B)
        ])

    def _masks(self, valid: torch.Tensor, step: int, count: int, stream: int, row0: int = 0) -> torch.Tensor:
        assert row0 + valid.shape[0] <= _STRIDE
        return torch.stack([
            random_positions(valid[b], count, counter_generator(self.seed, step, stream=stream + row0 + b))
            for b in range(valid.shape[0])
        ])

    def apply(self, batch: Batch, step: int, row0: int = 0) -> Batch:
        """The batch of `step` with the configured masks (the same object when nothing is random). row0: the index of the
        batch's first row in the step's batch, so that row b draws from the generator of row row0 + b (data parallelism:
        a rank's rows are rows row0 .. of the step's batch, ced/train/data_parallel.py, D2; 0 for the whole batch)."""
        if not self.active:
            return batch
        changes = {}
        if self.answer is not None:
            changes["dec_mask"] = self.answer_masks(batch, step, row0)
            answers = changes["dec_mask"]
            if self.sinks:
                sink = sink_positions(batch.doc_ids, self.sinks)
                changes["target_valid"] = batch.target_valid & ~(sink & ~changes["dec_mask"])  # sinks are not scored
                changes["dec_mask"] = changes["dec_mask"] | sink
            if self.replay:  # the positions before each answer span: decoder slots, not scored (answers stay scored)
                replay = replay_positions(batch.doc_ids, answers, self.replay)
                changes["target_valid"] = changes.get("target_valid", batch.target_valid) & ~(replay & ~answers)
                changes["dec_mask"] = changes["dec_mask"] | replay
        elif self.dec_count is not None:
            changes["dec_mask"] = self._masks(batch.target_valid, step, self.dec_count, DEC_POSITION_STREAM, row0)
        if self.aux_count == 0:  # aux_positions "none" (what random_positions gives for a count of 0)
            changes["aux_mask"] = torch.zeros_like(batch.target_valid)
        elif self.aux_count is not None:
            changes["aux_mask"] = self._masks(batch.target_valid, step, self.aux_count, AUX_POSITION_STREAM, row0)
        return dataclasses.replace(batch, **changes)

    def shortfall_keys(self) -> list[str]:
        """The keys shortfall() returns (configuration, not data), in its order."""
        return [k for k, n in (("dec_positions_short", self.dec_count), ("aux_positions_short", self.aux_count))
                if n is not None]

    def shortfall(self, batch: Batch) -> dict[str, int]:
        """Positions missing from the configured counts, summed over rows (rows with fewer valid
        targets than the count). Zero in normal packed data; the trainer logs it."""
        out = {}
        if self.dec_count is not None:
            out["dec_positions_short"] = int((self.dec_count - batch.dec_mask.sum(1)).clamp(min=0).sum())
        if self.aux_count is not None:
            out["aux_positions_short"] = int((self.aux_count - batch.aux_mask.sum(1)).clamp(min=0).sum())
        return out
