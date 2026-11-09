"""Deterministic synthetic packed batches and the pretraining decoder-span sampler.

`SyntheticPackedData.batch(step)` is a pure function of (seed, step), so resuming a run needs only
the step counter. Documents come from a fixed random first-order Markov chain over the vocabulary,
so a small model has something learnable and loss curves mean something. The real tokenized-shard
pipeline replaces this in Phase 0; it must keep the same Batch fields and the same span rules.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

from ced.config import DataConfig


@dataclass
class Batch:
    tokens: torch.Tensor  # (B, L) inputs
    targets: torch.Tensor  # (B, L) next tokens
    target_valid: torch.Tensor  # (B, L) bool; False where the target belongs to the next document
    doc_ids: torch.Tensor  # (B, L) document id per position, increasing along the row
    dec_mask: torch.Tensor  # (B, L) bool; decoder positions (exactly `capacity` per row)
    # (B, L) bool, positions where the aux head is trained (LossConfig.aux_positions "random",
    # ced/train/positions.py); None: the context positions (v1).
    aux_mask: torch.Tensor | None = None
    # (B, L) bool, True at positions of chat (role-structured) documents (the shard packer); None: no chat documents
    # (synthetic data). Read only by the answer-span positions (DataConfig.decoder_positions "answer_spans"), which
    # take a chat document's answers from its roles; nothing else uses it, and batch_hash does not include it.
    chat: torch.Tensor | None = None
    # Segment-based pass counts only (model.loop.segment_passes; ced/train/pass_positions.py). generated,
    # next_generated: (B, L) bool, the shards' assistant flags as the segment state machine reads them: the input
    # token at t is a generated assistant token; its next token is one, in the same document (the shard packer, when
    # asked: ShardPacker(flags=True)); None: no flags (synthetic data), the state machine reads the role headers.
    # pass_counts: (B, L) int8, the passes of every position of the batch (Trainer.batch_at, from the switch step
    # on); None: no schedule, or a single-pass step. Nothing else reads them, and batch_hash does not include them
    # (they are functions of the tokens, the masks and the config).
    generated: torch.Tensor | None = None
    next_generated: torch.Tensor | None = None
    pass_counts: torch.Tensor | None = None
    # Token Superposition Training only (TrainConfig.tst; ced/train/tst.py). In a bag phase of bag size s, `tokens` is
    # (B, L, s): position j holds the s input tokens of its bag; `targets` / `target_valid` are the first target of
    # each position. multi_targets, multi_valid: (B, L, K), the K targets each position is scored against (a bag phase:
    # the next bag, K = s; a next-token phase with the multi-hot aux, TSTConfig.aux_multi_hot: the tokens t + 1 ..
    # t + K) and whether each is in the position's document (the base's target_valid rule per token); entry 0 is
    # `targets` / `target_valid`. None: one target per position (every batch without TST). batch_hash does not include
    # them (they are functions of the stream position the tokens already identify).
    multi_targets: torch.Tensor | None = None
    multi_valid: torch.Tensor | None = None
    # The per-token loss mask (DataConfig.loss_mask). (B, L) bool: whether position t's TARGET (token t + 1) keeps its
    # loss, a function of that token's place in the data stream (the shard packer). The trainer scores position t only
    # where target_valid & loss_mask (Trainer.loss_valid): every loss branch and every denominator. None (the default,
    # "all"): every valid target, every batch before the key. batch_hash does not include it (the arms of one comparison
    # read the same rows); the run's data identity names the mode (Trainer.data_state).
    loss_mask: torch.Tensor | None = None


def counter_generator(seed: int, step: int, stream: int) -> torch.Generator:
    """A generator whose state depends only on (seed, step, stream): counter-based randomness.

    The three integers are hashed together, so distinct triples never share a generator. (An
    earlier arithmetic mix aliased (step, stream) with (step + 1, stream - 101), which repeated
    rows across steps at batch size >= 102; found in review, 2026-10-02.)
    """
    digest = hashlib.blake2b(f"{seed}:{step}:{stream}".encode(), digest_size=8).digest()
    g = torch.Generator()
    g.manual_seed(int.from_bytes(digest, "little") & (2**63 - 1))
    return g


def draw_span_length(g: torch.Generator, lo: int, hi: int, distribution: str, median: float, sigma: float) -> int:
    """One span length in [lo, hi].

    Log-normal lengths are truncated, not clamped, so the shape inside [lo, hi] is the fitted one.
    The draw uses the inverse CDF restricted to [lo - 0.5, hi + 0.5] (the interval that rounds into
    [lo, hi]). This is the same distribution as rejection sampling, in one draw even when the
    interval holds little mass (short rows).
    """
    if distribution == "uniform":
        return int(torch.randint(lo, hi + 1, (1,), generator=g))
    if distribution != "lognormal":
        raise ValueError(f"unknown span distribution {distribution!r}")
    mu = math.log(median)
    a = torch.special.ndtr(torch.tensor((math.log(max(lo - 0.5, 1e-9)) - mu) / sigma, dtype=torch.float64))
    b = torch.special.ndtr(torch.tensor((math.log(hi + 0.5) - mu) / sigma, dtype=torch.float64))
    u = a + (b - a) * torch.rand((), generator=g, dtype=torch.float64)
    length = round(math.exp(mu + sigma * float(torch.special.ndtri(u))))
    return min(max(length, lo), hi)


def sample_spans(
    length: int,
    n_dec: int,
    g: torch.Generator,
    *,
    distribution: str = "lognormal",
    median: float = 230.0,
    sigma: float = 1.16,
    span_min: int = 16,
    span_max: int = 4096,
) -> torch.Tensor:
    """Mark exactly n_dec positions of a row of `length` as decoder positions, in spans.

    Lengths are drawn from the configured distribution within [span_min, min(n_dec, span_max)].
    Drawing continues until n_dec positions are covered. A span that would overshoot is cut to the
    remainder, and a span that would leave a tail shorter than span_min absorbs that tail, so there
    are no tiny trailing spans (a span can then exceed span_max by less than span_min).

    Spans are placed uniformly at random, with at least one context position between neighbouring
    spans when the row has room. Placement uses only (length, RNG), never the tokens or the
    document boundaries: a placement that depended on where a document ends would leak that future
    information to earlier positions through the kind of KV they offer.
    """
    mask = torch.zeros(length, dtype=torch.bool)
    if n_dec == 0:
        return mask
    assert n_dec <= length
    lo = max(1, min(span_min, n_dec))
    hi = max(lo, min(span_max, n_dec))
    lengths, remaining = [], n_dec
    while remaining > 0:
        span = draw_span_length(g, lo, hi, distribution, median, sigma)
        if span >= remaining or remaining - span < lo:
            span = remaining
        lengths.append(span)
        remaining -= span

    n = len(lengths)
    free = length - n_dec
    inner_gap = 1 if free >= n - 1 else 0  # keep spans apart when possible
    extra = free - inner_gap * (n - 1)
    cuts = sorted(torch.randint(0, extra + 1, (n,), generator=g).tolist())
    gaps = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, n)]  # the trailing gap is implicit

    pos = 0
    for i, span in enumerate(lengths):
        pos += gaps[i] + (inner_gap if i > 0 else 0)
        mask[pos : pos + span] = True
        pos += span
    assert int(mask.sum()) == n_dec
    return mask


class SyntheticPackedData:
    def __init__(self, cfg: DataConfig, vocab_size: int, capacity: int):
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.capacity = capacity
        # Fixed Markov chain: each token has 4 likely successors (80% of the mass, one of the 4
        # chosen uniformly) plus 20% uniform noise. Sampled as that mixture, so memory is O(V), not
        # the O(V^2) of an explicit transition matrix (65 GB at a 128k vocabulary).
        g = counter_generator(cfg.seed, step=-1, stream=0)
        self.successors = torch.randint(0, vocab_size, (vocab_size, 4), generator=g).tolist()

    def _row(self, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """L + 1 tokens and their document ids for one packed row."""
        n = self.cfg.seq_len + 1
        tokens = [0] * n
        docs = torch.empty(n, dtype=torch.long)
        # All random numbers of the row in four draws; the chain itself is a cheap sequential loop.
        follow = (torch.rand(n, generator=g) < 0.8).tolist()
        which = torch.randint(0, 4, (n,), generator=g).tolist()
        uniform = torch.randint(0, self.vocab_size, (n,), generator=g).tolist()
        lengths = torch.randint(self.cfg.doc_len_min, self.cfg.doc_len_max + 1, (n,), generator=g).tolist()
        pos, doc = 0, 0
        succ = self.successors
        while pos < n:
            end = min(n, pos + lengths[doc])
            tokens[pos] = uniform[pos]
            for t in range(pos + 1, end):
                tokens[t] = succ[tokens[t - 1]][which[t]] if follow[t] else uniform[t]
            docs[pos:end] = doc
            pos, doc = end, doc + 1
        return torch.tensor(tokens, dtype=torch.long), docs

    def batch(self, step: int, first: int = 0, count: int | None = None) -> Batch:
        """The step's rows first .. first + count - 1 (default: all data.batch_size of them). Row b is drawn from the
        generators of (seed, step, 1000 + b) and (seed, step, 2000 + b) whatever rows are asked for, so a data-parallel
        rank's rows (ced/train/data_parallel.py, D2) are the single process's rows of the same index."""
        rows = []
        for b in range(first, self.cfg.batch_size if count is None else first + count):
            g_tokens = counter_generator(self.cfg.seed, step, stream=1000 + b)
            g_spans = counter_generator(self.cfg.seed, step, stream=2000 + b)
            seq, docs = self._row(g_tokens)
            c = self.cfg
            dec = sample_spans(
                c.seq_len, self.capacity, g_spans, distribution=c.span_distribution,
                median=c.span_median, sigma=c.span_sigma, span_min=c.span_min, span_max=c.span_max,
            )
            rows.append((seq, docs, dec))
        seq = torch.stack([r[0] for r in rows])
        docs = torch.stack([r[1] for r in rows])
        return Batch(
            tokens=seq[:, :-1],
            targets=seq[:, 1:],
            target_valid=docs[:, 1:] == docs[:, :-1],
            doc_ids=docs[:, :-1],
            dec_mask=torch.stack([r[2] for r in rows]),
        )
