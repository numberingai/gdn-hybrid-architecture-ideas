"""Tokenized shard sets and the deterministic packer for real training data.

The on-disk format and its writer are in `ced/data/shard_format.py` (numpy only).

`ShardPacker.batch(step)` is a pure function of (seed, step), like the synthetic data: resuming
needs only the step counter. Documents are concatenated in a per-epoch permutation and cut into
rows of L + 1 tokens (consecutive rows overlap by one token, the last target of a row being the
first input of the next).

Decoder positions (decided 2026-10-03):
- chat documents use roles: position t is a decoder position iff the token at t + 1 is generated
  by the assistant in the same document;
- non-chat documents use random spans: exactly round(p * L) span positions are placed over the
  row from the counter RNG alone (log-normal lengths, `ced.train.data.sample_spans`), and the ones
  that fall on non-chat positions are kept. Placement never looks at tokens or boundaries; the
  only thing used is the kind of the document a position belongs to, known from its start.
The row's decoder capacity (e.g. 0.5 * L) is enforced by the model's keep-first-C selection, and the
trainer logs how many positions were dropped.

Two or more shard sets (DataConfig.extra_shard_dirs; the two-set loader): the packer reads a ShardUnion
(ced/data/shard_format.py), whose documents are shard_dir's then each extra set's. Nothing in this file
changes for it:
- the per-seed epoch permutation is torch.randperm over the union's documents (stream 3 of the data
  seed, as for one set), so the documents of the sets are interleaved at random and every one of them
  is read once an epoch; an epoch is the union's tokens;
- the stream has no set boundary: chat_whole fills before a chat document with the next non-chat
  document of the permutation, from either set, and target_valid is false at every document start, a
  document of one set followed by a document of the other included (a document never spans two sets);
- the union is, batch for batch and bit for bit, ONE set written with the concatenated document lists (tests/test_two_set_loader.py),
  and its fingerprint covers every set in order. With DataConfig.union_order "sets_in_order" (long-data, 2026-10-30; section 12) the
  FIRST epoch is read set by set instead (_sets_in_order_layout): the first set in the epoch-0 permutation and chat_whole layout of a
  packer over it alone, token for token, then the next set in a permutation of its own (UNION_SET_STREAM), so no document is read twice
  before every set has been read once and a run that trained on the first set alone goes on in the union where it stopped. Epochs from
  the second are one permutation over every document, as above. "permute" (the default) changes nothing in this file's behaviour.

The per-token loss mask (DataConfig.loss_mask). "all" (the default): nothing below runs and the batches are those of before
(Batch.loss_mask None). Another mode: the set's invariants are checked when the packer is made (check_loss_spans; ValueError
by name), and every row also carries "loss_mask", position t True when its TARGET (stream token t + 1) keeps its loss:
- "assistant": the target is in a non-chat document, or it is an assistant token (generated.bin);
- a span mode: the target's loss-span class (loss_spans.bin, read with the tokens; zeros for a set or union part without the
  file) is one the mode keeps (LOSS_SPAN_KEEP).
A pure function of the stream position, like every field of a row: a row of another length (the shape schedule, a stage at
another row length) is a slice of the same mask, and a row cut never moves a head or a tail (the classes are fixed per
document when the set is built).
"""

from __future__ import annotations

import numpy as np
import torch

from ced.config import DataConfig
from ced.data.shard_format import LOSS_MASK_ALL, LOSS_MASK_MODES, LOSS_SPAN_KEEP, check_loss_spans, loss_span_table
from ced.data.shard_format import UNION_ORDER_PERMUTE, UNION_ORDER_SETS, UNION_ORDERS
from ced.data.shard_format import ShardSet, ShardUnion, ShardWriter, open_shards  # noqa: F401 (re-exported)
from ced.train.data import Batch, counter_generator, sample_spans

# The counter-RNG stream of the permutation of a union's k-th set (k >= 1) in epoch 0 under union_order "sets_in_order": UNION_SET_STREAM + k.
# The first set uses stream 3 of epoch 0, exactly as a packer over that set alone does (so its documents come in the same order, bit for bit).
# Streams are hashed with (data seed, epoch), so no other generator in use shares it (ced/train/data.py counter_generator).
UNION_SET_STREAM = 3_000_000


def check_extra_shard_dirs(cfg: DataConfig) -> None:
    """DataConfig.extra_shard_dirs: a list of directory names (strings), read with data.kind "shards" only, and not
    with data.fit_decoder_capacity (ShardPacker._generated_per_doc reads the generated flags over the span from the
    first to the last chat document in one array; across two sets that span is most of the union, about 100 GB as
    int64 for the item-6 pair: not built). ValueError otherwise; nothing for [] (every run before the key)."""
    extra = cfg.extra_shard_dirs
    if not isinstance(extra, (list, tuple)) or not all(isinstance(d, str) and d for d in extra):
        raise ValueError(f"data.extra_shard_dirs {extra!r}: a list of shard set directories ([] for none)")
    if not extra:
        return
    if cfg.kind != "shards":
        raise ValueError(f"data.extra_shard_dirs {list(extra)!r} is read with data.kind 'shards', not {cfg.kind!r}")
    if not cfg.shard_dir:
        raise ValueError("data.extra_shard_dirs: the first set is data.shard_dir, which is not set")
    if cfg.fit_decoder_capacity:
        raise ValueError("data.extra_shard_dirs is not built for data.fit_decoder_capacity (see check_extra_shard_dirs)")


def check_union_order(cfg: DataConfig) -> None:
    """DataConfig.union_order: one of UNION_ORDERS, and "sets_in_order" only with data.extra_shard_dirs (it orders the sets of a union; with
    one set there is nothing to order). ValueError by name otherwise; nothing for "permute" (every run before the key)."""
    order = cfg.union_order
    if order not in UNION_ORDERS:
        raise ValueError(f"data.union_order {order!r}: one of {UNION_ORDERS}")
    if order != UNION_ORDER_PERMUTE and not cfg.extra_shard_dirs:
        raise ValueError(f"data.union_order {order!r} orders the sets of a union: data.extra_shard_dirs is empty (one set)")


class ShardPacker:
    """Rows of L + 1 tokens cut from a per-epoch permutation of the documents."""

    def __init__(self, shards: ShardSet, cfg: DataConfig, span_fraction: float = 0.25, capacity: int | None = None,
                 flags: bool = False):
        self.shards = shards
        # flags (segment-based pass counts, ced/train/pass_positions.py): batches also carry the assistant flags of
        # every input token and of its next token (Batch.generated / next_generated). Off: the batches of before.
        self.flags = flags
        self.cfg = cfg
        self.L = cfg.seq_len
        self.span_fraction = span_fraction
        self._epoch_cache: dict[int, dict[str, np.ndarray]] = {}
        self.chat_cut = 0  # chat documents the last computed layout could not keep whole
        self.capacity = capacity
        if cfg.fit_decoder_capacity:
            assert cfg.chat_whole and capacity, "fit_decoder_capacity needs chat_whole and the decoder capacity"
        self._doc_generated = None
        # The per-token loss mask (module docstring): None for "all" (nothing more is read or built); otherwise the mode,
        # the set's invariants checked here (check_loss_spans: ValueError by name) and their data identity kept for the
        # trainer (Trainer.data_state); a span mode also reads the classes with the tokens (_span_table: kept classes).
        mode = cfg.loss_mask
        if mode not in LOSS_MASK_MODES:
            raise ValueError(f"data.loss_mask {mode!r}: one of {LOSS_MASK_MODES}")
        self.loss_mask = None if mode == LOSS_MASK_ALL else mode
        self.loss_spans_state = None if self.loss_mask is None else check_loss_spans(shards, mode)
        self._span_table = loss_span_table(mode) if mode in LOSS_SPAN_KEEP else None

    def _generated_per_doc(self) -> np.ndarray:
        """Generated (assistant) tokens per document; zero outside chat documents."""
        if self._doc_generated is None:
            s = self.shards
            out = np.zeros(s.n_docs, dtype=np.int64)
            chat = np.flatnonzero(s.doc_chat)
            if len(chat):
                # One pass over the span of chat documents; documents in between are not chat and
                # have no generated tokens, so each sum is exactly its chat document's count.
                lo, hi = int(s.doc_starts[chat[0]]), int(s.doc_starts[chat[-1] + 1])
                flags = np.asarray(s.generated[lo:hi], dtype=np.int64)
                out[chat] = np.add.reduceat(flags, s.doc_starts[chat] - lo)
            self._doc_generated = out
        return self._doc_generated

    def _epoch(self, epoch: int) -> dict[str, np.ndarray]:
        """The epoch's layout: the stream is a sequence of segments, each a piece of one document
        ("doc", "offset" into it, "length"), with "cum" the cumulative stream offsets. Without
        chat_whole every segment is a whole document, in permutation order."""
        if epoch not in self._epoch_cache:
            if epoch == 0 and getattr(self.shards, "order", UNION_ORDER_PERMUTE) == UNION_ORDER_SETS:
                doc, offset, length = self._sets_in_order_layout()  # a union of "sets_in_order": the sets one after the other
            else:
                g = counter_generator(self.cfg.seed, epoch, stream=3)
                perm = torch.randperm(self.shards.n_docs, generator=g).numpy()
                if self.cfg.chat_whole:
                    doc, offset, length = self._chat_whole_layout(perm)
                else:
                    doc, offset, length = perm, np.zeros_like(perm), np.diff(self.shards.doc_starts)[perm]
            cum = np.concatenate([[0], np.cumsum(length)])
            assert cum[-1] == self.shards.n_tokens
            self._epoch_cache = {epoch: {"doc": doc, "offset": offset, "length": length, "cum": cum}}
        return self._epoch_cache[epoch]

    def _sets_in_order_layout(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Epoch 0 of a union read with union_order "sets_in_order" (ShardUnion.order): the segments of its first set, laid out exactly
        as a packer over that set alone lays out its epoch 0 (the same permutation, counter_generator(data seed, 0, stream 3), and the
        same chat_whole layout, from stream position 0), then those of each further set k laid out the same way from the stream position
        where the sets before end (so the row boundaries of chat_whole stay at multiples of L), in a permutation of its own
        (counter_generator(data seed, 0, UNION_SET_STREAM + k)). Every document is read once; the stream's first n_0 tokens (n_0: the
        first set's) are the first set's epoch-0 stream token for token, and a chat document at the end of a set never takes a filler
        from the next one (a set is laid out on its own). self.chat_cut is the sum over the sets."""
        parts, bounds = self.shards.parts, np.cumsum([0] + [p.n_docs for p in self.shards.parts])
        outs, pos, cut = [], 0, 0
        for k, p in enumerate(parts):
            g = counter_generator(self.cfg.seed, 0, stream=3 if k == 0 else UNION_SET_STREAM + k)
            perm = torch.randperm(p.n_docs, generator=g).numpy() + int(bounds[k])  # the union's document indices
            if self.cfg.chat_whole:
                doc, offset, length = self._chat_whole_layout(perm, start=pos)
                cut += self.chat_cut
            else:
                doc, offset, length = perm, np.zeros_like(perm), np.diff(self.shards.doc_starts)[perm]
            outs.append((doc, offset, length))
            pos += int(length.sum())
        self.chat_cut = cut
        return tuple(np.concatenate([o[i] for o in outs]).astype(np.int64) for i in range(3))

    def _chat_whole_layout(self, perm: np.ndarray, start: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segments (doc, offset, length) such that no chat document of at most L + 1 tokens
        crosses a row boundary (see DataConfig.chat_whole). Longer chat documents, and a chat
        document near the end of the epoch when no non-chat document is left to fill with, are
        placed as they come (counted in self.chat_cut). start: the stream position of the first segment
        (0 for every epoch of this packer; ced/train/tst.py's OrderedPacker lays a document order out after a prefix
        of `start` tokens, so that the row boundaries stay at multiples of L of its stream)."""
        L = self.L
        lengths = np.diff(self.shards.doc_starts).tolist()
        chat = self.shards.doc_chat.astype(bool)
        fit = self.cfg.fit_decoder_capacity
        gen = self._generated_per_doc().tolist() if fit else None
        row_of_gen, row_gen = -1, 0  # assistant tokens already placed in row row_of_gen
        perm_list = perm.tolist()
        nonchat = [j for j, d in enumerate(perm_list) if not chat[d]]
        used = bytearray(len(perm_list))
        nc = 0  # next candidate in `nonchat`
        out_doc, out_off, out_len = [], [], []
        pos, cut = start, 0
        for j, d in enumerate(perm_list):
            if used[j]:
                continue
            used[j] = 1
            n = lengths[d]
            tail = None
            over_capacity = False
            if fit and chat[d]:
                g = gen[d]
                used_here = row_gen if row_of_gen == pos // L else 0
                over_capacity = used_here > 0 and used_here + g > self.capacity
            if chat[d] and n <= L + 1 and (n > (pos // L) * L + L + 1 - pos or over_capacity):
                need = (pos // L + 1) * L - pos
                while need > 0:
                    while nc < len(nonchat) and (nonchat[nc] <= j or used[nonchat[nc]]):
                        nc += 1
                    if nc == len(nonchat):
                        cut += 1
                        break
                    k = nonchat[nc]
                    used[k] = 1
                    d2 = perm_list[k]
                    take = min(need, lengths[d2])
                    out_doc.append(d2), out_off.append(0), out_len.append(take)
                    pos, need = pos + take, need - take
                    if take < lengths[d2]:
                        tail = (d2, take, lengths[d2] - take)
            elif chat[d] and n > L + 1:
                cut += 1
            if fit and chat[d]:
                row = pos // L
                row_gen = (row_gen if row_of_gen == row else 0) + gen[d]
                row_of_gen = row
            out_doc.append(d), out_off.append(0), out_len.append(n)
            pos += n
            if tail is not None:
                out_doc.append(tail[0]), out_off.append(tail[1]), out_len.append(tail[2])
                pos += tail[2]
        self.chat_cut = cut
        return np.array(out_doc, dtype=np.int64), np.array(out_off, dtype=np.int64), np.array(out_len, dtype=np.int64)

    def _stream(self, start: int, length: int):
        """Tokens, byte lengths, generated flags, chat flags and unique document ids (and, under a span mode of the loss
        mask, the loss-span classes) for stream positions [start, start + length), wrapping into later epochs."""
        s = self.shards
        spans = self._span_table is not None  # a span mode of the loss mask: the classes too ("spans")
        out = {k: [] for k in ("tokens", "bytes", "generated", "chat", "doc") + (("spans",) if spans else ())}
        pos, remaining = start, length
        stride = 2 * s.n_docs if self.cfg.chat_whole else s.n_docs  # segments per epoch, at most
        while remaining > 0:
            epoch, offset = divmod(pos, s.n_tokens)
            lay = self._epoch(epoch)
            cum = lay["cum"]
            i = int(np.searchsorted(cum, offset, side="right") - 1)  # segment index
            doc = int(lay["doc"][i])
            within = offset - cum[i]
            take = min(remaining, int(cum[i + 1] - offset))
            a = int(s.doc_starts[doc] + lay["offset"][i] + within)
            out["tokens"].append(np.asarray(s.tokens[a : a + take], dtype=np.int64))
            out["bytes"].append(np.asarray(s.byte_lens[a : a + take], dtype=np.int64) if s.byte_lens is not None
                                else np.zeros(take, dtype=np.int64))
            out["generated"].append(np.asarray(s.generated[a : a + take], dtype=bool))
            out["chat"].append(np.full(take, bool(s.doc_chat[doc])))
            out["doc"].append(np.full(take, epoch * stride + i, dtype=np.int64))  # increasing, never reused
            if spans:  # a set without loss_spans.bin is class 0 everywhere (a union's part without one: ZerosArray)
                out["spans"].append(np.asarray(s.loss_spans[a : a + take]) if s.loss_spans is not None
                                    else np.zeros(take, dtype=np.uint8))
            pos, remaining = pos + take, remaining - take
        return {k: np.concatenate(v) for k, v in out.items()}

    def row(self, index: int, length: int | None = None) -> dict[str, torch.Tensor]:
        """Row `index` of the stream cut into rows of `length` tokens (default, and every run without a shape
        schedule: the packer's own L): the stream's tokens [index * length, index * length + length].

        A row of another length (TrainConfig.shape_schedule; ced/train/shape_schedule.py) is a piece of the SAME
        stream: the layout (the document order, and chat_whole's fill) is the one of rows of L, and with length
        dividing L the rows index * (L // length) + q, q = 0 .. L / length - 1, are row `index` of L cut into
        consecutive pieces. Every field of a piece is the slice of the long row's field (tokens, targets,
        target_valid, doc_ids, chat; tests/test_shape_schedule.py), so over a run the same tokens are inputs and the
        same tokens are targets, in the same order. What a cut does to a document that crosses it:
        - the document goes on in the next piece under the same document id, and that piece is another row: the
          model starts it at the piece's position 0 with no earlier context (as it starts a document that crosses a
          row boundary of L; chat_whole keeps chat documents of at most L + 1 tokens off the boundaries of L only,
          so a shorter piece does cut them);
        - no target is lost: the last position of a piece predicts the first token of the next piece, valid when
          both are the same document (the slice of the long row's target_valid), and the first position of a piece
          predicts its second token from the first alone;
        - the assistant flags (generated, next_generated) and the chat flag are per token and go with their tokens.
        The data's own decoder mask is drawn for the piece (its length, a generator of its own per length); it is
        not a slice of the long row's mask."""
        own = length is None or length == self.L
        L = self.L if own else length
        x = self._stream(index * L, L + 1)
        same_doc = x["doc"][1:] == x["doc"][:-1]
        role = x["chat"][:-1] & x["generated"][1:] & same_doc
        c = self.cfg
        spans = sample_spans(
            # stream 4: the rows of L (every run before the shape schedule); another length has its own stream, so no
            # two (length, index) pairs share a generator (ced/train/positions.py lists the streams in use)
            L, round(self.span_fraction * L), counter_generator(c.seed, index, stream=4 if own else 10_000_000 + L),
            distribution=c.span_distribution, median=c.span_median, sigma=c.span_sigma,
            span_min=c.span_min, span_max=c.span_max,
        ).numpy()
        kept_spans = spans & ~x["chat"][:-1]
        if c.fit_decoder_capacity:
            excess = int(role.sum()) + int(kept_spans.sum()) - self.capacity
            if excess > 0:  # spans yield to assistant positions, from the end of the row
                kept_spans[np.flatnonzero(kept_spans)[-excess:]] = False
        dec = role | kept_spans
        flags = {}
        if self.flags:  # the input token is generated; its next token is, in the same document (`role`)
            flags = {"generated": torch.from_numpy(x["chat"][:-1] & x["generated"][:-1]),
                     "next_generated": torch.from_numpy(role.copy())}
        if self.loss_mask is not None:  # the per-token loss mask (module docstring), of each position's TARGET token
            if self._span_table is not None:
                keep = self._span_table[x["spans"][1:]]
            else:  # "assistant": the target is in a non-chat document, or the assistant's token
                keep = ~x["chat"][1:] | x["generated"][1:]
            flags["loss_mask"] = torch.from_numpy(np.ascontiguousarray(keep))
        return {
            **flags,
            "tokens": torch.from_numpy(x["tokens"][:-1]),
            "targets": torch.from_numpy(x["tokens"][1:]),
            "target_valid": torch.from_numpy(same_doc),
            "target_bytes": torch.from_numpy(x["bytes"][1:]),
            "doc_ids": torch.from_numpy(x["doc"][:-1]),
            "dec_mask": torch.from_numpy(dec),
            "chat": torch.from_numpy(x["chat"][:-1].copy()),  # the input position's document is a chat document
        }

    def batch(self, step: int) -> Batch:
        return self.rows(step * self.cfg.batch_size, self.cfg.batch_size)

    def rows(self, first: int, count: int, length: int | None = None) -> Batch:
        """`count` consecutive rows of `length` tokens (default: the packer's L) from row index `first`, as a Batch:
        the stream's tokens [first * length, (first + count) * length]. batch(step) is rows(step * batch_size,
        batch_size); a shape schedule reads other counts and lengths of the same stream (row)."""
        if length is not None and length != self.L and self.cfg.fit_decoder_capacity:
            raise ValueError("rows of another length than data.seq_len are not built for data.fit_decoder_capacity (its "
                             "layout fits the assistant tokens of rows of data.seq_len)")
        rows = [self.row(first + b, length) for b in range(count)]
        keys = ("tokens", "targets", "target_valid", "doc_ids", "dec_mask", "chat")
        if self.flags:
            keys += ("generated", "next_generated")
        if self.loss_mask is not None:  # DataConfig.loss_mask (module docstring); absent ("all"): Batch.loss_mask None
            keys += ("loss_mask",)
        return Batch(**{k: torch.stack([r[k] for r in rows]) for k in keys})
