"""Segment-based pass counts on the CPU: which positions run which pass (LoopConfig.segment_passes; the mechanism is
ced/model/pass_schedule.py).

"Positions per pass come from the data loader on the CPU in training, with fixed slot sizes per pass (a few bucket
sizes) so nothing recompiles." This file is that loader side, and the streaming form generation uses. No model code,
no device: numpy and CPU tensors. Nothing here names an architecture.

The pass count of position t is map[segment of t], where the segment of t is what ced.eval.token_types.segment_positions
gives: a function of the tokens up to t and of whether the next token is a generated assistant token. A position whose
next token is not generated, and one the state machine cannot place ("unplaced"), is "other". With
LoopConfig.unscored_passes = 1, a decoder position that is not scored (a sink, a replay-window position) counts 1
whatever its segment.

Three implementations of the same labels, checked against each other and against token_types.segment_positions
(tests/test_seg_passes_positions.py):

    segment_labels   one document piece at once, driven by events (the five deciding control tokens and the starts of
                     generated runs), numpy between them: the training loader's and the no-cache forward's. A row of
                     8,192 tokens costs well under a millisecond without a chat document and a few per chat document.
    SegmentStream    one token at a time: generation, and every forward with a cache.
    token_types.segment_positions   the readable reference (unchanged).

Which tokens are generated (the "assistant flags"):
- the training loader has them from the shards (Batch.generated / next_generated: ced/train/shards.py);
- a caller that has none (the eval rows and the probe sets store none) gets them from the stream itself: the chat
  template writes "<|im_start|>assistant\\n" before every assistant message and the message ends with its <|im_end|>
  (tools/data/chat_format.py), so a token is generated iff it lies between such a header and that <|im_end|>
  (`stream_flags`, `RoleStream`). On a whole document these are the shards' flags exactly. A row that starts inside
  a message has no header for it: without flags those positions are "other" up to the next header.

PassScheduler holds the map and the vocabulary's bytes and gives per-position counts, (B, L); PassPlanner turns the
counts of the decoder slots (CED v2: `plan`) or of every position (the control with a second pass through its top
block: `plan_positions`) into the fixed-shape PassPlan of one forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ced.eval import token_types as tt
from ced.model.capacity import select_slots
from ced.model.pass_schedule import (BUCKET_FRACTIONS, PassPlan, PassSchedule, bucket_sizes, build_plan, schedule_of,
                                     unit_counts)

cf = tt.cf
N_SPECIAL = tt.N_SPECIAL
SEG = tt.SEG
_START, _REASON, _CONTENT, _HEAD, _BODY, _UNPLACED = range(6)  # token_types' states, in its order
assert (_START, _REASON, _CONTENT, _HEAD, _BODY, _UNPLACED) == (tt._ST_START, tt._ST_REASON, tt._ST_CONTENT, tt._ST_HEAD,
                                                               tt._ST_BODY, tt._ST_UNPLACED)
_SEG_OF = np.array([tt._ST_TO_SEG[s] for s in range(6)], dtype=np.int8)
_IM_START, _IM_END = tt.ID[cf.IM_START], tt.ID[cf.IM_END]
_THINK, _THINK_END = tt.ID[cf.THINK], tt.ID[cf.THINK_END]
_CALL, _CALL_END = tt.ID[cf.TOOL_CALL], tt.ID[cf.TOOL_CALL_END]
_DECIDING = np.zeros(N_SPECIAL, dtype=bool)
_DECIDING[[_IM_END, _THINK, _THINK_END, _CALL, _CALL_END]] = True
ASSISTANT_HEADER = b"assistant\n"  # the role line after <|im_start|> (chat_format.message_pieces)


def _transition(state: int, x: int) -> int:
    """The state after a deciding control token x (token_types.segment_positions' rules); _HEAD means a new call."""
    if x == _IM_END:
        return _START
    if x == _THINK:
        return _REASON if state in (_START, _UNPLACED, _REASON) else _UNPLACED
    if x == _THINK_END:
        return _CONTENT if state in (_REASON, _UNPLACED) else _UNPLACED
    if x == _CALL:
        return _HEAD if state in (_START, _CONTENT, _UNPLACED) else _UNPLACED
    return _CONTENT if state in (_HEAD, _BODY, _UNPLACED) else _UNPLACED  # </tool_call>


def segment_labels(ids, generated, next_generated, table: tt.TokenTable) -> np.ndarray:
    """(n,) int8: token_types.SEGMENTS index of every position whose next token is generated, else -1, for ONE
    document piece (a document, or its part inside a row). ids (n,); generated (n,) bool: token t is generated;
    next_generated (n,) bool: token t + 1 is (the last entry: the token after the piece, in the same document).
    Equal to token_types.segment_positions(ids + [any], generated + [next_generated[-1]])[:n]."""
    ids = np.asarray(ids)
    a = np.asarray(generated, dtype=bool)
    nx = np.asarray(next_generated, dtype=bool)
    n = len(ids)
    out = np.full(n, -1, dtype=np.int8)
    if n == 0:
        return out
    out[nx & ~a] = SEG["content"]  # the position before a message's first token
    if not a.any():
        return out
    special = ids < N_SPECIAL
    deciding = np.zeros(n, dtype=bool)
    deciding[special] = _DECIDING[ids[special]]
    plain = np.concatenate([[0], np.cumsum(~special)])  # plain[t]: tokens that are not control tokens before t
    edge = np.diff(np.concatenate([[0], a.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)
    events = np.flatnonzero(deciding & a)
    lab = np.empty(n, dtype=np.int8)
    token_bytes = table.token_bytes

    for s, e in zip(starts.tolist(), ends.tolist()):
        state = _UNPLACED if s == 0 else _START  # a piece that starts inside a message: its state is unknown
        at = s  # tokens before `at` are labelled
        scan = None
        for q in [*events[np.searchsorted(events, s) : np.searchsorted(events, e)].tolist(), e]:
            # tokens at .. q - 1 are not deciding tokens: the state moves only from START (to CONTENT at the first
            # token that is not a control token) and inside a call head (the name scan)
            if at < q:
                if state == _HEAD:
                    t = at
                    while t < q and state == _HEAD:
                        x = int(ids[t])
                        r = -1 if x < N_SPECIAL else scan.feed(token_bytes[x])
                        if r < 0:  # a control token, or bytes that cannot be the start of a name: the body starts here
                            state = _BODY
                            break
                        lab[t] = SEG["call_head"]
                        t += 1
                        if r > 0:  # this token held the name's last byte: the body starts after it
                            state = _BODY
                    lab[t:q] = _SEG_OF[state]
                else:
                    lab[at:q] = _SEG_OF[state]
                    if state == _START and plain[q] > plain[at]:
                        state = _CONTENT
            if q == e:
                break
            state = _transition(state, int(ids[q]))
            if state == _HEAD:
                scan = tt._NameScan()
            lab[q] = _SEG_OF[state]
            at = q + 1
    on = a & nx
    out[on] = lab[on]
    return out


class SegmentStream:
    """token_types.segment_positions one token at a time, for ONE document piece: feed(x, generated, next_generated)
    returns the SEGMENTS index of the position of token x, or -1 when its next token is not generated. The first
    token fed is the piece's first (a generated first token starts in the unknown state, as a row cut inside a
    message). copy() gives an independent stream in the same state (a cache that is extended twice)."""

    __slots__ = ("first", "prev", "state", "scan", "done")

    def __init__(self):
        self.first, self.prev, self.state, self.scan, self.done = True, False, None, None, False

    def copy(self) -> "SegmentStream":
        c = SegmentStream()
        c.first, c.prev, c.state, c.done = self.first, self.prev, self.state, self.done
        if self.scan is not None:
            c.scan = tt._NameScan()
            c.scan.st, c.scan.k = self.scan.st, self.scan.k
        return c

    def feed(self, x: int, generated: bool, next_generated: bool, table: tt.TokenTable) -> int:
        first, self.first = self.first, False
        if not generated:
            self.state, self.prev = None, False
            return SEG["content"] if next_generated else -1
        if first or not self.prev:
            self.state = _UNPLACED if first else _START
        self.prev = True
        state = self.state
        if state == _HEAD and self.done:
            state = _BODY
        if x < N_SPECIAL and _DECIDING[x]:
            state = _transition(state, x)
            if state == _HEAD:
                self.scan, self.done = tt._NameScan(), False
        elif state == _HEAD:
            r = -1 if x < N_SPECIAL else self.scan.feed(table.token_bytes[x])
            if r < 0:
                state = _BODY
            elif r > 0:
                self.done = True
        elif state == _START and x >= N_SPECIAL:
            state = _CONTENT
        self.state = state
        return int(_SEG_OF[state]) if next_generated else -1


# ---------------------------------------------------------------------------------------------------------------------
# assistant flags from the stream, for callers that have none


def _header_end(ids, p: int, n: int, table: tt.TokenTable) -> tuple[int, bool]:
    """After <|im_start|> at p: (the index of the first token after the role line, whether the role is assistant).
    The role line is the tokens up to the one that ends with the first newline; a control token, a line longer than
    the header, or bytes after the newline in the same token end the header as "not assistant"."""
    buf = b""
    t = p + 1
    while t < n:
        x = int(ids[t])
        if x < N_SPECIAL:
            return t, False
        buf += table.token_bytes[x]
        t += 1
        if b"\n" in buf:
            return t, buf == ASSISTANT_HEADER
        if len(buf) >= len(ASSISTANT_HEADER):
            return t, False
    return n, False


def stream_flags(ids, table: tt.TokenTable) -> tuple[np.ndarray, np.ndarray]:
    """(generated, next_generated), each (n,) bool, of one document piece read from its own tokens (module
    docstring): token t is generated iff it lies after an "<|im_start|>assistant\\n" header and up to and including
    that message's <|im_end|>; next_generated[t] says the same of token t + 1 from the tokens up to t alone (the
    header just ended, or the message has not). A piece that starts inside a message has no header for it: not
    generated until the next header."""
    ids = np.asarray(ids)
    n = len(ids)
    nxt = np.zeros(n, dtype=bool)
    starts = np.flatnonzero(ids == _IM_START)
    if len(starts):
        ends = np.flatnonzero(ids == _IM_END)
        for j, p in enumerate(starts.tolist()):
            body, assistant = _header_end(ids, p, n, table)
            if not assistant:
                continue
            stop = int(starts[j + 1]) if j + 1 < len(starts) else n  # the next header ends the message at the latest
            k = np.searchsorted(ends, body)
            if k < len(ends) and ends[k] < stop:
                stop = int(ends[k])  # the message's <|im_end|>: generated itself, and nothing after it
            # positions body - 1 .. stop - 1 are followed by a generated token (the header's last token included)
            nxt[body - 1 : stop] = True
    gen = np.zeros(n, dtype=bool)
    gen[1:] = nxt[:-1]
    return gen, nxt


class RoleStream:
    """stream_flags one token at a time: feed(x) returns (generated, next_generated) of token x."""

    __slots__ = ("mode", "buf")
    OUT, HEADER, BODY = 0, 1, 2

    def __init__(self):
        self.mode, self.buf = self.OUT, b""

    def copy(self) -> "RoleStream":
        c = RoleStream()
        c.mode, c.buf = self.mode, self.buf
        return c

    def feed(self, x: int, table: tt.TokenTable) -> tuple[bool, bool]:
        generated = self.mode == self.BODY
        if x == _IM_START:
            self.mode, self.buf = self.HEADER, b""
        elif self.mode == self.HEADER:
            if x < N_SPECIAL:
                self.mode = self.OUT
            else:
                self.buf += table.token_bytes[x]
                if b"\n" in self.buf:
                    self.mode = self.BODY if self.buf == ASSISTANT_HEADER else self.OUT
                elif len(self.buf) >= len(ASSISTANT_HEADER):
                    self.mode = self.OUT
        elif self.mode == self.BODY and x == _IM_END:
            self.mode = self.OUT
        return generated, self.mode == self.BODY


# ---------------------------------------------------------------------------------------------------------------------
# counts per position


def piece_bounds(doc_ids: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every document's stretch of a row."""
    d = np.asarray(doc_ids)
    cuts = np.flatnonzero(d[1:] != d[:-1]) + 1
    edges = [0, *cuts.tolist(), len(d)]
    return list(zip(edges[:-1], edges[1:]))


@dataclass
class StreamState:
    """The streaming state of a batch of rows between the chunks of a cached forward: per row the document id of
    its last token, the two state machines (None before the first token) and whether the next token to be fed is
    generated. Never changed in place: feed returns a new one, so a cache may be extended more than once."""

    doc: list
    role: list
    seg: list
    gen: list


class PassScheduler:
    """Per-position pass counts from a PassSchedule and the vocabulary's bytes (None for a schedule without a map,
    which reads no token)."""

    def __init__(self, schedule: PassSchedule, table: tt.TokenTable | None):
        assert schedule is not None
        assert schedule.counts is None or table is not None, "a segment map reads the vocabulary's bytes: give the TokenTable"
        self.schedule = schedule
        self.table = table
        c = schedule.counts
        # indexed by a SEGMENTS index or -1: reasoning, content, call_head, call_body, unplaced -> other, and -1 -> other
        self._by_label = None if c is None else np.array(
            [c["reasoning"], c["content"], c["call_head"], c["call_body"], c["other"], c["other"]], dtype=np.int8)
        assert c is None or tt.SEGMENTS == ("reasoning", "content", "call_head", "call_body", "unplaced")

    @classmethod
    def from_config(cls, loop, vocab_size: int | None = None) -> "PassScheduler | None":
        """The scheduler of a LoopConfig (None without a schedule). With a map, loop.segment_tokenizer is read."""
        schedule = schedule_of(loop)
        if schedule is None:
            return None
        table = None
        if schedule.counts is not None:
            path = getattr(loop, "segment_tokenizer", None)
            if not path:
                raise ValueError("loop.segment_passes needs loop.segment_tokenizer: the tokenizer.json whose vocabulary "
                                 "bytes the segment state machine reads")
            table = tt.TokenTable.from_tokenizer_json(path)
            if vocab_size is not None and len(table.token_bytes) != vocab_size:
                raise ValueError(f"loop.segment_tokenizer {path}: {len(table.token_bytes)} entries, the model's "
                                 f"vocabulary has {vocab_size}")
        return cls(schedule, table)

    # -- labels ---------------------------------------------------------------------------------------------------

    def labels(self, tokens, doc_ids, generated=None, next_generated=None) -> np.ndarray:
        """(B, L) int8: the segment of every position (SEGMENTS index, -1 where the next token is not generated),
        document piece by document piece. generated / next_generated (B, L) bool: the assistant flags of each input
        token and of its next token (the loader's); None: read from the stream (stream_flags)."""
        tok = np.asarray(tokens)
        doc = np.asarray(doc_ids)
        gen = None if generated is None else np.asarray(generated, dtype=bool)
        nxt = None if next_generated is None else np.asarray(next_generated, dtype=bool)
        assert (gen is None) == (nxt is None), "give both assistant flags or neither"
        B, L = tok.shape
        out = np.full((B, L), -1, dtype=np.int8)
        if gen is None and not (tok == _IM_START).any():
            return out  # no header anywhere: no token is known to be generated
        if gen is not None and not nxt.any():
            return out
        for b in range(B):
            if gen is not None and not nxt[b].any():
                continue
            for s, e in piece_bounds(doc[b]):
                ids = tok[b, s:e]
                if gen is None:
                    g, x = stream_flags(ids, self.table)
                else:
                    g, x = gen[b, s:e], nxt[b, s:e]
                if x.any():
                    out[b, s:e] = segment_labels(ids, g, x, self.table)
        return out

    def counts_of_labels(self, labels: np.ndarray) -> np.ndarray:
        return self._by_label[labels]

    def position_counts(self, tokens, doc_ids, dec_mask=None, scored=None, generated=None, next_generated=None) -> torch.Tensor:
        """(B, L) int8 CPU tensor: the pass count of every position (meaningful where the decoder runs). With a map:
        the count of the position's segment; without: the model's loop count. With unscored_passes, a decoder
        position (dec_mask) that is not scored (scored (B, L) bool: where the target counts; None: every position
        is) counts unscored_passes. All arguments are host tensors or arrays."""
        s = self.schedule
        B, L = np.asarray(tokens).shape
        if s.counts is None:
            counts = np.full((B, L), s.dense, dtype=np.int8)
        else:
            counts = self._by_label[self.labels(tokens, doc_ids, generated, next_generated)]
        if s.unscored is not None and scored is not None:
            unscored = ~np.asarray(scored, dtype=bool)
            if dec_mask is not None:
                unscored &= np.asarray(dec_mask, dtype=bool)
            counts = np.where(unscored, np.int8(s.unscored), counts)
        return torch.from_numpy(np.ascontiguousarray(counts))

    # -- streaming ------------------------------------------------------------------------------------------------

    def start(self, batch: int) -> StreamState:
        return StreamState(doc=[None] * batch, role=[None] * batch, seg=[None] * batch, gen=[False] * batch)

    def feed(self, state: StreamState | None, tokens, doc_ids, dec_mask=None, scored=None, next_generated=None
             ) -> tuple[torch.Tensor, StreamState]:
        """The next chunk of the same rows: ((B, n) int8 counts of its positions, the state after it). The assistant
        flags come from the stream (RoleStream), or, with next_generated (B, n) bool, from the caller: a generation
        harness knows whether it will ask for a token after each one it feeds, and a fed token is then generated iff
        the previous position's flag said so (a document's first token is not). The counts equal position_counts
        over the whole rows with the same flags."""
        s = self.schedule
        tok = np.asarray(tokens)
        doc = np.asarray(doc_ids)
        B, n = tok.shape
        state = self.start(B) if state is None else state
        counts = np.full((B, n), s.dense, dtype=np.int8)
        new = StreamState(doc=list(state.doc), role=list(state.role), seg=list(state.seg), gen=list(state.gen))
        if s.counts is not None:
            table, by_label = self.table, self._by_label
            flags = None if next_generated is None else np.asarray(next_generated, dtype=bool)
            for b in range(B):
                last, gen = new.doc[b], new.gen[b]
                role = None if new.role[b] is None else new.role[b].copy()
                seg = None if new.seg[b] is None else new.seg[b].copy()
                row_t, row_d = tok[b].tolist(), doc[b].tolist()
                for t in range(n):
                    if seg is None or row_d[t] != last:  # a new document: both machines start again
                        role, seg, last, gen = RoleStream(), SegmentStream(), row_d[t], False
                    x = row_t[t]
                    g, nx = role.feed(x, table)  # kept in step whoever supplies the flags
                    if flags is not None:
                        g, nx = gen, bool(flags[b, t])
                    gen = nx
                    counts[b, t] = by_label[seg.feed(x, g, nx, table)]
                new.doc[b], new.role[b], new.seg[b], new.gen[b] = last, role, seg, gen
        if s.unscored is not None and scored is not None:
            unscored = ~np.asarray(scored, dtype=bool)
            if dec_mask is not None:
                unscored &= np.asarray(dec_mask, dtype=bool)
            counts = np.where(unscored, np.int8(s.unscored), counts)
        return torch.from_numpy(np.ascontiguousarray(counts)), new


class PassPlanner:
    """Counts per position -> the fixed-shape plan of one forward (pass_schedule.build_plan) for an arch whose dense
    pass runs at the first `capacity` decoder positions of each row (capacity.select_slots: CED v2). multiple: the
    slot sizes are multiples of it (decoder.capacity_multiple); fractions: TrainConfig.pass_slot_buckets."""

    def __init__(self, multiple: int, fractions=BUCKET_FRACTIONS):
        self.multiple = int(multiple)
        self.fractions = tuple(float(f) for f in fractions)
        self._sizes: dict[int, list[int]] = {}

    def sizes(self, capacity: int) -> list[int]:
        if capacity not in self._sizes:
            self._sizes[capacity] = bucket_sizes(capacity, self.multiple, self.fractions)
        return self._sizes[capacity]

    def plan(self, pass_counts: torch.Tensor, dec_mask: torch.Tensor, capacity: int, row_gather: bool = False) -> PassPlan:
        """pass_counts, dec_mask (B, L) host tensors -> the plan over the capacity decoder slots of each row."""
        slots = select_slots(dec_mask, capacity)
        counts = unit_counts(pass_counts, slots.index, slots.valid)
        assert not bool(((counts == 0) & slots.valid).any()), "a decoder position without a pass count"
        return build_plan(counts, self.sizes(capacity), row_gather=row_gather)

    def plan_positions(self, pass_counts: torch.Tensor, row_gather: bool = False) -> PassPlan:
        """pass_counts (B, L) host tensor -> the plan for an arch whose dense first pass runs at EVERY position of a
        row (the control with a second pass through its top block, ced/model/control.py: a unit is a position,
        C = L). The slot sizes are the same fractions of L, in multiples of `multiple`."""
        counts = pass_counts.to(torch.long)
        assert bool((counts >= 1).all()), "a position without a pass count"
        return build_plan(counts, self.sizes(counts.shape[1]), row_gather=row_gather)
