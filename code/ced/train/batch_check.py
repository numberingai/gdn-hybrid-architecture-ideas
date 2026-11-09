"""Loader-side validation of a training batch and of its varlen boundaries, on the host, before anything reaches a kernel.

Why. The kernel gates (a GPU run) found that the kernels take, without an
error:
- a head target id outside [0, V): id V gives ce = inf; id -1 or -3 gives ce = lse (the target term silently dropped);
- a cu_seqlens that leaves the last tokens uncovered, starts after 0 or decreases: outputs unwritten or wrong.

What is checked, and where:
- check_batch (Trainer.batch_at, every training batch, every data source: the shard packer, a shape schedule's rows, TST's
  bag rows and its multi-hot targets, synthetic data):
  * shapes: tokens (B, L), or (B, L, s) in a TST bag phase; targets, target_valid, doc_ids, dec_mask (B, L); aux_mask
    and loss_mask (B, L) or None; multi_targets and multi_valid (B, L, K) or None;
  * dtypes: ids are integer tensors, masks are bool;
  * every input id and every target id (targets and, with TST, multi_targets: each of a position's K targets) in [0, V),
    V = model.vocab_size. Our batches have NO ignore value: a target that does not count is marked by target_valid /
    multi_valid, and its id is still a real token of the stream (the next document's first token). The loss reads ids it
    does not count (the chunked head gathers at every position; the fused head at the selected ones), so an id outside
    [0, V) is refused at every position, counted or not;
  * doc_ids >= 0 and non-decreasing along every row: the contract of ced/model/sequence.py sequence_context ("document
    ids increase along a row ... and are never reused in the same row"). Then the row's segments, which are its
    cu_seqlens (ced/train/varlen.py _boundaries: a start at the row's position 0 and at every id change), cover exactly
    [0, L] with strictly increasing boundaries, and the attention's id-equality mask and the GDN's document starts agree.
    A negative id is the empty-slot marker (-1) of ced/train/varlen.py slot_docs and the "no previous document" value of
    sequence_context.
- check_cu_seqlens (ced/train/varlen.py varlen_for_batch, every cu_seqlens the fused GDN kernels get: the encoder's over
  the flattened B x L positions, CED v1's decoder slots' over B x C): a 1-D int32 array; first 0; last rows x N; strictly
  increasing; every row start r x N a boundary (no segment crosses from one row into the next).

How. Whole-tensor reductions on the host tensors (aminmax, all over a comparison, searchsorted at the row starts): fixed
shapes, no boolean-mask indexing and no nonzero. Only when a check fails are the row and the position located (argmax
over per-row flags), for the message. Nothing is read from or sent to the device, and no tensor is changed: a batch that
passes is the same object, so a run is the run it was, bit for bit. A failure raises BatchError, whose `info` names the
field, the rule, the row, the position and the value; the trainer adds the step and the row's place in the data stream
(Trainer.check_batch) and halts the run through the stop's report path (Trainer.fit, "bad_batch").
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

INT_DTYPES = (torch.int64, torch.int32, torch.int16, torch.uint8, torch.int8)


class BatchError(ValueError):
    """A training batch (or its varlen boundaries) broke the loader's contract. `info`: {"field", "rule", "row",
    "position", "value", ...}, plain JSON values; the trainer adds the step and the row's identity."""

    def __init__(self, message: str, info: dict):
        super().__init__(message)
        self.info = info


def _host(t: torch.Tensor) -> np.ndarray:
    """The host tensor's numpy view (no copy). The reductions run in numpy, on the calling thread: torch's would wake its
    intra-op thread pool for a few hundred thousand elements, which on a busy host costs a scheduler slice (measured on
    the control VM: 1 to 27 ms a batch with torch)."""
    return t.detach().cpu().numpy()


def _first_true(flags: np.ndarray) -> tuple[int, int]:
    """(row, flat position within the row) of the first True of a (B, ...) bool array that has one, by argmax over
    per-row flags (fixed shapes; only called on a failure)."""
    f = flags.reshape(flags.shape[0], -1)
    r = int(np.argmax(f.any(axis=1)))
    return r, int(np.argmax(f[r]))


def _fail(field: str, rule: str, row: int | None = None, position=None, value=None, **more) -> None:
    info = {"field": field, "rule": rule, "row": row, "position": position, "value": value, **more}
    where = "" if row is None else f" at row {row}" + ("" if position is None else f", position {position}")
    shown = "" if value is None else f" (value {value!r})"
    raise BatchError(f"batch {field}{where}: {rule}{shown}", info)


def _check_ids(field: str, t: torch.Tensor, vocab_size: int) -> None:
    """Every id of t (B, L) or (B, L, K) in [0, vocab_size)."""
    if t.dtype not in INT_DTYPES:
        _fail(field, f"ids must be an integer tensor, not {t.dtype}")
    if t.numel() == 0:
        return
    a = _host(t)
    if int(a.min()) >= 0 and int(a.max()) < vocab_size:
        return
    r, p = _first_true((a < 0) | (a >= vocab_size))
    value = int(a.reshape(a.shape[0], -1)[r, p])
    position = p if t.dim() == 2 else [p // t.shape[2], p % t.shape[2]]  # (position, k) of a (B, L, K) tensor
    _fail(field, f"id outside [0, {vocab_size})", r, position, value)


def _check_mask(field: str, t: torch.Tensor | None, shape: tuple) -> None:
    if t is None:
        return
    if tuple(t.shape) != shape:
        _fail(field, f"shape {tuple(t.shape)}, expected {shape}")
    if t.dtype != torch.bool:
        _fail(field, f"a mask must be bool, not {t.dtype}")


def check_doc_ids(doc: torch.Tensor, field: str = "doc_ids") -> None:
    """doc (B, L) integer: >= 0 and non-decreasing along every row (module docstring)."""
    if doc.dtype not in INT_DTYPES:
        _fail(field, f"document ids must be an integer tensor, not {doc.dtype}")
    if doc.dim() != 2 or doc.shape[1] == 0 or doc.shape[0] == 0:
        _fail(field, f"shape {tuple(doc.shape)}, expected (B, L) with B, L >= 1")
    d = _host(doc)
    if int(d[:, 0].min()) < 0:  # with the rows non-decreasing (below), the first id is each row's smallest
        r = int(np.argmin(d[:, 0]))
        _fail(field, "negative document id at the row's start (-1 marks an empty slot; ids are >= 0)", r, 0, int(d[r, 0]))
    if d.shape[1] > 1:
        steps = d[:, 1:] >= d[:, :-1]
        if not bool(steps.all()):
            r, p = _first_true(~steps)
            _fail(field, f"document id decreases from position {p} to {p + 1} ({int(d[r, p])} -> {int(d[r, p + 1])}): ids "
                         "must not decrease along a row, or the row's segments (its cu_seqlens) would not cover it in order",
                  r, p + 1, int(d[r, p + 1]))


def check_batch(batch, vocab_size: int) -> None:
    """The contract of a training batch (ced/train/data.py Batch; module docstring). Raises BatchError; returns None and
    changes nothing when the batch is valid."""
    tok = batch.tokens
    if tok.dim() not in (2, 3):
        _fail("tokens", f"shape {tuple(tok.shape)}, expected (B, L) or (B, L, s)")
    B, L = int(tok.shape[0]), int(tok.shape[1])
    for field in ("targets", "doc_ids"):
        t = getattr(batch, field)
        if tuple(t.shape) != (B, L):
            _fail(field, f"shape {tuple(t.shape)}, expected {(B, L)}")
    for field in ("target_valid", "dec_mask", "aux_mask", "loss_mask"):  # loss_mask: None without DataConfig.loss_mask
        _check_mask(field, getattr(batch, field), (B, L))
    mt, mv = batch.multi_targets, batch.multi_valid
    if (mt is None) != (mv is None):
        _fail("multi_targets", "multi_targets and multi_valid come together")
    if mt is not None:
        if mt.dim() != 3 or tuple(mt.shape[:2]) != (B, L):
            _fail("multi_targets", f"shape {tuple(mt.shape)}, expected {(B, L)} + (K,)")
        _check_mask("multi_valid", mv, tuple(mt.shape))
    _check_ids("tokens", tok, vocab_size)
    _check_ids("targets", batch.targets, vocab_size)
    if mt is not None:
        _check_ids("multi_targets", mt, vocab_size)
    check_doc_ids(batch.doc_ids)


def check_cu_seqlens(cu, rows: int, n: int, name: str = "enc") -> None:
    """The boundaries a fused GDN kernel gets for `rows` rows of `n` positions flattened (module docstring). cu: a CPU
    torch tensor or a numpy array."""
    c = cu.numpy() if isinstance(cu, torch.Tensor) else np.asarray(cu)
    field = f"cu_seqlens[{name}]"
    if c.ndim != 1 or c.dtype != np.int32:
        _fail(field, f"a 1-D int32 array, not {c.dtype} of shape {c.shape}")
    total = rows * n
    if total >= 2**31:
        _fail(field, f"{rows} x {n} positions do not fit int32 boundaries")
    if len(c) < rows + 1:
        _fail(field, f"{len(c)} boundaries for {rows} rows: every row starts a segment, so at least {rows + 1}")
    if int(c[0]) != 0:
        _fail(field, f"starts at {int(c[0])}, not 0: positions 0 .. {int(c[0]) - 1} would be in no segment", 0, 0, int(c[0]))
    if int(c[-1]) != total:
        last = int(c[-1])
        row = min(max(last, 0) // n, rows - 1) if n else 0
        _fail(field, f"ends at {last}, not {rows} x {n} = {total}: the segments do not cover exactly the rows' positions",
              row, last - row * n, last)
    d = np.diff(c)
    if not bool((d > 0).all()):
        k = int(np.argmax(d <= 0))  # boundaries k and k + 1: not strictly increasing
        row = min(max(int(c[k]), 0) // n, rows - 1) if n else 0
        _fail(field, f"boundary {k + 1} ({int(c[k + 1])}) is not above boundary {k} ({int(c[k])}): boundaries must "
                     "strictly increase (a segment of length 0 or less)", row, int(c[k]) - row * n, int(c[k + 1]))
    starts = np.arange(rows, dtype=np.int64) * n
    at = np.minimum(np.searchsorted(c, starts), len(c) - 1)
    ok = c[at] == starts
    if not bool(ok.all()):
        r = int(np.argmax(~ok))
        _fail(field, f"row {r}'s start ({r} x {n} = {int(starts[r])}) is not a boundary: a segment would cross from row "
                     f"{r - 1} into row {r}", r, 0, int(starts[r]))


def row_hashes(batch) -> list[str]:
    """A short content hash of every row of the batch (blake2b-64 over the row's tokens, targets, document ids, decoder
    and aux masks and, with TST, its multi-hot targets), for a halt record: two runs read the same row iff the hashes
    agree. Host work over the batch; only called when a run halts."""
    fields = [t for t in (batch.tokens, batch.targets, batch.doc_ids, batch.dec_mask, batch.aux_mask, batch.multi_targets)
              if t is not None]
    out = []
    for b in range(int(batch.tokens.shape[0])):
        h = hashlib.blake2b(digest_size=8)
        for t in fields:
            h.update(t[b].contiguous().numpy().tobytes())
        out.append(h.hexdigest())
    return out
