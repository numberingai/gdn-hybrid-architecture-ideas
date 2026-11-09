"""Activation checkpointing of the layers of a training forward (TrainConfig.activation_checkpoint; the long-context
grid).

Off by default ("none"): every layer call is the call it always was. "layer": in a training forward (grad enabled,
no cache) each encoder layer and each CED v2 decoder layer runs under torch.utils.checkpoint (non-reentrant): the
tensors a layer saves for its backward are dropped as the forward goes, and the layer is run a second time, from its
kept input, when the backward reaches it. The function and its gradients are the same; the cost is one more forward
of every layer, the gain is that only one layer's saved tensors exist at a time instead of all layers'.

Why it is needed (measured): the saved tensors are about 27 kB per token and layer at 60M, so a micro-batch of one
16,384-token row peaks at 10.96 GiB, one 32,768-token row at 19.32 GiB (of 21.62), and one 65,536-token row would
need about 36 GiB.

What is kept per checkpointed layer: its input (B, L, d) and references to its other arguments (the sequence
context, the memory KeySet); what a layer returns besides the stream (an encoder layer's cache entry) stays alive as
before. No RNG is drawn inside a layer, so the RNG state is neither saved nor restored (preserve_rng_state False).

"host" (option B): "layer", and each layer's kept input lives in host memory instead of on the device. With "layer",
the kept inputs of all layer calls of a forward stay on the device until the backward reaches them: 40 calls of a (1,
262,144, 2,560) fp32 stream are 100 GiB at 256K (exact). With "host":
- in the forward, torch.utils.checkpoint saves the layer's arguments through torch.autograd.graph.saved_tensors_hooks.
  The hook that run_layer puts around the call copies the stream x into a host buffer of a HostPool and hands autograd
  a handle (_Kept) in its place. Nothing holds the device tensor any more once the layer has run, so the device frees
  it. The other arguments (the context, the memory keys, records) pass through untouched and are kept by reference, as
  in "layer";
- in the backward, when the recomputation of the layer asks for its input, the handle copies the buffer back into a
  new device tensor, and the layer is recomputed from it. A copy keeps the bits: the recomputed layer is the same
  layer, and losses, gradients and parameters are those of "layer" and "none" bit for bit
  (tests/test_host_inputs.py);
- both copies block (copy_ with non_blocking False): the device is idle while a copy runs. "host_overlap" runs
  them beside the compute instead;
- the buffers are made on first use, one per layer call of a forward, pinned when the stream is on a CUDA device,
  and reused by every later forward of the same shapes (with one row shape nothing is allocated after the first
  micro-batch). A buffer goes back to the pool when its input has been copied back, or when autograd drops the handle
  (a layer the backward never reaches). Buffers are kept per shape and never freed: a run whose micro-batch shapes
  change (a shape schedule's phases, micro_batch_by_t with different row counts, the pass schedule's slot buckets)
  pins the sum over its shapes. The pool counts the bytes it requests; torch 2.14.1's pinned allocator rounds each
  request up to a power of two (2.5 GiB takes 4 GiB), unless PYTORCH_ALLOC_CONF sets pinned_max_round_threshold_mb.
  The trainer opens the pool around each training forward (host_inputs, Trainer.train_step); the mode on the modules
  and the open pool must agree (active() raises otherwise), so a "host" forward never silently runs as "layer" or the
  reverse. The model, the state dict, the checkpoint files and the loss are those of the other modes.

"host_overlap": "host" with the copies on a side stream, beside the compute. The same hooks, pool, buffers and
guards; the same training bit for bit (a copy is exact on whatever stream it runs). What changes
(HostPool(overlap=True); the side stream is _CudaLane on CUDA, _EmuLane elsewhere):
- forward: the copy of layer k's input to the host is queued on the side stream (after the side stream has waited
  for the compute stream to reach the layer's start, so x is complete), and runs while layer k computes.
  x.record_stream(side) stops the caching allocator from giving x's memory to another tensor before that copy has
  read it (x is freed as soon as the caller lets go of it, usually before the copy has run);
- backward: when the recomputation of layer k asks for its input, the copy back of layer k - 1's input (the previous
  handle of the same forward) is queued on the side stream into a new device tensor, and runs while layer k
  recomputes and runs its backward (prefetch). Layer k - 1's recomputation then makes the compute stream wait for
  that copy's event (a wait on the device, not on the host) before it reads the tensor. The target is
  record_stream'ed on the side stream too (a prefetched input whose layer the backward never reaches is freed while
  the copy may still run). The top layer's input, the first one the backward asks for, is not prefetched: it is
  copied when asked. A call may keep several inputs or none (C2L, below: R1-entry's closing layer keeps its stream and
  its block's input; R1's block-first layer keeps nothing on the host). The prefetch is per call: the first restore
  of a call's recomputation queues the copies back of every kept input of the call before it in the forward (its
  group, in pack order, which is the order its recomputation asks for them), and a call that keeps nothing on the host
  is passed over (the chain links the kept inputs, not the calls);
- a host buffer goes back to the pool when its copy back has been queued. Every copy runs on the one side stream, in
  queue order, so a later copy into the same buffer cannot pass the copy back that reads it;
- at most `lead` copies (2 by default) are in flight: before it queues another, the host waits for the oldest
  (an event's synchronize; the rate limit of FSDP's limit_all_gathers). Without it the host, which runs ahead of the
  device, would free every x of a forward before the device had copied any, and the allocator, which reuses a
  record_stream'ed block only once the side stream's event has been seen complete, would take new memory for each:
  no saving. With it the device holds about three kept inputs at a time (in use, in flight each way) instead of two;
- the order of the backward is not assumed: a prefetch is only ever started for a handle not yet restored, a
  handle that was not prefetched copies when asked, and one prefetched and never asked for gives its buffer back
  when autograd drops it (counters prefetched, prefetch_hits, prefetch_unused). On the CPU (the tests) the side
  stream is emulated by a queue that runs a copy only when something waits for it: the latest a device may run it. A
  read before its wait sees the poison; a device tensor a queued copy uses that was freed before the copy ran (a
  missing record_stream) is an error; a copy queued without the side stream first waiting for the compute stream is
  an error (tests/test_host_overlap.py).

Not for: the second-pass control and Looped-CED v1 (their layers return records other code reads back), ASEntmax
(its training statistics count every forward of a layer), and anything with a cache (inference). The trainer refuses
those (ced/train/trainer.py); here a layer call with a cache or without grad is simply the plain call. The trainer
refuses "host" and "host_overlap" wherever it refuses "layer".

The compiled layers (TrainConfig.compile "all") are called from inside the checkpoint, so each layer's compiled
graph is what runs both times. Checked on the CPU with aot_eager and Inductor (tests/test_long_context_grid.py,
tests/test_host_inputs.py); before any run uses it; for "host".

"block": one checkpoint per BLOCK, a hybrid group of full_attn_interval layers ([GDN, GDN, GDN, global] at 60M and at
the 3.8B shape), for arch "control" with one pass through the stack, with or without the depth mix
(model.encoder.attn_res "none", "block" = R1, "block_entry" = R1-entry). In a training forward the block's layers run
inside one torch.utils.checkpoint call (non-reentrant, run_block below) and are run a second time, all of them, when
the backward reaches the block. What the call keeps:
- the base: the stream where the block begins (one (B, L, d) tensor per block);
- R1 and R1-entry: references to the block representations b_0 .. b_{n-1} the block reads (a Python list of the very
  tensors the walk holds, not copies; they are alive anyway, because every later block reads them), and nothing of
  the block's own partial sums, entry read or inner stream, which the recomputation forms again.
The function and its gradients are the same bit for bit (non-reentrant checkpointing leaves the autograd graph as it
is and the recomputed saved tensors equal the dropped ones). The cost is one more forward of every layer, as "layer";
the peak is one BLOCK's saved tensors during its recomputation and backward (4 layers' worth), instead of one layer's.
Monitoring between the layers of a block (the stream-RMS log, the activation maxima) is read in the block's first
forward only (FirstForward below), never again in the recomputation.
Refused (ced/train/trainer.py) for every other form: ced_v2, the second pass, the looped middle, a looped encoder,
the residual scalars, Canon layers, ASEntmax, NVFP4, and a layout that is not a whole number of blocks.

"layer", "host" and "host_overlap" with the depth mix (-layer checkpoints inside a block, for 128K and 256K rows):
the stack of the depth mix
(attn_res.BlockAttnRes.stack) runs each LAYER of a block as its own checkpointed call (Walk.run / EntryWalk.run with
ckpt), not nested inside a block checkpoint (nesting would run each layer's forward a third time). The calls, their
order and the compiled graphs are those of the stack without checkpointing. What each layer call keeps:
- R1 ("block"): the block's partial sum where the layer begins (moved to the host under "host"), and references to the
  finished blocks b_0 .. b_{n-1}. In a block's first layer the input is b_{n-1} itself: held by reference, never moved.
- R1-entry ("block_entry"; its inner layers are the base's): the plain stream where the layer begins (moved under
  "host"); the block's closing layer also keeps the block's input `start`, moved like the stream (run_kept: both are
  top-level arguments of the call). A finished block (the input of a block's opening layer; b_0 as block 1's input
  and start) is held by reference, never moved.
- the base: unchanged ("layer", "host" and "host_overlap" as above). Under "host_overlap" (C1b) the moved inputs take
the side stream as the base's do: a call moves none, one or two of them, and the prefetch queues all of the previous
call's (above). Merged with C1b on 2026-10-15; R1 and R1-entry under "host_overlap" are tested against no
checkpointing bit for bit in tests/test_c2l_layer_in_block.py. The sources' gradients are not touched: each layer's
backward makes them as it does without checkpointing. Built for the forms the depth mix is built for
(attn_res.check_attn_res: arch "control", one pass, whole blocks); the trainer refuses the rest by name, as before.

The architecture arms (2026-10-16; section 13; tests/test_arch_arms_ckpt.py):
- the GDN-side arms (ced/model/gdn_arms.py): "layer" and "block" with A and CCR, "host" and "host_overlap" with A. A
  layer call keeps what R1-entry's keeps and holds A's sources (finished blocks) and CCR's incoming record by
  reference; a layer's own record is an output of its checkpointed call (or of its block's), never side state, so the
  recomputation forms it for its own saved tensors only. CCR with a host mode is refused (its record would stay on the
  device);
- LDM (attn_res "ldm"): "layer" (its call keeps the stream and the memory M) and "block" (the block keeps both where it
  begins); the host modes refused (M would stay on the device);
- R1-entry's token queries and heads: every mode; under "host" / "host_overlap" an opening layer with a token query
  also moves `end`, the stream where the block before ended (a top-level argument kept by its checkpoint alone, as the
  closing layer's block input).
"""

from __future__ import annotations

import collections
import contextlib
import time
import weakref

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils.checkpoint import checkpoint

MODES = ("none", "layer", "block", "host", "host_overlap")
CHECKPOINTED = ("layer", "host", "host_overlap")  # the modes whose layer calls run under torch.utils.checkpoint ("block" checkpoints whole blocks: run_block)
HOST = ("host", "host_overlap")  # the modes whose kept layer inputs live in a HostPool (overlap: "host_overlap")

_open: list = []  # the HostPool of the training forward running now (host_inputs); empty otherwise


def set_activation_checkpoint(model: torch.nn.Module, mode: str) -> int:
    """Set the mode on every module of `model` that has an `activation_checkpoint` attribute (the encoder, CED v2);
    returns how many were set. A plain attribute: never in the state dict or the model config."""
    if mode not in MODES:
        raise ValueError(f"activation_checkpoint {mode!r}: one of {MODES}")
    n = 0
    for m in model.modules():
        if hasattr(m, "activation_checkpoint"):
            m.activation_checkpoint = mode
            n += 1
    return n


def open_pool() -> HostPool | None:
    """The HostPool of the training forward running now, or None (no host_inputs block is open)."""
    return _open[-1] if _open else None


@contextlib.contextmanager
def host_inputs(pool: HostPool | None):
    """Around one training forward of a model in mode "host" or "host_overlap" (Trainer.train_step): every checkpointed
    layer call inside moves its kept input to `pool`. With None, nothing (the modes "none" and "layer"). The backward
    needs no open block: each kept input carries its own way back."""
    if pool is None:
        yield
        return
    if _open:
        raise RuntimeError("activation_checkpoint 'host': a host_inputs block is already open (one training forward "
                           "at a time)")
    _open.append(pool)
    pool._last = None  # the prefetch chain ("host_overlap") is per forward
    try:
        yield
    finally:
        pool._last = None
        _open.pop()


def active(mode: str, x: torch.Tensor) -> bool:
    """Whether a layer call on the stream x is checkpointed: the mode is on, grad is enabled and x is part of a
    graph (a training forward). Evaluation and cached inference run under no_grad and never are. A checkpointed call
    in mode "host" or "host_overlap" needs an open pool (host_inputs) of that kind (HostPool.overlap) and one in mode
    "layer" must not have one: a mismatch raises. The per-layer form only: under "block" no single layer call is
    checkpointed (block_active)."""
    if mode not in CHECKPOINTED or not torch.is_grad_enabled() or not x.requires_grad:
        return False
    if (mode in HOST) != bool(_open):
        raise RuntimeError(f"activation_checkpoint {mode!r} in a training forward " +
                           ("without an open host pool: run the forward inside act_checkpoint.host_inputs(pool) "
                            "(Trainer.train_step does)" if mode in HOST else
                            "inside an open host pool (host_inputs): the model's mode is not 'host' or 'host_overlap'"))
    if _open and _open[-1].overlap != (mode == "host_overlap"):
        raise RuntimeError(f"activation_checkpoint {mode!r} in a training forward inside an open host pool with "
                           f"overlap {_open[-1].overlap}: 'host' takes HostPool(), 'host_overlap' HostPool(overlap=True)")
    return True


def block_active(mode: str, x: torch.Tensor) -> bool:
    """Whether the stack runs its blocks checkpointed ("block"): as active, for the block form."""
    return mode == "block" and torch.is_grad_enabled() and x.requires_grad


def run_layer(layer, x: torch.Tensor, *args, **kwargs):
    """layer(x, *args, **kwargs) under torch.utils.checkpoint: same outputs, same gradients. x is the layer's
    stream input (the tensor that is kept); the other arguments are passed through as they are. With an open host
    pool (mode "host"), x is kept in the pool's host memory until the backward reaches the layer."""
    return run_kept(layer, (x,), x, *args, **kwargs)


def run_kept(fn, moved: tuple, *args, **kwargs):
    """fn(*args, **kwargs) under torch.utils.checkpoint (non-reentrant, no RNG state), with its kept inputs named:
    `moved` holds the top-level tensor arguments of the call that nothing but this checkpoint keeps alive (a layer's
    stream input; under the depth mix also R1-entry's block input, C2L). With an open host pool (mode "host" or
    "host_overlap") those are kept in the pool's host memory until the backward reaches the call. Every other argument
    is held by reference, as in "layer": a finished block of the depth mix, which the stack holds on the device anyway
    (moving it would free nothing and cost two copies), the context, a list of tensors. () moves nothing."""
    pool = open_pool()
    if pool is None or not moved:
        return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)
    with pool.offload(*moved):
        return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)


def run_block(fn, *args):
    """fn(*args) under torch.utils.checkpoint (non-reentrant, no RNG state): one block of the stack ("block"). The
    arguments are held by reference until the backward has run the block again: tensors and lists of tensors are
    not copied, so a list of block representations costs nothing beyond the tensors themselves."""
    return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)


class FirstForward:
    """Monitoring inside a checkpointed block: `on` while the block's first forward runs; the caller sets it off
    when run_block returns, so the recomputation in the backward (the same function with the same arguments, this
    object among them) reads and records nothing a second time."""

    def __init__(self):
        self.on = True


def _deref(ref):
    """A weak reference's object, or None (no reference, or its object is gone)."""
    return ref() if ref is not None else None


def _same_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    """a is b, or a view of exactly b's elements (the pack hook may be handed an alias of the argument)."""
    return a is b or (a.device == b.device and a.dtype == b.dtype and a.shape == b.shape and a.stride() == b.stride()
                      and a.data_ptr() == b.data_ptr())


def _dense(t: torch.Tensor) -> bool:
    """Whether t's elements fill a block of memory of exactly t.numel() elements, in some order of its dimensions
    (row-major, or e.g. transposed): then a buffer with t's strides holds it with nothing in between."""
    if t.is_contiguous():
        return True
    expect = 1
    for stride, size in sorted((st, n) for st, n in zip(t.stride(), t.shape) if n != 1):
        if stride != expect:
            return False
        expect *= size
    return True


def _poison(t: torch.Tensor) -> None:
    with torch.no_grad():
        t.fill_(float("nan") if t.is_floating_point() or t.is_complex() else -1)


def _probe(t: torch.Tensor) -> None:
    """(reuse_probe) A tensor of t's size and layout made on the compute stream and filled with NaN at once, then
    dropped: it takes memory the allocator holds free now. A device tensor whose queued copy has not run yet must not
    be among it (record_stream); if it were, that copy would read or leave NaN."""
    with torch.no_grad():
        _poison(torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device))


class HostPool:
    """Host buffers for the kept layer inputs of modes "host" and "host_overlap" (module docstring): made on first use,
    pinned for a CUDA stream, reused by every later forward of the same shapes. Counters (stats()): buffers made and
    their bytes, buffers in use now and at most, inputs moved, inputs copied back, inputs left on the device (a stream
    whose elements do not fill its memory densely, e.g. a strided slice: kept as in "layer"; never seen in our layers).
    A dense stream in another order than row-major (a transposed one) is moved with its strides, so it comes back in the
    same layout.

    overlap (mode "host_overlap"): the copies run on a side stream beside the compute (_CudaLane on CUDA, _EmuLane
    elsewhere), with at most `lead` copies in flight before the host waits for the oldest (module docstring). More
    counters then: prefetched (copies back started by the restore of the layer above), prefetch_hits (restores that
    found their copy started), prefetch_unused (prefetched inputs whose layer the backward never reached), cpu_waits
    (times the host waited for a copy before it queued another), and with track most_pending (at a copy to the host,
    the most earlier kept inputs of the pool whose device storage was still alive: on the CPU, those whose emulated
    copy had not run yet).

    Checks for the tests and the GPU job, off in runs:
      poison          a buffer is filled with NaN when it goes back to the pool, and the device tensor a kept input is
                      copied back into is filled with NaN before the copy: a missed or partial copy-back, or a buffer
                      read after it was given back (by a later layer or a later step), shows as NaN. With overlap the
                      returned buffer is filled through the side stream (a copy from a NaN device tensor queued after
                      the copy back that reads it), so the host never waits for it and never writes a buffer a copy
                      is reading.
      poison_sources  the device-side stream x is filled with NaN right after its layer's forward (the moment the
                      device may reuse its memory): a recomputation that read the device copy instead of the host copy
                      shows as NaN. Only where nothing else reads x after its layer: the base's stack, CED v2 and the
                      two-pass models do not; the residual scalars' x0 and the loop input mix's p are read again by
                      design (tests/test_host_inputs.py). With overlap the compute stream first waits for x's copy to
                      the host (the allocator, too, reuses x's memory only after it).
      track           counts the kept inputs whose device storage is still alive when they are copied back
                      (alive_at_restore): 0 means the device really let go of every one of them.
      timing          the time of every copy (CUDA events on the device, on the side stream with overlap; the host
                      clock on the CPU): copy_stats().
      reuse_probe     (overlap) before each copy to the host, _probe on the compute stream: with a missing
                      record_stream the probe could take the memory of an x whose copy has not run, and fill it with
                      NaN first. Meaningful on CUDA; the CPU's emulated lane catches a missing record_stream itself.
      delay_cycles    (overlap, CUDA) the side stream spins this many GPU clock cycles (torch.cuda._sleep) before each
                      copy: every copy runs late, so a read that does not wait for its copy, or a reuse of memory a copy
                      is not protected against, meets it while it is still pending.
    """

    def __init__(self, *, poison: bool = False, poison_sources: bool = False, track: bool = False,
                 timing: bool = False, overlap: bool = False, lead: int = 2, reuse_probe: bool = False,
                 delay_cycles: int = 0):
        if lead < 1:
            raise ValueError(f"HostPool lead {lead}: at least 1 copy in flight")
        self.poison, self.poison_sources, self.track, self.timing = poison, poison_sources, track, timing
        self.overlap, self.lead, self.reuse_probe, self.delay_cycles = overlap, lead, reuse_probe, delay_cycles
        self._free: dict[tuple, list[torch.Tensor]] = {}
        self._copies: list[tuple] = []
        self._lane = None  # (overlap) the side stream, made at the first copy (_lane_for)
        self._inflight: collections.deque = collections.deque()  # (overlap) events of the copies queued, oldest first
        self._nan: dict[tuple, torch.Tensor] = {}  # (overlap, poison) a NaN device tensor per buffer key
        self._last = None  # (overlap) a weak reference to the last handle of the forward running now (prefetch chain)
        self._live = weakref.WeakSet() if (overlap and track) else None  # (overlap, track) handles alive
        self.buffers = 0  # host buffers made
        self.bytes = 0  # their size
        self.in_use = 0  # buffers holding a kept input now
        self.most_in_use = 0
        self.moved = 0  # kept inputs copied to the host
        self.restored = 0  # kept inputs copied back to the device
        self.kept_on_device = 0  # kept inputs left on the device (not dense)
        self.alive_at_restore = 0  # (track) kept inputs whose device storage was alive when copied back
        self.prefetched = 0  # (overlap) copies back started by the restore of the layer above
        self.prefetch_hits = 0  # (overlap) restores that found their copy back started
        self.prefetch_unused = 0  # (overlap) prefetched inputs given back without being asked for
        self.cpu_waits = 0  # (overlap) times the host waited for the oldest copy in flight before queuing another
        self.most_pending = 0  # (overlap, track) the most earlier kept inputs alive on the device at a copy to the host

    def stats(self) -> dict:
        keys = ("buffers", "bytes", "in_use", "most_in_use", "moved", "restored", "kept_on_device", "alive_at_restore")
        if self.overlap:
            keys += ("prefetched", "prefetch_hits", "prefetch_unused", "cpu_waits") + (("most_pending",) if self.track else ())
        return {k: getattr(self, k) for k in keys}

    @contextlib.contextmanager
    def offload(self, *moved: torch.Tensor):
        """Around one torch.utils.checkpoint call whose kept inputs are `moved` (its stream input x; C2L: also
        R1-entry's block input): the saved-tensor hooks that move them to this pool and bring them back. Every other
        saved argument passes through untouched. Inside the checkpointed forward torch.utils.checkpoint's own hooks are
        on top, so the layer's own saved tensors never reach these."""
        # Autograd keeps the hook functions as long as it keeps what they packed (until the backward has passed the
        # layer), so pack must not hold x itself: it reads x through this list, which is emptied when the call returns.
        stream, sources = list(moved), []
        group = []  # (overlap) this call's handles, weak, in pack order: its recomputation asks for all of them

        def pack(t):
            if not stream or not any(_same_tensor(t, s) for s in stream):
                return t  # a shared argument: kept by reference, as in mode "layer"
            if not _dense(t):
                self.kept_on_device += 1
                return t
            kept = _Kept(self, t)
            if self.overlap:  # the prefetch chain: the backward is expected to ask for the previous call's handles next
                kept.prev, self._last = self._last, weakref.ref(kept)
                kept.group = group
                group.append(self._last)
            if self.poison_sources:
                sources.append((t, kept))
            return kept

        def unpack(obj):
            return obj.restore() if isinstance(obj, _Kept) else obj

        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                yield
            for t, kept in sources:  # the layer's forward has run: from here on the device may reuse x's memory
                self._poison_source(t, kept)
        finally:
            stream.clear()
            sources.clear()

    def _poison_source(self, t: torch.Tensor, kept: _Kept) -> None:
        """(poison_sources) Fill the device-side x with NaN. With overlap, after its copy to the host has read it: the
        compute stream waits for that copy's event first, as the allocator would before giving x's memory away."""
        if kept.sent is not None:
            self._lane.compute_waits(kept.sent)
        _poison(t)

    def _lane_for(self, device: torch.device) -> _CudaLane | _EmuLane:
        lane = self._lane
        if lane is None:
            lane = self._lane = _CudaLane(device) if device.type == "cuda" else _EmuLane(device)
        elif lane.device != device:
            raise RuntimeError(f"activation_checkpoint 'host_overlap': one device per HostPool ({lane.device}, not {device})")
        return lane

    def _make_room(self, lane) -> None:
        """(overlap) Before another copy is queued: forget the copies known to be done, then wait on the host for the
        oldest until fewer than `lead` are in flight (module docstring: what this bounds)."""
        q = self._inflight
        while q and lane.done(q[0]):
            q.popleft()
        while len(q) >= self.lead:
            lane.cpu_waits(q.popleft())
            self.cpu_waits += 1

    def drain(self) -> None:
        """(overlap) Wait on the host for every copy queued so far (the tests, before they read a buffer; the GPU job
        after its steps). Nothing to do without overlap: every copy has finished when it returns."""
        if self._lane is not None:
            self._lane.cpu_waits(self._lane.mark())
            self._inflight.clear()

    def _take(self, key: tuple, pin: bool) -> torch.Tensor:
        free = self._free.get(key)
        if free:
            buf = free.pop()
        else:
            shape, stride, dtype, _ = key
            buf = torch.empty_strided(shape, stride, dtype=dtype, device="cpu", pin_memory=pin)
            self.buffers += 1
            self.bytes += buf.nbytes
        self.in_use += 1
        self.most_in_use = max(self.most_in_use, self.in_use)
        return buf

    def _give(self, key: tuple, buf: torch.Tensor) -> None:
        """A buffer back to the pool. With overlap its copy back may still be queued: the side stream runs every copy
        in queue order, so the next copy into this buffer (queued later) cannot pass it, and the poison goes through
        the side stream too."""
        if self.poison:
            if self._lane is not None:
                self._poison_on_lane(key, buf)
            else:
                _poison(buf)
        self._free.setdefault(key, []).append(buf)
        self.in_use -= 1

    def _poison_on_lane(self, key: tuple, buf: torch.Tensor) -> None:
        nan = self._nan.get(key)
        if nan is None:
            shape, stride, dtype, device = key
            nan = self._nan[key] = torch.empty_strided(shape, stride, dtype=dtype, device=device)
            _poison(nan)
        self._lane.side_waits_compute()
        self._lane.to_host(buf, nan)  # nan lives in the pool: nothing to protect

    def _start(self, device: torch.device, lane=None):
        if not self.timing:
            return None
        if device.type == "cuda":
            ev = torch.cuda.Event(enable_timing=True)
            ev.record(lane.side if lane is not None else None)
            return ev
        return time.perf_counter()

    def _stop(self, direction: str, nbytes: int, start, device: torch.device, lane=None) -> None:
        if start is None:
            return
        if device.type == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record(lane.side if lane is not None else None)
        else:
            end = time.perf_counter()  # with the emulated lane the copy has only been queued: bytes, not seconds
        self._copies.append((direction, nbytes, start, end))

    def copy_stats(self) -> dict:
        """(timing) {"out": [bytes, seconds], "back": [bytes, seconds]}: the copies to the host and back since the
        last call, with their summed durations (CUDA events: the transfers alone, not the wait for the work queued
        before them). Empty lists' worth of zeros without timing."""
        out = {"out": [0, 0.0], "back": [0, 0.0]}
        for direction, nbytes, start, end in self._copies:
            if isinstance(start, float):
                seconds = end - start
            else:
                end.synchronize()
                seconds = start.elapsed_time(end) / 1000.0
            out[direction][0] += nbytes
            out[direction][1] += seconds
        self._copies = []
        return out


class _Kept:
    """One kept layer input in a HostPool buffer: what autograd holds in its place between the forward and the
    backward (the saved-tensor hooks' packed object). With overlap also: the event of its copy to the host (sent), the
    previous handle of its forward (prev, weak), the handles of its own checkpoint call (group: weak references in pack
    order, shared by the call's handles; None for a handle made outside a call, which is then a call of its own), and
    once its copy back is queued, the device tensor it goes into (target) and that copy's event (back)."""

    __slots__ = ("pool", "key", "buf", "device", "source", "released", "prev", "group", "sent", "back", "target",
                 "__weakref__")

    def __init__(self, pool: HostPool, t: torch.Tensor):
        self.released = True  # until the buffer is taken (a failed allocation gives nothing back)
        self.pool, self.device = pool, t.device
        self.prev = self.group = self.sent = self.back = self.target = None  # prev, group: the pack hook (overlap)
        self.key = (tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
        lane = pool._lane_for(t.device) if pool.overlap else None
        if lane is not None:
            pool._make_room(lane)
            if pool._live is not None:
                pool.most_pending = max(pool.most_pending, sum(
                    1 for h in pool._live if not h.released and h.source is not None and not h.source.expired()))
            if pool.reuse_probe:
                _probe(t)
        self.buf = pool._take(self.key, pin=t.is_cuda)
        self.released = False
        if lane is not None:
            lane.side_waits_compute()  # x is complete on the compute stream
            lane.delay(pool.delay_cycles)
        start = pool._start(t.device, lane)
        self._copy_out(t)
        pool._stop("out", t.nbytes, start, t.device, lane)
        if lane is not None:
            lane.keep(t)  # record_stream: x's memory is not reused before this copy has read it
            self.sent = lane.mark()
            pool._inflight.append(self.sent)
        pool.moved += 1
        self.source = StorageWeakRef(t.untyped_storage()) if pool.track else None
        if pool._live is not None:
            pool._live.add(self)

    def _copy_out(self, t: torch.Tensor) -> None:
        """Device to host. "host": blocking, the host copy is complete when this returns. "host_overlap": queued on
        the side stream, complete at the event self.sent."""
        if self.pool.overlap:
            self.pool._lane.to_host(self.buf, t)
            return
        with torch.no_grad():
            self.buf.copy_(t)

    def _copy_back(self, out: torch.Tensor) -> None:
        """Host to device. "host": blocking, the buffer may be reused when this returns. "host_overlap": queued on the
        side stream, complete at the event self.back."""
        if self.pool.overlap:
            self.pool._lane.to_device(out, self.buf)
            return
        with torch.no_grad():
            out.copy_(self.buf)

    def restore(self) -> torch.Tensor:
        """The kept input, copied back into a new device tensor; the buffer goes back to the pool. With overlap, the
        copy queued by the restore of the layer above when there was one (else queued now), the compute stream made to
        wait for it, and the copies back of the previous call's handles queued (prefetch; _below)."""
        if self.released:
            raise RuntimeError("activation_checkpoint 'host': a kept layer input was asked for after its host buffer "
                               "went back to the pool (a second backward over the same graph, retain_graph, is not "
                               "supported in this mode)")
        pool = self.pool
        if pool.overlap:
            return self._restore_overlapped()
        if self.source is not None and not self.source.expired():
            pool.alive_at_restore += 1
        out = torch.empty_strided(self.buf.shape, self.buf.stride(), dtype=self.buf.dtype, device=self.device)
        if pool.poison:
            _poison(out)
        start = pool._start(self.device)
        self._copy_back(out)
        pool._stop("back", out.nbytes, start, self.device)
        pool.restored += 1
        self.release()
        return out

    def _restore_overlapped(self) -> torch.Tensor:
        pool = self.pool
        if self.back is None:
            self._fetch()
        else:
            pool.prefetch_hits += 1
        self._wait_back()
        out, self.target = self.target, None
        # after the wait: on the CPU the emulated copy to the host has run by now and let go of x
        if self.source is not None and not self.source.expired():
            pool.alive_at_restore += 1
        pool.restored += 1
        self.release()  # the copy back is queued: later copies into this buffer queue behind it
        for h in self._below():
            if not h.released and h.back is None:
                h._fetch()  # runs while this layer is recomputed and its backward runs
                pool.prefetched += 1
        return out

    def _below(self) -> list:
        """(overlap) The handles the backward is expected to ask for after this call's: every kept input of the call
        before it in the forward (its group, in pack order; one handle for a handle made outside a call), reached
        through prev past this call's own earlier handles. A call that kept nothing on the host made no handle and is
        passed over. Clears self.prev: whichever handle of a call is asked for first queues the prefetch; a later one
        of the same call stops at an earlier handle already restored (its prev cleared) or reaches the same handles,
        already queued."""
        prev, self.prev = _deref(self.prev), None
        while prev is not None and self.group is not None and prev.group is self.group:
            prev = _deref(prev.prev)
        if prev is None:
            return []
        if prev.group is None:
            return [prev]
        return [h for h in map(_deref, prev.group) if h is not None]

    def _fetch(self) -> None:
        """(overlap) Queue the copy back on the side stream into a new device tensor (self.target), made on the
        compute stream; its event is self.back."""
        pool = self.pool
        lane = pool._lane
        pool._make_room(lane)
        out = torch.empty_strided(self.buf.shape, self.buf.stride(), dtype=self.buf.dtype, device=self.device)
        if pool.poison:
            _poison(out)
        lane.side_waits_compute()  # out's memory (and its poison) are the compute stream's until here
        lane.delay(pool.delay_cycles)
        start = pool._start(self.device, lane)
        self._copy_back(out)
        pool._stop("back", out.nbytes, start, self.device, lane)
        lane.keep(out)  # record_stream: if out is dropped before the copy has run, its memory waits for the copy
        self.back = lane.mark()
        pool._inflight.append(self.back)
        self.target = out

    def _wait_back(self) -> None:
        """(overlap) The compute stream waits for the copy back (a wait on the device; on the CPU the emulated copy
        runs now)."""
        self.pool._lane.compute_waits(self.back)

    def release(self) -> None:
        if not self.released:
            self.released = True
            buf, self.buf = self.buf, None
            self.pool._give(self.key, buf)

    def __del__(self):  # autograd dropped the handle without asking for it (a layer the backward never reached)
        if not getattr(self, "released", True):
            if getattr(self, "target", None) is not None:
                self.pool.prefetch_unused += 1
            self.release()


class _CudaLane:
    """The side stream of "host_overlap" on a CUDA device (module docstring). Every copy is queued on the one side
    stream, so copies run in queue order; the compute stream is torch.cuda.current_stream at each call (the autograd
    engine runs a backward node on its forward's stream)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.side = torch.cuda.Stream(device)

    def side_waits_compute(self) -> None:
        self.side.wait_stream(torch.cuda.current_stream(self.device))

    def delay(self, cycles: int) -> None:
        if cycles:
            with torch.cuda.stream(self.side):
                torch.cuda._sleep(cycles)

    def to_host(self, buf: torch.Tensor, x: torch.Tensor) -> None:
        with torch.no_grad(), torch.cuda.stream(self.side):
            buf.copy_(x, non_blocking=True)

    def to_device(self, out: torch.Tensor, buf: torch.Tensor) -> None:
        with torch.no_grad(), torch.cuda.stream(self.side):
            out.copy_(buf, non_blocking=True)

    def keep(self, t: torch.Tensor) -> None:
        t.record_stream(self.side)

    def mark(self):
        ev = torch.cuda.Event()
        ev.record(self.side)
        return ev

    def compute_waits(self, ev) -> None:
        torch.cuda.current_stream(self.device).wait_event(ev)

    def cpu_waits(self, ev) -> None:
        ev.synchronize()

    def done(self, ev) -> bool:
        return ev.query()


class _EmuCopy:
    __slots__ = ("host", "device", "strong", "to_host")

    def __init__(self, host: torch.Tensor, device: torch.Tensor, to_host: bool):
        self.host, self.device, self.strong, self.to_host = host, weakref.ref(device), None, to_host


class _EmuLane:
    """The side stream of "host_overlap" off CUDA (the CPU tests): a queue whose copies run only when something waits
    for them (compute_waits, cpu_waits), all earlier ones first: the latest a device may run them. So a read that does
    not wait for its copy sees what was there before (the poison). Like the caching allocator, the queue does not keep
    a device tensor alive: it holds it weakly unless keep (record_stream) was called, and a copy whose device tensor is
    gone when it runs raises (on a device its memory could have been given to another tensor). A copy queued without
    side_waits_compute since the last one raises (on a device it could read x before the compute stream wrote it).
    The compute stream is the host here: what it does has happened when the call returns, so a host write racing a
    copy the compute stream waits for cannot show (the poison of a returned buffer goes through the queue for that
    reason)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.ops: collections.deque = collections.deque()  # queued and not yet run
        self.queued = 0  # ops queued so far: the mark of the latest
        self.ran = 0  # ops run so far
        self._synced = False

    def side_waits_compute(self) -> None:
        self._synced = True

    def delay(self, cycles: int) -> None:
        pass  # every copy already runs as late as it may

    def _queue(self, op: _EmuCopy) -> None:
        if not self._synced:
            raise RuntimeError("activation_checkpoint 'host_overlap': a copy was queued on the side stream without the "
                               "side stream first waiting for the compute stream (side_waits_compute)")
        self._synced = False
        self.ops.append(op)
        self.queued += 1

    def to_host(self, buf: torch.Tensor, x: torch.Tensor) -> None:
        self._queue(_EmuCopy(buf, x, to_host=True))

    def to_device(self, out: torch.Tensor, buf: torch.Tensor) -> None:
        self._queue(_EmuCopy(buf, out, to_host=False))

    def keep(self, t: torch.Tensor) -> None:
        # the memory, not the tensor: a detached alias, so the queue never holds an autograd graph (record_stream holds
        # no reference at all; it only delays the reuse of the memory)
        for op in self.ops:
            if op.device() is t:
                op.strong = t.detach()

    def mark(self) -> int:
        return self.queued

    def _run_to(self, mark: int) -> None:
        while self.ran < mark:
            op = self.ops.popleft()
            self.ran += 1
            dev = op.strong if op.strong is not None else op.device()
            if dev is None:
                raise RuntimeError("activation_checkpoint 'host_overlap': a device tensor was freed before the side "
                                   "stream's copy " + ("from" if op.to_host else "into") + " it ran; without "
                                   "record_stream (keep) the allocator could have given its memory to another tensor")
            with torch.no_grad():
                if op.to_host:
                    op.host.copy_(dev)
                else:
                    dev.copy_(op.host)

    def compute_waits(self, mark: int) -> None:
        self._run_to(mark)

    def cpu_waits(self, mark: int) -> None:
        self._run_to(mark)

    def done(self, mark: int) -> bool:
        return self.ran >= mark
