"""The asynchronous checkpoint save.

OFF BY DEFAULT (TrainConfig.ckpt_async False): Trainer.save_checkpoint is the synchronous save it always was, and nothing
here is imported.

ON, a save has three parts (Trainer.save_checkpoint, Trainer._ckpt_settle):
1. THE SNAPSHOT (snapshot; the training thread, the only part that holds training up). The checkpoint's state (the same
   Trainer.state_dict the synchronous save writes) is copied into host memory: one arena (Arena), page-locked on CUDA and
   plain on the CPU, carved into one buffer for each distinct storage of the state, so that the copy has the state's own
   storage structure. A device storage is copied with a non_blocking copy on the current stream, and one event is
   recorded after the copies: every later kernel on that stream (the next step's, and the side streams', which wait on it)
   runs after them, so no step changes a tensor before it is copied, and the host does not wait for the copies. A host
   storage is copied at once. Every container of the state (the dicts, the lists, the optimizer's per-parameter state,
   the config, the RNG states) is copied too, by one pickle round trip with the tensors swapped for their copies, so
   nothing a later step changes reaches the file. The strings stay the state's own objects (_skeleton: pickle memoizes a
   string by its identity, and torch.save adds strings of its own, such as the location tag "cpu", to the same stream).
2. THE WRITE (Writer; one background thread, one write at a time). The thread makes no CUDA call: the training thread
   waits for the copies' event at the next step boundary (Writer.release, from busy, poll and wait; the copies ran
   before that step's kernels, so the wait is nothing) and only then lets the thread go: N1's CUDA graph capture runs in
   the global mode, in each process's third step, and torch documents that mode as erroring on actions in other threads
   (Writer's docstring). The thread then torch.saves the copy to the synchronous save's own temporary name (<step file>.tmp). The file
   is the synchronous save's bit for bit:
   - the same pickled structure (the round trip keeps every type, every shared reference and the state dicts'
     _metadata);
   - the same storages in the same order, each with its own bytes;
   - the same archive name (torch.save takes it from the file name);
   - the device storages' location tags (torch.serialization.location_tag of the ORIGINAL storage, e.g. "cuda:0", not
     the copy's "cpu"), given by a tagger registered ahead of torch's own that answers for the copy's storages only.
     The byte writes run with the GIL released (measured on the VM: a Python loop in the main thread kept 99.7 % of its
     rate during a 7.1 s background save of 2.1 GB), so the thread does not slow the training loop's host work.
3. THE COMMIT (the training thread, at a step boundary, once every rank's write is done and the ranks agree on it):
   os.replace of the temporary file to the step file, the `latest` pointer, the rolling checkpoints pruned, as the
   synchronous save does them. So `latest` only ever names a whole file. A process killed during a write leaves at most a
   temporary file, never read and removed by the run's next process at its first save.

WITH THE SHARDED OPTIMIZER (TrainConfig.optimizer_sharding "zero1"; ced/train/zero1.py; its checkpoint is written by
parameter: one optimizer file a rank beside rank 0's step file), every rank copies its own optimizer shard (the payload of
zero1.Zero1.save_shard) and its thread writes it to the synchronous save's temporary name of that file (rank 0's thread
writes its shard, then its step file: one write, Writer.start's `more`). The ranks agree that every rank's write is done
(fit's per-step MAX, as above); then each rank renames its own shard, a barrier, rank 0 renames its step file and moves
`latest` (Trainer._ckpt_commit): N2's order, rank 0's step file only once every rank's optimizer file is there. A crash
(an exception leaving fit on one rank) commits nothing: the agreement that every rank's shard is written cannot be had on
the way out. A stop between the shards and rank 0's step file leaves temporary files and an optimizer directory that
nothing names; `latest` names the last whole set, and a later commit's sweep (N2's) or the step's next save removes or
rewrites them. The files are the synchronous sharded save's bit for bit (tests/test_async_save_zero1.py).

What only a GPU shows: the page-locked arena and its allocation time, the copies' ordering against the next step
on the device, the location tags of real CUDA storages, the stall's length.
"""

from __future__ import annotations

import io
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

ALIGN = 64  # bytes: every buffer of the arena starts on a 64-byte boundary
TAG_PRIORITY = 9  # ahead of torch's CPU tagger (10); a priority no torch package uses (each must be unique: the registry sorts tuples)


class CheckpointWriteError(RuntimeError):
    """A background checkpoint write failed (the disk full, for example). Raised on every rank at the same step, before
    the file is committed: `latest` still names the last whole checkpoint."""


# ------------------------------------------------------------------------------------------------
# the location tags of the copy's storages (part 2 of the module docstring)

_TAGS: dict[int, str] = {}  # a copy's storage (its StorageImpl, _cdata) -> its original's location tag; during a write
_TAGS_LOCK = threading.Lock()
_REGISTERED = False


def _tag(storage) -> str | None:
    """torch.serialization's tagger for the copies: the original's tag for a storage a write registered, None for every
    other storage (torch's own taggers then answer as always)."""
    if not _TAGS:
        return None
    try:
        key = storage._cdata
    except AttributeError:
        return None
    with _TAGS_LOCK:
        return _TAGS.get(key)


def _untag(storage, location):  # the load side: never ours (the tags are torch's own device strings)
    return None


def _register() -> None:
    global _REGISTERED
    if not _REGISTERED:
        torch.serialization.register_package(TAG_PRIORITY, _tag, _untag)
        _REGISTERED = True


# ------------------------------------------------------------------------------------------------
# the snapshot (part 1)


class Arena:
    """Host memory for the snapshots: one tensor of bytes (page-locked when `pin`), kept for the process and reused by
    every save (its layout is the same from save to save), carved into buffers that each have a storage of their own
    (torch.frombuffer over a slice: the same memory, its own StorageImpl), so that torch.save writes one record for each,
    as it does for the original storages. One allocation, not one a storage: PyTorch's page-locked allocator rounds every
    block up to a power of two (ATen/core/CachingHostAllocator.h, roundSize), which one block a tensor would pay on every
    tensor."""

    def __init__(self, pin: bool):
        self.pin = pin
        self.memory: torch.Tensor | None = None
        self._view = None

    def carve(self, sizes: list[int]) -> list[torch.Tensor]:
        offsets, total = [], 0
        for n in sizes:
            offsets.append(total)
            total += -(-n // ALIGN) * ALIGN
        if self.memory is None or self.memory.numel() < total:
            self.memory = self._view = None  # the old block first: at 4B it is 30.8 GB (modelled)
            self.memory = torch.empty(max(total, 1), dtype=torch.uint8, pin_memory=self.pin)
            self._view = memoryview(self.memory.numpy())
        return [torch.frombuffer(self._view[o:o + n], dtype=torch.uint8) if n else torch.empty(0, dtype=torch.uint8)
                for o, n in zip(offsets, sizes)]

    @property
    def nbytes(self) -> int:
        return 0 if self.memory is None else self.memory.numel()


@dataclass
class Snapshot:
    """A state copied into host memory: `state` (the copy), `event` (the CUDA events after the device copies; empty on
    the CPU), `tags` (each copied device storage's original location tag), its bytes and storages, and the seconds the
    training thread spent making it."""

    state: object
    events: list = field(default_factory=list)
    tags: dict = field(default_factory=dict)
    nbytes: int = 0
    storages: int = 0
    seconds: float = 0.0


def _skeleton(state) -> tuple[bytes, list, list]:
    """The state pickled with every tensor replaced by its index i >= 0 (the first time it is seen), and every str and
    bytes object by -1 - j; the tensors in that order (the order torch.save meets them in), and the str and bytes objects.

    The strings are kept as the very objects of the state, not copies. pickle memoizes a str by its identity, and
    torch.save pickles strings of its own into the same stream: a storage's location tag ("cpu" is torch's interned
    literal), "storage". Where the state holds the same object (a config's device "cpu", an interned literal), the
    synchronous file has a back-reference there. A copied string would be a new object, and the file would differ
    (measured: 8 bytes of data.pkl, scripts/async_save_check.py's dry mode before this). Strings and bytes are immutable,
    so the copy may share them."""
    tensors: list = []
    index: dict = {}
    atoms: list = []
    atom_index: dict = {}

    class Collect(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, torch.Tensor):
                i = index.get(id(obj))
                if i is None:
                    i = index[id(obj)] = len(tensors)
                    tensors.append(obj)
                return i
            if type(obj) in (str, bytes):  # integer ids only: an id is pickled with persistent_id on for its parts
                j = atom_index.get(id(obj))
                if j is None:
                    j = atom_index[id(obj)] = len(atoms)
                    atoms.append(obj)
                return -1 - j
            if isinstance(obj, (torch.UntypedStorage, torch.storage.TypedStorage)):
                raise TypeError("the asynchronous save copies tensors; this checkpoint state holds a storage of its own")
            return None

    buf = io.BytesIO()
    Collect(buf, protocol=2).dump(state)
    return buf.getvalue(), tensors, atoms


def _refusal(t: torch.Tensor) -> str | None:
    if type(t) not in (torch.Tensor, torch.nn.Parameter):
        return f"a tensor of type {type(t).__name__}"
    if t.layout != torch.strided:
        return f"a {t.layout} tensor"
    if t.is_quantized:
        return "a quantized tensor"
    if t.is_conj() or t.is_neg():
        return "a tensor with its conjugate or negative bit set"
    if t.device.type == "meta":
        return "a meta tensor"
    if any(isinstance(v, torch.Tensor) for v in t.__dict__.values()):
        return "a tensor with tensor attributes"
    return None


def _bytes_of(storage) -> torch.Tensor:
    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)


def snapshot(state, arena: Arena) -> Snapshot:
    """The state copied into the arena (module docstring, part 1). The copy's tensors are views over the arena's buffers,
    one buffer for each distinct storage of the state, with each tensor's own offset, size, stride, dtype, requires_grad
    and Python attributes; a parameter stays a parameter. Refused by name (TypeError): a tensor torch.save would write in
    another way (a subclass, a sparse, quantized, conjugated or meta tensor), or a storage on its own."""
    t0 = time.perf_counter()
    skeleton, tensors, atoms = _skeleton(state)
    slot_of: dict = {}
    originals: list = []
    for t in tensors:
        why = _refusal(t)
        if why is not None:
            raise TypeError(f"the asynchronous save does not copy {why} (shape {tuple(t.shape)}); set ckpt_async false")
        st = t.untyped_storage()
        if st._cdata not in slot_of:
            slot_of[st._cdata] = len(originals)
            originals.append(st)
    buffers = arena.carve([st.nbytes() for st in originals])
    devices, tags = set(), {}
    for st, buf in zip(originals, buffers):
        if st.device.type == "cpu":
            buf.copy_(_bytes_of(st))
        else:
            buf.copy_(_bytes_of(st), non_blocking=True)
            devices.add(st.device)
            tags[buf.untyped_storage()._cdata] = torch.serialization.location_tag(st)
    events = []
    for dev in sorted(devices, key=str):  # after every device copy: the training thread waits for these (Writer.release)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(dev))
        events.append(ev)
    copies = []
    for t in tensors:
        buf = buffers[slot_of[t.untyped_storage()._cdata]]
        c = torch.empty(0, dtype=t.dtype).set_(buf.untyped_storage(), t.storage_offset(), t.size(), t.stride())
        if isinstance(t, torch.nn.Parameter):
            c = torch.nn.Parameter(c, requires_grad=t.requires_grad)
        elif t.requires_grad:
            c.requires_grad_(True)
        if t.__dict__:
            c.__dict__.update(t.__dict__)
        copies.append(c)

    class Rebuild(pickle.Unpickler):
        def persistent_load(self, pid):
            return copies[pid] if pid >= 0 else atoms[-1 - pid]

    copy = Rebuild(io.BytesIO(skeleton)).load()
    return Snapshot(state=copy, events=events, tags=tags, nbytes=sum(st.nbytes() for st in originals),
                    storages=len(originals), seconds=time.perf_counter() - t0)


# ------------------------------------------------------------------------------------------------
# the write (part 2)


def _torch_save(obj, path: Path) -> None:
    """The write itself: torch.save (a name of its own so that a test can slow it or make it fail)."""
    torch.save(obj, path)


class Writer:
    """One background write at a time. Every method runs on the training thread; the thread it starts makes no CUDA
    call. start(snapshot, tmp) starts a thread that waits until the snapshot's device copies are released, then
    torch.saves the copy to `tmp`. release() waits for the copies' events (on the training thread) and lets the write go:
    busy(), poll() and wait() release first, so fit's next step boundary (_ckpt_slots, _ckpt_settle) starts the write;
    there the copies are done (they were queued before the step's kernels, whose metrics the host has read), so the wait
    is nothing. Why not in the thread: N1 captures its CUDA graphs in the default global mode (ced/train/cuda_graphs.py,
    torch.cuda.graph without capture_error_mode), in each process's third step, and torch's documentation of that mode
    says '"global" will error on actions in other threads' (torch/cuda/graphs.py, capture_error_mode); whether an event
    synchronize is such an action is unverified (CUDA's "potentially unsafe API calls" are not listed), so the thread
    makes none. A save after a process's second step would otherwise put the thread's synchronize, and the events'
    destruction, inside that capture. A release whose synchronize raises lets the thread go with the copies unconfirmed:
    it writes nothing and reports the failure. On the CPU there are no events and the thread goes at once. busy() says
    whether the thread still runs; wait() joins it and returns its outcome. A failed write removes its temporary file.
    The thread is not a daemon, so a process that ends while it runs ends after it.

    One write may hold several files (start's `more`; the sharded optimizer's rank 0, N2 with N8: its optimizer shard, then
    its step file): the thread writes them in order, under one release and one outcome; a failure of any removes every
    temporary file of the write."""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._outcome: dict | None = None
        self._gate: dict | None = None  # {"go": threading.Event, "ok": the copies confirmed} of the current write
        self._events: list = []  # the current write's CUDA events, until release

    def start(self, snap: Snapshot, tmp: Path, more: list | tuple = ()) -> None:
        """Start the write of `snap` to `tmp`, then of each (snapshot, temporary file) of `more`, in order."""
        if self.busy():
            raise RuntimeError("a checkpoint write is already running: one write at a time (Trainer._ckpt_settle first)")
        _register()
        self._outcome = None
        jobs = [(snap, Path(tmp))] + [(s, Path(p)) for s, p in more]
        self._events = []
        for s, _ in jobs:  # the events stay on the training thread
            self._events, s.events = self._events + list(s.events), []
        self._gate = {"go": threading.Event(), "ok": True}
        if not self._events:
            self._gate["go"].set()
        self._thread = threading.Thread(target=self._run, args=(jobs, self._gate),
                                        name="ced-checkpoint-writer", daemon=False)
        self._thread.start()

    def release(self) -> None:
        """The training thread: wait for the current write's device copies and let the write go (nothing without one)."""
        events, self._events = self._events, []
        gate = self._gate
        ok = False
        try:
            for ev in events:
                ev.synchronize()
            ok = True
        finally:
            if gate is not None and not gate["go"].is_set():
                gate["ok"] = ok
                gate["go"].set()

    def _run(self, jobs: list, gate: dict) -> None:
        t0 = time.perf_counter()
        nbytes = sum(snap.nbytes for snap, _ in jobs)
        try:
            gate["go"].wait()
            t0 = time.perf_counter()  # write_s: from the release (the training thread's next step boundary)
            if not gate["ok"]:
                raise RuntimeError("the device copies were not confirmed (their event's synchronize failed on the "
                                   "training thread); nothing written")
            for snap, tmp in jobs:
                keys = list(snap.tags)
                with _TAGS_LOCK:
                    _TAGS.update(snap.tags)
                try:
                    _torch_save(snap.state, tmp)
                finally:
                    with _TAGS_LOCK:
                        for k in keys:
                            _TAGS.pop(k, None)
            self._outcome = {"ok": True, "error": None, "write_s": time.perf_counter() - t0, "bytes": nbytes}
        except BaseException as error:  # noqa: BLE001 - every failure is reported to the training thread (wait)
            for _, tmp in jobs:  # every file of the write: a write is whole or not there
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            self._outcome = {"ok": False, "error": f"{type(error).__name__}: {error}",
                             "write_s": time.perf_counter() - t0, "bytes": nbytes}
        finally:
            for snap, _ in jobs:
                snap.state = None  # the copy's views go with the write; the arena stays for the next save

    def busy(self) -> bool:
        self.release()
        return self._thread is not None and self._thread.is_alive()

    def poll(self) -> tuple[bool, dict | None]:
        """(running, outcome), after release: the outcome once the thread has ended (read after is_alive, so an ended
        thread's outcome is always there), None while it runs or before any write."""
        self.release()
        if self._thread is None:
            return False, None
        if self._thread.is_alive():
            return True, None
        return False, self._outcome

    def wait(self) -> dict | None:
        """Release, join the thread; its outcome ({"ok", "error", "write_s", "bytes"}), None when no write was started."""
        self.release()
        if self._thread is None:
            return None
        self._thread.join()
        return self._outcome
