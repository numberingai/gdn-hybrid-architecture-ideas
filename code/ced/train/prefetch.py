"""The data prefetch: the NEXT step's input built on a background thread while this step runs (TrainConfig.data_prefetch).

WHAT IS PREFETCHED. One StepInput a step: Trainer.batch_at(step) (the rows, the decoder / aux positions, the masks, the pass
counts, the loader's validation), the loss-valid mask, the host numbers the step and its metrics read from the batch
(HostNumbers: the counts, the batch hash, the position shortfall, the valid-target count; under data parallelism also the
micro-batches' boundary check), and on CUDA a device copy of every tensor the micro-batches read, made from pinned host memory
on a side stream. All of it is a pure function of the step (batch_at is: it reads the stream at the step's place and draws
from counter generators of (seed, step, row), never from a global generator), so the input built early is the input built on
the main thread, bit for bit, on every rank.

THE PIECES.
- Build: a thread started by submit(step) (one at a time; it ends with its job: no idle thread exists between steps) runs the
  Trainer's build function under the BUILD LOCK. The shard packer keeps a one-epoch cache that a read of another epoch replaces,
  so no two builds run at once: a build the main thread makes itself (the first step of a process, a step nothing was submitted
  for) takes the same lock.
- Stage (CUDA only): after the build, the same thread copies each tensor the micro-batches read to the device, from a pinned
  host buffer (pin_memory), non-blocking, on the stage's own stream, and records an event after the last copy. The thread's
  CUDA work is under the CUDA LOCK, which GraphRunner holds around its two captures (quiet()): a capture in the default
  "global" error mode refuses unsafe CUDA calls made by ANY thread meanwhile, and a pin_memory is one.
- Wait: take(step) joins the thread (the host wait; zero when the build is done, which is the point), then Staged.wait() makes
  the stream the step runs on wait for the event (a device-side wait, no host stall) and records every staged tensor as used
  on that stream, so the allocator does not give its block to the stage stream's next copy while the step still reads it. The
  tensors are reachable only through Staged.twin(host_tensor), which raises PrefetchRace before the wait has been made: a
  read of the next batch before its copy is complete is an error, not stale numbers.
- Resume: the prefetch holds no state the stream depends on. The stream is a function of the step (in the checkpoint) and the
  data identity (Trainer.data_state, in the checkpoint and checked on a load); a prefetch in flight when a checkpoint is saved
  is for the step the checkpoint names, built from the same two things, and a process that resumes starts with none (its first
  step builds on the main thread). Trainer.load_state_dict and fit's exit call discard(): the thread finished and its result
  dropped, so nothing built for the old state outlives it. take(step) also refuses a result built for another step.

Errors. A build that raises (BatchError from the loader's validation, a stream end, a step before data_start_step) stores the
exception on its job; take() of THAT step raises it, from the same place the main-thread build raised it, and a job that is
never taken is dropped with its error.

Off (TrainConfig.data_prefetch False, the default): nothing here is imported and train_step is the step it was.
"""

from __future__ import annotations

import atexit
import threading
import time
import weakref
from dataclasses import dataclass, field

import torch


class PrefetchRace(RuntimeError):
    """A staged tensor was read before the stream that reads it had waited for the copy (Staged.wait)."""


class NotStagedError(LookupError):
    """A tensor was asked for by Staged.twin that the build did not list among those the micro-batches read."""


@dataclass
class HostNumbers:
    """What the step reads from the host batch besides its tensors (Trainer._host_numbers): the loss counts, the batch hash,
    the position samplers' shortfall and the number of valid targets. Each is the value the main thread computes."""

    counts: dict
    hash: str
    short: dict
    n_target_valid: int


@dataclass
class StepInput:
    """One step's input. to_stage: the host tensors that get a device twin (every tensor the micro-batches move to the device,
    each once). staged: their device twins (None on the CPU, or before the stage ran). stats: the milliseconds the build, the
    stage and the take's wait cost, and whether the take found the job done for this step (a hit)."""

    step: int
    batch: object
    valid_loss: torch.Tensor
    numbers: HostNumbers
    to_stage: list = field(default_factory=list)
    staged: "Staged | None" = None
    stats: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------------------------
# the copy engines


class CudaStage:
    """The device copy on CUDA: pinned host buffers, a side stream of this process, one event after the last copy."""

    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.index is None:
            # review fix: made on the main thread, whose current device a run may have set (torch.cuda.set_device); the build
            # thread starts with device 0, and `torch.cuda.device(torch.device("cuda"))` there means "the thread's current"
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.stream = torch.cuda.Stream(self.device)

    def copy(self, hosts: list) -> tuple[list, list, object]:
        """(device twins, the pinned buffers, the event): the copies are enqueued on the side stream, non-blocking; the
        caller keeps the pinned buffers until the twins are dropped. Runs on the build thread, under the CUDA lock."""
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            pinned = [h.contiguous().pin_memory() for h in hosts]
            twins = [p.to(self.device, non_blocking=True) for p in pinned]
            event = torch.cuda.Event()
            event.record(self.stream)
        return twins, pinned, event

    def wait(self, twins: list, event) -> None:
        """The stream the step runs on (the current stream of this thread) waits for the copies, on the device; every twin
        is recorded as used on that stream (the allocator keeps its block from the stage stream until that stream's work
        so far is done). The work of the capture stream is joined to this stream by GraphRunner._on_stream, so a twin read
        there is covered too."""
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(event)
        for t in twins:
            t.record_stream(stream)


class EmulatedStage:
    """A copy engine for the CPU tests: the twins start as zeros and a thread fills them after `delay` seconds, setting an
    event; wait() blocks the host on that event. A read of a twin before the wait sees the zeros: the race the guard in
    Staged.twin exists to make an error. Nothing of the real engine's streams, only its order of events."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.copies = 0
        self.waits = 0

    def copy(self, hosts: list) -> tuple[list, list, object]:
        twins = [torch.zeros_like(h) for h in hosts]
        done = threading.Event()

        def fill():
            time.sleep(self.delay)
            for t, h in zip(twins, hosts):
                t.copy_(h)
            done.set()

        self.copies += 1
        threading.Thread(target=fill, daemon=True, name="emulated-copy").start()
        return twins, [], done

    def wait(self, twins: list, done) -> None:
        self.waits += 1
        done.wait()


def make_stage(device):
    """The copy engine of a device: CudaStage on CUDA, None on the CPU (the host tensors are the device's)."""
    return CudaStage(device) if torch.device(device).type == "cuda" else None


class Staged:
    """The device twins of a step's input tensors. twin(host) is the only way to them, and refuses before wait()."""

    def __init__(self, stage, hosts: list, twins: list, pinned: list, event):
        self._stage, self._hosts, self._twins, self._pinned, self._event = stage, hosts, twins, pinned, event
        self._index = {id(h): i for i, h in enumerate(hosts)}  # hosts stays referenced: the ids stay theirs
        self.waited = False

    def wait(self) -> None:
        """Before the first use: the consuming stream waits for the copies (idempotent)."""
        if not self.waited:
            self._stage.wait(self._twins, self._event)
            self.waited = True

    def twin(self, host: torch.Tensor) -> torch.Tensor:
        if not self.waited:
            raise PrefetchRace("a staged tensor was read before the stream that reads it had waited for its copy (Staged.wait)")
        i = self._index.get(id(host))
        if i is None:
            raise NotStagedError(f"a tensor of shape {tuple(host.shape)} and dtype {host.dtype} is not among those the build "
                                 "staged: add it to StepInput.to_stage")
        return self._twins[i]


# ------------------------------------------------------------------------------------------------------------------
# the prefetcher


class _Job:
    def __init__(self, step: int):
        self.step = step
        self.thread: threading.Thread | None = None
        self.result: StepInput | None = None
        self.error: BaseException | None = None


class BatchPrefetcher:
    """One step ahead. build(step) -> StepInput is the Trainer's host-side function of the step; stage is make_stage's
    engine (None: no device copy)."""

    def __init__(self, build, stage=None):
        self._build = build
        self._stage = stage
        self._build_lock = threading.Lock()
        self._cuda_lock = threading.RLock()
        self._job: _Job | None = None
        self.counts = {"submitted": 0, "hits": 0, "misses": 0, "dropped": 0}
        # a process that ends with a job in flight (a script that calls train_step and exits) joins it first: a daemon thread
        # inside a torch call when the interpreter finalises is how such a process would crash. A weak reference: a trainer
        # that is dropped takes its prefetcher with it.
        atexit.register(lambda ref=weakref.ref(self): ref() is not None and ref().discard())

    # -- the work of one step's input (on the build thread, or on the caller's for a miss)
    def _make(self, step: int) -> StepInput:
        t0 = time.perf_counter()
        with self._build_lock:
            si = self._build(step)
        t1 = time.perf_counter()
        if self._stage is not None and si.to_stage:
            with self._cuda_lock:
                twins, pinned, event = self._stage.copy(si.to_stage)
            si.staged = Staged(self._stage, si.to_stage, twins, pinned, event)
        si.stats = {"build_ms": 1e3 * (t1 - t0), "stage_ms": 1e3 * (time.perf_counter() - t1)}
        return si

    def _run(self, job: _Job) -> None:
        try:
            job.result = self._make(job.step)
        except Exception as error:  # noqa: BLE001 - kept for take(): raised there if this step is ever taken
            job.error = error

    # -- the trainer's calls
    def submit(self, step: int) -> None:
        """Start building `step`'s input on a thread, unless it is already (or a job for another step is first dropped)."""
        job = self._job
        if job is not None:
            if job.step == step:
                return
            self.discard()
        job = _Job(step)
        job.thread = threading.Thread(target=self._run, args=(job,), daemon=True, name=f"batch-prefetch-{step}")
        self._job = job
        self.counts["submitted"] += 1
        job.thread.start()

    def take(self, step: int) -> StepInput:
        """`step`'s input: the submitted job's if it is for this step (joined; its error raised), else built here. The
        consuming stream has waited for the device copies when this returns."""
        t0 = time.perf_counter()
        job, self._job = self._job, None
        hit = job is not None and job.step == step
        if job is not None:
            job.thread.join()
        if hit:
            if job.error is not None:
                self.counts["hits"] += 1
                raise job.error
            if job.result is None:
                raise RuntimeError(f"the prefetch thread for step {step} ended without a result")
            si = job.result
            self.counts["hits"] += 1
        else:
            if job is not None:
                self.counts["dropped"] += 1
            si = self._make(step)
            self.counts["misses"] += 1
        if si.staged is not None:
            si.staged.wait()
        si.stats.update(hit=hit, wait_ms=1e3 * (time.perf_counter() - t0))
        return si

    def discard(self) -> None:
        """The job in flight finished and its result dropped: nothing built for an earlier state outlives the call."""
        job, self._job = self._job, None
        if job is not None:
            job.thread.join()
            self.counts["dropped"] += 1

    def quiet(self):
        """A context in which no CUDA call of the build thread is made (CUDA graph captures; module docstring)."""
        return self._cuda_lock

    @property
    def pending(self) -> int | None:
        """The step of the job in flight or done and not yet taken (None: none)."""
        return None if self._job is None else self._job.step
