"""parts D1 (launch and process groups), D2 (per-rank data and token-count weighting) and D3 (gradient sync).

OFF BY DEFAULT. Without torchrun's environment (no WORLD_SIZE, or WORLD_SIZE 1) from_environment returns None, the trainer
gets no DataParallel and every line of it is today's (tests/test_data_parallel.py, T6, against main's tree in its own
process). Nothing here is imported into a run's step unless a DataParallel exists.

D1, LAUNCH AND PROCESS GROUPS (from_environment, scripts/train.py). torchrun sets RANK, LOCAL_RANK, WORLD_SIZE,
LOCAL_WORLD_SIZE, MASTER_ADDR and MASTER_PORT. On CUDA the rank's device is cuda:LOCAL_RANK (torch.cuda.set_device before
anything is built); when the process sees one device only it uses cuda:0: either the node's ranks share it (one MIG slice:
"two processes on one slice", where NCCL is refused by name and gloo carries it), or each rank's CUDA_VISIBLE_DEVICES names
a device of its own (one MIG slice's UUID a process: the two-slice check), which an explicit CED_DP_BACKEND=nccl states.
torchrun gives every worker the same environment, so the second form needs ENV_DEVICES ("CED_DP_DEVICES" = a comma list of
the node's devices, one a local rank: MIG UUIDs, GPU UUIDs or indices): each rank sets its own CUDA_VISIBLE_DEVICES to
entry LOCAL_RANK before CUDA initialises, and then uses cuda:0 with any backend (the two-slice probe,
scripts/dp_two_slice_probe.py: a CUDA process enumerates one MIG instance, so a node with two slices otherwise puts both
ranks on the first). Unset (the B200: every GPU visible, cuda:LOCAL_RANK), nothing of it runs. The mapping itself is
device_index (pure; tests/test_dp_two_slice_probe.py). Two groups:
- the TENSOR group (the default group): NCCL on CUDA, gloo on the CPU. It carries the gradient all-reduce, the step's
  packed metric sums and the debug replica check. ENV_BACKEND ("CED_DP_BACKEND") overrides its backend: "gloo" makes the
  one-slice check's tensor all-reduce run on gloo with CUDA tensors (gloo's CUDA all_reduce stages through the host);
- the HOST group (a gloo group of every rank): the step's host integers (the token counts, the bad-batch flag, the rows'
  hashes, which parameters got a gradient), so agreeing on them never waits on the GPU.

D2, PER-RANK DATA AND TOKEN-COUNT WEIGHTING (Trainer.batch_at, Trainer._dp_batch). Rank r of W reads rows r B/W ..
(r + 1) B/W - 1 of the step's B rows (DataParallel.rows): the shard stream's rows step B + r B/W onwards
(ShardPacker.rows), the synthetic source's rows of the same global index (SyntheticPackedData.batch's first / count), and
the decoder, aux and answer positions drawn for the GLOBAL row index (PositionSampler.apply's row0). So the ranks' rows
are disjoint, their union is the single process's batch in order, and every row is the single process's row bit for bit
(T3). The step's host counts (n_valid, n_dec, n_aux: Trainer.batch_counts, loss masks included) are summed over the ranks
on the host group before the first micro-batch, so each rank's micro-batch loss is its local sum over the GLOBAL count,
exactly the single process's micro-batch loss, and the SUM of the ranks' gradients is the single process's gradient, up
to the order of the sum over micro-batches (T1, T2). micro_share (MoE's balance loss) is taken over the global rows. A
rank runs K/W micro-batches. A batch that fails the loader's validation (check_batch; with the fused GDN kernels on CUDA
also the boundary arrays of every micro-batch, checked before the first) on ANY rank halts EVERY rank before any
micro-batch: the flag travels in the same host reduction, and each rank raises a BatchError naming the failing rank.

D3, GRADIENT SYNC (GradBuffer, Trainer._dp_sync). One flat gradient buffer per dtype (float32 master weights: one) with
every parameter's .grad a view into it (tensor of its own over the buffer's storage, 256-byte aligned, with its own
version counter: saves its buffer for backward); the persistent float32 buffers (ced/train/wgrad_accum.py) are these
views, and so are N1's persistent gradients (ced/train/cuda_graphs.py), which are the views the process's first step
leaves. After the micro-batch loop, before the gradient metrics and the clip: one SUM all_reduce of each flat buffer on
the tensor group, then the step's loss, metric sums, dropped-slot count and (with model.fp8.nonfinite_flag) the MXFP8
flag's bits in ONE packed float64 SUM all_reduce (a float32 sum of two ranks rounded from float64 is the float32 sum
itself: 53 >= 2 x 24 + 2). Replicated Muon, AdamW and the matrix optimizers then run unchanged on identical gradients. No
NCCL determinism pins; what is checked is that the ranks stay equal: ENV_CHECK_EVERY
("CED_DP_CHECK_EVERY" = N) compares the ranks' parameter checksums before the process's first step and after every N-th,
and raises ReplicaMismatch on every rank on a difference.

Which parameters have a gradient. Today a parameter that no micro-batch's backward reaches keeps .grad None, and the clip
and the optimizers skip it (no weight decay, no state: pass 2's tensors before a loop switch step, for example). With the
views attached up front, GradBuffer records which parameters the backward reached (a post-accumulate-grad hook, removed
for the rest of the step once every parameter is reached: in practice after the first micro-batch); the ranks take the
union (a MAX on the host group) and the parameters no rank reached get .grad None again after the all-reduce, as in the
single process. Under CUDA graphs the set is fixed after the process's first step (N1's rule: "the set of parameters with
a gradient is the run's"), and no hook runs.

Not built under data parallelism, refused by name (refusals): the batch-shape schedule, a TST stage (below), a
pass count per row (model.loop.t_sampling "sequence"), the segment pass schedule, Looped-CED v1, the asentmax normaliser
(its training statistics are per process), a batch that is not a whole number of micro-batches on every rank.

TST_DP, TOKEN SUPERPOSITION TRAINING UNDER DATA PARALLELISM (tests/test_dp_tst.py). Until then the one refusal above was the only
guard: lifted alone, each rank would have read ALL B rows of the step (TSTPlan.batch_at knew no rank) and normalised its bag loss by
its own rows' weights (TSTPlan.weights). What is built, all of it a function of the step so that no message decides anything:
- the rows (D2): rank r of W reads bag rows r B/W .. (r + 1) B/W - 1 of the step's B, that is stream base rows R(k) + (r B/W) s
  onward (TSTPlan.batch_at first / count, Trainer._build_batch), with the decoder and aux positions drawn for the global row
  index as always; every rank reads the same global batch as the single process, split;
- the loss weights: the bag phase's denominator sum v w and its decoder / other split are the GLOBAL batch's. Each rank counts
  the positions with k valid targets (k = 0 .. s), off and on the decoder mask (TSTPlan.host_counts) and the ranks add the
  integers in the step's one host reduction (Trainer._dp_batch); every rank then forms the same totals from the same histogram
  (TSTPlan.weights summed). The multi-hot aux's valid-target count travels the same way. The step number goes in the same sum, and
  a rank at another update halts every rank by name before any collective of the step (so every rank switches at the same update);
- the switch (TSTPlan.switch): every rank resets (or keeps, tst.keep_optimizer_state) its optimizer's state at the same step with
  no message. Replicated, each rank holds all of it; under ZeRO-1 each tensor's state lives on its owner and is reset there; the
  event's count of states is the ranks' counts added (a host reduction at a switch step only);
- the LR (lr_at, lr_factor) is a function of the step; the checkpoint carries the step and the data identity (tst_schedule), so a
  resume on either side of the switch, at any world size, switches exactly where the uninterrupted run does.

D4, AGREEMENT AND RANK 0'S DUTIES (Trainer.fit). Every rank acts at the same step: the bad-batch flag and the non-finite
check read the global step's numbers (D2, D3); a SIGTERM on any rank and rank 0's checkpoint clock are agreed in ONE MAX
on the host group after each step (DataParallel.agree). Rank 0 (DataParallel.lead) writes metrics.jsonl, the halt
records and checkpoints, logs and talks to W&B, and runs the evals and readouts while every other rank waits at a
barrier (DataParallel.barrier) on the host group, whose timeout is ENV_WAIT_SECONDS (default 2 hours; the tensor group
keeps torch's default, since no rank waits on it while rank 0 works). The replica check runs once before every process's
first step (a resume included), and every N-th step with ENV_CHECK_EVERY.

D5, THE CHECKPOINT AT ANY WORLD SIZE (Trainer.save_checkpoint, Trainer._dp_load). Rank 0 writes today's single file (the
state is replicated) with every rank's CUDA generator state; each rank's CUDA generator is seeded from rank_seed(seed,
rank, step) at a data-parallel start and whenever a resume changes the world size, restored from the file at the same
world size, and every re-derivation is recorded in the run's history (Trainer.dp_info).

E1, THE EVALS SHARDED BY ROWS (ced/train/sharded_eval.py; OFF BY DEFAULT). With ENV_SHARD_EVALS ("CED_DP_SHARD_EVALS" = 1)
the periodic and final evals on the eval sets (TrainConfig.eval.sets) are every rank's work instead of rank 0's: each rank
scores its own contiguous block of the sets' batches, the per-row sums go to rank 0 on the host group
(DataParallel.gather_to_lead), and rank 0 joins them and writes the records, which are the single-process eval's bit for
bit. Like the world size it is not in the config (it changes where the rows are scored, not what is computed), and it is
read under data parallelism only; the start record names it.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import torch

ENV_BACKEND = "CED_DP_BACKEND"  # the tensor group's backend: "nccl" (CUDA's default) or "gloo"
ENV_DEVICES = "CED_DP_DEVICES"  # one CUDA device a local rank, comma-separated: rank LOCAL_RANK's CUDA_VISIBLE_DEVICES (D1)
ENV_CHECK_EVERY = "CED_DP_CHECK_EVERY"  # the replica check every N steps (unset or 0: only once, before a process's first step)
# D4: how long a rank waits on the host group (rank 0's evals, readouts and synchronous save, while the others wait at a
# barrier). A rank that crashes closes its connections, so the others fail at once; the timeout only ends a true hang.
# Default 2 hours (an assumption: the 4B's final evals on one GPU are not measured).
ENV_WAIT_SECONDS = "CED_DP_WAIT_SECONDS"
DEFAULT_WAIT_SECONDS = 7200
# E1: the evals sharded by rows over the ranks (module docstring, E1): "1" on; unset or "0" off (rank 0 evaluates alone)
ENV_SHARD_EVALS = "CED_DP_SHARD_EVALS"
BACKENDS = ("nccl", "gloo")


class ReplicaMismatch(RuntimeError):
    """The ranks' parameters differ (the debug replica check, ENV_CHECK_EVERY). Raised on every rank."""


@dataclass
class DataParallel:
    """This process's place among the data-parallel ranks, and its two process groups (module docstring, D1). Without
    groups (connected False) it is a data layout only: which rows the rank reads (the tests of D2's data use it so)."""

    rank: int
    world: int
    local_rank: int = 0
    device: torch.device = torch.device("cpu")
    backend: str = "gloo"
    check_every: int = 0
    tensor_group: object = None
    host_group: object = None
    connected: bool = False
    shard_evals: bool = False  # E1 (ENV_SHARD_EVALS): the eval sets' rows scored by every rank, joined on rank 0

    def __post_init__(self):
        if not (isinstance(self.world, int) and self.world >= 2 and isinstance(self.rank, int) and 0 <= self.rank < self.world):
            raise ValueError(f"data parallelism: rank {self.rank!r} of world size {self.world!r} (world size >= 2)")

    # --- D2: the rank's rows --------------------------------------------------------------------------------------------
    def rows(self, batch_size: int) -> tuple[int, int]:
        """(first, count): the rank's rows of a batch of batch_size rows, rows first .. first + count - 1."""
        if batch_size % self.world:
            raise ValueError(f"data.batch_size {batch_size} is not a whole number of rows a rank at world size {self.world}")
        count = batch_size // self.world
        return self.rank * count, count

    # --- the host group -------------------------------------------------------------------------------------------------
    def _need(self) -> None:
        if not self.connected:
            raise RuntimeError("data parallelism: this DataParallel has no process groups (a data layout only)")

    def sum_host(self, values: list[int]) -> list[int]:
        """Each entry summed over the ranks (int64, on the host group)."""
        return self._reduce_host(values, "SUM")

    def max_host(self, values: list[int]) -> list[int]:
        """Each entry's maximum over the ranks (int64, on the host group)."""
        return self._reduce_host(values, "MAX")

    def _reduce_host(self, values: list[int], op: str) -> list[int]:
        import torch.distributed as dist

        self._need()
        t = torch.tensor(values, dtype=torch.int64)
        dist.all_reduce(t, op=getattr(dist.ReduceOp, op), group=self.host_group)
        return t.tolist()

    def gather_host(self, obj) -> list:
        """Every rank's obj (a picklable host value), in rank order (on the host group: failure paths, a checkpoint's
        per-rank RNG states, a halt record's rows)."""
        import torch.distributed as dist

        self._need()
        out = [None] * self.world
        dist.all_gather_object(out, obj, group=self.host_group)
        return out

    def gather_to_lead(self, obj) -> list | None:
        """Every rank's obj (a picklable host value) on rank 0, in rank order; None on the other ranks (on the host group:
        the sharded evals' per-row sums, E1)."""
        import torch.distributed as dist

        self._need()
        out = [None] * self.world if self.lead else None
        dist.gather_object(obj, out, dst=0, group=self.host_group)
        return out

    # --- D4: agreement and rank 0's duties ------------------------------------------------------------------------------
    @property
    def lead(self) -> bool:
        """Rank 0: the rank that writes the run's files (metrics.jsonl, halt records, checkpoints), logs, talks to W&B and
        runs the evals and readouts."""
        return self.rank == 0

    def barrier(self) -> None:
        """Every rank waits here for every other (on the host group, whose timeout is ENV_WAIT_SECONDS): after rank 0's
        duties (an eval, a readout, a checkpoint's files), so that no rank runs ahead into a collective while rank 0
        works."""
        import torch.distributed as dist

        self._need()
        dist.barrier(group=self.host_group)

    def agree(self, stop: bool, lead_due: bool, extra: list[int] | None = None):
        """D4's per-step agreement, one MAX on the host group: (a stop was requested on ANY rank (SIGTERM), rank 0's
        clock says a checkpoint is due). Every rank gets the same two answers; lead_due is read from rank 0 only.
        extra (asynchronous save's slots: Trainer._ckpt_slots; None otherwise): more integers in the same MAX, whose
        maxima come back as a third item (a list); without it the payload and the answer are as before."""
        got = self.max_host([int(bool(stop)), int(bool(lead_due)) if self.lead else 0] + [int(x) for x in (extra or [])])
        if extra is None:
            return bool(got[0]), bool(got[1])
        return bool(got[0]), bool(got[1]), [int(x) for x in got[2:]]

    # --- the tensor group -----------------------------------------------------------------------------------------------
    def all_reduce_(self, tensor: torch.Tensor, op: str = "SUM") -> torch.Tensor:
        """tensor reduced over the ranks in place (on the tensor group)."""
        import torch.distributed as dist

        self._need()
        dist.all_reduce(tensor, op=getattr(dist.ReduceOp, op), group=self.tensor_group)
        return tensor

    def close(self) -> None:
        """The process groups destroyed (scripts/train.py, at the end of the process)."""
        import torch.distributed as dist

        if self.connected and dist.is_initialized():
            dist.destroy_process_group()
        self.connected = False


def check_every_from_env() -> int:
    raw = os.environ.get(ENV_CHECK_EVERY, "").strip()
    if not raw:
        return 0
    try:
        n = int(raw)
    except ValueError:
        n = -1
    if n < 0:
        raise ValueError(f"{ENV_CHECK_EVERY}={raw!r}: 0 (off) or a number of optimizer steps")
    return n


def shard_evals_from_env() -> bool:
    raw = os.environ.get(ENV_SHARD_EVALS, "").strip()
    if raw not in ("", "0", "1"):
        raise ValueError(f"{ENV_SHARD_EVALS}={raw!r}: 1 (the evals sharded by rows over the ranks) or 0 / unset (rank 0 alone)")
    return raw == "1"


def wait_seconds_from_env() -> float:
    raw = os.environ.get(ENV_WAIT_SECONDS, "").strip()
    if not raw:
        return float(DEFAULT_WAIT_SECONDS)
    try:
        s = float(raw)
    except ValueError:
        s = -1.0
    if not s > 0:
        raise ValueError(f"{ENV_WAIT_SECONDS}={raw!r}: a number of seconds above 0")
    return s


def init(rank: int, world: int, local_rank: int, device: torch.device, backend: str, init_method: str = "env://",
         check_every: int | None = None, wait_seconds: float | None = None,
         shard_evals: bool | None = None) -> DataParallel:
    """Join the process group as rank `rank` of `world` (the tensor group on `backend`) and make the host group (gloo,
    with its own timeout: wait_seconds, default ENV_WAIT_SECONDS, D4's waits for rank 0). The caller has set the CUDA
    device (from_environment does). init_method: torchrun's "env://" (MASTER_ADDR, MASTER_PORT), or a "tcp://host:port"
    the tests give. shard_evals (E1): default ENV_SHARD_EVALS."""
    import datetime

    import torch.distributed as dist

    if backend not in BACKENDS:
        raise ValueError(f"{ENV_BACKEND}={backend!r}: one of {BACKENDS}")
    if backend == "nccl" and device.type != "cuda":
        raise ValueError(f"data parallelism: backend 'nccl' carries CUDA tensors; this rank's device is {device}: use gloo")
    wait = wait_seconds_from_env() if wait_seconds is None else float(wait_seconds)
    shard = shard_evals_from_env() if shard_evals is None else bool(shard_evals)
    kwargs = {"device_id": device} if backend == "nccl" else {}
    dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world, **kwargs)
    host = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=wait))
    dp = DataParallel(rank=rank, world=world, local_rank=local_rank, device=device, backend=backend,
                      check_every=check_every_from_env() if check_every is None else check_every,
                      tensor_group=None, host_group=host, connected=True, shard_evals=shard)
    # E1: every rank shards the evals or none does (a rank that did not would wait at a barrier while the others gather)
    agreed = dp.sum_host([int(shard)])[0]
    if agreed not in (0, world):
        dp.close()
        raise ValueError(f"{ENV_SHARD_EVALS}: {agreed} of {world} ranks shard the evals; every rank or none")
    return dp


def rank_devices(env, local_rank: int, local_world: int) -> str | None:
    """This rank's own CUDA device from ENV_DEVICES (entry LOCAL_RANK of the comma list), to become its CUDA_VISIBLE_DEVICES;
    None when the variable is unset or empty (every rank sees what the node shows it: the B200). Refused by name: an empty
    entry, an entry named twice, fewer entries than the node's ranks."""
    raw = (env.get(ENV_DEVICES) or "").strip()
    if not raw:
        return None
    items = [x.strip() for x in raw.split(",")]
    if any(not x for x in items) or len(set(items)) != len(items):
        raise ValueError(f"{ENV_DEVICES}={raw!r}: one device a local rank, comma-separated, none empty or named twice")
    if len(items) < local_world or local_rank >= len(items):
        raise ValueError(f"{ENV_DEVICES}={raw!r}: {len(items)} devices for {local_world} ranks on this node "
                         f"(local rank {local_rank})")
    return items[local_rank]


def device_index(n_visible: int, local_rank: int, local_world: int, backend: str, asked: str = "",
                 own_device: bool = False) -> int:
    """D1's rank-to-device mapping on CUDA: the index of this rank's device among the n_visible devices its process sees.
    Several visible (every GPU of the node, the B200): cuda:LOCAL_RANK. One visible: cuda:0, either because every rank of
    the node shares it (one MIG slice, the one-slice check: gloo) or because each rank sees its own (own_device: ENV_DEVICES
    narrowed it; or CUDA_VISIBLE_DEVICES set per process). The default NCCL is refused on one visible device unless the
    rank's device is known to be its own (own_device) or NCCL was asked for by name (asked == "nccl"): the shared case is
    the one this code cannot tell from the other (NCCL itself refuses two ranks on one device at its init)."""
    if n_visible > 1:
        if local_rank >= n_visible:
            raise ValueError(f"data parallelism: local rank {local_rank} on a node with {n_visible} visible CUDA devices")
        return local_rank
    if backend == "nccl" and local_world > 1 and not asked and not own_device:
        raise ValueError(f"data parallelism: {local_world} ranks on this node see one CUDA device each; NCCL "
                         f"takes one device a rank: set {ENV_BACKEND}=gloo when the ranks share it (the one-slice "
                         f"check), or {ENV_BACKEND}=nccl when each rank's device is its own (the two-slice check), or "
                         f"{ENV_DEVICES} to give each rank its own")
    return 0


def from_environment(device_name: str = "auto", env=None) -> DataParallel | None:
    """D1: this process's DataParallel from torchrun's environment, its device set and its groups made; None without one
    (no WORLD_SIZE, or WORLD_SIZE 1: today's single process, nothing is initialised). device_name: TrainConfig.device
    ("auto": CUDA when available). With ENV_DEVICES set, this rank's CUDA_VISIBLE_DEVICES (the process's environment,
    which CUDA reads when it initialises) is set to its own entry first; CUDA must not have been initialised before
    (a process that then sees more than one device is refused by name)."""
    env = os.environ if env is None else env
    world = int(env.get("WORLD_SIZE", "1") or 1)
    if world <= 1:
        return None
    rank = int(env["RANK"])
    local_rank = int(env.get("LOCAL_RANK", rank))
    local_world = int(env.get("LOCAL_WORLD_SIZE", world))
    own = rank_devices(env, local_rank, local_world)
    if own is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = own
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    kind = torch.device(device_name).type
    asked = env.get(ENV_BACKEND, "").strip()  # the backend named in the environment ("" when it is the default)
    backend = asked or ("nccl" if kind == "cuda" else "gloo")
    if kind == "cuda":
        n = torch.cuda.device_count()
        if own is not None and n != 1:
            raise ValueError(f"data parallelism: {ENV_DEVICES} gives local rank {local_rank} the device {own!r}, but this "
                             f"process sees {n} CUDA devices: CUDA was initialised before data_parallel.from_environment")
        index = device_index(n, local_rank, local_world, backend, asked, own is not None)
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    else:
        device = torch.device(kind)
    return init(rank, world, local_rank, device, backend)


# ------------------------------------------------------------------------------------------------
# D3: the flat gradient buffer


class GradBuffer:
    """One flat gradient buffer per (device, dtype) of the parameters that require a gradient, in named_parameters
    order, with a contiguous view of it for every parameter (module docstring, D3). begin_step attaches every view as its
    parameter's .grad and zeroes the buffers; while tracking, a post-accumulate-grad hook records which parameters the
    backward reached; detach_untouched gives the others .grad None again."""

    ALIGN = 64  # elements: every view starts on a 256-byte boundary (float32), as the allocator aligns a tensor of its own

    def __init__(self, named_params, layout=None):
        """layout: the sharded optimizer's owner-ordered layout (ced/train/zero1.py ShardLayout, over these parameters in
        this order, one device and dtype): each view at its offset, the buffer W equal slices. None: model order."""
        self.names, self.params = [], []
        for name, p in named_params:
            if p.requires_grad:
                self.names.append(name)
                self.params.append(p)
        sizes: dict = {}
        offsets = []
        if layout is not None:
            key = (self.params[0].device, self.params[0].dtype)
            assert all((p.device, p.dtype) == key for p in self.params) and len(layout.offsets) == len(self.params)
            offsets, sizes = list(layout.offsets), {key: layout.numel}
        for p in (self.params if layout is None else ()):
            key = (p.device, p.dtype)
            offsets.append(sizes.get(key, 0))
            sizes[key] = -(-(sizes.get(key, 0) + p.numel()) // self.ALIGN) * self.ALIGN
        self.flats = {key: torch.zeros(n, dtype=key[1], device=key[0]) for key, n in sizes.items()}
        # Each view is a tensor of its own over the flat buffer's storage (Tensor.set_), not an autograd view of it: a view
        # shares its base's version counter, the linear saves its buffer for backward (wgrad_accum.py
        # _FusedWgradLinearFunction), so one layer's in-place accumulation would fail every other layer's saved-tensor
        # check. With set_ each gradient has its own counter, as a gradient tensor of its own has; the memory is shared.
        self.views = []
        for p, o in zip(self.params, offsets):
            flat = self.flats[(p.device, p.dtype)]
            v = torch.empty(0, dtype=p.dtype, device=p.device)
            v.set_(flat.untyped_storage(), o, p.shape, torch.empty(p.shape, device="meta").stride())
            self.views.append(v)
        self._index = {id(p): i for i, p in enumerate(self.params)}
        self.touched = bytearray(len(self.params))
        self._left = len(self.params)
        self._handles: list = []

    def __len__(self) -> int:
        return len(self.params)

    def numel(self) -> int:
        return sum(f.numel() for f in self.flats.values())

    def view_of(self, p: torch.Tensor) -> torch.Tensor:
        """The view that is p's gradient (buffers: ced/train/wgrad_accum.py WgradAccum's grad_of)."""
        return self.views[self._index[id(p)]]

    def begin_step(self, track: bool = True) -> None:
        """Every view attached as its parameter's .grad, the buffers zeroed (one memset each), the record of reached
        parameters emptied, and (track) the hooks that fill it armed."""
        for p, v in zip(self.params, self.views):
            if p.grad is not v:
                p.grad = v
        for f in self.flats.values():
            f.zero_()
        self.touched = bytearray(len(self.params))
        self._left = len(self.params)
        if track:
            self._arm()
        else:
            self.stop_tracking()

    def _arm(self) -> None:
        if self._handles:
            return

        def make(i):
            def hook(_p):
                if not self.touched[i]:
                    self.touched[i] = 1
                    self._left -= 1
            return hook

        self._handles = [p.register_post_accumulate_grad_hook(make(i)) for i, p in enumerate(self.params)]

    def after_backward(self) -> None:
        """After a micro-batch's backward: once every parameter has been reached, the hooks are removed for the rest of
        the step (they cannot add anything; about 1.2 us a parameter a micro-batch on the VM's CPU, measured)."""
        if self._handles and self._left == 0:
            self.stop_tracking()

    def stop_tracking(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def touched_flags(self) -> list[int]:
        return list(self.touched)

    def detach_untouched(self, flags: list[int]) -> None:
        """.grad None for the parameters whose flag is 0 (no rank's backward reached them): as the single process leaves
        them. Their views stay zero and in the buffers."""
        for p, f in zip(self.params, flags):
            if not f:
                p.grad = None

    def all_reduce_(self, dp: DataParallel) -> None:
        """Each flat buffer summed over the ranks in place (one SUM all_reduce each, on the tensor group)."""
        for f in self.flats.values():
            dp.all_reduce_(f)


# ------------------------------------------------------------------------------------------------
# the debug replica check


def _bits(t: torch.Tensor) -> torch.Tensor:
    """t's bit patterns as an integer tensor of the same bytes (2-byte types as int16, others as int32 words)."""
    t = t.detach().contiguous().reshape(-1)
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def checksums(tensors) -> torch.Tensor:
    """Two int64 checksums a tensor, on its device: the sum of its bit patterns, and the sum of the products of each
    pattern's low and high 16 bits (exact integer sums, so the same whatever the reduction order; equal tensors give equal
    checksums, and a replica that has drifted changes them with overwhelming probability)."""
    out = []
    for t in tensors:
        x = _bits(t).to(torch.int64)
        lo, hi = x & 0xFFFF, (x >> 16) & 0xFFFF
        out.append(torch.stack([x.sum(), (lo * hi).sum()]))
    return torch.stack(out) if out else torch.zeros((0, 2), dtype=torch.int64)


def check_replicas(dp: DataParallel, named_params, step: int) -> None:
    """The ranks' parameter checksums compared (two all_reduces on the tensor group, MAX and MIN); on a difference every
    rank gathers them on the host group and raises ReplicaMismatch naming the first parameter that differs and each
    rank's checksums of it."""
    named = list(named_params)
    local = checksums([p for _, p in named])
    hi, lo = dp.all_reduce_(local.clone(), "MAX"), dp.all_reduce_(local.clone(), "MIN")
    differ = (hi != lo).any(dim=1)
    if not bool(differ.any()):
        return
    i = int(differ.nonzero()[0, 0])  # the failure path only
    every = dp.gather_host(local[i].cpu().tolist())
    raise ReplicaMismatch(f"data parallelism: the ranks' parameters differ after step {step}: {named[i][0]} has checksums "
                          + ", ".join(f"rank {r} {c}" for r, c in enumerate(every))
                          + f" ({int(differ.sum())} of {len(named)} tensors differ)")


# ------------------------------------------------------------------------------------------------
# what is not built under data parallelism


def refusals(trainer) -> list[str]:
    """Why the trainer's config is not built under data parallelism at its world size, by name ([] when it is)."""
    cfg, dp = trainer.cfg, trainer.dp
    W, B, micro = dp.world, cfg.data.batch_size, trainer.micro
    m = cfg.model
    out = []
    if B % W:
        out.append(f"data.batch_size {B}: not a whole number of rows a rank at world size {W}")
    elif (B // W) % micro:
        out.append(f"micro_batch_size {micro}: does not divide the {B // W} rows a rank reads at world size {W} (each rank "
                   "runs whole micro-batches of the single process's size)")
    elif any((B // W) % rows for rows in cfg.micro_batch_by_t):
        out.append(f"micro_batch_by_t {list(cfg.micro_batch_by_t)}: each entry must divide the {B // W} rows a rank reads")
    if m.arch == "looped_ced":
        out.append("arch 'looped_ced' (Looped-CED v1: trimmed slots by micro-batch, drafts, MoE)")
    if cfg.shape_schedule:
        out.append("shape_schedule (the batch-shape schedule: its phases place the steps in the stream by whole batches)")
    if cfg.tst.schedule and cfg.stage != "none":
        # Token Superposition Training itself is built under data parallelism (TST_DP below; tests/test_dp_tst.py); its stage after
        # the last switch (TrainConfig.stage "continue": a resume_from onto a new shard set, ced/train/tst.py) is not tested here
        out.append("tst.schedule in a stage (stage 'continue'; Token Superposition Training itself is built under data "
                   "parallelism, its stage after the last switch is not tested there)")
    if m.loop.t_sampling == "sequence":
        out.append("model.loop.t_sampling 'sequence' (a pass count per row: the step's multiset is drawn over the whole "
                   "batch)")
    if trainer.schedule is not None:
        out.append("model.loop.segment_passes / unscored_passes (the segment pass schedule)")
    if trainer._entmax:
        out.append("the asentmax normaliser (its training statistics are counted per process)")
    return out


def check(trainer) -> None:
    bad = refusals(trainer)
    if bad:
        raise ValueError(f"data parallelism (world size {trainer.dp.world}) is not built for: {'; '.join(bad)}")


# ------------------------------------------------------------------------------------------------
# D5: each rank's CUDA random stream


def rank_seed(seed: int, rank: int, step: int) -> int:
    """The seed of rank `rank`'s CUDA generator from optimizer step `step` on (D5): a 63-bit number from blake2b of (the
    run's seed, the rank, the step). Set at a data-parallel run's start (step 0) and whenever a resume changes the
    world size; at the same world size each rank restores its own saved state instead. Distinct ranks get independent
    streams (M1's review: with one seed every rank drew the same numbers for different rows, e.g. NVFP4's stochastic
    rounding)."""
    h = hashlib.blake2b(f"ced-dp-cuda-rng/{int(seed)}/{int(rank)}/{int(step)}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little") & (2**63 - 1)


def batch_digest(hex_digest: str) -> int:
    """A batch_hash (8 bytes, hex) as a signed int64, to travel in the host reduction."""
    return int.from_bytes(bytes.fromhex(hex_digest), "little", signed=True)


def combined_hash(digests: list[int]) -> str:
    """The step's batch hash under data parallelism: blake2b (8 bytes) of the ranks' batch hashes in rank order. It names
    the global batch at this world size (it is not the single process's batch_hash of the same rows)."""
    h = hashlib.blake2b(digest_size=8)
    for d in digests:
        h.update(int(d).to_bytes(8, "little", signed=True))
    return h.hexdigest()


def digest_hex(d: int) -> str:
    return int(d).to_bytes(8, "little", signed=True).hex()
