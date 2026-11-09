"""The sharded optimizer, ZeRO-1 by whole matrix.

OFF BY DEFAULT. TrainConfig.optimizer_sharding "replicated" (the default) is today's data parallelism (every rank holds the
whole optimizer state and steps every tensor); "zero1" turns this on, and is read under data parallelism only (one process
owns every tensor: the key changes nothing there). With "replicated" nothing here is imported into a run, and the key is
absent from a checkpoint's saved config, so every checkpoint of a replicated run is byte for byte what it was (T6).

WHAT IS SHARDED. Each rank OWNS whole parameter tensors (Muon, NorMuon and Hyperball need a whole matrix; ANVIL-II's units
stay inside one matrix): the owner alone keeps the tensor's optimizer state and updates it. The owners are assigned once a
process by the tensors' sizes (assign_owners: largest first, each to the rank with the fewest elements so far; the
embedding and the head are tensors like any other). Balanced by elements, not by state bytes, so that every rank's slice
of the flat buffers below is about the same length (their common length is the largest rank's, and every rank holds all
W slices of the gradients): at 4B the common slice is 0.1 % longer than the ideal 1/8 (scripts/zero1_4b_layout.py).

THE TWO FLAT BUFFERS, ORDERED BY OWNER (ShardLayout). The parameters' fp32 master weights and their gradients each live in
one flat buffer of W equal slices; slice r holds rank r's tensors, in model order, each at a 64-element (256-byte)
boundary, the slice padded to the common length. Every parameter IS its view of the parameter buffer (Tensor.set_ at the
trainer's start, before the layers are compiled and before any CUDA graph holds an address), and every gradient its view of
the gradient buffer (data_parallel.GradBuffer with this layout; buffers and N1's persistent gradients are those views, as
in stage 1).

THE STEP (Trainer.train_step, cuda_graphs.GraphRunner):
1. The micro-batches accumulate every rank's partial gradients in the gradient buffer, as in stage 1.
2. ONE reduce-scatter (in place) of the gradient buffer replaces stage 1's all-reduce: rank r's slice then holds the
   global gradient of its own tensors (the other slices hold this rank's partial sums, which nothing reads). The step's
   loss and metric sums keep their packed all-reduce.
3. The gradient metrics that read whole gradients (pass 2's, the residual scalars', the injection's and the input mix's
   gradient norms; the exact-zero counts) are computed by each tensor's owner and gathered in one packed SUM all-reduce of
   per-tensor values, zero where a rank does not own the tensor (x + 0 = x: an exact gather): gather_values.
4. The clip: each owner computes its tensors' norms as torch.nn.utils.clip_grad_norm_ computes them (torch._foreach_norm,
   or vector_norm a tensor where foreach is not supported); one SUM all-reduce of the vector of per-tensor norms (an exact
   gather, as above) gives every rank the same vector clip_grad_norm_ stacks, in the same order (the parameters with a
   gradient, in model order), and so the same total norm and clip coefficient, bit for bit; each owner scales its own
   gradients. The plan's "one scalar all-reduce of the shards' squared norms" would round differently from the replicated
   clip; the vector costs one all-reduce of one float a tensor (about 650 at 4B) instead of one.
5. The optimizer steps the owned tensors only: every other tensor's .grad is None during the step (the optimizers skip a
   tensor without a gradient: no state is made for it and it is not updated), and is attached again after it.
6. ONE MAX on the host group (the one host collective N2 adds a step): whether any rank's optimizer step raised, and
   whether any rank's ANVIL-II matrices shipped their end-of-run blend under the per-bank renorm (anvil2_decon "bank":
   one ratio a bank over every matrix's rows of it). The renorm is then made from every owner's per-matrix sums, gathered
   exactly and added on every rank in the replicated order (bank_decon, bank_gather: bit for bit; the plan allowed
   "within rounding"), then ONE all-gather (in place) of the parameter buffer: every rank has every updated weight, the
   same bits (an all-gather copies; gloo keeps the sign of -0.0, measured). On a failure no rank all-gathers, and every
   rank takes the trainer's failure path at the same update with the step's (global) loss and gradient norm.
So a sharded step does what the replicated step does, operation for operation on every tensor, on another rank: T7
(tests/test_zero1.py) checks it bit for bit against the replicated run for Muon, NorMuon, Hyperball over Muon and ANVIL-II
(the style and the record's per-bank renorm), on the CPU with gloo. Under cuda_graphs "step" the vector of norms is
gathered eagerly before the optimizer graph, whose replay then makes the total norm, the clip and the orthogonalisation of
the owned matrices.

THE CHECKPOINT, BY PARAMETER (save_shard, manifest, load_for): every rank writes its owned tensors' optimizer state, keyed by
parameter name, to <checkpoints>/step_<step>.optim/rank<r>-of-<W>.pt; then rank 0 writes the step's file as today (model,
config, RNG, data state, ...) whose "optimizer" entry holds the optimizer's param_groups, an empty "state" and the manifest
"by_parameter" (the world size, the files, each saver index's parameter name and owner). Any world size, sharded or
replicated, loads any such file set: it reads the state of the parameters it owns (torch.load with mmap, so a rank pages
in only what it takes), and a replicated process takes all of them (full_optimizer_state: also what a script that reads
state["optimizer"]["state"] needs). A sharded process loads a replicated file the same way, keeping its own tensors' state.
A file set that is not the manifest's (a file it names missing, or a file of another world size under its name) is refused
by name, a missing file on every rank (states_by_name).

NOT BUILT, REFUSED BY NAME (check): optimizers other than Muon (MuonWithAdamW), NorMuon, Hyperball over Muon and ANVIL-II
(T7's four: KL-SOAP, Shampoo, AdamW, our optimizer's flags and its baselines are per-tensor too, but untested here);
Hyperball over another rule; parameters of more than one dtype or device (the clip's norms are one vector); the tensor
group on gloo with CUDA tensors (the one-slice check of stage 1: its reduce-scatter and all-gather there are untested).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

MODES = ("replicated", "zero1")
ALIGN = 64  # elements: every tensor starts on a 256-byte boundary of its flat buffer (data_parallel.GradBuffer.ALIGN)
SHARD_SUFFIX = ".optim"  # step_<step>.pt's shard directory: step_<step>.optim/
BY_PARAMETER = "by_parameter"  # the manifest's key in a checkpoint's "optimizer" entry


def shard_dir(path: Path) -> Path:
    """The directory of a checkpoint file's optimizer shards (step_<step>.pt -> step_<step>.optim)."""
    return Path(path).with_suffix(SHARD_SUFFIX)


def shard_name(rank: int, world: int) -> str:
    return f"rank{rank:03d}-of-{world:03d}.pt"


def aligned(n: int) -> int:
    return -(-int(n) // ALIGN) * ALIGN


def assign_owners(sizes: list[int], world: int, groups: list[list[int]] | None = None) -> list[int]:
    """The rank that owns each tensor: largest first (ties by model order), each to the rank whose slice has the fewest
    (aligned) elements so far (ties to the lower rank). A pure function of the sizes and the world size, so every rank
    computes the same assignment.
    groups (model.fused_projections, ced/model/fused_proj.py; None or [] for every other run, and then the loop is the one
    it always was): lists of tensor indices, consecutive in model order, that one rank owns together; such a group is ONE
    unit of the order (its summed aligned size, ranked by its first index), so its tensors land back to back in that
    rank's slice (the slice holds its tensors in model order) and the group's concatenated weight is a view of the flat
    buffer when every part but the last is a whole number of ALIGN-element blocks."""
    load = [0] * world
    out = [0] * len(sizes)
    if not groups:
        for i in sorted(range(len(sizes)), key=lambda i: (-sizes[i], i)):
            r = min(range(world), key=lambda r: (load[r], r))
            out[i] = r
            load[r] += aligned(sizes[i])
        return out
    member = {i for g in groups for i in g}
    units = [list(g) for g in groups] + [[i] for i in range(len(sizes)) if i not in member]
    for u in sorted(units, key=lambda u: (-sum(sizes[i] for i in u), u[0])):
        r = min(range(world), key=lambda r: (load[r], r))
        for i in u:
            out[i] = r
            load[r] += aligned(sizes[i])
    return out


class ShardLayout:
    """The owner-ordered layout of the flat parameter and gradient buffers (module docstring): owners[i], offsets[i] (the
    element offset of tensor i, in model order), seg (the common slice length), numel (world x seg). groups: see
    assign_owners (None: the layout it always was)."""

    def __init__(self, sizes: list[int], world: int, groups: list[list[int]] | None = None):
        self.world = world
        self.sizes = [int(n) for n in sizes]
        self.owners = assign_owners(self.sizes, world, groups)
        local = [0] * world
        rel = [0] * len(sizes)
        for i, n in enumerate(self.sizes):  # within a slice: the owner's tensors in model order
            r = self.owners[i]
            rel[i] = local[r]
            local[r] += aligned(n)
        self.seg = max(max(local), ALIGN)
        self.offsets = [self.owners[i] * self.seg + rel[i] for i in range(len(sizes))]
        self.numel = world * self.seg
        self.loads = local  # each rank's aligned elements (before the padding to seg)

    def describe(self) -> dict:
        total = sum(self.sizes)
        return {"world_size": self.world, "tensors": len(self.sizes), "elements": total, "slice": self.seg,
                "padding": self.numel - total, "per_rank_elements": [sum(n for n, o in zip(self.sizes, self.owners) if o == r)
                                                                     for r in range(self.world)],
                "per_rank_tensors": [self.owners.count(r) for r in range(self.world)]}


# ------------------------------------------------------------------------------------------------
# what is built


def refusals(trainer) -> list[str]:
    """Why the trainer's config is not built under optimizer_sharding "zero1", by name ([] when it is)."""
    from ced.train.matrix_optim import MatrixWithAdamW
    from ced.train.muon import MuonWithAdamW

    opt = trainer.optimizer
    out = []
    if isinstance(opt, MatrixWithAdamW):
        rules = {g.get("rule") for g in opt.param_groups if g.get("kind") == "matrix"}
        hyper = any(g.get("hyperball") for g in opt.param_groups if g.get("kind") == "matrix")
        ours = [k for k in ("m1", "m2", "m3", "m4") if opt.ours.get(k)]
        if ours:
            out.append(f"optim.ours {ours} (our optimizer's flags: not tested sharded)")
        for rule in sorted(r for r in rules if r is not None):
            if rule == "muon" and not hyper:
                out.append("the matrix rule 'muon' without Hyperball (optim.name 'muon' is MuonWithAdamW)")
            elif rule not in ("muon", "normuon", "anvil2"):
                out.append(f"the matrix rule {rule!r} (built for Muon, NorMuon, Hyperball over Muon and ANVIL-II: T7)")
            elif hyper and rule != "muon":
                out.append(f"Hyperball over {rule!r} (built: Hyperball over Muon)")
    elif not isinstance(opt, MuonWithAdamW):
        out.append(f"the optimizer {type(opt).__name__} (optim.name {trainer.cfg.optim.name!r}; built for Muon, NorMuon, "
                   "Hyperball over Muon and ANVIL-II: T7)")
    keys = {(p.device, p.dtype) for p in trainer.model.parameters() if p.requires_grad}
    if len(keys) > 1:
        out.append(f"parameters of {len(keys)} (device, dtype) pairs (the clip's norms are one vector)")
    dp = getattr(trainer, "dp", None)
    if dp is not None and dp.backend == "gloo" and torch.device(dp.device).type == "cuda":
        out.append("the tensor group on gloo with CUDA tensors (CED_DP_BACKEND=gloo, the one-slice check): its reduce-scatter "
                   "and all-gather on CUDA tensors are not tested (stage 1 there needs only the all-reduce); N2 runs on NCCL")
    return out


def check_mode(mode) -> None:
    if mode not in MODES:
        raise ValueError(f"optimizer_sharding {mode!r}: one of {MODES}")


def check(trainer) -> None:
    bad = refusals(trainer)
    if bad:
        raise ValueError(f"optimizer_sharding 'zero1' is not built for: {'; '.join(bad)}. Leave it 'replicated'")


# ------------------------------------------------------------------------------------------------
# the sharded optimizer of one rank


class Zero1:
    """One rank's part of the sharded optimizer (module docstring). Made by Trainer.__init__ under data parallelism with
    optimizer_sharding "zero1", after the model and the optimizer and before the gradient buffer, and the compile: it
    seats every parameter in the flat parameter buffer."""

    def __init__(self, model: torch.nn.Module, optimizer, dp):
        self.dp = dp
        self.optimizer = optimizer
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.names = [n for n, _ in named]
        self.params = [p for _, p in named]
        self.index = {id(p): i for i, p in enumerate(self.params)}
        # model.fused_projections (ced/model/fused_proj.py): each fused group one owner's, back to back; [] otherwise
        groups = None
        if any(getattr(m, "fused_proj", None) for m in model.modules()):
            from ced.model import fused_proj

            groups = fused_proj.index_groups(model, self.params)
        self.layout = ShardLayout([p.numel() for p in self.params], dp.world, groups)
        self.owner = {id(p): r for p, r in zip(self.params, self.layout.owners)}
        p0 = self.params[0]
        self.device, self.dtype = p0.device, p0.dtype
        # the flat parameter buffer: every parameter seated in it (its values copied first; the old storage is freed)
        self.flat = torch.zeros(self.layout.numel, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            for p, o in zip(self.params, self.layout.offsets):
                n = p.numel()
                self.flat[o:o + n].copy_(p.detach().reshape(-1))
                p.set_(self.flat.untyped_storage(), o, p.shape, torch.empty(p.shape, device="meta").stride())
        seg = self.layout.seg
        self.segment = self.flat[dp.rank * seg:(dp.rank + 1) * seg]
        self._norm_vec = None  # the clip's vector of per-tensor norms (a persistent buffer: the optimizer graph reads it)
        self._norm_index = None  # (positions of the owned tensors in it, the owned tensors), by the clip set
        self._clip_key = None
        from ced.train.matrix_optim import MatrixWithAdamW

        if isinstance(optimizer, MatrixWithAdamW) and optimizer.anvil.get("decon") == "bank":
            optimizer.bank_deferred = True  # ANVIL-II's per-bank renorm: the owners' sums gathered (after_step, bank_decon)

    # --- ownership ---------------------------------------------------------------------------------------------------
    def owns(self, p) -> bool:
        return self.owner.get(id(p)) == self.dp.rank

    def owned_names(self) -> list[str]:
        return [n for n, p in zip(self.names, self.params) if self.owns(p)]

    # --- the collectives ---------------------------------------------------------------------------------------------
    def _group(self):
        return self.dp.tensor_group

    def reduce_scatter_(self, flat: torch.Tensor) -> None:
        """The gradient buffer reduced over the ranks into this rank's slice, in place (module docstring, step 2)."""
        import torch.distributed as dist

        seg = self.layout.seg
        out = flat[self.dp.rank * seg:(self.dp.rank + 1) * seg]
        fn = getattr(dist, "reduce_scatter_single", None) or dist.reduce_scatter_tensor
        fn(out, flat, op=dist.ReduceOp.SUM, group=self._group())

    def all_gather_(self) -> None:
        """Every rank's slice of the parameter buffer to every rank, in place (step 6): the updated weights."""
        import torch.distributed as dist

        fn = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor
        fn(self.flat, self.segment, group=self._group())

    def gather_values(self, params: list, fn) -> list:
        """[fn(p.grad) for p in params] as every rank would compute it with the whole gradients: each owner computes its
        tensors' values (0-dim tensors in the gradients' dtype, the one dtype of a sharded run), one SUM all-reduce of a
        float64 vector with zeros elsewhere (an exact gather: x + 0 = x, and every float32 or bfloat16 value is exact in
        float64), each value back in the gradients' dtype. Every p must have a gradient (the caller passes tensors whose
        .grad is not None, the same set on every rank)."""
        if not params:
            return []
        vec = torch.zeros(len(params), dtype=torch.float64, device=self.device)
        for i, p in enumerate(params):
            if self.owns(p):
                vec[i] = fn(p.grad).to(torch.float64)
        self.dp.all_reduce_(vec)
        return [vec[i].to(self.dtype) for i in range(len(params))]

    def gather_rows(self, params: list, fn, width: int, none_row: list) -> torch.Tensor:
        """torch.stack([fn(p.grad) if p.grad is not None else none_row for p in params]) as every rank would compute it with
        the whole gradients (int64 rows: the exact-zero counts, ced/train/zero_wgrad.py counts): each owner's rows, zeros
        elsewhere, one SUM all-reduce. `params` and which of them have a gradient are the same on every rank."""
        out = torch.zeros((len(params), width), dtype=torch.int64, device=self.device)
        for i, p in enumerate(params):
            if self.owns(p):
                out[i] = fn(p.grad) if p.grad is not None else torch.tensor(none_row, dtype=torch.int64, device=self.device)
        self.dp.all_reduce_(out)
        return out

    # --- the clip ----------------------------------------------------------------------------------------------------
    def gather_norms(self, clip_set: list) -> torch.Tensor:
        """Step 4's vector: the norm of every gradient of clip_set (the parameters with a gradient, in model order, the
        same on every rank) as clip_grad_norm_ computes it, gathered from the owners into a persistent buffer, which is
        returned. Run before the owners' gradients are scaled."""
        from torch.utils._foreach_utils import _has_foreach_support

        key = tuple(id(p) for p in clip_set)
        if key != self._clip_key:  # once a process in practice (the set of tensors with a gradient is the run's)
            pos = [i for i, p in enumerate(clip_set) if self.owns(p)]
            self._norm_index = (torch.tensor(pos, dtype=torch.long, device=self.device), [clip_set[i] for i in pos])
            self._norm_vec = torch.zeros(len(clip_set), dtype=self.dtype, device=self.device)
            self._clip_key = key
        vec = self._norm_vec
        vec.zero_()
        pos, owned = self._norm_index
        if owned:
            grads = [p.grad for p in owned]
            if _has_foreach_support(grads, grads[0].device):
                norms = torch._foreach_norm(grads, 2.0)
            else:
                norms = [torch.linalg.vector_norm(g, 2.0) for g in grads]
            vec.index_copy_(0, pos, torch.stack(norms).to(vec.dtype))
        self.dp.all_reduce_(vec)
        return vec

    def clip_from_norms(self, clip_set: list, max_norm: float) -> torch.Tensor:
        """The total norm of the gathered vector (clip_grad_norm_'s vector_norm of the stacked norms) and the owners'
        gradients scaled by clip_grad_norm_'s coefficient (torch.nn.utils.clip_grads_with_norm_). Returns the total norm.
        No collective: the step graph of cuda_graphs "step" holds it."""
        if not clip_set:
            return torch.tensor(0.0)
        total = torch.linalg.vector_norm(self._norm_vec, 2.0)
        owned = self._norm_index[1]
        if owned:
            clip_grads_with_norm_(owned, max_norm, total)
        return total

    def clip(self, clip_set: list, max_norm: float) -> torch.Tensor:
        """clip_grad_norm_(clip_set, max_norm) over the shards (module docstring, step 4): the total norm, the same bits on
        every rank, and the owned gradients scaled."""
        if not clip_set:
            return torch.tensor(0.0)
        self.gather_norms(clip_set)
        return self.clip_from_norms(clip_set, max_norm)

    # --- the optimizer step ------------------------------------------------------------------------------------------
    def detach_others(self) -> list:
        """Every tensor this rank does not own gets .grad None for the optimizer step; returns [(p, its gradient)] to put
        back after it (attach)."""
        saved = []
        for p in self.params:
            if p.grad is not None and not self.owns(p):
                saved.append((p, p.grad))
                p.grad = None
        return saved

    @staticmethod
    def attach(saved: list) -> None:
        for p, g in saved:
            p.grad = g

    def after_step(self, error: BaseException | None):
        """After this rank's optimizer step (module docstring, step 6). One MAX on the host group: whether any rank's step
        raised, and whether any rank's ANVIL-II matrices shipped their end-of-run blend with the per-bank renorm (deferred
        to here: bank_decon). If none raised: the renorm when due, then the all-gather of the updated weights, and None.
        If one raised: no all-gather, and every rank's exception: its own, or a RuntimeError naming the failing ranks and
        the first one's error (the failure path's host gather)."""
        with torch.no_grad():
            return self._after_step(error)

    def _after_step(self, error):
        opt = self.optimizer
        pending = getattr(opt, "_bank_pending", None) or []
        failed, shipped = self.dp.max_host([int(error is not None), int(bool(pending))])
        if failed:
            every = self.dp.gather_host(None if error is None else f"{type(error).__name__}: {error}")
            bad = [r for r, e in enumerate(every) if e is not None]
            if error is not None:
                return error
            return RuntimeError(f"the sharded optimizer step failed on rank(s) {bad} of {self.dp.world}: {every[bad[0]]}")
        if shipped:
            self.bank_decon(opt, pending)
        self.all_gather_()
        return None

    # --- ANVIL-II's per-bank renorm (anvil2_decon "bank"; ced/train/matrix_optim.py) ---------------------------------
    def bank_decon(self, opt, pending: list) -> None:
        """ANVIL-II's per-bank renorm at the run's last step under sharding (MatrixWithAdamW.bank_deferred): for every matrix
        group with anvil2_decon "bank", in order, every rank takes part (bank_gather) and each owner applies the ratios to
        its shipped matrices (MatrixWithAdamW._apply_bank_ratio). Deferring it from the matrix group's step to the end of
        the optimizer step changes nothing: it touches only the group's matrices, which nothing after the group reads."""
        mine = {id(group): vals for group, vals in pending}
        for group in opt.param_groups:
            if group.get("kind") != "matrix" or group.get("rule") != "anvil2" or opt.anvil.get("decon") != "bank":
                continue
            norms, shipped = self.bank_gather(opt, group, mine.get(id(group), []))
            if norms is not None and shipped:
                opt._apply_bank_ratio(norms, shipped)
        opt._bank_pending = []

    def bank_gather(self, opt, group: dict, vals: list):
        """The per-bank sums of one matrix group: vals = [(p, which, [the sum of squares of each of p's segments, in
        float64])] of this rank's matrices that shipped (MatrixWithAdamW._bank_add, deferred; which 0 before the blend, 1
        after). One SUM all-reduce of [pre, post] a segment of every matrix of the group (zeros where a rank does not own
        it: an exact gather), the shipped matrices' indices gathered on the host group, then every rank adds the shipped
        matrices' sums bank by bank in the group's order, as the replicated _bank_add accumulates them: the same float64
        additions in the same order, so the same sums and ratios. Returns ({bank: [pre, post]}, this rank's shipped
        matrices in the group's order), or (None, []) when none shipped."""
        mats = group["params"]
        segs = [opt.matrix_info[p]["segments"] or [] for p in mats]
        base, at = [], 0
        for s in segs:
            base.append(at)
            at += 2 * len(s)
        vec = torch.zeros(max(at, 1), dtype=torch.float64, device=self.device)
        index = {id(p): j for j, p in enumerate(mats)}
        mine = set()
        for p, which, sums in vals:
            j = index[id(p)]
            mine.add(j)
            for s, v in enumerate(sums):
                vec[base[j] + 2 * s + which] = v
        self.dp.all_reduce_(vec)
        shipped = sorted(set().union(*self.dp.gather_host(sorted(mine))))
        if not shipped:
            return None, []
        norms: dict = {}
        for j in shipped:
            for s, (_, _, bank) in enumerate(segs[j]):
                sums = norms.setdefault(bank, [torch.zeros((), dtype=torch.float64, device=self.device)] * 2)
                sums[0] = sums[0] + vec[base[j] + 2 * s]
                sums[1] = sums[1] + vec[base[j] + 2 * s + 1]
        return norms, [mats[j] for j in sorted(mine)]

    # --- the checkpoint, by parameter ---------------------------------------------------------------------------------
    def index_names(self, model: torch.nn.Module) -> list[str]:
        """The parameter name of each index of the optimizer's state_dict (its param_groups' order)."""
        names = {id(p): n for n, p in model.named_parameters()}
        return [names[id(p)] for g in self.optimizer.param_groups for p in g["params"]]

    def shard_state(self, model: torch.nn.Module) -> dict:
        """This rank's optimizer state, keyed by parameter name (the owned tensors that have state)."""
        names = self.index_names(model)
        sd = self.optimizer.state_dict()
        return {names[i]: st for i, st in sd["state"].items()}

    def shard_path(self, path: Path) -> Path:
        """This rank's optimizer file of the checkpoint `path`: step_<step>.optim/rank<r>-of-<W>.pt."""
        return shard_dir(path) / shard_name(self.dp.rank, self.dp.world)

    def shard_payload(self, model: torch.nn.Module, step: int) -> dict:
        """What this rank's optimizer file holds (save_shard writes it; the asynchronous save, N8, copies it)."""
        return {"rank": self.dp.rank, "world_size": self.dp.world, "step": int(step), "state": self.shard_state(model)}

    def save_shard(self, path: Path, model: torch.nn.Module, step: int) -> Path:
        """This rank's file of the checkpoint `path` (module docstring): its owned tensors' state by name, written
        atomically. Returns the file's path."""
        out = self.shard_path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".pt.tmp")
        torch.save(self.shard_payload(model, step), tmp)
        os.replace(tmp, out)
        return out

    def manifest(self, path: Path, model: torch.nn.Module) -> dict:
        """The "optimizer" entry of rank 0's file: param_groups, an empty state and the manifest (module docstring)."""
        if path is None:
            raise ValueError("a sharded run's state_dict is its checkpoint's manifest: it needs the checkpoint file's path "
                             "(Trainer.save_checkpoint passes it)")
        sd = self.optimizer.state_dict()
        names = self.index_names(model)
        owner = {n: self.owner[id(p)] for n, p in zip(self.names, self.params)}
        d = shard_dir(path).name
        return {"state": {}, "param_groups": sd["param_groups"],
                BY_PARAMETER: {"world_size": self.dp.world, "files": [f"{d}/{shard_name(r, self.dp.world)}"
                                                                      for r in range(self.dp.world)],
                               "names": names, "owners": [owner.get(n) for n in names],
                               "layout": self.layout.describe()}}


def clip_grads_with_norm_(params: list, max_norm: float, total_norm: torch.Tensor) -> None:
    """torch.nn.utils.clip_grads_with_norm_ (the second half of clip_grad_norm_)."""
    fn = getattr(torch.nn.utils, "clip_grads_with_norm_", None)
    if fn is None:  # an older torch: its private name
        from torch.nn.utils.clip_grad import _clip_grads_with_norm_ as fn
    fn(params, max_norm, total_norm)


# ------------------------------------------------------------------------------------------------
# loading a file set at any world size


def is_by_parameter(opt_state: dict) -> bool:
    return isinstance(opt_state, dict) and BY_PARAMETER in opt_state


def states_by_name(opt_state: dict, source: Path, want=None) -> dict:
    """{name: state entry} of a by-parameter checkpoint (opt_state: its "optimizer" entry; source: its file), read from the
    shard files with mmap (each rank reads only the files that hold names it wants); want (a set of names) limits it to
    those (a sharded rank's own tensors), None takes all.

    A shard set that is not the manifest's is refused by name (ValueError), as the launcher refuses it (scripts/launch_main.py
    shard_files, restore_shards; the two-slice probe's P3 refusal leg, scripts/dp_two_slice_probe.py): a manifest that does
    not name one file a rank of its world size; a file it names that is missing, checked for EVERY file before any is read,
    so that every rank refuses at the same place (not only the ranks whose tensors the file holds, while the others go on
    into the first collective), the message naming what the directory holds instead (a set of another world size is
    "rank<r>-of-<W'>.pt"); and a file whose own rank and world size (its payload's, Zero1.shard_payload) are not its place in
    the manifest: a file of another world size under the manifest's name, whose state would otherwise load silently for
    the names it shares and leave every other owned tensor without its state."""
    man = opt_state[BY_PARAMETER]
    base = Path(source).parent
    names, owners = man["names"], man["owners"]
    world, files = int(man["world_size"]), list(man["files"])
    if len(files) != world or len(set(files)) != world:
        raise ValueError(f"the checkpoint {source} is written by parameter, but its manifest names {files} at world size "
                         f"{world}: not one file a rank")
    absent = [f for f in files if not (base / f).exists()]
    if absent:
        first = base / absent[0]
        held = sorted(p.name for p in first.parent.iterdir()) if first.parent.is_dir() else None
        raise ValueError(f"the checkpoint {source} is written by parameter at world size {man['world_size']}: its "
                         f"optimizer shard {first} is missing"
                         + (f" ({len(absent)} of its {world} shards are)" if len(absent) > 1 else "")
                         + ("; the directory does not exist" if held is None else
                            f"; {first.parent.name}/ holds {held[:8] + (['...'] if len(held) > 8 else []) if held else 'nothing'}"))
    need_files = sorted({owners[i] for i, n in enumerate(names) if owners[i] is not None and (want is None or n in want)})
    out: dict = {}
    for r in need_files:
        f = base / files[r]
        part = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(part, dict) or part.get("rank") != r or part.get("world_size") != world:
            got = (part.get("rank"), part.get("world_size")) if isinstance(part, dict) else (None, None)
            raise ValueError(f"the checkpoint {source} is written by parameter at world size {world}: its optimizer shard "
                             f"{f} holds rank {got[0]} of world size {got[1]}, not rank {r} of {world}")
        for n, st in part["state"].items():
            if want is None or n in want:
                out[n] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    return out


def full_optimizer_state(opt_state: dict, source: Path) -> dict:
    """A by-parameter checkpoint's optimizer state as torch's Optimizer.state_dict() of the saver: {"state": {index: entry},
    "param_groups": ...}, every shard read. For scripts that read state["optimizer"]["state"] of a sharded run's file."""
    if not is_by_parameter(opt_state):
        return opt_state
    by_name = states_by_name(opt_state, source)
    names = opt_state[BY_PARAMETER]["names"]
    return {"state": {i: by_name[n] for i, n in enumerate(names) if n in by_name}, "param_groups": opt_state["param_groups"]}


def load_for(opt_state: dict, source: Path | None, optimizer, model: torch.nn.Module, z1: Zero1 | None) -> dict:
    """The optimizer state a process loads (Trainer.load_state_dict), as torch's state_dict in THIS optimizer's index
    space: from a by-parameter file set (each saved name mapped to this optimizer's index of that parameter), or from a
    replicated file (its own indices: today's mapping, also for a grown trunk, whose index i is this optimizer's index i).
    With z1, only the indices of the tensors this rank owns (a replicated file's other states are dropped on the host,
    before anything reaches the device)."""
    params = [p for g in optimizer.param_groups for p in g["params"]]
    if is_by_parameter(opt_state):
        if source is None:
            raise ValueError("a checkpoint written by parameter (optimizer_sharding 'zero1') is loaded with its file's path "
                             "(its optimizer shards are beside it)")
        own_names = {id(p): n for n, p in model.named_parameters()}
        index = {own_names[id(p)]: i for i, p in enumerate(params)}
        want = None if z1 is None else {own_names[id(p)] for p in params if z1.owns(p)}
        by_name = states_by_name(opt_state, source, want)
        missing = sorted(n for n in by_name if n not in index)
        if missing:
            raise ValueError(f"the checkpoint's optimizer state names parameters this model does not have: {missing[:8]}")
        state = {index[n]: st for n, st in by_name.items()}
        return {"state": state, "param_groups": opt_state["param_groups"]}
    if z1 is None:
        return opt_state
    keep = {i for i, p in enumerate(params) if z1.owns(p)}
    return {"state": {i: st for i, st in opt_state["state"].items() if i in keep}, "param_groups": opt_state["param_groups"]}
