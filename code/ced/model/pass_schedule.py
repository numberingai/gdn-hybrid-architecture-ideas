"""Segment-based pass counts: the part shared by every model family (2026-10-11).

What is here (nothing in this file names an architecture):

1. The config (LoopConfig.segment_passes, unscored_passes): `schedule_of` validates it and returns a PassSchedule, or
   None when neither key is set. None is "no schedule": a model then runs the code it always ran.
2. The plan of one forward (PassPlan): for every pass p >= 2, which positions run it, as fixed-shape index tensors.
   It is built on the host (`build_plan`) from the pass count of each "unit" of the dense pass:

       counts (B, C)   the passes unit c of row b runs: 0 = not a unit (an empty slot), 1, 2 or 3

   A unit is whatever the dense first pass of the arch runs on: a decoder slot of CED v2 (C = the decoder capacity),
   a position of the plain hybrid (C = L). Pass 1 is the arch's dense pass over all C units, unchanged. Pass p >= 2
   runs on K_p "pass slots" per row: the units with count >= p, in order, padded to K_p. K_p is the smallest of a
   fixed set of sizes (`bucket_sizes`: fractions of C) that holds every row's units of the pass; a pass no row needs
   is not in the plan. So no shape depends on the data beyond the choice among the fixed sizes.
3. The device side: `take` gathers a pass's input from the previous pass's layout (pass 3's slots are a subset of
   pass 2's, so pass 3 reads pass 2's output at its slots), `readout` gives every unit the state of the LAST pass it
   ran. Both are gathers with an exact inverse map (capacity.gather_rows), so their backward is a gather too: no
   scatter-add, no sort under torch.use_deterministic_algorithms, no boolean-mask indexing, no nonzero.

What an arch must provide to use it (CED v2's decoder was the first, ced/model/ced_v2.py `_decode_scheduled`; the
plain hybrid with a second pass through its top block is the second, ced/model/control.py `scheduled_passes`, where a
unit is a position, C = L, and a GDN layer of the block reads pass 1's state after the slot's own position):
- a dense first pass over the C units that also returns what the later passes read (pass 1's K/V per layer; for a
  GDN layer its pass-1 state), and that is its single-pass model bit for bit;
- `later_pass(state at K slots, the slots' positions, pass index)`: the shared block applied to the gathered slots,
  each slot reading pass 1's keys of its row under the mask row of ITS position, writing nothing;
- per pass its own gate and loop-index embedding (ced/model/loop.py GatedSecondPass builds 2 or 3 passes), started so
  that the pass is the identity;
- the head on `readout([z_1, z_2, ...], plan)`.
The position counts come from ced/train/pass_positions.py (the segment state machine on the CPU; the loader in
training, a streaming form in generation), which is arch-neutral too.

Masks: a pass's attention masks depend on its slots only, so an arch builds them once per pass and shares them
across the layers of the block; PassPlan.same[j] says that pass j + 2 runs at exactly the previous pass's slots (the
all-2 map through this path), where the previous pass's masks are reused as they are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ced.model.capacity import gather_rows, gather_slots, slot_inverse

# The keys of LoopConfig.segment_passes. The first four are the segments of ced.eval.token_types.SEGMENTS (the
# position's NEXT token is a generated assistant token in that segment); "other" is every other position: its next
# token is not a generated assistant token (prompt, tool output, plain web or code text), or the state machine cannot
# place it (token_types' "unplaced").
SEGMENT_KEYS = ("reasoning", "call_head", "content", "call_body", "other")
PASS_COUNTS = (1, 2, 3)
DEFAULT_COUNT = 2  # "default all 2": a key the map leaves out
# The slot sizes of a later pass, as fractions of C (TrainConfig.pass_slot_buckets; the default of the model's own
# host-side planning in evaluation and generation).
BUCKET_FRACTIONS = (1.0, 0.5, 0.25, 0.125, 0.0625)


@dataclass(frozen=True)
class PassSchedule:
    """A validated pass schedule (schedule_of).

    counts: None without a map, else {key: passes} over every key of SEGMENT_KEYS;
    unscored: None, or 1: an unscored unit (a sink, a replay-window position) runs pass 1 only;
    dense: the passes of a scored unit when there is no map (the model's loop count, loop.t_max)."""

    counts: dict | None
    unscored: int | None
    dense: int

    @property
    def max_passes(self) -> int:
        """The passes the model is built for: the map's largest value, or the loop count without a map."""
        return self.dense if self.counts is None else max(self.counts.values())

    def of(self, key: str) -> int:
        return self.dense if self.counts is None else self.counts[key]


def schedule_of(loop) -> PassSchedule | None:
    """The pass schedule a LoopConfig asks for, or None when it sets neither segment_passes nor unscored_passes
    (every config before these keys). Raises ValueError on a map that is not {SEGMENT_KEYS: 1 | 2 | 3}."""
    seg, unscored = getattr(loop, "segment_passes", None), getattr(loop, "unscored_passes", None)
    if seg is None and unscored is None:
        return None
    counts = None
    if seg is not None:
        if not isinstance(seg, dict):
            raise ValueError(f"loop.segment_passes {seg!r}: a map from {SEGMENT_KEYS} to 1, 2 or 3")
        unknown = sorted(set(seg) - set(SEGMENT_KEYS))
        if unknown:
            raise ValueError(f"loop.segment_passes: unknown segment(s) {unknown}; the segments are {SEGMENT_KEYS}")
        for k, v in seg.items():
            if isinstance(v, bool) or v not in PASS_COUNTS:
                raise ValueError(f"loop.segment_passes[{k!r}] = {v!r}: 1, 2 or 3")
        counts = {k: int(seg.get(k, DEFAULT_COUNT)) for k in SEGMENT_KEYS}
    if unscored is not None and (isinstance(unscored, bool) or unscored != 1):
        raise ValueError(f"loop.unscored_passes {unscored!r}: None (as a scored slot) or 1 (pass 1 only)")
    return PassSchedule(counts=counts, unscored=unscored, dense=int(loop.t_max))


def built_passes(loop) -> int:
    """How many passes a model of this LoopConfig has tensors for: loop.t_max, or with a segment map at least that
    and the map's largest value (so a map with a 3 adds the third pass's gate and loop-index embedding, and a map
    whose largest value is 1 or 2 adds nothing to the two-pass model)."""
    sched = schedule_of(loop)
    return int(loop.t_max) if sched is None else max(int(loop.t_max), sched.max_passes)


def bucket_sizes(capacity: int, multiple: int, fractions=BUCKET_FRACTIONS) -> list[int]:
    """The slot sizes a later pass may run at for a dense pass of `capacity` units, ascending: each fraction of the
    capacity rounded up to a multiple of `multiple` (at least one multiple, at most the capacity), and the capacity
    itself, which holds any row."""
    assert capacity >= 1 and multiple >= 1, (capacity, multiple)
    sizes = {capacity}
    for f in fractions:
        if not 0.0 < f <= 1.0:
            raise ValueError(f"pass_slot_buckets: fraction {f!r} outside (0, 1]")
        sizes.add(min(capacity, max(multiple, math.ceil(f * capacity / multiple) * multiple)))
    return sorted(sizes)


@dataclass
class PassPlan:
    """Which units run which pass in one forward (module docstring). Later pass j (0-based) is pass j + 2.

    capacity:  C, the units of the dense first pass per row
    index[j]:  (B, K_j) long: for each slot of the pass, the slot of the PREVIOUS layout whose state it continues: a
               unit in [0, C) for pass 2, a pass-2 slot in [0, K_0) for pass 3; the previous layout's size at a padding
               slot (the gathers read their fill row there)
    inverse[j]: (B, K_prev) long: the slot of this pass that continues each slot of the previous layout, K_j where
               none (the exact inverse of index[j]: the gather's backward)
    slot[j]:   (B, K_j) long: the unit in [0, C) each slot belongs to, C at a padding slot (for the slot's position,
               document, key count: metadata gathered from the dense layout)
    unit_slot[j]: (B, C) long: the slot of this pass that holds each unit, K_j where none (the exact inverse of
               slot[j]; for pass 2 it is inverse[0]): for an arch whose later pass gathers pass-1 records of the
               dense layout at its slots with a gather as the backward (the control's GDN state read)
    valid[j]:  (B, K_j) bool
    same[j]:   the pass runs at exactly the previous layout's slots, in the same order (its masks are that layout's)
    rows[j]:   None (every row has its slots in the tensors above: training), or (B_j,) long, the rows that have a
               slot in this pass when some have none (cached generation: the block then runs on those rows only)
    readout:   (B, C) long: the row of cat([dense, pass 2, pass 3], dim 1) that holds each unit's last pass
    readout_inverse: (B, C + sum K) long: the unit that reads each of those rows, C where none
    counts:    (B, C) long: the passes of each unit (0: not a unit)
    """

    capacity: int
    index: list = field(default_factory=list)
    inverse: list = field(default_factory=list)
    slot: list = field(default_factory=list)
    unit_slot: list = field(default_factory=list)
    valid: list = field(default_factory=list)
    same: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    readout: torch.Tensor | None = None
    readout_inverse: torch.Tensor | None = None
    counts: torch.Tensor | None = None

    @property
    def n_later(self) -> int:
        """The passes beyond the first that run in this forward."""
        return len(self.index)

    @property
    def sizes(self) -> list[int]:
        """K_j of every later pass that runs."""
        return [int(i.shape[1]) for i in self.index]

    def to(self, device, non_blocking: bool = False) -> "PassPlan":
        def mv(t):
            return None if t is None else t.to(device, non_blocking=non_blocking)

        return PassPlan(capacity=self.capacity, index=[mv(t) for t in self.index], inverse=[mv(t) for t in self.inverse],
                        slot=[mv(t) for t in self.slot], unit_slot=[mv(t) for t in self.unit_slot],
                        valid=[mv(t) for t in self.valid], same=list(self.same),
                        rows=[mv(t) for t in self.rows], readout=mv(self.readout),
                        readout_inverse=mv(self.readout_inverse), counts=mv(self.counts))

    def stats(self) -> dict:
        """Host-side counts for logging (call on the CPU plan): units, the sum of their pass counts, and per later
        pass [slots filled, slots allocated]."""
        units = int((self.counts > 0).sum())
        return {"units": units, "passes": int(self.counts.sum()),
                "slots": {str(j + 2): [int(v.sum()), int(v.numel())] for j, v in enumerate(self.valid)}}


def build_plan(counts: torch.Tensor, sizes: list[int], row_gather: bool = False) -> PassPlan:
    """The plan of one forward from the pass count of every unit. Host side: CPU tensors, called by the data path.

    counts (B, C) integer: 0 where the unit does not exist (an empty decoder slot), else its passes (1, 2 or 3).
    sizes: the allowed slot sizes, ascending, the last one >= C (bucket_sizes). Each later pass takes the smallest
    size that holds every row's units of the pass; a pass with no unit in any row is left out, and so is every pass
    after it (a unit of pass p + 1 is a unit of pass p).
    row_gather (cached generation): also record, per pass, the rows that have a slot when some have none."""
    counts = counts.detach().to("cpu", torch.long)
    B, C = counts.shape
    assert sizes and sizes[-1] >= C and list(sizes) == sorted(sizes), (sizes, C)
    units = torch.arange(C).expand(B, C)
    plan = PassPlan(capacity=C, counts=counts)
    readout = units.clone()
    prev_member, prev_inv, prev_K, offset = counts >= 1, None, C, C
    p = 2
    while True:
        member = counts >= p
        n = int(member.sum(1).max()) if B else 0
        if n == 0:
            break
        K = next(s for s in sizes if s >= n)
        rank = member.long().cumsum(1) - 1
        dest = torch.where(member, rank, torch.full_like(rank, K))  # column K is a trash column
        slot = torch.full((B, K + 1), C, dtype=torch.long).scatter(1, dest, units)[:, :K]  # ascending units, then C
        inv_units = slot_inverse(slot, C)  # (B, C): this pass's slot of each unit, K where none
        if prev_inv is None:  # pass 2 continues the dense layout
            index, inverse = slot, inv_units
        else:  # a later pass continues the previous pass's slots, of which its own are a subset
            index = torch.cat([prev_inv, torch.full((B, 1), prev_K, dtype=torch.long)], dim=1).gather(1, slot)
            inverse = slot_inverse(index, prev_K)
        plan.index.append(index)
        plan.inverse.append(inverse)
        plan.slot.append(slot)
        plan.unit_slot.append(inv_units)
        plan.valid.append(slot < C)
        plan.same.append(K == prev_K and bool((member == prev_member).all()))
        has = member.any(dim=1)
        plan.rows.append(has.nonzero().squeeze(1) if row_gather and not bool(has.all()) else None)
        readout = torch.where(member, offset + inv_units, readout)
        prev_member, prev_inv, prev_K, offset = member, inv_units, K, offset + K
        p += 1
    plan.readout = readout
    plan.readout_inverse = slot_inverse(readout, offset)
    return plan


def unit_counts(pass_counts: torch.Tensor, index: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(B, C) long: the pass count of each dense unit from per-position counts (B, L): the count at the unit's
    position (index, capacity.select_slots: L at an empty slot), 0 at an empty slot."""
    c = gather_slots(pass_counts.to(torch.long), index, fill=0)
    return torch.where(valid, c, torch.zeros_like(c))


def take(x: torch.Tensor, index: torch.Tensor, inverse: torch.Tensor, fill=0) -> torch.Tensor:
    """x (B, N, ...) at the slots `index` (B, K) (N: the fill row), with the exact inverse map `inverse` (B, N) so
    that the backward is a gather (capacity.gather_rows). A pass's input from the previous layout:
    take(Y_prev, plan.index[j], plan.inverse[j])."""
    return gather_rows(x, index, inverse, fill)


def take_meta(x: torch.Tensor, slot: torch.Tensor, fill) -> torch.Tensor:
    """Integer or bool metadata of the dense layout x (B, C) at a pass's slots (plan.slot[j]); `fill` at padding."""
    return gather_slots(x, slot, fill=fill)


def mask_rows(allow: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    """Rows `slot` (B, K) of a dense attention mask (B, N, S) of the previous layout: a later pass's mask over the
    same keys, a gather and no rebuild. A padding slot (N) gets the empty-slot row, key 0 only, which is what
    attention.slots_allowed gives a slot of document -1."""
    B, _, S = allow.shape
    first = torch.zeros(B, 1, S, dtype=torch.bool, device=allow.device)
    first[:, :, 0] = True
    return torch.cat([allow, first], dim=1).gather(1, slot[:, :, None].expand(-1, -1, S))


def readout(states: list, plan: PassPlan) -> torch.Tensor:
    """states = [z_1 (B, C, ...), z_2 (B, K_0, ...), ...], one per pass that ran: every unit's state after the LAST
    pass it ran, (B, C, ...). One gather with an exact inverse (each row of the concatenation has at most one
    reader)."""
    assert len(states) == plan.n_later + 1, (len(states), plan.n_later)
    if len(states) == 1:
        return states[0]
    return gather_rows(torch.cat(states, dim=1), plan.readout, plan.readout_inverse)


def readout_upto(states: list, plan: PassPlan) -> torch.Tensor:
    """readout over the first len(states) passes only: every unit's state after min(its count, len(states)) passes
    (the readout "after loop k" of a model's draft outputs). No inverse map: evaluation only."""
    out = states[0]
    C = plan.capacity
    for j, z in enumerate(states[1:]):
        inv = slot_inverse(plan.slot[j], C)  # (B, C): the pass slot of each unit, K where none
        K = z.shape[1]
        picked = gather_slots(z, inv.clamp(max=K), fill=0)
        out = torch.where((inv < K).view(*inv.shape, *([1] * (z.dim() - 2))), picked, out)
    return out


def select_rows(x, rows: torch.Tensor):
    """Rows `rows` of a tensor, or of every tensor field of a dataclass-like record (a KeySet, a slot context, key
    metadata); None stays None. For a pass that only some rows of a generation batch run."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.index_select(0, rows)
    import dataclasses

    return dataclasses.replace(x, **{f.name: select_rows(getattr(x, f.name), rows) for f in dataclasses.fields(x)
                                     if torch.is_tensor(getattr(x, f.name))})


def restore_rows(x: torch.Tensor, rows: torch.Tensor, batch: int) -> torch.Tensor:
    """The inverse of select_rows for a pass's output: x (B_j, ...) back at its rows of a (B, ...) tensor, zeros
    elsewhere (rows without a slot in the pass: every slot of theirs is padding and nothing reads it)."""
    out = x.new_zeros(batch, *x.shape[1:])
    return out.index_copy(0, rows, x)
