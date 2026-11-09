"""A fast attention path with exactly the semantics of the reference `attend` in attention.py.

The reference builds a dense (B, Lq, S) boolean mask with `allowed(...)` and materialises full
(B, H, Lq, S) score tensors. That is right but slow and memory-bound at 8k-16k rows. This module
computes the same function from the KeySet metadata instead of a materialised mask:

    allow[b, q, s] = keys.ok[b, s] & (keys.idx[b, s] <= q_idx[b, q]) & (keys.doc[b, s] == q_doc[b, q])
                     | (s == q + self_block_offset)            # only when self_block_offset is given

The second line is the decoder's self-mask (decoder.py): the queries' own keys `own_now` are the
last block of the key set, starting at column self_block_offset, and every slot may attend to
itself. For a valid slot that key is already allowed; for an empty slot (doc = -1, ok = False) it
is the only allowed key, which keeps its softmax row finite. In the encoder the self term would
be redundant, so callers pass self_block_offset=None there.

Slot reads over keys the slots did not write (attention.slots_allowed: the v2 decoder over the
memory, the second pass over pass-1 keys) have no own keys; with empty_reads_first an empty slot
(q_doc < 0) instead reads key 0:

    allow[b, q, s] |= (q_doc[b, q] < 0) & (s == 0)

Two backends:

- "flex": torch.nn.attention.flex_attention with a mask_mod that closes over the idx/doc/ok
  tensors, and a BlockMask from create_block_mask. Under torch.compile (always on CUDA here) it
  is a fused block-sparse kernel that skips fully masked 128 x 128 tiles, which is where documents
  and causality pay off. Without compile, PyTorch runs a dense math fallback that ignores the
  block structure and evaluates mask_mod everywhere: correct, but 3-4x slower than the reference
  on this CPU. Use it only for tests.
- "sdpa": F.scaled_dot_product_attention with the dense boolean mask. Simpler, no block
  sparsity, a fallback if flex misbehaves.

What PyTorch 2.14.1 supports on this CPU (checked on the control VM, 2026-10-03):
- flex forward, eager: float32, float64, bfloat16. Compiled: float32, float16, bfloat16 only
  (float64 raises a lowering error).
- flex backward on CPU: refused by the public front end ("does not support backward on CPU").
  The higher-order op does have a dense math backward; the tests reach it by bypassing that one
  check, which verifies our mask and GQA plumbing for gradients. Compiled CPU backward fails in
  Inductor. The real (Triton) backward exists only on GPU and must be checked there; the tests
  run on CUDA too when a GPU is present.
- sdpa: forward and backward in float32 and float64.

GQA: key/value head j serves query heads j * group ... (j + 1) * group - 1, the reference's
repeat_interleave convention. flex's enable_gqa and sdpa's enable_gqa use the same mapping
(query head h reads kv head h // group); the tests check it with distinct per-head values.

Precision: q, k, v keep their dtype. In float32 and float64 that matches the reference up to
summation order. In bfloat16 the reference upcasts q, k, v to float32 (exact) and computes P @ V
in float32, while fused kernels round P to bfloat16 before P @ V, so the two differ at bfloat16
rounding level; logit matching for bfloat16 runs needs a bfloat16 tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (AuxRequest, BlockMask, _dense_to_ordered, _ordered_to_dense, create_block_mask,
                                               flex_attention)

from ced.config import FLEX_TILE_MODES
from ced.model.attention import NOT_CAPPED, KeySet, allowed, far_keys, slots_allowed
from ced.model.norms import compute_dtype

_compiled_flex = None


def _flex_kernel(compiled: bool):
    """flex_attention, wrapped in torch.compile once if asked and not already inside a compile."""
    global _compiled_flex
    if not compiled or torch.compiler.is_compiling():
        return flex_attention
    if _compiled_flex is None:
        _compiled_flex = torch.compile(flex_attention, dynamic=False)
    return _compiled_flex


# AttentionConfig.flex_tiles: "classified" is flex's own BlockMask (full tiles are skipped by the mask and visited AFTER the partial
# ones); "indexed" is the same mask with the full tiles folded into the partial pass, every tile visited in ascending key-tile order.
TILE_MODES = FLEX_TILE_MODES


def check_tiles(tiles: str) -> str:
    assert tiles in TILE_MODES, f"flex_tiles {tiles!r}: one of {TILE_MODES}"
    return tiles


def fold_full_blocks(bm: BlockMask) -> BlockMask:
    """The BlockMask bm with its FULL tiles moved into the partial list: one list per query block, every tile that holds an allowed
    pair, in ascending key-tile order (the stable order of flex's own _dense_to_ordered), and no full list. The set of allowed pairs is
    unchanged (the mask_mod is the same and is now evaluated on every visited tile; where a tile was full it is all True).

    Why: flex's forward kernel visits a query block's partial tiles first and its full tiles after, with an online softmax whose result
    depends on the visiting order at rounding level. Whether a tile is full is a property of ALL 128 rows of the query block, including
    the rows after a given row, so the document boundaries after t can change the bits of the rows up to t. With the fold, a row visits
    its tiles in ascending order whatever the other rows of its block do. A tile that holds a pair for another row of the block only
    (nothing for this row) is visited with every score -inf: the running maximum, the sum and the accumulator are unchanged to the bit
    (alpha = 1, p = 0), so extra or missing empty tiles change no row's result. The backward's transposed lists (q_indices) are rebuilt
    from the folded lists (BlockMask.from_kv_blocks), so the backward is folded too. Fixed shapes (a scatter, a stable argsort and sums:
    what create_block_mask itself does), no nonzero."""
    if bm.full_kv_num_blocks is None:
        return bm  # no full tiles: the partial list is already in ascending order
    num_cols = (bm.seq_lengths[1] + bm.BLOCK_SIZE[1] - 1) // bm.BLOCK_SIZE[1]
    dense = (_ordered_to_dense(bm.kv_num_blocks, bm.kv_indices, num_cols)
             | _ordered_to_dense(bm.full_kv_num_blocks, bm.full_kv_indices, num_cols))
    num, idx = _dense_to_ordered(dense)
    return BlockMask.from_kv_blocks(num, idx, None, None, BLOCK_SIZE=bm.BLOCK_SIZE, mask_mod=bm.mask_mod, seq_lengths=bm.seq_lengths)


def with_tiles(bm: BlockMask, tiles: str) -> BlockMask:
    """bm as the tile mode asks: unchanged for "classified" (the default: the object create_block_mask built), folded for "indexed"."""
    return bm if check_tiles(tiles) == "classified" else fold_full_blocks(bm)


def mask_mod_for(
    keys: KeySet, q_idx: torch.Tensor, q_doc: torch.Tensor, self_block_offset: int | None = None,
    empty_reads_first: bool = False,
):
    """The flex mask_mod: the reference `allowed` mask (plus the self-mask, or the empty-slot rule
    of attention.slots_allowed) for one (b, q, s).

    It closes over the (B, S) and (B, Lq) metadata tensors. flex evaluates it on scalar index
    tensors (vmapped in eager, inlined into the kernel when compiled). In kernels, indices past a
    length that is not a multiple of the block size are wrapped modulo the length before they
    reach mask_mod, and those entries are masked out afterwards, so the tensor reads stay in bounds.
    """
    k_ok, k_idx, k_doc = keys.ok, keys.idx, keys.doc

    def mask_mod(b, h, q, s):
        allow = k_ok[b, s] & (k_idx[b, s] <= q_idx[b, q]) & (k_doc[b, s] == q_doc[b, q])
        if self_block_offset is not None:
            allow = allow | (s == q + self_block_offset)
        if empty_reads_first:
            allow = allow | ((q_doc[b, q] < 0) & (s == 0))
        return allow

    return mask_mod


def make_block_mask(
    keys: KeySet,
    q_idx: torch.Tensor,
    q_doc: torch.Tensor,
    self_block_offset: int | None = None,
    block_size: int = 128,
    empty_reads_first: bool = False,
    tiles: str = "classified",
) -> BlockMask:
    """BlockMask for flex. One mask serves every head (H=None). tiles: AttentionConfig.flex_tiles ("classified": flex's own mask, as
    always; "indexed": fold_full_blocks of it).

    The mask depends only on metadata, so it can be built once per forward and shared: all
    encoder global layers of a chunk share one, and all T x n_layers decoder attention calls share
    another (memory, own_past and own_now metadata are the same in every loop and layer).

    create_block_mask evaluates mask_mod on the whole (B, Lq, S) grid and reduces it to tiles. In
    eager mode that materialises a (B, Lq, S) bool tensor; torch.compile(create_block_mask) avoids
    it on GPU.
    """
    B, Lq = q_idx.shape
    bm = create_block_mask(
        mask_mod_for(keys, q_idx, q_doc, self_block_offset, empty_reads_first),
        B=B,
        H=None,
        Q_LEN=Lq,
        KV_LEN=len(keys),
        device=q_idx.device,
        BLOCK_SIZE=block_size,
    )
    return bm if tiles == "classified" else with_tiles(bm, tiles)


_compiled_create_block_mask = None


def block_mask_builder(compiled: bool):
    """create_block_mask, wrapped in torch.compile once if asked and not already inside a compile. Compiled it
    never materialises the (B, Lq, S) mask, which eager create_block_mask evaluates on the whole grid. Dynamo
    guards a mask_mod argument on its code object, so a fresh closure per forward (over new metadata tensors)
    reuses the compiled graph."""
    global _compiled_create_block_mask
    if not compiled or torch.compiler.is_compiling():
        return create_block_mask
    if _compiled_create_block_mask is None:
        _compiled_create_block_mask = torch.compile(create_block_mask, dynamic=False)
    return _compiled_create_block_mask


def slot_block_mask(keys, q_idx: torch.Tensor, q_doc: torch.Tensor, block_size: int = 128,
                    compiled: bool | None = None) -> BlockMask:
    """The BlockMask of a slot read (attend_fast with empty_reads_first: attention.slots_allowed) of slot queries at
    q_idx / q_doc (B, Lq) over keys: a KeySet, or any key metadata with ok, idx, doc (B, S) and a length
    (ced_v2.JoinedKeyMeta). The mask_mod is mask_mod_for's over those tensors, so the BlockMask is the one
    attend_fast builds for the same read (the same block tensors, the same mask_mod): built once per forward, it
    serves every decoder layer that makes the read (decoder optimisation r1, ced_v2.CEDv2.decode_fast).
    compiled: build with the compiled create_block_mask (default: CUDA yes, CPU no; ignored inside a compile)."""
    B, Lq = q_idx.shape
    compiled = q_idx.is_cuda if compiled is None else compiled
    return block_mask_builder(compiled)(
        mask_mod_for(keys, q_idx, q_doc, None, True), B=B, H=None, Q_LEN=Lq, KV_LEN=len(keys), device=q_idx.device,
        BLOCK_SIZE=block_size,
    )


def dense_mask(
    keys: KeySet, q_idx: torch.Tensor, q_doc: torch.Tensor, self_block_offset: int | None = None,
    empty_reads_first: bool = False,
) -> torch.Tensor:
    """(B, Lq, S) bool: the reference `allowed` mask, with the self-mask as in decoder.py, or the
    empty-slot rule of attention.slots_allowed."""
    if empty_reads_first:
        assert self_block_offset is None, "the empty-slot rule is for slot reads, which have no own keys"
        return slots_allowed(keys, q_idx, q_doc)
    allow = allowed(keys, q_idx, q_doc)
    if self_block_offset is not None:
        Lq, S = allow.shape[1], allow.shape[2]
        rows = torch.arange(Lq, device=allow.device)
        cols = torch.arange(S, device=allow.device)
        allow = allow | (cols[None, :] == rows[:, None] + self_block_offset)[None]
    return allow


def attend_fast(
    q: torch.Tensor,
    keys: KeySet,
    q_idx: torch.Tensor,
    q_doc: torch.Tensor,
    self_block_offset: int | None = None,
    backend: str = "flex",
    block_mask: BlockMask | None = None,
    compiled: bool | None = None,
    empty_reads_first: bool = False,
    dense: torch.Tensor | None = None,
    return_lse: bool = False,
    tiles: str = "classified",
    row_term: str = "stored_o",
):
    """Same result as attend(q, keys, allowed(keys, q_idx, q_doc) [| self-mask]), or with
    empty_reads_first, attend(q, keys, slots_allowed(keys, q_idx, q_doc)).
    dense (sdpa only): that mask, (B, Lq, S) bool, built by the caller for the same metadata and shared by every
    layer that makes the read (the segment pass schedule); None builds it here, as before.
    return_lse (the trunk copy's fused path, ced/kernels/cp_trunk_fused.py): (output, the read's log-sum-exp (B, Hq, Lq)
    float32 detached) on flex (the same kernel call with the log-sum-exp requested), (output, None) on sdpa.
    tiles (flex, when this call builds the block mask): AttentionConfig.flex_tiles; a caller-built block_mask is used as given.
    row_term (AttentionConfig.row_term): "stored_o" (default): the call as it always was. "pdp" (flex; the encoder's own read of a row): the same
    forward values, and with a gradient needed on a 16-bit query dtype (where flex's backward, the compiled kernel's and the dense math
    fallback's alike, forms its row term from the stored output) the call's (output, log-sum-exp) are tied to a pre-pass that gives the backward
    rowsum(P . dP) (ced/kernels/attn_row_term.py). At float32 / float64 and without a gradient it is inert.

    q:      (B, Lq, Hq, hd); keys.k / keys.v: (B, S, Hkv, hd)
    q_idx, q_doc: (B, Lq) absolute index and document id of each query
    self_block_offset: column where the queries' own keys start (decoder: len(memory) +
        len(own_past)); query i may then always attend to column self_block_offset + i.
    block_mask: a precomputed make_block_mask(...) for the same metadata (flex only).
    compiled: wrap flex in torch.compile. Default: on CUDA yes, on CPU no (see the module
        docstring for what compiled flex supports on CPU). Ignored inside an outer compile.
        The compiled kernel specialises on shapes (dynamic=False). A training step has two or
        three fixed shapes; cached decoding grows S every step, so it would recompile until
        dynamo's recompile limit and then silently run eager flex (dense, slower than the
        reference on CPU). Use sdpa or the reference for cached decoding.

    Every query row must allow at least one key, as in the reference; for the self-mask case it
    always does.
    """
    assert getattr(keys, "k_far", None) is None, NOT_CAPPED
    B, Lq, Hq, hd = q.shape
    Hkv = keys.k.shape[2]
    assert Hq % Hkv == 0, (Hq, Hkv)
    assert q_idx.shape == q_doc.shape == (B, Lq)
    if self_block_offset is not None:
        assert 0 <= self_block_offset <= len(keys) - Lq, (self_block_offset, len(keys), Lq)
        assert not empty_reads_first, "the empty-slot rule is for slot reads, which have no own keys"
    scale = hd**-0.5
    qt, kt, vt = q.transpose(1, 2), keys.k.transpose(1, 2), keys.v.transpose(1, 2)  # (B, H, len, hd)

    assert dense is None or backend == "sdpa", "a caller-built dense mask is for the sdpa backend"
    if backend == "flex":
        if block_mask is None:
            block_mask = make_block_mask(keys, q_idx, q_doc, self_block_offset, empty_reads_first=empty_reads_first, tiles=tiles)
        compiled = q.is_cuda if compiled is None else compiled
        if row_term != "stored_o":  # AttentionConfig.row_term "pdp" (ced/kernels/attn_row_term.py)
            from ced.kernels import attn_row_term as RT

            RT.check_row_term(row_term)
            if RT.applies(q, keys.k, keys.v):
                if self_block_offset is not None or empty_reads_first:
                    raise ValueError("attention.row_term 'pdp' is built for the encoder's own read of a row, not for a decoder read "
                                     "(a self-mask or a slot read)")
                if not torch.compiler.is_compiling():  # a data-dependent check: never part of a compiled graph
                    assert bool((keys.idx == torch.arange(len(keys), device=keys.idx.device)).all()), \
                        "attention.row_term 'pdp' reads a key set with one entry per row position (keys.idx = 0 .. S - 1)"
                out, aux = _flex_kernel(compiled)(qt, kt, vt, block_mask=block_mask, scale=scale, enable_gqa=Hq != Hkv,
                                                  return_aux=AuxRequest(lse=True))
                out, lse = RT.tie_read(out, aux.lse, qt, kt, vt, keys.doc, scale, RT.KIND_PLAIN, 0)
                if return_lse:
                    return out.transpose(1, 2), lse.detach().float()
                return out.transpose(1, 2)
        if return_lse:
            out, aux = _flex_kernel(compiled)(qt, kt, vt, block_mask=block_mask, scale=scale, enable_gqa=Hq != Hkv,
                                              return_aux=AuxRequest(lse=True))
            return out.transpose(1, 2), aux.lse.detach().float()
        out = _flex_kernel(compiled)(qt, kt, vt, block_mask=block_mask, scale=scale, enable_gqa=Hq != Hkv)
    elif backend == "sdpa":
        assert block_mask is None, "block_mask is for the flex backend"
        mask = (dense_mask(keys, q_idx, q_doc, self_block_offset, empty_reads_first) if dense is None else dense)[:, None]  # (B, 1, Lq, S)
        out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask, scale=scale, enable_gqa=Hq != Hkv)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    if return_lse:
        return out.transpose(1, 2), None
    return out.transpose(1, 2)


# ---------------------------------------------------------------------------------------------------
# the rotary distance cap (AttentionConfig.rope_max_distance; ced/model/attention.py, "Rotary distance cap")
# ---------------------------------------------------------------------------------------------------
#
# The reference (attention.attend_capped) holds two full (B, H, Lq, S) logit tensors. The paths here hold none.
#
# flex: two reads with the model's own kernel, joined exactly. The softmax over a query's allowed keys splits over
# any partition of them: with N the keys within the cap and F the keys beyond it,
#
#     out = (e^lse_N out_N + e^lse_F out_F) / (e^lse_N + e^lse_F) = out_N + sigmoid(lse_F - lse_N) (out_F - out_N)
#
# where out_X and lse_X are the softmax output and the log-sum-exp of the logits over X alone. So:
#     near read:  flex(q,     keys.k,     v)  under  allowed & distance <= C       (the plain logits)
#     far read:   flex(q_far, keys.k_far, v)  under  allowed & distance >  C       (the capped logits)
# each with its own BlockMask and both returning their log-sum-exp (float32 inside the kernel). The two BlockMasks
# depend on the metadata alone, so a caller may build them once and pass them to every layer that makes the same read
# (capped_block_masks / CappedMasks below: the encoder's global layers, ced/model/encoder.py). Head dimension,
# GQA mapping and scale are the plain read's; the kernel never holds more than its 128 x 128 tiles, and autograd
# keeps q, k, v, the output and the log-sum-exp of each read, as for the plain read. The tiles give the block
# structure for free: a tile whose pairs are all beyond the cap is one plain matmul in the far read only, a tile
# within the cap is in the near read only, and the tiles the distance-C diagonal crosses are partial in both.
# A query with no key beyond the cap (document position <= C, and every empty slot) would have an empty row in the
# far read; it reads key 0 there instead and that part is dropped from the join (weight exactly 0), so no row of
# either kernel is empty and no -inf enters the join. Which rows those are is read from the queries' document
# positions: a query at position p > C has its document's first key p > C back, and a query at p <= C has no key
# further back than p. That holds for every key set built under the cap, which has one ok entry per position of
# the row (a layer's own keys with their cache, CED v2's memory): keys.idx is 0 .. S - 1 in every row.
#
# sdpa (dense, the fallback): one call over a doubled key set. Each key appears twice, [k, 0] (its near copy) and
# [0, k_far] (its far copy), against the query [q, q_far], with the values repeated; the mask lets a query see the
# near copy of a key within the cap and the far copy of a key beyond it, so every pair is scored once, with the
# right logit, in one softmax.
#
# A key set of at most C + 1 entries has no pair beyond the cap (one entry per row position), so the capped read is
# then the plain read: attend_fast on the same q and keys, bit for bit, by a test on shapes alone.


def capped_mask_mods(keys, q_idx: torch.Tensor, q_doc: torch.Tensor, q_pos: torch.Tensor, max_distance: int,
                     empty_reads_first: bool = False):
    """(near, far): the flex mask_mods of the capped read's two parts (the comment above) for one (b, q, s), over
    the key metadata (a KeySet, or anything with ok, idx, doc (B, S)) and the queries' q_idx, q_doc, q_pos (B, Lq).

        near = allowed & (q_idx - k_idx <= max_distance)   [| the empty-slot rule: an empty slot reads key 0]
        far  = allowed & (q_idx - k_idx >  max_distance)    | (q_pos <= max_distance) & (s == 0)

    The last term gives a query with no key beyond the cap one key to read in the far part (dropped in the join).
    Only the metadata tensors are captured and every comparison is made inside, as mask_mod_for does."""
    k_ok, k_idx, k_doc = keys.ok, keys.idx, keys.doc

    def near(b, h, q, s):
        allow = k_ok[b, s] & (k_idx[b, s] <= q_idx[b, q]) & (k_doc[b, s] == q_doc[b, q])
        allow = allow & (q_idx[b, q] - k_idx[b, s] <= max_distance)
        if empty_reads_first:
            allow = allow | ((q_doc[b, q] < 0) & (s == 0))
        return allow

    def far(b, h, q, s):
        allow = k_ok[b, s] & (k_idx[b, s] <= q_idx[b, q]) & (k_doc[b, s] == q_doc[b, q])
        allow = allow & (q_idx[b, q] - k_idx[b, s] > max_distance)
        return allow | ((q_pos[b, q] <= max_distance) & (s == 0))

    return near, far


def capped_dense_masks(keys: KeySet, q_idx: torch.Tensor, q_doc: torch.Tensor, q_pos: torch.Tensor, max_distance: int,
                       empty_reads_first: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """(near, far), each (B, Lq, S) bool: capped_mask_mods' two masks from the reference mask functions."""
    allow = allowed(keys, q_idx, q_doc)
    beyond = far_keys(keys, q_idx, max_distance)
    first = torch.arange(len(keys), device=q_idx.device) == 0
    near = allow & ~beyond
    if empty_reads_first:
        near = near | ((q_doc < 0)[:, :, None] & first[None, None, :])
    far = (allow & beyond) | ((q_pos <= max_distance)[:, :, None] & first[None, None, :])
    return near, far


def has_far_keys(q_pos: torch.Tensor, max_distance: int) -> torch.Tensor:
    """(B, Lq) bool: the query has a key beyond the cap, read from its document position (the comment above)."""
    return q_pos > max_distance


@dataclass(frozen=True)
class CappedMasks:
    """The two flex BlockMasks of one capped read (capped_mask_mods), built once from its metadata and shared by every
    layer that makes the same read. The masks depend on the metadata alone: the keys' ok / idx / doc, the queries'
    idx / doc / pos, the cap and the empty-slot rule. A layer that is handed one makes the read attend_capped_fast
    would make with masks of its own, bit for bit: the same block tensors and the same mask_mod functions over
    tensors of the same values.

    near, far: the BlockMasks; has_far (B, Lq) bool: has_far_keys(q_pos, max_distance), the rows whose far part counts
    in the join; max_distance, empty_reads_first: what they were built for (checked at every use, on Python values);
    meta: (keys.ok, keys.idx, keys.doc, q_idx, q_doc, q_pos), the tensors they were built from, compared with the
    read's own whenever the read runs eagerly (never inside a compiled graph, like the keys.idx check)."""

    near: BlockMask
    far: BlockMask
    has_far: torch.Tensor
    max_distance: int
    empty_reads_first: bool
    meta: tuple
    tiles: str = "classified"

    def check(self, keys, q_idx: torch.Tensor, q_doc: torch.Tensor, q_pos: torch.Tensor, max_distance: int,
              empty_reads_first: bool, tiles: str = "classified") -> None:
        """Refuse a read these masks were not built for. Shapes and settings always; the metadata's values when the
        call runs eagerly (a data-dependent check: never part of a compiled graph)."""
        B, Lq = q_idx.shape
        S = keys.idx.shape[1]
        assert self.max_distance == max_distance and self.empty_reads_first == empty_reads_first, \
            f"shared capped masks built for cap {self.max_distance} (empty_reads_first {self.empty_reads_first}), " \
            f"read with cap {max_distance} (empty_reads_first {empty_reads_first})"
        assert self.tiles == tiles, f"shared capped masks built with flex_tiles {self.tiles!r}, read with {tiles!r}"
        for bm in (self.near, self.far):
            assert bm.kv_num_blocks.shape[0] == B and tuple(bm.seq_lengths) == (Lq, S), \
                f"shared capped masks of shape {tuple(bm.shape)}, read of {(B, Lq, S)}"
        assert self.has_far.shape == (B, Lq), (tuple(self.has_far.shape), (B, Lq))
        if not torch.compiler.is_compiling():
            for name, built, now in zip(("keys.ok", "keys.idx", "keys.doc", "q_idx", "q_doc", "q_pos"), self.meta,
                                        (keys.ok, keys.idx, keys.doc, q_idx, q_doc, q_pos)):
                assert built is now or (built.shape == now.shape and bool((built == now).all())), \
                    f"shared capped masks were built for another {name} than this read's (a mask of another batch?)"


def capped_block_masks(keys, q_idx: torch.Tensor, q_doc: torch.Tensor, q_pos: torch.Tensor, max_distance: int,
                       empty_reads_first: bool = False, compiled: bool | None = None, block_size: int = 128,
                       tiles: str = "classified") -> CappedMasks:
    """The CappedMasks of the capped read of queries at q_idx / q_doc / q_pos (B, Lq) over `keys` (a KeySet, or any key
    metadata with ok, idx, doc (B, S)): the masks attend_capped_fast builds for that read (capped_mask_mods' functions
    over these tensors, the same create_block_mask arguments), for a caller that makes the read in several layers.
    compiled: build with the compiled create_block_mask (block_mask_builder; default: CUDA yes, CPU no; ignored
    inside a compile). keys must hold one entry per row position (keys.idx = 0 .. S - 1; attend_capped_fast's rule),
    checked here whenever the call runs eagerly and the build is not compiled (a host sync: not in a training step)."""
    B, Lq = q_idx.shape
    S = keys.idx.shape[1]
    assert q_idx.shape == q_doc.shape == q_pos.shape and keys.ok.shape == keys.idx.shape == keys.doc.shape == (B, S)
    assert S > max_distance + 1, "a key set of at most cap + 1 entries has no pair beyond the cap: the plain read, no masks"
    compiled = q_idx.is_cuda if compiled is None else compiled
    if not compiled and not torch.compiler.is_compiling():
        assert bool((keys.idx == torch.arange(S, device=keys.idx.device)).all()), \
            "the rotary distance cap's fast path reads a key set with one entry per row position (keys.idx = 0 .. S - 1)"
    near, far = capped_mask_mods(keys, q_idx, q_doc, q_pos, max_distance, empty_reads_first)
    size = dict(B=B, H=None, Q_LEN=Lq, KV_LEN=S, device=q_idx.device, BLOCK_SIZE=block_size)
    build = block_mask_builder(compiled)
    return CappedMasks(near=with_tiles(build(near, **size), tiles), far=with_tiles(build(far, **size), tiles),
                       has_far=has_far_keys(q_pos, max_distance), max_distance=max_distance, empty_reads_first=empty_reads_first,
                       meta=(keys.ok, keys.idx, keys.doc, q_idx, q_doc, q_pos), tiles=tiles)


def attend_capped_fast(
    q: torch.Tensor,
    q_far: torch.Tensor,
    keys: KeySet,
    q_idx: torch.Tensor,
    q_doc: torch.Tensor,
    q_pos: torch.Tensor,
    max_distance: int,
    backend: str = "flex",
    compiled: bool | None = None,
    empty_reads_first: bool = False,
    block_masks: CappedMasks | None = None,
    return_lse: bool = False,
    tiles: str = "classified",
    row_term: str = "stored_o",
):
    """Same result as attention.attend_capped(q, q_far, keys, allowed(keys, q_idx, q_doc), far_keys(keys, q_idx,
    max_distance)), or with empty_reads_first with slots_allowed, without a (Lq, S) tensor (flex).
    return_lse (the trunk copy's fused path): (output, the read's log-sum-exp (B, Hq, Lq) float32 detached: on flex the near read's,
    joined with the far read's by logaddexp where the row has keys beyond the cap (the join's own rule), or None on sdpa).

    q, q_far: (B, Lq, Hq, hd); keys.k, keys.k_far, keys.v: (B, S, Hkv, hd); q_idx, q_doc, q_pos: (B, Lq), the queries' absolute row index,
    document id and document position. keys must hold one ok entry per position of the row, keys.idx = 0 .. S - 1 (the comment above); checked
    whenever the call runs eagerly. compiled (flex): as attend_fast. block_masks (flex only): the two parts' BlockMasks built by the caller for
    this read's metadata (capped_block_masks), shared by every layer that makes the same read; None: the two parts build their own here, as
    every run before 2026-10-11 did. Either way the same block tensors reach the same two kernel calls: the result is equal bit for bit
    (tests/test_rc_tune.py). row_term (AttentionConfig.row_term, flex): "stored_o" (default): the two calls as they always were. "pdp": each
    call's (output, log-sum-exp) is tied to the pre-pass of ced/kernels/attn_row_term.py (the near call KIND_NEAR, the far call KIND_FAR; the
    join below then reads the tied log-sum-exps, the same values), where flex's backward forms its row term from the stored output (a 16-bit
    query dtype, a gradient needed: attn_row_term.applies); inert elsewhere."""
    assert keys.k_far is not None, "attend_capped_fast needs the unrotated keys (KeySet.k_far)"
    B, Lq, Hq, hd = q.shape
    S, Hkv = len(keys), keys.k.shape[2]
    assert Hq % Hkv == 0, (Hq, Hkv)
    assert q.shape == q_far.shape and keys.k.shape == keys.k_far.shape
    assert q_idx.shape == q_doc.shape == q_pos.shape == (B, Lq)
    if block_masks is not None:  # built by the caller for this read: settings and shapes always, values when eager
        assert backend == "flex", "block_masks are for the flex backend"
        block_masks.check(keys, q_idx, q_doc, q_pos, max_distance, empty_reads_first, tiles)
    elif not torch.compiler.is_compiling():  # a data-dependent check: never part of a compiled graph
        assert bool((keys.idx == torch.arange(S, device=keys.idx.device)).all()), \
            "the rotary distance cap's fast path reads a key set with one entry per row position (keys.idx = 0 .. S - 1)"
    if S <= max_distance + 1:  # no pair can be beyond the cap: the plain read, bit for bit
        assert block_masks is None, "a key set of at most cap + 1 entries is read by the plain path: no capped masks"
        rt = {} if row_term == "stored_o" else {"row_term": row_term}
        if return_lse:
            return attend_fast(q, keys.without_far(), q_idx, q_doc, backend=backend, compiled=compiled,
                               empty_reads_first=empty_reads_first, return_lse=True, **rt)
        return attend_fast(q, keys.without_far(), q_idx, q_doc, backend=backend, compiled=compiled,
                           empty_reads_first=empty_reads_first, **rt)
    scale = hd**-0.5
    gqa = Hq != Hkv
    vt = keys.v.transpose(1, 2)
    if backend == "flex":
        kernel = _flex_kernel(q.is_cuda if compiled is None else compiled)
        lse = AuxRequest(lse=True)
        tie = False  # AttentionConfig.row_term "pdp" on the compiled kernel (ced/kernels/attn_row_term.py)
        if row_term != "stored_o":
            from ced.kernels import attn_row_term as RT

            RT.check_row_term(row_term)
            tie = RT.applies(q, keys.k, keys.v)
            if tie and empty_reads_first:
                raise ValueError("attention.row_term 'pdp' is built for the encoder's own read of a row, not for a slot read")
        if block_masks is None:  # this read's own masks, built here (two builds per call)
            near, far = capped_mask_mods(keys, q_idx, q_doc, q_pos, max_distance, empty_reads_first)
            size = dict(B=B, H=None, Q_LEN=Lq, KV_LEN=S, device=q_idx.device, BLOCK_SIZE=128)
            o_near, a_near = kernel(q.transpose(1, 2), keys.k.transpose(1, 2), vt,
                                    block_mask=with_tiles(create_block_mask(near, **size), tiles),
                                    scale=scale, enable_gqa=gqa, return_aux=lse)
            o_far, a_far = kernel(q_far.transpose(1, 2), keys.k_far.transpose(1, 2), vt,
                                  block_mask=with_tiles(create_block_mask(far, **size), tiles),
                                  scale=scale, enable_gqa=gqa, return_aux=lse)
            has_far = has_far_keys(q_pos, max_distance)
        else:  # the caller's masks, shared by every layer making this read: nothing is built or sorted here
            o_near, a_near = kernel(q.transpose(1, 2), keys.k.transpose(1, 2), vt, block_mask=block_masks.near,
                                    scale=scale, enable_gqa=gqa, return_aux=lse)
            o_far, a_far = kernel(q_far.transpose(1, 2), keys.k_far.transpose(1, 2), vt, block_mask=block_masks.far,
                                  scale=scale, enable_gqa=gqa, return_aux=lse)
            has_far = block_masks.has_far
        lse_near, lse_far = a_near.lse, a_far.lse
        if tie:  # each call's (output, log-sum-exp) through the identity node whose backward gives the kernel rowsum(P . dP)
            o_near, lse_near = RT.tie_read(o_near, lse_near, q.transpose(1, 2), keys.k.transpose(1, 2), vt, keys.doc, scale,
                                           RT.KIND_NEAR, max_distance)
            o_far, lse_far = RT.tie_read(o_far, lse_far, q_far.transpose(1, 2), keys.k_far.transpose(1, 2), vt, keys.doc, scale,
                                         RT.KIND_FAR, max_distance)
        # the join, in at least float32: out_N + w (out_F - out_N), w = sigmoid(lse_F - lse_N), 0 without far keys
        dt = compute_dtype(q)
        w = torch.sigmoid(lse_far.to(dt) - lse_near.to(dt))  # (B, Hq, Lq)
        w = torch.where(has_far[:, None, :], w, torch.zeros((), dtype=dt, device=q.device))
        near_out = o_near.to(dt)
        out = (near_out + w[..., None] * (o_far.to(dt) - near_out)).to(q.dtype)
        if return_lse:  # the whole read's log-sum-exp: logaddexp of the two parts where the far part counts (the join's rule)
            ln, lf = lse_near.detach().float(), lse_far.detach().float()
            return out.transpose(1, 2), torch.where(has_far[:, None, :], torch.logaddexp(ln, lf), ln)
    elif backend == "sdpa":
        near, far = capped_dense_masks(keys, q_idx, q_doc, q_pos, max_distance, empty_reads_first)
        far = far & has_far_keys(q_pos, max_distance)[:, :, None]  # one softmax: no far-part stand-in key
        zero = torch.zeros_like(keys.k)
        q2 = torch.cat([q, q_far], dim=-1)  # (B, Lq, Hq, 2 hd)
        k2 = torch.cat([torch.cat([keys.k, zero], dim=-1), torch.cat([zero, keys.k_far], dim=-1)], dim=1)  # (B, 2 S, Hkv, 2 hd)
        v2 = torch.cat([keys.v, keys.v], dim=1)
        mask = torch.cat([near, far], dim=-1)[:, None]  # (B, 1, Lq, 2 S)
        out = F.scaled_dot_product_attention(q2.transpose(1, 2), k2.transpose(1, 2), v2.transpose(1, 2), attn_mask=mask,
                                             scale=scale, enable_gqa=gqa)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    if return_lse:
        return out.transpose(1, 2), None
    return out.transpose(1, 2)
