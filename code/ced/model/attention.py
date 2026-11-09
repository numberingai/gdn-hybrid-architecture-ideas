"""Gated grouped-query softmax attention with QK-norm and partial RoPE.

The encoder's global layers and the decoder layers use the same module. What differs is the set
of keys a query may look at, which is passed in as a `KeySet`:

- encoder global layer: the keys of the same layer at earlier positions of the same document;
- decoder layer: memory keys of context positions plus own keys of decoder positions
  (see ced/model/decoder.py);
- slot reads (`attend_slots`): queries at gathered slots over a key set they did not write, the
  v2 decoder's memory (ced/model/ced_v2.py) or a layer's pass-1 keys (ced/model/second_pass.py).

Temperature (AttentionConfig.temperature; the length-failure test of 2026-10-09): the logits are
q.k * head_dim^-0.5 * tau. tau multiplies the query after QK-norm and RoPE (GatedAttention.queries), so
the reference, flex and sdpa paths and cached decoding all get it from the same place, unchanged:

    fixed:    tau = 1 (no parameter)
    learned:  tau_h = 1 + w_h                                   w: temp_w (n_q_heads,), init 0
    ssmax:    tau_{t,h} = (1 + w_h) ln(n_t) / ln(ref)           n_t = pos_t + 1 (or the counted n_keys),
                                                                ref = temperature_ref_len

ssmax (Scalable-Softmax, arXiv 2501.19399) takes n_t from the query's document position: the keys a query
sees are its document's positions s <= t, one each, so n_t = pos_t + 1 wherever that holds. It holds for the
encoder's global layers (and their second-pass reads), CED v2's decoder memory read, and v1's decoder (a
context position offers its memory entry, a decoder position its own key). Where it does not, the caller
passes the exact count as n_keys (B, L) (queries / read; key_count_factor): CED v2's full decoder layers with
self-KV, which see the memory entries of s <= t AND the own keys of the valid decoder slots of t's document
at s <= t (ced/model/ced_v2.py slot_key_counts; stage 2 + Scalable-Softmax). Such a module has
needs_key_count set (a plain attribute), and a query without n_keys is refused there. The diagnostics' key
windows (ced/eval/attn_diag.py) and CED v2's local decoder layers (a bounded read whose count is not
computed) refuse ssmax.

Normaliser (AttentionConfig.normalizer; the three-variant attention experiment): "softmax"

    c_{t,h} = 1 + softplus(x_t . w_b,h + b_b,h) * ln(pos_t + 2) ** (2 tanh(x_t . w_g,h + b_g,h))

with x_t the attention input at the query (the output of the pre-attention norm: EncoderLayer.mixer_in_norm,
CED v2's MemoryDecoderLayer.attn_in_norm), pos_t the query's document position (ln(pos + 2) = ln(n_t + 1), n_t
its visible keys, as the authors' code), in float32 (autocast off), multiplied into q where the temperature is
applied (after QK-norm and RoPE; 1/sqrt(head_dim) kept). beta_proj and gamma_proj are Linear(d_model, n_q_heads)
with bias, weights N(0, 0.02) and bias 0 (the authors' implicit init), on AdamW without weight decay
(ced/train/optim.py). The attention itself is the exact reference of ced/model/entmax.py on every path
(GatedAttention.forward with or without a cache, read / slot reads with the empty-slot rule); the configured
backend (flex / sdpa) is not used for it. Refused with a non-fixed temperature (one scale on q at a time), with
CED v2's decoder self-KV and in v1's decoder (ced/model/decoder.py builds its own mask), and by the decode engine.

Rotary distance cap (AttentionConfig.rope_max_distance = C;, and the scoring scripts' --rope-clamp). The rotary part
of q. k depends only on the relative distance d = pos_q - pos_k (rotations are orthogonal: R(a) q. R(b) k = R(a - b)
q. k). With the cap it uses min(d, C):

    d <= C:  logit = R(pos_q) q . R(pos_k) k     the plain logit: both rotated at their own document positions
    d >  C:  logit = R(C) q . k                  the query rotated by the constant C against the UNROTATED key,
                                                 which is the plain logit of a key exactly C positions back

times head_dim^-0.5, q and k after QK-norm; the dimensions that do not rotate are the same in both lines. One softmax
per query over the same keys as without the cap (the mask is unchanged), the model's temperature on both rotations of q
(tau multiplies q; ssmax's key count stays pos + 1), the same values, gate and projection.
- Where the unrotated keys come from: they are kept. A key set built under the cap carries `k_far`, the keys as the
  QK-norm gives them, next to `k`, the same keys after rotation (KeySet.k_far). That is the tensor the rotation reads,
  so it costs no computation and is exact; rotating the stored keys back would need every key's document position and
  add a rounding (in bf16 a second one); a far logit built from the rotated key would need a per-pair rotation by
  pos_q - C - pos_k, which no plain matmul gives. The cached key sets later chunks read (GatedAttention.forward with a
  cache) and the memory CED v2's decoder reads (decoder.MemoryKV) carry it the same way; KeySet.cat joins it.
- The distance is read from the absolute row indices: d = q_idx - keys.idx. For a key the mask allows (the query's
  document, at or before it) that is pos_q - pos_k, because a document's positions are consecutive in the row; any
  other key is masked, so its distance is never read.
- `attend_capped` below is the reference: both logit tensors in full and a `where`. The flex and sdpa backends run
  fast_attention.attend_capped_fast. On flex the read's two BlockMasks depend on the positions alone, so the encoder
  builds them once per forward for all its global layers (Encoder.share_capped_masks) instead of twice in every
  layer: the same block tensors, the same result. Backend "fa3" (ced/kernels/flash_capped.py) runs the same read as
  two FlashAttention-3 varlen calls merged by their log-sum-exps (one call without the cap), with varlen metadata
  the encoder builds once per forward (Encoder.share_flash_varlen); built for the control's encoder global layers
  without a cache, refused elsewhere by name (control.check_fa3, GatedAttention._fa3, GatedAttention.read). Backend
  "fa4"
  (ced/kernels/flash_fa4.py) runs the same two calls, merge and metadata on FlashAttention-4, covered and refused alike.
- The rotation tables of an encoder global layer's own read (cos and sin at the positions, and at the one position
  C) are tensors: built once per forward by the encoder for all its global layers (CappedRotation,
  Encoder.share_capped_rotation) and applied by rope.rotate_with_tables. Computed inside a compiled layer they cost
  three quarters of the cap's step time (CappedRotation's docstring). A read without them (a slot read, a layer
  called on its own) rotates with rope.apply_rope as before. A key set no longer than C + 1 holds no pair beyond the cap, and the fast path is
  then the plain read bit for bit (the reference is too: `where` picks every plain logit).
- Without rotary dimensions (rope_fraction 0.0) the cap has nothing to act on and the layer runs the plain code.
- Refused: the asentmax normaliser; CED v2's decoder with self-KV (stage 1's own keys, stage 2's local layers) and
  its fast decoder path (decoder_opt); Looped-CED v1 and the second-pass control (ced/model/control.py build_model).
  Under the cap GatedAttention.queries / keys_values raise, so a caller that forms its own attention from them (the
  diagnostics' explicit rows, the cost measurements) fails instead of running without the cap, and the plain attention
  functions refuse a key set that carries k_far.

Grid KV sharing (AttentionConfig.kv_sharing "grid"). The n_q_heads = R x C x P query heads sit on a grid of R x C cells with P = heads_per_cell
consecutive heads to a cell: head h is in the cell c = h // P, the grid cell (i, j) = (c // C, c % C) (P = 1: head h = i C + j; P = 2 on 2 x 2:
heads 0-1 are cell (0,0), 2-3 (0,1), 4-5 (1,0), 6-7 (1,1)). The layer projects ONE key component and ONE value component per row (row_dim numbers)
and per column (col_dim numbers); the heads of cell (i, j) read [row i; column j] as their key and as their value (head_dim = row_dim + col_dim, the
layer's own head dimension), so R + C components give R C distinct keys. A THIRD AXIS (AttentionConfig.grid_depth, depth_dim; both 0 by default,
which is the grid exactly as it was) adds D components of depth_dim numbers: the cells are R x C x D (row-major, the last axis the fastest: head h
is in the cell c = h // P = (i, j, k) = (c // (C D), (c // D) % C, c % D)), a head reads [row i; column j; depth k] (head_dim = row_dim + col_dim +
depth_dim), the cache is R row_dim + C col_dim + D depth_dim numbers wide (rows', columns', then depth components), and R + C + D components give R
C D distinct keys; every other statement below holds for it unchanged (GridSpec.axes, assemble_grid and grid_spec are the only code that knows the
number of axes). What is stored is what is projected, RAW:
- The cache (the layer's KeySet) holds the RAW COMPONENTS, before any norm and before rotary: k and v are each one "head" of width
  R row_dim + C col_dim (+ D depth_dim) (rows first, then columns, then depth), the shape of a gqa cache with one KV head, so everything that slices, concatenates
  or batches a cache along the batch or the sequence axis handles it unchanged; and KeySet.pos (B, S), the rotary position each entry
  is rotated at when it is read (the document position; TST's token-scale rotary s x pos). k_far is not stored (it is derived at the
  read, below). Nothing assembled, normed or rotated is ever stored.
- The READ assembles, norms and rotates, per head, with each entry's own position (GatedAttention._grid_keys, _grid_read): the head's key [row
  i; column j (depth k)] (assemble_grid, pure indexing) is normed over the WHOLE head_dim by the layer's ONE k_norm (the gqa layer's module:
  RMSNorm over head_dim, one gain vector of head_dim numbers, the same init), then rotated in its leading rotary_dims(head_dim, rope_fraction)
  dimensions (apply_rope, or under the cap the shared CappedRotation tables; k_far under the cap is the normed key before rotation); the value
  [row i; column j] is not normed, as no value is. That is the gqa layer's sequence of operations on a head's key, so the grid layer IS the gqa
  layer with n_kv_heads = n_q_heads whose k_proj / v_proj rows are block-assembled [row i; column j] (tests/test_grid_kv.py: logits and
  gradients, QK-norm on), and a grid differs from the GQA control of the same cache in the sharing alone. The ordinary attention kernel
  (reference, flex, sdpa; capped or plain) runs on the read's per-head keys with n_kv_heads = n_q_heads. q_norm runs over the whole head_dim as
  it always did. The rotary distance cap, TST's token-scale rotary, the NoPE layers of the by-layer mix, the output gate and the temperature
  (tau multiplies q) are the ones of the gqa layer, in the same places.
- Cost, recorded as implementation debt, not as a cost of the idea (the cost rule of 2026-10-25): a read of a cached key set (prefill
  chunks, generation steps) assembles, norms and rotates every cached entry again, since nothing assembled is stored; a training forward
  does so once per layer on its own chunk. The cache's numbers are the components' alone (plus the metadata every KeySet has).
- keys_values / capped_keys_values (the callers that form their own attention) return the ASSEMBLED, normed and rotated per-head keys and
  the assembled values.
- Refused where it is not built (control.check_kv_sharing; GatedAttention.read): the second pass and every slot read, the other
  arches, the decoder, asentmax, the FlashAttention backends, the row term "pdp", the fused projections, FP8 / NVFP4 / MX, the Copy-Paste head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ced.config import FLEX_TILE_MODES, AttentionConfig
from ced.model.entmax import entmax_attention
from ced.model.norms import RMSNorm, compute_dtype
from ced.model.rope import apply_rope, rope_tables, rotary_dims, rotate_with_tables
from ced.model.sequence import SeqContext


@dataclass
class KeySet:
    """Keys and values a query may attend to, plus the metadata that builds the mask.

    k, v: (B, S, H_kv, head_dim)
    idx:  (B, S) absolute index in the row of the position that produced the entry
    doc:  (B, S) document id of that position
    ok:   (B, S) bool, False for entries that must never be attended (for example empty
          capacity slots, or a decoder position's memory entry)
    k_far: None, or under the rotary distance cap (AttentionConfig.rope_max_distance; the module docstring)
          (B, S, H_kv, head_dim): the same keys before rotation, which the keys beyond the cap are scored with
    canon_input, canon_doc: None, or in the cache of an encoder global layer that has a Canon layer
          (EncoderConfig.canon_kernel = K; ced/model/canon.py) (B, K-1, d_model) and (B, K-1): the last K - 1 inputs of
          that Canon layer and their document ids, which the next chunk's convolution reads. Not keys: no read looks
          at them, `cat` and `without_far` leave them out, and the encoder layer sets them on the key set it returns
          (with_canon).
    cp_q, cp_k: None, or on the key set the Copy-Paste pointer's shared layer returns (model.cp_head.pointer_input
          "shared"; GatedAttention.cp_query): (B, L, head_dim) each, that chunk's query of the shared head (QK-normed;
          after the layer's temperature with model.cp_head.shared_ssmax) and its key head's key, for the pointer. Not
          keys of a cache: no read looks at them, `cat`, `without_far` and `with_canon` leave them out, and
          cp_head.pointer_inputs drops them from the cache entry it hands on.
    tc_q, tc_q_far: None, or on the key set an encoder global layer returns under the Copy-Paste head's trunk copy
          (model.cp_head.trunk_copy; GatedAttention.trunk_copy; ced/model/cp_trunk.py): (B, L, Hq, head_dim) each, that chunk's
          queries exactly as the layer's attention read them (after QK-norm, RoPE and the temperature), and under the rotary cap
          the far rotation (else None). Not keys of a cache: no read looks at them, `cat`, `without_far` and `with_canon` leave
          them out, and cp_trunk.trunk_inputs drops them from the cache entry it hands on.
    tc_lse: None, or with tc_q under the trunk copy's fused paths (model.cp_head.kernel "fused" or "fused2"; GatedAttention.trunk_lse,
          the flex backend, a read without a cache): (B, Hq, L) float32, detached, the log-sum-exp of that read's own softmax per query
          (under the rotary cap the near and far reads' logaddexp), which the fused copy kernels take as their offset
          (ced/kernels/cp_trunk_fused.py, cp_trunk_fused2.py).
          Left out like tc_q.
    pos:  None (every gqa layer's key set), or in the cache of a grid KV sharing layer (AttentionConfig.kv_sharing "grid"; the module
          docstring) (B, S) integer: the rotary position the entry's key is rotated at when the layer reads it (the document position;
          TST's token-scale rotary s x position). That cache holds the RAW components (k, v of width R row_dim + C col_dim (+ D depth_dim), no norm, no
          rotary, k_far None), and the read norms and rotates the assembled per-head keys at these positions. Carried like idx and doc
          by `cat`, `without_far`, `with_canon` and a row or sequence slice (mtp_spec.truncate_keys, pass_schedule.select_rows).
    """

    k: torch.Tensor
    v: torch.Tensor
    idx: torch.Tensor
    doc: torch.Tensor
    ok: torch.Tensor
    k_far: torch.Tensor | None = None
    canon_input: torch.Tensor | None = None
    canon_doc: torch.Tensor | None = None
    cp_q: torch.Tensor | None = None
    cp_k: torch.Tensor | None = None
    tc_q: torch.Tensor | None = None
    tc_q_far: torch.Tensor | None = None
    tc_lse: torch.Tensor | None = None
    pos: torch.Tensor | None = None

    def with_canon(self, canon_input: torch.Tensor, canon_doc: torch.Tensor) -> "KeySet":
        """The same key set carrying a Canon layer's history (the class docstring): an encoder global layer's cache."""
        return KeySet(k=self.k, v=self.v, idx=self.idx, doc=self.doc, ok=self.ok, k_far=self.k_far,
                      canon_input=canon_input, canon_doc=canon_doc, pos=self.pos)

    @staticmethod
    def cat(*sets: "KeySet | None") -> "KeySet":
        sets = [s for s in sets if s is not None]
        far = [s.k_far for s in sets]
        if any(f is not None for f in far):
            assert all(f is not None for f in far), \
                "KeySet.cat: some key sets carry the unrotated keys of the rotary distance cap (k_far) and some do " \
                "not; a cache built without the cap cannot be continued under it"
        pos = [s.pos for s in sets]
        if any(p is not None for p in pos):
            assert all(p is not None for p in pos), \
                "KeySet.cat: some key sets carry the rotary positions of a grid KV sharing cache (pos) and some do not"
        return KeySet(
            k=torch.cat([s.k for s in sets], dim=1),
            v=torch.cat([s.v for s in sets], dim=1),
            idx=torch.cat([s.idx for s in sets], dim=1),
            doc=torch.cat([s.doc for s in sets], dim=1),
            ok=torch.cat([s.ok for s in sets], dim=1),
            k_far=torch.cat(far, dim=1) if far and far[0] is not None else None,
            pos=torch.cat(pos, dim=1) if pos and pos[0] is not None else None,
        )

    def without_far(self) -> "KeySet":
        """The same key set without k_far: what the plain attention functions take (they refuse a key set that
        carries it). For a read in which no key is beyond the cap."""
        return KeySet(k=self.k, v=self.v, idx=self.idx, doc=self.doc, ok=self.ok, pos=self.pos)

    def __len__(self) -> int:
        return self.k.shape[1]


def allowed(keys: KeySet, q_idx: torch.Tensor, q_doc: torch.Tensor) -> torch.Tensor:
    """(B, Lq, S) mask: the key is valid, not in the future, and in the query's document."""
    return (
        keys.ok[:, None, :]
        & (keys.idx[:, None, :] <= q_idx[:, :, None])
        & (keys.doc[:, None, :] == q_doc[:, :, None])
    )


def attend(q: torch.Tensor, keys: KeySet, allow: torch.Tensor) -> torch.Tensor:
    """Softmax attention of q (B, Lq, Hq, hd) over keys, restricted by allow (B, Lq, S).

    Every query row must allow at least one key (callers guarantee a query can always see itself),
    otherwise the softmax row would be NaN. Key/value head j serves query heads
    j * group ... (j + 1) * group - 1, as in the Hugging Face repeat_kv convention.
    """
    assert getattr(keys, "k_far", None) is None, NOT_CAPPED
    hq, hkv = q.shape[2], keys.k.shape[2]
    group = hq // hkv
    k = keys.k.repeat_interleave(group, dim=2)
    v = keys.v.repeat_interleave(group, dim=2)
    dtype = compute_dtype(q)
    scores = torch.einsum("bqhd,bshd->bhqs", q.to(dtype), k.to(dtype)) * q.shape[-1] ** -0.5
    scores = scores.masked_fill(~allow[:, None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqs,bshd->bqhd", probs, v.to(dtype)).to(q.dtype)


# ---------------------------------------------------------------------------------------------------
# grid KV sharing (AttentionConfig.kv_sharing "grid"; the module docstring)
# ---------------------------------------------------------------------------------------------------

KV_SHARINGS = ("gqa", "grid")


@dataclass(frozen=True)
class GridSpec:
    """The grid of a layer with kv_sharing "grid": rows x cols cells (x depth cells with a third axis) of heads_per_cell consecutive query
    heads (head h is in the cell c = h // heads_per_cell = i * cols + j, the grid cell (i, j); with a third axis c = (i * cols + j) * depth +
    k, the cell (i, j, k); with heads_per_cell 1, head h = c), one key and one value component of row_dim numbers per row, of col_dim numbers
    per column and, with the third axis, of depth_dim numbers per depth index. The key of a cell's heads is [row i ; column j] (with a third
    axis [row i ; column j ; depth k]), head_dim = row_dim + col_dim (+ depth_dim) wide. The cache holds `width` numbers a token for the keys
    and the same for the values: the rows' components first (rows * row_dim), then the columns' (cols * col_dim), then the depth components
    (depth * depth_dim). depth = depth_dim = 0 (the default) is the two-axis grid, exactly as it was before the third axis existed."""

    rows: int
    cols: int
    row_dim: int
    col_dim: int
    heads_per_cell: int = 1
    depth: int = 0
    depth_dim: int = 0

    @property
    def axes(self) -> tuple[tuple[int, int], ...]:
        """((count, dim) of each axis in the key's order: the rows, the columns, and with a third axis the depth)."""
        return ((self.rows, self.row_dim), (self.cols, self.col_dim)) + (((self.depth, self.depth_dim),) if self.depth else ())

    @property
    def n_axes(self) -> int:
        return len(self.axes)

    @property
    def n_cells(self) -> int:
        n = 1
        for count, _ in self.axes:
            n *= count
        return n

    @property
    def n_heads(self) -> int:
        return self.n_cells * self.heads_per_cell

    def cell_of(self, head: int) -> tuple[int, ...]:
        """The grid cell of query head `head`: cell c = head // heads_per_cell, row-major with the last axis the fastest: (i, j) = (c // cols,
        c % cols), or with a third axis (i, j, k) = (c // (cols * depth), (c // depth) % cols, c % depth)."""
        c = head // self.heads_per_cell
        out = []
        for count, _ in reversed(self.axes[1:]):
            out.append(c % count)
            c //= count
        return (c, *reversed(out))

    @property
    def head_dim(self) -> int:
        return sum(dim for _, dim in self.axes)

    @property
    def row_width(self) -> int:
        return self.rows * self.row_dim

    @property
    def col_width(self) -> int:
        return self.cols * self.col_dim

    @property
    def depth_width(self) -> int:
        return self.depth * self.depth_dim

    @property
    def width(self) -> int:
        return sum(count * dim for count, dim in self.axes)


def grid_spec(cfg: AttentionConfig) -> GridSpec | None:
    """The GridSpec of an attention config: None for kv_sharing "gqa" (whose grid keys must then be 0, heads_per_cell 1: a value would be
    ignored), else the grid, checked (ValueError, by name): n_q_heads = rows x cols (x depth) x heads_per_cell, head_dim = row_dim + col_dim
    (+ depth_dim), n_kv_heads 1, every size >= 1; the third axis' two keys, grid_depth and depth_dim, are both 0 (two axes) or both >= 1.
    (The rotary part of a head is rotated at the read, on the assembled head, as a gqa head's is, so it may reach into the later components:
    no further condition.)"""
    sharing = cfg.kv_sharing
    if sharing not in KV_SHARINGS:
        raise ValueError(f"attention.kv_sharing {sharing!r}: one of {KV_SHARINGS}")
    keys = (cfg.grid_rows, cfg.grid_cols, cfg.row_dim, cfg.col_dim)
    third = (cfg.grid_depth, cfg.depth_dim)
    if sharing == "gqa":
        if any(keys) or any(third) or cfg.heads_per_cell != 1:
            raise ValueError(f"attention.grid_rows / grid_cols / row_dim / col_dim {keys}, grid_depth / depth_dim {third} and heads_per_cell "
                             f"{cfg.heads_per_cell} are the grid's; kv_sharing is 'gqa', which would ignore them: set kv_sharing 'grid' or leave "
                             "the keys 0 and heads_per_cell 1")
        return None
    if not all(type(v) is int and v >= 1 for v in (*keys, cfg.heads_per_cell)):
        raise ValueError(f"attention.kv_sharing 'grid': grid_rows, grid_cols, row_dim, col_dim, heads_per_cell are integers >= 1, got {(*keys, cfg.heads_per_cell)}")
    if not all(type(v) is int and v >= 0 for v in third) or (third[0] == 0) != (third[1] == 0):
        raise ValueError(f"attention.kv_sharing 'grid': grid_depth and depth_dim (the third axis) are integers set together, both 0 (two axes) or both >= 1, "
                         f"got grid_depth {cfg.grid_depth}, depth_dim {cfg.depth_dim}")
    g = GridSpec(*keys, cfg.heads_per_cell, *third)
    if cfg.n_q_heads != g.n_heads:
        shape = " x ".join(["grid_rows", "grid_cols"] + (["grid_depth"] if g.depth else []) + ["heads_per_cell"])
        sizes = " x ".join(str(n) for n in [g.rows, g.cols] + ([g.depth] if g.depth else []) + [g.heads_per_cell])
        raise ValueError(f"attention.kv_sharing 'grid': n_q_heads {cfg.n_q_heads} must equal {shape} = {sizes} = {g.n_heads}")
    if cfg.head_dim != g.head_dim:
        dims = " + ".join(["row_dim", "col_dim"] + (["depth_dim"] if g.depth else []))
        sizes = " + ".join(str(dim) for _, dim in g.axes)
        raise ValueError(f"attention.kv_sharing 'grid': head_dim {cfg.head_dim} must equal {dims} = {sizes} = {g.head_dim} "
                         f"(a head's key is [{' ; '.join(['row', 'column'] + (['depth'] if g.depth else []))}])")
    if cfg.n_kv_heads != 1:
        raise ValueError(f"attention.kv_sharing 'grid': n_kv_heads must be 1 (the cache is one head of {g.width} components' numbers), got {cfg.n_kv_heads}")
    return g


def assemble_grid(comp: torch.Tensor, g: GridSpec) -> torch.Tensor:
    """The per-head tensor of a grid layer from its components: comp (B, S, 1, g.width) (rows' components, then columns', then with a third axis
    the depth components) to (B, S, g.n_heads, g.head_dim), where the heads of cell (i, j) (heads (i * cols + j) * heads_per_cell, ... +
    heads_per_cell - 1) are [comp of row i ; comp of column j], and with a third axis those of cell (i, j, k) [row i ; column j ; depth k].
    Pure indexing (an expand and a cat): the values are the stored numbers, nothing is computed."""
    B, S, one, W = comp.shape
    assert one == 1 and W == g.width, (tuple(comp.shape), g)
    P = g.heads_per_cell
    counts = [count for count, _ in g.axes]
    parts, start = [], 0
    for a, (count, dim) in enumerate(g.axes):
        shape = [B, S] + [1] * len(counts) + [1, dim]
        shape[2 + a] = count
        parts.append(comp[..., start : start + count * dim].reshape(*shape).expand(B, S, *counts, P, dim))
        start += count * dim
    return torch.cat(parts, dim=-1).reshape(B, S, g.n_heads, g.head_dim)


# ---------------------------------------------------------------------------------------------------
# the rotary distance cap (AttentionConfig.rope_max_distance; the module docstring)
# ---------------------------------------------------------------------------------------------------

# SeqContext.memo key of the shared BlockMasks of the encoder's capped own read (fast_attention.CappedMasks; written by
# Encoder.share_capped_masks once per forward, read by GatedAttention.forward in every global layer)
CAPPED_MASKS_KEY = "capped_own_read_masks"
# SeqContext.memo key of the capped read's rotation tables (CappedRotation below; written by
# Encoder.share_capped_rotation once per forward, read by GatedAttention.forward in every global layer)
CAPPED_ROTATION_KEY = "capped_rotation_tables"
# SeqContext.memo key of the rotary positions under Token Superposition Training's token-scale rotary
# (SeqContext.rope_scale > 1; TSTConfig.token_scale_rope): rope_scale x ctx.pos, written by Encoder.share_capped_rotation
# once per forward (absent with rope_scale 1, every forward before TST).
ROPE_POS_KEY = "tst_rope_positions"
# SeqContext.memo key of attention backends "fa3" and "fa4"'s varlen metadata (ced/kernels/flash_capped.py FlashVarlen;
# written by Encoder.share_flash_varlen once per forward, read by GatedAttention.forward in every global layer). Absent on
# every other backend.
FLASH_VARLEN_KEY = "fa3_varlen"
# the backends that read through that metadata: FlashAttention-3 (ced/kernels/flash_capped.py) and FlashAttention-4
# (ced/kernels/flash_fa4.py)
FLASH_BACKENDS = ("fa3", "fa4")
FA3_NOT_COVERED = ("attention backend 'fa3' covers the encoder global layers' own read in a training forward without a "
                   "cache (Encoder.forward builds its varlen metadata)")


def flash_not_covered(backend: str) -> str:
    """FA3_NOT_COVERED for backend "fa3" or "fa4"."""
    return FA3_NOT_COVERED.replace("'fa3'", repr(backend))


def rotary_positions(ctx: SeqContext) -> torch.Tensor:
    """The positions the capped-rotary layers rotate at: ctx.pos, or under token-scale rotary (ctx.rope_scale = s > 1;
    TSTConfig.token_scale_rope) s x ctx.pos, a bag's document position in tokens (the memo's tensor when the encoder
    built it for this forward)."""
    if ctx.rope_scale == 1:
        return ctx.pos
    pos = ctx.memo.get(ROPE_POS_KEY)
    return ctx.pos * ctx.rope_scale if pos is None else pos


def mask_cap(cap: int, rope_scale: int) -> int:
    """The capped read's cap in positions of the row when the rotary angle of a step is rope_scale tokens
    (TSTConfig.token_scale_rope): a key d positions back is near while rope_scale x d <= cap, i.e. d <= cap // rope_scale;
    the far keys are then rotated by the cap itself, in tokens. cap for rope_scale 1."""
    out = cap // rope_scale
    assert out >= 1, f"rope_max_distance {cap} under token-scale rotary at {rope_scale} tokens a position: no near key"
    return out


@dataclass(frozen=True)
class CappedRotation:
    """The rotation tables of a capped read, computed once from the positions and handed to every layer that rotates
    at them.

    cos, sin (B, L, 1, r): rope.rope_tables at the queries' / keys' document positions `pos` (B, L); far_cos, far_sin
    (1, 1, 1, r): the same at the one position max_distance, which rotates q_far. All in `cos.dtype`, at least float32
    (norms.compute_dtype of what is rotated): float32 and float64 models rotate exactly as rope.apply_rope does, bit
    for bit; under bfloat16 autocast the rotation is computed in float32 and rounded once (rope.rotate_with_tables).

    Why (measured, runs rc-prof2 and rc-prof2b): computed inside a compiled layer, the tables are not tensors at all.
    Inductor inlines their float64 pow / cos / sin into the kernels that use them, for every element. One backward
    kernel of the capped layer took 8.2 ms per layer call that way (0.8 ms in the layer without the cap), three
    quarters of the cap's cost per step; with the tables passed in it is gone. Why float32 under bfloat16 autocast
    (read in Inductor's source, codegen/triton.py to_dtype): inside a fused kernel a cast to bfloat16 is a cast to
    float32, values are rounded to bfloat16 only where a kernel stores them. So the compiled layers always rotated
    with float32-precision tables; float32 tables are the values they used."""

    cos: torch.Tensor
    sin: torch.Tensor
    far_cos: torch.Tensor
    far_sin: torch.Tensor
    pos: torch.Tensor
    max_distance: int
    # (head_dim, rope_fraction, rope_theta) the tables were computed for: a layer with other rotary settings that is
    # handed this context must not rotate with them (the tables alone do not say which theta they hold)
    rotary: tuple = ()

    def check(self, pos: torch.Tensor, max_distance: int, dtype: torch.dtype, rotary: tuple | None = None) -> None:
        """Refuse a use these tables were not built for: the cap, the rotary settings (when the caller gives its
        own), the shape and the dtype always; the positions' values when the call runs eagerly (a data-dependent
        check: never part of a compiled graph)."""
        assert self.max_distance == max_distance, f"rotation tables built for cap {self.max_distance}, used with {max_distance}"
        assert rotary is None or tuple(rotary) == self.rotary, \
            f"rotation tables built for (head_dim, rope_fraction, rope_theta) = {self.rotary}, used by a layer with {tuple(rotary)}"
        assert self.cos.shape[:2] == pos.shape and self.cos.dtype == dtype, \
            f"rotation tables of shape {tuple(self.cos.shape)} / {self.cos.dtype}, used at positions {tuple(pos.shape)} in {dtype}"
        if not torch.compiler.is_compiling():
            assert self.pos is pos or bool((self.pos == pos).all()), \
                "rotation tables were built for other positions than these (tables of another batch?)"


def _capped_rotation_tensors(pos: torch.Tensor, max_distance: int, head_dim: int, fraction: float, theta: float,
                             dtype: torch.dtype):
    cos, sin = rope_tables(pos, head_dim, fraction, theta, dtype)
    far_cos, far_sin = rope_tables(pos.new_full((1, 1), max_distance), head_dim, fraction, theta, dtype)
    return cos, sin, far_cos, far_sin


_compiled_capped_rotation = None


def capped_rotation(pos: torch.Tensor, cfg: AttentionConfig, dtype: torch.dtype, compiled: bool = False) -> CappedRotation:
    """The CappedRotation of positions pos (B, L) for attention config cfg (its cap, head_dim, rope_fraction,
    rope_theta) in dtype. compiled: build through a compiled function (ignored inside a compile): with compiled
    layers the tables are then computed by the same generated float64 code that computed them inside the layers'
    kernels before."""
    global _compiled_capped_rotation
    cap = rope_cap(cfg)
    assert cap is not None, "capped_rotation is for an attention config with rope_max_distance and rotary dimensions"
    build = _capped_rotation_tensors
    if compiled and not torch.compiler.is_compiling():
        if _compiled_capped_rotation is None:
            _compiled_capped_rotation = torch.compile(_capped_rotation_tensors, dynamic=False)
        build = _compiled_capped_rotation
    cos, sin, far_cos, far_sin = build(pos, cap, cfg.head_dim, cfg.rope_fraction, cfg.rope_theta, dtype)
    return CappedRotation(cos=cos, sin=sin, far_cos=far_cos, far_sin=far_sin, pos=pos, max_distance=cap,
                          rotary=(cfg.head_dim, cfg.rope_fraction, cfg.rope_theta))

NOT_CAPPED = ("this key set carries the unrotated keys of the rotary distance cap (KeySet.k_far): this attention path "
              "does not implement AttentionConfig.rope_max_distance (use attend_capped / capped_attention)")


def rope_cap(cfg: AttentionConfig) -> int | None:
    """The distance cap an attention config asks for, or None: rope_max_distance when the head has rotary
    dimensions. NoPE (rope_fraction 0.0) has nothing to cap: None, and the layer runs the plain code."""
    C = cfg.rope_max_distance
    if C is None:
        return None
    assert isinstance(C, int) and not isinstance(C, bool) and C >= 1, \
        f"rope_max_distance {C!r}: None, or an integer >= 1 (a distance in tokens)"
    return C if rotary_dims(cfg.head_dim, cfg.rope_fraction) > 0 else None


def far_keys(keys: KeySet, q_idx: torch.Tensor, max_distance: int) -> torch.Tensor:
    """(B, Lq, S) bool: key s lies more than max_distance positions before the query, by absolute row index. For
    a key the mask allows (the query's document, at or before it) that is pos_q - pos_k > max_distance, since a
    document's positions are consecutive in the row; any other key is masked and its entry is never read."""
    return q_idx[:, :, None] - keys.idx[:, None, :] > max_distance


def attend_capped(q: torch.Tensor, q_far: torch.Tensor, keys: KeySet, allow: torch.Tensor, far: torch.Tensor) -> torch.Tensor:
    """`attend` with the rotary distance capped: the reference, the ground truth of AttentionConfig.rope_max_distance.

    One softmax per query over the allowed keys, exactly as `attend`; only the logit of a key beyond the cap differs:

        far[t, s] False:  q[t] . keys.k[s]            both rotated at their own positions: attend's logit
        far[t, s] True:   q_far[t] . keys.k_far[s]    the query rotated by the cap against the unrotated key

    q, q_far: (B, Lq, Hq, hd), the same normed query rotated at its position / by the cap (GatedAttention.capped_queries);
    keys.k, keys.k_far: (B, S, Hkv, hd), rotated at the key's position / not rotated; allow, far: (B, Lq, S) bool
    (allowed or slots_allowed; far_keys). Both logit tensors are computed in full and `where` picks per pair, so with
    no pair beyond the cap this is `attend` bit for bit. Same GQA mapping, scale, dtype and row rule as `attend`."""
    assert keys.k_far is not None, "attend_capped needs the unrotated keys (KeySet.k_far)"
    hq, hkv = q.shape[2], keys.k.shape[2]
    group = hq // hkv
    k = keys.k.repeat_interleave(group, dim=2)
    k_far = keys.k_far.repeat_interleave(group, dim=2)
    v = keys.v.repeat_interleave(group, dim=2)
    dtype = compute_dtype(q)
    near_scores = torch.einsum("bqhd,bshd->bhqs", q.to(dtype), k.to(dtype)) * q.shape[-1] ** -0.5
    far_scores = torch.einsum("bqhd,bshd->bhqs", q_far.to(dtype), k_far.to(dtype)) * q.shape[-1] ** -0.5
    scores = torch.where(far[:, None, :, :], far_scores, near_scores)
    scores = scores.masked_fill(~allow[:, None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqs,bshd->bqhd", probs, v.to(dtype)).to(q.dtype)


def capped_attention(
    q: torch.Tensor, q_far: torch.Tensor, keys: KeySet, q_idx: torch.Tensor, q_doc: torch.Tensor, q_pos: torch.Tensor,
    max_distance: int, backend: str = "reference", empty_reads_first: bool = False, block_masks=None, return_lse: bool = False,
    tiles: str = "classified",
    row_term: str = "stored_o",
):
    """The attention of queries at q_idx / q_doc / q_pos (B, Lq) over `keys` with the rotary distance capped at
    max_distance: the mask of `allowed` (a layer's own read) or, with empty_reads_first, `slots_allowed` (a slot
    read: an empty slot reads key 0). backend "reference": attend_capped; "flex" / "sdpa":
    fast_attention.attend_capped_fast, which also reads the queries' document positions q_pos.
    block_masks (flex only): the read's two BlockMasks built by the caller (fast_attention.CappedMasks), shared by
    every layer that makes the same read; None: the read builds its own. The same result bit for bit.
    return_lse (the trunk copy's fused path): (output, the read's log-sum-exp (B, Hq, Lq) float32 detached, or None where the backend
    gives none: the reference and sdpa); the output is the same.
    tiles (flex): AttentionConfig.flex_tiles ("classified", flex's own tile masks, the default; "indexed", the full tiles folded in).
    The block_masks of a caller must have been built with the same tiles (CappedMasks.check).
    row_term (AttentionConfig.row_term): "stored_o" (default) calls as it always did; "pdp" is passed to attend_capped_fast (flex; the
    reference's backward is already rowsum(P . dP), so there it is inert; sdpa is refused by name at the model's build, control.check_row_term)."""
    if backend == "reference":
        assert block_masks is None, "block_masks are for the flex backend"
        rule = slots_allowed if empty_reads_first else allowed
        out = attend_capped(q, q_far, keys, rule(keys, q_idx, q_doc), far_keys(keys, q_idx, max_distance))
        return (out, None) if return_lse else out
    from ced.model.fast_attention import attend_capped_fast

    kw = {} if tiles == "classified" else {"tiles": tiles}  # the default adds no argument: the calls are the calls they always were
    if row_term != "stored_o":  # the same for the row term (AttentionConfig.row_term)
        kw["row_term"] = row_term
    if return_lse:
        return attend_capped_fast(q, q_far, keys, q_idx, q_doc, q_pos, max_distance, backend=backend,
                                  empty_reads_first=empty_reads_first, block_masks=block_masks, return_lse=True, **kw)
    if block_masks is None:  # the call of the code before the shared masks
        return attend_capped_fast(q, q_far, keys, q_idx, q_doc, q_pos, max_distance, backend=backend,
                                  empty_reads_first=empty_reads_first, **kw)
    return attend_capped_fast(q, q_far, keys, q_idx, q_doc, q_pos, max_distance, backend=backend,
                              empty_reads_first=empty_reads_first, block_masks=block_masks, **kw)


def slots_allowed(keys: KeySet, q_idx: torch.Tensor, q_doc: torch.Tensor) -> torch.Tensor:
    """(B, C, S) mask for slot queries over a key set they did not write: the `allowed` rule, plus
    one rule for empty slots (q_doc < 0), which match no key: they read key 0 only.

    That keeps every softmax row finite (key 0 always exists: a key set holds at least the current
    chunk). An empty slot's output is never used, and no valid slot reads it (slots write no keys).
    """
    first = torch.arange(len(keys), device=q_idx.device) == 0
    return allowed(keys, q_idx, q_doc) | ((q_doc < 0)[:, :, None] & first[None, None, :])


def attend_slots(
    q: torch.Tensor, keys: KeySet, q_idx: torch.Tensor, q_doc: torch.Tensor, backend: str = "reference",
    block_mask=None, allow: torch.Tensor | None = None,
) -> torch.Tensor:
    """Slot queries q (B, C, Hq, hd) at positions q_idx / documents q_doc (B, C) over `keys`:
    softmax over the keys at s <= t of the slot's document (ok entries only), empty slots as in
    `slots_allowed`. backend "reference" (dense mask, ground truth), "flex" or "sdpa".
    block_mask (flex only): the flex BlockMask of this read built by the caller (fast_attention.slot_block_mask
    of the same key metadata and queries), shared by every layer that makes the same read; None builds it here.
    allow ("reference" and "sdpa" only): the dense mask of this read built by the caller, slots_allowed(keys, q_idx,
    q_doc) of the same metadata, (B, C, S) bool, shared the same way (the segment pass schedule builds one per pass:
    ced/model/pass_schedule.py); None builds it here, as before."""
    if backend == "reference":
        assert block_mask is None, "block_mask is for the flex backend"
        return attend(q, keys, slots_allowed(keys, q_idx, q_doc) if allow is None else allow)
    from ced.model.fast_attention import attend_fast

    if allow is not None:
        assert backend == "sdpa" and block_mask is None, "a caller-built dense mask is for the reference and sdpa backends"
        return attend_fast(q, keys, q_idx, q_doc, backend=backend, empty_reads_first=True, dense=allow)
    return attend_fast(q, keys, q_idx, q_doc, backend=backend, empty_reads_first=True, block_mask=block_mask)


TEMPERATURES = ("fixed", "learned", "ssmax")
NORMALIZERS = ("softmax", "asentmax")
ASENTMAX_INIT_STD = 0.02  # beta_proj / gamma_proj weights (the authors' implicit Gemma2 init); bias 0
_BACKEND_NOTED: set = set()  # backends already reported as unused by asentmax (printed once per process)


def ssmax_factor(pos: torch.Tensor, ref_len: int, dtype: torch.dtype) -> torch.Tensor:
    """ln(n) / ln(ref_len) for n = pos + 1 (B, L), in dtype. Computed as 1 + ln(n / ref_len) / ln(ref_len),
    which is exactly 1 at n = ref_len (ln(1) = 0), so ssmax at init equals the fixed temperature there bit
    for bit. 0 at pos 0, where the query sees one key."""
    n = (pos + 1).to(dtype)
    return 1.0 + torch.log(n / ref_len) / math.log(ref_len)


def key_count_factor(n_keys: torch.Tensor, ref_len: int, dtype: torch.dtype) -> torch.Tensor:
    """ln(n) / ln(ref_len) for an exact key count n_keys (B, L) (integers, >= 1 at real queries), in dtype:
    ssmax_factor's formula with n given instead of pos + 1 (key_count_factor(pos + 1) is ssmax_factor(pos) bit for
    bit), so it is exactly 1 at n = ref_len."""
    n = n_keys.to(dtype)
    return 1.0 + torch.log(n / ref_len) / math.log(ref_len)


class GatedAttention(nn.Module):
    """Projections of one gated GQA attention layer (Qwen3-Next / Qwen3.5 layout, no biases).

    q_proj produces the query and, for the elementwise gate, the gate, interleaved per head.
    QK-norm (zero-centred RMSNorm over head_dim, shared across heads) comes before RoPE.

    own_keys=False builds a query-only layer (no k_proj, v_proj, k_norm): the v2 decoder's
    attention, whose keys and values are the encoder-written memory only.

    Temperature (module docstring): "learned" and "ssmax" add temp_w (n_q_heads,), zero-centred like the
    norm gains; "fixed" adds nothing. query_scale (inference only, diagnostics): a plain attribute, None
    or a callable f(pos) -> (B, L), (B, L, 1, 1) or (B, L, Hq, 1); when set, q is also multiplied by
    f(pos) where tau is applied. It is not a parameter or buffer, so never in the state dict.
    """

    def __init__(self, d_model: int, cfg: AttentionConfig, norm_eps: float, own_keys: bool = True):
        super().__init__()
        assert cfg.n_q_heads % cfg.n_kv_heads == 0
        assert cfg.temperature in TEMPERATURES, f"unknown attention temperature {cfg.temperature!r}"
        assert cfg.temperature_ref_len >= 2, cfg.temperature_ref_len
        assert cfg.flex_tiles in FLEX_TILE_MODES, f"unknown flex_tiles {cfg.flex_tiles!r}: one of {FLEX_TILE_MODES}"
        self.cfg = cfg
        # Grid KV sharing (AttentionConfig.kv_sharing; the module docstring): None for "gqa" (every layer before the key), else the
        # GridSpec. A plain attribute, never in the state dict. ValueError by name on a grid that does not fit the config.
        self.grid = grid_spec(cfg)
        assert self.grid is None or own_keys, "grid KV sharing is for a layer with its own keys"
        hq, hkv, hd = cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim
        kv_width = hkv * hd if self.grid is None else self.grid.width  # the grid's components: rows' and columns', for k and for v
        q_out = hq * hd * (2 if cfg.gate == "elementwise" else 1)
        self.q_proj = nn.Linear(d_model, q_out, bias=False)
        self.g_proj = nn.Linear(d_model, hq, bias=False) if cfg.gate == "headwise" else None
        self.k_proj = nn.Linear(d_model, kv_width, bias=False) if own_keys else None
        self.v_proj = nn.Linear(d_model, kv_width, bias=False) if own_keys else None
        self.o_proj = nn.Linear(hq * hd, d_model, bias=False)
        self.q_norm = RMSNorm(hd, norm_eps) if cfg.qk_norm else nn.Identity()
        if own_keys:  # the grid too: ONE k_norm over the assembled per-head key (head_dim), the gqa layer's module and init (the module docstring)
            self.k_norm = RMSNorm(hd, norm_eps) if cfg.qk_norm else nn.Identity()
        self.temp_w = nn.Parameter(torch.zeros(hq)) if cfg.temperature != "fixed" else None
        self.query_scale = None
        # ssmax where a query does not see pos + 1 keys (CED v2's full decoder layers with self-KV; set by
        # MemoryDecoderLayer): every query must then come with its exact key count n_keys (module docstring), and one
        # without is refused. A plain attribute, never in the state dict.
        self.needs_key_count = False
        assert cfg.normalizer in NORMALIZERS, f"unknown attention normalizer {cfg.normalizer!r}"
        # The rotary distance cap (module docstring): validated here (rope_cap); the layer reads it at every call
        # (self.rope_cap()), so no parameter, buffer or attribute is added and None changes nothing.
        assert rope_cap(cfg) is None or cfg.normalizer == "softmax", \
            f"rope_max_distance is built for the softmax normaliser, not {cfg.normalizer!r}"
        if cfg.normalizer == "asentmax":
            assert cfg.temperature == "fixed", \
                f"normalizer asentmax has its own learned scale on q; temperature {cfg.temperature!r} is refused (use 'fixed')"
            assert cfg.asentmax_chunk >= 0, cfg.asentmax_chunk
            # ASEntmax's scale heads (module docstring). Built only for asentmax: softmax models keep their
            # parameters, state dict and RNG use. Built and initialised here without drawing from the global
            # RNG (fork_rng restores the CPU stream), so every other parameter of an asentmax model gets exactly
            # the values of the softmax model of the same seed (the arm differs from its baseline by the
            # normaliser only); the model's init_weights skips them and draws them from a generator of their
            # own (ced/model/looped_ced.py init_asentmax_parameters), N(0, 0.02) whatever the model's init_std.
            with torch.random.fork_rng(devices=[]):
                self.beta_proj = nn.Linear(d_model, hq, bias=True)
                self.gamma_proj = nn.Linear(d_model, hq, bias=True)
                self.reset_asentmax_parameters()
            if cfg.backend != "reference" and cfg.backend not in _BACKEND_NOTED:
                _BACKEND_NOTED.add(cfg.backend)
                print(f"[attention] normalizer asentmax: backend {cfg.backend!r} is not used; every asentmax layer runs "
                      "the reference entmax attention (ced/model/entmax.py)", flush=True)
        # asentmax training statistics (ced/model/entmax.py EntmaxStats; set by the trainer): a plain attribute,
        # never in the state dict
        self.entmax_stats = None
        # The Copy-Paste pointer's shared head (model.cp_head.pointer_input "shared"; ced/model/cp_head.py
        # attach_cp_head): None (every layer of every other model), or (query head, inherit the temperature) on the one
        # layer whose head the pointer reads; forward then returns that head's query and key on its key set (KeySet.cp_q,
        # cp_k). A plain attribute, never in the state dict; the layer's own output is the same either way.
        self.cp_query = None
        # The Copy-Paste head's trunk copy (model.cp_head.trunk_copy; ced/model/cp_trunk.py; cp_head.attach_cp_head sets it on every
        # encoder global layer): False (every layer of every other model), or True: forward also returns this chunk's queries as its
        # attention read them (and the far rotation under the rotary cap) on its key set (KeySet.tc_q, tc_q_far). A plain attribute,
        # never in the state dict; the layer's output and its key set's keys are the same either way.
        self.trunk_copy = False
        # The trunk copy's fused paths (model.cp_head.kernel "fused" or "fused2"; attach_cp_head sets it with trunk_copy): False, or
        # True: a read without a cache on the flex backend also returns its own log-sum-exp per query (KeySet.tc_lse), the fused copy
        # kernels' offset. The same flex calls with the log-sum-exp requested (the capped read already requests it): the output is the same.
        self.trunk_lse = False

    def _tile_kw(self) -> dict:
        """The flex_tiles argument of the flex reads: none for the default "classified" (the calls are then the calls they always were), else
        {"tiles": ...} (AttentionConfig.flex_tiles; ced/model/fast_attention.py fold_full_blocks)."""
        return {} if self.cfg.flex_tiles == "classified" else {"tiles": self.cfg.flex_tiles}

    def reset_asentmax_parameters(self, generator: torch.Generator | None = None) -> None:
        """beta_proj / gamma_proj: weights N(0, 0.02) (from generator when given, drawn on its device and
        copied), bias 0."""
        for lin in (self.beta_proj, self.gamma_proj):
            w = lin.weight
            if generator is None or w.is_meta or w.device == generator.device:
                nn.init.normal_(w, mean=0.0, std=ASENTMAX_INIT_STD, generator=generator)
            else:
                with torch.no_grad():
                    w.copy_(torch.empty(w.shape, dtype=w.dtype, device=generator.device).normal_(
                        0.0, ASENTMAX_INIT_STD, generator=generator))
            nn.init.zeros_(lin.bias)

    def asentmax_scale(self, x: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(c, beta, gamma), each (B, L, Hq) in compute_dtype(x) (float32 under autocast: autocast is off here):
        beta = softplus(x . w_b + b_b), gamma = 2 tanh(x . w_g + b_g), c = 1 + beta * ln(pos + 2) ** gamma."""
        dt = compute_dtype(x)
        with torch.autocast(x.device.type, enabled=False):
            xf = x.to(dt)
            beta = F.softplus(F.linear(xf, self.beta_proj.weight.to(dt), self.beta_proj.bias.to(dt)))
            gamma = 2.0 * torch.tanh(F.linear(xf, self.gamma_proj.weight.to(dt), self.gamma_proj.bias.to(dt)))
            log_n = torch.log((pos + 2).to(dt))[:, :, None]  # ln(pos + 2) >= ln 2
            c = 1.0 + beta * torch.pow(log_n, gamma)
        return c, beta, gamma

    def query_factor(self, pos: torch.Tensor, dtype: torch.dtype, n_keys: torch.Tensor | None = None
                     ) -> torch.Tensor | None:
        """The factor on q at document positions pos (B, L): tau (module docstring) times query_scale(pos)
        when set, in dtype, broadcastable to (B, L, Hq, 1); None for the fixed temperature without an
        override (q is then left untouched). n_keys (B, L): the exact number of keys each query sees, for ssmax
        where that is not pos + 1 (ssmax only; required when needs_key_count is set)."""
        assert n_keys is None or self.cfg.temperature == "ssmax", \
            f"n_keys is ssmax's key count; temperature {self.cfg.temperature!r} does not read it"
        assert n_keys is None or n_keys.shape == pos.shape, (tuple(n_keys.shape), tuple(pos.shape))
        f = None
        if self.cfg.temperature != "fixed":
            f = (1.0 + self.temp_w.to(dtype))[:, None]  # (Hq, 1)
            if self.cfg.temperature == "ssmax" and n_keys is None:
                assert not self.needs_key_count, \
                    "ssmax here needs the exact key count n_keys: these queries do not see pos + 1 keys (CED v2's " \
                    "decoder with self-KV: ced_v2.slot_key_counts)"
                f = f * ssmax_factor(pos, self.cfg.temperature_ref_len, dtype)[:, :, None, None]
            elif self.cfg.temperature == "ssmax":
                f = f * key_count_factor(n_keys, self.cfg.temperature_ref_len, dtype)[:, :, None, None]
        if self.query_scale is not None:
            s = self.query_scale(pos).to(dtype)
            s = s[:, :, None, None] if s.ndim == 2 else s
            assert s.ndim == 4 and s.shape[:2] == pos.shape and s.shape[3] == 1, \
                f"query_scale(pos) must be (B, L), (B, L, 1, 1) or (B, L, Hq, 1); got {tuple(s.shape)}"
            f = s if f is None else f * s
        return f

    def queries(self, x: torch.Tensor, pos: torch.Tensor, n_keys: torch.Tensor | None = None,
                q_out: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(q, gate): q (B, L, Hq, hd) after QK-norm, RoPE and the temperature (and, for asentmax, the scale c).
        n_keys: see query_factor. q_out: q_proj(x) when the caller has it (model.fused_projections "qkv"; None: computed
        here, as always)."""
        q, gate, _ = self._project_queries(x, pos, n_keys, q_out)
        return q, gate

    def _fused_outputs(self, x: torch.Tensor) -> tuple[torch.Tensor | None, tuple | None]:
        """(q_proj(x), (k_proj(x), v_proj(x))) as views of one GEMM's output (model.fused_projections "qkv";
        ced/model/fused_proj.py) in a training forward; (None, None) otherwise: the separate projections, as always."""
        if not (getattr(self, "fused_proj", None) and self.training):
            return None, None
        from ced.model import fused_proj

        out = fused_proj.outputs(self, x)
        return out["q_proj"], (out["k_proj"], out["v_proj"])

    def rope_cap(self) -> int | None:
        """This layer's rotary distance cap (AttentionConfig.rope_max_distance; None without one, and for NoPE)."""
        return rope_cap(self.cfg)

    def _row_term_kw(self) -> dict:
        """The keyword the flex reads of forward() take for AttentionConfig.row_term: {} at the default "stored_o" (the calls are then the
        calls of every run before the key), {"row_term": "pdp"} otherwise (ced/kernels/attn_row_term.py)."""
        return {} if self.cfg.row_term == "stored_o" else {"row_term": self.cfg.row_term}

    def _row_term_unbuilt(self, x: torch.Tensor) -> bool:
        """Whether a slot read of this layer (read()) would train through the compiled flex kernel with AttentionConfig.row_term 'pdp' set: a
        read the option is not built for (its backward would form its row term from the stored output without saying so). CUDA training
        only: nobody trains a slot read on the CPU, and the CPU tests run it in float64, where the option is inert."""
        return (self.cfg.row_term != "stored_o" and self.cfg.backend == "flex" and x.is_cuda and x.requires_grad
                and torch.is_grad_enabled())

    def _normed_queries(self, x: torch.Tensor, q_out: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(q, gate): the projected queries after QK-norm, before RoPE and the temperature, and the output gate.
        q_out: q_proj(x) when the caller has it (model.fused_projections "qkv"); None: computed here, as always."""
        B, L, _ = x.shape
        hq, hd = self.cfg.n_q_heads, self.cfg.head_dim
        gate = None
        if self.cfg.gate == "elementwise":
            q, gate = (self.q_proj(x) if q_out is None else q_out).view(B, L, hq, 2 * hd).chunk(2, dim=-1)
        else:
            q = (self.q_proj(x) if q_out is None else q_out).view(B, L, hq, hd)
            if self.cfg.gate == "headwise":
                gate = self.g_proj(x).view(B, L, hq, 1)
        return self.q_norm(q), gate

    def _project_queries(self, x: torch.Tensor, pos: torch.Tensor, n_keys: torch.Tensor | None = None,
                         q_out: torch.Tensor | None = None):
        """(q, gate, scale_info): scale_info is asentmax's (c, beta, gamma), for the training statistics; None
        for softmax. n_keys: see query_factor. q_out: see _normed_queries."""
        assert self.rope_cap() is None, \
            "this layer has a rotary distance cap (rope_max_distance): its queries come in two rotations " \
            "(capped_queries); a caller of queries() would run without the cap"
        q, gate = self._normed_queries(x, q_out)
        q = apply_rope(q, pos, self.cfg.rope_fraction, self.cfg.rope_theta)
        # The temperature (and the diagnostics' override) as a per-position, per-head factor on q, in at least
        # float32, before any attention kernel; skipped entirely for "fixed" without an override.
        f = self.query_factor(pos, compute_dtype(q), n_keys)
        if self.cfg.normalizer == "asentmax":
            # ASEntmax's c (module docstring) at the same place, with any diagnostics override (fixed temperature)
            c, beta, gamma = self.asentmax_scale(x, pos)
            scale = c[..., None] if f is None else c[..., None] * f.to(c.dtype)
            return (q.to(scale.dtype) * scale).to(q.dtype), gate, (c, beta, gamma)
        if f is not None:
            q = (q.to(f.dtype) * f).to(q.dtype)
        return q, gate, None

    def _pointer_queries(self, x: torch.Tensor, pos: torch.Tensor, q_out: torch.Tensor | None = None
                         ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """(q, gate, q_plain) of the Copy-Paste pointer's shared layer (cp_query; the softmax normaliser, no rotary cap):
        q and gate are queries(x, pos)'s, computed by _project_queries' statements in its order (the same values bit for
        bit: tests/test_cp_mix_shared.py), and q_plain is q before the temperature (after QK-norm and RoPE, which a NoPE
        layer does not rotate). q_out: see _normed_queries."""
        assert self.rope_cap() is None and self.cfg.normalizer == "softmax", \
            "the pointer's shared head is a softmax layer without a rotary cap (cp_head.shared_layer_index)"
        q, gate = self._normed_queries(x, q_out)
        q = apply_rope(q, pos, self.cfg.rope_fraction, self.cfg.rope_theta)
        q_plain = q
        f = self.query_factor(pos, compute_dtype(q))
        if f is not None:
            q = (q.to(f.dtype) * f).to(q.dtype)
        return q, gate, q_plain

    def keys_values(self, x: torch.Tensor, pos: torch.Tensor, kv_out: tuple | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """kv_out: (k_proj(x), v_proj(x)) when the caller has them (model.fused_projections "qkv"); None: computed here.
        A grid layer (kv_sharing "grid") returns the ASSEMBLED per-head keys and values (B, L, n_q_heads, head_dim)."""
        assert self.rope_cap() is None, \
            "this layer has a rotary distance cap (rope_max_distance): its keys are kept in two forms " \
            "(capped_keys_values); a caller of keys_values() would run without the cap"
        if self.grid is not None:
            assert kv_out is None, "the grid layer's projections are not fused (control.check_kv_sharing)"
            k_raw, v_raw = self._grid_raw(x)
            k, _ = self._grid_keys(k_raw, pos)
            return k, assemble_grid(v_raw, self.grid)
        B, L, _ = x.shape
        hkv, hd = self.cfg.n_kv_heads, self.cfg.head_dim
        k = self.k_norm((self.k_proj(x) if kv_out is None else kv_out[0]).view(B, L, hkv, hd))
        k = apply_rope(k, pos, self.cfg.rope_fraction, self.cfg.rope_theta)
        v = (self.v_proj(x) if kv_out is None else kv_out[1]).view(B, L, hkv, hd)
        return k, v

    def _rotate(self, x: torch.Tensor, pos: torch.Tensor, rotation: "CappedRotation | None", far: bool) -> torch.Tensor:
        """x (normed queries or keys) rotated at its positions pos, or with far at the one position C. With the caller's
        tables (checked against pos, the cap and the dtype): rope.rotate_with_tables. Without: rope.apply_rope, the
        code of every capped read before the tables were shared, unchanged."""
        C = self.rope_cap()
        if rotation is None:
            at = pos.new_full((1, 1), C) if far else pos
            return apply_rope(x, at, self.cfg.rope_fraction, self.cfg.rope_theta)
        rotation.check(pos, C, compute_dtype(x), rotary=(x.shape[-1], self.cfg.rope_fraction, self.cfg.rope_theta))
        return rotate_with_tables(x, rotation.far_cos, rotation.far_sin) if far else rotate_with_tables(x, rotation.cos, rotation.sin)

    def capped_queries(self, x: torch.Tensor, pos: torch.Tensor, n_keys: torch.Tensor | None = None,
                       rotation: "CappedRotation | None" = None, factor_pos: torch.Tensor | None = None,
                       q_out: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """(q, q_far, gate) under the rotary distance cap C (the module docstring): q is what `queries` gives without
        the cap, the normed query rotated at its document position; q_far is the same normed query rotated by C, one
        constant angle for every position, which scores the keys more than C back. The temperature (and a
        query_scale override) multiplies both, so a key's logit is scaled the same on either side of the cap.
        rotation: the rotation tables of these positions built by the caller (CappedRotation; the encoder builds them
        once per forward for all its global layers); None: rope.apply_rope computes them here, as before.
        factor_pos: the document positions the temperature reads (query_factor), when they are not the rotation's
        `pos` (TST's token-scale rotary rotates at s x pos and counts keys in positions); None: pos, as before.
        q_out: see _normed_queries."""
        C = self.rope_cap()
        assert C is not None, "capped_queries is for a layer with rope_max_distance"
        qn, gate = self._normed_queries(x, q_out)
        q = self._rotate(qn, pos, rotation, far=False)
        q_far = self._rotate(qn, pos, rotation, far=True)
        f = self.query_factor(pos if factor_pos is None else factor_pos, compute_dtype(q), n_keys)
        if f is not None:
            q = (q.to(f.dtype) * f).to(q.dtype)
            q_far = (q_far.to(f.dtype) * f).to(q_far.dtype)
        return q, q_far, gate

    def capped_keys_values(self, x: torch.Tensor, pos: torch.Tensor, rotation: "CappedRotation | None" = None,
                           kv_out: tuple | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(k, k_far, v) under the rotary distance cap: k is what `keys_values` gives without the cap (rotated at
        the key's document position); k_far is the same normed key before rotation (KeySet.k_far).
        rotation: as capped_queries. kv_out: see keys_values (a grid layer returns the assembled per-head k, k_far and v)."""
        assert self.rope_cap() is not None, "capped_keys_values is for a layer with rope_max_distance"
        if self.grid is not None:
            assert kv_out is None, "the grid layer's projections are not fused (control.check_kv_sharing)"
            k_raw, v_raw = self._grid_raw(x)
            k, k_far = self._grid_keys(k_raw, pos, rotation, capped=True)
            return k, k_far, assemble_grid(v_raw, self.grid)
        B, L, _ = x.shape
        hkv, hd = self.cfg.n_kv_heads, self.cfg.head_dim
        k_far = self.k_norm((self.k_proj(x) if kv_out is None else kv_out[0]).view(B, L, hkv, hd))
        k = self._rotate(k_far, pos, rotation, far=False)
        v = (self.v_proj(x) if kv_out is None else kv_out[1]).view(B, L, hkv, hd)
        return k, k_far, v

    def _grid_raw(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(k, v) of a grid layer in the CACHE form: the projected key and value components, RAW (no norm, no rotary), each
        (B, L, 1, g.width): the rows' components, then the columns', then with a third axis the depth components (the module docstring)."""
        B, L, _ = x.shape
        return self.k_proj(x).view(B, L, 1, self.grid.width), self.v_proj(x).view(B, L, 1, self.grid.width)

    def _grid_keys(self, k_raw: torch.Tensor, pos: torch.Tensor, rotation: "CappedRotation | None" = None, capped: bool = False
                   ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(k, k_far) per-head keys (B, S, n_q_heads, head_dim) of a grid layer from raw key components k_raw (B, S, 1, g.width) at the
        entries' rotary positions pos (B, S): the cell's [row i ; column j] assembled, normed over the whole head by the ONE k_norm, then
        rotated. These are the statements of the gqa layer's keys_values (not capped) and capped_keys_values (capped), applied to the
        assembled head: k_norm, then apply_rope; or under the cap k_far = the normed key, k = _rotate(k_far) (by the caller's tables when
        `rotation` is given: _rotate checks them against pos). k_far is None when not capped."""
        kn = self.k_norm(assemble_grid(k_raw, self.grid))
        if capped:
            return self._rotate(kn, pos, rotation, far=False), kn
        return apply_rope(kn, pos, self.cfg.rope_fraction, self.cfg.rope_theta), None

    def _grid_read(self, keys: KeySet, rotation: "CappedRotation | None" = None, capped: bool = False) -> KeySet:
        """The key set a grid layer's attention reads, from its key set of raw components (the cache cat this chunk): per-head keys
        (assembled, normed, rotated at keys.pos: _grid_keys), the assembled values, the same metadata. The component key set stays the
        layer's cache. rotation (capped only): the rotation tables of keys.pos (CappedRotation; checked against it by _rotate), or None."""
        assert keys.pos is not None, "a grid layer's key set carries the rotary position of every entry (KeySet.pos)"
        k, k_far = self._grid_keys(keys.k, keys.pos, rotation, capped)
        return KeySet(k=k, v=assemble_grid(keys.v, self.grid), idx=keys.idx, doc=keys.doc, ok=keys.ok, k_far=k_far)

    def _forward_grid(self, x: torch.Tensor, ctx: SeqContext, cache: KeySet | None) -> tuple[torch.Tensor, KeySet]:
        """forward() of a grid layer (kv_sharing "grid"), the three cases of forward() in one place: the capped read (rotary distance cap),
        its TST token-scale form, and the plain read (a NoPE layer of the by-layer mix, or rotary without a cap). The queries are the gqa
        layer's own. The layer's cache is the KeySet of RAW components with their rotary positions (this chunk's cat the cache's); the read
        assembles, norms and rotates the per-head keys from it at every entry's own position (_grid_read) and runs the same attention
        functions as the gqa layer's read, with n_kv_heads = n_q_heads, on the same shared masks. The rotation tables of the context are
        the chunk's: the queries use them as always, and the keys of a read without a cache (keys.pos is the chunk's positions); with a
        cache the keys rotate with tables of their own positions. Returns (output, the key set of components)."""
        cap = self.rope_cap()
        ok = torch.ones_like(ctx.doc, dtype=torch.bool)
        if cap is None:
            q, gate = self.queries(x, ctx.pos)
            k, v = self._grid_raw(x)
            keys = KeySet.cat(cache, KeySet(k=k, v=v, idx=ctx.idx, doc=ctx.doc, ok=ok, pos=ctx.pos))
            read = self._grid_read(keys)
            if self.cfg.backend == "reference":
                o = attend(q, read, allowed(read, ctx.idx, ctx.doc))
            else:
                from ced.model.fast_attention import attend_fast

                o = attend_fast(q, read, ctx.idx, ctx.doc, backend=self.cfg.backend, **self._tile_kw(), **self._row_term_kw())
            return self.output(o, gate), keys
        scaled = ctx.rope_scale != 1  # TST's token-scale rotary: a training forward without a cache, the near / far split at cap // rope_scale
        assert not scaled or cache is None, "token-scale rotary (TST) is a training forward without a cache"
        rotation = ctx.memo.get(CAPPED_ROTATION_KEY)
        rot_pos = rotary_positions(ctx) if scaled else ctx.pos
        q, q_far, gate = self.capped_queries(x, rot_pos, rotation=rotation, factor_pos=ctx.pos if scaled else None)
        k, v = self._grid_raw(x)
        keys = KeySet.cat(cache, KeySet(k=k, v=v, idx=ctx.idx, doc=ctx.doc, ok=ok, pos=rot_pos))
        read_rotation = rotation
        if cache is not None and rotation is not None:
            # the context's tables are this chunk's; the cached entries rotate at their own positions with tables built the same way
            # (rope_tables, in the dtype the context's tables hold), so a cached read rotates as the uncached read of the same row does
            read_rotation = capped_rotation(keys.pos, self.cfg, rotation.cos.dtype)
        read = self._grid_read(keys, read_rotation, capped=True)
        masks = ctx.memo.get(CAPPED_MASKS_KEY) if (cache is None and self.cfg.backend == "flex") else None
        o = capped_attention(q, q_far, read, ctx.idx, ctx.doc, ctx.pos, mask_cap(cap, ctx.rope_scale) if scaled else cap, self.cfg.backend,
                             block_masks=masks, **self._tile_kw(), **self._row_term_kw())
        return self.output(o, gate), keys

    def _fa3(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx: SeqContext, cache: KeySet | None,
             q_far: torch.Tensor | None = None, k_far: torch.Tensor | None = None, cap: int | None = None) -> torch.Tensor:
        """Backends "fa3" and "fa4": this layer's own read over its own keys (no cache) with the forward's varlen
        metadata (SeqContext.memo[FLASH_VARLEN_KEY], built by Encoder.share_flash_varlen). cap None: the plain read; else
        the capped read with q_far, k_far (ced/kernels/flash_capped.py flash_attention; on "fa4" ced/kernels/flash_fa4.py
        flash_attention, with AttentionConfig.fa4_deterministic). Anything else is refused by name."""
        not_covered = flash_not_covered(self.cfg.backend)
        if cache is not None:
            raise ValueError(f"{not_covered}: not a forward with a cache (prefill + steps, cached generation, chunked "
                             "scoring); score with backend 'flex', 'sdpa' or 'reference' (an implementation field: the "
                             "same weights)")
        meta = ctx.memo.get(FLASH_VARLEN_KEY)
        if meta is None:
            raise ValueError(f"{not_covered}: this call has no varlen metadata in its context (a layer called outside "
                             "Encoder.forward: the MTP block, the second pass, a looped middle, a diagnostic)")
        if self.cfg.backend == "fa4":
            from ced.kernels.flash_fa4 import flash_attention as fa4_attention

            return fa4_attention(q, k, v, meta, q_far, k_far, cap, deterministic=self.cfg.fa4_deterministic)
        from ced.kernels.flash_capped import flash_attention

        return flash_attention(q, k, v, meta, q_far, k_far, cap)

    def output(self, o: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
        if gate is not None and getattr(self, "mx_attn_gate", False) and self.training:  # model.fp8.chains attn_gate
            from ced.model import mx_wire

            return mx_wire.gate_proj(self, o, gate)
        if gate is not None:
            o = o * torch.sigmoid(gate)
        return self.o_proj(o.flatten(2))

    def forward(
        self, x: torch.Tensor, ctx: SeqContext, cache: KeySet | None = None
    ) -> tuple[torch.Tensor, KeySet]:
        """Causal self-attention over this chunk plus the cached keys of earlier chunks.

        Returns the output and the key set including this chunk (the cache for the next chunk).
        """
        if self.grid is not None:  # grid KV sharing: the components are the cache, the heads are assembled for the read
            return self._forward_grid(x, ctx, cache)
        cap = self.rope_cap()
        if cap is not None and ctx.rope_scale != 1:  # TST's token-scale rotary (TSTConfig.token_scale_rope)
            return self._forward_token_scale(x, ctx, cache, cap)
        # model.fused_projections "qkv": q_proj, k_proj and v_proj as one GEMM in a training forward (None, None: the
        # separate projections below, as always)
        qo, kvo = self._fused_outputs(x)
        if cap is not None:  # the rotary distance cap (module docstring); the key set carries the unrotated keys
            # the rotation tables of this context's positions, when the caller built them once for every global
            # layer of this forward (Encoder.share_capped_rotation); without them apply_rope computes them, as before
            rotation = ctx.memo.get(CAPPED_ROTATION_KEY)
            q, q_far, gate = self.capped_queries(x, ctx.pos, rotation=rotation, q_out=qo)
            k, k_far, v = self.capped_keys_values(x, ctx.pos, rotation=rotation, kv_out=kvo)
            ok = torch.ones_like(ctx.doc, dtype=torch.bool)
            keys = KeySet.cat(cache, KeySet(k=k, v=v, idx=ctx.idx, doc=ctx.doc, ok=ok, k_far=k_far))
            if self.trunk_copy:  # (model.cp_head.trunk_copy): this chunk's queries, both rotations, as the read below takes them
                keys = KeySet(k=keys.k, v=keys.v, idx=keys.idx, doc=keys.doc, ok=keys.ok, k_far=keys.k_far, tc_q=q, tc_q_far=q_far)
            # The read's two flex BlockMasks, when the caller built them once for every global layer of this forward
            # (Encoder.share_capped_masks puts them in the context's per-forward memo; they are this read's: the
            # layer's own keys without a cache, i.e. this context's positions). Without them, as before, the read
            # builds its own. The key set with a cache is longer than the context: never the shared masks.
            if self.cfg.backend in FLASH_BACKENDS:  # FlashAttention-3 / -4: the near and far calls merged (flash_capped)
                o = self._fa3(q, k, v, ctx, cache, q_far, k_far, cap)
                return self.output(o, gate), keys
            masks = ctx.memo.get(CAPPED_MASKS_KEY) if cache is None and self.cfg.backend == "flex" else None
            if self.trunk_copy and self.trunk_lse and cache is None and self.cfg.backend == "flex":
                # (model.cp_head.kernel "fused" / "fused2"): the same read, with its log-sum-exp for the fused copy kernel
                o, lse = capped_attention(q, q_far, keys, ctx.idx, ctx.doc, ctx.pos, cap, self.cfg.backend, block_masks=masks,
                                          return_lse=True, **self._tile_kw(), **self._row_term_kw())
                keys = KeySet(k=keys.k, v=keys.v, idx=keys.idx, doc=keys.doc, ok=keys.ok, k_far=keys.k_far, tc_q=keys.tc_q,
                              tc_q_far=keys.tc_q_far, tc_lse=lse)
                return self.output(o, gate), keys
            o = capped_attention(q, q_far, keys, ctx.idx, ctx.doc, ctx.pos, cap, self.cfg.backend, block_masks=masks,
                                 **self._tile_kw(), **self._row_term_kw())
            return self.output(o, gate), keys
        if self.cfg.normalizer == "asentmax":
            q, gate, info = self._project_queries(x, ctx.pos)
        elif self.cp_query is not None:  # the Copy-Paste pointer's shared head: the same q, and q before the temperature
            q, gate, q_plain = self._pointer_queries(x, ctx.pos, q_out=qo)
        else:
            q, gate = self.queries(x, ctx.pos, q_out=qo)
        k, v = self.keys_values(x, ctx.pos, kv_out=kvo)
        ok = torch.ones_like(ctx.doc, dtype=torch.bool)
        keys = KeySet.cat(cache, KeySet(k=k, v=v, idx=ctx.idx, doc=ctx.doc, ok=ok))
        if self.cp_query is not None:  # (model.cp_head.pointer_input "shared"): this chunk's head query and key
            h, inherit = self.cp_query
            kv = h // (self.cfg.n_q_heads // self.cfg.n_kv_heads)  # the key head GQA assigns query head h
            keys = KeySet(k=keys.k, v=keys.v, idx=keys.idx, doc=keys.doc, ok=keys.ok, k_far=keys.k_far,
                          cp_q=(q if inherit else q_plain)[:, :, h], cp_k=k[:, :, kv])
        if self.trunk_copy:  # (model.cp_head.trunk_copy): this chunk's queries as the read below takes them
            keys = KeySet(k=keys.k, v=keys.v, idx=keys.idx, doc=keys.doc, ok=keys.ok, k_far=keys.k_far, tc_q=q)
        if self.cfg.normalizer == "asentmax":
            o = entmax_attention(q, keys, ctx.idx, ctx.doc, False, self.cfg.asentmax_chunk, self.entmax_stats, info)
        elif self.cfg.backend == "reference":
            o = attend(q, keys, allowed(keys, ctx.idx, ctx.doc))
        elif self.cfg.backend in FLASH_BACKENDS:  # FlashAttention-3 / -4, one causal varlen call (ced/kernels/flash_capped.py)
            o = self._fa3(q, k, v, ctx, cache)
        elif self.trunk_copy and self.trunk_lse and cache is None and self.cfg.backend == "flex":
            # (model.cp_head.kernel "fused" / "fused2"): the same read, with its log-sum-exp for the fused copy kernel
            from ced.model.fast_attention import attend_fast

            o, lse = attend_fast(q, keys, ctx.idx, ctx.doc, backend=self.cfg.backend, return_lse=True, **self._tile_kw(), **self._row_term_kw())
            keys = KeySet(k=keys.k, v=keys.v, idx=keys.idx, doc=keys.doc, ok=keys.ok, k_far=keys.k_far, tc_q=keys.tc_q, tc_lse=lse)
        else:
            from ced.model.fast_attention import attend_fast

            o = attend_fast(q, keys, ctx.idx, ctx.doc, backend=self.cfg.backend, **self._tile_kw(), **self._row_term_kw())
        return self.output(o, gate), keys

    def _forward_token_scale(self, x: torch.Tensor, ctx: SeqContext, cache: KeySet | None, cap: int
                             ) -> tuple[torch.Tensor, KeySet]:
        """The capped read under Token Superposition Training's token-scale rotary (ctx.rope_scale = s > 1;
        TSTConfig.token_scale_rope). A position of the row is a bag of s tokens; the rotary angle is read in tokens and
        the cap C (rope_max_distance) stays in tokens:

            s d <= C:  logit = R(s pos_q) q . R(s pos_k) k     the plain logit, both rotated at s x their positions
            s d >  C:  logit = R(C) q . k                       the query rotated by C tokens against the unrotated key

        d = q_idx - k_idx in positions (bags). So the near / far split is the plain cap's at C // s positions
        (mask_cap), and the rotation is the plain cap's at the positions s x pos with the far rotation at C. The
        temperature (Scalable-Softmax) counts keys, so it reads the positions themselves (pos + 1 keys), not s x pos.
        Training forwards without a cache only (HybridLM.forward refuses the rest)."""
        assert cache is None, "token-scale rotary (TST) is a training forward without a cache"
        rot_pos = rotary_positions(ctx)
        rotation = ctx.memo.get(CAPPED_ROTATION_KEY)
        qo, kvo = self._fused_outputs(x)  # model.fused_projections "qkv" (forward)
        q, q_far, gate = self.capped_queries(x, rot_pos, rotation=rotation, factor_pos=ctx.pos, q_out=qo)
        k, k_far, v = self.capped_keys_values(x, rot_pos, rotation=rotation, kv_out=kvo)
        ok = torch.ones_like(ctx.doc, dtype=torch.bool)
        keys = KeySet(k=k, v=v, idx=ctx.idx, doc=ctx.doc, ok=ok, k_far=k_far)
        if self.cfg.backend in FLASH_BACKENDS:  # the near / far split at cap // rope_scale positions, as below
            return self.output(self._fa3(q, k, v, ctx, cache, q_far, k_far, mask_cap(cap, ctx.rope_scale)), gate), keys
        masks = ctx.memo.get(CAPPED_MASKS_KEY) if self.cfg.backend == "flex" else None
        o = capped_attention(q, q_far, keys, ctx.idx, ctx.doc, ctx.pos, mask_cap(cap, ctx.rope_scale), self.cfg.backend,
                             block_masks=masks, **self._tile_kw(), **self._row_term_kw())
        return self.output(o, gate), keys

    def read(self, x: torch.Tensor, pos: torch.Tensor, idx: torch.Tensor, doc: torch.Tensor, keys: KeySet,
             block_mask=None, n_keys: torch.Tensor | None = None, allow: torch.Tensor | None = None) -> torch.Tensor:
        """Queries from x (B, C, d) at slots with document-relative positions pos, absolute indices
        idx and documents doc (B, C), over a key set the slots do not write (attend_slots).
        Returns the gated, projected output (B, C, d). block_mask: see attend_slots (flex, softmax only).
        allow: see attend_slots (a caller-built dense mask; reference and sdpa, softmax without the rotary cap).
        n_keys (B, C): the exact number of keys each slot sees, for ssmax where that is not pos + 1 (query_factor;
        CED v2's full decoder layers with self-KV).
        Under the rotary distance cap `keys` must carry the unrotated keys (KeySet.k_far: decoder.MemoryKV writes
        them under the same config), and a caller-built block_mask is refused (the capped read builds its own two)."""
        if self.grid is not None:
            raise ValueError("attention.kv_sharing 'grid' is built for the encoder's own read of a row (GatedAttention.forward, with or without a "
                             "cache), not for a slot read (GatedAttention.read: a decoder, the second pass): the key set it would read holds "
                             "components, not per-head keys")
        if self.cfg.backend in FLASH_BACKENDS:
            raise ValueError(f"{flash_not_covered(self.cfg.backend)}: not a slot read (GatedAttention.read: CED v2's "
                             "decoder, the second pass, v1's decoder); set backend 'flex', 'sdpa' or 'reference' for it")
        if self._row_term_unbuilt(x):
            raise ValueError("attention.row_term 'pdp' is built for the encoder's own read of a row (GatedAttention.forward without a "
                             "cache), not for a slot read (GatedAttention.read: a decoder, the second pass): its backward would form "
                             "its row term from the stored output without saying so")
        cap = self.rope_cap()
        assert allow is None or (cap is None and self.cfg.normalizer == "softmax"), \
            "a caller-built dense mask is for the plain softmax read (no rotary distance cap, no asentmax)"
        if cap is not None:
            assert block_mask is None, "the rotary distance cap builds its own block masks: no caller-built block_mask"
            assert keys.k_far is not None, \
                "this layer has a rotary distance cap (rope_max_distance) but the key set it reads was built " \
                "without the unrotated keys (KeySet.k_far): the writer of these keys does not implement the cap"
            q, q_far, gate = self.capped_queries(x, pos, n_keys)
            o = capped_attention(q, q_far, keys, idx, doc, pos, cap, self.cfg.backend, empty_reads_first=True)
            return self.output(o, gate)
        if self.cfg.normalizer == "asentmax":  # the slot rule: an empty slot reads key 0
            assert block_mask is None, "asentmax runs the reference entmax attention: no flex block_mask"
            assert n_keys is None, "asentmax reads ln(pos + 2), not a key count"
            q, gate, info = self._project_queries(x, pos)
            return self.output(entmax_attention(q, keys, idx, doc, True, self.cfg.asentmax_chunk, self.entmax_stats, info), gate)
        q, gate = self.queries(x, pos) if n_keys is None else self.queries(x, pos, n_keys)
        if allow is not None:
            return self.output(attend_slots(q, keys, idx, doc, self.cfg.backend, block_mask=block_mask, allow=allow), gate)
        return self.output(attend_slots(q, keys, idx, doc, self.cfg.backend, block_mask=block_mask), gate)
