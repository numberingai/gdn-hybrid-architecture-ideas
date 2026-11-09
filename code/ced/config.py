"""Configuration dataclasses for the model and the training loop.

Every dimension is a config value. Nothing about the full-size model is hard-coded in the
model code: the same classes build the tiny CPU test model and the ~3.8B model.

Configs round-trip through plain dicts (and YAML) with `to_dict` / `from_dict`.
"""

from __future__ import annotations

import dataclasses
import math
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# AttentionConfig.flex_tiles: the order in which flex visits a query block's key tiles (ced/model/fast_attention.py fold_full_blocks)
FLEX_TILE_MODES = ("classified", "indexed")


@dataclass
class AttentionConfig:
    """Softmax attention shape, shared by the encoder's global layers and the decoder."""

    n_q_heads: int = 16
    n_kv_heads: int = 2
    head_dim: int = 256
    # Sigmoid output gate on the attention output before o_proj (arXiv 2505.06708).
    # "elementwise": one gate per output element (the spec's ~89M per layer implies this).
    # "headwise": one gate per head. "none": no gate.
    gate: str = "elementwise"
    qk_norm: bool = True
    # Fraction of each head's dimensions that get rotary embeddings. 0.0 means NoPE.
    rope_fraction: float = 0.25
    rope_theta: float = 1e7
    # Distance cap of the rotary part (also the scoring scripts' --rope-clamp, set on a loaded checkpoint). None:
    # plain rotary, every run before 2026-10-11 (no parameter either way: the state dict, outputs and RNG use are
    # unchanged). An integer C >= 1: the rotary part of a logit uses the relative distance min(pos_q - pos_k, C) for
    # a key of the query's document at or before it: pos_q - pos_k <= C: q rotated at pos_q against k rotated at
    # pos_k (the plain logit); pos_q - pos_k > C: q rotated by C against the UNROTATED k (the plain logit of a key
    # exactly C back). The non-rotating dimensions, the mask, the normaliser and the temperature are untouched (the
    # temperature multiplies both rotations of q; ssmax's key count stays pos + 1). ced/model/attention.py
    # attend_capped is the reference, ced/model/fast_attention.py attend_capped_fast the flex / sdpa path. Without
    # rotary dimensions
    # (rope_fraction 0.0) there is nothing to cap and the value is inert. Built for arch "control" and for "ced_v2"
    # without decoder self-KV; refused with the asentmax normaliser and everywhere else (ced/model/control.py).
    rope_max_distance: int | None = None
    # "reference" (dense mask, ground truth), "flex" or "sdpa" (ced/model/fast_attention.py), or "fa3" (FlashAttention-3
    # varlen calls, the capped read as two calls with an exact log-sum-exp merge; ced/kernels/flash_capped.py). "fa3" is
    # built for the control's encoder global layers in the training forward without a cache (plain causal, capped
    # rotary, the NoPE mix, TST's token-scale rotary), on CUDA in bf16 / fp16, and refused everywhere else by name
    # (ced/model/control.py check_fa3). "fa4" (ced/kernels/flash_fa4.py): the same reads, merge and metadata as "fa3" on
    # FlashAttention-4 (the CuTe DSL; SM90 and SM100 / SM110), covered and refused as "fa3" is.
    backend: str = "reference"
    # backends "fa3" and "fa4" only: the capacity of their varlen metadata, in segments (documents and row starts) a row
    # of the micro-batch; B x this many segments a forward, empty segments padding it (fixed shapes: the kernels plan's
    # section 4, rule 1). A micro-batch with more runs at this doubled (flash_capped.capacity_for), never stops. Inert
    # on every other backend, and not refused there: the scorers switch a loaded config's backend and keep the rest. An
    # implementation field (ced/train/trainer.py IMPLEMENTATION_FIELDS), like backend.
    fa3_docs_per_row: int = 64
    # backend "fa4" only: FlashAttention-4's deterministic backward (its dQ / dK / dV semaphores on SM90; on SM100 / SM110
    # at head dimension 256 its dedicated backward, deterministic by construction per its source). true (the default):
    # deterministic, what TrainConfig.deterministic requires (the slices' bit-for-bit resume: refused by name otherwise,
    # ced/kernels/flash_fa4.py check_deterministic). false: the faster backward with atomic accumulation, for a run with
    # train.deterministic false. Inert on every other backend. An implementation field (the same function; only the
    # summation order of the gradients differs).
    fa4_deterministic: bool = True
    # backend "flex" only: the order in which a query block visits its key tiles. "classified" (the default, every run before
    # 2026-10-26, bit for bit): flex's own BlockMask, in which a tile whose 128 x 128 pairs are all allowed is a FULL tile; the
    # kernel visits the partial tiles first and the full tiles after them, so a row's accumulation order (an online softmax:
    # rounding differs with the order) depends on whether OTHER rows of its 128-row query block, possibly after it, start a new
    # document. "indexed": every tile that holds an allowed pair is visited as a partial tile, in ascending key-tile order
    # (fast_attention.fold_full_blocks), so a row's order and bits depend on the rows up to itself only; the document mask is
    # evaluated on every tile (a cost). Inert on every other backend (the scorers switch a loaded config's backend and keep the
    # rest). An implementation field (ced/train/trainer.py IMPLEMENTATION_FIELDS): the same function, a different rounding.
    flex_tiles: str = "classified"
    # The row term of the attention backward's softmax gradient dz = P (dP - delta). "stored_o" (the default, every run before the
    # key): delta = rowsum(dO. O) from the stored output, which the compiled flex kernel keeps in the query's dtype (bf16): a fixed
    # per-row bias dO. (O_stored - O) in delta that bites on peaky rows. "pdp": delta = rowsum(P. dP) from the same dP, as the
    # reference's autograd softmax backward forms it: on the compiled flex kernel (backend "flex" on CUDA) a pre-pass kernel
    # (ced/kernels/attn_row_term.py, attn_row_term_triton.py) forms that row sum in the backward and hands the difference to flex's own
    # backward through the log-sum-exp's gradient; the forward values and the parameters are unchanged. On backend "reference" the
    # backward is already that form (the key is inert); "sdpa" and the FlashAttention backends are refused by name
    # (ced/model/control.py check_row_term). Built for the encoder's global layers' own read (Encoder.forward without a cache); refused
    # on the decoder's attention. Not an implementation field: it changes the gradients (by the bias above), not the function. Absent
    # from a checkpoint's saved config at the default (ced/train/trainer.py SAVED_ATTENTION_DEFAULTS).
    row_term: str = "stored_o"
    # Softmax temperature on the logits q.k * head_dim^-0.5, applied by scaling the query after QK-norm and
    # RoPE (ced/model/attention.py, GatedAttention.queries), so every backend and cached decoding get it:
    # - "fixed": none (every run before 2026-10-09; no parameter, the state dict unchanged);
    # - "learned": tau_h = 1 + w_h, one zero-centred parameter per query head (temp_w, init 0; AdamW without
    #   weight decay).
    # - "ssmax": Scalable-Softmax (arXiv 2501.19399), tau_{t,h} = (1 + w_h) ln(n_t) / ln(temperature_ref_len)
    #   with n_t = pos_t + 1, the number of keys query t sees (causal, same document): s_h = (1 + w_h) /
    #   ln(ref) in the paper's notation, so tau = 1 at n = ref at init. CED v2's decoder with self-KV: its full
    #   layers count n_t exactly (pos_t + 1 memory entries plus the valid decoder slots of t's document at s <= t;
    #   ced/model/ced_v2.py slot_key_counts), its local layers take DecoderConfig.local_temperature. Refused where a
    #   query's key count is not counted (the diagnostics' key windows, a local decoder layer).
    temperature: str = "fixed"
    # ssmax's reference key count (tau = 1 + w_h there). Set per role: model.encoder.attention and
    # model.decoder.attention are separate configs (stage 2 + ssmax: 8192 in the encoder, 16384 in the decoder, where
    # the last query of an 8k row with the decoder everywhere sees 8192 memory entries plus 8192 own keys).
    temperature_ref_len: int = 8192
    # The normaliser over the logits (the three-variant attention experiment):
    # - "softmax": every run before 2026-10-09 (no parameter, the state dict, outputs and RNG use unchanged);
    # - "asentmax": ASEntmax (arXiv 2506.16640; "Our implementation choices"): alpha = 1.5 entmax, p = [z/2 -
    #   tau]_+^2, of the logits of the query scaled by c_{t,h} = 1 + softplus(x_t. w_b,h + b_b,h) * ln(pos_t + 2)^(2
    #   tanh(x_t. w_g,h + b_g,h)), x_t the attention input (after the pre-attention norm), applied to q after
    #   QK-norm and RoPE; beta_proj / gamma_proj (Linear(d_model, n_q_heads), bias, N(0, 0.02), bias 0) on AdamW
    #   without weight decay. The exact, slow reference (ced/model/entmax.py) on every path: `backend` is ignored.
    #   Refused with a non-fixed temperature, with CED v2's decoder self-KV and in v1's decoder.
    normalizer: str = "softmax"
    # asentmax: query rows per chunk of the reference (memory; results change at rounding level only). 0: auto,
    # as many rows as keep one (n_q_heads, rows, keys) float32 score tensor at 2^25 elements (entmax.chunk_rows).
    asentmax_chunk: int = 0
    # How the global attention layer's query heads share keys and values:
    # - "gqa" (the default, every run before this key): n_kv_heads key/value heads, query head h reads head h // (n_q_heads / n_kv_heads).
    #   The five grid keys below must be 0 (heads_per_cell 1). No parameter, no state-dict entry, no random number changes: the model is the model it was.
    # - "grid": the query heads are laid out on a grid_rows x grid_cols grid of CELLS, heads_per_cell consecutive query heads to a cell: head h
    #   sits in the cell c = h // heads_per_cell, which is the grid cell (i, j) = (c // grid_cols, c % grid_cols) (with heads_per_cell 1, head h =
    #   i * grid_cols + j; with 2 on a 2 x 2 grid of 8 heads, heads 0-1 are cell (0,0), 2-3 (0,1), 4-5 (1,0), 6-7 (1,1)). The layer projects
    #   and caches ONE key component and ONE value component per row (row_dim numbers each) and per column (col_dim numbers each):
    #   grid_rows * row_dim + grid_cols * col_dim numbers per token for the keys and the same for the values (k_proj, v_proj are
    #   Linear(d_model, that)). The heads of cell (i, j) read the concatenation [row i ; column j] as their key and as their value, so
    #   grid_rows + grid_cols components give grid_rows * grid_cols distinct keys. head_dim (the width of a query head, of the output gate
    #   and of a head's output) must equal row_dim + col_dim (so the head dimension is the layer's own, not narrowed by the grid); n_q_heads
    #   must equal grid_rows * grid_cols * heads_per_cell; n_kv_heads must be 1 (the cache is one "head" of components). The cache holds the
    #   RAW components (before any norm and before rotary) and each entry's rotary position, never the assembled heads; at read time each
    #   head's key [row i ; column j] is assembled, normed over the whole head_dim by ONE k_norm (the gqa layer's module and init), and rotated
    #   (the leading rotary_dims(head_dim, rope_fraction) dimensions) at the entry's own position, as a gqa head's key is, and the ordinary
    #   attention kernel runs on them (ced/model/attention.py GridSpec, assemble_grid). So the grid is the gqa layer with n_kv_heads =
    #   n_q_heads whose k_proj / v_proj rows are block-assembled, and differs from a GQA control of the same cache in the sharing alone.
    #   Built for arch "control" (control.check_kv_sharing).
    # - THREE AXES: grid_depth and depth_dim (both 0, the default: the two-axis grid above, exactly as it was) add a THIRD axis of grid_depth
    #   components of depth_dim numbers each. Then the cells are grid_rows x grid_cols x grid_depth (row-major, the third axis the fastest: with
    #   heads_per_cell P, head h is in the cell c = h // P = (i, j, k) = (c // (grid_cols grid_depth), (c // grid_depth) % grid_cols, c %
    #   grid_depth)), a head reads [row i; column j; depth k] as its key and as its value, head_dim must equal row_dim + col_dim + depth_dim,
    #   n_q_heads grid_rows x grid_cols x grid_depth x heads_per_cell, and k_proj / v_proj have grid_rows row_dim + grid_cols col_dim + grid_depth
    #   depth_dim outputs (rows', columns', then depth components). Both keys must be set together (both 0 or both >= 1) and are 0 under kv_sharing
    #   "gqa". Every other rule above is unchanged.
    kv_sharing: str = "gqa"
    grid_rows: int = 0
    grid_cols: int = 0
    row_dim: int = 0
    col_dim: int = 0
    heads_per_cell: int = 1
    grid_depth: int = 0
    depth_dim: int = 0


@dataclass
class GDNNegEigConfig:
    """Negative eigenvalues, the negative-eigenvalue form of the GDN memory-rule screen (ced/model/gdn_negeig.py).

    beta_max: the top of the write strength's range, beta = beta_max x sigmoid(in_proj_b(x)). 1 (the default; every run
    before this key): beta = sigmoid(b) in (0, 1), the layer as it was, bit for bit (the line is not changed). 2: beta =
    2 sigmoid(b) in (0, 2) (Grazzi et al. arXiv 2411.12537, Eq. 2): a step's transition alpha (I - beta k k^T) has the
    eigenvalue alpha (1 - beta) in (-alpha, alpha) along k (was (0, alpha)), a reflection at beta = 2. No parameter, no
    state-dict entry, no random number. It acts in GatedDeltaNet.gates and gates_from, before every kernel; every kernel
    takes beta as an input and none clamps it or assumes beta <= 1. Built for kernel "ced" with chunk_backend "triton"
    and edges "plain" (R1-entry's plain path; its GPU values check passed), and for the reference
    kernels "chunked" and "recurrence"; refused at build with kernel "fla", chunk_backend "flashqla" and edges other
    than "plain" (not checked at beta > 1), and in a TST bag phase with the write-strength power
    (tst.gdn_resolution_gates "decay_beta": its 1 - (1 - beta)^s is written for beta = sigmoid(b)). An int, 1 or 2 (YAML
    and --set give 2; a bool is refused)."""

    beta_max: int = 1


@dataclass
class KDAConfig:
    """Off by default (decay "scalar"): the layer as it was, bit for bit; the other keys are read only with decay "channel" (refused
    at their non-default values otherwise).

    decay "scalar": GDN's one log decay per value head and token (every model before this key). "channel": KDA (Kimi
                    Linear, arXiv 2510.26692, Eq. 1), S_t = (I - beta k k^T) Diag(a_t) S_{t-1} + beta k v^T with one log decay per
                    value head, key channel and token, g = -exp(A_log) softplus(u + dt_bias), u = the low-rank decay projection of x
                    (fla 0.5.2's KimiDeltaAttention gate). It replaces in_proj_a and the per-head dt_bias by that projection and a
                    per-channel dt_bias (H_v d_k), flat as fla's; A_log stays one per value head. proj the decay projection's form
                    (both are built): "shared" (the default; Kimi Linear / fla's f_proj: d -> rank -> H_v d_k, one bottleneck for
                    every head) or "per_head" (the EDA paper's "rank-16 (per-head)" gate generator, arXiv 2606.26560 Appendix A: d ->
                    rank x H_v, then rank -> d_k for each value head on its own rank slice). rank the bottleneck; 0 is the form's
                    own: shared v_head_dim (fla's head_v_dim, 128 here), per_head 16 (EDA's). floor_channels the horizon floor: the
                    first n key channels of every value head take g = ln(1 - 1/H_i) sigmoid(u + b) (b: those channels' entries of
                    dt_bias, zeros at the start), H_i log-spaced from floor_min to floor_max tokens, so a >= 1 - 1/H_i and the
                    horizon 1 / (1 - a) >= H_i at every token; the other channels keep KDA's gate. 0 (the default): no floor. The
                    floor is a buffer, never a weight. floor_min, floor_max the floored channels' horizons in tokens (the survey's
                    proposal: 64 and 262,144). kernel the fused path's kernel (CUDA training forward, GDNConfig.kernel "fla" /
                    "ced"): "chunk_kda" (the default; every run before this key): fla 0.5.2's chunk_kda, on the 2T interleave when
                    GDNConfig.erase is on, the layer as it was. "k1": our per-channel-decay multi-row WY chunk kernel
                    (ced/kernels/gdn_k1.py): r = 2 rows a token with the erase (both rank-1 updates of a token in one WY form, no
                    interleave), r = 1 without (they release on chunk_kda at 60M). Read only with decay "channel"; the reference
                    paths (recurrence, chunked, the CPU) ignore it. "k2" (with decay "scalar"): our scalar-decay multi-row WY chunk
                    kernel (ced/kernels/gdn_k2.py) for the layers on GDN's decay with more than one rank-1 update a token: the erase
                    (r = 2) or Gated DeltaProduct (r = dprod.n_h, at most 3), every row of a token in one WY form, no interleave;
                    refused by name on any other scalar-decay layer (gdn_kda.check_k2). The default there is today's route (the 2T or
                    n_h T interleave into our scalar kernel)."""

    decay: str = "scalar"
    proj: str = "shared"
    rank: int = 0
    floor_channels: int = 0
    floor_min: float = 64.0
    floor_max: float = 262144.0
    kernel: str = "chunk_kda"


@dataclass
class EraseConfig:
    """Off by default (mode "none"): the layer as it was, bit for bit; rank is read only with mode "eda" (refused at another value
    otherwise).

    mode   "none": no erase (every model before this key). "eda": EDA (arXiv 2606.26560, Eq. 8), S_t = (I - beta k k^T)
           (I - gamma e e^T) D_t S_{t-1} + beta k v^T: after the layer's decay D_t (GDN's scalar, or KDA's per-channel decay with
           its floor when GDNConfig.kda is on) and before the delta write, the state is erased along a unit direction e_t of its
           own (one per key head, from a low-rank projection of x) with strength gamma_t in [0, 1] (one per value head:
           c(s) sigmoid(W_gamma x), s zero-initialised, c the inward straight-through clamp of ced/model/gdn_erase.py).
    rank   the erase-address projection's bottleneck per key head (d -> rank x H_k, then rank -> d_k on each key head's own slice):
           16, the EDA paper's "rank-16 (per-head) erase-address projection" (its Appendix A, read by the survey)."""

    mode: str = "none"
    rank: int = 16


@dataclass
class DeltaProductConfig:
    """Gated DeltaProduct for a GDN layer (GDNConfig.dprod; ced/model/gdn_dprod.py). Siems et al. arXiv 2502.10297, Eq. 3: n_h delta-rule
    (generalised Householder) steps a token, each with its own key, value and beta, the decay once a token (before the first step),
    the output read after the last step. Every step's beta = M sigmoid(b), M = 2 by default (beta in (0, 2); gdn_dprod.BETA_MAX);
    beta_max 1: M = 1 (below).

    n_h: 1 (the default; every model before this key) is Gated DeltaNet, the layer as it was: no tensor, no operation, no random
    number. n_h >= 2: step 1 is the layer's own (in_proj_qkv's k and v, in_proj_b's beta); steps 2 .. n_h read n_h - 1 more key / value
    projections with their own short conv (dprod.in_proj_kv, dprod.conv_weight) and beta projections (dprod.in_proj_b). The arm's value
    is 2. Built for arch "control"'s encoder GDN layers only (check_config).

    route: how the FUSED path (CUDA, kernel "ced", the training and scoring forward from an empty state) reads the n_h = 2 steps out.
    "interleave" (the default; every run before this key): our chunk kernels on the 2T interleave of the two steps, the readout done
    on both sub-steps of a token and kept on the second (ced/model/gdn_dprod.py interleave()). "one-readout": the same chunk kernels
    with the readout done ONCE a token, q and g read at the token rows and no interleave copy of them (ced/kernels/gdn_1r.py). The
    same function; read only with n_h 2 and kernel "ced" (refused by name otherwise, gdn_dprod.build / check_config); the reference
    paths (recurrence, chunked, the CPU, a carried state) ignore it. The route reads every step's beta as given (dprod_forward forms
    it, beta_max and beta_init included), so beta_max and beta_init compose with it.

    PLACEMENT (the three keys below): which GDN layers and which value heads of a layer carry the extra steps. Read only with
    model.encoder.gdn.dprod.n_h >= 2 (a non-default value with n_h 1 is refused by name: it would be a silent no-op). The
    defaults (layers (), heads 0, heads_by_layer {}) are every value head of every encoder GDN layer, the Gated DeltaProduct
    form as built, bit for bit. A layer that does not carry has no `dprod` module and is R1-entry's layer; a layer that carries
    some of its heads runs those through the n_h path and the others through the plain path (two kernel calls, the outputs
    joined along the head axis before the gated norm), so a head that does not carry is Gated DeltaNet's head exactly (beta =
    sigmoid(b), one step), not an n_h head with beta 0. The kernels take whole key groups only (a key head serves H_v / H_k
    value heads), so a set of heads is a union of key groups.

    layers          the GDN ordinals (0-based among the encoder's GDN layers: 0 .. 14 at 60M, 0 .. 29 at 4B), distinct and
                    ascending, that carry the extra steps; () = every encoder GDN layer. Always a tuple here.
    heads           0 = every value head of a carrying layer. m > 0: the first m value heads of every carrying layer, m a
                    multiple of H_v / H_k (whole key groups) and at most H_v (m = H_v is every head). Read against each layer's
                    own shape (EncoderConfig.gdn_shape_by_layer).
    heads_by_layer  {ordinal: value-head indices}: that layer's own set (distinct, ascending, whole key groups; its ordinal is
                    among `layers` when `layers` is not ()); it wins over `heads` for that layer. {} = none. Keys are ints
                    here (a JSON round trip's digit strings are turned back into ints when it is set).

    THE SECOND STEP'S CONSTRUCTION (the six keys below). Read only with n_h 2 (a non-default value with n_h 1 is refused by name). The
    defaults (key proj, value proj, beta proj, conv true) are the Gated DeltaProduct form as built, bit for bit: the second step's key,
    value and beta come from the full projections dprod.in_proj_kv, dprod.conv_weight, dprod.in_proj_b.

    key how step 2's key is made. proj: the Gated DeltaProduct form's (a full projection, short conv, SiLU, L2 norm). lowrank: k2 =
           L2norm(k1 + U_h W_h x), W_h d -> r per key head (dprod.k_down), U_h r -> d_k (dprod.k_up), r = `rank` (C1). prev: k2_t = k1_{t-1},
           the previous token's key after conv, SiLU and norm, zero at a document start and at the first token of a sequence, so the second
           step is the identity there (C2). perm: k2 = P k1, P a fixed permutation of a head's d_k channels (`perm`) (C3, permutation). rot:
           k2 = Q k1, Q a fixed seeded d_k x d_k rotation, one for every head of a layer (C3, rotation). The three fixed forms add no
           parameter to the key (a permutation is a roll or a seeded draw, Q a seeded QR; both are buffers of the layer's state, not
           parameters). value step 2's value. proj: the Gated DeltaProduct form's full projection. v1: the first step's value (the default of
           every derived key: one value, stored under a second address). beta step 2's strength: 2 sigmoid(w_b x) from its own N = H_v
           projection (dprod.in_proj_b), as the Gated DeltaProduct form's. Only proj is built. rank lowrank's r (1 .. d_k); read with key
           lowrank only. perm perm's P: roll (a cyclic shift of each head's channels by d_k / 2) or random (a fixed permutation drawn from the
           layer's own stream, a persistent buffer: it comes back with the checkpoint); read with key perm only. conv the Gated DeltaProduct
           form's short conv on the second step's channels: true with key proj and with no stride or summary; false with a derived key (it has
           no channels of its own) and with the strided or summarised second step of the variants (below), which refuse conv true.

    THE SECOND STEP ON A SUBSET OF THE TOKENS (the keys stride, summary, conv, first_beta; ced/model/gdn_dprod_sparse.py, its docstring). Read only
    with n_h 2 and key proj (refused by name with n_h 1, with a derived key, with the placement keys, with beta_init and with route "one-readout":
    the module reads none of them).

    stride the second step on the tokens with p mod stride = stride - 1 (p the packed-row position); D: the group size. summary E: "none" (the
    carrying token's own x); D: "xmean" (the group's mean of x) or "kvmean" (the mean of k1 and v1). first_beta step 1's beta on a token without the
    second step: "carry" = sigmoid(b) (R1-entry's token), "all" = beta_max sigmoid(b). SPARSE_PATH (ced/model/gdn_dprod_sparse_fast.py): how the layer
    above builds and unbuilds its inserted-row layout. Not set (None, the default): "fast" for the variant "xmean", "fast" for the variant "kvmean"
    (its GPU values check passed), "plain" for every other form. "plain" (set; the path of every run before the key): insert_rows / take_tokens
    as built. "fast": the same function with the copies at their fewest passes, values and gradients bit for bit the plain path's. "rows": "fast", and
    on the fused path (kernel "ced", a row length that is a multiple of the stride, not D "kvmean") the input preparation writes q, k, v in the row
    layout (ced/kernels/gdn_prep_rows.py), the same bits again; read only with the strided or summarised second step (refused by name with n_h 1, with
    the keys above at their defaults, with a derived key, with route "one-readout")."""

    n_h: int = 1
    route: str = "interleave"
    beta_max: int = 2  # M: 2 or 1 (beta in (0, 1), arm dprodb1)
    beta_init: float | None = None  # None, or p in (0, 1) with beta_max 1: b + logit(p) on every step, fixed (gdn_dprod.beta_offset)
    layers: tuple = ()
    heads: int = 0
    heads_by_layer: dict = field(default_factory=dict)
    key: str = "proj"
    value: str = "proj"
    beta: str = "proj"
    rank: int = 16
    perm: str = "roll"
    conv: bool = True  # the Gated DeltaProduct form's short conv on the second step's channels; False with a derived key, and must be False with stride > 1 or a summary
    stride: int = 1  # the second step on the tokens with p mod stride = stride - 1 (p the packed-row position); D: the group size
    summary: str = "none"  # E: "none" (the carrying token's own x); D: "xmean" (the group's mean of x) or "kvmean" (the mean of k1 and v1)
    first_beta: str = "carry"  # step 1's beta on a token without the second step: "carry" = sigmoid(b) (R1-entry's token), "all" = beta_max sigmoid(b)
    sparse_path: str | None = None  # the inserted-row layout's copies (docstring, last line): None (not set: "fast" for xmean and kvmean, "plain" for the other forms), "plain" (insert_rows / take_tokens as built), "fast" (same values, fewest passes) or "rows" (fast, and the fused prep writes q, k, v in the row layout)
    def __setattr__(self, name, value):
        if name == "layers" and isinstance(value, list):
            value = tuple(value)  # YAML and --set give lists; one type in memory, in to_dict and in checkpoints
        elif name == "heads_by_layer" and isinstance(value, dict):
            value = _int_table(value)
        object.__setattr__(self, name, value)


@dataclass
class GDNConfig:
    """Gated DeltaNet layer shape. The head counts are deliberately config-only (undecided)."""

    n_k_heads: int = 16
    n_v_heads: int = 16
    k_head_dim: int = 128
    v_head_dim: int = 128
    conv_kernel: int = 4
    # Gate initialization: "fla" (Mamba2-style dt in [1e-3, 1e-1]) or "hf" (dt_bias = 1, as in the
    # Hugging Face Qwen3.5 code). See scripts/gdn_init_report.py for the decay values each gives.
    init: str = "fla"
    # Delta-rule kernel: "recurrence" (the per-token reference loop, ground truth), "chunked"
    # (ced/model/gdn_chunked.py: the same function in WY/UT chunked form, pure PyTorch), or "fla"
    # (flash-linear-attention's fused Triton kernel for the CUDA training forward; falls back to
    # "chunked" elsewhere, see ced/model/gdn.py), or "ced" (our kernel for the same function,
    # ced/kernels/gdn.py; same fallback).
    kernel: str = "recurrence"
    chunk_size: int = 64
    # Activation of the output gate in the GatedRMSNorm after the delta rule: RMSNorm(o) * act(z).
    # "silu" (v1, the Hugging Face Qwen3.5 code) or "sigmoid" (architecture spec v2, Qwen3.8 form).
    # Applied outside the delta-rule kernel, so every kernel uses the same gate.
    output_gate: str = "silu"
    # Dtype of q, k, v and beta at the fused delta-rule kernel ("fla" or "ced") under bf16 autocast.
    # "float32": as before (the short conv's float32 weight promotes q, k, v to float32; beta is cast
    # to q's dtype). "bfloat16" (2026-10-07): q, k, v (after the conv, SiLU, L2 norm and q scale,
    # computed in float32) and beta reach the kernel in bf16; g (log decay) stays float32, and inside
    # the kernel the recurrent state and every accumulation stay float32. It applies only where the
    # in-projection's output is bf16 (bf16 autocast) and the fused kernel runs: float32 and float64
    # runs and the reference paths (recurrence, chunked, cached inference) keep their own precision.
    input_dtype: str = "float32"
    # How our kernel (kernel "ced", impl "triton") forms the within-chunk cumulative log decay that every decay inside a
    # 64-token chunk is an exponent of. "plain" (the default; every run before this key): one float32 cumulative sum,
    # whose rounding is about 2^-24 of the chunk's |sum of g| in every pair's exponent (the kernel gates' gate50
    # finding). "compensated": F3: the sum is a float pair (hi, lo) from a TwoSum scan and each exponent is (hi_i - hi_j)
    # + (lo_i - lo_j) (ced/kernels/gdn_triton.py). No parameter, no state-dict entry, no random number. Kernel "fla" has
    # no such option: "compensated" with it is refused at build. The reference forms (recurrence; chunked, also kernel
    # "ced"'s CPU fallback) never difference a cumulative sum and ignore the key.
    decay_cumsum: str = "plain"
    # C. "plain" (the default; every run before this key): the layer as it was. "fused" (kernel "ced" only): the training
    # path's input preparation (short conv, SiLU, L2 norm, q scale) runs in our chunk kernels' prologues, so q, k and v
    # are never written or saved; the varlen metadata is formed on the device (no host read); one torch.library custom op
    # each way (ced/kernels/gdn_edges.py), called where today's two opaque calls are, with today's graph breaks, so the
    # compiled graphs around it are today's. The same function with the same arithmetic as "plain", built to be bit for
    # bit, forward, backward and a training step (checked on a GPU: tests/gpu/test_gdn_edges_gpu.py). "fused-traced": the
    # same op traced into the layer's graph (no graph break: one compiled graph a layer); Inductor then adds the four
    # input projections' gradients of the layer input in one place, so that sum's rounding order is not today's (a
    # tolerance change, measured by the same check). Only the one-pass training path takes either (no keep, valid, state,
    # GDN-side arms); everywhere else, and on the CPU, the layer runs as with "plain". No parameter, no state-dict entry.
    # (Not "off" / "on": YAML and the --set overrides read those as booleans.) C2: "once" (kernel "ced" only), the op in
    # "fused"'s place with its graph breaks, whose first chunk kernel forms q, k and v once a chunk with qkv_prep's
    # statements, stores them once and does the solve and w / u on them (prep, intra and wu in one kernel), and whose WY
    # backward folds in the dk sum (ced/kernels/gdn_c2.py). Today's statements on today's operands, but a GPU may round
    # the solve's and the L2 norm's sums (and, with float32 kernel inputs, w / u's products) in another order (other tile
    # layouts): a numerics change, judged against the float64 reference
    # (scripts/gdn_c2_check.py) and by the 60M band check. Refused with TrainConfig.cuda_graphs, as "fused".
    edges: str = "plain"
    # (in_proj_a and in_proj_b as one GEMM, the plan's step 4 of C, is not a key: it cannot be bit for bit today's two
    # GEMMs, so the build stopped there; section 4) D: which kernels run the delta-rule core of kernel "ced" on the CUDA
    # training path. "triton" (the default; every run before this key): our Triton chunk kernels, the layer as it was.
    # "flashqla": FlashQLA 0.1.3's TileLang kernels (flash-qla, MIT, Qwen) in their exact mode (auto_cp False: no intra-card
    # context parallelism), pinned by sha256 and installed outside the environment (ced/kernels/gdn_flashqla.py); the same
    # function (our gates, beta, varlen boundaries, grouped value heads, no initial or final state, scale 1) with FlashQLA's
    # algebra and rounding: a numerics change, judged against the float64 reference and by the 60M band check. Everything
    # around the core is the layer's own (the fused input preparation, the gates, the output norm). With decay_cumsum
    # "compensated", F3 is ported into FlashQLA's SM90 and SM100 kernels (ced/kernels/flashqla_f3/, the diff named there;
    # the SM100 copies lowered on the CPU only, unrun); FlashQLA's SM120 path (chunks of 32) and other GPUs are refused by
    # name. Built for kernel "ced", input_dtype "bfloat16" (FlashQLA takes bf16 or fp16 only), edges "plain", K = V = 128;
    # refused by name with CCR's writer, CUDA graphs, a CPU tensor or a float32 kernel input.
    chunk_backend: str = "triton"
    # The negative-eigenvalue form, negative eigenvalues (GDNNegEigConfig above): model.encoder.gdn.negeig.beta_max, 1 (the
    # default: the layer as it was) or 2 (beta in (0, 2)).
    negeig: GDNNegEigConfig = field(default_factory=GDNNegEigConfig)
    # The GDN memory-rule the KDA forms (KDAConfig above; ced/model/gdn_kda.py): per-channel decay and the horizon floor.
    # The default (decay "scalar") is the layer as it was: no tensor, no operation and no random number added.
    kda: KDAConfig = field(default_factory=KDAConfig)
    # The GDN memory-rule the KDA + erase and EDA forms (EraseConfig above; ced/model/gdn_erase.py): EDA's separate erase
    # direction, on either decay. The default (mode "none") is the layer as it was: no tensor, no operation and no random number
    # added.
    erase: EraseConfig = field(default_factory=EraseConfig)
    # Gated DeltaProduct: its own sub-section, DeltaProductConfig above. n_h 1 (the default): the layer as it was, bit
    # for bit.
    dprod: DeltaProductConfig = field(default_factory=DeltaProductConfig)

    def __post_init__(self):
        # One generic hook for every sub-section (negeig, kda, erase, dprod); a second __post_init__ would replace this one.
        # A sub-section is this config's own: dataclasses.replace (the decoder's GDNConfig made from the encoder's,
        # tiny_v2_config) passes the same object, which would tie the two layers' settings together; each gets a copy.
        # A sub-section given as a dict (GDNConfig(dprod={"n_h": 2})) becomes its dataclass, as gdnrule-dprod's hook did.
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, dict) and dataclasses.is_dataclass(f.default_factory):
                object.__setattr__(self, f.name, f.default_factory(**v))
            elif dataclasses.is_dataclass(v) and not isinstance(v, type):
                object.__setattr__(self, f.name, dataclasses.replace(v))


WRITE_MIX_STORAGES = ("float32", "bf16")  # EncoderConfig.write_mix_storage's values (ced/kernels/write_mix.py STORAGES is this tuple)


def check_write_mix_storage(name) -> str:
    """EncoderConfig.write_mix_storage's value: one of WRITE_MIX_STORAGES, else ValueError by name (the string itself, not a look-alike)."""
    if type(name) is not str or name not in WRITE_MIX_STORAGES:
        raise ValueError(f"model.encoder.write_mix_storage {name!r}: one of {WRITE_MIX_STORAGES}")
    return name


RESIDUAL_DTYPES = ("float32", "bf16")  # EncoderConfig.residual_dtype's values


def check_residual_dtype(name) -> str:
    """EncoderConfig.residual_dtype's value: one of RESIDUAL_DTYPES, else ValueError by name (the string itself, not a look-alike)."""
    if type(name) is not str or name not in RESIDUAL_DTYPES:
        raise ValueError(f"model.encoder.residual_dtype {name!r}: one of {RESIDUAL_DTYPES}")
    return name


@dataclass
class EncoderConfig:
    n_layers: int = 36
    # Layer i is a global attention layer iff (i + 1) % full_attn_interval == 0 (the Qwen3-Next rule),
    # so 4 gives the 3:1 GDN:global pattern [G, G, G, A] * n. 0 means GDN only.
    full_attn_interval: int = 4
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    gdn: GDNConfig = field(default_factory=GDNConfig)
    # Loop the encoder instead of (or as well as) the decoder: the "YOCO-U" arm (arXiv 2604.01220,
    # which loops the half that runs on every token). Each encoder loop keeps its own GDN states and
    # global-attention KV; the loop boundary reuses LoopConfig's injection / hyper-connection /
    # loop-embedding settings; the decoder's memory KV is built from the final loop's output only.
    loops: int = 1
    # A mix of position schemes by layer. Indices among the GLOBAL attention layers, bottom first (0 = the lowest
    # global layer, layer 3 of the 20 at 60M; 4 = layer 19), whose attention runs with NO rotary part: such a layer's
    # attention config is `attention` with rope_fraction 0.0 (global_attention below), so it runs the plain read as a
    # layer of the NoPE arm does and has nothing to cap. The other global layers use `attention` as given, with its
    # rope_max_distance. () (the default): every global layer uses `attention`, every run before this key (no parameter
    # either way: the state dict, the parameter count and the RNG use are unchanged). All of them: the NoPE arm's
    # model, bit for bit. Always a tuple of ints here (a list from YAML or --set is turned into one when it is set), so
    # configs compare equal however they were written. Built for arch "control" with a rotary distance cap on
    # `attention` and no second pass (loop.pass2_gate "none"); refused elsewhere (ced/model/control.py
    # check_nope_global_layers).
    nope_global_layers: tuple[int, ...] = ()
    # Per-layer residual scalars with x0 re-injection. "none" (the default): no tensor, no operation; every model before
    # this key. "skip_x0": every layer i of the stack computes x <- a_i x + b_i x0 + f_i(x) where x is the layer's input,
    # f_i(x) = layer_i(x) - x is everything the layer adds (its mixer branch and its FFN branch, the FFN reading the
    # unscaled x plus the mixer branch, as before), and x0 is the stack's input: the embedding output (after embed_norm),
    # the same per-position tensor at every layer. a_i and b_i are one trained scalar each per layer, stored zero-centred
    # as the norm gains are: encoder.skip_gain (a_i = 1 + skip_gain[i]) and encoder.x0_gain (b_i = x0_gain[i]), both zeros
    # at the start, where the model is the model without the key bit for bit. 2 x n_layers parameters (40 at 60M). On
    # AdamW at the run's LR, without weight decay (ced/train/optim.py). Not `model.residual_scale`: that is the fixed
    # multiplier on the residual BRANCHES of a looped block. Built for arch "control" with one pass (no loop.pass2_gate,
    # encoder.loops 1) and without NVFP4; refused elsewhere (ced/model/control.py check_layer_scalars). Two more forms
    # (section 4, rows (ii) and (iv)). Both compute the same x <- a_i x + b_i x0 + f_i(x), start at a = 1 and b = 0 (the
    # model without the key, bit for bit) and keep the parameter names; they differ in how a and b follow from the stored
    # tensors (ced/model/encoder.py layer_scalar_terms, the one place). In both, b_0 is fixed at 0: layer 0's input is x0
    # itself, so a_0 and b_0 would be one parameter and a bound on a_0 alone would bound nothing. "skip_x0_span" the
    # span-bounded form. encoder.skip_gain stores u_1 .. u_L (L = n_layers). With S_0 = 0 and S_k = (ln R / 2) tanh(u_k),
    # R = layer_scalars_span, a_i = exp(S_{i+1} - S_i). The product of a over ANY run of consecutive layers i .. k is
    # exp(S_{k+1} - S_i), inside (1/R, R) at any depth; from the bottom to any layer it is inside (1/sqrt(R), sqrt(R)). u
    # is read clamped to [-3, 3]
    #                   (tanh 0.995) and the trainer puts the stored value back into that range after every optimizer
    #                   step, so a value pushed against the bound cannot travel into the flat part of the tanh. b_i =
    #                   encoder.x0_gain[i] for i >= 1, free as in "skip_x0". 2 x n_layers stored values. "x0" mHC's
    #                   constraint at one stream: a_i = 1 exactly (no tensor and no product for it: encoder.skip_gain does
    #                   not exist), x <- x + b_i x0 + f_i(x), b_i = encoder.x0_gain[i] for i >= 1. n_layers stored values.
    #                   The composite gain of the identity path is 1 at any depth. One more form, and where the sum is
    #                   computed (and the arms "R-a: a only (b fixed at 0)", "R-b: b only (a fixed at 1)"): "skip" a only:
    #                   x <- a_i x + f_i(x), a_i = 1 + encoder.skip_gain[i] in every layer, b_i = 0 exactly (no tensor and
    #                   no product for it: encoder.x0_gain does not exist and x0 is not read). n_layers stored values. Arm
    #                   R-a. ("x0" above is -b.) In every form the scalar terms are the last part of the layer's own
    #                   residual add (ced/model/encoder.py EncoderLayer.forward calls apply_layer_scalars: out + ((a - 1)
    #                   x + b x0), out = x + mixer branch + MLP branch), inside a compiled layer's graph, in at least
    #                   float32. The definition is the one the runs resid-mxl-a-s1/-s2/-s3 trained with (there the same
    #                   sum was formed after the layer call).
    layer_scalars: str = "none"
    # The width R of the span bound of layer_scalars "skip_x0_span": every product of a over consecutive layers stays
    # inside (1/R, R). 4.0 (the proposal's choice: it holds the period-four pattern seed 1 found and a trend of about
    # 2 across the stack). Read by that form only; any other value with another form is refused (check_layer_scalars).
    layer_scalars_span: float = 4.0
    # Canon layers (arXiv 2512.17351, "Canon-A"; ced/model/canon.py): the kernel size K of a residual, causal, depthwise
    # convolution on the input of every GLOBAL attention layer, after its input norm: h'_t = h_t + sum_{j=0..K-1} w_j *
    # h_{t-j} (per channel; inputs of t's own document only) with the taps w (d_model, K) zero at the start, so the model
    # starts as the model without the key. 0 (the default): no Canon layer, every model before this key (no parameter, no
    # module, no random number). The 60M screen runs 4, the source's kernel. Each such layer adds K * d_model parameters
    # (AdamW, with weight decay, as the GDN layers' depthwise convolution). Built for arch "control" without a second
    # pass; refused elsewhere (ced/model/control.py check_canon). The decoder's Canon layers wait with stage 2.
    canon_kernel: int = 0
    # The start of the Canon taps (ced/model/canon.py, "The source's start"). "zeros" (the default): the taps are zeros and
    # draw no random number, every model, config and checkpoint before this key, bit for bit. "uniform": every tap is drawn
    # uniform in +-1/sqrt(canon_kernel) (+-0.5 at kernel 4), the start the source's code leaves (PyTorch's Conv1d default),
    # so the model is NOT the model without the Canon layers at step 0. The taps come from a generator of their own, one
    # per Canon layer, seeded from the run's seed and the layer's index in the stack (canon.canon_generator), never from
    # the global generator: every other parameter starts at the value it has with "zeros" and without the key. Read at
    # model build only: a checkpoint's taps are loaded like any weight. Any other value, and "uniform" without a kernel, is
    # refused at model build (ced/model/control.py check_canon).
    canon_init: str = "zeros"
    # Block Attention Residuals (arXiv 2603.15031, section 3.2; ced/model/attn_res.py). "none" (the default): no tensor,
    # no module, no random number; every model, config and checkpoint before this key. "block": the stack's residual sum
    # is replaced by softmax attention over depth. Every sublayer (a layer's mixer and its MLP, each as it is: pre-norm,
    # mixer or MLP, output norm) reads h = sum_v alpha_v v over its sources v: the embedding output, the finished blocks'
    # sums and its own block's partial sum, with alpha = softmax(w^T RMSNorm(v)), w a (d,) pseudo-query of its own that
    # starts at zero (uniform weights), and the stream the final norm reads is the same attention over the embedding and
    # all blocks. A block is one hybrid group of full_attn_interval layers. Adds 2 x d_model values per reader (the query
    # and the key norm's gain), 2 x n_layers readers: 40,960 at 60M. AdamW at the run's LR with the run's weight decay
    # (ced/train/optim.py). NOT the model without the key at the start: the first step already reads an equal-weight mean
    # of its sources where the base reads their sum. Built for arch "control" with one pass, no layer_scalars, no Canon
    # layer, no NVFP4; refused elsewhere (ced/model/attn_res.py check_attn_res). "block_entry" (arm R1-entry): the same
    # softmax attention over depth, but only where a block begins and at the output. Block n >= 2 starts from h = sum_v
    # alpha_v v over the embedding output and the sums of the blocks before it (R1's first read of the block); block 1
    # starts from the embedding output; inside a block the base's plain residual add runs, x <- x + f(x); a block's sum
    # is what it added to its own input; the final norm reads R1's own read over the embedding and all blocks. One reader
    # per block: 2 x d_model values each, 5,120 at 60M. Not a form of the paper. The same optimizer rule, the same
    # refusals and the same blocks as "block"; NOT the model without the key at the start (a block reads the mean of the
    # earlier blocks, not their sum). "ldm" (ced/model/ldm.py): the base's residual stream x <- x + delta, and beside it
    # a per-position associative memory M of ldm_groups groups, each a (ldm_key_dim x d_model / ldm_groups) matrix. Every
    # sublayer reads x + gamma (g q)^T M (q, g token-dependent: an address network of x), and every sublayer's output
    # delta is written into M with the gated delta rule M <- lam M + k_w (v_w - (lam M)^T k_w)^T (k from the updated
    # stream x + delta, v = delta); the embedding output is the first write, and the final norm reads x + its own read of
    # M. Not R1-entry: no reader of R1-entry exists in this form. Its tensors are drawn from a generator of their own
    # (every other tensor starts as in the base). NOT the model without the key at the start.
    attn_res: str = "none"
    # Grouped depth mixing on R1-entry (ced/model/attn_res.py): every reader of R1-entry ("block_entry") splits its query
    # and its sources' channels into attn_res_heads contiguous slices, each slice with its own softmax over the sources
    # (the keys' RMS taken over all d_model channels, as the paper's Eq. 3). 1 (the default): one softmax per reader,
    # R1-entry and every model before this key, bit for bit. No tensor is added. A divisor of d_model; a value other than
    # 1 is built for "block_entry" only (refused elsewhere, check_attn_res).
    attn_res_heads: int = 1
    # Token-dependent queries for R1-entry's readers: "static" (the default): a (d,) pseudo-query per reader, R1-entry and every model
    # before this key, bit for bit. "token": each reader's query
    # is w(x) = W2 SiLU(W1 RMSNorm(x) + b1) + b2, LDM's address network (arXiv 2610.18809, section 3.2) with a d_model
    # output, x the stream at the read (the stream where the block before ended; at the output the stream after the
    # last layer); W2 and b2 start at zero (b2 plays the static query), so the model starts as R1-entry. Built for
    # "block_entry" only (refused elsewhere, check_attn_res).
    attn_res_query: str = "static"
    # The hidden width of that query network: 0 (the default) is d_model // 32, LDM's ratio (32 at d 1,024 and 64 at
    # 2,048 in the paper's configs). Read with attn_res_query "token" only; any other value without it is refused.
    attn_res_query_hidden: int = 0
    # E (ced/kernels/entry_mix.py): false (the default): today's path, every model, config and checkpoint before this key,
    # bit for bit. true: in a forward with gradients on, R1-entry's entry read and the opening layer's input norm run as
    # one Triton kernel each way, and the output read as the same kernel without a norm; evaluation, scoring and generation
    # keep today's path. The same model, another rounding, so its gate is the 60M band check. CUDA
    # (or TRITON_INTERPRET=1) only, refused by name elsewhere; built for attn_res "block_entry" with R1-entry's own readers
    # (refused with every other value and with the arms' options: ced/model/attn_res.py check_attn_res).
    attn_res_fused: bool = False
    # Linear Depth Memory's shape (attn_res "ldm" only; any value other than the default without it is refused):
    # ldm_groups G (the paper's main runs 16), ldm_key_dim K (the paper's 8), ldm_hidden the address networks' hidden
    # width (0: d_model // 32, the paper's ratio).
    ldm_groups: int = 8
    ldm_key_dim: int = 8
    ldm_hidden: int = 0
    # The gain of every norm INSIDE the layers (corrected by). "channel" (the default): y = x / rms(x) * (1 + w), w of
    # shape (d,), every model, config and checkpoint before this key, bit for bit. "scalar": the same norms with w of
    # shape (1,): one gain per norm, zero at the start (so the model at step 0 is the per-channel model: every gain is
    # 1), AdamW with the run's weight decay as every norm gain (ced/train/optim.py: one-dimensional, the decayed AdamW
    # group). The norms it covers, in every layer (ced/model/encoder.py EncoderLayer): the four Peri-LN norms
    # mixer_in_norm, mixer_out_norm, mlp_in_norm and mlp_out_norm, and in a GDN layer the gated output norm inside the
    # mixer (mixer.norm, ced/model/gdn.py: its output is what out_proj reads). Per-channel always: the embedding norm and
    # the final norm (outside the layers), the attention's QK-norm (q_norm) and Block AttnRes's key norms
    # (attn_res.*.key_norm: the keys of a depth attention, read as QK-norm is). Built for arch "control" (any residual
    # form: the base, R1 and R1-entry read the layer's own norms; ced/model/attn_res.py); refused elsewhere
    # (ced/model/control.py check_block_norm_gain).
    block_norm_gain: str = "channel"
    # The GDN-side architecture arms (ced/model/gdn_arms.py has the forms). Both off by default: no parameter, no module, no
    # random number, no operation; every model, config and checkpoint before these keys, bit for bit. gdn_write_mix,
    # "Write-shallow / read-deep GDN": "none" (the default), "layer" or "head". On, in every GDN layer but layer 0 (whose
    # input is the embedding output itself), the inputs of the k, v, beta and decay projections become x_w = x + lambda (m' -
    # x): x the layer's normed input (what the projections read now), m' the layer's own softmax depth mix over [embedding,
    # finished blocks] (R1-entry's stored sources), put through the layer's own input norm; q and everything else keep x.
    # lambda is zero at the start ("layer": one scalar a layer; "head": one per key head for the k channels and one per value
    # head for the v channels, beta and the decay), so the model starts as R1-entry. Built for arch "control" with attn_res
    # "block_entry"; refused elsewhere.
    gdn_write_mix: str = "none"
    # gdn_write_variant, the wsl variants (ced/model/gdn_arms.py "The wsl variants"): which of a GDN layer's projections read a depth-mixed input
    # under the write mix's "layer" form. "none" (the default): the write mix as built (wsl), the k, v, beta and decay rows read x_w and the q rows x;
    # every model, config and checkpoint before this key, bit for bit. "qmix" (wsl-qmix): the q rows also read a depth-mixed input, x_q = x + lambda_q
    # (m'_q - x), with m'_q the layer's own SECOND softmax depth mix over the same sources (its own pseudo-query and key-norm gain, zeros at the
    # start) through the layer's input norm, and lambda_q its own scalar, zero at the start. "korig" (wsl-korig): the k rows read x, as the q rows do;
    # only v, beta and the decay read x_w. Either starts as R1-entry. "qshare" (wsl-qshare): the q rows read x_w too, the same mixed input the k, v,
    # beta and decay rows read, with no extra mix and no tensor added (all of in_proj_qkv is one projection of x_w; the output gate z keeps x); it
    # starts as R1-entry. Read only with gdn_write_mix "layer"; refused otherwise, and with CCR and the CP head, by name; "korig" also with the fused
    # write mix (attn_res_fused), which "qshare" takes, and "qmix" too since 2026-10-21 (its group read forms each layer's m' and m'_q in one pass
    # over the sources) (gdn_arms.check_gdn_arms). The write-quantity split on qshare, the q rows reading x_w in both: "qshare_kv", q, k and v read
    # x_w (one projection of x_w), beta and the decay read x; "qshare_bd", q, beta and the decay read x_w, k and v read x. No tensor added; each
    # starts as R1-entry; refused as "qshare" is, and taken by the fused write mix as "qshare" is. "vonly": ONLY the v rows read x_w; q, k, beta and
    # the decay read x (z always does). No tensor added; it starts as R1-entry; refused as "korig" is, with the fused write mix and with model.fp8, by
    # name. "kqmix" (arm KQ): qmix's tensors and its second mix, whose x_kq = x + lambda_q (m'_q - x) the q AND k rows read; the v rows, beta and the
    # decay read x_w; z reads x. It starts as R1-entry; refused as "korig" is, with the fused write mix and with model.fp8, by name.
    gdn_write_variant: str = "none"
    # gdn_qmix_select_step, qmix's query mix restricted to 2 static sources a layer (ced/model/gdn_arms.py "qmix with a static top-2 query mix"): 0
    # (the default): the wsl variant qmix as built, every model, config and checkpoint before this key, bit for bit. S >= 1: under gdn_write_variant
    # "qmix" the query mix is qmix's full softmax over every source until the start of optimizer step S; there, once, each GDN layer with more than 2
    # sources keeps the 2 sources with the largest token-mean query-mix weight on the step's first micro-batch (ties to the lower index; the statistic
    # is gdn_arms.select_qmix_pairs'), stored in the state dict (the buffer gdn_arms.qmix_keep, (2,) int64, -1 before the choice), and from then on
    # its query mix is the per-token softmax over those 2 sources only (the others neither read nor given weight). A layer with 2 or fewer sources
    # keeps all of them. Read only with the variant "qmix"; refused by name with the fused write mix (attn_res_fused), mix_fold, model.fp8 and the CP
    # head, and by the trainer with data parallelism, eval.readout_every (its scoring twin keeps the flag it was built with) and the host
    # activation-checkpoint modes (gdn_arms.check_write_variant, Trainer).
    gdn_qmix_select_step: int = 0
    # mix_fold (build (a) of the 4B-step audit; ced/kernels/fold_mix.py): false (the default): today's path, every model, config and checkpoint before
    # this key, bit for bit. true: in a forward with gradients on, under bf16 autocast, on CUDA, a GDN layer with the write mix's "layer" form
    # (gdn_write_variant "none" or "qmix") forms x_w (and qmix's x_q) in the load of ONE Triton GEMM over its q, k, v, a and b rows, so x_w and x_q
    # are never written; the depth mixes themselves stay in the group read. The same model, another rounding (its gate: the 4x check, then the 60M
    # band check). Refused by name with any other the write mix form or variant (gdn_arms.check_mix_fold). One key, three forms: false (the default);
    # true or "lerp", the form above (bf16 GEMMs); "producer", for model.fp8 on MXFP8 with 'in_norm' among model.fp8.chains: the layer's input-norm
    # pass (the chain's kernel) forms x_w (and qmix's x_q) from m' in its registers and writes their MXFP8 operands, both orientations, once
    # (ced/kernels/mx_mix_chain.py). Same model as the MXFP8 arms (model.fp8 without the key), another rounding. "producer-rc": the producer form, and
    # the fused write mix's group read (attn_res_fused) does not store the block's extra mixes: its backward recomputes them from the sources it
    # re-reads (ced/kernels/write_mix_rc.py); refused without attn_res_fused (gdn_arms.check_mix_fold). "producer-p2", "producer-p3"
    # (ced/model/mx_p2.py): that and the layer's backward as one kernel pass, and a and b in the norm pass.
    mix_fold: bool | str = False
    # write_mix_storage: the dtype the fused write mix's group read (attn_res_fused with gdn_write_mix "layer", wsl and qmix;
    # ced/kernels/write_mix.py) stores the block's NQ - 1 extra mixes in, for the backward. "float32" (the default): the sources' dtype (float32;
    # 30,720 bytes a token and extra mix at D 2,560, forward and backward together), the exact gradient, THE RULE's pass. "bf16": the GEMMs' 16-bit
    # storage, 10,240 bytes a token less per extra mix and failing THE RULE by 400 to 1,500 times at qmix's three query mixes; ONLY on this explicit
    # value, refused by name unless the fused write mix is on (attn_res_fused with gdn_write_mix "layer") and refused with the recompute forms of
    # mix_fold, which store no mix (attn_res.check_fused). The environment variable CED_WRITE_MIX_STORAGE no longer exists: a process that has it set
    # is refused by name (write_mix.refuse_legacy_env). Runs files of runs not yet submitted that use the write mix pin "float32" explicitly.
    write_mix_storage: str = "float32"
    # write_mix_row_term (CroissAttn fix for the fused write mix's softmaxes): "stored_o" (the default, every run before the key, bit for bit): the
    # group read's backward forms each extra query's row term sum_i a_ij (dm_j. v_i) as dm_j. h_j from the mix h_j AS STORED, and the extra queries'
    # mixes are stored in the GEMMs' dtype (bf16 under autocast): a per-row bias dm_j. (h_j - bf16(h_j)) that does not cancel
    # (ced/kernels/write_mix.py, "STORAGE OF THE EXTRA QUERIES' MIXES"). "pdp": the extra queries' mixes are stored in the sources' dtype (float32),
    # so the stored h_j is the accumulated mix and dm_j. h_j is sum_i a_ij da_ij to float32 summation (the unfused path's autograd form); 10,240 more
    # bytes a token and extra query at D 2,560. Read with the fused write mix (attn_res_fused with gdn_write_mix "layer") only; refused otherwise by
    # name (gdn_arms.check_write_mix_row_term); with mix_fold "producer-rc" (and the forms built on it) the extra mixes are recomputed in float32 in
    # the backward and the row term is already exact: accepted, inert. Since write_mix_storage "float32" became the default (the comment above,
    # 2026-10-26) the default stores the extra mixes in the sources' dtype as well: "pdp" then equals the default, and "stored_o" differs from it only
    # with write_mix_storage "bf16". "pdp" with write_mix_storage "bf16" asks for two opposite storages: refused by name
    # (gdn_arms.check_write_mix_row_term).
    write_mix_row_term: str = "stored_o"
    # residual_dtype (the floor report's candidate dense.stream_bf16_only, section 2.5): the precision the residual stream is HELD at between the norm
    # kernels of the fused design. "float32" (the default; every model, config and checkpoint before the key, bit for bit; absent from a saved config
    # at its default): the stream is float32, as the trainer's float32 embedding output and the promotion `float32 + bfloat16 -> float32` make it.
    # "bf16": the stream is rounded to bfloat16 wherever the fused design stores it (the embedding stage's stream copy and each block's input x_n^0,
    # which are the entry mix's output; the stream after each sublayer's add; not the block's last add, which is consumed in the same epilogue by the
    # block sum), and its gradient with it; the depth-mix SOURCES stay float32 (b_0 the embedding output; b_n = x_n^S - x_n^0 formed in float32 from
    # the rounded x_n^0 and the unrounded last sum), so they inherit the stream's rounding and nothing else, and the extra mixes' storage
    # (write_mix_storage) is untouched. The tensors stay float32 (the rounding is `x.to(bfloat16).to(x.dtype)`, ced/model/encoder.py round_stream):
    # the reference runs on every path (autocast on or off, float64 tests) and the values are those of a bfloat16 tensor. Built for attn_res "none"
    # and "block_entry" on the plain path; refused by name where another path would skip it (attn_res_fused, model.fp8 chains, mix_fold,
    # layer_scalars, the loops, the gated second pass, the CP head: attn_res.check_residual_dtype).
    residual_dtype: str = "float32"
    # gdn_ccr_spacing, arm CCR ("Checkpoint cross-read"): 0 (the default) is off; C > 0 is the checkpoint spacing in
    # tokens of a document. On, each GDN layer after the first also reads the PREVIOUS GDN layer's state at its last
    # checkpoint (the state after the first C * floor(p / C) tokens of the token's document, p the token's position in
    # it; zero in a document's first C tokens) with its own q, through a per-value-head gate that is zero at the start.
    # The fused training path reads our kernel's chunk-start states, so C must be the kernel's chunk (64) there; the
    # runs use 64 and inference reads checkpoints at the same spacing. Built for arch "control" with attn_res "none" or
    # "block_entry"; refused elsewhere.
    gdn_ccr_spacing: int = 0
    # CCR v2. Read only with gdn_ccr_spacing > 0; at their defaults CCR is v1, bit for bit. gdn_ccr_target: which GDN layer's
    # checkpoint a reader reads. "previous" (the default: v1, the previous GDN layer in stack order); "lowest" (the model's
    # lowest GDN layer, which itself reads nothing); "group_first" (the first GDN layer of the reader's hybrid group of
    # full_attn_interval layers, which itself reads nothing).
    gdn_ccr_target: str = "previous"
    # gdn_ccr_translator: "none" (the default: v1, the read uses the reader's own q) or "key_head": a d_k x d_k query
    # translator M per reader layer and key head, identity at the start; the read uses q_x = M q, with q the reader's
    # L2-normalised, scaled query as v1 reads it (M acts after the norm and the scale). Stored as M - I (zeros).
    gdn_ccr_translator: str = "none"
    # gdn_ccr_gate_init: the start of every CCR reader's gate (H_v,), 0.0 by default (v1, ccr-nowd: the model starts as
    # the trunk, bit for bit). Read only with gdn_ccr_spacing > 0; a finite number; no random number is drawn
    # (torch.full).
    gdn_ccr_gate_init: float = 0.0
    # gdn_ccr_gate_floor: a floor on every CCR reader's gate, 0.0 by default (no floor: the gates are free, as in v1 and
    # ccr-v2, bit for bit). With a floor f > 0 the trainer puts every gate back to at least f after every optimizer step
    # (gdn_arms.floor_gates_, a clamp in place: projected AdamW), so a gate can rise above f by its gradient and cannot
    # fall below it. Read only with gdn_ccr_spacing > 0; a finite number >= 0, at most gdn_ccr_gate_init (the start must
    # lie at or above the floor), and only with optim.zero_gate_decay "none" (weight decay would pull every gate onto the
    # floor whatever the loss says).
    gdn_ccr_gate_floor: float = 0.0
    # A GDN layer's own shape (design sections 4 and 5): {ordinal: (n_k_heads, n_v_heads, head_dim)}, the ordinal 0-based among
    # the encoder's GDN layers (0 .. 14 at 60M, 0 .. 29 at 4B); head_dim is both k_head_dim and v_head_dim of that layer. {} (the
    # default): every GDN layer has `gdn`'s shape, every run before this key (nothing is built, drawn or touched differently).
    # The other fields of `gdn` apply to every layer. It is how a compute-matched n_h = 2 model is cut between two head counts (8
    # of the 30 layers at 6 key /12 value heads and 22 at 7 / 14), and how a per-layer head table is written. Each shape must
    # satisfy the layer's own rules (n_v_heads a multiple of n_k_heads; our kernel 'ced' takes 128 and n_v_heads in (n_k_heads, 2
    # n_k_heads)). Built for arch "control" without the second pass, the MTP drafter, FP8 / NVFP4, fused projections or the
    # GDN-side arms; the decode engine, the GDN state diagnostics and the other tools that read one GDN shape refuse it by name
    # (ced/model/gdn_rules.py).
    gdn_shape_by_layer: dict = field(default_factory=dict)

    def __setattr__(self, name, value):
        if name == "nope_global_layers" and isinstance(value, list):
            value = tuple(value)  # YAML and --set give lists; one type in memory, in to_dict and in checkpoints
        elif name == "gdn_shape_by_layer" and isinstance(value, dict):
            value = _int_table(value)
        object.__setattr__(self, name, value)

    def layer_kinds(self) -> list[str]:
        kinds = []
        for i in range(self.n_layers):
            is_global = self.full_attn_interval > 0 and (i + 1) % self.full_attn_interval == 0
            kinds.append("attention" if is_global else "gdn")
        return kinds

    def global_layers(self) -> list[int]:
        """The stack indices of the global attention layers, bottom first."""
        return [i for i, kind in enumerate(self.layer_kinds()) if kind == "attention"]

    def nope_layers(self) -> tuple[int, ...]:
        """nope_global_layers checked: distinct integers among 0 .. (number of global layers) - 1, ascending.
        Raises ValueError otherwise (nothing is sorted or dropped silently)."""
        idx, n = self.nope_global_layers, len(self.global_layers())
        ok = isinstance(idx, tuple) and all(type(i) is int and 0 <= i < n for i in idx) and list(idx) == sorted(set(idx))
        if not ok:
            raise ValueError(f"model.encoder.nope_global_layers {idx!r}: distinct indices among the encoder's {n} global "
                             f"attention layers, bottom first (0 .. {n - 1}), in ascending order")
        return idx

    def global_attention(self) -> list[AttentionConfig]:
        """The attention config of every global layer, bottom first. A layer outside nope_global_layers gets
        `attention` itself (the same object, as every model before the key); a layer in it gets a copy without a
        rotary part: rope_fraction 0.0 and no rope_max_distance, which is the NoPE arm's attention config. The copies
        are made when the model is built: change `attention` (a backend, a scoring clamp) before building, as every
        script does."""
        nope = set(self.nope_layers())
        return [dataclasses.replace(self.attention, rope_fraction=0.0, rope_max_distance=None) if g in nope
                else self.attention for g in range(len(self.global_layers()))]

    def gdn_shapes(self) -> list:
        """The GDNConfig of every encoder GDN layer, bottom first, with the layer's own shape (gdn_shape_by_layer). With the table
        {} (the default) every entry is `gdn` itself, the same object (as global_attention returns `attention`): the layers are
        built from exactly the config they always were. A layer in the table gets a copy of `gdn` with its n_k_heads, n_v_heads
        and head size (both k_head_dim and v_head_dim); every other field is `gdn`'s. ValueError by name on a bad table."""
        n = self.gdn_count()
        table = self.gdn_shape_by_layer
        if not table:
            return [self.gdn] * n
        if not isinstance(table, dict):
            raise ValueError(f"model.encoder.gdn_shape_by_layer {table!r}: a mapping {{GDN ordinal: [n_k_heads, n_v_heads, head_dim]}}")
        bad = [k for k in table if type(k) is not int or not 0 <= k < n]
        if bad:
            raise ValueError(f"model.encoder.gdn_shape_by_layer keys {bad!r}: GDN ordinals, integers among the encoder's {n} GDN layers "
                             f"(0 .. {n - 1})")
        out = []
        for i in range(n):
            if i not in table:
                out.append(self.gdn)
                continue
            shape = table[i]
            ok = (isinstance(shape, (list, tuple)) and len(shape) == 3 and all(type(v) is int and v >= 1 for v in shape))
            if not ok:
                raise ValueError(f"model.encoder.gdn_shape_by_layer[{i}] {shape!r}: [n_k_heads, n_v_heads, head_dim], three integers >= 1")
            hk, hv, hd = shape
            if hv % hk:
                raise ValueError(f"model.encoder.gdn_shape_by_layer[{i}] {list(shape)!r}: n_v_heads must be a multiple of n_k_heads")
            out.append(dataclasses.replace(self.gdn, n_k_heads=hk, n_v_heads=hv, k_head_dim=hd, v_head_dim=hd))
        return out

    def gdn_count(self) -> int:
        """The number of GDN layers in the encoder (the ordinals of gdn_shape_by_layer and DeltaProductConfig.layers run 0 .. n - 1)."""
        return self.layer_kinds().count("gdn")

    def dprod_carry(self) -> list:
        """Which value heads of each encoder GDN layer carry Gated DeltaProduct's extra steps (DeltaProductConfig placement),
        bottom first: a list of one entry per GDN layer. With n_h 1 every entry is None (nothing carries; the placement keys
        must be at their defaults). Otherwise a tuple of value-head indices (against that layer's own shape): () for a layer
        that does not carry, every head 0 .. H_v - 1 for a layer that carries as the Gated DeltaProduct form does, a proper
        subset (whole key groups: every key head serves H_v / H_k consecutive value heads and all of them are in the set or
        none) for a layer that carries some of its heads. ValueError by name on a bad key."""
        d = self.gdn.dprod
        shapes = self.gdn_shapes()
        n = len(shapes)
        place = {"layers": d.layers != (), "heads": d.heads != 0, "heads_by_layer": d.heads_by_layer != {}}
        if d.n_h == 1:
            used = [k for k, on in place.items() if on]
            if used:
                raise ValueError(f"model.encoder.gdn.dprod.{', '.join(used)} set with n_h 1: the placement keys are read only with n_h >= 2 "
                                 "(Gated DeltaProduct; at n_h 1 they would change nothing)")
            return [None] * n
        layers = d.layers
        ok = isinstance(layers, tuple) and all(type(i) is int and 0 <= i < n for i in layers) and list(layers) == sorted(set(layers))
        if not ok:
            raise ValueError(f"model.encoder.gdn.dprod.layers {layers!r}: distinct GDN ordinals among the encoder's {n} GDN layers "
                             f"(0 .. {n - 1}) in ascending order, or () for every layer")
        if type(d.heads) is not int or d.heads < 0:
            raise ValueError(f"model.encoder.gdn.dprod.heads {d.heads!r}: an integer >= 0 (0: every value head of a carrying layer)")
        table = d.heads_by_layer
        if not isinstance(table, dict):
            raise ValueError(f"model.encoder.gdn.dprod.heads_by_layer {table!r}: a mapping {{GDN ordinal: [value heads]}}")
        carrying = set(layers) if layers else set(range(n))
        stray = [k for k in table if type(k) is not int or k not in carrying]
        if stray:
            raise ValueError(f"model.encoder.gdn.dprod.heads_by_layer keys {stray!r}: ordinals of layers that carry the extra steps "
                             f"({sorted(carrying)!r}; dprod.layers is {list(layers)!r})")
        out = []
        for i in range(n):
            if i not in carrying:
                out.append(())
                continue
            hk, hv = shapes[i].n_k_heads, shapes[i].n_v_heads
            g = hv // hk
            if i in table:
                heads = table[i]
                good = (isinstance(heads, (list, tuple)) and len(heads) > 0 and all(type(h) is int and 0 <= h < hv for h in heads)
                        and list(heads) == sorted(set(heads)))
                if not good:
                    raise ValueError(f"model.encoder.gdn.dprod.heads_by_layer[{i}] {heads!r}: distinct value heads among this layer's "
                                     f"{hv} (0 .. {hv - 1}) in ascending order, at least one")
                chosen = tuple(heads)
            elif d.heads:
                if d.heads % g or d.heads > hv:
                    raise ValueError(f"model.encoder.gdn.dprod.heads {d.heads}: a multiple of the layer's {g} value heads a key head "
                                     f"(whole key groups, which the kernels take) and at most its {hv} value heads (GDN layer {i})")
                chosen = tuple(range(d.heads))
            else:
                chosen = tuple(range(hv))
            split = [k for k in sorted({h // g for h in chosen}) if not all(h in chosen for h in range(k * g, (k + 1) * g))]
            if split:
                raise ValueError(f"model.encoder.gdn.dprod value heads {list(chosen)!r} of GDN layer {i}: key head(s) {split!r} would be "
                                 f"cut in half (a key head serves {g} consecutive value heads and the kernels take whole key groups)")
            out.append(chosen)
        return out

    def gdn_layers(self) -> list:
        """The GDNConfig each encoder GDN layer is built from, bottom first: gdn_shapes() with Gated DeltaProduct's placement applied.
        A layer that does not carry the extra steps gets the default DeltaProductConfig (n_h 1: it has no `dprod` module and is
        R1-entry's layer); a layer that carries gets the model's own DeltaProductConfig with the three placement keys at their
        defaults, every other key kept as set (n_h, and the keys of other builds, e.g. beta_max and beta_init of dprodb1: the carrying
        heads are dprod_carry()'s, handed over separately). With gdn_shape_by_layer {} and the placement keys at their defaults, every
        entry is `gdn` itself."""
        shapes = self.gdn_shapes()
        d = self.gdn.dprod
        carry = self.dprod_carry()
        if d.n_h == 1 or (d.layers == () and d.heads == 0 and d.heads_by_layer == {}):
            return shapes
        # (not DeltaProductConfig(n_h=...), which dropped every other key of the model's dprod)
        return [dataclasses.replace(cfg, dprod=dataclasses.replace(d, layers=(), heads=0, heads_by_layer={}) if c else DeltaProductConfig())
                for cfg, c in zip(shapes, carry)]


@dataclass
class DecoderConfig:
    n_layers: int = 4
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    # Decoder layer i reads the memory KV (and own per-loop KV) through softmax attention iff
    # (i + 1) % memory_attn_interval == 0; the other layers are Gated DeltaNet layers that run over
    # the sequence of decoder positions with per-loop recurrent state. 1 = every layer attends
    # (the original spec); 4 = the 2026-10-03 default for the primary candidate (decode reads the
    # memory KV in one layer in four).
    memory_attn_interval: int = 1
    gdn: GDNConfig = field(default_factory=GDNConfig)
    # Feed-forward: 0 = dense SwiGLU; > 0 = routed SwiGLU with this many experts, top-k each,
    # expert width ffn_expert_hidden (default ffn_hidden // top_k: same active width as dense).
    ffn_experts: int = 0
    ffn_top_k: int = 2
    ffn_expert_hidden: int | None = None
    moe_balance_weight: float = 0.01
    # Separate token-mixer weights for each loop instead of one shared copy: "none", "attention"
    # (only the memory-attention layers; the routed-FFN arm), or "all" (attention and GDN layers).
    untie_mixers: str = "none"
    router_z_weight: float = 0.0
    # Memory KV (keys/values that context positions offer the decoder). Shared across loops always;
    # shared across decoder layers unless memory_kv_per_layer (a ladder item).
    memory_kv_per_layer: bool = False

    def layer_kinds(self) -> list[str]:
        k = self.memory_attn_interval
        return ["memory_attention" if (i + 1) % k == 0 else "gdn" for i in range(self.n_layers)]
    # Fixed decoder capacity per packed row: C = min(L, round_up(ceil(p * L), multiple)).
    capacity_fraction: float = 0.25
    capacity_multiple: int = 128
    # CED v2 only: decoder self-KV. False (default): every decoder layer attends over the encoder-written memory
    # only, the v2 /the variant model bit for bit. True: every decoder layer also has k_proj / v_proj / k_norm, and
    # a slot attends in ONE softmax over the memory entries of s <= t plus that layer's own K/V of the decoder
    # positions s <= t, same document (ced/model/ced_v2.py). T = 1 or T = 2: pass 2 reads pass 1's K/V of the same
    # layer and writes none, with the gate of LoopConfig.pass2_gate at the loop boundary or, with "none", as the
    # plain loop. T > 2 is refused.
    self_kv: bool = False
    # CED v2 with self_kv only: the decoder read pattern, one letter per layer, repeated from the first decoder
    # layer: "F" (full) reads all memory entries s <= t plus the layer's own K/V of the earlier decoder
    # positions, as above (stage 1); "L" (local) reads, in one softmax, only the layer's own K/V at the first
    # n_sinks positions of t's document and at the positions t - local_window < s <= t (each key once where both
    # hold), no memory. The default "F" is every layer full: stage 1 bit for bit (tests/test_v5_stage2.py).
    # Stage 2 is "LLLF" (9 layers: full at layers 3 and 7, counted from 0).
    read_pattern: str = "F"
    local_window: int = 128
    n_sinks: int = 4
    # CED v2 with local layers only (read pattern; stage 2 + Scalable-Softmax): the attention temperature of the LOCAL
    # decoder layers. "inherit" (the default): decoder.attention.temperature, as every config before this field
    # (existing configs unchanged; ssmax there is refused as before). "fixed" / "learned": that mode on the local
    # layers whatever the full layers use (the stage 2 + ssmax config: "learned", tau = 1 + w_h, since a local query
    # reads at most n_sinks + local_window keys and the length term ln(n_t) has nothing to track). "ssmax" is refused:
    # the local read's key count is not counted (ced/model/ced_v2.py).
    local_temperature: str = "inherit"

    def read_kinds(self) -> list[str]:
        """Per decoder layer "full" or "local" from read_pattern (repeated from layer 0)."""
        p = self.read_pattern
        if not p or set(p) - {"L", "F"}:
            raise ValueError(f"decoder.read_pattern {p!r}: a non-empty string of 'L' (local) and 'F' (full)")
        return ["local" if p[i % len(p)] == "L" else "full" for i in range(self.n_layers)]


@dataclass
class LoopConfig:
    """How the shared decoder block is looped."""

    # Largest loop count the parameters support (loop embedding table, per-loop HC parameters).
    t_max: int = 3
    # Training: T is drawn once per optimizer step from t_choices with probabilities t_probs.
    t_choices: list[int] = field(default_factory=lambda: [3])
    t_probs: list[float] = field(default_factory=lambda: [1.0])
    # Inference and evaluation use one fixed T for the whole sequence.
    t_eval: int = 3
    # Input injection at every loop: "parcae" (stable diagonal form, arXiv 2604.12946) or "none". With recur_layers
    # (the control's looped middle, below) also "mix": the scalar input mix of recur_mix_init, and "depth_mix": R3,
    # the softmax mix over [the embedding's output, the prelude's output, the earlier passes' outputs] that every pass
    # reads (ced/model/recur_depth.py); both refused everywhere else.
    injection: str = "parcae"
    parcae_init_decay: float = math.sqrt(0.2)
    # Zero-initialized learned per-loop embedding added to the block input (loop-index conditioning).
    loop_embedding: bool = True
    # Loop-boundary hyper-connections: "none", "static" (Zhu et al. form as used in
    # arXiv 2604.21106 App. G.1) or "dynamic_diag" (Hyperloop, arXiv 2604.21254). Independent of
    # `injection`; the Phase 2 arms are injection-only, HC-only (injection "none") and both.
    hyper_connections: str = "none"
    # Readout of intermediate loops through the tied LM head (LoopSpec drafts from loop 1):
    # an auxiliary cross-entropy on the loop-k readouts (k < T) at decoder positions, with this
    # weight relative to the main loss. 0 = the readout norms exist but are not trained.
    draft_loss_weight: float = 0.0
    # Depth attention (Dreamer, arXiv 2601.21582): "none" or "layer". With "layer", every decoder
    # layer application attends, per position, over that position's states at all earlier layer
    # applications of this step (all loops so far, the current one included): the paper's
    # granularity, where a "depth" is one layer application. The coarser per-loop reading is nearly
    # degenerate at T = 2 (one key at loop 1, two at loop 2) and overlaps with injection and HC.
    depth_attention: str = "none"
    depth_attention_head_dim: int = 128
    depth_attention_rope_theta: float = 500.0
    # Lanes / streams: 2 in arXiv 2604.21106, 4 in Hyperloop.
    hc_streams: int = 2
    # Static HC write: "delta" (block output minus what was read; reduces to the plain loop at
    # init) or "full" (the block's full output stream).
    hc_write: str = "delta"
    # Dynamic diagonal HC: initial sigmoid(b_res), the carry-over weight of each stream.
    hc_res_init: float = 0.05
    # CED v2 with decoder self-KV. How pass 2 enters:
    # - "none" (the default): no gate, every config before this key (the plain loop of the v2 runs: at T = 2 the
    #   shared static module with one row per pass, pass 2 written at weight 1; with decoder self-KV that loop is
    #   built too, and it cannot grow from a T = 1 checkpoint).
    # - "hc" (the proposal's "hyper-connections on"): pass 2 has its own static hyper-connection scalars, a read
    #   (hc_streams), a mix (hc_streams x hc_streams) and a write (hc_streams), initialised at read = the lane mean,
    #   mix = identity, write = 0, so alpha = the mean write weight = 0.
    # - "scalar" (the proposal's "gate only"): one trainable scalar alpha, initialised 0; pass 2's read is fixed at
    #   the lane mean and its mix at identity. In both, pass 1's tensors are exactly the T = 1 model's (same names,
    #   shapes and init: loop_embedding (1, d), hyper_connections.read_weights / mix_weights / write_weights with one
    #   row), and pass 2's are NEW tensors: hyper_connections.pass2_read / pass2_mix / pass2_write ("hc") or
    #   hyper_connections.pass2_gate ("scalar"), and loop_embedding_pass2 (d, zeros; with loop_embedding). At alpha =
    #   0 the two-pass output equals the one-pass output bit for bit, so a T = 1 checkpoint grows to two passes
    #   without changing its function (TrainConfig.init_from, TrainConfig.loop_switch_step). Needs t_max = 2,
    #   hyper_connections "static" and hc_write "delta"; refused elsewhere (ced/model/ced_v2.py). Arch "control" (the
    #   loop on the plain hybrid; ced/model/control.py, "Two passes"): "scalar" only (the control has no
    #   hyper-connection lanes, so "hc" is refused), with pass2_layers below. The control reads none of the other loop
    #   keys without a gate, and with one only t_max (2), t_choices ([2]), t_eval (1 or 2) and loop_embedding; its new
    #   tensors are hyper_connections.pass2_gate (the scalar alpha, 0) and loop_embedding_pass2 (d, zeros): the stage
    #   2 build's names, so the trainer and the optimizer treat them alike.
    pass2_gate: str = "none"
    # Segment-based pass counts (ced/model/pass_schedule.py, ced/train/pass_positions.py). None (the default): not
    # set. Every decoder position runs the run's loop count through the dense code path: every config before this
    # key, bit for bit. A map {reasoning, call_head, content, call_body, other} -> 1, 2 or 3 (a key left out counts
    # 2): the pass count of the position whose NEXT token lies in that segment, as
    # ced.eval.token_types.segment_positions labels the position from the tokens up to it. "other" is every position
    # whose next token is not a generated assistant token (prompt, tool output, plain web or code text) and every
    # position the state machine cannot place. Pass 1 runs at every decoder position; pass p runs only at the
    # positions whose count is at least p, gathered into slots, reads pass 1's K/V of the same layer and writes
    # none; the head reads the last pass a position ran. The map's largest value is the number of passes the model
    # is built for, and 3 is allowed ONLY through this map ("A third pass defined like the second. The spec's ban on
    # T > 2 is lifted for this experiment only."): a third pass has its own gate in the run's gate form and its own
    # loop-index embedding, started as the identity. Needs the gated two-pass decoder (pass2_gate "hc" / "scalar",
    # t_max 2); refused elsewhere.
    segment_passes: dict | None = None
    # With segment_passes: the tokenizer.json (a ByteLevel BPE with the chat format's special tokens at ids 0 .. 31)
    # whose vocabulary bytes the segment state machine reads (the end of a tool name; the role header of a message
    # where the caller gives no assistant flags). Required with a map; a path, so the same file must exist where the
    # checkpoint is scored.
    segment_tokenizer: str | None = None
    # A slot rule for the restricted regimes (the decoder at the answers, the sinks and the replay window): None
    # (the default): an unscored decoder slot runs the passes a scored one would (today's behaviour). 1: a slot that
    # is not scored (a sink, a replay-window position) runs pass 1 only. It still writes its pass-1 K/V, which is all
    # another slot can read of it, so every scored logit, the loss and every gradient are unchanged.
    unscored_passes: int | None = None
    # Arch "control" with pass2_gate "scalar" only: the looped block, as the number of TOP layers of the stack the
    # second pass runs through (layers n_layers - pass2_layers .. n_layers - 1). Control (a) has 20 layers: 8 is
    # layers 12 to 19 (the approved block), 4 is layers 16 to 19 (the optional arm). 0 (the default): no loop, the
    # control of every run before this key. Refused: with a gate, a value outside 1 .. n_layers; without one, or on
    # another arch, any value but 0 (CED v2's second pass is its whole decoder).
    pass2_layers: int = 0
    # The loop in the published form, arch "control" only (ced/model/control.py, "The loop in the published form"). ()
    # (the default): no loop; every model before this key, bit for bit (state dict, parameter order, RNG use). [first,
    # end]: the layers first .. end - 1 of the stack are the looped middle: the layers below it (the prelude) run
    # once, the middle runs T times with shared weights, each pass over the whole sequence with its OWN GDN states and
    # its own K/V in its global layers, and the layers from `end` on (the coda) run once. Always a tuple of two ints
    # here (a list from YAML or --set is turned into one). With it the control reads these loop keys and no other:
    # injection ("parcae" / "none" / "mix" / "depth_mix"), parcae_init_decay, recur_inject_b, recur_state_init,
    # recur_prelude_norm, recur_mix_init (injection "mix" only), t_choices / t_probs with t_sampling (training),
    # t_eval (the pass count of a forward that is given none; any T >= 1 can be asked for: no parameter depends on the
    # pass index, so t_max is not read). Refused together with pass2_gate (the gated second pass: the other loop of
    # this arch), encoder.layer_scalars, model.nvfp4, segment_passes / unscored_passes and a looped encoder
    # (ced/model/control.py check_recur_loop).
    recur_layers: tuple[int, ...] = ()
    # With recur_layers and injection "parcae": the matrix B of u = decay * h + dt * (B e). "dense" (the published
    # form: a d x d matrix at the identity) or "diag" (a (d,) vector at ones).
    recur_inject_b: str = "dense"
    # With recur_layers and injection "parcae": the state h_0 the first pass's injection reads. "zeros" or "prelude" (h_0 = e).
    # Without injection the first pass reads the prelude's output itself and the key is not read.
    recur_state_init: str = "zeros"
    # With recur_layers and injection "parcae": e = RMSNorm(prelude output) or, False, the prelude's output as it is. Not read without injection (no tensor is added).
    recur_prelude_norm: bool = True
    # With recur_layers and injection "mix" (the late switch-on, "growth"; ced/model/encoder.py recur_input_mix): the
    # value the one scalar encoder.recur_mix starts at. Every pass after the first reads w e + (1 - w) z_{k-1}, with the
    # same w and the same e (the prelude's output); the first pass reads e itself, as without injection. At 1.0 a later
    # pass reads e exactly and repeats the first pass: the model at T passes is then the one-pass model bit for bit. A
    # finite number; not read with any other injection (no tensor is added there).
    recur_mix_init: float = 0.99
    # How a training step draws its pass counts from t_choices / t_probs. "step" (the default, every run before the
    # key): one T per optimizer step. "sequence" (recur_layers only): one T per ROW; every step's rows hold exactly
    # t_probs[i] x data.batch_size rows of t_choices[i] (the products must be whole numbers), in an order drawn from
    # the counter RNG of (seed, step), and the step's micro-batches are formed from rows of equal T
    # (ced/train/trainer.py Trainer.row_passes).
    t_sampling: str = "step"

    def __setattr__(self, name, value):
        if name == "recur_layers" and isinstance(value, list):
            value = tuple(value)  # YAML and --set give lists; one type in memory, in to_dict and in checkpoints
        object.__setattr__(self, name, value)


@dataclass
class ResidualScaleConfig:
    """Fixed multipliers on residual branches (not learned, not in the state dict).

    "none": 1.0 everywhere (default until Phase 1 settles it).
    "completep_effective_depth": every branch gets lam * (depth_base / (L_enc + t_ref * L_dec)).
    "loop_factored": decoder branches get lam / (t_ref * sqrt(L_dec)) (arXiv 2606.18524);
        encoder branches get lam * depth_base / L_enc (CompleteP on the encoder's own depth).
    "inverse_loops": the looped block's branches get lam / t_ref (1/r); the other block gets 1.
        The 2026-10-03 default for the primary candidate.
    t_ref is a fixed loop count (default: max of t_choices), never the sampled T, so the block
    computes the same function for every T.
    """

    mode: str = "none"
    lam: float = 1.0
    depth_base: float = 1.0
    t_ref: int | None = None
    # loop_factored only: reference unique depth. arXiv 2606.18524 trains lam * N^-1 * (L / l_ref)^-1/2
    # with l_ref = 12 and lam = 1; lam / (N * sqrt(L)) with lam = 1 was never trained by anyone.
    l_ref: float = 12.0


@dataclass
class SecondPassConfig:
    """Second-pass control: a zero-write second pass through the encoder's own layers at the decoder positions (ced/model/second_pass.py)."""

    # Zero-initialised learned vector added to the pass-2 input h1_top(t), telling the shared layers
    # which pass they are in. False: pass 2 starts from h1_top(t) alone.
    pass_embedding: bool = True


@dataclass
class NVFP4Config:
    """Opt-in NVFP4 fake quantisation of the layers' linears (ced/model/nvfp4.py, which gives the recipe: NVIDIA's
    arXiv 2509.25149; the screen). Emulated on bf16 GEMMs: it measures quality only, never speed. Off by
    default; when off no module is touched and the model is the model it was (tests/test_nvfp4_screen.py:
    logits, one step's loss and gradients equal main's bit for bit). Built for arch "control" with the softmax
    normaliser; refused elsewhere (ced/model/control.py, ced/model/nvfp4.py).

    mode: "none" (the default: the switch is off; not "off", which YAML and the --set overrides read as false); "all_linears";
    "all_but_gdn" (every linear except those inside GDN layers, read literally: a GDN layer's mixer projections and its FFN stay
    BF16); "all_but_gdn_mixer" (only a GDN layer's five mixer projections stay BF16); "mlp_only" (only the FFN's three linears, in
    the layers that are not kept; no mixer linear is touched: the NVFP4 MLP arm). The output head and the embeddings always stay
    high precision. keep_last_fraction: the last round(f * n_layers) layers keep all their linears in BF16 (the recipe's sensitive
    final layers, about 0.15 in the paper). keep_first_layers: likewise the first n layers. stochastic_rounding (of gradients), rht
    (random Hadamard transform on the Wgrad inputs), weight_2d (16x16 weight blocks): the recipe's parts, on by default, switchable
    for tests and ablations. The quantised path runs in training mode only: every eval-mode forward (in-training evals, the
    re-score, the probes, cached generation) is the BF16 model, as the paper evaluates."""

    mode: str = "none"
    keep_last_fraction: float = 0.0
    keep_first_layers: int = 0
    stochastic_rounding: bool = True
    rht: bool = True
    weight_2d: bool = True
    # Linears kept in BF16 by NAME, in every layer, on top of the mode and the kept layers. Each entry is a
    # module-name suffix matched at a dot boundary: "in_proj_a" keeps encoder.layers.<i>.mixer.in_proj_a of every GDN
    # layer ("mixer.in_proj_a" or the full name also work). () (the default): no linear is kept by name, every run
    # before this key, bit for bit. An entry that names no linear of the layers is refused (a typo would otherwise
    # train another arm under this arm's name). Always a tuple of str here (YAML and --set give lists).
    keep_names: tuple[str, ...] = ()
    # The quantiser recipe of the quantised linears (ced/model/nvfp4.py). "nvidia" (the default): NVIDIA's recipe
    # above (arXiv 2509.25149), every run before this key, bit for bit. "quartet2": the Quartet II scheme (arXiv
    # 2601.22813; the authors' pseudo-quantisation reference, IST-DASLab/Quartet-II kernels/python/quartet2/
    # pseudoquant.py): Fprop round-to-nearest with 1x16 blocks for activations AND weights and the "four over six"
    # block-scale choice; Dgrad and Wgrad re-quantise the forward's quantised operands and the gradient with MS-EDEN
    # (a random Hadamard transform of 128 along the GEMM's inner dimension, fresh signs every backward, round to
    # nearest, then a per-block correction of the E4M3 scale rounded stochastically). stochastic_rounding, rht and
    # weight_2d are the nvidia recipe's switches and must keep their defaults under "quartet2".
    recipe: str = "nvidia"
    # Linears that run FP8 instead of NVFP4, by NAME (the NVFP4 tail arms' N1). Matched as keep_names is (a module-name
    # suffix at a dot boundary, e.g. "out_proj"), and only where the mode and the kept layers would quantise the linear:
    # there it runs the FP8 control's recipe (ced/model/fp8.py, FP8Config scaling "rowwise": E4M3, one power-of-two scale
    # per row of each GEMM operand along its dot product) in all three GEMMs, under either recipe; in a kept layer it stays
    # BF16 like every other linear there. () (the default): no linear runs FP8, every run before this key, bit for bit (no
    # module attribute is set). Refused: a name in keep_names too, a name that matches no linear, in_proj_a / in_proj_b
    # (fp8.py keeps them in high precision), a matched linear whose sizes are not multiples of 16, and the key with mode
    # "none". Always a tuple of str here (YAML and --set give lists).
    fp8_names: tuple[str, ...] = ()
    # The same function as the NVFP4 tail arms' stage "continue" with precision_switch (bit for bit: the run with its quantised linears turned off in
    # memory at update until_step, tests/test_nvfp4_mlp.py), for the runs that could not take a stage when it was built (a stage with TST was built on
    # a new shard set only; since 2026-10-29 it also takes the source's own). 0 (the default): NVFP4 for every update, every run before this key, bit
    # for bit (no module attribute is set). Refused: a negative or non-integer value, a value at or beyond the run's steps, and the key with mode
    # "none". Evals are BF16 whatever the phase.
    until_step: int = 0

    def __setattr__(self, name, value):
        if name in ("keep_names", "fp8_names") and isinstance(value, list):
            value = tuple(value)  # YAML and --set give lists; one type in memory, in to_dict and in checkpoints
        object.__setattr__(self, name, value)


@dataclass
class FP8Config:
    """Opt-in FP8 for the layers' linears (ced/model/fp8.py, which states the recipes). The reference: a fake
    quantisation whose GEMMs run on the FP8 values in a high-precision GEMM. It measures quality only, never speed.
    Off by default; when off no module is touched and the model is the model it was (tests/test_fp8.py: main's record
    at 09544e4, bit for bit). Built for arch "control" with one pass and the softmax normaliser, with
    model.encoder.attn_res "none", "block" or "block_entry" and every activation checkpoint mode the trainer allows
    for them; refused elsewhere (ced/model/fp8.py check_fp8).

    mode: "none" (the default: the switch is off; not "off", which YAML and the --set overrides read as false) or
    "linears": every nn.Linear of the layers' mixers and FFNs (attention q/k/v/o, the elementwise gate inside q_proj, GDN
    in_proj_qkv / in_proj_z / out_proj, the FFN), except GDN's in_proj_a and in_proj_b (one output per value head; on
    AdamW) and any linear whose input or output size is not a multiple of 16 (an FP8 GEMM's alignment; e.g. the headwise
    attention gate). Embeddings, the output head, norms, the GDN delta rule, conv and state, softmax and the depth mix keep
    their precision. scaling: "rowwise" (torchao's ROWWISE recipe: E4M3 for activations, weights and gradients; one
    power-of-two scale per row of each GEMM operand along its dot-product dimension, every operand quantised afresh for
    each GEMM), "tensorwise" (Transformer Engine's Float8CurrentScaling with the HYBRID format: E4M3 forward, E5M2
    gradients; one scale per tensor, each tensor quantised once) or "mxfp8" (OCP MXFP8 as TE's MXFP8BlockScaling runs it,
    E4M3 everywhere, one E8M0 power-of-two scale per block of 32 elements along each GEMM's dot product, by TE's rounding;
    ced/model/mxfp8.py). All use current scaling: the scale comes from the tensor being quantised, so there is no amax
    history and nothing new in the state dict or the checkpoint. backend: "emulated" (the default: the fake quantisation of
    ced/model/fp8.py, on any device) or "te" (Transformer Engine's MXFP8 te.Linear behind the same modules, with scaling
    "mxfp8" only; refused by name without TE and below sm100, ced/model/fp8_te.py). The same parameters and state dict
    either way; read only with mode "linears". te_nonfinite_check: with backend "te" only (refused otherwise: the emulated
    MXFP8 quantiser has the non-finite policy built in, ced/model/mxfp8.py): every +-inf of a quantised linear's activation
    and of its output's gradient becomes NaN before Transformer Engine quantises it, so TE's saturation of an infinity
    cannot hide it from the trainer's finite check. Finite values bit for bit. Off by default: it costs a read and a write
    of each operand where Inductor does not fuse it (ced/model/fp8_te.py). The quantised path runs in training mode only:
    every eval-mode forward (in-training evals, the re-score, the probes, cached generation) is the BF16 model, as the
    NVFP4 screen read its arms. The tuning pass, each off by default (the model is main's bit for bit): quantiser: "torch"
    (the default: the emulated linear quantises with ced/model/mxfp8.py's PyTorch reference) or "triton" (backend
    "emulated" with scaling "mxfp8" only: the same operands from our Triton quantiser, ced/kernels/mx_ops.py, both copies
    in one read; the same bytes, so the same GEMMs: the vehicle of the tuned kernels' 60M band check). te_operands: "te"
    (the default: B's te.Linear, where Transformer Engine quantises x, W and dy itself) or "ours" (backend "te" only:: TE's
    GEMMs receive only MXFP8 operands our kernels made; a raw tensor handed to them is refused by name;
    ced/model/fp8_te.py). nonfinite_flag: with quantiser "triton" or te_operands "ours" only (refused otherwise):, every
    quantised operand ORs a bit into one device int32 when a block holds an inf or NaN, from the block amaxes the kernel
    computes anyway; the trainer zeroes it each step and reads it with the gradient norm, and a set bit stops the run there
    (ced/train/trainer.py, "The non-finite flag"). nonfinite_flag_form: "pure" or "ordered" (default before it, kept
    selectable: the flagged quantiser as an ordered effectful operator). "pure" (the compiled
    60M MXFP8 arm bit for bit as the flag off over 500 steps): the same kernel and the same OR into the flag, declared to
    the compiler as a pure operator, so the compiled graphs are the flag off's; a call whose outputs are all unused may
    then be dropped with its bit (ced/kernels/mx_ops.py, "The pure form"), and none is: every quantised operand feeds a
    GEMM (tests/test_mxfp8_flag_operands.py). Read only with nonfinite_flag and quantiser "triton" (ced/model/fp8.py
    flag_form_of); inert, and either value accepted, elsewhere (the flag off; te_operands "ours", whose linears run
    eagerly). A checkpoint written before the key trained "ordered" (checkpoint_config); one written before "pure" became
    the default carries its form, "ordered" unless set; a run resumes only in the form it trained
    (ced/train/trainer.py load_state_dict). chains: the fused chains wired into training (ced/model/mx_wire.py states the
    forms). A tuple of chain names, () by default (no module is touched: the model is main's bit for bit): "in_norm" (each
    layer's mixer input norm makes the operand of the mixer's quantised in-projections), "exit_entry" (the mixer's output
    norm, the residual add and the MLP's input norm in one pass, making the operand of w_gate and w_up), "swiglu" (silu(g)
    u makes w_down's operand), "swiglu_bwd" (the SwiGLU backward makes the dy operands of w_gate and w_up), "attn_gate" (o
    sigmoid(gate) makes o_proj's operand), "gdn_out_norm" (the GDN gated output norm makes out_proj's operand). With
    quantiser "triton" only (our kernels' operands, backend "emulated"); refused with nonfinite_flag (no chain passes the
    flag yet), and wherever a form is not built (ced/model/fp8.py check_fp8). Always a tuple of str here (YAML and --set
    give lists). folds: the B200 step audit's folds on G0, our GEMM with prologue and epilogue hooks
    (ced/kernels/g0_gemm.py, ced/model/g0_wire.py). A tuple of fold names, () by default (no module is touched: the model
    is main's bit for bit): "swiglu_bwd" (the audit's c.swiglu_bwd: the SwiGLU backward and the MXFP8 quantisation of dg
    and du in the epilogue of the w_down Dgrad, which G0 runs; the forward unchanged); "norm_mlp" (audit's b.prologue at
    the MLP's input norm, so h is never written; the norm's backward in one pass after the Dgrads; with "swiglu_bwd" the
    two combine). Built for the 4B config's path only (backend "te", te_operands "ours", scaling "mxfp8", mode "linears";
    the non-finite flag on or off); refused elsewhere by name (ced/model/g0_wire.py check_folds), with any of
    model.fp8.chains (in_norm and exit_entry are built on this path since, but not with the folds) and with
    model.fused_projections "gate_up" (both take the SwiGLU's w_gate and w_up GEMMs). Nothing in the state dict:
    checkpoints load either way, and at its default () the key is left out of a checkpoint's saved config
    (ced/train/trainer.py SAVED_FP8_DEFAULTS). Always a tuple of str here."""

    mode: str = "none"
    scaling: str = "rowwise"
    backend: str = "emulated"
    te_nonfinite_check: bool = False
    quantiser: str = "torch"
    te_operands: str = "te"
    nonfinite_flag: bool = False
    nonfinite_flag_form: str = "pure"  # ced/model/fp8.py FLAG_FORM_DEFAULT
    chains: tuple[str, ...] = ()
    folds: tuple[str, ...] = ()

    def __setattr__(self, name, value):
        if name in ("chains", "folds") and isinstance(value, list):
            value = tuple(value)  # YAML and --set give lists; one type in memory, in to_dict and in checkpoints
        object.__setattr__(self, name, value)


@dataclass
class MTPConfig:
    """Opt-in multi-token-prediction drafter on a FROZEN trunk (ced/model/mtp.py, which states the form). Off by
    default: no module is built, and the model, its state dict, parameter order and RNG use are the model without the
    key (tests/test_mtp.py).

    enabled: build the head as the model's `mtp` submodule (arch "control", one pass; refused elsewhere: mtp.check_mtp).
      It is trained only by scripts/train_mtp.py with every trunk tensor frozen; the trainer refuses the key.
    block: the head's one block. "attention" (the only form built): one of the trunk's own global-attention layers
      (ced/model/encoder.py EncoderLayer, Peri-LN, the trunk's model.encoder.attention settings and SwiGLU FFN of
      model.ffn_hidden), causal over the head's own positions, with a key/value cache in decoding."""

    enabled: bool = False
    block: str = "attention"


@dataclass
class CPHeadConfig:
    """Opt-in Copy-Paste head (CPLM: Godey & Artzi, arXiv 2610.19247; ced/model/cp_head.py states the form). The
    next-token distribution becomes p = (1 - lambda) p_LM + lambda p_CP, lambda = c (1 - s). Off by default: no tensor
    is built, no random number is drawn, and the model, its state dict, its outputs and its training are the model
    without the key, bit for bit (tests/test_cp_head.py, against main's tree 61920b2d).

    enabled: build the head (`model.cp_head`: W_Q, W_K (d_c x d_model), the sink key (d_c,) and the <copy> row of the LM
      head as its own vector w_copy (d_model,)). Arch "control" with one pass only, on the base (encoder.attn_res "none")
      or on R1-entry's own form ("block_entry"; the cp-r1e arms, 2026-10-18), there also with the write mix's "layer" form
      (encoder.gdn_write_mix "layer"; the wsl-nowd-cp-r1e arms, 2026-10-18); refused elsewhere: check_cp_head. d_c: the
      query/key width (the spec: 64). query_chunk: query positions per chunk of the pure-PyTorch reference (the [B,
      query_chunk, T] score blocks); any value >= 1, it need not divide T. It changes how the same numbers are computed,
      not what they are. force_lambda_zero: lambda = 0 everywhere: p = p_LM. The loss and every base tensor's gradient are
      the base model's bit for bit, and the CP tensors get no gradient. For the tests and the GPU check; the trainer
      applies the same rule by itself in every TST bag phase (s > 1). score: what the model's next-token output is at
      inference and in every scoring path: "mixture" (the default: log p, the trained distribution) or "lm" (p_LM alone:
      the LM head over the real vocabulary, without <copy>). The environment variable CED_CP_SCORE overrides it in a
      scoring process (cp_head.ENV_SCORE). Training always trains the mixture (under train_mode "mixture").

    cp-aux (ced/model/cp_aux.py states the form). Both keys at their defaults: the head as it was, bit for bit. train_mode:
    "mixture" (the default: the prediction is the mixture and the loss its CE, everything above) or "aux": the pointer as in cp-mix
    (pointer_input "mix" only), trained only through its own auxiliary loss, the cross-entropy of the pointer's softmax over {sink}
    u ctx(t) against the positions whose token equals the next token (the sink when none is in the context); the main loss is
    p_LM's CE alone, nothing is mixed into the prediction (no <copy> row is built), and the pointer is not run at evaluation: every
    scoring path, probe and generation reads p_LM alone (score is not read). aux_weight: the auxiliary loss's weight (train_mode
    "aux" only, where it must be > 0; 0.0, the default, otherwise). The auxiliary loss is averaged over the main loss's positions
    (the valid targets) and its weight multiplies that mean. train_mode "addon" (ced/model/cp_aux.py, "ADD-ON CP"): cp-mix's head
    whole (pointer_input "mix" only), the mixture at evaluation; in training the trunk and the LM head learn from R1-entry's own
    loss on p_LM alone, the pointer's inputs and the <copy> row's are detached from the trunk, and the pointer and the gate learn
    from the mixture's CE with p_LM's numbers detached; the head's gradients are clipped apart from the trunk's.

    The reference implementation's two collapse fixes (PR #379, NathanGodey/modded-nanogpt @ c44cc41e). Both off by
    default: off, the head is the head without the keys, bit for bit (tests/test_cp_collapse_fixes.py, against main's
    tree 120aa0f0). qk_norm: RMS-norm q and k over their d_c axis (eps 2^-23, the reference's float32 default), q times a
    learnable scalar gain (cp_head.qk_gain, stored zero-centred: the gain is 1 + qk_gain, 1 at the start), before the
    scores' 1/sqrt(d_c); the sink key is not normed (model/gpt.py L925-927). Every path that computes the pointer reads
    it. init: "spec" or "reference" (the record's: W_Q and W_K U(-sqrt(3/D), +sqrt(3/D)) with D = d_model, the
    projections' fan-in; the sink key N(0, 1/d_c); model/gpt.py L238-245). Both from the head's own generator
    (cp_head.own_generator). w_copy is today's draw under both (the record's <copy> is a row of its LM head under the
    head's own init: ours is N(0, init_std)).

    The fused pointer (ced/kernels/cp_pointer.py). Both keys at their defaults: the head as it was, bit for bit. kernel:
    "reference" (the pure-PyTorch head above: the ground truth, and the head of every CP run trained before this key) or
    "fused": the stacked q/k/<copy> GEMM, the pointer as a Triton kernel pair over the rows' document segments (no T x T
    tensor in HBM; one compiled kernel for every row length), the mixture inside the head's CE kernel. The same function
    within float32 rounding, not bit for bit (its band check decides). Training needs head_impl "fused". window: the
    pointer's reach. 0 (the default: the spec's "every earlier position of the document"): every position of the query's
    document up to and including the query itself, in the row. W >= 1: the band, the W latest positions of the document,
    the query itself included (idx_t - idx_tau < W); 2048 is the reference implementation's band (PR #379 COPY_BLOCK),
    an option only. Every path reads it: the training terms, scoring, cached generation, the readings. The CP runs
    trained before the key had no band (their head read every earlier position of the document): a config without the
    key keeps its meaning.

    The pointer's input (2026-10-18: two 60M arms on the R1-entry trunk, cp-mix and cp-shared;
    runs/arch-cp-mix-shared-60m.yaml). At the default the head is the head without the keys, bit for bit; the <copy> row
    and the sink are cp-r1e's under every value. pointer_input: what the pointer's query and key read. "final" (the
    default: h_t, the final norm's output, as every CP run before the key). "mix" (cp-mix): W_Q and W_K each read their
    own depth-mixing reader over R1-entry's output read's sources [b_0 (the embedding), b_1 .. b_N (the blocks' sums)],
    R1-entry's reader form (one (d,) pseudo-query, an RMSNorm on its keys, one softmax over the sources; zeros at the
    start), then an RMSNorm of their own before the projection (as a block's sublayer norms its read): R1-entry only.
    "shared" (cp-shared): no W_Q and W_K; the scores are one global attention head's own q. k / sqrt(head_dim) (its
    QK-normed q and its key head's QK-normed k, NoPE), times the pointer's gain 1 + qk_gain; the sink key lives in that
    head's key space (head_dim wide). R1-entry only, with qk_norm (the pointer's gain) and the trunk's attention QK-norm.
    shared_layer (with "shared"): the encoder stack index of that global layer; -1 (the default): the stack's last global
    layer. It must be a layer without a rotary part (a NoPE global layer: encoder.nope_global_layers): layer 19 at 60M.
    shared_head (with "shared"): the query head of that layer (0 .. n_q_heads - 1; the default 0); its key head is the one
    GQA assigns it (head // (n_q_heads / n_kv_heads)). shared_ssmax (with "shared"): false (the default): the pointer's
    logits and the sink's are the head's QK-normed q. k without the layer's temperature (cp-r1e's form: the length-robust
    gate is a separate decision); true: they inherit it, q after the layer's temperature (SSMax's (1 + w_h) ln(n_t) /
    ln(ref) when the layer has it), the sink's logit scaled with them.

    The length-robust pointer (design, candidates (a) and (b); ced/model/cp_head.py, "THE LENGTH RULE"). Both off by default:
    off, the head is the head without the keys, bit for bit. n_t is the number of context positions query t sees (its
    document's positions tau <= t in the row: pos_t + 1, the trunk's ssmax_factor count), ref the global layers'
    model.encoder.attention.temperature_ref_len (8192 at 60M). temperature: "none" (the default: no factor) or "ssmax":
    candidate (a), the global layers' Scalable-Softmax on the pointer's query, q_t <- sigma_t q_t with sigma_t = ln(n_t) /
    ln(ref) (attention.key_count_factor: exactly 1 at n = ref; 0 at the first position; unclipped, as the trunk's), after
    QK-norm and the gain, so the context logits and the sink's logit both scale. The trunk's learned per-head scale (1 + w_h)
    is the pointer's own gain 1 + qk_gain (one head): no new tensor. Refused with pointer_input "shared" (shared_ssmax is that
    form there) and with window (the count is not pos + 1). sink_count: false (the default) or true: candidate (b), e_sink_t <-
    e_sink_t + ln(n_t / ref), added after the gain and the scale (unclipped). No new tensor. Refused with window.

    The pointer's heads (ced/model/cp_head.py, "THE POINTER'S HEADS"; runs/arch-cp-grid-60m.yaml). Both at their defaults: the
    head is the head without the keys, bit for bit (tests/test_cp_grid.py, against main's tree 22a5fc03). SSMax on the pointer is
    the key above, temperature "ssmax". n_heads: the pointer's heads (1, the default: one head, every CP run before the key). H
    >= 2: each head h has its own depth-mixed query and key (its own PointerMix readers and norms, as cp-mix has one), its own
    W_Q / W_K rows (d_c x d_model each: d_c is the width of ONE head, 64 at 60M), its own QK-norm gain and its own sink key
    (d_c). Built with pointer_input "mix" and qk_norm only; refused with the fused kernel, a band and sink_count. combine: how
    the heads make one pointer. "and" (the default; cp-mix-mha): the heads' scores summed into one logit per position before one
    softmax over {sink} u ctx, e_tau = sum_h q_h. k_h,tau / sqrt(d_c), and the sink's logit the sum of the heads' sink logits,
    e_sink = sum_h q_h. k_sink,h / sqrt(d_c) (one H x d_c wide head with a block-wise QK-norm: a position must match every head).
    "or" (cp-mix-mho): each head its own softmax over {sink_h} u ctx, mixed by input-dependent weights pi_t = softmax_h(W_route
    h_t) (W_route (H, d_model), zeros at the start: the uniform mixture; it reads h_t, the final norm's output, the <copy> row's
    input). The mixture over {sink} u ctx is P(sink) = sum_h pi_h s_h, P(tau) = sum_h pi_h (1 - s_h) alpha_h(tau), so s = sum_h
    pi_h s_h, 1 - s = sum_h pi_h (1 - s_h) and p_CP = sum_h w_h p_CP,h with w_h = pi_h (1 - s_h) / (1 - s); lambda = c (1 - s)
    keeps its meaning. With n_heads 1 both are cp-mix exactly (tested).

    The trunk copy (ced/model/cp_trunk.py states the form; design note; runs/arch-cp-trunk-r1e-60m.yaml). All three at their defaults:
    the head is the head without the keys, bit for bit (tests/test_cp_trunk.py, against main's tree 23c18140). trunk_copy: "none" (the
    default) or the form: no pointer of its own (no W_Q, W_K, sink key: the keys d_c, qk_norm, init, kernel, window, pointer_input,
    temperature, sink_count, n_heads and combine stay at their defaults); s and p_CP are read from the encoder's global attention
    heads' own attention probabilities in the forward, shifted +1 (attending key j copies x_{j+1}; the diagonal and the document's
    first position are "no copy", CP's sink), mixed by learned weights over the heads (cp_head. trunk_logits, zeros); the <copy> row,
    lambda = c (1 - s) and the mixture are the head's. R1-entry's own form only. "all": every global head for the whole run. "subset":
    all heads for trunk_select_step steps, then the trunk_k heads with the largest learned logits (once; kept in the state dict as
    cp_head.trunk_keep). trunk_k ("subset" only): the subset's heads, a multiple of the global layers' query heads, fewer than all of
    them. trunk_select_step ("subset" only): the optimizer step at whose start the subset is selected, >= 1. kernel under the trunk
    copy: "reference" (cp_trunk.py's block function, the ground truth), "fused" (ced/kernels/cp_trunk_fused.py: one pass over the row
    a group of heads, forward and backward, against the flex read's log-sum-exp) or "fused2" (the trunk copy's only: the fast path,
    ced/kernels/cp_trunk_fused2.py, the same terms by the second kernels)."""

    enabled: bool = False
    d_c: int = 64
    query_chunk: int = 512
    force_lambda_zero: bool = False
    score: str = "mixture"
    train_mode: str = "mixture"  # cp-aux: "aux" (ced/model/cp_aux.py)
    aux_weight: float = 0.0  # cp-aux: the auxiliary loss's weight (train_mode "aux" only)
    qk_norm: bool = False
    init: str = "spec"
    kernel: str = "reference"
    window: int = 0
    pointer_input: str = "final"
    shared_layer: int = -1
    shared_head: int = 0
    shared_ssmax: bool = False
    temperature: str = "none"
    sink_count: bool = False
    n_heads: int = 1
    combine: str = "and"
    trunk_copy: str = "none"
    trunk_k: int = 0
    trunk_select_step: int = 0


@dataclass
class ModelConfig:
    # "looped_ced" (v1), "control" (our own encoder block stack: no loop, no encoder/decoder split;
    # controls (a) and (b) of v2), "ced_v2" (architecture spec v2: encoder + looped decoder that reads
    # only an encoder-written memory, ced/model/ced_v2.py) or "second_pass" (control (c),
    # ced/model/second_pass.py).
    arch: str = "looped_ced"
    vocab_size: int = 128000
    # True (v1): the LM heads use the input embedding matrix. False (v2): a separate output
    # embedding (vocab x d_model) shared by every head of the model. See head_weight() on the models.
    tie_embeddings: bool = True
    d_model: int = 2560
    ffn_hidden: int = 7168
    # The dense FFN of every encoder and decoder layer (ced/model/mlp.py, dense_ffn): "swiglu" (v1 and v2):
    # W_down (silu(W_gate x) * W_up x), three matrices; "relu2": W_down relu(W_up x)^2, two matrices (ReLU
    # squared, arXiv 2109.08668). A routed decoder FFN stays SwiGLU and is refused with it.
    ffn_activation: str = "swiglu"
    norm_eps: float = 1e-6
    # Peri-LN's optional norm on the embedding output.
    embed_norm: bool = True
    init_std: float = 0.02
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    residual_scale: ResidualScaleConfig = field(default_factory=ResidualScaleConfig)
    second_pass: SecondPassConfig = field(default_factory=SecondPassConfig)
    # NVFP4 fake quantisation of the layers' linears (NVFP4Config; off by default: nothing changes).
    nvfp4: NVFP4Config = field(default_factory=NVFP4Config)
    # FP8 for the layers' linears (FP8Config; off by default: nothing changes).
    fp8: FP8Config = field(default_factory=FP8Config)
    # The frozen-trunk MTP drafter (MTPConfig; off by default: nothing is built).
    mtp: MTPConfig = field(default_factory=MTPConfig)
    # The Copy-Paste head (CPHeadConfig; off by default: nothing is built).
    cp_head: CPHeadConfig = field(default_factory=CPHeadConfig)
    # other.concat of the B200 step audit (ced/model/fused_proj.py states it): the projections that read the same input
    # run as ONE GEMM in a training forward. A tuple of group names, () by default (no module is touched: the model is
    # main's bit for bit): "gate_up" (SwiGLU w_gate | w_up), "qkv" (attention q_proj | k_proj | v_proj), "qkv_z" (GDN
    # in_proj_qkv | in_proj_z), "ab" (GDN in_proj_b | in_proj_a, kept in high precision). The same parameters, names,
    # state dict and optimizer; the same function (the GEMM sums in another order: fused_proj.py). Refused by name with
    # any model.fp8.chains and, for "gate_up", with model.fp8.folds "swiglu_bwd" / "norm_mlp" (the SwiGLU runs one path:
    # ced/model/mlp.py); "qkv", "qkv_z" and "ab" meet the folds in no module. At its default () the key is left out of a
    # checkpoint's saved config (ced/train/trainer.py SAVED_MODEL_DEFAULTS). Always a tuple of str here (YAML and --set
    # give lists).
    fused_projections: tuple[str, ...] = ()

    def __setattr__(self, name, value):
        if name == "fused_projections" and isinstance(value, list):
            value = tuple(value)  # YAML and --set give lists; one type in memory, in to_dict and in checkpoints
        object.__setattr__(self, name, value)

    def t_ref(self) -> int:
        if self.residual_scale.t_ref is not None:
            return self.residual_scale.t_ref
        return max(self.loop.t_choices)


PASS2_GATES = ("none", "hc", "scalar")


def recur_loop(model: ModelConfig) -> bool:
    """Whether the config is the control with the loop in the published form (LoopConfig.recur_layers on arch
    "control"; ced/model/control.py, "The loop in the published form"): a prelude, a looped middle whose every pass
    writes its own states and K/V, a coda. Not the gated second pass (looped_control below)."""
    return model.arch == "control" and bool(model.loop.recur_layers)


def recur_mix(model: ModelConfig) -> bool:
    """Whether the config is the control with a looped middle whose later passes read the scalar input mix
    (LoopConfig.recur_layers with injection "mix"; ced/model/encoder.py recur_input_mix): the one form of the loop in
    the published form that can be switched on inside a run (TrainConfig.loop_switch_step), since at T = 1 it is the
    plain control and its one new tensor, encoder.recur_mix, is read by the later passes only."""
    return recur_loop(model) and model.loop.injection == "mix"


def looped_control(model: ModelConfig) -> bool:
    """Whether the config is the control with a gated second pass through its top block (LoopConfig.pass2_gate on
    arch "control"; ced/model/control.py, "Two passes"): the one control that takes a loop count."""
    return model.arch == "control" and model.loop.pass2_gate != "none"


def one_pass_model(model: ModelConfig) -> ModelConfig:
    """The single-pass (T = 1) model a gated two-pass model (LoopConfig.pass2_gate "hc" / "scalar") grows from: the
    same config with the loop at T = 1, no gate and no pass schedule (segment_passes, segment_tokenizer and
    unscored_passes are read by the later passes only). Every other key is kept as it is, residual_scale.t_ref included
    (a t_ref of None follows the loop count, so the decoder's residual scale would change with T: the growth checks
    compare ced.model.scaling.residual_scales of the two configs and refuse a change).
    Arch "control": the same config without the gate, without the looped block and without a pass schedule
    (pass2_gate "none", pass2_layers 0, segment_passes / segment_tokenizer None). The control without a gate reads no
    other loop key, and the control configs carry v2's (t_max 2, t_eval 2: configs/v2/control-a-60m.yaml), so they
    are left as they are: the single-pass model of a looped control is the control its trunk was trained as, key for
    key.
    Arch "control" with a looped middle and the scalar input mix (recur_mix above; the late switch-on of 2026-10-13):
    the plain control it grows from, which is that config without the looped middle and without the mix:
    recur_layers (), injection "none" (the value of a control that injects nothing: configs/v2/control-a-60m.yaml
    and every 60M control's config have it) and recur_mix_init at its default. The other loop keys the looped middle
    reads (t_choices, t_probs, t_sampling, t_eval) are left as they are: the plain control reads none of them, and a
    branch gives them the values its trunk's config carries (control (a): t_choices [2], t_eval 2). A looped middle
    with any other injection is returned as it is: it has no one-pass model here and no switch step
    (ced/train/trainer.py refuses one)."""
    if model.arch == "control":
        loop = dataclasses.replace(
            model.loop, pass2_gate="none", pass2_layers=0, segment_passes=None, segment_tokenizer=None,
            unscored_passes=None)
        if recur_mix(model):
            loop = dataclasses.replace(loop, recur_layers=(), injection="none", recur_mix_init=LoopConfig().recur_mix_init)
        return dataclasses.replace(model, loop=loop)
    loop = dataclasses.replace(model.loop, t_max=1, t_choices=[1], t_probs=[1.0], t_eval=1, pass2_gate="none",
                               segment_passes=None, segment_tokenizer=None, unscored_passes=None)
    return dataclasses.replace(model, loop=loop)


@dataclass
class DataConfig:
    # "synthetic" (ced/train/data.py) or "shards" (tokenized shard set, ced/train/shards.py)
    kind: str = "synthetic"
    shard_dir: str | None = None
    # Shards only: more shard sets read together with shard_dir as ONE training stream (the two-set loader;
    # ced/data/shard_format.py ShardUnion). The union's documents are shard_dir's, then each extra set's, in this
    # order, and the per-seed epoch permutation runs over all of them; the data identity (fingerprint) covers every
    # set and their order. [] (the default): shard_dir alone, every run before the key bit for bit.
    extra_shard_dirs: list[str] = field(default_factory=list)
    # The order the union's FIRST epoch is read in (with extra_shard_dirs only; ced/data/shard_format.py UNION_ORDERS; long-data,
    # 2026-10-30; section 12). "permute" (the default): one permutation over every document of every set, as the loader was
    # built, every run before the key bit for bit. "sets_in_order": shard_dir in its own epoch-0 permutation (the one a run on
    # shard_dir alone draws, token for token), then the first extra set in a permutation of its own, and so on, so that no
    # document is read twice before every set has been read once, and a run that trained on shard_dir alone goes on in the union
    # where it stopped (a stage "continue": the data relation "extends", ced/train/trainer.py stage_data_relation). Later epochs
    # are one permutation over every document in both orders. The order is part of the data identity (the union's fingerprint)
    # unless it is "permute".
    union_order: str = "permute"
    # Share of non-chat positions that get random decoder spans (shards only; chat documents use roles).
    span_fraction: float = 0.25
    seq_len: int = 64
    batch_size: int = 2
    doc_len_min: int = 8
    doc_len_max: int = 40
    # Pretraining decoder spans (decided 2026-10-03, from the measured assistant-turn lengths):
    # log-normal lengths with the given median and sigma (natural log), truncated by rejection to
    # [span_min, min(C, span_max)]. Exactly C decoder positions per row, placed independently of
    # tokens and document boundaries. "uniform" (lengths uniform in [span_min, span_max]) is kept
    # for comparisons.
    span_distribution: str = "lognormal"
    span_median: float = 230.0
    span_sigma: float = 1.16
    span_min: int = 16
    span_max: int = 4096
    seed: int = 1234
    # Shards only. True: a chat document (the v1 builder splits them into pieces of at most seq_len
    # tokens at turn boundaries) is never cut by a row boundary. When one does not fit in the rest
    # of the current row, that space is filled with the head of the next non-chat document of the
    # permutation, the chat document starts the next row, and that non-chat document's tail follows
    # it. Every token is still used once per epoch. False: plain concatenate-and-cut (v0 shards).
    chat_whole: bool = False
    # Shards only, with chat_whole. True: no row has more decoder positions than the decoder capacity
    # C (the trainer passes the C of its capacity formula, the same for both arms):
    # - a chat piece whose assistant tokens would push its row's assistant tokens past C starts the
    #   next row instead (like a piece that does not fit);
    # - random pretraining spans yield: when a row's assistant positions plus its kept span positions
    #   exceed C, span positions are removed from the end of the row until it fits.
    # The token stream and every assistant position are unchanged; without it, keep-first-C drops a
    # row's late decoder positions, which in chat rows are assistant tokens.
    fit_decoder_capacity: bool = False
    # Where the decoder runs in training (ced/train/positions.py):
    # - "data" (v1): the data's decoder mask, spans and roles from the packer (above);
    # - "random": exactly C positions per row (C = the decoder capacity), uniform without replacement among the row's
    #   valid-target positions, from the counter RNG of (seed, step, row), so a pure function of the step. A row with
    #   fewer valid targets than C gets all of them (logged). Chat and non-chat positions alike. The controls get the
    #   same mask; they train on every valid target and only report ce_dec there.
    # - "answer_spans" (stage 3 adaptation; ced/train/positions.py answer_span_mask): the decoder (its own K/V and the
    #   main loss) runs only on "answer" positions, and every other position has its memory entry only (no decoder K/V,
    #   no main loss). Answers are the assistant positions of chat documents (the packer's roles) and, in other
    #   documents, contiguous spans with the span_* lengths that start only where the document already has
    #   answer_min_context earlier positions in the row, after a geometric gap of mean answer_gap_mean such positions,
    #   and stop at the document's end. Whether a position is an answer depends only on the RNG and on the document
    #   structure up to it (never on where a document ends), so a slot never learns the future from which positions are
    #   slots. The trainer runs C = decoder_capacity(L, answer_capacity_fraction, model.decoder.capacity_multiple) slots
    #   per row (keep-first-C: answers past C become context, logged as n_dropped, and since they are a row's LATE
    #   answers, drops bias against long contexts; 1.0, the default, has none), passed to the model when it differs from
    #   the model's own capacity, so the model config (and init_from's strict check) stays the pretrained model's. CED
    #   v2 (and the controls, which only report ce_dec there). The answer_* fields below are read in this mode only.
    #   Defaults: the context before a first assistant turn and the gap between turns in the measured trajectories are
    #   about 0.6-1.4k and 0.1-0.6k tokens; with spans of mean about 412 tokens, a gap
    #   of mean 1,024 eligible positions puts about 29% of a long document's eligible positions in answers. answer_sinks
    #   ("Sinks": the restricted regime keeps them, and the adaptation run uses that regime): CED v2 also runs the
    #   decoder at each document's first decoder.n_sinks positions (document position < n_sinks in the row), which write
    #   their own K/V so the local layers see their sinks as in training (and the full layers see those keys too); they
    #   are not scored (their target_valid is cleared unless they are answers), as scripts/eval_probes.py --decoder-at
    #   answers_sinks. False: the regime without sinks. answer_replay_window ("replay window": the zero-shot condition
    #   is met, so the restricted regime of the adaptation run is answers + sinks + replay window): W > 0: CED v2 also
    #   runs the decoder at the min(W, available) positions of the same document immediately before each answer span's
    #   first position (before each assistant turn of a chat document, before each drawn span elsewhere), a bounded
    #   replay of the decoder's sliding-window state before generation. Those positions write decoder K/V like the sinks
    #   and are not scored; one that is itself an answer or a sink stays what it is. As scripts/eval_probes.py
    #   --decoder-at answers_sinks_replay --replay-window W. 0 (the default): none, the masks of every earlier run bit
    #   for bit. What stays causal and what does not: ced/train/positions.py, "Replay window".
    decoder_positions: str = "data"
    answer_min_context: int = 1024
    answer_gap_mean: float = 1024.0
    answer_capacity_fraction: float = 1.0
    answer_sinks: bool = True
    answer_replay_window: int = 0
    # The per-token loss mask. Shards only. Which targets keep their loss, decided per TARGET token
    # (position t keeps the loss on token t + 1) from that token's place in the data stream, always with target_valid:
    #   "all" (the default; not "off", which YAML reads as false): every valid target, every run before the key bit for bit.
    #   "assistant": in chat documents only the assistant's tokens (generated.bin); every target of other documents.
    #   "spans_full", "spans_head_tail", "spans_mask": by the token's loss-span class (loss_spans.bin; absent means class 0;
    #     ced/data/shard_format.py): MA-full keeps classes 0, 1 and 2; the head+tail arms (MA, MV, MN) keep 0 and 2; MA-mask
    #     keeps 0. Class 3 is never kept.
    # Every loss branch and every denominator reads the kept positions (the mean over them; a batch with none gives loss 0).
    # The decoder and aux positions, the pass schedule and every scorer read target_valid as before: scoring never applies
    # the training mask. With a mask on, the set's invariants are checked when it is opened (check_loss_spans), the run's
    # data identity names the mode and the span files (a resume on another is refused), and the metrics hold n_loss and
    # n_target_valid. Refused with synthetic data, TST (tst.schedule) and data.fit_decoder_capacity. A stage that
    # continues another run on a new shard set may set it (TrainConfig.stage).
    loss_mask: str = "all"


@dataclass
class OursConfig:
    """Our own optimizer on the ANVIL-II chassis, M1 to M4, and the two baselines of its mechanism test 1 (answers of
    2026-10-16, points 1 to 4; ced/train/matrix_optim.py). Every flag is off; every knob is read only with its flag
    (or its optim.name), so a config never names a setting that does nothing (ced/train/optim.py check_ours). All at
    the defaults: the run without the key, bit for bit (tests/test_our_optimizer.py)."""
    # M1, spherical geometry on the scale-invariant matrices of the Muon group (v_proj, k_proj, w_up, w_down, GDN
    # out_proj; ced/model/invariance.py names them): each keeps its Frobenius norm R of its first step and steps along
    # the arc by theta(t) = m1_theta x lr(t) / peak lr (the run's warm-up, plateau and decay): W <- cos(theta) W -
    # sin(theta) R O_t / ||O_t||, O_t the update's tangent part; no weight decay on them. Read under optim.name
    # "anvil2_muon" (the chassis) and "muon" (today's Muon through MatrixWithAdamW).
    m1: bool = False
    m1_theta: float = 0.0  # the peak angle in radians a step (> 0 with m1)
    # M2, the long slow rail on the chassis' momentum (optim.name "anvil2_muon" only): m_L <- lerp(m_L, G, 1 - beta_L(t)),
    # added to the chassis' look-ahead as alpha(t) x [m_L agrees in sign with the chassis' momentum] x m_L, with
    # AdEMAMix's alpha and beta schedulers (arXiv 2409.03137) over m2_warmup steps.
    m2: bool = False
    m2_beta: float = 0.0  # the horizon value; 0.0: by horizon, 1 - 1 / (m2_memory x the run's optimizer steps)
    m2_memory: float = 0.08  # the memory as a fraction of the run
    m2_alpha: float = 5.0  # AdEMAMix's alpha
    m2_warmup: int = 0  # T_alpha = T_beta in steps; 0: the run's optimizer steps (AdEMAMix: "we typically use T")
    # M3, sharp-subspace damping (optim.name "anvil2_muon" or "muon"): EMA factors L = E[G G^T], R = E[G^T G] of the raw
    # gradient; every m3_every steps the top-m3_k eigenvectors by subspace iteration; c = clip((mean eigenvalue /
    # lambda)^p, m3_cmin, 1) per side; O <- (I - U diag(1 - c_L) U^T) O (I - V diag(1 - c_R) V^T).
    m3: bool = False
    m3_p: float = 0.25
    m3_k: int = 16
    m3_every: int = 20
    m3_beta: float = 0.95
    m3_cmin: float = 0.1
    m3_snr: bool = False  # c scaled by sqrt(rho / (1 + rho)), rho M2's per-direction signal-to-noise (needs m2)
    # M4, phase-aware state: inside reset_moments() (the TST switch's one call) the long slow rail is zeroed and its
    # schedulers restart, the input-side factor (R = E[G^T G]) of the first m4_layers layers' matrices is re-warmed, the
    # other factors are kept; the chassis' fast rail is zeroed only with m4_reset_fast (the TST rule's reset; the design
    # note: "fast-rail reset is optional").
    m4: bool = False
    m4_reset_fast: bool = True
    m4_layers: int = 4
    # Diagnostics, read-only: M2's per-matrix noise estimate, M3's spectra and M1's angle every log_every optimizer steps
    # (metrics "ours"); the measured rotation of every Muon-group matrix every rotation_log_every steps under ANY
    # optimizer (metrics "rotation"; ced/train/trainer.py). 0: off.
    log_every: int = 0
    rotation_log_every: int = 0
    # Baselines of mechanism test 1, on today's Muon: Muon-LITE-L (optim.name "muon_lite_l"; arXiv 2602.22681 eq. 5 with
    # beta 0 and appendix B.1; github.com/SHUCHENZHU/LITE muonlite.py) and SAMuon-lite (optim.name "samuon_lite"; arXiv
    # 2608.25990 eq. 8 with k = 1, Algorithm 1 variant 2, appendix G).
    lite_chi: float = 2.0
    lite_ratio: float = 0.1
    samuon_gamma: float = 7.07
    samuon_warmup: float = 0.3
    samuon_iters: int = 10


@dataclass
class OptimConfig:
    # "adamw": AdamW on every parameter. "muon": Muon on the 2-D hidden weight matrices and AdamW
    # on the rest (embeddings/head, norms, GDN scalars, convolutions and gates, loop-boundary
    # parameters); see ced/train/optim.py. "kl_soap", "shampoo", "normuon", "anvil2_muon": another matrix optimizer
    # on Muon's tensors, the same AdamW groups (the keys after switch_warmup_steps; ced/train/matrix_optim.py).
    name: str = "adamw"
    lr: float = 3e-3
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 10
    # Warmup-stable-decay: linear decay to zero over the last decay_fraction of training.
    decay_fraction: float = 0.2
    # Muon only.
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    # Learning-rate multiplier of the AdamW groups when name == "muon" (Moonlight matches update RMS,
    # so 1.0 shares one LR), and under the matrix optimizers (name "kl_soap", "shampoo", "normuon", "anvil2_muon", or
    # hyperball true under any name, "adamw" included; ced/train/optim.py matrix_groups): there the matrix group runs at
    # lr and every AdamW group at lr x this key, so a screen of the matrix optimizer's LR can hold the AdamW groups' fixed.
    adamw_lr_scale: float = 1.0
    # The loop experiment. Learning-rate multiplier of pass 2's own tensors (LoopConfig.pass2_gate: the gate or pass
    # 2's hyper-connection scalars, and pass 2's loop-index embedding), on top of the run's LR at that step (and of
    # adamw_lr_scale under Muon). 1.0 (the default): the run's LR. No effect on a model without those tensors.
    pass2_lr_scale: float = 1.0
    # The LR schedule's length in REFERENCE steps (ced/train/trainer.py schedule_clock): None (the default) = `steps`,
    # the schedule of every run before this key. Set, the warmup-stable-decay schedule above is a function of the
    # planned wall-clock, not of the step index: warmup_steps and decay_fraction describe a run of schedule_steps
    # steps (the T = 1 arm of a pair, or the trunk), and this run's update k reads that schedule at the clock
    #   k                                                             for k < loop_switch_step (the trunk's own steps)
    #   S + (k - S) * (schedule_steps - S) / (steps - S),  S = loop_switch_step   from the switch on
    # so a run whose steps are slower (T = 2) follows the reference schedule at equal fractions of the planned
    # wall-clock: with f = S / schedule_steps and N2 = steps - S, at f + (1 - f) * (k - S) / N2.
    schedule_steps: int | None = None
    # Learning-rate multiplier of the residual scalars' a alone (model.encoder.layer_scalars with a stored
    # encoder.skip_gain: "skip_x0", "skip", "skip_x0_span"), on top of the LR the scalars have (the run's LR at that
    # step, times adamw_lr_scale under Muon; ced/train/optim.py). 1.0 (the default): encoder.skip_gain stays in the
    # AdamW group without weight decay beside encoder.x0_gain, the groups of every run before the key, tensor for
    # tensor and in the same order. Any other value: encoder.skip_gain alone moves to a group of its own, the LAST one
    # (same AdamW, betas, eps, schedule, no weight decay), whose LR is the run's times this; encoder.x0_gain (b) is
    # not touched. The low-LR variant sets 0.01. A finite number above 0; a value other than 1 on a model without
    # encoder.skip_gain is refused, so no config names an LR that nothing has.
    layer_scalar_a_lr_scale: float = 1.0
    # The optimizer of the dense matrix B of a looped middle's injection (model.loop.recur_layers with injection
    # "parcae" and recur_inject_b "dense": the tensor encoder.recur_injection.B, d x d, at the identity at the start).
    # Read under name "muon" only (ced/train/optim.py build_optimizer). "adamw" (the default: every run before the
    # key, the same groups in the same order): B in the AdamW group without weight decay, with A_log and dt_bias.
    # "muon": B is one more tensor of the Muon group of the layers' matrices: the same update, momentum, LR and LR
    # schedule, and the same weight decay as they get (optim.weight_decay: B is multiplied by 1 - lr x weight_decay at
    # every step, a pull towards zero, not towards its start, the identity). "muon_no_decay": the Muon update with the
    # run's LR in a group of its own with weight decay 0; the
    # group comes after every other group. A_log and dt_bias stay in the AdamW group without weight decay whatever the
    # value. Refused: a value other than "adamw" with name "adamw", or on a model without that d x d tensor (no looped
    # middle, no injection, B "diag").
    recur_b: str = "adamw"
    # An LR ramp after a loop switch step (TrainConfig.loop_switch_step S; the loop screen's variant (b), section 5.3,
    # reading 1). 0 (the default): none, every run before the key, bit for bit. N_w > 0: for the first N_w updates after
    # the switch (update index k, S <= k < S + N_w) the LR of EVERY parameter group is multiplied by (k - S + 1) / N_w,
    # and by nothing from update S + N_w on (the factor of update S + N_w - 1 is N_w / N_w, exactly 1). The ramp is on top
    # of the schedule the run reads by compute fraction (schedule_steps above); it starts at 1 / N_w of that LR, not at
    # zero. Refused without a switch step, and when the ramp would not end inside the run (N_w > steps - S). The T = 1
    # trunk of the run has 0 (ced.train.trainer.trunk_config), so a branch may set it against its trunk.
    switch_warmup_steps: int = 0
    # The methodology programme's matrix optimizers (ced/train/matrix_optim.py). Each is an optim.name of its own, opt-in:
    # "kl_soap", "shampoo", "normuon", "anvil2_muon". Each takes exactly the tensors "muon" puts on Muon (ced/train/optim.py
    # is_muon_param); every other tensor stays in the AdamW groups "muon" builds. The keys below are read only under the
    # name they name (or with hyperball); at their defaults no run before them changes. Hyperball (arXiv 2606.16899): the
    # matrix group's update normalised and the matrix projected back to a fixed Frobenius norm after every step, with no
    # weight decay on it. Off by default. Usable on every matrix optimizer, "muon" (then the framework's Muon, the same
    # update as MuonWithAdamW) and "adamw" (then Adam's update on the Muon group's matrices).
    hyperball: bool = False
    # The norm Hyperball keeps: 0.0 (the default) = each matrix's own Frobenius norm at its first optimizer step (its
    # initialisation: the paper's R = ||W_0||_F); a positive value = that norm for every wrapped matrix (each is
    # rescaled to it at its first step).
    hyperball_radius: float = 0.0
    # KL-SOAP (arXiv 2607.20548, Algorithm 2 / Table 6; NVIDIA-NeMo/Emerging-Optimizers KlSoapV3): the EMA coefficient of
    # the Kronecker factors (beta_kron), the eigenbasis refresh period F (1 = the paper's per-step QR, its fix of the
    # instability; 10 = the original SOAP's), and the refresh by a full eigendecomposition instead of one power
    # iteration + QR (Emerging-Optimizers' "realtime eigen" ReklsV3). betas and eps are optim.betas / optim.eps.
    soap_shampoo_beta: float = 0.95
    soap_precondition_frequency: int = 1
    soap_eigh: bool = False
    # Shampoo (Distributed Shampoo, arXiv 2309.06497, the AdamW-replacement configuration of its README: Adam grafting,
    # betas = optim.betas, grafting epsilon = optim.eps): the epsilon added to the factors before the inverse root, the
    # period of the inverse-root refresh, and the step before which the grafted Adam update is used alone (-1 = the
    # refresh period, the source's default).
    shampoo_epsilon: float = 1e-12
    shampoo_precondition_frequency: int = 100
    shampoo_start_step: int = -1
    # NorMuon (arXiv 2510.05491): the EMA coefficient of the per-neuron second moment (beta2). Momentum, Nesterov and
    # the orthogonaliser's steps are the muon_* keys above.
    normuon_beta2: float = 0.95
    # The muP-correct GDN grouping (with "W_alpha/W_beta as hidden weights with Adam LR ∝ 1/fan-in; A_log and dt_bias at
    # Θ(1); arXiv 2606.04048"; and "Do not change the current ladder's groups"). Off (the default): the groups of every run
    # before the key, tensor for tensor. On, under every optim.name: every GDN layer's in_proj_a and in_proj_b (W_alpha,
    # W_beta: d_model -> one value a head) leave their AdamW group for groups of their own at the AdamW LR times
    # gdn_mup_base_width / fan_in (fan_in = d_model), with the weight decay they had; A_log and dt_bias leave theirs for a
    # group of their own at the AdamW LR with no width factor (and no weight decay, as before). The base width is the 60M
    # base's d_model, 512, where the LR 0.003 was measured, so at 60M the option moves tensors between groups and changes no
    # number; at 200M (d_model 768) the gates' LR is 2/3 of today's. ced/train/optim.py gdn_mup_groups. And their init: as
    # drawn, times sqrt(gdn_mup_base_width / fan_in) (std 0.02 at 512, 0.0163 at 768; ced/train/optim.py gdn_mup_init_,
    # applied by the trainer after the model is built). At the base width the option still changes no number.
    gdn_mup: bool = False
    gdn_mup_base_width: int = 512
    # Weight decay on the zero-initialised gates whose zero means "off". "decay" (the default): the rule, the groups of every run before the key,
    # tensor for tensor. "none": the tensors named by ced/train/optim.py ZERO_GATE_SUFFIXES (the GDN-side arms' ccr_gate and write_lambda, and the wsl
    # variant qmix's qmix_lambda) move to their optimizer's AdamW group WITHOUT weight decay, at the same LR scale; nothing else moves. New arms set
    # it to "none" in their runs file.
    zero_gate_decay: str = "decay"
    # Weight decay on the write mix's depth-mix pseudo-queries. "decay" (the default): R1-entry's readers' rule, the groups of
    # every run before the key, tensor for tensor. "none_qmix": the wsl variant qmix's read-mix query
    # (encoder.layers.<i>.gdn_arms.qmix_query.query) moves to its optimizer's AdamW group WITHOUT weight decay at the same LR
    # scale; the write query keeps wsl-nowd's decay. "none_write_qmix": the write query (gdn_arms.write_query.query) too. The
    # key-norm gains (key_norm.weight) and R1-entry's own readers keep their decay under every value. A value other than
    # "decay" on a model with none of the named tensors is refused. ced/train/optim.py MIX_QUERY_DECAY.
    mix_query_decay: str = "decay"
    # CCR v2's query translators (model.encoder.gdn_ccr_translator "key_head"; ced/model/gdn_arms.py). The translators are
    # always in an AdamW group of their own at the readers' LR (the run's LR x adamw_lr_scale under Muon and the matrix
    # optimizers; the run's LR under "adamw"). "none" (the default): that group has weight decay 0. "identity": the run's
    # weight decay, which pulls the stored M - I toward zero, so M toward the identity. Read only when the model has
    # translators; a value other than "none" without them is refused.
    ccr_translator_decay: str = "none"
    # ANVIL-II fidelity (the port's note). Read under optim.name "anvil2_muon" only; every key at its default is today's
    # anvil2_muon, bit for bit. anvil2_preset: "style" (today: the keys below one by one) or "record" (the four pieces that
    # touch only the Muon group on, and the schedule's points at the record's own step numbers; the keys of those pieces
    # must then stay at their defaults; ced/train/optim.py anvil2_settings).
    anvil2_preset: str = "style"
    # The record's cascade operation for operation: the first Gram on the unnormalised blend, and each map's two products
    # as addmm with beta / alpha (one rounding each; two kernels on a matrix of more than 1,024 rows in the record's storage).
    anvil2_single_rounding: bool = False
    # The record's tail average TAVG (window 250, every 4th step, rate 4/53) on the V/O and MLP rows, which at the end replaces
    # them before the EMA blend of 0.3493 (0.55 on the Q/K rows); ced/train/optim.py anvil2_matrix_roles names the rows.
    anvil2_tavg: bool = False
    # The renorm after the end-of-run blend: "matrix" (today: each matrix to its own pre-blend norm) or "bank" (one ratio for
    # all rows of a record bank: Q/K, V/O, MLP).
    anvil2_decon: str = "matrix"
    # "ours" (today: 0.2 sqrt(max(m, n)) and decay wd x lr x lr / peak) or "record" (the record's lr x max(1, rows / cols)^0.5
    # in its storage, x 2 on the MLP output matrix, and its decay 2.25 x lr x that on the aligned elements; optim.weight_decay
    # is then not read for the matrices).
    anvil2_scale: str = "ours"
    # Where the record's schedule points (240, 50, 514, 298, 250 of its 1,194 steps) sit in this run: "scaled" (x steps
    # /1194), "absolute" (the record's step numbers; a run of more than 514 steps), "record_tokens" (the share of the
    # record's tokens at each point), or None (the default): the preset's, "scaled" under "style" (today) and "absolute"
    # under "record". A named basis is read under either preset.
    anvil2_points: str | None = None
    # TailEMA on the input embedding and the output head (the record's: window 298, blend 0.65, renorm): tensors outside the
    # Muon group, so a key of its own under either preset. Off.
    anvil2_tail_ema: bool = False
    # Q and K orthogonalised two heads at a time, as the record's qk_bank (R:2111-2122): the query rows of each two
    # adjacent heads of q_proj (its gate rows one more unit), each two adjacent KV heads of k_proj, and the q and k rows
    # of each two adjacent key heads of a GDN in_proj_qkv (its v rows one more unit) are each their own cascade, lane
    # equaliser and update scale (ced/train/optim.py anvil2_qk_units). A separate key under either preset. Off: today,
    # bit for bit.
    anvil2_qk_head_pairs: bool = False
    # The LR schedule's decay ends at lr_floor x the peak LR instead of 0 (the record's LR_FLOOR 0.30, R:3108 and 3164:
    # lr = peak x (f + (1 - f) x the linear decay)). Every group of every optimizer. 0.0 (the default): today, bit for bit.
    lr_floor: float = 0.0
    # KL-SOAP's seeded start: the first step's KL weights at the gradient's RMS, and the larger factor of a rectangular
    # matrix one-sided until it can have full rank (ced/train/matrix_optim.py _rule_kl_soap_seeded). Read under optim.name
    # "kl_soap" only. Off.
    soap_seeded_start: bool = False
    # Our optimizer M1 to M4 on the ANVIL-II chassis and the baselines of its test 1 (OursConfig above). All off.
    ours: OursConfig = field(default_factory=OursConfig)


@dataclass
class LossConfig:
    aux_weight: float = 1.0
    z_loss: float = 1e-4
    # "global_token_mean": (sum_dec CE + w_aux * sum_aux CE) / number of valid targets.
    # "per_head_mean": mean_dec CE + w_aux * mean_aux CE (v2: 0.1 x the aux head's mean).
    # The z-loss term z_loss * lse^2 is added to each position's CE inside the same sums.
    normalization: str = "global_token_mean"
    # Where the aux (encoder) head of a model with decoder slots is trained (ced/train/positions.py):
    # - "context" (v1): every valid target that does not run the decoder;
    # - "random": exactly round(aux_fraction * L) positions per row, uniform among the row's valid targets, from a
    #   counter-RNG stream independent of the decoder positions' (so the two sets overlap at random). Fewer valid
    #   targets: all of them (logged).
    # - "none" (stage 3 adaptation): no aux-head loss position at all (an all-False aux mask), so with the decoder at
    #   the answer positions only the context positions are scored by nothing; needs aux_weight 0. With
    #   decoder_opt "exact" / "compiled" and the fused head the aux head is then not computed (Trainer._skip_aux).
    # Models with one head (the controls) ignore both.
    aux_positions: str = "context"
    aux_fraction: float = 0.5
    # The cross-entropy's form (ced/train/ce_fix.py).
    # - "standard" (the default; every run before the key): ce = lse - z_target, the target's gradient softmax_t - 1, which
    #   rounds to 0 when the target's margin passes the float32 cliff (about 16.6; bf16: about 5.5).
    # - "fixed": ce = softplus(LSE_{j != t}(z_j) - z_t); the target's gradient -sigmoid(LSE_{j != t}(z_j) - z_t) (the others'
    #   unchanged). Training-loss paths only (the chunked and fused heads at either head_precision, the draft heads, Token
    #   Superposition's K hard-target terms); the eval scorers stay "standard". Refused by name with head_impl quack_lce /
    #   liger_flce / cce, model.cp_head and model.mtp (ced/train/ce_fix.py check_config). Not bit for bit with "standard"
    #   (the others' elements are the same numbers; the target's differs only past the cliff, the loss value by float32
    #   rounding); with "standard" every path is main's, bit for bit.
    ce_form: str = "standard"


@dataclass
class WandbConfig:
    mode: str = "disabled"  # online | offline | disabled
    project: str = "looped-ced"
    entity: str | None = None
    run_name: str | None = None


@dataclass
class EvalConfig:
    # name -> path of an eval set saved by tools/data (ced.eval.span_bpb.EvalRows as a torch file).
    sets: dict = field(default_factory=dict)
    every_steps: int = 0  # periodic eval on the first periodic_max_rows rows; 0 = only the final eval
    periodic_max_rows: int = 64
    batch_size: int = 1
    # decoder_at "rows_sinks_replay" (decoder_at is below; "replay window": the restricted regime is answers + sinks +
    # replay window): "rows_sinks" plus the decoder at the min(replay_window, available) positions of the same document
    # before each run of consecutive scored positions of the rows (not scored), so dec_bpb is at the same positions as
    # "rows" and "rows_sinks". replay_window is read in that regime only and must be >= 1 there
    # (ced/eval/span_bpb.evaluate's replay_window; 128 is stage 2's local window and scripts/eval_probes.py's default).
    # scripts/rescore.py may also set a last-turn name here ("turn", "turn128": ced/eval/replay_window.py; the
    # candidate regime). No training config sets one, and no key was added for it: a checkpoint written by this code is
    # read by the code before it, and the other way round.
    replay_window: int | str = 128
    # Where CED v2's decoder runs in these evals (ced/eval/span_bpb.evaluate's decoder_at): None (the default, every
    # run so far) and "every": at every valid target; "rows": at the rows' own decoder positions only (spans, or the
    # assistant tokens of role rows), the restricted regime of stage 3, scored there (dec_bpb); "rows_sinks": "rows"
    # plus each document's first decoder.n_sinks positions (not scored), the restricted regime with the sinks. Other
    # arches: None.
    decoder_at: str | None = None
    # The in-loop lite long-context eval (ced/train/periodic_eval.py run_lc_eval): a trace over steps of what the
    # long-context suite measures, for the stopping-signal question. Forward only; training is the same bit for bit with
    # it on or off. lc_every_steps: 0 (the default) = off; else every this many optimizer steps, and at step 0 and at the
    # end. lc_dir: the suite's eval set (web.pt, code_repo.pt, agent.pt); the tool that makes it is not part of this release.
    # lc_length: the row length L of the stage (0: data.seq_len); one of lc_base_len x {1, 2, 4, 8}. lc_docs: documents
    # per set: the first lc_docs of each set that fill a row of L (else the longest others). lc_base_len: the pretraining
    # row length the suite's position buckets are scaled from (tests: small). lc_mf_items: 0 (the default) = no multi-fact
    # part; else the multi-fact task in the trace too: the first lc_mf_items items of every
    # (probe, k) cell of length L of the suite's multi-fact set, <lc_dir>/multifact/t64k (mf_chain and mf_concat,
    # k = 2 / 4 / 8: 6 cells, so 4 is 24 forwards), scored with the suite's own multi-fact scoring.
    lc_every_steps: int = 0
    lc_dir: str | None = None
    lc_length: int = 0
    lc_docs: int = 12
    lc_base_len: int = 8192
    lc_mf_items: int = 0
    # The readout inside training (ced/train/readout.py; the loop screen, section 5.7, way A): exact match at T = 1 and
    # at T = 2 on a fixed sample of held-out DEV items of the synthetic eval sets, scored forward only by the scorer's
    # own code (ced/eval/loop_passes.py) and written to metrics.jsonl as {"event": "readout", "step"...}. Training is the
    # same bit for bit with it on or off. readout_every: 0 (the default) = off; R > 0: after every R-th optimizer step
    # (step % R == 0), and also at the run's loop switch step (the state a branch starts from) and at its last step. A
    # step whose record is in metrics.jsonl already is not scored again (a resume). readout_sets: the sample, a list of
    # [eval directory, task, depths, items per depth]: the first `items per depth` items of each listed depth of <eval
    # directory>/<task>.dev.pt, in file order (no random draw). One row of the record per entry, named <set>/<task> (the
    # set is the directory's parent name when the directory is a tokenizer directory such as
    # .../syn-reason-deep-v1/t64k). Read once, at the start of the run. readout_batch_tokens: padded tokens a scoring
    # forward (rows = this // the padded item length; 16384 is the scoring runs' value). Smaller: less memory beside the
    # training state, more forwards. A model without a loop is scored at T = 1 only. Arch "control" (the plain control
    # and the loop in the published form); needs metrics_jsonl. The eval section is free between a branch and its trunk
    # (BRANCH_SECTIONS).
    readout_every: int = 0
    readout_sets: list = field(default_factory=list)
    readout_batch_tokens: int = 16384
    # The bpb readout inside training (ced/train/bpb_readout.py): bits per byte at T = 1 (and for a looped model also at T
    # = 2) on a fixed held-out sample of eval rows, scored by the reference scorer's own function
    # (ced.eval.span_bpb.evaluate) on the training model (its compiled layers where the run compiles them), one record
    # {"event": "bpb_readout", "step"...} in metrics.jsonl. Training is the same bit for bit with it on or off.
    # bpb_readout_every: 0 (the default) = off; R > 0: at every step s >= bpb_readout_from with (s - from) % R == 0, and
    # at the run's last step (never step 0). A due step whose record is in metrics.jsonl already is not scored again.
    # bpb_readout_from: the first readout step (a stage C or a loop branch: its first step, read before any update).
    # bpb_readout_sets: {name: path of an EvalRows file}, the sets the rows come from (the four headline sets: web, code,
    # agent, general_web). Read once, at the start of the run. bpb_readout_rows: rows in all, split equally over the sets
    # (256: 64 a set); bpb_readout_seed: the fixed seed of the choice. Which rows of a set is a function of (seed, set
    # name, the set's row count) alone, so every arm that reads the same files scores the same rows (the record carries
    # each subset's fingerprint).
    bpb_readout_every: int = 0
    bpb_readout_from: int = 0
    bpb_readout_sets: dict = field(default_factory=dict)
    bpb_readout_rows: int = 256
    bpb_readout_seed: int = 0


BAG_INPUTS = ("mean", "sum_sqrt")  # TSTConfig.bag_input


@dataclass
class TSTConfig:
    """Token Superposition Training (TST; arXiv 2605.06546, prior art PLT arXiv 2407.12665; 2026-10-15;
    ced/train/tst.py has the rules, the design). Off by default (schedule []): nothing in ced/train/tst.py runs
    and every run is the run it was, bit for bit.

    schedule: [[start_step, s]...]: from optimizer step start_step on, one position of a row is the mean of s contiguous token
      embeddings (a "bag") and predicts the next bag of s tokens (mean CE through the normal head; 1/i weights normalised to 1
      for s >= 8); s = 1 is plain next-token training. The first start is 0, starts increase, two neighbours never have the same
      s. At each later start (a "switch") every Muon momentum buffer and every AdamW m, v and step count is reset (AdamW's bias
      correction restarts at t = 1; keep_optimizer_state true keeps them); nothing else changes (the LR schedule is a function of
      the step alone). [[0, 1]] is the base bit for bit. gdn_decay_rescale: at a switch from s to s', A_log += ln(s' / s) in
      every GDN layer, so the log-decay g becomes g s' / s: the decay per token (g / s per position of s tokens) is unchanged.
      token_scale_rope: in a bag phase the capped-rotary global layers rotate at s x (the bag's document position), and the cap
      rope_max_distance stays in tokens: a key d bags back is near while s d <= cap. phase1_exclude_sources: source names (shard
      meta "sources") the FIRST phase does not read; the later phases read the rest of the same per-seed permutation, the two
      queues merged in proportion to their tokens, so no document is read twice and none is skipped. aux_multi_hot: K > 1: in
      next-token (s = 1) phases, the logits at t are also scored against the tokens t + 2 .. t + K (the mean CE of the K - 1),
      weight aux_weight at the first next-token step after a bag phase, annealed linearly to 0 at the decay start int(steps x (1
      - decay_fraction)). 0 (the default): off.

    The redesign for the GDN hybrid (2026-10-16). Each off by default; all off is today's TST bit for bit.
    gdn_resolution_gates (E): in a bag phase of size s every GDN layer's kernel gets g_bag = s g (g = -exp(A_log)
    softplus(in_proj_a(x) + dt_bias), the per-token log decay) and, with "decay_beta", the write strength beta_bag = 1 -
    (1 - beta)^s (beta = sigmoid(in_proj_b(x))): the position's transition is the token's raised to the power s. "decay":
    g alone. "none" (the default): off. Next-token phases are untouched. gdn_conv_reinit (G): at each switch INTO
    next-token (s' = 1), every GDN layer's short-conv weight is drawn again from its own init, U(-1/sqrt(K), 1/sqrt(K)),
    from a counter generator of (seed, switch step); its moments are reset with every other moment at the switch.
    bag_lr_power (L): p > 0: in a bag phase of size s the LR of every group is the schedule's times s^p (0.5: sqrt(s)). 0
    (the default): off. rewarm_steps (R): N > 0: from each switch into next-token at step S, the LR of every group is the
    schedule's times
      (k - S + 1) / N for S <= k < S + N (the moment reset as always). 0 (the default): off.
    output_only (output-only TST, arXiv 2605.06546 section 5.2): a phase with s > 1 reads the base's token rows (one
      token a position, the base's stream position) and scores each position's logits against its next s tokens
      t + 1 .. t + s with the bag loss (bag_weights(s)). False (the default): bags in and out.

    The TST core. Each off at its default; all off is today's TST bit for bit. bag_input: how a bag's s token embeddings become one position. "mean"
    (the default): their mean, as every run so far (the paper's own listing is the audit's). "sum_sqrt": their sum divided by sqrt(s)
    (variance-preserving; the redesign's H). Both are formed in float32 (the embedding table's dtype, the master weights'; "sum_sqrt" raises a
    lower-precision table to float32 itself), BEFORE model.embed_norm. With embed_norm (RMSNorm, every config we train) the two differ only through
    the norm's eps (sqrt(s) x the mean, and RMSNorm divides by the scale): the choice is a numerical no-op there. keep_optimizer_state: False (the
    default): at every switch every Muon momentum buffer and every AdamW m, v and step count is reset (the module docstring of ced/train/tst.py, THE
    SWITCH; today's behaviour). True: the switch leaves the optimizer's state as it is, at every switch of the schedule: (arXiv 2605.06546
    main.tex:190, which says nothing of the optimizer; a TorchTitan checkpoint carries it); the LR schedule, the weights and the data are the switch's
    as before. Refused with gdn_conv_reinit (the re-drawn conv weights would carry the old weights' moments), with optim.ours.m4 (M4 acts inside the
    reset call, which keep never makes: it would never fire) and with a schedule that has no switch (the key acts at a switch). The keep-moments test
    arm is this key; runs/tst-keep-60m.yaml, tests/test_tst_keep.py."""

    schedule: list = field(default_factory=list)
    gdn_decay_rescale: bool = False
    token_scale_rope: bool = False
    phase1_exclude_sources: list = field(default_factory=list)
    aux_multi_hot: int = 0
    aux_weight: float = 0.1
    gdn_resolution_gates: str = "none"
    gdn_conv_reinit: bool = False
    bag_lr_power: float = 0.0
    rewarm_steps: int = 0
    output_only: bool = False
    bag_input: str = "mean"
    keep_optimizer_state: bool = False


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    steps: int = 100
    seed: int = 0
    dtype: str = "float32"  # parameter (master weight) dtype
    log_every: int = 10
    ckpt_every_steps: int = 50
    ckpt_every_minutes: float = 7.5
    ckpt_keep: int = 3
    # The asynchronous checkpoint save (ced/train/async_save.py has the rules). False (the default; every run before the
    # key, bit for bit): the synchronous save. Absent from a checkpoint's saved config at this value, so a checkpoint is
    # byte for byte what it was. True: fit's periodic saves copy the state into host memory (page-locked on CUDA) and write
    # it in a background thread; the file and `latest` are committed at a step boundary once every rank agrees the write is
    # done; the stop's and the end's saves wait for theirs. The files are the synchronous save's bit for bit
    # (tests/test_async_save.py). With optimizer_sharding "zero1" every rank writes its optimizer shard in the background
    # and renames it at the agreed commit before rank 0's step file (tests/test_async_save_zero1.py).
    ckpt_async: bool = False
    # The data prefetch (ced/train/prefetch.py has the mechanics). False (the default; every run before the key, bit for
    # bit): the step's input (the rows, their positions and masks, the host numbers the step reads) is built on the main
    # thread when the step starts. True: a background thread builds the NEXT step's input while this step runs (the same
    # function of the step, so the same rows in the same order on every rank, with a resume from any checkpoint the same
    # stream), and on CUDA copies the rows to the device on a side stream from pinned memory, the compute stream waiting for
    # the copy just before the step's first use. Absent from a checkpoint's saved config at False (a checkpoint is byte for
    # byte what it was); a resume that changes it needs --allow-config-change, as ckpt_async does. Not part of the data
    # identity (Trainer.data_state): it changes no number. A top-level key and not data.prefetch so that the saved config of
    # an off run stays main's (saved_config_dict drops top-level keys at their defaults only).
    data_prefetch: bool = False
    # GPU path. device "auto" = CUDA when available. autocast: "none" or "bfloat16" (the model body;
    # the LM head always runs in float32 or wider). micro_batch_size: rows per forward/backward,
    # gradients accumulated over data.batch_size rows (None = all at once). tf32: allow TF32 for
    # float32 matmuls on CUDA (this is what the float32 LM head uses). deterministic:
    # torch.use_deterministic_algorithms (the launcher also sets CUBLAS_WORKSPACE_CONFIG).
    device: str = "auto"
    autocast: str = "none"
    micro_batch_size: int | None = None
    tf32: bool = False
    deterministic: bool = False
    # With deterministic on, PyTorch also fills every new uninitialised tensor (torch.empty and the like)
    # with NaN (integers: their maximum) unless this is false
    # (torch.utils.deterministic.fill_uninitialized_memory). Every kernel here writes what it later
    # reads, so the fill changes no value; it costs one write pass over each such buffer (the fused GDN
    # kernel's and the fused head's buffers among them: about 3-4% of a v2-arm step). Determinism is
    # unaffected: the kernels are the same, and a kernel that read uninitialised memory would show up as
    # a bitwise difference between runs with and without the fill (checked per arm before the v2 configs
    # turned it off).
    fill_uninitialized_memory: bool = True
    # Append one JSON line of metrics per step to <run_dir>/metrics.jsonl.
    metrics_jsonl: bool = False
    # torch.compile (CUDA only): "none"; "head" = the chunked LM-head function; "all" = the head and every
    # encoder/decoder layer (in place: parameter names and checkpoints do not change). Same math,
    # different kernels: a run uses one setting throughout, and it is checked before use.
    compile: str = "none"
    # Compute the LM head only at positions whose loss counts (Looped-CED: valid context positions for
    # the aux head, filled valid slots for the decoder and draft heads; control: valid targets), gathered
    # into a buffer padded to a multiple of the head chunk. Masked positions contribute exactly zero to
    # the loss and gradients, so this removes work without changing the function. Positions are found
    # on the host from the batch (like the varlen boundaries).
    head_select: bool = False
    # "chunked": ced/train/chunked_loss.token_ce (checkpointed chunks; autograd recomputes the logits). "fused":
    # ced/train/fused_head.fused_linear_ce (logits once per chunk, gradients formed in the forward pass; same matmul
    # precision). "fused" requires head_select. "quack_lce": QuACK 0.6.5's bf16 linear cross-entropy (its scaled-exp pipeline:
    # the CE fused into the logits GEMM's epilogue) with our glue for the z-loss and the per-row weights
    # (ced/train/quack_head.py). Requires head_select and head_precision "bf16"; SM90 only (refused on a B200 and on the CPU
    # without the emulation); refused with model.cp_head and tst.schedule. Not bit for bit with "fused": its 60M band check is
    # its gate. "liger_flce" / "cce": a portable fused linear cross-entropy, Liger-Kernel 0.8.4's or Apple's Cut Cross-Entropy
    # (git b7a02791), both Triton, with our z-loss and per-row weights (ced/train/portable_head.py). Both round the logits to
    # bf16. Require head_select and head_precision "bf16"; refused with model.cp_head, tst.schedule / tst.aux_multi_hot, CUDA
    # graphs and the CPU without the emulation; liger_flce also with loss.z_loss 0, cce under deterministic true (its
    # backward's locked adds are not bit for bit). Not bit for bit with "fused": a 60M band check is each one's gate.
    head_impl: str = "chunked"
    # The precision of the LM head's three matmuls in the training loss (ced/train/head_precision.py). "fp32" (the
    # default; every run before the key): float32 operands, TF32 on CUDA when tf32 is on, the code as it was. "bf16": the
    # operands (h and W for the logits, the logit gradient and W for dh, the logit gradient and h for dW) rounded once to
    # bf16, the products accumulated in float32 into float32 logits, dh and dW; softmax, log-sum-exp, CE, z-loss and the
    # logit gradient in float32. Both head_impl values, every head of the trainer's loss (the aux, decoder and draft
    # heads, the Copy-Paste mixture's LM rows, Token Superposition's multi-target head). Not bit for bit: the 60M band
    # check (scripts/band_check_60m.py) is its gate. Scores never read it: the final eval comes from the reference head
    # in float32; the eval's fused-kernel check reads it.
    head_precision: str = "fp32"
    # Positions per chunk of the fused head (head_impl "fused"; ced/train/fused_head.py, whose CHUNK is the default).
    # 2048 (the default; every run before the key): the head as it was. Another value: the same function with other
    # summation orders (cuBLAS's kernel at the chunk's shape; dW summed over fewer, longer chunks), so not bit for bit
    # with 2048: the 60M band check is its gate. Every head of the fused loss reads it (the aux, decoder and draft
    # heads, the Copy-Paste mixture, Token Superposition's multi-target head), at either head_precision, and the eval's
    # fused-kernel check; refused by name with any other head_impl when it is not 2048. Memory a call: the chunk's
    # float32 logits, plus at "bf16" their bf16 gradient: 6 bytes x chunk x V (3.1 GB at 8,192 x 64,000, against 0.8 GB
    # at 2,048; exact).
    head_chunk: int = 2048
    # Looped-CED: run the decoder on the first ceil(n / m) * m slots of each micro-batch instead of
    # all C, where n is the micro-batch's largest number of kept decoder positions and m this value
    # (0 = off). Empty slots are a suffix of the slot sequence and no filled slot reads them (causal
    # mixers, document id -1), so the function is unchanged; only the empty tail is not computed.
    # Must be a multiple of decoder.capacity_multiple. A handful of distinct slot counts results
    # (C / m at most), each a separate compiled shape.
    trim_decoder_slots: int = 0
    # CED v2's decoder fast path (decoder optimisation r1, 2026-10-09). Training and no-cache scoring forwards only;
    # cached inference keeps the old path. Same function in every mode. "none": the decoder code of every run before r1
    # (the default: existing configs run unchanged; not "off", which YAML and the --set overrides read as false). "exact":
    # the same values bit for bit (losses, gradients, parameters; tests/test_decoder_opt.py): the slot gather with an
    # inverse-map backward (a gather instead of the deterministic sort-based scatter-add), the flex BlockMask of each
    # decoder read kind built once per forward and shared by the layers (instead of once per layer), and the aux head
    # skipped where it has weight 0 and no positions (its parameters' gradients set to the zeros it gave). "compiled":
    # "exact" plus the decoder glue (memory KV, slot gather, injection norm, hyper-connections, loop embedding, final
    # norm) as compiled regions under compile "all" on CUDA: the same function, not the same bits.
    decoder_opt: str = "none"
    # Activation checkpointing of the layers in training forwards (the long-context grid; ced/model/act_checkpoint.py). Rows of 32k and 64k tokens on
    # one 21.6 GiB MIG slice: a micro-batch of one 32,768-token row peaks at 19.3 GiB at 60M and one 65,536-token row would need about 36 GiB. "none"
    # (the default; not "off", which YAML and the --set overrides read as false): every run so far, unchanged. "layer": every encoder layer and every
    # CED v2 decoder layer of a training forward runs under torch.utils.checkpoint: the same loss and the same gradients, one more forward of each
    # layer per step, and only one layer's saved tensors alive at a time. Arches "control" and "ced_v2" with the softmax normaliser; refused
    # elsewhere. At T = 2 it covers both passes: CED v2's decoder layers of each pass, and the looped control's block layers of pass 1 (their records
    # kept) and of pass 2. An implementation choice, not part of the model config (init_from and checkpoints do not depend on it); evaluation and
    # cached inference never use it. "block": one checkpoint per hybrid group of full_attn_interval layers, for arch "control" with one pass, with or
    # without model.encoder.attn_res ("block", "block_entry"): the same loss and gradients bit for bit; a block keeps its input (the base) or
    # references to the block outputs b_0 .. it reads (the depth mix), and one block's saved tensors are alive at a time in the backward. Refused
    # elsewhere by name (Trainer._check_block_checkpoint). "host": "layer" with each layer's kept input in host memory between the forward and the
    # backward: the same training bit for bit as "layer" and "none", the device holds about two kept inputs at a time instead of one per layer call, a
    # pool of pinned host buffers (one per layer call, made in the first step) holds the rest, and two blocking copies of the stream per layer call
    # are added to the step. Refused wherever "layer" is. "host_overlap": "host" with the two copies on a side stream beside the compute: layer k's
    # input goes to the host while layer k runs and comes back (prefetched) while layer k + 1 is recomputed and runs its backward. The same training
    # bit for bit as "host"; about one kept input more on the device at a time (in use, in flight each way). Refused wherever "layer" is. "layer",
    # "host" and "host_overlap" with model.encoder.attn_res "block" / "block_entry": each layer of each block under its own checkpoint (not nested in
    # a block checkpoint); a layer keeps the block's partial sum (R1) or the plain stream (R1-entry, whose closing layer also keeps the block's
    # input), on the host under "host" and "host_overlap", and references to the finished blocks. The same loss and gradients bit for bit. For the
    # 128K and 256K rows of the main run. With the Copy-Paste head's pointer reading other than h (model.cp_head.pointer_input "mix" / "shared"):
    # "layer" with cp-mix only ("mix", train_mode "mixture", kernel "reference", one pointer head: n_heads 1, combine "and"; the long-row pair at 32k;
    # tests/test_cp_mix_ckpt.py); every other mode and form refused by name (Trainer.__init__), the cp-grid arms' heads (cp_head.multi_head) among
    # them.
    activation_checkpoint: str = "none"
    # Initial weights of a NEW run (the length-extension test): the path of a checkpoint file whose MODEL weights a
    # run starts from, when its run directory has no checkpoint yet (scripts/train.py; Trainer.init_from_checkpoint).
    # Strict: the source's model config must equal this one (only the implementation choices attention.backend,
    # gdn.kernel and gdn.chunk_size may differ) and every parameter name and shape must match; anything else is
    # refused. One growth is accepted: a gated two-pass model (model.loop.pass2_gate) from the checkpoint of its
    # single-pass model (ced.config.one_pass_model: the same config at T = 1 without the gate): every tensor of the
    # source is loaded strictly, only pass 2's own tensors are new, at their function-preserving init. The optimizer
    # state, the LR schedule, the step counter, the RNG states and the data stream start fresh from this config (step
    # 0 of its shards). A run directory that has a checkpoint resumes from it instead, so a continuation evicted
    # after its first checkpoint continues; the source is not read again. The source's path, step and sha256 are kept
    # in every checkpoint and in metrics.jsonl's start events.
    init_from: str | None = None
    # Add-on CP as a post-hoc stage on a finished trunk (ced/train/freeze_trunk.py has the rules). False (the default): every run before the
    # key, bit for bit (tests/test_freeze_trunk_off.py). True: every tensor outside the add-on head (model.cp_head.*) is frozen
    # (requires_grad False, the tied matrix included), so the trunk's forward builds no graph, no gradient or optimizer state exists for a
    # trunk tensor, the logged grad_norm (the trunk's) is 0 and the head trains alone on the mixture loss. The run starts from init_from's
    # weights (a source equal to this model with model.cp_head off grows the head: its tensors are new at their own-generator init, recorded
    # as "grown"; the trunk is loaded strictly). Refused by name (ced/train/freeze_trunk.py check_config): without init_from; with
    # resume_from; without model.cp_head.train_mode "addon"; with a tst.schedule other than empty or [[0, 1]]; with data parallelism,
    # cuda_graphs, activation_checkpoint, shape_schedule, precision_switch, the sharded optimizer, wgrad_accum or
    # model.encoder.gdn_qmix_select_step. The trunk's sha256 is taken when the weights are loaded and checked at every save and every load; a
    # per-step check refuses a trunk tensor that has a gradient, a state, or requires a gradient again. Absent from a checkpoint's saved
    # config at False (a checkpoint of a run that does not set it is byte for byte what it was).
    freeze_trunk: bool = False
    # Steps at which a checkpoint is written and KEPT, without stopping (nested budgets of the long-context grid;
    # tools/analysis/lc_grid.py): besides the rolling checkpoints (ckpt_every_steps, ckpt_every_minutes, ckpt_keep),
    # the trainer writes step_<s>.pt exactly at each listed step s (0 < s <= steps), never prunes it, and never writes
    # it a second time (a resume that passes the step again leaves the file as it is). A shorter budget's run branches
    # from such a file with resume_from. No effect on the training itself.
    checkpoint_steps: list[int] = field(default_factory=list)
    # A NEW run that continues ANOTHER run's training (a nested budget's branch): the path of a checkpoint file whose
    # WHOLE training state (model, optimizer, step counter, RNG states) this run starts from, when its run directory
    # has no checkpoint yet (scripts/train.py; Trainer.resume_from_checkpoint). Strict, refused otherwise: the source's
    # config must equal this one apart from steps, checkpoint_steps, resume_from and the eval and wandb sections; its
    # data must be this run's (shard fingerprint, data seed, row shape); and the two LR schedules must agree on every
    # update the source has made. The run is then, bit for bit, the run this config trains from its own start
    # (tests/test_long_context_grid.py), at the cost of its remaining steps only. A run directory that has a checkpoint
    # resumes from it instead, so an evicted branch continues; the source's path, step and sha256 are kept in every
    # checkpoint and in metrics.jsonl's start events. With stage "continue" (below) the source may also differ in the
    # stage's keys and its shard set: a stage C on other data.
    resume_from: str | None = None
    # The loop experiment's branch inside one pretraining run (phase A / phase B): the optimizer step S at which the loop is
    # switched on. 0 (the default): none, the loop count of every step is drawn from model.loop.t_choices as before. S > 0 (a
    # gated two-pass model only, LoopConfig.pass2_gate): the updates with index k < S run ONE pass (T = 1) and the updates
    # from S on draw T from t_choices. Before S pass 2's tensors are not part of any forward: they get no gradient, no weight
    # decay and no optimizer state, and the run is bit for bit the T = 1 trunk (ced.train.trainer.trunk_config(this config))
    # run alone (tests/test_loop_t2.py). So a run with resume_from = the trunk's kept checkpoint at step S
    # (checkpoint_steps) continues the trunk's whole training state (weights, optimizer state, RNG, data position; pass 2's
    # tensors new, with fresh optimizer state) and is, bit for bit, the run this config trains from its own start
    # (Trainer.resume_from_checkpoint). The LR after the switch: optim.schedule_steps. The branch's config must be the trunk's
    # apart from the loop keys, steps, checkpoint_steps, resume_from, the eval and wandb sections, and micro_batch_size /
    # activation_checkpoint (a two-pass step needs more memory; trainer.TRUNK_FREE_FIELDS). The control with a gated second
    # pass through its top block (arch "control", LoopConfig.pass2_gate "scalar" with pass2_layers) switches on the same way:
    # its trunk is the plain control of the same config (one_pass_model), and C1 `lpa-b20` continues C0 `lpa-0`'s kept
    # checkpoint of step 5,190. The control with a looped middle and the scalar input mix (model.loop.recur_layers with
    # injection "mix"; the late switch-on of 2026-10-13) switches on the same way: before S every step runs one pass, which is
    # the plain control's forward bit for bit; its trunk is the plain control (one_pass_model: no looped middle, injection
    # "none"); the one new tensor is encoder.recur_mix (w), which gets no gradient and no optimizer state before S. At the
    # switch the model is the trunk's exactly only at w = 1; at the configured start (0.99) it is near it, by design
    # (ced/model/encoder.py recur_input_mix). No other form of the looped middle takes a switch step.
    loop_switch_step: int = 0
    # A stage of a run that continues another run's kept checkpoint on other data. An explicit, narrow opt-in; off by default. "none" (the default):
    # every run before the key. resume_from refuses another config and other data as always. "continue": this run continues resume_from's whole
    # training state (weights, Muon and AdamW state, step counter, RNG states), as resume_from does, and may differ from the source in exactly these
    # keys besides resume_from's own (steps, checkpoint_steps, resume_from, the eval and wandb sections): optim.decay_fraction, data.shard_dir, the
    # shard set itself (its fingerprint), stage and data_start_step, and on a new shard set data.loss_mask. The LR of every update the source made
    # must be this run's (the check of resume_from), so a stage A of S steps at decay_fraction 0 continued by a stage C of N steps at decay_fraction f
    # trains exactly the LR of one N-step run at f. The data seed, the row length and the rows a step are the source's (unless stage_shape is
    # "tokens", below: then the row length and the rows a step may change at the source's tokens a step). With a loop switch step the source is the T
    # = 1 trunk as for any loop branch.
    stage: str = "none"
    # A stage that continues its source in another PRECISION (the NVFP4 tail arms). False (the default): resume_from
    # refuses a source whose model.nvfp4 or model.fp8 differs, as always. True (with stage "continue" only; refused
    # otherwise): the source may also differ in model.nvfp4, model.fp8 and this key. Nothing else changes: the parameters,
    # their names, the state dict, the optimizer groups and state and the RNG states are the same under every precision
    # (both switches turn the layers' nn.Linear into a subclass in place and add no state), so the run is its source's
    # training state continued with this run's linears. The source's and this run's precision are recorded in resumed_from
    # ("precision").
    precision_switch: bool = False
    # The optimizer step at which this run's data stream starts (its row 0): update k reads the stream's batch k - D,
    # rows and decoder / aux positions alike (ShardPacker.batch and PositionSampler.apply at k - D). 0 (the default):
    # batch k, every run before the key. D > 0 only with stage "continue"; there it must be 0 (or the source's own D)
    # when the shard set is the source's (the stream goes on where the source is) and the source's step when the shard
    # set is new (the new stream starts at its beginning at this run's first update, for the run's data seed).
    data_start_step: int = 0
    # Segment-based pass counts (model.loop.segment_passes / unscored_passes; ced/train/pass_positions.py): the slot
    # sizes a later pass may run at, as fractions of the decoder capacity C, each rounded up to a multiple of
    # model.decoder.capacity_multiple. Per forward (a micro-batch) and per pass, the smallest size that holds every
    # row's positions of that pass is used, and a pass no row needs is not run. Fixed sizes, so a compiled layer sees
    # at most len(pass_slot_buckets) call shapes beyond today's. How a step is computed, not what: the logits do not
    # depend on it. Read only with a pass schedule.
    pass_slot_buckets: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.25, 0.125, 0.0625])
    # A batch-shape schedule (batch warmup and short-to-long rows; ced/train/shape_schedule.py has the rules and
    # the design). [] (the default): none, every step reads data.batch_size rows of data.seq_len tokens, as every run
    # before the key. Otherwise the phases BEFORE the config's own shape, in order, each [until, rows, seq_len]:
    # until the share of the run's tokens at which the phase ends (0 < until < 1, increasing); rows rows per
    # optimizer step in the phase; seq_len the row length in the phase (a divisor of data.seq_len). After the last
    # listed phase the run reads the config's own shape to its end. `steps` stays the run's length in REFERENCE steps
    # (steps x data.batch_size x data.seq_len tokens, the unscheduled run's): the number of optimizer steps follows
    # from the phases, the LR schedule and the periodic evals are read at tokens seen, and the rows are the
    # unscheduled run's token stream in its order (a shorter row is a cut of its packed row). Shard data and arch
    # "control" without a second pass. A schedule whose every phase is the config's own shape is no schedule, bit for
    # bit (tests/test_shape_schedule.py).
    shape_schedule: list = field(default_factory=list)
    # The LR of an optimizer step under a shape_schedule (ced/train/shape_schedule.py, "The LR by rows"). "clock" (the
    # default): the config's schedule read at tokens seen, whatever the step's size: every run before the key, bit for
    # bit. A phase with fewer tokens a step makes more updates at that LR. "rows" the linear scaling rule: the LR of the
    # reference step whose tokens the step reads, times the share of a reference step's tokens the step reads (rows x
    # seq_len of the phase over data.batch_size x data.seq_len). The LR summed over the optimizer steps of a reference
    # step is then the unscheduled run's LR of that step, in every phase; where a phase has one optimizer step to a
    # reference step it is "clock" bit for bit. "rows" needs a shape_schedule (refused without one), and a run cannot
    # change the value on a resume.
    shape_lr: str = "clock"
    # Rows per forward/backward by pass count (the loop in the published form, model.loop.recur_layers; a step at more
    # passes keeps more activations): entry T - 1 is the micro-batch size of the rows that run T passes. [] (the
    # default): micro_batch_size at every T, every run before the key. A pass count beyond the list uses its last
    # entry. How a step is held in memory, not what it computes: the step's gradient is the same sum over its rows
    # (the order of the sum over micro-batches aside). Each entry at most data.batch_size.
    micro_batch_by_t: list[int] = field(default_factory=list)
    # The size of the residual stream by layer, in the metrics. 0 (the default): off, every run before the key. N > 0:
    # on every N-th optimizer step the metrics hold "stream_rms": n_layers + 1 numbers, the root mean square over
    # tokens and dimensions of the stream ENTERING each encoder layer (entry 0 is the embedding output x0) and of the
    # stream after the last layer, before the final norm (the definition of scripts/resid_stream_rms.py), on the step's
    # first micro-batch. Read under no_grad from the forward the step already runs, between the layer calls: no extra
    # forward, no random number, nothing written to the model. Losses, gradients, the optimizer's state, the RNG state
    # and the checkpoints' tensors are those of the run without it, bit for bit (tests/test_resid_forms.py). Built for
    # arch "control" with one pass through the stack (with or without model.encoder.layer_scalars); refused elsewhere,
    # so no run sets it and logs nothing (ced/train/trainer.py).
    log_stream_rms_every: int = 0
    # Exact-zero weight gradients by linear (ced/train/zero_wgrad.py). 0 (the default): off, every run before the key. N
    # > 0: on every N-th optimizer step the metrics hold "zero_wgrad": {"zeros": [...], "zero_rows": [...]}, per linear
    # of the layers' mixers and FFNs (the list and each weight's size are in the start record), the number of elements of
    # its weight's gradient that are exactly 0.0 and of its output rows whose gradient is 0.0 throughout, read from the
    # step's gradient as the micro-batches accumulated it, before clipping. Read only (comparisons and sums on the
    # gradients, one host read after the update): the losses, the gradients, the optimizer's state, the RNG state and the
    # checkpoints' tensors are those of the run without it, bit for bit. Arch "control"; refused elsewhere.
    log_zero_wgrad_every: int = 0
    # CUDA graphs for the training step (ced/train/cuda_graphs.py has the rules). "none" (the default; a string, since YAML reads
    # "off" as false): every run before the key, bit for bit. "microbatch": each micro-batch's forward and backward is one CUDA
    # graph replay over static input buffers (fixed shapes: micro_batch_size rows of data.seq_len tokens; the documents of a row in
    # fixed-capacity device metadata; the head at every position of the micro-batch); the gradients stay in place and are
    # accumulated there. The first two steps of a process run eagerly (the second on the capture stream, as the warm-up), the graph
    # is captured at the third. A micro-batch that does not fit the graph's shapes runs eagerly, the same function, and is counted.
    # "step": "microbatch" plus the gradient clip and Muon's momentum and orthogonalisation as a second graph; Muon's LR step and
    # AdamW (which read the step's LR and step count on the host) run eagerly after it. The losses, gradient norms, parameters and
    # optimizer state are those of "none", bit for bit (tests/test_n1_cuda_graphs.py on the CPU, where the same code runs
    # uncaptured; checked on a GPU slice). Built for the plain control (arch "control", one pass, attn_res none / block /
    # block_entry), Muon, the fused head; refused elsewhere by name. Since (2026-10-27) also for the write mix's "layer" write mix
    # (variants none and qmix), E (attn_res_fused) and tst.schedule (the options the 4B plan uses; ced/train/cuda_graphs.py, "THE
    # WRITE MIXES AND TST": a graph and a set of buffers a TST phase).
    cuda_graphs: str = "none"
    # The capacity of the fixed-capacity varlen metadata of the GDN kernels under cuda_graphs, in segments (documents and
    # row starts) a row: cu_seqlens of micro_batch_size x this + 2 entries, the chunk index of rows x seq_len / 64 + that
    # many chunks. A micro-batch with more segments runs eagerly (counted in the step's "cuda_graphs" metrics). In a TST bag phase of size s
    # a row covers s times the tokens, so the capacity is this times s.
    cuda_graphs_docs_per_row: int = 64
    # How the layers' weight gradients accumulate over a step's micro-batches (ced/train/wgrad_accum.py has the rules).
    # "autograd" (the default; every run before the key, bit for bit): the wgrad GEMM writes a bf16 dW under autocast, the
    # cast's backward makes it float32 and autograd adds it into .grad. "fp32_fused": every bias-free nn.Linear of the
    # layers' mixers and FFNs accumulates its weight gradient in float32 inside the wgrad GEMM (beta = 1) into a persistent
    # float32 .grad, zeroed in place each step: no bf16 dW, no cast, no separate add; one bf16 rounding of each
    # micro-batch's gradient fewer (not bit for bit: the 60M band check). Built for the plain control; composes with
    # cuda_graphs; refused elsewhere by name.
    wgrad_accum: str = "autograd"
    # How the optimizer state is held under data parallelism (ced/train/zero1.py has the rules). "replicated" (the default;
    # every run before the key, bit for bit): every rank holds the whole optimizer state and steps every tensor. Absent from
    # a checkpoint's saved config at this value, so a checkpoint is byte for byte what it was. "zero1": ZeRO-1 by whole
    # matrix: each rank owns whole tensors (balanced by elements), keeps their state alone and steps them alone; the
    # gradients are reduce-scattered and the updated fp32 weights all-gathered; the clip, the gradient metrics and
    # ANVIL-II's per-bank renorm gather the owners' numbers exactly, so the step is the replicated step bit for bit
    # (tests/test_zero1.py, T7); the checkpoint is written by parameter (one optimizer file a rank) and loads at any world
    # size, sharded or not. Read under data parallelism only (one process owns every tensor). Built for Muon, NorMuon,
    # Hyperball over Muon and ANVIL-II; refused elsewhere by name. A resume may change it (DP_RESUME_FREE: how the state is
    # held, not what it computes).
    optimizer_sharding: str = "replicated"
    # Token Superposition Training (TSTConfig above; ced/train/tst.py). Off by default (tst.schedule []).
    tst: TSTConfig = field(default_factory=TSTConfig)
    # A stage at another row shape (ced/train/trainer.py, "A stage at another row shape"). With stage "continue" only.
    # "source" (the default; a string, since YAML reads "off" as false): the stage keeps its source's row length and rows a
    # step, every run before the key bit for bit. "tokens": the stage may set data.seq_len and data.batch_size (and
    # micro_batch_size and activation_checkpoint, how a step is held in memory) differently from its source, at the
    # source's tokens a step (6 x 32,768 = 24 x 8,192 = 196,608): the LR clock stays the source's step counter (update k
    # has read k x the same tokens under either shape), the new shard set is read from its beginning in rows of the stage's
    # length (a shape change on the source's own set is refused), and a resume inside the stage is a plain resume. Refused
    # by name with tst.schedule, loop_switch_step, the slot arches and data.fit_decoder_capacity (shape_schedule is refused
    # with any stage).
    stage_shape: str = "source"


# --------------------------------------------------------------------------------------------
# dict / YAML round trip
# --------------------------------------------------------------------------------------------


def to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)


def from_dict(cls, data: dict | None):
    """Build dataclass `cls` from a (possibly partial) nested dict. Unknown keys are an error."""
    if data is None:
        return cls()
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            kwargs[name] = from_dict(hint, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def checkpoint_config(data: dict) -> dict:
    """A checkpoint's config dict (state["config"]) as this code reads it, before from_dict: a key whose default has
    changed since a checkpoint without it was written gets the value that checkpoint trained with, not the new default
    (from_dict fills a missing key with today's). One key: model.fp8.nonfinite_flag_form, "ordered" for a config written
    before the key existed (ordered form was the only one). A new dict; `data` is not changed."""
    fp8 = (data.get("model") or {}).get("fp8")
    if not isinstance(fp8, dict) or "nonfinite_flag_form" in fp8:
        return data
    return {**data, "model": {**data["model"], "fp8": {**fp8, "nonfinite_flag_form": "ordered"}}}


def load_yaml(path: str | Path, cls=TrainConfig):
    with open(path) as f:
        return from_dict(cls, yaml.safe_load(f))


def save_yaml(cfg, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)


# --------------------------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------------------------


def tiny_data_config(**overrides) -> DataConfig:
    """Synthetic data for the tiny model: rows of 64 tokens, spans scaled down to fit C = 16."""
    cfg = DataConfig(seq_len=64, batch_size=4, span_median=5.0, span_min=2, span_max=12)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def tiny_model_config(**loop_overrides) -> ModelConfig:
    """The tiny CPU test model (~0.5M parameters)."""
    attn = AttentionConfig(n_q_heads=4, n_kv_heads=2, head_dim=16, rope_theta=1e4)
    cfg = ModelConfig(
        vocab_size=256,
        d_model=64,
        ffn_hidden=192,
        encoder=EncoderConfig(
            n_layers=8,
            full_attn_interval=4,
            attention=attn,
            gdn=GDNConfig(n_k_heads=2, n_v_heads=4, k_head_dim=16, v_head_dim=16, conv_kernel=4),
        ),
        decoder=DecoderConfig(
            n_layers=2,
            attention=dataclasses.replace(attn),
            gdn=GDNConfig(n_k_heads=2, n_v_heads=4, k_head_dim=16, v_head_dim=16, conv_kernel=4),
            capacity_fraction=0.25,
            capacity_multiple=8,
        ),
        loop=LoopConfig(t_max=3, t_choices=[2, 3], t_probs=[0.5, 0.5], t_eval=3),
    )
    for key, value in loop_overrides.items():
        setattr(cfg.loop, key, value)
    return cfg


def tiny_control_config() -> ModelConfig:
    """Tiny non-looped control: the same blocks, 10 encoder layers, no decoder."""
    cfg = tiny_model_config()
    cfg.arch = "control"
    cfg.encoder.n_layers = 10
    return cfg


def tiny_v2_config() -> ModelConfig:
    """The tiny CPU test model of architecture spec v2 (arch "ced_v2"), sigmoid GDN output gate, 4:1
    query/KV grouping with partial RoPE, a 3-layer memory-only decoder looped T = 2 with static
    hyper-connections (2 streams, delta write), the loop embedding, no injection, and 1/T residual
    scaling on the decoder."""
    attn = AttentionConfig(n_q_heads=4, n_kv_heads=1, head_dim=16, rope_theta=1e4)
    gdn = GDNConfig(n_k_heads=2, n_v_heads=4, k_head_dim=16, v_head_dim=16, conv_kernel=4, output_gate="sigmoid")
    return ModelConfig(
        arch="ced_v2",
        vocab_size=256,
        d_model=64,
        ffn_hidden=192,
        tie_embeddings=False,
        encoder=EncoderConfig(n_layers=8, full_attn_interval=4, attention=attn, gdn=gdn),
        decoder=DecoderConfig(
            n_layers=3,
            attention=dataclasses.replace(attn),
            gdn=dataclasses.replace(gdn),
            capacity_fraction=0.25,
            capacity_multiple=8,
        ),
        loop=LoopConfig(t_max=2, t_choices=[2], t_probs=[1.0], t_eval=2, injection="none",
                        hyper_connections="static", hc_streams=2, hc_write="delta"),
        residual_scale=ResidualScaleConfig(mode="inverse_loops", t_ref=2),
    )


def tiny_second_pass_config() -> ModelConfig:
    """Tiny control (c): the v2 encoder (8 layers) with a zero-write second pass at the slots."""
    cfg = tiny_v2_config()
    cfg.arch = "second_pass"
    return cfg


def tiny_v2_control_config() -> ModelConfig:
    """Tiny controls (a)/(b): the plain stack with the v2 layer settings (untied, sigmoid gate)."""
    cfg = tiny_v2_config()
    cfg.arch = "control"
    cfg.encoder.n_layers = 10
    return cfg


def _int_table(table: dict) -> dict:
    """{ordinal: list of ints} with the one in-memory form of a by-layer table (DeltaProductConfig.heads_by_layer,
    EncoderConfig.gdn_shape_by_layer): keys that are digit strings (a JSON round trip writes dict keys as strings) become ints,
    list values become tuples (YAML and --set give lists), so configs compare equal however they were written. Anything else is
    left as given for the model's checks to refuse by name (gdn_dprod.check_config)."""
    return {(int(k) if isinstance(k, str) and k.isdigit() else k): (tuple(v) if isinstance(v, list) else v)
            for k, v in table.items()}
