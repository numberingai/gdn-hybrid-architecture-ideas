"""The GDN-side architecture arms: A (write-shallow / read-deep GDN) and CCR (checkpoint cross-read).

No paper is named for either arm; both are built from these words.

Keys (EncoderConfig): gdn_write_mix ("none" | "layer" | "head") for A, gdn_ccr_spacing (0 off, else C) for CCR. Both
off by default: build() then adds nothing (no module, no tensor, no random number) and no forward reads this module.

A, as built
-----------
In a GDN layer i >= 1 of block n (blocks of full_attn_interval layers, as R1-entry's):

    x     = mixer_in_norm(stream)                          what the GDN's projections read today
    m     = softmax depth mix over [b_0, ..., b_{n-1}]     R1-entry's stored sources: the embedding output and the
                                                           finished blocks' sums; attn_res.depth_mix with a DepthQuery
                                                           of the layer's own (zeros: uniform at the start); in block 1
                                                           the one source b_0 with weight 1 (no query)
    m'    = mixer_in_norm(m)                               the layer's own input norm, so m' lives where x lives
    x_w   = x + lambda (m' - x)                            per head: lambda of the head the channel belongs to
    k, v  = the k and v channels of in_proj_qkv(x_w);  beta = sigmoid(in_proj_b(x_w));  decay from in_proj_a(x_w)
    q, z  = from x, as today

lambda: "layer", one scalar; "head", H_k + H_v values, entry i < H_k the key head i (the k channels of that head),
entry H_k + j the value head j (its v channels, its beta and its decay). Zero at the start, so the layer's function is
R1-entry's. Layer 0 has none: its input is the embedding output itself (m = b_0, x = norm(b_0)), so x_w = x always.
Two implementations of the same function: write_inputs (the fast path, every forward) and write_inputs_reference
(explicit x_w per head, the ground truth); "layer" costs no extra matmul (the q rows read x, the k/v rows x_w), "head"
one more matmul of the k, v, a and b rows on (m' - x) (linearity: W x_w^h = W x + lambda_h W (m' - x)).

The wsl variants, as built (EncoderConfig.gdn_write_variant)
------------------------------------------------------------------------------------------------------------------------ Read
only with gdn_write_mix "layer" (one lambda a layer); "none" (the default) is A above, bit for bit.

    qmix: x_q = x + lambda_q (m'_q - x) m'_q = mixer_in_norm(softmax depth mix over [b_0 ..
                                               b_{n-1}] with the layer's own SECOND DepthQuery, qmix_query); in block 1 the one source b_0
                                               (so m'_q = m'); lambda_q = qmix_lambda (1,), zeros q = the q rows of in_proj_qkv(x_q); k, v,
                                               beta, decay from x_w as in A; z from x korig: k = the k rows of in_proj_qkv(x), as q; v,
                                               beta, decay from x_w as in A vonly: the V-only variant: ONLY the v rows of in_proj_qkv read
                                               x_w; q, k, beta and the decay read x (the q and k rows ONE matmul on x as korig's, the v rows
                                               a second on x_w; in_proj_a and in_proj_b on x as qshare_kv's); z from x. korig with beta and
                                               the decay moved back to x; qshare_kv's gates with only v mixed kqmix: arm KQ: qmix's two
                                               mixes and tensors (READ_MIX_VARIANTS); the q AND k rows read x_kq = x + lambda_q (m'_q - x),
                                               one matmul (W[:2*nq]); the v rows, beta and the decay read x_w; z from x. qmix with the k
                                               rows moved to the read mix

    qshare: q = the q rows of in_proj_qkv(x_w): the q rows read the same x_w as k, v, beta and the decay (so all
            of in_proj_qkv is ONE projection of x_w); z from x

    The write-quantity split on qshare. In both forms the q rows read x_w, as in qshare; z reads x: qshare_kv: q, k, v =
    in_proj_qkv(x_w) (ONE projection of x_w, as qshare); beta and the decay from x (in_proj_b(x), in_proj_a(x), as R1-entry's)
    qshare_bd: q = the q rows of in_proj_qkv(x_w); k, v = the k and v rows of in_proj_qkv(x) (two matmuls: wsl's split with the
    two operands swapped); beta and the decay from x_w (as in A)

    qmix with a static top-2 query mix (EncoderConfig.gdn_qmix_select_step = S >= 1, under the variant "qmix"): until the start of
    optimizer step S every layer's m'_q is qmix's, the softmax over all its sources. At the start of step S, once, each layer with more
    than 2 sources keeps the 2 sources with the largest TOKEN-MEAN query-mix weight (select_qmix_pairs: the mean over the tokens of the
    step's first micro-batch of the per-token softmax weights, in float64; ties to the lower source index;
    tools/analysis/qmix_mix_read.py's top-2 rule), written into the buffer qmix_keep ((2,) int64, in the state dict: a resume after S
    keeps the choice; -1 before it), and from then on m'_q is the PER-TOKEN softmax over those 2 sources' logits only
    (attn_res.depth_mix_pair: the other sources are neither read nor given weight). Everything else is qmix's, the same parameters; the
    buffer is the only addition. A layer with 2 or fewer sources keeps all of them (no buffer). Off (0, the default): qmix as built, bit
    for bit.

All six start as R1-entry (lambda and lambda_q zero). qmix's own tensors: qmix_lambda (every GDN layer but layer 0) and qmix_query.query
/ qmix_query.key_norm.weight (blocks 2 and up, as the write query); korig, vonly, qshare, qshare_kv and qshare_bd add none. qshare,
qshare_kv and qshare_bd are the variants the fused write mix takes (attn_res_fused; ced/kernels/write_mix.py): its group read already
forms every m' x_w is made from, and these variants only choose which projection rows read the x_w the layer forms from it
(write_inputs), so no kernel changes. qmix takes it too: the block's group read also forms every fused layer's m'_q, NQ 1 + 2 G queries
over the sources read once.

On MXFP8 ------------------------------------------------------------------------------------------------------------------------ Wired:
the write mix's "layer" form with the variants MX_VARIANTS, "none" (wsl, wsl-nowd: the decay-free lambda is an optimizer setting) and
"qmix", with model.fp8 on the MXFP8 scaling, the emulated backend (either quantiser) or Transformer Engine on our operands (te_operands
"ours"). No new option: model.fp8 selects the path, and a model without it is today's bit for bit (write_inputs reads `mx_rows` first and
takes today's statements when it is None). check_fp8_forms refuses every other form by name (CCR, the "head" form, korig, vonly, qshare,
qshare_kv, qshare_bd, mix_fold, another scaling, TE's own quantiser, widths that are not whole blocks of 32); model.nvfp4 and the norm
chains stay refused (mx_wire.check_chains).

What runs. In a training forward with in_proj_qkv a quantised FP8Linear (mx_rows), the two projections of write_inputs are that
linear's own quantised path (FP8Linear.quantised, the code the trunk's in-projections run) on a row range of its weight:

    q rows   W[:nq]  on x (wsl) or x_q (qmix)         k, v rows   W[nq:]  on x_w

each input quantised once by the linear's quantiser (both orientations from one read, the flag's bits, the saved column-wise copy),
as the trunk's input is once a linear. The ranges start and end on blocks of 32, so the weight's bytes, scales and both copies are
those of the whole weight's quantisation. x_w and x_q are formed in float32 (the lerp of write_inputs, an addcmul) and quantised as
they were formed, not through a bf16 copy (the trunk's rule, ced/model/fp8.py: Fprop quantises the activation as it arrives); the
fused write mix's group read hands m' and the block's opening x in bf16, and under qmix's second round x_w and x_q themselves
("lerped"): those are quantised as they came. in_proj_a and in_proj_b are high precision (apply_fp8 keeps them: one output per value
head, and they train under AdamW): they read x_w's one cast to the GEMMs' dtype. attn_res_fused is allowed with model.fp8 for these
forms (attn_res.check_fused): the group read's outputs feed the quantised projections; E alone stays refused with fp8. In evaluation,
scoring, prefill / extend and generation the quantised path is off (training mode only) and the layer is the BF16 model bit for bit.
The head-by-head reference (write_inputs_reference) runs each head's rows through the same quantised linear on its own x_w^h.

The producer form (EncoderConfig.mix_fold "producer"; mx_wire.py "THE PRODUCER FORM OF THE ARMS' MIXES"): with model.fp8 on MXFP8 and
"in_norm" among model.fp8.chains, the layer's input-norm pass forms x_w (and qmix's x_q) from m' in its registers and writes the operands
of x, x_w and x_q once (ced/kernels/mx_mix_chain.py); write_inputs then hands them to the two row ranges of in_proj_qkv
(FP8Linear.quantised(... op=)) and gives in_proj_a / in_proj_b the one copy of x_w in the GEMMs' dtype, shared as the cast above is. The
lerp fold (mix_fold true / "lerp") is the bf16 GEMM's and stays refused with model.fp8.

CCR, as built
-------------
The checkpoint of a token at position p of its document is the state after the document's first C floor(p / C)
tokens: zero for p < C. Every GDN layer l after the first adds to its delta-rule output, before the gated output norm,

    o_t  +=  gate (*) ( S^{l'}_{ckpt(t)} )^T q_t        l' the previous GDN layer, gate (H_v,) zeros at the start

with q_t layer l's own (normalised, scaled) query, value head j reading key head j // (H_v / H_k). Three paths:
- fused (training and scoring: kernel "ced", varlen boundaries, no cache): the writer l' calls our kernel through
  ced_delta_rule_states, which returns h, the state at the start of every chunk of C = 64 tokens, chunks aligned to
  every segment start of cu_seqlens (row and document starts) and kept by the kernel for its own backward anyway. The
  reader lays its q out by chunk (a gather with the exact inverse map) and does ONE batched matmul per chunk and head,
  q_chunk (C x d_k) @ h_chunk (d_k x d_v); dL/dh goes back into the writer's state-gradient recursion through the
  kernel's existing EXT input (the second-pass read's). No intra-chunk recompute. C must be the kernel's chunk.
- non-fused (CPU, cached chunks: prefill / extend, decoding): the writer forms the state after every position of its
  call (the recurrence, or the chunked form with all_states) and gathers each position's checkpoint
  (boundary_states); the cache carries one more tensor per writer, GDNState.ckpt, the checkpoint the next token reads.
- reference: the same as the non-fused path with GDN kernel "recurrence"; boundary_states_reference is the gather
  written as a loop over positions.

CCR v2, as built
-----------------------------------------------------------------------------------------------------------------------
Two keys, read only with gdn_ccr_spacing > 0; at their defaults CCR is v1 bit for bit (tests/test_ccr_v2.py):
- gdn_ccr_target, the GDN layer a reader reads (ccr_sources): "previous" (v1: the previous GDN layer in stack order);
  "lowest" (the model's lowest GDN layer: every other GDN layer reads it; it reads nothing); "group_first" (the first GDN
  layer of the reader's hybrid group of full_attn_interval layers: every other GDN layer of the group reads it; it reads
  nothing). The writers are the layers some reader reads; the walks hand on "the last record written" (hand_record): a
  layer that writes replaces it, a layer that does not leaves it, and a layer that does not read is called without it
  (record_in). Under "previous" that is v1's chain (every GDN layer but the last writes; the last's None ends it). Each
  record names its writer (CCRChunks.source / CCRBoundary.source) and the reader checks it against its own ccr_source.
- gdn_ccr_translator "key_head": per reader layer and key head a d_k x d_k query translator M, identity at the start;
  the read uses q_x = M q_t in place of q_t:  o_t += gate (*) ( S^{src}_{ckpt(t)} )^T (M_h q_t),  h the value head's key
  head. q_t is the reader's query as v1's read takes it: L2-normalised and scaled by d_k^-1/2 (M acts after the norm and
  the scale: on the runs' path the fused prep kernel hands out only that q), so at M = I the read is v1's, and M can
  rotate the query into the source's key space and also set the read's size by the query's direction (the read is
  linear, so this is a direction-dependent scale, not a temperature; a norm after M would remove it, and the gate, one
  scalar a value head, cannot give it). Stored as D = M - I ((H_k, d_k, d_k), zeros), and
  formed as q + q D^T: at the start D q is exactly zero, so the read is v1's bit for bit on every device and precision
  (TF32 and bf16 included); and weight decay on D, when asked for, pulls M toward the identity
  (optim.ccr_translator_decay; ced/train/optim.py). One translator per key head, not per value head: value heads 2i and
  2i + 1 of the source were written with the source's key head i, one key space, which one M maps into.
- gdn_ccr_gate_init: the start of every reader's gate, 0.0 by default (v1's zero start; the model starts as the trunk).
  The ccr-v2 entries use 0.09 (tools/analysis/ccr_v2_gate_init.py).

The gate floor
------------------------------------------------------------------------------------------------------------------------
- gdn_ccr_gate_floor: 0.0 by default (no floor; nothing below runs and every number is as before). f > 0: after every
  optimizer step the trainer calls floor_gates_, which clamps every reader's gate in place to at least f (projected AdamW:
  the update is the optimizer's, then the gate is put back into [f, inf)). The forward never reads f: a model with a floor
  computes what the same tensors compute without one, so causality, cached generation and the paths' agreement are the
  target's own. A gate the gradient pushes down sits at f (its Adam moments keep the push, and it rises again once the
  gradient turns); a gate the gradient pushes up rises above f. The sign is the start's (positive): the floor is a bound
  on g, not on |g|; M (per key head) carries a sign for both value heads of its key head. Refused by name: a value that is
  not a finite number >= 0, a nonzero floor with CCR off, a floor above gdn_ccr_gate_init (the start must lie at or above
  it), and (ced/train/trainer.py) a floor with optim.zero_gate_decay other than "none".

Optimizer (ced/train/optim.py, module docstring): every tensor of the arms, write_lambda, write_query and ccr_gate, gets R1-entry's
rule for its readers' tensors: AdamW at the run's LR (times adamw_lr_scale under Muon) in the group WITH the run's weight decay (until
then write_lambda and ccr_gate had none). Under optim.zero_gate_decay "none" the zero-initialised gates (ccr_gate, write_lambda and the
qmix variant's qmix_lambda; optim.ZERO_GATE_SUFFIXES) have none; the depth queries (write_query, qmix_query) keep the rule above. Under
optim.mix_query_decay "none_qmix" qmix_query's pseudo-query has none, under "none_write_qmix" write_query's too
(optim.MIX_QUERY_DECAY); their key-norm gains keep the rule above under every value.

Activation checkpointing (TrainConfig.activation_checkpoint; ced/model/act_checkpoint.py; section 13):
- "layer": each layer call of the walk (attn_res.EntryWalk._run_arms on R1-entry, Encoder._stack_arms on the plain
  stack) runs under its own checkpoint with the arguments it takes without one: the stream, the write mix's sources
  (the finished blocks, held by reference: the walk holds them anyway) and CCR's record of the previous GDN layer
  (held by reference). A layer's own record is an OUTPUT of its checkpointed call (in its new cache entry): the walk
  takes it out after the call returns, so the recomputation in the backward (the same function on the same arguments)
  makes the record again for its own saved tensors only and hands it to no one; nothing outside the call is written
  twice.
- "block": one checkpoint per hybrid group (attn_res.walk_block, Encoder._stack_blocks): the record of the block's last
  GDN layer is an output of the block's call (the walk it returns, or the plain stack's (stream, record) pair) and the
  input of the next block's call; inside a block the records are handed as without checkpointing.
- "host" and "host_overlap": the write mix only (its kept inputs are R1-entry's: the stream of a layer inside a
  block moves, the finished blocks it mixes are held by reference). Refused with CCR (check_checkpoint_mode): the
  record (the writer's chunk-start states h, one fp32 stream vector a token a writer at d_k 128) would stay on the
  device, held by its reader's checkpoint until the reader's backward; moving it to the host is state handling that is
  not built. The monitoring reads (the stream RMS, the activation maxima) stay in the first forward only, as for
  R1-entry.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ced.config import ModelConfig
from ced.model.norms import compute_dtype

WRITE_MIX = ("none", "layer", "head")
WRITE_VARIANTS = ("none", "qmix", "korig", "qshare", "qshare_kv", "qshare_bd", "vonly", "kqmix")  # EncoderConfig.gdn_write_variant (module docstring)
FUSED_VARIANTS = ("none", "qshare", "qshare_kv", "qshare_bd", "qmix")  # the values the fused write mix takes (attn_res_fused; check_write_variant)
SPLIT_VARIANTS = ("qshare_kv", "qshare_bd")  # the write-quantity split on qshare (module docstring): the q rows read x_w in both
MX_VARIANTS = ("none", "qmix")  # the wsl variants wired on model.fp8 (module docstring, "On MXFP8"): wsl, wsl-nowd and qmix, the write mix's "layer" form
READ_MIX_VARIANTS = ("qmix", "kqmix")  # the variants with a SECOND depth mix (qmix_lambda, qmix_query; x_read): qmix's feeds the q rows, kqmix's the q and k rows (arm KQ)
CCR_TARGETS = ("previous", "lowest", "group_first")  # EncoderConfig.gdn_ccr_target (CCR v2; module docstring)
CCR_TRANSLATORS = ("none", "key_head")  # EncoderConfig.gdn_ccr_translator
# what a layer's GDNArms may hold (state-dict suffixes)
NAMES = ("write_query", "write_lambda", "ccr_gate", "ccr_translator_delta", "qmix_query", "qmix_lambda")
TRANSLATOR = "ccr_translator_delta"  # the stored D = M - I of CCR v2's translator (gdn_ccr_translator "key_head")


def write_on(cfg: ModelConfig) -> bool:
    return cfg.encoder.gdn_write_mix != "none"


def ccr_on(cfg: ModelConfig) -> bool:
    return cfg.encoder.gdn_ccr_spacing != 0


def any_on(cfg: ModelConfig) -> bool:
    return write_on(cfg) or ccr_on(cfg)


def write_variant(cfg: ModelConfig) -> str:
    """EncoderConfig.gdn_write_variant ("none" in every model before the key; read with gdn_write_mix "layer" only)."""
    return cfg.encoder.gdn_write_variant


def translator_on(cfg: ModelConfig) -> bool:
    return ccr_on(cfg) and cfg.encoder.gdn_ccr_translator != "none"


def gate_floor_on(cfg: ModelConfig) -> bool:
    """Whether the trainer holds the CCR gates at or above model.encoder.gdn_ccr_gate_floor (CCR on and a floor > 0)."""
    return ccr_on(cfg) and cfg.encoder.gdn_ccr_gate_floor != 0


def ccr_sources(cfg: ModelConfig) -> dict[int, int]:
    """{reader's layer index: the layer index of the GDN layer it reads}, in stack order, by model.encoder.gdn_ccr_target
    ({} with CCR off). "previous": every GDN layer but the first reads the GDN layer before it (v1); "lowest": every GDN
    layer but the lowest reads the lowest; "group_first": every GDN layer of a hybrid group (full_attn_interval layers)
    but the group's first GDN layer reads that one. The layers read are the writers (ccr_writers)."""
    if not ccr_on(cfg):
        return {}
    gdn = gdn_layers(cfg)
    target = cfg.encoder.gdn_ccr_target
    if target == "previous":
        return {gdn[n]: gdn[n - 1] for n in range(1, len(gdn))}
    if target == "lowest":
        return {i: gdn[0] for i in gdn[1:]}
    assert target == "group_first", target
    S = cfg.encoder.full_attn_interval
    first: dict = {}
    out = {}
    for i in gdn:
        g = i // S
        if g in first:
            out[i] = first[g]
        else:
            first[g] = i
    return out


def ccr_writers(cfg: ModelConfig) -> list[int]:
    """The GDN layers some reader reads (ccr_sources' values), in stack order."""
    return sorted(set(ccr_sources(cfg).values()))


def check_gdn_arms(cfg: ModelConfig) -> None:
    """Refuse the two keys wherever they are not built, by name (ValueError). Off: nothing is checked but the values."""
    e = cfg.encoder
    if e.gdn_write_mix not in WRITE_MIX:
        raise ValueError(f"model.encoder.gdn_write_mix {e.gdn_write_mix!r}: one of {WRITE_MIX}")
    if isinstance(e.gdn_ccr_spacing, bool) or not isinstance(e.gdn_ccr_spacing, int) or e.gdn_ccr_spacing < 0:
        raise ValueError(f"model.encoder.gdn_ccr_spacing {e.gdn_ccr_spacing!r}: 0 (off) or a positive spacing in tokens")
    if e.gdn_ccr_target not in CCR_TARGETS:
        raise ValueError(f"model.encoder.gdn_ccr_target {e.gdn_ccr_target!r}: one of {CCR_TARGETS}")
    if e.gdn_ccr_translator not in CCR_TRANSLATORS:
        raise ValueError(f"model.encoder.gdn_ccr_translator {e.gdn_ccr_translator!r}: one of {CCR_TRANSLATORS}")
    if not ccr_on(cfg) and (e.gdn_ccr_target, e.gdn_ccr_translator) != ("previous", "none"):
        raise ValueError(f"model.encoder.gdn_ccr_target {e.gdn_ccr_target!r} / gdn_ccr_translator {e.gdn_ccr_translator!r} "
                         "are read with model.encoder.gdn_ccr_spacing > 0 (CCR on); with CCR off leave them 'previous' / 'none'")
    gi = e.gdn_ccr_gate_init
    if isinstance(gi, bool) or not isinstance(gi, (int, float)) or not math.isfinite(gi):
        raise ValueError(f"model.encoder.gdn_ccr_gate_init {gi!r}: the start of the CCR gates, a finite number (0.0: v1's zero start)")
    if not ccr_on(cfg) and gi != 0:
        raise ValueError(f"model.encoder.gdn_ccr_gate_init {gi!r} is read with model.encoder.gdn_ccr_spacing > 0 (CCR on); "
                         "with CCR off leave it 0.0")
    gf = e.gdn_ccr_gate_floor
    if isinstance(gf, bool) or not isinstance(gf, (int, float)) or not math.isfinite(gf) or gf < 0:
        raise ValueError(f"model.encoder.gdn_ccr_gate_floor {gf!r}: the floor of the CCR gates, a finite number >= 0 "
                         "(0.0: no floor)")
    if not ccr_on(cfg) and gf != 0:
        raise ValueError(f"model.encoder.gdn_ccr_gate_floor {gf!r} is read with model.encoder.gdn_ccr_spacing > 0 (CCR on); "
                         "with CCR off leave it 0.0")
    if gf != 0 and gf > gi:  # (no floor: any start, a negative one included, as before)
        raise ValueError(f"model.encoder.gdn_ccr_gate_floor {gf!r} is above model.encoder.gdn_ccr_gate_init {gi!r}: the gates "
                         "would start below their floor; the start must lie at or above it")
    check_write_variant(cfg)
    check_mix_fold(cfg)
    check_write_mix_row_term(cfg)
    if not any_on(cfg):
        return
    what = (f"model.encoder.gdn_write_mix {e.gdn_write_mix!r}" if write_on(cfg) else
            f"model.encoder.gdn_ccr_spacing {e.gdn_ccr_spacing!r}")
    if cfg.arch != "control":
        raise ValueError(f"{what} is built for arch 'control', not {cfg.arch!r}")
    if write_on(cfg) and e.attn_res != "block_entry":
        raise ValueError(f"model.encoder.gdn_write_mix {e.gdn_write_mix!r} reads R1-entry's stored sources: it is built with "
                         f"model.encoder.attn_res 'block_entry', not {e.attn_res!r}")
    if ccr_on(cfg) and e.attn_res not in ("none", "block_entry"):
        raise ValueError(f"model.encoder.gdn_ccr_spacing {e.gdn_ccr_spacing!r} is built with model.encoder.attn_res 'none' "
                         f"or 'block_entry', not {e.attn_res!r}")
    if "gdn" not in e.layer_kinds():
        raise ValueError(f"{what}: the encoder has no GDN layer")
    if e.loops != 1 or cfg.loop.pass2_gate != "none" or cfg.loop.recur_layers:
        raise ValueError(f"{what} is built for one pass through the stack (encoder.loops 1, loop.pass2_gate 'none', no "
                         "loop.recur_layers): another pass would need records and sources of its own")
    if e.layer_scalars != "none" or e.canon_kernel:
        raise ValueError(f"{what} is not built together with model.encoder.layer_scalars or canon_kernel: nothing tests the pair")
    if cfg.nvfp4.mode not in ("none", "mlp_only"):
        # "mlp_only" (the NVFP4 MLP arm, 2026-10-27) leaves every mixer linear a plain nn.Linear: the arms' mixers are untouched
        raise ValueError(f"{what} is not built together with model.nvfp4: nothing tests the pair")
    check_fp8_forms(cfg, what)  # model.fp8: the wired forms only (module docstring, "On MXFP8")
    if cfg.mtp.enabled:
        raise ValueError(f"{what} is not built together with model.mtp (its drafter computes the GDN layers itself)")
    if ccr_on(cfg) and e.gdn.kernel == "fla":
        raise ValueError("model.encoder.gdn_ccr_spacing needs the chunk states of our kernel: GDN kernel 'ced' (or the "
                         "reference kernels 'recurrence' / 'chunked'), not 'fla'")
    if ccr_on(cfg) and e.gdn.chunk_backend != "triton":  # D (GDNConfig.chunk_backend): refused here, before any kernel
        raise ValueError(f"model.encoder.gdn_ccr_spacing needs the chunk states of our Triton kernel: GDN chunk_backend "
                         f"{e.gdn.chunk_backend!r} has none (set chunk_backend 'triton')")
    if ccr_on(cfg) and not ccr_sources(cfg):
        raise ValueError(f"model.encoder.gdn_ccr_target {e.gdn_ccr_target!r}: no GDN layer reads another in this stack "
                         f"(GDN layers {gdn_layers(cfg)}, full_attn_interval {e.full_attn_interval})")


def check_fp8_forms(cfg: ModelConfig, what: str) -> None:
    """The GDN-side arms with model.fp8 (module docstring, "On MXFP8"): built for the write mix's "layer" form with the variants
    MX_VARIANTS ("none": wsl and wsl-nowd; "qmix"), on MXFP8 with the emulated backend or Transformer Engine on our operands. Every
    other form is refused by name (ValueError) with the reason: CCR, the write mix's "head" form, the variants korig, vonly, qshare,
    qshare_kv and qshare_bd, model.encoder.mix_fold (the lerp fold's GEMMs are bf16), a scaling other than "mxfp8" (the other
    scalings quantise a tensor with one scale, and a row range of the weight would be another tensor), TE's own quantiser
    (te_operands "te": te.Linear holds the whole weight), and widths that are not whole MX blocks. fp8 off: nothing is checked."""
    q, e = cfg.fp8, cfg.encoder
    if q.mode == "none":
        return
    why = []
    if ccr_on(cfg):
        why.append(f"CCR (model.encoder.gdn_ccr_spacing {e.gdn_ccr_spacing!r}): its readers are not wired")
    if write_on(cfg):
        if e.gdn_write_mix != "layer":
            why.append(f"arm A's {e.gdn_write_mix!r} form (only 'layer' is wired: the 'head' form's projections of the difference are not)")
        elif e.gdn_write_variant not in MX_VARIANTS:
            why.append(f"the wsl variant {e.gdn_write_variant!r} (wired: {MX_VARIANTS}; the others' projections are not)")
        if mix_fold_form(e) == "lerp":  # ('producer' is the MXFP8 form: check_mix_fold says what it needs)
            why.append("model.encoder.mix_fold true / 'lerp' (the lerp fold's GEMMs read bf16 operands: they are not the MXFP8 "
                       "linear; the producer form is)")
    if q.scaling != "mxfp8":
        why.append(f"model.fp8.scaling {q.scaling!r} (wired: 'mxfp8'; a row range of the weight is not the tensor the other "
                   "scalings quantise with one scale)")
    if getattr(q, "backend", "emulated") == "te" and getattr(q, "te_operands", "te") != "ours":
        why.append("model.fp8.backend 'te' with te_operands 'te' (Transformer Engine's own te.Linear holds the whole in_proj_qkv "
                   "weight; wired: te_operands 'ours')")
    g = e.gdn
    nq = g.n_k_heads * g.k_head_dim
    widths = {"d_model": cfg.d_model, "the q rows (n_k_heads x k_head_dim)": nq,
              "the k, v rows (n_k_heads x k_head_dim + n_v_heads x v_head_dim)": nq + g.n_v_heads * g.v_head_dim}
    bad = {k: v for k, v in widths.items() if v % 32}
    if bad and write_on(cfg):
        why.append(f"widths that are not whole blocks of 32 (the arms' two projections of in_proj_qkv are row ranges of its weight: "
                   f"{bad})")
    if why:
        raise ValueError(f"{what} with model.fp8 (mode {q.mode!r}) is not built for: " + "; ".join(why) +
                         f". Built on MXFP8: arm A's 'layer' form with gdn_write_variant in {MX_VARIANTS} (wsl, wsl-nowd, qmix)")


QMIX_KEEP = 2  # the sources a layer's static query mix keeps


def check_qmix_select(cfg: ModelConfig) -> None:
    """EncoderConfig.gdn_qmix_select_step (module docstring): 0 (off, the default: nothing else is checked) or the optimizer step
    S >= 1 at whose start each layer's 2 sources are chosen. Refused by name (ValueError): a value that is not a non-negative int (a
    bool included), the key with a variant other than "qmix" (or another write mix), and with the paths that form the query mix
    elsewhere: the fused write mix (attn_res_fused: the group read's kernel mixes all n sources), mix_fold and model.fp8 (their kernels
    and tests take the full query mix). The reference path trains it. The CP head is refused by check_write_variant (every variant
    is), data parallelism and CUDA graphs by the trainer (the choice is one process's statistic and the switch is a Python change)."""
    e = cfg.encoder
    s = e.gdn_qmix_select_step
    if isinstance(s, bool) or not isinstance(s, int) or s < 0:
        raise ValueError(f"model.encoder.gdn_qmix_select_step {s!r}: 0 (off) or the optimizer step >= 1 at whose start each layer's "
                         f"{QMIX_KEEP} sources are chosen")
    if s == 0:
        return
    what = f"model.encoder.gdn_qmix_select_step {s!r}"
    if e.gdn_write_variant != "qmix" or e.gdn_write_mix != "layer":
        raise ValueError(f"{what} restricts the wsl variant qmix's query mix: it is read with model.encoder.gdn_write_variant 'qmix' "
                         f"(and gdn_write_mix 'layer'), not {e.gdn_write_variant!r} / {e.gdn_write_mix!r}")
    if e.attn_res_fused:
        raise ValueError(f"{what} is not built together with model.encoder.attn_res_fused (the fused group read mixes all the sources; "
                         "the kernel that reads only the chosen 2 is not built): the reference path trains it")
    if mix_fold_form(e) != "none":
        raise ValueError(f"{what} is not built together with model.encoder.mix_fold {e.mix_fold!r}: nothing tests the pair")
    if cfg.fp8.mode != "none":
        raise ValueError(f"{what} is not built together with model.fp8 (mode {cfg.fp8.mode!r}): nothing tests the pair")


def check_write_variant(cfg: ModelConfig) -> None:
    """EncoderConfig.gdn_write_variant (the wsl variants; module docstring): its value, and where it is not built, by name
    (ValueError). "none" (the default): nothing else is checked. A variant is built for the write mix's "layer" form on R1-entry
    (check_gdn_arms checks the rest of A's conditions); refused with "head" (its per-head lambdas name the k heads korig takes
    off x_w, and qmix's q lambda would need a form of its own: neither asked), with CCR and with the CP head (nothing tests the
    pairs). korig and vonly are also refused with the fused write mix (attn_res_fused: their pairs have no fused test) and with
    MXFP8 (check_fp8_forms). qshare takes it (FUSED_VARIANTS): its q rows read the x_w the layer forms from the group read's m',
    so the group read is wsl's, query for query (tests/test_wsl_qshare.py); so do qshare_kv and qshare_bd, which only choose
    which projection rows read that x_w (tests/test_wsl_split.py). qmix takes it too: its block's group read also forms each
    fused layer's query mix m'_q, the same sources read once for both (NQ 1 + 2 G, write_mix.QMIX_QUERIES; attn_res.check_fused
    refuses a block with more GDN layers than that holds; tests/test_qmix_fused.py)."""
    e = cfg.encoder
    v = e.gdn_write_variant
    if v not in WRITE_VARIANTS:
        raise ValueError(f"model.encoder.gdn_write_variant {v!r}: one of {WRITE_VARIANTS}")
    check_qmix_select(cfg)
    if v == "none":
        return
    what = f"model.encoder.gdn_write_variant {v!r}"
    if e.gdn_write_mix != "layer":
        raise ValueError(f"{what} is read with model.encoder.gdn_write_mix 'layer' (arm A, one lambda a layer), not "
                         f"{e.gdn_write_mix!r}")
    if ccr_on(cfg):
        raise ValueError(f"{what} is not built together with model.encoder.gdn_ccr_spacing {e.gdn_ccr_spacing!r} (CCR): "
                         "nothing tests the pair")
    if e.attn_res_fused and v not in FUSED_VARIANTS:
        raise ValueError(f"{what} is not built together with model.encoder.attn_res_fused (the fused write mix, "
                         "ced/kernels/write_mix.py): the reference path trains it; "
                         f"of the variants only {FUSED_VARIANTS[1:]} take the fused write mix")
    if cfg.cp_head.enabled:
        raise ValueError(f"{what} is not built together with model.cp_head: nothing tests the pair")


MIX_FOLD_FORMS = (False, True, "lerp", "producer", "producer-rc", "producer-p2", "producer-p3")  # EncoderConfig.mix_fold; True is "lerp" (the first form)


def mix_fold_form(e) -> str:
    """The form EncoderConfig.mix_fold `e` (an EncoderConfig) names: "none" (False, the default), "lerp" (True or "lerp": the lerps
    in the projection GEMM's load, ced/kernels/fold_mix.py) or "producer" (the input norm's pass forms the mixed operands,
    ced/kernels/mx_mix_chain.py). Anything else: ValueError."""
    v = e.mix_fold
    if v is False:
        return "none"
    if v is True or v == "lerp":
        return "lerp"
    if isinstance(v, str) and v in ("producer", *RC_FORMS):  # (the recompute and the layer-side values are the producer form plus flags: mix_rc, mix_p)
        return "producer"
    raise ValueError(f"model.encoder.mix_fold {v!r}: true or false (true is 'lerp'), or 'lerp', 'producer' or 'producer-rc' (and, on top of 'producer-rc', 'producer-p2' and 'producer-p3')")


def mix_rc(e) -> bool:
    """Whether EncoderConfig.mix_fold recomputes the group read's extra mixes: 'producer-rc' and the values built on it ('producer-p2', 'producer-p3':
    mix_p, below): the producer form, and the fused write mix's group read recomputes the block's extra mixes in the backward instead of storing them
    (ced/kernels/write_mix_rc.py). False for every other value (mix_fold_form says what is valid)."""
    return isinstance(e.mix_fold, str) and e.mix_fold in RC_FORMS


def check_mix_fold(cfg: ModelConfig) -> None:
    """EncoderConfig.mix_fold (build (a) of the 4B-step audit): false (the default), "lerp" (true is "lerp": the lerps in the
    projection GEMM's input load, ced/kernels/fold_mix.py; bf16 GEMMs) or "producer" (the input norm's pass forms the mixed
    operands and writes them once in MXFP8, ced/kernels/mx_mix_chain.py; model.fp8). Both are read with the write mix's "layer"
    form and the variants whose lerps the fold forms ("none": wsl, wsl-nowd; "qmix"); refused by name with every other form
    (ValueError). "producer" is also refused without model.fp8 on the MXFP8 scaling with our operands (the emulated backend on
    our Triton quantiser, or Transformer Engine on our operands) and 'in_norm' among model.fp8.chains: the norm pass is the
    chain's. False (the default): nothing else is checked."""
    e = cfg.encoder
    form = mix_fold_form(e)
    if form == "none":
        return
    if e.gdn_write_mix != "layer" or e.gdn_write_variant not in ("none", "qmix"):
        raise ValueError(f"model.encoder.mix_fold {e.mix_fold!r} forms arm A's lerps x_w (and qmix's x_q) in the projection GEMM's "
                         f"load or the input norm's pass: it is built with gdn_write_mix 'layer' and gdn_write_variant 'none' or "
                         f"'qmix', not {e.gdn_write_mix!r} / {e.gdn_write_variant!r}")
    if form != "producer":
        return
    q = cfg.fp8
    ours = (getattr(q, "backend", "emulated") == "emulated" and getattr(q, "quantiser", "torch") == "triton") or (
        getattr(q, "backend", "emulated") == "te" and getattr(q, "te_operands", "te") == "ours")
    if q.mode != "linears" or q.scaling != "mxfp8" or not ours or "in_norm" not in tuple(getattr(q, "chains", ())):
        raise ValueError(
            f"model.encoder.mix_fold 'producer' writes the arms' mixed operands in MXFP8 from the input norm's pass: it needs "
            f"model.fp8 mode 'linears', scaling 'mxfp8', our operands (backend 'emulated' with quantiser 'triton', or backend "
            f"'te' with te_operands 'ours') and 'in_norm' among model.fp8.chains; got mode {q.mode!r}, scaling {q.scaling!r}, "
            f"backend {getattr(q, 'backend', 'emulated')!r}, quantiser {getattr(q, 'quantiser', 'torch')!r}, te_operands "
            f"{getattr(q, 'te_operands', 'te')!r}, chains {list(getattr(q, 'chains', ()))!r}")
    if mix_rc(e) and not (e.attn_res_fused and e.attn_res == "block_entry"):
        raise ValueError(f"model.encoder.mix_fold {e.mix_fold!r} recomputes the fused write mix's group read: it needs model.encoder.attn_res "
                         f"'block_entry' with attn_res_fused true (the group read is the only place a mix is stored for the backward; the "
                         f"unfused write mix's depth_mix keeps its own autograd tensors and is not built), got attn_res {e.attn_res!r}, "
                         f"attn_res_fused {e.attn_res_fused!r}")


WRITE_MIX_ROW_TERMS = ("stored_o", "pdp")


def check_write_mix_row_term(cfg: ModelConfig) -> None:
    """EncoderConfig.write_mix_row_term (the CroissAttn fix for the fused write mix; ced/kernels/write_mix.py): "stored_o" (the default, nothing else
    is checked) or "pdp", read with the fused write mix (attn_res_fused with gdn_write_mix "layer": the group read is the only place a mix is stored for the
    backward); a value that is neither, and "pdp" without the fused write mix, are refused by name (ValueError). With mix_fold "producer-rc" and the forms built
    on it the extra mixes are recomputed in float32 in the backward: the row term is already exact, and "pdp" is accepted and inert there."""
    e = cfg.encoder
    if e.write_mix_row_term not in WRITE_MIX_ROW_TERMS:
        raise ValueError(f"model.encoder.write_mix_row_term {e.write_mix_row_term!r}: one of {WRITE_MIX_ROW_TERMS} ('stored_o': the extra queries' mixes in "
                         "the GEMMs' dtype and the row term from them as stored, as always; 'pdp': stored in the sources' dtype, the row term exact)")
    if e.write_mix_row_term == "stored_o":
        return
    if e.write_mix_storage != "float32":  # (the merge of croissattn-fix with the float32 default of write_mix_storage, 2026-10-26)
        raise ValueError(f"model.encoder.write_mix_row_term {e.write_mix_row_term!r} stores the extra queries' mixes in the sources' dtype, which is what "
                         f"model.encoder.write_mix_storage 'float32' (the default) does; with write_mix_storage {e.write_mix_storage!r} the two options ask for "
                         "opposite storages: leave one of them at its default")
    if not (e.attn_res_fused and e.gdn_write_mix == "layer" and e.attn_res == "block_entry"):
        raise ValueError(f"model.encoder.write_mix_row_term {e.write_mix_row_term!r} is read by the fused write mix's group read: it needs "
                         f"model.encoder.attn_res 'block_entry', attn_res_fused true and gdn_write_mix 'layer' (without them the depth mixes run on the plain "
                         f"path, whose backward is already rowsum(P . dP)), got attn_res {e.attn_res!r}, attn_res_fused {e.attn_res_fused!r}, "
                         f"gdn_write_mix {e.gdn_write_mix!r}")


CHECKPOINT_MODES = ("none", "layer", "block", "host", "host_overlap")  # act_checkpoint.MODES, every one built for A
CCR_CHECKPOINT_MODES = ("none", "layer", "block")  # with CCR: the record stays on the device (module docstring)


def check_checkpoint_mode(cfg: ModelConfig, mode: str) -> None:
    """TrainConfig.activation_checkpoint with the GDN-side arms (module docstring): every mode with the write mix;
    "none", "layer" and "block" with CCR, whose record the host modes would leave on the device (refused by name,
    ValueError). Off: nothing is checked."""
    if not any_on(cfg):
        return
    if ccr_on(cfg) and mode not in CCR_CHECKPOINT_MODES:
        raise ValueError(f"activation_checkpoint {mode!r} is not built together with model.encoder.gdn_ccr_spacing "
                         f"{cfg.encoder.gdn_ccr_spacing!r} (CCR): its record, the writer's chunk-start states, is held on "
                         "the device by its reader's checkpoint and moving it to the host is not built; "
                         f"{', '.join(repr(m) for m in CCR_CHECKPOINT_MODES)} are")
    if mode not in CHECKPOINT_MODES:
        raise ValueError(f"activation_checkpoint {mode!r}: one of {CHECKPOINT_MODES}")


def refuse_gdn_arms(cfg: ModelConfig, what: str) -> None:
    """For code that computes a GDN layer itself (the decode engine, the MTP drafter, a cost harness): with an arm on
    it would compute another model under this model's name. ValueError, by name."""
    if any_on(cfg):
        raise ValueError(f"{what} computes the GDN layers itself and does not run the GDN-side arms "
                         f"(model.encoder.gdn_write_mix {cfg.encoder.gdn_write_mix!r}, gdn_ccr_spacing "
                         f"{cfg.encoder.gdn_ccr_spacing!r}; ced/model/gdn_arms.py): use the model's own forward or prefill / extend")


def gdn_layers(cfg: ModelConfig) -> list[int]:
    return [i for i, kind in enumerate(cfg.encoder.layer_kinds()) if kind == "gdn"]


class GDNArms(nn.Module):
    """What one GDN layer holds of the arms (attached as layer.gdn_arms, the layer's last submodule).

    write_mix: the config's value for this layer ("none" in layer 0). write_query: the layer's DepthQuery over [b_0 ..
    b_{n-1}] (None in block 1: one source). write_lambda: (1,) or (H_k + H_v), zeros. ccr_read: the layer reads a GDN
    layer's checkpoint (ccr_sources; v1: every GDN layer but the first); ccr_gate (H_v,) zeros. ccr_write: a later GDN
    layer reads this layer's checkpoints (ccr_writers; v1: every GDN layer but the last). spacing: C. CCR v2, plain
    attributes (never in the state dict): layer, this layer's index in the stack; ccr_source, the index of the layer it
    reads (None if it reads none); ccr_target, the config's target; translator, the config's translator in a reader ("none"
    elsewhere). ccr_translator_delta (H_k, d_k, d_k), zeros: the stored M - I of the query translator (a reader with
    gdn_ccr_translator "key_head"). gate_init: the gates' start (EncoderConfig.gdn_ccr_gate_init, 0.0 by default).
    gate_floor: a plain attribute (never in the state dict), EncoderConfig.gdn_ccr_gate_floor (0.0: no floor), the bound
    floor_gates_ holds a reader's gate at after every optimizer step. variant (EncoderConfig.gdn_write_variant; the wsl
    variants, module docstring): a plain attribute, "none" in every layer without the write mix's "layer" form; under
    "qmix" also qmix_lambda (1,) zeros and qmix_query, a DepthQuery of its own over the same sources (None in block 1, as
    write_query). No random number: zeros, or the gates at gate_init."""

    def __init__(self, d: int, eps: float, hk: int, hv: int, write_mix: str, write_query: bool, ccr_read: bool,
                 ccr_write: bool, spacing: int, layer: int | None = None, ccr_source: int | None = None,
                 ccr_target: str = "previous", translator: str = "none", dk: int | None = None, gate_init: float = 0.0,
                 gate_floor: float = 0.0, variant: str = "none"):
        super().__init__()
        self.gate_floor = float(gate_floor) if ccr_read else 0.0
        from ced.model.attn_res import DepthQuery

        assert variant == "none" or write_mix == "layer", "the wsl variants are read with arm A's 'layer' form (check_write_variant)"
        self.write_mix = write_mix
        self.variant = variant
        self.write_query = DepthQuery(d, eps) if write_query else None
        # the fused write mix (model.encoder.attn_res_fused with gdn_write_mix "layer"; ced/kernels/write_mix.py): a plain
        # attribute (no tensor; never in the state dict), set by build on the layers whose m' the block's group read forms
        # (those with a write query: blocks 2 and up); False in every other model
        self.fused_write = False
        # EncoderConfig.write_mix_storage (ced/kernels/write_mix.py "STORAGE"): the dtype name the fused group read stores the block's extra mixes in,
        # "float32" (the default) or "bf16" (explicit opt-in); a plain attribute (never in the state dict), set by build, read by
        # attn_res.EntryWalk._group_read
        self.write_mix_storage = "float32"
        # model.encoder.mix_fold (ced/kernels/fold_mix.py): a plain attribute (never in the state dict), set by build; False in every
        # other model
        self.mix_fold = False
        # EncoderConfig.gdn_qmix_select_step (module docstring): qmix_static, a plain attribute (never in the state dict), True once
        # this layer's 2 sources are chosen (select_qmix_pairs, or a load of a state dict whose qmix_keep holds a choice): read_source
        # then mixes those 2 sources only. False in every other model and before the choice. qmix_keep, the (2,) int64 buffer of the
        # choice (-1 before it), exists only in a layer that has more than 2 sources under the option (register_qmix_keep, called by
        # build); nothing else is added.
        self.qmix_static = False
        if write_mix != "none":
            self.write_lambda = nn.Parameter(torch.zeros(1 if write_mix == "layer" else hk + hv))
        if variant in READ_MIX_VARIANTS:  # the second mix (module docstring; qmix: the q rows', kqmix: the q and k rows'): registered after write_lambda and write_query
            self.qmix_lambda = nn.Parameter(torch.zeros(1))
            self.qmix_query = DepthQuery(d, eps) if write_query else None
        self.ccr_read, self.ccr_write, self.spacing = ccr_read, ccr_write, spacing
        self.layer, self.ccr_source, self.ccr_target = layer, ccr_source, ccr_target
        self.n_sources = 0  # set by build under the option: the sources this layer's query mix reads (0: not under the option)
        self.translator = translator if ccr_read else "none"
        if ccr_read:
            # gate_init (EncoderConfig.gdn_ccr_gate_init): 0.0, v1's zero start, or a small nonzero start; torch.full
            # draws no random number either
            self.ccr_gate = nn.Parameter(torch.zeros(hv) if gate_init == 0 else torch.full((hv,), float(gate_init)))
            if self.translator == "key_head":  # registered after the gate: the state dict's order (parameter_names)
                assert dk is not None, "a translator is d_k x d_k"
                self.ccr_translator_delta = nn.Parameter(torch.zeros(hk, dk, dk))

    def register_qmix_keep(self, n_sources: int) -> None:
        """Under EncoderConfig.gdn_qmix_select_step, in a layer whose query mix reads more than QMIX_KEEP sources: the buffer qmix_keep
        ((2,) int64, -1 = not chosen; persistent: the state dict keeps the choice) and the post-load hook that sets qmix_static from
        it, so that every loader of a state dict (the trainer's resume, the scorers, the suite) reads the form the weights were
        trained in. Called by build; a layer with QMIX_KEEP or fewer sources gets nothing (all its sources stay)."""
        self.n_sources = n_sources
        if n_sources <= QMIX_KEEP or self.qmix_query is None:
            return
        self.register_buffer("qmix_keep", torch.full((QMIX_KEEP,), -1, dtype=torch.long))
        self.register_load_state_dict_post_hook(_sync_qmix_static)

    def chosen_sources(self) -> list | None:
        """The 2 chosen sources (ascending) as a list, or None before the choice (one host read of the buffer; for logs and tests)."""
        keep = getattr(self, "qmix_keep", None)
        if keep is None or int(keep[0]) < 0:
            return None
        return [int(i) for i in keep.detach().cpu().tolist()]


def _sync_qmix_static(module: "GDNArms", incompatible_keys) -> None:
    """GDNArms.register_qmix_keep's post-load hook: qmix_static is True exactly when the loaded qmix_keep holds a choice."""
    chosen = module.chosen_sources()
    if chosen is not None and not 0 <= chosen[0] < chosen[1] < module.n_sources:
        raise ValueError(f"layer {module.layer}: qmix_keep {chosen} is not 2 ascending sources of its {module.n_sources}")
    module.qmix_static = chosen is not None


def build(cfg: ModelConfig, layers) -> bool:
    """Encoder.__init__'s call, after attn_res.build. Off: nothing is touched, returns False. On: every GDN layer that
    takes part gets its GDNArms as its last submodule (zeros only), and its mixer the plain attribute write_mix (so that
    GatedDeltaNet.gates refuses the write input it cannot see). Returns True."""
    check_gdn_arms(cfg)
    if not any_on(cfg):
        return False
    e = cfg.encoder
    S = e.full_attn_interval
    gdn = gdn_layers(cfg)
    g = e.gdn
    sources, writers = ccr_sources(cfg), set(ccr_writers(cfg))
    for i in gdn:
        write = e.gdn_write_mix if (write_on(cfg) and i > 0) else "none"
        ccr_read = i in sources
        ccr_write = i in writers
        if write == "none" and not ccr_read and not ccr_write:
            continue
        layers[i].gdn_arms = GDNArms(cfg.d_model, cfg.norm_eps, g.n_k_heads, g.n_v_heads, write,
                                     write != "none" and i // S >= 1, ccr_read, ccr_write, e.gdn_ccr_spacing,
                                     layer=i, ccr_source=sources.get(i), ccr_target=e.gdn_ccr_target,
                                     translator=e.gdn_ccr_translator, dk=g.k_head_dim, gate_init=e.gdn_ccr_gate_init,
                                     gate_floor=e.gdn_ccr_gate_floor,
                                     variant=e.gdn_write_variant if write != "none" else "none")
        layers[i].mixer.write_mix = write
        layers[i].mixer.arms_attached = True
        # the fused write mix (attn_res.check_fused accepts attn_res_fused with gdn_write_mix "layer" only)
        layers[i].gdn_arms.fused_write = bool(e.attn_res_fused and write == "layer" and layers[i].gdn_arms.write_query is not None)
        layers[i].gdn_arms.write_mix_storage = e.write_mix_storage  # (attn_res.check_fused has validated it)
        if layers[i].gdn_arms.fused_write:  # a process that still sets CED_WRITE_MIX_STORAGE is refused here, before the first step
            from ced.kernels.write_mix import refuse_legacy_env  # noqa: PLC0415

            refuse_legacy_env()
        # (check_mix_fold: the variants the fold forms; only the lerp form is the GEMM's: the producer form is the norm pass's)
        layers[i].gdn_arms.mix_fold = bool(mix_fold_form(e) == "lerp" and write == "layer")
        if e.gdn_qmix_select_step:  # (check_qmix_select: the variant qmix): the buffer of the choice where the layer has more than 2 sources
            layers[i].gdn_arms.register_qmix_keep(i // S + 1)
        # 'producer-rc' (and p2, p3): the block's group read recomputes the extra mixes in the backward; mix_p: 'producer-p2' / 'producer-p3', the layer-side level (plain attributes, never in the state dict)
        layers[i].gdn_arms.mix_rc, layers[i].gdn_arms.mix_p = bool(mix_rc(e) and layers[i].gdn_arms.fused_write), (mix_p(e) if write == "layer" else 0)
        # write_mix_row_term "pdp": the group read stores the extra queries' mixes in the sources' dtype (a plain attribute, read by attn_res.EntryWalk._group_read)
        layers[i].gdn_arms.hx_source = bool(e.write_mix_row_term == "pdp" and layers[i].gdn_arms.fused_write)
    if mix_fold_form(e) == "lerp":  # the fold's pins read here, eagerly: read first inside a compiled layer's trace, the file's stat breaks the graph
        from ced.kernels import fold_mix

        fold_mix.pins()
    return True


@torch.no_grad()
def floor_gates_(encoder) -> None:
    """After an optimizer step (the trainer calls it when gate_floor_on): every CCR reader's gate clamped in place to at
    least its gate_floor (EncoderConfig.gdn_ccr_gate_floor), one clamp_ a reader on a (H_v,) tensor: fixed shapes, no
    host read (no device sync), and the tensor keeps its storage, so a captured graph or a compiled layer reads the new
    values. A reader without a floor (gate_floor 0.0) is not touched."""
    for layer in encoder.layers:
        arms = getattr(layer, "gdn_arms", None)
        if arms is not None and arms.ccr_read and arms.gate_floor:
            arms.ccr_gate.clamp_(min=arms.gate_floor)


def fused_write_here(arms: GDNArms) -> bool:
    """Whether this layer's m' comes from its block's group read (ced/kernels/write_mix.py): the fused write mix on and a
    forward with gradients on (training, and activation checkpointing's recomputation); evaluation, scoring, prefill /
    extend and the diagnostics (no_grad) keep today's path, as E's reads (attn_res.fused_here)."""
    return bool(getattr(arms, "fused_write", False)) and torch.is_grad_enabled()


def parameter_names(cfg: ModelConfig) -> list[str]:
    """The names, in state-dict order, of the tensors the two keys add. [] when both are off."""
    if not any_on(cfg):
        return []
    S, names, gdn = cfg.encoder.full_attn_interval, [], gdn_layers(cfg)
    sources = ccr_sources(cfg)
    qmix = write_on(cfg) and write_variant(cfg) in READ_MIX_VARIANTS
    for i in gdn:
        p = f"encoder.layers.{i}.gdn_arms"
        if write_on(cfg) and i > 0:
            names.append(f"{p}.write_lambda")
            if qmix:
                names.append(f"{p}.qmix_lambda")
        if i in sources:
            names.append(f"{p}.ccr_gate")
            if translator_on(cfg):
                names.append(f"{p}.{TRANSLATOR}")
        if write_on(cfg) and i > 0 and i // S >= 1:
            names += [f"{p}.write_query.query", f"{p}.write_query.key_norm.weight"]
            if qmix:
                names += [f"{p}.qmix_query.query", f"{p}.qmix_query.key_norm.weight"]
    return names


# ------------------------------------------------------------------------------------------------
# A: the write inputs


def write_source(arms: GDNArms, sources: list) -> torch.Tensor:
    """m: the layer's softmax depth mix over its sources (b_0 alone in block 1, weight 1)."""
    from ced.model.attn_res import depth_mix

    if arms.write_query is None:
        assert len(sources) == 1, "a block-1 layer's write mix reads the embedding output alone"
        return sources[0]
    assert len(sources) >= 2, "a layer of block n >= 2 mixes the embedding output and the finished blocks"
    return depth_mix(sources, arms.write_query)


def read_source(arms: GDNArms, sources: list) -> torch.Tensor:
    """m_q (the wsl variant "qmix"; module docstring): the layer's SECOND softmax depth mix over the same sources, with its
    own DepthQuery (qmix_query); b_0 alone in block 1, weight 1, as write_source."""
    from ced.model.attn_res import depth_mix

    assert arms.variant in READ_MIX_VARIANTS, "the read mix is the qmix and kqmix variants'"
    if arms.qmix_query is None:
        assert len(sources) == 1, "a block-1 layer's read mix reads the embedding output alone"
        return sources[0]
    assert len(sources) >= 2, "a layer of block n >= 2 mixes the embedding output and the finished blocks"
    if arms.qmix_static:  # EncoderConfig.gdn_qmix_select_step: after the choice, the per-token softmax over the layer's 2 chosen sources
        from ced.model.attn_res import depth_mix_pair

        return depth_mix_pair(sources, arms.qmix_query, arms.qmix_keep)
    return depth_mix(sources, arms.qmix_query)


def write_mixes(norm, arms: GDNArms, sources: list) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(m', m'_q): the write mix's normed write mix over `sources` through the layer's input norm `norm`
    (EncoderLayer.mixer_in_norm), and under the variant "qmix" its normed read mix (in block 1 the same tensor as m': both
    are norm(b_0)); None otherwise. The one place the forward (EncoderLayer.gdn_arms_mixer) and the suite's GDN walk
    (ced/eval/gdn_diag.py) form them."""
    x_write = norm(write_source(arms, sources))
    if arms.variant not in READ_MIX_VARIANTS:
        return x_write, None
    return x_write, (x_write if arms.qmix_query is None else norm(read_source(arms, sources)))


def _head_lambdas(mixer, arms: GDNArms) -> tuple[torch.Tensor, torch.Tensor]:
    """(lambda of every key head (H_k,), lambda of every value head (H_v,)); "layer": the one value for all."""
    hk, hv = mixer.cfg.n_k_heads, mixer.cfg.n_v_heads
    lam = arms.write_lambda
    if arms.write_mix == "layer":
        return lam.expand(hk), lam.expand(hv)
    return lam[:hk], lam[hk:]


def _compute_dtype(t: torch.Tensor) -> torch.Tensor:
    """t in the dtype the projections compute in: autocast's, cast ONCE here and shared by the projections that read it
    (autocast would cast it again for every one and keep each copy for the backward); t itself without autocast. The
    same rounding as autocast's own cast, so the projections' outputs are the same."""
    dev = t.device.type
    if torch.is_autocast_enabled(dev):
        return t.to(torch.get_autocast_dtype(dev))
    return t


def mx_rows(mixer):
    """The mixer's in_proj_qkv as the quantised linear (ced/model/fp8.py FP8Linear, training mode, a quantised layer) when this call
    runs on MXFP8, else None: a model without model.fp8 (a plain nn.Linear), an evaluation or generation forward (the BF16 model bit
    for bit) and a layer apply_fp8 kept in high precision all take the bf16 projections. The arms' projections are row ranges of
    its weight (write_inputs), so the layer must hold no bias and its row ranges whole MX blocks (check_fp8_forms)."""
    m = mixer.in_proj_qkv
    now = getattr(m, "quantised_now", None)
    if now is None or not now():
        return None
    assert m.bias is None, "the arms' projections are row ranges of in_proj_qkv's weight: no bias"
    return m


def write_inputs(mixer, arms: GDNArms, x: torch.Tensor, x_write: torch.Tensor, x_read: torch.Tensor | None = None,
                 lerped: bool = False):
    """The fast path: (u_pre (B, L, 2 H_k d_k + H_v d_v), a_lin (B, L, H_v), b_lin (B, L, H_v)), the in-projection's pre-conv output and the
    decay / beta pre-activations, with the q rows reading x and the k, v, a, b rows reading x_w. "layer": x_w = x + lambda (x_write - x)
    (one addcmul, which keeps only the difference for its backward), and the q and k/v rows of in_proj_qkv as two matmuls (the same
    multiply-adds as one). The wsl variants (arms.variant; module docstring), "layer" only: "korig", the q and k rows read x (one matmul)
    and the v rows x_w (another); "vonly", the same two matmuls as korig's and beta and the decay read x (x cast once, as "qshare_kv"'s),
    so only the v rows read x_w; "qmix", the q rows read x_q = x + lambda_q (x_read - x) (x_read: m'_q, the normed read mix; given under
    "qmix" only), formed as x_w is; "qshare", every row of in_proj_qkv reads x_w: ONE matmul, R1-entry's projection with x_w in place of
    x (the same multiply-adds; one GEMM where wsl has two, and x is not read by it); "qshare_kv", the same ONE matmul on x_w, and beta
    and the decay read x (x cast once, as the q rows' x under "none"); "qshare_bd", the q rows read x_w and the k and v rows x (two
    matmuls, wsl's split with the operands swapped), beta and the decay x_w. "head": by linearity, P(x) + lambda_head (*) P(x_write - x)
    for the k, v, a, b rows (one more projection of the difference). The inputs of the projections are cast once (_compute_dtype). The
    outputs are in the projections' own dtype (bf16 under autocast, as without the arm: the fused kernel's input dtype follows it). The
    fused write mix (arms.fused_write; ced/kernels/write_mix.py): m' arrives in the GEMMs' dtype (bf16 under autocast), and so does a
    block's opening layer's x (E's normed output); x_w is still formed in at least float32 (x and m' widened first, lambda never rounded
    to bf16) and cast once, as without it. A no-op on float32 / float64 inputs (evaluation's unfused path, the float64 tests). Under
    "qshare" the q rows read that x_w with the k and v rows; under "qshare_kv" and "qshare_bd" the rows that read x read it as it came
    (as the q rows under "none"). Under "qmix" m'_q arrives from the same group read in the GEMMs' dtype and is widened as m' is: x_q = x
    + lambda_q (m'_q - x) in at least float32, cast once. lerped (qmix's block-opening layer on the fused write mix's second round;
    ced/kernels/write_mix.py group_read): x_write and x_read are x_w and x_q themselves, formed by the group read from the norms' float32
    outputs and rounded once (the kernel takes lambda and lambda_q and returns their gradients); the projections read them as they came,
    and no addcmul is formed here. On MXFP8 (module docstring, "On MXFP8"; training mode with in_proj_qkv a quantised FP8Linear,
    `mx_rows`): the two projections of in_proj_qkv are FP8Linear.quantised on the rows' ranges W[:nq] and W[nq:], each on the tensor as
    it was formed (x_w and x_q in float32, quantised once each by the linear's own quantiser; x as it came), and in_proj_a / in_proj_b
    read x_w's one cast to the GEMMs' dtype. Every other call is the path above, statement for statement."""
    hk, dk, dv = mixer.cfg.n_k_heads, mixer.cfg.k_head_dim, mixer.cfg.v_head_dim
    nq = hk * dk
    W = mixer.in_proj_qkv.weight
    variant = arms.variant
    assert (x_read is not None) == (variant in READ_MIX_VARIANTS), "the qmix and kqmix variants' layers take their read mix, and only they"
    mx = mx_rows(mixer)  # in_proj_qkv as the quantised linear in this call, else None (every call off MXFP8)
    hold = getattr(mixer.in_proj_qkv, "mx_hold", None)
    if mx is not None and hold is not None and hold.opw is not None:
        # the producer form (EncoderConfig.mix_fold 'producer'; ced/kernels/mx_mix_chain.py): the layer's input-norm pass made the
        # operands of x, x_w and (qmix) x_q (mx_wire.mix_mixer sets the layer's hold). x, x_write and x_read are that pass's autograd
        # edges, whose values nothing reads: the q rows take the operand of x (wsl) or x_q (qmix), the k, v rows x_w's, and
        # in_proj_a / in_proj_b the copy of x_w the pass wrote (FP8Linear.forward: hold.low)
        assert not lerped and arms.write_mix == "layer" and variant in MX_VARIANTS, (arms.write_mix, variant, lerped)
        from ced.model import mx_p2, mx_wire  # noqa: PLC0415

        qmix = variant == "qmix"
        u = torch.cat([mx.quantised(mx_p2.q_edge(hold, x_read if qmix else x), W[:nq], own=False, op=hold.opq if qmix else hold.op),
                       mx.quantised(x_write, W[nq:], own=False, op=hold.opw)], dim=-1)
        if hold.ab is not None or hold.ea is not None:  # 'producer-p2' / 'producer-p3' (ced/model/mx_p2.py): an edge each for a and b, or a and b out of the pass
            return u, *mx_p2.a_b(mixer, hold)
        xw = mx_wire.LowRead.apply(x_write, hold.low)  # in_proj_a / in_proj_b read x_w's ONE copy in the GEMMs' dtype, shared as the unwired path's one cast is
        hold.low = None  # (the linears below take the copy as given; mx_wire.mix_mixer clears the rest of the hold)
        return u, mixer.in_proj_a(xw), mixer.in_proj_b(xw)
    if getattr(arms, "mix_fold", False) and not lerped:  # model.encoder.mix_fold: the lerps in the GEMM's load (None: today's path)
        from ced.kernels import fold_mix

        assert mx is None, "model.encoder.mix_fold is refused with model.fp8 (check_fp8_forms)"
        folded = fold_mix.lerp_inputs(mixer, arms, x, x_write, x_read)
        if folded is not None:
            return folded
    if lerped:
        assert variant == "qmix" and arms.write_mix == "layer" and getattr(arms, "fused_write", False), \
            "x_w and x_q from the group read are qmix's on the fused write mix"
        if mx is not None:  # MXFP8: the group read's x_w and x_q (rounded once by the kernel), quantised as they came
            u = torch.cat([mx.quantised(x_read, W[:nq], own=False), mx.quantised(x_write, W[nq:], own=False)], dim=-1)
            xw = _compute_dtype(x_write)
            return u, mixer.in_proj_a(xw), mixer.in_proj_b(xw)
        xw, xr = _compute_dtype(x_write), _compute_dtype(x_read)
        u = torch.cat([F.linear(xr, W[:nq]), F.linear(xw, W[nq:])], dim=-1)
        return u, mixer.in_proj_a(xw), mixer.in_proj_b(xw)
    x_q = x  # what the q rows read (the opening layer's bf16 normed input under the fused write mix, as it came)
    if getattr(arms, "fused_write", False):
        assert variant in FUSED_VARIANTS, f"the fused write mix is built with the variants {FUSED_VARIANTS} only (check_write_variant)"
        dt = compute_dtype(x)
        x, x_write = x.to(dt), x_write.to(dt)
        if x_read is not None:  # qmix: m'_q from the same group read, widened as m' is (x_q formed in float32, cast once)
            x_read = x_read.to(dt)
    d = x_write - x
    if mx is not None:  # MXFP8: the write mix's "layer" form with the variants MX_VARIANTS (check_fp8_forms refuses every other)
        assert arms.write_mix == "layer" and variant in MX_VARIANTS, (arms.write_mix, variant)
        xw = torch.addcmul(x, d, arms.write_lambda.to(x.dtype))  # x_w as formed (>= float32): the quantiser reads it, not a bf16 copy
        if variant == "none":  # the q rows read x as it came (the opening layer's bf16 normed input under the fused write mix)
            xr = x_q
        else:  # x_q = x + lambda_q (m'_q - x), formed as x_w is
            xr = torch.addcmul(x, x_read - x, arms.qmix_lambda.to(x.dtype))
        u = torch.cat([mx.quantised(xr, W[:nq], own=False), mx.quantised(xw, W[nq:], own=False)], dim=-1)
        xw = _compute_dtype(xw)  # in_proj_a / in_proj_b are high precision: one cast of x_w to the GEMMs' dtype, as the bf16 path's
        return u, mixer.in_proj_a(xw), mixer.in_proj_b(xw)
    if arms.write_mix == "layer":
        xw = _compute_dtype(torch.addcmul(x, d, arms.write_lambda.to(x.dtype)))  # x + lambda d: x itself at lambda = 0
        if variant == "none":
            u = torch.cat([F.linear(_compute_dtype(x_q), W[:nq]), F.linear(xw, W[nq:])], dim=-1)
        elif variant == "qshare":  # every row of in_proj_qkv reads x_w (one matmul): q shares the write side's input
            u = F.linear(xw, W)
        elif variant == "korig":  # the q and k rows read x (one matmul), the v rows x_w
            u = torch.cat([F.linear(_compute_dtype(x_q), W[: 2 * nq]), F.linear(xw, W[2 * nq :])], dim=-1)
        elif variant == "qshare_kv":  # q, k and v read x_w (one matmul, qshare's); beta and the decay read x (R1-entry's)
            xc = _compute_dtype(x_q)
            return F.linear(xw, W), mixer.in_proj_a(xc), mixer.in_proj_b(xc)
        elif variant == "vonly":  # the V-only variant: q and k read x (one matmul, korig's), v reads x_w (another); beta and the decay read x (qshare_kv's)
            xc = _compute_dtype(x_q)
            u = torch.cat([F.linear(xc, W[: 2 * nq]), F.linear(xw, W[2 * nq :])], dim=-1)
            return u, mixer.in_proj_a(xc), mixer.in_proj_b(xc)
        elif variant == "qshare_bd":  # q reads x_w, k and v read x (two matmuls); beta and the decay read x_w (below)
            u = torch.cat([F.linear(xw, W[:nq]), F.linear(_compute_dtype(x_q), W[nq:])], dim=-1)
        elif variant == "kqmix":  # arm KQ: the q and k rows read x_kq = x + lambda_q (m'_q - x) (one matmul), the v rows x_w; beta and the decay x_w (below)
            xr = _compute_dtype(torch.addcmul(x, x_read - x, arms.qmix_lambda.to(x.dtype)))
            u = torch.cat([F.linear(xr, W[: 2 * nq]), F.linear(xw, W[2 * nq :])], dim=-1)
        else:
            assert variant == "qmix", variant
            # x_q = x + lambda_q (m'_q - x): x itself at lambda_q = 0, as x_w is (the same addcmul, its own lambda)
            xr = _compute_dtype(torch.addcmul(x, x_read - x, arms.qmix_lambda.to(x.dtype)))
            u = torch.cat([F.linear(xr, W[:nq]), F.linear(xw, W[nq:])], dim=-1)
        return u, mixer.in_proj_a(xw), mixer.in_proj_b(xw)
    assert arms.write_mix == "head" and variant == "none", (arms.write_mix, variant)
    lam_k, lam_v = _head_lambdas(mixer, arms)
    dl = _compute_dtype(d)
    u = mixer.in_proj_qkv(x)
    lam_kv = torch.cat([lam_k.repeat_interleave(dk), lam_v.repeat_interleave(dv)])
    kv = u[..., nq:] + F.linear(dl, W[nq:]) * lam_kv
    a0, b0 = mixer.in_proj_a(x), mixer.in_proj_b(x)
    a = a0 + mixer.in_proj_a(dl) * lam_v
    b = b0 + mixer.in_proj_b(dl) * lam_v
    return torch.cat([u[..., :nq], kv.to(u.dtype)], dim=-1), a.to(a0.dtype), b.to(b0.dtype)


def write_inputs_reference(mixer, arms: GDNArms, x: torch.Tensor, x_write: torch.Tensor, x_read: torch.Tensor | None = None):
    """The ground truth of write_inputs, head by head: x_w^h = x + lambda_h (x_write - x) written out for every key head and every value head, and
    each head's rows of the projections applied to its own x_w^h. Same results. The wsl variants written out the same way: under "qmix" each key
    head's q rows read x_q = x + lambda_q (x_read - x); under "korig" each key head's k rows read x itself; under "vonly" each key head's q and k
    rows and each value head's beta and decay rows read x itself and only each value head's v rows read x_w^j; under "qshare" each key head's q
    rows read x_w^h = x + lambda_h (x_write - x), its k rows' input; under "qshare_kv" as "qshare", and each value head's beta and decay rows read
    x itself; under "qshare_bd" each key head's q rows read x_w^h, each key head's k rows and each value head's v rows read x itself, beta and the
    decay x_w^j. On MXFP8 (mx_rows; write_inputs' docstring) each head's rows of in_proj_qkv go through FP8Linear.quantised on that head's range of
    the weight, on the head's own x_w^h (the heads' widths whole MX blocks, so each head's blocks are the fast path's): the reference of the
    quantised path, in the model's dtype (float64 in the tests); beta and the decay stay plain F.linear."""
    hk, hv, dk, dv = mixer.cfg.n_k_heads, mixer.cfg.n_v_heads, mixer.cfg.k_head_dim, mixer.cfg.v_head_dim
    nq = hk * dk
    W = mixer.in_proj_qkv.weight
    variant = getattr(arms, "variant", "none")
    assert (x_read is not None) == (variant in READ_MIX_VARIANTS)
    mx = mx_rows(mixer)
    assert mx is None or (dk % 32 == 0 and dv % 32 == 0), "a head's rows are whole MX blocks (check_fp8_forms)"

    def lin(xin: torch.Tensor, lo: int, hi: int) -> torch.Tensor:  # rows lo:hi of in_proj_qkv on xin
        return F.linear(xin, W[lo:hi]) if mx is None else mx.quantised(xin, W[lo:hi], own=False)

    lam_k, lam_v = _head_lambdas(mixer, arms)
    if variant in READ_MIX_VARIANTS:  # qmix and kqmix: each key head's q rows read x_q = x + lambda_q (x_read - x)
        q = torch.cat([lin(x + arms.qmix_lambda[0] * (x_read - x), i * dk, (i + 1) * dk) for i in range(hk)], dim=-1)
    elif variant in ("qshare", *SPLIT_VARIANTS):
        q = torch.cat([lin(x + lam_k[i] * (x_write - x), i * dk, (i + 1) * dk) for i in range(hk)], dim=-1)
    else:
        q = lin(x, 0, nq)
    ks, vs, a, b = [], [], [], []
    for i in range(hk):
        if variant == "kqmix":  # arm KQ: each key head's k rows read the q rows' input x_kq
            xw = x + arms.qmix_lambda[0] * (x_read - x)
        else:
            xw = x if variant in ("korig", "qshare_bd", "vonly") else x + lam_k[i] * (x_write - x)
        ks.append(lin(xw, nq + i * dk, nq + (i + 1) * dk))
    for j in range(hv):
        xw = x + lam_v[j] * (x_write - x)
        xv = x if variant == "qshare_bd" else xw  # the v rows' input
        xg = x if variant in ("qshare_kv", "vonly") else xw  # beta's and the decay's input
        vs.append(lin(xv, 2 * nq + j * dv, 2 * nq + (j + 1) * dv))
        a.append(F.linear(xg, mixer.in_proj_a.weight[j : j + 1]))
        b.append(F.linear(xg, mixer.in_proj_b.weight[j : j + 1]))
    return torch.cat([q, *ks, *vs], dim=-1), torch.cat(a, dim=-1), torch.cat(b, dim=-1)


# ------------------------------------------------------------------------------------------------
# CCR: the records a writer hands its reader, and the reads


@dataclass
class CCRChunks:
    """Fused path: h (1, NT, H_v, d_k, d_v), the writer's state at the start of every chunk of the kernel's layout
    (ced_delta_rule_states); spacing C (the kernel's chunk); source, the writer's layer index (its readers check it)."""

    h: torch.Tensor
    spacing: int
    source: int | None = None


@dataclass
class CCRBoundary:
    """Non-fused paths: states (B, L, H_v, d_k, d_v), the writer's checkpoint of every position (boundary_states);
    source, the writer's layer index."""

    states: torch.Tensor
    source: int | None = None


def boundary_states(states: torch.Tensor, pos: torch.Tensor, ckpt0: torch.Tensor | None, C: int):
    """Each position's checkpoint, from the states after every position of a call (fixed shapes: gathers and wheres).

    states (B, L, H, d_k, d_v): the state after every position of the call; pos (B, L): positions in the document;
    ckpt0 (B, H, d_k, d_v) or None: the checkpoint carried by the cache (the state after the first C floor(n / C)
    tokens of the document the call continues, n its tokens before the call; None: no cache, zeros).
    Returns (per-position checkpoints (B, L, H, d_k, d_v), the checkpoint the next token after the call reads).

    For position t (index in the call) with r = pos_t mod C, its checkpoint is the state after index j = t - r - 1 if
    j >= 0; the cache's if j < 0 (then the document began before the call and that is its checkpoint); zero if
    pos_t - r = 0 (the document's first C tokens)."""
    B, L = pos.shape
    tail = states.shape[2:]
    expand = (lambda m: m.view(B, L, *([1] * len(tail))))  # noqa: E731
    r = pos % C
    j = torch.arange(L, device=pos.device)[None, :] - r - 1
    gathered = torch.gather(states, 1, j.clamp(min=0).view(B, L, *([1] * len(tail))).expand(B, L, *tail))
    zero = torch.zeros((), dtype=states.dtype, device=states.device)
    before = zero if ckpt0 is None else ckpt0[:, None].to(states.dtype)
    out = torch.where(expand(j >= 0), gathered, before)
    out = torch.where(expand(pos - r > 0), out, zero)
    # the next token: position pos_last + 1, index L
    p1 = pos[:, -1] + 1
    r1 = p1 % C
    j1 = L - r1 - 1
    last = torch.gather(states, 1, j1.clamp(min=0).view(B, 1, *([1] * len(tail))).expand(B, 1, *tail))[:, 0]
    one = lambda m: m.view(B, *([1] * len(tail)))  # noqa: E731
    nxt = torch.where(one(j1 >= 0), last, zero if ckpt0 is None else ckpt0.to(states.dtype))
    nxt = torch.where(one(p1 - r1 > 0), nxt, zero)
    return out, nxt


def boundary_states_reference(states: torch.Tensor, pos: torch.Tensor, ckpt0: torch.Tensor | None, C: int):
    """boundary_states written as a loop over rows and positions (the ground truth; slow)."""
    B, L = pos.shape
    zero = states.new_zeros(states.shape[2:])
    out = []
    for b in range(B):
        row = []
        for t in range(L):
            p = int(pos[b, t])
            c = C * (p // C)  # the checkpoint: after the document's first c tokens
            if c == 0:
                row.append(zero)
            elif t - (p - c) - 1 >= 0:  # the document's token c - 1 is in this call, at index t - (p - c) - 1
                row.append(states[b, t - (p - c) - 1])
            else:
                row.append(ckpt0[b].to(states.dtype))
        out.append(torch.stack(row))
    p1 = [int(pos[b, -1]) + 1 for b in range(B)]
    nxt = []
    for b in range(B):
        c = C * (p1[b] // C)
        idx = L - (p1[b] - c) - 1
        nxt.append(zero if c == 0 else states[b, idx] if idx >= 0 else ckpt0[b].to(states.dtype))
    return torch.stack(out), torch.stack(nxt)


def kernel_chunk() -> int:
    """The chunk of our kernel's chunk states (ced/kernels/gdn.py CHUNK, 64): the fused path's checkpoint spacing."""
    from ced.kernels import gdn as kernels

    return kernels.CHUNK


def take_record(cache):
    """(the CCR record a GDN layer's call left in its new cache entry, the entry without it): the walks hand the record
    to the next GDN layer and keep the entry (the cache: S, conv history and the checkpoint) without it."""
    return cache.ccr, dataclasses.replace(cache, ccr=None)


def record_in(arms: GDNArms, rec):
    """What a walk hands a GDN layer with the arms as its `ccr` argument: the walk's record if the layer reads one, else
    None (v1: only the first GDN layer reads none, and the walk's record is None there anyway; v2: a writer of
    "group_first" after the first group reads none while the walk holds the group before's record)."""
    return rec if arms.ccr_read else None


def hand_record(arms: GDNArms, rec, cache):
    """After a GDN layer's call: (the record the walk holds for the next GDN layer, the cache entry without a record).
    A writer's own record replaces the walk's; a layer that writes none leaves it (so the lowest's record reaches every
    reader under "lowest", the group's first's every reader of the group under "group_first"). Under "previous" the
    layer's own (None in the last GDN layer) always replaces it: v1's chain, as it was."""
    own, cache = take_record(cache)
    if arms.ccr_write or arms.ccr_target == "previous":
        return own, cache
    assert own is None, "a GDN layer that no reader reads made a CCR record"
    return rec, cache


LAYOUT_KEY = "gdn_ccr_layout"


def chunk_layout(cu_seqlens_cpu: torch.Tensor, T: int, C: int, device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Host side, from the CPU boundaries (no device sync): the kernel's chunk layout (ced/kernels/gdn.py chunk_index:
    segments in turn, chunks of C within each segment, h[c] the state at chunk c's start). Returns (pad (NT C,) long:
    the flattened position in each chunk slot, T for a slot past its segment's end; slot (T,) long: each position's
    slot c C + offset; NT). Both moved to `device` in one copy."""
    cu = cu_seqlens_cpu.long().cpu()
    lens = cu[1:] - cu[:-1]
    counts = (lens + C - 1) // C
    first = torch.cat([counts.new_zeros(1), counts.cumsum(0)])  # each segment's first global chunk
    nt = int(first[-1])
    seg = torch.repeat_interleave(torch.arange(len(lens)), lens)  # (T,)
    off = torch.arange(T) - cu[:-1][seg]
    slot = (first[:-1][seg] + off // C) * C + off % C
    pad = torch.full((nt * C,), T, dtype=torch.long)
    pad[slot] = torch.arange(T)
    host = torch.cat([pad, slot])
    dev = host.to(device, non_blocking=True)
    return dev[: nt * C], dev[nt * C :], nt


@torch.compiler.disable  # host-side layout from the varlen boundaries and two inverse gathers: an opaque op, as the kernel
def fused_read(q: torch.Tensor, h: torch.Tensor, ctx, C: int) -> torch.Tensor:
    """The fused path's read: for every position t, h[chunk(t)]^T q_t, one batched matmul per chunk and head.

    q (B, L, H_k, d_k): the reader's query (in the kernel's dtype); h (1, NT, H_v, d_k, d_v): the writer's chunk-start
    states; ctx: the forward's SeqContext (cu_seqlens_cpu, its memo for the layout, shared by every reader of a forward).
    Returns (B, L, H_v, d_v) in h's dtype. Gradients: dq through the inverse gathers, dh = q_chunk^T do_chunk."""
    from ced.model.capacity import gather_rows

    B, L, hk, dk = q.shape
    T = B * L
    nt, hv, dv = h.shape[1], h.shape[2], h.shape[4]
    key = (LAYOUT_KEY, C, T)
    if key not in ctx.memo:
        cpu = ctx.cu_seqlens_cpu if ctx.cu_seqlens_cpu is not None else ctx.cu_seqlens.cpu()
        ctx.memo[key] = chunk_layout(cpu, T, C, q.device)
    pad, slot, n = ctx.memo[key]
    assert n == nt, f"the writer's chunk states hold {nt} chunks; the layout of these boundaries at C = {C} has {n}"
    qf = q.reshape(1, T, hk, dk).to(h.dtype)
    qp = gather_rows(qf, pad[None], slot[None])  # (1, NT C, H_k, d_k); every position read by exactly one slot
    qp = qp.view(nt, C, hk, dk).permute(0, 2, 1, 3).contiguous()  # (NT, H_k, C, d_k): one copy, kept once below
    hg = h[0].view(nt, hk, hv // hk, dk, dv)
    # one batched matmul per value head of a key head's group (G = H_v / H_k: 2 in every shape we run): a broadcast
    # matmul over the group would materialise and keep G copies of the laid-out q for the backward; these keep one
    out = torch.stack([torch.matmul(qp, hg[:, :, j]) for j in range(hv // hk)], dim=2)  # (NT, H_k, G, C, d_v)
    out = out.permute(0, 3, 1, 2, 4).reshape(1, nt * C, hv, dv)
    return gather_rows(out, slot[None], pad[None]).view(B, L, hv, dv)


def translator_matrix(arms: GDNArms) -> torch.Tensor:
    """M = I + D (H_k, d_k, d_k) of a reader with a translator, float64 and detached: what the readings print and apply."""
    D = getattr(arms, TRANSLATOR).detach().double()
    return D + torch.eye(D.shape[-1], dtype=D.dtype, device=D.device)


def translate_query(arms: GDNArms, q: torch.Tensor) -> torch.Tensor:
    """q_x = M_h q_t for every position and key head, formed as q + q D_h^T (D = M - I, the stored tensor); q itself when
    the reader has no translator. q (..., H_k, d_k), the reader's normalised, scaled query; returns q's shape and dtype.
    At D = 0 the product is exactly zero, so q_x is q bit for bit."""
    if arms.translator == "none":
        return q
    D = getattr(arms, TRANSLATOR)
    return q + torch.einsum("...hk,hjk->...hj", q, D.to(q.dtype)).to(q.dtype)


def translate_query_reference(arms: GDNArms, q: torch.Tensor) -> torch.Tensor:
    """translate_query written out head by head with M = I + D: (M_h q)_j = sum_k M_h[j, k] q_k (the ground truth)."""
    if arms.translator == "none":
        return q
    M = getattr(arms, TRANSLATOR).to(q.dtype) + torch.eye(q.shape[-1], dtype=q.dtype, device=q.device)
    return torch.stack([q[..., h, :] @ M[h].T for h in range(q.shape[-2])], dim=-2)


def ccr_cross(mixer, arms: GDNArms, rec, q: torch.Tensor, ctx) -> torch.Tensor:
    """The reader's cross term before its gate: (B, L, H_v, d_v), ckpt(t)^T q_x,t of its source's record, with
    q_x = M q (translate_query; q itself without a translator). q (B, L, H_k, d_k) in key heads."""
    hv, hk = mixer.cfg.n_v_heads, mixer.cfg.n_k_heads
    assert rec.source is None or rec.source == arms.ccr_source, (
        f"layer {arms.layer} reads layer {arms.ccr_source}'s checkpoints (gdn_ccr_target {arms.ccr_target!r}); the walk "
        f"handed it layer {rec.source}'s")
    q = translate_query(arms, q)
    if isinstance(rec, CCRChunks):
        assert rec.spacing == arms.spacing
        return fused_read(q, rec.h, ctx, arms.spacing)
    assert isinstance(rec, CCRBoundary), f"a CCR reader needs the previous GDN layer's record, not {type(rec).__name__}"
    qr = q.repeat_interleave(hv // hk, dim=2).to(rec.states.dtype)
    # contiguous, as the fused path's read returns it (einsum may return permuted strides); the paths without the fused
    # kernel only (CPU, cached chunks)
    return torch.einsum("blhkv,blhk->blhv", rec.states, qr).contiguous()


# ------------------------------------------------------------------------------------------------
# EncoderConfig.gdn_qmix_select_step: the weights' record, the choice, the log (module docstring)


def record_query_weights(rec: dict, i: int, layer, sources: list) -> None:
    """attn_res._monitor's call, before layer i, on a forward whose encoder has qmix_record set (the trainer sets it for one
    micro-batch): rec[i] = (the token-sums of the layer's query-mix weights over `sources`, the tokens), as the layer is about to use
    them: the full softmax before the choice, the chosen pair's (zero on the others) after it. A layer without a query mix adds
    nothing. Under no_grad, outside the compiled layers: a second, eager read of the mix, on monitored forwards only."""
    from ced.model.attn_res import depth_weight_sums

    arms = getattr(layer, "gdn_arms", None)
    if arms is None or arms.variant != "qmix" or arms.qmix_query is None or sources is None:
        return
    assert len(sources) >= 2, "a layer with a query mix reads the embedding output and a finished block at least"
    rec[i] = depth_weight_sums(sources, arms.qmix_query, arms.qmix_keep if arms.qmix_static else None)


def read_query_weights(rec: dict) -> dict:
    """{layer: (token-mean weights as a list of Python floats over its sources, tokens)} from a record: the sums divided by the tokens
    in float64, one host read for all the layers."""
    if not rec:
        return {}
    layers = sorted(rec)
    flat = torch.cat([rec[i][0] for i in layers]).cpu()
    out, at = {}, 0
    for i in layers:
        sums, tokens = rec[i]
        out[i] = ((flat[at:at + sums.numel()] / tokens).tolist(), tokens)
        at += sums.numel()
    return out


def qmix_pending(encoder) -> bool:
    """Whether a layer is waiting for its choice: it has the buffer (a query mix over more than QMIX_KEEP sources under
    EncoderConfig.gdn_qmix_select_step) and has not chosen. False in every model without the option."""
    return any(a is not None and hasattr(a, "qmix_keep") and not a.qmix_static
               for a in (getattr(layer, "gdn_arms", None) for layer in encoder.layers))


@torch.no_grad()
def select_qmix_pairs(encoder, rec: dict) -> dict:
    """The choice, once (module docstring): in every layer that has the buffer qmix_keep and has not chosen, the QMIX_KEEP sources
    with the largest token-mean query-mix weight in `rec` (a record of the full softmax: the forward ran before any choice), ties to the
    lower source index (a sort on (-weight, index), the rule of tools/analysis/qmix_mix_read.py's top-k), written ascending into
    qmix_keep in place (the state dict keeps it) and qmix_static set. Returns {layer: {"pair": [a, b], "weights": [token-mean weight of
    every source], "tokens": N}} for the layers chosen now, in stack order. Deterministic: float64 sums of the same weights."""
    means = read_query_weights(rec)
    out = {}
    for i, layer in enumerate(encoder.layers):
        arms = getattr(layer, "gdn_arms", None)
        if arms is None or not hasattr(arms, "qmix_keep") or arms.qmix_static:
            continue
        assert i in means, f"layer {i} has no recorded weights: the choice needs a monitored forward before any choice"
        w, tokens = means[i]
        assert len(w) == arms.n_sources, (i, len(w), arms.n_sources)
        pair = sorted(sorted(range(len(w)), key=lambda s: (-w[s], s))[:QMIX_KEEP])
        arms.qmix_keep.copy_(torch.tensor(pair, dtype=torch.long, device=arms.qmix_keep.device))
        arms.qmix_static = True
        out[i] = {"pair": pair, "weights": w, "tokens": tokens}
    return out


def qmix_weights_metrics(encoder, rec: dict) -> dict:
    """The log of a monitored forward's record (the trainer, metrics["qmix_weights"]): {"layers": [i], "n_sources": [n], "weights":
    [[token-mean weight of every source]], "tokens": N, "chosen": [[a, b] or None]} in stack order; before the choice the full
    weights of every source, after it the chosen pair's (zero on the others), as the layer used them."""
    means = read_query_weights(rec)
    layers = sorted(means)
    chosen = []
    for i in layers:
        arms = encoder.layers[i].gdn_arms
        chosen.append(arms.chosen_sources() if hasattr(arms, "qmix_keep") else None)
    return {"layers": layers, "n_sources": [len(means[i][0]) for i in layers],
            "weights": [[round(x, 8) for x in means[i][0]] for i in layers], "tokens": means[layers[0]][1] if layers else 0,
            "chosen": chosen}


@torch.no_grad()
def summary(encoder) -> dict | None:
    """The arms' tensors as they are now, for the logs (the trainer, every log_every steps): per layer that has them,
    in stack order, "write_lambda" (the values), "write_query_norm" (||w|| of the write mix's query: how far its depth
    weights have moved from uniform; None in block 1), "ccr_gate" (the values). With the wsl variant "qmix" also
    "qmix_lambda" and "qmix_query_norm", the same for the q rows' own mix (absent in every other model). With CCR v2's translator also
    "ccr_translator_dev" (||M - I||_F per key head) and "ccr_translator_dev_2" (||M - I||_2, the largest singular value,
    per key head: the most M stretches or turns any one query direction; it is at most the Frobenius norm and equals it
    when M - I has rank one), None for a layer without one. Python floats; one host read (and one copy of M - I to the
    host for the singular values)."""
    rows = []
    for i, layer in enumerate(encoder.layers):
        arms = getattr(layer, "gdn_arms", None)
        if arms is None:
            continue
        rows.append((i, arms))
    if not rows:
        return None
    out = {"layers": [i for i, _ in rows]}
    if any(a.write_mix != "none" for _, a in rows):
        out["write_lambda"] = [a.write_lambda.detach().double().cpu().tolist() if a.write_mix != "none" else None for _, a in rows]
        out["write_query_norm"] = [float(a.write_query.query.detach().double().norm()) if a.write_query is not None else None
                                   for _, a in rows]
    if any(a.variant in READ_MIX_VARIANTS for _, a in rows):  # the second mix (qmix: the q rows'; kqmix: the q and k rows'): its lambda and its query
        out["qmix_lambda"] = [a.qmix_lambda.detach().double().cpu().tolist() if a.variant in READ_MIX_VARIANTS else None for _, a in rows]
        out["qmix_query_norm"] = [float(a.qmix_query.query.detach().double().norm())
                                  if a.variant in READ_MIX_VARIANTS and a.qmix_query is not None else None for _, a in rows]
    if any(a.ccr_read for _, a in rows):
        out["ccr_gate"] = [a.ccr_gate.detach().double().cpu().tolist() if a.ccr_read else None for _, a in rows]
    if any(a.translator != "none" for _, a in rows):  # CCR v2: ||M - I||_F and ||M - I||_2 per key head (0 at the start)
        out["ccr_translator_dev"] = [getattr(a, TRANSLATOR).detach().double().flatten(1).norm(dim=1).cpu().tolist()
                                     if a.translator != "none" else None for _, a in rows]
        # the spectral norm on the host in float64 (LAPACK; no device solver is started for a log line)
        out["ccr_translator_dev_2"] = [torch.linalg.matrix_norm(getattr(a, TRANSLATOR).detach().double().cpu(), ord=2).tolist()
                                       if a.translator != "none" else None for _, a in rows]
    return out


# ------------------------------------------------------------------------------------------------ EncoderConfig.mix_fold's ladder above 'producer'
# (ced/model/mx_p2.py). Each value is the one before it plus one thing: "producer" the norm pass forms the mixed operands (mx_wire.MXInNormMix); its
# backward is autograd's formulas of the unwired statements "producer-rc" + the fused write mix's group read recomputes the block's extra mixes in its
# backward (ced/kernels/write_mix_rc.py) "producer-p2" + the layer's backward is ONE kernel pass (ced/kernels/mx_mix_bwd_triton.py), the consumers'
# bf16 Dgrad outputs handed to it as written "producer-p3" + in_proj_a / in_proj_b are computed in the norm pass from the float32 x_w
# (ced/kernels/mx_mix_p3_triton.py) They are the producer form plus flags (mix_fold_form says "producer" for all four): mix_rc on the group read's
# layers, mix_p on the arms layers. (Defined here, after the code that reads them at call time, so that no line of the module above moves.)

RC_FORMS = ("producer-rc", "producer-p2", "producer-p3")  # the values that recompute the group read's extra mixes (mix_rc)
P_FORMS = {"producer-p2": 2, "producer-p3": 3}  # the layer-side level (mix_p): 2, the one-pass backward; 3, that and a / b in the norm pass


def mix_p(e) -> int:
    """The layer-side level of EncoderConfig.mix_fold `e`: 2 for 'producer-p2' (the layer's backward as ONE kernel pass), 3 for 'producer-p3' (that, and
    in_proj_a / in_proj_b computed in the norm pass), 0 for every other value (mix_fold_form says what is valid)."""
    return P_FORMS.get(e.mix_fold, 0) if isinstance(e.mix_fold, str) else 0
