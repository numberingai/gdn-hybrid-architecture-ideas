"""The non-looped control: our own encoder block stack with one LM head.

No loop and no encoder/decoder split: every position runs every layer and is scored by one head. It
uses exactly the same blocks, norms and initialisation as the encoder of Looped-CED (v1) and of CED
v2, so the comparison changes one thing. v1: its depth matches Looped-CED's non-embedding parameter
count (see ced/utils/params.py), tied head. v2: controls
(a) (time-matched depth) and (b) (parameter-matched depth) are this class with the v2 layer
settings (sigmoid GDN gate) and an untied output embedding.

Per-layer residual scalars with x0 re-injection (EncoderConfig.layer_scalars "skip_x0"): every layer of the stack
computes x <- a x + b x0 + f(x), x0 the embedding output, a and b one trained scalar per layer that start at 1 and 0
(ced/model/encoder.py apply_layer_scalars, called by each layer as the last part of its own residual add, so inside a
compiled layer's graph; encoder.skip_gain and encoder.x0_gain). Off by default: the model above, bit for bit. One pass
only (check_layer_scalars). Two more forms of the same layer rule (ced/model/encoder.py layer_scalar_terms):
"skip_x0_span", where every product of a over consecutive layers stays inside (1/R, R), R =
EncoderConfig.layer_scalars_span, and "x0", where a is 1 exactly. Both fix b of layer 0 at 0 and start as the model
above, bit for bit.

Two passes (the loop on the plain hybrid; sections 2 and 3). Off by default (LoopConfig.pass2_gate "none"): the model
above, bit for bit, with the same parameters (tests/test_loop_control.py, against main's code). With pass2_gate
"scalar" and pass2_layers n, the top n layers of the stack (the block: layers N - n .. N - 1) run a second time, behind
the gate of the stage 2 build's "scalar" form:

    z_1 = Stack(embed(x))                      pass 1: the single-pass model, every position
    s_2 = Block(z_1 + e_2)                     pass 2: the block's layers again, writing nothing
    z_2 = z_1 + alpha * (s_2 - z_1)            alpha: one trained scalar; e_2: a (d,) embedding; both start at 0,
    logits = OutEmbed(RMSNorm_final(z_2))      where the model is the single-pass model exactly

In pass 2 the block's layers are the same layers (weights, norms, residual scale 1) in zero-write mode
(ced/model/encoder.py EncoderLayer.zero_write / zero_write_all: control (c)'s second pass):
- a full-attention layer puts pass 2's own query at t (rotary at t's document position, Scalable-Softmax with the
  key count pos_t + 1) over THAT layer's pass-1 keys and values at s <= t of t's document, t's own included: the
  causal and document mask and the key counts of pass 1. No key or value is projected.
- a GDN layer reads the state pass 1 left after t (t's own write included) with pass 2's query, whose short
  convolution reads pass 1's inputs at t-3 .. t-1 and pass 2's at t. No key, value, write strength or decay is
  formed; the state and the convolution history stay pass 1's.
Pass 2 writes nothing. So the output at t depends only on pass-1 quantities at s <= t and on z_1(t), never on which
other positions run pass 2; the cache for later chunks is pass 1's (ControlCache: one K/V set per attention layer
and token, one state per GDN layer, whatever T); and read tokens need one pass.
Where pass 2 runs: at every position of the chunk by default (pass2_mask None: training and scoring; fixed shapes,
nothing gathered: zero_write_all), or at the first pass2_capacity positions of each row that pass2_mask marks (cached
generation: the positions whose next token is generated; the slots of ced/model/capacity.py: zero_write), and the
other positions keep z_1. prefill and extend take their decoder mask as that mask (None: every position).
Pass 1's records are not detached: pass 2's loss trains pass 1's keys, values and states (full backprop through both
passes). A forward at T = 1 of a two-pass model is its pass 1 alone (the trainer's steps before a loop switch step).
New tensors (513 at d = 512): hyper_connections.pass2_gate (alpha) and loop_embedding_pass2 (e_2, with
loop.loop_embedding), the stage 2 build's names (ced/model/loop.py PASS2_PARAMETERS); pass2_parameter_names(),
alpha() and hyper_connections.is_identity() are what the trainer reads (ced/train/trainer.py).
Layer checkpointing (ced/model/act_checkpoint.py) wraps every layer call of both passes. Monitoring adds
"act_max/pass2.enc<i>" per block attention layer and "loop_usage" (the mean of ||z_2 - z_1|| / ||z_2||).
Refused (check_looped_control): the "hc" form (no lanes), a block outside 1 .. N layers, t_max other than 2, a
looped encoder, the asentmax normaliser, NVFP4 fake quantisation (pass 2 applies part of a layer's in-projection
directly), and T > 2 in a forward (a third pass exists only through the segment map below).

Segment-based pass counts (LoopConfig.segment_passes / unscored_passes; quoted in ced/model/pass_schedule.py, which
holds everything that is not specific to a model: the plan, its gathers, the slot sizes; the counts come from
ced/train/pass_positions.py). Neither key set: nothing below runs and the model is the code above bit for bit
(HybridLM.schedule is None). With a schedule (the gated control only), a forward at T = 2 runs `scheduled_passes`. A
unit of the plan is a POSITION of the chunk (C = L: pass 1 runs the whole stack at every one):

    pass 1    every position, exactly the single-pass control (Encoder.forward_keep): it writes each layer's K/V and
              states and, in cached inference, the cache. z_1 is every position's pass-1 state.
    pass p    (p = 2, 3) only the positions whose count is >= p, gathered into K_p pass slots (pass_schedule.take:
              pass 2 from z_1, pass 3 from z_2 at its own slots): u = z_{p-1} + e_p through the block's layers in
              zero-write mode at the slots (EncoderLayer.zero_write), then z_p = z_{p-1} + alpha_p (u - z_{p-1}).
              A full-attention layer's K_p queries read that layer's pass-1 K/V over the whole row under the mask
              rows and Scalable-Softmax key counts of THEIR positions. A GDN layer's slot reads the state pass 1 left
              after the slot's own position (GatedDeltaNet.zero_write_read), with pass 1's q inputs at t-3 .. t-1 in
              its short convolution. Nothing is projected into a key, value or state and nothing is written: no K/V,
              no recurrent state, no convolution history. A padding slot computes on zeros and nothing reads it.
              A pass that every position runs (the map all-2, or a row of one count) is not gathered: it is the dense
              call above (EncoderLayer.zero_write_all), so all-2 through the schedule is the dense form bit for bit.
    head      RMSNorm_final of pass_schedule.readout: per position the state after the LAST pass it ran (z_1 where
              the count is 1), so the loss trains the head on pass 1's output there.

What pass 1 keeps for a GDN layer's read, and what the read costs, depends on the kernel (GDNPass1):
- "ced" on CUDA in a training forward (the runs): the chunk states h (the state at the start of every chunk of 64
  tokens) and v_new, which the kernel's own backward keeps anyway (ced/kernels/gdn.py, keep "all"). The read puts the
  K_p queries back at their positions (zero elsewhere), runs the read kernel over the WHOLE row and gathers the K_p
  outputs: the state read of a gathered pass costs one read over L positions per GDN layer whatever K_p is; only
  the rest of the layer (projections, norms, FFN) runs on K_p slots. An O(K_p) read needs a kernel of its own.
- "fla", and "chunked" (the CPU's fast path): pass 1's k, v, g, beta; the read is one more delta-rule call over the
  row with the queries at their positions.
- "recurrence" (the reference; tests and one-token decode steps): the state after EVERY position of the chunk,
  (B, L, H_v, d_k, d_v), gathered at the slots: O(K_p) to read, L states to keep (262 kB a position at 60M), so
  only for short chunks. A one-token step keeps one state per row, which is the cache's own.
Nothing a later pass computes is read by another position, so a position's output does not depend on any other
position's count, and the cache after a chunk is the T = 1 cache whatever the counts (tests/test_seg_control.py; the
mutation check there makes pass 2 write and shows the test fail). A third pass exists only through the map ("A third
pass defined like the second. The spec's ban on T > 2 is lifted for this experiment only."): its gate
hyper_connections.pass3_gate and its loop-index embedding loop_embedding_pass3 start at zero, so at switch-on every
count gives the single-pass output bit for bit.
The plan comes from the caller (`passes`: the trainer's, built by the data path on the CPU over all L positions),
from per-position counts (`pass_counts`), or from the model's own host-side scheduler over the tokens (streaming
across the chunks of a cached forward: ControlCache.pass_state). In prefill / extend the decoder mask marks the
positions whose logits are read: every other position runs pass 1 only, whatever its segment (a read token costs one
pass). A forward whose plan holds no later pass (every count 1) is the T = 1 forward and keeps no record.
Masks: pass 1 is the single-pass control's code (each of its attention layers builds its own mask, as the trunk
does). Each later pass builds ONE mask for the block's attention layers (the flex BlockMask, or the dense mask of
the reference and sdpa backends); a pass at the previous pass's own slots reuses that pass's, and a dense mask of
pass 3 is a row gather of pass 2's. Under the rotary distance cap the capped read builds its own two masks per layer
(no caller-built mask), and so does flex over a cache (its key count grows each step).

The loop in the published form (LoopConfig.recur_layers). A second, separate loop of this arch; the two are refused
together and share no code path. Off by default (recur_layers ()): nothing below runs and the model is the code above
bit for bit (HybridLM.recur is None). With recur_layers [first, end] the stack is cut in three (control (a): [4, 16],
"prelude = group 1 (layers 0-3), looped middle = groups 2-4 (layers 4-15), coda = group 5 (layers 16-19)"):

    p      = Prelude(embed(x))                 layers 0 .. first - 1, once
    z_k    = Middle(u_k), k = 1 .. T           layers first .. end - 1, the same weights at every pass
    logits = OutEmbed(RMSNorm_final(Coda(z_T)))   layers end .. N - 1, once

"Every pass writes its own GDN state and K/V. Within each pass, attention is ordinary causal attention over that
pass's own K/V.": every layer call of every pass is the single-pass model's layer call on that pass's stream over the
whole chunk (ced/model/encoder.py Encoder.forward_recur), so a pass reads nothing another pass wrote; it takes only
the stream z_{k-1} of the pass before, position by position. The position settings are the stack's (the layers are
the same layers: nope_global_layers keeps its meaning). What enters a pass:
- loop.injection "parcae" (the Parcae form, ced/model/loop.py RecurInjection): e = RMSNorm(p)
  (recur_prelude_norm), z_0 = 0 (recur_state_init "zeros") or e ("prelude"), and u_k = decay * z_{k-1} + dt * (B e):
  the injection comes first and the block runs on the injected state. With the layers' own residual stream, z_k =
  u_k + f(u_k): with f read at u_k, not at z_{k-1}. Also at k = 1 and at T = 1 (one pass is Middle(dt * B e) under
  "zeros").
- loop.injection "none": u_1 = p and u_k = z_{k-1}: the plain stack with its middle repeated, the model of N +
  (T - 1)(end - first) layers with tied weights. No tensor is added: exactly the plain control's parameters.
- loop.injection "mix" (the late switch-on, "growth"; ced/model/encoder.py recur_input_mix): u_1 = p, as without
  injection, and u_k = w p + (1 - w) z_{k-1} for k >= 2, with the same w and the same e = p at every later pass.
  One tensor is added, the 0-dim encoder.recur_mix (w, at loop.recur_mix_init): the plain control's parameters plus
  one. At T = 1 the model is the plain control bit for bit (w is not read). At w = 1 a later pass reads p itself
  and repeats the first pass, so the model at any T is the one-pass model; at 0.99 it is near it. No norm at the
  loop boundary, nothing of the Parcae form (no A, no B, no decay, no start state), and the layers keep their
  width: every other line of this forward is the forward of injection "none".
- loop.injection "depth_mix" (R3, depth mixing at the loop boundary; ced/model/recur_depth.py has the
  form): EVERY pass reads a softmax mix over its candidates, per position: u_k = sum over V_k of alpha_v v with alpha =
  softmax over V_k of q^T RMSNorm(v), V_1 = [x0, p] and V_k = [x0, p, z_1... z_{k-1}], x0 the embedding's output; the
  coda reads z_T. One reader for every pass (R1's DepthQuery, encoder.recur_depth): a (d,) pseudo-query and the (d,)
  gain of its key norm, zeros at the start: the plain control's parameters plus 2 d. The layers keep their plain
  residual add and their width. At its start every candidate has equal weight (pass 1 reads (x0 + p) / 2), so this model
  is NOT the plain control at any T, T = 1 included, and it takes no switch step: it trains from scratch. No loop-index
  embedding, no hyper-connection, no gate, no draft loss, no residual scale on the middle. T is a forward argument: any
  T >= 1, by default loop.t_eval, or the environment variable CED_LOOP_T where a caller gives none (the scoring scripts:
  one switch for scoring a checkpoint at another pass count; it is one of the CED_* variables the resumable scoring keys
  on). Full backprop through all passes. Cached generation (prefill / extend): ControlCache.encoder holds one entry per
  layer APPLICATION (first + T (end - first) + N - end of them, in running order: every pass of the middle has its own
  GDN states and K/V), and a cached sequence keeps the T it was prefilled with. dec_mask is not read: every position
  runs every pass, a read token included (a later pass at position t needs that pass's keys and states of every earlier
  position). Refused (check_recur_loop): any other arch, pass2_gate and its keys, encoder.layer_scalars,
  encoder.canon_kernel, model.nvfp4, a looped encoder, a bad range, an unknown injection or state init, injection "mix"
  or a recur_mix_init other than the default on a model without that mix, injection "depth_mix" on a model without a
  looped middle, without a prelude (first = 0: x0 and p would be one stream) or with a pass count per row (t_sampling
  "sequence": nothing tests the pair). encoder.attn_res (R1, Block AttnRes inside the stack) stays refused together with
  any looped middle (attn_res.check_attn_res), this one included.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn as nn

import os

from ced.config import BAG_INPUTS, EncoderConfig, ModelConfig
from ced.model import act_checkpoint, pass_schedule
from ced.model.attention import slots_allowed
from ced.model.attn_res import check_attn_res
from ced.model.capacity import (gather_slots, gather_slots_inv, scatter_slots, scatter_slots_inv, select_slots,
                                 slot_inverse)
from ced.model.decoder import SlotContext
from ced.model.encoder import Encoder, record_activation_max
from ced.model.loop import LATER_PASS_PARAMETERS, PASS2_PARAMETERS, ScalarSecondPassGate  # noqa: F401 (PASS2_PARAMETERS re-exported)
from ced.model.looped_ced import ModelOutput, init_weights, lm_head, output_embedding
from ced.model.norms import RMSNorm
from ced.model.scaling import residual_scales
from ced.model.sequence import SeqContext, SeqState, sequence_context


@dataclass
class ControlCache:
    """seq: index / document bookkeeping; encoder: per layer, a GDNState or the global layer's KeySet (pass 1's:
    a second pass writes nothing); T: the pass count prefill ran with (extend's default; the cache itself is the
    same whatever T; 1 for a control without a second pass); pass_state: with a segment pass schedule whose counts
    the model's own scheduler reads from the tokens (ced/train/pass_positions.py PassScheduler.feed), the state of
    the segment state machine after this chunk, a host object; None otherwise (no schedule, T = 1, or the caller gave
    the counts). Not a tensor: the K/V and states of the cache are the T = 1 cache whatever the counts.
    With the loop in the published form (loop.recur_layers): encoder holds one entry per layer APPLICATION in running
    order (the prelude's layers, each pass of the middle, the coda's), every pass with its own GDN states and K/V, and
    T is the pass count every chunk of the sequence runs."""

    seq: SeqState
    encoder: list
    T: int = 1
    pass_state: object | None = None
    # The Copy-Paste head's copy cache (ModelConfig.cp_head; ced/model/cp_head.py CopyCache: the keys, token ids,
    # document ids and indices of every position so far); None for a model without the head.
    copy: object | None = None


def check_looped_control(cfg: ModelConfig) -> None:
    """Refuse the second-pass keys (LoopConfig.pass2_gate, pass2_layers) wherever the looped control is not built
    (the module docstring, "Two passes"), so that no model ignores a key it was given. ValueError."""
    lc = cfg.loop
    scheduled = lc.segment_passes is not None or lc.unscored_passes is not None
    if cfg.arch != "control":
        if lc.pass2_layers:
            raise ValueError(f"loop.pass2_layers {lc.pass2_layers!r} is the looped block of arch 'control', not for arch "
                             f"{cfg.arch!r} (CED v2's second pass is its whole decoder): leave it 0")
        if scheduled and cfg.arch != "ced_v2":  # CED v2 checks its own conditions (ced/model/ced_v2.py)
            raise ValueError("loop.segment_passes / unscored_passes are built for arch 'ced_v2' (the gated two-pass "
                             f"decoder with self-KV) and arch 'control' (loop.pass2_gate 'scalar'), not {cfg.arch!r}")
        return
    if lc.pass2_gate == "none":
        if lc.pass2_layers:
            raise ValueError(f"loop.pass2_layers {lc.pass2_layers!r} without loop.pass2_gate 'scalar': the control's top "
                             "block is looped only behind the gate; set both, or leave pass2_layers 0")
        if scheduled:
            raise ValueError("loop.segment_passes / unscored_passes on arch 'control' say where the later passes of its "
                             "top block run: they need the gated second pass (loop.pass2_gate 'scalar' with "
                             "pass2_layers); this control has none")
        return
    pass_schedule.schedule_of(lc)  # a map that is not {segment: 1 | 2 | 3}, or an unscored rule other than 1: ValueError
    if lc.pass2_gate != "scalar":
        raise ValueError(f"loop.pass2_gate {lc.pass2_gate!r} on arch 'control': 'none' or 'scalar' (the control has no "
                         "hyper-connection lanes, so the 'hc' form does not exist for it)")
    n = cfg.encoder.n_layers
    if type(lc.pass2_layers) is not int or not 1 <= lc.pass2_layers <= n:
        raise ValueError(f"loop.pass2_layers {lc.pass2_layers!r}: with loop.pass2_gate on arch 'control', the number of "
                         f"top layers the second pass runs through, 1 .. {n} (encoder.n_layers)")
    if lc.t_max != 2:
        raise ValueError(f"loop.pass2_gate on arch 'control' is built for T = 2 (loop.t_max = 2), not t_max = {lc.t_max}")
    if lc.t_eval not in (1, 2):
        raise ValueError(f"loop.t_eval {lc.t_eval!r}: 1 or 2 for the control with a second pass (loop.pass2_gate)")
    if cfg.encoder.loops != 1:
        raise ValueError("loop.pass2_gate on arch 'control' is not built for a looped encoder (encoder.loops > 1)")
    if cfg.encoder.attention.normalizer != "softmax":
        raise ValueError("loop.pass2_gate on arch 'control' is built for the softmax normaliser, not "
                         f"{cfg.encoder.attention.normalizer!r} (its training statistics would count pass 2's reads)")
    if cfg.nvfp4.mode != "none":
        raise ValueError(f"loop.pass2_gate on arch 'control' is not built for model.nvfp4 (mode {cfg.nvfp4.mode!r}): pass "
                         "2 applies the q rows of a GDN layer's in-projection directly, outside the quantised linear")


RECUR_INJECTIONS = ("parcae", "none", "mix", "depth_mix")
RECUR_STATE_INITS = ("zeros", "prelude")
T_SAMPLINGS = ("step", "sequence")
# The one switch for scoring a checkpoint with a looped middle at another pass count: where a caller gives no T, the
# forward runs this many passes instead of loop.t_eval (HybridLM.passes). Read at every forward, by that model only.
ENV_LOOP_T = "CED_LOOP_T"


def check_recur_loop(cfg: ModelConfig) -> None:
    """Refuse the loop in the published form (LoopConfig.recur_layers; the module docstring) wherever it is not built,
    so that no model ignores the key, and refuse its own keys' bad values. ValueError."""
    from ced.model.loop import RECUR_B_FORMS

    lc = cfg.loop
    if lc.t_sampling not in T_SAMPLINGS:
        raise ValueError(f"loop.t_sampling {lc.t_sampling!r}: one of {T_SAMPLINGS}")
    r = lc.recur_layers
    mix0 = type(lc)().recur_mix_init  # the default start of the scalar input mix
    if isinstance(lc.recur_mix_init, bool) or not isinstance(lc.recur_mix_init, (int, float)) or not math.isfinite(lc.recur_mix_init):
        raise ValueError(f"loop.recur_mix_init {lc.recur_mix_init!r}: the start of the scalar input mix's w, a finite number")
    if lc.recur_mix_init != mix0 and not (r and lc.injection == "mix"):
        raise ValueError(f"loop.recur_mix_init {lc.recur_mix_init!r} is the start of the scalar input mix of a looped middle "
                         f"(loop.recur_layers with loop.injection 'mix'); this model has none: leave it {mix0}")
    if not r:
        if r != ():
            raise ValueError(f"loop.recur_layers {r!r}: () (no loop) or [first, end]")
        if lc.t_sampling != "step":
            raise ValueError(f"loop.t_sampling {lc.t_sampling!r} draws a pass count per row for the loop in the "
                             "published form (loop.recur_layers): this model has none; leave it 'step'")
        if lc.injection == "mix":
            raise ValueError("loop.injection 'mix' is the scalar input mix of a looped middle (loop.recur_layers on arch "
                             "'control'): this model has no looped middle and would ignore it")
        if lc.injection == "depth_mix":
            raise ValueError("loop.injection 'depth_mix' is the depth mix at the boundary of a looped middle (R3: "
                             "loop.recur_layers on arch 'control'): this model has no looped middle and would ignore it")
        return
    if cfg.arch != "control":
        raise ValueError(f"loop.recur_layers {list(r)!r} is the looped middle of arch 'control', not for arch {cfg.arch!r}")
    n = cfg.encoder.n_layers
    if not (isinstance(r, tuple) and len(r) == 2 and all(type(i) is int for i in r) and 0 <= r[0] < r[1] <= n):
        raise ValueError(f"loop.recur_layers {r!r}: [first, end] with 0 <= first < end <= {n} (encoder.n_layers): the "
                         "layers first .. end - 1 are the looped middle")
    if lc.pass2_gate != "none" or lc.pass2_layers or lc.segment_passes is not None or lc.unscored_passes is not None:
        raise ValueError("loop.recur_layers (the loop in the published form) and the gated second pass (loop.pass2_gate, "
                         "pass2_layers, segment_passes, unscored_passes) are two loops of arch 'control': set one")
    if cfg.encoder.loops != 1:
        raise ValueError(f"loop.recur_layers is not built for a looped encoder (encoder.loops {cfg.encoder.loops})")
    if cfg.encoder.layer_scalars != "none":
        raise ValueError(f"loop.recur_layers is not built together with model.encoder.layer_scalars "
                         f"({cfg.encoder.layer_scalars!r}): nothing tests the pair")
    if cfg.nvfp4.mode != "none":
        raise ValueError(f"loop.recur_layers is not built together with model.nvfp4 (mode {cfg.nvfp4.mode!r}): nothing "
                         "tests the pair")
    if cfg.encoder.canon_kernel:
        raise ValueError(f"loop.recur_layers is not built together with model.encoder.canon_kernel ({cfg.encoder.canon_kernel!r}, "
                         "the Canon layers): nothing tests the pair (a pass would carry its own Canon history)")
    if cfg.encoder.attention.normalizer != "softmax":
        raise ValueError("loop.recur_layers is built for the softmax normaliser, not "
                         f"{cfg.encoder.attention.normalizer!r} (its training statistics would count every pass)")
    if lc.injection not in RECUR_INJECTIONS:
        raise ValueError(f"loop.injection {lc.injection!r} with loop.recur_layers: one of {RECUR_INJECTIONS}")
    if lc.injection == "depth_mix":
        # R3 (ced/model/recur_depth.py): the candidates of pass 1 are the embedding's output and the prelude's output
        if r[0] == 0:
            raise ValueError(f"loop.injection 'depth_mix' with loop.recur_layers {list(r)!r}: the depth mix reads the "
                             "embedding's output and the prelude's output as two candidates; without a prelude (first = 0) "
                             "they are one stream. Give the middle a prelude of at least one layer")
        if lc.t_sampling != "step":
            raise ValueError(f"loop.injection 'depth_mix' is not built together with loop.t_sampling {lc.t_sampling!r} (a "
                             "pass count per row): nothing tests the pair. Leave it 'step'")
    if lc.recur_inject_b not in RECUR_B_FORMS:
        raise ValueError(f"loop.recur_inject_b {lc.recur_inject_b!r}: one of {RECUR_B_FORMS}")
    if lc.recur_state_init not in RECUR_STATE_INITS:
        raise ValueError(f"loop.recur_state_init {lc.recur_state_init!r}: one of {RECUR_STATE_INITS}")
    if type(lc.recur_prelude_norm) is not bool:
        raise ValueError(f"loop.recur_prelude_norm {lc.recur_prelude_norm!r}: true or false")
    if not 0.0 < lc.parcae_init_decay < 1.0:
        raise ValueError(f"loop.parcae_init_decay {lc.parcae_init_decay!r}: the injection's initial decay, in (0, 1)")
    if type(lc.t_eval) is not int or lc.t_eval < 1:
        raise ValueError(f"loop.t_eval {lc.t_eval!r}: the pass count of a forward that is given none, an integer >= 1")
    ok = (isinstance(lc.t_choices, list) and lc.t_choices and all(type(t) is int and t >= 1 for t in lc.t_choices)
          and isinstance(lc.t_probs, list) and len(lc.t_probs) == len(lc.t_choices))
    if not ok:
        raise ValueError(f"loop.t_choices {lc.t_choices!r} / t_probs {lc.t_probs!r}: pass counts >= 1 and one "
                         "probability each")


def env_loop_t() -> int | None:
    """CED_LOOP_T as a pass count, or None when it is not set. ValueError on anything but an integer >= 1."""
    text = os.environ.get(ENV_LOOP_T, "").strip()
    if not text:
        return None
    if not text.isdigit() or int(text) < 1:
        raise ValueError(f"{ENV_LOOP_T}={text!r}: a pass count, an integer >= 1")
    return int(text)


class HybridLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.arch == "control"
        check_looped_control(cfg)
        check_recur_loop(cfg)
        self.cfg = cfg
        eps_enc, _ = residual_scales(cfg)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_norm = RMSNorm(cfg.d_model, cfg.norm_eps) if cfg.embed_norm else nn.Identity()
        self.encoder = Encoder(cfg, eps_enc)
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.out_embed = output_embedding(cfg)
        init_weights(self, cfg.init_std)
        if cfg.encoder.attn_res == "ldm" or cfg.encoder.attn_res_query == "token":
            # The architecture arms' address networks (Linear Depth Memory; R1-entry's token-dependent queries): drawn
            # from a generator of their own (ced/model/ldm.py init_own_parameters), after every other tensor, which
            # therefore starts as in the model without them. Off: this line is not reached.
            from ced.model.ldm import init_own_parameters

            init_own_parameters(self, cfg.init_std)
        # EDA's erase direction (GDNConfig.erase; ced/model/gdn_erase.py): its tensors drawn from a generator of their own after every
        # other tensor, which therefore starts as in the model without them. Off (mode "none"): no such tensor, nothing drawn.
        from ced.model.gdn_erase import init_own_parameters as init_erase

        init_erase(self, cfg.init_std)
        # Gated DeltaProduct's extra steps (GDNConfig.dprod; ced/model/gdn_dprod.py): drawn from a generator of their own after
        # every other tensor, which therefore starts as in the model without them. Off (n_h 1): no such tensor, nothing drawn.
        from ced.model.gdn_dprod import init_own_parameters as init_dprod

        init_dprod(self, cfg.init_std)
        # Two passes (module docstring): only with a gate. Without one no tensor and no module is added, so the state
        # dict, the parameter order and the RNG use are the control's as before. With one, the single-pass model's
        # tensors keep their names, shapes, relative order and initial values, and pass 2's own are added (zeros,
        # drawn from no RNG): a T = 1 checkpoint loads strictly into the others.
        self.gated = cfg.loop.pass2_gate != "none"
        # The loop in the published form (module docstring): (first, end) of the looped middle, or None (no such
        # loop: every line below that names it is skipped). Its tensors, if any, are the encoder's (the injection).
        self.recur = tuple(cfg.loop.recur_layers) or None
        # the first layer of the looped block (N without a gate: no layer)
        self.pass2_from = cfg.encoder.n_layers - (cfg.loop.pass2_layers if self.gated else 0)
        # Segment-based pass counts (module docstring; ced/model/pass_schedule.py). schedule None: no schedule, the
        # code as it was. n_passes: the passes this model has tensors for: 1 without a gate, 2 with one, 3 when the
        # segment map holds a 3 (the ONLY way to a third pass).
        self.schedule = pass_schedule.schedule_of(cfg.loop) if self.gated else None
        self.n_passes = pass_schedule.built_passes(cfg.loop) if self.gated else 1
        if self.gated:
            self.hyper_connections = ScalarSecondPassGate(self.n_passes)
            if cfg.loop.loop_embedding:
                self.loop_embedding_pass2 = nn.Parameter(torch.zeros(cfg.d_model))
                if self.n_passes > 2:  # a third pass (the segment map only): its own loop-index embedding, zeros
                    self.loop_embedding_pass3 = nn.Parameter(torch.zeros(cfg.d_model))
        # The segment pass schedule's host side, plain attributes (never in the state dict), as CED v2's:
        # pass_scheduler reads per-position counts from the tokens when a caller gives none (built from
        # loop.segment_tokenizer at first use; a test or a trainer may set it); pass_buckets are the slot-size
        # fractions of the plans the model builds ITSELF (evaluation, scoring, generation): the full size only by
        # default, so that scoring a scheduled model traces no more call shapes per row shape than the dense
        # two-pass model does; last_plan is the plan of the last scheduled forward (monitoring and tests).
        self.pass_scheduler = None
        self.pass_buckets = (1.0,)
        self.last_plan = None
        # Monitoring (off by default; no effect on outputs): with collect_stats, every forward fills
        # self.stats with the max |activation| at the input of each global-attention layer (at T = 2 also of the
        # block's attention layers in pass 2, "act_max/pass2.enc<i>", and "loop_usage").
        self.collect_stats = False
        self.stats: dict = {}
        if cfg.cp_head.enabled:
            # The Copy-Paste head (ModelConfig.cp_head; ced/model/cp_head.py): registered after every other tensor and
            # drawn from a generator of its own, so the base tensors' names, order, initial values and the global RNG
            # are the model's without the key. Off: this line is not reached.
            from ced.model.cp_head import attach_cp_head, check_cp_head

            check_cp_head(cfg)
            attach_cp_head(self)

    def lm_head_weight(self) -> torch.Tensor:
        """The (V, d) LM head matrix over the real vocabulary: the input embedding when tied, else the output
        embedding. With a Copy-Paste head this is p_LM's head (the <copy> row is cp_head.w_copy)."""
        return self.embed.weight if self.out_embed is None else self.out_embed.weight

    def head_weight(self) -> torch.Tensor:
        """The (V, d) LM head matrix (lm_head_weight). A model with a Copy-Paste head whose next-token distribution is
        the mixture (cp_head.score "mixture") refuses it in eval mode: softmax(h W^T) is not its distribution, so a
        scoring path that applies the bare head to hidden states must use ced/model/cp_head.py head_token_ce /
        head_logprobs_at (or lm_head_weight() for p_LM on purpose). Training mode (the trainer's loss) gets the matrix."""
        if not self.training and getattr(self, "cp_head", None) is not None:
            from ced.model.cp_head import cp_head_of, score_mode

            if cp_head_of(self) is not None and score_mode(self) == "mixture":  # (cp-aux: no mixture; cp_head_of is None)
                raise ValueError("this model has a Copy-Paste head: its next-token distribution is the mixture, not "
                                 "softmax(h W^T). Score it with ced.model.cp_head.head_token_ce / head_logprobs_at, or "
                                 "the forward's logits (log p); lm_head_weight() is p_LM's head (CED_CP_SCORE=lm)")
        return self.lm_head_weight()

    def embed_tokens(self, tokens: torch.Tensor, bag_input: str = "mean") -> torch.Tensor:
        """The embedding stage's output, the stream's input: embed_norm(embed(tokens)) for tokens (B, L), the code of every forward before
        Token Superposition Training. For bags (B, L, s) (TrainConfig.tst; ced/train/tst.py) one position is the MEAN of its s token
        embeddings (no positional encoding before the mean: the model has none there), then the same norm: embed_norm(mean_k
        embed(tokens[... k])). The mean is taken in the embedding's own dtype (float32 master weights; the embedding is never autocast),
        so in float32 in every run we train ("the bag mean computed in fp32"; the paper's Listing 2, arXiv 2605.06546 main.tex:629-650,
        sums in float32, divides by the bag size and casts back to the embedding dtype: the same mean, without the cast;
        TSTConfig.bag_input). bag_input "sum_sqrt" (TSTConfig.bag_input, the redesign's H): the SUM of the s embeddings over sqrt(s),
        summed in at least float32 whatever the table's dtype (a bf16 table is summed in float32; float32 and float64 are as they are),
        then the same norm. RMSNorm divides by the input's RMS, so for the norm that follows (model.embed_norm, true in every config we
        train) sqrt(s) x the mean is the same input up to the norm's eps (ced/eval/tst_diag.py bag_norm_eps_change): the two differ by
        the eps term alone, and by sqrt(s) when embed_norm is off. "mean" (the default) is the code above, untouched."""
        if tokens.dim() == 3:
            if bag_input not in BAG_INPUTS:
                raise ValueError(f"bag_input {bag_input!r}: one of {BAG_INPUTS}")
            if bag_input == "sum_sqrt":
                e = self.embed(tokens)
                e = e.to(torch.promote_types(e.dtype, torch.float32))  # at least float32 (a no-op for float32 and float64 tables)
                return self.embed_norm(e.sum(dim=2) / math.sqrt(tokens.shape[2]))
            return self.embed_norm(self.embed(tokens).mean(dim=2))
        return self.embed_norm(self.embed(tokens))

    def _embedded(self, tokens: torch.Tensor, bag_input: str = "mean") -> torch.Tensor:
        """embed_tokens(tokens), and with a non-default bag_input embed_tokens(tokens, bag_input): the default call is the
        one-argument call it always was (a tool that replaces embed_tokens with a one-argument function still works)."""
        return self.embed_tokens(tokens) if bag_input == "mean" else self.embed_tokens(tokens, bag_input)

    def pass2_parameter_names(self) -> list[str]:
        """The names of the later passes' own tensors (LoopConfig.pass2_gate "scalar"; [] without a gate): the only
        tensors a looped control has that its single-pass model does not (ced.config.one_pass_model): pass 2's and,
        with a 3 in the segment map, pass 3's."""
        return [n for n, _ in self.named_parameters() if n in LATER_PASS_PARAMETERS] if self.gated else []

    def alpha(self) -> torch.Tensor | None:
        """The pass-2 gate alpha of z_2 = z_1 + alpha (Block(z_1) - z_1) (0-dim, detached: a monitor), or None
        without a gate."""
        return self.hyper_connections.alpha().detach() if self.gated else None

    def recur_mix(self) -> torch.Tensor | None:
        """The scalar w of the input mix of a looped middle (loop.injection "mix": a later pass reads
        w e + (1 - w) z_{k-1}; 0-dim, detached: a monitor), or None for every other model."""
        w = getattr(self.encoder, "recur_mix", None)
        return None if w is None else w.detach()

    def recur_depth(self):
        """The reader of the depth mix at the loop boundary (loop.injection "depth_mix": R3; an attn_res.DepthQuery,
        the encoder's `recur_depth`), or None for every other model."""
        return getattr(self.encoder, "recur_depth", None)

    def recur_summary(self) -> dict | None:
        """The injection of a looped middle as it is now, for the logs (loop.RecurInjection.summary: the decay's and
        dt's min / mean / max, B's drift from its start and its size); None without a loop or without injection."""
        inject = getattr(self.encoder, "recur_injection", None)
        return None if inject is None else inject.summary()

    def alphas(self) -> list[torch.Tensor]:
        """The gate of every later pass in order (pass 2's, then pass 3's with a third pass); [] without a gate."""
        return [self.hyper_connections.alpha(k).detach() for k in range(1, self.n_passes)] if self.gated else []

    def passes(self, T: int | None) -> int:
        """The pass count of a forward: 1 without a gate (T is then ignored, as it always was); with one, T, by
        default loop.t_eval. With a looped middle (loop.recur_layers): T, any integer >= 1; where the caller gives
        none, CED_LOOP_T if it is set, else loop.t_eval."""
        if self.recur is not None:
            if T is None:
                T = env_loop_t() or self.cfg.loop.t_eval
            assert type(T) is int and T >= 1, f"T={T!r}: a pass count >= 1"
            return T
        if not self.gated:
            return 1
        T = self.cfg.loop.t_eval if T is None else T
        assert T in (1, 2), f"T={T}: the control with a second pass runs T = 1 or T = 2 (T > 2 is not built)"
        return T

    def second_pass(self, z1: torch.Tensor, ctx: SeqContext, records: list, stats: dict | None,
                    pass2_mask: torch.Tensor | None, capacity: int | None, checkpoint: bool):
        """Pass 2 of the block, and the gate (module docstring, "Two passes"): from pass 1's output z_1 (B, L, d) and
        the block layers' pass-1 records to (z, kept, n_dropped). z (B, L, d) is z_2 where pass 2 ran and z_1
        elsewhere; kept (B, L) bool marks where it ran; n_dropped (B,) counts marked positions without a slot.

        pass2_mask None: every position, nothing gathered (EncoderLayer.zero_write_all). pass2_mask (B, L) bool: the
        first `capacity` marked positions of each row (default L: all of them), gathered into slots
        (EncoderLayer.zero_write, as control (c)); an empty slot computes on zeros and is written nowhere.
        checkpoint: each layer call under activation checkpointing (a training forward without a cache)."""
        B, L, _ = z1.shape
        e2 = getattr(self, "loop_embedding_pass2", None)
        block = list(self.encoder.layers)[self.pass2_from:]
        if pass2_mask is None:
            u = z1 if e2 is None else z1 + e2
            for i, (layer, record) in enumerate(zip(block, records), start=self.pass2_from):
                if stats is not None and layer.kind == "attention":
                    record_activation_max(stats, f"pass2.enc{i}", u)
                u = act_checkpoint.run_layer(layer.zero_write_all, u, ctx, record) if checkpoint else \
                    layer.zero_write_all(u, ctx, record)
            z = self.hyper_connections(z1, u)
            kept = torch.ones(B, L, dtype=torch.bool, device=z1.device)
            n_dropped = torch.zeros(B, dtype=torch.long, device=z1.device)
        else:
            slots = select_slots(pass2_mask, L if capacity is None else capacity)
            # Off the reference GDN kernel the gathers at the slots use the slots' inverse map, so that their backward
            # is a gather too (capacity.gather_rows: the same values and gradients), as control (c) does.
            inverse = slot_inverse(slots.index, L) if self.cfg.encoder.gdn.kernel != "recurrence" else None
            slot_ctx = SlotContext(
                idx=gather_slots(ctx.idx, slots.index, fill=-1),
                doc=gather_slots(ctx.doc, slots.index, fill=-1),
                pos=gather_slots(ctx.pos, slots.index, fill=0),
                valid=slots.valid,
                inverse=inverse,
            )
            z1_slots = gather_slots(z1, slots.index) if inverse is None else gather_slots_inv(z1, slots.index, inverse)
            u = z1_slots if e2 is None else z1_slots + e2
            for i, (layer, record) in enumerate(zip(block, records), start=self.pass2_from):
                if stats is not None and layer.kind == "attention":
                    record_activation_max(stats, f"pass2.enc{i}", u, slots.valid)
                u = act_checkpoint.run_layer(layer.zero_write, u, slot_ctx, slots.index, record) if checkpoint else \
                    layer.zero_write(u, slot_ctx, slots.index, record)
            z2_slots = self.hyper_connections(z1_slots, u)
            if inverse is None:
                z = scatter_slots(z1, slots.index, z2_slots)
            else:  # the same values through the inverse map: z_2 where a slot holds the position, z_1 elsewhere
                z = torch.where(slots.kept[..., None], scatter_slots_inv(z2_slots, slots.index, inverse), z1)
            kept, n_dropped = slots.kept, slots.n_dropped
        if stats is not None:
            from ced.model.ced_v2 import loop_usage

            stats["loop_usage"] = loop_usage(z1, z, kept)
        return z, kept, n_dropped

    # -- the segment pass schedule (module docstring, "Segment-based pass counts") ------------------------------

    def _mask_share(self, block: list, cached: bool) -> str:
        """Which mask the block's attention layers share in a later pass: "flex" one BlockMask, "dense" one
        (B, slots, keys) bool mask (the reference and sdpa backends), "none" each read builds its own (no attention
        layer in the block; the rotary distance cap, whose read builds its own two; flex over a cache, whose key
        count grows). Read from the layers' own attention, which is the backend the read runs on (a scoring script
        may have switched it on the modules), not from the model config."""
        attn = next((layer.mixer for layer in block if layer.kind == "attention"), None)
        if attn is None or attn.rope_cap() is not None:
            return "none"
        if attn.cfg.backend == "flex":
            return "none" if cached else "flex"
        return "dense"

    @staticmethod
    def _pass_mask(share: str, keys, q_idx: torch.Tensor, q_doc: torch.Tensor) -> dict:
        """The mask keyword of one pass's attention reads, built ONCE for the block's attention layers: queries at
        q_idx / q_doc over the key metadata `keys` (a pass-1 KeySet of the block: every attention layer of the stack
        has the same idx, doc and ok)."""
        if share == "flex":
            from ced.model.fast_attention import slot_block_mask

            return {"block_mask": slot_block_mask(keys, q_idx, q_doc)}
        if share == "dense":
            return {"allow": slots_allowed(keys, q_idx, q_doc)}
        return {}

    def scheduled_passes(self, z1: torch.Tensor, ctx: SeqContext, records: list, stats: dict | None, plan,
                         checkpoint: bool, cached: bool) -> torch.Tensor:
        """The later passes of a segment pass schedule (module docstring) and the head's input: from pass 1's output
        z_1 (B, L, d), the block layers' pass-1 records and `plan` (a pass_schedule.PassPlan over the L positions
        of the chunk, on the device, with at least one later pass) to the state after the LAST pass each position
        ran, (B, L, d). checkpoint: each layer call under activation checkpointing (a training forward without a
        cache). cached: the forward reads or returns a cache (the mask kind, _mask_share)."""
        B, L, _ = z1.shape
        block = list(self.encoder.layers)[self.pass2_from:]
        kinds = [layer.kind for layer in block]
        keys = records[kinds.index("attention")] if "attention" in kinds else None
        share = self._mask_share(block, cached)
        # off the reference GDN kernel the gathers at the slots use the slots' inverse map (the same values and
        # gradients, a gather as the backward), as second_pass does
        with_inverse = self.cfg.encoder.gdn.kernel != "recurrence"
        states = [z1]
        # The layout the previous pass ran at: its state, its slots (None: every position of the chunk, in row
        # order) and the mask of its attention reads (None: none built yet).
        x_prev, slots_prev, mask_prev = z1, None, None
        for j in range(plan.n_later):
            k = j + 1  # the pass's 0-based index: pass k + 1
            rows = plan.rows[j]
            if plan.same[j] and rows is None:
                # the previous layout's own slots, in the same order: nothing gathered, the same masks
                x, slots = x_prev, slots_prev
                if mask_prev is None:  # pass 2 at every position: its mask is built here, once
                    at = ctx if slots is None else slots
                    mask_prev = self._pass_mask(share, keys, at.idx, at.doc)
                mask = mask_prev
            else:
                slot = plan.slot[j]
                x = pass_schedule.take(x_prev, plan.index[j], plan.inverse[j])
                slots = SlotContext(idx=pass_schedule.take_meta(ctx.idx, slot, -1),
                                    doc=pass_schedule.take_meta(ctx.doc, slot, -1),
                                    pos=pass_schedule.take_meta(ctx.pos, slot, 0), valid=plan.valid[j],
                                    inverse=plan.unit_slot[j] if with_inverse else None)
                if rows is not None:  # a generation batch in which only some rows run this pass: built for them below
                    mask = None
                elif share == "dense" and mask_prev is not None:  # rows of the previous pass's mask: no rebuild
                    mask = {"allow": pass_schedule.mask_rows(mask_prev["allow"], plan.index[j])}
                else:
                    mask = self._pass_mask(share, keys, slots.idx, slots.doc)
            e = getattr(self, f"loop_embedding_pass{k + 1}", None)
            u = x if e is None else x + e
            run_slots, run_index, run_records, run_mask = slots, plan.slot[j], records, mask
            if rows is not None:
                assert all(getattr(r, "fused_kernel", None) is None for r in records), \
                    "a pass on some rows only (PassPlan.rows) is for cached generation, not for a fused-kernel forward"
                sel = lambda t: pass_schedule.select_rows(t, rows)  # noqa: E731
                u, run_index = sel(u), sel(run_index)
                run_slots = SlotContext(idx=sel(slots.idx), doc=sel(slots.doc), pos=sel(slots.pos), valid=sel(slots.valid),
                                        inverse=sel(slots.inverse))
                run_records = [sel(r) for r in records]
                run_mask = self._pass_mask(share, sel(keys), run_slots.idx, run_slots.doc)
            for i, (layer, record) in enumerate(zip(block, run_records), start=self.pass2_from):
                attention = layer.kind == "attention"
                if stats is not None and attention:
                    record_activation_max(stats, f"pass{k + 1}.enc{i}", u, None if slots is None else run_slots.valid)
                kw = run_mask if attention else {}
                if slots is None:  # every position, in row order: the dense call (nothing gathered or scattered)
                    call, args = layer.zero_write_all, (ctx, record)
                else:
                    call, args = layer.zero_write, (run_slots, run_index, record)
                u = act_checkpoint.run_layer(call, u, *args, **kw) if checkpoint else call(u, *args, **kw)
            if rows is not None:
                u = pass_schedule.restore_rows(u, rows, B)
            z = self.hyper_connections(x, u, k)
            if stats is not None:  # loop usage of this pass, over its own slots
                from ced.model.ced_v2 import loop_usage

                with torch.no_grad():
                    on = torch.ones(B, L, dtype=torch.bool, device=z1.device) if slots is None else slots.valid
                    stats["loop_usage" if k == 1 else f"loop_usage_pass{k + 1}"] = loop_usage(x, z, on)
            states.append(z)
            x_prev, slots_prev, mask_prev = z, slots, (mask if rows is None else None)
        if all(plan.same) and all(r is None for r in plan.rows):
            return states[-1]  # every position ran every pass of the plan: the last state is the readout
        return pass_schedule.readout(states, plan)

    def scheduler(self):
        """The host-side scheduler of this model's pass schedule (ced/train/pass_positions.py PassScheduler; None
        without a schedule): the attribute pass_scheduler when set, else built from the config at first use (a
        segment map reads loop.segment_tokenizer then)."""
        if self.schedule is not None and self.pass_scheduler is None:
            from ced.train.pass_positions import PassScheduler

            self.pass_scheduler = PassScheduler.from_config(self.cfg.loop, self.cfg.vocab_size)
        return self.pass_scheduler

    def _pass_plan(self, tokens, doc_ids, where, passes, pass_counts, scored, cache, return_cache):
        """(plan on the device, the scheduler's state after this chunk or None) for a scheduled forward at T = 2
        (forward's docstring gives the order of the sources). A unit is a position of the chunk. A plan the model
        builds itself is built on the host; with a cache, rows that have no slot in a pass are left out of that pass
        (PassPlan.rows). where (B, L) bool, optional: the positions whose logits are read; the others count 1."""
        B, L = tokens.shape
        if passes is not None:
            assert pass_counts is None, "give passes or pass_counts, not both"
            assert where is None, "a caller's plan already says where the later passes run: no pass2_mask with it"
            assert passes.capacity == L and passes.readout.shape[0] == B, \
                f"the plan is for {tuple(passes.readout.shape)} positions, this forward has {(B, L)}"
            return passes, None
        streaming = cache is not None or return_cache
        state = None
        if pass_counts is None:
            scheduler = self.scheduler()
            host = lambda t: None if t is None else t.detach().cpu().numpy()  # noqa: E731
            if streaming:
                prev = cache.pass_state if cache is not None else None
                if cache is not None and prev is None and self.schedule.counts is not None:
                    raise ValueError("this cache has no segment state (its earlier chunks ran at T = 1 or were given "
                                     "pass_counts): give pass_counts for this chunk too")
                pass_counts, state = scheduler.feed(prev, host(tokens), host(doc_ids), None, host(scored))
            else:
                pass_counts = scheduler.position_counts(host(tokens), host(doc_ids), None, host(scored))
        counts = pass_counts.detach().to("cpu", torch.long)
        assert counts.shape == (B, L), f"pass_counts {tuple(counts.shape)}: one count per position, {(B, L)}"
        if where is not None:  # a position whose logits are not read runs pass 1 only
            counts = torch.where(where.detach().cpu(), counts, torch.ones_like(counts))
        assert int(counts.min()) >= 1, "a position without a pass count"
        assert int(counts.max()) <= self.n_passes, f"a pass count above the model's {self.n_passes} passes"
        sizes = pass_schedule.bucket_sizes(L, self.cfg.decoder.capacity_multiple, self.pass_buckets)
        plan = pass_schedule.build_plan(counts, sizes, row_gather=streaming)
        return plan.to(tokens.device), state

    def forward(
        self,
        tokens: torch.Tensor,
        doc_ids: torch.Tensor,
        dec_mask: torch.Tensor | None = None,
        T: int | None = None,
        capacity: int | None = None,
        cache: ControlCache | None = None,
        return_cache: bool = False,
        return_hidden: bool = False,
        varlen: dict | None = None,
        pass2_mask: torch.Tensor | None = None,
        pass2_capacity: int | None = None,
        passes=None,
        pass_counts: torch.Tensor | None = None,
        scored: torch.Tensor | None = None,
        rope_scale: int = 1,
        gdn_scale: int = 1,
        gdn_beta_power: bool = False,
        bag_input: str = "mean",
    ) -> ModelOutput:
        """Same signature as LoopedCED.forward; dec_mask and capacity are ignored, and so is T without a second pass.

        Token Superposition Training (TrainConfig.tst; ced/train/tst.py): tokens may be bags (B, L, s), one position
        the mean of s token embeddings (embed_tokens), on the one-pass stack only (no second pass, no looped middle, no
        cache). rope_scale (TSTConfig.token_scale_rope): the capped-rotary global layers rotate at rope_scale x the
        document position and keep the cap in tokens (SeqContext.rope_scale; ced/model/attention.py). 1 (the default):
        nothing is set, and the forward is the forward it was. gdn_scale (TSTConfig.gdn_resolution_gates, "E"): the GDN
        layers' gates describe one token and a position stands for gdn_scale of them: g = gdn_scale x the token's log
        decay and, with gdn_beta_power, beta = 1 - (1 - beta)^gdn_scale (SeqContext.gdn_scale; ced/model/gdn.py
        GatedDeltaNet.gates). 1 (the default): nothing is set. bag_input (TSTConfig.bag_input): how a bag becomes a
        position, "mean" (the default, embed_tokens) or "sum_sqrt".
        return_hidden: return the head input (hidden_aux) instead of logits. varlen: optional
        precomputed segment boundaries for the fused GDN kernel ({"enc": cu_seqlens}).

        With a second pass (module docstring, "Two passes"): T = 1 or 2 (default loop.t_eval). At T = 2 pass 2 runs
        at every position (pass2_mask None), or at the first pass2_capacity positions of each row that pass2_mask
        (B, L) bool marks (default: all of them; 0: none, pass 1 alone), and the head reads z_2 there and z_1
        elsewhere. kept / n_dropped then say where pass 2 ran and how many marked positions had no slot (without a
        second pass, and at T = 1, they are zeros as before). The cache is pass 1's whatever T.

        A model with a segment pass schedule (loop.segment_passes / unscored_passes; module docstring) at T = 2
        runs, at each position, the passes of one of, in this order: passes (a pass_schedule.PassPlan over the L
        positions of this forward, on the device: the trainer's, built by the data path); pass_counts ((B, L)
        integer: the passes of every position); or the model's own scheduler over the tokens (pass_scheduler), with
        `scored` ((B, L) bool, optional: where the target counts) for the unscored rule. pass2_mask then marks the
        positions whose logits are read: every other position runs pass 1 only (pass2_capacity is not used). kept
        says where a later pass ran. A model without a schedule refuses passes and pass_counts and ignores scored."""
        T = self.passes(T)
        assert self.schedule is not None or (passes is None and pass_counts is None), \
            "passes / pass_counts are for a model with a pass schedule (loop.segment_passes / unscored_passes)"
        # The segment pass schedule: the plan of this forward's later passes (None without a schedule, and at T = 1,
        # which is pass 1 alone: every call below is then the call it always was). It is built before pass 1: a
        # forward whose plan holds no later pass keeps no record.
        plan = pass_state = None
        if self.schedule is not None and T == 2:
            assert pass2_capacity is None, \
                "pass2_capacity is the dense second pass's slot count; under a pass schedule the plan sets the slots"
            plan, pass_state = self._pass_plan(tokens, doc_ids, pass2_mask, passes, pass_counts, scored, cache, return_cache)
            self.last_plan = plan
            two = plan.n_later > 0
        else:
            two = T == 2 and not (pass2_mask is not None and pass2_capacity == 0)
        ctx, seq_state = sequence_context(doc_ids, cache.seq if cache else None)
        if varlen is not None:
            ctx.cu_seqlens, ctx.cu_seqlens_cpu = varlen["enc"], varlen.get("enc_cpu")
            if varlen.get("enc_chunks") is not None:  # CUDA graphs' fixed-capacity chunk index (ced/train/cuda_graphs.py)
                ctx.gdn_chunks = varlen["enc_chunks"]
        stats = {} if self.collect_stats else None
        B, L = tokens.shape[:2]
        if tokens.dim() == 3 or rope_scale != 1 or gdn_scale != 1:  # TST's bags / token-scale rotary / E: one pass
            assert self.recur is None and not two and cache is None and not return_cache, \
                "bags of tokens (TST), rope_scale and gdn_scale run on the one-pass stack without a cache"
            if rope_scale != 1:
                ctx.rope_scale = int(rope_scale)
            if gdn_scale != 1:  # TST (E): the GDN gates of a position of gdn_scale tokens
                ctx.gdn_scale, ctx.gdn_beta_power = int(gdn_scale), bool(gdn_beta_power)
        if self.recur is not None:
            # The loop in the published form (module docstring): the prelude, T passes of the middle, the coda. Every
            # position runs every pass; pass2_mask / pass2_capacity belong to the other loop and are refused.
            assert pass2_mask is None and pass2_capacity is None, \
                "pass2_mask / pass2_capacity are the gated second pass's; a looped middle runs every pass at every position"
            assert cache is None or cache.T == T, \
                f"this cache was built at T = {cache.T}; its sequence continues at that pass count, not T = {T}"
            h, encoder_caches = self.encoder.forward_recur(
                self._embedded(tokens, bag_input), ctx, cache.encoder if cache else None, stats, T
            )
            kept = torch.zeros(B, L, dtype=torch.bool, device=tokens.device)
            n_dropped = torch.zeros(B, dtype=torch.long, device=tokens.device)
        elif not two:
            h, encoder_caches = self.encoder(
                self._embedded(tokens, bag_input), ctx, cache.encoder if cache else None, stats=stats
            )
            kept = torch.zeros(B, L, dtype=torch.bool, device=tokens.device)
            n_dropped = torch.zeros(B, dtype=torch.long, device=tokens.device)
        elif plan is not None:  # the segment pass schedule: pass 1 as it is, then each later pass at its own slots
            z1, encoder_caches, records = self.encoder.forward_keep(
                self._embedded(tokens, bag_input), ctx, cache.encoder if cache else None, stats, self.pass2_from
            )
            checkpoint = cache is None and act_checkpoint.active(self.encoder.activation_checkpoint, z1)
            h = self.scheduled_passes(z1, ctx, records, stats, plan, checkpoint, cached=cache is not None or return_cache)
            kept = plan.counts >= 2
            n_dropped = torch.zeros(B, dtype=torch.long, device=tokens.device)
        else:
            z1, encoder_caches, records = self.encoder.forward_keep(
                self._embedded(tokens, bag_input), ctx, cache.encoder if cache else None, stats, self.pass2_from
            )
            checkpoint = cache is None and act_checkpoint.active(self.encoder.activation_checkpoint, z1)
            h, kept, n_dropped = self.second_pass(z1, ctx, records, stats, pass2_mask, pass2_capacity, checkpoint)
        if stats is not None:
            self.stats = stats
        hidden = self.final_norm(h)
        copy_cache = None
        pointer = None
        # cp-aux (model.cp_head.train_mode "aux"; ced/model/cp_aux.py): the pointer is dropped at evaluation (eval mode: its inputs
        # are not formed, the logits are p_LM's, no copy cache); in training mode its inputs are formed as cp-mix's.
        aux = getattr(self, "cp_head", None) is not None and self.cp_head.cfg.train_mode == "aux"
        if getattr(self, "cp_head", None) is not None and hasattr(self.cp_head, "n_trunk"):
            # The trunk copy (model.cp_head.trunk_copy; ced/model/cp_trunk.py): the global layers' queries and keys from their key
            # sets (the queries dropped from the cache entries), the keys' tokens and starts from this chunk and the copy cache.
            from ced.model.cp_trunk import trunk_inputs

            assert tokens.dim() == 2, "the trunk copy reads token rows, not TST's bags"
            pointer, encoder_caches = trunk_inputs(self, ctx, encoder_caches, tokens, cache.copy if cache else None)
        elif getattr(self, "cp_head", None) is not None and self.cp_head.cfg.pointer_input != "final":
            # The Copy-Paste pointer's own inputs (model.cp_head.pointer_input "mix" / "shared"; ced/model/cp_head.py):
            # cp-mix's two depth mixes of the stack's sources, or cp-shared's head query and key from its layer's key set
            # (dropped from the cache entry). Under "final" (and without the head) this block is not entered.
            from ced.model.cp_head import SOURCES_KEY, pointer_inputs

            if aux and not self.training:
                ctx.memo.pop(SOURCES_KEY, None)  # cp-aux at evaluation: the stack's sources are dropped, the pointer not run
            else:
                pointer, encoder_caches = pointer_inputs(self, ctx, encoder_caches)
        if return_hidden:
            logits = None
        elif getattr(self, "cp_head", None) is not None and not aux:
            # The Copy-Paste head (ced/model/cp_head.py): the logits are log p over V of the mixture (or the LM logits
            # under score "lm"), from this chunk's keys and the cache's. Off: the line below, as before.
            from ced.model.cp_head import forward_logits

            assert tokens.dim() == 2, "the Copy-Paste head's output is for token rows, not TST's bags"
            logits, copy_cache = forward_logits(self, hidden, tokens, doc_ids, ctx.idx, cache.copy if cache else None,
                                                pointer=pointer)
        else:
            logits = lm_head(hidden, self.head_weight())
        if return_cache and copy_cache is None and getattr(self, "cp_head", None) is not None and not aux:
            raise ValueError("a cache of a Copy-Paste model is built by a forward that returns logits (prefill / extend)")
        return ModelOutput(
            logits=logits,
            hidden_aux=hidden if return_hidden else None,
            kept=kept,
            n_dropped=n_dropped,
            cache=(ControlCache(seq=seq_state, encoder=encoder_caches, T=T, pass_state=pass_state, copy=copy_cache)
                   if return_cache else None),
            cp_pointer=pointer if return_hidden else None,
        )

    def _pass2_at(self, dec_mask: torch.Tensor | None, T: int | None) -> dict:
        """prefill / extend: forward's pass-2 arguments for a decoder mask. {}: every position (and nothing to say
        without a second pass or at T = 1); else the mask, with as many slots as the fullest row has marked positions
        (0: pass 1 alone). A host read of the mask: inference only."""
        if dec_mask is None or self.recur is not None or self.passes(T) == 1:
            return {}  # (a looped middle runs every pass at every position: the mask is not read)
        if self.schedule is not None:  # the plan sets the slots; the mask says whose logits are read
            return {"pass2_mask": dec_mask}
        return {"pass2_mask": dec_mask, "pass2_capacity": int(dec_mask.sum(1).max())}

    @torch.no_grad()
    def prefill(self, tokens, doc_ids, dec_mask=None, T=None, **schedule) -> ModelOutput:
        """Teacher-forced pass over a prompt that returns pass 1's caches. With a second pass at T = 2 (default
        loop.t_eval), pass 2 runs where dec_mask marks (the positions whose next token is generated; None: every
        position) and nowhere else: read tokens get one pass. A mask that marks nothing runs pass 1 alone, so the
        read tokens of a prompt can be prefilled at the T = 1 cost and its last token extended with pass 2.
        With a segment pass schedule a marked position runs the passes of its segment and an unmarked one pass 1
        only; schedule: forward's pass_counts / scored."""
        return self.forward(tokens, doc_ids, T=T, return_cache=True, **self._pass2_at(dec_mask, T), **schedule)

    @torch.no_grad()
    def extend(self, tokens, doc_ids, dec_mask, cache: ControlCache, T: int | None = None, **schedule) -> ModelOutput:
        """The next chunk (often one token) of the same rows. With a second pass: T defaults to the cache's (the pass
        count prefill ran with), and pass 2 runs where dec_mask marks (None: every position), over pass 1's K/V and
        states of the cached positions and of this chunk; it adds nothing to the cache. schedule: forward's
        pass_counts / scored (a scheduled model; without them the counts continue from the cache's own segment
        state). With a looped middle (loop.recur_layers): T is the cache's (another T is refused), and every pass of
        the chunk reads and extends its own entries of the cache."""
        T = cache.T if T is None else T
        return self.forward(tokens, doc_ids, T=T, cache=cache, return_cache=True, **self._pass2_at(dec_mask, T), **schedule)


# Arches whose forward runs a decoder (or a second pass) at slots, the first C decoder positions of
# each row, and has two heads: return_hidden gives hidden_aux (B, L, d) at every position and
# hidden_dec (B, C, d) at the slots. The others ("control") have one head at every position.
SLOT_ARCHES = ("looped_ced", "ced_v2", "second_pass")
# Arches with a loop count T: drawn per training step from loop.t_choices, loop.t_eval at inference.
LOOPED_ARCHES = ("looped_ced", "ced_v2")


def has_loop(cfg: ModelConfig) -> bool:
    """Whether a model of this config takes a loop count T: the LOOPED_ARCHES, the control with a gated second
    pass through its top block (ced.config.looped_control) and the control with a looped middle (ced.config.recur_loop).
    A control without either ignores T, as it always did."""
    return cfg.arch in LOOPED_ARCHES or (cfg.arch == "control" and (cfg.loop.pass2_gate != "none" or bool(cfg.loop.recur_layers)))


def check_rope_cap(cfg: ModelConfig) -> None:
    """Refuse a rotary distance cap (AttentionConfig.rope_max_distance; ced/model/attention.py, "Rotary distance
    cap") wherever it is not built, so that no model ignores one it was given. Built: the control's global layers
    (encoder.attention), and CED v2's encoder global layers (encoder.attention) and decoder memory reads
    (decoder.attention, without decoder self-KV). A cap on a config without rotary dimensions is inert (NoPE)."""
    from ced.model.attention import rope_cap

    enc, dec = cfg.encoder.attention, cfg.decoder.attention
    if enc.rope_max_distance is None and dec.rope_max_distance is None:
        return
    for name, a in (("encoder", enc), ("decoder", dec)):
        try:
            rope_cap(a)  # the value itself: None, or an integer >= 1
        except AssertionError as e:
            raise ValueError(f"model.{name}.attention.{e}") from None
        if a.rope_max_distance is not None and a.normalizer != "softmax":
            raise ValueError(f"model.{name}.attention.rope_max_distance is built for the softmax normaliser, not "
                             f"{a.normalizer!r}")
    if cfg.arch not in ("control", "ced_v2"):
        raise ValueError(f"rope_max_distance is built for arch 'control' and 'ced_v2', not {cfg.arch!r}")
    if cfg.arch == "control" and dec.rope_max_distance is not None:
        raise ValueError("model.decoder.attention.rope_max_distance on arch 'control': the control builds no decoder "
                         "and would ignore it; set model.encoder.attention.rope_max_distance")
    if cfg.arch == "ced_v2" and cfg.decoder.self_kv and dec.rope_max_distance is not None:
        raise ValueError("model.decoder.attention.rope_max_distance is not built for the decoder with self-KV "
                         "(stage 1's own keys, stage 2's local layers); model.encoder.attention.rope_max_distance alone "
                         "is")


def check_row_term(cfg: ModelConfig) -> None:
    """Refuse AttentionConfig.row_term "pdp" (the CroissAttn fix: delta = rowsum(P. dP) in the attention backward; ced/kernels/attn_row_term.py)
    wherever it is not built, so that no model runs the stored-output row term under its name. Built: the encoder's global layers' own read
    (Encoder.forward without a cache), on backend "flex" (flex's backward, the compiled kernel's and the dense math fallback's alike, forms delta
    from the stored output; the pre-pass runs on CUDA, its PyTorch route elsewhere) and on backend "reference" (the autograd softmax backward is
    already that form: inert). Refused, by name: any other backend ("sdpa": the native kernels' backward is not read; "fa3" / "fa4":
    FlashAttention's preprocess forms delta from O, `dlse` is an input of FA4's backward and the same node would carry it); the decoder's
    attention (every decoder read is a slot read); the control's second pass (loop.pass2_gate: slot reads); the asentmax normaliser (the exact
    entmax reference); a head dimension the pre-pass kernel does not take. The scorers switch a loaded config's backend after the build and run
    no backward: the key is then inert."""
    from ced.kernels.attn_row_term import check_row_term as check_value, kernel_head_dim_ok

    enc, dec = cfg.encoder.attention, cfg.decoder.attention
    check_value(enc.row_term)
    check_value(dec.row_term)
    if dec.row_term != "stored_o":
        raise ValueError("model.decoder.attention.row_term 'pdp' is not built: every decoder read is a slot read over keys it did not write "
                         "(set model.encoder.attention.row_term alone)")
    if enc.row_term == "stored_o":
        return
    if enc.backend not in ("flex", "reference"):
        raise ValueError(f"model.encoder.attention.row_term 'pdp' is built for backend 'flex' (the compiled kernel's stored-output row term) "
                         f"and 'reference' (already rowsum(P . dP)), not {enc.backend!r}"
                         + (": FlashAttention-4's repair is its O residual (o_lo), which the dense interface of the pinned release does "
                            "not take" if enc.backend in ("fa3", "fa4") else ""))
    if enc.normalizer != "softmax":
        raise ValueError(f"model.encoder.attention.row_term 'pdp' is built for the softmax normaliser, not {enc.normalizer!r}")
    if cfg.loop.pass2_gate != "none":
        raise ValueError(f"model.encoder.attention.row_term 'pdp' is not built with loop.pass2_gate {cfg.loop.pass2_gate!r} (the second "
                         "pass reads through slot reads)")
    if enc.backend == "flex" and not kernel_head_dim_ok(enc.head_dim):
        raise ValueError(f"model.encoder.attention.row_term 'pdp': head_dim {enc.head_dim} is not one the pre-pass kernel takes (a power of two, "
                         "16 to 256)")


def check_kv_sharing(cfg: ModelConfig) -> None:
    """Refuse grid KV sharing (AttentionConfig.kv_sharing "grid"; ced/model/attention.py, "Grid KV sharing") wherever it is not built, so that no model
    runs gqa under its name or ignores a grid it was given. The values themselves are checked for every arch (attention.grid_spec: n_q_heads = rows x
    cols (x depth) x heads_per_cell, head_dim = row_dim + col_dim (+ depth_dim), n_kv_heads 1; third axis' keys grid_depth and depth_dim, are 0 under
    gqa and set together under the grid; the three-axis grid is built wherever the two-axis one is). Built: arch "control" with one pass, the
    encoder's global layers' own read (GatedAttention.forward with or without a cache) on the backends "reference", "flex" and "sdpa", with the
    softmax normaliser and the stored-output row term, the rotary distance cap, TST's token-scale rotary and the by-layer mix of NoPE layers,
    R1-entry. Refused, by name: the decoder's attention (every decoder read is a slot read over keys it did not write: the key set it would read holds
    components); any other arch; the control's second pass and every slot read (loop.pass2_gate), the looped middle (loop.recur_layers) and the looped
    encoder (encoder.loops); the FlashAttention backends; asentmax; the row term "pdp"; the fused projections; FP8, NVFP4 and the MX chains; the
    Copy-Paste head; the MTP drafter; Canon layers."""
    from ced.model.attention import grid_spec

    enc, dec = cfg.encoder.attention, cfg.decoder.attention
    for name, a in (("encoder", enc), ("decoder", dec)):
        try:
            grid_spec(a)  # the values: ValueError by name
        except ValueError as e:
            raise ValueError(f"model.{name}.{e}") from None
    if dec.kv_sharing != "gqa":
        raise ValueError("model.decoder.attention.kv_sharing 'grid' is not built: every decoder read is a slot read over keys it did not "
                         "write (set model.encoder.attention.kv_sharing alone)")
    if enc.kv_sharing == "gqa":
        return
    why = []
    if cfg.arch != "control":
        why.append(f"arch {cfg.arch!r}")
    if cfg.loop.pass2_gate != "none":
        why.append(f"loop.pass2_gate {cfg.loop.pass2_gate!r} (the second pass reads through slot reads)")
    if cfg.loop.recur_layers:
        why.append(f"loop.recur_layers {list(cfg.loop.recur_layers)!r}")
    if cfg.encoder.loops != 1:
        why.append(f"encoder.loops {cfg.encoder.loops!r}")
    if enc.backend not in ("reference", "flex", "sdpa"):
        why.append(f"attention backend {enc.backend!r} (built for 'reference', 'flex' and 'sdpa')")
    if enc.normalizer != "softmax":
        why.append(f"the normalizer {enc.normalizer!r}")
    if enc.row_term != "stored_o":
        why.append(f"attention.row_term {enc.row_term!r}")
    if getattr(cfg, "fused_projections", ()):
        why.append(f"model.fused_projections {list(cfg.fused_projections)!r}")
    if cfg.fp8.mode != "none" or cfg.nvfp4.mode != "none":
        why.append("model.fp8 / model.nvfp4")
    if cfg.cp_head.enabled:
        why.append("model.cp_head")
    if cfg.mtp.enabled:
        why.append("model.mtp")
    if cfg.encoder.canon_kernel:
        why.append(f"encoder.canon_kernel {cfg.encoder.canon_kernel!r}")
    if why:
        raise ValueError("model.encoder.attention.kv_sharing 'grid' is built for arch 'control' with one pass, the backends reference / "
                         "flex / sdpa, the softmax normalizer and the stored-output row term, without fused projections, FP8 / NVFP4, "
                         "the Copy-Paste head, the MTP drafter or Canon layers; not built together with: " + "; ".join(why))


def check_fa3(cfg: ModelConfig) -> None:
    """Refuse attention backends "fa3" (FlashAttention-3 varlen calls: ced/kernels/flash_capped.py) and "fa4" (the same calls
    on FlashAttention-4: ced/kernels/flash_fa4.py) wherever they are not built, so that no model runs another path under
    their names. fa3_docs_per_row (both backends' capacity) and fa4_deterministic are checked for every arch (an integer
    >= 1; a bool).

    Built: arch "control", the encoder's global layers (encoder.attention), their own read in Encoder.forward without a
    cache: plain causal (no cap, or NoPE), the rotary distance cap (rope_max_distance), the by-layer mix
    (nope_global_layers), TST's bags and token-scale rotary, any temperature (it scales q before the call), the output
    gate, and every option that only changes the stream around the layer (attn_res, layer_scalars, block_norm_gain,
    canon_kernel, FP8 / NVFP4 linears, the GDN options, the Copy-Paste head); head dimensions 8 to 256 in steps of 8;
    CUDA, bf16 or fp16 inputs (checked at the call: the kernel).
    Refused here, by name: the decoder's attention (decoder.attention.backend: every decoder read is a slot read over
    keys it did not write); any arch but "control" (CED v2's memory and decoder reads, v1's decoder, the second-pass
    arch); the control's second pass (loop.pass2_gate: zero-write slot reads); a looped middle (loop.recur_layers:
    Encoder.forward_recur); a looped encoder (encoder.loops); the asentmax normaliser (the exact entmax reference on
    every path); the MTP drafter (model.mtp: its attention block runs outside Encoder.forward); head dimensions the
    kernel does not take (FA4 on SM100 / SM110 takes fewer than on SM90: 8 to 128 or 256, checked at the call). Refused
    at the call, by name (GatedAttention._fa3, flash_capped.kernels / FA3Kernels, flash_fa4.kernels / FA4Kernels): a
    forward with a cache (prefill + steps, cached generation, chunked scoring), a slot read, the CPU, float32 inputs, a
    GPU the kernel is not built for, a missing build. On "fa4" a capped layer's near-call backward at head dimension 256 on
    SM100 / SM110 (FA4 4.0.0b35 asserts it out) is refused by name when the Trainer is built on such a GPU
    (check_fa4_device) and at the call (flash_fa4.window_backward_refusal); this check knows no device. The scorers switch
    a loaded config's backend (an implementation field) and run their own paths."""
    from ced.model.attention import FLASH_BACKENDS

    enc, dec = cfg.encoder.attention, cfg.decoder.attention
    for name, a in (("encoder", enc), ("decoder", dec)):
        D = a.fa3_docs_per_row
        if isinstance(D, bool) or not isinstance(D, int) or D < 1:
            raise ValueError(f"model.{name}.attention.fa3_docs_per_row {D!r}: an integer >= 1 (segments a row of the "
                             "micro-batch: the capacity of backend 'fa3''s varlen metadata)")
        if not isinstance(a.fa4_deterministic, bool):
            raise ValueError(f"model.{name}.attention.fa4_deterministic {a.fa4_deterministic!r}: true or false (backend "
                             "'fa4''s deterministic backward)")
    if dec.backend in FLASH_BACKENDS:
        raise ValueError(f"model.decoder.attention.backend {dec.backend!r} is not built: every decoder read is a slot read "
                         "over keys it did not write (CED v2's memory, v1's decoder), and the control builds no decoder")
    if enc.backend not in FLASH_BACKENDS:
        return
    raise ValueError(f"model.encoder.attention.backend {enc.backend!r} is not part of this code export (the FlashAttention-3 and -4 "
                     "kernels are left out); use backend 'flex', 'sdpa' or 'reference'")

def check_fa4_device(cfg: ModelConfig, capability: tuple[int, int] | None, row_positions: int,
                     rope_scales: tuple[int, ...] = (1,)) -> None:
    """Refuse, by name and before any step, attention backend "fa4" training on a GPU where the pinned FlashAttention-4 lacks a
    backward the model will ask for: a capped global layer's near call (a causal window of rope_max_distance keys,
    flash_fa4.window_backward_refusal) at head dimension 256 on SM100 / SM110 (FA4 4.0.0b35). Not refused at a pin that runs
    it (flash_fa4.SM100_HD256_WINDOW_BWD: FA4 main 965c9212, the A build of 2026-10-19). The Trainer calls it with its
    device's capability (None off CUDA: nothing to refuse), the longest row in positions (data.seq_len: a shape schedule's
    phases are divisors of it, and TST's rows keep it) and the rope scales its forwards run (TST's token-scale rotary: s of
    each bag phase, the near window then cap // s positions). A row no longer than the window + 1 never runs the near call
    (the capped read is the plain read there), so it is not refused; nor is a NoPE layer, any other backend, head dimension
    or GPU. The call itself refuses the same backward again (FA4Kernels.bwd), for every caller outside the Trainer."""
    # Public export: backend 'fa4' is refused at the model build (check_attention_backends), so there is nothing to check here.
    return


def check_nope_global_layers(cfg: ModelConfig) -> None:
    """Refuse the by-layer mix of position schemes (EncoderConfig.nope_global_layers) wherever it is not built, so that
    no model ignores one it was given. The indices themselves are checked for every arch (EncoderConfig.nope_layers).
    Built: arch "control" whose encoder.attention has a rotary distance cap (rope_max_distance on a config with rotary
    dimensions). Refused: any other arch (CED v2's decoder memory, the second pass and v1's decoder read keys an
    encoder layer's config no longer describes alone); the key without a cap (that would be plain rotary under NoPE
    layers, which nobody asked for and nothing checks; with rope_fraction 0.0 every global layer is NoPE already); the
    control with a gated second pass through its top block (loop.pass2_gate, so loop.pass2_layers and a segment pass
    schedule too). Why: the two were built on separate branches and nothing tests them together. One known fault: a
    scheduled pass shares one mask among the block's attention layers, chosen from the block's FIRST attention layer
    (HybridLM._mask_share); with a NoPE layer below a capped one in the block, the capped layer is handed a mask its
    read refuses (GatedAttention.read: "a caller-built dense mask is for the plain softmax read"). so the pair is
    refused for every split, not built."""
    from ced.model.attention import rope_cap

    nope = cfg.encoder.nope_layers()  # ValueError on a bad list
    if not nope:
        return
    if cfg.arch != "control":
        raise ValueError(f"model.encoder.nope_global_layers is built for arch 'control', not {cfg.arch!r}")
    if cfg.loop.pass2_gate != "none":
        raise ValueError("model.encoder.nope_global_layers is not built together with the control's second pass "
                         f"(loop.pass2_gate {cfg.loop.pass2_gate!r}; loop.pass2_layers, loop.segment_passes and "
                         "loop.unscored_passes need that gate): the later passes' shared mask is chosen for one "
                         "position scheme. Leave the gate 'none' or the split empty")
    a = cfg.encoder.attention
    if rope_cap(a) is None:
        raise ValueError("model.encoder.nope_global_layers mixes capped rotary layers with NoPE layers: it needs "
                         "model.encoder.attention.rope_max_distance on a config with rotary dimensions (got "
                         f"rope_max_distance {a.rope_max_distance!r}, rope_fraction {a.rope_fraction!r}). Plain rotary "
                         "under NoPE layers is not built; with rope_fraction 0.0 every global layer is NoPE already")


def check_layer_scalars(cfg: ModelConfig) -> None:
    """Refuse the per-layer residual scalars (EncoderConfig.layer_scalars: x <- a x + b x0 + f(x) in every layer of the
    stack, in the form "skip_x0", "skip_x0_span" or "x0"; ced/model/encoder.py layer_scalar_terms and
    apply_layer_scalars) wherever they are not built, so that no model ignores the key. Built: arch "control" with
    one pass. Refused, each because nothing tests it and a code path would skip the scalars: any other arch (CED v2's
    and v1's decoders and the second-pass control walk layers of their own); the control's second pass
    (loop.pass2_gate: the block's zero-write calls, EncoderLayer.zero_write / zero_write_all, apply no scalar); a
    looped encoder (x0 would be each loop's input); NVFP4 fake quantisation (the two were built apart). The span
    bound's width (EncoderConfig.layer_scalars_span) is checked for every arch: a finite number above 1, and the
    default wherever the form does not read it, so that no config names a bound its model does not have."""
    from ced.model.encoder import LAYER_SCALARS

    mode = cfg.encoder.layer_scalars
    if mode not in LAYER_SCALARS:
        raise ValueError(f"model.encoder.layer_scalars {mode!r}: one of {LAYER_SCALARS}")
    span = cfg.encoder.layer_scalars_span
    if type(span) not in (int, float) or not (math.isfinite(span) and span > 1.0):
        raise ValueError(f"model.encoder.layer_scalars_span {span!r}: the width R of the span bound, a finite number "
                         "above 1 (every product of a over consecutive layers stays inside (1/R, R))")
    if mode != "skip_x0_span" and span != EncoderConfig().layer_scalars_span:
        raise ValueError(f"model.encoder.layer_scalars_span {span!r} is read by layer_scalars 'skip_x0_span' only; with "
                         f"layer_scalars {mode!r} nothing is bounded: leave it at {EncoderConfig().layer_scalars_span}")
    if mode == "none":
        return
    if cfg.arch != "control":
        raise ValueError(f"model.encoder.layer_scalars is built for arch 'control', not {cfg.arch!r}")
    if cfg.loop.pass2_gate != "none":
        raise ValueError("model.encoder.layer_scalars is not built together with the control's second pass "
                         f"(loop.pass2_gate {cfg.loop.pass2_gate!r}): pass 2's zero-write layer calls apply no scalar. "
                         "Leave the gate 'none' or layer_scalars 'none'")
    if cfg.encoder.loops != 1:
        raise ValueError(f"model.encoder.layer_scalars is not built for a looped encoder (encoder.loops {cfg.encoder.loops})")
    if cfg.nvfp4.mode != "none":
        raise ValueError(f"model.encoder.layer_scalars is not built together with model.nvfp4 (mode {cfg.nvfp4.mode!r}): "
                         "nothing tests the pair")


def check_block_norm_gain(cfg: ModelConfig) -> None:
    """Refuse the scalar gain of the norms inside the layers (EncoderConfig.block_norm_gain "scalar": the four Peri-LN
    norms and the GDN mixer's gated output norm; ced/model/norms.py RMSNorm, ced/model/encoder.py EncoderLayer)
    wherever it is not tested, so that no config names a model that was not built as named. The value is checked for
    every arch: "channel" (the default) or "scalar". Built and tested: arch "control" with one pass, in the three
    residual forms (the base, R1 "block" and R1-entry "block_entry": both read the layer's own norm modules,
    ced/model/attn_res.py) and with NVFP4 or FP8 on. Refused, because nothing tests the pair: any other arch, a looped
    encoder, the control's second pass (loop.pass2_gate), a looped middle (loop.recur_layers), the per-layer residual
    scalars and Canon layers."""
    gain = cfg.encoder.block_norm_gain
    if gain not in ("channel", "scalar"):
        raise ValueError(f"model.encoder.block_norm_gain {gain!r}: 'channel' (a per-channel gain per norm, the default) "
                         "or 'scalar' (one gain per norm)")
    if gain == "channel":
        return
    if cfg.arch != "control" or cfg.encoder.loops != 1:
        raise ValueError(f"model.encoder.block_norm_gain 'scalar' is built for arch 'control' with encoder.loops 1, not "
                         f"arch {cfg.arch!r} with encoder.loops {cfg.encoder.loops}")
    others = {"loop.pass2_gate": (cfg.loop.pass2_gate, "none"), "loop.recur_layers": (tuple(cfg.loop.recur_layers), ()),
              "model.encoder.layer_scalars": (cfg.encoder.layer_scalars, "none"),
              "model.encoder.canon_kernel": (cfg.encoder.canon_kernel, 0)}
    for key, (value, off) in others.items():
        if value != off:
            raise ValueError(f"model.encoder.block_norm_gain 'scalar' is not built together with {key} {value!r}: "
                             "nothing tests the pair")


def check_canon(cfg: ModelConfig) -> None:
    """Refuse Canon layers (EncoderConfig.canon_kernel; ced/model/canon.py) wherever they are not built, so that no model
    ignores a kernel it was given. The value is checked for every arch: 0 (no Canon layer) or an integer >= 2. Built:
    arch "control" with at least one global attention layer and one pass, the softmax normaliser, any position scheme
    of its global layers (rotary, capped rotary, NoPE, the by-layer mix) and any temperature. Refused, by name:
    - any other arch: CED v2's memory, the second-pass arch and v1's decoder read an encoder layer's keys or redo its
      projections outside EncoderLayer.forward, where the Canon layer runs, and the decoder's own Canon layers wait
      with stage 2;
    - the control with a gated second pass through its top block (loop.pass2_gate): the later passes' zero-write
      reads form their queries without the Canon layer, and nothing defines what its history is in a pass that writes
      nothing;
    - an encoder without a global layer (full_attn_interval 0): the key would add nothing;
    - the asentmax normaliser (its scale heads read the attention input), NVFP4 fake quantisation and the residual
      scalars (encoder.layer_scalars, merged into main beside this key): nothing tests them together with a Canon
      layer.
    The start of the taps (EncoderConfig.canon_init) is checked for every arch too: "zeros" or "uniform", any other
    value is refused, and "uniform" without a kernel is refused, so no config says its taps start uniform when it has
    none."""
    from ced.model.canon import CANON_INITS

    K = cfg.encoder.canon_kernel
    if isinstance(K, bool) or not isinstance(K, int) or K < 0 or K == 1:
        raise ValueError(f"model.encoder.canon_kernel {K!r}: 0 (no Canon layer) or an integer >= 2 (the kernel size)")
    init = cfg.encoder.canon_init
    if not isinstance(init, str) or init not in CANON_INITS:
        raise ValueError(f"model.encoder.canon_init {init!r}: one of {CANON_INITS} (the start of the Canon taps)")
    if K == 0:
        if init != "zeros":
            raise ValueError(f"model.encoder.canon_init {init!r} is the start of the Canon taps, and this model has no Canon "
                             "layer (model.encoder.canon_kernel 0): leave it 'zeros' or set the kernel")
        return
    if cfg.arch != "control":
        raise ValueError(f"model.encoder.canon_kernel is built for arch 'control', not {cfg.arch!r}")
    if cfg.loop.pass2_gate != "none":
        raise ValueError("model.encoder.canon_kernel is not built together with the control's second pass "
                         f"(loop.pass2_gate {cfg.loop.pass2_gate!r}): the later passes' reads do not run a Canon layer. "
                         "Leave the gate 'none' or the kernel 0")
    if not cfg.encoder.global_layers():
        raise ValueError("model.encoder.canon_kernel puts a Canon layer on every global attention layer, and this "
                         f"encoder has none (full_attn_interval {cfg.encoder.full_attn_interval!r})")
    if cfg.encoder.attention.normalizer != "softmax":
        raise ValueError("model.encoder.canon_kernel is built for the softmax normaliser, not "
                         f"{cfg.encoder.attention.normalizer!r}: nothing tests them together")
    if cfg.nvfp4.mode != "none":
        raise ValueError(f"model.encoder.canon_kernel is not built together with model.nvfp4.mode {cfg.nvfp4.mode!r}: "
                         "nothing tests them together")
    if cfg.encoder.layer_scalars != "none":
        raise ValueError("model.encoder.canon_kernel is not built together with model.encoder.layer_scalars "
                         f"{cfg.encoder.layer_scalars!r}: nothing tests them together")


def build_model(cfg: ModelConfig) -> nn.Module:
    from ced.model.looped_ced import LoopedCED

    if cfg.decoder.self_kv and cfg.arch != "ced_v2":
        # Decoder self-KV is built in CED v2 only; any other model would ignore the flag and train
        # without it under a self-KV label.
        raise ValueError(f"decoder.self_kv is a CED v2 option, not for arch {cfg.arch!r}")
    if cfg.decoder.local_temperature != "inherit" and "local" not in cfg.decoder.read_kinds():
        # The local layers' temperature (stage 2 + ssmax) set on a model without local layers would be ignored.
        raise ValueError(f"decoder.local_temperature {cfg.decoder.local_temperature!r} is the temperature of local "
                         f"decoder layers; this model has none (arch {cfg.arch!r}, read_pattern "
                         f"{cfg.decoder.read_pattern!r}): leave it 'inherit'")
    if "local" in cfg.decoder.read_kinds() and cfg.arch != "ced_v2":
        # The local read pattern likewise exists in CED v2 only.
        raise ValueError(f"decoder.read_pattern {cfg.decoder.read_pattern!r} (local layers) is a CED v2 option, "
                         f"not for arch {cfg.arch!r}")
    if cfg.loop.pass2_gate != "none" and cfg.arch not in ("ced_v2", "control"):
        # The gated second pass is built in CED v2's decoder and through the control's top block (the loop on
        # the plain hybrid; HybridLM, "Two passes").
        raise ValueError(f"loop.pass2_gate {cfg.loop.pass2_gate!r} is built for arch 'ced_v2' and 'control', not for "
                         f"arch {cfg.arch!r}")
    check_looped_control(cfg)  # loop.pass2_layers, and the control's own conditions on a gate
    check_recur_loop(cfg)  # loop.recur_layers (the loop in the published form) and loop.t_sampling
    check_rope_cap(cfg)
    check_fa3(cfg)  # attention backends "fa3" / "fa4" (ced/kernels/flash_capped.py, flash_fa4.py): refused where not built
    check_row_term(cfg)  # attention.row_term "pdp" (ced/kernels/attn_row_term.py): refused where not built
    check_kv_sharing(cfg)  # attention.kv_sharing "grid" (ced/model/attention.py): refused where not built
    check_nope_global_layers(cfg)
    check_layer_scalars(cfg)
    check_canon(cfg)
    check_block_norm_gain(cfg)  # model.encoder.block_norm_gain: one gain per norm inside the layers
    check_attn_res(cfg)  # model.encoder.attn_res (Block Attention Residuals; ced/model/attn_res.py)
    from ced.model.gdn_arms import check_gdn_arms

    check_gdn_arms(cfg)  # model.encoder.gdn_write_mix / gdn_ccr_spacing (the GDN-side arms; ced/model/gdn_arms.py)
    from ced.model.gdn_erase import check_model as check_gdn_erase

    check_gdn_erase(cfg)  # model.encoder.gdn.erase / decoder.gdn.erase (EDA's erase direction; ced/model/gdn_erase.py)
    from ced.model.gdn_dprod import check_config as check_gdn_dprod

    check_gdn_dprod(cfg)  # model.encoder.gdn.dprod / decoder.gdn.dprod (Gated DeltaProduct; ced/model/gdn_dprod.py)
    from ced.model.mtp import attach_mtp, check_mtp

    check_mtp(cfg)  # model.mtp (the frozen-trunk MTP drafter; ced/model/mtp.py): its values, and refused where not built
    from ced.model.cp_head import check_cp_head

    check_cp_head(cfg)  # model.cp_head (the Copy-Paste head; ced/model/cp_head.py): its values, refused where not built
    from ced.model.fp8 import apply_fp8, check_fp8

    check_fp8(cfg)  # model.fp8: its values for every arch; with the switch on, refused where it is not built
    from ced.model import fused_proj

    fused_proj.check_config(cfg)  # model.fused_projections (other.concat): refused where it is not built; () checks nothing
    if cfg.fp8.mode != "none":
        # Opt-in FP8 for the layers' linears (FP8Config; ced/model/fp8.py). Off: this branch is not entered and no module
        # is touched. On: the control's layer linears become FP8Linear in place, after the model is built and initialised
        # (the same parameters, names, state dict and RNG use).
        model = HybridLM(cfg)
        apply_fp8(model, cfg.fp8)
        if fused_proj.names_of(cfg):  # one GEMM a shared input (ced/model/fused_proj.py); () (the default): not entered
            fused_proj.apply(model, fused_proj.names_of(cfg))
        return model
    if cfg.nvfp4.mode == "none" and getattr(cfg.nvfp4, "fp8_names", ()):
        # NVFP4Config.fp8_names names linears that run FP8 instead of NVFP4: without NVFP4 nothing would read it (the
        # FP8 control's precision is model.fp8). Off (the default ()): not entered.
        raise ValueError(f"model.nvfp4.fp8_names {list(cfg.nvfp4.fp8_names)!r} is read only with model.nvfp4.mode other "
                         "than 'none' (for FP8 on every linear use model.fp8)")
    if cfg.nvfp4.mode == "none" and getattr(cfg.nvfp4, "until_step", 0):
        # NVFP4Config.until_step ends the NVFP4 phase at an update: without NVFP4 there is no phase (the default 0 is not entered).
        raise ValueError(f"model.nvfp4.until_step {cfg.nvfp4.until_step!r} is read only with model.nvfp4.mode other than 'none'")
    if cfg.nvfp4.mode != "none":
        # Opt-in NVFP4 fake quantisation (NVFP4Config; ced/model/nvfp4.py). Off: this branch is not entered and no
        # module is touched. On: the control's layer linears become NVFP4Linear in place, after the model is built
        # and initialised (the same parameters, names, state dict and RNG use); refused where it is not built.
        from ced.model.nvfp4 import apply_nvfp4, check_nvfp4

        check_nvfp4(cfg)
        model = HybridLM(cfg)
        apply_nvfp4(model, cfg.nvfp4)
        return model
    if cfg.arch == "looped_ced":
        return LoopedCED(cfg)
    if cfg.arch == "control":
        if fused_proj.names_of(cfg):  # one GEMM a shared input (ced/model/fused_proj.py); () (the default): not entered
            model = attach_mtp(HybridLM(cfg)) if cfg.mtp.enabled else HybridLM(cfg)
            fused_proj.apply(model, fused_proj.names_of(cfg))
            return model
        if cfg.mtp.enabled:
            # The MTP drafter (ced/model/mtp.py): the trunk built and initialised exactly as without the key, then the
            # head as `model.mtp`, drawn after it. Off: this branch is not entered.
            return attach_mtp(HybridLM(cfg))
        return HybridLM(cfg)
    if cfg.arch == "ced_v2":
        from ced.model.ced_v2 import CEDv2

        return CEDv2(cfg)
    if cfg.arch == "second_pass":
        from ced.model.second_pass import SecondPassLM

        return SecondPassLM(cfg)
    raise ValueError(f"unknown arch {cfg.arch!r}")
