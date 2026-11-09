"""A resumable training loop, CPU (tests) and CUDA (the cluster runs).

Resumability is the point: preemptible machines and cluster runs can stop at any step, so a run must be able to
continue with exactly the same results. Everything that affects the trajectory is either saved in
the checkpoint (model, optimizer, step, RNG states, W&B run id, the data identity) or a pure
function of (seed, step) (the data batch and the sampled loop count T). The data pipeline has no
other state: `ShardPacker.batch(step)` is computed from the step alone, and the checkpoint records
the shard set's fingerprint and the data seed, which a resume checks. With data.extra_shard_dirs the
shard set is the union of two or more sets (ced/data/shard_format.py ShardUnion; its fingerprint
covers every set and their order), read by the same packer.

Checkpoints are written every `ckpt_every_steps` steps, every `ckpt_every_minutes` minutes, and
immediately when SIGTERM arrives (the eviction notice), always atomically (write, then rename).

The loss is computed from the models' hidden states with the LM head (model.head_weight(): the
tied input embedding in v1, the separate output embedding in v2) in chunks
(ced/train/chunked_loss.py), so no full (B, L, V) logits exist. A step can be split into
micro-batches; the loss of each is normalised by the whole batch's counts, which are known on the
host from the batch itself, so the accumulated gradient equals the one-batch gradient.

Models with decoder slots (Looped-CED v1, CED v2, the second-pass control) have two heads:

    loss = sum_dec obj / N + w_aux * sum_aux obj / N            normalization "global_token_mean"
    loss = sum_dec obj / n_dec + w_aux * sum_aux obj / n_aux    normalization "per_head_mean" (v2)
    obj  = CE + z_loss * lse^2

where "dec" is the main head at the valid filled slots, "aux" the aux head at the aux positions
(v1: the valid positions that do not run the decoder; v2: a random 50% of the valid targets,
LossConfig.aux_positions), and N the number of valid targets. The control has one head:
sum over valid targets of obj / N. Decoder and aux positions are set host-side per batch
(ced/train/positions.py).

The loop experiment (phase A / phase B). A gated two-pass CED v2 (model.loop.pass2_gate; ced/model/ced_v2.py, "Two
passes with self-KV") is its single-pass model plus pass 2's own tensors (the gate or pass 2's hyper-connection
scalars, and pass 2's loop-index embedding), which start as the identity. What the trainer adds for it, all off by
default:
- pass 2's tensors are in AdamW groups of their own at optim.pass2_lr_scale x the run's LR (ced/train/optim.py);
- growing a T = 1 checkpoint to two passes: init_from (the weights; a fresh optimizer) and, inside one pretraining
  run, loop_switch_step S with resume_from = the T = 1 trunk's kept checkpoint of step S (the trunk's whole training
  state). A run with a switch step trains one pass for its first S updates, where pass 2's tensors get no gradient,
  no weight decay and no optimizer state, so those S steps are the trunk run alone bit for bit (trunk_config), and
  the branch from the trunk's checkpoint is the run from its own start bit for bit;
- one LR schedule against planned wall-clock (optim.schedule_steps; schedule_clock): a T = 2 step is slower, so a
  two-pass run reads the schedule of its T = 1 arm (or trunk) at equal fractions of the planned wall-clock;
- "alpha", "lr_pass2" and "grad_norm_pass2" in the metrics of every step, and pass 2's tensors ("pass2") every
  log_every steps.

Segment-based pass counts (model.loop.segment_passes / unscored_passes; quoted in ced/model/pass_schedule.py;
the model side is ced/model/ced_v2.py, "Segment-based pass counts"). Off by default. With a schedule, from the
switch step on (from step 0 without one):
- batch_at gives every position of the batch its pass count on the CPU (Batch.pass_counts): the count of its
  segment, read by the state machine of ced/train/pass_positions.py from the tokens and the shards' assistant
  flags, and 1 at an unscored decoder slot with unscored_passes;
- per micro-batch the plan of the later passes is built on the CPU with fixed slot sizes (pass_slot_buckets) and
  goes to the model with the tensors of the micro-batch (`passes`): no shape of the step depends on the data beyond
  the choice among those sizes;
- the loss is the same code: the main head reads, per slot, the last pass it ran;
- "passes_mean" (the mean pass count over the batch's decoder positions), "pass_slots" ({pass: [filled,
  allocated]} over the micro-batches) and "pass_fill" in the metrics of every step; with a third pass "alpha_pass3".
The control with a second pass through its top block (arch "control"; ced/model/control.py, "Segment-based pass
counts") takes the same schedule through the same code: a unit is then every position of a row (the later passes'
slot sizes are fractions of L, and "passes_mean" is the mean over all positions), the plan is
PassPlanner.plan_positions, and with unscored_passes the unscored positions are those without a valid target.
A map with a 3 adds pass 3's own tensors, which are pass 2's in every respect here: the same optimizer groups, new
with pass 2's when a T = 1 checkpoint or trunk grows. Before the switch step a scheduled run is its T = 1 trunk bit
for bit, as any gated run. The cumulative "flops" of a scheduled run are counted at the dense loop count.

A batch-shape schedule (TrainConfig.shape_schedule; ced/train/shape_schedule.py; batch warmup and short-to-long
rows). Off by default, and then nothing below runs. With one, a step reads the rows and the row length of its phase
from the same token stream in the same order (ShardPacker.rows), in micro-batches of the config's tokens; cfg.steps
stays the run's length in reference steps (the config's own B x L tokens each) and the run ends after
Trainer.total_steps optimizer steps, when it has read cfg.steps x B x L tokens. Read at tokens seen: the LR (lr_at:
the config's schedule at the step's clock), the periodic evals (every eval.every_steps reference steps, on the
config's own row length as always) and the FLOPs. Left counting optimizer steps: the optimizer's own state, weight
decay, log_every, ckpt_every_steps, checkpoint_steps, --until-step and the checkpoint files' names. Every metrics
record of a scheduled run carries "tokens_seen", "ref_step", "phase", "rows" and "seq_len". TrainConfig.shape_lr
"rows" (the default "clock" is the sentence above) scales that LR with the step's tokens: the LR of the reference
step the step reads, divided by the phase's optimizer steps to a reference step, in every parameter group (lr_at;
ced/train/shape_schedule.py, "The LR by rows"). Refused without a schedule; a checkpoint is continued under the value
it was trained with (load_state_dict).

The loop on the plain hybrid: the control with a gated second pass through its top block (model.loop.pass2_gate
"scalar" with pass2_layers on arch "control"; ced/model/control.py, "Two passes") is trained by the same code: its two
new tensors carry the stage 2 build's names, its single-pass model is ced.config.one_pass_model's (the control without
the gate and the block), and it takes a loop count like the looped arches (ced.model.control.has_loop). Everything
above applies to it unchanged; compile "all" also compiles the block layers' pass-2 calls (compile_layers).

The per-layer residual scalars (model.encoder.layer_scalars "skip_x0"; ced/model/encoder.py apply_layer_scalars). Off by
default: nothing below is logged and the step is the step it was. With them: "grad_norm_layer_scalars" in the metrics of
every step (the norm of the two vectors' gradient before clipping; they are inside the global clip, so a scalar gradient
that dominates the clipped norm shows), and every log_every steps "layer_scalars": {"a": [a_i], "b": [b_i]} per layer,
bottom first, after that step's update. The scalar terms are part of each layer's own residual add (EncoderLayer.forward),
so with compile "all" they are inside the layers' compiled graphs, and the gradient norm above is read on the host after
the optimizer step, with the step's other metrics. The form "skip" (a only; no encoder.x0_gain) logs the same two keys, b
= 0.0 in every layer; with optim.layer_scalar_a_lr_scale other than 1 the metrics also hold "lr_layer_scalar_a", the LR of
encoder.skip_gain's group. The forms "skip_x0_span" and "x0" (ced/model/encoder.py layer_scalar_terms) log the same two
keys: "layer_scalars" holds the EFFECTIVE a and b of x <- a x + b x0 + f(x) (for the span form a_i = exp(S_{i+1} - S_i),
not the stored u; a = 1.0 in every layer of "x0"; b of layer 0 is 0.0 in both), and "grad_norm_layer_scalars" the gradient
norm of the tensors the form stores (x0_gain alone for "x0"). After every optimizer step the span form's stored u is put
back into [-3, 3] (Encoder.project_layer_scalars: a clamp in place, part of that form's training; the other forms and the
model without the scalars are not touched). Likewise, with model.encoder.gdn_ccr_gate_floor f > 0 (CCR's gate floor;
ced/model/gdn_arms.py floor_gates_), every CCR reader's gate is put back to at least f after every optimizer step (a clamp
in place, no host read); the trainer refuses the floor unless optim.zero_gate_decay is "none". Without a floor nothing
runs.

The stream's size by layer (TrainConfig.log_stream_rms_every N). Off by default (0): nothing below runs. With N > 0, on
every N-th optimizer step the metrics hold "stream_rms": n_layers + 1 numbers, the root mean square over tokens and
dimensions of the stream entering each encoder layer (entry 0: the embedding output) and of the stream after the last
layer, before the final norm, on the step's first micro-batch. It is read under no_grad from the forward the step runs
anyway (Encoder._stack calls record_stream_rms between the layer calls, outside any compiled layer and any checkpointed
call; under activation_checkpoint "block" inside a block's first forward only, never in its recomputation), not from a
second forward: the stack exposes every boundary there, a second forward would cost one more forward of a micro-batch and
would run the layers' kernels once more. Nothing of the step changes: the read is on the detached stream, writes to no
tensor the step uses and draws no random number, so the losses, the gradients, the optimizer's state, the RNG state and
the checkpoints' tensors are those of the run without it, bit for bit (tests/test_resid_forms.py; only the config in the
checkpoint differs, by the key). For arch "control" with one pass through the stack, with or without layer scalars;
refused elsewhere, where the stack runs through another function and nothing would be logged. Under Block AttnRes
(model.encoder.attn_res "block"; ced/model/attn_res.py BlockAttnRes.stack) there is no running stream: the n_layers + 1
numbers are the RMS of what each layer's MIXER reads (its depth attention's output; entry 0 is the embedding output, as
above) and of the stream after the last sublayer, which the final norm reads. Such a run's metrics also hold, every
log_every steps, "attn_res": {"query_norm": [||w|| per reader, in running order, the output's last], "key_gain_dev": [max
|gain - 1| of each reader's key norm]} after that step's update (BlockAttnRes.summary: a pseudo-query's norm is how far
it has moved from zero, where its weights are equal). Read from the parameters alone. Under R1-entry
(model.encoder.attn_res "block_entry") a block has a running stream of its own: the n_layers + 1 numbers are the RMS of
the stream entering each layer (in a block's first layer the block's entry read; entry 0 the embedding output) and of the
output's read; "attn_res" then holds n_blocks values per key (the blocks' entries, the output's last).

qmix's static top-2 query mix (model.encoder.gdn_qmix_select_step S; ced/model/gdn_arms.py "qmix with a static top-2 query mix"). Off by
default (0): nothing below runs. With it, at the start of the first step >= S, once, train_step runs ONE extra forward of that step's first
micro-batch (the training forward's own call, discarded; Trainer._qmix_choose) with the encoder's qmix_record set, so that every query mix's
token-sums of weights are recorded (attn_res._monitor, an eager read under no_grad outside the compiled layers); each layer with more than 2
sources then takes the 2 with the largest token-mean weight (ties to the lower source) into its state-dict buffer qmix_keep, and its query
mix is the per-token softmax over those 2 from then on. The metrics hold "qmix_selection" (the step, the layers, their pairs, the token-means
each was chosen on) at that step, and, on every log_stream_rms_every-th step, "qmix_weights" (each layer's token-mean weight over every
source, as the layer used it: the full softmax before the choice, the pair's after). Refused with data parallelism, eval.readout_every and
activation_checkpoint host /host_overlap.

The loop in the published form (model.loop.recur_layers on arch "control"; ced/model/control.py). Off by default:
nothing below runs and a step is the step it was. With it the model takes a pass count like the looped arches
(has_loop), and:
- model.loop.t_sampling "step" (the default): one T per optimizer step, drawn as always (loop_count);
- "sequence": one T per ROW. Every step's rows hold exactly t_probs[i] x data.batch_size rows of t_choices[i] (the
  multiset; refused unless each product is a whole number), in an order drawn from the counter RNG of (seed, step)
  (row_passes: a pure function of the step, so a resume needs only the step). The rows and their order are the
  data's as always; only which rows share a forward changes: the step's micro-batches are formed from rows of equal
  T (micro_plan), so every forward has one T and a fixed shape, and nothing is masked or gathered on the device;
- micro_batch_by_t: the rows per forward by pass count (a step at more passes keeps more activations).
The loss of every micro-batch is normalised by the whole batch's counts, as always, so the accumulated gradient is the
same sum over the step's rows whatever the grouping (tests/test_loop_published.py). Metrics of every step: "T" (the
mean over the rows under "sequence"), with "sequence" also "T_rows"; with injection "grad_norm_injection", and every
log_every steps "recur" (the decay's and dt's min / mean / max, B's drift from its start and its size) and "loop_usage" (of the
step's first forward when it runs two passes or more; with a pass count per row the first forward holds the one-pass
rows, so it is not logged). The cumulative "flops" are counted at the mean pass count of t_choices / t_probs (the
count is linear in T).

The late switch-on of a looped middle ("growth"). A looped middle with the scalar input mix (model.loop.injection
"mix"; ced/model/encoder.py recur_input_mix) takes the switch-on the gated second pass has, through the same code
(loop_switch_step S, resume_from, trunk_config, optim.schedule_steps), and no other form of the looped middle does:
- before S every step runs ONE pass (loop_count), which is the plain control's forward bit for bit, and the one new
  tensor, encoder.recur_mix (w), gets no gradient, no weight decay and no optimizer state: those S steps are the
  trunk run alone (trunk_config: the plain control, ced.config.one_pass_model);
- resume_from = the trunk's kept checkpoint of step S continues the trunk's whole training state (weights, Muon and
  AdamW state, RNG states, the data position), with w new at its configured start and without optimizer state; the
  run is then, bit for bit, the run its config trains from its own start;
- unlike a gate at 0, the mix at its start is NOT the trunk's function: at w = 1 the model at two passes is the
  trunk exactly, at the configured 0.99 it is near it. Nothing is refused for that; the record of the resume carries
  the start value ("recur_mix_init").
- w is in the AdamW group of a later pass's own tensors (ced/train/optim.py: no weight decay, the run's LR x
  optim.pass2_lr_scale, 1.0 by default);
- metrics of every step: "w" (after that update), "grad_norm_w" (the norm of its gradient before clipping; 0 before
  the switch) and "lr_w" (the LR its group had).
A pass count per row (t_sampling "sequence") is refused with a switch step: nothing runs or tests the pair.

R3, the depth mix at the loop boundary (model.loop.injection "depth_mix" on a looped middle; ced/model/recur_depth.py):
every pass reads a softmax mix over [the embedding's output x0, the prelude's output p, the earlier passes' outputs].
It trains from scratch by the code above (one T per step from t_choices; no switch step: at its start it is not the
plain control, and loop_switch_step is refused for it as for every looped middle but the scalar mix). Its reader's two
tensors are AdamW tensors with the run's weight decay (ced/train/optim.py). Every log_every steps the metrics hold
- "depth_mix": {"pass1": [the mean weight of x0, of p], "pass2": [of x0, of p, of z1], ...}: the mean over the
  positions of the step's first forward (the forward's own weights, before this update) of each candidate's weight at
  each pass that forward ran. Read from the monitored forward the step runs anyway (the model's collect_stats; the
  weights are formed a second time under no_grad, recur_depth.mix_weights), so the losses, the gradients, the
  optimizer's state and the RNG state are those of the run without the logging, bit for bit (tests/test_loop_r3.py);
- "depth_mix_query": {"query_norm": ||q||, "key_gain_dev": max |gain - 1| of its key norm} after this update (the
  query's norm is how far it has moved from zero, where every candidate has equal weight).

The loop screen's trainer changes (stage 2; the build's note is). Each is off by default or changes no number of a
run whose losses are finite.
- The stop on a non-finite loss (w stays unbounded at x10 and x100, with a stop on a non-finite loss). fit() reads
  the loss and the gradient norm train_step has already put on the host. When either is not finite the run ends at once
  (Trainer._check_finite): no checkpoint is written (`latest` stays the last one, whose weights are finite), the step's
  own metrics record is not written, one record {"event": "diverged", "step": k...} goes to metrics.jsonl, the marker
  file <run_dir>/diverged.json is written ("diverged at step k", the non-finite values, the last finite steps, and w's
  last values when the model has w), and fit() raises Diverged; scripts/train.py turns that into the exit status
  EXIT_DIVERGED (86). k counts updates as metrics.jsonl does: the update whose record would have carried "step": k. A
  run directory that holds the marker is never trained again: fit() raises Diverged at once, and scripts/train.py exits
  with the same status before it builds anything, so a restart of the process (the run's backoff, a resubmission) costs
  seconds and trains nothing. Nothing is read from the device for it and nothing of a step changes: a run with finite
  losses is the run it was.
- The LR ramp after a switch (optim.switch_warmup_steps N_w; 0 = off): for the updates k with S <= k < S + N_w, S the
  loop switch step, lr_of multiplies the schedule's LR by (k - S + 1) / N_w, so every parameter group has that
  factor (train_step sets group["lr"] = lr x the group's multiplier). The factor of update S + N_w - 1 is exactly 1.
  A pure function of the step, as the schedule is: a resume inside the ramp needs nothing but the step.
- The readout inside training (eval.readout_every R; 0 = off; ced/train/readout.py): after every R-th update, at the
  switch step and at the last step, exact match at T = 1 and T = 2 on a fixed sample of dev items, one record
  {"event": "readout", "step", ...} in metrics.jsonl. It runs between two steps on a scoring twin of the model (the
  scoring runs' eager path on the trainer's own parameter tensors), outside train_step, so "step_seconds" never holds
  its time, and it changes nothing of the training state (tests/test_loop_readout.py).

Exact-zero weight gradients by linear (TrainConfig.log_zero_wgrad_every N; ced/train/zero_wgrad.py; the NVFP4 tail arms'
counter). Off by default (0): nothing below runs. With N > 0, on every N-th optimizer step, after the micro-batches and
before clipping, the number of exactly-zero elements and of all-zero output rows of every layer linear's weight gradient is
counted on the device; the host reads them after the update as "zero_wgrad" {"zeros", "zero_rows"}, in the order of the
start record's "zero_wgrad" {"every", "modules", "numel", "rows"}. Read only: the run is the run without it, bit for bit
(tests/test_nvfp4_arms.py).

A stage that continues another run on other data (TrainConfig.stage "continue" and data_start_step;
tests/test_stage_continue.py). Off by default ("none", 0): nothing below runs and resume_from is what it was.
- What it is: resume_from (a NEW run from another run's kept checkpoint of step S: weights, Muon and AdamW state, step
  counter, RNG states), which may then differ from its source in STAGE_FREE besides BRANCH_FIELDS and BRANCH_SECTIONS:
  optim.decay_fraction, data.shard_dir, stage, data_start_step and (on a new shard set) data.loss_mask, and the shard set
  itself (the data identity's fingerprint; stage_data_relation). Nothing else: not the data seed, the row length, the rows
  a step (unless stage_shape is "tokens": below, "A stage at another row shape"), the model, the other optimizer keys.
  Without the opt-in a plain resume_from, and a loop branch from its trunk, refuse other data and another schedule exactly
  as before.
- The LR: lr_of over the run's own steps and decay_fraction, as for any run; resume_from's check that the source made
  every one of its updates at this run's LR stays. So a stage A of S steps at decay_fraction 0 (S before the whole run's
  decay start) continued by a stage C of N steps at decay_fraction f has, at every update, the LR of one N-step run at f.
- The data: update k reads the stream's batch k - D (data_step; rows, decoder and aux positions alike), D =
  data_start_step. On the source's shard set D is the source's own (0): the stream goes on, and stage A + stage C is the
  uninterrupted run bit for bit. On a new shard set D must be S: the new stream starts at its beginning (batch 0 for the
  run's data seed) at the stage's first update. D is part of the data identity a checkpoint records (data_state), so a
  stop and resume inside the stage resumes the new stream at k - D, from its own checkpoint, as any resume.
- With loop_switch_step S (a loop branch at the stage boundary) the source is the T = 1 trunk as for any branch; the
  trunk's schedule-length check gives way to the update-by-update LR check (a stage's source is cut at S). The branch
  then reads base stage C's batches at base stage C's LR, step for step.
- In another precision (TrainConfig.precision_switch, off by default; the NVFP4 tail arms): the stage may also differ
  from its source in PRECISION_FREE (model.nvfp4, model.fp8 and the key itself). Both switches turn the layers' nn.Linear
  into a subclass in place and add no state, so the source's weights, Muon and AdamW state and RNG states load as they
  are, and the stage is the source's training state continued with this run's linears: bit for bit the source's run with
  its precision flipped in memory at step S (tests/test_nvfp4_arms.py). Recorded in resumed_from as "precision"
  {"source", "this"}. Refused without stage "continue", and with model.cp_head (the head is built in BF16 only:
  check_cp_head refuses model.nvfp4 and model.fp8 with it). Off: the refusal of another precision is what it was. With a
  stage at another row shape
  (stage_shape "tokens") and a per-token loss mask on the new set, the free keys add up (STAGE_FREE, STAGE_SHAPE_FREE,
  PRECISION_FREE) and the stage is the same stage from the source labelled in the new precision, bit for bit
  (tests/test_nvfp4_c3c4_cp.py).
- NVFP4 for the first N updates of one run (model.nvfp4.until_step N, the 4B schedule's 66.8 %): at the start of update N train_step turns the
  quantised linears off in memory (ced/model/nvfp4.py set_phase), the same function as a stage "continue" with precision_switch and not tied to a
  stage, so it runs with TST (when it was built a TST stage was on a new shard set only; since 2026-10-29 it also takes the source's own). The run's
  LR, data stream and RNG go on; after a resume past N the first step turns them off before it trains. "nvfp4_switch" in the metrics of update N. Off
  (0): nothing runs, no module attribute is set.

The bpb readout inside training (eval.bpb_readout_every R; 0 = off; ced/train/bpb_readout.py): from eval.bpb_readout_from
on, every R steps and at the last step, bpb at T = 1 (and T = 2 for a looped model) on a fixed seeded sample of the run's
eval rows, by ced.eval.span_bpb.evaluate on the training model (its compiled layers where the run compiles them), one
record {"event": "bpb_readout", "step", ...} in metrics.jsonl with the per-row sums a paired report needs. Between two
steps, outside train_step; the training is the same bit for bit with it on or off.

Token Superposition Training (TrainConfig.tst; ced/train/tst.py has the rules). Off by default (tst.schedule []): self.tst is None and nothing below
runs. With a schedule: batch_at reads the step's rows from TSTPlan.batch_at (bags of s tokens in a bag phase, the base's rows in a next-token phase,
each from where the steps before it stopped in the stream); train_step starts a later phase's first step with the switch (TSTPlan.switch: the
optimizer's state emptied, unless tst.keep_optimizer_state, and with tst.gdn_decay_rescale the GDN A_log shifted) and scores a batch with several
targets per position (Batch.multi_targets) through the multi-target head (_micro_loss_tst); with tst.token_scale_rope a bag phase's forward gets
rope_scale = s, and with tst.gdn_resolution_gates
(the redesign's E) gdn_scale = s (TSTPlan.gdn_kwargs), and with tst.bag_input "sum_sqrt" bag_input = "sum_sqrt" (TSTPlan.bag_kwargs);
tst.bag_lr_power (L) and tst.rewarm_steps (R) multiply the step's LR of every group by TSTPlan.lr_factor (the "lr" metric is the LR the
groups got). Metrics of every step: "tst_s" and "tst_phase"; a bag step's ce_dec / ce_ctx / ce_all are its weighted bag CE (the loss
without the z-term) over the decoder-mask / other / all positions, and "ce_bag_first" the CE of each position's first target; with the
multi-hot aux "aux_lambda" and "ce_multi" (the mean CE of the valid targets t + 2 .. t + K). A switch step's metrics carry "tst_switch"
and fit() writes it as the record {"event": "tst_switch"} before the step's own. With tst.schedule [[0, 1]] every step is the base's
step bit for bit (tests/test_tst.py, and against main's tree: tests/golden/make_tst_off_golden.py). A stage (stage "continue") of a TST
run is built after the schedule's last switch only, on a new shard set or on the source's own: on a new set its update D + j reads the
new set's batch j as the base's stage does, on the source's own (data_start_step 0) the stream goes on (ced/train/tst.py, "A STAGE
AFTER THE LAST SWITCH"; tests/test_tst_stage.py, tests/test_tst_stage_own_set.py); resume_from_checkpoint's one addition for it is
TSTPlan.check_stage_source(step) after the data check, before any state is loaded. Under data parallelism (ced/train/data_parallel.py,
TST_DP): _build_batch gives a rank its bag rows (TSTPlan.batch_at first / count), _dp_batch adds the step number, a sharded optimizer's
switch count and the loss weights' integers (TSTPlan.host_counts) over the ranks in its one host reduction, and train_step forms the
global bag totals from them (TSTPlan.weights summed); one process runs none of it.

The input guards. Always on; no option. A run whose batches are valid and whose losses and gradient norms are finite is
the run it was, bit for bit (tests/test_train_input_guards.py against main's tree).
- The finite check is the stop on a non-finite loss above (Trainer._check_finite), unchanged in what it reads and
  when: the loss and the gradient norm that train_step already reads on the host for its metrics, after
  optimizer.step(), so no device sync is added. The update of the failing step is therefore made in device memory, but
  nothing is ever written from that state: the check is the first thing fit() does with the step's numbers, before the
  step's metrics record, any checkpoint (rolling, kept, on SIGTERM, final), eval, readout or W&B log, and the process
  then exits 86. The persisted state is the last checkpoint, written before the bad gradient existed; a resume from it
  (once the marker is removed by hand) is the run that never halted, bit for bit. The record now also names the step's
  batch: "batch_hash", "rows" (row_record: where the rows come from in the data stream, a content hash of every row,
  the first and last document id of every row) and "last_finite_step" (k - 1: every update is checked before the next).
  An optimizer that fails on the non-finite gradient itself (Shampoo's and KL-SOAP's torch.linalg.eigh refuses a NaN
  matrix) would end the step before fit() reads its numbers: train_step catches an exception of optimizer.step(), reads
  the loss and the gradient norm there (on that failure path only), and when either is not finite hands them to the
  same check, whose record then also carries "optimizer_error" (Trainer._optimizer_failed); with both finite the error
  is raised as it was (tests/test_train_input_guards.py).
- The non-finite flag (FP8Config.nonfinite_flag, off by default). With it on (model.fp8.quantiser "triton" or
  model.fp8.te_operands "ours"), every MXFP8 operand our kernels quantise in the step ORs a bit into one int32 on the
  device (ced/kernels/mx_ops.py FLAG: 1 an activation, 2 a weight, 4 an output gradient) when one of its blocks holds an
  inf or NaN; train_step zeroes it first (a device memset) and brings it to the host inside the gradient norm's own read
  (one copy of two float64 numbers: no other sync), as metrics["mxfp8_nonfinite"]; the finite check treats a set bit as
  a non-finite step: the same halt, the record naming "mxfp8_nonfinite_flag" and the operand kinds, even when the GEMMs
  hid the value (a hardware MMA that saturates). Off: nothing of it exists and every step is main's. Its form
  (FP8Config.nonfinite_flag_form, read only with quantiser "triton": ced/model/fp8.py flag_form_of) is "pure" by default
  since, "ordered" before. A run keeps the form it trained: load_state_dict refuses a checkpoint whose flagged form
  differs from this config's (a checkpoint written before the key trained "ordered": ced/config.py checkpoint_config),
  and init_from and resume_from compare the form as flag_form_of reads it (the key is inert, and not compared, where the
  flag is off).
- The loader's validation (ced/train/batch_check.py): batch_at checks every batch on the host before it is returned
  (input and target ids in [0, V), TST's multi-hot targets too; document ids >= 0 and non-decreasing along each row,
  so each row's segments cover it exactly; shapes and dtypes), and ced/train/varlen.py checks every cu_seqlens it
  builds for the fused GDN kernels. A failure raises BatchError naming the update, the row, its stream row and the
  position. fit() ends the run on it through the same path as the stop: the record {"event": "bad_batch", "step": k,
  ...} in metrics.jsonl and in the marker <run_dir>/diverged.json, no checkpoint, and BadBatch (a Diverged), so
  scripts/train.py exits 86 and the controller reports and delivers it as it does a diverged run.

The per-token loss mask (DataConfig.loss_mask; tests/test_loss_mask.py). Off by default ("all"): Batch.loss_mask is None,
loss_valid(batch) is batch.target_valid itself, and every line below reads what it read before (tests/test_loss_mask.py
against main's tree). With a mode, the shard packer gives every batch a (B, L) bool loss_mask (whether position t's target
keeps its loss; ced/train/shards.py) and train_step reads ONE value, valid_loss = target_valid & loss_mask, in the three
places that decide the loss: batch_counts (n_valid, n_dec and n_aux count the kept positions, so every denominator follows:
the one-head mean, both normalisations of the two heads, the draft terms), _head_selection (the selected and fused heads'
positions, the slots' targets and the aux positions) and _micro_loss (the unfused heads, the slots' on_dec / on_aux). The
loss is then the mean over the kept positions; a batch with none has loss 0 (the max(1, n) of the denominators). What stays
on target_valid: the decoder and aux position samplers (so the arms of one comparison share them), the pass schedule,
varlen, the halt records' row hashes, batch_hash (the arms read the same rows) and every scorer (the training mask is never
applied in scoring). Metrics with a mask: "n_loss" (kept targets) and "n_target_valid" every step; the data identity
(data_state) holds "loss_mask" and, for a span mode, "loss_spans" (each part's span classes: version, head, tail, class
counts), so a resume with another mode or span file is refused, and a stage on a NEW shard set may set its own mask
(STAGE_FREE, stage_data_relation). Refused by name: synthetic data, TST, data.fit_decoder_capacity. With the Copy-Paste
head (model.cp_head) the mask needs no line of its own: _cp_loss reads the same selection (sel["all"], from valid_loss)
and the same denominator (n_valid), so the mixture's CE, its z-term and the CP tensors' gradients come from the kept
positions only; the CP head's context (its keys, every position of the row) is not masked, as attention's is not
(tests/test_cp_c3c4.py).

A stage at another row shape (TrainConfig.stage_shape "tokens"; tests/test_stage_shape.py). Off by default ("source"):
nothing below runs and every refusal of a stage is what it was. With it, a stage (stage "continue") may also differ from
its source in STAGE_SHAPE_FREE: data.seq_len and data.batch_size at the source's tokens a step (stage_data_relation: rows
x row length equal, e.g. 6 x 32,768 = 24 x 8,192), and micro_batch_size and activation_checkpoint, which hold the same
step in memory. What follows:
- the state at the switch is the source's (weights, Muon and AdamW state, step counter, RNG states), as for any stage;
- the LR clock is the step counter: update k has read k x the same tokens under either shape, so resume_from's check (the
  source made every update before S at this run's LR) is unchanged;
- the new shard set is read from its beginning (data_start_step = the source's step S) in rows of the stage's length: update
  S + j reads the packer's batch j of that length (its rows, its decoder positions); a shape change on the source's own
  set is refused (its stream goes on in rows of the source's length);
- a stop inside the stage resumes from the stage's own checkpoint (its data identity has the stage's shape), or, before
  its first checkpoint, starts from the source again: both the uninterrupted stage bit for bit;
- the record: resume_info["stage"]["shape"] = {"source": [rows, length], "this_run": [rows, length], "tokens_per_step",
  "memory": {micro_batch_size, activation_checkpoint: [source, this run]}}; scripts/train.py prints it;
- the FLOPs log (flops_done): the source's updates are counted at the row length they were trained at, along the chain
  of sources (_stage_shape_segments), this run's own at its own; logging only.
Refused by name with tst.schedule, loop_switch_step, the slot arches (their capacity follows the row length),
data.fit_decoder_capacity and model.cp_head with kernel "reference" (its training terms run one query block shape per
query_chunk of the row: 64 at 32,768 / 512, over the trainer's recompile limit of 32 for the compiled block on CUDA;
tests/test_cp_c3c4.py); with kernel "fused" (one kernel for every row length) it is built, and model.cp_head.kernel is one
of STAGE_SHAPE_FREE (tests/test_cp_fused.py); shape_schedule is refused with any stage.

Data parallelism (Trainer(dp=...); ced/train/data_parallel.py states the design). Off by default (dp None:
scripts/train.py without torchrun's environment): nothing below runs and every step is what it was
(tests/test_data_parallel.py, T6, against main's tree). With a DataParallel of world size W (built: D1 to D3):
- the device is the rank's (DataParallel.device); batch_at gives the rank's B / W rows of the step's batch, positions drawn
  for their global row index (D2);
- train_step sums the step's host counts over the ranks before the first micro-batch (_dp_batch), so every micro-batch's
  loss is its local sum over the global counts; a batch failing the loader's validation on any rank raises BatchError on
  every rank there; the micro-batches are the rank's K / W;
- the gradients are views of one flat buffer (data_parallel.GradBuffer; buffers and N1's persistent gradients are
  those views), summed over the ranks after the micro-batch loop with the step's metric sums (_dp_sync), before the
  gradient metrics and the clip; the replicated optimizers run unchanged;
- the metrics are the global step's ("tokens", the counts, the sums, "batch_hash" over the ranks' rows, with
  "batch_hash_ranks" and "world_size"); the monitored first micro-batch and the stream RMS are each rank's first
  micro-batch (rank 0's is the single process's);
- D4, fit's agreement and rank 0's duties: every rank runs fit and acts at the same step. A bad batch on any rank and a
  non-finite step halt every rank (their decisions are the global step's: the host reduction's flag, the summed loss and
  gradient norm, the OR-ed MXFP8 flag); the halt record names every rank's rows ("rows_by_rank", gathered on this path
  only). A SIGTERM on any rank stops every rank after one checkpoint, and a checkpoint is due by rank 0's clock: both in
  ONE MAX over the ranks on the host group after each step (DataParallel.agree). Rank 0 alone writes metrics.jsonl, the
  halt records and the marker, logs, talks to W&B, and runs the evals and readouts while the others wait at a barrier on
  the host group (_lead_duty; its timeout is data_parallel.ENV_WAIT_SECONDS). With the evals sharded (E1,
  data_parallel.ENV_SHARD_EVALS, off by default; ced/train/sharded_eval.py) the periodic and final evals on the eval sets
  are every rank's work instead (_eval_duty): each rank scores its block of the sets' batches, rank 0 joins and writes
  them (the single process's records bit for bit); the readouts stay rank 0's. The ranks' parameters are compared once
  before each process's first step (data_parallel.check_replicas; again every N-th step with ENV_CHECK_EVERY);
- D5, the checkpoint at any world size: every rank's CUDA generator state is gathered and rank 0 writes today's single
  file (the state is replicated), with "cuda_rng" None and "data_parallel": the world size, each rank's CUDA state, the
  history of world sizes (dp_info). Each rank's CUDA generator is seeded from (seed, rank, step) at a data-parallel start
  and whenever a resume changes the world size (data_parallel.rank_seed), and restored from the file at the same world
  size; every re-derivation is recorded in dp_info's history, which the checkpoints and the start records keep. The
  world size is not in the config, so scripts/train.py's config_differences never sees it; DP_RESUME_FREE
  (micro_batch_size, activation_checkpoint, optimizer_sharding) may differ on a resume that is data-parallel or changes
  the world size, recorded likewise. One process writes "data_parallel" only for a run that ran on several ranks before:
  every other checkpoint is byte for byte what it was.
- N2, the sharded optimizer (TrainConfig.optimizer_sharding "zero1"; ced/train/zero1.py; off by default): each rank owns
  whole tensors and keeps their optimizer state alone; the gradient buffer and the parameters live in owner-ordered flat
  buffers, the all-reduce becomes a reduce-scatter, the owners alone step their tensors and an all-gather hands every rank
  the updated weights; the clip's per-tensor norms, the gradient metrics that read whole gradients and ANVIL-II's per-bank
  sums are gathered exactly from the owners, so the step is the replicated step bit for bit (tests/test_zero1.py, T7).
  Its checkpoint is written by parameter: one optimizer file a rank beside rank 0's step file (zero1.shard_dir), loaded at
  any world size, sharded or not (zero1.load_for).

The asynchronous save (TrainConfig.ckpt_async; ced/train/async_save.py has the mechanics; off by default, and then every
line here is the synchronous save). fit's periodic saves snapshot the state into host memory and write it in a background
thread (save_checkpoint with wait False); the file and `latest` are COMMITTED by the training thread at a step boundary
once the write is done (_ckpt_settle): in one process when the thread has ended, under data parallelism when every rank
agrees, from slots added to fit's one MAX a step (DataParallel.agree's extra: [this rank's write still running, this
rank's write failed]), so every rank commits at the same step and keeps the same record of commits. One write at a time:
a save due while the last is not committed waits for it first. A write that fails raises CheckpointWriteError on every
rank at the same step, and `latest` still names the last whole file. The stop's save (a SIGTERM on any rank) and the
end's wait for their write: a write in flight when the stop comes is finished and committed first, then the stop's step
is saved and committed, so a kill at any moment leaves `latest` naming a whole file. A halt
(a non-finite step, a bad batch) commits a write in flight before it writes its record, which names that checkpoint; a
crash commits rank 0's finished write on the way out (fit's finally). The files are the synchronous save's bit for bit
(tests/test_async_save.py). With the sharded optimizer (N2) every rank copies and writes its own optimizer shard in the
background too; at the agreed commit each rank renames its shard before rank 0's step file (_ckpt_commit), and a crash
commits nothing (the agreement cannot be had on the way out); the files are the synchronous sharded save's bit for bit
and load at any world size as N2's do (tests/test_async_save_zero1.py). Each commit is one {"event": "checkpoint"} line in metrics.jsonl: the checkpoint's step, the
step it was committed after, the seconds waited for the last write, the snapshot's and the write's seconds, the bytes
("on_exit": true on the crash's commit).

The data prefetch (TrainConfig.data_prefetch; ced/train/prefetch.py has the mechanics; off by default, and then train_step is the
step it was and nothing of the module is imported). train_step takes its input (batch_at, the loss-valid mask, the host numbers its
metrics read, the device twins of the tensors the micro-batches read) from a prefetcher that built it on a thread while the step
before ran, and starts the build of the next step's input once it has its own. The input is a pure function of the step, so the
stream is the off run's bit for bit on every rank and a resume (the checkpoint holds the step and the data identity, nothing of the
prefetch) yields the same stream; the device copy is made on a side stream from pinned memory and the stream the step runs on waits
for its event before the first use (Staged.wait; a read before it raises PrefetchRace). The graph runner's captures hold the
prefetcher's CUDA lock. Metrics add "prefetch" (build, stage and wait milliseconds, hit) only when on.

The post-hoc add-on stage (TrainConfig.freeze_trunk; ced/train/freeze_trunk.py has the rules; off by default, and then the trainer is main's bit
for bit: the only code that runs is the key's type check). On: every tensor outside model.cp_head.* has requires_grad False (Trainer.__init__,
after the head's clip lists are made), so train_step trains the head alone through its own code: no trunk graph, no trunk gradient or optimizer
state, grad_norm 0 (the trunk's clip is skipped; cp_grad_norm is the head's). init_from_checkpoint accepts the head-less source (the head's
tensors "grown", the trunk strict) and takes the trunk's sha256; the checkpoint holds it ("freeze_trunk") and every save and load checks it;
train_step checks before the clip that no trunk tensor has a gradient and after the update that none has state (TrunkTouched). The start record of
metrics.jsonl gains "freeze_trunk" (the frozen and the trained tensors and parameters, the start checksum) only when on. After
the process's first update the trunk is hashed once (a write in place by another part of the step is seen on that update, not at the next save);
the clamps that follow an update (project_layer_scalars, floor_gates_) and the scalars' gradient norm are not run on a frozen trunk; init_from
refuses a source the model would have to cast; a resume checks the file's trunk record against its init_from record and its optimizer for trunk
state.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import random
import shutil
import signal
import time
from pathlib import Path

import torch

from ced.config import (TrainConfig, checkpoint_config, from_dict, looped_control, one_pass_model, recur_loop, recur_mix,
                        to_dict)
from ced.data.shard_format import LOSS_MASK_ALL, LOSS_MASK_MODES
from ced.model import act_checkpoint, pass_schedule
from ced.model import recur_depth
from ced.model.capacity import decoder_capacity, gather_slots, select_slots
from ced.model.control import LOOPED_ARCHES, SLOT_ARCHES, build_model, has_loop  # noqa: F401 (LOOPED_ARCHES re-exported)
from ced.model.encoder import layer_scalar_parameters
from ced.model import nvfp4 as nvfp4_mod
from ced.model.fp8 import flag_form_of
from ced.model.norms import compute_dtype
from ced.train.batch_check import BatchError, row_hashes
from ced.train.batch_check import check_batch as check_batch_contract
from ced.train.ce_fix import check_config as check_ce_form
from ced.train.chunked_loss import head_positions, selected_ce, token_ce, token_multi_ce
from ced.train import freeze_trunk as freeze
from ced.train.fused_head import fused_linear_ce, fused_linear_multi_ce
from ced.train.head_precision import check_head_precision

# Public export: the heads that load the CuTe DSL runtime (head_impl 'quack_lce') and the Liger / CCE heads are not part of this export.
# The configs here use head_impl 'fused'; the stand-ins below refuse any other choice by name.
PORTABLE_HEADS: tuple = ()


def _head_not_in_export(*_args, **_kwargs):
    raise ValueError("head_impl 'quack_lce', 'liger_flce' and 'cce' are not part of this code export; use head_impl 'fused' or 'chunked'")


check_quack_head = quack_linear_ce = check_portable_head = portable_linear_ce = _head_not_in_export

from ced.train.data import Batch, SyntheticPackedData, counter_generator
from ced.train.optim import NO_DECAY_SUFFIXES, RECUR_MIX, build_optimizer, gdn_mup_init_, is_muon_param, is_pass2_param, param_groups  # noqa: F401 (re-exported)
from ced.train.positions import PositionSampler
from ced.train.shape_schedule import SHAPE_LR_MODES, ShapeSchedule
from ced.train.zero1 import SHARD_SUFFIX as ZERO1_SHARD_SUFFIX
from ced.train.zero1 import shard_dir as zero1_shard_dir


def check_head_chunk(cfg: TrainConfig) -> int:
    """TrainConfig.head_chunk (the fused head's positions per chunk; ced/config.py): a positive whole number, and a value
    other than fused_head.CHUNK (2,048, the default) only with head_impl "fused", the one head that reads it (any other
    head_impl would ignore it silently). Returns it."""
    from ced.train.fused_head import CHUNK

    chunk = cfg.head_chunk
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError(f"head_chunk {chunk!r}: a positive whole number of positions (default {CHUNK})")
    if chunk != CHUNK and cfg.head_impl != "fused":
        raise ValueError(f"head_chunk {chunk} is read by head_impl 'fused' only (the fused head's chunk); head_impl "
                         f"{cfg.head_impl!r} would ignore it: leave head_chunk at {CHUNK}")
    return chunk


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def uses_fla(cfg) -> bool:
    return cfg.model.encoder.gdn.kernel in ("fla", "ced") or cfg.model.decoder.gdn.kernel in ("fla", "ced")


def batch_hash(batch) -> str:
    """A short hash of what the step trains on: tokens, document ids, decoder positions and (when
    set) aux positions. Two runs saw the same data order iff their per-step hashes agree."""
    h = hashlib.blake2b(digest_size=8)
    for t in (batch.tokens, batch.doc_ids, batch.dec_mask, batch.aux_mask):
        if t is not None:
            h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()


def configure_compile() -> None:
    """torch.compile settings that make the compiled path the same in every process:
    - static activation shapes (no automatic dynamic shapes; the few shape variants are separate
      graphs), with the varlen tensors explicitly dynamic (ced/train/varlen.py:to_device);
    - hitting the recompile limit is an error, never a silent fallback to eager (a fallback depends
      on the process's history, so a resumed process would run different kernels);
    - Inductor's deterministic mode: no on-device benchmarking whose choice would change numerics.
    The launcher keeps the compile caches on <data volume> (TORCHINDUCTOR_CACHE_DIR) for compile time."""
    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config

    dynamo_config.automatic_dynamic_shapes = False
    dynamo_config.recompile_limit = 32
    dynamo_config.fail_on_recompile_limit_hit = True
    inductor_config.deterministic = True


def compile_layers(model: torch.nn.Module, **compile_kwargs) -> None:
    """torch.compile, in place, every per-layer computation of a training step (compile "all"):
    - every encoder layer's forward (for the second-pass control this is pass 1, run with keep=True);
    - CED v2's decoder layers (and v1's), which run T times at the slots;
    - the second-pass control's pass 2: each encoder layer's zero_write, the same layers at the slots;
    - the looped control's pass 2 (ced/model/control.py, "Two passes"): the block layers' zero_write_all
      (every position: training and scoring) and zero_write (marked positions: cached generation). The layers
      below the block get no pass-2 call and none is compiled for them. So every arm's measured step time, which
      sets the time-matched depths and step counts, comes from the same treatment of its layers: v2's decoder
      and (c)'s pass 2 are both compiled. compile_kwargs go to torch.compile (default dynamic=False)."""
    compile_kwargs.setdefault("dynamic", False)
    for layer in list(model.encoder.layers) + list(getattr(model, "decoder_layers", [])):
        layer.compile(**compile_kwargs)
    if model.cfg.arch == "second_pass":
        for layer in model.encoder.layers:
            # An instance attribute shadows the method; the class method stays the eager reference.
            layer.zero_write = torch.compile(layer.zero_write, **compile_kwargs)
    if looped_control(model.cfg):
        for layer in list(model.encoder.layers)[model.pass2_from:]:
            layer.zero_write_all = torch.compile(layer.zero_write_all, **compile_kwargs)
            layer.zero_write = torch.compile(layer.zero_write, **compile_kwargs)


def aux_positions(valid: torch.Tensor, kept: torch.Tensor, aux_mask: torch.Tensor | None) -> torch.Tensor:
    """Where the aux head's loss counts: valid targets in aux_mask (LossConfig.aux_positions
    "random"), or v1's context positions, the valid targets that do not run the decoder."""
    return valid & (aux_mask if aux_mask is not None else ~kept)


# Model config fields that choose how the same function is computed, not what it is (AttentionConfig.backend and asentmax_chunk,
# GDNConfig.kernel, chunk_size and decay_cumsum): the only model config differences init_from accepts. decay_cumsum: the
# rounding of the kernel's cumulative log decay, the same delta rule; so a run launched with F3 on may start from a checkpoint
# trained with it off (a stage C, a growth), and the reverse. fa3_docs_per_row: the capacity of the kernel's varlen metadata,
# the same function. edges: where the GDN input preparation runs (before our chunk kernels or in their prologues), the same
# function and parameters; a run with "fused" may start from a "plain" one. fa4_deterministic: FlashAttention-4's deterministic
# backward or its atomic one, the same function (only the summation order of the gradients differs). chunk_backend: which
# kernels run the GDN delta-rule core (our Triton kernels or FlashQLA's), the same function and parameters; a run with
# "flashqla" may start from a "triton" one. kernel in model.cp_head: the reference head or the fused kernels, the same function
# and parameters within float32 rounding; so the long-row measurement's runs (init_from an 8k final trained with "reference", at
# 32k rows with "fused") start. model.cp_head.window is not one: it changes what the pointer reads. flex_tiles: the order in
# which flex visits a query block's key tiles ("classified" is flex's own, "indexed" folds the full tiles into the partial pass
# in ascending order); the same function, a different rounding.
IMPLEMENTATION_FIELDS = ("backend", "kernel", "chunk_size", "asentmax_chunk", "decay_cumsum", "fa3_docs_per_row", "edges",
                         "fa4_deterministic", "chunk_backend", "flex_tiles")
IMPLEMENTATION_SECTIONS = ("attention.", "gdn.")  # the config sections whose IMPLEMENTATION_FIELDS are free
CP_IMPLEMENTATION_FIELDS = ("kernel",)  # the same in model.cp_head
# The same in the encoder's own section. attn_res_fused: whether R1-entry's entry read and input norm run as one kernel,
# the same function and parameters; a run with it on may start from a checkpoint trained with it off, and the reverse.
ENCODER_IMPLEMENTATION_FIELDS = ("attn_res_fused",)
# The same in the model's own section. fused_projections (other.concat; ced/model/fused_proj.py): the projections that
# read one input as one GEMM, the same function and parameters (the GEMM's order of summation differs); a run with it on
# may start from a checkpoint trained with it off, and the reverse.
MODEL_IMPLEMENTATION_FIELDS = ("fused_projections",)
# The same in the GDN layers' dprod section. route (GDNConfig.dprod.route; ced/kernels/gdn_1r.py): whether the fused path reads the n_h = 2
# steps out on every sub-step of the interleave or once a token, the same function and parameters (the GPU check matches it to the float64
# reference); a run with "one-readout" may start from a checkpoint trained with "interleave", and the reverse.
DPROD_IMPLEMENTATION_FIELDS = ("route",)


def fp8_comparable(fp8: dict) -> dict:
    """A model.fp8 config dict with its nonfinite_flag_form as the model reads it (ced/model/fp8.py flag_form_of): None
    where the form is inert (the flag off, or no operator form: quantiser other than "triton"). So two configs differ
    in the form only where it runs: the key's default changed from "ordered" to "pure", and every flag-off checkpoint
    written until then carries "ordered"."""
    return {**fp8, "nonfinite_flag_form": flag_form_of(fp8)}


def model_differences(source: dict, own: dict, prefix: str = "model.") -> list[str]:
    """Dotted model config keys whose values differ between an init_from source and this run, apart
    from IMPLEMENTATION_FIELDS inside attention and GDN configs, ENCODER_IMPLEMENTATION_FIELDS in the encoder's,
    CP_IMPLEMENTATION_FIELDS in model.cp_head and MODEL_IMPLEMENTATION_FIELDS in the model's own section."""
    out = []
    for key in sorted(set(source) | set(own)):
        name = f"{prefix}{key}"
        a, b = source.get(key), own.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            if name == "model.fp8":  # the flag's form as it is read (fp8_comparable)
                a, b = fp8_comparable(a), fp8_comparable(b)
            out += model_differences(a, b, name + ".")
        elif a != b and not (key in IMPLEMENTATION_FIELDS and prefix.endswith(IMPLEMENTATION_SECTIONS)) and not (
                key in ENCODER_IMPLEMENTATION_FIELDS and prefix == "model.encoder.") and not (
                key in CP_IMPLEMENTATION_FIELDS and prefix == "model.cp_head.") and not (
                key in MODEL_IMPLEMENTATION_FIELDS and prefix == "model.") and not (
                key in DPROD_IMPLEMENTATION_FIELDS and prefix.endswith("gdn.dprod.")):
            out.append(f"{name}: source {a!r} -> this run {b!r}")
    return out


def lr_schedule(optim, total: int, step: int | float) -> float:
    """Warmup-stable-decay, the LR of the update with index `step` (0-based) of a run of `total` steps: linear warmup
    over optim.warmup_steps updates, constant, then linear decay to zero from int(total * (1 - decay_fraction)) (to
    optim.lr_floor x the peak when that is set: peak x (floor + (1 - floor) x the decay's fraction left)).
    `step` may be a fraction (schedule_clock: a run whose steps are not the schedule's): the warmup is then capped at
    its last integer value (min below; for an integer step < warmup_steps it changes nothing)."""
    decay_start = int(total * (1.0 - optim.decay_fraction))
    if step < optim.warmup_steps:
        return optim.lr * min(step + 1, optim.warmup_steps) / optim.warmup_steps
    if step < decay_start:
        return optim.lr
    frac = max(0.0, (total - step) / max(1, total - decay_start))
    floor = getattr(optim, "lr_floor", 0.0)
    if floor:  # OptimConfig.lr_floor: the decay ends at floor x the peak (the record's LR_FLOOR), not at 0
        return optim.lr * (floor + (1.0 - floor) * frac)
    return optim.lr * frac


def schedule_total(cfg: TrainConfig) -> int:
    """The length of a run's LR schedule in reference steps: optim.schedule_steps, or its own steps."""
    return cfg.steps if cfg.optim.schedule_steps is None else cfg.optim.schedule_steps


def schedule_clock(cfg: TrainConfig, step: int) -> int | float:
    """Where update `step` (0-based) of the run sits on its LR schedule, in reference steps (OptimConfig.schedule_steps;
    loop experiment: equal compute is wall-clock, and a T = 2 step is slower than a T = 1 step). Without
    schedule_steps: `step` itself, every run before the key. With it, N = schedule_steps reference steps stand for the
    run's planned wall-clock, and with S = loop_switch_step (0: none):

        clock(k) = k                                    k < S     (the trunk's own steps: one reference step each)
        clock(k) = S + (k - S) * (N - S) / (steps - S)  k >= S    (the steps - S remaining steps share the N - S
                                                                   reference steps left)

    In wall-clock fractions, f = S / N and N2 = steps - S: the run reads the reference schedule at
    f + (1 - f) * (k - S) / N2. Two arms of a pair with the same N (the T = 1 arm: steps = N, clock(k) = k) get
    the same LR at the same fraction of the planned wall-clock."""
    N, S = cfg.optim.schedule_steps, cfg.loop_switch_step
    if N is None or step < S:
        return step
    return S + (step - S) * (N - S) / (cfg.steps - S)


def switch_ramp(cfg: TrainConfig, step: int) -> float | None:
    """The factor of the LR ramp after a loop switch step at update `step` (OptimConfig.switch_warmup_steps N_w):
    (step - S + 1) / N_w for S <= step < S + N_w, S = loop_switch_step; None everywhere else and with N_w = 0 (no
    factor: the LR is the schedule's own number, not that number times 1.0)."""
    n, S = cfg.optim.switch_warmup_steps, cfg.loop_switch_step
    if not n or not S <= step < S + n:
        return None
    return (step - S + 1) / n


def lr_of(cfg: TrainConfig, step: int) -> float:
    """The LR of update `step` of the run `cfg` describes: lr_schedule over schedule_total at schedule_clock, times
    the ramp after a switch where one is set and still rising (switch_ramp)."""
    lr = lr_schedule(cfg.optim, schedule_total(cfg), schedule_clock(cfg, step))
    ramp = switch_ramp(cfg, step)
    return lr if ramp is None else lr * ramp


# The stop on a non-finite loss (module docstring). EXIT_DIVERGED is the exit status of scripts/train.py for a run that
# stopped this way (and for any later start in its run directory): not 0 (the run is not finished and is not scored),
# not 1 or 2 (a Python error, a usage error), not above 128 (a signal).
EXIT_DIVERGED = 86
DIVERGED_MARKER = "diverged.json"  # in the run directory; its presence stops every later fit() there
DIVERGED_EVENT = "diverged"  # the "event" of the record in metrics.jsonl
FINITE_KEYS = ("loss", "grad_norm")  # the step metrics that must be finite
# The step metrics that must be finite when a step logs them (absent: nothing changes). Add-on CP's head gradient norm
# (model.cp_head.train_mode "addon"; ced/model/cp_aux.py, "ADD-ON CP"): the head is clipped apart from the trunk, so grad_norm no
# longer holds the head's gradient; without this a non-finite head gradient would pass the step's check, the head's tensors would
# be NaN after the update and the step's checkpoint could keep them (cp-mix's one global norm stops at that step; review 2026-10-22).
FINITE_IF_LOGGED = ("cp_grad_norm",)
# What the marker keeps of the last finite steps (a short tail; the whole record is in metrics.jsonl) and of w.
RECENT_STEPS = 8
RECENT_KEYS = ("loss", "grad_norm", "lr", "w", "grad_norm_w", "lr_w")


BAD_BATCH_EVENT = "bad_batch"  # the "event" of the record of a run that a batch failing the loader's validation ended


class Diverged(RuntimeError):
    """A training step's loss or gradient norm was not finite: the run has ended (Trainer._check_finite), or its run
    directory holds the marker of a run that ended so. `record` is the marker's content."""

    def __init__(self, record: dict):
        super().__init__(record.get("message", "diverged"))
        self.record = record


class BadBatch(Diverged):
    """A training batch failed the loader's validation (ced/train/batch_check.py): the run has ended through the stop's
    path (Trainer._halt_bad_batch: the record {"event": "bad_batch"} in metrics.jsonl and in the marker, no checkpoint),
    so scripts/train.py exits with EXIT_DIVERGED and every later start in the run directory exits at once."""


def diverged_marker(run_dir: str | Path) -> dict | None:
    """The content of <run_dir>/diverged.json, or None when the run has no such marker. A marker that cannot be parsed
    still counts: the run stopped on a non-finite loss, and the file says no more than that."""
    path = Path(run_dir) / DIVERGED_MARKER
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
        assert isinstance(record, dict)
    except (OSError, ValueError, AssertionError):
        record = {}
    record.setdefault("message", f"diverged (the marker {path} could not be read)")
    return record


def _quiet(*args, **kwargs) -> None:
    """The log of every data-parallel rank but rank 0 (D4: rank 0 logs)."""


def _plain(x):
    """A metric as valid JSON: a finite number as it is, a non-finite float as its name ("nan", "inf", "-inf")."""
    if isinstance(x, float) and not math.isfinite(x):
        return repr(x)
    return x


# The halt path's pieces, shared by the Trainer and the MTP drafter's trainer (ced/train/mtp_train.py): one finite
# check, one record of each kind, one marker writer, one way to name a batch's row. Each Trainer method below that
# uses them does what it did before they were split out (tests/test_train_input_guards.py).


def nonfinite_metrics(metrics: dict) -> dict:
    """The step metrics among FINITE_KEYS, and among FINITE_IF_LOGGED when the step logs them, that are not finite (host numbers;
    empty: the step passes the finite check)."""
    bad = {k: metrics[k] for k in FINITE_KEYS if not math.isfinite(metrics[k])}
    bad.update({k: metrics[k] for k in FINITE_IF_LOGGED if k in metrics and not math.isfinite(metrics[k])})
    return bad


# The MXFP8 non-finite flag (FP8Config.nonfinite_flag; module docstring, "The non-finite flag"): its bits in the step's
# metrics (MX_FLAG_KEY, present only with the flag on) and its name in a halt record (MX_FLAG_NAME).
MX_FLAG_KEY = "mxfp8_nonfinite"
MX_FLAG_NAME = "mxfp8_nonfinite_flag"


def mx_flag_text(bits: int) -> str:
    """The flag's value in a halt record: its bits and the operands they name."""
    from ced.kernels.mx_chains import flag_text

    return f"0x{int(bits):x}: a block holding an inf or NaN in {flag_text(int(bits))}"


def diverged_record(k: int, bad: dict, metrics: dict, recent, latest: Path | None) -> dict:
    """The record of a run the finite check ends at update k (as metrics.jsonl counts it): the non-finite values (`bad`,
    nonfinite_metrics), the update's own numbers, the last finite updates (`recent`: their metrics rows, oldest first),
    the last finite step, the latest checkpoint and the exit status. The caller adds what it knows of the step (its
    batch, w)."""
    last = [{key: _plain(v) for key, v in row.items()} for row in recent]
    return {
        "event": DIVERGED_EVENT, "step": k, "time": time.time(),
        "message": f"diverged at step {k}: " + ", ".join(f"{name} {value!r}" for name, value in bad.items()),
        "nonfinite": {name: repr(value) for name, value in bad.items()},
        # the diverged update's own numbers (after it: w is what the update left), then the last finite updates
        "at": {key: _plain(metrics[key]) for key in RECENT_KEYS if key in metrics},
        "last_finite": last,
        # every update is checked before the next one runs, so the one before is the last finite (0: none was)
        "last_finite_step": k - 1,
        "latest_checkpoint": None if latest is None else latest.name,
        "exit_status": EXIT_DIVERGED,
    }


def bad_batch_record(error: BatchError, k: int, last_finite_step: int, latest: Path | None) -> dict:
    """The record of a run that a batch failing the loader's validation ends before update k (as metrics.jsonl counts
    it): the error's info (field, rule, row, position, value, and where the row is in the data stream), the last
    finite step (the updates made), the latest checkpoint and the exit status."""
    info = json.loads(json.dumps(error.info, default=repr))  # plain JSON whatever the info holds
    return {
        "event": BAD_BATCH_EVENT, "step": k, "time": time.time(),
        "message": f"bad batch at step {k}: {error}",
        "error": info,
        "last_finite_step": last_finite_step,
        "latest_checkpoint": None if latest is None else latest.name,
        "exit_status": EXIT_DIVERGED,
    }


def write_halt(run_dir: Path, record: dict, metrics_jsonl: bool = True) -> None:
    """The record of a run that ends on its own check (a non-finite loss or gradient norm, a bad batch): one line in
    <run_dir>/metrics.jsonl (when the run writes it) and the marker file <run_dir>/diverged.json, written atomically.
    Strict JSON (allow_nan False)."""
    text = json.dumps(record, allow_nan=False)
    if metrics_jsonl:
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(text + "\n")
    tmp = run_dir / (DIVERGED_MARKER + ".tmp")
    tmp.write_text(text + "\n")
    os.replace(tmp, run_dir / DIVERGED_MARKER)


def halt_message(kind: str, record: dict, run_dir: Path) -> str:
    """The log line of a halt ("DIVERGED" or "BAD BATCH")."""
    return (f"{kind}: {record['message']}. The run ends here: no checkpoint is written (the latest is "
            f"{record['latest_checkpoint']}), the marker is {run_dir / DIVERGED_MARKER}, exit status {EXIT_DIVERGED}")


def locate_batch_error(error: BatchError, step: int, src: dict) -> BatchError:
    """error with the update (as metrics.jsonl counts it, step + 1; step 0-based) and the failing row's place in the
    data stream added to its message and info. src: where the update's rows come from (a row_source dict, or {"error":
    ...} when it could not be described)."""
    info = {**error.info, "step": step + 1}
    info["source"] = src
    row = info.get("row")
    where = ""
    if row is not None and "first_row" in src:
        info["stream_row"] = src["first_row"] + row * src["bag"]
        where = (f"; row {row} is stream row {info['stream_row']} of {src['length']} tokens"
                 + (f" (a bag row over stream rows {info['stream_row']} .. {info['stream_row'] + src['bag'] - 1})"
                    if src["bag"] > 1 else "") + f", data seed {src['data_seed']}")
    elif row is not None and "data_step" in src:
        where = f"; synthetic row {row} of data step {src['data_step']}, data seed {src['data_seed']}"
    if row is not None and "batch_row0" in src:  # data parallelism: the rank's row r is row batch_row0 + r of the batch
        info["batch_row"] = src["batch_row0"] + row
        where += f" (rank {src['rank']} of {src['world_size']}: row {info['batch_row']} of the step's batch)"
    return BatchError(f"update {step + 1}: {error}{where}", info)


def micro_batch_error(error: BatchError, rows) -> BatchError:
    """A boundary error of a micro-batch (varlen_for_batch) with its row named as a row of the step's batch: rows is the
    micro-batch's rows of the batch (a slice or indices), or None (the error as it is)."""
    r = error.info.get("row")
    if r is not None and rows is not None:  # the micro-batch's row r as a row of the step's batch
        batch_row = (rows.start or 0) + r if isinstance(rows, slice) else int(rows[r])
        error = BatchError(f"{error} (row {r} of the micro-batch is row {batch_row} of the batch)",
                           {**error.info, "row": batch_row, "micro_batch_row": r})
    return error


def describe_rows(step: int, batch, row_source) -> dict:
    """What a halt record says of update `step`'s batch (step 0-based): its rows' source (row_source(step)), a content
    hash of every row (batch_check.row_hashes) and the first and last document id of every row. Host work, only when a
    run halts; a failure to describe is noted in the record, never raised."""
    out: dict = {}
    try:
        out["source"] = row_source(step)
    except Exception as e:  # noqa: BLE001
        out["source"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        out["hashes"] = row_hashes(batch)
        out["doc_ids"] = [[int(batch.doc_ids[b, 0]), int(batch.doc_ids[b, -1])] for b in range(batch.doc_ids.shape[0])]
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# The arches a gated second pass is built for (model.loop.pass2_gate; ced/model/control.py build_model refuses the
# others): CED v2's decoder at T = 2 ("hc" / "scalar") and the control's top block ("scalar" with loop.pass2_layers).
GATED_ARCHES = ("ced_v2", "control")

# What a nested budget's branch may change against the run it continues (TrainConfig.resume_from): the length of the
# schedule, the kept checkpoints, the pointer itself, and the sections that never touch the training state.
BRANCH_FIELDS = ("steps", "checkpoint_steps", "resume_from")
BRANCH_SECTIONS = ("eval", "wandb")

# A stage that continues another run on other data (TrainConfig.stage "continue"; module docstring, "A stage that
# continues another run"). STAGE_FREE: the dotted keys such a stage may set differently from its source besides
# BRANCH_FIELDS and BRANCH_SECTIONS. The shard set itself (its fingerprint) is checked by stage_data_relation.
STAGE_NONE, STAGE_CONTINUE = "none", "continue"
STAGES = (STAGE_NONE, STAGE_CONTINUE)
# data.loss_mask (the per-token loss mask): a stage on a NEW shard set may train with another mask than its source (the step-2
# lane's F8: masked arms from the unmasked r1e-mxl-a step 4,865); on the source's own set the stream goes on and so does its
# mask (stage_data_relation).
STAGE_FREE = ("optim.decay_fraction", "data.shard_dir", "stage", "data_start_step", "data.loss_mask", "stage_shape")
# A stage whose data.union_order is "sets_in_order" may also name the union: its source trained on the union's first set alone
# (data.extra_shard_dirs [] and union_order "permute" there). The data identity decides whether that is allowed (stage_data_relation:
# the relation "extends", or "same" for a source that is the union itself); a stage with any other order keeps both keys fixed, as
# every stage before: "a stage-continue run refuses a change of data.extra_shard_dirs".
STAGE_UNION_FREE = ("data.extra_shard_dirs", "data.union_order")
# A stage at another row shape (TrainConfig.stage_shape "tokens"; module docstring, "A stage at another row shape"): the
# keys it may also set differently from its source, at the source's tokens a step: the row length and the rows a step,
# and how a step is held in memory (the same pair TRUNK_FREE_FIELDS lets a loop branch change).
STAGE_SHAPE_SOURCE, STAGE_SHAPE_TOKENS = "source", "tokens"
STAGE_SHAPES = (STAGE_SHAPE_SOURCE, STAGE_SHAPE_TOKENS)
STAGE_SHAPE_FREE = ("data.seq_len", "data.batch_size", "micro_batch_size", "activation_checkpoint",
                    "model.cp_head.kernel")
# model.cp_head.kernel (the fused Copy-Paste head) is how the head computes, not what: a CP source trained at 8k with the
# reference kernel continues at a long row with the fused one (the reference's compiled block cannot run there). The same
# keys, the same function within float32 rounding. A stage that continues its source in another precision
# (TrainConfig.precision_switch, with stage "continue"; the NVFP4 tail arms' N2, N2-FP8 and N3): the dotted keys it may
# set differently from its source besides STAGE_FREE. Both subtrees turn the layers' nn.Linear into a subclass in place
# and add no state, so the source's model and optimizer state load into this run's model as they are.
PRECISION_FREE = ("model.nvfp4", "model.fp8", "precision_switch")


def branch_differences(source: dict, own: dict, prefix: str = "", free: tuple = ()) -> list[str]:
    """Dotted config keys whose values differ between a resume_from source and this run, apart from BRANCH_FIELDS
    and BRANCH_SECTIONS (top level only) and the dotted keys in `free` (STAGE_FREE under stage "continue"; () else)."""
    out = []
    for key in sorted(set(source) | set(own)):
        name = f"{prefix}{key}"
        if not prefix and (key in BRANCH_FIELDS or key in BRANCH_SECTIONS):
            continue
        if name in free:
            continue
        a, b = source.get(key), own.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            if name == "model.fp8":  # the flag's form as it is read (fp8_comparable)
                a, b = fp8_comparable(a), fp8_comparable(b)
            out += branch_differences(a, b, name + ".", free)
        elif a != b:
            out.append(f"{name}: source {a!r} -> this run {b!r}")
    return out


DATA_POSITION_KEYS = ("fingerprint", "data_start_step")  # what a stage may change in the data identity (data_state)
DATA_EXTENDS_KEY = "extends"  # data_state of a union read with union_order "sets_in_order": its first set's fingerprint (absent otherwise)
# The per-token loss mask's entries of the data identity (Trainer.data_state; absent with data.loss_mask "all"): a stage on
# a new shard set may change them, a stage on the source's set may not (stage_data_relation).
DATA_MASK_KEYS = ("loss_mask", "loss_spans")
DATA_SHAPE_KEYS = ("seq_len", "batch_size")  # what a stage at another row shape changes in it (stage_shape "tokens")


def stage_data_relation(saved: dict | None, own: dict, step: int, shape: bool = False) -> tuple[str, list[str]]:
    """Under stage "continue": how this run's data stream (its data identity, Trainer.data_state) relates to the
    source's (`saved`, the source checkpoint's data_state, written at its step `step`). Returns (relation, problems):
    relation "same" when the shard set is the source's (equal fingerprints; synthetic data has none and is "same"),
    "extends" when this run reads a union (data.union_order "sets_in_order") whose first epoch begins with the first epoch of the
    source's own shard set (the union's mark DATA_EXTENDS_KEY is the source's fingerprint: the stream goes on where the source is,
    as on its own set), "new" when it is another; problems is empty when the stage may continue the source, else the reasons it may not:
    - the data kind, the data seed, the row length, the rows a step or a shape schedule differ (only the shard set and
      the stream's start may change);
    - the shard set is the source's and data_start_step is not the source's (0 for every run before the key): the stream
      goes on where the source is, so that a stage C on the source's data is the uninterrupted run bit for bit; likewise for
      "extends", with data_start_step 0 in both;
    - the shard set is new and data_start_step is not `step`: the new stream starts at its beginning (row 0) at this
      run's first update, the source's step;
    - the per-token loss mask (DATA_MASK_KEYS: its mode and span files) differs on the source's own shard set (on a new
      set it may: a masked stage from an unmasked source).
    With `shape` (TrainConfig.stage_shape "tokens") the row length and the rows a step (DATA_SHAPE_KEYS) may differ too,
    and two more problems are checked: the tokens a step (rows x row length) must be the source's, and a shape change on
    the source's own shard set is refused (under chat_whole the epoch's layout depends on the row length, so the source's
    stream has no continuation in rows of another length; a reshaped stage reads a new set from its beginning)."""
    if saved is None:
        return "", ["the source checkpoint records no data identity (data_state): it cannot be checked"]
    same = saved.get("fingerprint") == own.get("fingerprint")
    # "extends" (data.union_order "sets_in_order"): this run reads a union whose first epoch begins with the first epoch of ONE set
    # (own["extends"], that set's fingerprint) and the source trained on exactly that set: its stream goes on where the source is
    # (DATA_EXTENDS_KEY is the union's mark; the source has none)
    extends = not same and own.get(DATA_EXTENDS_KEY) is not None and own.get(DATA_EXTENDS_KEY) == saved.get("fingerprint")
    # the mark is never reported as a difference (a union run vs its source without one; refused by name below when it cannot go on);
    # the loss mask is free on a new set only: on the source's set, or one that extends it, the stream goes on and so does its mask
    free = DATA_POSITION_KEYS + (DATA_EXTENDS_KEY,) + (DATA_MASK_KEYS if not same and not extends else ())
    if shape:
        free += DATA_SHAPE_KEYS
    a = {k: v for k, v in saved.items() if k not in free}
    b = {k: v for k, v in own.items() if k not in free}
    problems = [f"data {k}: source {a.get(k)!r} -> this run {b.get(k)!r} (a stage changes only the shard set and where "
                "its stream starts)" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    if shape:
        src_rows, src_len, rows, length = (saved.get("batch_size"), saved.get("seq_len"), own.get("batch_size"),
                                           own.get("seq_len"))
        if not all(isinstance(x, int) for x in (src_rows, src_len, rows, length)) or src_rows * src_len != rows * length:
            problems.append(f"stage_shape 'tokens' keeps the source's tokens a step: the source reads {src_rows} x {src_len}"
                            f" tokens a step, this run {rows} x {length}")
        if extends and (src_rows, src_len) != (rows, length):   # the union's first epoch begins with the source's stream, laid out in rows of the source's length
            problems.append(f"stage_shape 'tokens': the union extends the source's shard set, whose stream goes on in rows of {src_len} tokens (its "
                            f"epoch layout depends on the row length); rows of {length} need a new shard set, read from its beginning")
        if same and (src_rows, src_len) != (rows, length):
            problems.append(f"stage_shape 'tokens': the shard set is the source's, whose stream goes on in rows of "
                            f"{src_len} tokens (its epoch layout depends on the row length); rows of {length} need a new "
                            "shard set, read from its beginning")
    d_src, d_own = saved.get("data_start_step", 0), own.get("data_start_step", 0)
    if same and d_own != d_src:
        problems.append(f"the shard set is the source's, so its stream goes on where the source is: data_start_step must "
                        f"be the source's ({d_src}), not {d_own}")
    if extends:
        if d_src != 0 or d_own != 0:
            problems.append(f"the union extends the source's shard set (union_order 'sets_in_order'), so the source's stream goes on "
                            f"where it is: both runs need data_start_step 0 (the source's is {d_src}, this run's {d_own}); a source "
                            "whose stream started later is not built")
    elif not same and own.get(DATA_EXTENDS_KEY) is not None:
        problems.append(f"data.union_order 'sets_in_order' continues a source trained on the union's first set alone (fingerprint "
                        f"{own.get(DATA_EXTENDS_KEY)}) or on the union itself (fingerprint {own.get('fingerprint')}); the source's "
                        f"data is {saved.get('fingerprint')}. A union as a new set, read from its beginning, is not built")
    elif not same and d_own != step:
        problems.append(f"the shard set is new (fingerprint {saved.get('fingerprint')} -> {own.get('fingerprint')}): its "
                        f"stream starts at its beginning at this run's first update, so data_start_step must be the "
                        f"source's step {step}, not {d_own}")
    return ("same" if same else "extends" if extends else "new"), problems


def trunk_config(cfg: TrainConfig) -> TrainConfig:
    """The T = 1 trunk of a run with a loop switch step (TrainConfig.loop_switch_step; the loop experiment's part B):
    the same config with the single-pass model (ced.config.one_pass_model), no switch step, and the LR schedule's
    reference length as its step count (schedule_steps None; pass2_lr_scale 1.0: the trunk has no such tensor;
    switch_warmup_steps 0: the trunk has no switch to ramp after). Up
    to its step S the switched run is this run bit for bit, and its resume_from accepts this run's kept checkpoint
    at step S (Trainer.resume_from_checkpoint). For a looped middle with the scalar input mix (the late switch-on of
    2026-10-13) the single-pass model is the plain control (one_pass_model: no looped middle, injection "none").
    So the keys a branch may set against its trunk, besides the growth itself, BRANCH_FIELDS, BRANCH_SECTIONS (the
    eval section holds the readout's keys) and TRUNK_FREE_FIELDS, are exactly these: optim.schedule_steps,
    optim.pass2_lr_scale, optim.switch_warmup_steps and, for the scalar input mix, model.loop.recur_mix_init."""
    optim = dataclasses.replace(cfg.optim, schedule_steps=None, pass2_lr_scale=1.0, switch_warmup_steps=0)
    return dataclasses.replace(cfg, model=one_pass_model(cfg.model), optim=optim, loop_switch_step=0, steps=schedule_total(cfg))


# What a loop branch may set differently from its T = 1 trunk besides the growth itself (and BRANCH_FIELDS /
# BRANCH_SECTIONS): how a step is held in memory, not what it computes. A two-pass step needs more memory than the
# trunk's single-pass step (ced/model/ced_v2.py, "Two passes with self-KV"), so the branch may need a smaller
# micro-batch or layer checkpointing where the trunk did not. activation_checkpoint is the same training bit for bit
# (tests/test_long_context_grid.py, tests/test_loop_t2.py); micro_batch_size gives the same gradient up to the order of
# a sum over micro-batches. With both equal to the trunk's, the branch is bit for bit the run its config trains from
# its own start; with another value it is the trunk's exact state at the switch step, continued under that value.
# pass_slot_buckets (the segment pass schedule's slot sizes) is read by a scheduled branch only and changes no logit.
TRUNK_FREE_FIELDS = ("micro_batch_size", "activation_checkpoint", "pass_slot_buckets")
# Data parallelism (D5): what a resume may set differently from the checkpoint when the world size changes or the run is
# data-parallel, as a loop branch may (TRUNK_FREE_FIELDS' first two: how a step is held in memory). A rank's rows are
# data.batch_size / W, so another world size can need another micro-batch size; activation_checkpoint is bit for bit.
# Recorded in dp_info's history; scripts/train.py's prepare lets them differ there without --allow-config-change.
DP_RESUME_FREE = ("micro_batch_size", "activation_checkpoint", "optimizer_sharding")
# The keys a checkpoint's saved config leaves out at their default (state_dict), with that default: so a checkpoint of a
# run that does not set them is byte for byte what it was before the key.
SAVED_CONFIG_DEFAULTS = {"optimizer_sharding": "replicated", "ckpt_async": False, "data_prefetch": False, "freeze_trunk": False}
# The same in the model's section: model.fused_projections at its default () is not saved, so every checkpoint without
# it is byte for byte what it was.
SAVED_MODEL_DEFAULTS = {"fused_projections": ()}
# The same for keys under model.fp8: absent at their default, so a checkpoint of a run without them is byte for byte
# main's.
SAVED_FP8_DEFAULTS = {"folds": ()}
# The same for keys under loss (loss.ce_form): absent at "standard", so a checkpoint of a run that does not use the option is byte
# for byte main's (tests/test_loss_mask.py, test_train_input_guards.py, test_mtp_guards.py and test_main_4b_config.py compare the
# saved config key by key).
SAVED_LOSS_DEFAULTS = {"ce_form": "standard"}
# The same for keys under data (data.union_order): absent at "permute", so a checkpoint of a run that does not use the option is byte
# for byte main's.
SAVED_DATA_DEFAULTS = {"union_order": "permute"}
# The same for keys under model.encoder.attention and model.decoder.attention (attention.row_term): absent at "stored_o", so a checkpoint of a run
# that does not use the option is byte for byte main's. The same for the grid KV sharing keys (attention.kv_sharing and its five grid keys): absent at
# "gqa", 0 and heads_per_cell 1. The third axis' two keys (attention.grid_depth and depth_dim) are absent at 0: a checkpoint of any run without the
# third axis is byte for byte what it was.
SAVED_ATTENTION_DEFAULTS = {"row_term": "stored_o", "kv_sharing": "gqa", "grid_rows": 0, "grid_cols": 0, "row_dim": 0, "col_dim": 0,
                            "heads_per_cell": 1, "grid_depth": 0, "depth_dim": 0}
# The same for model.encoder.write_mix_row_term (CroissAttn fix for the fused write mix) and for model.encoder.residual_dtype.
SAVED_ENCODER_DEFAULTS = {"write_mix_row_term": "stored_o", "residual_dtype": "float32"}


def saved_config_dict(cfg: TrainConfig) -> dict:
    """The config a checkpoint saves: to_dict(cfg) without the keys of SAVED_CONFIG_DEFAULTS (and the model's
    SAVED_MODEL_DEFAULTS, SAVED_FP8_DEFAULTS, the loss section's SAVED_LOSS_DEFAULTS and the attention sections'
    SAVED_ATTENTION_DEFAULTS) at their defaults (from_dict gives them back)."""
    out = to_dict(cfg)
    loss = out.get("loss")
    if isinstance(loss, dict):
        for key, default in SAVED_LOSS_DEFAULTS.items():
            if loss.get(key) == default:
                del loss[key]
    data = out.get("data")
    if isinstance(data, dict):
        for key, default in SAVED_DATA_DEFAULTS.items():
            if data.get(key) == default:
                del data[key]
    for section in ("encoder", "decoder"):
        attn = ((out.get("model") or {}).get(section) or {}).get("attention")
        if isinstance(attn, dict):
            for key, default in SAVED_ATTENTION_DEFAULTS.items():
                if attn.get(key) == default:
                    del attn[key]
    encoder = (out.get("model") or {}).get("encoder")
    if isinstance(encoder, dict):
        for key, default in SAVED_ENCODER_DEFAULTS.items():
            if encoder.get(key) == default:
                del encoder[key]
    for key, default in SAVED_CONFIG_DEFAULTS.items():
        if out.get(key) == default:
            del out[key]
    model = out.get("model")
    if isinstance(model, dict):
        for key, default in SAVED_MODEL_DEFAULTS.items():
            if key in model and tuple(model[key]) == default:
                del model[key]
    fp8 = (out.get("model") or {}).get("fp8")
    if isinstance(fp8, dict):
        for key, default in SAVED_FP8_DEFAULTS.items():
            if key in fp8 and tuple(fp8[key]) == default:
                del fp8[key]
    return out


def trunk_differences(source: TrainConfig, own: TrainConfig, free: tuple = ()) -> tuple[list[str], dict]:
    """(differences, memory changes) between a resume_from source and the T = 1 trunk of the switched run `own`
    (trunk_config): the dotted keys that differ apart from BRANCH_FIELDS, BRANCH_SECTIONS, TRUNK_FREE_FIELDS and `free`
    (STAGE_FREE for a loop branch that is also a stage, TrainConfig.stage "continue"), and {field: [the trunk's value,
    this run's]} for the TRUNK_FREE_FIELDS that differ."""
    a, b = to_dict(source), to_dict(trunk_config(own))
    changes = {f: [a.get(f), b.get(f)] for f in TRUNK_FREE_FIELDS if a.get(f) != b.get(f)}
    for f in TRUNK_FREE_FIELDS:
        a.pop(f, None), b.pop(f, None)
    return branch_differences(a, b, free=free), changes


def growth_problems(source_model, own_model, model: torch.nn.Module) -> list[str]:
    """Why growing the single-pass model `source_model` (a ModelConfig) into the gated two-pass `own_model`, built as
    `model`, would change its function; [] when it would not ("With alpha=0, the model at the start of phase B is
    exactly the phase-A model"). Checked:
    - the residual scales are the same (a residual_scale.t_ref of None follows the loop count, and v2's T = 2
      config sets 2: either would run the shared block at half its residual scale from the switch on);
    - pass 2 is the identity as built: every pass-2 write weight (or the gate) is 0 and its mix is the identity."""
    from ced.model.scaling import residual_scales

    problems = []
    a, b = residual_scales(source_model), residual_scales(own_model)
    if a != b:
        problems.append(f"the (encoder, decoder) residual scales change from {a} to {b}: set model.residual_scale.t_ref "
                        "to the single-pass model's loop count (1), so that the block stays the same function")
    if not model.hyper_connections.is_identity():
        problems.append("pass 2 is not at its function-preserving init (write / gate 0, mix identity)")
    return problems


def monitoring_metrics(stats: dict) -> dict:
    """Step metrics from a model's monitoring stats (one host sync): loop_usage = mean over valid slots
    of ||z_T - z_{T-1}|| / ||z_T|| (CED v2, T >= 2), and the max |activation| of the residual stream
    at the input of each global layer, overall
    (max_act_pre_global) and per layer (act_max_pre_global, keyed "enc<i>" for encoder layers,
    "dec<i>.loop<k>" for CED v2's decoder layers, "pass2.enc<i>" for control (c)'s pass 2; see
    ced/model/encoder.py, record_activation_max). With the depth mix at the loop boundary (R3;
    ced/model/recur_depth.py) also "depth_mix": {"pass<k>": [the mean weight of each candidate of pass k over the
    forward's positions]}; no model without it has the key."""
    out = {}
    mix = recur_depth.stats_weights(stats)
    if mix:
        out["depth_mix"] = {k: [round(x, 6) for x in v] for k, v in mix.items()}
    if "loop_usage" in stats:
        out["loop_usage"] = float(stats["loop_usage"])
    if "loop_usage_pass3" in stats:  # a third pass (the segment pass schedule), over the slots that ran it
        out["loop_usage_pass3"] = float(stats["loop_usage_pass3"])
    acts = {k.split("/", 1)[1]: float(v) for k, v in stats.items() if k.startswith("act_max/")}
    if acts:
        out["max_act_pre_global"] = max(acts.values())
        out["act_max_pre_global"] = acts
    return out


def entmax_module_name(name: str) -> str:
    """Short names of attention modules in the "entmax" metrics, as the diagnostics': "enc<i>" (encoder layer i's
    global attention), "dec<i>" (CED v2 decoder layer i's read); any other module keeps its full name."""
    parts = name.split(".")
    if len(parts) == 4 and parts[0] == "encoder" and parts[1] == "layers" and parts[3] == "mixer":
        return f"enc{parts[2]}"
    if len(parts) == 3 and parts[0] == "decoder_layers" and parts[2] == "attn":
        return f"dec{parts[1]}"
    return name


def attach_entmax_stats(model: torch.nn.Module) -> list:
    """An EntmaxStats on every asentmax attention module of the model (GatedAttention.entmax_stats);
    [(short name, stats)]. Empty for softmax models."""
    from ced.model.attention import GatedAttention
    from ced.model.entmax import EntmaxStats

    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, GatedAttention) and mod.cfg.normalizer == "asentmax":
            mod.entmax_stats = EntmaxStats(mod.cfg.n_q_heads, mod.cfg.n_kv_heads)
            out.append((entmax_module_name(name), mod.entmax_stats))
    return out


def entmax_metrics(entmax: list) -> dict:
    """The log step's ASEntmax statistics (one host sync per module; ced/model/entmax.py EntmaxStats.summary):
    "entmax": {module: summary} plus overall scalars: entmax_support (mean support size over every module's
    heads, the log step's snapshot), entmax_collapsed (the share of all heads whose mean support <= 2),
    entmax_dead (mean over query heads of the window-mean dead-key share), entmax_dead_max (its largest window
    max), entmax_dead_kv (the same mean per KV head: a key dead for every query head of its group, i.e. one that
    gets no gradient in that layer; equal to entmax_dead without GQA)."""
    per = {name: st.summary() for name, st in entmax}
    means = [v for s in per.values() for v in s.get("support", {}).get("mean", [])]
    dead = [v for s in per.values() for v in s.get("dead_keys", {}).get("mean", [])]
    dead_max = [v for s in per.values() for v in s.get("dead_keys", {}).get("max", [])]
    dead_kv = [v for s in per.values() for v in s.get("dead_keys", {}).get("kv_heads", s.get("dead_keys", {})).get("mean", [])]
    return {"entmax": per,
            "entmax_support": sum(means) / len(means) if means else None,
            "entmax_collapsed": sum(m <= 2 for m in means) / len(means) if means else None,
            "entmax_dead": sum(dead) / len(dead) if dead else None,
            "entmax_dead_max": max(dead_max) if dead_max else None,
            "entmax_dead_kv": sum(dead_kv) / len(dead_kv) if dead_kv else None}


# The keys a config may leave unset (null) on purpose until they are chosen, and that no run trains without: the run's
# length (steps; the LR schedule is read over it), the warm-up, the kept checkpoints and the data. Every config before
# this check either set them or failed later on the None (a comparison with None, ShardSet(None)); only the message is
# new.
UNSET_CHOICES = ("steps", "optim.warmup_steps", "checkpoint_steps", "data.shard_dir")


def unset_choices(cfg: TrainConfig) -> list[str]:
    """The keys of UNSET_CHOICES that `cfg` leaves unset (None), in that order. data.shard_dir counts with data.kind
    "shards" alone, and not when data.extra_shard_dirs names sets (check_extra_shard_dirs refuses that case by name)."""
    values = {"steps": cfg.steps, "optim.warmup_steps": cfg.optim.warmup_steps, "checkpoint_steps": cfg.checkpoint_steps,
              "data.shard_dir": (cfg.data.shard_dir if cfg.data.kind == "shards" and not cfg.data.extra_shard_dirs else "")}
    return [k for k in UNSET_CHOICES if values[k] is None]


class Trainer:
    _mx_flag = None  # the MXFP8 non-finite flag (__init__ sets it with FP8Config.nonfinite_flag); None: off
    _nvfp4_until = 0  # NVFP4Config.until_step (__init__ sets it): the update at which the quantised path ends; 0: it never does
    _nvfp4_on = True  # whether the quantised path is on (train_step flips it at update _nvfp4_until, and after a resume past it)

    def __init__(self, cfg: TrainConfig, run_dir: str | Path, dp=None):
        unset = unset_choices(cfg)
        if unset:  # before anything is built (UNSET_CHOICES)
            raise ValueError(f"not set: {', '.join(unset)}. This config leaves them unset (null) until they are chosen: "
                             "set each one (scripts/train.py --set KEY=VALUE, or in the config) before training")
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        # Data parallelism (ced/train/data_parallel.py; module docstring, "Data parallelism"): this process's rank, world
        # size and process groups, or None (one process: every line below is what it was). The rank's device is its own.
        self.dp = dp
        self.device = resolve_device(cfg.device) if dp is None else dp.device
        self.dtype = getattr(torch, cfg.dtype)
        freeze.check_config(cfg, dp is not None)  # TrainConfig.freeze_trunk (ced/train/freeze_trunk.py): its refusals, by name
        if cfg.head_impl == "quack_lce" and "flashqla" in (cfg.model.encoder.gdn.chunk_backend,
                                                           cfg.model.decoder.gdn.chunk_backend):
            # D (GDNConfig.chunk_backend; sections 6 and 13): QuACK loads from FA4's pinned set, FlashQLA from its own,
            # and one process imports one tvm_ffi: built only while the two sets install the same apache-tvm-ffi (and
            # torch-c-dlpack-ext) wheel (one pin, 0.1.12, since 2026-10-18) and are the two sets tested together;
            # refused by name otherwise (gdn_flashqla.pair_refusal)
            from ced.kernels.gdn_flashqla import pair_refusal

            mismatch = pair_refusal()
            if mismatch is not None:
                raise ValueError(f"head_impl 'quack_lce' is not built together with GDN chunk_backend 'flashqla' while "
                                 f"{mismatch} (QuACK loads from FA4's pinned set); use another head_impl, or chunk_backend "
                                 "'triton'")
        if cfg.head_impl == "quack_lce":  # QuACK's bf16 linear-CE (ced/train/quack_head.py): its refusals, by name
            check_quack_head(cfg, self.device)
        elif cfg.head_impl in PORTABLE_HEADS:  # Liger's / CCE's fused linear-CE (ced/train/portable_head.py): its refusals
            check_portable_head(cfg, self.device)
        elif cfg.head_impl not in ("chunked", "fused") or (cfg.head_impl == "fused" and not cfg.head_select):
            raise ValueError(f"head_impl {cfg.head_impl!r}: 'chunked', or 'fused' with head_select")
        check_head_precision(cfg.head_precision)  # every head of the loss reads it, with either head_impl
        check_ce_form(cfg)  # loss.ce_form "fixed": the heads it covers; the others refused by name (ced/train/ce_fix.py)
        check_head_chunk(cfg)  # the fused head's chunk (TrainConfig.head_chunk): head_impl "fused" reads it
        if cfg.trim_decoder_slots % cfg.model.decoder.capacity_multiple:
            raise ValueError("trim_decoder_slots must be a multiple of decoder.capacity_multiple")
        # Trimming is a Looped-CED (v1) option. The v2 slot models run exactly C slots (random
        # positions) and refuse it; a control has no decoder slots, so for it the setting is a no-op,
        # as it always was in v1 (the v1 GPU checks pass the fast-path settings to both models).
        if cfg.trim_decoder_slots and cfg.model.arch in ("ced_v2", "second_pass"):
            raise ValueError("trim_decoder_slots is a Looped-CED (v1) option")
        if cfg.decoder_opt not in ("none", "exact", "compiled"):
            raise ValueError(f"decoder_opt {cfg.decoder_opt!r}: 'none', 'exact' or 'compiled'")
        if cfg.decoder_opt != "none" and cfg.model.arch != "ced_v2":
            raise ValueError(f"decoder_opt {cfg.decoder_opt!r} is CED v2's decoder fast path, not for arch {cfg.model.arch!r}")
        if cfg.decoder_opt != "none" and cfg.model.decoder.attention.rope_max_distance is not None:
            # the fast path's shared BlockMasks and its memory glue do not carry the rotary distance cap (ced_v2.py)
            raise ValueError(f"decoder_opt {cfg.decoder_opt!r} does not implement model.decoder.attention.rope_max_distance: "
                             "use decoder_opt 'none'")
        # A batch-shape schedule (module docstring): the validated phases, or None (no schedule: nothing below that
        # names it runs). total_steps: the optimizer steps of the run, cfg.steps without a schedule.
        self.shape = ShapeSchedule.from_config(cfg)
        self.total_steps = cfg.steps if self.shape is None else self.shape.total_steps
        if self.shape is not None:
            self._check_shape_schedule()
        # The LR of a scheduled run's steps (TrainConfig.shape_lr): "rows" has no meaning without phases.
        if cfg.shape_lr not in SHAPE_LR_MODES:
            raise ValueError(f"shape_lr {cfg.shape_lr!r}: one of {SHAPE_LR_MODES}")
        if cfg.shape_lr != "clock" and self.shape is None:
            raise ValueError(f"shape_lr {cfg.shape_lr!r} scales the LR of a shape_schedule's phases with their rows: this config "
                             "has no shape_schedule (every step is the config's own, and its LR the schedule's): leave it 'clock'")
        kept = cfg.checkpoint_steps
        if not isinstance(kept, list) or any(type(k) is not int or not 0 < k <= self.total_steps for k in kept):
            raise ValueError(f"checkpoint_steps {kept!r}: a list of steps s with 0 < s <= steps ({self.total_steps})")
        self._kept_steps = frozenset(kept)  # checkpoints written exactly there and never pruned
        # The loop experiment: the switch step, the schedule's reference length, pass 2's LR multiplier.
        S, N = cfg.loop_switch_step, cfg.optim.schedule_steps
        # a gated second pass: CED v2's decoder at T = 2, or the control's top block (the looped control)
        gated = cfg.model.arch in GATED_ARCHES and cfg.model.loop.pass2_gate != "none"
        if type(S) is not int or not 0 <= S < cfg.steps:
            raise ValueError(f"loop_switch_step {S!r}: an optimizer step S with 0 <= S < steps ({cfg.steps})")
        # a looped middle whose later passes read the scalar input mix (the late switch-on of 2026-10-13): the one
        # form of the loop in the published form that switches on (module docstring)
        mix = recur_mix(cfg.model)
        if S and not (gated or mix):
            raise ValueError(f"loop_switch_step {S} switches a gated second pass on (model.loop.pass2_gate 'hc' or "
                             "'scalar' on arch 'ced_v2'; 'scalar' with pass2_layers on arch 'control'): this model has "
                             "none (nor a looped middle with the scalar input mix, model.loop.recur_layers with "
                             "injection 'mix', the other model that takes a switch step)")
        if S and mix and cfg.model.loop.t_sampling != "step":
            raise ValueError(f"loop_switch_step {S} with model.loop.t_sampling {cfg.model.loop.t_sampling!r}: a looped "
                             "middle switches on with one pass count per step ('step'); a count per row is not built "
                             "with a switch step")
        if N is not None and (type(N) is not int or N <= S):
            raise ValueError(f"optim.schedule_steps {N!r}: None, or the schedule's length in reference steps, an integer "
                             f"greater than loop_switch_step ({S})")
        if not cfg.optim.pass2_lr_scale > 0:
            raise ValueError(f"optim.pass2_lr_scale {cfg.optim.pass2_lr_scale!r}: a positive multiplier")
        # The LR ramp after the switch (optim.switch_warmup_steps; module docstring): refused where it would ramp nothing
        # (no switch step) and where it would not end inside the run.
        ramp = cfg.optim.switch_warmup_steps
        if type(ramp) is not int or ramp < 0:
            raise ValueError(f"optim.switch_warmup_steps {ramp!r}: 0 (no ramp) or a number of updates after loop_switch_step")
        if ramp and not S:
            raise ValueError(f"optim.switch_warmup_steps {ramp} ramps the LR over the first updates after a loop switch step: "
                             "this config has none (loop_switch_step 0): leave it 0")
        if ramp > cfg.steps - S:
            raise ValueError(f"optim.switch_warmup_steps {ramp}: the run has {cfg.steps - S} updates after its switch step "
                             f"({S}), fewer than the ramp")
        # The readout inside training (eval.readout_every; ced/train/readout.py): the keys and what it is built for,
        # checked here; the sample's files are read at the start of fit().
        from ced.train.readout import check_readout_config

        check_readout_config(cfg)
        # The bpb readout inside training (eval.bpb_readout_every; ced/train/bpb_readout.py): its keys, checked here; the
        # sets are read at the start of fit().
        from ced.train.bpb_readout import check_config as check_bpb_readout

        check_bpb_readout(cfg, self.total_steps)
        # A stage that continues another run (TrainConfig.stage, data_start_step; module docstring): the keys, and what
        # the opt-in is built for. The source itself is checked by resume_from_checkpoint.
        if cfg.stage not in STAGES:
            raise ValueError(f"stage {cfg.stage!r}: one of {STAGES}")
        D = cfg.data_start_step
        if type(D) is not int or not 0 <= D < cfg.steps:
            raise ValueError(f"data_start_step {D!r}: the optimizer step at which this run's data stream starts, an "
                             f"integer with 0 <= D < steps ({cfg.steps})")
        if D and cfg.stage != STAGE_CONTINUE:
            raise ValueError(f"data_start_step {D} starts the data stream at a later step: that is a stage that continues "
                             "another run (stage 'continue' with resume_from); leave it 0")
        if cfg.stage == STAGE_CONTINUE:
            if not cfg.resume_from:
                raise ValueError("stage 'continue' continues another run's training state: set resume_from (the source's "
                                 "kept checkpoint)")
            if cfg.shape_schedule:
                raise ValueError("stage 'continue' is not built together with shape_schedule")
        # A stage at another row shape (TrainConfig.stage_shape; module docstring, "A stage at another row shape"): the key,
        # and what it is built for. The source's shape and tokens a step are checked by resume_from_checkpoint.
        if cfg.stage_shape not in STAGE_SHAPES:
            raise ValueError(f"stage_shape {cfg.stage_shape!r}: one of {STAGE_SHAPES}")
        if cfg.stage_shape == STAGE_SHAPE_TOKENS:
            what = "stage_shape 'tokens'"
            if cfg.stage != STAGE_CONTINUE:
                raise ValueError(f"{what} changes the row shape of a stage that continues another run: set stage "
                                 f"'continue' (this config has stage {cfg.stage!r})")
            if cfg.tst.schedule:
                raise ValueError(f"{what} is not built together with tst.schedule (TST's bag phases place the steps in the "
                                 "stream by the row length)")
            if cfg.loop_switch_step:
                raise ValueError(f"{what} is not built together with loop_switch_step {cfg.loop_switch_step} (a loop branch's "
                                 "source is its T = 1 trunk, at the trunk's shape)")
            if cfg.model.arch in SLOT_ARCHES:
                raise ValueError(f"{what} is not built for arch {cfg.model.arch!r}: its decoder capacity, slots and aux "
                                 "positions follow the row length")
            if cfg.data.fit_decoder_capacity:
                raise ValueError(f"{what} is not built together with data.fit_decoder_capacity (its layout fits the "
                                 "assistant tokens of rows of the source's length)")
            if cfg.model.cp_head.enabled and cfg.model.cp_head.kernel != "fused":
                raise ValueError(f"{what} is not built together with model.cp_head kernel 'reference' (the CP terms run "
                                 "one query block shape per model.cp_head.query_chunk of the row: 64 at 32,768 / 512, over "
                                 "the trainer's recompile limit of 32 for the compiled block; model.cp_head.kernel 'fused' "
                                 "runs every row length as one kernel)")
        # The per-token loss mask (DataConfig.loss_mask; module docstring, "The per-token loss mask"): the mode, and what it
        # is not built for, by name. The set's own invariants are checked by the shard packer when it opens the set.
        mask = cfg.data.loss_mask
        if mask not in LOSS_MASK_MODES:
            raise ValueError(f"data.loss_mask {mask!r}: one of {LOSS_MASK_MODES}")
        if mask != LOSS_MASK_ALL:
            if cfg.data.kind != "shards":
                raise ValueError(f"data.loss_mask {mask!r} reads the shard set's assistant flags or loss-span classes: "
                                 f"data.kind {cfg.data.kind!r} has none (leave it 'all'; a scorer that switches to synthetic "
                                 "data resets it, scripts/rescore.py)")
            if cfg.tst.schedule:
                raise ValueError(f"data.loss_mask {mask!r} is not built together with tst.schedule (TST builds its own "
                                 "batches with several targets a position)")
            if cfg.data.fit_decoder_capacity:
                raise ValueError(f"data.loss_mask {mask!r} is not built together with data.fit_decoder_capacity")
        # A stage in another precision (TrainConfig.precision_switch; PRECISION_FREE): only on a stage continuation.
        if type(cfg.precision_switch) is not bool:
            raise ValueError(f"precision_switch {cfg.precision_switch!r}: true or false")
        if cfg.precision_switch and cfg.stage != STAGE_CONTINUE:
            raise ValueError("precision_switch lets a stage continue its source in another precision (model.nvfp4, "
                             "model.fp8): it is built for stage 'continue' with resume_from only")
        if cfg.precision_switch and cfg.model.cp_head.enabled:
            raise ValueError("precision_switch is not built together with model.cp_head (the head is built in BF16 only: "
                             "model.nvfp4 and model.fp8 are refused with it, so a CP run has no other precision to switch to)")
        # NVFP4 for the first until_step updates only (NVFP4Config.until_step): inside the run's steps.
        until = cfg.model.nvfp4.until_step
        if until and not 0 < until < cfg.steps:
            raise ValueError(f"model.nvfp4.until_step {until!r}: the update at which NVFP4 ends, inside the run (0 < until_step < steps "
                             f"{cfg.steps})")
        self._nvfp4_until = until if cfg.model.nvfp4.mode != "none" else 0
        # Segment-based pass counts (module docstring): the validated schedule, or None (no schedule: nothing below
        # that names it runs).
        self.schedule = pass_schedule.schedule_of(cfg.model.loop)
        if self.schedule is not None and not (gated and (looped_control(cfg.model) or cfg.model.decoder.self_kv)):
            raise ValueError("model.loop.segment_passes / unscored_passes are built for the gated two-pass decoder with "
                             "self-KV (arch 'ced_v2', model.decoder.self_kv, model.loop.pass2_gate 'hc' or 'scalar') and "
                             "for the control with a second pass through its top block (arch 'control', "
                             "model.loop.pass2_gate 'scalar' with pass2_layers)")
        buckets = cfg.pass_slot_buckets
        if not isinstance(buckets, list) or any(isinstance(f, bool) or not isinstance(f, (int, float)) or not 0 < f <= 1
                                                for f in buckets):
            raise ValueError(f"pass_slot_buckets {buckets!r}: a list of fractions in (0, 1]")
        # With decoder self-KV the loop count of a training step is fixed ("T=1 in pretraining phase A; T=2 from
        # phase B on"): a count drawn per step is not part of the spec and is refused. T = 1 before a switch is
        # loop_switch_step.
        loop = cfg.model.loop
        if cfg.model.arch == "ced_v2" and cfg.model.decoder.self_kv and list(loop.t_choices) != [loop.t_max]:
            raise ValueError(f"model.loop.t_choices {loop.t_choices!r} with decoder self-KV: the loop count is fixed, "
                             f"[{loop.t_max}] (loop.t_max); T = 1 before a switch-on is loop_switch_step")
        # The looped control likewise trains at a fixed T = 2 (the approved arms: pass 2 at every position from the
        # switch step on); a count drawn per step is not part of the proposal and is refused.
        if looped_control(cfg.model) and list(loop.t_choices) != [2]:
            raise ValueError(f"model.loop.t_choices {loop.t_choices!r} on the control with a second pass (loop.pass2_gate): "
                             "the loop count is fixed, [2]; T = 1 before a switch-on is loop_switch_step")
        # The loop in the published form (module docstring): the rows' pass counts of a step under t_sampling
        # "sequence" (the multiset, in t_choices order: None otherwise), and the micro-batch size by pass count.
        self.recur = recur_loop(cfg.model)
        self._row_multiset = None
        if loop.t_sampling == "sequence":
            if not self.recur:
                raise ValueError("model.loop.t_sampling 'sequence' is built for the loop in the published form "
                                 "(model.loop.recur_layers on arch 'control')")
            B, multiset = cfg.data.batch_size, []
            for t, prob in zip(loop.t_choices, loop.t_probs):
                rows = prob * B
                if abs(rows - round(rows)) > 1e-9 or round(rows) < 0:
                    raise ValueError(f"model.loop.t_sampling 'sequence': t_probs {loop.t_probs!r} x data.batch_size {B} must "
                                     f"be whole numbers of rows (T = {t}: {rows})")
                multiset += [t] * round(rows)
            if len(multiset) != B:
                raise ValueError(f"model.loop.t_sampling 'sequence': t_probs {loop.t_probs!r} give {len(multiset)} rows of "
                                 f"data.batch_size {B} (they must sum to 1)")
            self._row_multiset = multiset
        by_t = cfg.micro_batch_by_t
        if not isinstance(by_t, list) or any(type(m) is not int or not 1 <= m <= cfg.data.batch_size for m in by_t):
            raise ValueError(f"micro_batch_by_t {by_t!r}: a list of rows per forward by pass count (entry T - 1), each "
                             f"1 .. data.batch_size ({cfg.data.batch_size})")
        if by_t and not self.recur:
            raise ValueError("micro_batch_by_t is read with the loop in the published form (model.loop.recur_layers on "
                             "arch 'control'); set micro_batch_size")
        self.autocast_dtype = None if cfg.autocast == "none" else getattr(torch, cfg.autocast)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
            torch.backends.cudnn.allow_tf32 = cfg.tf32
        if cfg.deterministic:
            torch.use_deterministic_algorithms(True)
            # process-wide, like the setting above (TrainConfig.fill_uninitialized_memory)
            torch.utils.deterministic.fill_uninitialized_memory = cfg.fill_uninitialized_memory

        # The model is built and initialised on the CPU, so the initial weights do not depend on the
        # device; then moved.
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        if cfg.model.mtp.enabled:  # the frozen-trunk MTP drafter (ced/model/mtp.py) has its own training script
            raise ValueError("model.mtp.enabled: the MTP drafter is trained on a frozen trunk by scripts/train_mtp.py; "
                             "this trainer would train the trunk and leave the head without a loss")
        self.model = build_model(cfg.model).to(self.dtype).to(self.device)
        if cfg.model.fused_projections:  # one GEMM a shared input (ced/model/fused_proj.py); () (the default): not entered
            from ced.model import fused_proj

            fused_proj.check_train_config(cfg)
            fused_proj.seat(self.model)  # moving the model gave each parameter a storage of its own: one buffer a group again
        if cfg.model.fp8.mode != "none" and cfg.model.fp8.backend == "te":  # Transformer Engine's MXFP8:
            from ced.model.fp8_te import check_te_backend  # refused by name without TE and below sm100

            check_te_backend(self.device)
        # The MXFP8 non-finite flag (FP8Config.nonfinite_flag; module docstring, "The non-finite flag"): None when off
        self._mx_flag = None
        if cfg.model.fp8.mode != "none" and cfg.model.fp8.nonfinite_flag:
            from ced.kernels import mx_ops

            self._mx_flag = mx_ops.flag_for(self.device)
        if cfg.model.encoder.attention.backend == "fa3":  # FA3 has no deterministic backward above head dimension 128
            from ced.kernels.flash_capped import check_deterministic

            check_deterministic(cfg.model.encoder.attention.head_dim, cfg.deterministic)
        if cfg.model.encoder.attention.backend == "fa4":  # FA4's backward is deterministic only with fa4_deterministic
            from ced.kernels.flash_fa4 import check_deterministic as check_fa4_deterministic
            from ced.kernels.flash_fa4 import device_capability
            from ced.model.control import check_fa4_device

            check_fa4_deterministic(cfg.model.encoder.attention.fa4_deterministic, cfg.deterministic)
            # FA4 4.0.0b35 has no windowed backward at head dimension 256 on SM100 / SM110 (B200): a capped layer's near
            # call there is refused by name before any step; the rope scales are TST's token-scale rotary's bag sizes
            # (the near window cap // s), else 1
            tst = cfg.tst
            scales = tuple(s for start, s in tst.schedule if start < cfg.steps) if tst.token_scale_rope else ()
            check_fa4_device(cfg.model, device_capability(self.device), cfg.data.seq_len, scales or (1,))
        if cfg.optim.gdn_mup:  # the gates' init with their LR (OptimConfig.gdn_mup; ced/train/optim.py gdn_mup_init_)
            gdn_mup_init_(self.model, cfg.optim.gdn_mup_base_width)
        arch = cfg.model.arch
        if arch == "ced_v2":  # the decoder fast path (TrainConfig.decoder_opt; ced/model/ced_v2.py): a plain attribute
            self.model.decoder_opt = cfg.decoder_opt
        self.is_v1 = arch == "looped_ced"  # v1-only paths: decoder GDN varlen, trimmed slots, drafts, MoE
        self.has_slots = arch in SLOT_ARCHES  # two heads: the main head at the slots, the aux head
        self.is_looped = has_loop(cfg.model)  # a loop count T per step (the looped arches; the control with a second pass)
        self.optimizer = build_optimizer(self.model, cfg.optim, total_steps=self.total_steps)
        # Our optimizer's read-only logs (OptimConfig.ours): the measured rotation of every Muon-group matrix under any
        # optimizer (rotation_log_every), grouped by the invariance map's roles; the optimizer's own diagnostics
        # (log_every). Both off by default: nothing is copied or read.
        self._rot_params = None
        if cfg.optim.ours.rotation_log_every:
            from ced.model.invariance import scale_invariance_map

            inv = scale_invariance_map(self.model)
            self._rot_params = [(inv[n]["role"], p) for n, p in self.model.named_parameters() if is_muon_param(n, p)]
        # Every arm uses the same capacity formula, so with the same seed the control's data has the
        # same decoder positions and its CE is reported at the same positions as the slot models'.
        d = cfg.model.decoder
        # _model_capacity: the slots the model runs by default (its config); capacity: the slots training runs, the same
        # except with answer spans (DataConfig.decoder_positions "answer_spans", stage 3), whose capacity comes from the
        # data config so that the model config stays the pretrained model's; it is then passed to the model explicitly.
        self._model_capacity = decoder_capacity(cfg.data.seq_len, d.capacity_fraction, d.capacity_multiple)
        self.capacity = self._model_capacity
        if cfg.data.decoder_positions == "answer_spans":
            if not 0.0 < cfg.data.answer_capacity_fraction <= 1.0:
                raise ValueError(f"data.answer_capacity_fraction {cfg.data.answer_capacity_fraction} outside (0, 1]")
            self.capacity = decoder_capacity(cfg.data.seq_len, cfg.data.answer_capacity_fraction, d.capacity_multiple)
        self.positions = PositionSampler.from_config(cfg, self.capacity)
        # With a shape schedule: the sampler of each phase's row length (the config's own for rows of data.seq_len).
        self._phase_positions = None if self.shape is None else [self._positions_for(ph.seq_len) for ph in self.shape.phases]
        if cfg.data.extra_shard_dirs != []:  # the two-set loader (DataConfig.extra_shard_dirs): its refusals by name
            from ced.train.shards import check_extra_shard_dirs

            check_extra_shard_dirs(cfg.data)
        if cfg.data.union_order != "permute":  # DataConfig.union_order: its refusals by name (nothing for the default)
            from ced.train.shards import check_union_order

            check_union_order(cfg.data)
        if cfg.data.kind == "shards":
            from ced.train.shards import ShardPacker, ShardSet, open_shards

            # with a segment map the batches also carry the shards' assistant flags (Batch.generated / next_generated)
            with_flags = {"flags": True} if self.schedule is not None and self.schedule.counts is not None else {}
            # extra_shard_dirs [] (every run before the key): ShardSet(shard_dir), as always; else the union of the sets
            shards = (ShardSet(cfg.data.shard_dir) if not cfg.data.extra_shard_dirs
                      else open_shards(cfg.data.shard_dir, cfg.data.extra_shard_dirs, order=cfg.data.union_order))
            self.data = ShardPacker(shards, cfg.data, cfg.data.span_fraction, capacity=self.capacity, **with_flags)
        else:
            self.data = SyntheticPackedData(cfg.data, cfg.model.vocab_size, self.capacity)
        # Shards and eval sets built with a tokenizer of another vocabulary are refused (64k ids are
        # valid 128k ids, so nothing else would notice); scripts/rescore.py goes through here too.
        self.tokenizer_checks = self._check_tokenizers()
        # Token Superposition Training (TrainConfig.tst; module docstring): the plan, or None (off: nothing of it runs).
        self.tst = None
        if cfg.tst.schedule:
            from ced.train.tst import TSTPlan

            self.tst = TSTPlan(self)
        elif dataclasses.asdict(cfg.tst) != dataclasses.asdict(type(cfg.tst)()):
            raise ValueError(f"tst options without tst.schedule ({dataclasses.asdict(cfg.tst)}): set the schedule, or leave "
                             "the section at its defaults")
        # The segment pass schedule's CPU side: the scheduler (per-position counts; with a map it loads the
        # vocabulary's bytes from model.loop.segment_tokenizer) and the planner (the fixed-size plan of a micro-batch).
        # The model gets the same scheduler for the forwards that bring no plan (evaluation).
        self.pass_scheduler = self.pass_planner = None
        if self.schedule is not None:
            from ced.train.pass_positions import PassPlanner, PassScheduler

            self.pass_scheduler = PassScheduler.from_config(cfg.model.loop, cfg.model.vocab_size)
            self.pass_planner = PassPlanner(cfg.model.decoder.capacity_multiple, buckets)
            self.model.pass_scheduler = self.pass_scheduler  # the in-training evals: the model's own plans, full size
        self.micro = cfg.micro_batch_size or cfg.data.batch_size
        assert cfg.data.batch_size % self.micro == 0, "batch_size must be a multiple of micro_batch_size"
        self.fused_varlen = uses_fla(cfg) and self.device.type == "cuda"
        # The aux-head skip of the decoder fast path (decoder_opt "exact" / "compiled"): a step whose batch has no aux
        # loss position (n_aux 0: the decoder on every valid target with aux_positions "context") does not compute
        # the aux head or its norm. With no position every aux weight is 0, so the head adds exactly zero to the loss,
        # the metrics and the gradients; the aux-only parameters (the aux norm) get the zero gradient it gave
        # (train_step), so the optimizer and the gradient clipping see the same tensors.
        self._skip_aux = (cfg.decoder_opt != "none" and arch == "ced_v2" and cfg.head_select
                          and cfg.head_impl in ("fused", "quack_lce", *PORTABLE_HEADS))
        # ASEntmax's training statistics (AttentionConfig.normalizer "asentmax"): one EntmaxStats per asentmax
        # attention module, recorded inside the entmax attention during training forwards (support sizes, dead keys,
        # the scale c) and logged under "entmax" every log_every steps (entmax_metrics). Attached before the layers are
        # compiled; the attention call is never traced (ced/model/entmax.py).
        self._entmax = attach_entmax_stats(self.model)
        # Data parallelism (D3): what is not built under it refused by name, then one flat gradient buffer with every
        # parameter's .grad a view into it (data_parallel.GradBuffer). None in one process.
        self._dp_grads = None
        self._dp_mx_flag = None  # the MXFP8 non-finite flag OR-ed over the ranks (train_step, with the flag on)
        self._dp_checked = False  # the replica check before the process's first step (data_parallel.ENV_CHECK_EVERY)
        self._tst_switch_pending = None  # TST's switch under the sharded optimizer: (event, count key, this rank's count) until _dp_batch adds the ranks'
        # The sharded optimizer (TrainConfig.optimizer_sharding; ced/train/zero1.py): this rank's part of it, or None
        # ("replicated", or one process: every line below is what it was). Made before the gradient buffer, whose layout is
        # its owner order, and before the compile: it seats every parameter in its flat parameter buffer.
        self._z1 = None
        from ced.train import zero1 as zero1_mod

        zero1_mod.check_mode(cfg.optimizer_sharding)
        if dp is not None:
            from ced.train import data_parallel

            data_parallel.check(self)
            layout = None
            if cfg.optimizer_sharding == "zero1":
                zero1_mod.check(self)
                self._z1 = zero1_mod.Zero1(self.model, self.optimizer, dp)
                layout = self._z1.layout
            self._dp_grads = data_parallel.GradBuffer(self.model.named_parameters(), layout=layout)
        if cfg.model.fused_projections and cfg.model.fp8.mode != "none" and cfg.model.fp8.backend == "te":
            from ced.model import fused_proj

            bad = fused_proj.unseated(self.model)  # TE's fused GEMMs read each group as one buffer, never a copy
            if bad:
                raise fused_proj.NotSeated(f"model.fused_projections: {len(bad)} groups are not one buffer after the "
                                           f"optimizer's seating ({bad[:3]}): ZeRO-1 keeps a group adjacent only when "
                                           "each part is a whole number of 64-element blocks")
        # TrainConfig.wgrad_accum (ced/train/wgrad_accum.py): "fp32_fused" turns the layers' bias-free linears into
        # FusedWgradLinear (before the layers are compiled) and attaches their persistent float32 gradients; None with
        # "autograd" (nothing is changed or made).
        from ced.train.wgrad_accum import WgradAccum
        from ced.train.wgrad_accum import check_config as check_wgrad_accum

        check_wgrad_accum(cfg)
        grad_of = None if self._dp_grads is None else self._dp_grads.view_of  # data parallelism: the buffers are views
        self._wgrad = (WgradAccum(self.model, cfg.wgrad_accum, self.device, grad_of=grad_of)
                       if cfg.wgrad_accum != "autograd" else None)
        self._set_activation_checkpoint(cfg.activation_checkpoint)
        self._apply_compile(cfg.compile)
        # The Copy-Paste head (model.cp_head; ced/model/cp_head.py; None without it: nothing below runs). The loss is the
        # mixture's CE (_cp_loss) in every step with one target a position; a TST bag phase (s > 1) is the base's bag
        # loss with lambda = 0, the CP tensors getting no gradient (their optimizer state starts at their first one).
        self._cp = getattr(self.model, "cp_head", None)
        self._cp_fn = None  # the CP head's block function in the training loss (None: cp_head.chunk_terms, eager)
        # add-on CP (model.cp_head.train_mode "addon"): (the trunk's and LM head's tensors, the head's), each clipped by its own norm in
        # the step; None otherwise (the one clip over every tensor, as before)
        self._cp_clip = None
        self._trunk: freeze.TrunkGuard | None = None  # TrainConfig.freeze_trunk: the frozen trunk's guard (None: nothing is frozen)
        if self._cp is not None:
            if not cfg.head_select:
                raise ValueError("model.cp_head is built for the head at the valid targets (head_select true), the "
                                 "base's setting")
            if cfg.tst.aux_multi_hot:
                raise ValueError("model.cp_head is not built together with tst.aux_multi_hot (several targets a "
                                 "position in a next-token phase): nothing defines the mixture for them")
            if cfg.model.cp_head.kernel == "fused" and cfg.head_impl != "fused":
                raise ValueError(f"model.cp_head.kernel 'fused' is built for head_impl 'fused' (the mixture inside the "
                                 f"head's CE kernel), not {cfg.head_impl!r}")
            from ced.model import cp_head as cp_mod

            if os.environ.get(cp_mod.ENV_PROBE, "").strip() not in ("", "none"):
                # the length-robust probe is inference only (cp_head.py, "THE INFERENCE-ONLY PROBE"): a training process
                # with it set would read it in its in-training evals
                raise ValueError(f"{cp_mod.ENV_PROBE}={os.environ[cp_mod.ENV_PROBE]!r} is set: the Copy-Paste head's probe "
                                 "is inference only (a scoring process's switch); unset it to train")
            # The block function of the CP terms in the training loss: torch.compile'd on CUDA under compile "head" /
            # "all" (Inductor fuses the masks, the where and the two logsumexps; the same function, compared with the
            # eager one by the GPU check), unless CED_CP_EAGER is set (the eager reference: the check's fallback).
            # model.cp_head.kernel "fused" never reads it (its own kernels).
            if (self.device.type == "cuda" and cfg.compile != "none" and not cp_mod.env_eager()
                    and cfg.model.cp_head.kernel == "reference"):
                # (the pointer's heads under combine "or": its own block function, chunk_terms_or, which runs chunk_terms a
                # head; cp_head.block_function; compiled once a process, as chunk_terms)
                # (the trunk copy, model.cp_head.trunk_copy: cp_trunk.trunk_block, a code object of its own, so its at most 16
                # shapes at 8,192 (ced/model/cp_trunk.py, "THE REFERENCE PATH") have their own recompile budget)
                if hasattr(self._cp, "n_trunk"):
                    from ced.model import cp_trunk

                    compiled = getattr(cp_trunk, "_TRUNK_BLOCK_COMPILED", None)
                    if compiled is None:
                        compiled = cp_trunk._TRUNK_BLOCK_COMPILED = torch.compile(cp_trunk.trunk_block, dynamic=False)
                elif cp_mod.block_function(self._cp) is cp_mod.chunk_terms_or:
                    compiled = getattr(cp_mod, "_CHUNK_TERMS_OR_COMPILED", None)
                    if compiled is None:
                        compiled = cp_mod._CHUNK_TERMS_OR_COMPILED = torch.compile(cp_mod.chunk_terms_or, dynamic=False)
                else:
                    compiled = getattr(cp_mod, "_CHUNK_TERMS_COMPILED", None)
                    if compiled is None:
                        compiled = cp_mod._CHUNK_TERMS_COMPILED = torch.compile(cp_mod.chunk_terms, dynamic=False)
                self._cp_fn = compiled
            # Activation checkpointing with the pointer's input (model.cp_head.pointer_input): "final" takes every mode the
            # trunk takes. cp-mix ("mix", trained as the mixture, the reference kernel, one pointer head) takes "layer":
            # R1-entry's layers under their own checkpoints hand [b_0 .. b_N] to the pointer's readers as ordinary outputs, and
            # the pointer's query blocks keep their own checkpoints; training is bit for bit with and without it
            # (tests/test_cp_mix_ckpt.py; the long-row pair at 32k, section 3 and "Build (2026-10-20)"). Every other pair is
            # refused by name: "block", "host" and "host_overlap" with "mix", every mode with "shared", cp-aux (train_mode
            # "aux") with any mode, and the pointer's heads (cp_head.multi_head: model.cp_head.n_heads > 1, the grid's AND and
            # OR arms, or combine "or" at n_heads 1, whose block function is chunk_terms_or) with any mode: nothing tests them.
            cpc = cfg.model.cp_head
            if cpc.trunk_copy != "none":
                # the trunk copy (ced/model/cp_trunk.py): the global layers hand their queries over on their key sets; built and tested
                # without activation checkpointing (R1-entry's 60M entries run "none") and without TST (its bag phases' token-scale
                # rotary read takes another path through the capped layers)
                if cfg.activation_checkpoint != "none":
                    raise ValueError(f"model.cp_head.trunk_copy {cpc.trunk_copy!r} is not built together with activation_checkpoint "
                                     f"{cfg.activation_checkpoint!r}: nothing tests the pair (built: 'none')")
                if cfg.tst.schedule or cfg.tst.aux_multi_hot:
                    raise ValueError(f"model.cp_head.trunk_copy {cpc.trunk_copy!r} is not built together with Token Superposition "
                                     "Training (tst.schedule / tst.aux_multi_hot): nothing tests the pair")
                if cfg.stage_shape != STAGE_SHAPE_SOURCE:
                    raise ValueError(f"model.cp_head.trunk_copy {cpc.trunk_copy!r} is not built together with stage_shape "
                                     f"{cfg.stage_shape!r}: nothing tests the pair")
            heads = cp_mod.multi_head(cpc)
            built = (cpc.pointer_input == "mix" and cfg.activation_checkpoint == "layer" and cpc.kernel == "reference"
                     and cpc.train_mode == "mixture" and not heads)
            if cpc.pointer_input != "final" and cfg.activation_checkpoint != "none" and not built:
                form = f"model.cp_head.pointer_input {cpc.pointer_input!r}" + (
                    f" with model.cp_head.train_mode {cpc.train_mode!r}" if cpc.train_mode != "mixture" else "") + (
                    f" with model.cp_head.n_heads {cpc.n_heads} with combine {cpc.combine!r}" if heads else "")
                raise ValueError(f"{form} is not built together with activation_checkpoint {cfg.activation_checkpoint!r}: "
                                 "nothing tests the pair (built: 'layer' with pointer_input 'mix', train_mode 'mixture', "
                                 "kernel 'reference', n_heads 1, combine 'and')")
            # cp-mix's readers (model.cp_head.pointer_input "mix"; cp_head.PointerMix, or cp_head.HeadsMix with the pointer's
            # heads: a PointerMix a head): compiled in place on CUDA under the same conditions as the block function (Inductor
            # fuses the reads of the same sources), eager otherwise; the state dict names do not change.
            if (cfg.model.cp_head.pointer_input == "mix" and self.device.type == "cuda" and cfg.compile != "none"
                    and not cp_mod.env_eager()):
                self._cp.mix.compile(dynamic=False)
            if cpc.train_mode == "addon":
                # add-on CP (ced/model/cp_aux.py, "ADD-ON CP"): built and tested in one process, without CUDA graphs and with one target a
                # position in a next-token step (activation checkpointing is refused above: "built" is train_mode "mixture" only; TST's
                # bag phases train the trunk on the bag loss with the head idle: cp_aux.py, "ADD-ON CP IN THE STACK"). The head's tensors
                # are clipped by their own norm, the rest by theirs (the step's clip), so the trunk's update is the arm without the head's.
                from ced.model.cp_aux import check_addon_train

                check_addon_train(cfg, self.dp is not None)
                head_ids = {id(p) for p in self._cp.parameters()}
                self._cp_clip = ([p for p in self.model.parameters() if id(p) not in head_ids], list(self._cp.parameters()))
                if cfg.freeze_trunk:  # the post-hoc stage (ced/train/freeze_trunk.py): every tensor outside the head requires no gradient
                    self._trunk = freeze.TrunkGuard(self.model)
        if cfg.freeze_trunk and self._trunk is None:  # the key on and nothing frozen would be a full run under its name
            raise ValueError("freeze_trunk: this model has no add-on head to train (arch "
                             f"{cfg.model.arch!r} builds none for model.cp_head): nothing was frozen")
        # Monitoring: models with collect_stats record loop usage and the activation maxima before
        # global layers on one micro-batch every log_every steps.
        self._stats_model = self.model if hasattr(self.model, "collect_stats") else None
        # The attention temperature gains (AttentionConfig.temperature "learned" / "ssmax"): exempt from weight decay,
        # so their per-head tau = 1 + w is logged every log_every steps (2026-10-09). None in the fixed mode.
        self._temp_w = [(n[: -len(".temp_w")], p) for n, p in self.model.named_parameters() if n.endswith(".temp_w")]
        # Pass 2's own tensors of a gated two-pass model (LoopConfig.pass2_gate; [] otherwise): their own optimizer
        # groups (ced/train/optim.py), logged every step (alpha) and every log_every steps (all of them).
        self._pass2 = [(n, p) for n, p in self.model.named_parameters() if is_pass2_param(n)]
        # The per-layer residual scalars (model.encoder.layer_scalars; None without them): the encoder that holds
        # them, for the logs (module docstring).
        encoder = getattr(self.model, "encoder", None)
        self._scalars = encoder if getattr(encoder, "layer_scalars", False) else None
        # The tensors the scalars' form stores, in the form's order (both vectors; x0_gain alone for "x0").
        self._scalar_params = ([getattr(encoder, n) for n in layer_scalar_parameters(encoder.scalar_form)]
                               if self._scalars is not None else [])
        # The stream's size by layer (TrainConfig.log_stream_rms_every; module docstring): the encoder whose _stack
        # records it on the first forward of a logging step; None with the option off.
        every = cfg.log_stream_rms_every
        if type(every) is not int or every < 0:
            raise ValueError(f"log_stream_rms_every {every!r}: 0 (off) or a number of optimizer steps")
        if every and (arch != "control" or looped_control(cfg.model) or self.recur or cfg.model.encoder.loops != 1):
            raise ValueError("log_stream_rms_every is built for arch 'control' with one pass through the stack (no "
                             "model.loop.pass2_gate, no model.loop.recur_layers, encoder.loops 1): this model runs its "
                             f"layers through another function (arch {arch!r}), where nothing would be logged. Leave it 0")
        self._rms_encoder = encoder if every else None
        # Exact-zero weight gradients by linear (TrainConfig.log_zero_wgrad_every; ced/train/zero_wgrad.py): the weights
        # whose gradients every N-th step counts, and the start record's description of them. None with the option off.
        zw_every = cfg.log_zero_wgrad_every
        if type(zw_every) is not int or zw_every < 0:
            raise ValueError(f"log_zero_wgrad_every {zw_every!r}: 0 (off) or a number of optimizer steps")
        self._zw: list | None = None
        self._zw_desc: dict | None = None
        if zw_every:
            if arch != "control":
                raise ValueError(f"log_zero_wgrad_every is built for arch 'control' (its layers' mixers and FFNs), not "
                                 f"{arch!r}: leave it 0")
            from ced.train import zero_wgrad

            mods = zero_wgrad.wgrad_modules(self.model)
            self._zw = [m.weight for _, m in mods]
            self._zw_desc = {"every": zw_every, **zero_wgrad.describe(mods)}
        # Block AttnRes (model.encoder.attn_res; None without it): its pseudo-queries' norms in the logs (module docstring).
        self._attn_res = getattr(encoder, "attn_res", None)
        # The GDN-side arms (model.encoder.gdn_write_mix / gdn_ccr_spacing; ced/model/gdn_arms.py): every log_every steps
        # the learned lambdas, the write mix's query norms and CCR's gates, "gdn_arms". None without them: nothing logged.
        self._gdn_arms = encoder if getattr(encoder, "gdn_arms_on", False) else None
        # The CCR gates' floor (model.encoder.gdn_ccr_gate_floor; ced/model/gdn_arms.py floor_gates_): the encoder whose
        # readers' gates are put back to at least the floor after every optimizer step. None without a floor: nothing runs,
        # and the step is the step it was.
        from ced.model.gdn_arms import gate_floor_on

        self._ccr_floor = encoder if gate_floor_on(cfg.model) else None
        if self._ccr_floor is not None and cfg.optim.zero_gate_decay != "none":
            raise ValueError(f"model.encoder.gdn_ccr_gate_floor {cfg.model.encoder.gdn_ccr_gate_floor!r} is built with "
                             f"optim.zero_gate_decay 'none', not {cfg.optim.zero_gate_decay!r}: weight decay would pull every "
                             "gate onto the floor whatever the loss says, and a gate at the floor would no longer read as the "
                             "loss closing it")
        # The injection of a looped middle (model.loop.recur_layers with injection "parcae"; None otherwise), for the logs.
        self._injection = getattr(encoder, "recur_injection", None) if self.recur else None
        # The scalar w of a looped middle's input mix (injection "mix"; None otherwise), for the logs (module docstring).
        self._mix = getattr(encoder, "recur_mix", None) if self.recur else None
        # The reader of the depth mix at the loop boundary (injection "depth_mix": R3; None otherwise), for the logs.
        self._depth = getattr(encoder, "recur_depth", None) if self.recur else None
        self._flops_per_token = None
        self.step = 0
        self.wandb_run_id: str | None = None
        self._wandb = None
        self._stop_requested = False
        # The asynchronous save (TrainConfig.ckpt_async; module docstring): the save whose file is not committed yet (its
        # step, keep, path, whether this process writes it, timings), this process's writer thread and host arena (made at
        # the first save that writes), and the commits made, as (checkpoint step, the step after which it was committed).
        self._ckpt_pending: dict | None = None
        self._ckpt_writer = None
        self._ckpt_arena = None
        self._ckpt_shard_arena = None  # the sharded optimizer's (N2): this rank's optimizer shard's host arena
        self._ckpt_commits: list[tuple[int, int]] = []
        # The last finite steps as fit() saw them (loss, gradient norm, LR, w): the tail a run that stops on a
        # non-finite loss writes into its record (module docstring). Host numbers the step had already read.
        self._recent: collections.deque = collections.deque(maxlen=RECENT_STEPS)
        # The batch of the last train_step (the host tensors batch_at returned; a reference, no copy), so that a stop on
        # a non-finite loss names the rows it trained on (module docstring, "The input guards"). Never saved.
        self._step_batch: Batch | None = None
        # The data prefetch (TrainConfig.data_prefetch; module docstring; ced/train/prefetch.py): the prefetcher (made at the
        # first step that uses it) and the input the running step reads (its batch, loss-valid mask, host numbers and device
        # twins). Both None with the option off, and then nothing of ced.train.prefetch is imported. Never saved.
        if not isinstance(cfg.data_prefetch, bool):
            raise ValueError(f"data_prefetch {cfg.data_prefetch!r}: true or false")
        self._prefetcher = None
        self._input = None
        # Where this run's initial weights came from (init_from_checkpoint), kept in its checkpoints.
        self.init_info: dict | None = None
        # The run whose training this run continues (resume_from_checkpoint), kept in its checkpoints.
        self.resume_info: dict | None = None
        # CUDA graphs for the training step (TrainConfig.cuda_graphs; ced/train/cuda_graphs.py): the keys are checked for
        # every run; the graph runner exists only with the option on (None: train_step is the step it was).
        from ced.train.cuda_graphs import GraphRunner
        from ced.train.cuda_graphs import check_config as check_cuda_graphs

        check_cuda_graphs(cfg)
        self._graphs = GraphRunner(self) if cfg.cuda_graphs != "none" else None
        # qmix's static query mix (model.encoder.gdn_qmix_select_step S; ced/model/gdn_arms.py): at the start of step S, once, each
        # layer's 2 sources are chosen from the token-mean weights of this step's first micro-batch (train_step), and the weights are
        # logged on the log_stream_rms_every cadence. The encoder that holds the layers; None with the option off: nothing below runs
        # and the step is the step it was.
        self._qmix = encoder if cfg.model.encoder.gdn_qmix_select_step else None
        if self._qmix is not None:
            if self.dp is not None:
                raise ValueError("model.encoder.gdn_qmix_select_step is not built with data parallelism: the choice is one process's "
                                 "statistic over its own rows (every rank would choose its own pair)")
            # (CUDA graphs: refused already, cuda_graphs.check_config names the GDN-side arms)
            if cfg.activation_checkpoint in ("host", "host_overlap"):
                raise ValueError(f"model.encoder.gdn_qmix_select_step is not built with activation_checkpoint {cfg.activation_checkpoint!r}: "
                                 "the choice runs one extra forward whose kept inputs the host pool would hold: nothing tests the pair")
            if cfg.eval.readout_every:  # (ced/train/readout.py scoring_twin)
                raise ValueError("model.encoder.gdn_qmix_select_step is not built with eval.readout_every: the readout's scoring twin is built "
                                 "once and keeps each layer's qmix_static flag as it was then (the full query mix) while its tensors are the "
                                 "trainer's, so after the choice it would score another function than the one training")
        self.host_enqueue_s: float | None = None  # train_step's host time to the first device wait (timing runs)
        # Data parallelism (D5; module docstring, "Data parallelism"): the run's record of its world sizes and of each
        # re-derivation of the ranks' CUDA random streams (kept in its checkpoints as "data_parallel" and in its start
        # records), and at a data-parallel start each rank's CUDA generator seeded from (seed, rank, step 0)
        # (data_parallel.rank_seed). A resume replaces it (load_state_dict). None in one process for a run that never ran
        # on several ranks: nothing of it is written or read.
        self.dp_info: dict | None = None
        if dp is not None:
            self.dp_info = {"world_size": dp.world, "history": []}
            self._dp_rng_event(0, None, "start")

    def _check_shape_schedule(self) -> None:
        """What a batch-shape schedule is built for (TrainConfig.shape_schedule); anything else is refused by name."""
        cfg = self.cfg
        if self.recur_loop_set():
            raise ValueError("shape_schedule is not built together with model.loop.recur_layers (the loop in the "
                             "published form): nothing tests the pair")
        if cfg.model.arch != "control" or looped_control(cfg.model):
            raise ValueError("shape_schedule is built for arch 'control' without a second pass (one head on every valid "
                             f"target, no decoder slots), not arch {cfg.model.arch!r}"
                             + (" with model.loop.pass2_gate" if cfg.model.arch == "control" else ""))
        if cfg.loop_switch_step or cfg.optim.schedule_steps is not None:
            raise ValueError("shape_schedule reads the LR at tokens seen: not with loop_switch_step or optim.schedule_steps")
        if cfg.resume_from:
            raise ValueError("shape_schedule is not built for resume_from (a branch of another run)")
        if cfg.eval.lc_every_steps:
            raise ValueError("shape_schedule is not built for eval.lc_every_steps (the in-loop long-context eval counts "
                             "optimizer steps)")
        if cfg.data.decoder_positions == "answer_spans":
            raise ValueError("shape_schedule is not built for data.decoder_positions 'answer_spans'")
        if cfg.data.fit_decoder_capacity and any(ph.seq_len != cfg.data.seq_len for ph in self.shape.phases):
            raise ValueError("shape_schedule with another row length is not built for data.fit_decoder_capacity")

    def recur_loop_set(self) -> bool:
        """Whether the run's model has a looped middle (read from the config: usable before self.recur is set)."""
        return recur_loop(self.cfg.model)

    def _positions_for(self, seq_len: int) -> PositionSampler:
        """The position sampler of rows of `seq_len` tokens: the config's own for its own row length, else the same
        rule at that length (the decoder capacity formula at seq_len; random positions are then exactly that many a
        row). For the control the decoder mask only says where ce_dec is reported."""
        if seq_len == self.cfg.data.seq_len:
            return self.positions
        d = self.cfg.model.decoder
        cfg = dataclasses.replace(self.cfg, data=dataclasses.replace(self.cfg.data, seq_len=seq_len))
        return PositionSampler.from_config(cfg, decoder_capacity(seq_len, d.capacity_fraction, d.capacity_multiple))

    def _check_tokenizers(self) -> list[str]:
        """ced/data/tokenizers.py: the shard set's meta.json tokenizer and each eval set's
        tokenizer_sha256.txt against the model's vocabulary. Raises ValueError on a known mismatch;
        returns one note per checked input."""
        from ced.data.tokenizers import check_tokenizer, eval_set_tokenizer

        vocab, notes = self.cfg.model.vocab_size, []
        if self.cfg.data.kind == "shards":
            extra = self.cfg.data.extra_shard_dirs  # the union's sets share one tokenizer (ShardUnion refuses others)
            notes.append(check_tokenizer(self.data.shards.meta.get("tokenizer_sha256"), vocab,
                                         f"shards {self.cfg.data.shard_dir}" + "".join(f" + {d}" for d in extra)))
        for name, path in (self.cfg.eval.sets or {}).items():
            notes.append(check_tokenizer(eval_set_tokenizer(path), vocab, f"eval set {name} ({path})"))
        return notes

    def _set_activation_checkpoint(self, mode: str) -> None:
        """TrainConfig.activation_checkpoint (ced/model/act_checkpoint.py): "layer" runs each encoder layer and each CED
        v2 decoder layer of a training forward under torch.utils.checkpoint. Refused where a layer's forward is not a
        pure function of its arguments or returns records other code reads back: the second-pass control and Looped-CED
        v1 (not built for them), and ASEntmax (its statistics count every forward of a layer). The control with a gated
        second pass is built for it: the block layers' pass-1 calls (which return their records) and their pass-2 calls
        are each checkpointed (ced/model/control.py; tests/test_loop_control.py). "block" runs each block of
        full_attn_interval layers as one checkpointed call: built and tested for arch "control" with one pass through
        the stack, with model.encoder.attn_res "none", "block" or "block_entry" (tests/test_c2_block_ckpt.py); refused
        for every other form by name. "host" is "layer" with each kept layer input in host memory (self.host_pool,
        opened around every training forward by train_step; tests/test_host_inputs.py). "host_overlap" is "host" with
        the copies on a side stream beside the compute (HostPool(overlap=True); tests/test_host_overlap.py). Both
        refused wherever "layer" is. "layer", "host" and "host_overlap" with attn_res "block" (R1) or "block_entry"
        (R1-entry): every layer of every block under its own checkpoint (ced/model/act_checkpoint.py;
        tests/test_c2l_layer_in_block.py). The architecture arms (2026-10-16; tests/test_arch_arms_ckpt.py): "layer"
        and "block" with every arm (A, CCR, LDM, R1-entry's token queries and heads); "host" and "host_overlap" with A,
        the token queries and the heads, refused by name with CCR (gdn_arms.check_checkpoint_mode) and LDM
        (attn_res.check_checkpoint_mode), whose record or memory would stay on the device."""
        from ced.model.act_checkpoint import HOST, MODES, HostPool, set_activation_checkpoint

        self.host_pool = None
        if mode not in MODES:
            raise ValueError(f"activation_checkpoint {mode!r}: one of {MODES}")
        if mode == "none":
            return
        # The GDN-side arms (ced/model/gdn_arms.py): "layer" and "block" for A and CCR (CCR's records are outputs of
        # the checkpointed calls), "host" / "host_overlap" for A only (tests/test_arch_arms_ckpt.py); CCR with a host
        # mode refused by name.
        from ced.model.gdn_arms import check_checkpoint_mode as check_arms_mode

        check_arms_mode(self.cfg.model, mode)
        arch = self.cfg.model.arch
        if arch not in ("control", "ced_v2"):
            raise ValueError(f"activation_checkpoint {mode!r} is built for arch 'control' and 'ced_v2', not {arch!r}")
        if self._entmax:
            raise ValueError(f"activation_checkpoint {mode!r} is not for the asentmax normaliser (its training "
                             "statistics would count the recomputed forwards)")
        if self.cfg.model.nvfp4.mode != "none":
            # NVFP4 fake quantisation draws its stochastic rounding inside each layer's backward; the checkpointed
            # path (a second forward, the backward run from inside the checkpoint) was not checked with it.
            raise ValueError(f"activation_checkpoint {mode!r} is not built for model.nvfp4 (mode "
                             f"{self.cfg.model.nvfp4.mode!r}): use activation_checkpoint 'none'")
        if mode == "block":
            self._check_block_checkpoint()
        elif self.cfg.model.encoder.attn_res != "none":
            # C2L ("layer" / "host" / "host_overlap" with Block AttnRes: per-layer checkpoints inside the depth mix's
            # blocks; tests/test_c2l_layer_in_block.py): built and tested for exactly the forms the depth mix itself is
            # built for, which check_attn_res names (arch "control", one pass through the stack, whole blocks; every
            # other form refused by name).
            from ced.model.attn_res import check_attn_res

            check_attn_res(self.cfg.model)
        # the architecture arms' depth mixers (2026-10-16; LDM, R1-entry's token queries and heads): every mode but the
        # host modes with LDM (attn_res.check_checkpoint_mode)
        from ced.model.attn_res import check_checkpoint_mode

        check_checkpoint_mode(self.cfg.model, mode)
        set_activation_checkpoint(self.model, mode)
        if mode in HOST:
            self.host_pool = HostPool(overlap=mode == "host_overlap")

    def _check_block_checkpoint(self) -> None:
        """activation_checkpoint "block" only where it is built and tested (ced/model/act_checkpoint.py): arch
        "control", one pass through the stack (Encoder._stack or BlockAttnRes.stack), blocks of full_attn_interval
        layers. ValueError by name for every other form: the other paths (the second pass, the looped middle, a looped
        encoder) do not run the block form, and the residual scalars and Canon layers are untested with it."""
        m = self.cfg.model
        e = m.encoder
        what = "activation_checkpoint 'block'"
        if m.arch != "control":
            raise ValueError(f"{what} is built for arch 'control' (one pass through the encoder stack), not {m.arch!r}: "
                             "use activation_checkpoint 'layer' or 'none'")
        if m.loop.pass2_gate != "none":
            raise ValueError(f"{what} is not built for the control's second pass (model.loop.pass2_gate "
                             f"{m.loop.pass2_gate!r}): use activation_checkpoint 'layer' or 'none'")
        if m.loop.recur_layers:
            raise ValueError(f"{what} is not built for the loop in the published form (model.loop.recur_layers "
                             f"{list(m.loop.recur_layers)!r}): use activation_checkpoint 'layer' or 'none'")
        if e.loops != 1:
            raise ValueError(f"{what} is not built for a looped encoder (model.encoder.loops {e.loops})")
        if e.layer_scalars != "none":
            raise ValueError(f"{what} is not built together with model.encoder.layer_scalars ({e.layer_scalars!r}): "
                             "use activation_checkpoint 'layer' or 'none'")
        if e.canon_kernel:
            raise ValueError(f"{what} is not built together with model.encoder.canon_kernel ({e.canon_kernel!r}): "
                             "use activation_checkpoint 'layer' or 'none'")
        if e.full_attn_interval <= 0 or e.n_layers % e.full_attn_interval:
            raise ValueError(f"{what}: a block is one hybrid group of full_attn_interval layers; n_layers {e.n_layers} "
                             f"is not a whole number of groups of full_attn_interval {e.full_attn_interval}")

    def _apply_compile(self, mode: str) -> None:
        """Set the head function this trainer uses (process-wide: one trainer per process in runs) and
        compile the layers in place for "all" (compile_layers)."""
        assert mode in ("none", "head", "all"), mode
        from ced.train import chunked_loss

        if mode == "none" or self.device.type != "cuda":
            chunked_loss._chunk_ce = chunked_loss._CHUNK_CE_EAGER
            chunked_loss._chunk_ce_fixed = chunked_loss._CHUNK_CE_FIXED_EAGER
            return
        configure_compile()
        if self.schedule is not None:
            # A later pass runs at up to 1 + len(pass_slot_buckets) slot sizes, each one more call shape of both
            # decoder layer kinds next to the T = 1 call and pass 1's (and one more per row shape an in-training eval
            # scores at). Keep the recompile limit above that count; reaching it stays an error, never a fallback.
            import torch._dynamo.config as dynamo_config

            dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 96)
        compiled = getattr(chunked_loss, "_CHUNK_CE_COMPILED", None)
        if compiled is None:
            compiled = chunked_loss._CHUNK_CE_COMPILED = torch.compile(chunked_loss._CHUNK_CE_EAGER, dynamic=False)
        chunked_loss._chunk_ce = compiled
        if self.cfg.loss.ce_form == "fixed":  # loss.ce_form "fixed": its chunk function compiled the same way, only then
            compiled_fixed = getattr(chunked_loss, "_CHUNK_CE_FIXED_COMPILED", None)
            if compiled_fixed is None:
                compiled_fixed = chunked_loss._CHUNK_CE_FIXED_COMPILED = torch.compile(
                    chunked_loss._CHUNK_CE_FIXED_EAGER, dynamic=False)
            chunked_loss._chunk_ce_fixed = compiled_fixed
        if mode == "all":
            compile_layers(self.model)
            if self.cfg.decoder_opt == "compiled":  # the decoder fast path's glue as compiled regions (ced_v2.py)
                self.model.compile_decoder_glue()

    # ----------------------------------------------------------------------------------------

    def lr_at(self, step: int) -> float:
        """Warmup-stable-decay: linear warmup, constant, then linear decay to zero (lr_schedule), read at the run's
        schedule clock (schedule_clock: the step itself unless optim.schedule_steps is set). With a shape schedule:
        the config's schedule over cfg.steps reference steps, read at the reference steps of tokens seen before the
        update (ShapeSchedule.at(step).clock; an integer, and then the unscheduled run's LR bit for bit, wherever a
        phase has one optimizer step to a reference step). With shape_lr "rows": the config's schedule at the whole
        reference step whose tokens the update reads, divided by the phase's optimizer steps to a reference step
        (ShapeSchedule.lr_point): the steps of a reference step share the unscheduled run's LR of that step."""
        if self.shape is not None:
            if self.cfg.shape_lr == "rows":
                ref, per_ref = self.shape.lr_point(step)
                return lr_schedule(self.cfg.optim, self.cfg.steps, ref) / per_ref
            return lr_schedule(self.cfg.optim, self.cfg.steps, self.shape.at(step).clock)
        return lr_of(self.cfg, step)

    def loop_count(self, step: int) -> int:
        """T for this optimizer step, drawn from a counter-based generator (same on every rank). Before a loop
        switch step (TrainConfig.loop_switch_step) it is 1: the trunk's single pass."""
        if step < self.cfg.loop_switch_step:
            return 1
        loop = self.cfg.model.loop
        g = counter_generator(self.cfg.seed, step, stream=1)
        i = int(torch.multinomial(torch.tensor(loop.t_probs, dtype=torch.float64), 1, generator=g))
        return loop.t_choices[i]

    def row_passes(self, step: int) -> list[int] | None:
        """model.loop.t_sampling "sequence" (the loop in the published form; None otherwise): the pass count of every
        row of the step's batch, in row order. The step's multiset (t_probs[i] x data.batch_size rows of t_choices[i])
        in an order drawn from the counter RNG of (seed, step): a pure function of the step."""
        if self._row_multiset is None:
            return None
        g = counter_generator(self.cfg.seed, step, stream=2)
        order = torch.randperm(len(self._row_multiset), generator=g).tolist()
        out = [0] * len(order)
        for slot, row in enumerate(order):
            out[row] = self._row_multiset[slot]
        return out

    def micro_rows(self, T: int | None) -> int:
        """Rows per forward/backward at pass count T: micro_batch_by_t's entry (its last beyond the list), else
        micro_batch_size."""
        by_t = self.cfg.micro_batch_by_t
        if not by_t or T is None:
            return self.micro
        return by_t[min(T, len(by_t)) - 1]

    def micro_plan(self, step: int, B: int, T: int | None, micro: int) -> list:
        """The micro-batches of a step as (rows, T): rows a slice of the batch's B rows, or, when the rows of a
        forward are not consecutive, their indices (a LongTensor, ascending); T the pass count of that forward.
        Without a looped middle: consecutive slices of `micro` rows at the step's T, as every run before it.
        With one: under t_sampling "step" consecutive slices of micro_rows(T) rows; under "sequence" the rows of each
        pass count (ascending T, rows in batch order) in groups of micro_rows(T). A group may be smaller than its
        size when the rows of a T are not a multiple of it; the shapes are the same at every step."""
        if not self.recur:
            return [(slice(m0, m0 + micro), T) for m0 in range(0, B, micro)]
        rows_t = self.row_passes(step)
        if rows_t is None:
            m = self.micro_rows(T)
            return [(slice(m0, min(m0 + m, B)), T) for m0 in range(0, B, m)]
        plan = []
        for t in sorted(set(rows_t)):
            rows = [r for r in range(B) if rows_t[r] == t]
            m = self.micro_rows(t)
            plan += [(torch.tensor(rows[i : i + m], dtype=torch.long), t) for i in range(0, len(rows), m)]
        return plan

    def mean_passes(self) -> float:
        """The mean pass count of a training step of a looped middle: the expectation of t_choices under t_probs
        (exact at every step under t_sampling "sequence", whose rows hold the multiset)."""
        loop = self.cfg.model.loop
        return sum(t * prob for t, prob in zip(loop.t_choices, loop.t_probs)) / sum(loop.t_probs)

    def flops_per_token(self, T: int | None = None) -> float:
        """Analytic training FLOPs per token of this config (ced.utils.params.flops_per_token) at its largest loop
        count, or at the loop count T (the single-pass steps before a loop switch step). With a looped middle
        (model.loop.recur_layers): at the MEAN pass count of t_choices / t_probs (the count is linear in T)."""
        from ced.utils.params import flops_per_token

        c = self.cfg
        aux = self.positions.aux_count / c.data.seq_len if self.positions.aux_count is not None else None
        if T is not None:
            return flops_per_token(c.model, c.data.seq_len, self.capacity / c.data.seq_len, T, aux_fraction=aux)["training"]
        if self._flops_per_token is None:
            if self.recur:
                one, two = (flops_per_token(c.model, c.data.seq_len, self.capacity / c.data.seq_len, t, aux_fraction=aux)["training"]
                            for t in (1, 2))
                self._flops_per_token = one + (self.mean_passes() - 1.0) * (two - one)
                return self._flops_per_token
            T = max(c.model.loop.t_choices) if self.is_looped else 1
            self._flops_per_token = flops_per_token(c.model, c.data.seq_len, self.capacity / c.data.seq_len, T,
                                                    aux_fraction=aux)["training"]
        return self._flops_per_token

    def data_step(self, step: int) -> int:
        """Where update `step` reads the data stream: step - data_start_step (TrainConfig.data_start_step; the step
        itself for every run before the key). ValueError for an update before the stream starts: a stage that
        continues another run on new data never trains those (they are the source's)."""
        k = step - self.cfg.data_start_step
        if k < 0:
            raise ValueError(f"update {step} is before this run's data stream starts (data_start_step "
                             f"{self.cfg.data_start_step}): those updates are the source's")
        return k

    def batch_at(self, step: int) -> Batch:
        """The training batch of `step`: the data's rows with the configured decoder and aux positions
        (ced/train/positions.py). A pure function of the step: of data_step(step), the stream's position, which is the
        step itself unless the run is a stage whose stream starts later (TrainConfig.data_start_step D: the stream's
        batch step - D, rows and positions alike, so its first update reads the stream's first batch for the data
        seed). With a segment pass schedule, on a step that runs more than one pass, also every position's pass count
        (Batch.pass_counts; module docstring). Under Token Superposition Training: TSTPlan.batch_at (module docstring).
        Every batch passes the loader's validation before it is returned (check_batch; module docstring, "The input
        guards"): BatchError otherwise. The check reads the batch and changes nothing."""
        batch = self._build_batch(step)
        self.check_batch(batch, step)
        return batch

    def _build_batch(self, step: int) -> Batch:
        """batch_at's batch, before the loader's validation."""
        if self.tst is not None:
            if self.dp is None:
                return self.tst.batch_at(step)
            # data parallelism (D2) with TST: this rank's rows of the step's batch (bag rows in a bag phase)
            first, count = self.dp.rows(self.cfg.data.batch_size)
            return self.tst.batch_at(step, first, count)
        if self.shape is None and self.dp is not None:
            # data parallelism (D2): this rank's rows of the step's batch, with the positions drawn for their global index
            k = self.data_step(step)
            first, count = self.dp.rows(self.cfg.data.batch_size)
            if hasattr(self.data, "rows"):  # the shard stream: rows k B + first onwards
                rows = self.data.rows(k * self.cfg.data.batch_size + first, count)
            else:  # synthetic: the step's rows first .. first + count - 1
                rows = self.data.batch(k, first=first, count=count)
            batch = self.positions.apply(rows, k, row0=first)
        elif self.shape is None:
            k = self.data_step(step)  # the step itself unless the stream starts later (data_start_step)
            batch = self.positions.apply(self.data.batch(k), k)
        else:
            # the rows and the row length of the step's phase, from the unscheduled run's stream in its order
            if not hasattr(self.data, "rows"):
                raise ValueError("shape_schedule reads the shard stream (data.kind 'shards'): synthetic data has no stream")
            sh = self.shape.at(step)
            batch = self._phase_positions[sh.phase].apply(self.data.rows(sh.first_row, sh.rows, sh.seq_len), step)
        if self.schedule is not None and self.loop_count(step) > 1:
            flags = {} if batch.generated is None else {"generated": batch.generated.numpy(),
                                                        "next_generated": batch.next_generated.numpy()}
            # The units of a later pass: CED v2's decoder positions (dec_mask), or every position of the control
            # with a second pass (its dec_mask only says where ce_dec is reported: an unscored position is then one
            # without a valid target, whatever the mask).
            units = batch.dec_mask.numpy() if self.has_slots else None
            counts = self.pass_scheduler.position_counts(batch.tokens.numpy(), batch.doc_ids.numpy(), units,
                                                         batch.target_valid.numpy(), **flags)
            batch = dataclasses.replace(batch, pass_counts=counts)
        return batch

    def check_batch(self, batch: Batch, step: int) -> None:
        """The loader's validation of update `step`'s batch (step 0-based, as batch_at's; ced/train/batch_check.py
        check_batch against the model's vocabulary). A failure raises BatchError with the update (as metrics.jsonl counts
        it, step + 1) and the failing row's place in the data stream (row_source) added to its message and info."""
        try:
            check_batch_contract(batch, self.cfg.model.vocab_size)
        except BatchError as error:
            raise self._batch_error(error, step) from None

    def _batch_error(self, error: BatchError, step: int) -> BatchError:
        """error with the update and the row's identity in the data stream added (row_source; best effort: a failure to
        describe the source is noted, never raised; locate_batch_error)."""
        try:
            src = self.row_source(step)
        except Exception as e:  # noqa: BLE001 - naming the row must never hide the error itself
            src = {"error": f"{type(e).__name__}: {e}"}
        return locate_batch_error(error, step, src)

    def row_source(self, step: int) -> dict:
        """Where the rows of update `step`'s batch (step 0-based) come from in the data stream, for a batch error and a
        halt record: {"data", "data_seed"} and, for shards, "first_row", "length" and "bag": row b of the batch is the
        stream's row first_row + b x bag of `length` tokens (ShardPacker.row; with bag s > 1, a TST bag phase, it covers
        the s stream rows from there; with TST option (d) "stream" says which of the two streams); for synthetic data
        "data_step": row b is drawn from the counter RNG of (data seed, data_step, 1000 + b). A pure function of the step,
        as batch_at is."""
        c = self.cfg
        out = {"data": c.data.kind, "data_seed": c.data.seed}
        if self.tst is not None:
            phase = self.tst.sched.phase_at(step)
            bag = self.tst.s_in(phase)
            first_row = self.tst.first_row(step)
            if self.dp is not None:  # data parallelism: this rank's row r is row row0 + r of the batch, i.e. stream row first_row + (row0 + r) x bag
                row0 = self.dp.rows(c.data.batch_size)[0]
                out.update(rank=self.dp.rank, world_size=self.dp.world, batch_row0=row0)
                first_row += row0 * bag
            out.update(first_row=first_row, length=c.data.seq_len, bag=bag)
            if self.tst.split is not None:
                out["stream"] = "first" if phase.index == 0 else "rest"
            return out
        if self.shape is not None:
            sh = self.shape.at(step)
            out.update(first_row=sh.first_row, length=sh.seq_len, bag=1)
            return out
        k = self.data_step(step)
        row0 = 0
        if self.dp is not None:  # data parallelism: this rank's rows start at row row0 of the step's batch
            row0 = self.dp.rows(c.data.batch_size)[0]
            out.update(rank=self.dp.rank, world_size=self.dp.world, batch_row0=row0)
        if c.data.kind != "shards":
            out["data_step"] = k
            return out
        out.update(first_row=k * c.data.batch_size + row0, length=c.data.seq_len, bag=1)
        return out

    def row_record(self, step: int, batch: Batch) -> dict:
        """What a halt record says of update `step`'s batch (step 0-based): its rows' source (row_source), a content
        hash of every row (batch_check.row_hashes) and the first and last document id of every row (describe_rows)."""
        return describe_rows(step, batch, self.row_source)

    @staticmethod
    def loss_valid(batch: Batch) -> torch.Tensor:
        """The positions whose target the loss counts: target_valid, and with a per-token loss mask (Batch.loss_mask;
        DataConfig.loss_mask) target_valid & loss_mask. Without a mask it is batch.target_valid itself (the same tensor
        object), so every branch below reads what it read before."""
        return batch.target_valid if batch.loss_mask is None else batch.target_valid & batch.loss_mask

    def batch_counts(self, batch: Batch) -> dict[str, int]:
        """Host side, from the whole batch: n_valid (valid targets), n_dec (positions the main head
        scores: valid kept slots, or for the control valid decoder-mask positions) and n_aux (the aux
        head's loss positions; for the control, the other valid targets). "Valid" is loss_valid: with a per-token loss
        mask only the kept targets count, so every denominator is the number of kept positions."""
        valid = self.loss_valid(batch)
        if self.has_slots:
            kept = select_slots(batch.dec_mask, self.capacity).kept
            n_dec, n_aux = int((valid & kept).sum()), int(aux_positions(valid, kept, batch.aux_mask).sum())
        else:
            n_dec, n_aux = int((valid & batch.dec_mask).sum()), int((valid & ~batch.dec_mask).sum())
        return {"n_valid": int(valid.sum()), "n_dec": n_dec, "n_aux": n_aux}

    # --- the data prefetch (TrainConfig.data_prefetch; ced/train/prefetch.py): everything below runs only with the option on,
    # except _check_micro_varlen, which _dp_batch's check always used.

    def _host_numbers(self, batch: Batch, step: int):
        """What train_step and _dp_batch read from update `step`'s host batch besides its tensors, each by the expression the
        main thread uses: the loss counts (batch_counts), the batch hash, the position samplers' shortfall (the step's phase
        sampler of a shape schedule; under data parallelism the config's) and the valid-target count. A pure function of the
        batch, so the prefetch thread can make it."""
        from ced.train.prefetch import HostNumbers

        shape = None if (self.shape is None or self.dp is not None) else self.shape.at(step)
        positions = self.positions if shape is None else self._phase_positions[shape.phase]
        return HostNumbers(counts=self.batch_counts(batch), hash=batch_hash(batch), short=positions.shortfall(batch),
                           n_target_valid=int(batch.target_valid.sum()))

    def _check_micro_varlen(self, batch: Batch, step: int) -> None:
        """The fused GDN kernels' boundary arrays of every micro-batch of update `step`, checked on the host before any
        micro-batch runs (_dp_batch: a bad one must stop every rank before the first collective of the step). BatchError,
        naming the row, as varlen_for_batch raises it."""
        from ced.train.varlen import varlen_for_batch

        T = self.loop_count(step) if self.is_looped else None
        for mb, _ in self.micro_plan(step, batch.tokens.shape[0], T, self.micro):
            try:
                varlen_for_batch(batch.doc_ids[mb], None, None)
            except BatchError as e:
                raise self._batch_error(micro_batch_error(e, mb), step) from None

    def _step_input(self, step: int):
        """Update `step`'s input as the prefetch builds it, on its thread or (for a step nothing was submitted for) on the
        main thread: batch_at, under data parallelism with the fused kernels the micro-batches' boundary check, the loss-valid
        mask, the host numbers, and the host tensors the micro-batches move to the device (each once, by identity). A pure
        function of the step; BatchError as batch_at raises it."""
        from ced.train.prefetch import StepInput

        batch = self.batch_at(step)
        if self.dp is not None and self.fused_varlen:
            self._check_micro_varlen(batch, step)
        valid = self.loss_valid(batch)
        seen, stage = set(), []
        for t in (batch.tokens, batch.targets, valid, batch.doc_ids, batch.dec_mask, batch.aux_mask, batch.multi_targets,
                  batch.multi_valid):
            if t is not None and id(t) not in seen:
                seen.add(id(t))
                stage.append(t)
        return StepInput(step=step, batch=batch, valid_loss=valid, numbers=self._host_numbers(batch, step), to_stage=stage)

    def _prefetch(self):
        """The prefetcher (made at its first use; its copy engine is the device's)."""
        if self._prefetcher is None:
            from ced.train import prefetch

            self._prefetcher = prefetch.BatchPrefetcher(self._step_input, prefetch.make_stage(self.device))
        return self._prefetcher

    def _take_input(self, step: int):
        """Update `step`'s input for the running step (self._input): the prefetched one, or built here when nothing was
        submitted for the step (a process's first step, a step after a jump). BatchError as batch_at raises it."""
        self._input = self._prefetch().take(step)
        return self._input

    def _prefetch_next(self) -> None:
        """Start building the next update's input on the prefetch thread (not past the run's last update)."""
        if self.step + 1 < self.total_steps:
            self._prefetch().submit(self.step + 1)

    def _prefetch_discard(self) -> None:
        """Nothing built for an earlier state outlives this (a checkpoint load, fit's exit): the thread finished, its result
        dropped. No-op without a prefetcher."""
        if self._prefetcher is not None:
            self._prefetcher.discard()
        self._input = None

    def _prefetch_quiet(self):
        """A context in which the prefetch thread makes no CUDA call (GraphRunner's captures); a null context without one."""
        return contextlib.nullcontext() if self._prefetcher is None else self._prefetcher.quiet()

    def _micro_to(self, mb):
        """The function that moves a host tensor's micro-batch rows to the device (None stays None): the tensor's rows copied
        now, or, when the step's input was staged, its device twin's rows (the copy made on the side stream; reading it before
        the wait raises PrefetchRace)."""
        staged = None if self._input is None else self._input.staged
        if staged is None:
            return lambda t: None if t is None else t[mb].to(self.device, non_blocking=True)
        # the rows of a slice are cloned (a device copy of a few kilobytes): the model then gets what `.to` gave it, a tensor of its
        # own at storage offset 0, not a view of the step's twin that the next micro-batch's rows share the storage of
        if isinstance(mb, slice):
            return lambda t: None if t is None else staged.twin(t)[mb].clone()
        # rows that are not consecutive (the looped middle under t_sampling "sequence"): the index tensor goes to the device ONCE,
        # asynchronously. Indexing a CUDA tensor with a CPU index tensor copies the index inside the indexing op, with the
        # stream synchronised (PyTorch's blocking H2D copy), which would be six drains of the launch queue a micro-batch.
        # An index gather is a copy already: no clone.
        rows = mb.to(self.device, non_blocking=True)
        return lambda t: None if t is None else staged.twin(t)[rows]

    def _micro_capacity(self, dec_mask: torch.Tensor) -> int:
        """Host side: the number of decoder slots this micro-batch runs (cfg.trim_decoder_slots, v1)."""
        m = self.cfg.trim_decoder_slots
        if not (self.is_v1 and m):
            return self.capacity
        n = min(int(dec_mask.sum(1).max()), self.capacity)
        return min(self.capacity, max(m, -(-n // m) * m))

    def _varlen(self, doc_ids: torch.Tensor, dec_mask: torch.Tensor, capacity: int | None = None, rows=None,
                step: int | None = None) -> dict | None:
        """Host-computed segment boundaries for the fused GDN kernel (ced/train/varlen.py): the
        encoder's always (CED v2's encoder; the second pass's extra kernel call uses pass 1's); the
        decoder slots' only for v1, whose decoder has GDN layers. varlen_for_batch checks every boundary array it
        builds (BatchError); rows (the micro-batch's rows of the batch: a slice or indices) and step (0-based) only name
        the failing row in that error."""
        if not self.fused_varlen:
            return None
        from ced.train.varlen import to_device, varlen_for_batch

        capacity = self.capacity if capacity is None else capacity
        try:
            host = varlen_for_batch(doc_ids, dec_mask if self.is_v1 else None, capacity if self.is_v1 else None)
        except BatchError as error:
            error = micro_batch_error(error, rows)  # the micro-batch's row as a row of the step's batch
            raise (error if step is None else self._batch_error(error, step)) from None
        return to_device(host, self.device)

    def _qmix_choose(self, autocast, forward) -> dict:
        """qmix's static query mix, the choice (model.encoder.gdn_qmix_select_step; ced/model/gdn_arms.py select_qmix_pairs): at the
        start of step S, with the weights the last update left, ONE extra forward of this step's first micro-batch (the training
        forward's own call, `forward`, run in the same autocast and discarded) with the encoder's qmix_record set, so that every query
        mix's token-sums are recorded as the full softmax (the forward runs before any choice); then each layer with more than 2
        sources takes the 2 with the largest token-mean weight (ties to the lower source), written into its state-dict buffer. The same
        compiled graphs as the training forward (grad mode on, nothing run backward), so the choice recompiles nothing. Returns the
        step's log record {"step", "layers", "pair", "weights" (the token-means the choice read, every source), "tokens"}; the data of
        the micro-batch is the step's own (a function of the step and the seed), so a resume before S chooses the same pair."""
        from ced.model import gdn_arms

        enc = self._qmix
        # The step's own monitors (train_step arms them before this call, for the step's first micro-batch: the stream's RMS by layer,
        # the activation statistics) belong to the training forward alone: this forward would append its values to the same lists a
        # second time (a stream_rms of 2 (n_layers + 1) values on a step where the choice and the stream log coincide). Set aside for
        # the length of this forward, put back after it.
        held_rms = enc.stream_rms
        held_collect = self._stats_model.collect_stats if self._stats_model is not None else None
        enc.stream_rms = None
        if self._stats_model is not None:
            self._stats_model.collect_stats = False
        enc.qmix_record = {}
        try:
            with autocast, act_checkpoint.host_inputs(self.host_pool):
                forward()
        finally:
            rec, enc.qmix_record = enc.qmix_record, None
            enc.stream_rms = held_rms
            if self._stats_model is not None:
                self._stats_model.collect_stats = held_collect
        chosen = gdn_arms.select_qmix_pairs(enc, rec)
        layers = list(chosen)
        record = {"step": self.step, "layers": layers, "pair": [chosen[i]["pair"] for i in layers],
                  "weights": [chosen[i]["weights"] for i in layers], "tokens": chosen[layers[0]]["tokens"] if layers else 0}
        print(f"[qmix_select] step {self.step}: each layer's 2 sources (layer: pair, of n) "
              f"{ {i: (chosen[i]['pair'], len(chosen[i]['weights'])) for i in layers} } on the token-mean query-mix weights of "
              f"{record['tokens']} tokens", flush=True)
        return record

    def _head_selection(self, valid: torch.Tensor, dec_mask: torch.Tensor, capacity: int | None = None,
                        aux_mask: torch.Tensor | None = None) -> dict | None:
        """Host side, per micro-batch: the positions each head must score (cfg.head_select): the aux
        head at its loss positions ("aux") and the main head at the valid filled slots ("dec"); the
        control's one head at every valid target ("all")."""
        if not self.cfg.head_select:
            return None
        dev = self.device
        move = lambda pair: tuple(x.to(dev, non_blocking=True) for x in pair)  # noqa: E731
        if self.has_slots:
            slots = select_slots(dec_mask, self.capacity if capacity is None else capacity)
            slot_target_valid = gather_slots(valid, slots.index, fill=False) & slots.valid
            on_aux = aux_positions(valid, slots.kept, aux_mask)
            return {"aux": move(head_positions(on_aux)), "dec": move(head_positions(slot_target_valid))}
        return {"all": move(head_positions(valid)), "dec_mask": dec_mask.to(dev, non_blocking=True)}

    def _two_head_loss(self, sum_dec: torch.Tensor, sum_aux: torch.Tensor, norms: dict) -> torch.Tensor:
        """The batch loss share of the two heads' summed objectives (see the module docstring)."""
        loss_cfg = self.cfg.loss
        if loss_cfg.normalization == "global_token_mean":
            return (sum_dec + loss_cfg.aux_weight * sum_aux) / norms["n_valid"]
        if loss_cfg.normalization == "per_head_mean":
            return sum_dec / norms["n_dec"] + loss_cfg.aux_weight * sum_aux / norms["n_aux"]
        raise ValueError(f"unknown normalization {loss_cfg.normalization!r}")

    def _micro_loss(self, out, targets, valid, dec_mask, norms: dict, sel: dict | None = None,
                    aux_mask: torch.Tensor | None = None, skip_aux: bool = False,
                    cp_rows: tuple | None = None) -> tuple[torch.Tensor, dict]:
        """This micro-batch's share of the batch loss, and its metric sums (0-dim tensors).
        aux_mask: the micro-batch's aux positions (device), or None for v1's context positions.
        skip_aux: the aux head has no position here and is not computed (Trainer._skip_aux; fused head only).
        cp_rows: (tokens, doc_ids) of the micro-batch for the Copy-Paste head (model.cp_head; None without it)."""
        if sel is not None:
            if cp_rows is not None:
                return self._cp_loss(out, targets, norms, sel, cp_rows)
            return self._micro_loss_selected(out, targets, norms, sel, skip_aux=skip_aux)
        assert not skip_aux, "the aux-head skip is for the fused head (head_select)"
        c = self.cfg
        W = self.model.head_weight()
        z = c.loss.z_loss
        zero = torch.zeros((), device=targets.device)
        hp = c.head_precision
        cf = c.loss.ce_form  # loss.ce_form (ced/train/ce_fix.py): "standard" is main's code, bit for bit
        ce, lse = token_ce(out.hidden_aux, W, targets, precision=hp, ce_form=cf)
        obj = ce + z * lse.pow(2)
        sums = {}
        if self.has_slots:
            on_aux = aux_positions(valid, out.kept, aux_mask)
            t_slots = gather_slots(targets, out.slot_index, fill=0)
            on_dec = gather_slots(valid, out.slot_index, fill=False) & out.slot_valid
            ce_d, lse_d = token_ce(out.hidden_dec, W, t_slots, precision=hp, ce_form=cf)
            obj_d = ce_d + z * lse_d.pow(2)
            sum_dec = torch.where(on_dec, obj_d, zero).sum()
            sum_aux = torch.where(on_aux, obj, zero).sum()
            loss = self._two_head_loss(sum_dec, sum_aux, norms)
            sums["ce_dec"] = torch.where(on_dec, ce_d, zero).sum()
            sums["ce_aux"] = torch.where(on_aux, ce, zero).sum()
            sums["lse_sq"] = torch.where(on_dec, lse_d.pow(2), zero).sum() + torch.where(on_aux, lse.pow(2), zero).sum()
            draft_w = c.model.loop.draft_loss_weight
            for i, h in enumerate(out.hidden_drafts or []):
                ce_k, _ = token_ce(h, W, t_slots, precision=hp, ce_form=cf)
                d_sum = torch.where(on_dec, ce_k, zero).sum()
                loss = loss + draft_w * d_sum / norms["n_dec"]
                sums[f"ce_draft_loop{i + 1}"] = d_sum
            if out.aux_loss is not None:
                loss = loss + out.aux_loss * norms["micro_share"]
                sums["moe_balance"] = out.aux_loss.detach() * norms["micro_share"]
        else:
            # One head: a plain token mean. The data's decoder mask only selects where ce_dec is reported.
            loss = torch.where(valid, obj, zero).sum() / norms["n_valid"]
            sums["ce_dec"] = torch.where(valid & dec_mask, ce, zero).sum()
            sums["ce_ctx"] = torch.where(valid & ~dec_mask, ce, zero).sum()
            sums["lse_sq"] = torch.where(valid, lse.pow(2), zero).sum()
        return loss, sums

    def _micro_loss_selected(self, out, targets, norms: dict, sel: dict, skip_aux: bool = False,
                             rows: dict | None = None) -> tuple[torch.Tensor, dict]:
        """_micro_loss computed only at the positions whose loss counts (cfg.head_select). Every term
        is the same sum over the same positions; masked positions are simply not computed.
        rows: None (every caller but add-on CP), or a dict that receives the one-head control's per-row "ce" and "lse" at the
        selected positions (zero at padding; detached), the numbers the loss was made of: nothing computed changes."""
        if self.cfg.head_impl in ("fused", "quack_lce", *PORTABLE_HEADS):  # quack_lce, liger_flce, cce: the fused head's
            # weights, their kernels
            return self._micro_loss_fused(out, targets, norms, sel, skip_aux=skip_aux, rows=rows)
        assert not skip_aux, "the aux-head skip is for the fused head"
        c = self.cfg
        W = self.model.head_weight()
        z = c.loss.z_loss
        hp = c.head_precision
        cf = c.loss.ce_form
        sums = {}
        if self.has_slots:
            ce, lse = selected_ce(out.hidden_aux, W, targets, *sel["aux"], precision=hp, ce_form=cf)
            t_slots = gather_slots(targets, out.slot_index, fill=0)
            ce_d, lse_d = selected_ce(out.hidden_dec, W, t_slots, *sel["dec"], precision=hp, ce_form=cf)
            sum_aux = (ce + z * lse.pow(2)).sum()
            sum_dec = (ce_d + z * lse_d.pow(2)).sum()
            loss = self._two_head_loss(sum_dec, sum_aux, norms)
            sums["ce_dec"], sums["ce_aux"] = ce_d.sum(), ce.sum()
            sums["lse_sq"] = lse_d.pow(2).sum() + lse.pow(2).sum()
            for i, h in enumerate(out.hidden_drafts or []):
                ce_k, _ = selected_ce(h, W, t_slots, *sel["dec"], precision=hp, ce_form=cf)
                d_sum = ce_k.sum()
                loss = loss + c.model.loop.draft_loss_weight * d_sum / norms["n_dec"]
                sums[f"ce_draft_loop{i + 1}"] = d_sum
            if out.aux_loss is not None:
                loss = loss + out.aux_loss * norms["micro_share"]
                sums["moe_balance"] = out.aux_loss.detach() * norms["micro_share"]
        else:
            idx, real = sel["all"]
            ce, lse = selected_ce(out.hidden_aux, W, targets, idx, real, precision=hp, ce_form=cf)
            loss = (ce + z * lse.pow(2)).sum() / norms["n_valid"]
            if rows is not None:  # add-on CP: the per-row numbers, read only
                rows.update(ce=ce.detach(), lse=lse.detach())
            on_dec = sel["dec_mask"].reshape(-1).index_select(0, idx) & real
            zero = torch.zeros((), dtype=ce.dtype, device=ce.device)
            sums["ce_dec"] = torch.where(on_dec, ce, zero).sum()
            sums["ce_ctx"] = torch.where(~on_dec, ce, zero).sum()
            sums["lse_sq"] = lse.pow(2).sum()
        return loss, sums

    def _micro_loss_fused(self, out, targets, norms: dict, sel: dict, skip_aux: bool = False,
                          rows: dict | None = None) -> tuple[torch.Tensor, dict]:
        """_micro_loss_selected with the fused head: each head call gets its per-position loss
        weights up front (the same normalisers and coefficients), and returns its weighted sum.
        skip_aux: the aux head has no position (every weight 0) and is not called: its loss term, its ce sum and
        its lse^2 sum are exactly 0, so they are left out (x + 0.0 is x for the positive sums kept).
        rows: as _micro_loss_selected's (the one-head control: the head call's own ce and lse, which carry no gradient)."""
        c = self.cfg
        W = self.model.head_weight()
        z = c.loss.z_loss

        def head(hidden, tgt, idx, real, a_scale, b_scale, zc=z):  # zc: the z-loss coefficient, b_scale / a_scale
            h = hidden.reshape(-1, hidden.shape[-1]).index_select(0, idx)
            t = tgt.reshape(-1).index_select(0, idx)
            w = real.to(compute_dtype(h))
            if c.head_impl == "quack_lce":  # QuACK's bf16 linear-CE with our glue (ced/train/quack_head.py)
                loss, ce, lse = quack_linear_ce(h, W, t, w * a_scale, w * b_scale)
            elif c.head_impl in PORTABLE_HEADS:  # Liger's / CCE's fused linear-CE: a = real a_scale, b = real zc a_scale
                loss, ce, lse = portable_linear_ce(c.head_impl, h, W, t, real, a_scale, zc)
            else:
                loss, ce, lse = fused_linear_ce(h, W, t, w * a_scale, w * b_scale, chunk=c.head_chunk,
                                                precision=c.head_precision, ce_form=c.loss.ce_form)
            zero = torch.zeros((), dtype=ce.dtype, device=ce.device)
            return loss, torch.where(real, ce, zero), torch.where(real, lse, zero)

        sums = {}
        if self.has_slots:
            t_slots = gather_slots(targets, out.slot_index, fill=0)
            if c.loss.normalization == "global_token_mean":
                dec_scale = aux_scale = 1.0 / norms["n_valid"]
            elif c.loss.normalization == "per_head_mean":
                dec_scale, aux_scale = 1.0 / norms["n_dec"], 1.0 / norms["n_aux"]
            else:
                raise ValueError(f"unknown normalization {c.loss.normalization!r}")
            aux_scale *= c.loss.aux_weight
            loss_d, ce_d, lse_d = head(out.hidden_dec, t_slots, *sel["dec"], dec_scale, z * dec_scale)
            if skip_aux:
                loss = loss_d
                sums["ce_dec"], sums["ce_aux"] = ce_d.sum(), torch.zeros((), dtype=ce_d.dtype, device=ce_d.device)
                sums["lse_sq"] = lse_d.pow(2).sum()
            else:
                loss_a, ce, lse = head(out.hidden_aux, targets, *sel["aux"], aux_scale, z * aux_scale)
                loss = loss_d + loss_a
                sums["ce_dec"], sums["ce_aux"] = ce_d.sum(), ce.sum()
                sums["lse_sq"] = lse_d.pow(2).sum() + lse.pow(2).sum()
            draft_scale = c.model.loop.draft_loss_weight / norms["n_dec"]
            for i, h in enumerate(out.hidden_drafts or []):
                loss_k, ce_k, _ = head(h, t_slots, *sel["dec"], draft_scale, 0.0, zc=0.0)
                loss = loss + loss_k
                sums[f"ce_draft_loop{i + 1}"] = ce_k.sum()
            if out.aux_loss is not None:
                loss = loss + out.aux_loss * norms["micro_share"]
                sums["moe_balance"] = out.aux_loss.detach() * norms["micro_share"]
        else:
            idx, real = sel["all"]
            if "a_scale" in norms:  # CUDA graphs (ced/train/cuda_graphs.py): the same two numbers, in device memory
                loss, ce, lse = head(out.hidden_aux, targets, idx, real, norms["a_scale"], norms["b_scale"])
            else:
                scale = 1.0 / norms["n_valid"]
                loss, ce, lse = head(out.hidden_aux, targets, idx, real, scale, z * scale)
            if rows is not None:  # add-on CP: the per-row numbers, read only
                rows.update(ce=ce.detach(), lse=lse.detach())
            on_dec = sel["dec_mask"].reshape(-1).index_select(0, idx) & real
            zero = torch.zeros((), dtype=ce.dtype, device=ce.device)
            sums["ce_dec"] = torch.where(on_dec, ce, zero).sum()
            sums["ce_ctx"] = torch.where(~on_dec, ce, zero).sum()
            sums["lse_sq"] = lse.pow(2).sum()
        return loss, sums

    def _cp_loss(self, out, targets, norms: dict, sel: dict, cp_rows: tuple) -> tuple[torch.Tensor, dict]:
        """_micro_loss_selected of the one-head control with the Copy-Paste head (model.cp_head; ced/model/cp_head.py):
        the same positions (the valid targets, sel["all"]), the same weights (a = 1 / n_valid, b = z a at a real
        position, 0 at padding) and the same z-loss term on the LM head's LSE_V; the CE is the mixture's -log p(y).
        The CP head's terms (log s, log(1 - s), log p_CP(y)) at every position of the rows, query block by block under
        activation checkpointing, taken at the selected positions; z_copy = w_copy . h there.
        head_impl "fused": fused_linear_cp_ce (the base's kernel and a per-row correction; ced/train/fused_head.py).
        "chunked": the reference head's LSE_V and z_y (chunked_loss.token_lse_zy) into cp_head.mixture_log_prob, autograd.
        model.cp_head.force_lambda_zero: lambda = 0 and the CP head is not evaluated (its tensors get no gradient):
        "fused" runs fused_linear_cp_ce's lambda = 0 path, the base's loss and gradients bit for bit; "chunked" runs the
        base's own code (_micro_loss_selected). Metric sums: the base's (ce_dec / ce_ctx of the mixture, lse_sq) and
        cp_lambda (lambda) and ce_lm (p_LM's CE) over the valid targets. "Valid" is loss_valid (a per-token loss mask:
        the kept targets): sel["all"] and norms["n_valid"] are built from it, so a masked position adds nothing to the
        loss, the counts or any gradient (tests/test_cp_c3c4.py)."""
        from ced.model.cp_head import cp_terms, mixture_log_prob, mixture_parts
        from ced.train.chunked_loss import token_lse_zy
        from ced.train.fused_head import fused_linear_cp_ce

        c = self.cfg
        if c.model.cp_head.train_mode == "aux":  # cp-aux (ced/model/cp_aux.py): p_LM's loss plus the pointer's own loss
            return self._cp_aux_loss(out, targets, norms, sel, cp_rows)
        if c.model.cp_head.train_mode == "addon":  # add-on CP (ced/model/cp_aux.py): p_LM's loss plus the mixture on a stopped trunk
            return self._cp_addon_loss(out, targets, norms, sel, cp_rows)
        force = c.model.cp_head.force_lambda_zero
        if force and c.head_impl != "fused":
            return self._micro_loss_selected(out, targets, norms, sel)
        tokens, doc_ids = cp_rows
        z = c.loss.z_loss
        idx, real = sel["all"]
        hidden = out.hidden_aux
        h = hidden.reshape(-1, hidden.shape[-1]).index_select(0, idx)
        t = targets.reshape(-1).index_select(0, idx)
        w = real.to(compute_dtype(h))
        scale = 1.0 / norms["n_valid"]
        a, b = w * scale, w * (z * scale)
        W = self.model.lm_head_weight()
        if force:
            loss, ce, lse, ce_lm, lam = fused_linear_cp_ce(h, W, t, a, b, force_zero=True, chunk=c.head_chunk,
                                                           precision=c.head_precision)
        elif c.model.cp_head.kernel == "fused" and not hasattr(self._cp, "n_trunk"):
            # (the trunk copy's fused path, model.cp_head.trunk_copy with kernel "fused", is cp_terms' own below: cp_trunk.trunk_terms)
            # the fused path (ced/kernels/cp_pointer.py): the stacked projection GEMM and the pointer kernels at every
            # position of the rows, taken at the selected positions; the mixture inside the CE kernel (mix "kernel")
            from ced.kernels.cp_pointer import fused_terms

            with torch.autocast(device_type=hidden.device.type, enabled=False):
                terms = fused_terms(self._cp, hidden, tokens, targets, doc_ids)
                ls, l1s, lp, z_c = (x.reshape(-1).index_select(0, idx) for x in terms)
            loss, ce, lse, ce_lm, lam = fused_linear_cp_ce(h, W, t, a, b, z_c, ls, l1s, lp, chunk=c.head_chunk,
                                                           precision=c.head_precision, mix="kernel")
        else:
            head = self._cp
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                terms = cp_terms(head, hidden, tokens, targets, doc_ids, checkpoint=True, fn=self._cp_fn,
                                 pointer=out.cp_pointer)
                ls, l1s, lp = (x.reshape(-1).index_select(0, idx) for x in terms)
                z_c = head.copy_logit(h)
            if c.head_impl == "fused":
                loss, ce, lse, ce_lm, lam = fused_linear_cp_ce(h, W, t, a, b, z_c, ls, l1s, lp, chunk=c.head_chunk,
                                                               precision=c.head_precision)
            else:
                lse, z_y = token_lse_zy(h, W, t, precision=c.head_precision)
                ce = -mixture_log_prob(lse, z_y, z_c, ls, l1s, lp)
                loss = (a * ce + b * lse * lse).sum()
                ce, lse = ce.detach(), lse.detach()
                ce_lm = (lse - z_y).detach()
                lam = mixture_parts(lse, z_c.detach(), ls.detach(), l1s.detach())[0].exp()
        zero = torch.zeros((), dtype=ce.dtype, device=ce.device)
        ce, lse = torch.where(real, ce, zero), torch.where(real, lse, zero)
        on_dec = sel["dec_mask"].reshape(-1).index_select(0, idx) & real
        sums = {"ce_dec": torch.where(on_dec, ce, zero).sum(), "ce_ctx": torch.where(~on_dec, ce, zero).sum(),
                "lse_sq": lse.pow(2).sum(), "cp_lambda": torch.where(real, lam, zero).sum(),
                "ce_lm": torch.where(real, ce_lm, zero).sum()}
        return loss, sums

    def _cp_aux_loss(self, out, targets, norms: dict, sel: dict, cp_rows: tuple) -> tuple[torch.Tensor, dict]:
        """_cp_loss under model.cp_head.train_mode "aux" (cp-aux; ced/model/cp_aux.py states the form): the main loss is p_LM's,
        R1-entry's own code (_micro_loss_selected: the same positions, weights, z-loss and metric sums); plus aux_weight x the mean
        over the same valid targets (sel["all"], norms["n_valid"]) of the pointer's own cross-entropy (cp_aux.aux_ce of cp_terms'
        three terms, query block by block under activation checkpointing as in _cp_loss). The pointer's tensors are only in the
        second term. Metric sums: the base's, cp_aux (L_aux) and cp_aux_sink (the targets that are the sink) over the valid targets."""
        from ced.model.cp_aux import aux_ce
        from ced.model.cp_head import cp_terms

        loss, sums = self._micro_loss_selected(out, targets, norms, sel)
        tokens, doc_ids = cp_rows
        idx, real = sel["all"]
        hidden = out.hidden_aux
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            terms = cp_terms(self._cp, hidden, tokens, targets, doc_ids, checkpoint=True, fn=self._cp_fn,
                             pointer=out.cp_pointer)
            ce_aux, sink = aux_ce(*(x.reshape(-1).index_select(0, idx) for x in terms))
            zero = torch.zeros((), dtype=ce_aux.dtype, device=ce_aux.device)
            ce_aux = torch.where(real, ce_aux, zero)  # padding rows of the selection add nothing (and no gradient)
            loss = loss + ce_aux.sum() * (self.cfg.model.cp_head.aux_weight / norms["n_valid"])
        sums["cp_aux"] = ce_aux.detach().sum()
        sums["cp_aux_sink"] = (sink & real).to(ce_aux.dtype).sum()
        return loss, sums

    def _cp_addon_loss(self, out, targets, norms: dict, sel: dict, cp_rows: tuple) -> tuple[torch.Tensor, dict]:
        """_cp_loss under model.cp_head.train_mode "addon" (add-on CP; ced/model/cp_aux.py, "ADD-ON CP"): the main loss is R1-entry's,
        its own code (_micro_loss_selected: the same positions, weights, z-loss and metric sums), which also hands over its per-row CE_LM
        and LSE_V (no gradient); plus the mean over the same valid targets (sel["all"], norms["n_valid"]) of the mixture's CE,
        -cp_aux.addon_log_prob(LSE_V, -CE_LM, z_copy, log s, log(1 - s), log p_CP(y)), with p_LM's two numbers detached, z_copy =
        w_copy . h on h detached, and cp_terms over the forward's pointer inputs (their sources detached in the forward: cp_head.
        pointer_inputs), query block by block under activation checkpointing as in _cp_loss. So the mixture's gradient reaches the head's
        tensors only, and the trunk's and the LM head's gradients are the main loss's. Metric sums: the base's (ce_dec / ce_ctx of p_LM),
        ce_mix (the mixture's CE) and cp_lambda (lambda) over the valid targets."""
        from ced.model.cp_aux import addon_log_prob
        from ced.model.cp_head import cp_terms, mixture_parts

        rows: dict = {}
        loss, sums = self._micro_loss_selected(out, targets, norms, sel, rows=rows)
        tokens, doc_ids = cp_rows
        idx, real = sel["all"]
        hidden = out.hidden_aux.detach()  # the <copy> row's input (and cp_terms' shapes: the pointer reads out.cp_pointer)
        head = self._cp
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            terms = cp_terms(head, hidden, tokens, targets, doc_ids, checkpoint=True, fn=self._cp_fn, pointer=out.cp_pointer)
            ls, l1s, lp = (x.reshape(-1).index_select(0, idx) for x in terms)
            h = hidden.reshape(-1, hidden.shape[-1]).index_select(0, idx)
            z_c = head.copy_logit(h)
            dt = ls.dtype
            lse, log_lm = rows["lse"].to(dt), -rows["ce"].to(dt)  # p_LM's numbers, detached (the head call's own)
            zero = torch.zeros((), dtype=dt, device=ls.device)
            ce_mix = torch.where(real, -addon_log_prob(lse, log_lm, z_c.to(dt), ls, l1s, lp), zero)
            loss = loss + ce_mix.sum() / norms["n_valid"]
            lam = mixture_parts(lse, z_c.detach().to(dt), ls.detach(), l1s.detach())[0].exp()
        sums["ce_mix"] = ce_mix.detach().sum()
        sums["cp_lambda"] = torch.where(real, lam, zero).sum()
        return loss, sums

    def _micro_loss_tst(self, out, multi_targets, multi_valid, sel: dict, w: torch.Tensor, totals: dict, norms: dict,
                        s: int) -> tuple[torch.Tensor, dict]:
        """A micro-batch's share of the loss of a batch with K targets per position (Token Superposition Training;
        ced/train/tst.py, "THE LOSS"): one logit row per selected position, the normal head (the fused head's multi-target
        form, or the chunked one), weights a = v w / (the batch's denominator), b = z x (the position's main weight):
        - a bag phase (s > 1): w = bag_weights(s), the denominator sum v w over the batch (totals["w_total"]);
        - the multi-hot aux (s = 1): w = [1, lam / (K - 1), ...], the denominator n_valid, b from the main target only.
        Metric sums: ce_dec / ce_ctx (a bag phase: the weighted bag CE; the aux: the main target's CE), lse_sq,
        ce_first (each position's first target), and with the aux ce_multi (the valid aux targets' CE)."""
        c = self.cfg
        W = self.model.head_weight()
        z = c.loss.z_loss
        idx, real = sel["all"]
        K = multi_targets.shape[-1]
        h = out.hidden_aux.reshape(-1, out.hidden_aux.shape[-1]).index_select(0, idx)
        t = multi_targets.reshape(-1, K).index_select(0, idx)
        v = multi_valid.reshape(-1, K).index_select(0, idx) & real[:, None]
        dtype = compute_dtype(h)
        vw = v.to(dtype) * w.to(device=h.device, dtype=dtype)
        if s > 1:
            total = totals["w_total"]
            a = total(vw) if callable(total) else vw / total  # CUDA graphs: the divisor in device memory divides as the Python float does
            b = z * a.sum(1)
        else:
            a = vw / norms["n_valid"]
            b = z * a[:, 0]
        if c.head_impl == "fused":
            loss, ce, lse = fused_linear_multi_ce(h, W, t, a, b, chunk=c.head_chunk, precision=c.head_precision,
                                                  ce_form=c.loss.ce_form)
        else:
            ce, lse = token_multi_ce(h, W, t, precision=c.head_precision, ce_form=c.loss.ce_form)
            loss = (a * ce).sum() + (b * lse * lse).sum()
        zero = torch.zeros((), dtype=ce.dtype, device=ce.device)
        on_dec = sel["dec_mask"].reshape(-1).index_select(0, idx) & real
        ce, lse = ce.detach(), lse.detach()
        main = (vw * ce).sum(1) if s > 1 else torch.where(v[:, 0], ce[:, 0], zero)
        sums = {"ce_dec": torch.where(on_dec, main, zero).sum(), "ce_ctx": torch.where(~on_dec & real, main, zero).sum(),
                "lse_sq": torch.where(real, lse.pow(2), zero).sum(),
                "ce_first": torch.where(v[:, 0], ce[:, 0], zero).sum()}
        if s == 1:
            sums["ce_multi"] = torch.where(v[:, 1:], ce[:, 1:], zero).sum()
        return loss, sums

    def train_step(self) -> dict:
        if self._graphs is not None and self._graphs.first_step_pending:
            # CUDA graphs: a process's first step (the code below, as without them) runs on the graph runner's capture
            # stream, so every autograd accumulator a capture adds into is made there (ced/train/cuda_graphs.py, "Streams")
            with self._graphs.first_step():
                return self.train_step()
        start = time.perf_counter()
        c = self.cfg
        if self._trunk is not None:
            self._trunk.check_loaded()  # freeze_trunk: the finished trunk is in (init_from_checkpoint, or a resume)
        if self._cp is not None and self._cp.trunk_select_due(self.step):
            # the trunk copy's subset (model.cp_head.trunk_copy "subset"; ced/model/cp_trunk.py): once, at the start of step
            # trunk_select_step, the trunk_k heads with the largest learned logits; kept in the state dict (cp_head.trunk_keep)
            w = self._cp.trunk_log_weights().detach().double().exp().cpu()
            from ced.model.cp_trunk import head_ranking

            rank = head_ranking(self._cp)  # the ranking the selection reads (logged with this step's metrics)
            picked = self._cp.select_trunk_subset()
            self._trunk_selection = {"selected": picked, "rank": rank}
            print(f"[cp_trunk] step {self.step}: the subset's heads {picked} (global head = layer order x heads + head), their "
                  f"weights before the selection {[round(float(w[h]), 6) for h in picked]}", flush=True)
        if self._mx_flag is not None:  # the MXFP8 non-finite flag: this step's quantisations OR into it from zero
            self._mx_flag.zero_()
        nvfp4_event = None
        if self._nvfp4_until:  # NVFP4Config.until_step: the quantised path ends at this update (also after a resume past it)
            on = self.step < self._nvfp4_until
            if on != self._nvfp4_on:
                nvfp4_mod.set_phase(self.model, on)
                self._nvfp4_on = on
                if self.step == self._nvfp4_until:
                    nvfp4_event = self.step
        # Token Superposition Training: a later phase's first step starts with the switch (TSTPlan.switch), before its
        # batch; None at every other step and without TST.
        tst_event = self.tst.switch(self, self.step) if self.tst is not None else None
        dp_host = None
        self._input = None  # the data prefetch's input of this step (TrainConfig.data_prefetch); None with the option off
        if self.dp is None:
            batch = self._take_input(self.step).batch if c.data_prefetch else self.batch_at(self.step)
        else:  # data parallelism (D2): this rank's rows, and the step's host numbers summed over the ranks (_dp_batch)
            batch, dp_host = self._dp_batch()
        self._step_batch = batch  # a reference for a halt record (module docstring, "The input guards"); no copy
        if self._input is not None:  # the next update's input is built while this one runs (module docstring; prefetch.py)
            self._prefetch_next()
        nums = None if self._input is None else self._input.numbers  # the host numbers the prefetch built (else read below)
        T = self.loop_count(self.step) if self.is_looped else None
        lr = self.lr_at(self.step)
        if self.tst is not None:  # TST's (L) bag-phase multiplier and (R) re-warm after a switch: None where neither acts
            factor = self.tst.lr_factor(self.step)
            if factor is not None:
                lr = lr * factor
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)

        # Batch-level normalisers, from the host batch (the slot selection is keep-first-C).
        B = batch.tokens.shape[0]
        # The positions the loss counts (loss_valid): target_valid itself without a per-token loss mask, else the kept
        # targets. The one value the counts, the head selection and every loss branch read below.
        valid_loss = self.loss_valid(batch) if nums is None else self._input.valid_loss
        # under data parallelism the counts of the GLOBAL batch (summed over the ranks): each rank's loss is its local
        # sum over them, and the sum of the ranks' gradients the single process's (D2)
        counts = (self.batch_counts(batch) if nums is None else nums.counts) if dp_host is None else dp_host["counts"]
        n_dec, n_aux = counts["n_dec"], counts["n_aux"]
        rows_all = B if self.dp is None else B * self.dp.world  # the step's rows over every rank (micro_share)
        norms = {k: max(1, v) for k, v in counts.items()}
        random_aux = self.has_slots and batch.aux_mask is not None
        skip_aux = self._skip_aux and n_aux == 0  # no aux loss position in the whole batch (host counts)
        # Monitoring on the first micro-batch of every log_every-th step (the steps fit() prints).
        collect = self._stats_model is not None and (self.step + 1) % c.log_every == 0
        entmax_log = (self.step + 1) % c.log_every == 0
        stats: dict = {}
        # The stream's size by layer, on the first micro-batch of every log_stream_rms_every-th step (module docstring).
        rms_log = self._rms_encoder is not None and (self.step + 1) % c.log_stream_rms_every == 0
        stream_rms: list = []
        qmix_chosen: dict | None = None  # qmix's static query mix: this step's choice (the log record), and the weights' record of the
        qmix_record: dict | None = None  # monitored forward (module docstring of gdn_arms; both None at every other step)

        draft = self.is_v1 and c.model.loop.draft_loss_weight > 0
        # CUDA graphs (TrainConfig.cuda_graphs; ced/train/cuda_graphs.py): from the process's second step on, the
        # persistent gradients are zeroed in place and the micro-batches run through the graph runner
        graphs = self._graphs is not None and self._graphs.active
        if graphs:
            self._graphs.zero_grads()
        elif self._dp_grads is not None:  # data parallelism (D3): every .grad a view of the flat buffer, zeroed in place
            self._dp_grads.begin_step()  # (buffers are views of it: nothing more to attach or zero)
        else:
            self.optimizer.zero_grad(set_to_none=True)
            if self._wgrad is not None:  # the persistent float32 gradients re-attached and zeroed in place
                self._wgrad.start_step()
        total_loss = torch.zeros((), device=self.device)
        sums: dict[str, torch.Tensor] = {}
        n_dropped = torch.zeros((), dtype=torch.long, device=self.device)
        pass_stats = {"units": 0, "passes": 0, "slots": {}}  # the segment pass schedule: host counts of this step's plans
        autocast = torch.autocast(self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None)
        # rows per forward/backward: the config's, or with a shape schedule the phase's (the config's tokens a micro-batch)
        shape = None if self.shape is None else self.shape.at(self.step)
        micro = self.micro if shape is None else shape.micro
        # Token Superposition Training: the step's bag size, and the weights of a batch with several targets a position
        # (TSTPlan.weights: a bag phase, or the multi-hot aux); tst_s None without TST
        tst_s = self.tst.sched.s_at(self.step) if self.tst is not None else None
        tst_w = tst_totals = None
        if batch.multi_targets is not None:
            # (data parallelism: the totals of the global batch, from the integers added over the ranks in _dp_batch)
            tst_w, tst_totals = self.tst.weights(batch, self.step, None if dp_host is None else dp_host["tst_counts"])
        # the step's forwards: (rows, T) each. Without a looped middle: consecutive slices of `micro` rows at the
        # step's T, as always (micro_plan).
        plan_mb = self.micro_plan(self.step, B, T, micro)
        row_t = self.row_passes(self.step)
        if graphs:  # the step's micro-batches through the graph runner: the loop below runs none
            total_loss, sums, n_dropped, stats = self._graphs.micro_batches(
                batch, valid_loss, norms, plan_mb, collect, c.loss.z_loss, None if self._input is None else self._input.staged,
                tst=None if tst_w is None else (tst_s, tst_w, tst_totals))  # TST: a bag step's s, weights and host totals
            plan_mb = []
        for m0, (mb, T_mb) in enumerate(plan_mb):  # m0: the forward's index in the step (0: the monitored one)
            to = self._micro_to(mb)
            tokens, targets, valid, doc_ids, dec, aux = (to(t) for t in (batch.tokens, batch.targets, valid_loss,
                                                                          batch.doc_ids, batch.dec_mask, batch.aux_mask))
            norms["micro_share"] = tokens.shape[0] / rows_all
            kwargs = {"T": T_mb} if self.is_looped else {}
            if draft:
                kwargs["return_drafts"] = True
            if skip_aux:
                kwargs["aux"] = False
            if tst_s is not None and tst_s > 1 and self.tst.t.token_scale_rope:  # TST (b): rotary at s x the position
                kwargs["rope_scale"] = tst_s
            if tst_s is not None and tst_s > 1:  # TST (E): the GDN gates of a position of s tokens ({} when off)
                kwargs.update(self.tst.gdn_kwargs(self.step))
                kwargs.update(self.tst.bag_kwargs(self.step))  # TST (bag_input): {} unless "sum_sqrt"
            cap = self._micro_capacity(batch.dec_mask[mb])
            if cap != self._model_capacity:  # v1's trimmed slots, or the answer-span capacity (else the model's default)
                kwargs["capacity"] = cap
            if batch.pass_counts is not None:
                # the later passes' slots of this micro-batch, on the CPU and in fixed sizes (module docstring)
                if self.has_slots:
                    plan = self.pass_planner.plan(batch.pass_counts[mb], batch.dec_mask[mb], cap)
                else:  # the control with a second pass: every position of the row is a unit
                    plan = self.pass_planner.plan_positions(batch.pass_counts[mb])
                st = plan.stats()
                pass_stats["units"] += st["units"]
                pass_stats["passes"] += st["passes"]
                for name, (filled, allocated) in st["slots"].items():
                    have = pass_stats["slots"].setdefault(name, [0, 0])
                    have[0] += filled
                    have[1] += allocated
                kwargs["passes"] = plan.to(self.device, non_blocking=True)
            if collect and m0 == 0:
                self._stats_model.collect_stats = True
            if rms_log and m0 == 0:  # the stream's size by layer, read inside this forward (Encoder._stack)
                self._rms_encoder.stream_rms = []
            if self._qmix is not None and m0 == 0:
                from ced.model.gdn_arms import qmix_pending

                if self.step >= c.model.encoder.gdn_qmix_select_step and qmix_pending(self._qmix):
                    # the choice, at the start of the step, on this micro-batch: one extra forward (the call below, discarded)
                    qmix_chosen = self._qmix_choose(autocast, lambda: self.model(
                        tokens, doc_ids, dec, return_hidden=True,
                        varlen=self._varlen(batch.doc_ids[mb], batch.dec_mask[mb], cap, mb, self.step), **kwargs))
                if rms_log:  # the weights as the layers use them, read inside this step's forward (attn_res._monitor)
                    self._qmix.qmix_record = {}
            for _, st in self._entmax:  # every training forward; the support / scale snapshot on the log step's first
                st.active, st.snapshot = True, entmax_log and m0 == 0
            try:
                # host_inputs: the kept layer inputs of activation_checkpoint "host" / "host_overlap" go to the host
                # pool (else nothing)
                with autocast, act_checkpoint.host_inputs(self.host_pool):
                    out = self.model(tokens, doc_ids, dec, return_hidden=True,
                                     varlen=self._varlen(batch.doc_ids[mb], batch.dec_mask[mb], cap, mb, self.step), **kwargs)
            finally:
                if rms_log and m0 == 0:
                    stream_rms, self._rms_encoder.stream_rms = self._rms_encoder.stream_rms, None
                if self._qmix is not None and rms_log and m0 == 0:
                    qmix_record, self._qmix.qmix_record = self._qmix.qmix_record, None
                if collect and m0 == 0:
                    stats, self._stats_model.stats = self._stats_model.stats, {}
                    self._stats_model.collect_stats = False
                for _, st in self._entmax:
                    st.active = st.snapshot = False
                    st.end_microbatch()
            sel = self._head_selection(valid_loss[mb], batch.dec_mask[mb], cap,
                                       batch.aux_mask[mb] if batch.aux_mask is not None else None)
            if batch.multi_targets is not None:  # TST: several targets a position (a bag phase; the multi-hot aux)
                loss, part = self._micro_loss_tst(out, to(batch.multi_targets), to(batch.multi_valid), sel, tst_w,
                                                  tst_totals, norms, tst_s)
            else:
                loss, part = self._micro_loss(out, targets, valid, dec, norms, sel, aux_mask=aux, skip_aux=skip_aux,
                                              cp_rows=(tokens, doc_ids) if self._cp is not None else None)
            loss.backward()
            if self._dp_grads is not None:  # data parallelism: the record of reached parameters (GradBuffer)
                self._dp_grads.after_backward()
            total_loss += loss.detach()
            for k, v in part.items():
                sums[k] = sums.get(k, 0) + v.detach()
            n_dropped += out.n_dropped.sum()
        if self.dp is not None:  # data parallelism (D3): the gradients and the step's sums over the ranks (_dp_sync)
            total_loss, sums, n_dropped = self._dp_sync(total_loss, sums, n_dropped, graphs)
        if skip_aux:  # the zero gradient the skipped aux head gave its own parameters (Trainer._skip_aux)
            for p in self.model.aux_only_parameters():
                if p.requires_grad and p.grad is None:
                    p.grad = torch.zeros_like(p)
        # The sharded optimizer (ced/train/zero1.py; None without it): from here to the optimizer step only each tensor's
        # owner holds its global gradient (the reduce-scatter of _dp_sync), so the reads of whole gradients below are each
        # owner's values gathered exactly (Zero1.gather_values / gather_rows), the same numbers on every rank.
        z1 = self._z1
        # pass 2's own tensors: the norm of their gradient before clipping (0 while they get none: before a loop
        # switch step); read only, the clipping below is over every parameter as before
        if z1 is None:
            pass2_grads = [p.grad for _, p in self._pass2 if p.grad is not None]
            grad_norm_pass2 = float(torch.linalg.vector_norm(torch.stack([g.norm() for g in pass2_grads]))) if pass2_grads else 0.0
        else:
            pass2_norms = z1.gather_values([p for _, p in self._pass2 if p.grad is not None], lambda g: g.norm())
            grad_norm_pass2 = float(torch.linalg.vector_norm(torch.stack(pass2_norms))) if pass2_norms else 0.0
        # the residual scalars' gradient norm before clipping (read only, as pass 2's above). Formed here, from the
        # gradients as the backward left them, and kept as a 0-dim tensor on the device: the host reads it (float) with
        # the step's other metrics, AFTER the optimizer step. A float() here is a GPU sync that the step without the
        # scalars does not have: the host would wait for the last backward before it launches the clip's and the
        # optimizer's kernels. The number is the same either way (the clip scales the gradients after this tensor's
        # kernels, in the device's own order). MEASURED (2026-10-13, one slice): that sync is worth
        # 0.010 s of a 2.79 s step (legs old-nolog and base-sync); it was NOT the 0.31 s the unfused scalars cost. That
        # was device time of the scalar products and sums over the stream outside the compiled layers, forward and
        # backward (the profiles of legs prof-old and prof-fused: aten::mul 205 -> 33 ms a step).
        grad_norm_scalars = None
        if self._scalars is not None and self._trunk is not None:  # freeze_trunk: the scalars are frozen, they have no gradient
            grad_norm_scalars = torch.zeros((), device=self.device)
        elif self._scalars is not None:  # over the tensors the form stores (skip_gain, x0_gain, or the one a form has)
            grad_norm_scalars = torch.linalg.vector_norm(torch.stack(
                [p.grad.norm() for p in self._scalar_params] if z1 is None else
                z1.gather_values(self._scalar_params, lambda g: g.norm())))
        # the injection's gradient norm before clipping (a looped middle with injection; read only, as pass 2's above)
        grad_norm_injection = None
        if self._injection is not None:
            inj = ([p.grad.norm() for p in self._injection.parameters() if p.grad is not None] if z1 is None else
                   z1.gather_values([p for p in self._injection.parameters() if p.grad is not None], lambda g: g.norm()))
            grad_norm_injection = float(torch.linalg.vector_norm(torch.stack(inj or [torch.zeros((), device=self.device)])))
        # the input mix's w: the size of its gradient before clipping (a looped middle with injection "mix"; read
        # only). A 0-dim tensor on the device; the host reads it after the optimizer step, with the step's other
        # metrics (as the scalars' above). None while w gets no gradient (the one-pass steps before a switch).
        grad_w = None
        if self._mix is not None and self._mix.grad is not None:
            grad_w = (self._mix.grad.detach().abs().clone() if z1 is None else
                      z1.gather_values([self._mix], lambda g: g.detach().abs().reshape(()))[0])
        # Exact-zero weight gradients by linear (TrainConfig.log_zero_wgrad_every; ced/train/zero_wgrad.py): counted on
        # the device from the accumulated gradients before clipping, read by the host after the update. Read only.
        zero_wgrad_counts = None
        if self._zw is not None and (self.step + 1) % c.log_zero_wgrad_every == 0:
            from ced.train import zero_wgrad

            zero_wgrad_counts = (zero_wgrad.counts([p.grad for p in self._zw]) if z1 is None else
                                 z1.gather_rows(self._zw, lambda g: zero_wgrad.counts([g])[0], 2, [-1, -1]))
        # cuda_graphs "step": the clip and Muon's momentum and orthogonalisation as one graph, the rest of the optimizer
        # step eagerly after it (ced/train/cuda_graphs.py)
        step_graph = graphs and self._graphs.mode == "step"
        if self._trunk is not None:
            self._trunk.check_grads()  # freeze_trunk: before the clip and the update, so a touched trunk is never stepped
        z1_saved = None
        if z1 is not None:
            # the sharded optimizer: the clip over every tensor with a gradient (the same set on every rank, in model order,
            # as clip_grad_norm_ takes it) from the owners' norms; the optimizer step over this rank's tensors only (every
            # other tensor's .grad None meanwhile, attached again after the step's all-gather)
            z1.clip_set = [p for p in self.model.parameters() if p.grad is not None]
            if step_graph:  # the norms gathered eagerly, before the optimizer graph that reads them
                z1.gather_norms(z1.clip_set)
            z1_saved = z1.detach_others()
        if step_graph:
            grad_norm = self._graphs.clip_and_orthogonalise()
        elif z1 is not None:
            grad_norm = z1.clip(z1.clip_set, c.optim.grad_clip)
        elif self._cp_clip is not None:
            # add-on CP (ced/model/cp_aux.py, "ADD-ON CP"): the trunk's and LM head's tensors clipped by their own norm (the same list in
            # the same order as R1-entry's model.parameters(), so R1-entry's number and scale), then the head's by its own
            # freeze_trunk: the trunk has no gradient, so its norm is 0 (a device scalar, as every other path's)
            grad_norm = (torch.zeros((), device=self.device) if self._trunk is not None
                         else torch.nn.utils.clip_grad_norm_(self._cp_clip[0], c.optim.grad_clip))
            cp_grad_norm = torch.nn.utils.clip_grad_norm_(self._cp_clip[1], c.optim.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.optim.grad_clip)
        rot_every = c.optim.ours.rotation_log_every
        rot_before = ([p.detach().clone() for _, p in self._rot_params]
                      if rot_every and (self.step + 1) % rot_every == 0 else None)
        if z1 is None:
            try:
                if step_graph:
                    self._graphs.optimizer_rest()
                else:
                    self.optimizer.step()
            except Exception as error:  # noqa: BLE001 - raised again unless the step's numbers are not finite
                halted = self._optimizer_failed(error, total_loss, grad_norm, lr)
                if halted is None:
                    raise
                return halted
        else:
            failure = None
            try:
                if step_graph:
                    self._graphs.optimizer_rest()
                else:
                    self.optimizer.step()
            except Exception as error:  # noqa: BLE001 - every rank learns of it before the all-gather (Zero1.after_step)
                failure = error
            # one MAX on the host group (did any rank's step raise), then the all-gather of the updated weights; on a
            # failure every rank takes the failure path below with the same step numbers
            failure = z1.after_step(failure)
            z1.attach(z1_saved)
            if failure is not None:
                halted = self._optimizer_failed(failure, total_loss, grad_norm, lr)
                if halted is None:
                    raise failure
                return halted
        if self._trunk is not None:
            self._trunk.check_state(self.optimizer)  # freeze_trunk: no optimizer state was made for a trunk tensor
            self._trunk.after_first_step()  # and, once a process, the trunk hashed: nothing else in the step wrote to it in place
        rotation = self._rotation_read(rot_before) if rot_before is not None else None
        if self._scalars is not None and self._trunk is None:  # (freeze_trunk: nothing trains the scalars, nothing to put back)
            # the span-bounded form's stored values back into the range it reads them in; every other form: nothing
            self._scalars.project_layer_scalars()
        if self._ccr_floor is not None and self._trunk is None:
            # the CCR gates back to at least model.encoder.gdn_ccr_gate_floor (a clamp in place; ced/model/gdn_arms.py)
            from ced.model.gdn_arms import floor_gates_

            floor_gates_(self._ccr_floor)
        self.step += 1
        for _, st in self._entmax:
            st.end_step()
        if self._graphs is not None:  # after the process's first step, its gradients are kept (cuda_graphs)
            self._graphs.end_step()
        # The host's time to enqueue the step: from its start to the first read below that waits for the device (an
        # attribute for the timing runs, not a metric; on the slice the GPU is slower than the host, so with a full
        # launch queue this is bounded below by the GPU's time)
        self.host_enqueue_s = time.perf_counter() - start

        n_tokens = batch.tokens.numel() if self.dp is None else batch.tokens.numel() * self.dp.world  # the step's tokens
        sum_dec = float(sums["ce_dec"])
        sum_aux = float(sums["ce_aux" if self.has_slots else "ce_ctx"])
        metrics = {"loss": float(total_loss), "ce_dec": sum_dec / max(1, n_dec)}
        if random_aux:  # the aux head's own positions overlap the slots: no "context" or "all" split
            metrics.update(ce_aux=sum_aux / max(1, n_aux), lse_sq=float(sums["lse_sq"]) / max(1, n_dec + n_aux),
                           n_dec=n_dec, n_aux=n_aux)
        else:
            metrics.update(ce_ctx=sum_aux / max(1, n_aux), ce_all=(sum_dec + sum_aux) / max(1, n_dec + n_aux),
                           lse_sq=float(sums["lse_sq"]) / max(1, n_dec + n_aux), n_dec=n_dec, n_ctx=n_aux)
        if nvfp4_event is not None:  # the update at which NVFP4 ended (NVFP4Config.until_step); absent at every other step
            metrics["nvfp4_switch"] = nvfp4_event
        if tst_s is not None:  # Token Superposition Training (module docstring)
            metrics.update(tst_s=tst_s, tst_phase=self.tst.sched.phase_at(self.step - 1).index)
            if tst_event is not None:
                metrics["tst_switch"] = tst_event
            if tst_totals is not None and tst_s > 1:  # a bag step: the weighted bag CE (the loss without the z-term)
                wd, wc = tst_totals["w_dec"], tst_totals["w_ctx"]
                metrics.update(ce_dec=sum_dec / wd if wd else 0.0, ce_ctx=sum_aux / wc if wc else 0.0,
                               ce_all=(sum_dec + sum_aux) / max(wd + wc, 1e-30),
                               ce_bag_first=float(sums["ce_first"]) / max(1, n_dec + n_aux))
            elif tst_totals is not None:  # the multi-hot aux
                metrics.update(aux_lambda=tst_totals["lambda"], ce_multi=float(sums["ce_multi"]) / max(1, tst_totals["n_aux"]))
        if row_t is not None:  # a pass count per row: the step's mean, and the rows' own
            T = sum(row_t) / len(row_t)
        if self._mx_flag is None:
            norm_host = float(grad_norm)
        else:  # the flag comes to the host in the gradient norm's own read (one copy, both numbers exact in float64)
            # (under data parallelism the flag OR-ed over the ranks, _dp_sync)
            flag_dev = self._mx_flag if self.dp is None else self._dp_mx_flag
            norm_host, flag_host = torch.stack([grad_norm.detach().double().reshape(()),
                                                flag_dev.double()]).tolist()
        metrics.update({
            "lr": lr,
            "T": T or 0,
            "grad_norm": norm_host,
            "n_dropped": int(n_dropped),
            "tokens": n_tokens,
            # under data parallelism: the hash of the ranks' batch hashes (data_parallel.combined_hash), and each rank's
            "batch_hash": (batch_hash(batch) if nums is None else nums.hash) if dp_host is None else dp_host["batch_hash"],
        })
        if dp_host is not None:
            metrics.update(batch_hash_ranks=dp_host["batch_hash_ranks"], world_size=self.dp.world)
        if self._cp_clip is not None:  # add-on CP: the head's own gradient norm before its clip (grad_norm is the trunk's)
            metrics["cp_grad_norm"] = float(cp_grad_norm)
        if self._mx_flag is not None:  # the bits set this step (0: every quantised block finite); _check_finite reads it
            metrics[MX_FLAG_KEY] = int(flag_host)
        if batch.loss_mask is not None:  # the per-token loss mask: the kept targets and the valid ones (host counts)
            metrics.update(n_loss=counts["n_valid"], n_target_valid=(
                (int(batch.target_valid.sum()) if nums is None else nums.n_target_valid) if dp_host is None
                else dp_host["n_target_valid"]))
        metrics.update(((self.positions if shape is None else self._phase_positions[shape.phase]).shortfall(batch)
                        if nums is None else nums.short) if dp_host is None else dp_host["shortfall"])
        if self._input is not None:  # the data prefetch's milliseconds (build, device copy, the take's wait) and whether it hit
            metrics["prefetch"] = dict(self._input.stats)
        if rotation is not None:  # OptimConfig.ours.rotation_log_every: this update's measured rotation by role
            metrics["rotation"] = rotation
        ours_every = c.optim.ours.log_every
        if ours_every and self.step % ours_every == 0 and hasattr(self.optimizer, "ours_diagnostics"):
            metrics["ours"] = self.optimizer.ours_diagnostics()  # OptimConfig.ours.log_every: M1's angle, M2, M3
        if shape is not None:
            # After this update: the tokens read so far and the same in reference steps (the config's own B x L tokens
            # each), so a scheduled run's curve is read next to the unscheduled run's at equal tokens (whose record at
            # step s has read s x B x L tokens); and what this step read.
            seen = shape.tokens_before + n_tokens
            metrics.update(tokens_seen=seen, ref_step=seen / self.shape.tokens_per_ref, phase=shape.phase, rows=shape.rows,
                           seq_len=shape.seq_len)
        for k, v in sums.items():
            if k.startswith("ce_draft"):
                metrics[k] = float(v) / max(1, n_dec)
            elif k == "moe_balance":
                metrics[k] = float(v)
            elif k in ("cp_lambda", "ce_lm", "cp_aux", "cp_aux_sink", "ce_mix"):  # the Copy-Paste head: mean lambda and p_LM's CE over
                # the valid targets; cp-aux (ced/model/cp_aux.py): the mean auxiliary CE and the share of targets that are the sink;
                # add-on CP: the mixture's mean CE (ce_dec / ce_ctx are p_LM's there)
                metrics[k] = float(v) / max(1, n_dec + n_aux)
        metrics.update(monitoring_metrics(stats))
        if self._temp_w and self.step % c.log_every == 0:
            # "temperature": {attention module: [tau_h per query head]}, tau = 1 + temp_w (ssmax's ln(n) / ln(ref) on top)
            taus = {n: (1.0 + p.detach().double()).cpu().tolist() for n, p in self._temp_w}
            metrics["temperature"] = {n: [round(t, 6) for t in v] for n, v in taus.items()}
            metrics.update(tau_min=min(min(v) for v in taus.values()), tau_max=max(max(v) for v in taus.values()))
        if self._cp is not None and self._cp.cfg.qk_norm and self.step % c.log_every == 0:
            # The Copy-Paste head's QK-norm gain (model.cp_head.qk_norm): the pointer's temperature, 1 + qk_gain, trained
            # without weight decay (ced/train/optim.py), so it is logged as temp_w is
            # (the pointer's heads, model.cp_head.n_heads > 1: one gain a head, a list in head order)
            gains = (1.0 + self._cp.qk_gain.detach().double()).reshape(-1)
            metrics["cp_qk_gain"] = float(gains[0]) if gains.numel() == 1 else [float(g) for g in gains]
        if self._cp is not None and hasattr(self._cp, "n_trunk") and self.step % c.log_every == 0:
            # the trunk copy's weights over every global head (0 for a head out of the subset), in global head order
            from ced.model.cp_trunk import head_weights

            metrics["cp_trunk_w"] = [round(float(x), 6) for x in head_weights(self._cp)]
        if self._cp is not None and hasattr(self._cp, "n_trunk"):
            # the heads' ranking by their learned logits (the selection's order) every cp_trunk.RANK_EVERY steps
            from ced.model.cp_trunk import RANK_EVERY, head_ranking

            if self.step % RANK_EVERY == 0:
                metrics["cp_trunk_rank"] = head_ranking(self._cp)
            sel = getattr(self, "_trunk_selection", None)
            if sel is not None:  # the subset's selection step: the selected heads and the ranking they were read from
                metrics["cp_trunk_selected"] = sel["selected"]
                metrics["cp_trunk_rank"] = sel["rank"]
                self._trunk_selection = None
        if self._cp is not None and self._cp.cfg.pointer_input == "mix" and self.step % c.log_every == 0:
            # cp-mix's two readers (cp_head.PointerMix): their pseudo-queries' norms, as R1-entry's readers' are logged
            # (attn_res.BlockAttnRes.summary's query_norm): how far each has moved from its uniform start. The pointer's heads
            # (cp_head.HeadsMix): [q_0, k_0, q_1, k_1, ...], head by head
            m = self._cp.mix
            pairs = [(m.read_q, m.read_k)] if hasattr(m, "read_q") else [(x.read_q, x.read_k) for x in m.heads]
            metrics["cp_mix_query_norm"] = [float(torch.linalg.vector_norm(r.query.detach().double())) for p in pairs for r in p]
        if self._pass2:
            # The gate alpha of z_2 = z_1 + alpha (Block(z_1) - z_1) after this update, the LR its group had (the
            # run's LR x optim.pass2_lr_scale) and the gradient norm of pass 2's own tensors before clipping (they are
            # inside the global clip): every step, so that an arm whose alpha ends at its summed-LR bound shows, and a
            # gate gradient that dominates the clipped norm at switch-on shows too. Every log_every steps also pass
            # 2's own tensors: "pass2": {name: values}, the loop-index embedding as its RMS.
            metrics["alpha"] = float(self.model.alpha())
            if self.model.n_passes > 2:  # a third pass (the segment pass schedule): its gate
                metrics["alpha_pass3"] = float(self.model.alphas()[1])
            metrics["grad_norm_pass2"] = grad_norm_pass2
            metrics["lr_pass2"] = next(g["lr"] for g in self.optimizer.param_groups if g.get("pass2"))
            if self.step % c.log_every == 0:
                metrics["pass2"] = {n.rsplit(".", 1)[-1]: (round(float(p.detach().double().pow(2).mean().sqrt()), 8)
                                                           if n.startswith("loop_embedding_pass") else p.detach().double().cpu().tolist())
                                    for n, p in self._pass2}
        if row_t is not None:
            metrics["T_rows"] = row_t
        if self._injection is not None:
            # The injection of a looped middle: its gradient norm before clipping every step; every log_every steps
            # the decay's and dt's min / mean / max over the channels, B's drift from its start and B's size, after this update.
            metrics["grad_norm_injection"] = grad_norm_injection
            if self.step % c.log_every == 0:
                metrics["recur"] = self.model.recur_summary()
        if self._mix is not None:
            # The input mix of a looped middle, every step: w of z = w e + (1 - w) h after this update, the norm of
            # its gradient before clipping (0.0 while it gets none: the one-pass steps before a switch) and the LR
            # its group had (the run's LR x optim.pass2_lr_scale).
            metrics["w"] = float(self._mix.detach())
            metrics["grad_norm_w"] = 0.0 if grad_w is None else float(grad_w)
            metrics["lr_w"] = next(g["lr"] for g in self.optimizer.param_groups if g.get("pass2"))
        if self._depth is not None and self.step % c.log_every == 0:
            # R3's reader after this update: how far its pseudo-query and its key-norm gain have moved from their
            # start (the weights of this step's first forward are "depth_mix", from the monitoring stats above).
            metrics["depth_mix_query"] = {k: round(v, 6) for k, v in recur_depth.summary(self._depth).items()}
        if self._scalars is not None:
            # The residual scalars: their gradient norm before clipping every step; every log_every steps a_i and b_i
            # of every layer after this update (the effective a and b of the form: a_i = 1 + skip_gain[i] for
            # "skip_x0"; Encoder.layer_scalar_values), so a run's metrics say whether they moved.
            metrics["grad_norm_layer_scalars"] = float(grad_norm_scalars)  # the host read, after the update (see above)
            a_group = next((g for g in self.optimizer.param_groups if g.get("layer_scalar_a")), None)
            if a_group is not None:  # optim.layer_scalar_a_lr_scale other than 1: the LR encoder.skip_gain had this step
                metrics["lr_layer_scalar_a"] = a_group["lr"]
            if self.step % c.log_every == 0:
                metrics["layer_scalars"] = {k: [round(x, 8) for x in v] for k, v in self._scalars.layer_scalar_values().items()}
        if self._attn_res is not None and self.step % c.log_every == 0:
            # Block AttnRes: how far every reader's pseudo-query and key-norm gain have moved, after this update.
            # Every key of the summary but the labels: "query_norm" and "key_gain_dev" for R1 and R1-entry (the
            # metrics they always logged); the architecture arms' summaries add their own keys (attn_res.summary).
            s = self._attn_res.summary(self.model.encoder)
            metrics["attn_res"] = {k: [round(x, 6) for x in v] for k, v in s.items() if k != "readers"}
        if self._gdn_arms is not None and self.step % c.log_every == 0:
            # The GDN-side arms after this update: per layer that has them, lambda, the write query's norm, CCR's gates.
            from ced.model.gdn_arms import summary as gdn_arms_summary

            metrics["gdn_arms"] = gdn_arms_summary(self._gdn_arms)
        if rms_log:
            # The stream's size by layer on this step's first micro-batch, before this update (the forward's weights):
            # n_layers + 1 values, the stream entering each layer and the stream after the last (one host read).
            metrics["stream_rms"] = [round(x, 6) for x in torch.stack(stream_rms).double().cpu().tolist()]
        if self._qmix is not None:
            # qmix's static query mix (model.encoder.gdn_qmix_select_step): on the log_stream_rms_every cadence, each query mix's
            # token-mean weights over its sources on this step's first micro-batch, as the layer used them (the full softmax before the
            # choice; after it the chosen pair's, zero on the others), and at the choice step the pair each layer took with the weights
            # it was chosen on (the full softmax's, before the choice).
            from ced.model.gdn_arms import qmix_weights_metrics

            if qmix_record is not None:
                metrics["qmix_weights"] = qmix_weights_metrics(self._qmix, qmix_record)
            if qmix_chosen is not None:
                metrics["qmix_selection"] = qmix_chosen
        if zero_wgrad_counts is not None:
            # The exact zeros of this step's weight gradients by linear, in the start record's module order (one host read).
            from ced.train import zero_wgrad

            metrics["zero_wgrad"] = zero_wgrad.read(zero_wgrad_counts)
        if batch.pass_counts is not None:
            # The segment pass schedule: the mean pass count over the batch's decoder positions, and per later pass the
            # slots filled and allocated over the step's micro-batches (a pass absent from every micro-batch has no key).
            metrics["passes_mean"] = pass_stats["passes"] / max(1, pass_stats["units"])
            metrics["pass_slots"] = pass_stats["slots"]
            metrics["pass_fill"] = {k: round(v[0] / max(1, v[1]), 6) for k, v in pass_stats["slots"].items()}
        if self._entmax and self.step % c.log_every == 0:
            metrics.update(entmax_metrics(self._entmax))
        if self._graphs is not None:  # CUDA graphs: the replays, eager micro-batches, misses and captures so far
            metrics["cuda_graphs"] = self._graphs.metrics()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        metrics.update(step_seconds=seconds, tokens_per_s=n_tokens / seconds)
        if self.device.type == "cuda":  # the process's peaks so far (periodic evals included): read only
            metrics.update(peak_alloc_gib=round(torch.cuda.max_memory_allocated(self.device) / 2**30, 3),
                           peak_reserved_gib=round(torch.cuda.max_memory_reserved(self.device) / 2**30, 3))
        if self.dp is not None and self.dp.check_every and self.step % self.dp.check_every == 0:
            # data parallelism's debug replica check (data_parallel.ENV_CHECK_EVERY): the ranks' parameters after this
            # update compared; ReplicaMismatch on every rank on a difference. Outside the step's timings.
            from ced.train.data_parallel import check_replicas

            check_replicas(self.dp, self.model.named_parameters(), self.step)
        return metrics

    # ----------------------------------------------------------------------------------------
    # data parallelism (ced/train/data_parallel.py: D2 and D3; module docstring, "Data parallelism")

    def _dp_batch(self) -> tuple[Batch, dict]:
        """This rank's batch of the step (batch_at: its rows of the global batch), and the step's host numbers summed over
        the ranks on the host group, in ONE reduction before the first micro-batch: the bad-batch flag, the counts
        (batch_counts, loss masks included), n_target_valid, the position samplers' shortfall, and each rank's batch hash
        in its own slot. A batch that fails the loader's validation on any rank (with the fused GDN kernels also a
        micro-batch's boundary arrays, which the single process checks in the micro-batch loop) raises BatchError on
        every rank, before any micro-batch runs (_dp_batch_error). Before the process's first step, with the replica
        check on, the ranks' parameters are compared first."""
        from ced.train import data_parallel as D

        dp = self.dp
        if not self._dp_checked:  # the ranks start equal (a resume read the same file everywhere): checked once a process
            D.check_replicas(dp, self.model.named_parameters(), self.step)  # then every N-th
        self._dp_checked = True  # step with data_parallel.ENV_CHECK_EVERY
        error = None
        try:
            if self.cfg.data_prefetch:  # the input built by the prefetch thread (batch_at and the boundary check below, the
                batch = self._take_input(self.step).batch  # host numbers); a step nothing was submitted for is built here
            else:
                batch = self.batch_at(self.step)
                if self.fused_varlen:  # the micro-batches' boundary arrays (_varlen's check), here before any micro-batch
                    self._check_micro_varlen(batch, self.step)
        except BatchError as e:
            batch, error = None, e
        positions = self.positions
        short_keys = positions.shortfall_keys()  # the same on every rank (the sampler's configuration)
        # Token Superposition Training: [the step number (every rank must be at the same update: the switch is a function of it),
        # the optimizer states this rank reset or kept at this step's switch (only a sharded optimizer's count needs the sum), the
        # integers of the loss weights of a bag phase or the multi-hot aux (TSTPlan.host_counts)], added over the ranks in the same
        # reduction. A FIXED width (TSTPlan.host_width, zeros after the step's own integers; a rank without a batch sends zeros),
        # so a rank at another update still sends a vector of the others' length. Absent without TST: the reduction is what it was.
        pending, self._tst_switch_pending = getattr(self, "_tst_switch_pending", None), None
        tst_n = 0 if self.tst is None else 2 + self.tst.host_width
        tst_local = [0] * tst_n
        if error is None:
            if self.tst is not None:
                try:
                    counts_ = self.tst.host_counts(batch, self.step)
                    tst_local = [self.step, 0 if pending is None else pending[2]] + counts_ + [0] * (self.tst.host_width - len(counts_))
                except BatchError as e:
                    batch, error = None, self._batch_error(e, self.step)
        if error is None:
            nums = None if self._input is None else self._input.numbers  # the prefetch thread's, the same expressions
            counts = self.batch_counts(batch) if nums is None else nums.counts
            short = positions.shortfall(batch) if nums is None else nums.short
            n_tv = int(batch.target_valid.sum()) if nums is None else nums.n_target_valid
            digest = D.batch_digest(batch_hash(batch) if nums is None else nums.hash)
            local = [0, counts["n_valid"], counts["n_dec"], counts["n_aux"], n_tv] + [short[k] for k in short_keys]
        else:
            digest = 0
            local = [1, 0, 0, 0, 0] + [0] * len(short_keys)
            tst_local = [0] * tst_n
        n_base = len(local)
        local = local + tst_local
        slots = [0] * dp.world
        slots[dp.rank] = digest
        total = dp.sum_host(local + slots)
        if total[0]:
            raise self._dp_batch_error(error)
        digests = total[len(local):]
        host = {"counts": dict(zip(("n_valid", "n_dec", "n_aux"), total[1:4])), "n_target_valid": total[4],
                "shortfall": dict(zip(short_keys, total[5:n_base])), "batch_hash": D.combined_hash(digests),
                "batch_hash_ranks": [D.digest_hex(d) for d in digests]}
        if self.tst is not None:
            if total[n_base] != dp.world * self.step:  # every rank's step number, added: they are all the same or the sum is not
                raise RuntimeError(f"data parallelism: the ranks are not at the same update (their step numbers add to "
                                   f"{total[n_base]}, not {dp.world} x {self.step}); the TST schedule's switch is a function "
                                   "of the step and would fall on different updates")
            if pending is not None:  # the sharded optimizer: the event's count is the ranks' counts added (TSTPlan.switch)
                pending[0][pending[1]] = total[n_base + 1]
            host["tst_counts"] = total[n_base + 2:n_base + 2 + self.tst.host_count_size(self.step)]
        return batch, host

    def _dp_batch_error(self, error: BatchError | None) -> BatchError:
        """The BatchError every rank raises when a rank's batch failed the loader's validation: each failing rank's error
        is gathered on the host group (this failure path only); a failing rank raises its own with its rank added, every
        other rank one that names the failing ranks and the first one's message."""
        dp = self.dp
        mine = None if error is None else {"message": str(error), "info": json.loads(json.dumps(error.info, default=repr))}
        every = dp.gather_host(mine)
        bad = [r for r, e in enumerate(every) if e is not None]
        where = {"rank": dp.rank, "world_size": dp.world, "failing_ranks": bad}
        if error is not None:
            return BatchError(f"{error} [rank {dp.rank} of {dp.world}]", {**error.info, **where})
        first = every[bad[0]]
        return BatchError(f"update {self.step + 1}: the batch of rank {bad[0]} of {dp.world} failed the loader's validation "
                          f"(failing ranks {bad}); every rank halts: {first['message']}",
                          {**first["info"], **where, "failing_rank_info": first["info"]})

    def _dp_sync(self, total_loss: torch.Tensor, sums: dict, n_dropped: torch.Tensor, graphs: bool) -> tuple:
        """D3, after the micro-batch loop and before the gradient metrics and the clip: which parameters got a gradient on
        any rank (a MAX on the host group of GradBuffer's record; under CUDA graphs the set N1 fixed after the process's
        first step, so nothing is exchanged), .grad None for the others; one SUM all_reduce of each flat gradient buffer;
        then the step's loss, its metric sums, the dropped slots and the MXFP8 flag's bits in ONE packed float64 SUM
        all_reduce. Returns (total_loss, sums, n_dropped) of the global step, each in its own dtype (the graph runner's
        accumulators are not written)."""
        dp, buf = self.dp, self._dp_grads
        if not graphs:
            buf.stop_tracking()
            buf.detach_untouched(dp.max_host(buf.touched_flags()))
        if self._z1 is None:
            buf.all_reduce_(dp)
        else:  # the sharded optimizer (N2): the buffer reduce-scattered, each rank's slice its own tensors' global gradient
            for f in buf.flats.values():
                self._z1.reduce_scatter_(f)
        keys = sorted(sums)
        parts = [total_loss.detach().double().reshape(1)] + [sums[k].detach().double().reshape(1) for k in keys]
        parts.append(n_dropped.double().reshape(1))
        bits = 8  # FLAG's bits (ced/kernels/mx_ops.py: 1, 2, 4), each summed over the ranks, then OR-ed back
        if self._mx_flag is not None:
            f = self._mx_flag.to(torch.int64).reshape(1)
            parts.append(torch.cat([(f >> i) & 1 for i in range(bits)]).double())
        packed = dp.all_reduce_(torch.cat(parts))
        out_loss = packed[0].to(total_loss.dtype)
        out_sums = {k: packed[1 + i].to(sums[k].dtype) for i, k in enumerate(keys)}
        out_dropped = packed[1 + len(keys)].to(n_dropped.dtype)
        if self._mx_flag is not None:
            ranks = packed[2 + len(keys):]
            weights = torch.tensor([1 << i for i in range(bits)], dtype=torch.int64, device=ranks.device)
            self._dp_mx_flag = ((ranks > 0).to(torch.int64) * weights).sum().to(self._mx_flag.dtype)
        return out_loss, out_sums, out_dropped

    # D4 and D5 (module docstring, "Data parallelism"): rank 0's duties, the ranks' CUDA random streams

    @property
    def lead(self) -> bool:
        """Whether this process writes the run's files, logs and talks to W&B: always in one process; rank 0 under data
        parallelism (D4)."""
        return self.dp is None or self.dp.lead

    def _lead_duty(self, fn):
        """D4: fn() (an eval, a readout) run by rank 0 while every other rank waits at a barrier on the host group; in one
        process fn() itself. Returns fn's result on rank 0 (None on the others)."""
        if self.dp is None:
            return fn()
        out = fn() if self.dp.lead else None
        self.dp.barrier()
        return out

    def _eval_duty(self, fn):
        """The periodic and final evals (run_evals): with the evals sharded over the ranks (E1, DataParallel.shard_evals;
        ced/train/sharded_eval.py) every rank runs fn (it scores the rank's block of every set's batches; rank 0 joins and
        writes), then every rank waits at the barrier for rank 0's writes; otherwise _lead_duty(fn). Returns fn's result
        on rank 0 (None on the others)."""
        if self.dp is None or not self.dp.shard_evals:
            return self._lead_duty(fn)
        out = fn()
        self.dp.barrier()
        return out if self.dp.lead else None

    def _dp_world(self) -> tuple[int, int]:
        return (1, 0) if self.dp is None else (self.dp.world, self.dp.rank)

    def _dp_rng_event(self, step: int, from_world: int | None, why: str, changes: dict | None = None,
                      derive: bool = True) -> None:
        """D5: an entry in dp_info's history at optimizer step `step` (the world size it starts, the one before, why, the
        resume's changes of DP_RESUME_FREE), and with derive each rank's CUDA generator seeded from (seed, rank, step)
        (data_parallel.rank_seed; on a CUDA device only: "cuda" False records seeds that were not set, since a CPU run has
        no CUDA generator and its CPU generator is the replicated one)."""
        from ced.train.data_parallel import rank_seed

        W, rank = self._dp_world()
        event: dict = {"step": int(step), "world_size": W, "from_world_size": from_world, "why": why}
        if derive:
            seeds = [rank_seed(self.cfg.seed, r, step) for r in range(W)]
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(seeds[rank])
            event.update(cuda=self.device.type == "cuda", cuda_rng_seeds=seeds)
        if changes:
            event["changes"] = changes
        self.dp_info["history"].append(event)

    def _dp_own_rng(self):
        """This rank's CUDA generator state (its own device's only: get_rng_state_all would make a context on every
        visible device), None on the CPU."""
        return torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None

    def _dp_record(self, cuda_rng_ranks=None) -> dict:
        """The checkpoint's "data_parallel" entry: the world size it was written at, each rank's CUDA generator state in
        rank order (None on the CPU, or in one process, where "cuda_rng" holds it as always), the history."""
        return {"world_size": self._dp_world()[0], "cuda_rng_ranks": cuda_rng_ranks,
                "history": [dict(e) for e in self.dp_info["history"]]}

    def _dp_load(self, state: dict) -> None:
        """D5, at the end of load_state_dict when this process or the checkpoint ran data-parallel: the record of world
        sizes taken over from the file; at the same world size each rank's CUDA generator restored from the file (one
        process: "cuda_rng", already restored), at another (or without a saved state for this rank) re-derived from
        (seed, rank, the checkpoint's step) and recorded. The changes of DP_RESUME_FREE (micro_batch_size,
        activation_checkpoint: how a step is held in memory) are recorded with it."""
        saved = state.get("data_parallel")
        W, rank = self._dp_world()
        W0 = 1 if saved is None else int(saved["world_size"])
        step = int(state["step"])
        self.dp_info = {"world_size": W, "history": [dict(e) for e in (saved or {}).get("history", [])]}
        own = to_dict(self.cfg)
        source = checkpoint_config(state.get("config") or {})
        source = {**{k: v for k, v in SAVED_CONFIG_DEFAULTS.items() if k not in source}, **source}  # a key saved absent
        changes = {f: [source.get(f), own.get(f)] for f in DP_RESUME_FREE if f in source and source.get(f) != own.get(f)}
        if W0 != W:
            self._dp_rng_event(step, W0, "the world size changed", changes)
            return
        ranks = None if saved is None else saved.get("cuda_rng_ranks")
        if self.dp is not None and self.device.type == "cuda":
            if ranks is not None and len(ranks) == W and ranks[rank] is not None:
                torch.cuda.set_rng_state(ranks[rank], self.device)
            else:  # a checkpoint written without the ranks' CUDA states (on the CPU): derived as at a start
                self._dp_rng_event(step, W0, "no saved CUDA generator state for this rank", changes)
                return
        if changes:
            self._dp_rng_event(step, W0, "the resume changed how a step is held in memory", changes, derive=False)

    # checkpoints

    def data_state(self) -> dict:
        """What identifies the data stream; the position in it is the step (less data_start_step, which is part of the
        identity when it is not 0, and absent otherwise: every checkpoint before the key compares equal)."""
        state = {"kind": self.cfg.data.kind, "seed": self.cfg.data.seed, "seq_len": self.cfg.data.seq_len,
                 "batch_size": self.cfg.data.batch_size}
        if self.cfg.data.kind == "shards":
            state["fingerprint"] = self.data.shards.fingerprint()
            extends = getattr(self.data.shards, "extends_fingerprint", None)
            if extends is not None:  # a union read with union_order "sets_in_order": the first set it begins with (stage_data_relation)
                state[DATA_EXTENDS_KEY] = extends
        if self.cfg.data_start_step:  # a stage whose stream starts later (TrainConfig.data_start_step)
            state["data_start_step"] = self.cfg.data_start_step
        if self.shape is not None:  # the step's place in the stream follows from the phases (absent without a schedule)
            state["shape_schedule"] = [list(e) for e in self.cfg.shape_schedule]
        if self.tst is not None:  # TST's bag phases and (d)'s sources place the steps in the stream (absent otherwise)
            state.update(self.tst.data_state())
        if self.cfg.data.loss_mask != LOSS_MASK_ALL:  # the per-token loss mask and its span files (absent with "all")
            state["loss_mask"] = self.cfg.data.loss_mask
            spans = getattr(self.data, "loss_spans_state", None)  # the packer's check_loss_spans record
            if spans is not None and any(p is not None for p in spans["parts"]):
                state["loss_spans"] = spans["parts"]
        return state

    def state_dict(self, cuda_rng_ranks=None, path: Path | None = None) -> dict:
        """The checkpoint's content. Under data parallelism (D5) the state is rank 0's (the replicas are equal), "cuda_rng"
        is None and "data_parallel" holds the world size, every rank's CUDA generator state (cuda_rng_ranks, gathered by
        save_checkpoint) and the history of world sizes; one process writes "data_parallel" only for a run that ran on
        several ranks before (dp_info), so every other checkpoint is the one it always was. The config is saved without
        the keys of SAVED_CONFIG_DEFAULTS at their defaults. Under the sharded optimizer (N2) "optimizer" is the manifest
        of the file set written by parameter (ced/train/zero1.py; path: the checkpoint's file, beside which the optimizer
        shards are)."""
        state = {
            "step": self.step,
            "config": saved_config_dict(self.cfg),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self._z1 is None else self._z1.manifest(path, self.model),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" and self.dp is None else None,
            "python_rng": random.getstate(),
            "wandb_run_id": self.wandb_run_id,
            "data_state": self.data_state(),
            "init_from": self.init_info,
            "resumed_from": self.resume_info,
        }
        if self.dp_info is not None:
            state["data_parallel"] = self._dp_record(cuda_rng_ranks)
        if self._trunk is not None:  # freeze_trunk: the trunk's checksum now and at its start (refused if the trunk moved); absent without
            state["freeze_trunk"] = self._trunk.record()
        return state

    def load_state_dict(self, state: dict, grown: list[str] | None = None, check_data: bool = True,
                        source: Path | None = None) -> None:
        """Load a checkpoint's whole training state. grown (resume_from_checkpoint's T = 1 trunk only): the names of
        pass 2's own tensors, which the state does not have: they keep their built init and get no optimizer state
        (it is created at their first gradient, like any parameter's), and the optimizer's other state is the trunk's.
        check_data False: the caller has checked the data identity itself (resume_from_checkpoint under stage
        "continue", by stage_data_relation); every other load refuses a checkpoint of other data, as always.
        source: the checkpoint's file, read for a file set written by parameter (the sharded optimizer, N2: its optimizer
        shards are beside it; ced/train/zero1.py load_for), which any process loads, sharded or not; a sharded process
        keeps only its own tensors' state of either kind of file. Every other load is what it was."""
        saved = state.get("data_state")
        if check_data and saved is not None and saved != self.data_state():
            raise ValueError(f"the checkpoint was trained on different data: {saved} != {self.data_state()}")
        # The LR rule of a scheduled run (TrainConfig.shape_lr) is part of what the checkpoint's optimizer state was
        # made with: a run goes on under the rule it started with. A checkpoint written before the key is "clock".
        saved_lr = (state.get("config") or {}).get("shape_lr", "clock")
        if saved_lr != self.cfg.shape_lr:
            raise ValueError(f"the checkpoint was trained with shape_lr {saved_lr!r}, this config has {self.cfg.shape_lr!r}: a "
                             "run keeps the LR rule it started with")
        # The MXFP8 non-finite flag's form (FP8Config.nonfinite_flag_form; module docstring, "The non-finite flag"):
        # its default changed from "ordered" to "pure", so a flagged run resumed from its runs file would change form
        # silently. A run keeps the form it trained (a checkpoint without the key trained "ordered"); where either
        # side runs no form (the flag off, or no operator form) nothing is compared.
        saved_fp8 = (checkpoint_config(state.get("config") or {}).get("model") or {}).get("fp8")
        saved_form = flag_form_of(saved_fp8) if isinstance(saved_fp8, dict) else None
        own_form = flag_form_of(self.cfg.model.fp8)
        if saved_form is not None and own_form is not None and saved_form != own_form:
            raise ValueError(f"the checkpoint was trained with the MXFP8 non-finite flag's {saved_form!r} form, this "
                             f"config runs {own_form!r}: a run keeps the form it trained (set "
                             f"model.fp8.nonfinite_flag_form {saved_form!r}; its default is 'pure' "
                             "in the current code)")
        from ced.train import zero1 as zero1_mod

        if self._z1 is not None or zero1_mod.is_by_parameter(state.get("optimizer")):  # N2: absent from every other load
            state = {**state, "optimizer": zero1_mod.load_for(state["optimizer"], source, self.optimizer, self.model,
                                                               self._z1)}
        if grown:
            self._load_grown(state, grown)
        else:
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
        if self._trunk is not None:  # freeze_trunk: the loaded trunk is the checkpoint's recorded one, which becomes this run's start
            self._trunk.verify_load(state.get("freeze_trunk"), state.get("init_from"), self.optimizer)
        elif state.get("freeze_trunk") is not None:
            raise ValueError("the checkpoint is a freeze_trunk run's (its trunk is frozen and its optimizer holds the head's state only): "
                             "this config has freeze_trunk false")
        self.step = state["step"]
        self._prefetch_discard()  # a prefetch built for the state before this load does not outlive it (prefetch.py, "Resume")
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        self.wandb_run_id = state["wandb_run_id"]
        self.init_info = state.get("init_from")
        self.resume_info = state.get("resumed_from")
        if self.dp is not None or state.get("data_parallel") is not None:  # D5: absent from every other load
            self._dp_load(state)

    def _load_grown(self, state: dict, grown: list[str]) -> None:
        """The model and optimizer state of a single-pass trunk into this gated two-pass trainer. Model: every tensor
        of the trunk strictly (names and shapes), the tensors of `grown` left as built. Optimizer: pass 2's groups come
        last (ced/train/optim.py), so the trunk's groups hold the same tensors in the same order as this optimizer's
        first groups, and its state is taken parameter by parameter; the parameters of `grown` start without state."""
        missing, unexpected = self.model.load_state_dict(state["model"], strict=False)
        if sorted(missing) != sorted(grown) or unexpected:
            raise ValueError(f"the trunk's model state does not match: missing {sorted(missing)} (expected {sorted(grown)}), "
                             f"not in this model {sorted(unexpected)}")
        own = self.optimizer.state_dict()["param_groups"]
        src = state["optimizer"]["param_groups"]
        n_new = sum(1 for g in own if g.get("pass2"))
        sizes = [len(g["params"]) for g in src]
        if len(src) != len(own) - n_new or sizes != [len(g["params"]) for g in own[: len(src)]]:
            raise ValueError(f"the trunk's optimizer groups {sizes} do not match this optimizer's "
                             f"{[len(g['params']) for g in own]} (the last {n_new} are pass 2's)")
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        for i, st in state["optimizer"]["state"].items():  # the trunk's index i is this optimizer's index i
            for t in st.values():
                if isinstance(t, torch.Tensor) and t.dim() and t.shape != params[i].shape:
                    raise ValueError(f"the trunk's optimizer state {i} has shape {tuple(t.shape)}, this parameter "
                                     f"{tuple(params[i].shape)}")
        self.optimizer.load_state_dict({"state": state["optimizer"]["state"], "param_groups": src + own[len(src):]})

    def _growth(self, source_model, what: str) -> list[str]:
        """The names of pass 2's own tensors when this run's model is the gated two-pass growth of the single-pass
        model config `source_model` (ced.config.one_pass_model of this run's equals it apart from
        IMPLEMENTATION_FIELDS); [] when it is not such a growth. A growth that would change the function is refused
        (growth_problems).

        The add-on head's growth (TrainConfig.freeze_trunk; ced/train/freeze_trunk.py): with freeze_trunk on and a source that is this
        run's model with model.cp_head off (the source's cp_head section replaces this run's; everything else equal apart from
        IMPLEMENTATION_FIELDS), the head's own tensors (the state-dict names under cp_head.) are the growth: new at their own-generator
        init, the trunk loaded strictly. Without freeze_trunk the old refusal stands."""
        own = self.cfg.model
        if self.cfg.freeze_trunk and own.cp_head.enabled and not source_model.cp_head.enabled:
            if model_differences(to_dict(source_model), to_dict(dataclasses.replace(own, cp_head=source_model.cp_head))):
                return []
            return [n for n in self.model.state_dict() if not freeze.is_trunk(n)]
        if recur_mix(own):
            # A looped middle with the scalar input mix grown from the plain control (module docstring, "The late
            # switch-on"): the one new tensor is w. No identity is asked of it: at its configured start the model is
            # near the source's, not equal (by design); the layers and their residual scales are the source's.
            if model_differences(to_dict(source_model), to_dict(one_pass_model(own))):
                return []
            from ced.model.scaling import residual_scales

            a, b = residual_scales(source_model), residual_scales(own)
            if a != b:
                raise ValueError(f"{what}: the (encoder, decoder) residual scales change from {a} to {b} when the "
                                 "looped middle is switched on: the layers would not be the source's")
            return [RECUR_MIX]
        if own.arch not in GATED_ARCHES or own.loop.pass2_gate == "none":
            return []
        if model_differences(to_dict(source_model), to_dict(one_pass_model(own))):
            return []
        problems = growth_problems(source_model, own, self.model)
        if problems:
            raise ValueError(f"{what}: growing the single-pass model to two passes would change its function:\n  "
                             + "\n  ".join(problems))
        return self.model.pass2_parameter_names()

    def init_from_checkpoint(self, path: str | Path | None = None) -> dict:
        """Start this NEW run (step 0) from the MODEL weights of the checkpoint file `path` (default
        cfg.init_from); everything else stays as built from this run's config: a fresh optimizer (no
        state), the LR schedule and step counter from 0, the RNG states of cfg.seed, the data stream
        from step 0. Strict: refused (ValueError) unless the source's model config equals this run's
        (apart from IMPLEMENTATION_FIELDS) and its model state has exactly this model's parameter names and shapes. One
        growth is accepted: this run's model is a gated two-pass model (model.loop.pass2_gate) and the source's is its
        single-pass model (ced.config.one_pass_model); then every tensor of the source is loaded as strictly, and only
        pass 2's own tensors are new, at their function-preserving init (the names are recorded as "grown"); refused if
        the growth would change the function (growth_problems). The file is read once; its sha256 is computed from the
        same bytes. Returns the record kept in this run's checkpoints and metrics start events (self.init_info)."""
        path = Path(path if path is not None else self.cfg.init_from)
        if self.step != 0 or any(self.optimizer.state.values()):
            raise ValueError("init_from starts a new run: this trainer has already trained or resumed")
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        state = torch.load(io.BytesIO(data), weights_only=False, map_location="cpu")
        del data
        source_cfg = from_dict(TrainConfig, checkpoint_config(state["config"]))
        diffs = model_differences(to_dict(source_cfg.model), to_dict(self.cfg.model))
        grown = self._growth(source_cfg.model, f"init_from {path}") if diffs else []
        if diffs and not grown:
            raise ValueError(f"init_from {path}: the source's model config differs:\n  " + "\n  ".join(diffs))
        src, own = state["model"], self.model.state_dict()
        problems = [f"missing in the source: {k}" for k in own if k not in src and k not in grown]
        problems += [f"not in this model: {k}" for k in src if k not in own]
        problems += [f"shape of {k}: source {tuple(src[k].shape)} -> this model {tuple(v.shape)}"
                     for k, v in own.items() if k in src and src[k].shape != v.shape]
        if problems:
            raise ValueError(f"init_from {path}: the model state does not match:\n  " + "\n  ".join(problems))
        if self._trunk is not None:
            # freeze_trunk keeps the finished trunk as it is: a tensor this model holds in another dtype would be the file's
            # values rounded, a trunk that is not the source's, and the stage's final could never equal it
            cast = [f"{k}: source {src[k].dtype} -> this model {v.dtype}" for k, v in own.items()
                    if freeze.is_trunk(k) and k in src and src[k].dtype != v.dtype]
            if cast:
                raise ValueError(f"init_from {path}: freeze_trunk loads the source's trunk bytes as they are, and {len(cast)} tensors would be "
                                 f"cast (this run's dtype {self.cfg.dtype!r}): " + "; ".join(cast[:3]))
        if grown:
            missing, unexpected = self.model.load_state_dict(src, strict=False)
            assert sorted(missing) == sorted(grown) and not unexpected, (missing, unexpected)
        else:
            self.model.load_state_dict(src, strict=True)
        self.init_info = {
            "path": str(path), "sha256": sha, "step": int(state["step"]), "source_steps": source_cfg.steps,
            "source_seed": source_cfg.seed, "source_data_state": state.get("data_state"),
            "source_init_from": state.get("init_from"), "tensors": len(src),
        }
        if grown:  # the T = 1 checkpoint grown to two passes: pass 2's own tensors, new at their init
            self.init_info["grown"] = grown
            if self._mix is not None:  # the input mix's w as the run starts with it (not the identity unless 1.0)
                self.init_info["recur_mix_init"] = float(self._mix.detach())
        if self._trunk is not None:  # freeze_trunk: the loaded trunk's checksum is the stage's start, beside the source file's own
            sha, count = self._trunk.set_start()
            source_sha, _ = freeze.trunk_sha256(src.items())
            self.init_info["freeze_trunk"] = {"trunk_sha256": sha, "tensors": count, "source_trunk_sha256": source_sha,
                                              "equal_to_source": sha == source_sha}
            if sha != source_sha:  # the dtypes are equal (refused above), so the load did not keep the file's bytes
                raise ValueError(f"init_from {path}: the trunk loaded ({sha}) is not the source file's ({source_sha}): freeze_trunk starts from "
                                 "the finished trunk as it is")
        return self.init_info

    def resume_from_checkpoint(self, path: str | Path | None = None) -> dict:
        """Start this NEW run from the WHOLE training state of another run's checkpoint file `path` (default
        cfg.resume_from): model, optimizer state, step counter and RNG states, as a resume of that run would load
        them, but under this run's config and in this run's directory. For a nested budget's branch: the longer run's
        kept checkpoint (cfg.checkpoint_steps) at this run's decay start, continued with this run's step count.
        Strict (ValueError): the source's config must equal this run's apart from BRANCH_FIELDS and BRANCH_SECTIONS;
        the data must be the same (load_state_dict); the source must have steps left to train here; and lr_schedule
        must give both runs the same LR at every update the source has made. With these the run equals, bit for bit,
        the one this config trains on its own. The file is read once; its sha256 is computed from the same bytes.
        Returns the record kept in this run's checkpoints and metrics start events (self.resume_info).

        The loop experiment's branch (TrainConfig.loop_switch_step S > 0): the source may instead be the T = 1 TRUNK
        of this run, a run of trunk_config(this config) (the single-pass model, the schedule's reference length as
        its steps), at exactly step S. Its whole state is continued as above (the weights, the optimizer state, the
        RNG states and the data position are the trunk's); only pass 2's own tensors are new, at their
        function-preserving init and without optimizer state. Refused: a source at another step than S, a trunk whose
        schedule length is not this run's reference length (optim.schedule_steps), and a growth that would change the
        function (growth_problems). The run is then, bit for bit, the run this config trains from its own start,
        whose first S steps are the trunk's (tests/test_loop_t2.py). The branch may hold its steps in memory
        differently from the trunk (TRUNK_FREE_FIELDS: micro_batch_size, activation_checkpoint; recorded as
        "trunk_changes"): a two-pass step needs more memory than the trunk's.

        A stage (TrainConfig.stage "continue"; module docstring): the keys of STAGE_FREE may differ too (against the
        source, or against the trunk with a switch step), the trunk's schedule length is not compared (the LR check
        above is), and the data identity is checked by stage_data_relation instead of equality (the same shard set with
        the source's data_start_step, or a new one with data_start_step = the source's step). Recorded as "stage": the
        relation ("same" / "new"), data_start_step, both data identities, both decay fractions, the first update's LR."""
        path = Path(path if path is not None else self.cfg.resume_from)
        if self.shape is not None:
            raise ValueError("resume_from is not built for a run with a shape_schedule")
        if self.step != 0 or any(self.optimizer.state.values()):
            raise ValueError("resume_from starts a new run: this trainer has already trained or resumed")
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        state = torch.load(io.BytesIO(data), weights_only=False, map_location="cpu")
        del data
        source_cfg = from_dict(TrainConfig, checkpoint_config(state["config"]))
        # A stage (TrainConfig.stage "continue"; module docstring): STAGE_FREE may differ too, and the data identity is
        # checked by stage_data_relation instead of equality. Every other run: as always.
        stage = self.cfg.stage == STAGE_CONTINUE
        # A stage at another row shape (stage_shape "tokens"): STAGE_SHAPE_FREE too, and the data relation checks the
        # tokens a step instead of the row shape (stage_data_relation)
        shape = stage and self.cfg.stage_shape == STAGE_SHAPE_TOKENS
        free = (STAGE_FREE + (STAGE_SHAPE_FREE if shape else ())) if stage else ()
        if stage and self.cfg.data.union_order == "sets_in_order":  # the union may differ from the source's set; the data relation decides
            free = free + STAGE_UNION_FREE
        precision = stage and self.cfg.precision_switch is True  # TrainConfig.precision_switch: PRECISION_FREE too
        if precision:
            free = free + PRECISION_FREE
        diffs = branch_differences(to_dict(source_cfg), to_dict(self.cfg), free=free)
        step = int(state["step"])
        grown: list[str] = []
        S = self.cfg.loop_switch_step
        if diffs and S:
            # not this run's own config: the source must then be this run's T = 1 trunk, whose state at step S is
            # this run's state at its switch step
            trunk_diffs, memory_changes = trunk_differences(source_cfg, self.cfg, free=free)
            if trunk_diffs:
                raise ValueError(f"resume_from {path}: the source's config is neither this run's nor its T = 1 trunk's "
                                 "(ced.train.trainer.trunk_config); against the trunk it differs:\n  " + "\n  ".join(trunk_diffs))
            if step != S:
                raise ValueError(f"resume_from {path}: the trunk's checkpoint is at step {step}, and this run switches "
                                 f"the loop on at loop_switch_step {S}: branch from the trunk's checkpoint of step {S}")
            # A stage's source is a stage of its own, cut at S (its steps are its stage's, not the schedule's length):
            # the LR check below, update by update, is what ties the two schedules together there.
            if not stage and schedule_total(source_cfg) != schedule_total(self.cfg):
                raise ValueError(f"resume_from {path}: the trunk's LR schedule has {schedule_total(source_cfg)} steps and "
                                 f"this run's reference schedule {schedule_total(self.cfg)} (optim.schedule_steps): the "
                                 "branch follows the trunk's schedule, so set optim.schedule_steps to the trunk's steps")
            grown = self._growth(source_cfg.model, f"resume_from {path}")
            diffs = [] if grown else diffs
        if diffs:
            raise ValueError(f"resume_from {path}: the source's config differs:\n  " + "\n  ".join(diffs))
        if not 0 < step < self.cfg.steps:
            raise ValueError(f"resume_from {path}: the source is at step {step}; this run has {self.cfg.steps} steps")
        first = next((k for k in range(step) if lr_of(source_cfg, k) != lr_of(self.cfg, k)), None)
        if first is not None:
            raise ValueError(f"resume_from {path}: the source's LR schedule ({source_cfg.steps} steps) differs from this "
                             f"run's ({self.cfg.steps} steps) at update {first}, before the source's step {step}")
        relation = None
        if stage:
            relation, problems = stage_data_relation(state.get("data_state"), self.data_state(), step, shape=shape)
            if problems:
                raise ValueError(f"resume_from {path}: stage 'continue' cannot continue the source on this data:\n  "
                                 + "\n  ".join(problems))
            if relation == "extends":
                self._check_extends_position(step, path)
            if self.tst is not None:  # TST: a stage on the source's own set starts after the schedule's last switch (check_stage_source)
                self.tst.check_stage_source(step)
        self.load_state_dict(state, grown=grown, check_data=not stage, source=path)
        self.wandb_run_id = None  # a new run
        self.resume_info = {
            "path": str(path), "sha256": sha, "step": step, "source_steps": source_cfg.steps,
            "source_checkpoint_steps": list(source_cfg.checkpoint_steps), "source_resumed_from": state.get("resumed_from"),
        }
        if stage:  # what the stage changed against its source (absent for every other resume_from)
            self.resume_info["stage"] = {
                "kind": STAGE_CONTINUE, "data": relation, "data_start_step": self.cfg.data_start_step,
                "source_data_state": state.get("data_state"), "data_state": self.data_state(),
                "source_decay_fraction": source_cfg.optim.decay_fraction, "decay_fraction": self.cfg.optim.decay_fraction,
                "lr_first_update": lr_of(self.cfg, step)}
            if shape:  # stage_shape "tokens": the rows a step and the row length, the source's and this run's
                src = state["data_state"]
                self.resume_info["stage"]["shape"] = {
                    "source": [src["batch_size"], src["seq_len"]], "this_run": [self.cfg.data.batch_size, self.cfg.data.seq_len],
                    "tokens_per_step": self.cfg.data.batch_size * self.cfg.data.seq_len,
                    "memory": {f: [getattr(source_cfg, f), getattr(self.cfg, f)] for f in ("micro_batch_size", "activation_checkpoint")}}
        if precision:  # the source's precision and this run's (TrainConfig.precision_switch; absent with it off)
            self.resume_info["precision"] = {
                "source": {"nvfp4": to_dict(source_cfg.model.nvfp4), "fp8": to_dict(source_cfg.model.fp8)},
                "this": {"nvfp4": to_dict(self.cfg.model.nvfp4), "fp8": to_dict(self.cfg.model.fp8)}}
        if grown:  # the T = 1 trunk grown to two passes at its switch step: pass 2's own tensors, new at their init
            self.resume_info["grown"] = grown
            if memory_changes:  # TRUNK_FREE_FIELDS this run sets differently from its trunk: {field: [trunk, this run]}
                self.resume_info["trunk_changes"] = memory_changes
            if self._mix is not None:  # the input mix's w as the branch starts with it (not the identity unless 1.0)
                self.resume_info["recur_mix_init"] = float(self._mix.detach())
        return self.resume_info

    def _check_extends_position(self, step: int, path: Path) -> None:
        """The relation "extends" (stage_data_relation): the source trained on the union's first set alone, and this run's stream is
        that set's epoch-0 stream token for token from position 0 to the set's end (ShardPacker._sets_in_order_layout), then the next
        sets'. So the stream goes on where the source is only while the source's updates 0 .. step - 1 read inside the first set's first
        epoch: the last token they touch (the rows read, times the row length, plus the target of the last row's last position) is at
        most the first set's token count. ValueError by name otherwise (the source's stream wrapped into a second epoch of its set,
        which this run's stream does not)."""
        L, B = self.cfg.data.seq_len, self.cfg.data.batch_size
        rows = self.tst.sched.rows_before(step, B) if self.tst is not None else step * B
        need, first_set = rows * L + 1, int(self.data.shards.parts[0].n_tokens)
        if need > first_set:
            raise ValueError(f"resume_from {path}: the source's {step} updates read {rows} rows of {L} tokens ({need} tokens with the "
                             f"last target), more than the {first_set} tokens of the union's first set: the source's stream wrapped "
                             "into a second epoch of that set, which the union's stream does not repeat")

    def save_checkpoint(self, keep: int | None = None, wait: bool = True) -> Path:
        """The checkpoint of this step: <run_dir>/checkpoints/step_<step>.pt, the `latest` pointer, the rolling ones
        pruned. Under data parallelism (D5) every rank calls it at the same step: each rank's CUDA generator state is
        gathered on the host group, rank 0 writes today's single file (the state is replicated, so the file does not
        depend on the world size) and every rank waits at a barrier until it is written; every rank returns its path.

        With TrainConfig.ckpt_async (module docstring, "The asynchronous save") the state is copied into host memory here
        and written by a background thread, and the file and `latest` are committed at a step boundary (_ckpt_settle).
        wait True (the default; fit's stop and end): returns once this checkpoint is committed (and, under data
        parallelism, after a barrier), as the synchronous save does. wait False (fit's other saves): returns after the
        copy. Without ckpt_async `wait` is not read."""
        if self.cfg.ckpt_async:
            return self._save_async(keep, wait)
        if self.dp is not None:
            ranks = self.dp.gather_host(self._dp_own_rng())
            ranks = None if all(r is None for r in ranks) else ranks
            target = self.ckpt_dir / f"step_{self.step:08d}.pt"
            if self._z1 is not None:
                # the sharded optimizer (N2): every rank writes its own tensors' optimizer state first (not over a kept
                # checkpoint written before: as its step file, below), then rank 0 the step file with the manifest
                if not (self.step in self._kept_steps and target.exists()):
                    self._z1.save_shard(target, self.model, self.step)
                self.dp.barrier()
            path = (self._save_files(keep, lambda: self.state_dict(cuda_rng_ranks=ranks, path=target))
                    if self.dp.lead else None)
            self.dp.barrier()
            return path if path is not None else target
        return self._save_files(keep, self.state_dict)

    def _save_files(self, keep: int | None, state_of) -> Path:
        """save_checkpoint's files (one process, or rank 0): state_of() is the state, read only when the file is
        written."""
        keep = self.cfg.ckpt_keep if keep is None else keep
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.ckpt_dir / f"step_{self.step:08d}.pt"
        # A kept checkpoint (cfg.checkpoint_steps) is written ONCE. A run that passes its step again (it was stopped
        # between this file and the `latest` pointer, and resumed from an earlier rolling checkpoint) leaves the file
        # as it is: a nested budget's branch may already have started from it, and its record names the file's sha256.
        if not (self.step in self._kept_steps and path.exists()):
            tmp = path.with_suffix(".pt.tmp")
            torch.save(state_of(), tmp)
            os.replace(tmp, path)  # atomic: a reader never sees a half-written checkpoint
        self._point_latest(path, keep)
        return path

    def _point_latest(self, path: Path, keep: int) -> None:
        """The `latest` pointer to `path` (written, then renamed), then the rolling checkpoints beyond `keep` pruned: the
        last part of the synchronous save, and of the asynchronous save's commit."""
        (self.ckpt_dir / "latest.tmp").write_text(path.name)
        os.replace(self.ckpt_dir / "latest.tmp", self.ckpt_dir / "latest")
        kept = {f"step_{k:08d}.pt" for k in self._kept_steps}  # cfg.checkpoint_steps: never pruned
        rolling = [p for p in sorted(self.ckpt_dir.glob("step_*.pt")) if p.name not in kept]
        for old in rolling[:-keep]:
            old.unlink()
            shards = zero1_shard_dir(old)  # a checkpoint written by parameter (N2): its optimizer shards go with it
            if shards.is_dir():
                shutil.rmtree(shards)
        # N2: the optimizer files of an earlier step whose step file was never written (the processes stopped between
        # every rank's optimizer file and rank 0's step file: a spot eviction during a save; about 15 GiB a set at 4B).
        # Nothing reads them, and a clock-timed run need not pass that step again, so they go here. A later step's are
        # left (a resume from an earlier checkpoint rewrites them when it gets there). None in a run without the sharded
        # optimizer.
        for stale in self.ckpt_dir.glob(f"step_*{ZERO1_SHARD_SUFFIX}"):
            if stale.name < zero1_shard_dir(path).name and not stale.with_suffix(".pt").exists():
                shutil.rmtree(stale)

    # The asynchronous save (TrainConfig.ckpt_async; module docstring, "The asynchronous save"; ced/train/async_save.py)

    def _save_async(self, keep: int | None, wait: bool) -> Path:
        """save_checkpoint with ckpt_async. The last save is committed first (one write at a time: it waits for that
        write if it still runs); then each rank's CUDA generator state is gathered (D5), and the process that writes
        (one process, or rank 0) copies the state into its host arena and starts the write of the synchronous save's
        temporary file, unless this is a kept checkpoint written before (written once, as in the synchronous save). Every
        rank records the save as pending; with `wait` it is committed before returning.

        With the sharded optimizer (N2; module docstring of ced/train/async_save.py): every rank also copies its own
        optimizer shard into a second arena and its thread writes the shard's temporary file (rank 0's: the shard, then
        the step file); the commit renames the shards before rank 0's step file (_ckpt_commit)."""
        from ced.train import async_save

        t0 = time.perf_counter()
        self._ckpt_settle(block=True)
        waited = time.perf_counter() - t0
        path = self.ckpt_dir / f"step_{self.step:08d}.pt"
        ranks = None
        if self.dp is not None:
            ranks = self.dp.gather_host(self._dp_own_rng())
            ranks = None if all(r is None for r in ranks) else ranks
        pending = {"step": self.step, "keep": self.cfg.ckpt_keep if keep is None else keep, "path": path,
                   "write": False, "wait_s": waited, "snapshot_s": 0.0, "bytes": 0}
        sharded = self._z1 is not None  # N2: every rank writes its optimizer shard (the same on every rank)
        if self.lead or sharded:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            if self._ckpt_writer is None:
                # this process's first write: the temporary files a killed process left (a write in flight when it was
                # killed) go first; nothing else writes in this directory (one writing process a run; under N2 the
                # other ranks write only inside the optimizer directories, which this does not touch)
                if self.lead:
                    for stale in self.ckpt_dir.glob("step_*.pt.tmp"):
                        stale.unlink()
                self._ckpt_writer = async_save.Writer()
                self._ckpt_arena = async_save.Arena(pin=self.device.type == "cuda")
            if not (self.step in self._kept_steps and path.exists()):
                jobs = []
                if sharded:  # this rank's optimizer shard, under the synchronous sharded save's own temporary name
                    shard = self._z1.shard_path(path)
                    shard.parent.mkdir(parents=True, exist_ok=True)
                    if self._ckpt_shard_arena is None:
                        self._ckpt_shard_arena = async_save.Arena(pin=self.device.type == "cuda")
                    jobs.append((async_save.snapshot(self._z1.shard_payload(self.model, self.step), self._ckpt_shard_arena),
                                 shard.with_suffix(".pt.tmp")))
                    pending["shard"] = shard
                if self.lead:
                    state = (self.state_dict(path=path) if ranks is None else
                             self.state_dict(cuda_rng_ranks=ranks, path=path))
                    jobs.append((async_save.snapshot(state, self._ckpt_arena), path.with_suffix(".pt.tmp")))
                self._ckpt_writer.start(*jobs[0], more=jobs[1:])
                pending.update(write=True, snapshot_s=sum(s.seconds for s, _ in jobs), bytes=sum(s.nbytes for s, _ in jobs))
                pending["tmps"] = [tmp for _, tmp in jobs]
        self._ckpt_pending = pending
        if wait:
            self._ckpt_settle(block=True)
            if self.dp is not None:
                self.dp.barrier()
        return path

    def _ckpt_slots(self) -> list[int]:
        """This rank's part of fit's per-step agreement (DataParallel.agree's extra): [its write still runs, its write
        failed]. A rank that writes nothing sends [0, 0]."""
        if self._ckpt_pending is None or not self._ckpt_pending["write"] or self._ckpt_writer is None:
            return [0, 0]
        running, outcome = self._ckpt_writer.poll()
        return [int(running), int(outcome is not None and not outcome["ok"])]

    def _ckpt_settle(self, block: bool, agreed: list[int] | None = None, raise_errors: bool = True) -> str | None:
        """Commit the pending save once its write is done (module docstring, "The asynchronous save"); nothing without one.

        block False (fit, after each step): commit only if the write is done: in one process if the thread has ended;
        under data parallelism if `agreed` (the MAX of every rank's _ckpt_slots, from fit's agree) says no write runs.
        block True (a new save, the stop's and the end's, a halt): wait for this process's write, then under data
        parallelism one MAX on the host group of the ranks' failure flags (every rank reaches it at the same step).
        A failed write: nothing is committed, the pending save is dropped, and CheckpointWriteError is raised (on every
        rank: the agreement carries the flag), naming the failing ranks; with raise_errors False its text is returned
        instead (a halt's record keeps it). The commit: _ckpt_commit."""
        pending = self._ckpt_pending
        if pending is None:
            return None
        writer = self._ckpt_writer if pending["write"] else None
        if self.dp is None:
            if not block and writer is not None and writer.busy():
                return None
            outcome = None if writer is None else writer.wait()
            failed = outcome is not None and not outcome["ok"]
            errors = [outcome["error"]] if failed else []
        else:
            if block:
                outcome = None if writer is None else writer.wait()
                own = int(outcome is not None and not outcome["ok"])
                failed = bool(self.dp.max_host([own])[0])
            else:
                if agreed is None:
                    raise ValueError("_ckpt_settle under data parallelism without blocking needs fit's agreed slots")
                if agreed[0]:
                    return None
                failed = bool(agreed[1])
                outcome = None if writer is None else writer.wait()  # ended: its slot said so
            errors = []
            if failed:
                mine = outcome["error"] if outcome is not None and not outcome["ok"] else None
                errors = [f"rank {r}: {e}" for r, e in enumerate(self.dp.gather_host(mine)) if e is not None]
        self._ckpt_pending = None
        if failed:
            for tmp in pending.get("tmps", ()):  # this rank's written files of the failed save (N2: another rank's failed)
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
            name = self._latest_name()
            text = (f"the background write of checkpoint {pending['path'].name} failed ({'; '.join(errors)}); nothing was "
                    "committed: " + ("there is no `latest` yet" if name is None else f"`latest` still names {name}"))
            if raise_errors:
                from ced.train.async_save import CheckpointWriteError

                raise CheckpointWriteError(text)
            return text
        self._ckpt_commit(pending, outcome)
        return None

    def _ckpt_commit(self, pending: dict, outcome: dict | None, on_exit: bool = False) -> None:
        """The commit of a pending save whose write is done (or that wrote nothing: a kept checkpoint written before). By
        the process that writes the run's files (one process, or rank 0): the temporary file renamed to the step file,
        `latest` and the pruning (_point_latest), and one {"event": "checkpoint"} line in metrics.jsonl ("on_exit": true
        when fit's finally made it, _ckpt_on_exit). On the calling process: the commit recorded in _ckpt_commits.

        With the sharded optimizer (N2), on every rank at the same step (the agreement, or a blocking settle, which every
        rank reaches together; never on the way out, _ckpt_on_exit): each rank renames its own optimizer shard, then a
        barrier, so rank 0 renames its step file only once every rank's shard is in place (N2's order). No second
        barrier: a rank's next decision on that file (a kept step's check in _save_async) follows the save's gather of
        the generator states, which rank 0 reaches only after this commit, and the waiting saves end on a barrier."""
        if self._z1 is not None:
            if pending["write"]:
                os.replace(pending["shard"].with_suffix(".pt.tmp"), pending["shard"])
            self.dp.barrier()
        if self.lead:
            if pending["write"]:
                os.replace(pending["path"].with_suffix(".pt.tmp"), pending["path"])
            self._point_latest(pending["path"], pending["keep"])
            if self.cfg.metrics_jsonl:
                event = {
                    "event": "checkpoint", "step": pending["step"], "committed_after_step": self.step,
                    "file": pending["path"].name, "written": pending["write"], "wait_s": pending["wait_s"],
                    "snapshot_s": pending["snapshot_s"], "write_s": None if outcome is None else outcome["write_s"],
                    "bytes": pending["bytes"], "time": time.time()}
                if on_exit:
                    event["on_exit"] = True
                self._write_jsonl("metrics.jsonl", event)
        self._ckpt_commits.append((pending["step"], self.step))

    def _ckpt_on_exit(self, log=print) -> None:
        """fit's finally with ckpt_async: a save still pending here means fit is leaving on an exception (its stop's and
        end's saves wait for theirs, and the halts commit theirs). This process's write is waited for; the process that
        writes the run's files commits it if it succeeded (_ckpt_commit, its metrics line marked "on_exit"; a crash
        after a finished copy keeps that checkpoint: in today's single-file format rank 0 is the only writer, so no
        agreement is needed). With the sharded optimizer (N2) nothing is committed: every rank writes a shard, and the
        agreement that all are written cannot be had by a process leaving on an exception (it makes no collective: the
        others may never come), so `latest` keeps naming the last whole set and the temporary files are left to the next
        save (async_save's module docstring). Never raises."""
        pending = self._ckpt_pending
        if pending is None:
            return
        self._ckpt_pending = None
        try:
            outcome = self._ckpt_writer.wait() if pending["write"] and self._ckpt_writer is not None else None
            if self._z1 is not None:
                log(f"checkpoint {pending['path'].name}: not committed on the way out: a sharded save (optimizer_sharding "
                    "zero1) is committed only when every rank agrees its write is done; "
                    + ("there is no `latest` yet" if self._latest_name() is None else
                       f"`latest` still names {self._latest_name()}")
                    + ("" if outcome is None or outcome["ok"] else f" (this rank's write failed: {outcome['error']})"))
            elif self.lead and (outcome is None or outcome["ok"]):
                self._ckpt_commit(pending, outcome, on_exit=True)
            elif outcome is not None:
                log(f"checkpoint {pending['path'].name}: the background write failed ({outcome['error']}); not committed")
        except Exception as error:  # noqa: BLE001 - an exception is already leaving fit; this one must not replace it
            log(f"checkpoint {pending['path'].name}: not committed on the way out ({type(error).__name__}: {error})")

    def _latest_name(self) -> str | None:
        latest = self.latest_checkpoint()
        return None if latest is None else latest.name

    def latest_checkpoint(self) -> Path | None:
        pointer = self.ckpt_dir / "latest"
        return self.ckpt_dir / pointer.read_text().strip() if pointer.exists() else None

    @staticmethod
    def saved_config(path: str | Path) -> TrainConfig:
        state = torch.load(path, weights_only=False, map_location="cpu")
        return from_dict(TrainConfig, checkpoint_config(state["config"]))

    @staticmethod
    def saved_config_and_world(path: str | Path) -> tuple[TrainConfig, int]:
        """saved_config, and the world size the checkpoint was written at (D5: 1 without "data_parallel"); one read."""
        state = torch.load(path, weights_only=False, map_location="cpu")
        return (from_dict(TrainConfig, checkpoint_config(state["config"])),
                int((state.get("data_parallel") or {}).get("world_size", 1)))

    @classmethod
    def from_checkpoint(cls, path: str | Path, run_dir: str | Path | None = None, cfg: TrainConfig | None = None,
                        dp=None) -> "Trainer":
        """Rebuild a trainer from a checkpoint. By default the checkpoint's own config is used; pass
        `cfg` to continue under a different (compatible) config. dp: the process's data parallelism (Trainer)."""
        path = Path(path)
        state = torch.load(path, weights_only=False, map_location="cpu")
        trainer = cls(cfg or from_dict(TrainConfig, checkpoint_config(state["config"])), run_dir or path.parent.parent, dp=dp)
        trainer.load_state_dict(state, source=path)
        return trainer

    # ----------------------------------------------------------------------------------------
    # main loop

    def _request_stop(self, signum, frame) -> None:
        self._stop_requested = True

    def _clock(self) -> float:
        """fit's clock for the checkpoint by minutes (ckpt_every_minutes): time.monotonic(). Under data parallelism rank
        0's decides for every rank (D4)."""
        return time.monotonic()

    def _init_wandb(self) -> None:
        w = self.cfg.wandb
        if w.mode == "disabled":
            return
        import wandb

        if self.wandb_run_id is None:
            self.wandb_run_id = wandb.util.generate_id()
        self._wandb = wandb.init(
            project=w.project, entity=w.entity, name=w.run_name, id=self.wandb_run_id,
            resume="allow", mode=w.mode, config=to_dict(self.cfg), dir=str(self.run_dir),
        )

    def _write_jsonl(self, name: str, record: dict) -> None:
        if not self.lead:  # data parallelism (D4): rank 0 writes the run's records
            return
        with open(self.run_dir / name, "a") as f:
            f.write(json.dumps(record) + "\n")

    def run_evals(self, periodic: bool, log=print) -> dict:
        """Evaluate on the configured eval sets: the first periodic_max_rows rows during training,
        all rows at the end. Results go to <run_dir>/eval/step_<step>.json and evals.jsonl."""
        from ced.train.periodic_eval import run_eval_sets

        results = run_eval_sets(self, periodic)
        if results:
            keys = ("dec_bpb", "ctx_bpb", "all_bpb", "aux_bpb", "aux_all_bpb", "loop1_bpb", "loop1_all_bpb")
            summary = {name: ({k: r[k] for k in keys if r.get(k) is not None} if "dec_bpb" in r else r)
                       for name, r in results.items()}
            seen = {} if self.shape is None else {"tokens_seen": self.shape.tokens_seen(self.step)}  # a scheduled run's place in tokens
            self._write_jsonl("evals.jsonl", {"step": self.step, "periodic": periodic, "flops": self.flops_done(), **seen, **summary})
            log(f"eval step {self.step} ({'periodic' if periodic else 'final'}): "
                + "  ".join(f"{n} dec {s['dec_bpb']:.4f} all {s['all_bpb']:.4f}"
                            + "".join(f" {k.split('_')[0]} {s[k]:.4f}" for k in ("aux_bpb", "loop1_bpb") if k in s)
                            for n, s in summary.items() if "dec_bpb" in s))
        return results

    def run_lc_eval(self, log=print, only_if_missing: bool = False, abortable: bool = False) -> dict | None:
        """The in-loop lite long-context eval (EvalConfig.lc_every_steps, 0 = off; ced/train/periodic_eval.py): when
        it is due at this step, score the fixed long-context subset forward only and record it as "lc/..." keys.
        It leaves the training state as it was. only_if_missing: not when this step's record exists; abortable: a
        stop request abandons it."""
        if not self.cfg.eval.lc_every_steps:
            return None
        from ced.train.periodic_eval import run_lc_eval

        return run_lc_eval(self, log=log, only_if_missing=only_if_missing, abortable=abortable)

    def readout_start(self) -> dict | None:
        """With the readout inside training on (eval.readout_every; ced/train/readout.py): read its sample and build its
        scoring twin (the first call; a wrong sample raises), and return what the start record says of it. None when
        the readout is off."""
        if not self.cfg.eval.readout_every:
            return None
        from ced.train.readout import describe

        return describe(self)

    def run_readout(self, log=print, abortable: bool = False) -> dict | None:
        """The readout inside training (eval.readout_every, 0 = off; ced/train/readout.py): when it is due at this step
        and the step has no record yet, exact match at T = 1 and T = 2 on the fixed dev sample, recorded as
        {"event": "readout"} in metrics.jsonl. It leaves the training state as it was. abortable: a stop request
        abandons it."""
        if not self.cfg.eval.readout_every:
            return None
        from ced.train.readout import run_readout

        return run_readout(self, log=log, abortable=abortable)

    def bpb_readout_start(self) -> dict | None:
        """With the bpb readout on (eval.bpb_readout_every; ced/train/bpb_readout.py): read its sets and choose its rows
        (the first call; a wrong set raises), and return what the start record says of it. None when it is off."""
        if not self.cfg.eval.bpb_readout_every:
            return None
        from ced.train.bpb_readout import describe

        return describe(self)

    def run_bpb_readout(self, log=print, abortable: bool = False) -> dict | None:
        """The bpb readout (eval.bpb_readout_every, 0 = off; ced/train/bpb_readout.py): when it is due at this step and
        the step has no record yet, bpb at T = 1 (and T = 2 for a looped model) on the fixed rows, recorded as
        {"event": "bpb_readout"} in metrics.jsonl. It leaves the training state as it was. abortable: a stop request
        abandons it."""
        if not self.cfg.eval.bpb_readout_every:
            return None
        from ced.train.bpb_readout import run

        return run(self, log=log, abortable=abortable)

    def _optimizer_failed(self, error: Exception, total_loss, grad_norm, lr: float) -> dict | None:
        """optimizer.step() raised (module docstring, "The input guards"). A non-finite gradient can make the optimizer
        itself fail before fit() reads the step's numbers: Shampoo's inverse roots and KL-SOAP's eigenbases come from
        torch.linalg.eigh (ced/train/matrix_optim.py eigh_descending), which refuses a NaN matrix in float32 and again in
        float64. The run would then crash with no record. Here the host reads the step's loss and gradient norm (a device
        read on this failure path only; the normal step is unchanged). When either is not finite the update counts as
        made, as for any non-finite step (its state is never saved), and the step's numbers go back to fit(), whose
        finite check ends the run with the record, which also names the optimizer's error, and exit status 86. None
        when both are finite: the error is the optimizer's own, and the caller raises it as it was."""
        loss, norm = float(total_loss), float(grad_norm)
        # the failure path's own read of the MXFP8 flag (under data parallelism the flag OR-ed over the ranks, _dp_sync)
        flag = int(self._mx_flag if self.dp is None else self._dp_mx_flag) if self._mx_flag is not None else 0
        if math.isfinite(loss) and math.isfinite(norm) and not flag:
            return None
        self.step += 1
        out = {"loss": loss, "grad_norm": norm, "lr": lr, "optimizer_error": f"{type(error).__name__}: {error}"[:2000]}
        if self._mx_flag is not None:
            out[MX_FLAG_KEY] = flag
        return out

    @torch.no_grad()
    def _rotation_read(self, before: list) -> dict:
        """The rotation log (OptimConfig.ours.rotation_log_every; mechanism test 3's "rotation logs (the angle per step per
        matrix group)"): for every Muon-group matrix, the angle between W before and after this optimizer step,
        acos(<W0, W1> / (|W0| |W1|)) in float64, and the norm ratio |W1| / |W0|; per role of ced/model/invariance.py
        (attention.o_proj, mlp.w_up, gdn.out_proj, ...): [mean angle, min, max, mean norm ratio, mean RMS of W1, count].
        Read only: the weights, the optimizer and the RNG are untouched; one host read."""
        rows = []
        for (_, p), w0 in zip(self._rot_params, before):
            a, b = w0.double(), p.detach().double()
            n0, n1 = a.norm(), b.norm()
            cos = ((a * b).sum() / (n0 * n1).clamp_min(1e-300)).clamp(-1.0, 1.0)
            rows.append(torch.stack([torch.arccos(cos), n1 / n0.clamp_min(1e-300), n1 / math.sqrt(b.numel())]))
        host = torch.stack(rows).cpu().tolist()
        out: dict = {}
        for (role, _), (ang, ratio, rms) in zip(self._rot_params, host):
            out.setdefault(role, []).append((ang, ratio, rms))
        return {role: [sum(v[0] for v in vals) / len(vals), min(v[0] for v in vals), max(v[0] for v in vals),
                       sum(v[1] for v in vals) / len(vals), sum(v[2] for v in vals) / len(vals), len(vals)]
                for role, vals in out.items()}

    def _check_finite(self, metrics: dict, log=print) -> None:
        """The stop on a non-finite loss (module docstring). With the step's loss and gradient norm finite: nothing,
        apart from keeping the step in the tail of the last finite steps. Otherwise the run ends here: the record
        {"event": "diverged"} in metrics.jsonl, the marker file <run_dir>/diverged.json, no checkpoint, and Diverged.
        Reads only the numbers train_step has put on the host. With model.fp8.nonfinite_flag, a set bit of the MXFP8
        non-finite flag (metrics[MX_FLAG_KEY]) ends the run the same way, named in the record as MX_FLAG_NAME with the
        kinds of operand that held the non-finite block, whatever the loss and the gradient norm (module docstring)."""
        bad = nonfinite_metrics(metrics)
        if metrics.get(MX_FLAG_KEY):
            bad[MX_FLAG_NAME] = mx_flag_text(metrics[MX_FLAG_KEY])
        if not bad:
            self._recent.append({"step": self.step, **{k: metrics[k] for k in RECENT_KEYS if k in metrics}})
            return
        k = self.step  # the update just made: the one whose metrics record would have carried "step": k
        # N8: a background write in flight is finished and committed first (every rank reaches this at the same update),
        # so the record names the latest whole checkpoint; a failed write is named in the record instead of raised
        write_error = self._ckpt_settle(block=True, raise_errors=False) if self.cfg.ckpt_async else None
        record = diverged_record(k, bad, metrics, self._recent, self.latest_checkpoint())
        if write_error is not None:
            record["checkpoint_write_error"] = write_error
        last = record["last_finite"]
        if "optimizer_error" in metrics:  # optimizer.step() failed on the non-finite step (_optimizer_failed)
            record["optimizer_error"] = metrics["optimizer_error"]
        batch = self._step_batch  # the batch this update trained on (train_step); its rows, for whoever reads the record
        if batch is not None:
            record["batch_hash"] = batch_hash(batch)
            record["rows"] = self.row_record(k - 1, batch)
        if self.dp is not None:
            # data parallelism (D4): every rank reaches this at the same update (its loss, gradient norm and MXFP8 flag are
            # the global step's, D3), so every rank's rows are gathered (a host collective, on this path only); the record
            # names the step's global batch ("batch_hash": the hash of the ranks' hashes) and rank 0 writes it
            from ced.train.data_parallel import batch_digest, combined_hash

            rows = self.dp.gather_host(None if batch is None else {"batch_hash": batch_hash(batch),
                                                                   "rows": self.row_record(k - 1, batch)})
            record.update(world_size=self.dp.world, rows_by_rank=rows)
            if all(r is not None for r in rows):
                record["batch_hash"] = combined_hash([batch_digest(r["batch_hash"]) for r in rows])
        if self._mix is not None:  # the input mix's w: after the last finite update, and its path over the tail
            record["w"] = {"last_finite": last[-1]["w"] if last else None, "path": [[row["step"], row["w"]] for row in last],
                           "after_the_diverged_update": _plain(metrics.get("w"))}
        self._write_halt(record)
        log(halt_message("DIVERGED", record, self.run_dir))
        raise Diverged(record)

    def _write_halt(self, record: dict) -> None:
        """The record of a run that ends on its own check (a non-finite loss or gradient norm, a bad batch): one line in
        metrics.jsonl (when the run writes it) and the marker file <run_dir>/diverged.json, written atomically. Strict JSON
        (allow_nan False; write_halt). Under data parallelism rank 0 writes them (D4); every rank halts, and every rank
        leaves only after rank 0's write (one barrier on the host group:, the review the). Every rank reaches this at the
        same update (a bad batch on any rank and a non-finite step are the global step's decisions, D2 and D3), so the
        barrier cannot wait for a rank that is not coming. Without it the other ranks exit 86 at once, and a launcher that
        then stops the group (torchrun: SIGTERM to every worker still running, SIGKILL after its shutdown timeout; another
        launcher may kill at once, or a node of a multi-node run) can end rank 0 before the marker is written: the next
        start would train again instead of exiting 86."""
        if self.lead:
            write_halt(self.run_dir, record, self.cfg.metrics_jsonl)
        if self.dp is not None:
            self.dp.barrier()

    def _halt_bad_batch(self, error: BatchError, log=print) -> None:
        """A batch failed the loader's validation (module docstring, "The input guards"): the run ends here through the
        stop's path. The record {"event": "bad_batch", "step": k, ...} goes to metrics.jsonl and the marker; no
        checkpoint is written; BadBatch is raised (scripts/train.py: exit status EXIT_DIVERGED). k is the update the batch
        was for (as metrics.jsonl counts it); the check runs before the update is made (batch_at), or, for a boundary
        array of the fused GDN kernels, between two micro-batches of it: either way no update was made with the batch."""
        # N8: as _check_finite, a background write in flight is committed first (every rank halts at this update)
        write_error = self._ckpt_settle(block=True, raise_errors=False) if self.cfg.ckpt_async else None
        record = bad_batch_record(error, self.step + 1, self.step, self.latest_checkpoint())
        if write_error is not None:
            record["checkpoint_write_error"] = write_error
        self._write_halt(record)
        log(halt_message("BAD BATCH", record, self.run_dir))
        raise BadBatch(record)

    def step_flops(self) -> float:
        return self.flops_per_token() * self.cfg.data.batch_size * self.cfg.data.seq_len

    def _phase_flops_per_token(self, seq_len: int) -> float:
        """Analytic training FLOPs per token at rows of `seq_len` tokens (a shape schedule's phases; the earlier updates of a
        stage at another row shape): flops_per_token of the config's own length, else the same count at that length, at
        the loop count flops_per_token takes (T = 1 without a loop: every shape schedule, which refuses loops; a looped
        model's largest T, a looped middle's mean pass count; T = 1 alone undercounted a
        looped model's earlier updates)."""
        if seq_len == self.cfg.data.seq_len:
            return self.flops_per_token()
        cache = self.__dict__.setdefault("_phase_flops", {})
        if seq_len not in cache:
            from ced.utils.params import flops_per_token

            c, pos = self.cfg, self._positions_for(seq_len)
            cap = decoder_capacity(seq_len, c.model.decoder.capacity_fraction, c.model.decoder.capacity_multiple)
            aux = pos.aux_count / seq_len if pos.aux_count is not None else None
            at = lambda T: flops_per_token(c.model, seq_len, cap / seq_len, T, aux_fraction=aux)["training"]  # noqa: E731
            if self.recur:  # as flops_per_token: linear in T, at the mean pass count
                one, two = at(1), at(2)
                cache[seq_len] = one + (self.mean_passes() - 1.0) * (two - one)
            else:
                cache[seq_len] = at(max(c.model.loop.t_choices) if self.is_looped else 1)
        return cache[seq_len]

    def eval_due(self) -> bool:
        """Whether a periodic eval belongs after the update just made: every eval.every_steps steps; with a shape
        schedule every eval.every_steps REFERENCE steps of tokens read (the unscheduled run's eval points in tokens)."""
        every = self.cfg.eval.every_steps
        if not every:
            return False
        if self.shape is None:
            return self.step % every == 0
        ref = self.shape.ref_done(self.step)
        return ref is not None and ref % every == 0

    def _stage_shape_segments(self) -> list | None:
        """The updates of the step counter before this run's own first update, as [(first, end, row length)], back along
        the chain of sources (resume_info, then each source's own record "source_resumed_from"), so that each is counted
        at the row length it was trained at: a stage's source at its recorded data identity's ("source_data_state"), any
        other source (a resume_from branch, whose config is the run's) at the row length of the run it was continued by.
        None when every one of them was at this run's row length: every run without a stage at another row shape (stage_shape
        "tokens") anywhere in its chain, whose FLOPs are step_flops x step, as always. The chain is walked through every
        record, not only the reshaped stages' (a stage at its source's shape after a reshaped
        one counted the reshaped stage's source at the later length)."""
        out, length, rec = [], self.cfg.data.seq_len, self.resume_info
        while isinstance(rec, dict) and rec.get("step") is not None:
            stage = rec.get("stage") or {}
            length = int((stage.get("source_data_state") or {}).get("seq_len", length))
            src = rec.get("source_resumed_from")
            lo = int(src["step"]) if isinstance(src, dict) and src.get("step") is not None else 0
            out.append((lo, int(rec["step"]), length))
            rec = src
        return None if all(L == self.cfg.data.seq_len for _, _, L in out) else out

    def flops_done(self) -> float:
        """Analytic training FLOPs of the updates made so far: step_flops x step, with the single-pass steps before a
        loop switch step (TrainConfig.loop_switch_step) counted at T = 1, and in a stage at another row shape
        (stage_shape "tokens") the source's updates at the row length they were trained at (_stage_shape_segments)."""
        S = self.cfg.loop_switch_step
        if self.shape is not None:
            # per phase: the tokens read in it so far x the analytic FLOPs per token at its row length
            total, last = 0.0, len(self.shape.phases) - 1
            for ph in self.shape.phases:
                done = max(self.step - ph.first_step, 0)
                if ph.index < last:
                    done = min(done, ph.steps)
                total += done * ph.rows * ph.seq_len * self._phase_flops_per_token(ph.seq_len)
            return total
        if not S:
            before = self._stage_shape_segments()
            if before is None:  # no other row length in the chain: as always
                return self.step_flops() * self.step
            # a stage at another row shape in the chain: the sources' updates (before this run's first, the source's step
            # B) at the row length they were trained at, this run's own at its own (the tokens a step are the same
            # throughout: a reshaped stage keeps them, any other continuation keeps the shape)
            T, B = self.cfg.data.batch_size * self.cfg.data.seq_len, before[0][1]
            done = sum(max(0, min(self.step, hi) - lo) * T * self._phase_flops_per_token(L) for lo, hi, L in before)
            return done + self.step_flops() * max(0, self.step - B)
        one = self.flops_per_token(T=1) * self.cfg.data.batch_size * self.cfg.data.seq_len
        return one * min(self.step, S) + self.step_flops() * max(0, self.step - S)

    def fit(self, until_step: int | None = None, log=print) -> None:
        """Train until `until_step` (default: the run's optimizer steps, cfg.steps without a shape schedule),
        checkpointing as configured.

        Under data parallelism (D4; module docstring, "Data parallelism") every rank runs this loop and acts at the same
        step: a bad batch on any rank and a non-finite step halt every rank (their decisions are the global step's, D2
        and D3); a SIGTERM on any rank stops every rank after a checkpoint (the stop flag, a MAX over the ranks every
        step); a checkpoint is due by rank 0's clock (in the same MAX). Rank 0 alone writes metrics.jsonl, the halt
        records and the checkpoints, logs, talks to W&B, and runs the evals and readouts while the others wait at a
        barrier on the host group."""
        dp = self.dp
        if not self.lead:  # data parallelism: rank 0 logs
            log = _quiet
        until_step = self.total_steps if until_step is None else until_step
        marker = diverged_marker(self.run_dir)
        if marker is not None:  # a run that stopped on a non-finite loss is never trained again (module docstring)
            raise Diverged(marker)
        if self.lead:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        # The readout inside training (eval.readout_every): its sample is read and its scoring twin built here, so a
        # wrong sample stops the run before it writes or trains anything. None with the readout off (and on every rank
        # but rank 0, which runs it).
        readout = self.readout_start() if self.lead else None
        # The bpb readout (eval.bpb_readout_every): its sets are read and its rows chosen here, likewise.
        bpb_readout = self.bpb_readout_start() if self.lead else None
        if self.lead:
            self._init_wandb()
        nvfp4 = getattr(self.model, "nvfp4_summary", None)  # set by apply_nvfp4; None with the switch off
        fp8 = getattr(self.model, "fp8_summary", None)  # set by apply_fp8 (ced/model/fp8.py); None with the switch off
        if self.cfg.metrics_jsonl:
            start = {"event": "start", "step": self.step, "time": time.time(), "flops_per_step": self.step_flops()}
            if readout is not None:  # the sample and the cadence of the readout; absent with it off
                start["readout"] = readout
            if bpb_readout is not None:  # the rows and the cadence of the bpb readout; absent with it off
                start["bpb_readout"] = bpb_readout
            if self.init_info is not None:  # the run's initial weights (init_from), on every start, resumes included
                start["init_from"] = self.init_info
            if self.resume_info is not None:  # the run this one continues (resume_from), likewise
                start["resumed_from"] = self.resume_info
            if self.cfg.data.kind == "shards" and self.cfg.data.extra_shard_dirs:  # the union's sets; absent with one set
                start["shard_sets"] = self.data.shards.describe()
            if nvfp4:  # which linears are fake-quantised (ced/model/nvfp4.py); absent with the switch off
                start["nvfp4"] = {k: nvfp4[k] for k in ("mode", "kept_layers", "high_precision", "quantised_param_share")}
                # the linears kept by name and the recipe (NVFP4Config.keep_names, .recipe): absent at their defaults
                start["nvfp4"].update({k: nvfp4[k] for k in ("keep_names", "kept_by_name", "recipe") if k in nvfp4})
                # the linears that run FP8 by name (NVFP4Config.fp8_names) and their share: absent at the default
                start["nvfp4"].update({k: nvfp4[k] for k in ("fp8_names", "fp8", "fp8_param_share") if k in nvfp4})
                # the update at which the quantised path ends (NVFP4Config.until_step): absent at the default
                start["nvfp4"].update({k: nvfp4[k] for k in ("until_step",) if k in nvfp4})
            if fp8:  # which linears run FP8 GEMMs in training forwards (ced/model/fp8.py); absent with the switch off
                start["fp8"] = {k: fp8[k] for k in ("mode", "scaling", "high_precision", "quantised_param_share")}
                if fp8.get("backend", "emulated") != "emulated":  # Transformer Engine; absent at the default
                    start["fp8"]["backend"] = fp8["backend"]
                if fp8.get("te_nonfinite_check"):  # TE's operands checked for infinities (B-fixes); absent at the default
                    start["fp8"]["te_nonfinite_check"] = True
                # the tuning pass's, the chains' wired sites and the G0 folds'; absent at their defaults
                for key in ("quantiser", "te_operands", "nonfinite_flag", "nonfinite_flag_form", "chains", "folds"):
                    if key in fp8:
                        start["fp8"][key] = fp8[key]
            if self._trunk is not None:  # the post-hoc stage: what is frozen and what trains, the trunk's start checksum; absent without it
                start["freeze_trunk"] = self._trunk.describe()
            if self.tst is not None:  # Token Superposition Training's phases and options; absent without it
                start["tst"] = self.tst.describe()
            if self._zw_desc is not None:  # the zero-wgrad counter's cadence and module list; absent with it off
                start["zero_wgrad"] = self._zw_desc
            if self._wgrad is not None:  # the fused linears and its GEMM route; absent with wgrad_accum "autograd"
                start["wgrad_accum"] = self._wgrad.describe()
            if self.shape is not None:  # the phases and the run's optimizer steps; absent without a schedule
                start.update(shape_schedule=self.shape.describe(), total_steps=self.total_steps,
                             tokens_seen=self.shape.tokens_seen(self.step))
                if self.cfg.shape_lr != "clock":  # the LR scaled with the rows; absent under the default
                    start["shape_lr"] = self.cfg.shape_lr
            if self.dp_info is not None:  # data parallelism (D5): the world sizes, the ranks' CUDA streams; absent otherwise
                rec = self._dp_record()
                start["data_parallel"] = {"world_size": rec["world_size"], "history": rec["history"],
                                          **({} if dp is None else {"shard_evals": dp.shard_evals})}  # E1's option
                if dp is not None:
                    start["data_parallel"].update(backend=dp.backend, check_every=dp.check_every)
                if self._z1 is not None:  # the sharded optimizer (N2): its layout; absent with "replicated"
                    start["data_parallel"]["zero1"] = self._z1.layout.describe()
            self._write_jsonl("metrics.jsonl", start)
        if nvfp4:
            log(f"nvfp4 {nvfp4['mode']}: {len(nvfp4['quantised'])} linears fake-quantised in training forwards "
                f"({100 * nvfp4['quantised_param_share']:.1f}% of the layers' linear weights), kept layers "
                f"{nvfp4['kept_layers']}"
                + (f", {len(nvfp4['kept_by_name'])} kept by name {nvfp4['keep_names']}" if "keep_names" in nvfp4 else "")
                + (f", recipe {nvfp4['recipe']}" if "recipe" in nvfp4 else "")
                + (f", {len(nvfp4['fp8'])} in FP8 (rowwise) by name {nvfp4['fp8_names']} "
                   f"({100 * nvfp4['fp8_param_share']:.1f}%)" if "fp8_names" in nvfp4 else "")
                + (f", NVFP4 for updates 0..{nvfp4['until_step'] - 1} only" if "until_step" in nvfp4 else "") + "; eval forwards are BF16")
        if fp8:
            log(f"fp8 {fp8['mode']} ({fp8['scaling']}): {len(fp8['quantised'])} linears fake-quantised in training "
                f"forwards ({100 * fp8['quantised_param_share']:.1f}% of the layers' linear weights), "
                f"{len(fp8['high_precision'])} kept in high precision; eval forwards are BF16")
        if dp is not None:  # rank 0's start (the readouts' samples, W&B, the start record) before any rank trains
            dp.barrier()
        previous_handler = signal.signal(signal.SIGTERM, self._request_stop)
        last_save = self._clock()
        ev = self.cfg.eval
        # Data parallelism (D4): the evals and readouts between steps are rank 0's duty, the other ranks at one barrier
        # (_lead_duty), and only when one of them is on (the config's: the same on every rank). stopped: the stop the
        # ranks agreed on (one process reads self._stop_requested where it always did).
        duties = dp is None or bool(ev.lc_every_steps or ev.readout_every or ev.bpb_readout_every)
        stopped = False

        def opening():
            # The lite long-context eval: step 0 of a new run; on a resume, only a due step that has no record yet.
            self.run_lc_eval(log, only_if_missing=True)
            # The readout: a due step that has no record yet (a new branch at its switch step; a resumed run whose
            # readout a stop request abandoned).
            self.run_readout(log)
            # The bpb readout likewise (a stage C or a loop branch at its first step, before any update of its own).
            self.run_bpb_readout(log)

        def between_steps():
            if self.step < self.total_steps:  # in-loop (the end of the run has its own, after the final checkpoint)
                self.run_lc_eval(log, abortable=True)
            self.run_readout(log, abortable=True)  # between two steps; a stop request abandons it and the save below follows
            self.run_bpb_readout(log, abortable=True)  # the same

        try:
            if duties:
                self._lead_duty(opening)
            while self.step < until_step:
                try:
                    metrics = self.train_step()
                except BatchError as error:  # a batch that fails the loader's validation ends the run here (BadBatch)
                    self._halt_bad_batch(error, log)
                self._check_finite(metrics, log)  # a non-finite loss or gradient norm ends the run here (Diverged)
                if "tst_switch" in metrics:  # Token Superposition Training: the switch's record, before the step's own
                    ev_ = metrics["tst_switch"]
                    if ev_.get("keep_optimizer_state"):  # tst.keep_optimizer_state: the switch kept every optimizer state
                        kept_ = f"optimizer state KEPT ({ev_['optimizer_states_kept']} tensors' states)"
                    else:
                        kept_ = f"optimizer state reset ({ev_['optimizer_states_reset']} tensors' states)"
                    log(f"step {ev_['step']:6d}  TST switch: s {ev_['from_s']} -> {ev_['to_s']}; {kept_}"
                        + (f"; GDN A_log shifted by {ev_['gdn_decay_rescale']['A_log_shift']:+.6f}" if "gdn_decay_rescale" in ev_ else ""))
                    if self.cfg.metrics_jsonl:
                        self._write_jsonl("metrics.jsonl", {**ev_, "time": time.time()})
                if self.cfg.metrics_jsonl:
                    self._write_jsonl("metrics.jsonl", {"step": self.step, "flops": self.flops_done(), **metrics})
                if duties:
                    self._lead_duty(between_steps)
                if self.step % self.cfg.log_every == 0 or self.step == until_step:
                    aux = "ce_aux" if "ce_aux" in metrics else "ce_ctx"
                    extra = "".join(f"  {k} {metrics[k]:.4g}" for k in ("alpha", "w", "loop_usage", "max_act_pre_global", "tau_min", "tau_max",
                                                                         "entmax_support", "entmax_collapsed", "entmax_dead",
                                                                         "entmax_dead_kv")
                                    if metrics.get(k) is not None)
                    log(f"step {self.step:6d}  loss {metrics['loss']:.4f}  ce_dec {metrics['ce_dec']:.4f}  "
                        f"{aux} {metrics[aux]:.4f}  T {metrics['T']}  lr {metrics['lr']:.2e}  "
                        f"dropped {metrics['n_dropped']}{extra}  {metrics['tokens_per_s']:.0f} tok/s")
                short = {k: metrics[k] for k in ("dec_positions_short", "aux_positions_short") if metrics.get(k)}
                if short:  # rows with fewer valid targets than the position counts: all of them were taken
                    log(f"step {self.step:6d}  rows short of valid targets: " + ", ".join(f"{k} {v}" for k, v in short.items()))
                clock_due = None
                if dp is not None:  # D4: a stop on any rank, and rank 0's clock, agreed in one MAX on the host group
                    if self.cfg.ckpt_async:  # N8: and the ranks' background writes, in the same MAX
                        stopped, clock_due, agreed = dp.agree(
                            self._stop_requested, self._clock() - last_save > 60 * self.cfg.ckpt_every_minutes,
                            extra=self._ckpt_slots())
                    else:
                        stopped, clock_due = dp.agree(self._stop_requested,
                                                      self._clock() - last_save > 60 * self.cfg.ckpt_every_minutes)
                if self.cfg.ckpt_async:  # N8: a finished background write is committed here (every rank at this step)
                    self._ckpt_settle(block=False, agreed=None if dp is None else agreed)
                due = (
                    self.step % self.cfg.ckpt_every_steps == 0
                    or (self._clock() - last_save > 60 * self.cfg.ckpt_every_minutes if dp is None else clock_due)
                    or (self._stop_requested if dp is None else stopped)
                    or self.step in self._kept_steps  # cfg.checkpoint_steps: saved exactly here, and kept
                )
                if due:
                    # N8: the stop's save waits for its write (and any write in flight before it); the others do not
                    self.save_checkpoint(wait=bool(self._stop_requested if dp is None else stopped))
                    last_save = self._clock()
                if self._stop_requested if dp is None else stopped:
                    log(f"SIGTERM received: checkpoint saved at step {self.step}, exiting")
                    break
                if ev.sets and self.eval_due() and self.step < self.total_steps:
                    self._eval_duty(lambda: self.run_evals(periodic=True, log=log))
                # Logging comes after the save and can never stop training: W&B may already be gone
                # when the eviction notice arrives.
                self._wandb_call("log", metrics, step=self.step, log=log)
            if not (self._stop_requested if dp is None else stopped):
                self.save_checkpoint()
                if ev.sets and self.step >= self.total_steps:
                    self._eval_duty(lambda: self.run_evals(periodic=False, log=log))
                # the end of the run (due when step = cfg.steps)
                self._lead_duty(lambda: self.run_lc_eval(log, only_if_missing=True))
        finally:
            self._prefetch_discard()  # the data prefetch's thread finished and its result dropped (no-op with the option off)
            if self.cfg.ckpt_async:  # N8: a write still pending (fit is leaving on an exception): finished and committed,
                self._ckpt_on_exit(log)  # under fit's handler (a SIGTERM here only sets the flag)
            signal.signal(signal.SIGTERM, previous_handler)
            self._wandb_call("finish", log=log)
            self._wandb = None

    def _wandb_call(self, method: str, *args, log=print, **kwargs) -> None:
        """Best-effort W&B call: on any error, warn once and continue without W&B."""
        if self._wandb is None:
            return
        try:
            getattr(self._wandb, method)(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - logging must never break training or saving
            log(f"W&B {method} failed ({type(error).__name__}: {error}); continuing without W&B")
            self._wandb = None
