"""CUDA graphs for the training step (TrainConfig.cuda_graphs).

What is captured. "microbatch": one micro-batch's forward, loss and backward, with the step's metric sums, as ONE CUDA
graph over static input buffers; every micro-batch of every later step is a replay of it (one launch on the host instead
of the thousands of kernel launches, Python and dispatch of an eager micro-batch). "step" adds a second graph: the
gradient clip (torch.nn.utils.clip_grad_norm_, as the eager step calls it) and Muon's momentum update and Polar Express
(MuonWithAdamW.muon_orthogonalise). Muon's LR step and AdamW (muon_apply, adamw_step) read the step's LR and AdamW's step
count on the host and run eagerly after it (their device-side form would round differently: section "Not captured").

The rules that make a replay the eager micro-batch, bit for bit:
- Fixed shapes. A micro-batch is micro_batch_size rows of data.seq_len tokens (a run's shape is fixed). The head runs at N positions,
  head_positions' padded size for a micro-batch whose every position counts (rows x seq_len rounded up to a multiple of HEAD_CHUNK = 1,024: 8,192 for
  one row of 8,192), the valid targets first, as head_positions orders them: that is the head's own padded size whenever the micro-batch has fewer
  than HEAD_CHUNK positions without a valid target (a real row has one per document end), so the head's GEMMs and sums are today's. The GDN kernels'
  document boundaries are carried at a fixed capacity (ced/train/varlen.py fixed_varlen_host; ced/kernels/gdn.py fixed_chunk_index): cu_seqlens of S +
  2 entries (S = rows x cuda_graphs_docs_per_row segments, padded with empty ones and a sentinel), the chunk index at nt_max = rows x seq_len / 64 + S
  chunks; every real program of every GDN kernel computes what it does over the exact boundaries and the padding programs touch nothing (masked), so
  the GDN outputs and gradients are bitwise the exact call's. Attention: flex builds its BlockMasks on the device from the document ids, the FA3
  metadata (which backend "fa4" reads too) is fixed-capacity already (ced/kernels/flash_capped.py). A micro-batch that does not fit (more segments
  than S, or a head size other than N) runs eagerly with its exact shapes, the same function: counted as a miss in the step's "cuda_graphs" metrics.
- No host values inside the graph. The loss's normalisers (1 / n_valid and z / n_valid, Python floats today) are put in
  a device buffer once a step, computed on the host in double exactly as today and rounded once to float32 (what the
  kernels do with the Python float), so every product is today's. Host-to-device copies of the inputs happen before each
  replay, from pinned memory, on the replay's stream (so a copy never overtakes the replay that still reads the buffer).
- Persistent gradients. The gradients made by the process's first step are kept (never set to None again), zeroed at
  the start of each step and accumulated in place by every micro-batch (eager or replayed: the autograd engine adds in
  place into an existing gradient). Today the first micro-batch's gradient is assigned, not added to zero: the two differ
  only where that gradient is exactly -0.0 (0.0 + -0.0 = +0.0). Nothing the run computes from the gradients keeps that
  sign: the clip squares them, Muon adds them to its momentum buffer (which starts at +0.0 and can never be -0.0, since
  x + y is -0.0 only when both are), and AdamW lerps them into exp_avg (likewise) and squares them into exp_avg_sq. So
  the losses, gradient norms, parameters and optimizer state are bit for bit those of cuda_graphs "none"; p.grad itself
  is equal in value (tests/test_n1_cuda_graphs.py compares the parameters and the optimizer state bitwise and the
  gradients by value, against main's record).
- Warm-up. A process's first step is today's step (cuda_graphs "none"'s code; it allocates the gradients, the optimizer
  state, compiles the layers), run on the capture stream (first_step; section "Streams"). Its second step runs the graph
  path eagerly on the capture stream (every kernel the graph will hold is compiled and loaded, cuBLAS has its workspace
  on that stream). The third step's first fitting micro-batch is captured, then replayed; so is the optimizer graph. A
  capture runs no kernel: the micro-batch is then replayed like every other. A resume is a new process: the same two
  eager steps first.
- Streams (a first GPU run failed every 4B-width and floor leg at the capture: "autograd node
  'torch::autograd::AccumulateGrad' has a stale reference to the default stream (stream 0) from warmup"). A parameter's
  AccumulateGrad node records the current stream when it is made (the first forward that uses the parameter while no
  node of it is alive) and keeps that stream while it lives; a capture refuses a backward that adds into a node made on
  the default stream (torch/autograd/graph.py set_override_stale_capture_stream; PyTorch's own recipe warms up on the
  capture stream: torch/cuda/graphs.py make_graphed_callables, "Need to use the same stream for warmup and capture").
  The node of step 1's forward stays alive while anything holds that step's autograd graph, and every later forward
  reuses it. torch.compile holds it: the compile of each layer kind's first call (and its recompile for the fixed
  metadata in the warm-up) leaves Dynamo's tracer objects in reference cycles that keep the real tensors at the layer's
  graph break (the GDN kernel call's inputs) and so their autograd history (the embedding, the norms, the first GDN
  layer's projections) until Python's cyclic collector runs, which a short step need not trigger in time. Measured on
  the CPU with the layers compiled (aot_eager) and the GDN kernels routed to the emulations: after the compiling step
  4,700 objects in cycles holding 3 tensors with autograd history, after the recompiling warm-up 4,170 (3 tensors),
  after the third step none; with the collector off, the nodes of 9 parameters made by step 1 still alive at step 4,
  with it on none (tests/test_n1_cuda_graphs.py checks the case). The fix: every forward of a graph-path process
  runs on the capture stream (first_step for the first step, _on_stream for the warm-up and every eager micro-batch,
  misses included), so every node a capture can meet was made on its stream, whatever holds it; and _check_accumulators,
  at the warm-up step's start and at the capture step's start, refuses by name any live node made on another stream
  (the stream each node was made on is recorded by a forward hook during the first two steps; on the CPU the "stream" is
  the runner's own scope, so the CPU tests check the same rule).
- Monitoring. The first micro-batch of a log_every-th step records the monitoring statistics (collect_stats: the maxima
  of the detached stream before each global layer, read only). With graphs every micro-batch records them, in the graph
  (a few reductions a micro-batch), and the step copies out its first micro-batch's: the same values, and no eager
  micro-batch beside the graph's memory pool (an eager micro-batch needs its own activation memory: at 4B about one
  micro-batch's activations more).
- A miss runs eagerly beside the graph's pool and needs its own activation memory: expected never with
  cuda_graphs_docs_per_row 64 on real rows (the run counts them); on a slice at 3 rows of the 60M a miss would not fit.
- Compiled layers (2026-10-17). A capture on CUDA needs compile "all" (check_capture): uncompiled layers run the
  attention path's data-dependent checks, a device read each forward. The optimizer graph refuses a replay if the
  clip's max norm, Muon's momentum / nesterov / ns_steps or a momentum buffer's address differs from its capture
  (GraphRunner._opt_key): they are baked into the graph, and none changes in a run today.

MXFP8. model.fp8 was refused here as a whole since N1, with no cause named (N1 was built and checked for the BF16 control).
It is built now for the two MXFP8 paths the 4B plan uses, and fp8_refusals refuses every other: backend "emulated" with
quantiser "triton" (our Triton quantiser, ced/kernels/mx_ops.py; the non-finite flag off, or on in its default form "pure")
and backend "te" with te_operands "ours" (Transformer Engine's GEMMs on our quantiser's operands, ced/model/fp8_te.py
OursTELinear: the B200 harness's MXFP8 legs). What a capture needs, item by item:
- shapes: every quantiser output, block scale, padded TE scale and Triton grid follows from its operand's shape, fixed at a
  call site; the launch config is the committed table's pick for (device, dtype, features), read on the host;
- host syncs: none on either path (no .item(), no device value in a Python condition);
- amax / scale state: none. Current scaling: each replay's kernels compute every block's scale from the tensor itself;
  Transformer Engine's autocast and quantisers never run with te_operands "ours";
- the non-finite flag: one int32 that the kernels find when they are launched, so a capture holds its address.
  _flag_ptr checks before every replay step that the kernels' flag is the trainer's and has not moved. train_step zeroes
  it before the micro-batches and reads it with the gradient norm after them, both eagerly, outside the graphs, ordered
  with the replays by the stream;
- allocator: the quantiser's outputs, TE's padded scales and GEMM outputs come from the graph's pool at the capture and
  every replay rewrites them; what lives across steps (the flag, the kernels' never-written stand-in flag of FLAG_BIT 0,
  the compiled Triton kernels; TE's cuBLAS workspace, which TE makes at its first GEMM: a judgement, TE's source not read
  here) is made by the process's first step, before any capture;
- RNG: none (MXFP8 rounds to nearest).
So no captured statement changes: both paths run the code they run uncaptured, and cuda_graphs "none" is main's code. The
CPU checks the graph path uncaptured against "none" bit for bit (tests/test_cuda_graphs_mxfp8.py, TE through a stand-in);
the run's part "mx" (scripts/n1_cuda_graphs_check.py) checks the capture against "none" on a slice, bit for bit, and with
te_operands "ours" (sm100 only) on a B200.

Not captured (and why):
- the batch itself (Trainer.batch_at: the shard packer on the host), the head positions and the boundaries (host numpy
  from the host batch), and their copies: host work before each replay, a few hundred microseconds a micro-batch;
- Muon's LR step and AdamW: lr, weight decay and the bias corrections are Python floats of the step; with them in device
  memory p.add_(O, alpha) would become another kernel (p + O * s, two roundings instead of one) and the step would leave
  the bit-for-bit path. They are once an optimizer step (1/64 of the micro-batches at K = 64);
- the step's metrics: one wait for the GPU at the end of the step (the losses, the gradient norm), as today.

THE WRITE MIXES AND TST. The 4B draft configuration had to set cuda_graphs none because check_config refused three things by name:
the GDN-side arms (the write mix and the query mix), E
(attn_res_fused) and tst.schedule. All three are taken now, in the forms the draft uses; the rest of each area is still refused by name.
What was found:
- The write mixes, E and the custom ops. No statement of the fused path reads a device value on the host (ced/kernels/write_mix.py,
  write_mix_triton.py, entry_mix.py, attn_res.py EntryWalk._group_read, gdn_arms.write_inputs): the ops allocate their outputs (the
  float32 extra mixes included: write_mix_storage float32, write_mix_row_term pdp is a dtype choice at build) from the capture's pool, take
  their launch from the committed pins (a host lookup) and their grids from shapes; the Triton kernels' alignment specialisation reads
  data_ptr on the host (the pool's blocks are as aligned as the warm-up's). What stays refused is what does read the host: CCR
  (gdn_arms.chunk_layout copies the boundaries to the host), qmix's static choice (gdn_qmix_select_step: a host read and a write into a
  buffer mid-run), the variants and forms nothing tests (the "head" form, korig, vonly, qshare*, mix_fold). The capture itself is the run's
  (scripts/graphs_wmix_tst_check.py, legs wm, mx, e).
- TST changes what a captured micro-batch is, in four ways, and a run crosses one switch (or several):
  1. The shapes. A bag step's tokens are (rows, L, s), its targets (rows, L, s) with their validity, and the loss is the multi-target
     head (Trainer._micro_loss_tst) with the bag weights (s,) and ONE host divisor for the whole step (the sum of the valid targets'
     weights over the batch). A graph captured in one phase is not the other's, so the runner has PHASES: NEXT_TOKEN and ("bag", s). Each
     has its own static buffers, accumulators, micro-batch graph and metadata capacity; entering another phase releases the previous
     phase's graphs (the micro-batch graph and the optimizer graph, which shares its pool, and what they hold; the allocator gets the
     pool back, which the next phase's eager step needs) and allocates the new phase's buffers.
  2. The warm-up. A phase captures after ONE eager graph-path step of its own (capturing: steps >= WARMUP_STEPS and phase_steps >= 1): its
     kernels (the multi-target head, the embedding of bags, the compiled layers for the new metadata's shapes) are compiled and loaded by
     that step, as the process's second step does for the first phase. Timeline of a 12-update run with the switch at 6: update 0 today's
     step, 1 eager (graph path), 2 captures the bag phase's two graphs and replays, 3 to 5 replay, 6 the switch and the next-token
     phase eagerly, 7 captures, 8 to 11 replay (scripts/graphs_wmix_tst_check.py judges exactly these counts).
  3. The divisor. Python floats are baked into a capture. The bag loss divides by a host float (a = v w / w_total); the graph holds it in a
     device cell the step rewrites (DeviceDivisor) and divides as the eager step does ON THIS DEVICE. How a device divides a float32
     tensor by a Python float is not assumed: divisor_form MEASURES it at the runner's construction (divisor_probe: 64 divisors against
     four forms, "divide", "mul", "divide64", "mul64", and two diagnostics) and takes the first form that is the eager division bit for
     bit on every divisor; a device where none is refuses the bag loss's graph by name, with the table. (The first GPU run
     is why: the source of older releases reads "a * reciprocal(b)" in float32 for a CPU-scalar divisor on
     CUDA, and torch 2.14.1+cu130 on the H100 is not that, on the probes; the table of the second run names what it is.) The bag weights
     are a float64 device buffer made once per phase (their host copy would be a pageable host-to-device copy inside the capture).
  4. The documents. A bag row covers s x L tokens, about s times a row's documents (ced/train/tst.py bag_row; a bag position's document is
     its last input token's): the GDN kernels' fixed-capacity metadata holds cuda_graphs_docs_per_row x s segments a row in a bag phase,
     so the micro-batches fit the graph (a miss runs eagerly beside the pool and needs its own activations: at 4B an out-of-memory
     risk). The metadata's shapes differ by phase, so the compiled layers take one more entry per layer kind at the switch (static shapes,
     configure_compile: automatic_dynamic_shapes off, the recompile limit raised to 64 by the runner). The FA3 / FA4 capacity
     (attention.fa3_docs_per_row x rows, flash_capped.flash_varlen) is the model's, checked on the host in _fill as before: a bag row
     with more documents than it holds stops the run by name, which is the FA4 build's to raise (GraphRunner.fa3_capacity, read in _fill only).
- The optimizer state at the switch is reset OUTSIDE the graphs, eagerly, before the step's batch (TSTPlan.switch), and IN PLACE
  (MuonWithAdamW.reset_moments zeroes every moment: the momentum buffers keep their addresses, which is all the optimizer graph holds of
  them, _opt_key), equal to emptying the state bit for bit (tests/test_matrix_optim.py). The runner releases the optimizer graph with the
  micro-batch graph at a phase change anyway and captures it again in the new phase. A reset that made new tensors (state.clear()) would
  change _opt_key and _check_opt refuses the replay by name; tst.keep_optimizer_state makes no call at all. The LR factors (bag_lr_power,
  rewarm_steps) are host floats of the eager LR step; gdn_conv_reinit and gdn_decay_rescale write parameters in place at the switch.
- Refused by name (tst_refusals): token_scale_rope, gdn_resolution_gates, bag_input other than "mean", aux_multi_hot and output_only: each
  changes what a captured forward or loss computes in a bag phase (another rotary form, the GDN gates' scale, another embedding, a
  next-token step with several targets and a weight that changes every step) and nothing tests it under a capture.

On the CPU (the tests) nothing is captured: the same code runs uncaptured from the process's second step on, so the CPU
rehearsal checks everything but the capture itself against cuda_graphs "none" bit for bit.
"""

from __future__ import annotations

import contextlib

import torch

MODES = ("none", "microbatch", "step")
WARMUP_STEPS = 2  # steps of a process before the first capture (module docstring, "Warm-up")
ALIGN = 32  # int32 entries (128 bytes) between the regions of the metadata buffer
# the names of the streams a parameter's AccumulateGrad node can be made on (module docstring, "Streams"), and the key of
# the record kept in the node's metadata dict (it lives as long as the node: torch.autograd.graph.get_gradient_edge)
CAPTURE_STREAM = "the capture stream"
DEFAULT_STREAM = "the default stream (stream 0)"
MADE = "ced.cuda_graphs.made"
# The write mix's forms and variants the capture is built for, the phases of a step (the key of what a captured micro-batch bakes
# in), and what a bag phase's divisor of the loss is (module docstring, "THE WRITE MIXES AND TST")
WRITE_MIX_FORMS = ("none", "layer")  # EncoderConfig.gdn_write_mix: the 4B's wsl-nowd write mix is "layer"; "head" is not built here
WRITE_VARIANTS = ("none", "qmix")  # EncoderConfig.gdn_write_variant: wsl (and wsl-nowd) and qmix (gdn_arms.MX_VARIANTS); the rest are refused
NEXT_TOKEN = ("next",)  # the phase key of a step of token rows (one target a position): the base's step
DIVIDE_FORMS = ("divide", "mul", "divide64", "mul64")  # how a device divides a float32 tensor by a Python float (divisor_form), cheapest first


def check_config(cfg) -> None:
    """ValueError, by name, for a cuda_graphs setting or a config it is not built for. Off ("none"): only the keys."""
    mode, D = cfg.cuda_graphs, cfg.cuda_graphs_docs_per_row
    if mode not in MODES:
        raise ValueError(f"cuda_graphs {mode!r}: one of {MODES}")
    if isinstance(D, bool) or not isinstance(D, int) or D < 1:
        raise ValueError(f"cuda_graphs_docs_per_row {D!r}: an integer >= 1 (segments a row of the GDN kernels' fixed-capacity "
                         "metadata)")
    if mode == "none":
        return
    m, e = cfg.model, cfg.model.encoder
    refused = [
        (m.arch != "control", f"arch {m.arch!r} (built for arch 'control')"),
        (m.loop.pass2_gate != "none", "model.loop.pass2_gate (the control's second pass)"),
        (bool(m.loop.recur_layers), "model.loop.recur_layers (a looped middle)"),
        (e.loops != 1, f"model.encoder.loops {e.loops}"),
        (e.attn_res not in ("none", "block", "block_entry"), f"model.encoder.attn_res {e.attn_res!r}"),
        (e.attn_res_heads != 1 or e.attn_res_query != "static", "R1-entry's heads or token queries (attn_res_heads, attn_res_query)"),
        (e.gdn_write_mix not in WRITE_MIX_FORMS or bool(e.gdn_ccr_spacing) or e.gdn_write_variant not in WRITE_VARIANTS
         or bool(e.gdn_qmix_select_step) or bool(e.mix_fold),
         "the GDN-side arms (gdn_write_mix, gdn_ccr_spacing) other than arm A's 'layer' write mix with the variant none or qmix, "
         "refused with them: gdn_write_variant other than those, gdn_qmix_select_step, mix_fold (CCR reads the chunk metadata on the "
         "host, qmix's static choice is a host read and a buffer change, the other forms and the folds are not tested under a capture)"),
        (e.layer_scalars != "none", f"model.encoder.layer_scalars {e.layer_scalars!r}"),
        (bool(e.canon_kernel), "model.encoder.canon_kernel"),
        (e.attention.normalizer != "softmax", f"the {e.attention.normalizer!r} normaliser"),
        (e.gdn.kernel == "fla", "GDN kernel 'fla' (fla makes its chunk index on the host; use kernel 'ced')"),
        (e.gdn.edges != "plain", f"model.encoder.gdn.edges {e.gdn.edges!r} (the edges ops of builds C and C2 form "
                                 "their own chunk metadata; not built together with the graphs' fixed-capacity metadata)"),
        (e.gdn.chunk_backend not in ("triton", "flashqla") or (e.gdn.chunk_backend == "flashqla" and _tokens(cfg) < 64),
         f"model.encoder.gdn.chunk_backend {e.gdn.chunk_backend!r} with {_tokens(cfg)} tokens a micro-batch (FlashQLA's "
         "captured form, fold-gdncore, prefetches the 64 tokens before an empty padding segment's end: T >= 64)"),
        (m.nvfp4.mode != "none", "model.nvfp4 (its stochastic rounding draws random numbers in the step)"),
        *((True, why) for why in fp8_refusals(m.fp8)),  # model.fp8: the 4B plan's MXFP8 paths only (module docstring, "MXFP8")
        (m.cp_head.enabled, "model.cp_head"),
        (not (cfg.head_select and cfg.head_impl == "fused"), "a head other than head_select with head_impl 'fused'"),
        (cfg.optim.name != "muon" or cfg.optim.hyperball, f"optim.name {cfg.optim.name!r} (built for Muon with AdamW)"),
        (cfg.optim.ours.rotation_log_every != 0 or cfg.optim.ours.log_every != 0, "optim.ours's logs"),
        (cfg.activation_checkpoint != "none", f"activation_checkpoint {cfg.activation_checkpoint!r}"),
        *((True, why) for why in tst_refusals(cfg.tst)),  # TST: the schedule and the options the 4B plan uses (module docstring, "TST")
        (bool(cfg.shape_schedule), "shape_schedule"),
        (cfg.log_stream_rms_every != 0, "log_stream_rms_every"),
        (cfg.data.loss_mask != "all", f"data.loss_mask {cfg.data.loss_mask!r} (the head's size follows the kept targets: every "
                                      "micro-batch would miss the graph's fixed head; a fixed head at those positions is not "
                                      "bit for bit, a reduction of another length)"),
        (cfg.data.batch_size % (cfg.micro_batch_size or cfg.data.batch_size) != 0, "a batch that is not whole micro-batches"),
    ]
    bad = [why for cond, why in refused if cond]
    if bad:
        raise ValueError(f"cuda_graphs {mode!r} is built for the plain control (arch 'control', one pass, attn_res none / block "
                         f"/ block_entry), Muon with AdamW, the fused head and no activation checkpoint, in BF16 or on the 4B "
                         f"plan's MXFP8 paths; not for: {'; '.join(bad)}. Leave it 'none'")


def fp8_refusals(q) -> list[str]:
    """Why the graphs are not built for model.fp8 q (an FP8Config), by name: [] with the switch off and on the two MXFP8
    paths the 4B plan uses (module docstring, "MXFP8"): mode "linears", scaling "mxfp8", no chains but in_norm and
    exit_entry (ced/model/mx_wire.py check_chains says where they are built), and either backend "emulated" with
    quantiser "triton" (our Triton quantiser; the non-finite flag off, or on in its default form "pure") or backend "te"
    with te_operands "ours" (Transformer Engine's GEMMs on our quantiser's operands, as the B200 harness runs it; the
    flag on or off). Every other combination stays refused: no capture of it is tested.

    The layer chains in_norm and exit_entry (model.fp8.chains) on either path: each is a functional custom operator
    (ced/kernels/mx_chain_ops.py) whose outputs follow from its input's shape; the TE form's flag is found when the kernel
    runs (mx_ops._flag_of), the same flag _flag_ptr checks; the layer's MXHold is a Python attribute set and cleared around
    the mixer and MLP calls inside the captured forward (no tensor lives in it between calls). The other chains stay refused
    (no capture of them is tested). (This paragraph stood in the module docstring; the merge review of the five folds moved it
    here so that check_config's lines stay where the B200 step audit cites them:152 and:159: tests/test_b200_step_audit.py,
    tests/test_gdn_flashqla_graphs.py.)"""
    if q.mode == "none":
        return []
    out = []
    if q.scaling != "mxfp8":
        out.append(f"model.fp8.scaling {q.scaling!r} (D1a's emulated FP8 recipe: no capture of it is tested; the graphs are "
                   "built for 'mxfp8')")
    backend, triton = getattr(q, "backend", "emulated"), getattr(q, "quantiser", "torch") == "triton"
    if backend == "emulated" and q.scaling == "mxfp8" and not triton:
        out.append("model.fp8.quantiser 'torch' (the PyTorch reference quantiser: no capture of it is tested; the graphs are "
                   "built for our Triton quantiser, 'triton', the same operand bytes)")
    if triton and getattr(q, "nonfinite_flag", False) and getattr(q, "nonfinite_flag_form", "pure") != "pure":
        out.append(f"model.fp8.nonfinite_flag_form {q.nonfinite_flag_form!r} (the flagged quantiser as an ordered effectful "
                   "operator: no capture of it is tested; the default form 'pure' is)")
    if backend == "te" and getattr(q, "te_operands", "te") != "ours":
        out.append(f"model.fp8.te_operands {q.te_operands!r} (Transformer Engine's own te.Linear quantises inside TE under "
                   "TE's autocast, whose capture recipe is TE's own, make_graphed_callables, not this runner: no capture of it "
                   "is tested; the graphs are built for te_operands 'ours')")
    chains = tuple(getattr(q, "chains", ()))
    if chains and not set(chains) <= {"in_norm", "exit_entry"}:  # the layer chains are built
        out.append(f"model.fp8.chains {list(q.chains)} (the fused chains other than in_norm and exit_entry: no capture of "
                   "them is tested; in_norm and exit_entry are built for the graphs on both paths)")
    return out


def tst_refusals(t) -> list[str]:
    """Why the graphs are not built for the TST options t (a TSTConfig), by name: [] for the 4B draft's (the schedule alone, the bag
    mean, no token-scale rotary, no resolution gates) and for the options that act on the host or on parameters in place at a switch
    (keep_optimizer_state, bag_lr_power, rewarm_steps, gdn_conv_reinit, gdn_decay_rescale, phase1_exclude_sources). Every option that
    changes what a captured forward or loss computes in a bag phase is refused: nothing tests it under a capture (module docstring,
    "THE WRITE MIXES AND TST")."""
    out = []
    if t.token_scale_rope:
        out.append("tst.token_scale_rope (the bag phase's capped-rotary layers take another rotary form, rope_scale: not tested under a capture)")
    if t.gdn_resolution_gates != "none":
        out.append(f"tst.gdn_resolution_gates {t.gdn_resolution_gates!r} (the bag phase's GDN gates take gdn_scale / gdn_beta_power: "
                   "not tested under a capture)")
    if t.bag_input != "mean":
        out.append(f"tst.bag_input {t.bag_input!r} (the bag phase's embedding takes another form: not tested under a capture)")
    if t.aux_multi_hot:
        out.append("tst.aux_multi_hot (next-token steps with several targets and a weight that changes every step: not built)")
    if t.output_only:
        out.append("tst.output_only (bag phases that read token rows with several targets: not built)")
    return out


def _tokens(cfg) -> int:
    """Tokens a micro-batch: rows x seq_len, the T of the GDN kernels' flattened call.

    FlashQLA (model.encoder.gdn.chunk_backend "flashqla"): its captured form takes the same fixed-capacity metadata as our
    Triton core (ced/kernels/gdn_flashqla.py, "THE CAPTURED FORM": the graphs' own index attached as FlashQLA's prepared
    metadata, no host read, grids and allocations from the fixed capacity), so check_config refuses it only below 64
    tokens a micro-batch (its backward's prefetch). Everything else the graphs refuse stays refused for it as for "triton"
    (the second pass, CCR, the edges, fla); FlashQLA's own refusals (bf16 inputs, K = V = 128, SM90 / SM100) are its
    layer's."""
    return (cfg.micro_batch_size or cfg.data.batch_size) * cfg.data.seq_len


def check_capture(cfg, device_type: str) -> None:
    """ValueError, by name, for a setting a CUDA capture cannot take, beyond check_config (2026-10-17). The attention path's
    data-dependent checks read the device when a layer runs uncompiled (ced/model/fast_attention.py capped_block_masks:
    bool((keys.idx == arange).all()) on every forward whose layers are not compiled; CappedMasks.check and
    attend_capped_fast's own check; ced/model/attention.py RotaryContext.check): a host sync in the step, an error inside
    a capture. Under compile "all" they are skipped (torch.compiler.is_compiling() in the layers, the compiled mask build
    outside them). So a capture needs compile "all"; the CPU runs the graph path uncaptured and is not refused."""
    if cfg.cuda_graphs != "none" and device_type == "cuda" and cfg.compile != "all":
        raise ValueError(f"cuda_graphs {cfg.cuda_graphs!r} on CUDA is built for compile 'all', not {cfg.compile!r}: uncompiled "
                         "layers run the attention path's data-dependent checks (a device read each forward, an error inside a "
                         "capture)")


_FORMS: dict = {}
_PROBES: dict = {}


def divisor_probe(device) -> dict:
    """What divisor_form measured on `device`'s type: {"form": the form or None, "matches": {candidate: the divisors (of "n") whose result
    equalled the eager division bit for bit}, "n": the number of divisors}. Runs the probe once per device type."""
    key = torch.device(device).type
    if key in _PROBES:
        return _PROBES[key]
    import numpy as np

    g = torch.Generator().manual_seed(7)
    x = (torch.rand(1 << 14, generator=g) + 0.25).to(device)
    divisors = [3.0, 7.0, 0.1, 1.0 / 3.0, 12345.678, 0.7, 51.2, 1e-3, 2.5e5, 9.99, 4096.5, 1.7]
    divisors += [float(v) for v in torch.rand(52, generator=g) * 1e4 + 0.01]
    ones = torch.ones((), device=device)
    matches = {k: 0 for k in ("divide", "mul", "divide64", "mul64", "recip32", "recip64")}  # the last two: diagnostics, never chosen
    for f in divisors:
        want = (x / f).view(torch.int32)  # the eager step's own division, on this device
        f32 = torch.tensor(f, dtype=torch.float32, device=device)
        f64 = torch.tensor(f, dtype=torch.float64, device=device)
        with np.errstate(all="ignore"):
            inv32 = torch.tensor(float(np.float32(1) / np.float32(f)), dtype=torch.float32, device=device)
            inv64 = torch.tensor(float(np.float32(1.0 / f)), dtype=torch.float32, device=device)
        got = {"divide": x / f32,  # x / float32(f)
               "mul": x * (ones / f),  # x times the multiplier this device's own division of 1 by f makes (measured)
               "divide64": (x.double() / f64).float(),  # computed in double, rounded once
               "mul64": (x.double() * (1.0 / f64)).float(),  # times the double reciprocal, rounded once
               "recip32": x * inv32,  # x * (float32(1) / float32(f)): "compute a * reciprocal(b)" in float32
               "recip64": x * inv64}  # x * float32(1.0 / f)
        for k, v in got.items():
            matches[k] += int(bool(torch.equal(want, v.view(torch.int32))))
    n = len(divisors)
    form = next((k for k in DIVIDE_FORMS if matches[k] == n), None)  # the first (cheapest) form that is the eager division on every probe
    _PROBES[key] = {"form": form, "matches": matches, "n": n}
    return _PROBES[key]


def divisor_form(device) -> str:
    """How PyTorch divides a float32 tensor by a Python float on `device`, MEASURED (module docstring, "THE WRITE MIXES AND TST"):
    one of DIVIDE_FORMS: "divide" (x / float32(f): the CPU's division), "mul" (x times the multiplier the device's own division of 1 by
    f makes: the CUDA kernel's "a * reciprocal(b)" for a CPU-scalar divisor, whatever precision it forms the reciprocal in), "divide64" or
    "mul64" (computed in double, rounded once). A bag step's loss divides by a host float (Trainer._micro_loss_tst); the graph holds it in
    device memory and must divide as the eager step does. The first form that equals the eager division bit for bit on every probe wins;
    ValueError, with the table, for a device where none does (a graph of a bag step would not be the eager step's). Cached per device
    type; the probe's host reads happen at the runner's construction, never inside a capture."""
    key = torch.device(device).type
    if key in _FORMS:
        return _FORMS[key]
    probe = divisor_probe(device)
    if probe["form"] is None:
        raise ValueError(f"cuda_graphs: on {key} PyTorch's division of a float32 tensor by a Python float is none of {DIVIDE_FORMS} on the probes "
                         f"(divisors matched out of {probe['n']}, by candidate: {probe['matches']}): a graph of a bag step (TST) would not "
                         "divide as the eager step does")
    _FORMS[key] = probe["form"]
    return probe["form"]


class DeviceDivisor:
    """A bag step's loss divisor (Trainer._micro_loss_tst: the sum of the valid targets' weights over the step's whole batch, a host
    float64) held in device memory, so a captured micro-batch reads it from a cell the step rewrites. Called on the weights, it divides
    as the eager step's division by the Python float does on this device (divisor_form). The cell is float32 for "divide" and "mul",
    float64 for "divide64" and "mul64"; what it holds: host_value(total, form), or for "mul" what the device's own division of 1 by the
    total gives (GraphRunner._set_divisor)."""

    def __init__(self, cell: torch.Tensor, form: str):
        assert form in DIVIDE_FORMS, form
        self.cell, self.form = cell, form

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:  # the eager division of a float64 model is by the float64 total, in double
            raise ValueError(f"cuda_graphs: the bag loss's divisor cell is built for float32 weights; these are {x.dtype}")
        if self.form == "divide":
            return x / self.cell
        if self.form == "mul":
            return x * self.cell
        if self.form == "divide64":
            return (x.double() / self.cell).float()
        return (x.double() * self.cell).float()

    @staticmethod
    def host_value(total: float, form: str) -> float:
        """The number the cell holds for the host divisor `total` when the host can form it: float32(total) ("divide"), the total
        ("divide64"), its double reciprocal ("mul64"). ValueError for "mul": the multiplier is measured on the device."""
        import numpy as np

        if form == "divide":
            return float(np.float32(total))
        if form == "divide64":
            return float(total)
        if form == "mul64":
            return 1.0 / float(total) if total else float("inf")
        raise ValueError(f"the cell of form {form!r} is not formed on the host")


class GraphRunner:
    """The graph path of Trainer.train_step (module docstring). One per trainer; Trainer.train_step runs the process's
    first step inside first_step, and calls zero_grads, micro_batches, clip_and_orthogonalise / optimizer_rest ("step")
    and end_step."""

    def __init__(self, trainer):
        from ced.train.chunked_loss import HEAD_CHUNK
        from ced.train.muon import MuonWithAdamW
        from ced.train.varlen import fixed_capacity

        cfg = trainer.cfg
        check_config(cfg)
        check_capture(cfg, trainer.device.type)
        if not isinstance(trainer.optimizer, MuonWithAdamW):
            raise ValueError(f"cuda_graphs {cfg.cuda_graphs!r} is built for MuonWithAdamW, not {type(trainer.optimizer).__name__}")
        if trainer._stats_model is not None and trainer._stats_model is not trainer.model:
            raise ValueError("cuda_graphs: the monitored model must be the trained model")
        self.t = trainer
        self.mode = cfg.cuda_graphs
        self.capture = trainer.device.type == "cuda"  # the CPU runs the same code uncaptured (module docstring)
        self.rows, self.L = trainer.micro, cfg.data.seq_len
        # the head's fixed size: head_positions' padded size for a micro-batch whose every position counts (rows x seq_len
        # rounded up to a multiple of HEAD_CHUNK, at least HEAD_CHUNK; 8,192 for a row of 8,192)
        self.N = max(HEAD_CHUNK, -(-(self.rows * self.L) // HEAD_CHUNK) * HEAD_CHUNK)
        self.varlen = trainer.fused_varlen  # the GDN kernels' boundaries (CUDA with GDN kernel "ced")
        self.docs = cfg.cuda_graphs_docs_per_row  # segments a row of the GDN metadata holds (a bag phase: times its s)
        self.S, self.nt_max = fixed_capacity(self.rows, self.L, self.docs)
        # the phase of a step (module docstring, "THE WRITE MIXES AND TST"): what a captured micro-batch bakes in. A run without TST has
        # one (NEXT_TOKEN); a TST run has the bag size's and the next-token one. The buffers, the accumulators, the micro-batch graph
        # and the metadata capacity (S, nt_max) below are the current phase's; entering another phase releases the graphs.
        self.phase: tuple | None = None
        self.phase_s = 1  # the bag size of the phase (1: token rows)
        self.phase_steps = 0  # graph-path steps that ended in this phase: a phase captures after one eager step of its own
        self.bag = False  # the phase reads bags (multi-target loss, 3-D tokens)
        self.totals: dict | None = None  # a bag step's loss divisor in device memory (DeviceDivisor)
        self.div_form: str | None = divisor_form(trainer.device) if cfg.tst.schedule else None
        self._phase_open = False  # micro_batches ran in this step (end_step counts it for the phase)
        # FA3's and FA4's own fixed capacity (ced/kernels/flash_capped.py): the graph path gives the model no host
        # boundaries, so an over-full micro-batch is told here, on the host, before the copies: _fill counts it a miss
        # (it runs eagerly, where flash_varlen takes the doubled capacity), as it counts one over the GDN capacity
        att = cfg.model.encoder.attention
        self.fa3_capacity = self.rows * att.fa3_docs_per_row if att.backend in ("fa3", "fa4") else None
        self.flash_backend = att.backend
        self.steps = 0  # steps this runner has seen end in this process
        self.grads: list | None = None  # [(p, grad)]: the persistent gradients (after the process's first step)
        self.ptrs: list | None = None  # the parameters' and gradients' storage, checked before every replay step
        self.buf: dict | None = None  # the static inputs
        self.acc: dict | None = None  # the static accumulators: "loss", "n_dropped" and the loss's metric sums
        self.norms: dict | None = None
        self.graph = self.opt_graph = None
        self.graph_stats: dict = {}
        self.orth: list | None = None
        self.opt_norm = None
        self.opt_key: tuple | None = None  # what the optimizer graph baked in at its capture (_opt_key)
        self.stream = torch.cuda.Stream(trainer.device) if self.capture else None
        self.counts = {"replays": 0, "eager": 0, "misses": 0, "captures": 0, "phases": 0, "released": 0}
        # module docstring, "Streams": inside first_step (the process's first step); the depth of _on_stream (on the
        # CPU, the "stream" the labels name); the forward hook that labels the nodes; the checks' results (the run reads)
        self._in_first_step = False
        self._scope = 0
        self._hook = None
        self.checks: dict = {}
        if cfg.tst.schedule:  # what the divisor probe found on this device (the run reads it; divisor_probe)
            self.checks["divisor"] = {"form": self.div_form, **{k: v for k, v in divisor_probe(trainer.device).items() if k != "form"}}
        if self.capture and cfg.compile != "none":
            # the layers see a second form of the varlen metadata (the fixed one, from the second step on): one more
            # compiled entry a layer kind. Keep the recompile limit above that count; reaching it stays an error, never a
            # fallback (configure_compile)
            import torch._dynamo.config as dynamo_config

            dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 64)

    # --- the step's frame -------------------------------------------------------------------------------------------
    @property
    def active(self) -> bool:
        """Whether this step runs the graph path: from the process's second step on (the first made the gradients)."""
        return self.grads is not None

    @property
    def capturing(self) -> bool:
        """Whether this step replays (or captures) graphs: CUDA, after the process's warm-up steps and after one eager step of the
        current phase (a phase's own kernels, the head's multi-target form of a bag phase, the new metadata's compiled layers, are
        compiled and loaded by that step before the capture, as the process's second step does for the first phase)."""
        return self.capture and self.steps >= WARMUP_STEPS and self.phase_steps >= 1

    # --- the streams (module docstring, "Streams") -------------------------------------------------------------------
    @property
    def first_step_pending(self) -> bool:
        """Whether Trainer.train_step is the process's first step and has not entered first_step yet."""
        return not self.active and not self._in_first_step

    @contextlib.contextmanager
    def first_step(self):
        """Around the process's first step (Trainer.train_step: today's code, run inside this): on the capture stream, so
        that the AccumulateGrad nodes its forwards make, and the persistent gradients, are made there; the labels on."""
        self._in_first_step = True
        try:
            self._start_labels()
            with self._on_stream():
                yield
        finally:
            self._in_first_step = False

    @contextlib.contextmanager
    def _on_stream(self):
        """Eager work of the graph path on the capture stream, ordered after the current stream's work and before its
        next (the pattern of PyTorch's warm-up: torch/cuda/graphs.py make_graphed_callables). On the CPU: the runner's
        scope alone (nothing is captured; the labels name it)."""
        prev = torch.cuda.current_stream() if self.capture else None
        if self.capture:
            self.stream.wait_stream(prev)
        self._scope += 1
        try:
            with torch.cuda.stream(self.stream) if self.capture else contextlib.nullcontext():
                yield
        finally:
            self._scope -= 1
            if self.capture:
                prev.wait_stream(self.stream)

    def _stream_name(self) -> str:
        """The stream the work being enqueued now runs on: the capture stream, the default stream, or another."""
        if not self.capture:
            return CAPTURE_STREAM if self._scope else DEFAULT_STREAM
        s = torch.cuda.current_stream()
        if s == self.stream:
            return CAPTURE_STREAM
        if s == torch.cuda.default_stream(s.device):
            return DEFAULT_STREAM
        return f"stream {s.stream_id} (neither the capture stream nor the default stream)"

    def _parameters(self):
        return [(n, p) for n, p in self.t.model.named_parameters() if p.requires_grad]

    def _label(self, module, args, output) -> None:
        """Forward hook of the first two steps: each parameter's live AccumulateGrad node gets, once, the step and the
        stream it was made on (the forward that just ran made it, or an earlier one did and labelled it). A parameter
        without a live node gets a new one, labelled and freed at once (nothing holds it)."""
        if not torch.is_grad_enabled() or (self.capture and torch.cuda.is_current_stream_capturing()):
            return
        from torch.autograd.graph import get_gradient_edge

        made = (self.steps + 1, self._stream_name())
        for _, p in self._parameters():
            get_gradient_edge(p).node.metadata.setdefault(MADE, made)

    def _start_labels(self) -> None:
        if self._hook is None:
            self._hook = self.t.model.register_forward_hook(self._label)

    def _stop_labels(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _check_accumulators(self, where: str) -> None:
        """Refuse, by name, any parameter whose AccumulateGrad node is alive and was made on a stream other than the
        capture stream: a backward that adds into it cannot be captured (module docstring, "Streams"). Records what it
        saw in self.checks[where] (live labelled nodes, by stream)."""
        from torch.autograd.graph import get_gradient_edge

        live: dict = {}
        for name, p in self._parameters():
            made = get_gradient_edge(p).node.metadata.get(MADE)
            if made is not None:
                live.setdefault(made, []).append(name)
        self.checks[where] = {f"step {s}, {stream}": len(names) for (s, stream), names in sorted(live.items())}
        bad = {k: v for k, v in live.items() if k[1] != CAPTURE_STREAM}
        if bad:
            lines = "; ".join(f"{len(names)} made on {stream} by a forward before the end of step {s} ({', '.join(names[:8])}"
                              f"{', ...' if len(names) > 8 else ''})" for (s, stream), names in sorted(bad.items()))
            raise RuntimeError(f"cuda_graphs: at {where}, live AccumulateGrad nodes not made on the capture stream: {lines}. "
                               "A capture whose backward adds into them fails (\"stale reference to the default stream\"). "
                               "Every forward of the graph path runs on the capture stream (GraphRunner.first_step, "
                               "_on_stream); this one ran outside it and something still holds its autograd graph (a tensor "
                               "with its history, or a reference cycle such as torch.compile's compile-time frames)")

    def zero_grads(self) -> None:
        """The step's start: every persistent gradient zeroed in place (re-attached first, should anything have set a
        parameter's .grad to None or another tensor between two steps)."""
        for p, g in self.grads:
            if p.grad is not g:
                p.grad = g
        torch._foreach_zero_([g for _, g in self.grads])

    def end_step(self) -> None:
        """The step's end. After the process's first step (today's code, which assigned every gradient), its gradients
        become the persistent ones: the set of parameters with a gradient is the run's (no data-dependent branch in the
        models this is built for), and a parameter without one keeps none (it is not clipped or updated, as today)."""
        if self.grads is None:
            self.grads = [(p, p.grad) for p in self.t.model.parameters() if p.grad is not None]
        self.steps += 1
        if self._phase_open:  # a graph-path step ended in the current phase
            self.phase_steps += 1
            self._phase_open = False

    def _storage(self) -> list:
        out = [(p.data_ptr(), g.data_ptr()) for p, g in self.grads]
        flag = self._flag_ptr()
        return out if flag is None else out + [(flag, None)]

    def _flag_ptr(self) -> int | None:
        """The address of the MXFP8 non-finite flag (model.fp8.nonfinite_flag; None without it), which a captured quantiser
        kernel holds (module docstring, "MXFP8"). The kernels find the flag when they are launched, so at the capture: by
        their tensor's device (ced/kernels/mx_ops.py _flag_of; quantiser "triton") or as mx_ops.FLAG (te_operands "ours").
        Both must be the trainer's flag, which train_step zeroes before the micro-batches and reads after them, outside
        the graphs; else a replay would OR its bits into a tensor the step never reads. Refused by name."""
        f = self.t._mx_flag
        if f is None:
            return None
        from ced.kernels import mx_ops

        if mx_ops._FLAGS.get(f.device) is not f or mx_ops.FLAG is not f:
            raise RuntimeError("cuda_graphs: the MXFP8 non-finite flag the quantiser kernels find (ced.kernels.mx_ops _FLAGS "
                               "and FLAG) is not the trainer's: a captured kernel would OR its bit into a tensor the step "
                               "neither zeroes nor reads")
        return f.data_ptr()

    def _check_storage(self) -> None:
        """A captured graph holds the addresses of the parameters, gradients, optimizer state and the MXFP8 non-finite
        flag: refuse to replay it if any parameter, gradient or the flag has moved (nothing in the trainer moves them
        after the warm-up steps)."""
        now = self._storage()
        if self.ptrs is None:
            self.ptrs = now
        elif now != self.ptrs:
            raise RuntimeError("cuda_graphs: a parameter or gradient tensor, or the MXFP8 non-finite flag, was reallocated "
                               "after the graphs were captured")

    # --- the micro-batches ------------------------------------------------------------------------------------------
    def _phase_key(self, tst) -> tuple:
        """The phase of a step: NEXT_TOKEN for rows of one target, ("bag", s) for a bag step (tst: (s, w, totals) of
        Trainer.train_step, or None)."""
        if tst is None:
            return NEXT_TOKEN
        if tst[0] <= 1:
            raise ValueError("cuda_graphs: a next-token step with several targets a position (tst.aux_multi_hot) is not built "
                             "(check_config refuses it)")
        return ("bag", int(tst[0]))

    def _enter_phase(self, key: tuple, batch) -> None:
        """Make `key` the current phase (module docstring, "THE WRITE MIXES AND TST"): nothing when it is. Otherwise the previous
        phase's graphs are released (the micro-batch graph and the optimizer graph, which shares its pool: both are captured
        again, the optimizer graph's momentum buffers being the same tensors, zeroed in place by the switch), the buffers
        and the metadata capacity are the new phase's, and its first graph-path step runs eagerly (capturing)."""
        if key == self.phase:
            return
        if self.phase is not None:
            self._release_graphs()
        self.phase, self.phase_steps = key, 0
        self.bag = key[0] == "bag"
        self.phase_s = key[1] if self.bag else 1
        # the segments a row can hold grow with the tokens it covers: a bag row of s x L tokens holds about s times the documents of a
        # row of L (ced/train/tst.py bag_row), so the GDN metadata's capacity per row is cuda_graphs_docs_per_row x s
        from ced.train.varlen import fixed_capacity

        self.docs = self.t.cfg.cuda_graphs_docs_per_row * self.phase_s
        self.S, self.nt_max = fixed_capacity(self.rows, self.L, self.docs)
        self.buf = self.acc = None
        self.graph_stats = {}
        self.totals = None
        self._alloc(batch)
        self.counts["phases"] += 1

    def _release_graphs(self) -> None:
        """The phase switch: let go of the micro-batch graph, the optimizer graph and what they hold (the pool's memory returns to the
        allocator, which the next phase's eager step needs: at 4B one micro-batch's activations). A host wait first: nothing of the
        graphs is in flight (the step before ended with the host's read of its metrics), and the wait makes it certain."""
        if self.capture:
            torch.cuda.synchronize(self.t.device)
        self.graph = self.opt_graph = None
        self.graph_stats = {}
        self.orth = self.opt_norm = self.opt_key = None
        import gc

        gc.collect()
        if self.capture:
            torch.cuda.empty_cache()
        self.counts["released"] += 1

    def _alloc(self, batch) -> None:
        dev = self.t.device
        r, L = self.rows, self.L
        row = tuple(batch.tokens.shape[1:])  # (L,) or a bag phase's (L, s)
        self.buf = {"tokens": torch.zeros(r, *row, dtype=batch.tokens.dtype, device=dev),
                    "doc_ids": torch.zeros(r, L, dtype=batch.doc_ids.dtype, device=dev),
                    "dec": torch.zeros(r, L, dtype=torch.bool, device=dev),
                    "idx": torch.zeros(self.N, dtype=torch.long, device=dev),
                    "real": torch.zeros(self.N, dtype=torch.bool, device=dev),
                    "scales": torch.zeros(2, dtype=torch.float32, device=dev)}
        if self.bag:  # the targets of a bag position (s a position) and their validity; the weights of the s targets, fixed for the phase
            self.buf["mt"] = torch.zeros(r, L, self.phase_s, dtype=batch.multi_targets.dtype, device=dev)
            self.buf["mv"] = torch.zeros(r, L, self.phase_s, dtype=torch.bool, device=dev)
            from ced.train.tst import bag_weights

            self.buf["w"] = torch.tensor(bag_weights(self.phase_s), dtype=torch.float64, device=dev)
            if self.div_form in ("divide64", "mul64"):  # the double forms' cell
                self.buf["div64"] = torch.zeros((), dtype=torch.float64, device=dev)
            cell = self.buf["div64"] if self.div_form in ("divide64", "mul64") else self.buf["scales"][0]
            self.totals = {"w_total": DeviceDivisor(cell, self.div_form)}
        else:
            self.buf["targets"] = torch.zeros(r, L, dtype=batch.targets.dtype, device=dev)
            self.buf["valid"] = torch.zeros(r, L, dtype=torch.bool, device=dev)
        self.meta_layout = None
        if self.varlen:
            S, nt = self.S, self.nt_max
            up = lambda n: -(-n // ALIGN) * ALIGN  # noqa: E731
            o_ci = up(S + 2)
            o_co = o_ci + up(2 * nt)
            self.meta_layout = (o_ci, o_co, o_co + up(S + 2))
            meta = torch.zeros(self.meta_layout[2], dtype=torch.int32, device=dev)
            self.buf["meta"] = meta
            # the views the graph reads: cu (S + 2), chunk_indices (nt_max, 2), chunk_offsets (S + 2)
            self.buf["varlen"] = {"enc": meta[: S + 2], "enc_cpu": None,
                                  "enc_chunks": (meta[o_ci: o_ci + 2 * nt].view(nt, 2), meta[o_co: o_co + S + 2], nt)}
        else:
            self.buf["varlen"] = None

    def _set_divisor(self, total: float) -> None:
        """The step's bag divisor into the phase's cell (DeviceDivisor), in the form this device divides: float32(total) or the double total
        or its double reciprocal from the host; for "mul", the multiplier the device's own division of 1 by the total makes (one eager
        scalar operation on the current stream, ordered before the replays; nothing is read back)."""
        form = self.div_form
        if form == "mul":
            self.buf["scales"][0].copy_(torch.ones((), device=self.t.device) / total)
        elif form == "divide":
            self._copy(self.buf["scales"], torch.tensor([DeviceDivisor.host_value(total, form), 0.0], dtype=torch.float32))
        else:
            self._copy(self.buf["div64"], torch.tensor(DeviceDivisor.host_value(total, form), dtype=torch.float64))

    def _copy(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        if self.capture and src.is_cuda:  # a device twin the data prefetch staged (ced/train/prefetch.py): device to device
            dst.copy_(src, non_blocking=True)
        elif self.capture:
            dst.copy_(src.pin_memory(), non_blocking=True)
        else:
            dst.copy_(src)

    def _fill(self, batch, mb, valid_loss, staged=None) -> bool:
        """Host work of one micro-batch and its copies into the static inputs. False (nothing copied): it does not fit
        the graph's shapes. staged: the step's device twins (the data prefetch's, already waited for): the row tensors
        are then copied from their twins' rows, device to device, with no pinned allocation or host copy on this thread; the
        head positions and the boundary metadata are host values of this micro-batch and are copied as always. A bag phase
        copies the bags (rows x L x s tokens) and their s targets and validity a position, not the one target a position."""
        from ced.train.chunked_loss import head_positions

        idx, real = head_positions(valid_loss[mb])  # the head's positions as the eager step finds them
        if idx.numel() != self.N:
            return False
        meta = None
        if self.varlen:
            import numpy as np

            from ced.train.batch_check import BatchError
            from ced.train.trainer import micro_batch_error
            from ced.train.varlen import fixed_varlen_host, varlen_for_batch

            try:  # the exact boundaries, checked as the eager step checks them (BatchError names the row)
                host = varlen_for_batch(batch.doc_ids[mb], None, None)
            except BatchError as error:
                raise self.t._batch_error(micro_batch_error(error, mb), self.t.step) from None
            n_seg = len(host["enc"]) - 1
            if self.fa3_capacity is not None and n_seg > self.fa3_capacity:
                # Over the graph's fixed attention capacity (a TST bag micro-batch holds about 4 times the documents of a
                # next-token one). Not a stop (it was flash_varlen's host refusal here, and a stop is deterministic: a resume
                # meets the same batch): a miss, run eagerly with its exact boundaries, where flash_varlen takes the capacity
                # tier that holds it (flash_capped.capacity_for). One graph a tier is the graph builder's to add;
                # capacity_for(n, rows, fa3_docs_per_row) names the tier.
                return False
            fixed = fixed_varlen_host(host["enc"].numpy(), self.rows, self.L, self.docs)
            if fixed is None:
                return False
            o_ci, o_co, n = self.meta_layout
            meta = np.zeros(n, dtype=np.int32)
            meta[: self.S + 2] = fixed["cu"]
            meta[o_ci: o_ci + 2 * self.nt_max] = fixed["ci"].reshape(-1)
            meta[o_co: o_co + self.S + 2] = fixed["co"]
        b = self.buf
        rows = [("tokens", batch.tokens), ("doc_ids", batch.doc_ids), ("dec", batch.dec_mask)]
        rows += [("mt", batch.multi_targets), ("mv", batch.multi_valid)] if self.bag else [("targets", batch.targets),
                                                                                           ("valid", valid_loss)]
        for name, src in rows:
            self._copy(b[name], (src if staged is None else staged.twin(src))[mb])
        self._copy(b["idx"], idx)
        self._copy(b["real"], real)
        if meta is not None:
            self._copy(b["meta"], torch.from_numpy(meta))
        return True

    def _exact_inputs(self, batch, mb, valid_loss) -> dict:
        """A micro-batch that does not fit: its inputs as the eager step makes them (its own head size and boundaries)."""
        from ced.train.chunked_loss import head_positions

        t = self.t
        to = lambda x: x[mb].to(t.device, non_blocking=True)  # noqa: E731
        idx, real = head_positions(valid_loss[mb])
        out = {"tokens": to(batch.tokens), "doc_ids": to(batch.doc_ids), "dec": to(batch.dec_mask),
               "idx": idx.to(t.device, non_blocking=True), "real": real.to(t.device, non_blocking=True),
               "varlen": t._varlen(batch.doc_ids[mb], batch.dec_mask[mb], t.capacity, mb, t.step)}
        if self.bag:  # (the weights are the phase's own, the divisor is the step's: both stay in the phase's buffers)
            out.update(mt=to(batch.multi_targets), mv=to(batch.multi_valid), w=self.buf["w"])
        else:
            out.update(targets=to(batch.targets), valid=to(valid_loss))
        return out

    def _micro(self, I: dict) -> dict:
        """One micro-batch's forward, loss and backward, and its metric sums into the accumulators: the body of
        Trainer.train_step's loop for the configs this is built for, over the inputs I (the static buffers, or a missed
        micro-batch's own tensors). Every micro-batch records the monitoring statistics (collect_stats: the maxima of the
        detached stream before each global layer, read only, outside the compiled layers), so that one graph serves the
        monitored first micro-batch of a log_every-th step too; returns them (a dict of device tensors; {} for a model
        without monitoring). A bag phase (TST) scores each position against its s targets (Trainer._micro_loss_tst) with the
        phase's weights and the step's divisor, both in device memory."""
        from ced.model import act_checkpoint

        t = self.t
        stats = {}
        autocast = torch.autocast(t.device.type, dtype=t.autocast_dtype, enabled=t.autocast_dtype is not None)
        monitor = t._stats_model is not None
        if monitor:
            t._stats_model.collect_stats = True
        try:
            with autocast, act_checkpoint.host_inputs(t.host_pool):
                out = t.model(I["tokens"], I["doc_ids"], I["dec"], return_hidden=True, varlen=I["varlen"])
        finally:
            if monitor:
                stats, t._stats_model.stats = t._stats_model.stats, {}
                t._stats_model.collect_stats = False
        sel = {"all": (I["idx"], I["real"]), "dec_mask": I["dec"]}
        if self.bag:
            loss, part = t._micro_loss_tst(out, I["mt"], I["mv"], sel, I["w"], self.totals, self.norms, self.phase_s)
        else:
            loss, part = t._micro_loss(out, I["targets"], I["valid"], I["dec"], self.norms, sel)
        loss.backward()
        if self.acc is None:  # made at the first micro-batch of the phase's eager step, never inside a capture
            assert not (self.capture and torch.cuda.is_current_stream_capturing())
            self.acc = {"loss": torch.zeros((), device=t.device), "n_dropped": torch.zeros((), dtype=torch.long, device=t.device),
                        **{k: torch.zeros_like(v.detach()) for k, v in part.items()}}
        acc = self.acc
        acc["loss"] += loss.detach()
        for k, v in part.items():
            acc[k] += v.detach()
        acc["n_dropped"] += out.n_dropped.sum()
        return stats

    def _miss(self, batch, mb, valid_loss) -> dict:
        """A micro-batch that does not fit the graph's shapes: eagerly, with its exact shapes (the caller runs it on the
        capture stream: a miss before the capture must not make a node on another stream)."""
        self.counts["misses"] += 1
        self.counts["eager"] += 1
        return self._micro(self._exact_inputs(batch, mb, valid_loss))

    def _capture_micro(self) -> None:
        g = torch.cuda.CUDAGraph()
        try:
            # the data prefetch's thread makes no CUDA call during a capture (it pins host memory and enqueues copies, and the
            # default "global" capture mode refuses such a call from any thread; ced/train/prefetch.py, "Stage")
            with self.t._prefetch_quiet(), torch.cuda.graph(g, stream=self.stream):
                # the statistics are outputs of the graph (each replay rewrites them): held here, so they stay allocated
                self.graph_stats = self._micro(self.buf)
        except Exception as e:
            raise RuntimeError(f"cuda_graphs: the capture of the micro-batch graph failed ({type(e).__name__}: {e})") from e
        self.graph = g
        self.counts["captures"] += 1

    def micro_batches(self, batch, valid_loss, norms: dict, plan_mb, collect: bool, z: float, staged=None, tst=None):
        """Every micro-batch of the step (Trainer.train_step's loop). norms: the step's host counts (Trainer.train_step's,
        max(1, n) each); z: the z-loss weight. staged: the step's device twins when the data prefetch made them
        (ced/train/prefetch.py Staged; None: the micro-batch's rows are copied from the host batch as always). tst: None for rows of
        one target, else (s, w, totals) of a bag step (TSTPlan.weights: the per-target weights and the host totals, whose "w_total"
        is the loss's divisor). Returns (total_loss, sums, n_dropped, stats) as the eager loop leaves them."""
        t = self.t
        self._enter_phase(self._phase_key(tst), batch)
        self._phase_open = True
        if self.capturing:
            self._check_storage()
        if self.acc is not None:
            for v in self.acc.values():
                v.zero_()
        if self.bag:  # the bag loss's divisor (Trainer._micro_loss_tst: a = v w / w_total), in device memory, in the form this device divides
            self._set_divisor(tst[2]["w_total"])
        else:
            # the loss's normalisers of the fused head (Trainer._micro_loss_fused), computed as there in double, rounded
            # once to float32 as the kernels round the Python float, in device memory for the graph
            a = 1.0 / norms["n_valid"]
            self._copy(self.buf["scales"], torch.tensor([a, z * a], dtype=torch.float32))
        # micro_share over the step's rows on every rank (data parallelism, ced/train/data_parallel.py D2; one process: B)
        dp = getattr(t, "dp", None)
        rows_all = batch.tokens.shape[0] * (1 if dp is None else dp.world)
        self.norms = {**norms, "micro_share": self.rows / rows_all, "a_scale": self.buf["scales"][0],
                      "b_scale": self.buf["scales"][1]}
        stats: dict = {}
        # module docstring, "Streams": the live AccumulateGrad nodes before the warm-up step and before the capture (then
        # the labels stop: every later forward is a replay or an eager micro-batch on the capture stream)
        if self.steps in (1, WARMUP_STEPS):
            self._check_accumulators("the warm-up step's start" if self.steps == 1 else "the capture step's start")
            if self.steps == WARMUP_STEPS:
                self._stop_labels()
        replaying = self.capturing
        for m0, (mb, _T) in enumerate(plan_mb):
            monitored = collect and m0 == 0  # its statistics are the step's (Trainer.train_step: the first micro-batch's)
            if not replaying:  # the warm-up step (and every step on the CPU): the micro-batch eagerly, on the capture stream
                with self._on_stream():
                    if self._fill(batch, mb, valid_loss, staged):
                        self.counts["eager"] += 1
                        got = self._micro(self.buf)
                    else:
                        got = self._miss(batch, mb, valid_loss)
            elif not self._fill(batch, mb, valid_loss, staged):  # the copies on the replay's stream; a miss: the capture stream
                with self._on_stream():
                    got = self._miss(batch, mb, valid_loss)
            else:
                if self.graph is None:
                    self._capture_micro()
                self.graph.replay()
                self.counts["replays"] += 1
                # the replay's statistics, copied out before the next replay rewrites them (stream order)
                got = {k: v.clone() for k, v in self.graph_stats.items()} if monitored else {}
            if monitored:
                stats = got
        acc = self.acc
        return acc["loss"], {k: v for k, v in acc.items() if k not in ("loss", "n_dropped")}, acc["n_dropped"], stats

    # --- the optimizer ("step") ---------------------------------------------------------------------------------------
    def _opt(self):
        t = self.t
        z1 = getattr(t, "_z1", None)
        if z1 is None:
            norm = torch.nn.utils.clip_grad_norm_(t.model.parameters(), t.cfg.optim.grad_clip)
        else:  # the sharded optimizer (ced/train/zero1.py): the total norm from the gathered vector, the owned gradients
            norm = z1.clip_from_norms(z1.clip_set, t.cfg.optim.grad_clip)  # scaled; train_step gathered the norms
        return norm, t.optimizer.muon_orthogonalise()  # (the other tensors' .grad is None here: the owned matrices)

    def _opt_key(self) -> tuple:
        """The host values and addresses the optimizer graph holds (2026-10-17): the clip's max norm, each Muon group's
        momentum, Nesterov flag and Polar Express step count (Python values read at the capture), and the Muon
        momentum buffers it updates in place. None of them changes in a run today (no schedule sets them; the state is
        loaded before the process's first step); a replay after one changed would silently run the captured ones."""
        opt = self.t.optimizer
        groups = tuple((g["momentum"], g["nesterov"], g["ns_steps"]) for g in opt.param_groups if g["kind"] == "muon")
        bufs = tuple(opt.state[p]["momentum_buffer"].data_ptr() if "momentum_buffer" in opt.state.get(p, {}) else None
                     for g in opt.param_groups if g["kind"] == "muon" for p in g["params"] if p.grad is not None)
        z1 = getattr(self.t, "_z1", None)
        if z1 is not None:  # the sharded optimizer: the gathered norms' buffer the graph reads, and the clip set's tensors
            return (self.t.cfg.optim.grad_clip, groups, bufs, z1._norm_vec.data_ptr(), z1._clip_key)
        return self.t.cfg.optim.grad_clip, groups, bufs

    def _check_opt(self) -> None:
        """Refuse to replay the optimizer graph when anything _opt_key names differs from its capture."""
        if self._opt_key() != self.opt_key:
            raise RuntimeError("cuda_graphs 'step': the clip's max norm, a Muon group's momentum / nesterov / ns_steps or a Muon "
                               "momentum buffer changed after the optimizer graph was captured")

    def clip_and_orthogonalise(self):
        """cuda_graphs "step": the gradient clip and Muon's momentum and Polar Express (eager in the warm-up, then one
        replay). Returns the gradient norm (a device tensor, as clip_grad_norm_ returns it)."""
        if not self.capturing:
            with self._on_stream():
                norm, self.orth = self._opt()
            return norm
        if self.opt_graph is None:
            g = torch.cuda.CUDAGraph()
            try:  # in the micro-batch graph's pool: replayed after it, never at the same time
                with self.t._prefetch_quiet(), torch.cuda.graph(g, pool=self.graph.pool() if self.graph is not None else None,
                                                                stream=self.stream):
                    self.opt_norm, self.orth = self._opt()
            except Exception as e:
                raise RuntimeError(f"cuda_graphs: the capture of the optimizer graph failed ({type(e).__name__}: {e})") from e
            self.opt_graph = g
            self.opt_key = self._opt_key()
            self.counts["captures"] += 1
        self._check_opt()
        self.opt_graph.replay()
        return self.opt_norm

    def optimizer_rest(self) -> None:
        """cuda_graphs "step": the rest of MuonWithAdamW.step, eagerly: Muon's LR step from the factors, then AdamW."""
        self.t.optimizer.muon_apply(self.orth)
        self.t.optimizer.adamw_step()

    def metrics(self) -> dict:
        """The step's "cuda_graphs" metrics: the process's counts so far, and the phase the step ran in (NEXT_TOKEN, or ("bag", s)) with
        the graph-path steps that ended in it so far."""
        return {"mode": self.mode, "captured": self.capturing, **self.counts, "phase": None if self.phase is None else list(self.phase),
                "phase_steps": self.phase_steps}

