"""Optimizers: AdamW, and Muon for hidden weight matrices with AdamW for everything else.

Parameter roles ("AdamW for embeddings, LM head, norms, and GDN scalars, convolutions, and gates"):
- Muon: 2-D weight matrices of hidden layers (attention and GDN projections, FFN, memory KV,
  loop-boundary projections; in CED v2 also W_mem and the decoder's query-only attention), except
  the GDN gate projections in_proj_a / in_proj_b, which produce per-head scalars and are "gates" in
  the sense above.
- AdamW: the token embedding (= the tied LM head in v1), the untied output embedding (v2: `out_embed`,
  the LM head of every head), every norm gain, GDN A_log / dt_bias and the depthwise conv, the Parcae
  injection, hyper-connection parameters, loop embeddings, the second-pass control's pass embedding,
  routers, the attention temperature gains (temp_w, one per query head; AttentionConfig.temperature
  "learned" / "ssmax"), and ASEntmax's scale heads beta_proj / gamma_proj (AttentionConfig.normalizer
  "asentmax": Linear(d_model, n_q_heads) with bias, per-head scale heads, not hidden-state maps), weights
  and biases.

Weight decay: none on the embeddings (input and output), GDN A_log / dt_bias, Parcae and hyper-connection
parameters, the attention temperature gains and ASEntmax's scale heads (NO_DECAY_SUFFIXES;: the fixed scale
caps the achievable logits, so the parameter that lifts the cap is not pulled back; beta_proj / gamma_proj: the
same reasoning, and the authors' synthetic runs use no weight decay); norm gains do get decay (they are
zero-centred, so decay pulls the gain to 1), and so do the loop and pass embeddings (zero-initialised: decay
pulls them to their initial value).

Pass 2's own tensors (LoopConfig.pass2_gate, the loop experiment; ced/model/loop.py PASS2_PARAMETERS): AdamW, as
the spec writes ("New scalars (alpha, loop-index embedding) are on AdamW with their own LR"), in two groups of
their own that come AFTER the groups above, so that the other groups hold the same tensors in the same order as the
single-pass model's optimizer. Their LR is the run's LR at that step times OptimConfig.pass2_lr_scale (times
adamw_lr_scale under Muon), 1.0 by default. Weight decay follows the rules above: none on the gate and on pass 2's
hyper-connection scalars (hyper-connection parameters), the run's on pass 2's loop-index embedding (a loop
embedding). A model without those tensors gets exactly the groups it always had. A third pass's own tensors
(LoopConfig.segment_passes with a 3; ced/model/loop.py PASS3_PARAMETERS) are treated as pass 2's in every respect:
the same two groups (so a model without a third pass has the groups it had), the same LR multiplier, the same
weight decay rule.

The per-layer residual scalars (EncoderConfig.layer_scalars "skip_x0"): encoder.skip_gain and encoder.x0_gain, one
(n_layers,) vector each. One-dimensional, so never on Muon: AdamW at the run's LR (times adamw_lr_scale under Muon,
1.0 in every 60M config) with the run's betas and schedule, in the group WITHOUT weight decay (NO_DECAY_SUFFIXES), the
rule of the gates and the hyper-connection weights: they weight residual paths, and a screen of whether they move is
not to pull them back. (Left out of the list they would get the run's weight decay, 0.165 at 60M: stored zero-centred,
that would pull a to 1 and b to 0, as it pulls a norm gain to 1.) They are inside the global gradient clip like every
parameter. A model without them gets exactly the groups it always had. The forms "skip_x0_span" and "x0"
(ced/model/encoder.py layer_scalar_terms) keep the two names, so nothing here names them: the span form's stored u
(encoder.skip_gain) and both forms' encoder.x0_gain are in that same group; "x0" has no encoder.skip_gain at all.
Weight decay on u would pull the cumulative gains to 0, every a to 1: the same reason. The form "skip" has
encoder.skip_gain alone, in that same group.

OptimConfig.layer_scalar_a_lr_scale. "The scalar LR" here is the LR of the group above: the run's LR at that step (the
run's warmup, plateau and decay), times adamw_lr_scale under Muon. 1.0 (the default): nothing changes, every group holds
the tensors it held, in the same order. Any other value: encoder.skip_gain (SCALAR_A: the stored a - 1, or the span
form's u) leaves that group for one of its own, the LAST group (after pass 2's and after a "muon_no_decay" B), so every
other group holds what it held without it: AdamW, the run's betas and eps, weight decay 0, and an LR of the run's LR
times adamw_lr_scale (under Muon) times this key; the group is marked "layer_scalar_a". encoder.x0_gain stays where it
was, at the scalar LR. A value other than 1 on a model without encoder.skip_gain is refused, as is a value that is not a
finite number above 0. Exact arithmetic at 60M (configs/v2/control-a-60m.yaml: LR 0.003, 6,487 steps whose LRs sum to
16.602, see the Canon paragraph below): at 0.01 the peak LR of a is 0.00003 and the summed LR 0.166. AdamW moves a stored
value by about its LR a step at most, so over the whole run a can leave 1 by about 0.17 at most, whatever the gradient
says: the key slows a AND bounds it.

The Canon taps (EncoderConfig.canon_kernel; ced/model/canon.py): encoder.layers.<i>.canon.conv_weight, (d_model, K).
Named as the GDN layers' depthwise convolution and treated the same way, whatever their start (EncoderConfig.canon_init):
AdamW, never Muon (is_muon_param: "conv_weight"), at the run's LR, in the group WITH the run's weight decay. The decay is
decoupled: a step multiplies every tap by 1 - lr x weight_decay (MuonWithAdamW.step; torch.optim.AdamW the same), so it
pulls the taps toward zero. With canon_init "zeros" that is toward their start. With "uniform" (taps in +-0.5 at K = 4)
it is AWAY from their start, and the optimizer does not keep that start. The size at 60M, exact arithmetic on
configs/v2/control-a-60m.yaml (LR 0.003, weight decay 0.165, warmup 610 steps, 6,487 steps, linear decay over the last
20%; tests/test_canon.py redoes it):
- a step at the peak LR multiplies a tap by 1 - 0.000495: a pull of 0.00025 on a tap of 0.5 and of 0.00014 on a tap at the
  start's RMS of 0.289, against an Adam step of about the LR, 0.003 (the pull is 8% and 5% of it);
- the decay alone halves a tap in 1,400 steps at the peak LR; over the whole schedule (the 6,487 LRs sum to 16.602) it
  leaves x 0.0646 of a tap that no gradient holds: x 0.86 by step 610, x 0.52 by 1,622, x 0.23 by 3,244;
- over the first 40 steps (the GPU smoke's) the LRs sum to 0.00403 and the decay takes x 0.99933 (0.9993348).
So the size of the taps at the end of a uniform run is what the loss gradient holds against the decay, not the start.
(The source decays its Canon weights too: one AdamW group over every parameter, default 0.1.)

The injection of a looped middle (LoopConfig.recur_layers with injection "parcae"; ced/model/loop.py RecurInjection):
encoder.recur_injection.A_log, .dt_bias and .B fall under the rules above by their names: no weight decay (".A_log",
".dt_bias", "injection.B"), and B, a d x d matrix, on
AdamW, not Muon (is_muon_param: "injection", the rule of every loop-boundary parameter). The norm on the prelude's
output (encoder.recur_norm.weight) is a norm gain like any other: AdamW with the run's weight decay.

OptimConfig.recur_b, under name "muon" only. "adamw" (the default) is the paragraph above: the groups every config
had before the key, tensor for tensor and in the same order. "muon": the dense B
(RECUR_B, d x d) is one more tensor of the Muon group, where its place in the model's parameter order puts it (after
the layers' matrices): the Muon update, momentum, LR and LR schedule of the layers' matrices, and their weight decay
too (MuonWithAdamW.step: B <- (1 - lr x weight_decay) B - lr x 0.2 sqrt(d) x O; the decay pulls B towards zero, and B
starts at the identity). "muon_no_decay": the same Muon update in a group of its own, the last one, with weight decay
0. A_log and dt_bias stay where they were in every case. A value other than "adamw" is refused under name "adamw" and
on a model that has no dense B.

The scalar input mix of a looped middle (LoopConfig.recur_layers with injection "mix"; ced/model/encoder.py
recur_input_mix; the late switch-on of 2026-10-13): encoder.recur_mix (RECUR_MIX), the one scalar w of
z = w e + (1 - w) h. It is a later pass's own tensor in every respect that matters here (only the passes after the
first read it; it gets no gradient before a loop switch step; it is the one new tensor when a plain control's
checkpoint grows), so it is treated as pass 2's own tensors are (is_grown_param): AdamW in a group of its own that
comes AFTER the groups above, marked "pass2", so that the other groups hold the plain control's tensors in the plain
control's order and the trunk's optimizer state is taken index by index. NO weight decay (".recur_mix" in
NO_DECAY_SUFFIXES: a decay would pull w towards 0, which is the loop without injection, away from its start). Its LR
is the run's LR at that step times adamw_lr_scale (under Muon) times OptimConfig.pass2_lr_scale, 1.0 by default: the
LR every AdamW scalar of the model has (the norm gains, the GDN scalars, the temperature gains), with the run's
warmup, plateau and decay. Nothing is tuned for it. AdamW moves a stored value by about its LR a step at most, so
after a switch w can leave its start by at most the sum of the remaining LRs. A model without it gets exactly the
groups it always had.

Block Attention Residuals (EncoderConfig.attn_res "block"; ced/model/attn_res.py): per reader a pseudo-query
(encoder.layers.<i>.attn_res.pre_mixer.query, .pre_mlp.query and, in the last layer, .out.query; (d_model,), zeros
at the start) and its key norm's gain (...key_norm.weight). Nothing here names them: one-dimensional, so never on Muon; AdamW
at the run's LR (times adamw_lr_scale under Muon) in the group WITH the run's weight decay. For a query the decay pulls
toward zero, which is its start (equal weights over the sources); for the gain toward 1, as for every norm gain. The
paper (arXiv 2603.15031) and its repository state no optimizer group, LR or weight decay for either; its pseudocode
holds the query as a Linear weight. A model without the key gets exactly the groups it always had.
R1-entry (EncoderConfig.attn_res "block_entry"): the same rule for its readers' tensors
(encoder.layers.<i>.attn_res.entry.query and .out.query, and their key_norm.weight): one-dimensional, AdamW at the
run's LR with the run's weight decay. Nothing here names them either.

The architecture arms' depth mixers (2026-10-16; ced/model/ldm.py, ced/model/attn_res.py). Every tensor under
".attn_res." is on AdamW at the run's LR (times adamw_lr_scale under Muon), never Muon, two-dimensional or not
(is_muon_param): the address networks of Linear Depth Memory (EncoderConfig.attn_res "ldm") and of R1-entry's
token-dependent queries
(attn_res_query "token") are narrow matrices (hidden x d_model, G K x hidden, G x hidden; d_model x hidden for a query),
the depth mixer's controller as R1's pseudo-queries are, and the paper trains them with AdamW. Weight decay, the paper's
training recipe (github.com/davidgonmar/depth-memory at 9a6865feeb43, paper-experiments/pretrain/train.py
configure_weight_decay and LDM_CONTROL_NO_DECAY: "the forgetting logits and the read gains are not decayed. Decay would
pull the gate biases towards a gate of one, so they are excluded too"; everything else is decayed there, the address
networks' first and address-head biases included, nn/ldm/parameters/mixer.py: "Weights and address-network biases go to
the decay bank and the gate bias to the no-decay bank, the paper recipe's partition"): none on LDM's read gains
(".read_gain"), its forgetting logits (".lam_logit") and its gate heads' biases (".net.gate.bias"); the run's on every
other new tensor, the address networks' weights and their ".net.inp.bias" / ".net.head.bias" included. So the token
query's static part (".net.head.bias", R1-entry's w) is decayed as R1-entry's pseudo-query is, and the intermediate arm
differs from R1-entry only by the token-dependent part. The input norms' gains follow this codebase's rule (zero-centred,
decayed toward a gain of 1, their start), where the paper's recipe leaves its norm gains undecayed. (2026-10-16: the
build had put ".net.inp.bias" and ".net.head.bias" in the no-decay class.) A model without the arms gets exactly the
groups it always had: no tensor of R1 or R1-entry is two-dimensional or ends with these names.

The GDN-side arms (EncoderConfig.gdn_write_mix "A" and gdn_ccr_spacing "CCR"; ced/model/gdn_arms.py; section 13). R1-entry's
own mixing parameters are its readers' tensors (encoder.layers.<i>.attn_res.entry.query.out.query and their key_norm.weight):
one-dimensional, so never on Muon; AdamW at the run's LR (times adamw_lr_scale under Muon) with the run's betas, eps and
schedule, in the AdamW group WITH the run's weight decay (the paragraph on Block Attention Residuals above: nothing here names
them). The arms' tensors get exactly that, by the same rule (nothing here names them either): the write mix's write_lambda
((1,) or (H_k + H_v), zeros) and its write_query.query and write_query.key_norm.weight (a DepthQuery, R1-entry's reader), and
CCR's ccr_gate ((H_v,), zeros). Under name "muon" that is the second group (kind "adamw", lr_scale adamw_lr_scale,
weight_decay optim.weight_decay); under "adamw" the first (param_groups); under the matrix optimizers matrix_groups'
"adam_decay" group. The decay pulls each toward zero, which is its start (lambda 0 and the gates 0: the arms off; the write
query 0: uniform weights) and the key-norm gain toward 1. A model without the arms gets exactly the groups it always had: no
other tensor of any model ends with those two names (tests/test_arch_arms_decay.py). OptimConfig.zero_gate_decay "none":
write_lambda and ccr_gate (ZERO_GATE_SUFFIXES) go to the AdamW group WITHOUT weight decay at the same lr_scale; the write
query and its key-norm gain stay where put them; "decay", the default, is the rule. OptimConfig.mix_query_decay
(MIX_QUERY_DECAY): "decay", the default, is the rule for the depth-mix pseudo-queries, every group as before. "none_qmix": the
wsl variant qmix's read-mix pseudo-query
(gdn_arms.qmix_query.query) goes to the AdamW group WITHOUT weight decay at the same lr_scale, and the write query keeps
wsl-nowd's decay. "none_write_qmix": the write query (gdn_arms.write_query.query) goes there too. The key-norm gains and
R1-entry's own readers keep their decay under every value; a value other than "decay" on a model with none of the named
queries is refused (check_mix_query_decay). CCR v2's query translators (model.encoder.gdn_ccr_translator "key_head"):
the stored D = M - I of every reader, (H_k, d_k, d_k), never on Muon, in ONE AdamW group of its own, the last
(take_ccr_translator, marked "ccr_translator"), at the readers' LR (the run's LR x adamw_lr_scale; the run's LR under
"adamw") with the run's betas, eps and schedule. optim.ccr_translator_decay "none" (the default): weight decay 0.
"identity": the run's weight decay on D, which is decay toward M = I: supported without an optimizer feature, because D
is what is stored (no decay target is built here; Adam's update is the same for D and M). A model without translators:
no group is added.

R3, the depth mix at the loop boundary (LoopConfig.recur_layers with injection "depth_mix"; ced/model/recur_depth.py):
ONE reader, encoder.recur_depth.query and encoder.recur_depth.key_norm.weight ((d_model,) each, zeros at the start).
It is R1's reader (attn_res.DepthQuery) and falls under R1's rule above with no line of its own: one-dimensional, so
never on Muon; AdamW at the run's LR (times adamw_lr_scale under Muon) in the group WITH the run's weight decay (the
decay pulls the query toward zero, its start, where every candidate has equal weight, and the gain toward 1). They
are not a later pass's own tensors (every pass reads them, the first too; is_grown_param is False), so they sit in the
ordinary AdamW group, after the layers' tensors in the model's parameter order, and optim.pass2_lr_scale does not
reach them. A model without the injection gets exactly the groups it always had.

The Copy-Paste head (ModelConfig.cp_head; ced/model/cp_head.py). R1-entry's treatment of its small new projections (the
token-dependent queries' address networks, narrow matrices, and its readers' vectors; the paragraphs above, and the
arms-200m-prereq merge 009464fd, "AdamW at the run's LR with weight decay 0.165" at 60M): AdamW, never Muon, at the
run's LR times adamw_lr_scale (1.0 at 60M) with the run's betas, eps and schedule, in the group WITH the run's weight
decay. So cp_head.w_q and cp_head.w_k ((d_c, d_model) matrices: is_muon_param refuses "cp_head.") and cp_head.k_sink
((d_c,)) are in the AdamW decay group ("adam_decay" under the matrix optimizers). The <copy> row, cp_head.w_copy
((d_model,)), is in the LM head's group: the head (out_embed.weight, or embed.weight when tied) is in the AdamW group
WITHOUT weight decay ("embed.weight" in NO_DECAY_SUFFIXES), so w_copy is there too (its name is in NO_DECAY_SUFFIXES),
at the same LR. The four come last in the model's parameter order, so every group holds the base tensors in the base
order, followed by the head's. A model without the head gets exactly the groups it always had. With
model.cp_head.qk_norm (the reference's QK-norm), the query gain cp_head.qk_gain ((1,), stored zero-centred: the gain is
1 + qk_gain) is the pointer's temperature: after the norm the pointer's scores are at most sqrt(d_c) x the gain in
size, and the gain is the only tensor that lifts that cap. It trains WITHOUT weight decay (its name is in
NO_DECAY_SUFFIXES): AdamW at the run's LR (times adamw_lr_scale) with the run's betas and eps, in the group without
decay, one-dimensional so never Muon. That is the reference's rule for it (training.py L93-95: "copy_qk_gain", Adam,
"wd_mul": 0.0) and the model's rule for the attention temperature gains (temp_w). It comes after w_copy in the
parameter order, so it is the no-decay group's last tensor. The betas are the run's (the reference's are (0.9, 0.99)):
the spec keeps the run's AdamW for the head's tensors (rows 9 and 10). The trainer logs the gain every log_every steps
(cp_qk_gain), as it logs temp_w.
"""

from __future__ import annotations

import math

import torch

from ced.config import OptimConfig
from ced.model.loop import LATER_PASS_PARAMETERS

NO_DECAY_SUFFIXES = (
    "embed.weight",
    ".A_log",
    ".dt_bias",
    "injection.B",
    "hyper_connections.read_weights",
    "hyper_connections.mix_weights",
    "hyper_connections.write_weights",
    "hyper_connections.bias_pre",
    "hyper_connections.bias_post",
    "hyper_connections.bias_res",
    "hyper_connections.scale_pre",
    "hyper_connections.scale_post",
    "hyper_connections.scale_res",
    # a gated second pass (LoopConfig.pass2_gate): pass 2's hyper-connection scalars, or its gate alpha
    "hyper_connections.pass2_read",
    "hyper_connections.pass2_mix",
    "hyper_connections.pass2_write",
    "hyper_connections.pass2_gate",
    # a third pass (LoopConfig.segment_passes with a 3): the same tensors of pass 3
    "hyper_connections.pass3_read",
    "hyper_connections.pass3_mix",
    "hyper_connections.pass3_write",
    "hyper_connections.pass3_gate",
    ".temp_w",  # the attention temperature gains (AttentionConfig.temperature "learned" / "ssmax")
    # the per-layer residual scalars (EncoderConfig.layer_scalars): encoder.skip_gain and encoder.x0_gain
    ".skip_gain",
    ".x0_gain",
    ".recur_mix",  # the scalar input mix of a looped middle (LoopConfig.injection "mix"): encoder.recur_mix
    # ASEntmax's scale heads (AttentionConfig.normalizer "asentmax")
    ".beta_proj.weight",
    ".beta_proj.bias",
    ".gamma_proj.weight",
    ".gamma_proj.bias",
    # (the GDN-side arms' tensors are not named here: R1-entry's rule, module docstring)
    # the architecture arms' depth mixers (2026-10-16; ced/model/ldm.py): LDM's read gains, forgetting logits and gate
    # heads' biases, the paper's training recipe's no-decay class (paper-experiments/pretrain/train.py
    # LDM_CONTROL_NO_DECAY); the address networks' other biases are decayed there and here (module docstring)
    ".read_gain",
    ".lam_logit",
    ".net.gate.bias",
    # the Copy-Paste head's <copy> row (ced/model/cp_head.py): the LM head's group, which has no weight decay
    "cp_head.w_copy",
    # the Copy-Paste head's QK-norm gain (model.cp_head.qk_norm): the pointer's temperature, not pulled back (module
    # docstring; the reference's wd_mul 0.0, and the rule of temp_w above)
    "cp_head.qk_gain",
)

# The zero-initialised gates whose zero means "off" (OptimConfig.zero_gate_decay; 2026-10-17): the GDN-side arms' CCR gate ((H_v,),
# zeros: no cross-read) and the write mix's write lambda ((1,) or (H_k + H_v), zeros: no depth mix). Under zero_gate_decay "none" they
# join NO_DECAY_SUFFIXES' group; under "decay" (the default) they stay in the decay group the rule (module docstring). the write mix's
# write_query (a DepthQuery: zero means uniform weights, not off) and its key-norm gain are not gates and keep their group either way. A
# future arm's zero-initialised gate or mixing scalar whose zero means "off" ("any future zero-init gate") is named here by its build,
# and the arm's runs set optim.zero_gate_decay=none.
# Canon's taps (conv_weight, a (d_model, K) filter, zeros under canon_init "zeros": h' = h), the loop and pass embeddings and the zero-initialised output matrices
# are decayed and not named here ("Matrices keep their groups unchanged"). The wsl variant "qmix" (EncoderConfig.gdn_write_variant) adds
# the q rows' lambda, qmix_lambda ((1,), zeros: the q rows read x), named here by its build as the rule above says; its read mix's
# DepthQuery (qmix_query: zero is uniform weights, not off) keeps the write query's group under either value of this key (the
# pseudo-queries' own key is optim.mix_query_decay, MIX_QUERY_DECAY below). No other tensor of any model ends with that name, so every
# model without the variant keeps its groups. The Copy-Paste head's trunk copy (model.cp_head.trunk_copy; ced/model/cp_trunk.py) adds
# its mixing logits over the global heads, cp_head.trunk_logits ((H_total,), zeros: uniform weights), named here by its build as the
# rule above says ("Zero-initialised gates and mixing weights get no weight decay"): under zero_gate_decay "none" (the trunk-copy arms'
# setting) the AdamW group without decay. No other tensor of any model has that name, so every model without the key keeps its groups.
ZERO_GATE_SUFFIXES = (".gdn_arms.ccr_gate", ".gdn_arms.write_lambda", ".gdn_arms.qmix_lambda", "cp_head.trunk_logits")
# EDA's erase direction (GDNConfig.erase; ced/model/gdn_erase.py) adds the erase gate's scale, erase.scale ((H_v,), zeros: gamma = 0,
# the erase off), named here by its build as the rule above says; its arms set optim.zero_gate_decay=none. No other tensor of any
# model has that name, so every model without the key keeps its groups.
ZERO_GATE_SUFFIXES += (".erase.scale", ".dprod.beta_const")  # + the kvmean arm's learned beta constant (ced/model/gdn_dprod_sparse.py, (H_v,) zeros: beta2 = beta_max / 2), named by its build
ZERO_GATE_DECAY_MODES = ("decay", "none")

# Arm A's depth-mix pseudo-queries (OptimConfig.mix_query_decay).
# - "decay" (the default): R1-entry's readers' rule for both queries, the groups of every run before the key;
# - "none_qmix": only the wsl variant qmix's own read-mix query loses its decay, so qmix's new tensors (lambda_q under
#   zero_gate_decay "none", and this query) carry no decay and every tensor wsl-nowd has keeps wsl-nowd's group;
# - "none_write_qmix": the write mix's write query too (wsl-nowd's write query HAS decay, so a run under this value differs
#   from wsl-nowd in its decay as well as in its rows). The pseudo-query (".query", (d_model,), zeros: uniform weights over the
#   sources) only. Its key-norm gain (".key_norm.weight", zero-centred) keeps the norm gains' rule under every value: the mix's
#   logits are sum_c w_c (1 + g_c) RMSNorm(b)_c, so with w free the gain's pull toward 1 caps nothing w cannot carry, and the
#   gain is a norm gain like every other. R1-entry's own readers (".attn_res.") keep their decay: they are the control's
#   tensors. A value other than "decay" on a model that has none of its tensors is refused (check_mix_query_decay), as
#   optim.ccr_translator_decay is.
MIX_QUERY_WRITE = ".gdn_arms.write_query.query"
MIX_QUERY_QMIX = ".gdn_arms.qmix_query.query"
MIX_QUERY_DECAY = {"decay": (), "none_qmix": (MIX_QUERY_QMIX,), "none_write_qmix": (MIX_QUERY_WRITE, MIX_QUERY_QMIX)}

# CCR v2's query translators (model.encoder.gdn_ccr_translator "key_head"; ced/model/gdn_arms.py TRANSLATOR): the stored D =
# M - I, (H_k, d_k, d_k) a reader, zeros. Always one AdamW group of their own, the last (take_ccr_translator), at the
# readers' LR: R1-entry's readers' (the run's LR x adamw_lr_scale under "muon" and the matrix optimizers, the run's LR under
# "adamw"), with the run's betas, eps and schedule. optim.ccr_translator_decay "none" (the default): weight decay 0.
# "identity": the run's weight decay; on D that is decay toward M = I, the start (no new optimizer feature: the decay target
# is the identity because D, not M, is stored; Adam's update is the same for D and M).
CCR_TRANSLATOR_SUFFIX = ".gdn_arms.ccr_translator_delta"
CCR_TRANSLATOR_DECAY_MODES = ("none", "identity")


def take_ccr_translator(groups: list[dict], model: torch.nn.Module, o: OptimConfig, lr_scale: float, **extra) -> list[dict]:
    """`groups` with CCR v2's translators (CCR_TRANSLATOR_SUFFIX) taken out of the groups that hold them and put in one
    AdamW group of their own at the end (marked "ccr_translator"; weight decay by optim.ccr_translator_decay; the LR
    multiplier lr_scale). Every other tensor stays in its group, in its place. A model without translators: `groups`
    as they are (and optim.ccr_translator_decay must be "none": a key that would do nothing is refused)."""
    mode = o.ccr_translator_decay
    if mode not in CCR_TRANSLATOR_DECAY_MODES:
        raise ValueError(f"optim.ccr_translator_decay {mode!r}: one of {CCR_TRANSLATOR_DECAY_MODES}")
    mats = [p for n, p in model.named_parameters() if n.endswith(CCR_TRANSLATOR_SUFFIX)]
    if not mats:
        if mode != "none":
            raise ValueError(f"optim.ccr_translator_decay {mode!r} is read for CCR v2's query translators "
                             "(model.encoder.gdn_ccr_translator 'key_head'); this model has none: leave it 'none'")
        return groups
    ids = {id(p) for p in mats}
    assert sum(1 for g in groups for p in g["params"] if id(p) in ids) == len(mats), "a translator is in no group, or in two"
    for g in groups:
        g["params"] = [p for p in g["params"] if id(p) not in ids]
    wd = 0.0 if mode == "none" else o.weight_decay
    return groups + [{"params": mats, "weight_decay": wd, "lr_scale": lr_scale, "ccr_translator": True, **extra}]


def no_decay_suffixes(o: OptimConfig | None) -> tuple:
    """The name suffixes of the groups WITHOUT weight decay: NO_DECAY_SUFFIXES, plus ZERO_GATE_SUFFIXES under
    optim.zero_gate_decay "none", plus optim.mix_query_decay's pseudo-queries (MIX_QUERY_DECAY) under a value other than
    "decay". With o None or both defaults, NO_DECAY_SUFFIXES itself (the groups of before)."""
    mode = "decay" if o is None else o.zero_gate_decay
    if mode not in ZERO_GATE_DECAY_MODES:
        raise ValueError(f"optim.zero_gate_decay {mode!r}: one of {ZERO_GATE_DECAY_MODES}")
    suffixes = NO_DECAY_SUFFIXES + ZERO_GATE_SUFFIXES if mode == "none" else NO_DECAY_SUFFIXES
    queries = "decay" if o is None else o.mix_query_decay
    if queries not in MIX_QUERY_DECAY:
        raise ValueError(f"optim.mix_query_decay {queries!r}: one of {tuple(MIX_QUERY_DECAY)}")
    return suffixes + MIX_QUERY_DECAY[queries] if queries != "decay" else suffixes


def check_mix_query_decay(model: torch.nn.Module, o: OptimConfig) -> None:
    """Refuse optim.mix_query_decay other than "decay" on a model that has none of the pseudo-queries it names (a key that
    would do nothing): "none_qmix" needs the wsl variant qmix's read-mix queries, "none_write_qmix" the write mix's write
    queries (with or without qmix's)."""
    mode = o.mix_query_decay
    if mode not in MIX_QUERY_DECAY:
        raise ValueError(f"optim.mix_query_decay {mode!r}: one of {tuple(MIX_QUERY_DECAY)}")
    named = MIX_QUERY_DECAY[mode]
    if named and not any(n.endswith(named) for n, _ in model.named_parameters()):
        raise ValueError(f"optim.mix_query_decay {mode!r} is read for arm A's depth-mix pseudo-queries {named} "
                         "(model.encoder.gdn_write_mix 'layer' or 'head'; 'none_qmix' with gdn_write_variant 'qmix' or 'kqmix'); "
                         "this model has none: leave it 'decay'")


# 2-D weights that stay on AdamW under Muon (besides the embedding).
ADAMW_MATRIX_SUFFIXES = ("in_proj_a.weight", "in_proj_b.weight", "beta_proj.weight", "gamma_proj.weight")
# Per-channel decay (GDNConfig.kda.decay "channel"; ced/model/gdn_kda.py): the low-rank decay projection that replaces in_proj_a is
# the gate's projection, so its two matrices take in_proj_a's place: AdamW, never Muon, in the group WITH the run's weight decay
# (in_proj_a.weight's group). Its per-channel dt_bias (the floor's b too) is ".dt_bias": no weight decay. No model without the key
# has a tensor of these names.
ADAMW_MATRIX_SUFFIXES += (".kda.f_down.weight", ".kda.f_up.weight")
# EDA's erase direction (GDNConfig.erase.mode "eda"; ced/model/gdn_erase.py): the erase address's low-rank pair and the erase gate's
# projection are gate projections like in_proj_a / in_proj_b: AdamW, never Muon, in the group WITH the run's weight decay ("Matrices
# keep their groups unchanged"). Its zero-initialised scale is named in ZERO_GATE_SUFFIXES below. No model without the key has a
# tensor of these names.
ADAMW_MATRIX_SUFFIXES += (".erase.e_down.weight", ".erase.e_up.weight", ".erase.in_proj_gamma.weight")
# Gated DeltaProduct's low-rank second key (GDNConfig.dprod.key "lowrank", arm C1; ced/model/gdn_dprod_c.py): the correction's pair k_down (d -> r H_k) and
# k_up ((H_k d_k, r)) is EDA's e_down / e_up in shape and role (a thin projection that makes an address): AdamW, never Muon, in the group WITH the run's
# weight decay. Its beta projection dprod.in_proj_b.weight is in_proj_b.weight's above. No model without the key has a tensor of these names.
ADAMW_MATRIX_SUFFIXES += (".dprod.k_down.weight", ".dprod.k_up.weight")

# The residual scalars' a as stored (EncoderConfig.layer_scalars; ced/model/encoder.py): the one tensor
# OptimConfig.layer_scalar_a_lr_scale moves to a group of its own (module docstring).
SCALAR_A = "encoder.skip_gain"

# The dense matrix B of a looped middle's injection (ced/model/loop.py RecurInjection with b_form "dense"), by its name
# in the control, and the values of OptimConfig.recur_b (module docstring).
RECUR_B = "encoder.recur_injection.B"
RECUR_B_MODES = ("adamw", "muon", "muon_no_decay")


# The scalar w of a looped middle's input mix (LoopConfig.recur_layers with injection "mix"; ced/model/encoder.py
# recur_input_mix), by its name in the control (module docstring).
RECUR_MIX = "encoder.recur_mix"


def is_pass2_param(name: str) -> bool:
    """One of the later passes' own tensors of a gated CED v2: pass 2's (ced/model/loop.py PASS2_PARAMETERS) or,
    with a third pass, pass 3's (PASS3_PARAMETERS)."""
    return name in LATER_PASS_PARAMETERS


def is_grown_param(name: str) -> bool:
    """A tensor that only the passes after the first read, new when a one-pass checkpoint grows: a gated model's
    later-pass tensors (is_pass2_param) or the scalar input mix of a looped middle (RECUR_MIX). These are the tensors
    of the groups pass2_groups builds. No model has both kinds (ced/model/control.py refuses the two loops together)."""
    return is_pass2_param(name) or name == RECUR_MIX


def pass2_groups(model: torch.nn.Module, weight_decay: float, lr_scale: float, **extra) -> list[dict]:
    """The AdamW groups of pass 2's own tensors (module docstring): the loop-index embedding of pass 2 with the run's
    weight decay, the gate / hyper-connection scalars without; [] for a model that has none. lr_scale is the groups'
    multiplier on the run's LR (the trainer sets group["lr"] = lr * group["lr_scale"] every step).
    The scalar input mix of a looped middle (RECUR_MIX) is such a tensor (is_grown_param): one group, without weight
    decay."""
    decay = [p for n, p in model.named_parameters() if is_grown_param(n) and not n.endswith(NO_DECAY_SUFFIXES)]
    no_decay = [p for n, p in model.named_parameters() if is_grown_param(n) and n.endswith(NO_DECAY_SUFFIXES)]
    groups = [{"params": decay, "weight_decay": weight_decay, "lr_scale": lr_scale, "pass2": True, **extra},
              {"params": no_decay, "weight_decay": 0.0, "lr_scale": lr_scale, "pass2": True, **extra}]
    return [g for g in groups if g["params"]]


def param_groups(model: torch.nn.Module, weight_decay: float, suffixes: tuple = NO_DECAY_SUFFIXES) -> list[dict]:
    """suffixes: the no-decay class (no_decay_suffixes(o); NO_DECAY_SUFFIXES by default, the groups of before)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if is_grown_param(name):  # pass 2's own tensors, or the input mix's w: their own groups (pass2_groups)
            continue
        (no_decay if name.endswith(suffixes) else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def is_muon_param(name: str, p: torch.nn.Parameter) -> bool:
    """2-D weights of hidden linear layers. Not: the token embedding (= tied head), the loop
    embedding (an embedding table), GDN depthwise convolutions and gate projections, ASEntmax's scale
    heads (beta_proj / gamma_proj), loop-boundary parameters (injection, hyper-connections) and routers."""
    if p.ndim != 2 or name.endswith(("embed.weight", "conv_weight") + ADAMW_MATRIX_SUFFIXES):
        return False
    # ".attn_res.": a depth mixer's tensors (Block AttnRes's are one-dimensional; the architecture arms' address
    # networks, ced/model/ldm.py, are narrow matrices: AdamW, as the depth mixer's other tensors; module docstring)
    excluded = ("hyper_connections", "injection", "router", "loop_embedding", ".attn_res.")
    # "cp_head.": the Copy-Paste head's W_Q / W_K, R1-entry's treatment of small new projections (module docstring)
    return not any(x in name for x in excluded) and not name.startswith("cp_head.")


def recur_b_mode(model: torch.nn.Module, o: OptimConfig) -> str:
    """OptimConfig.recur_b, checked: one of RECUR_B_MODES; a value other than "adamw" needs name "muon" and a model
    whose RECUR_B is a matrix (a looped middle with injection "parcae" and recur_inject_b "dense"). A key that
    would do nothing is refused, so a config never says B is under Muon when it is not."""
    mode = o.recur_b
    if mode not in RECUR_B_MODES:
        raise ValueError(f"optim.recur_b {mode!r}: one of {RECUR_B_MODES}")
    if mode == "adamw":
        return mode
    if o.name != "muon":
        raise ValueError(f"optim.recur_b {mode!r} is read under optim.name 'muon', not {o.name!r} (there every tensor is on AdamW)")
    b = dict(model.named_parameters()).get(RECUR_B)
    if b is None or b.ndim != 2:
        raise ValueError(f"optim.recur_b {mode!r} is for the dense matrix {RECUR_B} of a looped middle's injection "
                         "(model.loop.recur_layers with injection 'parcae' and recur_inject_b 'dense'); this model has "
                         + ("none" if b is None else "a vector there (recur_inject_b 'diag')") + ": leave it 'adamw'")
    return mode


def scalar_a_lr_scale(model: torch.nn.Module, o: OptimConfig) -> float:
    """OptimConfig.layer_scalar_a_lr_scale, checked: a finite number above 0, and 1.0 on a model that has no SCALAR_A
    (no residual scalars, or the form "x0"). A key that would do nothing is refused, so a config never names an LR
    that no tensor has."""
    scale = o.layer_scalar_a_lr_scale
    if type(scale) not in (int, float) or not (math.isfinite(scale) and scale > 0):
        raise ValueError(f"optim.layer_scalar_a_lr_scale {scale!r}: a finite number above 0 (1.0: the scalars' own LR)")
    if scale != 1.0 and SCALAR_A not in dict(model.named_parameters()):
        raise ValueError(f"optim.layer_scalar_a_lr_scale {scale!r} is the LR multiplier of {SCALAR_A} (the residual "
                         "scalars' a: model.encoder.layer_scalars 'skip_x0', 'skip' or 'skip_x0_span'); this model has "
                         "no such tensor: leave it 1.0")
    return float(scale)


def take_scalar_a(groups: list[dict], model: torch.nn.Module, lr_scale: float, **extra) -> list[dict]:
    """`groups` with SCALAR_A taken out of the group that holds it and put in a group of its own at the end (no weight
    decay, the LR multiplier lr_scale, marked "layer_scalar_a"). Every other tensor stays in its group, in its place."""
    a = dict(model.named_parameters())[SCALAR_A]
    held = [g for g in groups if any(p is a for p in g["params"])]
    assert len(held) == 1 and held[0]["weight_decay"] == 0.0, f"{SCALAR_A} is not in exactly one group without weight decay"
    held[0]["params"] = [p for p in held[0]["params"] if p is not a]
    return groups + [{"params": [a], "weight_decay": 0.0, "lr_scale": lr_scale, "layer_scalar_a": True, **extra}]


# The methodology programme's matrix optimizers (ced/train/matrix_optim.py): optim.name values that put one of them on
# exactly the tensors "muon" puts on Muon (matrix_groups), with "muon"'s AdamW groups for everything else.
MATRIX_NAMES = ("kl_soap", "shampoo", "normuon", "anvil2_muon")
# The baselines of our optimizer's mechanism test 1 (OptimConfig.ours): MatrixWithAdamW rules on the same groups, not
# the optimizers (no Hyperball, no item-4 grid).
OURS_BASELINE_NAMES = ("muon_lite_l", "samuon_lite")

# The keys only one name reads (OptimConfig, after switch_warmup_steps), with their defaults: a non-default value under
# another name is refused, so a config never names a setting that no optimizer reads.
MATRIX_KEYS = {"kl_soap": {"soap_shampoo_beta": 0.95, "soap_precondition_frequency": 1, "soap_eigh": False,
                           "soap_seeded_start": False},
               "shampoo": {"shampoo_epsilon": 1e-12, "shampoo_precondition_frequency": 100, "shampoo_start_step": -1},
               "normuon": {"normuon_beta2": 0.95}}

# ANVIL-II's faithful pieces (ced/train/matrix_optim.py ANVIL_STYLE): the OptimConfig key of each, the name of its
# setting in MatrixWithAdamW.anvil, and its default (today's anvil2_muon).
ANVIL2_PRESETS = ("style", "record")
# anvil2_points' default None is "the preset's basis": "scaled" under "style" (today), "absolute" under "record"; a named
# basis is read under either preset.
ANVIL2_KEYS = {"anvil2_single_rounding": ("single_rounding", False), "anvil2_tavg": ("tavg", False),
               "anvil2_decon": ("decon", "matrix"), "anvil2_scale": ("scale", "ours"), "anvil2_points": ("points", None),
               "anvil2_tail_ema": ("tail_ema", False), "anvil2_qk_head_pairs": ("qk_pairs", False)}
MATRIX_KEYS["anvil2_muon"] = {"anvil2_preset": "style", **{key: default for key, (_, default) in ANVIL2_KEYS.items()}}
# A key's earlier default that a saved config may hold (TrainConfig is saved whole in every checkpoint and read back by
# Trainer.from_checkpoint and the analyses): an earlier version of the code wrote optim.anvil2_points "scaled", its
# default then. check_matrix_keys takes it as the default under any other optim.name, where the key is not read.
MATRIX_KEYS_EARLIER_DEFAULTS = {"anvil2_points": ("scaled",)}


def anvil2_settings(o: OptimConfig) -> dict:
    """MatrixWithAdamW.anvil from the config: each anvil2_* key's value under the preset "style" (today's default: every
    key at its default is today's anvil2_muon); under "record", ced/train/matrix_optim.py ANVIL_RECORD for the pieces that
    touch only the Muon group (the single-rounding cascade, TAVG with the per-bank blend, the per-bank renorm, the record's
    update scale and decay), whose keys must then be left at their defaults (the preset sets them; a config never names a
    value it does not get), and the record's own step numbers as the schedule's points unless optim.anvil2_points names a
    basis (its default None is the preset's: "scaled" under "style", "absolute" under "record"). anvil2_tail_ema,
    anvil2_qk_head_pairs and optim.lr_floor stay separate keys under either preset."""
    from ced.train.matrix_optim import ANVIL_RECORD

    if o.anvil2_preset not in ANVIL2_PRESETS:
        raise ValueError(f"optim.anvil2_preset {o.anvil2_preset!r}: one of {ANVIL2_PRESETS}")
    out = {name: getattr(o, key) for key, (name, _) in ANVIL2_KEYS.items()}
    if o.anvil2_preset == "record":
        named = [key for key, (name, default) in ANVIL2_KEYS.items()
                 if name in ANVIL_RECORD and name != "points" and getattr(o, key) != default]
        if named:
            raise ValueError(f"optim.anvil2_preset 'record' sets {sorted(ANVIL_RECORD)}: leave {named} at their defaults "
                             "(or use the preset 'style' with the keys one by one)")
        out.update({name: v for name, v in ANVIL_RECORD.items() if name != "points" or o.anvil2_points is None})
    elif out["points"] is None:
        out["points"] = "scaled"
    return out


def anvil2_matrix_roles(model: torch.nn.Module) -> dict:
    """Each matrix's rows by the record's bank (ced/train/matrix_optim.py BANKS), keyed by the tensor: (segments, mlp_out).
    The record's banks (R:2105-2175): Q and K ("qk": the plain tail-blend, 0.55), V and O ("vo") and the MLP ("mlp"): TAVG
    and 0.3493. Our matrices, row by row:
    - attention q_proj: per head hd query rows ("qk") and, with gate "elementwise", hd output-gate rows ("vo": the gate
      multiplies the attention output, on the value path; q_proj(x).view(.., hq, 2 hd).chunk(2) puts them after each
      head's query rows); k_proj "qk"; v_proj, o_proj and a headwise g_proj "vo";
    - GDN in_proj_qkv: the q and k channels "qk", the v channels "vo" ([q | k | v], ced/model/gdn.py); in_proj_z (the
      output gate) and out_proj "vo";
    - FFN (SwiGLU w_gate, w_up; ReLU2 w_up) "mlp"; w_down "mlp" and the MLP output matrix (mlp_out: the record's c_proj,
      2x LR, stored transposed).
    A matrix of the Muon group in none of these modules has no entry (the pieces that need a role refuse it)."""
    from ced.model.attention import GatedAttention
    from ced.model.gdn import GatedDeltaNet
    from ced.model.mlp import ReLU2, SwiGLU

    def whole(lin, bank):
        return [(0, lin.weight.shape[0], bank)]

    roles = {}
    for mod in model.modules():
        if isinstance(mod, GatedAttention):
            hq, hd = mod.cfg.n_q_heads, mod.cfg.head_dim
            if mod.cfg.gate == "elementwise":
                segs = [s for h in range(hq) for s in ((2 * h * hd, (2 * h + 1) * hd, "qk"),
                                                       ((2 * h + 1) * hd, (2 * h + 2) * hd, "vo"))]
            else:
                segs = whole(mod.q_proj, "qk")
            roles[mod.q_proj.weight] = (segs, False)
            for lin, bank in ((mod.g_proj, "vo"), (mod.k_proj, "qk"), (mod.v_proj, "vo"), (mod.o_proj, "vo")):
                if lin is not None:
                    roles[lin.weight] = (whole(lin, bank), False)
        elif isinstance(mod, GatedDeltaNet):
            qk = 2 * mod.cfg.n_k_heads * mod.cfg.k_head_dim
            roles[mod.in_proj_qkv.weight] = ([(0, qk, "qk"), (qk, mod.in_proj_qkv.weight.shape[0], "vo")], False)
            roles[mod.in_proj_z.weight] = (whole(mod.in_proj_z, "vo"), False)
            roles[mod.out_proj.weight] = (whole(mod.out_proj, "vo"), False)
        elif isinstance(mod, (SwiGLU, ReLU2)):
            for lin in (getattr(mod, "w_gate", None), mod.w_up):
                if lin is not None:
                    roles[lin.weight] = (whole(lin, "mlp"), False)
            roles[mod.w_down.weight] = (whole(mod.w_down, "mlp"), True)
    return roles


def _head_pairs(first: int, n_heads: int, hd: int, stride: int | None = None) -> list[list[tuple[int, int]]]:
    """Units of QK_HEADS_PER_UNIT adjacent heads: head h's rows start at first + h x stride (stride hd: contiguous heads)
    and are hd long; a lone last head (an odd count) is a unit of its own."""
    from ced.train.matrix_optim import QK_HEADS_PER_UNIT

    stride = hd if stride is None else stride
    out = []
    for h0 in range(0, n_heads, QK_HEADS_PER_UNIT):
        merged = []
        for h in range(h0, min(h0 + QK_HEADS_PER_UNIT, n_heads)):
            a, b = first + h * stride, first + h * stride + hd
            if merged and merged[-1][1] == a:  # adjacent rows: one slice
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        out.append(merged)
    return out


def anvil2_qk_units(model: torch.nn.Module) -> dict:
    """The units of OptimConfig.anvil2_qk_head_pairs (ced/train/matrix_optim.py anvil_matrix_info "units"), keyed by the
    tensor: the record orthogonalises Q and K two heads at a time (R:2111-2122: per layer, three groups of two adjacent
    query heads and three of two adjacent key heads, each [2 x 128, 768]; V, O and the MLP whole). Our matrices:
    - attention q_proj: the query rows of each two adjacent query heads one unit; with gate "elementwise" each head's
      128 gate rows sit after its query rows (q_proj(x).view(.., hq, 2 hd).chunk(2)), so a pair's query rows are two
      ranges, and the gate rows of all heads are one more unit (they are on the value path, the "vo" bank, whose
      matrices the record keeps whole: the reading of "its gate rows stay as today");
    - attention k_proj: each two adjacent KV heads one unit (at 60M one KV head and at 200M two: one unit, the whole
      matrix as today);
    - GDN in_proj_qkv ([q | k | v], ced/model/gdn.py): the q rows of each two adjacent key heads one unit, the k rows
      the same, the v rows one unit (the record's V is whole);
    - every other matrix: whole (no entry).
    A lone last head (an odd head count, e.g. the 200M GDN's 3 key heads) is a unit of its own. A matrix that comes out
    as one unit has no entry."""
    from ced.model.attention import GatedAttention
    from ced.model.gdn import GatedDeltaNet

    units = {}
    for mod in model.modules():
        if isinstance(mod, GatedAttention):
            hq, hd = mod.cfg.n_q_heads, mod.cfg.head_dim
            if mod.cfg.gate == "elementwise":
                q = _head_pairs(0, hq, hd, stride=2 * hd) + [[(2 * h * hd + hd, 2 * h * hd + 2 * hd) for h in range(hq)]]
            else:
                q = _head_pairs(0, hq, hd)
            units[mod.q_proj.weight] = q
            if mod.k_proj is not None and mod.grid is None:  # (a grid layer's k_proj holds components, not KV heads: it stays whole)
                units[mod.k_proj.weight] = _head_pairs(0, mod.cfg.n_kv_heads, hd)
        elif isinstance(mod, GatedDeltaNet):
            hk, dk = mod.cfg.n_k_heads, mod.cfg.k_head_dim
            m = mod.in_proj_qkv.weight.shape[0]
            units[mod.in_proj_qkv.weight] = _head_pairs(0, hk, dk) + _head_pairs(hk * dk, hk, dk) + [[(2 * hk * dk, m)]]
    return {p: u for p, u in units.items() if not (len(u) == 1 and u[0] == [(0, p.shape[0])])}


def anvil2_matrix_info(model: torch.nn.Module, matrices: list, qk_pairs: bool = False) -> dict:
    """matrix_optim.anvil_matrix_info of every matrix of the group that anvil2_matrix_roles names, keyed by the tensor;
    with qk_pairs (OptimConfig.anvil2_qk_head_pairs) the units of anvil2_qk_units."""
    from ced.train.matrix_optim import anvil_matrix_info

    roles = anvil2_matrix_roles(model)
    units = anvil2_qk_units(model) if qk_pairs else {}
    return {p: anvil_matrix_info(tuple(p.shape), *roles[p], units=units.get(p)) for p in matrices if p in roles}


# TailEMA's tensors (OptimConfig.anvil2_tail_ema; the record's "embed" and "lm_head"): the input embedding and the untied
# output head of the model, by their names.
TAIL_EMA_NAMES = ("embed.weight", "out_embed.weight")


def anvil2_tail_ema_params(model: torch.nn.Module) -> list:
    named = dict(model.named_parameters())
    params = [named[n] for n in TAIL_EMA_NAMES if n in named]
    if not params:
        raise ValueError(f"optim.anvil2_tail_ema blends the embedding and the head ({TAIL_EMA_NAMES}): this model has neither")
    return params


def check_lr_floor(o: OptimConfig) -> None:
    f = o.lr_floor
    if type(f) not in (int, float) or not (math.isfinite(f) and 0.0 <= f < 1.0):
        raise ValueError(f"optim.lr_floor {f!r}: a fraction of the peak LR in [0, 1) (0: the decay ends at zero, today)")



# The muP-correct GDN grouping (OptimConfig.gdn_mup; arXiv 2606.04048, Table 2 "muP formulation of Gated Delta Net under
# AdamW": W_alpha and W_beta in the "Hidden weights" column, learning rate 1 / n_{l-1}; a_log and b, learning rate 1).
GDN_MUP_BASE_WIDTH = 512  # the 60M base's d_model (configs/v2/control-a-60m.yaml), where LR 0.003 was measured
GDN_GATES = ("in_proj_a", "in_proj_b")  # W_alpha (decay) and W_beta (write strength), Linear(d_model, n_v_heads)
GDN_SCALARS = ("A_log", "dt_bias")  # a_log and the decay gate's bias b (ced/model/gdn.py: softplus(W_a x + dt_bias))


def gdn_mup_tensors(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """The tensors gdn_mup_groups moves, in the model's parameter order: (the gate projections' weights, the scalars) of
    every GatedDeltaNet (encoder and decoder). Only GatedDeltaNet's own tensors: a looped middle's Parcae injection also
    has an A_log and a dt_bias, which are not GDN gates and stay where they are."""
    from ced.model.gdn import GatedDeltaNet

    gates, scalars = set(), set()
    for m in model.modules():
        if isinstance(m, GatedDeltaNet):
            gates.update(id(getattr(m, n).weight) for n in GDN_GATES)
            scalars.update(id(getattr(m, n)) for n in GDN_SCALARS)
    params = list(model.parameters())
    return [p for p in params if id(p) in gates], [p for p in params if id(p) in scalars]


def gdn_mup_lr_scale(fan_in: int, base_width: int = GDN_MUP_BASE_WIDTH) -> float:
    """The gate projections' LR multiplier under the option: base_width / fan_in (Adam LR proportional to 1 / fan-in,
    equal to the run's at the base width)."""
    return base_width / fan_in


def gdn_mup_groups(groups: list[dict], model: torch.nn.Module, o: OptimConfig, base_lr_scale: float, **extra) -> list[dict]:
    """`groups` with OptimConfig.gdn_mup applied (the comment at OptimConfig.gdn_mup): the GDN gate projections and
    scalars taken out of the groups that hold them (every other tensor stays in its group, in its place) and appended as
    - one group per distinct fan-in of the gates (one per model in practice): AdamW, weight decay o.weight_decay (the gates
      sit in the decay group without the option), lr_scale base_lr_scale * gdn_mup_lr_scale(fan_in, o.gdn_mup_base_width),
      marked gdn_mup "gate" and "fan_in";
    - one group of the scalars: AdamW, weight decay 0 (NO_DECAY_SUFFIXES, as without the option), lr_scale base_lr_scale
      (no width factor: the scalars' Θ(1) LR), marked gdn_mup "scalar".
    base_lr_scale is the AdamW groups' multiplier of the optimizer (adamw_lr_scale under a matrix optimizer, 1.0 under
    optim.name "adamw"). The trainer sets group["lr"] = lr * group["lr_scale"] every step. Importable on its own (the muP
    coordinate check's branch): it reads o.gdn_mup_base_width and o.weight_decay only."""
    gates, scalars = gdn_mup_tensors(model)
    if not gates:
        raise ValueError("optim.gdn_mup groups a model's GDN gates and scalars: this model has no GatedDeltaNet layer")
    moved = {id(p) for p in gates + scalars}
    for g in groups:
        held = [p for p in g["params"] if id(p) in moved]
        if held:
            want = 0.0 if all(any(p is q for q in scalars) for p in held) else o.weight_decay
            assert g["weight_decay"] == want, "a GDN gate / scalar sits in a group with another weight decay than expected"
            g["params"] = [p for p in g["params"] if id(p) not in moved]
    out = [g for g in groups if g["params"] or g.get("kind") == "matrix"]  # a group left empty is dropped
    fans = sorted({p.shape[1] for p in gates})
    for fan_in in fans:
        out.append({"params": [p for p in gates if p.shape[1] == fan_in], "weight_decay": o.weight_decay,
                    "lr_scale": base_lr_scale * gdn_mup_lr_scale(fan_in, o.gdn_mup_base_width), "gdn_mup": "gate",
                    "fan_in": fan_in, **extra})
    out.append({"params": scalars, "weight_decay": 0.0, "lr_scale": base_lr_scale, "gdn_mup": "scalar", **extra})
    return out


def gdn_mup_init_(model: torch.nn.Module, base_width: int = GDN_MUP_BASE_WIDTH) -> None:
    """optim.gdn_mup's init: every GatedDeltaNet's in_proj_a and in_proj_b, as drawn (N(0, init_std), the same random numbers),
    multiplied by sqrt(base_width / fan_in): variance proportional to 1 / fan_in, anchored at the base width where it is
    today's (the paper's experiments: "the embedding layer and all input projections are initialized with standard deviation
    0.02" at its base width d0 and larger models "according to Tables 2 and 1"). Exactly scripts/mup_coord_check.py's
    grouping "mup_gdn_init". No random number is drawn, so every other tensor starts as without the option. In place, before
    the optimizer is built (Trainer.__init__)."""
    gates, _ = gdn_mup_tensors(model)
    with torch.no_grad():
        for p in gates:
            p.mul_(math.sqrt(base_width / p.shape[1]))


def check_gdn_mup(o: OptimConfig) -> None:
    b = o.gdn_mup_base_width
    if type(b) is not int or b < 1:
        raise ValueError(f"optim.gdn_mup_base_width {b!r}: a positive integer (the d_model at which the gates' LR is the run's)")
    if not o.gdn_mup and b != GDN_MUP_BASE_WIDTH:
        raise ValueError(f"optim.gdn_mup_base_width {b!r} is read with optim.gdn_mup true: leave it {GDN_MUP_BASE_WIDTH}")


def check_matrix_keys(o: OptimConfig) -> None:
    """Refuse a matrix optimizer's key away from its default under another name, Hyperball under a name without a matrix
    group, and hyperball_radius without Hyperball (the comment above MATRIX_KEYS)."""
    for name, keys in MATRIX_KEYS.items():
        for key, default in keys.items():
            if o.name != name and getattr(o, key) != default and \
                    getattr(o, key) not in MATRIX_KEYS_EARLIER_DEFAULTS.get(key, ()):
                raise ValueError(f"optim.{key} {getattr(o, key)!r} is read under optim.name {name!r}, not {o.name!r}: "
                                 f"leave it {default!r}")
    if o.hyperball and o.name not in MATRIX_NAMES + ("muon", "adamw"):
        raise ValueError(f"optim.hyperball wraps the matrix group of optim.name 'muon', 'adamw' or {MATRIX_NAMES}, "
                         f"not {o.name!r}")
    if not o.hyperball and o.hyperball_radius != 0.0:
        raise ValueError(f"optim.hyperball_radius {o.hyperball_radius!r} is read with optim.hyperball true: leave it 0.0")


def matrix_groups(model: torch.nn.Module, o: OptimConfig) -> list[dict]:
    """The groups of MatrixWithAdamW: the tensors and the order of MuonWithAdamW's groups under name "muon" (its Muon group
    becomes the "matrix" group; tests/test_matrix_optim.py checks it tensor for tensor on the 60M and 200M bases), without
    optim.recur_b (refused under these names). The matrix group's weight decay is 0 under Hyperball; "peak_lr" is the
    group's LR before the schedule (read by ANVIL-II's LR-tied decay)."""
    a_scale = scalar_a_lr_scale(model, o)
    suffixes = no_decay_suffixes(o)
    matrix, adam_decay, adam_no_decay = [], [], []
    for name, p in model.named_parameters():
        if is_grown_param(name):  # pass 2's own tensors, or the input mix's w: AdamW groups of their own, appended below
            continue
        if is_muon_param(name, p):
            matrix.append(p)
        elif name.endswith(suffixes):
            adam_no_decay.append(p)
        else:
            adam_decay.append(p)
    groups = [
        {"params": matrix, "kind": "matrix", "weight_decay": 0.0 if o.hyperball else o.weight_decay, "lr_scale": 1.0,
         "peak_lr": o.lr},
        {"params": adam_decay, "kind": "adamw", "weight_decay": o.weight_decay, "lr_scale": o.adamw_lr_scale},
        {"params": adam_no_decay, "kind": "adamw", "weight_decay": 0.0, "lr_scale": o.adamw_lr_scale},
    ]
    groups += pass2_groups(model, o.weight_decay, o.adamw_lr_scale * o.pass2_lr_scale, kind="adamw")
    if a_scale != 1.0:  # the residual scalars' a at its own LR (OptimConfig.layer_scalar_a_lr_scale): the last group
        groups = take_scalar_a(groups, model, o.adamw_lr_scale * a_scale, kind="adamw")
    return groups


# Our optimizer, M1 to M4 on the ANVIL-II chassis, and the baselines of its mechanism test 1 (OptimConfig.ours; ced/config.py
# OursConfig; ced/train/matrix_optim.py OURS_OFF).
OURS_FLAGS = ("m1", "m2", "m3", "m4")
# Each knob and what reads it: a flag, or a baseline's optim.name. A knob away from its default without its reader is
# refused (check_ours), so a config never names a setting that does nothing.
OURS_KNOBS = {"m1_theta": "m1", "m2_beta": "m2", "m2_memory": "m2", "m2_alpha": "m2", "m2_warmup": "m2",
              "m3_p": "m3", "m3_k": "m3", "m3_every": "m3", "m3_beta": "m3", "m3_cmin": "m3", "m3_snr": "m3",
              "m4_reset_fast": "m4", "m4_layers": "m4",
              "lite_chi": "muon_lite_l", "lite_ratio": "muon_lite_l",
              "samuon_gamma": "samuon_lite", "samuon_warmup": "samuon_lite", "samuon_iters": "samuon_lite"}
OURS_NAMES = ("muon", "anvil2_muon")  # where M1, M3 and M4 act (M2: "anvil2_muon" only)


def ours_active(o: OptimConfig) -> bool:
    """One of M1 to M4 on (the optimizer is then MatrixWithAdamW, under name "muon" too)."""
    return any(getattr(o.ours, f) for f in OURS_FLAGS)


def check_ours(o: OptimConfig) -> None:
    """OptimConfig.ours, checked: the flags under the names that hold them, every knob only with its reader, the
    diagnostics' periods non-negative integers (log_every only with a flag that has something to log)."""
    from ced.config import OursConfig

    u, d = o.ours, OursConfig()
    on = [f for f in OURS_FLAGS if getattr(u, f)]
    if on and o.name not in OURS_NAMES:
        raise ValueError(f"optim.ours {on} act on optim.name 'muon' (today's Muon) or 'anvil2_muon' (the chassis), not "
                         f"{o.name!r}")
    if u.m2 and o.name != "anvil2_muon":
        raise ValueError("optim.ours.m2 sits on the ANVIL-II chassis' rails: optim.name 'anvil2_muon'")
    if on and o.hyperball:
        raise ValueError("optim.ours and optim.hyperball: one geometry for the matrix group at a time")
    for knob, reader in OURS_KNOBS.items():
        if getattr(u, knob) != getattr(d, knob):
            used = getattr(u, reader) if reader in OURS_FLAGS else o.name == reader
            if not used:
                raise ValueError(f"optim.ours.{knob} {getattr(u, knob)!r} is read with "
                                 + (f"optim.ours.{reader}" if reader in OURS_FLAGS else f"optim.name {reader!r}")
                                 + f": leave it {getattr(d, knob)!r}")
    for key in ("log_every", "rotation_log_every"):
        v = getattr(u, key)
        if type(v) is not int or v < 0:
            raise ValueError(f"optim.ours.{key} {v!r}: 0 (off) or a period in optimizer steps")
    if u.log_every and not (u.m1 or u.m2 or u.m3):
        raise ValueError("optim.ours.log_every logs M1's angle, M2's noise estimate and M3's spectra: none is on")
    if u.m2 and not (0.0 < u.m2_memory <= 1.0):
        raise ValueError(f"optim.ours.m2_memory {u.m2_memory!r}: the slow rail's memory as a fraction of the run, (0, 1]")
    if u.m4 and (type(u.m4_layers) is not int or u.m4_layers < 1):
        raise ValueError(f"optim.ours.m4_layers {u.m4_layers!r}: the first group's layer count, >= 1")


def ours_settings(o: OptimConfig, total_steps: int | None) -> dict:
    """MatrixWithAdamW.ours from the config (ced/train/matrix_optim.py OURS_OFF's keys): the flags and their knobs; M2's
    beta by horizon when 0 (matrix_optim.m2_horizon_beta: 1 - 1 / (memory x the run's optimizer steps)) and its warm-up
    at the run's steps when 0 (AdEMAMix's T); the baselines' knobs under their names."""
    from ced.train.matrix_optim import OURS_OFF, m2_horizon_beta

    u = o.ours
    s = dict(OURS_OFF)
    if u.m1:
        s.update(m1=True, m1_theta=float(u.m1_theta))
    if u.m2:
        if type(total_steps) is not int or total_steps < 1:
            raise ValueError("optim.ours.m2 needs the run's optimizer steps")
        s.update(m2=True, m2_beta=float(u.m2_beta) or m2_horizon_beta(total_steps, u.m2_memory),
                 m2_alpha=float(u.m2_alpha), m2_warmup=int(u.m2_warmup) or total_steps)
    if u.m3:
        s.update(m3=True, m3_p=float(u.m3_p), m3_k=u.m3_k, m3_every=u.m3_every, m3_beta=float(u.m3_beta),
                 m3_cmin=float(u.m3_cmin), m3_snr=bool(u.m3_snr))
    if u.m4:
        s.update(m4=True, m4_reset_fast=bool(u.m4_reset_fast))
    if o.name == "muon_lite_l":
        s.update(lite_chi=float(u.lite_chi), lite_ratio=float(u.lite_ratio))
    if o.name == "samuon_lite":
        s.update(samuon_gamma=float(u.samuon_gamma), samuon_warmup=float(u.samuon_warmup), samuon_iters=u.samuon_iters)
    return s


def ours_m1_params(model: torch.nn.Module, matrices: list) -> list:
    """The map is the object tests/test_scale_invariance.py checks. None at all is refused."""
    from ced.model.invariance import m1_eligible, scale_invariance_map

    inv = scale_invariance_map(model)
    names = {id(p): n for n, p in model.named_parameters()}
    out = [p for p in matrices if m1_eligible(inv[names[id(p)]])]
    if not out:
        raise ValueError("optim.ours.m1: no matrix of the Muon group is scale-invariant in this model "
                         "(ced/model/invariance.py)")
    return out


def ours_m4_params(model: torch.nn.Module, matrices: list, layers: int) -> list:
    """M4's tensors: the matrices of the matrix group in encoder layers 0 .. layers - 1 (the first hybrid group: layers
    0-3 at 60M and 200M, where full_attn_interval is 4)."""
    names = {id(p): n for n, p in model.named_parameters()}
    first = tuple(f"encoder.layers.{i}." for i in range(layers))
    return [p for p in matrices if names[id(p)].startswith(first)]


def build_matrix_optimizer(model: torch.nn.Module, o: OptimConfig, total_steps: int | None = None) -> torch.optim.Optimizer:
    """MatrixWithAdamW for optim.name in MATRIX_NAMES, or for "muon" / "adamw" with optim.hyperball (matrix_groups), or
    for "muon" with one of our optimizer's flags (OptimConfig.ours)."""
    from ced.train.matrix_optim import NAME_TO_RULE, MatrixWithAdamW

    if o.recur_b != "adamw":
        raise ValueError(f"optim.recur_b {o.recur_b!r} is built for MuonWithAdamW (optim.name 'muon' without Hyperball)")
    groups = matrix_groups(model, o)
    if o.gdn_mup:  # the muP-correct GDN grouping (OptimConfig.gdn_mup): the last groups
        groups = gdn_mup_groups(groups, model, o, o.adamw_lr_scale, kind="adamw")
    groups = take_ccr_translator(groups, model, o, o.adamw_lr_scale, kind="adamw")  # CCR v2's M: the last group, if any
    anvil, info, tail = None, None, None
    if o.name == "anvil2_muon":  # the faithful pieces (OptimConfig.anvil2_*): no role map and no TailEMA tensor when off
        from ced.train.matrix_optim import ANVIL_STYLE

        anvil = anvil2_settings(o)
        if any(anvil[name] != default for name, default in ANVIL_STYLE.items()):
            info = anvil2_matrix_info(model, groups[0]["params"], qk_pairs=anvil["qk_pairs"])
        if anvil["tail_ema"]:
            tail = anvil2_tail_ema_params(model)
    extra = {}
    if ours_active(o) or o.name in OURS_BASELINE_NAMES:  # nothing is passed when every flag is off
        mats = groups[0]["params"]
        extra = {"ours": ours_settings(o, total_steps),
                 "m1_params": ours_m1_params(model, mats) if o.ours.m1 else None,
                 "m4_params": ours_m4_params(model, mats, o.ours.m4_layers) if o.ours.m4 else None,
                 "param_names": {id(p): n for n, p in model.named_parameters()}}
    return MatrixWithAdamW(groups, lr=o.lr, rule=NAME_TO_RULE[o.name], betas=tuple(o.betas), eps=o.eps,
                           momentum=o.muon_momentum, nesterov=o.muon_nesterov, ns_steps=o.muon_ns_steps,
                           hyperball=o.hyperball, hyperball_radius=o.hyperball_radius, normuon_beta2=o.normuon_beta2,
                           soap_shampoo_beta=o.soap_shampoo_beta, soap_frequency=o.soap_precondition_frequency,
                           soap_eigh=o.soap_eigh, shampoo_epsilon=o.shampoo_epsilon,
                           shampoo_frequency=o.shampoo_precondition_frequency, shampoo_start=o.shampoo_start_step,
                           total_steps=total_steps, anvil=anvil, matrix_info=info, tail_ema_params=tail,
                           soap_seeded_start=o.soap_seeded_start, **extra)


def reset_moments(optimizer: torch.optim.Optimizer) -> int:
    """What Token Superposition Training's switch resets: the optimizer's own reset_moments() (MatrixWithAdamW: the moments,
    not the preconditioners; MuonWithAdamW: everything it holds, which is moments only), else (torch.optim.AdamW) its
    whole state. Returns the number of tensors whose state was reset. For MuonWithAdamW and AdamW the next step is the
    one after optimizer.state.clear(), bit for bit."""
    if hasattr(optimizer, "reset_moments"):
        return optimizer.reset_moments()
    n = sum(1 for st in optimizer.state.values() if st)
    optimizer.state.clear()
    return n


def build_optimizer(model: torch.nn.Module, o: OptimConfig, total_steps: int | None = None) -> torch.optim.Optimizer:
    """total_steps: the run's optimizer steps (Trainer.total_steps), read by ANVIL-II only (its rail schedule and its EMA
    blend at the end); every other optimizer ignores it."""
    b_mode = recur_b_mode(model, o)
    a_scale = scalar_a_lr_scale(model, o)
    check_matrix_keys(o)
    check_gdn_mup(o)
    no_decay_suffixes(o)  # refuses an unknown optim.zero_gate_decay or optim.mix_query_decay under every optimizer
    check_mix_query_decay(model, o)
    check_lr_floor(o)
    check_ours(o)
    if o.name in MATRIX_NAMES + OURS_BASELINE_NAMES or o.hyperball or ours_active(o):
        return build_matrix_optimizer(model, o, total_steps)
    if o.name == "adamw":
        groups = param_groups(model, o.weight_decay, no_decay_suffixes(o)) + pass2_groups(model, o.weight_decay, o.pass2_lr_scale)
        if a_scale != 1.0:  # the residual scalars' a at its own LR (OptimConfig.layer_scalar_a_lr_scale): the last group
            groups = take_scalar_a(groups, model, a_scale)
        if o.gdn_mup:  # the muP-correct GDN grouping (OptimConfig.gdn_mup): the last groups
            groups = gdn_mup_groups(groups, model, o, 1.0)
        groups = take_ccr_translator(groups, model, o, 1.0)  # CCR v2's M: the last group, if any
        return torch.optim.AdamW(groups, lr=o.lr, betas=tuple(o.betas), eps=o.eps)
    if o.name == "muon":
        from ced.train.muon import MuonWithAdamW

        suffixes = no_decay_suffixes(o)
        muon, adam_decay, adam_no_decay, b_own = [], [], [], []
        for name, p in model.named_parameters():
            if is_grown_param(name):  # pass 2's own tensors, or the input mix's w: AdamW groups of their own, appended below
                continue
            if name == RECUR_B and b_mode != "adamw":  # the injection's dense B under Muon (OptimConfig.recur_b)
                (muon if b_mode == "muon" else b_own).append(p)
            elif is_muon_param(name, p):
                muon.append(p)
            elif name.endswith(suffixes):
                adam_no_decay.append(p)
            else:
                adam_decay.append(p)
        groups = [
            {"params": muon, "kind": "muon", "weight_decay": o.weight_decay, "lr_scale": 1.0},
            {"params": adam_decay, "kind": "adamw", "weight_decay": o.weight_decay, "lr_scale": o.adamw_lr_scale},
            {"params": adam_no_decay, "kind": "adamw", "weight_decay": 0.0, "lr_scale": o.adamw_lr_scale},
        ]
        groups += pass2_groups(model, o.weight_decay, o.adamw_lr_scale * o.pass2_lr_scale, kind="adamw")
        if b_own:  # "muon_no_decay": the last group, so every other group holds what it held
            groups.append({"params": b_own, "kind": "muon", "weight_decay": 0.0, "lr_scale": 1.0, "recur_b": True})
        if a_scale != 1.0:  # the residual scalars' a at its own LR (OptimConfig.layer_scalar_a_lr_scale): the last group
            groups = take_scalar_a(groups, model, o.adamw_lr_scale * a_scale, kind="adamw")
        if o.gdn_mup:  # the muP-correct GDN grouping (OptimConfig.gdn_mup): the last groups
            groups = gdn_mup_groups(groups, model, o, o.adamw_lr_scale, kind="adamw")
        groups = take_ccr_translator(groups, model, o, o.adamw_lr_scale, kind="adamw")  # CCR v2's M: the last group, if any
        return MuonWithAdamW(groups, lr=o.lr, betas=tuple(o.betas), eps=o.eps, momentum=o.muon_momentum,
                             nesterov=o.muon_nesterov, ns_steps=o.muon_ns_steps)
    raise ValueError(f"unknown optimizer {o.name!r}")
