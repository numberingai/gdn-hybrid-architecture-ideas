"""The Copy-Paste head (CPLM; ModelConfig.cp_head; off by default).

    h_t = final hidden state (after the final norm, the LM head's input).
    q_t = W_Q h_t, k_tau = W_K h_tau, W_Q, W_K in R^{d_c x D}, d_c = 64. Learnable sink key k_sink in R^{d_c}.
    e_tau = k_tau . q_t / sqrt(d_c) over context positions tau <= t, same document only, plus e_sink = k_sink . q_t / sqrt(d_c).
    s_t = softmax over {sink, 1..t}, evaluated at the sink.     alpha_t(tau) = softmax over {1..t} only (no sink).
    p_CP(v) = sum over tau <= t with x_tau = v of alpha_t(tau).
    One extra row <copy> in the LM head: c_t = softmax over V u {<copy>} at <copy>; p_LM = softmax over V only.
    lambda_t = c_t (1 - s_t); p(v) = (1 - lambda_t) p_LM(v) + lambda_t p_CP(v). Loss = standard CE on p; no auxiliary loss.
    Never materialise T x T x V; for the loss only p_CP(target) is needed.

Sources: arXiv 2610.19247 (Godey & Artzi, 13 Oct 2026; section 2.2, equations 2-7) and the modded-nanogpt record of PR
#379 (NathanGodey/modded-nanogpt @ c44cc41e, track_1_short/cplm_copy.py and model/gpt.py).

THE TENSORS (`model.cp_head`, built after every base tensor, drawn from a generator of their own: the base model's
parameters, state dict order and RNG use are the model without the key):
    w_q, w_k   (d_c, D)   the query and key projections, applied to h (the final norm's output)
    k_sink     (d_c,)     the sink key
    w_copy     (D,)       the <copy> row of the LM head, held as its own vector: the head's V-row GEMM is the base's
                          kernel unchanged, and the <copy> logit z_copy = w_copy . h is one more dot product.
                          LSE over V u {<copy>} = logaddexp(LSE_V, z_copy). (A tied head: the same vector, not a row
                          of the input embedding: <copy> is never an input token. The 60M base is untied.)
    qk_gain    (1,)       with model.cp_head.qk_norm only: the query gain, stored zero-centred (the gain is 1 + qk_gain)
The first four start N(0, init_std), the rule of the model's linears and of the head's rows (init_weights): the paper
gives no initialisation. With model.cp_head.init "reference", W_Q, W_K and k_sink start as the record's instead
(U(+-sqrt(3/D)) with D = d_model, and N(0, 1/d_c)); w_copy keeps its draw; qk_gain starts at 0 (a gain of 1).

THE REFERENCE'S TWO COLLAPSE FIXES (off by default). PR #379 (NathanGodey/modded-nanogpt @ c44cc41e): model/gpt.py
L925-927 q = rms_norm(q.float(), (d,)) * copy_qk_gain; k = rms_norm(k.float(), (d,)) (README: "Without it the pointer
logits blow up on some seeds"; train_gpt.py L15 sets CPLM_QK_NORM=1; copy_qk_gain: shape (1,), CPLM_QK_GAIN_INIT default
1, gpt.py L264-265) model/gpt.py L238-245 bound = sqrt(3) * model_dim^-0.5; the q/k bank U(-bound, bound); the sink
randn(d) * d^-0.5; from a generator of its own when CPLM_INIT_SEED is set ("the copy init draw, not the backbone's,
decided the bad seeds"; the shipped train_gpt.py leaves it unset). Ours is always the head's own generator
(own_generator), under both inits. Here: qk_norm normalises q and k over the d_c axis with torch's rms_norm at eps
QK_NORM_EPS (2^-23: the reference calls it on float32 with the default eps, finfo(float32).eps; fixed here so that the
float64 tests compute the same function), q times 1 + qk_gain; the scores' 1/sqrt(d_c) comes after, on q. k and on q.
k_sink as in the reference (cplm_copy.py L386: s0 = q. k_sink * d^-0.5 with the normed q); the sink key itself is not
normed. queries() and keys() apply it, so every path that computes the pointer reads it (the training terms, the compiled
block's inputs, the naive reference, scoring, the copy cache of cached generation, the position readings).

THE LOSS IN LOG SPACE (no T x T x V; never p_CP over V in training). Per query t with target y = x_{t+1}:
    ctx(t)   = {tau : idx_tau <= idx_t, doc_tau = doc_t}         (tau = t included: the spec's tau <= t)
    match(t) = {tau in ctx(t) : x_tau = y}
    LSE_ctx = logsumexp over ctx of e; log s = e_sink - logaddexp(e_sink, LSE_ctx); log(1 - s) = LSE_ctx - logaddexp(...)
    log p_CP(y) = logsumexp over match of e - LSE_ctx, -inf when match is empty (no NaN in value or gradient: a row
      without a match takes a dummy row of zeros and is replaced by the constant -inf, so its gradient is exactly 0)
    LSE_all = logaddexp(LSE_V, z_copy); log c = z_copy - LSE_all; log(1 - c) = LSE_V - LSE_all
    log lambda = log c + log(1 - s); log(1 - lambda) = logaddexp(log(1 - c), log c + log s)   (1 - lambda = (1 - c) + c s)
    log p(y) = logaddexp(log(1 - lambda) + z_y - LSE_V, log lambda + log p_CP(y))
cp_terms computes (log s, log(1 - s), log p_CP(y)) query chunk by query chunk: [B, Cq, S] score blocks, masks built
with comparisons and torch.where only (fixed shapes; no boolean indexing, no nonzero), the document ids the forward's
own (the packing metadata every attention layer masks with). It is the reference (ced/model/cp_head.py is pure
PyTorch); mixture_log_prob is the reference mixture; mixture_grads its closed-form gradient, which the trainer's fused
head uses (ced/train/fused_head.py fused_linear_cp_ce).

THE OUTPUT AT INFERENCE AND IN SCORING (CPHeadConfig.score, or CED_CP_SCORE in a scoring process): "mixture": log p
over V, so softmax(log p) = p and every reader of logits (CE, argmax, tempered softmax, per-token nats) reads the
mixture unchanged; p_CP over V is built per query chunk by scatter_add of alpha into V (fixed shapes). "lm": the LM
logits over V, p_LM alone. HybridLM.forward returns it as `logits`; head_token_ce and head_logprobs_at are the
scoring paths' entry points for a forward with return_hidden. HybridLM.head_weight() refuses a CP model in eval mode
under "mixture", so a scoring path that would apply the bare LM head to hidden states fails instead of scoring p_LM
by mistake (HybridLM.lm_head_weight() is the V rows, for p_LM on purpose).

CACHED GENERATION: CopyCache keeps, per earlier position, k_tau (d_c), its token id, its document id and its absolute
index; a chunk appends its own before its queries read (tau = t included). The mask is the training mask.

THE POINTER'S INPUT (model.cp_head.pointer_input: cp-mix and cp-shared on the R1-entry trunk). "final" (the
default) is everything above, bit for bit: q_t = W_Q h_t, k_tau = W_K h_tau. The other two change what the pointer's query
and key read and nothing else: the <copy> row (z_copy = w_copy. h_t), the sink (e_sink = q_t. k_sink / sqrt(width)), the
softmaxes, the mixture and the loss are the rows above unchanged. "mix" (cp-mix) x^q = RMSNorm_q(sum_n alpha^q_n b_n),
alpha^q = softmax_n(w_q^T RMSNorm^q_key(b_n)), and x^k the same with its own reader, over R1-entry's output read's sources
[b_0, b_1 .. b_N] (the embedding and every block's sum: attn_res.py, R1-entry): each a DepthQuery (R1-entry's reader: a (d,)
pseudo-query, zeros at the start, and an RMSNorm on its keys) and an RMSNorm of its own (gain 1 at the start) on the read, as
a block's sublayer norms its input. q_t = W_Q x^q_t, k_tau = W_K x^k_tau, with qk_norm as above. PointerMix holds the four
(cp_head.mix.*). At the start every weight is 1/(N + 1) and R1-entry's own output read is the same uniform mean, so x^q = x^k
= h (the final norm of that read, at its gain 1): the arm starts as cp-r1e. The stack hands its sources over in the forward's
context (the last layer's attn_res.keep_sources; SOURCES_KEY). "shared" (cp-shared) no W_Q, W_K: q_t = (1 + qk_gain) qhat_t,
k_tau = khat_tau, where qhat and khat are one global attention head's own QK-normed query and its key head's QK-normed key
(model.cp_head.shared_layer, a layer without a rotary part; shared_head), as the layer reads them; the scale
1/sqrt(head_dim); the sink key is head_dim wide. With shared_ssmax qhat is the query after the layer's temperature (SSMax:
                       (1 + w_h) ln(n_t) / ln(ref)), so the sink's logit scales with the context's; without it (the default)
                       the pointer's logits carry no temperature, as cp-r1e's. The layer's own attention is unchanged. The
                       layer hands qhat and khat over on the key set it returns (KeySet.cp_q / cp_k; attention.py).
The forward returns them as ModelOutput.cp_pointer (PointerInputs) and every function below takes them as `pointer`
(pointer_sources). The fused kernel, the band (window) and every trunk but R1-entry's own form are refused with them.
Cached generation: the readers are per position, so k_tau is formed once from position tau's own sources (or its own
key) and the copy cache keeps k_tau as above (d_c values a position under "mix", head_dim under "shared").

THE LENGTH RULE (model.cp_head.temperature and model.cp_head.sink_count; design, candidates (a) and (b)). n_t = the number
of context positions query t sees: its document's positions tau <= t (pos_t + 1, the trunk's ssmax_factor count; with the
forward's own keys doc_counts, else the context mask's count, which is the same number: tests/test_cp_length_robust.py). ref
= the head's ref_len, the global layers' model.encoder.attention.temperature_ref_len (attach_cp_head). sigma_t =
attention.key_count_factor(n_t, ref) = 1 + ln(n_t / ref) / ln(ref) = ln(n_t) / ln(ref), exactly 1 at n = ref. On the
pointer's query q_t (after queries(): QK-norm and the gain), per query, before the scores: temperature "ssmax" q_t <-
sigma_t q_t: e_tau and e_sink both scale (the trunk's Scalable-Softmax on q; its learned per-head scale 1 + w_h is the
pointer's own gain 1 + qk_gain, so no tensor is added) sink_count e_sink_t <- e_sink_t + ln(n_t / ref), after the gain and
the scale Both are unclipped (training). Every path that computes the pointer reads them through length_terms: cp_terms (the
training terms, chunk_terms' two optional inputs q_sink and sink_add), cp_terms_naive, chunk_alpha (logprobs_rows: scoring
and cached generation; position_readings), the fused path (ced/kernels/cp_pointer.py fused_terms: the scale on q and the add
on e_sink are formed before the pointer kernels, which take q and e_sink as inputs, so no kernel changes) and the mechanism
reader. Off (the defaults), length_terms returns q untouched and every path runs the statements it ran before the keys.

THE INFERENCE-ONLY PROBE (CopyPasteHead.probe, a plain attribute: never in the state dict; or CED_CP_PROBE in a scoring
process; refused with gradients enabled, so a training step never reads it). On a head trained without the keys (temperature
"none", sink_count false, window 0, pointer_input "final"), clipped at ref so that every number at n_t <= ref is the
unprobed head's bit for bit (the factor is exactly 1.0 and the add exactly 0.0 there):
    "A"    q_t <- max(1, sigma_t) q_t                     the pointer and the sink (candidate (a) at inference)
    "A'"   the same factor on the context logits only     the sink's logit from the unscaled q (a diagnostic)
    "B"    e_sink_t <- e_sink_t + max(0, ln(n_t / ref))   candidate (b) at inference
    "A+B"  both

THE POINTER'S HEADS (model.cp_head.n_heads and model.cp_head.combine; runs/arch-cp-grid-60m.yaml). n_heads 1 with combine "and" (the
defaults) is everything above, bit for bit (the statements of before the keys run). With H heads (pointer_input "mix" and qk_norm only),
head h has its own reads x^q_h, x^k_h = PointerMix_h(sources) (HeadsMix: H PointerMix modules, cp_head.mix.heads.<h>.*) its own W_Q, W_K
rows h d_c .. (h + 1) d_c - 1 of w_q / w_k (H d_c, d_model): q_h = rms_norm(W_Q,h x^q_h) (1 + qk_gain[h]), k_h = rms_norm(W_K,h x^k_h),
each over its own d_c axis its own sink k_sink[h d_c .. (h + 1) d_c - 1] (k_sink is (H d_c)) and every path holds q and k FLAT, (B, L, H
d_c) (the copy cache too: H d_c values a position), with the scale 1/sqrt(d_c), the width of one head. The length rule's factor
(temperature "ssmax") multiplies the flat q, so every head's q and every sink logit scale by the same sigma_t; each head's own gain is
its learned (1 + w_h), as the trunk's per-head temp_w. "and" (cp-mix-mha) e_tau = q. k_tau / sqrt(d_c) = sum_h q_h. k_h,tau / sqrt(d_c)
over the flat vectors: the heads' scores summed into one logit per position, one softmax over {sink} u ctx; e_sink = q. k_sink /
sqrt(d_c) = the sum of the heads' sink logits. Every function above runs on it unchanged (one head of width H d_c and scale 1/sqrt(d_c)):
chunk_terms, cp_terms_naive, chunk_alpha. At n_heads 1 it IS cp-mix. "or" (cp-mix-mho) head h alone: s_h, alpha_h, p_CP,h (chunk_terms /
chunk_alpha on its own slice of q, k and k_sink); log pi_t = log_softmax_h(W_route h_t) (head_router (H, d_model), zeros: the uniform
mixture at the start; it reads h_t, the final norm's output, the <copy> row's input). The pointer's distribution over {sink} u ctx is the
mixture P(sink) = sum_h pi_h s_h, P(tau) = sum_h pi_h (1 - s_h) alpha_h(tau), so log s = logsumexp_h(log pi_h + log s_h) log(1 - s) =
logsumexp_h(log pi_h + log(1 - s_h)) log w_h = log_softmax_h(log pi_h + log(1 - s_h)) (w_h = pi_h (1 - s_h) / (1 - s)) log p_CP(y) =
logsumexp_h(log w_h + log p_CP,h(y)) (-inf when nothing matches: every head shares the match set; a dummy 0 under torch.where keeps the
gradient exactly 0) alpha(tau) = sum_h w_h alpha_h(tau) (scoring, cached generation, the readings) and lambda = c (1 - s), p = (1 -
lambda) p_LM + lambda p_CP as above. At n_heads 1, log pi = 0 (route: a softmax over one head, without reading h or the router) and log w
= 0 exactly, and a logsumexp over one entry returns it: cp-mix bit for bit in value and gradient; head_router gets no gradient
(tests/test_cp_grid.py). The fused kernel, a band (window), sink_count and the inference-only probe are refused with the heads
(check_cp_head; active_probe refuses pointer_input "mix" already). head_readings is the hook of the six-cell reading (per head: the
gains; under "or" pi_t, s_h and w_h at every position).

THE TRUNK COPY (model.cp_head.trunk_copy "all" / "subset"; ced/model/cp_trunk.py states the form; design note). No pointer of its own: no
W_Q, W_K, sink key or QK-norm gain. s and p_CP are read from the encoder's global attention heads' own attention probabilities in the
forward (shifted +1: attending key j copies x_{j+1}; the diagonal and the document's first position are "no copy", CP's sink), mixed by
learned weights over the heads (trunk_logits (H_total,), zeros: uniform). The <copy> row, lambda = c (1 - s), the mixture and the loss
are the rows above unchanged, so mixture_log_prob, mixture_grads, the fused head and every scorer read (log s, log(1 - s), log p_CP(y))
as for every other form. "trunk_copy" "none" (the default): nothing of this exists and every path runs the statements of before the key.
The forward hands the heads' queries over on the global layers' key sets (KeySet.tc_q, tc_q_far) and returns cp_trunk.TrunkInputs as
ModelOutput.cp_pointer; cp_terms, cp_terms_naive, head_logprobs_at and forward_logits dispatch to cp_trunk; position_readings is refused
by name; head_readings gives the weights. The copy cache is cp_trunk.TrunkCache (the earlier positions' input tokens and start flags: the
keys are the attention's own cache).

THE TRAINING MODE (model.cp_head.train_mode; ced/model/cp_aux.py states the two other modes). "mixture" (the default) is everything above.
"aux" (cp-aux): the pointer without the <copy> row, trained by its own auxiliary loss, dropped at evaluation (cp_head_of returns None).
"addon" (add-on CP, the CP decomposition's arm (B), 2026-10-22): this head as built under "mix", every evaluation path above unchanged (the
mixture); in training the readers read the stack's sources detached (pointer_inputs), and the trainer's loss is R1-entry's on p_LM plus the
mixture's CE with p_LM's numbers and the <copy> row's input detached (Trainer._cp_addon_loss, cp_aux.addon_log_prob).
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from ced.config import CPHeadConfig, ModelConfig
from ced.model.norms import RMSNorm, compute_dtype

SCORES = ("mixture", "lm")
ENV_SCORE = "CED_CP_SCORE"  # a scoring process's override of CPHeadConfig.score
ENV_EAGER = "CED_CP_EAGER"  # set (non-empty): the trainer's CP terms run eager on CUDA too (Trainer: the compiled block)
PREFIX = "cp_head."  # the head's tensor names in the model's state dict
NAMES = ("w_q", "w_k", "k_sink", "w_copy")
QK_GAIN = "qk_gain"  # with model.cp_head.qk_norm: the query gain (1,), registered after NAMES
INITS = ("spec", "reference")
# model.cp_head.kernel: the pure-PyTorch head (this file) or the fused path (ced/kernels/cp_pointer.py). The reference is
# the ground truth; "fused" is accepted where its values match it (tests/test_cp_fused.py; the GPU check; the band check).
KERNELS = ("reference", "fused")
# The reference's QK-norm eps: F.rms_norm(q.float(), (d,)) with eps None, which is torch.finfo(float32).eps (module
# docstring). Fixed, whatever the compute dtype, so the float64 reference of the tests is the same function.
QK_NORM_EPS = float(torch.finfo(torch.float32).eps)
# The head's own generator (attach_cp_head): CPU, seeded with torch.initial_seed() + this (the run's seed: the trainer
# calls torch.manual_seed(seed) before building), as ced/model/ldm.py's own_generator. The global RNG is not used.
OWN_SEED_OFFSET = 0xC0_2610_19247
# model.cp_head.pointer_input (module docstring, "THE POINTER'S INPUT"): what the pointer's query and key read
POINTER_INPUTS = ("final", "mix", "shared")
SHARED_KEYS = ("shared_layer", "shared_head", "shared_ssmax")  # read with pointer_input "shared" only
# SeqContext.memo key under which R1-entry's stack hands its output read's sources [b_0 .. b_N] to the pointer's readers
# (pointer_input "mix"; attn_res.BlockAttnRes.stack and entry_reference_stack write it, pointer_inputs pops it)
SOURCES_KEY = "cp_pointer_sources"
DESCRIBE = {"mix": "its own depth mixes of R1-entry's sources", "shared": "a global attention head's own q and k"}
# model.cp_head.temperature (module docstring, "THE LENGTH RULE"): none, or the trunk's Scalable-Softmax on the pointer's q
TEMPERATURES = ("none", "ssmax")
# the inference-only probe's conditions (module docstring, "THE INFERENCE-ONLY PROBE"); None / "none": no probe
PROBES = ("A", "A'", "B", "A+B")
ENV_PROBE = "CED_CP_PROBE"  # a scoring process's probe condition (CopyPasteHead.probe wins when set)
DEFAULT_REF_LEN = 8192  # AttentionConfig.temperature_ref_len's default: a head built without a model config (attach_cp_head sets it)
# model.cp_head.combine (module docstring, "THE POINTER'S HEADS"): the heads' scores summed before one softmax, or each head its own
# softmax mixed by the router's weights
COMBINES = ("and", "or")
ROUTER = "head_router"  # combine "or": the router's (H, d_model) weights (state dict: after qk_gain, before the readers)


def multi_head(c: CPHeadConfig) -> bool:
    """The pointer's heads are not the one head of before the keys (n_heads != 1, or combine "or" at n_heads 1)."""
    return c.n_heads != 1 or c.combine != "and"


def check_cp_head(cfg: ModelConfig) -> None:
    """Refuse the keys of model.cp_head wherever the head is not built, and their bad values. ValueError."""
    from ced.model.cp_aux import check_cp_aux
    from ced.model.cp_trunk import check_trunk

    check_cp_aux(cfg)  # model.cp_head.train_mode / aux_weight (cp-aux; ced/model/cp_aux.py)
    check_trunk(cfg)  # model.cp_head.trunk_copy / trunk_k / trunk_select_step (the trunk copy; ced/model/cp_trunk.py)
    c = cfg.cp_head
    d = CPHeadConfig()
    if type(c.enabled) is not bool or type(c.force_lambda_zero) is not bool or type(c.qk_norm) is not bool:
        raise ValueError(f"model.cp_head.enabled {c.enabled!r} / force_lambda_zero {c.force_lambda_zero!r} / qk_norm "
                         f"{c.qk_norm!r}: true or false")
    if c.pointer_input not in POINTER_INPUTS:
        raise ValueError(f"model.cp_head.pointer_input {c.pointer_input!r}: one of {POINTER_INPUTS}")
    if type(c.shared_layer) is not int or c.shared_layer < -1:
        raise ValueError(f"model.cp_head.shared_layer {c.shared_layer!r}: a stack index of a global layer, or -1 (the "
                         "stack's last global layer)")
    if type(c.shared_head) is not int or c.shared_head < 0:
        raise ValueError(f"model.cp_head.shared_head {c.shared_head!r}: a query head index, an integer >= 0")
    if type(c.shared_ssmax) is not bool:
        raise ValueError(f"model.cp_head.shared_ssmax {c.shared_ssmax!r}: true or false")
    if c.pointer_input != "shared":
        named = [k for k in SHARED_KEYS if getattr(c, k) != getattr(d, k)]
        if named:
            raise ValueError(f"model.cp_head.{named[0]} without model.cp_head.pointer_input 'shared': the key would be "
                             "ignored")
    if type(c.d_c) is not int or c.d_c < 1:
        raise ValueError(f"model.cp_head.d_c {c.d_c!r}: the query/key width, an integer >= 1")
    if type(c.query_chunk) is not int or c.query_chunk < 1:
        raise ValueError(f"model.cp_head.query_chunk {c.query_chunk!r}: query positions per chunk, an integer >= 1")
    if c.score not in SCORES:
        raise ValueError(f"model.cp_head.score {c.score!r}: one of {SCORES}")
    if c.init not in INITS:
        raise ValueError(f"model.cp_head.init {c.init!r}: one of {INITS}")
    from ced.model.cp_trunk import TRUNK_ONLY_KERNELS

    if c.kernel not in KERNELS and not (c.trunk_copy != "none" and c.kernel in TRUNK_ONLY_KERNELS):
        # (the trunk copy's own values, model.cp_head.trunk_copy: cp_trunk.check_trunk has accepted them above)
        raise ValueError(f"model.cp_head.kernel {c.kernel!r}: one of {KERNELS}"
                         + (f" ({c.kernel!r} is the trunk copy's only: model.cp_head.trunk_copy)" if c.kernel in TRUNK_ONLY_KERNELS
                            else ""))
    if type(c.window) is not int or c.window < 0:
        raise ValueError(f"model.cp_head.window {c.window!r}: the pointer's reach in positions, an integer >= 1 (the band), "
                         "or 0 (every earlier position of the document: the spec's)")
    if c.temperature not in TEMPERATURES:
        raise ValueError(f"model.cp_head.temperature {c.temperature!r}: one of {TEMPERATURES}")
    if type(c.sink_count) is not bool:
        raise ValueError(f"model.cp_head.sink_count {c.sink_count!r}: true or false")
    if type(c.n_heads) is not int or c.n_heads < 1:
        raise ValueError(f"model.cp_head.n_heads {c.n_heads!r}: the pointer's heads, an integer >= 1")
    if c.combine not in COMBINES:
        raise ValueError(f"model.cp_head.combine {c.combine!r}: one of {COMBINES}")
    if not c.enabled:
        named = [k for k in ("d_c", "query_chunk", "force_lambda_zero", "score", "qk_norm", "init", "kernel", "window",
                             "pointer_input", "temperature", "sink_count", "n_heads", "combine") if getattr(c, k) != getattr(d, k)]
        if named:
            raise ValueError(f"model.cp_head.{named[0]} without model.cp_head.enabled: the key would be ignored")
        return
    if multi_head(c):
        # the pointer's heads (module docstring, "THE POINTER'S HEADS"): built on cp-mix's pointer only, on the pure-PyTorch head
        # (the fused kernel is not accepted and has one head), over every earlier position of the document; each head's gain is
        # the QK-norm's
        what = f"model.cp_head.n_heads {c.n_heads} with combine {c.combine!r}"
        if c.kernel != "reference":
            raise ValueError(f"model.cp_head.kernel {c.kernel!r} is not built together with {what} (the fused pointer has one "
                             "head; the grid's arms run on the reference path): use kernel 'reference'")
        if c.pointer_input != "mix":
            raise ValueError(f"{what} is built on cp-mix's pointer (model.cp_head.pointer_input 'mix') only, not on "
                             f"{c.pointer_input!r}: nothing tests the pair")
        if not c.qk_norm:
            raise ValueError(f"{what} is built with model.cp_head.qk_norm true (each head's own QK-norm gain)")
        if c.window:
            raise ValueError(f"model.cp_head.window {c.window!r} is not built together with {what}: nothing tests the pair")
        if c.sink_count:
            raise ValueError(f"model.cp_head.sink_count is not built together with {what}: whether ln(n / ref) is added once "
                             "or per head is not decided; nothing tests the pair")
        if c.train_mode != "mixture":
            raise ValueError(f"model.cp_head.train_mode {c.train_mode!r} ({'cp-aux' if c.train_mode == 'aux' else 'add-on CP'}) is "
                             f"not built together with {what}: the grid's heads are built on cp-mix's mixture; nothing tests the pair")
    if c.pointer_input != "final":
        # cp-mix and cp-shared (module docstring): built with the pure-PyTorch head over every earlier position of the
        # document only; the fused kernel forms q and k from h inside its stacked GEMM
        if c.kernel != "reference":
            raise ValueError(f"model.cp_head.kernel {c.kernel!r} is not built together with model.cp_head.pointer_input "
                             f"{c.pointer_input!r} (the fused path projects h itself): use kernel 'reference'")
        if c.window:
            raise ValueError(f"model.cp_head.window {c.window!r} is not built together with model.cp_head.pointer_input "
                             f"{c.pointer_input!r}: nothing tests the pair")
    if c.pointer_input == "shared":
        if c.d_c != d.d_c:
            raise ValueError(f"model.cp_head.d_c {c.d_c!r} with model.cp_head.pointer_input 'shared': the pointer's width "
                             "is the shared head's head_dim, so the key would be ignored")
        if not c.qk_norm:
            raise ValueError("model.cp_head.pointer_input 'shared' is built with model.cp_head.qk_norm true: the shared "
                             "head's q and k are the trunk's QK-normed ones and the pointer's gain (qk_gain) is the fix's")
    # the length rule (module docstring): n_t is pos_t + 1, which a band does not see; cp-shared's form of (a) is shared_ssmax
    for key, on in (("temperature", c.temperature != "none"), ("sink_count", c.sink_count)):
        if on and c.window:
            raise ValueError(f"model.cp_head.{key} {getattr(c, key)!r} is not built together with model.cp_head.window "
                             f"{c.window!r}: the length rule counts every earlier position of the document (pos + 1), which "
                             "a band does not see")
    if c.temperature != "none" and c.pointer_input == "shared":
        raise ValueError(f"model.cp_head.temperature {c.temperature!r} with model.cp_head.pointer_input 'shared': the shared "
                         "head's form of it is model.cp_head.shared_ssmax true (the layer's own SSMax on the pointer's q); "
                         "both would scale q twice")
    if c.kernel == "fused" and (c.d_c & (c.d_c - 1) or not 16 <= c.d_c <= 256):
        raise ValueError(f"model.cp_head.kernel 'fused' is built for d_c a power of two in 16 .. 256 (the kernels' tile "
                         f"width), not {c.d_c}")
    from ced.model.control import has_loop

    if cfg.arch != "control":
        raise ValueError(f"model.cp_head is built for arch 'control' (one pass), not for arch {cfg.arch!r}")
    if has_loop(cfg) or cfg.encoder.loops != 1:
        raise ValueError("model.cp_head is built for the one-pass control: no loop.pass2_gate, loop.recur_layers or "
                         "looped encoder")
    if cfg.mtp.enabled:  # the MTP drafter on a CP trunk: built for the forms ced/model/mtp.py check_mtp_on_cp names
        from ced.model.mtp import check_mtp_on_cp

        check_mtp_on_cp(cfg)
    others = {"model.nvfp4.mode": (cfg.nvfp4.mode, "none"), "model.fp8.mode": (cfg.fp8.mode, "none")}
    for key, (value, off) in others.items():
        if value != off:
            raise ValueError(f"model.cp_head is not built together with {key} {value!r}: nothing tests the pair (follow-ups such as "
                             "the MTP drafter need their own build)")
    check_cp_trunk(cfg)


# The depth mixers the head is built on (model.encoder.attn_res): the base's plain residual ("none") and R1-entry
# ("block_entry", the cp-r1e arms; runs/arch-cp-r1e-60m.yaml; tests/test_cp_r1e.py). The head reads h = final_norm(the
# encoder's output) and nothing inside the stack, so on R1-entry it reads the final norm of R1-entry's output read, the
# same interface. Only R1-entry's own form is built: its static readers, one softmax a reader, the unfused path; every
# other value is refused by name (nothing tests the pair).
CP_TRUNKS = ("none", "block_entry")
# The GDN-side arms with the head: the write mix in its "layer" form only (one lambda a layer, model.encoder.gdn_write_mix "layer": the
# key wsl and wsl-nowd set), on R1-entry only (A reads R1-entry's stored sources; check_gdn_arms refuses it elsewhere). The
# wsl-nowd-cp-r1e arms; runs/arch-wsl-nowd-cp-r1e-60m.yaml; tests/test_wsl_cp_r1e.py. A changes the inputs of the GDN layers' k, v, beta
# and decay projections inside the stack and the head reads only the final norm's output, so the two share no tensor. Refused by name
# with the head on every trunk: A's "head" form (gdn_write_mix "head", wsh) and CCR (gdn_ccr_spacing). Nothing tests either pair. A's
# "layer" form with cp-mix's pointer (pointer_input "mix"; the stack test of the 4B candidates; runs/arch-stack5-r1e-60m.yaml): built for
# cp-mix's own form only, one pointer head (n_heads 1, combine "and") trained as the mixture (train_mode "mixture"); A changes the GDN
# layers' k, v, beta and decay inputs inside the stack and leaves R1-entry's stored sources and its output read as they are, so the
# pointer's readers read the same [b_0 .. b_N] (the stack's last layer, a global layer, hands them over). Every other pointer input with
# A (cp-shared, the pointer's heads, cp-aux) stays refused by name: nothing tests those pairs (tests/test_stack5_r1e_model.py tests this
# one). The same pointer trained as add-on CP (train_mode "addon"; ced/model/cp_aux.py, "ADD-ON CP"; runs/arch-stack-addon-
# r1e-60m.yaml): the readers read the same [b_0 .. b_N], DETACHED (pointer_inputs), so the mixture's gradient reaches neither A's lambdas
# nor any other trunk tensor (tests/test_stack_addon_model.py).
CP_WRITE_MIX = ("none", "layer")
WRITE_MIX_MODES = ("mixture", "addon")  # the train modes of cp-mix's pointer built together with A's "layer" form


def mix_with_write_mix(cfg: ModelConfig) -> bool:
    """The one pointer input built together with the write mix's "layer" form: cp-mix's single-head pointer trained as the mixture,
    or as add-on CP (the same pointer on detached sources)."""
    c, e = cfg.cp_head, cfg.encoder
    return (c.pointer_input == "mix" and e.gdn_write_mix == CP_WRITE_MIX[1] and not multi_head(c)
            and c.train_mode in WRITE_MIX_MODES)


def check_cp_trunk(cfg: ModelConfig) -> None:
    """Refuse model.encoder.attn_res, R1-entry's options and the GDN-side arms with the head wherever the pair is
    not built (built: the base; R1-entry's own form; R1-entry's own form with "layer" form). ValueError."""
    e = cfg.encoder
    if e.attn_res not in CP_TRUNKS:
        raise ValueError(f"model.cp_head is not built together with model.encoder.attn_res {e.attn_res!r}: nothing tests "
                         f"the pair (built: {CP_TRUNKS}, the base and R1-entry)")
    if e.gdn_ccr_spacing != 0:
        raise ValueError(f"model.cp_head is not built together with model.encoder.gdn_ccr_spacing {e.gdn_ccr_spacing!r} (a "
                         "GDN-side arm) on any trunk: nothing tests the pair")
    if e.gdn_write_mix not in CP_WRITE_MIX:
        raise ValueError(f"model.cp_head is not built together with model.encoder.gdn_write_mix {e.gdn_write_mix!r} (a "
                         f"GDN-side arm) on any trunk: nothing tests the pair (built: gdn_write_mix {CP_WRITE_MIX[1]!r} on "
                         "R1-entry, the wsl-nowd-cp-r1e arms)")
    pointer = cfg.cp_head.pointer_input
    if pointer != "final" and e.attn_res != "block_entry":
        raise ValueError(f"model.cp_head.pointer_input {pointer!r} is built on R1-entry ('block_entry', its own form) only, "
                         f"not on model.encoder.attn_res {e.attn_res!r}: nothing tests the pair")
    if e.attn_res != "block_entry":
        if e.gdn_write_mix != "none":  # check_gdn_arms refuses it first (A reads R1-entry's sources); named here too
            raise ValueError(f"model.cp_head is not built together with model.encoder.gdn_write_mix {e.gdn_write_mix!r} on "
                             f"model.encoder.attn_res {e.attn_res!r}: the pair is built on R1-entry ('block_entry') only")
        return
    r1e = {"model.encoder.attn_res_heads": (e.attn_res_heads, 1),
           "model.encoder.attn_res_query": (e.attn_res_query, "static"),
           "model.encoder.attn_res_fused": (e.attn_res_fused, False)}
    for key, (value, off) in r1e.items():
        if value != off:
            raise ValueError(f"model.cp_head on R1-entry (model.encoder.attn_res 'block_entry') is built for R1-entry's own "
                             f"form only: not together with {key} {value!r} (nothing tests the pair)")
    if pointer != "final" and e.gdn_write_mix != "none" and not mix_with_write_mix(cfg):
        raise ValueError(f"model.cp_head.pointer_input {pointer!r} is not built together with model.encoder.gdn_write_mix "
                         f"{e.gdn_write_mix!r} (a GDN-side arm): nothing tests the pair (built: pointer_input 'mix' with "
                         "gdn_write_mix 'layer', one pointer head, train_mode 'mixture' or 'addon')")
    if pointer == "shared":
        shared_layer_index(cfg)


def shared_layer_index(cfg: ModelConfig) -> int:
    """The encoder stack index of the global layer whose head the pointer shares (model.cp_head.pointer_input "shared";
    shared_layer -1: the stack's last global layer). ValueError, by name, unless it is a global layer without a rotary
    part (NoPE: the score is the head's own q . k, with no distance cap or far rotation), with the trunk's QK-norm and
    the softmax normaliser, and shared_head is one of its query heads."""
    c, e = cfg.cp_head, cfg.encoder
    glob = e.global_layers()
    if not glob:
        raise ValueError("model.cp_head.pointer_input 'shared' needs a global attention layer in the stack: there is none")
    i = glob[-1] if c.shared_layer == -1 else c.shared_layer
    if i not in glob:
        raise ValueError(f"model.cp_head.shared_layer {i}: not a global attention layer of the stack (they are {glob})")
    att = e.global_attention()[glob.index(i)]
    if att.rope_fraction != 0.0:
        raise ValueError(f"model.cp_head.shared_layer {i} has a rotary part (rope_fraction {att.rope_fraction}): the pointer "
                         "shares a head of a NoPE global layer only (model.encoder.nope_global_layers), whose score is its "
                         "own q . k with no distance cap")
    if not att.qk_norm:
        raise ValueError(f"model.cp_head.pointer_input 'shared' is built on a layer with the attention's QK-norm "
                         f"(model.encoder.attention.qk_norm); layer {i} has none, so its scores are unbounded")
    if att.normalizer != "softmax":
        raise ValueError(f"model.cp_head.pointer_input 'shared' is built on the softmax normaliser, not "
                         f"{att.normalizer!r} (layer {i})")
    if c.shared_head >= att.n_q_heads:
        raise ValueError(f"model.cp_head.shared_head {c.shared_head}: layer {i} has {att.n_q_heads} query heads")
    return i


def env_eager() -> bool:
    return bool(os.environ.get(ENV_EAGER, "").strip())


def env_score() -> str | None:
    """CED_CP_SCORE, checked: None when unset, else one of SCORES."""
    text = os.environ.get(ENV_SCORE, "").strip()
    if not text:
        return None
    if text not in SCORES:
        raise ValueError(f"{ENV_SCORE}={text!r}: one of {SCORES}")
    return text


def env_probe() -> str | None:
    """CED_CP_PROBE, checked: None when unset or "none", else one of PROBES."""
    text = os.environ.get(ENV_PROBE, "").strip()
    if not text or text == "none":
        return None
    if text not in PROBES:
        raise ValueError(f"{ENV_PROBE}={text!r}: one of {PROBES} (or unset / 'none')")
    return text


class CopyPasteHead(nn.Module):
    """The head's own tensors (module docstring). Every computation runs in compute_dtype (float32, float64 in the CPU
    tests) with autocast off, as the LM head does (ced/model/looped_ced.py lm_head)."""

    def __init__(self, d_model: int, cfg: CPHeadConfig, eps: float = 1e-6, d_shared: int | None = None,
                 ref_len: int = DEFAULT_REF_LEN, n_trunk: int | None = None):
        """eps: the model's norm_eps (pointer_input "mix": the readers' key norms and the reads' norms, as R1-entry's).
        d_shared: the shared head's head_dim (pointer_input "shared": the pointer's width and the sink key's).
        ref_len: the length rule's ref (module docstring, "THE LENGTH RULE"): the global layers' temperature_ref_len.
        n_trunk: the global heads H_total (model.cp_head.trunk_copy only: the mixing logits' length)."""
        super().__init__()
        self.cfg = cfg
        # the length rule's ref and the inference-only probe (module docstring): plain attributes, never in the state dict
        self.ref_len = int(ref_len)
        assert self.ref_len >= 2, self.ref_len
        self.probe = None
        self.shared_layer = None
        from ced.model.cp_trunk import trunk_on

        assert trunk_on(cfg) == (n_trunk is not None), "n_trunk is the global heads' count: given with model.cp_head.trunk_copy only"
        if n_trunk is not None:  # the trunk copy (module docstring, "THE TRUNK COPY"): the <copy> row, the mixing logits, the subset
            self.n_trunk = int(n_trunk)
            self.w_copy = nn.Parameter(torch.empty(d_model))
            self.trunk_logits = nn.Parameter(torch.zeros(self.n_trunk))
            self.register_buffer("trunk_keep", torch.ones(self.n_trunk, dtype=torch.bool))
            self._active = None  # (trunk_keep's version, the heads in use as a list, their index tensor): a cache, no tensor state
            return
        d_c = cfg.d_c
        H = cfg.n_heads  # the pointer's heads (module docstring, "THE POINTER'S HEADS"); 1 before the key
        shared = cfg.pointer_input == "shared"
        assert shared == (d_shared is not None), "d_shared is the shared head's width: given with pointer_input 'shared' only"
        if not shared:  # H heads: head h's W_Q / W_K are rows h d_c .. (h + 1) d_c - 1
            self.w_q = nn.Parameter(torch.empty(H * d_c, d_model))
            self.w_k = nn.Parameter(torch.empty(H * d_c, d_model))
        width = d_shared if shared else d_c
        self.k_sink = nn.Parameter(torch.empty(H * width))  # H heads: head h's sink is its own d_c slice
        self.w_copy = nn.Parameter(torch.empty(d_model))
        if cfg.qk_norm:  # the reference's copy_qk_gain (shape (1,)), zero-centred as every norm gain of the model; one a head
            self.qk_gain = nn.Parameter(torch.zeros(H))
        self.scale = 1.0 / math.sqrt(width)  # the width of ONE head under the heads too (the "and" sum keeps each head's scale)
        if cfg.pointer_input == "mix":  # cp-mix's two readers and their reads' norms, after cp-r1e's tensors (zeros); one pair a head
            self.mix = PointerMix(d_model, eps) if H == 1 else HeadsMix(d_model, eps, H)
        if cfg.combine == "or":  # the router over the heads (zeros: the uniform mixture at the start); a tensor of the head's own,
            # so the state dict holds it after qk_gain and before the readers (a module's own tensors come before its children's)
            self.head_router = nn.Parameter(torch.zeros(H, d_model))
        # pointer_input "shared": the stack index of the shared head's layer (attach_cp_head sets it; a plain attribute)
        self.shared_layer = None

    @torch.no_grad()
    def reset_own(self, g: torch.Generator, std: float, init: str = "spec") -> None:
        """N(0, std) for every tensor, in the order w_q, w_k, k_sink, w_copy, from the generator g (CPU). init
        "reference" then draws W_Q, W_K and k_sink again, from the same generator, as the record does (model/gpt.py
        L238-245): one (2, d_c, D) bank U(-bound, bound), bound = sqrt(3) * D^-0.5 with D = d_model (the record's
        model_dim, the projections' fan-in), W_Q = bank[0], W_K = bank[1]; then k_sink = randn(d_c) * d_c^-0.5. w_copy
        keeps its N(0, std) draw, the same tensor under both inits. qk_gain (with qk_norm) starts at 0: a gain of 1.
        pointer_input "mix": the same draws (the readers and their norms start at zero and draw nothing), so W_Q, W_K,
        k_sink and w_copy are cp-r1e's of the same seed. "shared" (no W_Q, W_K): the same draws into stand-ins of cp-r1e's
        shapes, so w_copy is cp-r1e's of the same seed; then the head_dim-wide sink is drawn from the same generator by
        the same rule at its width: N(0, std) under "spec", randn(head_dim) * head_dim^-0.5 under "reference".
        n_heads H > 1 (module docstring, "THE POINTER'S HEADS"): head 0 (rows 0 .. d_c - 1 of W_Q, W_K and the sink's first d_c)
        and w_copy are drawn first, by the rows above in their order, so they are cp-mix's of the same seed bit for bit; then the
        other heads' rows by the same rule (N(0, std); under "reference" one (2, (H - 1) d_c, D) bank and randn((H - 1) d_c) x
        d_c^-0.5). The readers, the norms and the router start at zero and draw nothing."""
        if init not in INITS:
            raise ValueError(f"model.cp_head.init {init!r}: one of {INITS}")
        if hasattr(self, "n_trunk"):
            # the trunk copy (module docstring, "THE TRUNK COPY"): cp-r1e's W_Q, W_K and sink drawn into stand-ins and dropped, then
            # w_copy, so w_copy is cp-r1e's and cp-mix's of the same seed bit for bit; the mixing logits start at zero (uniform
            # weights) and every head is in use
            d_c, d_model = self.cfg.d_c, self.w_copy.shape[0]
            for shape in ((d_c, d_model), (d_c, d_model), (d_c,)):
                torch.randn(shape, generator=g, dtype=torch.float32)
            self.w_copy.copy_(torch.randn(self.w_copy.shape, generator=g, dtype=torch.float32).to(self.w_copy.dtype) * std)
            self.trunk_logits.zero_()
            self.trunk_keep.fill_(True)
            return
        shared = self.cfg.pointer_input == "shared"
        dt = self.w_copy.dtype
        d_c, d_model = self.cfg.d_c, self.w_copy.shape[0]
        if shared:  # cp-r1e's W_Q, W_K and sink, drawn and dropped: the generator's stream is cp-r1e's
            w_q, w_k, k_sink = (torch.empty(s, dtype=dt) for s in ((d_c, d_model), (d_c, d_model), (d_c,)))
        else:  # head 0's rows (all of them at H = 1)
            w_q, w_k, k_sink = self.w_q[:d_c], self.w_k[:d_c], self.k_sink[:d_c]
        for p in (w_q, w_k, k_sink, self.w_copy):
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype) * std)
        if init == "reference":
            bound = (3**0.5) * d_model**-0.5
            bank = torch.empty(2, d_c, d_model, dtype=torch.float32).uniform_(-bound, bound, generator=g)
            w_q.copy_(bank[0].to(w_q.dtype))
            w_k.copy_(bank[1].to(w_k.dtype))
            k_sink.copy_((torch.randn(d_c, generator=g, dtype=torch.float32) * d_c**-0.5).to(k_sink.dtype))
        if not shared and self.cfg.n_heads > 1:  # the other heads' rows, after head 0's and w_copy's draws
            rest = (self.w_q[d_c:], self.w_k[d_c:], self.k_sink[d_c:])
            for p in rest:
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype) * std)
            if init == "reference":
                rows = rest[0].shape[0]
                bound = (3**0.5) * d_model**-0.5
                bank = torch.empty(2, rows, d_model, dtype=torch.float32).uniform_(-bound, bound, generator=g)
                rest[0].copy_(bank[0].to(dt))
                rest[1].copy_(bank[1].to(dt))
                rest[2].copy_((torch.randn(rows, generator=g, dtype=torch.float32) * d_c**-0.5).to(dt))
        if shared:  # the sink in the shared head's key space, head_dim wide, after cp-r1e's draws
            width = self.k_sink.shape[0]
            draw = torch.randn(width, generator=g, dtype=torch.float32)
            if init == "reference":
                self.k_sink.copy_((draw * width**-0.5).to(self.k_sink.dtype))
            else:
                self.k_sink.copy_(draw.to(self.k_sink.dtype) * std)
        if self.cfg.qk_norm:
            self.qk_gain.zero_()
        if self.cfg.combine == "or":
            self.head_router.zero_()

    def _proj(self, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        dtype = compute_dtype(h)
        with torch.autocast(device_type=h.device.type, enabled=False):
            return torch.nn.functional.linear(h.to(dtype), w.to(dtype))

    def _proj_heads(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """The heads' projections (module docstring, "THE POINTER'S HEADS"): x (..., H, D), each head's own input, against
        w (H d_c, D): (..., H, d_c), head h through rows h d_c .. (h + 1) d_c - 1 only; in compute_dtype, autocast off."""
        H, d_c = self.cfg.n_heads, self.cfg.d_c
        assert x.shape[-2] == H, f"the heads' inputs {tuple(x.shape)}: (..., {H}, d_model)"
        dtype = compute_dtype(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return torch.einsum("...hd,hcd->...hc", x.to(dtype), w.to(dtype).view(H, d_c, -1))

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        """The reference's RMS norm over the d_c axis (module docstring), in x's dtype, autocast off."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            return torch.nn.functional.rms_norm(x, (x.shape[-1],), eps=QK_NORM_EPS)

    def queries(self, h: torch.Tensor) -> torch.Tensor:
        """q = W_Q h; with qk_norm, rms_norm(q) * (1 + qk_gain). h: the pointer's query input (pointer_sources: the
        final norm's output, or cp-mix's x^q). pointer_input "shared": h is the shared head's QK-normed query (B, L,
        head_dim) and q = h * (1 + qk_gain), in at least float32, autocast off."""
        self._no_trunk("queries")
        if self.cfg.pointer_input == "shared":
            dtype = compute_dtype(h)
            with torch.autocast(device_type=h.device.type, enabled=False):
                return h.to(dtype) * (1.0 + self.qk_gain.to(dtype))
        if self.cfg.n_heads != 1:
            # the heads (module docstring, "THE POINTER'S HEADS"): h (..., H, D) each head's own x^q; each head normed over its
            # own d_c axis and times its own gain; returned flat, (..., H d_c) (n_heads > 1 is built with qk_norm only)
            q = self._proj_heads(h, self.w_q)
            with torch.autocast(device_type=q.device.type, enabled=False):
                return (self._norm(q) * (1.0 + self.qk_gain.to(q.dtype))[:, None]).flatten(-2)
        q = self._proj(h, self.w_q)
        if not self.cfg.qk_norm:
            return q
        with torch.autocast(device_type=q.device.type, enabled=False):
            return self._norm(q) * (1.0 + self.qk_gain.to(q.dtype))

    def keys(self, h: torch.Tensor) -> torch.Tensor:
        """k = W_K h; with qk_norm, rms_norm(k) (no gain: the reference's). pointer_input "shared": h is the shared
        head's QK-normed key, returned in at least float32. n_heads > 1: h (..., H, D), each head's own x^k; each head's key
        normed over its own d_c axis; returned flat, (..., H d_c)."""
        self._no_trunk("keys")
        if self.cfg.pointer_input == "shared":
            return h.to(compute_dtype(h))
        if self.cfg.n_heads != 1:
            return self._norm(self._proj_heads(h, self.w_k)).flatten(-2)
        k = self._proj(h, self.w_k)
        return self._norm(k) if self.cfg.qk_norm else k

    def route(self, h: torch.Tensor) -> torch.Tensor:
        """combine "or" (module docstring, "THE POINTER'S HEADS"): log pi = log_softmax over the heads of W_route h, for h
        (..., D) the final norm's output: (..., H), in compute_dtype, autocast off. At n_heads 1 a softmax over one head is 1
        whatever W_route and h: log pi = 0 exactly, without reading either (so h gains no consumer and every gradient of the model
        is cp-mix's bit for bit; the one-head router gets none)."""
        dtype = compute_dtype(h)
        if self.cfg.n_heads == 1:
            return torch.zeros(*h.shape[:-1], 1, dtype=dtype, device=h.device)
        with torch.autocast(device_type=h.device.type, enabled=False):
            return torch.log_softmax(torch.nn.functional.linear(h.to(dtype), self.head_router.to(dtype)), dim=-1)

    def head_gains(self) -> torch.Tensor:
        """1 + qk_gain per head (H,), float64, detached: each head's learned scale on its QK-normed query (the bound of its
        context logits is sqrt(d_c) x this, times the length rule's sigma_t under temperature "ssmax")."""
        return (1.0 + self.qk_gain.detach().double()).reshape(-1)

    def pointer_parameters(self) -> list[nn.Parameter]:
        """The tensors of the head the pointer's terms (log s, log(1 - s), log p_CP) read: W_Q, W_K, k_sink and, with
        qk_norm, the gain (the tests' and the GPU check's gradients); cp-mix adds its readers' and norms' tensors;
        cp-shared has no W_Q, W_K (its other inputs are the shared layer's own tensors)."""
        if hasattr(self, "n_trunk"):  # the trunk copy: the mixing logits (its other inputs are the global layers' own tensors)
            return [self.trunk_logits]
        own = [] if self.cfg.pointer_input == "shared" else [self.w_q, self.w_k]
        mix = list(self.mix.parameters()) if self.cfg.pointer_input == "mix" else []
        route = [self.head_router] if self.cfg.combine == "or" else []  # the heads' router reads h; the pointer's terms read it
        return own + [self.k_sink] + ([self.qk_gain] if self.cfg.qk_norm else []) + mix + route

    def copy_logit(self, h: torch.Tensor) -> torch.Tensor:
        """z_copy = w_copy . h for h (..., D): (...), in compute_dtype."""
        return self._proj(h, self.w_copy[None, :]).squeeze(-1)

    # --- the trunk copy (module docstring, "THE TRUNK COPY"; ced/model/cp_trunk.py) --------------------------------------------

    def _no_trunk(self, what: str) -> None:
        if hasattr(self, "n_trunk"):
            raise ValueError(f"CopyPasteHead.{what}: this head copies from the trunk's own heads (model.cp_head.trunk_copy "
                             f"{self.cfg.trunk_copy!r}): it has no pointer query or key (ced/model/cp_trunk.py)")

    def active_heads(self) -> list:
        """The global heads in use (ascending ids): every head, or after a subset's selection the selected ones. Read from trunk_keep
        once per change of the buffer (its version counter: a load or a selection writes it in place), never a device read a forward."""
        v = (self.trunk_keep._version, self.trunk_keep.data_ptr())
        if self._active is None or self._active[0] != v or self._active[2].device != self.trunk_logits.device:
            keep = self.trunk_keep.detach().cpu().tolist()
            act = [i for i, k in enumerate(keep) if k]
            self._active = (v, act, torch.tensor(act, dtype=torch.long, device=self.trunk_logits.device))
        return self._active[1]

    def trunk_log_weights(self) -> torch.Tensor:
        """log w over the heads in use (len(active_heads()),), in their order: log_softmax of their logits (zeros: uniform)."""
        act = self.active_heads()
        logits = self.trunk_logits if len(act) == self.n_trunk else self.trunk_logits.index_select(0, self._active[2])
        return torch.log_softmax(logits.to(compute_dtype(logits)), dim=0)

    def trunk_select_due(self, step: int) -> bool:
        """Whether the subset is to be selected now (model.cp_head.trunk_copy "subset", not selected yet, step >= trunk_select_step)."""
        c = self.cfg
        return (hasattr(self, "n_trunk") and c.trunk_copy == "subset" and step >= c.trunk_select_step
                and len(self.active_heads()) == self.n_trunk)

    @torch.no_grad()
    def select_trunk_subset(self) -> list:
        """The subset's selection (model.cp_head.trunk_copy "subset"): the trunk_k heads with the largest learned logits, ties to the
        lower id (a stable sort), written into trunk_keep in place (the state dict keeps it). Returns the selected ids, ascending."""
        c = self.cfg
        assert hasattr(self, "n_trunk") and c.trunk_copy == "subset", "select_trunk_subset is the trunk copy's subset form"
        assert len(self.active_heads()) == self.n_trunk, "the subset is selected once"
        logits = self.trunk_logits.detach().double().cpu()
        order = sorted(range(self.n_trunk), key=lambda h: (-float(logits[h]), h))
        pick = sorted(order[: c.trunk_k])
        keep = torch.zeros(self.n_trunk, dtype=torch.bool)
        keep[torch.tensor(pick)] = True
        self.trunk_keep.copy_(keep.to(self.trunk_keep.device))
        return pick


class PointerMix(nn.Module):
    """cp-mix's pointer inputs (model.cp_head.pointer_input "mix"; module docstring): read_q and read_k are R1-entry's
    reader (attn_res.DepthQuery: a (d,) pseudo-query of zeros and an RMSNorm on its keys, one softmax over the sources),
    norm_q and norm_k the reads' own RMSNorms (gain 1 + w, w zeros), as a block's sublayer norms its read. forward(sources)
    = (norm_q(depth_mix(sources, read_q)), norm_k(depth_mix(sources, read_k))): attn_res.depth_mix, R1-entry's fast-path
    read, on [b_0 .. b_N]. The trainer compiles this module in place on CUDA (Trainer: the CP head's setup)."""

    def __init__(self, d: int, eps: float):
        super().__init__()
        from ced.model.attn_res import DepthQuery

        self.read_q = DepthQuery(d, eps)
        self.read_k = DepthQuery(d, eps)
        self.norm_q = RMSNorm(d, eps)
        self.norm_k = RMSNorm(d, eps)

    def forward(self, sources: list) -> tuple[torch.Tensor, torch.Tensor]:
        from ced.model.attn_res import depth_mix

        return self.norm_q(depth_mix(sources, self.read_q)), self.norm_k(depth_mix(sources, self.read_k))


class HeadsMix(nn.Module):
    """The pointer's heads' inputs (model.cp_head.n_heads > 1 with pointer_input "mix"; module docstring, "THE POINTER'S HEADS"):
    one PointerMix a head (heads.<h>: its own two readers and two norms, zeros at the start, so every head's x^q_h = x^k_h = h at
    step 0, as cp-mix's). forward(sources) = (x^q, x^k), each (B, L, H, D): head h's reads on axis -2."""

    def __init__(self, d: int, eps: float, heads: int):
        super().__init__()
        self.heads = nn.ModuleList(PointerMix(d, eps) for _ in range(heads))

    def forward(self, sources: list) -> tuple[torch.Tensor, torch.Tensor]:
        reads = [m(sources) for m in self.heads]
        return torch.stack([r[0] for r in reads], dim=-2), torch.stack([r[1] for r in reads], dim=-2)


@dataclass
class PointerInputs:
    """What the pointer's query and key read in one forward (model.cp_head.pointer_input other than "final";
    ModelOutput.cp_pointer; module docstring): q and k (B, L, d_model) under "mix" (x^q and x^k), (B, L, head_dim) under
    "shared" (the shared head's QK-normed query and key). Under "shared" both are None in a forward whose shared layer's
    attention was taken out (the residual diagnostics' skips through the reference stack): the pointer has no scores
    there, and head_token_ce scores p_LM alone (lambda = 0)."""

    q: torch.Tensor | None  # (B, L, H, d_model) under "mix" with n_heads H > 1 (HeadsMix: head h's read on axis -2)
    k: torch.Tensor | None


def pointer_sources(head: "CopyPasteHead", hidden: torch.Tensor, pointer: PointerInputs | None):
    """(the query input, the key input) of the pointer: (hidden, hidden) under pointer_input "final" (pointer must be
    None), else (pointer.q, pointer.k), checked: a forward's ModelOutput.cp_pointer is required (ValueError by name). A head
    of the trunk copy has no pointer inputs: refused by name (every reader that forms the pointer's q and k refuses it so)."""
    head._no_trunk("pointer_sources")
    form = head.cfg.pointer_input
    if form == "final":
        if pointer is not None:
            raise ValueError("pointer inputs were given to a Copy-Paste head whose pointer reads h (model.cp_head."
                             "pointer_input 'final')")
        return hidden, hidden
    if pointer is None:
        raise ValueError(f"this Copy-Paste head's pointer reads {DESCRIBE[form]} (model.cp_head.pointer_input {form!r}): "
                         "pass the forward's ModelOutput.cp_pointer as pointer=")
    if pointer.q is None:
        raise ValueError("the shared head's layer was taken out of this forward (a skip of its attention): the pointer has "
                         "no scores here")
    assert pointer.q.shape[:2] == hidden.shape[:2] == pointer.k.shape[:2], \
        f"pointer inputs {tuple(pointer.q.shape)} / {tuple(pointer.k.shape)} for hidden states {tuple(hidden.shape)}"
    return pointer.q, pointer.k


def trunk_pointer(head: "CopyPasteHead", pointer, targets: torch.Tensor | None = None):
    """The forward's cp_trunk.TrunkInputs of a trunk-copy head (ModelOutput.cp_pointer), checked: required, and of the forward whose
    queries the caller reads (ValueError by name)."""
    from ced.model.cp_trunk import TrunkInputs

    if not isinstance(pointer, TrunkInputs):
        raise ValueError("this Copy-Paste head copies from the trunk's own heads (model.cp_head.trunk_copy): pass the forward's "
                         "ModelOutput.cp_pointer (cp_trunk.TrunkInputs) as pointer=")
    if targets is not None:
        assert targets.shape == pointer.q_idx.shape, \
            f"targets {tuple(targets.shape)} for a forward of {tuple(pointer.q_idx.shape)} queries"
    return pointer


def pointer_inputs(model: nn.Module, ctx, caches: list) -> tuple[PointerInputs, list]:
    """The forward's PointerInputs and its layers' caches (HybridLM.forward, after the stack; pointer_input "mix" or
    "shared"). "mix": the stack's sources [b_0 .. b_N] from ctx.memo (SOURCES_KEY, popped) through cp_head.mix. "shared":
    the shared layer's key set's cp_q and cp_k (attention.py), which are then dropped from that cache entry (a cache keeps
    keys only); a layer whose attention was taken out (its cache entry None) gives PointerInputs(None, None)."""
    head = model.cp_head
    if head.cfg.pointer_input == "mix":
        sources = ctx.memo.pop(SOURCES_KEY, None)
        if sources is None:
            raise RuntimeError("model.cp_head.pointer_input 'mix': the stack handed over no sources (the last layer's "
                               "attn_res.keep_sources)")
        if head.cfg.train_mode == "addon":
            # add-on CP (ced/model/cp_aux.py, "ADD-ON CP"): the readers read the sources detached, so no gradient of the mixture loss
            # reaches a trunk tensor through the pointer (the values are the same: at evaluation the head is cp-mix's)
            sources = [s.detach() for s in sources]
        q, k = head.mix(sources)
        return PointerInputs(q, k), caches
    i = head.shared_layer
    kc = caches[i]
    if kc is None:
        return PointerInputs(None, None), caches
    assert kc.cp_q is not None and kc.cp_k is not None, "the shared layer returned no pointer query and key"
    out = list(caches)
    out[i] = dataclasses.replace(kc, cp_q=None, cp_k=None)
    return PointerInputs(kc.cp_q, kc.cp_k), out


def own_generator() -> torch.Generator:
    return torch.Generator().manual_seed((torch.initial_seed() + OWN_SEED_OFFSET) % (1 << 63))


def attach_cp_head(model: nn.Module) -> None:
    """model.cp_head = CopyPasteHead, registered after every other tensor of the model and drawn from own_generator()
    (the global RNG is not touched): the base tensors start as in the model without the key, bit for bit. The draw
    follows model.cp_head.init (CopyPasteHead.reset_own).
    pointer_input "mix": R1-entry's last layer is told to hand over its output read's sources (attn_res.EntryLayer.
    keep_sources); "shared": the shared layer's attention is told which head the pointer reads (GatedAttention.cp_query,
    (head, shared_ssmax)). Both are plain attributes (no tensor, no random number); under "final" nothing is set."""
    cfg = model.cfg
    c = cfg.cp_head
    from ced.model.cp_trunk import n_trunk_heads, trunk_on

    if trunk_on(c):  # the trunk copy (module docstring, "THE TRUNK COPY"): every global layer hands its queries over
        for i in cfg.encoder.global_layers():
            model.encoder.layers[i].mixer.trunk_copy = True
            # under model.cp_head.kernel "fused" or "fused2" also the read's log-sum-exp (the fused copy kernels' offset; flex reads
            # only)
            model.encoder.layers[i].mixer.trunk_lse = c.kernel in ("fused", "fused2")
        head = CopyPasteHead(cfg.d_model, c, eps=cfg.norm_eps, ref_len=cfg.encoder.attention.temperature_ref_len,
                             n_trunk=n_trunk_heads(cfg))
        head.reset_own(own_generator(), cfg.init_std, c.init)
        model.cp_head = head
        return
    d_shared = None
    if c.pointer_input == "shared":
        layer = shared_layer_index(cfg)
        att = model.encoder.layers[layer].mixer
        d_shared = att.cfg.head_dim
        att.cp_query = (c.shared_head, c.shared_ssmax)
    if c.train_mode == "aux":  # cp-aux (ced/model/cp_aux.py): the pointer without the <copy> row, cp-mix's draws
        from ced.model.cp_aux import AuxPointerHead as head_class
    else:
        head_class = CopyPasteHead
    head = head_class(cfg.d_model, c, eps=cfg.norm_eps, d_shared=d_shared,
                      ref_len=cfg.encoder.attention.temperature_ref_len)
    head.reset_own(own_generator(), cfg.init_std, c.init)
    if c.pointer_input == "shared":
        head.shared_layer = layer
    if c.pointer_input == "mix":
        model.encoder.layers[-1].attn_res.keep_sources = True
    model.cp_head = head


def cp_head_of(model: nn.Module) -> CopyPasteHead | None:
    """The head of a model (unwrapped: torch.compile's _orig_mod), or None. None also for cp-aux's head (model.cp_head.train_mode
    "aux"; ced/model/cp_aux.py): its pointer is dropped at evaluation, so every scoring path that asks reads the model as one without
    a head (p_LM alone). The trainer reads model.cp_head itself."""
    inner = getattr(model, "_orig_mod", model)
    head = getattr(inner, "cp_head", None)
    return None if head is not None and head.cfg.train_mode == "aux" else head


def score_mode(model: nn.Module) -> str:
    """'mixture' or 'lm' for a CP model: CED_CP_SCORE if set, else the config's."""
    inner = getattr(model, "_orig_mod", model)
    return env_score() or inner.cfg.cp_head.score


# ---------------------------------------------------------------------------------------------------------------------
# The length rule and the inference-only probe (module docstring)


def active_probe(head: "CopyPasteHead") -> str | None:
    """The probe condition in force for this head: head.probe when set (None or "none": off), else CED_CP_PROBE. Checked:
    a condition is refused (ValueError, by name) on a head trained with the length keys, a band, a pointer that does not
    read h, or with gradients enabled (inference only: no training step reads it)."""
    probe = head.probe if head.probe is not None else env_probe()
    if probe is None or probe == "none":
        return None
    if probe not in PROBES:
        raise ValueError(f"CopyPasteHead.probe {probe!r}: one of {PROBES} (or None / 'none')")
    c = head.cfg
    if c.temperature != "none" or c.sink_count or c.window or c.pointer_input != "final":
        raise ValueError(f"the inference-only probe {probe!r} is built for a head trained without the length rule, without a "
                         f"band and with pointer_input 'final' (this head: temperature {c.temperature!r}, sink_count "
                         f"{c.sink_count!r}, window {c.window!r}, pointer_input {c.pointer_input!r}): nothing tests the pair")
    if torch.is_grad_enabled():
        raise RuntimeError(f"the probe {probe!r} ({ENV_PROBE} or CopyPasteHead.probe) is inference only: gradients are "
                           "enabled (a training step, or a scorer outside torch.no_grad)")
    return probe


def length_on(head: "CopyPasteHead") -> bool:
    """The length rule or a probe changes this head's pointer (False: every path runs its statements of before the keys)."""
    c = head.cfg
    return c.temperature != "none" or c.sink_count or active_probe(head) is not None


def doc_counts(doc_ids: torch.Tensor) -> torch.Tensor:
    """n_t = pos_t + 1 (B, L) int64 for the forward's own positions: the trunk's document position (ced/model/sequence.py
    sequence_context, the count ssmax_factor reads)."""
    from ced.model.sequence import sequence_context

    return sequence_context(doc_ids, None)[0].pos + 1


def context_counts(keys: "CopyKeys", q_idx: torch.Tensor, q_doc: torch.Tensor, window: int = 0) -> torch.Tensor:
    """n_t (B, C) int64 for queries over `keys` (the copy cache, or any key set): the context mask's count (_context), which
    equals doc_counts where the keys are every earlier position of the row (tests/test_cp_length_robust.py)."""
    return _context(keys, q_idx, q_doc, window).sum(-1)


def length_terms(head: "CopyPasteHead", q: torch.Tensor, n: torch.Tensor | None):
    """(q_ctx, q_sink, sink_add) of the length rule (module docstring) for the pointer's queries q (B, C, d) (after
    queries()) with context counts n (B, C) (doc_counts or context_counts; None only when length_on is False):
        q_ctx     the query of the context logits e_tau (and of the sink unless q_sink is given)
        q_sink    None (the sink reads q_ctx), or the query of the sink's logit (probe "A'": the unscaled q)
        sink_add  None, or (B, C) added to e_sink after the scale
    Off: (q, None, None), q itself (no op). The factors are in q's dtype; n is clamped at 1 (a real query sees itself)."""
    c = head.cfg
    probe = active_probe(head)
    if probe is None and c.temperature == "none" and not c.sink_count:
        return q, None, None
    from ced.model.attention import key_count_factor

    ref = head.ref_len
    n = n.clamp_min(1)
    q_ctx, q_sink, add = q, None, None
    if probe is None:
        if c.temperature == "ssmax":
            q_ctx = q * key_count_factor(n, ref, q.dtype)[..., None]
        if c.sink_count:
            add = torch.log(n.to(q.dtype) / ref)
        return q_ctx, q_sink, add
    if probe in ("A", "A'", "A+B"):
        q_ctx = q * key_count_factor(n, ref, q.dtype).clamp_min(1.0)[..., None]
        if probe == "A'":
            q_sink = q
    if probe in ("B", "A+B"):
        add = torch.log(n.to(q.dtype) / ref).clamp_min(0.0)
    return q_ctx, q_sink, add


def sink_logit(q: torch.Tensor, k_sink: torch.Tensor, scale: float, q_sink: torch.Tensor | None = None,
               sink_add: torch.Tensor | None = None) -> torch.Tensor:
    """e_sink (B, C) = (q_sink or q) . k_sink * scale (+ sink_add): chunk_terms' and chunk_alpha's statement, the length
    rule's inputs at None giving the statement of before the keys."""
    e_sink = torch.matmul(q if q_sink is None else q_sink, k_sink) * scale
    return e_sink if sink_add is None else e_sink + sink_add


def pointer_queries(head: "CopyPasteHead", q: torch.Tensor, keys: "CopyKeys", q_idx: torch.Tensor, q_doc: torch.Tensor):
    """length_terms for queries over a key set (scoring, cached generation, the readings): n from the context mask's
    count; (q, None, None) without a count when the rule is off."""
    if not length_on(head):
        return q, None, None
    return length_terms(head, q, context_counts(keys, q_idx, q_doc, head.cfg.window))


# ---------------------------------------------------------------------------------------------------------------------
# The key side of a forward (or of a cache) and the chunked reference


@dataclass
class CopyKeys:
    """Keys and their metadata, (B, S, d_c) and (B, S) each: the positions a query may copy from. In a training or
    scoring forward the chunk's own positions; in cached generation every position so far (CopyCache)."""

    k: torch.Tensor
    tok: torch.Tensor
    doc: torch.Tensor
    idx: torch.Tensor

    def extend(self, other: "CopyKeys") -> "CopyKeys":
        return CopyKeys(*(torch.cat([a, b], dim=1) for a, b in
                          ((self.k, other.k), (self.tok, other.tok), (self.doc, other.doc), (self.idx, other.idx))))


def _neg_inf(like: torch.Tensor) -> torch.Tensor:
    return torch.full((), float("-inf"), dtype=like.dtype, device=like.device)


def _context(keys: CopyKeys, q_idx: torch.Tensor, q_doc: torch.Tensor, window: int = 0) -> torch.Tensor:
    """ctx[b, i, s]: key s is a context position of query i (idx_s <= idx_i and the same document; with a window W >= 1
    also idx_i - idx_s < W, model.cp_head.window). (B, C, S) bool."""
    ctx = (keys.idx[:, None, :] <= q_idx[:, :, None]) & (keys.doc[:, None, :] == q_doc[:, :, None])
    if window:
        ctx = ctx & (q_idx[:, :, None] - keys.idx[:, None, :] < window)
    return ctx


def chunk_terms(q, k, k_sink, k_tok, k_doc, k_idx, q_idx, q_doc, q_tgt, scale: float, window: int = 0, q_sink=None,
                sink_add=None):
    """(log s, log(1 - s), log p_CP(target)), each (B, C), for C queries q (B, C, d_c) over the keys k (B, S, d_c).

    The rows of the module docstring: ctx and match by comparisons, torch.where for the masking, logsumexp for both
    sums. A query always has itself in ctx (idx_t <= idx_t, its own document), so LSE_ctx is finite. A query with no
    matching key reads a dummy row of zeros (finite) whose result is replaced by the constant -inf: its gradient is 0,
    never NaN. Every op has a fixed shape. window (model.cp_head.window): 0, every earlier position of the document (the
    ops as they were before the key); W >= 1, ctx also needs idx_t - idx_tau < W. q_sink (B, C, d_c) and sink_add (B, C):
    the length rule's inputs (length_terms; None: the sink's logit is q . k_sink * scale, the statement of before them)."""
    neg = _neg_inf(q)
    s = torch.matmul(q, k.transpose(1, 2)) * scale  # (B, C, S)
    e_sink = sink_logit(q, k_sink, scale, q_sink, sink_add)  # (B, C)
    ctx = (k_idx[:, None, :] <= q_idx[:, :, None]) & (k_doc[:, None, :] == q_doc[:, :, None])
    if window:
        ctx = ctx & (q_idx[:, :, None] - k_idx[:, None, :] < window)
    match = ctx & (k_tok[:, None, :] == q_tgt[:, :, None])
    lse_ctx = torch.logsumexp(torch.where(ctx, s, neg), dim=-1)
    has = match.any(dim=-1)
    fill = torch.where(has, neg, torch.zeros((), dtype=s.dtype, device=s.device))[..., None]
    lse_match = torch.logsumexp(torch.where(match, s, fill), dim=-1)
    log_pcp = torch.where(has, lse_match - lse_ctx, neg)
    lse_all = torch.logaddexp(e_sink, lse_ctx)
    return e_sink - lse_all, lse_ctx - lse_all, log_pcp


def or_combine(log_pi, log_s_h, log_1ms_h, log_pcp_h):
    """The "or" pointer's (log s, log(1 - s), log p_CP(target)) (...) from the heads' own (..., H) and the router's log pi (..., H)
    (module docstring, "THE POINTER'S HEADS"). Every head shares the match set (the same keys and tokens), so a row has a match
    for all heads or for none; a row without one is the constant -inf, computed from a dummy 0 under torch.where (no NaN in a
    gradient). At H = 1 (log pi = 0) it returns the head's own three, bit for bit."""
    a = log_pi + log_1ms_h
    log_s = torch.logsumexp(log_pi + log_s_h, dim=-1)
    log_1ms = torch.logsumexp(a, dim=-1)
    log_w = torch.log_softmax(a, dim=-1)
    has = torch.isfinite(log_pcp_h[..., 0])
    safe = torch.where(has[..., None], log_pcp_h, torch.zeros((), dtype=log_pcp_h.dtype, device=log_pcp_h.device))
    log_pcp = torch.where(has, torch.logsumexp(log_w + safe, dim=-1), _neg_inf(log_pcp_h))
    return log_s, log_1ms, log_pcp


def chunk_terms_or(q, k, k_sink, k_tok, k_doc, k_idx, q_idx, q_doc, q_tgt, scale: float, log_pi):
    """chunk_terms of the "or" pointer (module docstring, "THE POINTER'S HEADS"): q (B, C, H, d_c), k (B, S, H, d_c), k_sink (H,
    d_c), log_pi (B, C, H); the rest as chunk_terms. Head h is chunk_terms on its own slices (the one-head reference, its masks
    and its -inf rule), then or_combine. (log s, log(1 - s), log p_CP(target)), each (B, C)."""
    parts = [chunk_terms(q[:, :, h], k[:, :, h], k_sink[h], k_tok, k_doc, k_idx, q_idx, q_doc, q_tgt, scale)
             for h in range(q.shape[2])]
    return or_combine(log_pi, *(torch.stack([p[i] for p in parts], dim=-1) for i in range(3)))


def block_function(head: "CopyPasteHead"):
    """The eager block function of the head's training terms: chunk_terms_or under combine "or", else chunk_terms (the trainer
    compiles this one on CUDA)."""
    return chunk_terms_or if head.cfg.combine == "or" else chunk_terms


def cp_terms(head: CopyPasteHead, hidden: torch.Tensor, tokens: torch.Tensor, targets: torch.Tensor,
             doc_ids: torch.Tensor, chunk: int | None = None, checkpoint: bool = False,
             keys: CopyKeys | None = None, q_idx: torch.Tensor | None = None, fn=None,
             pointer: PointerInputs | None = None):
    """(log s, log(1 - s), log p_CP(target)), each (B, L), for every position of a forward: hidden (B, L, D) the final
    norm's output, tokens / targets / doc_ids (B, L). keys: the key side (default: this forward's own positions,
    idx = 0 .. L - 1); q_idx: the queries' absolute indices (default 0 .. L - 1). chunk: query positions per block
    (default the config's query_chunk; need not divide L). checkpoint: each block under activation checkpointing (the
    training forward), so a block's [B, C, S] tensors exist only while its forward or backward runs. fn: the block
    function (default chunk_terms; the trainer passes a torch.compile'd chunk_terms on CUDA). pointer: the forward's
    PointerInputs (ModelOutput.cp_pointer) under pointer_input "mix" / "shared", None under "final" (pointer_sources).

    With the forward's own keys, the block of queries a .. a + C - 1 reads the keys 0 .. a + C - 1 only: a later key is
    never in a query's context (idx_tau <= idx_t), so the result is the same and the work is about half. With a window
    W (model.cp_head.window >= 1) it reads the keys max(0, a - W + 1) .. a + C - 1 only (an earlier key is outside every
    query's band). The block shapes are a function of (L, C, W) alone, never of the data.
    The trunk copy (model.cp_head.trunk_copy; module docstring, "THE TRUNK COPY"): cp_trunk.trunk_terms of the forward's TrunkInputs
    (pointer), its own block function (fn) and block size (chunk, default cp_trunk.TRUNK_CHUNK); hidden and tokens are read from it."""
    if hasattr(head, "n_trunk"):
        from ced.model.cp_trunk import trunk_terms

        return trunk_terms(head, trunk_pointer(head, pointer, targets), targets, chunk=chunk, checkpoint=checkpoint, fn=fn)
    B, L = tokens.shape
    q_src, k_src = pointer_sources(head, hidden, pointer)
    q = head.queries(q_src)
    own = keys is None
    if own:
        idx = torch.arange(L, device=tokens.device).expand(B, L)
        keys = CopyKeys(head.keys(k_src), tokens, doc_ids, idx)
    if q_idx is None:
        q_idx = torch.arange(L, device=tokens.device).expand(B, L)
    fn = block_function(head) if fn is None else fn
    k_sink = head.k_sink.to(q.dtype)
    C = chunk or head.cfg.query_chunk
    W = head.cfg.window
    q_sink = sink_add = None
    rule = length_on(head)
    if rule:  # the length rule (module docstring): n_t per query, then the factor on q and the add on e_sink (W is 0 here)
        if own:
            n = doc_counts(doc_ids)
        else:
            n = torch.cat([context_counts(keys, q_idx[:, a : a + C], doc_ids[:, a : a + C]) for a in range(0, L, C)], 1)
        q, q_sink, sink_add = length_terms(head, q, n)
    if head.cfg.combine == "or":
        # the "or" pointer (module docstring, "THE POINTER'S HEADS"): every head over the same blocks, then the router's mixture;
        # no band, no sink add and no probe here (check_cp_head, active_probe), so q_sink and sink_add are None
        assert q_sink is None and sink_add is None and not W, "the 'or' pointer has no band, sink add or probe"
        H, d_c = head.cfg.n_heads, head.cfg.d_c
        log_pi = head.route(hidden)
        parts = []
        for a in range(0, L, C):
            S = min(a + C, L) if own else keys.k.shape[1]
            qa = q[:, a : a + C]
            args = (qa.unflatten(-1, (H, d_c)), keys.k[:, :S].unflatten(-1, (H, d_c)), k_sink.view(H, d_c), keys.tok[:, :S],
                    keys.doc[:, :S], keys.idx[:, :S], q_idx[:, a : a + C], doc_ids[:, a : a + C], targets[:, a : a + C],
                    head.scale, log_pi[:, a : a + C])
            if checkpoint and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint as ckpt

                parts.append(ckpt(fn, *args, use_reentrant=False))
            else:
                parts.append(fn(*args))
        return tuple(torch.cat([p[i] for p in parts], dim=1) for i in range(3))
    parts = []
    for a in range(0, L, C):
        S = min(a + C, L) if own else keys.k.shape[1]
        lo = max(0, a - W + 1) if (W and own) else 0
        args = (q[:, a : a + C], keys.k[:, lo:S], k_sink, keys.tok[:, lo:S], keys.doc[:, lo:S], keys.idx[:, lo:S],
                q_idx[:, a : a + C], doc_ids[:, a : a + C], targets[:, a : a + C], head.scale) + ((W,) if W else ())
        if rule:
            args = args + ((0,) if not W else ()) + (None if q_sink is None else q_sink[:, a : a + C],
                                                     None if sink_add is None else sink_add[:, a : a + C])
        if checkpoint and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint as ckpt

            parts.append(ckpt(fn, *args, use_reentrant=False))
        else:
            parts.append(fn(*args))
    return tuple(torch.cat([p[i] for p in parts], dim=1) for i in range(3))


def cp_terms_naive(head: CopyPasteHead, hidden, tokens, targets, doc_ids, pointer: PointerInputs | None = None):
    """The T x T reference of the tests: every score of a row at once (B, L, L), the spec's softmaxes taken explicitly
    over that matrix, in log space. (B, L) each:

        log s        log_softmax over {sink} u ctx (the sink first, then the context positions), at the sink
        log(1 - s)   logsumexp over ctx of that same log_softmax (the context's share, never 1 - s by subtraction)
        log alpha    log_softmax over ctx alone (no sink)
        log p_CP     logsumexp over match of log alpha; -inf where nothing matches

    A masked entry is -inf before each log_softmax and logsumexp, so its weight and its gradient are exactly 0. A row
    without a match reads a row of zeros (finite) whose result is replaced by the constant -inf (the double where): its
    gradient is 0, never NaN. It replaces a probability form (softmax, sum, then log) that was wrong in float32 at
    trained-looking scales: 1 - s by subtraction lost digits when s is near 1, and a p_CP under 2^-126 had a gradient 1
    / p_CP that overflowed to inf, which the match mask's zeros turned into NaN. The trunk copy:
    cp_trunk.trunk_terms_naive (every head's full softmax, the masses summed by their definitions)."""
    if hasattr(head, "n_trunk"):
        from ced.model.cp_trunk import trunk_terms_naive

        return trunk_terms_naive(head, trunk_pointer(head, pointer, targets), targets)
    B, L = tokens.shape
    q_src, k_src = pointer_sources(head, hidden, pointer)
    q, k = head.queries(q_src), head.keys(k_src)
    t = torch.arange(L, device=tokens.device)
    ctx = (t[None, None, :] <= t[None, :, None]) & (doc_ids[:, None, :] == doc_ids[:, :, None])
    if head.cfg.window:  # model.cp_head.window: the band, idx_t - idx_tau < W
        ctx = ctx & (t[None, :, None] - t[None, None, :] < head.cfg.window)
    q_sink = sink_add = None
    if length_on(head):  # the length rule, n_t counted on this matrix's own context mask
        q, q_sink, sink_add = length_terms(head, q, ctx.sum(-1))
    if head.cfg.combine == "or":
        return _naive_or(head, q, k, hidden, tokens, targets, ctx)
    e = torch.matmul(q, k.transpose(1, 2)) * head.scale  # (B, L, L)
    e_sink = sink_logit(q, head.k_sink.to(q.dtype), head.scale, q_sink, sink_add)  # (B, L)
    match = ctx & (tokens[:, None, :] == targets[:, :, None])
    neg = _neg_inf(e)
    e_ctx = torch.where(ctx, e, neg)
    log_with_sink = torch.log_softmax(torch.cat([e_sink[..., None], e_ctx], dim=-1), dim=-1)  # sink first, then 1..t
    log_s = log_with_sink[..., 0]
    log_1ms = torch.logsumexp(torch.where(ctx, log_with_sink[..., 1:], neg), dim=-1)
    log_alpha = torch.log_softmax(e_ctx, dim=-1)
    has = match.any(-1)
    fill = torch.where(has, neg, torch.zeros((), dtype=e.dtype, device=e.device))[..., None]
    log_pcp = torch.where(has, torch.logsumexp(torch.where(match, log_alpha, fill), dim=-1), neg)
    return log_s, log_1ms, log_pcp


def _naive_or(head: CopyPasteHead, q, k, hidden, tokens, targets, ctx):
    """cp_terms_naive of the "or" pointer, in the other order from cp_terms (module docstring, "THE POINTER'S HEADS"): each head's
    log_softmax over {sink} u ctx (B, L, 1 + L), the MIXTURE over {sink} u ctx first, log P = logsumexp_h(log pi_h + that), then
    the spec's reads of P: log s = log P(sink), log(1 - s) = logsumexp over ctx of log P(tau), log p_CP = logsumexp over match of
    log P(tau) - log(1 - s); -inf where nothing matches (the double where)."""
    H, d_c = head.cfg.n_heads, head.cfg.d_c
    B, L = tokens.shape
    qh, kh = q.unflatten(-1, (H, d_c)), k.unflatten(-1, (H, d_c))
    k_sink = head.k_sink.to(q.dtype).view(H, d_c)
    neg = _neg_inf(q)
    log_pi = head.route(hidden)  # (B, L, H)
    live = torch.cat([torch.ones_like(ctx[..., :1]), ctx], dim=-1)  # {sink} u ctx: the entries of the pointer's distribution
    zero = torch.zeros((), dtype=q.dtype, device=q.device)
    per = []
    for h in range(H):
        e = torch.matmul(qh[:, :, h], kh[:, :, h].transpose(1, 2)) * head.scale
        e_sink = torch.matmul(qh[:, :, h], k_sink[h]) * head.scale
        lw = torch.log_softmax(torch.cat([e_sink[..., None], torch.where(ctx, e, neg)], dim=-1), dim=-1)  # sink first
        per.append(log_pi[..., h, None] + torch.where(live, lw, zero))  # a dummy 0 off the context: no NaN in a gradient
    # (B, L, 1 + L): the pointer's distribution over {sink} u ctx, -inf off the context
    log_p = torch.where(live, torch.logsumexp(torch.stack(per, dim=0), dim=0), neg)
    log_s = log_p[..., 0]
    log_ctx = torch.where(ctx, log_p[..., 1:], neg)
    log_1ms = torch.logsumexp(log_ctx, dim=-1)
    match = ctx & (tokens[:, None, :] == targets[:, :, None])
    has = match.any(-1)
    fill = torch.where(has, neg, torch.zeros((), dtype=q.dtype, device=q.device))[..., None]
    log_pcp = torch.where(has, torch.logsumexp(torch.where(match, log_ctx, fill), dim=-1) - log_1ms, neg)
    return log_s, log_1ms, log_pcp


def pointer_terms(head: CopyPasteHead, hidden, tokens, targets, doc_ids, pointer: PointerInputs | None = None):
    """(log s, log(1 - s), log p_CP(target), z_copy), each (B, L), at every position of a forward with its own keys:
    model.cp_head.kernel "reference", cp_terms and copy_logit (this file); "fused", ced.kernels.cp_pointer.fused_terms
    (the stacked GEMM, the pointer kernels). The scoring paths' entry point (head_token_ce); the trainer calls each path
    itself (Trainer._cp_loss)."""
    if head.cfg.kernel == "fused" and not hasattr(head, "n_trunk"):  # (pointer_input "final" only: check_cp_head; the trunk copy's
        # fused path is cp_terms' own: cp_trunk.trunk_terms)
        from ced.kernels.cp_pointer import fused_terms

        return fused_terms(head, hidden, tokens, targets, doc_ids)
    return (*cp_terms(head, hidden, tokens, targets, doc_ids, pointer=pointer), head.copy_logit(hidden))


# ---------------------------------------------------------------------------------------------------------------------
# The mixture


def mixture_log_prob(lse_v, z_y, z_c, log_s, log_1ms, log_pcp):
    """log p(y) of the mixture from per-position (LSE_V, z_y, z_copy, log s, log(1 - s), log p_CP(y)): the module
    docstring's rows, autograd-differentiable (the reference of the fused head's closed-form gradient)."""
    lse_all = torch.logaddexp(lse_v, z_c)
    log_c = z_c - lse_all
    log_1mc = lse_v - lse_all
    log_lam = log_c + log_1ms
    log_1mlam = torch.logaddexp(log_1mc, log_c + log_s)
    return torch.logaddexp(log_1mlam + z_y - lse_v, log_lam + log_pcp)


def mixture_parts(lse_v, z_c, log_s, log_1ms):
    """(log lambda, log(1 - lambda), log c) per position."""
    lse_all = torch.logaddexp(lse_v, z_c)
    log_c = z_c - lse_all
    log_1mc = lse_v - lse_all
    return log_c + log_1ms, torch.logaddexp(log_1mc, log_c + log_s), log_c


@torch.no_grad()
def mixture_grads(lse_v, z_y, z_c, log_s, log_1ms, log_pcp, a):
    """The loss a * (-log p(y)) per position and its gradient, in closed form (no autograd), for the fused head:

        ce          -log p(y)
        g_lse       d loss / d LSE_V            (the z-loss term is the caller's: 2 b LSE_V)
        g_zy        -d loss / d z_y             (the one-hot coefficient: the gradient at the target logit is -g_zy)
        g_zc, g_ls, g_l1s, g_lp                 d loss / d z_copy, log s, log(1 - s), log p_CP(y)

    With wA = p_LM-branch share of p(y), wB = 1 - wA, c = exp(log c), r = (1 - c) / (1 - lambda), 1 - r = c s / (1 - lambda):
        d log(1 - lambda) / d LSE_V = r c - (1 - r)(1 - c);   d log(1 - lambda) / d z_c = -r c + (1 - r)(1 - c)
        g_lse = -a [wA (r c - (1 - r)(1 - c) - 1) - wB (1 - c)];   g_zy = a wA
        g_zc = -a [wA (-r c + (1 - r)(1 - c)) + wB (1 - c)];  g_ls = -a wA (1 - r);  g_l1s = g_lp = -a wB
    Written so that the limit lambda = 0 (z_copy = -inf) is exact: g_lse = a and g_zy = a bit for bit, every CP
    gradient 0, ce = LSE_V - z_y (tests/test_cp_head.py)."""
    lse_all = torch.logaddexp(lse_v, z_c)
    log_c = z_c - lse_all
    log_1mc = lse_v - lse_all
    log_lam = log_c + log_1ms
    log_cs = log_c + log_s
    log_1mlam = torch.logaddexp(log_1mc, log_cs)
    A = log_1mlam + z_y - lse_v
    Bc = log_lam + log_pcp
    log_p = torch.logaddexp(A, Bc)
    wA, wB = torch.exp(A - log_p), torch.exp(Bc - log_p)
    c, one_mc = torch.exp(log_c), torch.exp(log_1mc)
    r, one_mr = torch.exp(log_1mc - log_1mlam), torch.exp(log_cs - log_1mlam)
    g_lse = -a * (wA * (r * c - one_mr * one_mc - 1) - wB * one_mc)
    g_zy = a * wA
    g_zc = -a * (wA * (one_mr * one_mc - r * c) + wB * one_mc)
    g_ls = -a * wA * one_mr
    g_lb = -a * wB
    return {"ce": -log_p, "g_lse": g_lse, "g_zy": g_zy, "g_zc": g_zc, "g_ls": g_ls, "g_l1s": g_lb, "g_lp": g_lb,
            "log_lam": log_lam, "log_1mlam": log_1mlam}


# ---------------------------------------------------------------------------------------------------------------------
# Scoring: p over V, and the CE of the mixture


def chunk_alpha(q, keys: CopyKeys, k_sink, q_idx, q_doc, scale: float, window: int = 0, q_sink=None, sink_add=None):
    """(alpha (B, C, S) over the context positions, log s (B, C), log(1 - s) (B, C), ctx (B, C, S)). window: the
    pointer's reach (model.cp_head.window; _context). q_sink, sink_add: the length rule's inputs (chunk_terms;
    pointer_queries gives them with q)."""
    neg = _neg_inf(q)
    e = torch.matmul(q, keys.k.transpose(1, 2)) * scale
    e_sink = sink_logit(q, k_sink, scale, q_sink, sink_add)
    ctx = _context(keys, q_idx, q_doc, window)
    masked = torch.where(ctx, e, neg)
    lse_ctx = torch.logsumexp(masked, dim=-1)
    alpha = torch.exp(masked - lse_ctx[..., None])
    lse_all = torch.logaddexp(e_sink, lse_ctx)
    return alpha, e_sink - lse_all, lse_ctx - lse_all, ctx


def head_alphas(head: CopyPasteHead, q, keys: CopyKeys, k_sink, q_idx, q_doc, h_q):
    """The "or" pointer's heads alone over `keys` (module docstring, "THE POINTER'S HEADS"): q (B, C, H d_c) flat (after the
    length rule), h_q (B, C, D) the final norm's output at the queries (the router's input). Returns (alpha_h (B, C, S, H),
    log s_h, log(1 - s_h), log pi, log w), the last four (B, C, H), and ctx (B, C, S): each head chunk_alpha on its own slices."""
    H, d_c = head.cfg.n_heads, head.cfg.d_c
    qh = q.unflatten(-1, (H, d_c))
    kh = keys.k.unflatten(-1, (H, d_c))
    ks = k_sink.view(H, d_c)
    parts = [chunk_alpha(qh[:, :, h], CopyKeys(kh[:, :, h], keys.tok, keys.doc, keys.idx), ks[h], q_idx, q_doc, head.scale,
                         head.cfg.window) for h in range(H)]
    alpha_h = torch.stack([p[0] for p in parts], dim=-1)
    log_s_h, log_1ms_h = (torch.stack([p[i] for p in parts], dim=-1) for i in (1, 2))
    log_pi = head.route(h_q)
    log_w = torch.log_softmax(log_pi + log_1ms_h, dim=-1)
    return alpha_h, log_s_h, log_1ms_h, log_pi, log_w, parts[0][3]


def pointer_alpha(head: CopyPasteHead, q, keys: CopyKeys, k_sink, q_idx, q_doc, q_sink=None, sink_add=None, h_q=None):
    """chunk_alpha of the head's pointer, whatever its form: (alpha (B, C, S), log s, log(1 - s), ctx). One head and "and" (one
    head of width H d_c): chunk_alpha itself. "or" (module docstring, "THE POINTER'S HEADS"): alpha = sum_h w_h alpha_h, log s =
    logsumexp_h(log pi_h + log s_h), log(1 - s) = logsumexp_h(log pi_h + log(1 - s_h)); h_q (B, C, D), the final norm's output at
    the queries, is the router's input (required)."""
    if head.cfg.combine != "or":
        return chunk_alpha(q, keys, k_sink, q_idx, q_doc, head.scale, head.cfg.window, q_sink, sink_add)
    assert q_sink is None and sink_add is None and h_q is not None, "the 'or' pointer: no sink add or probe; h_q is the router's input"
    alpha_h, log_s_h, log_1ms_h, log_pi, log_w, ctx = head_alphas(head, q, keys, k_sink, q_idx, q_doc, h_q)
    alpha = (alpha_h * torch.exp(log_w)[:, :, None, :]).sum(-1)
    return alpha, torch.logsumexp(log_pi + log_s_h, dim=-1), torch.logsumexp(log_pi + log_1ms_h, dim=-1), ctx


def mixture_logprobs_v(logits_v, z_c, log_s, log_1ms, p_cp):
    """log p over V (n, V) from the LM logits over V (n, V), z_copy (n,), log s, log(1 - s) (n,) and p_CP over V (n, V).
    log space throughout; a zero of p_CP is -inf through a safe log (no NaN in a gradient)."""
    lse_v = torch.logsumexp(logits_v, dim=-1)
    log_lam, log_1mlam, _ = mixture_parts(lse_v, z_c, log_s, log_1ms)
    pos = p_cp > 0
    log_pcp = torch.where(pos, torch.log(torch.where(pos, p_cp, torch.ones((), dtype=p_cp.dtype, device=p_cp.device))),
                          _neg_inf(p_cp))
    return torch.logaddexp((log_1mlam - lse_v)[:, None] + logits_v, log_lam[:, None] + log_pcp)


def logprobs_rows(head: CopyPasteHead, hidden_q, q_idx, q_doc, keys: CopyKeys, weight_v, chunk: int | None = None,
                  q_src: torch.Tensor | None = None):
    """log p over V at the queries: hidden_q (B, P, D) with their absolute indices and documents (B, P), over `keys`.
    Returns (B, P, V) in compute_dtype, chunk by chunk ([B, C, S] alpha, [B, C, V] p_CP). q_src (B, P, .): the pointer's
    query input at the same positions under pointer_input "mix" / "shared" (pointer_sources); None: hidden_q."""
    B, P, _ = hidden_q.shape
    V = weight_v.shape[0]
    dtype = compute_dtype(hidden_q)
    C = chunk or head.cfg.query_chunk
    out = []
    with torch.autocast(device_type=hidden_q.device.type, enabled=False):
        W = weight_v.to(dtype)
        k_sink = head.k_sink.to(dtype)
        for a in range(0, P, C):
            h = hidden_q[:, a : a + C].to(dtype)
            q = head.queries(h if q_src is None else q_src[:, a : a + C])
            q, q_sink, sink_add = pointer_queries(head, q, keys, q_idx[:, a : a + C], q_doc[:, a : a + C])
            alpha, log_s, log_1ms, _ = pointer_alpha(head, q, keys, k_sink, q_idx[:, a : a + C], q_doc[:, a : a + C], q_sink,
                                                     sink_add, h_q=h)
            n = h.shape[1]
            p_cp = torch.zeros(B, n, V, dtype=dtype, device=h.device)
            p_cp.scatter_add_(2, keys.tok[:, None, :].expand(B, n, keys.tok.shape[1]), alpha)
            logits = torch.nn.functional.linear(h, W)
            z_c = head.copy_logit(h)
            lp = mixture_logprobs_v(logits.reshape(B * n, V), z_c.reshape(-1), log_s.reshape(-1), log_1ms.reshape(-1),
                                    p_cp.reshape(B * n, V))
            out.append(lp.reshape(B, n, V))
    return torch.cat(out, dim=1)


def forward_keys(head: CopyPasteHead, hidden, tokens, doc_ids, idx) -> CopyKeys:
    return CopyKeys(head.keys(hidden), tokens, doc_ids, idx)


def head_token_ce(model: nn.Module, hidden: torch.Tensor, targets: torch.Tensor, tokens: torch.Tensor,
                  doc_ids: torch.Tensor, chunk: int = 1024,
                  pointer: PointerInputs | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The scoring paths' per-position (CE, LSE_V) of the model's next-token distribution, from a forward with
    return_hidden (hidden (B, L, D) at every position of the rows; tokens, targets, doc_ids (B, L)).

    A model without a CP head: ced.train.chunked_loss.token_ce(hidden, model.head_weight(), targets), the call every
    path made before, unchanged. A CP model: under score "lm" that same call on the V rows (p_LM alone); under
    "mixture" -log p(target) of the mixture (cp_terms, chunk by chunk, and the reference head's LSE_V and z_y in float32
    or wider), the same number as the mixture's log p over V at the target (tests/test_cp_head.py). pointer: the forward's
    ModelOutput.cp_pointer (pointer_sources). cp-shared in a forward whose shared layer's attention was taken out (the
    residual diagnostics' skips: PointerInputs(None, None)): the pointer has no scores, and the CE is p_LM's (lambda = 0)."""
    from ced.train.chunked_loss import token_ce, token_lse_zy

    inner = getattr(model, "_orig_mod", model)
    head = cp_head_of(inner)
    if head is None:
        return token_ce(hidden, inner.head_weight(), targets, chunk)
    W = inner.lm_head_weight()
    if score_mode(inner) == "lm":
        return token_ce(hidden, W, targets, chunk)
    if head.cfg.pointer_input == "shared" and pointer is not None and pointer.q is None:
        return token_ce(hidden, W, targets, chunk)  # the shared head's layer taken out: no pointer, p = p_LM
    lse, z_y = token_lse_zy(hidden, W, targets, chunk)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        log_s, log_1ms, log_pcp, z_c = pointer_terms(head, hidden, tokens, targets, doc_ids, pointer=pointer)
        logp = mixture_log_prob(lse, z_y, z_c.to(lse.dtype), log_s.to(lse.dtype), log_1ms.to(lse.dtype),
                                log_pcp.to(lse.dtype))
    return -logp, lse


def head_logprobs_at(model: nn.Module, hidden: torch.Tensor, tokens: torch.Tensor, doc_ids: torch.Tensor,
                     index: torch.Tensor, chunk: int | None = None, pointer: PointerInputs | None = None) -> torch.Tensor:
    """log p over V at the positions index (B, P) of the rows (hidden (B, L, D) from a forward with return_hidden;
    tokens, doc_ids (B, L)): (B, P, V), float32 or wider. A CP model only (the paths that call it check cp_head_of
    first). Under score "lm": log_softmax of the LM logits. pointer: the forward's ModelOutput.cp_pointer."""
    inner = getattr(model, "_orig_mod", model)
    head = cp_head_of(inner)
    assert head is not None, "head_logprobs_at is the CP model's path"
    B, L = tokens.shape
    D = hidden.shape[-1]
    h_q = torch.gather(hidden, 1, index[..., None].expand(*index.shape, D))
    W = inner.lm_head_weight()
    dtype = compute_dtype(hidden)
    if score_mode(inner) == "lm":
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return torch.log_softmax(torch.nn.functional.linear(h_q.to(dtype), W.to(dtype)), dim=-1)
    if hasattr(head, "n_trunk"):  # the trunk copy: the copy distribution at the rows from the forward's TrunkInputs
        from ced.model import cp_trunk

        return cp_trunk.logprobs_rows(head, trunk_pointer(head, pointer), h_q, index, W, chunk)
    idx = torch.arange(L, device=tokens.device).expand(B, L)
    q_src, k_src = pointer_sources(head, hidden, pointer)
    final = head.cfg.pointer_input == "final"
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        keys = forward_keys(head, hidden.to(dtype) if final else k_src, tokens, doc_ids, idx)
        if final:
            qs = None
        elif q_src.ndim == 3:
            qs = torch.gather(q_src, 1, index[..., None].expand(*index.shape, q_src.shape[-1]))
        else:  # the heads' inputs (B, L, H, D): the same positions on every head
            qs = torch.gather(q_src, 1, index[..., None, None].expand(*index.shape, *q_src.shape[2:]))
        return logprobs_rows(head, h_q, torch.gather(idx, 1, index), torch.gather(doc_ids, 1, index), keys, W, chunk,
                             q_src=qs)


# ---------------------------------------------------------------------------------------------------------------------
# Cached generation


@dataclass
class CopyCache:
    """The copy cache of a cached sequence: CopyKeys of every position so far (k_tau, token id, document id, absolute
    index). d_c values a position (64 at the spec: 128 B a token in BF16; held in compute_dtype here)."""

    keys: CopyKeys | None = None


def forward_logits(model: nn.Module, hidden: torch.Tensor, tokens: torch.Tensor, doc_ids: torch.Tensor,
                   idx: torch.Tensor, cache: CopyCache | None,
                   pointer: PointerInputs | None = None) -> tuple[torch.Tensor, CopyCache | None]:
    """The CP model's `logits` of a forward (HybridLM.forward): log p over V at every position of the chunk (score
    "mixture") or the LM logits (score "lm"), and the copy cache after the chunk (None when no cache is kept or
    returned: the caller passes `cache=None` and ignores it). idx (B, L): the chunk's absolute indices. pointer: this
    forward's PointerInputs under pointer_input "mix" / "shared" (the chunk's own key inputs; the cache holds the keys
    of the earlier positions, each formed once from its own inputs)."""
    inner = getattr(model, "_orig_mod", model)
    head = inner.cp_head
    if hasattr(head, "n_trunk"):  # the trunk copy: its own copy cache (cp_trunk.TrunkCache), built in trunk_inputs
        from ced.model import cp_trunk

        return cp_trunk.forward_logits(model, hidden, tokens, trunk_pointer(head, pointer))
    W = inner.lm_head_weight()
    dtype = compute_dtype(hidden)
    q_src, k_src = pointer_sources(head, hidden, pointer)
    final = head.cfg.pointer_input == "final"
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        own = forward_keys(head, hidden.to(dtype) if final else k_src, tokens, doc_ids, idx)
        keys = own if cache is None or cache.keys is None else cache.keys.extend(own)
        if score_mode(inner) == "lm":
            from ced.model.looped_ced import lm_head

            logits = lm_head(hidden, W)
        else:
            logits = logprobs_rows(head, hidden, idx, doc_ids, keys, W, q_src=None if final else q_src)
    return logits, CopyCache(keys=keys)


# ---------------------------------------------------------------------------------------------------------------------
# The readings tool's hook (the CP readings build, after this one)


@torch.no_grad()
def position_readings(model: nn.Module, hidden: torch.Tensor, tokens: torch.Tensor, targets: torch.Tensor,
                      doc_ids: torch.Tensor, chunk: int | None = None,
                      pointer: PointerInputs | None = None) -> dict[str, torch.Tensor]:
    """Per-position readings of the CP head for a batch, from a forward with return_hidden (hidden (B, L, D); tokens,
    targets, doc_ids (B, L)). Every value (B, L), float64 on the device:

        lam        lambda_t = c_t (1 - s_t)
        c          c_t, the <copy> share of the LM head's softmax over V u {<copy>}
        s          s_t, the sink's share of the CP head's softmax over {sink} u ctx
        copy_dist  sum over ctx(t) of alpha_t(tau) (idx_t - idx_tau): the alpha-weighted distance of the copied positions
        p_cp       p_CP(target) (0 where the target is not in ctx)
        logp       log p(target) of the mixture;   logp_lm   log p_LM(target)
        doc_pos    t's position in its document (0 at a document's first token)

    The model's score mode is not read: these are the mixture's parts. Chunked by query blocks ([B, C, L] alpha).
    pointer: the forward's ModelOutput.cp_pointer (pointer_sources)."""
    from ced.train.chunked_loss import token_lse_zy

    inner = getattr(model, "_orig_mod", model)
    head = cp_head_of(inner)
    assert head is not None, "position_readings needs a CP model"
    if hasattr(head, "n_trunk"):
        raise ValueError("position_readings is not built for the trunk copy (model.cp_head.trunk_copy): its readings are the per-head "
                         "selection diagnostic's (ced/eval/cp_head_selection.py) and head_readings' weights")
    B, L = tokens.shape
    dtype = compute_dtype(hidden)
    q_src, k_src = pointer_sources(head, hidden, pointer)
    final = head.cfg.pointer_input == "final"
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        h = hidden.to(dtype)
        idx = torch.arange(L, device=tokens.device).expand(B, L)
        keys = forward_keys(head, h if final else k_src, tokens, doc_ids, idx)
        k_sink = head.k_sink.to(dtype)
        C = chunk or head.cfg.query_chunk
        dist, s_all = [], []
        for a in range(0, L, C):
            q = head.queries(h[:, a : a + C] if final else q_src[:, a : a + C])
            q, q_sink, sink_add = pointer_queries(head, q, keys, idx[:, a : a + C], doc_ids[:, a : a + C])
            alpha, log_s, _, ctx = pointer_alpha(head, q, keys, k_sink, idx[:, a : a + C], doc_ids[:, a : a + C], q_sink,
                                                 sink_add, h_q=h[:, a : a + C])
            gap = (idx[:, a : a + C, None] - keys.idx[:, None, :]).to(dtype)
            dist.append((alpha * torch.where(ctx, gap, torch.zeros((), dtype=dtype, device=gap.device))).sum(-1))
            s_all.append(log_s.exp())
        log_s, log_1ms, log_pcp = cp_terms(head, h, tokens, targets, doc_ids, chunk=C, pointer=pointer)
        lse, z_y = token_lse_zy(h, inner.lm_head_weight(), targets)
        z_c = head.copy_logit(h)
        log_lam, _, log_c = mixture_parts(lse, z_c, log_s, log_1ms)
        logp = mixture_log_prob(lse, z_y, z_c, log_s, log_1ms, log_pcp)
    from ced.model.sequence import sequence_context

    ctx, _ = sequence_context(doc_ids, None)
    f64 = lambda t: t.double()  # noqa: E731
    return {"lam": f64(log_lam.exp()), "c": f64(log_c.exp()), "s": f64(torch.cat(s_all, 1)),
            "copy_dist": f64(torch.cat(dist, 1)), "p_cp": f64(log_pcp.exp()), "logp": f64(logp),
            "logp_lm": f64(z_y - lse), "doc_pos": ctx.pos}


@torch.no_grad()
def head_readings(model: nn.Module, hidden: torch.Tensor, tokens: torch.Tensor, doc_ids: torch.Tensor, chunk: int | None = None,
                  pointer: PointerInputs | None = None) -> dict[str, torch.Tensor]:
    """The pointer's heads, per head (module docstring, "THE POINTER'S HEADS"; the hook of the six-cell reading), from a forward
    with return_hidden (hidden (B, L, D); tokens, doc_ids (B, L); pointer: ModelOutput.cp_pointer). float64 on the device:

        gain    (H,)        1 + qk_gain of each head (its learned scale on the QK-normed query; head_gains)
        under combine "or", (B, L, H) each at every position:
        pi      the router's weights over the heads (softmax of W_route h_t)
        s_h     each head's own sink share
        w       each head's share of the copy, w_h = pi_h (1 - s_h) / (1 - s) (p_CP = sum_h w_h p_CP,h)

    The model's score mode is not read. A head of one form or "and" gives the gains only (its heads share one softmax)."""
    inner = getattr(model, "_orig_mod", model)
    head = cp_head_of(inner)
    assert head is not None, "head_readings needs a CP model"
    if hasattr(head, "n_trunk"):  # the trunk copy: the learned weights over every global head (0 for a head out of the subset)
        from ced.model.cp_trunk import head_weights

        return {"weights": head_weights(head), "keep": head.trunk_keep.detach().cpu().clone()}
    out = {"gain": head.head_gains() if head.cfg.qk_norm else torch.ones(head.cfg.n_heads, dtype=torch.float64)}
    if head.cfg.combine != "or":
        return out
    B, L = tokens.shape
    dtype = compute_dtype(hidden)
    q_src, k_src = pointer_sources(head, hidden, pointer)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        h = hidden.to(dtype)
        idx = torch.arange(L, device=tokens.device).expand(B, L)
        keys = forward_keys(head, k_src, tokens, doc_ids, idx)
        k_sink = head.k_sink.to(dtype)
        C = chunk or head.cfg.query_chunk
        got = {"pi": [], "s_h": [], "w": []}
        for a in range(0, L, C):
            q = head.queries(q_src[:, a : a + C])
            q, _, _ = pointer_queries(head, q, keys, idx[:, a : a + C], doc_ids[:, a : a + C])
            _, log_s_h, _, log_pi, log_w, _ = head_alphas(head, q, keys, k_sink, idx[:, a : a + C], doc_ids[:, a : a + C],
                                                          h[:, a : a + C])
            got["pi"].append(log_pi.exp())
            got["s_h"].append(log_s_h.exp())
            got["w"].append(log_w.exp())
    out.update({k: torch.cat(v, 1).double() for k, v in got.items()})
    return out
