"""The Copy-Paste head's training modes other than the mixture (model.cp_head.train_mode; "mixture", the default, is cp_head.py's).
cp-aux: the Copy-Paste pointer trained only through its own auxiliary loss (model.cp_head.train_mode "aux"; off by default). Add-on CP
(train_mode "addon"; off by default): the section "ADD-ON CP" at the end of this docstring.

Runs: runs/arch-cp-aux-60m.yaml (held). Tests: tests/test_cp_aux.py.

THE POINTER is cp-mix's (ced/model/cp_head.py, "THE POINTER'S INPUT", pointer_input "mix"): its two depth readers and their norms
(cp_head.mix), W_Q, W_K, the QK-norm and its gain, the sink key; the context ctx(t) = {tau : idx_tau <= idx_t, doc_tau = doc_t} (tau =
t included); the scores e_tau = q_t . k_tau / sqrt(d_c) and e_sink = q_t . k_sink / sqrt(d_c). The same tensors, the same draws (below)
and the same function (cp_head.cp_terms) as cp-mix.

THE AUXILIARY LOSS. The pointer's distribution at t is P_t = softmax over {sink} u ctx(t) of (e_sink, e_tau). Its target comes from the
mapping the head uses to turn positions into tokens, p_CP(v) = sum over tau in ctx(t) with x_tau = v of alpha_t(tau) (the pointer at tau
copies the token x_tau), taken at v = y_t = x_{t+1}, the next token:
    match(t)  = {tau in ctx(t) : x_tau = y_t}                    (cp_head.chunk_terms' own comparison)
    L_aux(t)  = -log sum over tau in match(t) of P_t(tau)        when match(t) is not empty
              = -log P_t(sink)                                    when it is empty (the next token is not in the context)
From cp_terms' three terms, exactly: log sum_{match} P_t = (LSE_match - LSE_ctx) + (LSE_ctx - LSE_all) = log p_CP(y) + log(1 - s), and
log P_t(sink) = log s; so L_aux = -(log p_CP(y) + log(1 - s)) where a match exists, else -log s (aux_ce). chunk_terms makes log p_CP(y)
the constant -inf exactly where match is empty, so isfinite(log p_CP) is the match test, and the branch not taken gets a zero gradient
(torch.where's backward; never NaN).
The loss of a micro-batch, n = the number of valid targets (the loss mask; the main loss's denominator):
    L = (1/n) sum_valid [CE_LM(t) + z LSE_V(t)^2]  +  aux_weight x (1/n) sum_valid L_aux(t)
The first term is R1-entry's loss computed by its own code (Trainer._micro_loss_selected); the second is averaged over the same
positions. Logged per step: cp_aux (the mean L_aux over the valid targets) and cp_aux_sink (the share of them whose target is the sink).

THE GRADIENTS. The main loss reads h (the final norm's output) through the LM head only. The pointer's tensors (W_Q, W_K, k_sink,
qk_gain, cp_head.mix.*) are not in it, so they learn from L_aux alone. (A stop-gradient form, the pointer's input detached so that the
trunk learns from p_LM alone)

THE GATE IS NOT BUILT. AuxPointerHead has no w_copy (the <copy> row), so nothing computes c_t or lambda_t. Why not built rather than
built and unused: nothing is mixed into the prediction, so a w_copy would be a parameter at its initial draw in the optimizer and the
state dict that every mixture reader (the CP readings, the mechanism reader) would read as if it were trained; not built, those paths
fail by name (copy_logit raises). The head's draws are cp-mix's: reset_own draws w_copy's N(0, std) into a stand-in that is dropped, so
W_Q, W_K and k_sink (and the readers' zero start) are cp-mix's of the same seed, bit for bit (tested).

EVALUATION ("the pointer dropped at evaluation"). HybridLM.forward in eval mode (model.eval(): every scorer, probe and generation) does
not form the pointer's inputs (the stack's sources are dropped from the forward's memo) and its logits are lm_head(h), p_LM; cached
generation keeps no copy cache. cp_head.cp_head_of returns None for this head, so every scoring path (span_bpb and the score run, the
probes, the long-context suite, the residual diagnostics, multiple choice, the CP readings' copy probe) reads the model as one without a
head: the scored bpb equals the model with the pointer removed, exactly (tests/test_cp_aux.py). In training mode (the trainer's forward)
the pointer's inputs are formed as cp-mix's are.

ADD-ON CP (model.cp_head.train_mode "addon"; the CP gain decomposition's arm (B)). Runs: runs/arch-cp-addon-r1e-60m.yaml (held). Tests:
tests/test_cp_addon.py. The head is cp-mix's, whole (CopyPasteHead with pointer_input "mix": the readers and norms, W_Q, W_K, the QK-norm
and its gain, the sink key AND the <copy> row w_copy; the same draws of the same seed), and so is everything at evaluation: the mixture p =
(1 - lambda) p_LM + lambda p_CP is the forward's output and every scorer's (cp_head_of returns the head), exactly cp-mix's code with these
tensors; CED_CP_SCORE=lm scores p_LM alone (the LM head, the pointer's terms not computed), as for cp-mix. Only training differs: the main
loss R1-entry's own: (1/n) sum_valid [CE_LM(t) + z LSE_V(t)^2], computed by R1-entry's code (Trainer._micro_loss_selected) on h, so the
trunk and the LM head get exactly R1-entry's gradient (the pointer adds no consumer of any trunk or LM-head tensor's value inside the
autograd graph); the pointer's its readers read the stack's sources [b_0 .. b_N] DETACHED (cp_head.pointer_inputs), and the <copy> row reads
h inputs detached (Trainer._cp_addon_loss): no gradient of the mixture loss reaches any trunk tensor; the mixture (1/n) sum_valid -log
p(y_t), the mixture's CE with cp-mix's mixture and mapping (cp_head.mixture_log_prob's rows) and loss p_LM's numbers DETACHED inside it: log
p_LM(y) = -CE_LM and LSE_V, both from the main loss's own head call, carry no gradient (addon_log_prob); so its gradient reaches the
pointer's tensors (W_Q, W_K, k_sink, qk_gain, cp_head.mix.*) through log s, log(1 - s), log p_CP(y), and w_copy (the gate's <copy> row)
through z_copy, and nothing else; the clip the trunk's and LM head's gradients are clipped by their own norm and the head's by its own
(Trainer's step: two clip_grad_norm_ calls at optim.grad_clip), so the pointer's gradient does not change the trunk's update through the
global norm either: the trunk's and LM head's updates are R1-entry's, bit for bit on the CPU (tests/test_cp_addon.py). grad_norm logs the
trunk's norm (R1-entry's number); cp_grad_norm the head's. The loss of a micro-batch is the sum of the two terms (their parameter sets are
disjoint, so the sum's gradient on each is its own term's). Logged per step beside the base's (ce_dec / ce_ctx are p_LM's, R1-entry's own
sums): ce_mix (the mixture's mean CE over the valid targets), cp_lambda (the mean lambda) and cp_grad_norm. Built for cp-mix's pointer
(pointer_input "mix", one head, the reference kernel, no length keys), without activation checkpointing, data parallelism, CUDA graphs or
TST's multi-hot aux: each refused by name.

ADD-ON CP IN THE STACK (runs/arch-stack-addon-r1e-60m.yaml; tests/test_stack_addon_model.py). Built the way cp-mix runs in the stack
(runs/arch-stack5-r1e-60m.yaml): Token Superposition Training (tst.schedule): a bag phase (s > 1) is TST's bag loss on the trunk and the LM
head, the trainer's own code (Trainer._micro_loss_tst); the head is not in the loss (the mixture is not defined over bags), so it gets no
gradient, no update and no optimizer state; the split clip clips the trunk's gradients by their own norm and finds none on the head
(cp_grad_norm 0). A next-token phase (s = 1) has one target a position, the base's next token y_t = x_{t+1}: the loss above, unchanged. So
the head trains from TST's switch on (4,865 of the 6,487 updates, as cp-mix's in the stack), and the trunk's and the LM head's updates are
the arm without the head's in both phases (tested bit for bit on the CPU). TST's multi-hot aux (several targets a position in a next-token
phase) stays refused for every CP head (the trainer, before this check: nothing defines the mixture for them). tied embeddings
(model.tie_embeddings): the LM head is the input embedding; LSE_V and log p_LM(y) are the tied head call's own numbers (the main loss's,
Trainer._micro_loss_selected's rows), detached as above; the <copy> row is the head's own w_copy. wsl (A, model.encoder.gdn_write_mix
"layer"): A changes the GDN layers' inputs inside the stack; the readers read R1-entry's stored sources [b_0 .. b_N] detached, so no
gradient of the mixture reaches A's lambdas (cp_head.mix_with_write_mix). 4:1 (model.encoder.nope_global_layers [4]): a config of the
trunk's global layers; the head reads nothing inside them.
"""

from __future__ import annotations

import math

import torch

from ced.config import CPHeadConfig, ModelConfig
from ced.model.cp_head import DEFAULT_REF_LEN, CopyPasteHead

TRAIN_MODES = ("mixture", "aux", "addon")  # model.cp_head.train_mode
METRICS = ("cp_aux", "cp_aux_sink")  # the trainer's per-step logs (means over the valid targets)
ADDON_METRICS = ("ce_mix", "cp_lambda")  # add-on CP's per-step logs (means over the valid targets), beside cp_grad_norm
ADDON_POINTERS = ("mix",)  # add-on CP is built for cp-mix's pointer


def aux_on(c: CPHeadConfig) -> bool:
    return c.enabled and c.train_mode == "aux"


def addon_on(c: CPHeadConfig) -> bool:
    """Add-on CP (module docstring, "ADD-ON CP")."""
    return c.enabled and c.train_mode == "addon"


def check_cp_aux(cfg: ModelConfig) -> None:
    """Refuse model.cp_head.train_mode and aux_weight wherever cp-aux is not built, and their bad values (ValueError, by name).
    Called first by cp_head.check_cp_head; the head's other keys are checked there."""
    c, d = cfg.cp_head, CPHeadConfig()
    if c.train_mode not in TRAIN_MODES:
        raise ValueError(f"model.cp_head.train_mode {c.train_mode!r}: one of {TRAIN_MODES}")
    w = c.aux_weight
    if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0:
        raise ValueError(f"model.cp_head.aux_weight {w!r}: the auxiliary loss's weight, a finite number >= 0")
    if not c.enabled:
        named = [k for k in ("train_mode", "aux_weight") if getattr(c, k) != getattr(d, k)]
        if named:
            raise ValueError(f"model.cp_head.{named[0]} without model.cp_head.enabled: the key would be ignored")
        return
    if c.train_mode in ("mixture", "addon"):
        if w != 0:
            raise ValueError(f"model.cp_head.aux_weight {w!r} without model.cp_head.train_mode 'aux': the key would be ignored")
        if c.train_mode == "addon":
            check_addon(cfg)
        return
    if w <= 0:
        raise ValueError("model.cp_head.aux_weight 0 with model.cp_head.train_mode 'aux': the pointer would get no gradient (its "
                         "only loss is the auxiliary one); give the weight, > 0")
    if c.pointer_input != "mix":
        raise ValueError(f"model.cp_head.train_mode 'aux' is built for model.cp_head.pointer_input 'mix' (the pointer as in cp-mix), "
                         f"not {c.pointer_input!r}: nothing tests the pair")
    if c.force_lambda_zero:
        raise ValueError("model.cp_head.force_lambda_zero with model.cp_head.train_mode 'aux': there is no lambda (nothing is mixed "
                         "into the prediction)")
    if c.score != d.score:
        raise ValueError(f"model.cp_head.score {c.score!r} with model.cp_head.train_mode 'aux': the key would be ignored (the pointer "
                         "is dropped at evaluation: every scoring path reads p_LM alone)")
    for key, on in (("temperature", c.temperature != d.temperature), ("sink_count", c.sink_count != d.sink_count)):
        if on:
            raise ValueError(f"model.cp_head.{key} {getattr(c, key)!r} with model.cp_head.train_mode 'aux': nothing tests the pair")


def check_addon(cfg: ModelConfig) -> None:
    """Refuse add-on CP (model.cp_head.train_mode "addon"; module docstring, "ADD-ON CP") wherever it is not built (ValueError, by name).
    Called by check_cp_aux for an enabled head under "addon" (aux_weight 0 checked there); the trainer refuses its training settings."""
    c, d = cfg.cp_head, CPHeadConfig()
    if c.pointer_input not in ADDON_POINTERS:
        raise ValueError(f"model.cp_head.train_mode 'addon' is built for model.cp_head.pointer_input 'mix' (cp-mix's pointer, on "
                         f"cp-mix's head), not {c.pointer_input!r}: nothing tests the pair")
    if c.force_lambda_zero:
        raise ValueError("model.cp_head.force_lambda_zero with model.cp_head.train_mode 'addon': the mixture is the pointer's and the "
                         "gate's only loss (with lambda forced to 0 they would get no gradient)")
    for key in ("temperature", "sink_count", "n_heads", "combine"):
        if getattr(c, key) != getattr(d, key):
            raise ValueError(f"model.cp_head.{key} {getattr(c, key)!r} with model.cp_head.train_mode 'addon': nothing tests the pair "
                             "(built: cp-mix's one-head pointer without the length keys)")


def check_addon_train(cfg, data_parallel: bool) -> None:
    """The training settings add-on CP is built for (the trainer calls it for a model under train_mode "addon"; ValueError, by name):
    one process (no data parallelism: the split clip is one process's), no CUDA graphs, one target a position in a next-token step (no
    TST multi-hot aux). TST's schedule is built (module docstring, "ADD-ON CP IN THE STACK": a bag phase trains the trunk on TST's bag loss
    with the head idle, a next-token phase is the loss above). cfg: the TrainConfig. Activation checkpointing is refused by the trainer's
    own check of the pointer's input (built there for train_mode "mixture" only)."""
    for key, value, off in (("data parallelism (a world size above 1)", bool(data_parallel), False),
                            ("cuda_graphs", cfg.cuda_graphs, "none"),
                            ("tst.aux_multi_hot", bool(cfg.tst.aux_multi_hot), False)):
        if value != off:
            raise ValueError(f"model.cp_head.train_mode 'addon' is not built together with {key} {value!r}: nothing tests the pair (the "
                             "split clip and the mixture's loss are built for one process and one target a position)")


def addon_log_prob(lse_v: torch.Tensor, log_lm_y: torch.Tensor, z_c: torch.Tensor, log_s: torch.Tensor, log_1ms: torch.Tensor,
                   log_pcp: torch.Tensor) -> torch.Tensor:
    """log p(y) of the mixture (cp_head.mixture_log_prob's rows) with p_LM given as its own numbers: lse_v = LSE_V and log_lm_y = log
    p_LM(y) = z_y - LSE_V (the main loss's -CE_LM). Add-on CP passes both DETACHED (module docstring, "ADD-ON CP"), so the gradient
    reaches z_copy and the pointer's three terms only. The same function as mixture_log_prob(lse_v, lse_v + log_lm_y, ...), written with
    log p_LM(y) itself so that no LSE_V - LSE_V cancellation rounds it (tests/test_cp_addon.py)."""
    lse_all = torch.logaddexp(lse_v, z_c)
    log_c = z_c - lse_all
    log_1mc = lse_v - lse_all
    log_lam = log_c + log_1ms
    log_1mlam = torch.logaddexp(log_1mc, log_c + log_s)
    return torch.logaddexp(log_1mlam + log_lm_y, log_lam + log_pcp)


class AuxPointerHead(CopyPasteHead):
    """cp-aux's head (module docstring): CopyPasteHead without the <copy> row (no w_copy), its draws cp-mix's."""

    def __init__(self, d_model: int, cfg: CPHeadConfig, eps: float = 1e-6, d_shared: int | None = None,
                 ref_len: int = DEFAULT_REF_LEN):
        super().__init__(d_model, cfg, eps=eps, d_shared=d_shared, ref_len=ref_len)
        del self.w_copy  # the gate is not built (module docstring, "THE GATE IS NOT BUILT")
        self.d_model = d_model

    @torch.no_grad()
    def reset_own(self, g: torch.Generator, std: float, init: str = "spec") -> None:
        """CopyPasteHead.reset_own's draws in its order, w_copy's into a stand-in that is dropped: the generator's stream, and so
        W_Q, W_K and k_sink, are cp-mix's of the same seed."""
        self.w_copy = torch.empty(self.d_model, dtype=self.k_sink.dtype)  # a plain tensor attribute, not a parameter
        try:
            super().reset_own(g, std, init)
        finally:
            del self.w_copy

    def copy_logit(self, h: torch.Tensor) -> torch.Tensor:
        raise ValueError("this Copy-Paste head is cp-aux's (model.cp_head.train_mode 'aux'): it has no <copy> row and nothing is "
                         "mixed into the prediction; read p_LM (the LM head) or the pointer's terms (cp_head.cp_terms, aux_ce)")


def aux_ce(log_s: torch.Tensor, log_1ms: torch.Tensor, log_pcp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(L_aux, sink) per position from cp_terms' (log s, log(1 - s), log p_CP(y)) (module docstring): L_aux = -(log p_CP(y) +
    log(1 - s)) where the next token is in the context, else -log s; sink (bool) marks the positions whose target is the sink."""
    has = torch.isfinite(log_pcp)
    return torch.where(has, -(log_pcp + log_1ms), -log_s), ~has
