"""Gated DeltaProduct for the GDN layers (GDNConfig.dprod).

THE ASK. Every arm "keeps R1-entry and changes only the GDN layers' memory update". The survey found that our kernel runs it as is
on the n_h T interleave.

THE RULE (Siems et al. arXiv 2502.10297, Eq. 3, in our layout S in R^{d_k x d_v}, o_t = S_t^T q_t). Per token t, with the layer's scalar
decay a_t = exp(g_t) once and n_h delta-rule steps j = 1 .. n_h, each with its own k_{t,j} (SiLU, L2 norm), v_{t,j} and beta_{t,j} = 2
sigmoid(b_{t,j}) in (0, 2) (BETA_MAX; the paper's gated setting, its allow_neg_eigval form, as fla's GatedDeltaProduct layer with
allow_neg_eigval=True: each step's transition I - beta k k^T has the eigenvalue 1 - beta in (-1, 1) along k). The key dprod.beta_max (2,
the default: the line above; 1: beta = sigmoid(b) in (0, 1), see beta_max_of at the end) sets M in beta = M sigmoid(b); n_h 1 (no key) is
unchanged:

    S <- a_t S                                                        (the decay, once, before the first step)
    S <- S + k_{t,j} (beta_{t,j} (v_{t,j} - S^T k_{t,j}))^T,  j = 1 .. n_h   (each step: S <- (I - beta k k^T) S + beta k v^T)
    o_t = S^T q_t                                                     (the read, once, after the last step)

so S_t = prod_{j = n_h .. 1} (I - beta_{t,j} k_{t,j} k_{t,j}^T) a_t S_{t-1}
       + sum_j [prod_{i > j} (I - beta_{t,i} k_{t,i} k_{t,i}^T)] beta_{t,j} k_{t,j} v_{t,j}^T     (the paper's Eq. 3, transposed).
n_h = 1 is Gated DeltaNet (the option off: the layer as it was, beta = sigmoid(b), nothing of this file runs). Step 1 is the
layer's own (in_proj_qkv's k and v after its short conv, in_proj_b's beta: 2 x gates()' sigmoid(b)); steps 2 .. n_h
read `dprod.in_proj_kv` (d -> (n_h - 1)(H_k d_k + H_v d_v), the channels of step j = 2 .. n_h in turn, each [k_j | v_j]),
`dprod.conv_weight` (their depthwise short conv, the layer's conv rule) and `dprod.in_proj_b` (d -> (n_h - 1) H_v, step-major).
That is fla's GatedDeltaProduct layer (fla/layers/gated_deltaproduct.py at v0.5.2: k_proj, v_proj and b_proj of n_h blocks, the
first block the first step; one q_proj; the gate once) with its first block in our in_proj_qkv / in_proj_b; GQA kept (value
head j reads key head j // (H_v / H_k), as the layer). The decay, the gates (with the evaluation floor and cap), the output gate
and norm are the layer's, unchanged: the census (scripts/gdn_g_dist.py --census, a scalar decay per head) reads them as for GDN.

THE PATHS (each the same function; the per-token loop is the ground truth):
- `dprod_recurrence`: the rule above, token by token and step by step (GDNConfig.kernel "recurrence", and a step of one token).
- `interleave` + ced/model/gdn_chunked.py chunked_delta_rule (kernels "chunked", "fla", "ced" off the fused path, L > 1): the
  n_h T interleave of fla's chunk_gated_delta_product (fla/ops/gated_delta_product/chunk.py at v0.5.2): sub-step n_h t + j - 1 is
  an ordinary gated delta-rule step with k_{t,j}, v_{t,j}, beta_{t,j}; the decay g_t on sub-step j = 1 and 0 (no decay) on the
  others; q_t on sub-step j = n_h and 0 on the others (their outputs are not read); a document start on sub-step j = 1 only.
- the fused path (CUDA, kernel "ced" (or "fla"), from an empty state with varlen boundaries: the training forward and the
  scoring forward): the same interleave into our kernel's ced_delta_rule (ced/kernels/gdn.py), which runs it AS IS: its
  contract is q, k (1, T, H_k, 128), v (1, T, H_v, 128), g and beta (1, T, H_v), H_v in (H_k, 2 H_k), varlen boundaries
  (ced/kernels/gdn.py:292, the forward's only assert), nothing about what a sub-step stands for, and beta only as a multiplier
  with no clamp at 1 (ced/model/gdn_negeig.py, THE KERNELS: read for, whose GPU values check runs our kernel at beta up to 2;
  this arm's own check, scripts/gdn_dprod_check.py, judges its path at beta in (0, 2)); the boundaries are n_h x the token
  boundaries (cu_seqlens x n_h, as fla's chunk_gated_delta_product), the outputs read at the last sub-step of each token. The
  input preparation stays one fused kernel (ced/kernels/gdn.py qkv_prep over [q | k_1 | v_1 | k_2 | v_2 ...] with the extra
  channels in its value slots, so one launch each way): the extra keys' L2 norm runs after it (float32, rounded to the kernel's
  input dtype), so with bf16 kernel inputs (GDNConfig.input_dtype "bfloat16") k_j (j >= 2) is rounded twice (the conv's SiLU
  stored in bf16, then the unit vector): a numerics difference from the reference, judged against float64 by the arm's GPU
  check (scripts/gdn_dprod_check.py; section 5).
- K2 on the fused path (GDNConfig.kda.kernel "k2"; ced/kernels/gdn_k2.py): the n_h steps of a token as n_h rows of one WY form,
  k, v and beta in the stacks' own row layout, q and g once a token: no interleave. The key at its default keeps the interleave
  above (the same lines as before the key).
- the one-readout route on the fused path (GDNConfig.dprod.route "one-readout", n_h 2, kernel "ced"; ced/kernels/gdn_1r.py): the same
  chunk kernels on the same sub-step rows of the (B, L, 2..) stacks, with the readout done once a token: q (B, L, H_k, d_k) and g
  (B, L, H_v) enter at the token rows (no q or g interleave copy, no readout or output on the sub-step with q = 0), the stacks k, v, beta
  in their own layout (a view). The FORM is in what the caller puts in the stacks: the Gated DeltaProduct form and the Gated DeltaProduct
  form with beta in [0, 1] (a smaller beta_max) differ in beta only; A (a subset of whole key groups) calls the route with those heads'
  slices; C (a second key from elsewhere) and E (the second step off the stride: beta_2 = 0 there) put their k_2, v_2, beta_2 in the
  stacks, however they were made. Off (the default "interleave"): the lines above, bit for bit. Fixed shapes throughout (stack and
  reshape; no index op, no boolean mask).

INITIALISATION. The extra tensors are drawn from a generator of their own (init_own_parameters, called by the model's build after
init_weights: seed torch.initial_seed() + SEED_OFFSET), so every tensor R1-entry has starts as in R1-entry of the same seed and the
global RNG is not touched (ced/model/cp_head.py's and ced/model/ldm.py's rule): in_proj_kv and in_proj_b N(0, init_std), the
trunk's rule for linears (and fla's initializer_range 0.02); conv_weight U(-1/sqrt(K), 1/sqrt(K)), the layer's own conv rule. A
layer built on its own (no model) draws them at construction from a fixed generator (STANDALONE_SEED), never the global one.
No zero-initialised gate: optim.zero_gate_decay moves no tensor of this arm. Optimiser groups by name (ced/train/optim.py), as
the first step's: dprod.in_proj_kv.weight a hidden matrix (Muon under name "muon"), dprod.in_proj_b.weight a gate projection
(ADAMW_MATRIX_SUFFIXES), dprod.conv_weight a depthwise conv (AdamW); all three with the run's weight decay.

PARAMETERS (exact): (n_h - 1) [d (H_k d_k + H_v d_v) + d H_v + K (H_k d_k + H_v d_v)] a layer: at 60M (d 512, 2 / 4 heads of 128,
K 4) 398,336 a layer, 5,975,040 over 15 GDN layers for n_h = 2 (the survey's +5.98M, tools/analysis/gdn_rules_cost.py gdp_step).

NOT BUILT (refused by name, check_config and dprod_forward): the decoder's GDN layers and arches other than "control"; the
GDN-side arms (write mix A, CCR); the zero-write second pass (keep); padding (valid); Token Superposition Training's bag phases
(SeqContext.gdn_scale != 1); CUDA graphs' fixed-capacity metadata (SeqContext.gdn_chunks); edges other than "plain" and
chunk_backend other than "triton" (their fused ops form q, k, v of one step); the MTP drafter and the decode engine
(ced/infer/mtp_spec.py and ced/infer/engine.py write the GDN step themselves: the drafter refused here, the engine by
ced/model/gdn_rules.py, as the GDN state walk, gdn_memory, gate_floor and gdn_force); negative eigenvalues (ced/model/gdn.py
_refuse_unbuilt_rule_pairs: this rule's beta is already in (0, 2)); FP8 / NVFP4 linears. F3 (decay_cumsum "compensated") passes
to our kernel as for GDN. Python 3.11 compatible.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ced.config import GDNConfig

SEED_OFFSET = 0xD9_0D_2026_1022  # the extra tensors' own stream: seed = torch.initial_seed() + this
STANDALONE_SEED = 0xD9_0D  # a layer built outside a model (benches, kernel checks): a fixed stream, not the global one
KERNELS = ("recurrence", "chunked", "fla", "ced")  # the GDN kernels this rule is built for
# every step's beta = M x sigmoid(b), M = BETA_MAX for dprod.beta_max 2 (default: beta in (0, 2); 2 is a power of two, exact) or 1 for beta_max
# 1 (beta in (0, 1)): beta_multiplier, at the end
BETA_MAX = 2
ROUTES = ("interleave", "one-readout")  # GDNConfig.dprod.route: the fused path's readout (the first is the default, every run before the key)
RAN = {"one_readout": 0}  # calls of the one-readout route's wrapper (the model-level tests and the GPU check read it)


def route_of(cfg: GDNConfig) -> str:
    """The fused path's route, GDNConfig.dprod.route, checked: one of ROUTES ("interleave" when the key is absent)."""
    r = getattr(getattr(cfg, "dprod", None), "route", "interleave")
    if r not in ROUTES:
        raise ValueError(f"GDNConfig.dprod.route {r!r}: one of {ROUTES} ('interleave': the n_h T interleave into our kernel, the "
                         "readout on every sub-step; 'one-readout': the readout once a token, ced/kernels/gdn_1r.py)")
    return r


def one_readout_refusals(cfg: GDNConfig) -> list[str]:
    """Why a layer of this GDNConfig cannot take route "one-readout" (empty: it can, or the route is "interleave"): the route is the chunk
    kernels of ced/kernels/gdn_triton.py with the readout once a token, so it needs kernel "ced", n_h 2, H_v in (H_k, 2 H_k), edges
    "plain", no second kernel named (kda.kernel "k2"); the head width 128 is the kernels' own assert at the first call (the CPU tests
    run the route through its reference at a tiny width)."""
    if route_of(cfg) == "interleave":
        return []
    n = n_steps(cfg)
    hk, hv = cfg.n_k_heads, cfg.n_v_heads
    refused = [(n != 2, f"n_h {n} (built for n_h 2: two sub-steps a token, the readout on the second)"),
               (cfg.kernel != "ced", f"kernel {cfg.kernel!r} (built on our kernel 'ced', ced/kernels/gdn_triton.py)"),
               (hv not in (hk, 2 * hk), f"{hk} key / {hv} value heads (the value heads must be 1 or 2 per key head)"),
               (cfg.edges != "plain", f"edges {cfg.edges!r}"),
               (cfg.chunk_backend != "triton", f"chunk_backend {cfg.chunk_backend!r}"),
               (cfg.kda.kernel == "k2", "kda.kernel 'k2' (K2 is the other dedicated n_h-step kernel; name one)")]
    return [why for cond, why in refused if cond]


CPU_KERNELS = ("recurrence", "chunked")  # the reference kernels of the CPU: the ones cpu_route maps for
CPU_ROUTE_NOTE = ("ROUTE: one-readout -> interleave for the CPU kernel {kernel!r} (the same function and parameters: GDNConfig.dprod.route is an "
                   "implementation field and the CPU kernels ignore it; the one-readout kernel's GPU check passed against the interleave "
                   "route; ced/model/gdn_dprod.py cpu_route)")


def cpu_route(cfg: GDNConfig) -> str | None:
    """THE ONE MAPPING of an analysis that reads a "one-readout" checkpoint on a CPU kernel. A run trained with GDNConfig.dprod.route
    "one-readout" is refused on every kernel but "ced" (one_readout_refusals), and the analyses in the CPU worker run the CPU kernels
    ("chunked", "recurrence": scripts/gdn_g_dist.py override_backends). The route is the same function and parameters as "interleave"
    (trainer.py DPROD_IMPLEMENTATION_FIELDS: the state dict loads either way) and the CPU kernels never read it, so a loader that has just
    replaced the layer's kernel calls this, once per GDN section, and builds on "interleave".

    cfg: a GDNConfig whose kernel has ALREADY been set to the CPU kernel. Returns None (and changes nothing) unless the route is
    "one-readout" and the kernel is one of CPU_KERNELS and the layer would take the route on kernel "ced" (every other refusal still fires
    at the build, by its own name); otherwise sets dprod.route to "interleave" and returns the line the analysis must print
    (CPU_ROUTE_NOTE). Never called by the training path: a run's own build keeps the refusal (tests/test_census_1r.py pins both)."""
    if route_of(cfg) != "one-readout" or cfg.kernel not in CPU_KERNELS:
        return None
    as_trained = copy.copy(cfg)
    as_trained.kernel = "ced"
    if one_readout_refusals(as_trained):
        return None
    cfg.dprod.route = "interleave"
    return CPU_ROUTE_NOTE.format(kernel=cfg.kernel)


def n_steps(cfg: GDNConfig) -> int:
    """The Householder steps a token, GDNConfig.dprod.n_h, checked: an int >= 1 (1: Gated DeltaNet, off)."""
    n = getattr(getattr(cfg, "dprod", None), "n_h", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"GDNConfig.dprod.n_h {n!r}: an integer >= 1 (1: Gated DeltaNet, the layer as it was; the arm: 2)")
    return n


def step_channels(cfg: GDNConfig) -> int:
    """The pre-conv channels of one extra step: [k_j (H_k d_k) | v_j (H_v d_v)]."""
    return cfg.n_k_heads * cfg.k_head_dim + cfg.n_v_heads * cfg.v_head_dim


def params_per_layer(d_model: int, cfg: GDNConfig) -> int:
    """The tensors one GDN layer adds (exact): (n_h - 1) [d c + d H_v + K c], c = step_channels."""
    n, c = n_steps(cfg) - 1, step_channels(cfg)
    return n * (d_model * c + d_model * cfg.n_v_heads + cfg.conv_kernel * c)


class DeltaProductSteps(nn.Module):
    """The tensors of steps 2 .. n_h of one GDN layer (GatedDeltaNet.dprod): in_proj_kv (d -> (n_h - 1) c, c = H_k d_k + H_v d_v,
    the channels of step j in turn, [k_j | v_j]), in_proj_b (d -> (n_h - 1) H_v, step-major) and conv_weight ((n_h - 1) c, K: the
    depthwise short conv of those channels, conv_weight[:, K-1] on the current input as the layer's). Built on the meta device and
    moved to `device` without a draw (no random number of the global stream), then drawn from a fixed generator (reset_own); the
    model's build draws them again from the run's own stream (init_own_parameters)."""

    def __init__(self, d_model: int, cfg: GDNConfig, device=None, dtype=None):
        super().__init__()
        self.n_extra = n_steps(cfg) - 1
        assert self.n_extra >= 1, "DeltaProductSteps is built for n_h >= 2"
        self.channels = self.n_extra * step_channels(cfg)
        dev = torch.device("cpu") if device is None else torch.device(device)
        self.in_proj_kv = nn.Linear(d_model, self.channels, bias=False, device="meta", dtype=dtype).to_empty(device=dev)
        self.in_proj_b = nn.Linear(d_model, self.n_extra * cfg.n_v_heads, bias=False, device="meta",
                                   dtype=dtype).to_empty(device=dev)
        # init_weights (ced/model/looped_ced.py) draws every other linear from the global stream; these two are drawn here
        self.in_proj_kv.own_init = True
        self.in_proj_b.own_init = True
        self.conv_weight = nn.Parameter(torch.empty(self.channels, cfg.conv_kernel, device=dev, dtype=dtype))
        self.reset_own(torch.Generator().manual_seed(STANDALONE_SEED), 0.02)

    @torch.no_grad()
    def reset_own(self, g: torch.Generator, std: float) -> None:
        """in_proj_kv, then in_proj_b: N(0, std); then conv_weight: U(-1/sqrt(K), 1/sqrt(K)) (the layer's conv rule), all from the
        CPU generator g in float32 and copied in. Tensors on the meta device are skipped (nothing to draw into)."""
        K = self.conv_weight.shape[1]
        bound = 1.0 / math.sqrt(K)
        for p, draw in ((self.in_proj_kv.weight, lambda s: torch.randn(s, generator=g, dtype=torch.float32) * std),
                        (self.in_proj_b.weight, lambda s: torch.randn(s, generator=g, dtype=torch.float32) * std),
                        (self.conv_weight, lambda s: (torch.rand(s, generator=g, dtype=torch.float32) * 2 - 1) * bound)):
            if p.is_meta:
                continue
            p.copy_(draw(p.shape).to(device=p.device, dtype=p.dtype))


def build(layer: nn.Module, d_model: int, cfg: GDNConfig, heads: tuple | None = None) -> DeltaProductSteps | None:
    """GatedDeltaNet's hook (its __init__): None for n_h 1 (the layer as it was: nothing built, nothing drawn); else the extra
    steps' tensors, after the layer-level refusals. `layer` is the GatedDeltaNet under construction (its in_proj_qkv gives the
    device and dtype). heads: the carrying value heads (None: every head; gdn_dprod_place.py)."""
    n = check_beta_max(cfg, n_steps(cfg))  # n_h, beta_max and beta_init first, then the route's refusals
    bad_route = one_readout_refusals(cfg)
    if bad_route:
        raise ValueError(f"GDNConfig.dprod.route 'one-readout' (ced/kernels/gdn_1r.py) is not built with: {'; '.join(bad_route)}")
    if n == 1:
        return _no_steps(heads, cfg)
    refused = [(cfg.kernel not in KERNELS, f"kernel {cfg.kernel!r} (built for {KERNELS})"),
               (cfg.edges != "plain", f"edges {cfg.edges!r} (the fused edges ops form q, k, v of one step in our chunk kernels)"),
               (cfg.chunk_backend != "triton", f"chunk_backend {cfg.chunk_backend!r} (built on our Triton kernel's call)")]
    bad = [why for cond, why in refused if cond]
    if bad:
        raise ValueError(f"GDNConfig.dprod.n_h {n} (Gated DeltaProduct, ced/model/gdn_dprod.py) is not built with: {'; '.join(bad)}")
    w = layer.in_proj_qkv.weight
    return _steps_of(d_model, cfg, w, heads)


def own_generator() -> torch.Generator:
    return torch.Generator().manual_seed((torch.initial_seed() + SEED_OFFSET) % (1 << 63))


def init_own_parameters(model: nn.Module, std: float) -> int:
    """Draw every DeltaProductSteps of `model`, in module order, from own_generator() (DeltaProductSteps.reset_own). Called by the
    model's build after init_weights; returns how many were drawn (0 in every model without the key: then nothing is drawn and
    no generator is made)."""
    steps = _own_steps(model)
    if not steps:
        return 0
    g = own_generator()
    for m in steps:
        m.reset_own(g, std)
    return len(steps)


def check_config(cfg) -> None:
    """The model-level refusals (ced/model/control.py build_model): GDNConfig.dprod.n_h checked in both GDN sections; n_h > 1 only
    in arch "control"'s encoder, without the GDN-side arms, the gated second pass, the MTP drafter or FP8 / NVFP4 linears."""
    enc, dec = (check_beta_max(g, n_steps(g)) for g in (cfg.encoder.gdn, cfg.decoder.gdn))  # n_h and beta_max, both sections
    if dec != 1:
        raise ValueError(f"model.decoder.gdn.dprod.n_h {dec}: Gated DeltaProduct is built for the encoder's GDN layers only")
    if route_of(cfg.decoder.gdn) != "interleave":
        raise ValueError(f"model.decoder.gdn.dprod.route {route_of(cfg.decoder.gdn)!r}: the decoder's GDN layers have no second step")
    bad_route = one_readout_refusals(cfg.encoder.gdn)
    if bad_route:
        raise ValueError(f"model.encoder.gdn.dprod.route 'one-readout' (ced/kernels/gdn_1r.py) is not built with: {'; '.join(bad_route)}")
    if _place_check(cfg, enc):
        return
    refused = [(cfg.arch != "control", f"arch {cfg.arch!r} (built for 'control')"),
               (cfg.encoder.gdn_write_mix != "none", f"model.encoder.gdn_write_mix {cfg.encoder.gdn_write_mix!r} (the GDN-side arm A)"),
               (bool(cfg.encoder.gdn_ccr_spacing), f"model.encoder.gdn_ccr_spacing {cfg.encoder.gdn_ccr_spacing!r} (the GDN-side arm CCR)"),
               (cfg.loop.pass2_gate != "none", f"model.loop.pass2_gate {cfg.loop.pass2_gate!r} (the zero-write second pass reads "
                                               "GDN's record)"),
               (bool(cfg.mtp.enabled), "model.mtp.enabled (the drafter writes the GDN step itself, ced/infer/mtp_spec.py)"),
               (cfg.fp8.mode != "none", f"model.fp8.mode {cfg.fp8.mode!r}"),
               (cfg.nvfp4.mode != "none", f"model.nvfp4.mode {cfg.nvfp4.mode!r}")]
    bad = [why for cond, why in refused if cond]
    if bad:
        raise ValueError(f"model.encoder.gdn.dprod.n_h {enc} (Gated DeltaProduct, ced/model/gdn_dprod.py) is not built with: "
                         f"{'; '.join(bad)}")


# --------------------------------------------------------------------------------------------------------------------------
# the rule: the reference loop and the interleave


def dprod_recurrence(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                     is_doc_start: torch.Tensor, S: torch.Tensor, all_states: bool = False):
    """Gated DeltaProduct, one token and one step at a time (the ground truth; module docstring, THE RULE).

    q: (B, L, H, d_k), at the value heads (repeated), scaled; k: (B, L, n_h, H, d_k) unit keys; v: (B, L, n_h, H, d_v);
    g: (B, L, H) log decay; beta: (B, L, n_h, H); is_doc_start: (B, L) bool; S: (B, H, d_k, d_v), the state before the first
    token. Per token: reset at a document start, decay by exp(g_t) once, then the n_h delta-rule steps in order, then read
    S^T q_t. Returns the outputs (B, L, H, d_v) and the final state; all_states: also the state after every token
    (B, L, H, d_k, d_v)."""
    n_h = k.shape[2]
    outputs, states = [], []
    # unbind once (indexing in the loop would make the backward O(L^2), as delta_rule_recurrence's note says)
    for q_t, k_t, v_t, g_t, b_t, start_t in zip(q.unbind(1), k.unbind(1), v.unbind(1), g.unbind(1), beta.unbind(1),
                                                is_doc_start.unbind(1)):
        S = S * (~start_t).to(S.dtype)[:, None, None, None]
        S = S * torch.exp(g_t)[:, :, None, None]
        for j in range(n_h):
            kj = k_t[:, j]
            predicted = torch.einsum("bhkv,bhk->bhv", S, kj)
            S = S + torch.einsum("bhk,bhv->bhkv", kj, b_t[:, j, :, None] * (v_t[:, j] - predicted))
        outputs.append(torch.einsum("bhkv,bhk->bhv", S, q_t))
        if all_states:
            states.append(S)
    if all_states:
        return torch.stack(outputs, dim=1), S, torch.stack(states, dim=1)
    return torch.stack(outputs, dim=1), S


def interleave(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
               is_doc_start: torch.Tensor | None = None):
    """The n_h T interleave (module docstring, THE PATHS): per token t the sub-steps n_h t .. n_h t + n_h - 1.

    q (B, L, Hq, d_k) -> (B, n_h L, Hq, d_k), q_t on the last sub-step, zeros before it; k (B, L, n_h, Hk, d_k) and
    v (B, L, n_h, Hv, d_v) -> (B, n_h L, ., .), step j on sub-step j; g (B, L, H) -> (B, n_h L, H), g_t on the first sub-step,
    zeros after it; beta (B, L, n_h, H) -> (B, n_h L, H); is_doc_start (B, L) -> (B, n_h L), on the first sub-step only.
    Returns (q, k, v, g, beta, is_doc_start or None). Stack and reshape only (fixed shapes, no index op)."""
    B, L = q.shape[:2]
    n_h = k.shape[2]
    flat = lambda t: t.reshape(B, n_h * L, *t.shape[3:])  # noqa: E731
    zq, zg = torch.zeros_like(q), torch.zeros_like(g)
    qi = flat(torch.stack([zq] * (n_h - 1) + [q], dim=2))
    gi = flat(torch.stack([g] + [zg] * (n_h - 1), dim=2))
    si = None
    if is_doc_start is not None:
        si = flat(torch.stack([is_doc_start] + [torch.zeros_like(is_doc_start)] * (n_h - 1), dim=2))
    return qi, flat(k), flat(v), gi, flat(beta), si


def last_substep(o: torch.Tensor, n_h: int) -> torch.Tensor:
    """(B, n_h L, H, d_v) interleaved outputs -> (B, L, H, d_v): each token's output, read on its last sub-step."""
    B, T = o.shape[:2]
    return o.reshape(B, T // n_h, n_h, *o.shape[2:])[:, :, n_h - 1]


# --------------------------------------------------------------------------------------------------------------------------
# the layer's forward with the extra steps (GatedDeltaNet.forward's hook)


def _refuse_call(layer, ctx, valid, keep, arms, x_write, ccr, x_read) -> None:
    refused = [(keep, "keep (the zero-write second pass, control (c) / loop.pass2_gate, reads GDN's one-step record)"),
               (arms is not None or x_write is not None or ccr is not None or x_read is not None or layer.arms_attached,
                "the GDN-side arms (ced/model/gdn_arms.py)"),
               (valid is not None, "padding (valid: decoder slots)"),
               (ctx.gdn_scale != 1, f"a Token Superposition Training bag phase (SeqContext.gdn_scale {ctx.gdn_scale})"),
               (ctx.gdn_chunks is not None, "CUDA graphs' fixed-capacity GDN metadata (TrainConfig.cuda_graphs: built for "
                                            "the token boundaries, not the n_h T interleave's)"),
               (bool(getattr(layer, "mx_out_norm", False)), "model.fp8.chains gdn_out_norm")]
    bad = [why for cond, why in refused if cond]
    if bad:
        raise ValueError(f"GDNConfig.dprod.n_h {layer.dprod.n_extra + 1} (Gated DeltaProduct, ced/model/gdn_dprod.py) is not "
                         f"built with: {'; '.join(bad)}")


@torch.compiler.disable  # never traced, as the kernels' own wrappers: the boundaries' host copy is a CPU tensor
def _fused_call(fused, q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu, n_h: int, **kw):
    """The fused delta rule on the interleave: the token boundaries (device and host copies) times n_h, the sub-steps' boundaries.
    kw: the layer's _ced_kw ({} for fla, refused with F3; {"f3": True} for our kernel with F3)."""
    cu_cpu = cu_seqlens_cpu * n_h if cu_seqlens_cpu is not None else None
    return fused(q, k, v, g, beta, cu_seqlens * n_h, cu_cpu, **kw)


def load_one_readout():
    """ced/kernels/gdn_1r.py's one_readout_delta_rule, imported at the first call (Triton is in the GPU runs' venv, not on the VM). A
    function, so the CPU tests can route the fused path through the reference (ced/kernels/gdn_1r_ref.py)."""
    from ced.kernels.gdn_1r import one_readout_delta_rule

    return one_readout_delta_rule


@torch.compiler.disable  # never traced: an opaque fused op on varlen inputs, as _fused_call
def _one_readout_call(q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu, **kw):
    """The one-readout route (module docstring): q (B, L, H_k, d_k) the readout queries, k (B, L, 2, H_k, d_k), v (B, L, 2, H_v, d_v), beta
    (B, L, 2, H_v) the layer's stacks (flattened to the sub-step rows: a view of a contiguous stack), g (B, L, H_v) the log decay of the
    tokens, cu_seqlens the TOKEN boundaries (device and host copies; the kernels take twice them themselves). kw: the layer's _ced_kw
    ({} or {"f3": True}). Returns (B, L, H_v, d_v)."""
    B, L = q.shape[:2]
    g_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    RAN["one_readout"] += 1
    o = load_one_readout()(q.reshape(1, B * L, *q.shape[2:]), k.reshape(1, 2 * B * L, *k.shape[3:]), v.reshape(1, 2 * B * L, *v.shape[3:]),
                           g.reshape(1, B * L, -1).to(g_dtype), beta.reshape(1, 2 * B * L, -1).to(q.dtype), cu_seqlens.long(),
                           cu_seqlens_cpu.long() if cu_seqlens_cpu is not None else None, **kw)
    return o.reshape(B, L, *o.shape[2:])


def _unit(x: torch.Tensor) -> torch.Tensor:
    """The layer's L2 norm (ced/model/gdn.py l2_normalize) in float32 at least, returned in x's dtype."""
    from ced.model.gdn import l2_normalize

    return l2_normalize(x.to(torch.promote_types(x.dtype, torch.float32))).to(x.dtype)


def dprod_forward(layer, x: torch.Tensor, ctx, state=None, valid=None, keep: bool = False, arms=None, x_write=None, ccr=None,
                  x_read=None, lerped: bool = False):
    """GatedDeltaNet.forward for a layer with Gated DeltaProduct (layer.dprod not None): the same inputs and result, (y, GDNState).
    The layer's own q, k_1, v_1 and the extra steps' k_j, v_j come from one causal conv over [in_proj_qkv(x) | in_proj_kv(x)]
    (the fused prep on the fused path), so the state's conv history holds both: GDNState.conv_input (B, K-1, C + (n_h - 1) c)."""
    from ced.model import gdn as G
    from ced.model.gdn_chunked import chunked_delta_rule

    _refuse_call(layer, ctx, valid, keep, arms, x_write, ccr, x_read)
    B, L, _ = x.shape
    cfg, steps = layer.cfg, layer.dprod
    hk, hv, dk, dv, K = cfg.n_k_heads, cfg.n_v_heads, cfg.k_head_dim, cfg.v_head_dim, cfg.conv_kernel
    n_x = steps.n_extra
    n_h = n_x + 1
    u_all = torch.cat([layer.in_proj_qkv(x), steps.in_proj_kv(x)], dim=-1)  # [q | k_1 | v_1 | k_2 | v_2 | ...]
    w_all = torch.cat([layer.conv_weight, steps.conv_weight], dim=0)
    if state is not None and state.conv_input.shape[-1] != u_all.shape[-1]:
        raise ValueError(f"this GDN cache holds {state.conv_input.shape[-1]} conv channels; a layer with Gated DeltaProduct "
                         f"n_h {n_h} carries {u_all.shape[-1]} (its own and the extra steps')")
    use_fused = cfg.kernel in ("fla", "ced") and state is None and ctx.cu_seqlens is not None and G.fla_supported(x)
    kernel_dtype = (torch.bfloat16 if use_fused and cfg.input_dtype == "bfloat16" and u_all.dtype == torch.bfloat16 else None)
    rest_heads = hv + n_x * (hk + hv)  # the prep's value slots: v_1, then k_j, v_j of every extra step (d_k = d_v there)
    if layer._fused_prep and state is None and ctx.cu_seqlens is not None and G.fla_supported(x) and L >= K - 1:
        # one fused prep over every channel (ced/kernels/gdn.py qkv_prep): q scaled and unit, k_1 unit, the rest SiLU(conv)
        q, k1, rest = G.qkv_prep(u_all, ctx.doc, w_all, hk, rest_heads, dk, out_dtype=kernel_dtype)
        rest = rest.reshape(B, L, rest_heads * dk)
        conv_input, conv_doc = u_all[:, L - (K - 1):], ctx.doc[:, L - (K - 1):]
        unit = _unit  # the extra keys leave the prep as SiLU(conv) in its output dtype: their norm in float32
    else:
        hist, hist_doc = (None, None) if state is None else (state.conv_input, state.conv_doc)
        u, conv_input, conv_doc = G.causal_doc_conv(u_all, ctx.doc, w_all, hist, hist_doc)
        u = F.silu(u)
        q, k1, rest = torch.split(u, [hk * dk, hk * dk, u.shape[-1] - 2 * hk * dk], dim=-1)
        q = G.l2_normalize(q.reshape(B, L, hk, dk)) * dk**-0.5
        k1 = G.l2_normalize(k1.reshape(B, L, hk, dk))
        unit = G.l2_normalize  # every key as the layer's k_1: in the conv's dtype
    parts = torch.split(rest, [hv * dv] + [hk * dk, hv * dv] * n_x, dim=-1)
    ks = [k1] + [unit(parts[1 + 2 * j].reshape(B, L, hk, dk)) for j in range(n_x)]
    vs = [parts[2 * j].reshape(B, L, hv, dv) for j in range(n_x + 1)]
    # beta = bmax sigmoid(b + boff) on EVERY step, bmax = beta_multiplier(cfg) (dprod.beta_max 2: BETA_MAX; 1: 1), boff = beta_offset(cfg)
    bmax, boff = beta_multiplier(cfg), beta_offset(cfg)  # (boff None: beta_init absent, b as it is. negeig with dprod is refused: gdn.py)
    g, beta1 = layer.gates(x) if boff is None else gates_offset(layer, x, boff)  # step 1's from gates(), the extra steps' from dprod.in_proj_b
    betas = [bmax * beta1] + list((bmax * torch.sigmoid(offset_b(steps.in_proj_b(x), boff))).reshape(B, L, n_x, hv).unbind(2))
    k = torch.stack(ks, dim=2)  # (B, L, n_h, H_k, d_k)
    v = torch.stack(vs, dim=2)  # (B, L, n_h, H_v, d_v)
    beta = torch.stack(betas, dim=2)  # (B, L, n_h, H_v)
    if kernel_dtype is not None:
        q, k, v, beta = q.to(kernel_dtype), k.to(kernel_dtype), v.to(kernel_dtype), beta.to(kernel_dtype)

    if use_fused and cfg.kda.kernel == "k2":
        # K2 (KDAConfig.kernel "k2"; ced/kernels/gdn_k2.py): the n_h steps of a token as n_h rows of one WY form, the stacks' own
        # row layout, no interleave; grouped heads kept
        from ced.model import gdn_kda

        o = gdn_kda.k2_fused(q, k, v, g, beta, ctx.cu_seqlens, ctx.cu_seqlens_cpu)
        S = None
    elif use_fused and route_of(cfg) == "one-readout":
        # the one-readout route (GDNConfig.dprod.route; ced/kernels/gdn_1r.py): our chunk kernels on the stacks' own sub-step rows, the
        # readout once a token, q and g at the token rows: no interleave copy of q or g, no readout on the sub-step with q = 0
        o = _one_readout_call(q, k, v, g, beta, ctx.cu_seqlens, ctx.cu_seqlens_cpu, **layer._ced_kw)
        S = None
    elif use_fused:
        # the n_h T interleave into the fused kernel as is (module docstring): boundaries x n_h, grouped heads kept
        qi, ki, vi, gi, bi, _ = interleave(q, k, v, g, beta)
        fused = G.fla_delta_rule if cfg.kernel == "fla" else G.ced_delta_rule
        o = last_substep(_fused_call(fused, qi, ki, vi, gi, bi, ctx.cu_seqlens, ctx.cu_seqlens_cpu, n_h, **layer._ced_kw), n_h)
        S = None  # training / scoring forward: the final state is not needed
    else:
        rep = hv // hk
        q_r, k_r = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=3)
        S0 = x.new_zeros(B, hv, dk, dv) if state is None else state.S
        if cfg.kernel in ("chunked", "fla", "ced") and L > 1:
            qi, ki, vi, gi, bi, si = interleave(q_r, k_r, v, g, beta, ctx.is_doc_start)
            oi, S = chunked_delta_rule(qi, ki, vi, gi, bi, si, S0, cfg.chunk_size)
            o = last_substep(oi, n_h)
        else:
            o, S = dprod_recurrence(q_r, k_r, v, g, beta, ctx.is_doc_start, S0)
    if layer.head_keep is not None:  # evaluation only: the zeroed heads' readouts are 0 (GatedDeltaNet.set_head_zero)
        o = layer.keep_heads(o)
    z = layer.in_proj_z(x).view(B, L, hv, dv)
    y = layer.out_proj(layer.norm(o, z).flatten(2))
    return y, G.GDNState(conv_input=conv_input, conv_doc=conv_doc, S=S)


# --------------------------------------------------------------------------------------------------------------------------
# GDNConfig.dprod.beta_max. Appended below the code so that no line above moves (tools/analysis/gdn_dprod_split.py cites this file's lines).

BETA_MAXES = (1, 2)  # the values built: 1 (beta = sigmoid(b) in (0, 1)) and 2 (beta = BETA_MAX sigmoid(b) in (0, 2), the default)


def beta_max_of(cfg: GDNConfig) -> int:
    """GDNConfig.dprod.beta_max, checked: the int 1 or 2 (not a bool, not 2.0), else ValueError naming the key. 2 is the default (every
    model before the key). A section without a `dprod` attribute reads as the default."""
    m = getattr(getattr(cfg, "dprod", None), "beta_max", 2)
    if isinstance(m, bool) or not isinstance(m, int) or m not in BETA_MAXES:
        raise ValueError(f"GDNConfig.dprod.beta_max {m!r}: 1 (every step's beta = sigmoid(b) in (0, 1): no reflection, no rotation) or 2 "
                         "(beta = 2 sigmoid(b) in (0, 2), the default)")
    return m


def check_beta_max(cfg: GDNConfig, n: int) -> int:
    """beta_max_of(cfg) checked against n = n_steps(cfg), returning n: beta_max 1 needs n_h >= 2, because n_h 1 is the layer as it was
    (its beta = sigmoid(b) from gates(), no dprod tensor, nothing of this file runs) and a key that scales nothing is refused, not ignored.
    beta_init (beta_init_of) is checked here too: with n_h 1 and with beta_max 2 it is refused by name."""
    m = beta_max_of(cfg)
    if n == 1 and m != 2:
        raise ValueError(f"GDNConfig.dprod.beta_max {m} with dprod.n_h 1: Gated DeltaNet (n_h 1) has no such key (its beta is sigmoid(b), "
                         "unscaled); beta_max needs n_h >= 2")
    p = beta_init_of(cfg)
    if p is not None and (n == 1 or m != 1):
        raise ValueError(f"GDNConfig.dprod.beta_init {p!r} with dprod.n_h {n} and dprod.beta_max {m}: the start beta is set for the arm "
                         "with beta in (0, 1) (n_h >= 2 and beta_max 1); with beta_max 2 the beta = 2 sigmoid(b) starts at 1 already")
    return n


def beta_multiplier(cfg: GDNConfig):
    """M in every step's beta = M sigmoid(b), M x sigmoid(b) on step 1 (from gates()) and on steps 2 .. n_h (dprod.in_proj_b) alike:
    beta_max 1 -> 1 (no factor: the multiplication by 1 is exact, so beta is the sigmoid's own bits), beta_max 2 -> BETA_MAX (this
    module's constant, 2: the line this function replaced, `BETA_MAX * ...`, bit for bit; the K1/K2 interp tests set BETA_MAX to
    read the layer as an older snapshot ran it)."""
    return 1 if beta_max_of(cfg) == 1 else BETA_MAX


# --------------------------------------------------------------------------------------------------------------------------
# GDNConfig.dprod.beta_init: the start beta 0.5 is offset to about 0.95, matching the beta in (0, 2) form (start beta 1) as closely as the sigmoid allows.
#
# What it is: a FIXED offset c = logit(beta_init) added to b = x W before the sigmoid on EVERY step, step 1's (layer.in_proj_b, through
# gates_from) and the extra steps' (dprod.in_proj_b) alike: beta = sigmoid(x W + c). b has no bias (both projections are bias=False), and this
# adds none: no parameter, no buffer, no tensor in the state dict, no optimiser group, no weight-decay rule (weights keep their groups and their
# decay: a decayed W pulls x W to 0, that is beta to beta_init, as pulls beta to 1). Not a trainable bias: a bias initialised to c with decay
# would drift to beta 0.5, one without decay is a new group; the offset is the same function at step 0 and keeps the parameter set, the
# checkpoint format, the optimiser and the resume path of the Gated DeltaProduct form with beta in [0, 1] as built.


def beta_init_of(cfg: GDNConfig) -> float | None:
    """GDNConfig.dprod.beta_init, checked: None (the default: no offset), else a real number p with 0 < p < 1 (an int or a float, not a
    bool, finite), else ValueError naming the key. A section without a `dprod` attribute reads as None."""
    p = getattr(getattr(cfg, "dprod", None), "beta_init", None)
    if p is None:
        return None
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not (0.0 < p < 1.0):  # NaN fails the range test too
        raise ValueError(f"GDNConfig.dprod.beta_init {p!r}: None (the default) or a number in (0, 1), the start value of every step's beta")
    return float(p)


def beta_offset(cfg: GDNConfig) -> float | None:
    """c = logit(beta_init) = ln(p) - ln(1 - p), the constant added to b on every step; None for beta_init None (nothing is added: the
    key-absent path runs the lines it always ran). beta_init 0.95 -> 2.944438979166440."""
    p = beta_init_of(cfg)
    return None if p is None else math.log(p) - math.log1p(-p)


def offset_b(b: torch.Tensor, c: float | None) -> torch.Tensor:
    """b + c in b's dtype (c a Python float: no tensor, no promotion), b itself for c None."""
    return b if c is None else b + c


def gates_offset(layer, x: torch.Tensor, c: float):
    """layer.gates(x) with b + c in place of b: (g, beta) with beta = sigmoid(in_proj_b(x) + c). gates_from is gates' lines on the two
    projections' outputs (ced/model/gdn.py: gates(x) == gates_from(in_proj_a(x), in_proj_b(x)) bit for bit); this layer has no KDA, no
    erase and no GDN-side arm (dprod's refusals), so gates() would not have refused."""
    return layer.gates_from(layer.in_proj_a(x), offset_b(layer.in_proj_b(x), c))


# -------------------------------------------------------------------------------------------------------------------------- PLACEMENT (per
# layer and per head, ced/model/gdn_dprod_place.py): three hooks at the end of this module, so that the Gated DeltaProduct form's
# statements above stay as they were (the design and split notes cite them). Lazy imports: gdn_dprod_place imports this module.


def _steps_of(d_model: int, cfg: GDNConfig, weight: torch.Tensor, heads):
    """build's last statement: DeltaProductSteps for every head (heads None or all of them: as before), PartialSteps for some; with a
    second step made from the first (dprod.key lowrank, prev, perm, rot; ced/model/gdn_dprod_c.py) DerivedSteps."""
    from ced.model import gdn_dprod_c as C
    from ced.model.gdn_dprod_place import steps_of

    if C.is_derived(cfg):
        return C.steps_of(d_model, cfg, weight, heads)
    C.check_second(cfg)
    return steps_of(d_model, cfg, weight, heads)


def _no_steps(heads, cfg):
    """build's n_h 1 return: None; carrying heads given to a layer that does not carry are refused, and so are the second step's keys."""
    from ced.model import gdn_dprod_c as C
    from ced.model.gdn_dprod_place import no_steps

    C.check_second(cfg)
    return no_steps(heads)


def _place_check(cfg, enc: int) -> bool:
    """check_config's early return: validates the placement keys and the shape table by name; True when n_h is 1 and there is no shape table.
    The second step's keys (dprod.key and the rest; ced/model/gdn_dprod_c.py) are checked first, in both GDN sections."""
    from ced.model import gdn_dprod_c as C
    from ced.model.gdn_dprod_place import place_check

    C.check_model(cfg)
    return place_check(cfg, enc)


def _own_steps(model: nn.Module) -> list:
    """init_own_parameters' modules: the extra steps of the Gated DeltaProduct form's form (DeltaProductSteps and
    PartialSteps) and of the derived forms (DerivedSteps), in module order; every one has reset_own(generator, std)."""
    from ced.model.gdn_dprod_c import DerivedSteps

    return [m for m in model.modules() if isinstance(m, (DeltaProductSteps, DerivedSteps))]


# -------------------------------------------------------------------------------------------------------------------------- THE SECOND STEP ON A
# SUBSET OF THE TOKENS (GDNConfig.dprod.stride / summary / conv / first_beta; ced/model/gdn_dprod_sparse.py). Appended below the code so that no line
# above moves (tools/analysis/gdn_dprod_split.py cites this file's lines): the two entry points `build` and `check_config` are rebound to wrappers
# that run the new keys' checks and, with the keys at their defaults, call the Gated DeltaProduct form's own functions, kept under the names below.

_arm6_build, _arm6_check_config = build, check_config


def build(layer: nn.Module, d_model: int, cfg: GDNConfig, *args, **kw):  # noqa: F811 (the rebinding described above)
    """GatedDeltaNet's hook: gdn_dprod_sparse.build_steps (the new keys' refusals; build when they are at their defaults)."""
    from ced.model.gdn_dprod_sparse import build_steps

    return build_steps(layer, d_model, cfg, *args, **kw)


def check_config(cfg) -> None:  # noqa: F811 (the rebinding described above)
    """The model-level refusals: gdn_dprod_sparse.check_model."""
    from ced.model.gdn_dprod_sparse import check_model

    check_model(cfg)
