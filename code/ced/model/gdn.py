"""Gated DeltaNet (arXiv 2412.06464), written as a plain sequential recurrence.

Layout follows the Hugging Face Qwen3.5 code (Qwen3_5GatedDeltaNet):

    u      = in_proj_qkv(x)                      # [q | k | v] channels
    u      = SiLU(causal depthwise conv_K(u))    # short conv over q, k, v (not over z, a, b)
    q, k   = L2norm(q), L2norm(k);  q *= dk^-0.5
    beta_t = sigmoid(in_proj_b(x))               # per value head
    g_t    = -exp(A_log) * softplus(in_proj_a(x) + dt_bias);  alpha_t = exp(g_t)
    S_t    = alpha_t S_{t-1};  S_t += k_t (x) beta_t (v_t - S_t^T k_t);  o_t = S_t^T q_t
    out    = out_proj(GatedRMSNorm(o, z = in_proj_z(x)))   # RMSNorm(o) * act(z), act = SiLU or sigmoid

S has shape (B, H_v, d_k, d_v). With more value heads than key heads, value head j reads key head
j // (H_v / H_k) (fla and HF convention).

Document isolation: the recurrent state is zeroed at each document start, and the conv window
only mixes inputs from the same document. Hugging Face's pure-torch fallback ignores cu_seqlens
and leaks both across documents; fla's varlen kernels do not.

This loop over time is the readable ground truth. Faster kernels must match it:
- kernel "chunked": our pure-PyTorch chunked form (gdn_chunked.py);
- kernel "ced": the same function with our kernels (ced/kernels/gdn.py; same conditions as "fla"),
  including the training-path input preparation (conv, SiLU, L2 norm) fused into one kernel each way;
  a drop-in replacement for "fla" everywhere (output gate, keep records, the second pass's extra call);
- kernel "fla": flash-linear-attention's fused Triton `chunk_gated_delta_rule` (CUDA, training
  forward from an empty state with precomputed varlen boundaries, see ced/train/varlen.py). In any
  other case (CPU, cached chunks with a carried state) "fla" falls back to "chunked" / the
  recurrence, so the inference path stays the reference.

Evaluation-only gate floor (GatedDeltaNet.set_alpha_floor; the gate-floor re-score): for chosen value heads,
alpha_t = max(alpha_t, floor), applied to g in `gates`, before the state recurrence of every kernel above (all of
them take g as an input: none computes the gate itself, and none was edited). Off by default (g_floor None): the
layer as it always was, bit for bit. Not a config field, not in a checkpoint.

Evaluation-only gate CAP (GatedDeltaNet.set_alpha_cap): for chosen value heads, alpha_t = min(alpha_t, cap_h), that is g_t
= min(g_t, ln cap_h), so the head's horizon h_t = 1 / (1 - alpha_t) is at most 1 / (1 - cap_h) at every token. The same
place as the floor (in `gates` and `gates_from`, after it: with both on one head the cap wins), so every kernel takes the
capped g and none was edited. Off by default (g_cap None): the layer as it always was, bit for bit. Not a config field,
not in a checkpoint.

Evaluation-only head zero (GatedDeltaNet.set_head_zero): for chosen value heads, the head's readout o_t = S_t^T q_t (the delta rule's
output, (B, L, H_v, d_v)) is multiplied by 0 BEFORE the gated output norm, the output gate and out_proj, so the head's slice of out_proj's
input is exactly 0 and the head contributes exactly nothing to the layer's output (RMSNorm(0) * act(z) = 0). The head's state still
evolves; only what is read from it is dropped. A fixed-shape per-head multiplier (1.0 keep, 0.0 zero), not an index. It acts in every
forward that forms o: this file's (the reference loop, the chunked rule, the fused kernels "ced" and "fla"), Gated DeltaProduct's, KDA's
and the erase's (gdn_dprod, gdn_kda, gdn_erase), the zero-write second pass's reads; the decode engine and the MTP drafter write the step
themselves and refuse by name. Off by default (head_keep None): the layer as it always was, bit for bit. Not a config field, not in a
checkpoint.

F3, the compensated cumulative sum (GDNConfig.decay_cumsum "compensated"): every call of our kernel "ced" (pass 1, the
read over its chunk states, the extra call) gets f3=True and forms its decays from a (hi, lo) float pair. "plain" (the
default): the calls are as they always were. The reference paths (recurrence, chunked) sum exponents directly and
ignore it; kernel "fla" refuses it.

D, the chunk backend (GDNConfig.chunk_backend): "triton" (the default) is the layer as it was. "flashqla": kernel "ced"'s
fused delta-rule calls (pass 1, and the second pass's extra call: the chunk-state read is our Triton kernel's, so
read_states is off) run FlashQLA's exact mode (flashqla_delta_rule; ced/kernels/gdn_flashqla.py), with F3 through its
patched kernels (SM90; SM100 lowered only); the fused input preparation, the gates and the output norm stay the layer's
own. Refused by name with CCR's writer and CUDA graphs; on the CPU the layer runs the reference path, as kernel "ced"
does.

Token Superposition Training's resolution-explicit gates (TSTConfig.gdn_resolution_gates, "E"; 2026-10-16): in a bag
phase of size s the forward's SeqContext carries gdn_scale = s, and `gates` returns g = s x the per-token log decay
(and, with gdn_beta_power, the write strength 1 - (1 - beta)^s). Like the floor, it acts in `gates`, before every kernel
(none was edited). gdn_scale 1 (every other forward): the gates as they always were, the same lines. E composes with F3:
E scales g in `gates`, F3 changes how our kernel sums that g; every kernel call takes the scaled g whichever sum it
uses.

The GDN-side arms (EncoderConfig.gdn_write_mix "A" and gdn_ccr_spacing "CCR"; ced/model/gdn_arms.py; the forward's
arms / x_write / ccr arguments, None in every model without them) compose with E and F3 (merge of arch-gdn-read,
2026-10-16; tests/test_arch_gdn_tst_f3.py). An A layer's decay and beta read x_w, so its gates come from `gates_from`
(the same lines as `gates` on in_proj_a(x_w) and in_proj_b(x_w)) with E's scale and write-strength flag, and `gates(x)`
refuses it. CCR's writer calls our kernel through ced_delta_rule_states with the same (scaled) g and F3's flag, and its
reader reads the chunk-start states that call forms. In a TST bag phase the spacing C counts positions (bags of s
tokens), as every other position of that phase does.

Negative eigenvalues (GDNConfig.negeig.beta_max; ced/model/gdn_negeig.py): with beta_max 2, `gates` and `gates_from`
return beta = 2 sigmoid(b) in (0, 2), before every kernel (none was edited: each takes beta as an input). beta_max 1 (the
default): the gate lines as they always were.

Per-channel decay (KDA) and the horizon floor (GDNConfig.kda, off by default; ced/model/gdn_kda.py): with kda.decay "channel" the
layer holds `kda` (a low-rank decay projection and a per-channel dt_bias) in place of in_proj_a and the per-head dt_bias, and its
forward is gdn_kda.forward (the first line of `forward`); gates() and gates_from() refuse it (their callers read a scalar per head)
and channel_gates() gives its (B, L, H_v, d_k) decay. Off ("scalar"): `kda` is None and nothing below differs from the layer as it
was.

EDA's erase direction (GDNConfig.erase, off by default; ced/model/gdn_erase.py): with erase.mode "eda" the layer holds `erase` (the
erase address's low-rank projection, the gate's projection and its zero-initialised scale) and its forward is gdn_erase.forward (the
first line of `forward`), on either decay (GDN's, or KDA's with `kda`); gates() refuses it (its callers walk the state without the
erase). Off ("none"): `erase` is None and nothing below differs from the layer as it was.

Gated DeltaProduct (GDNConfig.dprod.n_h; ced/model/gdn_dprod.py): n_h delta-rule steps a token, the decay once. n_h 1 (the
default): `self.dprod` is None and this file's code runs as it always did. n_h >= 2: the layer holds the extra steps' tensors
in `self.dprod` and its forward is gdn_dprod.dprod_forward (the reference loop, the chunked rule and our fused kernel on the
n_h T interleave); `gates` is unchanged (one scalar decay per value head and token).

Zero-write second pass (control (c), ced/model/second_pass.py): `forward(... keep=True)` also
returns what a second pass needs from this layer (GDNPass1), and `zero_write_read` computes the
second pass's read o2_t = S1_t^T q2_t at chosen positions without writing the state. With kernel
"ced" the read reuses pass 1's chunk states (one state recursion forward and one backward per layer
for both passes).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ced.config import GDNConfig
from ced.kernels import gdn_c2  # C2 (edges "once"): registers its custom ops at import, as gdn_edges
from ced.kernels import gdn_edges  # C (edges): registers its custom ops at import, never while a graph is traced
from ced.model import gdn_kda  # per-channel decay (GDNConfig.kda): imports nothing of this module at load
from ced.model import gdn_erase  # the erase direction (GDNConfig.erase): imports nothing of this module at load
from ced.model.capacity import gather_rows, gather_slots, gather_slots_inv, scatter_slots, scatter_slots_inv
from ced.model.norms import GatedRMSNorm
from ced.model import gdn_negeig
from ced.model.gdn_chunked import chunked_delta_rule
from ced.model.sequence import SeqContext

# GDNConfig.edges: "plain" (the layer as it was), "fused" (the fused op where today's two opaque calls are, today's graph
# breaks kept), "fused-traced" (the fused op traced into the layer's graph); C2: "once" (q, k, v formed once a chunk in
# the first kernel that reads it, ced/kernels/gdn_c2.py; the op where "fused"'s is, today's graph breaks)
EDGES_CHOICES = ("plain", "fused", "fused-traced", "once")
# gates() / gates_from() of a layer with per-channel decay (GDNConfig.kda): their callers read one decay per value head
# the decode engine and the MTP drafter write a GDN step themselves; they do not read the evaluation-only head zero (set_head_zero)
HEAD_ZERO_REFUSAL = ("this GDN layer has an evaluation-only head zero set (GatedDeltaNet.set_head_zero): "
                     "the decode engine and the MTP drafter write the GDN step themselves and would read the heads the zero "
                     "removes; take it off with set_head_zero() before decoding")
_PER_CHANNEL_REFUSAL = ("this GDN layer's decay is per key channel (GDNConfig.kda.decay 'channel'; ced/model/gdn_kda.py): gates(x) "
                        "and gates_from give one decay per value head, which this layer does not have; channel_gates(x) gives its "
                        "(B, L, H_v, d_k) decay. A tool that computes a GDN layer from gates (the decode engine, the MTP drafter, "
                        "the GDN state diagnostics) is not built for it")
# GDNConfig.chunk_backend: "triton" (our Triton chunk kernels, the layer as it was), "flashqla" (FlashQLA's TileLang kernels
# in their exact mode for the delta-rule core of kernel "ced"; ced/kernels/gdn_flashqla.py)
CHUNK_BACKENDS = ("triton", "flashqla")


def _refuse_unbuilt_rule_pairs(beta_max: int, cfg: GDNConfig, dprod) -> None:
    """Refuse two GDN memory-rule options on one layer unless the pair is built: only KDA (GDNConfig.kda.decay "channel") with the
    erase (GDNConfig.erase.mode "eda") is. Negative eigenvalues (negeig.beta_max 2) with KDA, the erase or DeltaProduct, and
    DeltaProduct (dprod.n_h >= 2) with KDA or the erase, are not: KDA's gate forms beta = sigmoid(b) (negeig's 2 sigmoid(b)
    would not hold there), the erase's paths were checked at beta <= 1, and the erase and KDA forwards never run DeltaProduct's
    extra steps. Negative eigenvalues with DeltaProduct: DeltaProduct already takes beta = 2 sigmoid(b) on every step
    (gdn_dprod.BETA_MAX), so negeig.beta_max 2 adds no range; with it gates() would hand step 1 2 sigmoid(b) and dprod_forward
    would double it again, to (0, 4), a rule nobody asked for: refused, not composed. Nothing is refused with at most one option
    on (every model before these keys has none)."""
    on = [name for cond, name in ((beta_max != 1, f"negeig.beta_max {beta_max}"),
                                  (cfg.kda.decay != "scalar", f"kda.decay {cfg.kda.decay!r}"),
                                  (cfg.erase.mode != "none", f"erase.mode {cfg.erase.mode!r}"),
                                  (dprod is not None, f"dprod.n_h {cfg.dprod.n_h}")) if cond]
    if len(on) > 1 and not (len(on) == 2 and cfg.kda.decay != "scalar" and cfg.erase.mode != "none"):
        why = ("; negeig with dprod: DeltaProduct already takes beta = 2 sigmoid(b) on every step, "
               "so beta_max 2 adds no range and would double step 1's beta again, to (0, 4)"
               if beta_max != 1 and dprod is not None else "")
        raise ValueError(f"GDNConfig {' with '.join(on)}: these GDN memory-rule options are not built together (only kda with "
                         f"erase is, arms 4 and 5); each forward reads its own option and would drop the other's setting{why}")


@dataclass
class GDNState:
    conv_input: torch.Tensor  # (B, K-1, C) last K-1 pre-conv inputs
    conv_doc: torch.Tensor  # (B, K-1) their document ids (-1 = none)
    S: torch.Tensor  # (B, H_v, d_k, d_v) recurrent state
    # The GDN-side arm CCR only (EncoderConfig.gdn_ccr_spacing; ced/model/gdn_arms.py); None in every other model:
    # ckpt (B, H_v, d_k, d_v): a writer layer's checkpoint that the next token's reader reads (the state after the first
    # C floor(p / C) tokens of the document, p the next token's position); ccr: the record this forward hands the next
    # GDN layer (the encoder's walk takes it out of the cache entry).
    ckpt: torch.Tensor | None = None
    ccr: object | None = None


@dataclass
class GDNPass1:
    """What a zero-write second pass needs from pass 1 of one GDN layer over a chunk of L positions.

    q_inputs: (B, K-1+L, H_k*d_k) pre-conv q-channel inputs (in_proj_qkv output): the K-1 entries
              of the conv history before the chunk, then the chunk's own
    q_docs:   (B, K-1+L) their document ids (-1 = none)

    Then either (pass 1 ran the per-token recurrence, the reference)
    states:   (B, L, H_v, d_k, d_v) the recurrent state after every position of the chunk
    or (pass 1 ran a chunked kernel) the inputs of one more delta-rule call over the chunk:
    k:        (B, L, H_k, d_k) L2-normalised keys (not yet repeated to the value heads)
    v:        (B, L, H_v, d_v); g, beta: (B, L, H_v); is_doc_start: (B, L) bool
    S0:       (B, H_v, d_k, d_v) the state before the chunk (chunked kernel)
    fused_kernel, cu_seqlens(_cpu): pass 1 ran this fused varlen kernel ("fla" or "ced") with these
              boundaries (None: the chunked kernel); the second pass's extra call runs the same one
    or (pass 1 ran kernel "ced" with its chunk states exposed: the default for "ced", see
    GatedDeltaNet.read_states) what the read needs, and no extra call is made:
    h:        (1, NT, H_v, d_k, d_v) pass 1's state at the start of every chunk of the flattened
              varlen sequence; v_new: (1, B*L, H_v, d_v) (ced/kernels/gdn.py ced_delta_rule_states);
              with k, g, fused_kernel "ced" and cu_seqlens(_cpu) above (v and beta are not kept)

    All tensors are pass 1's own autograd nodes, never detached: pass 2's loss trains pass 1.
    """

    q_inputs: torch.Tensor
    q_docs: torch.Tensor
    states: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    g: torch.Tensor | None = None
    beta: torch.Tensor | None = None
    is_doc_start: torch.Tensor | None = None
    S0: torch.Tensor | None = None
    fused_kernel: str | None = None
    cu_seqlens: torch.Tensor | None = None
    cu_seqlens_cpu: torch.Tensor | None = None
    h: torch.Tensor | None = None
    v_new: torch.Tensor | None = None


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def causal_doc_conv(
    u: torch.Tensor, doc: torch.Tensor, weight: torch.Tensor, hist: torch.Tensor | None = None,
    hist_doc: torch.Tensor | None = None, n_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Causal depthwise conv over [history ; u], using only inputs from the current document: the short conv of a GDN
    layer (GatedDeltaNet.short_conv) and of a Canon layer (ced/model/canon.py).

    u (B, L, C) inputs; doc (B, L) their document ids; weight (C, K): weight[:, K-1] multiplies the current input,
    weight[:, K-1-j] the input j steps back; hist (B, K-1, C) / hist_doc (B, K-1): the last K-1 inputs of earlier
    chunks and their document ids (None: no earlier chunk, zeros with document id -1, which no position has).
    A tap whose input is in another document than the position (or before the row) contributes nothing: every
    document is convolved as if it stood alone with K-1 zeros before it.

    Returns the conv output and the new history: the last K-1 inputs and their document ids,
    or, when only the first n_valid positions of each row are real (decoder slots), the last
    K-1 real ones.
    """
    B, L, C = u.shape
    K = weight.shape[1]
    if hist is None:
        hist = u.new_zeros(B, K - 1, C)
        hist_doc = torch.full((B, K - 1), -1, dtype=torch.long, device=u.device)
    full = torch.cat([hist, u], dim=1)  # (B, K-1+L, C)
    full_doc = torch.cat([hist_doc, doc], dim=1)
    out = torch.zeros_like(u)
    for j in range(K):  # tap j looks j steps back
        start = K - 1 - j
        shifted = full[:, start : start + L]
        same_doc = (full_doc[:, start : start + L] == doc).to(u.dtype)
        out = out + shifted * same_doc[..., None] * weight[:, K - 1 - j]
    if n_valid is None:
        return out, full[:, L:], full_doc[:, L:]
    # [history ; u] holds real entries up to index (K-1) + n_valid; keep the K-1 before that.
    take = n_valid[:, None] + torch.arange(K - 1, device=u.device)[None, :]  # (B, K-1)
    hist = torch.gather(full, 1, take[..., None].expand(B, K - 1, C))
    hist_doc = torch.gather(full_doc, 1, take)
    return out, hist, hist_doc


class GatedDeltaNet(nn.Module):
    def __init__(self, d_model: int, cfg: GDNConfig, norm_eps: float, norm_gain: str = "channel", dprod_heads: tuple | None = None):
        """norm_gain: the gain of the gated output norm (self.norm): "channel" (the default, a (v_head_dim,) gain: the
        layer it always was) or "scalar" (one gain; EncoderConfig.block_norm_gain). dprod_heads: the carrying value heads (gdn_dprod_place.py)."""
        super().__init__()
        assert cfg.n_v_heads % cfg.n_k_heads == 0, "value heads must be a multiple of key heads"
        self.cfg = cfg
        hk, hv, dk, dv = cfg.n_k_heads, cfg.n_v_heads, cfg.k_head_dim, cfg.v_head_dim
        self.conv_channels = 2 * hk * dk + hv * dv
        self.in_proj_qkv = nn.Linear(d_model, self.conv_channels, bias=False)
        self.in_proj_z = nn.Linear(d_model, hv * dv, bias=False)
        self.in_proj_b = nn.Linear(d_model, hv, bias=False)
        # GDNConfig.kda (ced/model/gdn_kda.py): None for the scalar decay (no tensor, no random number: the layer as it was);
        # with per-channel decay the gate's tensors, built where in_proj_a was, and no in_proj_a or per-head dt_bias
        self.kda = gdn_kda.build(d_model, cfg)
        if self.kda is None:
            self.in_proj_a = nn.Linear(d_model, hv, bias=False)
        # Depthwise causal conv weight: conv_weight[:, K-1] multiplies the current input,
        # conv_weight[:, K-1-j] the input j steps back.
        self.conv_weight = nn.Parameter(torch.empty(self.conv_channels, cfg.conv_kernel))
        self.A_log = nn.Parameter(torch.empty(hv))
        if self.kda is None:
            self.dt_bias = nn.Parameter(torch.empty(hv))
        self.norm = GatedRMSNorm(dv, norm_eps, gate=cfg.output_gate, gain=norm_gain)
        self.out_proj = nn.Linear(hv * dv, d_model, bias=False)
        self.reset_gate_parameters(cfg.init)
        assert cfg.input_dtype in ("float32", "bfloat16"), cfg.input_dtype
        # F3, the compensated cumulative sum of the decays in our kernel (GDNConfig.decay_cumsum). "plain": nothing
        # below changes (the kernel calls carry no extra argument). "compensated": every call of our kernel (pass 1,
        # the second pass's read, the extra call) gets f3=True.
        if cfg.decay_cumsum not in ("plain", "compensated"):
            raise ValueError(f"GDNConfig.decay_cumsum {cfg.decay_cumsum!r}: 'plain' or 'compensated'")
        if cfg.decay_cumsum == "compensated" and cfg.kernel == "fla":
            raise ValueError("GDNConfig.decay_cumsum 'compensated' (F3) is an option of our kernel 'ced'; kernel 'fla' has "
                             "no compensated sum (set decay_cumsum 'plain' to run fla)")
        self._ced_kw = {"f3": True} if cfg.decay_cumsum == "compensated" else {}
        # C (GDNConfig.edges): "plain", nothing below changes; "fused" / "fused-traced", the one-pass training path calls
        # the fused op (the input preparation in our chunk kernels' prologues, one custom op each way) in place of
        # qkv_prep + ced_delta_rule: "fused" keeps today's two graph breaks where those two calls were, "fused-traced"
        # traces the op into the layer's graph. C2 "once": the same place and graph breaks as "fused", its op forms q, k,
        # v once a chunk in the first kernel that reads it (ced/kernels/gdn_c2.py).
        if cfg.edges not in EDGES_CHOICES:
            raise ValueError(f"GDNConfig.edges {cfg.edges!r}: one of {EDGES_CHOICES}")
        if cfg.edges != "plain" and cfg.kernel != "ced":
            raise ValueError(f"GDNConfig.edges {cfg.edges!r} fuses the input preparation into our kernel 'ced'; this layer's "
                             f"kernel is {cfg.kernel!r}")
        self._fused_edges = cfg.edges != "plain"
        # D (GDNConfig.chunk_backend): "triton", nothing below changes; "flashqla", the fused delta-rule calls of kernel
        # "ced" (pass 1 and the second pass's extra call) run FlashQLA's exact mode (flashqla_delta_rule below),
        # everything around them is the layer's own
        if cfg.chunk_backend not in CHUNK_BACKENDS:
            raise ValueError(f"GDNConfig.chunk_backend {cfg.chunk_backend!r}: one of {CHUNK_BACKENDS}")
        if cfg.chunk_backend != "triton":
            refused = [(cfg.kernel != "ced", f"kernel {cfg.kernel!r} (it replaces the delta-rule core of our kernel 'ced')"),
                       (cfg.edges != "plain", f"edges {cfg.edges!r} (builds C and C2 fuse the input preparation into our "
                                              "Triton chunk kernels)"),
                       (cfg.input_dtype != "bfloat16", f"input_dtype {cfg.input_dtype!r} (FlashQLA takes bf16 or fp16 "
                                                       "q, k, v only: set input_dtype 'bfloat16')"),
                       (cfg.k_head_dim != 128 or cfg.v_head_dim != 128, f"head dimensions {cfg.k_head_dim} / "
                                                                        f"{cfg.v_head_dim} (FlashQLA takes K = V = 128)")]
            bad = [why for cond, why in refused if cond]
            if bad:
                raise ValueError(f"GDNConfig.chunk_backend {cfg.chunk_backend!r} is built for kernel 'ced', edges 'plain', "
                                 f"input_dtype 'bfloat16' and K = V = 128; not for: {'; '.join(bad)}")
        if self._fused_edges:
            gdn_edges.check_process()  # impl "triton", keep "all" (the per-process switches; read at construction too)
        # kernel "ced" also fuses the training-path input preparation (short conv, SiLU, L2 norm) into one
        # kernel each way; CED_GDN_PREP=torch keeps the PyTorch code (A/B runs; read at construction).
        self._fused_prep = cfg.kernel == "ced" and os.environ.get("CED_GDN_PREP", "fused") != "torch"
        # Control (c)'s second-pass read with kernel "ced": pass 1 exposes its chunk states and the read
        # reuses them (one state recursion per layer; ced/kernels/gdn.py ced_delta_rule_states, ced_read).
        # CED_SECOND_PASS_READ=two-call keeps the earlier path, one more full delta-rule call per layer
        # (A/B runs; read at construction). impl "fla-kernels" has no read: two calls.
        self.read_states = (cfg.kernel == "ced" and os.environ.get("CED_SECOND_PASS_READ", "states") != "two-call"
                            and _read_supported())
        if cfg.chunk_backend != "triton":  # FlashQLA exposes no chunk states of ours: the second pass makes its extra call
            self.read_states = False
        # Negative eigenvalues (GDNConfig.negeig; ced/model/gdn_negeig.py): 1, the gates below as they always were; 2, beta =
        # 2 sigmoid(b), refused here on the kernels not checked at beta > 1. A plain int attribute, never in a state dict.
        self._beta_max = gdn_negeig.check(cfg)
        # PyTorch's default Conv1d init: kaiming-uniform with fan_in = kernel size (depthwise).
        bound = 1.0 / math.sqrt(cfg.conv_kernel)
        nn.init.uniform_(self.conv_weight, -bound, bound)
        # Evaluation-only floor on the forget gate of chosen heads (set_alpha_floor; the gate-floor re-score). None
        # (the default, and every model as trained): gates() below is the code it always was, and nothing here is
        # read. A plain attribute: no parameter, no buffer, never in a state dict or a config.
        self.g_floor: torch.Tensor | None = None
        # Evaluation-only CAP on the forget gate of chosen heads (set_alpha_cap). None (the default, and every model as
        # trained): gates() below is the code it always was. A plain attribute like g_floor: no parameter, no buffer,
        # never in a state dict or a config.
        self.g_cap: torch.Tensor | None = None
        # Evaluation-only HEAD ZERO (set_head_zero): the per-value-head multiplier of the delta rule's readout, 1.0 keep and 0.0 zero.
        # None (the default, and every model as trained): the forward is the code it always was. A plain attribute like g_floor: no
        # parameter, no buffer, never in a state dict or a config.
        self.head_keep: torch.Tensor | None = None
        # The GDN-side EncoderConfig.gdn_write_mix; ced/model/gdn_arms.py: a plain attribute, "none" here and in every
        # model without the arm; gdn_arms.build sets it on the layers that take x_w, where gates(x) refuses.
        # arms_attached: the layer holds GDN-side arms (either), so a call without them (a reference stack, a tool that
        # calls the mixer itself) would compute another layer; forward refuses it. False in every model without them.
        self.write_mix = "none"
        self.arms_attached = False
        # GDNConfig.erase (ced/model/gdn_erase.py): None for mode "none" (the default: nothing built, no tensor, no random
        # number); else the erase's tensors (their own generator, never the global one); forward hands over.
        self.erase = gdn_erase.build(self, d_model, cfg)
        # Gated DeltaProduct (GDNConfig.dprod; ced/model/gdn_dprod.py): None for n_h 1 (the default: nothing built, no tensor,
        # no random number), else the extra steps' tensors (their own generator, never the global one); forward hands over.
        from ced.model.gdn_dprod import build as build_dprod

        self.dprod = build_dprod(self, d_model, cfg, dprod_heads)
        # The memory-rule arms were built one by one; a pair of them on one layer is refused unless built together (only kda with
        # erase is): see _refuse_unbuilt_rule_pairs. All off: nothing changes.
        _refuse_unbuilt_rule_pairs(self._beta_max, cfg, self.dprod)

    @torch.no_grad()
    def reset_gate_parameters(self, scheme: str) -> None:
        """Initialise A_log and dt_bias.

        "fla" (flash-linear-attention GatedDeltaNet, Mamba2-style):
            A ~ U(0, 16); dt ~ exp(U(log 1e-3, log 1e-1)), clamped to >= 1e-4;
            dt_bias = softplus^-1(dt) = dt + log(-expm1(-dt)).
        "hf" (Hugging Face Qwen3.5 _init_weights):
            A ~ U(0.01, 16); dt_bias = 1.
        """
        hv = self.cfg.n_v_heads
        if scheme == "fla":
            A = torch.empty(hv).uniform_(0, 16)
            dt = torch.exp(torch.rand(hv) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            dt = dt.clamp(min=1e-4)
            dt_bias = dt + torch.log(-torch.expm1(-dt))
        elif scheme == "hf":
            A = torch.empty(hv).uniform_(0.01, 16)
            dt_bias = torch.ones(hv)
        else:
            raise ValueError(f"unknown GDN init scheme {scheme!r}")
        self.A_log.copy_(torch.log(A))
        if self.kda is None:
            self.dt_bias.copy_(dt_bias)
        else:  # per-channel decay: its own dt_bias (H_v d_k,), the same law a channel (ced/model/gdn_kda.py)
            self.kda.reset_dt_bias(scheme)

    @torch.no_grad()
    def set_alpha_floor(self, heads=(), floor: float = 0.5) -> list[int]:
        """EVALUATION ONLY.

            alpha_t = max(alpha_t, floor),   written on the log decay the delta rule takes:  g_t = max(g_t, ln floor)

        (alpha_t = exp(g_t); the two lines are the same number up to the rounding of exp(ln floor)). The other heads
        of the layer are not touched (their floor is ln 0 = -inf, and max(g, -inf) is g bit for bit). It acts in
        `gates`, so BEFORE the state recurrence and in every path that calls it: the reference recurrence, the
        chunked rule, the fused kernels "ced" and "fla" (they take g as an input and are not edited), cached chunks
        (prefill / extend: a state carried from an earlier chunk was written under the floor of that chunk's
        forward), the zero-write reads of a second pass (they read pass 1's g), the decode engine and the GDN
        diagnostics (ced/eval/gdn_diag.py, gdn_wipe.py).

        heads: value-head indices 0 .. n_v_heads - 1; empty: the floor is taken off (g_floor None: the layer as it
        was). floor: in [0, 1]; 0 clamps nothing (ln 0 = -inf) and 1 stops the head from forgetting at all.
        Not a config field and not saved: a runtime setting of this module, lost when the model is rebuilt. Set it on
        a model that is already on its device. Nothing in the training code calls it.
        Returns the heads set, sorted."""
        heads = sorted({int(h) for h in heads})
        hv = self.cfg.n_v_heads
        if any(not 0 <= h < hv for h in heads):
            raise ValueError(f"value heads {heads}: this layer has heads 0 .. {hv - 1}")
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not 0.0 <= floor <= 1.0:
            raise ValueError(f"alpha floor {floor!r}: a number in [0, 1]")
        if not heads:
            self.g_floor = None
            return heads
        # float64 here and cast to g's dtype at use, so a float64 model is not handed a float32 ln floor
        g_floor = torch.full((hv,), float("-inf"), dtype=torch.float64, device=self.A_log.device)
        g_floor[heads] = math.log(floor) if floor > 0.0 else float("-inf")
        self.g_floor = g_floor
        return heads

    @torch.no_grad()
    def set_alpha_cap(self, caps: dict | None = None) -> dict:
        """EVALUATION ONLY.

            alpha_t = min(alpha_t, caps[h]),   written on the log decay the delta rule takes:  g_t = min(g_t, ln caps[h])

        so the head's per-token horizon h_t = 1 / (1 - alpha_t) is at most 1 / (1 - caps[h]) (cap 1/3: h_t <= 1.5 tokens).
        The other heads of the layer are not touched (their log cap is +inf, and min(g, +inf) is g bit for bit). It acts in `gates` and `gates_from`, after the floor (set_alpha_floor), so BEFORE the state recurrence and
        in every path that calls them: the reference recurrence, the chunked rule, the fused kernels "ced" and "fla"
        (they take g as an input and are not edited), cached chunks, the zero-write reads of a second pass, the decode
        engine and the GDN diagnostics.

        caps: {value head: cap}, heads 0 .. n_v_heads - 1, each cap in (0, 1] (0 would put ln 0 = -inf into every
        kernel's cumulative sums; 1 caps nothing, g <= 0 already); empty or None: the cap is taken off (g_cap None: the
        layer as it was). Not a config field and not saved: a runtime setting of this module, lost when the model is
        rebuilt. Set it on a model that is already on its device. Nothing in the training code calls it.
        Returns the caps set, {head: cap} sorted by head."""
        caps = dict(caps or {})
        hv = self.cfg.n_v_heads
        out: dict[int, float] = {}
        for h, c in sorted(caps.items()):
            if isinstance(h, bool) or not isinstance(h, int) or not 0 <= h < hv:
                raise ValueError(f"value head {h!r}: this layer has heads 0 .. {hv - 1}")
            if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0.0 < c <= 1.0:
                raise ValueError(f"alpha cap {c!r} for head {h}: a number in (0, 1]")
            out[h] = float(c)
        if not out:
            self.g_cap = None
            return out
        # float64 here and cast to g's dtype at use (as the floor): +inf where no cap, ln cap where one is set
        g_cap = torch.full((hv,), float("inf"), dtype=torch.float64, device=self.A_log.device)
        for h, c in out.items():
            g_cap[h] = math.log(c)
        self.g_cap = g_cap
        return out

    @torch.no_grad()
    def set_head_zero(self, heads=()) -> list[int]:
        """EVALUATION ONLY. For the value heads in `heads`, the head's readout of every later forward is multiplied by 0:

            o[:, :, h, :] = 0   before   y = out_proj(GatedRMSNorm(o, z).flatten(2))

        o is the delta rule's output o_t = S_t^T q_t of value head h ((B, L, H_v, d_v)), the head's own readout of its state. It is
        the place where a head's output is a separate tensor: after it the gated norm acts per head (RMSNorm over d_v, times act(z)
        of the head's own z) and out_proj mixes the heads, so zeroing o makes the head's d_v columns of out_proj's input exactly 0
        (RMSNorm(0) * gain * act(z) = 0, no bias in the norm) and its contribution to the layer's output exactly 0, while the other
        heads' slices are untouched. The head's state still evolves (it is read nowhere). The multiplier is a fixed-shape tensor
        over the heads (1.0 keep, 0.0 zero): no index, no mask. It acts in every forward of this layer that forms o (the reference
        loop, the chunked rule, the fused kernels "ced" and "fla", cached chunks, Gated DeltaProduct's, KDA's and the erase's forward,
        the zero-write second pass's reads); the decode engine and the MTP drafter write the step themselves and refuse.

        heads: value-head indices 0 .. n_v_heads - 1; empty: the zero is taken off (head_keep None: the layer as it was, bit for bit).
        Not a config field and not saved: a runtime setting of this module, lost when the model is rebuilt. Set it on a model that is
        already on its device. Nothing in the training code calls it. Returns the heads set, sorted."""
        heads = sorted({int(h) for h in heads})
        hv = self.cfg.n_v_heads
        if any(not 0 <= h < hv for h in heads):
            raise ValueError(f"value heads {heads}: this layer has heads 0 .. {hv - 1}")
        if not heads:
            self.head_keep = None
            return heads
        keep = torch.ones(hv, dtype=torch.float64, device=self.A_log.device)  # float64 here, cast to o's dtype at use (0 and 1 are exact)
        keep[heads] = 0.0
        self.head_keep = keep
        return heads

    def keep_heads(self, o: torch.Tensor) -> torch.Tensor:
        """o with the zeroed heads' readouts set to 0 (set_head_zero): o (..., H_v, d_v) times the multiplier (H_v, 1). Callers check
        `self.head_keep is not None` first, so a layer without the setting never reaches this line."""
        return o * self.head_keep.to(device=o.device, dtype=o.dtype)[:, None]

    def gates(self, x: torch.Tensor, scale: int = 1, beta_power: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (g, beta), each (B, L, H_v). The per-step decay is alpha = exp(g).
        With an evaluation-only floor set (set_alpha_floor; None in every model as trained, and then this is the two
        lines it always was): g = max(g, ln floor) in the floored heads, before anything reads it. With an
        evaluation-only cap set (set_alpha_cap; None in every model as trained): g = min(g, ln cap) in the capped heads,
        after the floor.

        scale (Token Superposition Training's resolution-explicit gates, TSTConfig.gdn_resolution_gates "E"; the forward
        passes SeqContext.gdn_scale): a position stands for `scale` tokens (a bag), and A_log / dt_bias / in_proj_a keep
        describing ONE token: g = scale x (-exp(A_log) softplus(in_proj_a(x) + dt_bias)), the decay of `scale` tokens
        (alpha^scale; S4ND's Delta / s rule, arXiv 2210.06583). With beta_power also the write strength:
        beta = 1 - (1 - sigmoid(b))^scale = -expm1(-scale softplus(b)), b = in_proj_b(x), formed in float32 at least
        (float64 in a float64 model) and returned in b's dtype. Why: a GDN step is S' = alpha (I - beta k k^T) S + beta k v^T with |k| = 1; its
        transition has the eigenvalue alpha (1 - beta) along k and alpha across it, and scale token steps with the same key are alpha^scale (I - (1 -
        (1 - beta)^scale) k k^T): the bag step with these two gates IS that transition. scale 1 (the default, and every caller but a TST bag phase
        with E on): the lines above, unchanged."""
        if self.kda is not None:
            raise ValueError(_PER_CHANNEL_REFUSAL)
        if self.erase is not None:
            raise ValueError(gdn_erase.GATES_REFUSAL)
        if self.write_mix != "none":
            raise ValueError("this GDN layer's decay and beta read x_w (the GDN-side arm A, model.encoder.gdn_write_mix "
                             f"{self.write_mix!r}; ced/model/gdn_arms.py), not x: gates(x) would give another layer's gates")
        g = -torch.exp(self.A_log) * F.softplus(self.in_proj_a(x) + self.dt_bias)
        if scale != 1:  # TST (E): the bag's log decay, s x the token's
            g = g * scale
        if self.g_floor is not None:  # evaluation only: alpha_t = max(alpha_t, floor) in the named heads
            g = torch.maximum(g, self.g_floor.to(device=g.device, dtype=g.dtype))
        if self.g_cap is not None:  # evaluation only: alpha_t = min(alpha_t, cap) in the named heads (after the floor)
            g = torch.minimum(g, self.g_cap.to(device=g.device, dtype=g.dtype))
        b = self.in_proj_b(x)
        if scale != 1 and beta_power:  # TST (E): the bag's write strength, 1 - (1 - beta)^s, in float32 at least
            if self._beta_max != 1:  # negative eigenvalues: that formula is written for beta = sigmoid(b)
                gdn_negeig.refuse_beta_power(self._beta_max)
            bf = b.to(torch.promote_types(b.dtype, torch.float32))
            beta = (-torch.expm1(-scale * F.softplus(bf))).to(b.dtype)
        else:
            beta = torch.sigmoid(b)
            if self._beta_max != 1:  # negative eigenvalues (GDNConfig.negeig): beta = 2 sigmoid(b) in (0, 2)
                beta = gdn_negeig.scale_beta(beta, self._beta_max)
        return g, beta

    def gates_from(self, a_lin: torch.Tensor, b_lin: torch.Tensor, scale: int = 1,
                   beta_power: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """gates() from the two projections' outputs (GDN-side, whose decay and beta read x_w; gates(x) is the code every
        other model runs): a_lin = in_proj_a(x_w), b_lin = in_proj_b(x_w). The same lines as gates(), in the same
        order, with these two in place of in_proj_a(x) and in_proj_b(x), so gates(x, scale, beta_power) ==
        gates_from(in_proj_a(x), in_proj_b(x), scale, beta_power) bit for bit (tests/test_arch_gdn_tst_f3.py).

        scale, beta_power: TST's resolution-explicit gates (E), as gates(): A and E compose (merge of arch-gdn-read,
        2026-10-16). E says how many tokens a position stands for and scales the gates whatever input the projections
        read; A says which input the decay and beta projections read (x_w). So in a TST bag phase with E on, an A layer's
        g is scale x (-exp(A_log) softplus(in_proj_a(x_w) + dt_bias)) and, with beta_power, its beta is
        1 - (1 - sigmoid(in_proj_b(x_w)))^scale; scale 1: the lines as the arm's build wrote them."""
        if self.kda is not None:
            raise ValueError(_PER_CHANNEL_REFUSAL)
        g = -torch.exp(self.A_log) * F.softplus(a_lin + self.dt_bias)
        if scale != 1:  # TST (E): the bag's log decay, s x the token's
            g = g * scale
        if self.g_floor is not None:  # evaluation only: alpha_t = max(alpha_t, floor) in the named heads
            g = torch.maximum(g, self.g_floor.to(device=g.device, dtype=g.dtype))
        if self.g_cap is not None:  # evaluation only: alpha_t = min(alpha_t, cap) in the named heads (after the floor)
            g = torch.minimum(g, self.g_cap.to(device=g.device, dtype=g.dtype))
        if scale != 1 and beta_power:  # TST (E): the bag's write strength, 1 - (1 - beta)^s, in float32 at least
            if self._beta_max != 1:  # negative eigenvalues: that formula is written for beta = sigmoid(b)
                gdn_negeig.refuse_beta_power(self._beta_max)
            bf = b_lin.to(torch.promote_types(b_lin.dtype, torch.float32))
            return g, (-torch.expm1(-scale * F.softplus(bf))).to(b_lin.dtype)
        if self._beta_max != 1:  # negative eigenvalues (GDNConfig.negeig): beta = 2 sigmoid(b) in (0, 2), as gates()
            return g, gdn_negeig.scale_beta(torch.sigmoid(b_lin), self._beta_max)
        return g, torch.sigmoid(b_lin)

    def channel_gates(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(g, beta) of a layer with per-channel decay (GDNConfig.kda.decay "channel"): g (B, L, H_v, d_k), beta (B, L, H_v)
        (ced/model/gdn_kda.py gates). A scalar-decay layer refuses: its gates are gates(x)."""
        if self.kda is None:
            raise ValueError("channel_gates reads a per-channel decay (GDNConfig.kda.decay 'channel'); this layer's decay is one "
                             "per value head: gates(x)")
        return gdn_kda.gates(self, x)

    def short_conv(
        self, u: torch.Tensor, ctx: SeqContext, state: GDNState | None, n_valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Causal depthwise conv over [history ; u], using only inputs from the current document
        (causal_doc_conv with this layer's weight and the state's history).

        Returns the conv output and the new history: the last K-1 inputs and their document ids,
        or, when only the first n_valid positions of each row are real (decoder slots), the last
        K-1 real ones.
        """
        hist, hist_doc = (None, None) if state is None else (state.conv_input, state.conv_doc)
        return causal_doc_conv(u, ctx.doc, self.conv_weight, hist, hist_doc, n_valid)

    def forward(
        self,
        x: torch.Tensor,
        ctx: SeqContext,
        state: GDNState | None = None,
        valid: torch.Tensor | None = None,
        keep: bool = False,
        arms=None,
        x_write: torch.Tensor | None = None,
        ccr=None,
        x_read: torch.Tensor | None = None,
        lerped: bool = False,
    ) -> tuple[torch.Tensor, GDNState] | tuple[torch.Tensor, GDNState, GDNPass1]:
        """valid (B, L) bool, optional: positions marked False are padding. They must be a suffix
        of each row (as decoder slots are), and they leave the state and the conv history untouched.
        The caller must not mark them as document starts.

        keep: also return a GDNPass1 record for a zero-write second pass (see zero_write_read). The
        layer's own outputs and state are the same either way.

        arms, x_write, ccr: the GDN-side arms (ced/model/gdn_arms.py; the encoder layer passes them). arms: the layer's
        GDNArms, None in every model without them, and then nothing below differs from the layer as it was. x_write: m', the
        normed write mix; the k, v, beta and decay inputs become x_w. ccr (arm CCR): the previous GDN layer's record
        (CCRChunks or CCRBoundary; v2: the record of the layer arms.ccr_source names, read with the translated query M q when
        the layer has a translator), read with this layer's q through arms.ccr_gate. A writer layer
        (arms.ccr_write) returns its own record in the new state's `ccr` and, off the fused path, its checkpoint in
        `ckpt`. x_read (the wsl variant "qmix", EncoderConfig.gdn_write_variant): m'_q, the q rows' own normed mix; the q
        rows then read x_q (gdn_arms.write_inputs). None in every other call. lerped (qmix's block-opening layer on the fused
        write mix's second round, ced/kernels/write_mix.py group_read): x_write and x_read are already x_w and x_q (the group
        read formed the lerps); False in every other call."""
        if self.erase is not None:  # the erase direction (GDNConfig.erase), on either decay: its own forward (ced/model/gdn_erase.py)
            return gdn_erase.forward(self, x, ctx, state, valid, keep, arms, x_write, ccr, x_read, lerped)
        if self.kda is not None:  # per-channel decay (GDNConfig.kda): its own forward (ced/model/gdn_kda.py)
            return gdn_kda.forward(self, x, ctx, state, valid, keep, arms, x_write, ccr, x_read, lerped)
        if self.dprod is not None:  # Gated DeltaProduct (GDNConfig.dprod; ced/model/gdn_dprod.py); None in every other layer
            from ced.model.gdn_dprod import dprod_forward

            return getattr(self.dprod, "forward_fn", dprod_forward)(self, x, ctx, state, valid=valid, keep=keep, arms=arms, x_write=x_write,
                                                                    ccr=ccr, x_read=x_read, lerped=lerped)
        B, L, _ = x.shape
        hk, hv, dk, dv = self.cfg.n_k_heads, self.cfg.n_v_heads, self.cfg.k_head_dim, self.cfg.v_head_dim
        assert not (keep and valid is not None), "a second pass runs over real positions only"
        if arms is not None:
            assert not keep and valid is None, "the GDN-side arms run on the encoder's one pass over real positions"
            assert (x_write is not None) == (arms.write_mix != "none"), "arm A's layers take x_write, and only they"
            assert (ccr is not None) == arms.ccr_read, "a CCR reader takes the previous GDN layer's record, and only it"
            from ced.model.gdn_arms import READ_MIX_VARIANTS

            assert (x_read is not None) == (arms.variant in READ_MIX_VARIANTS), "the qmix and kqmix variants' layers take x_read, and only they"
        else:
            assert x_write is None and ccr is None and x_read is None, "x_write / ccr / x_read belong to a layer with the GDN-side arms"
            assert not self.arms_attached, ("this GDN layer has GDN-side arms (ced/model/gdn_arms.py): it is called through "
                                            "its encoder layer's walk (EncoderLayer.gdn_arms_mixer), not on its own")

        n_valid = None if valid is None else valid.long().sum(1)
        K = self.cfg.conv_kernel
        fo = {}  # model.fused_projections "qkv_z" / "ab" (ced/model/fused_proj.py): {part: its view of one GEMM's output}
        if getattr(self, "fused_proj", None) and self.training and arms is None:
            from ced.model import fused_proj

            fo = fused_proj.outputs(self, x)
        if x_write is None:
            # pre-conv [q | k | v] inputs (also what keep records)
            u_pre = fo["in_proj_qkv"] if "in_proj_qkv" in fo else self.in_proj_qkv(x)
        else:  # the write mix: the q rows read x, the k and v rows x_w (and so do the decay and beta, below)
            from ced.model.gdn_arms import write_inputs

            if x_read is None:
                u_pre, a_lin, b_lin = write_inputs(self, arms, x, x_write)
            elif lerped:  # qmix on the fused write mix's second round: x_w and x_q from the group read
                u_pre, a_lin, b_lin = write_inputs(self, arms, x, x_write, x_read, lerped=True)
            else:  # the wsl variants "qmix" (the q rows read x_q) and "kqmix" (the q and k rows read x_kq)
                u_pre, a_lin, b_lin = write_inputs(self, arms, x, x_write, x_read)
        use_fused = (self.cfg.kernel in ("fla", "ced") and state is None and ctx.cu_seqlens is not None
                     and fla_supported(x))
        # input_dtype "bfloat16": under bf16 autocast (the in-projection's output is bf16), q, k, v and
        # beta reach the fused kernel in bf16 instead of float32 (g stays float32). Elsewhere (float32 /
        # float64 runs, the reference kernels) nothing changes. See GDNConfig.input_dtype.
        kernel_dtype = (torch.bfloat16 if use_fused and self.cfg.input_dtype == "bfloat16"
                        and u_pre.dtype == torch.bfloat16 else None)
        # C (edges "fused"): the one-pass training path only (no keep, no GDN-side arms; use_fused already means no
        # state, varlen boundaries and CUDA); q, k, v are formed inside the kernels, so none is made here
        edges = (self._fused_edges and use_fused and not keep and arms is None and valid is None and L >= K - 1)
        if edges:
            if self.cfg.edges in ("fused", "once"):
                # today's first graph break (the opaque qkv_prep call stood here): the compiled graphs, and so the order in
                # which their backwards add x's four input-projection gradients, stay today's (no-op when not compiling)
                torch._dynamo.graph_break()
            q = k = v = None
            conv_input, conv_doc = u_pre[:, L - (K - 1):], ctx.doc[:, L - (K - 1):]
        elif self._fused_prep and valid is None and state is None and ctx.cu_seqlens is not None and fla_supported(x) and L >= K - 1:
            # the same function, fused into one kernel each way (ced/kernels/gdn.py: qkv_prep); it
            # computes in float32 and stores q, k, v in kernel_dtype (None: promote(u_pre, conv_weight))
            q, k, v = qkv_prep(u_pre, ctx.doc, self.conv_weight, hk, hv, dk, out_dtype=kernel_dtype)
            conv_input, conv_doc = u_pre[:, L - (K - 1):], ctx.doc[:, L - (K - 1):]
        else:
            u, conv_input, conv_doc = self.short_conv(u_pre, ctx, state, n_valid)
            u = F.silu(u)
            q, k, v = torch.split(u, [hk * dk, hk * dk, hv * dv], dim=-1)
            q = l2_normalize(q.view(B, L, hk, dk)) * dk**-0.5
            k = l2_normalize(k.view(B, L, hk, dk))
            v = v.view(B, L, hv, dv)
            if kernel_dtype is not None:
                q, k, v = q.to(kernel_dtype), k.to(kernel_dtype), v.to(kernel_dtype)
        # TST's resolution-explicit gates (SeqContext.gdn_scale, E): set only in a TST bag phase with E on; every other
        # forward calls gates(x) as it always did. Every kernel below takes g and beta as inputs, so the reference, the
        # chunked and the fused (Triton) paths, forward and backward, all see the same scaled gates.
        if "in_proj_a" in fo:  # "ab": in_proj_a and in_proj_b as one GEMM; gates_from is gates' lines on their outputs
            g, beta = (self.gates_from(fo["in_proj_a"], fo["in_proj_b"]) if ctx.gdn_scale == 1
                       else self.gates_from(fo["in_proj_a"], fo["in_proj_b"], ctx.gdn_scale, ctx.gdn_beta_power))
        elif x_write is None:
            g, beta = self.gates(x) if ctx.gdn_scale == 1 else self.gates(x, ctx.gdn_scale, ctx.gdn_beta_power)
        else:  # the write mix: the decay and beta read x_w (its projections' outputs), with E's scale where both apply
            g, beta = (self.gates_from(a_lin, b_lin) if ctx.gdn_scale == 1
                       else self.gates_from(a_lin, b_lin, ctx.gdn_scale, ctx.gdn_beta_power))
        cross = None
        if ccr is not None:  # arm CCR: its source's checkpoint read with this layer's q (key heads here; v2: M q)
            from ced.model.gdn_arms import ccr_cross

            cross = ccr_cross(self, arms, ccr, q, ctx)
        writer = arms is not None and arms.ccr_write
        ccr_out = ckpt = None
        if valid is not None:  # padding: no decay (g = 0) and no write (beta = 0)
            g = g * valid[..., None].to(g.dtype)
            beta = beta * valid[..., None].to(beta.dtype)
        if kernel_dtype is not None:
            beta = beta.to(kernel_dtype)  # already bf16 under autocast (sigmoid of a bf16 projection)

        # What a second pass reads (keep): the state after every position (recurrence), or the
        # inputs of one more delta-rule call (chunked kernels).
        read = {}
        if use_fused:
            # fla (and ours, which computes fla's function) handles grouped value heads itself (value
            # head j reads key head j // (hv / hk)).
            # v is a view into the conv output u on the PyTorch prep path; the kernel would copy it to
            # a contiguous tensor on every call (and keep that copy for its backward). One copy here
            # serves both this call and, with keep, the second pass's call over the same v (no second
            # copy). The fused prep returns a contiguous v already (no-op).
            if not edges:
                v = v.contiguous()
            S = None  # training forward: the final state is not needed (no cache is built)
            if edges and ctx.gdn_chunks is not None:  # refused at the config too (cuda_graphs.check_config)
                raise ValueError("model.encoder.gdn.edges with TrainConfig.cuda_graphs: not built together (the fused edges op "
                                 "forms its own chunk metadata; the graphs' fixed-capacity metadata is for the plain call)")
            if edges:  # C: the input preparation and the delta rule in one custom op (no keep: nothing is recorded)
                # "fused": opaque, as ced_delta_rule here today (today's second graph break); "fused-traced": in the graph
                # "once" (C2): opaque in the same place, its kernels form q, k, v once a chunk (ced/kernels/gdn_c2.py)
                fused_op = {"fused": ced_edges_delta_rule_opaque, "fused-traced": ced_edges_delta_rule,
                            "once": ced_once_delta_rule_opaque}[self.cfg.edges]
                o = fused_op(u_pre, ctx.doc, self.conv_weight, g, beta, ctx.cu_seqlens, hk, hv, dk, out_dtype=kernel_dtype,
                             **self._ced_kw)
            elif ctx.gdn_chunks is not None:
                # CUDA graphs' fixed-capacity metadata (SeqContext.gdn_chunks): the plain one-pass call of kernel "ced", our
                # Triton core, or FlashQLA's captured form (chunk_backend "flashqla")
                assert self.cfg.kernel == "ced" and not (keep and self.read_states) and not writer, (
                    "the fixed-capacity GDN metadata (TrainConfig.cuda_graphs) is built for GDN kernel 'ced' on the plain "
                    "one-pass call, not for fla, the second pass's state read or CCR")
                fixed, name = (flashqla_delta_rule, "flashqla") if self.cfg.chunk_backend == "flashqla" else (ced_delta_rule,
                                                                                                             self.cfg.kernel)
                if keep and name == "flashqla":  # refused at the config too (cuda_graphs.check_config: model.loop.pass2_gate)
                    raise ValueError("GDN chunk_backend 'flashqla' with TrainConfig.cuda_graphs: the second pass's extra call "
                                     "is not built on the fixed-capacity metadata (one pass only)")
                o = fixed(q, k, v, g, beta, ctx.cu_seqlens, ctx.cu_seqlens_cpu, chunks=ctx.gdn_chunks, **self._ced_kw)
                read = dict(k=k, v=v, g=g, beta=beta, fused_kernel=name, cu_seqlens=ctx.cu_seqlens,
                            cu_seqlens_cpu=ctx.cu_seqlens_cpu)
            elif keep and self.read_states:
                # pass 1 hands its chunk states to the second pass's read (zero_write_read)
                o, h_chunks, v_new = ced_delta_rule_states(q, k, v, g, beta, ctx.cu_seqlens, ctx.cu_seqlens_cpu, **self._ced_kw)
                read = dict(k=k, g=g, h=h_chunks, v_new=v_new, fused_kernel="ced", cu_seqlens=ctx.cu_seqlens,
                            cu_seqlens_cpu=ctx.cu_seqlens_cpu)
            elif writer:
                # arm CCR's writer: the kernel's chunk-start states (kept for its own backward anyway) go to the reader;
                # their gradient comes back through the kernel's EXT input (the second-pass read's)
                if self.cfg.chunk_backend != "triton":
                    raise ValueError("the GDN-side arm CCR's writer reads the chunk states of our Triton kernel: "
                                     "GDNConfig.chunk_backend 'flashqla' has none (set chunk_backend 'triton')")
                assert self.cfg.kernel == "ced" and _read_supported(), \
                    "CCR's fused path reads the chunk states of our kernel: GDN kernel 'ced' with impl 'triton'"
                from ced.model.gdn_arms import CCRChunks, kernel_chunk

                assert arms.spacing == kernel_chunk(), (f"model.encoder.gdn_ccr_spacing {arms.spacing}: the fused path's "
                                                        f"checkpoints are the kernel's chunk starts, every {kernel_chunk()} tokens")

                o, h_chunks, _ = ced_delta_rule_states(q, k, v, g, beta, ctx.cu_seqlens, ctx.cu_seqlens_cpu, **self._ced_kw)
                ccr_out = CCRChunks(h=h_chunks, spacing=arms.spacing, source=arms.layer)
            else:
                fused = fla_delta_rule if self.cfg.kernel == "fla" else ced_delta_rule
                name = self.cfg.kernel
                if self.cfg.chunk_backend == "flashqla":  # D: FlashQLA for the core; the second pass's extra call too
                    fused, name = flashqla_delta_rule, "flashqla"
                o = fused(q, k, v, g, beta, ctx.cu_seqlens, ctx.cu_seqlens_cpu, **self._ced_kw)  # {} for fla (refused with F3)
                read = dict(k=k, v=v, g=g, beta=beta, fused_kernel=name, cu_seqlens=ctx.cu_seqlens,
                            cu_seqlens_cpu=ctx.cu_seqlens_cpu)
        else:
            k_heads = k
            q = q.repeat_interleave(hv // hk, dim=2)
            k = k.repeat_interleave(hv // hk, dim=2)
            S0 = x.new_zeros(B, hv, dk, dv) if state is None else state.S
            if writer:
                # arm CCR's writer off the fused path: the state after every position, then each position's checkpoint
                # (gdn_arms.boundary_states) and the one the next token after this call reads (the cache's ckpt)
                from ced.model.gdn_arms import CCRBoundary, boundary_states

                assert state is None or state.ckpt is not None, "this cache was written without CCR's checkpoints"
                if self.cfg.kernel in ("chunked", "fla", "ced") and L > 1:
                    o, S, after = chunked_delta_rule(q, k, v, g, beta, ctx.is_doc_start, S0, self.cfg.chunk_size,
                                                     all_states=True)
                else:
                    o, S, after = delta_rule_recurrence(q, k, v, g, beta, ctx.is_doc_start, S0, all_states=True)
                per_pos, ckpt = boundary_states(after, ctx.pos, None if state is None else state.ckpt, arms.spacing)
                ccr_out = CCRBoundary(states=per_pos, source=arms.layer)
            elif self.cfg.kernel in ("chunked", "fla", "ced") and L > 1:
                o, S = chunked_delta_rule(q, k, v, g, beta, ctx.is_doc_start, S0, self.cfg.chunk_size)
                read = dict(k=k_heads, v=v, g=g, beta=beta, is_doc_start=ctx.is_doc_start, S0=S0)
            elif keep:
                o, S, states = delta_rule_recurrence(q, k, v, g, beta, ctx.is_doc_start, S0, all_states=True)
                read = dict(states=states)
            else:
                o, S = delta_rule_recurrence(q, k, v, g, beta, ctx.is_doc_start, S0)

        if cross is not None:  # arm CCR: o_t += gate (*) ckpt^T q_t, per value head, before the gated output norm
            # in o's own dtype (bf16 from the fused kernel under the runs' autocast), as the write mix's projections keep
            # theirs: the float32 gate would otherwise promote the sum, and the norm would return and save float32
            # (tests/test_arch_gdn.py test_at_the_start_under_bf16_autocast_ccr_is_r1_entry_bit_for_bit)
            o = (o + arms.ccr_gate[:, None] * cross).to(o.dtype)
        if self.head_keep is not None:  # evaluation only: the zeroed heads' readouts are 0 (set_head_zero); None in every model as trained
            o = self.keep_heads(o)
        z = (fo["in_proj_z"] if "in_proj_z" in fo else self.in_proj_z(x)).view(B, L, hv, dv)
        if getattr(self, "mx_out_norm", False) and self.training:  # model.fp8.chains gdn_out_norm (ced/model/mx_wire.py)
            from ced.model import mx_wire

            y = mx_wire.gated_norm_proj(self, o, z)
        else:
            y = self.out_proj(self.norm(o, z).flatten(2))
        if arms is not None:
            return y, GDNState(conv_input=conv_input, conv_doc=conv_doc, S=S, ckpt=ckpt, ccr=ccr_out)
        new_state = GDNState(conv_input=conv_input, conv_doc=conv_doc, S=S)
        if not keep:
            return y, new_state
        # The q channels of [conv history ; this chunk], before the conv: the second pass's q conv
        # reads pass 1's inputs at t-K+1..t-1 from here.
        nq, K = hk * dk, self.cfg.conv_kernel
        if state is None:
            hist = u_pre.new_zeros(B, K - 1, nq)
            hist_doc = torch.full((B, K - 1), -1, dtype=torch.long, device=x.device)
        else:
            hist, hist_doc = state.conv_input[..., :nq], state.conv_doc
        record = GDNPass1(
            q_inputs=torch.cat([hist, u_pre[..., :nq]], dim=1), q_docs=torch.cat([hist_doc, ctx.doc], dim=1), **read
        )
        return y, new_state, record

    def zero_write_read(
        self, x: torch.Tensor, slot_index: torch.Tensor, slot_doc: torch.Tensor, slot_valid: torch.Tensor,
        rec: GDNPass1, slot_inverse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The zero-write second pass of this layer at C slots of a chunk (control (c), section 2). For
        a slot at position t:

            q_in2_t = in_proj_qkv(x_t) restricted to the q channels      (pass 2's own input at t)
            q2_t    = L2norm(SiLU(conv_q(pass-1 q inputs at t-K+1..t-1 of t's document ; q_in2_t))) * d_k^-0.5
            o2_t    = S1_t^T q2_t     S1_t: pass 1's state after position t (t's own write included)
            y2_t    = out_proj(GatedRMSNorm(o2_t, z2_t = in_proj_z(x_t)))

        Nothing is written: no state update and no conv history entry; pass 2's k, v, beta and g
        are never formed. The read uses the record's states (reference); or, after pass 1 on kernel
        "ced" with its chunk states exposed (the default for "ced"), the read over pass 1's chunk
        states h and v_new with q = q2 scattered to the slot positions (ced_read: the output kernel
        with q2; its backward feeds pass 1's one state-gradient recursion); or, after a chunked pass
        1 (or fla), one more delta-rule call over the chunk with pass 1's k, v, g, beta and q = q2
        scattered to the slot positions (zero elsewhere): that call's output at t is S1_t^T q_t,
        which at a slot is exactly o2_t. With pass 1 on a fused varlen kernel (fla or ced), the extra
        call runs the same kernel (same varlen boundaries).

        x: (B, C, d) pass 2's normed layer input at the slots; slot_index: (B, C) chunk-relative
        position of each slot (L for an empty slot); slot_doc: (B, C), -1 for empty slots;
        slot_valid: (B, C) bool. Returns (B, C, d). Empty slots get a finite output that no valid
        slot reads.

        slot_inverse: optional (B, L), the slot of each position (C where none; capacity.slot_inverse).
        After a chunked or fused pass 1 (not the reference), the gathers and the scatter at the slots
        then use it so that their backward is another gather (capacity.gather_rows): the same values
        and gradients without the deterministic scatter-add's sort.
        """
        B, C, _ = x.shape
        hk, hv, dk, dv = self.cfg.n_k_heads, self.cfg.n_v_heads, self.cfg.k_head_dim, self.cfg.v_head_dim
        K, nq = self.cfg.conv_kernel, hk * dk

        inverse = slot_inverse if rec.states is None else None  # the reference keeps plain gathers
        q_now = F.linear(x, self.in_proj_qkv.weight[:nq])
        w = self.conv_weight[:nq]
        out = q_now * w[:, K - 1]  # tap 0: pass 2's own input
        N = rec.q_inputs.shape[1]  # K-1 + L: [history ; chunk]
        for j in range(1, K):  # tap j looks j steps back, into pass 1's inputs
            take = slot_index + (K - 1 - j)  # position t - j in [history ; chunk]
            if inverse is None:
                past = torch.gather(rec.q_inputs, 1, take[..., None].expand(B, C, nq))
            else:
                # empty slots read the zero fill row (their same_doc is 0 either way); row p of
                # [history ; chunk] is read by the slot at position p - (K-1-j), if any
                s = K - 1 - j
                pad = lambda n: inverse.new_full((B, n), C)  # noqa: E731
                bwd = torch.cat([pad(s), inverse, pad(K - 1 - s)], dim=1)
                past = gather_rows(rec.q_inputs, torch.where(slot_valid, take, N), bwd)
            same_doc = ((torch.gather(rec.q_docs, 1, take) == slot_doc) & slot_valid).to(x.dtype)
            out = out + past * same_doc[..., None] * w[:, K - 1 - j]
        q2 = l2_normalize(F.silu(out).view(B, C, hk, dk)) * dk**-0.5

        if rec.states is not None:
            S_t = gather_slots(rec.states, slot_index)  # (B, C, H_v, d_k, d_v); zeros at empty slots
            o2 = torch.einsum("bchkv,bchk->bchv", S_t, q2.repeat_interleave(hv // hk, dim=2))
        else:
            L = rec.k.shape[1]
            q2 = q2.to(rec.k.dtype)  # pass 1's q and k share a dtype; so must q2 (mixed precision)
            if inverse is None:
                q_scattered = scatter_slots(q2.new_zeros(B, L, hk, dk), slot_index, q2)
            else:
                q_scattered = scatter_slots_inv(q2, slot_index, inverse)
            if rec.h is not None:
                o = ced_read(q_scattered, rec.k, rec.g, rec.h, rec.v_new, rec.cu_seqlens, rec.cu_seqlens_cpu, **self._ced_kw)
            elif rec.fused_kernel is not None:
                fused = {"fla": fla_delta_rule, "ced": ced_delta_rule, "flashqla": flashqla_delta_rule}[rec.fused_kernel]
                o = fused(q_scattered, rec.k, rec.v, rec.g, rec.beta, rec.cu_seqlens, rec.cu_seqlens_cpu,
                          **(self._ced_kw if rec.fused_kernel in ("ced", "flashqla") else {}))
            else:
                o, _ = chunked_delta_rule(
                    q_scattered.repeat_interleave(hv // hk, dim=2), rec.k.repeat_interleave(hv // hk, dim=2),
                    rec.v, rec.g, rec.beta, rec.is_doc_start, rec.S0, self.cfg.chunk_size,
                )
            o2 = gather_slots(o, slot_index) if inverse is None else gather_slots_inv(o, slot_index, inverse)
        if self.head_keep is not None:  # evaluation only: the zeroed heads' readouts are 0 here too (set_head_zero)
            o2 = self.keep_heads(o2)
        z2 = self.in_proj_z(x).view(B, C, hv, dv)
        return self.out_proj(self.norm(o2, z2).flatten(2))

    def zero_write_read_all(self, x: torch.Tensor, doc: torch.Tensor, rec: GDNPass1) -> torch.Tensor:
        """zero_write_read at EVERY position of the chunk, in row order: the looped control's second pass in
        training and scoring (ced/model/control.py, "Two passes"). The slots are the positions themselves, so
        nothing is gathered or scattered (fixed shapes, no index ops): pass 1's q inputs at t-j are a slice of the
        record, the query q2 is already laid out as the kernels want it, and their output is o2. The same function as
        zero_write_read(x, slot_index = 0 .. L-1, doc, all valid, rec), read for read:

            reference (rec.states):     o2_t = S1_t^T q2_t over the kept states
            kernel "ced", chunk states: ced_read(q2, ...) over pass 1's h and v_new
            fused / chunked:            one more delta-rule call over pass 1's k, v, g, beta with q = q2

        Nothing is written: no state update, no conv history entry, no k, v, beta or g of pass 2.
        x: (B, L, d) pass 2's normed layer input; doc: (B, L) the chunk's document ids. Returns (B, L, d)."""
        B, L, _ = x.shape
        hk, hv, dk, dv = self.cfg.n_k_heads, self.cfg.n_v_heads, self.cfg.k_head_dim, self.cfg.v_head_dim
        K, nq = self.cfg.conv_kernel, hk * dk

        q_now = F.linear(x, self.in_proj_qkv.weight[:nq])
        w = self.conv_weight[:nq]
        out = q_now * w[:, K - 1]  # tap 0: pass 2's own input
        for j in range(1, K):  # tap j looks j steps back, into pass 1's inputs
            s = K - 1 - j  # row t + s of [history ; chunk] is position t - j
            same_doc = (rec.q_docs[:, s : s + L] == doc).to(x.dtype)
            out = out + rec.q_inputs[:, s : s + L] * same_doc[..., None] * w[:, K - 1 - j]
        q2 = l2_normalize(F.silu(out).view(B, L, hk, dk)) * dk**-0.5

        if rec.states is not None:
            o2 = torch.einsum("bchkv,bchk->bchv", rec.states, q2.repeat_interleave(hv // hk, dim=2))
        else:
            q2 = q2.to(rec.k.dtype)  # pass 1's q and k share a dtype; so must q2 (mixed precision)
            if rec.h is not None:
                o2 = ced_read(q2, rec.k, rec.g, rec.h, rec.v_new, rec.cu_seqlens, rec.cu_seqlens_cpu, **self._ced_kw)
            elif rec.fused_kernel is not None:
                fused = {"fla": fla_delta_rule, "ced": ced_delta_rule, "flashqla": flashqla_delta_rule}[rec.fused_kernel]
                o2 = fused(q2, rec.k, rec.v, rec.g, rec.beta, rec.cu_seqlens, rec.cu_seqlens_cpu,
                           **(self._ced_kw if rec.fused_kernel in ("ced", "flashqla") else {}))
            else:
                o2, _ = chunked_delta_rule(
                    q2.repeat_interleave(hv // hk, dim=2), rec.k.repeat_interleave(hv // hk, dim=2),
                    rec.v, rec.g, rec.beta, rec.is_doc_start, rec.S0, self.cfg.chunk_size,
                )
        if self.head_keep is not None:  # evaluation only: the zeroed heads' readouts are 0 here too (set_head_zero)
            o2 = self.keep_heads(o2)
        z2 = self.in_proj_z(x).view(B, L, hv, dv)
        return self.out_proj(self.norm(o2, z2).flatten(2))


def fla_supported(x: torch.Tensor) -> bool:
    """The fused Triton kernels (fla's, and ours for kernel "ced": the delta rule and the input
    preparation) run on CUDA only. (A function, so CPU tests can route the fused branches through
    emulations of fla_delta_rule / ced_delta_rule / qkv_prep.)"""
    return x.is_cuda


@torch.compiler.disable  # never traced: an opaque fused op (its own forward and backward kernels)
def qkv_prep(u: torch.Tensor, doc: torch.Tensor, conv_weight: torch.Tensor, hk: int, hv: int, d: int,
             out_dtype: torch.dtype | None = None):
    """short_conv + SiLU + split + L2 norm (+ q scale) of the training path, fused (ced/kernels/gdn.py);
    q, k, v in out_dtype (None: promote(u, conv_weight), the PyTorch code's dtype)."""
    from ced.kernels.gdn import qkv_prep as fused

    return fused(u, doc, conv_weight, hk, hv, d, out_dtype=out_dtype)


@torch.compiler.disable  # never traced: its varlen inputs change length every batch (see ced/train/varlen.py)
def ced_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None = None,
    f3: bool = False,
    chunks: tuple | None = None,
) -> torch.Tensor:
    """fla_delta_rule's function with our kernel (ced/kernels/gdn.py), same arguments and result.
    g goes to the kernel in float32 (float64 with float64 inputs: our kernel computes in float64 then,
    so the log decay is not rounded to float32 first). f3: F3, the compensated cumulative sum of the decays
    (GDNConfig.decay_cumsum "compensated"); off, the kernel is called exactly as before (no f3 argument).
    chunks: CUDA graphs' fixed-capacity chunk index (SeqContext.gdn_chunks; ced/kernels/gdn.py fixed_chunk_index);
    None, the kernel is called exactly as before (no chunks argument)."""
    from ced.kernels.gdn import ced_delta_rule as kernel

    B, L = q.shape[:2]
    flat = lambda t: t.reshape(1, B * L, *t.shape[2:])  # noqa: E731
    g_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    o = kernel(flat(q), flat(k), flat(v), flat(g).to(g_dtype), flat(beta).to(q.dtype), cu_seqlens.long(),
               cu_seqlens_cpu.long() if cu_seqlens_cpu is not None else None, **({"f3": True} if f3 else {}),
               **({"chunks": chunks} if chunks is not None else {}))
    return o.reshape(B, L, *o.shape[2:])


@torch.compiler.disable  # never traced: its varlen inputs change length every batch (see ced/train/varlen.py)
def flashqla_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None = None,
    f3: bool = False,
    chunks: tuple | None = None,
) -> torch.Tensor:
    """ced_delta_rule's function with FlashQLA's exact mode (GDNConfig.chunk_backend "flashqla", D;
    ced/kernels/gdn_flashqla.py), the same arguments and result: q, k (B, L, H_k, d_k), v (B, L, H_v, d_v) in bf16 / fp16;
    g (log decay) and beta (B, L, H_v). g goes to the kernel in float32, beta in q's dtype. f3: F3 (decay_cumsum
    "compensated"), the pair formed by our Triton scan and FlashQLA's patched kernels. chunks: CUDA graphs' fixed-capacity
    metadata (SeqContext.gdn_chunks), FlashQLA's captured form; None, the kernel is called exactly as before (no chunks
    argument)."""
    from ced.kernels.gdn_flashqla import flashqla_delta_rule as kernel

    B, L = q.shape[:2]
    flat = lambda t: t.reshape(1, B * L, *t.shape[2:])  # noqa: E731
    o = kernel(flat(q), flat(k), flat(v), flat(g).float(), flat(beta).to(q.dtype), cu_seqlens,
               cu_seqlens_cpu.long() if cu_seqlens_cpu is not None else None, f3=bool(f3),
               **({"chunks": chunks} if chunks is not None else {}))
    return o.reshape(B, L, *o.shape[2:])


def ced_edges_delta_rule(u, doc, conv_weight, g, beta, cu_seqlens, hk: int, hv: int, d: int, out_dtype=None,
                         f3: bool = False) -> torch.Tensor:
    """C (GDNConfig.edges "fused-traced"): qkv_prep(u, ...) followed by ced_delta_rule with its q, k, v, as one call of
    our kernels whose q, k, v are never written (ced/kernels/gdn_edges.py fused_delta_rule; two torch.library custom
    ops, traceable: not torch.compiler.disable, so the layer compiles as one graph). u (B, L, C) the in_proj_qkv output,
    doc (B, L), g and beta (B, L, H_v) as the plain path passes them; cu_seqlens on the device (its host copy is not
    read). Returns o (B, L, H_v, d_v) in the kernel's input dtype, the values of the plain path. f3: as ced_delta_rule's."""
    return gdn_edges.fused_delta_rule(u, doc, conv_weight, g, beta, cu_seqlens, hk, hv, d, out_dtype=out_dtype, f3=f3)


@torch.compiler.disable  # "fused": opaque where ced_delta_rule is today, so the compiled graphs are today's
def ced_edges_delta_rule_opaque(u, doc, conv_weight, g, beta, cu_seqlens, hk: int, hv: int, d: int, out_dtype=None,
                                f3: bool = False) -> torch.Tensor:
    """ced_edges_delta_rule, never traced (GDNConfig.edges "fused")."""
    return gdn_edges.fused_delta_rule(u, doc, conv_weight, g, beta, cu_seqlens, hk, hv, d, out_dtype=out_dtype, f3=f3)


@torch.compiler.disable  # "once": opaque where ced_delta_rule is today, as "fused"
def ced_once_delta_rule_opaque(u, doc, conv_weight, g, beta, cu_seqlens, hk: int, hv: int, d: int, out_dtype=None,
                               f3: bool = False) -> torch.Tensor:
    """C2 (GDNConfig.edges "once"): qkv_prep(u, ...) followed by ced_delta_rule with its q, k, v, with q, k and v formed
    once a chunk in the first kernel that reads it (ced/kernels/gdn_c2.py once_delta_rule; two torch.library custom
    ops). The arguments and the result are ced_edges_delta_rule's; never traced."""
    return gdn_c2.once_delta_rule(u, doc, conv_weight, g, beta, cu_seqlens, hk, hv, d, out_dtype=out_dtype, f3=f3)


def _read_supported() -> bool:
    from ced.kernels.gdn import read_supported

    return read_supported()


@torch.compiler.disable  # never traced: its varlen inputs change length every batch (see ced/train/varlen.py)
def ced_delta_rule_states(q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu=None, f3: bool = False):
    """ced_delta_rule that also returns pass 1's chunk states for control (c)'s second-pass read (ced_read):
    (o (B, L, H_v, d_v), h (1, NT, H_v, d_k, d_v), v_new (1, B*L, H_v, d_v)); h and v_new stay in the
    kernel's flattened layout (only ced_read reads them). ced/kernels/gdn.py ced_delta_rule_states.
    f3: as ced_delta_rule's (off: the kernel is called exactly as before)."""
    from ced.kernels.gdn import ced_delta_rule_states as kernel

    B, L = q.shape[:2]
    flat = lambda t: t.reshape(1, B * L, *t.shape[2:])  # noqa: E731
    g_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    o, h, v_new = kernel(flat(q), flat(k), flat(v), flat(g).to(g_dtype), flat(beta).to(q.dtype), cu_seqlens.long(),
                         cu_seqlens_cpu.long() if cu_seqlens_cpu is not None else None, **({"f3": True} if f3 else {}))
    # h's chunk count NT (the chunks of 64 over every varlen segment) changes from batch to batch, and h
    # crosses compiled frames (the GDN forward returns it in GDNPass1; pass 2's compiled zero_write
    # passes it to ced_read). Under the trainer's static shapes (configure_compile) every new NT would
    # otherwise be a recompile of both frames, and the 33rd distinct NT in a process a hard failure
    # (fail_on_recompile_limit_hit). Marked dynamic in that length, as varlen.to_device marks cu_seqlens.
    torch._dynamo.maybe_mark_dynamic(h, 1)
    return o.reshape(B, L, *o.shape[2:]), h, v_new


@torch.compiler.disable  # never traced: its varlen inputs change length every batch (see ced/train/varlen.py)
def ced_read(q2, k, g, h, v_new, cu_seqlens, cu_seqlens_cpu=None, f3: bool = False):
    """Control (c)'s second-pass read over pass 1's chunk states (ced_delta_rule_states with the same k, g
    and boundaries): o (B, L, H_v, d_v), o_t = S1_t^T q2_t, S1_t pass 1's state after t. q2 (B, L, H_k, d_k)
    in k's dtype, zero where nothing reads. ced/kernels/gdn.py ced_read. f3: pass 1's (off: called as before)."""
    from ced.kernels.gdn import ced_read as kernel

    B, L = q2.shape[:2]
    flat = lambda t: t.reshape(1, B * L, *t.shape[2:])  # noqa: E731
    g_dtype = torch.float64 if q2.dtype == torch.float64 else torch.float32
    o = kernel(flat(q2), flat(k), flat(g).to(g_dtype), h, v_new, cu_seqlens.long(),
               cu_seqlens_cpu.long() if cu_seqlens_cpu is not None else None, **({"f3": True} if f3 else {}))
    return o.reshape(B, L, *o.shape[2:])


@torch.compiler.disable  # never traced: its varlen inputs change length every batch (see ced/train/varlen.py)
def fla_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> torch.Tensor:
    """The gated delta rule from an empty state with fla's fused chunked kernel.

    q, k: (B, L, H_k, d_k), already L2-normalised, q scaled by d_k^-0.5 (so scale=1 here);
    v: (B, L, H_v, d_v); g (log decay), beta: (B, L, H_v). Rows are flattened into one sequence and
    the state restarts at every boundary in cu_seqlens (row starts and document starts), exactly
    the reference's reset at is_doc_start. Returns outputs (B, L, H_v, d_v).
    """
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    B, L = q.shape[:2]
    flat = lambda t: t.reshape(1, B * L, *t.shape[2:])  # noqa: E731
    o, _ = chunk_gated_delta_rule(
        flat(q), flat(k), flat(v), flat(g).float(), flat(beta).to(q.dtype),
        scale=1.0, initial_state=None, output_final_state=False,
        cu_seqlens=cu_seqlens.long(), cu_seqlens_cpu=cu_seqlens_cpu.long() if cu_seqlens_cpu is not None else None,
    )
    return o.reshape(B, L, *o.shape[2:])


def delta_rule_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    is_doc_start: torch.Tensor,
    S: torch.Tensor,
    all_states: bool = False,
):
    """The gated delta rule, one token at a time.

    q, k: (B, L, H, d_k); v: (B, L, H, d_v); g, beta: (B, L, H); is_doc_start: (B, L) bool;
    S: (B, H, d_k, d_v) state before the first token. Returns outputs (B, L, H, d_v) and the final
    state. Per token: reset at a document start, decay by alpha = exp(g), write
    k (x) beta (v - S^T k), read S^T q.

    all_states: also return the state after every token, (B, L, H, d_k, d_v) (the reference read
    of the zero-write second pass).
    """
    outputs, states = [], []
    # Unbind the time axis once: indexing x[:, t] in the loop makes the backward O(L^2), because
    # each select's backward zero-fills a full-size gradient. Results are bitwise identical.
    steps = zip(q.unbind(1), k.unbind(1), v.unbind(1), g.unbind(1), beta.unbind(1), is_doc_start.unbind(1))
    for q_t, k_t, v_t, g_t, beta_t, start_t in steps:
        keep = (~start_t).to(S.dtype)[:, None, None, None]
        S = S * keep  # a new document starts from an empty state
        S = S * torch.exp(g_t)[:, :, None, None]
        predicted = torch.einsum("bhkv,bhk->bhv", S, k_t)
        delta = beta_t[:, :, None] * (v_t - predicted)
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, delta)
        outputs.append(torch.einsum("bhkv,bhk->bhv", S, q_t))
        if all_states:
            states.append(S)
    if all_states:
        return torch.stack(outputs, dim=1), S, torch.stack(states, dim=1)
    return torch.stack(outputs, dim=1), S
