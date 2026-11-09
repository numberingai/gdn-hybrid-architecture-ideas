"""EDA's separate erase direction in a Gated DeltaNet layer: GDN memory-rule the KDA + erase form (KDA + erase), 5 and 7 (EDA
alone: the erase on GDN's scalar decay).

THE ASK. The survey gives the mechanism, the 2T interleave and "our form" of the tensors. Config: GDNConfig.erase (ced/config.py
EraseConfig); off by default.

THE RULE (EDA, arXiv 2606.26560, Eq. 8), per value head, in our layout S in R^{d_k x d_v}, o_t = S_t^T q_t:

    S_t = (I - beta_t k_t k_t^T)(I - gamma_t e_t e_t^T) D_t S_{t-1} + beta_t k_t v_t^T

per token: reset at a document start; the decay S <- D_t S (GDN's scalar a_t, or KDA's Diag(a_t) with its floor: GDNConfig.kda,
ced/model/gdn_kda.py); the erase S <- S - gamma_t e_t (e_t^T S) (a gradient step on 1/2 ||S^T e_t||^2 along e_t, EDA Eqs. 10-11:
a delta-rule step with value 0); GDN's delta write S <- S + k_t (beta_t (v_t - S^T k_t))^T; the read S^T q_t. With gamma = 0 it
is the base layer's rule (GDN's or KDA's) exactly (tests/test_gdn_erase.py).

THE TENSORS (GatedDeltaNet.erase, an EraseGate; the survey's "our form", its section 3.3 (a); the paper gives no shapes):
- e_t = L2norm(W_up W_down x_t), one unit vector per KEY head: e_down d -> r H_k, e_up r -> d_k on each key head's own rank slice
  (block diagonal, one batched GEMM over the key heads; r = EraseConfig.rank = 16, the EDA paper's "rank-16 (per-head)
  erase-address projection", its Appendix A); normalised in float32 at least (l2_normalize, the layer's eps 1e-6). Value head j
  erases along the e of its key head j // (H_v / H_k), as it writes along that key head's k (GQA kept: the fused kernels take e
  at the key heads).
- gamma_t = c(s) sigmoid(W_gamma x_t), one per VALUE head: in_proj_gamma d -> H_v, s = `scale` (H_v,), zero-initialised.
- Parameters (exact): d r H_k + r H_k d_k + d H_v + H_v a layer; at 60M (d 512, 2 / 4 heads of 128) 22,532 a layer, 337,980 over
  the 15 GDN layers (0.53 % of R1-entry's 63.74M non-embedding; the survey's +0.34M); at 4B 481,300 a layer (the survey's).

THE GATE c (survey's: clamp(s, 0, 1) has a dead zone, so one update that takes s under 0 switches that head's erase off for the
rest of the run, with no weight decay and no gradient to bring it back; and on torch 2.14.1, the GPU runs' pin, its gradient is 0
at s = 0 itself (review, measured), so a zero-initialised plain clamp would never open): `inward_clamp`, a straight-through clamp
that passes the gradient only toward [0, 1] outside it. Forward: clamp(s, 0, 1) exactly. Backward: inside [0, 1], the bounds
included, the gradient as it is; below 0 only a gradient that raises s under descent (grad < 0); above 1 only one that lowers it
(grad > 0). So the head's erase can always reopen (the gradient toward it always passes: never switched off), and the gradient
that would take s further out is not applied (s does not drift away from the interval while the erase is shut: a plain
straight-through clamp would let it run to -X while e, which gets no gradient while gamma = 0, cannot learn). Tested
(tests/test_gdn_erase.py: the plain clamp's dead zone, this form's live gradient, s brought back from below 0 and held near 0). s
zero-initialised: gamma = 0 at the start, so each arm starts as its base arm (as, as, as R1-entry of the same seed: the same
initial tensors, below, and the same function). At s = 0 the gradient passes (inside the interval).

INITIALISATION. The erase's tensors are drawn from a generator of their own (init_own_parameters, called by the model's build after
init_weights; seed torch.initial_seed() + SEED_OFFSET; ced/model/gdn_dprod.py's and ced/model/ldm.py's rule), so every tensor of the
base arm starts as in the base arm of the same seed and the global random stream is not touched: e_down, e_up, in_proj_gamma N(0,
init_std) (the trunk's rule for linears, fla's 0.02), scale zeros. A layer built on its own (no model) draws them at construction
from a fixed generator (STANDALONE_SEED), never the global one.

WEIGHT DECAY (the standing rule): `.erase.scale` is in ZERO_GATE_SUFFIXES (ced/train/optim.py), so under optim.zero_gate_decay=none
(the arms' setting) it trains without weight decay; e_down, e_up and in_proj_gamma are the gate's matrices and go where in_proj_a
and in_proj_b go (ADAMW_MATRIX_SUFFIXES: AdamW, never Muon, with the run's decay).

THE PATHS (GDNConfig.kernel; each the same function, the per-token loop the ground truth):
- "recurrence" (and a step of one token): erase_recurrence, the rule above token by token.
- "chunked", and "fla" / "ced" off the CUDA training path (the CPU, cached chunks with a carried state): `interleave` into the
  chunked reference of the layer's decay (ced/model/gdn_chunked.py chunked_delta_rule for GDN's, gdn_kda.kda_chunked for KDA's):
  the 2T interleave of EDA's Sec. 3.4, Eq. 17: token t becomes sub-step 2t, the erase (k' = e_t, beta' = gamma_t, v' = 0, q' = 0,
  g' = g_t, a document start where t starts one) and sub-step 2t + 1, the write (k_t, beta_t, v_t, q_t, g' = 0, no start); the
  outputs are read on the write sub-steps, the state after the last one is the token's.
- the fused path (CUDA, kernel "fla" / "ced", from an empty state with varlen boundaries: the training and scoring forward): the
  same interleave into the fused kernel of the layer's decay, AS IS, with the boundaries x 2: fla 0.5.2's chunk_kda
  (gdn_kda.kda_fused) for KDA's decay; for GDN's our kernel (ced/model/gdn.py ced_delta_rule) with kernel "ced", fla's
  chunk_gated_delta_rule with kernel "fla". Both take q, k at the key heads (GVA), any beta in [0, 1] and g <= 0 per sub-step,
  nothing about what a sub-step stands for. With kernel "ced" the input preparation stays our fused qkv_prep. A fast path is
  accepted only when it matches the reference: scripts/gdn_erase_check.py on a GPU (float64 reference, the 4x criterion of
  ced/utils/criterion.py). For KDA's decay it is built as K1 (ced/kernels/gdn_k1.py, KDAConfig.kernel "k1": then call it with e and
  gamma, no interleave); for GDN's it is K2 (ced/kernels/gdn_k2.py, KDAConfig.kernel "k2" with decay "scalar": then calls it with e
  and gamma, no interleave). Fixed shapes throughout (stack and reshape; no index op, no boolean mask).

NOT BUILT, refused by name (check_config, check_model, forward): F3 (decay_cumsum "compensated"), the fused edges (edges !=
"plain"), FlashQLA (chunk_backend), CUDA graphs' fixed-capacity GDN metadata, a zero-write second pass (keep), decoder slots
(valid), the GDN-side arms, TST's bag phases, the decoder's GDN layers and arches other than "control", the MTP drafter, FP8 /
NVFP4 linears; and every tool that walks a GDN state from GatedDeltaNet.gates (the decode engine, the MTP drafter, the GDN state
diagnostics, gdn_memory): gates() refuses an erase layer (GATES_REFUSAL). The head census reads the layer's decay through
decay_gates (scripts/gdn_g_dist.py --census): the decay only, not the erase. Python 3.11 compatible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ced.model import gdn_kda

MODES = ("none", "eda")
DEFAULT_RANK = 16  # the EDA paper's "rank-16 (per-head) erase-address projection" (arXiv 2606.26560, Appendix A)
KERNELS = ("recurrence", "chunked", "fla", "ced")
SEED_OFFSET = 0xE7A5E_2026_1022  # the erase's own stream: seed = torch.initial_seed() + this
STANDALONE_SEED = 0xE7A5E  # a layer built outside a model (benches, kernel checks): a fixed stream, not the global one
# Calls of the fused path on the interleave (the GPU check reads it to see that the model ran the fused kernel, never a reference)
RAN = {"interleave_fused": 0}
GATES_REFUSAL = ("this GDN layer erases its state along a direction of its own (GDNConfig.erase.mode 'eda'; ced/model/gdn_erase.py): "
                 "gates(x) gives its decay and beta only, so a tool that walks the GDN state from them (the decode engine, the MTP "
                 "drafter, the GDN state diagnostics, gdn_memory) would compute another layer; it is not built for it. The layer's "
                 "decay alone: gdn_erase.decay_gates(layer, x)")


# ------------------------------------------------------------------------------------------------------------------------------
# the keys


def is_on(cfg) -> bool:
    """cfg: a GDNConfig. True when it erases (EraseConfig.mode "eda")."""
    return cfg.erase.mode == "eda"


def check_config(cfg) -> None:
    """GDNConfig.erase, checked (ValueError naming the key). With mode "none" rank must be at its default."""
    from ced.config import EraseConfig

    e = cfg.erase
    if not isinstance(e, EraseConfig):
        raise ValueError(f"GDNConfig.erase {e!r}: an EraseConfig")
    if e.mode not in MODES:
        raise ValueError(f"GDNConfig.erase.mode {e.mode!r}: one of {MODES}")
    if e.mode == "none":
        if e.rank != EraseConfig().rank:
            raise ValueError("GDNConfig erase.rank is read with erase.mode 'eda'; this layer's erase.mode is 'none'")
        return
    if type(e.rank) is not int or not 1 <= e.rank <= cfg.k_head_dim:
        raise ValueError(f"GDNConfig.erase.rank {e.rank!r}: an integer in [1, k_head_dim = {cfg.k_head_dim}]")
    refused = [(cfg.kernel not in KERNELS, f"kernel {cfg.kernel!r} (built for {KERNELS})"),
               (cfg.decay_cumsum != "plain", f"decay_cumsum {cfg.decay_cumsum!r} (F3: not checked on the interleave)"),
               (cfg.edges != "plain", f"edges {cfg.edges!r} (the fused edges ops form q, k, v of one step a token)"),
               (cfg.chunk_backend != "triton", f"chunk_backend {cfg.chunk_backend!r} (built on our Triton kernel's call)")]
    bad = [why for cond, why in refused if cond]
    if bad:
        raise ValueError(f"GDNConfig.erase.mode 'eda' (ced/model/gdn_erase.py) is not built with: {'; '.join(bad)}")


def check_model(cfg) -> None:
    """The model-level refusals (ced/model/control.py build_model): the erase only in arch "control"'s encoder, without the GDN-side
    arms, the gated second pass, the MTP drafter or FP8 / NVFP4 linears."""
    check_config(cfg.encoder.gdn)
    check_config(cfg.decoder.gdn)
    if is_on(cfg.decoder.gdn):
        raise ValueError("model.decoder.gdn.erase.mode 'eda': the erase direction is built for the encoder's GDN layers only")
    if not is_on(cfg.encoder.gdn):
        return
    refused = [(cfg.arch != "control", f"arch {cfg.arch!r} (built for 'control': its build draws the erase's own tensors)"),
               (cfg.encoder.gdn_write_mix != "none", f"model.encoder.gdn_write_mix {cfg.encoder.gdn_write_mix!r} (the GDN-side arm A)"),
               (bool(cfg.encoder.gdn_ccr_spacing), f"model.encoder.gdn_ccr_spacing {cfg.encoder.gdn_ccr_spacing!r} (the GDN-side arm "
                                                   "CCR)"),
               (cfg.loop.pass2_gate != "none", f"model.loop.pass2_gate {cfg.loop.pass2_gate!r} (the zero-write second pass reads "
                                               "GDN's record)"),
               (bool(cfg.mtp.enabled), "model.mtp.enabled (the drafter writes the GDN step itself, ced/infer/mtp_spec.py)"),
               (cfg.fp8.mode != "none", f"model.fp8.mode {cfg.fp8.mode!r}"),
               (cfg.nvfp4.mode != "none", f"model.nvfp4.mode {cfg.nvfp4.mode!r}")]
    bad = [why for cond, why in refused if cond]
    if bad:
        raise ValueError(f"model.encoder.gdn.erase.mode 'eda' (ced/model/gdn_erase.py) is not built with: {'; '.join(bad)}")


def added_parameters(cfg, d_model: int) -> dict:
    """The tensors one GDN layer gains with the erase (exact, from the shapes)."""
    hk, hv, dk, r = cfg.n_k_heads, cfg.n_v_heads, cfg.k_head_dim, cfg.erase.rank
    out = {"e_down": d_model * r * hk, "e_up": r * hk * dk, "in_proj_gamma": d_model * hv, "scale": hv}
    out["per_layer"] = sum(out.values())
    return out


# ------------------------------------------------------------------------------------------------------------------------------
# the gate


class _InwardClamp(torch.autograd.Function):
    """clamp(s, 0, 1) forward; backward the gradient inside [0, 1], and outside it only the part that moves s toward [0, 1] under
    descent (below 0: grad <= 0; above 1: grad >= 0). Module docstring, THE GATE."""

    @staticmethod
    def forward(ctx, s):
        ctx.save_for_backward(s)
        return s.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, grad):
        (s,) = ctx.saved_tensors
        return torch.where(s < 0, grad.clamp(max=0.0), torch.where(s > 1, grad.clamp(min=0.0), grad))


def inward_clamp(s: torch.Tensor) -> torch.Tensor:
    """The erase gate's scale c(s) in [0, 1] (module docstring, THE GATE): clamp(s, 0, 1) exactly, with a gradient that is never
    cut toward the interval."""
    return _InwardClamp.apply(s)


class EraseGate(nn.Module):
    """A GDN layer's erase tensors (GatedDeltaNet.erase): e_down (d -> r H_k), e_up ((H_k d_k, r): each key head's r -> d_k block),
    in_proj_gamma (d -> H_v) and scale (H_v,), zeros. Built on the meta device and moved to `device` without a draw (no random
    number of the global stream), then drawn from a fixed generator (reset_own); the model's build draws them again from the run's
    own stream (init_own_parameters)."""

    def __init__(self, d_model: int, cfg, device=None, dtype=None):
        super().__init__()
        check_config(cfg)
        assert is_on(cfg)
        self.hk, self.hv, self.dk, self.rank = cfg.n_k_heads, cfg.n_v_heads, cfg.k_head_dim, cfg.erase.rank
        dev = torch.device("cpu") if device is None else torch.device(device)

        def linear(n_in: int, n_out: int) -> nn.Linear:
            lin = nn.Linear(n_in, n_out, bias=False, device="meta", dtype=dtype).to_empty(device=dev)
            lin.own_init = True  # init_weights (ced/model/looped_ced.py) draws every other linear from the global stream
            return lin

        self.e_down = linear(d_model, self.rank * self.hk)
        self.e_up = linear(self.rank, self.hk * self.dk)  # (H_k d_k, r), applied block by block (direction)
        self.in_proj_gamma = linear(d_model, self.hv)
        self.scale = nn.Parameter(torch.zeros(self.hv, device=dev, dtype=dtype))
        self.reset_own(torch.Generator().manual_seed(STANDALONE_SEED), 0.02)

    @torch.no_grad()
    def reset_own(self, gen: torch.Generator, std: float) -> None:
        """e_down, e_up, then in_proj_gamma: N(0, std) from the CPU generator `gen` in float32, copied in; scale to zeros (no draw).
        Tensors on the meta device are skipped."""
        for p in (self.e_down.weight, self.e_up.weight, self.in_proj_gamma.weight):
            if not p.is_meta:
                p.copy_((torch.randn(p.shape, generator=gen, dtype=torch.float32) * std).to(device=p.device, dtype=p.dtype))
        if not self.scale.is_meta:
            self.scale.zero_()

    def direction(self, x: torch.Tensor) -> torch.Tensor:
        """e (..., H_k, d_k): the erase address of each key head, a unit vector (float32 at least; the layer's l2_normalize)."""
        from ced.model.gdn import l2_normalize

        z = self.e_down(x).unflatten(-1, (self.hk, self.rank))
        # one batched GEMM over the key heads (H_k, tokens, r) x (H_k, r, d_k): not z[..., None, :] @ W^T, which would broadcast W_up
        # to every token and save that copy for the backward (ced/model/gdn_kda.py KDAGate.decay_input, review fix 1 there)
        w = self.e_up.weight.to(z.dtype).view(self.hk, self.dk, self.rank)
        e = torch.einsum("...hr,hdr->...hd", z, w)
        return l2_normalize(e.to(torch.promote_types(e.dtype, torch.float32)))

    def strength(self, x: torch.Tensor) -> torch.Tensor:
        """gamma (..., H_v) in [0, 1]: inward_clamp(scale) x sigmoid(in_proj_gamma(x))."""
        return inward_clamp(self.scale) * torch.sigmoid(self.in_proj_gamma(x))

    def extra_repr(self) -> str:
        return f"rank={self.rank}, key_heads={self.hk}, value_heads={self.hv}"


def build(layer: nn.Module, d_model: int, cfg) -> EraseGate | None:
    """GatedDeltaNet's hook (the end of its __init__): None for mode "none" (the layer as it was: nothing built, nothing drawn);
    else the erase's tensors on the layer's device and dtype."""
    check_config(cfg)
    if not is_on(cfg):
        return None
    w = layer.in_proj_qkv.weight
    return EraseGate(d_model, cfg, device=w.device, dtype=w.dtype)


def own_generator() -> torch.Generator:
    return torch.Generator().manual_seed((torch.initial_seed() + SEED_OFFSET) % (1 << 63))


def init_own_parameters(model: nn.Module, std: float) -> int:
    """Draw every EraseGate of `model`, in module order, from own_generator() (EraseGate.reset_own). Called by the model's build
    after init_weights; returns how many were drawn (0 in every model without the key: then nothing is drawn and no generator is
    made)."""
    gates = [m for m in model.modules() if isinstance(m, EraseGate)]
    if not gates:
        return 0
    g = own_generator()
    for m in gates:
        m.reset_own(g, std)
    return len(gates)


def decay_gates(layer, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(g, beta) as an erase layer's forward forms them, the erase left out: KDA's per-channel gate (gdn_kda.gates: g (..., H_v,
    d_k)) or GDN's (GatedDeltaNet.gates_from on in_proj_a(x), in_proj_b(x): the same lines as gates(x), bit for bit; g (..., H_v)),
    with the evaluation-only floor and cap as there. The head census reads this (scripts/gdn_g_dist.py)."""
    if layer.kda is not None:
        return gdn_kda.gates(layer, x)
    return layer.gates_from(layer.in_proj_a(x), layer.in_proj_b(x))


# ------------------------------------------------------------------------------------------------------------------------------
# the rule: the reference loop and the 2T interleave


def erase_recurrence(q, k, v, g, beta, e, gamma, is_doc_start, S, all_states: bool = False):
    """The rule one token at a time (module docstring, THE RULE): the ground truth. q, k, e: (B, L, H, d_k) at the value heads
    (repeated); v: (B, L, H, d_v); g: (B, L, H) (GDN's scalar decay) or (B, L, H, d_k) (KDA's per channel); beta, gamma: (B, L, H);
    is_doc_start: (B, L) bool; S: (B, H, d_k, d_v), the state before the first token. Per token: reset at a document start, the
    decay, the erase S - gamma e (S^T e)^T, the write k (x) beta (v - S^T k), the read S^T q. Returns (outputs (B, L, H, d_v), the
    final state); all_states: also the state after every token (B, L, H, d_k, d_v)."""
    per_channel = g.dim() == 4
    outputs, states = [], []
    # unbind once (indexing in the loop would make the backward O(L^2), as delta_rule_recurrence's note says)
    steps = zip(q.unbind(1), k.unbind(1), v.unbind(1), g.unbind(1), beta.unbind(1), e.unbind(1), gamma.unbind(1),
                is_doc_start.unbind(1))
    for q_t, k_t, v_t, g_t, b_t, e_t, c_t, start_t in steps:
        S = S * (~start_t).to(S.dtype)[:, None, None, None]  # a new document starts from an empty state
        S = S * (torch.exp(g_t)[..., None] if per_channel else torch.exp(g_t)[:, :, None, None])
        erased = torch.einsum("bhkv,bhk->bhv", S, e_t)  # S^T e
        S = S - torch.einsum("bhk,bhv->bhkv", e_t, c_t[..., None] * erased)
        predicted = torch.einsum("bhkv,bhk->bhv", S, k_t)
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, b_t[..., None] * (v_t - predicted))
        outputs.append(torch.einsum("bhkv,bhk->bhv", S, q_t))
        if all_states:
            states.append(S)
    if all_states:
        return torch.stack(outputs, dim=1), S, torch.stack(states, dim=1)
    return torch.stack(outputs, dim=1), S


def interleave(q, k, v, g, beta, e, gamma, is_doc_start=None):
    """The 2T interleave (module docstring, THE PATHS): token t -> sub-step 2t, the erase (q' = 0, k' = e_t, v' = 0, g' = g_t,
    beta' = gamma_t, a document start where t starts one) and sub-step 2t + 1, the write (q_t, k_t, v_t, g' = 0, beta_t, no start).
    q, k, e: (B, L, H, d_k) (the key heads for the fused kernels, the value heads for the references; e in k's dtype); v: (B, L,
    H_v, d_v); g: (B, L, H_v) or (B, L, H_v, d_k); beta, gamma: (B, L, H_v) (gamma in beta's dtype); is_doc_start: (B, L) or None.
    Returns (q, k, v, g, beta, is_doc_start or None) of length 2L. A broadcast select and a reshape (fixed shapes, no index op):
    the values are the inputs' exactly, and each input's gradient is a sum over the pair (a contiguous tensor; the slices a stack's
    backward hands back are strided, which aot_eager's traced linear backwards read with a view: tests/test_gdn_erase.py, compile)."""
    B, L = q.shape[:2]

    def pair(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        write = (torch.arange(2, device=first.device) == 1).view(1, 1, 2, *([1] * (first.dim() - 2)))  # True on sub-step 2t + 1
        return torch.where(write, second.unsqueeze(2), first.unsqueeze(2)).reshape(B, 2 * L, *first.shape[2:])

    starts = None if is_doc_start is None else pair(is_doc_start, torch.zeros_like(is_doc_start))
    return (pair(torch.zeros_like(q), q), pair(e, k), pair(torch.zeros_like(v), v), pair(g, torch.zeros_like(g)),
            pair(gamma, beta), starts)


def write_substeps(o: torch.Tensor) -> torch.Tensor:
    """(B, 2L, H, d_v) interleaved outputs -> (B, L, H, d_v): each token's output, read on its write sub-step."""
    B, T = o.shape[:2]
    return o.reshape(B, T // 2, 2, *o.shape[2:])[:, :, 1]


def erase_wy_chunked(q, k, v, g, beta, e, gamma, is_doc_start, S, chunk_tokens: int = 32):
    """THE FUSED KERNEL'S ALGORITHM, in PyTorch (not on any model path): both rank-one updates of a token in one WY / UT form, a chunk
    of `chunk_tokens` tokens holding 2 x chunk_tokens rows (32 tokens = 64 rows, our kernel's tile), row 2j the erase of token j
    (key e_j, strength gamma_j, value 0) and row 2j + 1 its write (key k_j, strength beta_j, value v_j); the cumulative log decay c
    advances on the erase rows only (c_{2j} = c_{2j-1} + g_j, c_{2j+1} = c_{2j}); per channel (GDN's scalar decay is the same g in
    every channel). With entry state S_0:

        A[i, l] = b_i sum_d key_{i,d} key_{l,d} exp(c_{i,d} - c_{l,d}), l < i, the same document     (all 2C x 2C: e.e, e.k, k.e, k.k)
        T       = (I + A)^-1                                                                       (the solve, as our kernel's)
        W       = T diag(b) (exp(c) * keys)    (every row; the S_0 term in the first document only)
        U       = T[:, write] diag(beta) V     THE VALUE SKIP: V is 0 on the erase rows, so only T's write columns are read
        v_new   = U - W S_0
        o_j     = (exp(c_{2j+1}) * q_j)^T S_0 + sum_{l <= 2j+1} (sum_d q_{j,d} key_{l,d} exp(c_{2j+1,d} - c_{l,d})) v_new_l
                                               THE OUTPUT SKIP: q is 0 on the erase rows, so only the C write rows are queried
        S_C     = Diag(exp(c_last)) S_0 + sum_l (exp(c_last - c_l) * key_l) v_new_l^T

    q, k, e: (B, L, H, d_k) at the value heads; v: (B, L, H, d_v); g: (B, L, H) or (B, L, H, d_k); beta, gamma: (B, L, H);
    is_doc_start: (B, L); S: (B, H, d_k, d_v). Returns (outputs (B, L, H, d_v), the final state), the per-token loop's function
    (tests/test_gdn_erase.py, float64). Fixed chunks, documents by masks, the pairwise exponents summed directly and the double
    `where` (kda_chunked's conventions); padding rows carry no decay and no write."""
    from functools import reduce

    B, L, H, dk = q.shape
    dv = v.shape[-1]
    if g.dim() == 3:
        g = g[..., None].expand(B, L, H, dk)
    out_dtype = reduce(torch.promote_types, (t.dtype for t in (q, k, v, g, beta, e, gamma, S)))
    work = torch.promote_types(out_dtype, torch.float32)
    C = max(1, min(chunk_tokens, L))
    n = -(-L // C)
    pad = n * C - L
    R = 2 * C

    def per_head(x: torch.Tensor) -> torch.Tensor:
        """(B, L, H, ...) -> (B, H, n, C, ...), padded at the end of the row with zeros."""
        x = x.to(work).transpose(1, 2)
        trailing = x.dim() - 3
        return F.pad(x, (0, 0) * trailing + (0, pad)).reshape(B, H, n, C, *x.shape[3:])

    def rows(erase_part: torch.Tensor, write_part: torch.Tensor) -> torch.Tensor:
        """(B, H, n, C, ...) twice -> (B, H, n, 2C, ...): row 2j from the first, row 2j + 1 from the second."""
        return torch.stack([erase_part, write_part], dim=4).reshape(*erase_part.shape[:3], R, *erase_part.shape[4:])

    q, k, v, g, beta, e, gamma = map(per_head, (q, k, v, g, beta, e, gamma))
    keys, b, grow = rows(e, k), rows(gamma, beta), rows(g, torch.zeros_like(g))
    starts = F.pad(is_doc_start.to(torch.long), (0, pad)).reshape(B, 1, n, C)
    seg = rows(starts, torch.zeros_like(starts)).cumsum(-1)  # (B, 1, n, 2C): a document starts on its first token's erase row
    causal = torch.ones(R, R, dtype=torch.bool, device=q.device).tril()
    strictly_lower = causal.tril(-1)
    eye = torch.eye(R, dtype=work, device=q.device)
    zero = torch.zeros((), dtype=work, device=q.device)

    state = S.to(work)
    outputs = []
    for c_ in range(n):
        q_c, v_c, beta_c = q[:, :, c_], v[:, :, c_], beta[:, :, c_]
        key_c, b_c, g_c, seg_c = keys[:, :, c_], b[:, :, c_], grow[:, :, c_], seg[:, :, c_]
        pair_ok = (seg_c[..., :, None] == seg_c[..., None, :]) & causal  # (B, 1, R, R)
        in_first = (seg_c == 0).to(work)[..., None]  # (B, 1, R, 1)
        in_last = (seg_c == seg_c[..., -1:]).to(work)[..., None]
        carry = (seg_c[..., -1] == 0).to(work)[..., None]
        pair_exp = torch.where(strictly_lower[:, :, None], g_c[..., :, None, :], zero).cumsum(-3)  # (B, H, R, R, dk): c_i - c_l
        ok = pair_ok[..., None]
        pair_decay = torch.where(ok, torch.exp(torch.where(ok, pair_exp, zero)), zero)
        gam = torch.exp(g_c.cumsum(-2))  # (B, H, R, dk): exp(c_i)
        to_end = torch.exp(pair_exp[..., -1, :, :])  # (B, H, R, dk): exp(c_last - c_l)
        key_b = key_c * b_c[..., None]
        A = torch.einsum("bhid,bhjd,bhijd->bhij", key_b, key_c, pair_decay) * strictly_lower
        T = torch.linalg.solve_triangular(eye + A, eye.expand_as(A), upper=False, unitriangular=True)
        U = T.reshape(*T.shape[:-1], C, 2)[..., 1] @ (v_c * beta_c[..., None])  # the value skip: T's write columns only
        W = T @ (key_b * gam * in_first)
        written = U - W @ state  # (B, H, R, dv)
        # the output skip: the C write rows (row 2j + 1) only
        decay_w = pair_decay.reshape(B, H, C, 2, R, dk)[:, :, :, 1]  # (B, H, C, R, dk)
        gam_w = gam.reshape(B, H, C, 2, dk)[:, :, :, 1]
        first_w = in_first.reshape(B, 1, C, 2, 1)[:, :, :, 1]
        intra = torch.einsum("bhid,bhjd,bhijd->bhij", q_c, key_c, decay_w)  # (B, H, C, R)
        outputs.append((q_c * gam_w * first_w) @ state + intra @ written)
        state = state * (gam[..., -1, :] * carry)[..., None] + (key_c * to_end * in_last).transpose(-1, -2) @ written
    o = torch.stack(outputs, dim=2).reshape(B, H, n * C, dv)[:, :, :L].transpose(1, 2)
    return o.to(out_dtype), state.to(out_dtype)


# ------------------------------------------------------------------------------------------------------------------------------
# the layer's forward with the erase (GatedDeltaNet.forward's first line hands an erase layer here)


@torch.compiler.disable  # never traced, as the kernels' own wrappers: the boundaries' host copy is a CPU tensor
def _fused_call(per_channel: bool, kernel: str, q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu):
    """The fused kernel of the layer's decay on the 2T interleave, the boundaries (device and host copies) x 2: fla's chunk_kda
    (gdn_kda.kda_fused) for KDA's decay; our kernel (kernel "ced") or fla's chunk_gated_delta_rule (kernel "fla") for GDN's."""
    from ced.model import gdn as G

    RAN["interleave_fused"] += 1
    cu_cpu = cu_seqlens_cpu * 2 if cu_seqlens_cpu is not None else None
    if per_channel:
        return gdn_kda.kda_fused(q, k, v, g, beta, cu_seqlens * 2, cu_cpu)
    fused = G.fla_delta_rule if kernel == "fla" else G.ced_delta_rule
    return fused(q, k, v, g, beta, cu_seqlens * 2, cu_cpu)


def forward(layer, x, ctx, state=None, valid=None, keep=False, arms=None, x_write=None, ccr=None, x_read=None, lerped=False):
    """GatedDeltaNet.forward of an erase layer (layer.erase not None): the same inputs and result, (y, GDNState). R1-entry's input
    preparation, the layer's decay and beta (decay_gates: GDN's or KDA's), the erase's e and gamma, the paths above, R1-entry's
    output norm and projection."""
    from ced.model import gdn as G
    from ced.model.gdn_chunked import chunked_delta_rule

    refused = [(keep, "keep (the zero-write second pass reads GDN's one-step record)"),
               (valid is not None, "valid (decoder slots)"),
               (arms is not None or x_write is not None or ccr is not None or x_read is not None or layer.arms_attached,
                "the GDN-side arms (model.encoder.gdn_write_mix, gdn_ccr_spacing)"),
               (ctx.gdn_scale != 1, f"TST's resolution-explicit gates (a bag phase of {ctx.gdn_scale} tokens)"),
               (bool(getattr(layer, "mx_out_norm", False)), "model.fp8.chains gdn_out_norm")]
    bad = [why for cond, why in refused if cond]
    if bad:
        raise ValueError(f"GDNConfig.erase.mode 'eda' (ced/model/gdn_erase.py) is not built with: {'; '.join(bad)}")
    cfg = layer.cfg
    B, L, _ = x.shape
    hk, hv, dk, dv = cfg.n_k_heads, cfg.n_v_heads, cfg.k_head_dim, cfg.v_head_dim
    K = cfg.conv_kernel
    per_channel = layer.kda is not None
    u_pre = layer.in_proj_qkv(x)
    fused_ok = state is None and ctx.cu_seqlens is not None and G.fla_supported(x)
    use_fused = cfg.kernel in ("fla", "ced") and fused_ok
    # input_dtype "bfloat16": q, k, v, e, beta and gamma reach the fused kernel in bf16 under bf16 autocast (g stays float32), as on
    # the base layer (GDNConfig.input_dtype)
    kernel_dtype = (torch.bfloat16 if use_fused and cfg.input_dtype == "bfloat16" and u_pre.dtype == torch.bfloat16 else None)
    if layer._fused_prep and fused_ok and L >= K - 1:  # kernel "ced": our fused input preparation (the same function)
        q, k, v = G.qkv_prep(u_pre, ctx.doc, layer.conv_weight, hk, hv, dk, out_dtype=kernel_dtype)
        conv_input, conv_doc = u_pre[:, L - (K - 1):], ctx.doc[:, L - (K - 1):]
    else:
        u, conv_input, conv_doc = layer.short_conv(u_pre, ctx, state)
        u = F.silu(u)
        q, k, v = torch.split(u, [hk * dk, hk * dk, hv * dv], dim=-1)
        q = G.l2_normalize(q.view(B, L, hk, dk)) * dk**-0.5
        k = G.l2_normalize(k.view(B, L, hk, dk))
        v = v.view(B, L, hv, dv)
        if kernel_dtype is not None:
            q, k, v = q.to(kernel_dtype), k.to(kernel_dtype), v.to(kernel_dtype)
    k1 = use_fused and per_channel and cfg.kda.kernel == "k1"  # K1 (KDAConfig.kernel): both rank-1 updates in one WY form, r = 2
    k2 = use_fused and not per_channel and cfg.kda.kernel == "k2"  # K2: the same on GDN's scalar decay
    prologue = gdn_kda.k1_prologue(layer, x) if k1 else None
    if prologue is None:
        g, beta = decay_gates(layer, x)
    else:  # K1 forms g in its prologue: beta as gdn_kda.gates forms it
        g, beta = None, torch.sigmoid(layer.in_proj_b(x))
    e = layer.erase.direction(x).to(k.dtype)  # (B, L, H_k, d_k), unit, in k's dtype
    gamma = layer.erase.strength(x).to(beta.dtype)  # (B, L, H_v)
    if kernel_dtype is not None:
        beta, gamma = beta.to(kernel_dtype), gamma.to(kernel_dtype)
    if use_fused:
        if ctx.gdn_chunks is not None:
            raise ValueError("GDNConfig.erase.mode 'eda' with TrainConfig.cuda_graphs: not built (the fixed-capacity GDN metadata is "
                             "built for the token boundaries, not the 2T interleave's)")
        if k1:
            o = gdn_kda.k1_fused(q, k, v.contiguous(), g, beta, ctx.cu_seqlens, ctx.cu_seqlens_cpu, e=e, gamma=gamma, gate=prologue)
        elif k2:
            o = gdn_kda.k2_fused(q, k, v.contiguous(), g, beta, ctx.cu_seqlens, ctx.cu_seqlens_cpu, e=e, gamma=gamma)
        else:
            qi, ki, vi, gi, bi, _ = interleave(q, k, v.contiguous(), g, beta, e, gamma)
            o = write_substeps(_fused_call(per_channel, cfg.kernel, qi, ki, vi, gi, bi, ctx.cu_seqlens, ctx.cu_seqlens_cpu))
        S = None  # training / scoring forward: the final state is not needed (no cache is built)
    else:
        rep = hv // hk
        q_r, k_r, e_r = (t.repeat_interleave(rep, dim=2) for t in (q, k, e))
        S0 = x.new_zeros(B, hv, dk, dv) if state is None else state.S
        if cfg.kernel in ("chunked", "fla", "ced") and L > 1:
            qi, ki, vi, gi, bi, si = interleave(q_r, k_r, v, g, beta, e_r, gamma, ctx.is_doc_start)
            rule = gdn_kda.kda_chunked if per_channel else chunked_delta_rule
            oi, S = rule(qi, ki, vi, gi, bi, si, S0, cfg.chunk_size)
            o = write_substeps(oi)
        else:
            o, S = erase_recurrence(q_r, k_r, v, g, beta, e_r, gamma, ctx.is_doc_start, S0)
    if layer.head_keep is not None:  # evaluation only: the zeroed heads' readouts are 0 (GatedDeltaNet.set_head_zero)
        o = layer.keep_heads(o)
    z = layer.in_proj_z(x).view(B, L, hv, dv)
    y = layer.out_proj(layer.norm(o, z).flatten(2))
    return y, G.GDNState(conv_input=conv_input, conv_doc=conv_doc, S=S)


def scales(model: nn.Module) -> dict:
    """{erase module name: c(s) per value head}: each erase layer's gate scale as the forward applies it (0: that head's
    erase is shut). Read beside the head census for the erase arms (the census reads the decay only)."""
    return {n: [float(x) for x in m.scale.detach().clamp(0.0, 1.0).double()] for n, m in model.named_modules()
            if isinstance(m, EraseGate)}
