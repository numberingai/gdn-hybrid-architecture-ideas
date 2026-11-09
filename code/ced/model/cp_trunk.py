"""The Copy-Paste head's copy distribution read from the trunk's own global attention heads (model.cp_head.trunk_copy; off by default).

THE FORM. CP's mixture is unchanged (ced/model/cp_head.py: the <copy> row c_t = softmax over V u {<copy>} at <copy>, z_copy = w_copy. h_t,
lambda = c (1 - s), p = (1 - lambda) p_LM + lambda p_CP, the loss the CE of p). What changes is where s and p_CP come from: no W_Q, W_K or
sink key; the global layers' own attention probabilities A_h(t, j) over the document's keys j <= t (the forward's own q after QK-norm, RoPE
or NoPE and the temperature (SSMax), the forward's own keys, the scale head_dim^-0.5, the rotary cap's far rule, the causal and document
mask, query head h reading key head h // (Hq / Hkv)), not detached: SHIFT +1 a head attending key j gives its mass to the token x_{j+1}
(next_tokens: the row's input ids shifted left by one). CP's own pointer copies x_tau, the token AT the attended position (cp_head.py:9,
48-49, 1001, 1069): the trunk's selecting heads point at the " is" just before the value. VALID(t) { j: idx_j < idx_t STRICTLY, doc_j =
doc_t, j not its document's first position }. For j = t, x_{j+1} is the target itself: the diagonal is never a copy source. Nothing at or
after t + 1 enters (causality). NO COPY a head's mass on the diagonal and on the document's first position (the emergent sink of a softmax
head) is CP's sink: s_h = A_h(t, t) + A_h(t, start). It stays in the head's own normaliser (the attention's softmax). PER HEAD with m the
row's largest score over the document's keys j <= t and e_j = exp(score_j - m): S_nc = sum over no-copy keys, S_v = sum over VALID(t), S_m
= sum over VALID(t) with x_{j+1} = y_t; log s_h = log S_nc - log(S_nc + S_v), log(1 - s_h) = log S_v - log(S_nc + S_v), log M_h(y) = log
S_m - log(S_nc + S_v) (an empty or underflowing sum: -inf through a guarded log, gradient exactly 0). MIXED w =
softmax(cp_head.trunk_logits) over the heads in use (zeros at the start: uniform); log s = logsumexp_h(log w_h + log s_h), log(1 - s) =
logsumexp_h(log w_h + log(1 - s_h)), log M = logsumexp_h(log w_h + log M_h), log p_CP(y) = log M - log(1 - s) where a match exists, else
-inf. Every logsumexp whose entries can all be -inf is guarded. These three per-position terms (log s, log(1 - s), log p_CP(y)) are what
cp_head.cp_terms returns for every other form, so the mixture, its closed-form gradient in the fused head (ced/train/fused_head.py) and
every scorer read them unchanged.

THE TWO FORMS. "all": every head of every global layer. "subset" (trunk_k = k, trunk_select_step = S): all heads for the first S optimizer
steps, then once, at the start of step S, the k heads with the largest logits (CopyPasteHead.select_trunk_subset; never fixed by index),
kept in the state dict (cp_head.trunk_keep, a (H_total,) bool buffer: a resume after S keeps the selection), and from then on only those
heads, with w = softmax over their logits.

THE REFERENCE PATH (this file: pure PyTorch, the ground truth). The heads run in GROUPS: one global layer's query heads (Hq), or after a
subset's selection the selected heads in groups of Hq (trunk_k must be a multiple of Hq), so every group has the same shape. Query block by
query block (TRUNK_CHUNK, 1,024 by default: at 8,192 eight blocks, so the compiled block function sees at most 16 shapes: 8 key lengths x
{with the cap's far scores, without}); with the forward's own keys, the block of queries a .. a + C - 1 reads the keys 0 .. a + C - 1 only.
Each block's [B, G, C, S] scores exist only while its forward or backward runs (activation checkpointing in training). Peak at 60M
training (B 3, G 4, C 1,024, S up to 8,192, float32): 403 MB a score tensor, a few of them at once (modelled; the entry check measures).

WHERE THE INPUTS COME FROM. Each global layer's GatedAttention (attention.py, trunk_copy attribute set by attach_cp_head) returns its
query (after the temperature) and, under the rotary cap, its far query on the key set it returns (KeySet.tc_q, tc_q_far); trunk_inputs
collects them with the key set's own keys (k, k_far, idx, doc) and drops them from the cache entries. The copied tokens and the document
starts of the keys come from this forward's tokens and positions, and in cached generation from the copy cache (TrunkCache: the earlier
positions' input tokens and start flags; the keys themselves are the attention's own cache). A layer whose attention was taken out (the
residual diagnostics' skips: its cache entry None) contributes no copy: its heads abstain (s_h = 1).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from ced.config import CPHeadConfig, ModelConfig
from ced.model.norms import compute_dtype

TRUNK_FORMS = ("none", "all", "subset")
TRUNK_CHUNK = 1024  # query positions a block of the reference path (module docstring)
EVAL_ELEMENTS = 1 << 27  # the scorers' blocks: at most this many score elements (heads x rows x keys) at once
LOGITS = "trunk_logits"  # cp_head.trunk_logits (H_total,): the heads' mixing logits, zeros at the start
KEEP = "trunk_keep"  # cp_head.trunk_keep (H_total,) bool buffer: the heads in use (all before a subset's selection)
TRUNK_KEYS = ("trunk_copy", "trunk_k", "trunk_select_step")
# the keys of model.cp_head the trunk copy does not read: each must stay at its default (a set value would be ignored or is not built)
NOT_READ = ("d_c", "qk_norm", "init", "window", "pointer_input", "temperature", "sink_count", "n_heads", "combine",
            "query_chunk")  # (query_chunk: the pointer's block; the trunk copy's is TRUNK_CHUNK, review 2026-10-21)
# model.cp_head.kernel under the trunk copy: "reference" (the default: this file's block function, as merged) or "fused"
# (ced/kernels/cp_trunk_fused.py: one pass over the row a group, forward and backward, with the forward's own keys; a cached chunk reads
# this file's path) or "fused2" (fast path, the same the "build its fast path first": ced/kernels/cp_trunk_fused2.py, the same terms by
# the second kernels, a layer's query heads packed on its key head, the layer's own q and k without copies; a cached chunk reads this
# file's path)
TRUNK_KERNELS = ("reference", "fused", "fused2")
TRUNK_ONLY_KERNELS = ("fused2",)  # values of model.cp_head.kernel that exist for the trunk copy only (the pointer has no such kernel)


def fused_on(head) -> bool:
    """The trunk copy's fused path (model.cp_head.kernel "fused" with model.cp_head.trunk_copy)."""
    return hasattr(head, "n_trunk") and head.cfg.kernel == "fused"


def fused2_on(head) -> bool:
    """The trunk copy's fast path (model.cp_head.kernel "fused2" with model.cp_head.trunk_copy)."""
    return hasattr(head, "n_trunk") and head.cfg.kernel == "fused2"


def trunk_on(c: CPHeadConfig) -> bool:
    return c.trunk_copy != "none"


def check_trunk(cfg: ModelConfig) -> None:
    """The keys of model.cp_head.trunk_copy: their values, and every pair the trunk copy is not built for, refused by name. ValueError."""
    c, d = cfg.cp_head, CPHeadConfig()
    if c.trunk_copy not in TRUNK_FORMS:
        raise ValueError(f"model.cp_head.trunk_copy {c.trunk_copy!r}: one of {TRUNK_FORMS}")
    for key in ("trunk_k", "trunk_select_step"):
        v = getattr(c, key)
        if type(v) is not int or v < 0:
            raise ValueError(f"model.cp_head.{key} {v!r}: an integer >= 0")
    if c.trunk_copy != "subset":
        named = [k for k in ("trunk_k", "trunk_select_step") if getattr(c, k) != getattr(d, k)]
        if named:
            raise ValueError(f"model.cp_head.{named[0]} without model.cp_head.trunk_copy 'subset': the key would be ignored")
    if not trunk_on(c):
        return
    if not c.enabled:
        raise ValueError("model.cp_head.trunk_copy without model.cp_head.enabled: the key would be ignored")
    if c.kernel not in TRUNK_KERNELS:
        raise ValueError(f"model.cp_head.kernel {c.kernel!r} with model.cp_head.trunk_copy: one of {TRUNK_KERNELS}")
    for key in NOT_READ:
        if getattr(c, key) != getattr(d, key):
            raise ValueError(f"model.cp_head.{key} {getattr(c, key)!r} is not built together with model.cp_head.trunk_copy "
                             f"{c.trunk_copy!r}: the copy distribution is the trunk heads' own (no pointer q/k, no sink key, the "
                             "reference path), so the key would be ignored or is not built")
    if c.train_mode != "mixture":
        raise ValueError(f"model.cp_head.train_mode {c.train_mode!r} is not built together with model.cp_head.trunk_copy: the trunk "
                         "copy is trained as CP's mixture")
    e = cfg.encoder
    if e.attn_res != "block_entry":
        raise ValueError(f"model.cp_head.trunk_copy is built on R1-entry (model.encoder.attn_res 'block_entry') only, not on "
                         f"{e.attn_res!r}: the 60M runs use R1-entry; nothing tests another trunk")
    if e.gdn_write_mix != "none" or e.gdn_ccr_spacing != 0:
        raise ValueError("model.cp_head.trunk_copy is not built together with the GDN-side arms (model.encoder.gdn_write_mix / "
                         "gdn_ccr_spacing): nothing tests the pair")
    glob = e.global_layers()
    if not glob:
        raise ValueError("model.cp_head.trunk_copy needs global attention layers in the encoder stack: there are none")
    atts = e.global_attention()
    hq = {a.n_q_heads for a in atts}
    if len(hq) != 1:
        raise ValueError(f"model.cp_head.trunk_copy: the global layers have different query head counts {sorted(hq)}")
    for i, a in zip(glob, atts):
        if a.normalizer != "softmax":
            raise ValueError(f"model.cp_head.trunk_copy reads softmax attention; layer {i} has normalizer {a.normalizer!r}")
        from ced.model.attention import FLASH_BACKENDS

        if a.backend in FLASH_BACKENDS:
            raise ValueError(f"model.cp_head.trunk_copy is not built together with attention backend {a.backend!r} (layer {i}): its "
                             "layers' queries are handed over on the flex, sdpa and reference paths only")
    H = len(glob) * hq.pop()
    if c.trunk_copy == "subset":
        n_q = atts[0].n_q_heads
        if not 1 <= c.trunk_k < H or c.trunk_k % n_q:
            raise ValueError(f"model.cp_head.trunk_k {c.trunk_k!r}: the subset's heads, a multiple of the global layers' {n_q} query "
                             f"heads (the reference path's group) and fewer than the {H} global heads")
        if c.trunk_select_step < 1:
            raise ValueError(f"model.cp_head.trunk_select_step {c.trunk_select_step!r}: the optimizer step at whose start the subset "
                             "is selected from the learned weights, an integer >= 1")


def n_trunk_heads(cfg: ModelConfig) -> int:
    """H_total: the global layers' query heads, layer by layer bottom first (global head id = layer order x Hq + h)."""
    e = cfg.encoder
    return len(e.global_layers()) * e.global_attention()[0].n_q_heads


# ---------------------------------------------------------------------------------------------------------------------
# the forward's inputs


@dataclass
class TrunkLayer:
    """One global layer's part of a forward: its query (B, Lq, Hq, d) after QK-norm, RoPE and the temperature (the attention's own), its
    far query under the rotary cap, its key set's keys (B, S, Hkv, d) (k_far: the unrotated keys under the cap), the cap, the scale. q is
    None when the layer's attention was taken out of this forward (its heads abstain). lse (B, Hq, Lq) float32: the attention's own
    log-sum-exp of this forward's read (the flex backend under model.cp_head.kernel "fused" or "fused2": KeySet.tc_lse), else None."""

    stack: int
    order: int
    q: torch.Tensor | None
    q_far: torch.Tensor | None
    k: torch.Tensor | None
    k_far: torch.Tensor | None
    cap: int | None
    scale: float
    lse: torch.Tensor | None = None


@dataclass
class TrunkInputs:
    """What the copy reads in one forward (ModelOutput.cp_pointer under model.cp_head.trunk_copy). The queries are the forward's positions
    (q_idx, q_doc (B, Lq)); the keys every position the global layers' key sets hold (this forward's, after a cache's): k_idx, k_doc,
    k_tok (their input tokens), k_next (the token each key copies, x_{j+1}; -1 where it is not known yet), k_start (bool: the key is
    its document's first position). own: the keys are the queries' own positions (no cache)."""

    layers: list
    q_idx: torch.Tensor
    q_doc: torch.Tensor
    k_idx: torch.Tensor
    k_doc: torch.Tensor
    k_tok: torch.Tensor
    k_next: torch.Tensor
    k_start: torch.Tensor
    own: bool
    n_q: int


@dataclass
class TrunkCache:
    """The copy cache of a cached sequence under the trunk copy: every earlier position's input token and document-start flag (B, S).
    The keys are the global layers' own cache entries."""

    tok: torch.Tensor | None = None
    start: torch.Tensor | None = None


def next_tokens(tok: torch.Tensor) -> torch.Tensor:
    """The token each key position copies under the shift +1: x_{j+1} = tok[:, j + 1]; -1 at the last position (not known yet)."""
    return torch.cat([tok[:, 1:], torch.full_like(tok[:, :1], -1)], dim=1)


def trunk_inputs(model, ctx, caches: list, tokens: torch.Tensor, copy_cache: TrunkCache | None) -> tuple[TrunkInputs, list]:
    """The forward's TrunkInputs and its layers' caches without the handed-over queries (HybridLM.forward, after the stack)."""
    enc = model.encoder
    glob = model.cfg.encoder.global_layers()
    out = list(caches)
    layers = []
    meta = None
    n_q = None
    for order, i in enumerate(glob):
        mod = enc.layers[i].mixer
        n_q = mod.cfg.n_q_heads
        kc = caches[i]
        scale = mod.cfg.head_dim ** -0.5
        if kc is None:
            layers.append(TrunkLayer(i, order, None, None, None, None, None, scale))
            continue
        assert kc.tc_q is not None, f"global layer {i} handed over no query (GatedAttention.trunk_copy unset?)"
        cap = mod.rope_cap()
        assert (cap is None) == (kc.tc_q_far is None), f"global layer {i}: the far query and the cap disagree"
        layers.append(TrunkLayer(i, order, kc.tc_q, kc.tc_q_far, kc.k, kc.k_far, cap, scale, kc.tc_lse))
        meta = (kc.idx, kc.doc)
        out[i] = dataclasses.replace(kc, tc_q=None, tc_q_far=None, tc_lse=None)
    start_now = ctx.pos == 0
    if copy_cache is None or copy_cache.tok is None:
        tok, start = tokens, start_now
    else:
        tok, start = torch.cat([copy_cache.tok, tokens], 1), torch.cat([copy_cache.start, start_now], 1)
    own = copy_cache is None or copy_cache.tok is None
    if meta is None:  # every global layer taken out of this forward: the keys are its own positions
        assert own, "a cached forward with every global layer taken out"
        meta = (ctx.idx, ctx.doc)
    k_idx, k_doc = meta
    assert k_idx.shape == tok.shape, f"the key set holds {tuple(k_idx.shape)} positions, the copy cache and the chunk {tuple(tok.shape)}"
    inp = TrunkInputs(layers=layers, q_idx=ctx.idx, q_doc=ctx.doc, k_idx=k_idx, k_doc=k_doc, k_tok=tok, k_next=next_tokens(tok),
                      k_start=start, own=own, n_q=n_q)
    return inp, out


# ---------------------------------------------------------------------------------------------------------------------
# the groups of heads


@dataclass
class Group:
    """Heads of one call of the block function: ids (global head ids, ascending), qn / qf (B, G, Lq, d) their queries (qf: the far
    rotation, None for a group without the cap), kn / kf (B, G, S, d) their own key heads' keys, far_on (G,) bool (None: no cap), cap,
    scale. caps: the heads' caps as Python values (None for a head without one: the fused path's, read without a device sync); lse
    (B, G, Lq): the heads' attention log-sum-exp when every head's layer handed it over, else None (the fused path's offset)."""

    ids: list
    qn: torch.Tensor
    qf: torch.Tensor | None
    kn: torch.Tensor
    kf: torch.Tensor | None
    far_on: torch.Tensor | None
    cap: int
    scale: float
    caps: list = dataclasses.field(default_factory=list)
    lse: torch.Tensor | None = None


def _heads_of(t: torch.Tensor, n_q: int) -> torch.Tensor:
    """(B, L, H, d) -> (B, n_q, L, d) contiguous, a key head repeated for its query heads (GQA: query head h reads key head h // group)."""
    H = t.shape[2]
    if H != n_q:
        t = t.repeat_interleave(n_q // H, dim=2)
    return t.permute(0, 2, 1, 3).contiguous()


def groups_of(head, inp: TrunkInputs) -> list:
    """The groups of the heads in use (module docstring, "THE REFERENCE PATH"): one a present layer while every head is in use, else the
    selected heads in groups of Hq. The heads of an absent layer are in no group (they abstain)."""
    active = head.active_heads()
    n_q = inp.n_q
    present = [ly for ly in inp.layers if ly.q is not None]
    if len(active) == head.n_trunk:
        out = []
        for ly in present:
            ids = list(range(ly.order * n_q, (ly.order + 1) * n_q))
            far = ly.cap is not None
            out.append(Group(ids, _heads_of(ly.q, n_q), _heads_of(ly.q_far, n_q) if far else None, _heads_of(ly.k, n_q),
                             _heads_of(ly.k_far, n_q) if far else None,
                             torch.ones(n_q, dtype=torch.bool, device=ly.q.device) if far else None, ly.cap or 0, ly.scale,
                             caps=[ly.cap] * n_q, lse=ly.lse))
        return out
    by_order = {ly.order: ly for ly in present}
    have = [h for h in active if h // n_q in by_order]
    out = []
    for g0 in range(0, len(have), n_q):
        ids = have[g0 : g0 + n_q]
        lys = [by_order[h // n_q] for h in ids]
        caps = {ly.cap for ly in lys if ly.cap is not None}
        assert len(caps) <= 1, f"the selected heads' layers have different rotary caps {caps}"
        far = bool(caps)

        def pick(t_of, ly, h, kv: bool):
            t = t_of(ly)
            j = (h % n_q) // (n_q // t.shape[2]) if kv else h % n_q
            return t[:, :, j]

        qn = torch.stack([pick(lambda ly: ly.q, ly, h, False) for ly, h in zip(lys, ids)], 1).contiguous()
        kn = torch.stack([pick(lambda ly: ly.k, ly, h, True) for ly, h in zip(lys, ids)], 1).contiguous()
        qf = kf = far_on = None
        if far:
            qf = torch.stack([pick(lambda ly: ly.q_far if ly.cap is not None else ly.q, ly, h, False) for ly, h in zip(lys, ids)],
                             1).contiguous()
            kf = torch.stack([pick(lambda ly: ly.k_far if ly.cap is not None else ly.k, ly, h, True) for ly, h in zip(lys, ids)],
                             1).contiguous()
            far_on = torch.tensor([ly.cap is not None for ly in lys], dtype=torch.bool, device=qn.device)
        scales = {ly.scale for ly in lys}
        assert len(scales) == 1, scales
        lse = None
        if all(ly.lse is not None for ly in lys):
            lse = torch.stack([ly.lse[:, h % n_q] for ly, h in zip(lys, ids)], 1).contiguous()
        out.append(Group(ids, qn, qf, kn, kf, far_on, caps.pop() if far else 0, scales.pop(), caps=[ly.cap for ly in lys], lse=lse))
    return out


@dataclass
class Group2:
    """Heads of one call of the fast path (model.cp_head.kernel "fused2"; ced/kernels/cp_trunk_fused2.py): ids (global head ids,
    ascending), q / qf (B, L, H, d) their queries in the layer's own layout (qf: the far rotation, None for a group without the cap),
    k / kf (B, L, KV, d) their key heads (query head h reads key head h // (H / KV)), caps (KV,) a key head's cap (None: none), scale,
    lse (B, H, L) the heads' attention log-sum-exp when every head's layer handed it over, else None."""

    ids: list
    q: torch.Tensor
    qf: torch.Tensor | None
    k: torch.Tensor
    kf: torch.Tensor | None
    caps: list
    scale: float
    lse: torch.Tensor | None = None


def groups2_of(head, inp: TrunkInputs) -> list:
    """The fast path's groups (groups_of's heads in groups_of's order): while every head is in use, one a present layer with the
    layer's own q (B, L, Hq, d) and k (B, L, Hkv, d), not copied; after a subset's selection the selected heads stacked in groups of
    Hq on the head dimension (B, L, Hq, d), each with its own key head (one key head a query head)."""
    active = head.active_heads()
    n_q = inp.n_q
    present = [ly for ly in inp.layers if ly.q is not None]
    if len(active) == head.n_trunk:
        return [Group2(list(range(ly.order * n_q, (ly.order + 1) * n_q)), ly.q, ly.q_far, ly.k, ly.k_far, [ly.cap] * ly.k.shape[2],
                       ly.scale, ly.lse) for ly in present]
    by_order = {ly.order: ly for ly in present}
    have = [h for h in active if h // n_q in by_order]
    out = []
    for g0 in range(0, len(have), n_q):
        ids = have[g0 : g0 + n_q]
        lys = [by_order[h // n_q] for h in ids]
        far = any(ly.cap is not None for ly in lys)

        def kv_of(ly, h):
            return (h % n_q) // (n_q // ly.k.shape[2])

        q = torch.stack([ly.q[:, :, h % n_q] for ly, h in zip(lys, ids)], 2)
        k = torch.stack([ly.k[:, :, kv_of(ly, h)] for ly, h in zip(lys, ids)], 2)
        qf = kf = None
        if far:
            qf = torch.stack([(ly.q_far if ly.cap is not None else ly.q)[:, :, h % n_q] for ly, h in zip(lys, ids)], 2)
            kf = torch.stack([(ly.k_far if ly.cap is not None else ly.k)[:, :, kv_of(ly, h)] for ly, h in zip(lys, ids)], 2)
        scales = {ly.scale for ly in lys}
        assert len(scales) == 1, scales
        lse = None
        if all(ly.lse is not None for ly in lys):
            lse = torch.stack([ly.lse[:, h % n_q] for ly, h in zip(lys, ids)], 1)
        out.append(Group2(ids, q, qf, k, kf, [ly.cap for ly in lys], scales.pop(), lse))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the block function (the trainer compiles it on CUDA)


def _safe_log(x: torch.Tensor) -> torch.Tensor:
    """log x for x >= 0, -inf at 0 with a gradient of exactly 0 there (never inf x 0)."""
    pos = x > 0
    return torch.where(pos, torch.log(torch.where(pos, x, torch.ones((), dtype=x.dtype, device=x.device))),
                       torch.full((), float("-inf"), dtype=x.dtype, device=x.device))


def _masks(q_idx, q_doc, k_idx, k_doc, k_start, k_next, q_tgt):
    """(ctx, valid, match), each (B, C, S) bool: ctx the document's keys j <= t (the head's own softmax), valid the copy sources (j < t
    strictly, not the document's first position), match the valid keys whose copied token is the target."""
    ctx = (k_idx[:, None, :] <= q_idx[:, :, None]) & (k_doc[:, None, :] == q_doc[:, :, None])
    valid = ctx & (k_idx[:, None, :] < q_idx[:, :, None]) & ~k_start[:, None, :]
    match = valid & (k_next[:, None, :] == q_tgt[:, :, None])
    return ctx, valid, match


def head_scores(qn, qf, kn, kf, far_on, cap: int, q_idx, k_idx, scale: float):
    """The heads' scores (B, G, C, S) in the accumulation dtype (float32, or float64 for float64 inputs): q . k x scale, and under the cap
    q_far . k_far x scale for a key more than cap positions before the query on a head with far_on (attention.attend_capped's rule)."""
    dt = torch.float64 if qn.dtype == torch.float64 else torch.float32
    s = torch.matmul(qn.to(dt), kn.to(dt).transpose(-1, -2)) * scale
    if qf is not None:
        sf = torch.matmul(qf.to(dt), kf.to(dt).transpose(-1, -2)) * scale
        far = (q_idx[:, :, None] - k_idx[:, None, :] > cap)[:, None] & far_on[None, :, None, None]
        s = torch.where(far, sf, s)
    return s


def trunk_block(qn, qf, kn, kf, far_on, cap: int, q_idx, q_doc, k_idx, k_doc, k_start, k_next, q_tgt, scale: float):
    """(log s_h, log(1 - s_h), log M_h(target)), each (B, C, G), for one group's heads at C queries over S keys (module docstring, "PER
    HEAD"). Fixed shapes; masks by comparisons and torch.where only."""
    s = head_scores(qn, qf, kn, kf, far_on, cap, q_idx, k_idx, scale)
    ctx, valid, match = _masks(q_idx, q_doc, k_idx, k_doc, k_start, k_next, q_tgt)
    neg = torch.full((), float("-inf"), dtype=s.dtype, device=s.device)
    zero = torch.zeros((), dtype=s.dtype, device=s.device)
    c4 = ctx[:, None]
    m = torch.where(c4, s, neg).amax(-1, keepdim=True).detach()  # finite: the diagonal is always in ctx
    e = torch.exp(torch.where(c4, s - m, neg))
    s_nc = torch.where(c4 & ~valid[:, None], e, zero).sum(-1)
    s_v = torch.where(valid[:, None], e, zero).sum(-1)
    s_m = torch.where(match[:, None], e, zero).sum(-1)
    log_all = torch.log(s_nc + s_v)  # >= 0: the row's largest term is 1
    out = (_safe_log(s_nc) - log_all, _safe_log(s_v) - log_all, _safe_log(s_m) - log_all)
    return tuple(x.transpose(1, 2) for x in out)


# ---------------------------------------------------------------------------------------------------------------------
# the mixture over heads


def safe_lse(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """logsumexp over dim, -inf where every entry is -inf, with a gradient of exactly 0 there (a dummy 0 under torch.where)."""
    some = torch.isfinite(x).any(dim)
    y = torch.logsumexp(torch.where(some.unsqueeze(dim), x, torch.zeros((), dtype=x.dtype, device=x.device)), dim)
    return torch.where(some, y, torch.full((), float("-inf"), dtype=x.dtype, device=x.device))


def combine(log_w: torch.Tensor, ls: torch.Tensor, l1: torch.Tensor, lm: torch.Tensor):
    """(log s, log(1 - s), log p_CP(y)) (...) from the heads' own (..., H) and the weights' log w (H,) (module docstring, "MIXED")."""
    lw = log_w.to(ls.dtype)
    log_s, log_1ms, log_m = safe_lse(lw + ls), safe_lse(lw + l1), safe_lse(lw + lm)
    has = torch.isfinite(log_m)
    zero = torch.zeros((), dtype=ls.dtype, device=ls.device)
    log_pcp = torch.where(has, torch.where(has, log_m, zero) - torch.where(has, log_1ms, zero),
                          torch.full((), float("-inf"), dtype=ls.dtype, device=ls.device))
    return log_s, log_1ms, log_pcp


def _abstain(B: int, L: int, n: int, like: torch.Tensor):
    """An absent head's (log s_h, log(1 - s_h), log M_h): (0, -inf, -inf), (B, L, n)."""
    z = torch.zeros(B, L, n, dtype=like.dtype, device=like.device)
    return z, z - float("inf"), z - float("inf")


def trunk_terms(head, inp: TrunkInputs, targets: torch.Tensor, chunk: int | None = None, checkpoint: bool = False, fn=None):
    """(log s, log(1 - s), log p_CP(target)), each (B, Lq), at every query of a forward (module docstring). targets (B, Lq): the queries'
    targets. chunk: query positions a block (TRUNK_CHUNK). checkpoint: each group's block under activation checkpointing (the training
    forward). fn: the block function (trunk_block; the trainer passes a torch.compile'd one on CUDA).
    Under model.cp_head.kernel "fused" with the forward's own keys: each group's heads in one pass over the row (ced/kernels/
    cp_trunk_fused.py group_terms, forward and backward; chunk, checkpoint and fn are not read), mixed as here. Under "fused2" with
    the forward's own keys: the fast path (ced/kernels/cp_trunk_fused2.py group_terms on groups2_of's groups, the integer inputs built
    once for every group), mixed as here (the groups' columns concatenated when they are the heads in use in order)."""
    B, Lq = inp.q_idx.shape
    if fused2_on(head) and inp.own:
        from ced.kernels.cp_trunk_fused2 import Meta2
        from ced.kernels.cp_trunk_fused2 import group_terms as group_terms2

        meta = Meta2(inp.k_start, inp.k_next, inp.q_doc, targets)
        groups = groups2_of(head, inp)
        per = {id(g): list(group_terms2(g.q, g.qf, g.k, g.kf, g.caps, g.scale, meta, g.lse)) for g in groups}
        return mix_heads(head, inp, groups, per, B, Lq, cat=True)
    if fused_on(head) and inp.own:
        from ced.kernels.cp_trunk_fused import group_terms

        groups = groups_of(head, inp)
        per = {id(g): list(group_terms(g.qn, g.qf, g.kn, g.kf, g.caps, g.scale, inp.k_start, inp.k_next, inp.q_doc, targets, g.lse))
               for g in groups}
        return mix_heads(head, inp, groups, per, B, Lq)
    S_all = inp.k_idx.shape[1]
    fn = trunk_block if fn is None else fn
    C = chunk or TRUNK_CHUNK
    if chunk is None and not torch.is_grad_enabled():  # a scorer at a long row: blocks of at most EVAL_ELEMENTS score elements
        C = max(1, min(TRUNK_CHUNK, EVAL_ELEMENTS // max(1, B * inp.n_q * S_all)))
    groups = groups_of(head, inp)
    parts = {id(g): [] for g in groups}
    for a in range(0, Lq, C):
        S = min(a + C, Lq) if inp.own else S_all
        for g in groups:
            args = (g.qn[:, :, a : a + C], None if g.qf is None else g.qf[:, :, a : a + C], g.kn[:, :, :S],
                    None if g.kf is None else g.kf[:, :, :S], g.far_on, g.cap, inp.q_idx[:, a : a + C], inp.q_doc[:, a : a + C],
                    inp.k_idx[:, :S], inp.k_doc[:, :S], inp.k_start[:, :S], inp.k_next[:, :S], targets[:, a : a + C], g.scale)
            if checkpoint and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint as ckpt

                parts[id(g)].append(ckpt(fn, *args, use_reentrant=False))
            else:
                parts[id(g)].append(fn(*args))
    return mix_heads(head, inp, groups, {k: [torch.cat([p[i] for p in v], 1) for i in range(3)] for k, v in parts.items()}, B, Lq)


def mix_heads(head, inp: TrunkInputs, groups: list, per_group: dict, B: int, Lq: int, cat: bool = False):
    """combine() of the heads in use, in active order: a group's heads from its columns, an absent layer's heads abstaining. cat (the
    fast path): when the groups' ids in order are the heads in use, their (B, L, G) columns concatenated (the same values as the
    head-by-head stack below, in one op each way)."""
    active = head.active_heads()
    if cat and groups and [h for g in groups for h in g.ids] == list(active):
        ls, l1, lm = (torch.cat([per_group[id(g)][i] for g in groups], -1) for i in range(3))
        return combine(head.trunk_log_weights(), ls, l1, lm)
    like = None
    cols = {}
    for g in groups:
        ls, l1, lm = per_group[id(g)]
        like = ls
        for j, h in enumerate(g.ids):
            cols[h] = (ls[..., j], l1[..., j], lm[..., j])
    if like is None:  # every layer taken out
        like = torch.zeros(B, Lq, 1, dtype=torch.float64 if inp.k_idx.dtype == torch.float64 else torch.float32,
                           device=inp.q_idx.device)
    missing = [h for h in active if h not in cols]
    if missing:
        z, ninf, _ = _abstain(B, Lq, 1, like)
        for h in missing:
            cols[h] = (z[..., 0], ninf[..., 0], ninf[..., 0])
    ls, l1, lm = (torch.stack([cols[h][i] for h in active], -1) for i in range(3))
    return combine(head.trunk_log_weights(), ls, l1, lm)


def trunk_terms_naive(head, inp: TrunkInputs, targets: torch.Tensor):
    """The T x T reference of the tests: every head's full softmax over the document's keys j <= t (the attention's own), then the masses
    of the no-copy, valid and matching keys summed in probability space by their definitions (no block, no shared max), mixed by the
    weights: (log s, log(1 - s), log p_CP(target)) (B, Lq)."""
    B, Lq = inp.q_idx.shape
    groups = groups_of(head, inp)
    per = {}
    for g in groups:
        s = head_scores(g.qn, g.qf, g.kn, g.kf, g.far_on, g.cap, inp.q_idx, inp.k_idx, g.scale)
        ctx, valid, match = _masks(inp.q_idx, inp.q_doc, inp.k_idx, inp.k_doc, inp.k_start, inp.k_next, targets)
        p = torch.softmax(s.masked_fill(~ctx[:, None], float("-inf")), -1)
        zero = torch.zeros((), dtype=p.dtype)
        sums = [torch.where(m[:, None], p, zero).sum(-1).transpose(1, 2) for m in (ctx & ~valid, valid, match)]
        per[id(g)] = [_safe_log(x) for x in sums]
    return mix_heads(head, inp, groups, per, B, Lq)


# ---------------------------------------------------------------------------------------------------------------------
# scoring: the copy distribution over the keys and over V


def copy_rows(head, inp: TrunkInputs, rows: torch.Tensor, chunk: int | None = None):
    """The mixture's copy distribution at query rows `rows` (B, P) (indices into the forward's queries): yields per block of rows (a, n,
    mass (B, n, S) = sum_h w_h A_h(t, j) on the valid keys (0 elsewhere), log s (B, n), log(1 - s) (B, n)), in the accumulation dtype.
    mass summed over the keys is 1 - s; p_CP over V is mass scattered by k_next, over 1 - s."""
    B, P = rows.shape
    S = inp.k_idx.shape[1]
    groups = groups_of(head, inp)
    active = head.active_heads()
    log_w = head.trunk_log_weights()
    G = max(1, inp.n_q)
    C = chunk or max(1, min(TRUNK_CHUNK, EVAL_ELEMENTS // max(1, G * S * B)))
    for a in range(0, P, C):
        r = rows[:, a : a + C]
        n = r.shape[1]
        q_idx, q_doc = torch.gather(inp.q_idx, 1, r), torch.gather(inp.q_doc, 1, r)
        ctx, valid, _ = _masks(q_idx, q_doc, inp.k_idx, inp.k_doc, inp.k_start, inp.k_next, torch.zeros_like(q_idx))
        mass = None
        heads_s, heads_1ms = {}, {}
        for g in groups:
            gi = r[:, None, :, None].expand(B, g.qn.shape[1], n, g.qn.shape[-1])
            qn = torch.gather(g.qn, 2, gi)
            qf = None if g.qf is None else torch.gather(g.qf, 2, gi)
            s = head_scores(qn, qf, g.kn, g.kf, g.far_on, g.cap, q_idx, inp.k_idx, g.scale)
            logp = torch.log_softmax(s.masked_fill(~ctx[:, None], float("-inf")), -1)
            p = logp.exp()
            zero = torch.zeros((), dtype=p.dtype, device=p.device)
            pv = torch.where(valid[:, None], p, zero)
            s_v = _safe_log(pv.sum(-1))
            s_nc = _safe_log(torch.where(ctx[:, None] & ~valid[:, None], p, zero).sum(-1))
            w = torch.stack([log_w[active.index(h)] for h in g.ids]).to(p.dtype).exp()
            contrib = (pv * w[None, :, None, None]).sum(1)
            mass = contrib if mass is None else mass + contrib
            for j, h in enumerate(g.ids):
                heads_s[h], heads_1ms[h] = s_nc[:, j], s_v[:, j]
        if mass is None:
            dt = torch.float32
            mass = torch.zeros(B, n, S, dtype=dt, device=rows.device)
        z = torch.zeros(B, n, dtype=mass.dtype, device=mass.device)
        ls = torch.stack([heads_s.get(h, z) for h in active], -1)
        l1 = torch.stack([heads_1ms.get(h, z - float("inf")) for h in active], -1)
        lw = log_w.to(ls.dtype)
        yield a, n, mass, safe_lse(lw + ls), safe_lse(lw + l1)


def logprobs_rows(head, inp: TrunkInputs, hidden_q: torch.Tensor, rows: torch.Tensor, weight_v: torch.Tensor,
                  chunk: int | None = None) -> torch.Tensor:
    """log p over V of the mixture at query rows `rows` (B, P), hidden_q (B, P, D) the final norm's output there: (B, P, V) in the
    accumulation dtype (cp_head.logprobs_rows's form: p_CP over V by scatter_add of the copy mass into its copied tokens)."""
    from ced.model.cp_head import mixture_logprobs_v

    B, P, _ = hidden_q.shape
    V = weight_v.shape[0]
    out = []
    with torch.autocast(device_type=hidden_q.device.type, enabled=False):
        dtype = compute_dtype(hidden_q)
        W = weight_v.to(dtype)
        nxt = inp.k_next.clamp_min(0)
        for a, n, mass, log_s, log_1ms in copy_rows(head, inp, rows, chunk):
            h = hidden_q[:, a : a + n].to(dtype)
            mass = mass.to(dtype)
            m_v = torch.zeros(B, n, V, dtype=dtype, device=h.device)
            m_v.scatter_add_(2, nxt[:, None, :].expand(B, n, nxt.shape[1]), mass)
            one_ms = log_1ms.to(dtype).exp()
            pos = one_ms > 0
            p_cp = torch.where(pos[..., None], m_v / torch.where(pos, one_ms, torch.ones_like(one_ms))[..., None],
                               torch.zeros((), dtype=dtype, device=h.device))
            logits = torch.nn.functional.linear(h, W)
            z_c = head.copy_logit(h)
            lp = mixture_logprobs_v(logits.reshape(B * n, V), z_c.reshape(-1), log_s.to(dtype).reshape(-1),
                                    log_1ms.to(dtype).reshape(-1), p_cp.reshape(B * n, V))
            out.append(lp.reshape(B, n, V))
    return torch.cat(out, 1)


def forward_logits(model, hidden: torch.Tensor, tokens: torch.Tensor, inp: TrunkInputs):
    """cp_head.forward_logits under the trunk copy: log p over V at every position of the chunk (score "mixture") or the LM logits (score
    "lm"), and the copy cache after the chunk (TrunkCache)."""
    from ced.model.cp_head import score_mode

    inner = getattr(model, "_orig_mod", model)
    W = inner.lm_head_weight()
    if score_mode(inner) == "lm":
        from ced.model.looped_ced import lm_head

        logits = lm_head(hidden, W)
    else:
        B, L = tokens.shape
        rows = torch.arange(L, device=tokens.device).expand(B, L)
        logits = logprobs_rows(inner.cp_head, inp, hidden, rows, W)
    return logits, TrunkCache(tok=inp.k_tok, start=inp.k_start)


def head_weights(head) -> torch.Tensor:
    """The learned weights over every global head (H_total,), float64, detached: softmax over the heads in use, 0 for the others."""
    w = torch.zeros(head.n_trunk, dtype=torch.float64)
    act = head.active_heads()
    w[torch.tensor(act)] = head.trunk_log_weights().detach().double().exp().cpu()
    return w


RANK_EVERY = 100  # the trainer logs the heads' ranking (metrics.jsonl cp_trunk_rank) every RANK_EVERY optimizer steps (step 0 too)


def head_ranking(head) -> list:
    """The global heads ranked by their learned mixing logits, largest first, ties to the lower id: the subset selection's own order
    (CopyPasteHead.select_trunk_subset), so before the selection the ranking of the weights; after it the selected heads' order among
    themselves is their weights' (the others' logits no longer train)."""
    logits = head.trunk_logits.detach().double().cpu().tolist()
    return sorted(range(head.n_trunk), key=lambda h: (-logits[h], h))


@torch.no_grad()
def copy_alpha_rows(head, inp: TrunkInputs, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The copy distribution at query rows `rows` (B, P) in CP's convention (over the COPIED positions: alpha(tau) = mass(tau - 1) /
    (1 - s), the position whose token is copied; -inf at tau = 0): (log alpha (B, P, S) float64, log s (B, P) float64). The readers'
    hook (the per-head selection diagnostic)."""
    out_a, out_s = [], []
    for _a, _n, mass, log_s, log_1ms in copy_rows(head, inp, rows):
        m = mass.double()
        shifted = torch.cat([torch.zeros_like(m[..., :1]), m[..., :-1]], -1)
        la = _safe_log(shifted) - log_1ms.double()[..., None]
        out_a.append(torch.where(torch.isfinite(log_1ms.double())[..., None], la, torch.full_like(la, float("-inf"))))
        out_s.append(log_s.double())
    return torch.cat(out_a, 1), torch.cat(out_s, 1)
