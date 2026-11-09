"""The second Householder step on a subset of the tokens: a fixed stride and one step per chunk, built from the chunk's summary of the
lighter forms of Gated DeltaProduct (GDNConfig.dprod.stride / summary / conv / first_beta).

THE ASK. In the design's text the stride is s, because k is the key.

WHY A MODULE OF ITS OWN. Arm 6's code (ced/model/gdn_dprod.py) is what the screen's arm trained and the GPU values check passed on; the
design and split notes cite its lines. This module leaves its statements as they were: gdn_dprod.py ends with two appended wrappers
(`build` and `check_config` are rebound to functions that call `build_steps` and `check_model` at the end of this file, which call own
when the keys are at their defaults), and GatedDeltaNet.forward calls the function a layer's `dprod` names (`forward_fn`) in place of
dprod_forward. With the keys at their defaults (stride 1, summary "none", conv True) nothing here runs: the layer is the Gated
DeltaProduct form's, bit for bit (tests/test_gdn_dprod_sparse_off.py holds main's tree against this one).

THE RULE. Arm 6's recursion, with the second step on a subset T_c of the tokens:

    S_t = R(k2_t, v2_t, beta2_t) R(k1_t, v1_t, beta1_t) (a_t S_{t-1})     for t in T_c
    S_t =                        R(k1_t, v1_t, beta1_t) (a_t S_{t-1})     otherwise;       o_t = S_t^T q_t

R(k, v, beta) S = (I - beta k k^T) S + beta k v^T; the decay a_t = exp(g_t) once, before step 1; the read once, after the last update
the token carries. T_c = the tokens p with p mod s = s - 1 (p the position in the packed row: the cache's absolute index, ctx.idx),
for E and D alike: for D the carrying token is the last of each group of s tokens, and its second step is built from the group's
SUMMARY, not from its own x.
- E (summary "none"): k2 = unit(SiLU(W_k x_p)), v2 = SiLU(W_v x_p), beta2 = beta_max sigmoid(w_b. x_p), the carrying token's own x; the
  matrices are the Gated DeltaProduct form's `dprod.in_proj_kv` (d -> H_k d_k + H_v d_v) and `dprod.in_proj_b` (d -> H_v), run on the T/s
  carrying rows only, and there is NO short conv on those channels (conv False; a conv over the strided tokens would read tokens s apart,
  and over all T rows would cost the full T-row pass the stride avoids). s = 1 with conv False is the Gated DeltaProduct form without the
  conv; with the conv's taps set to the identity it is the Gated DeltaProduct form exactly (tests/test_gdn_dprod_sparse.py).
- D, "xmean": xbar_i = the mean of x over the tokens of group i that are in the carrying (last) token's document; then the same
  matrices and nonlinearities as E, on xbar_i. "kvmean" (no matrix, parameter-free): k2 = unit(mean of k1), v2 = mean of v1 over the same
  tokens, beta2 = beta_max sigmoid(c), c = `dprod.beta_const` (H_v,), zeros (beta2 = beta_max / 2 at the start, a learned constant a head).
- Step 1 on every token is the layer's own (in_proj_qkv's q, k, v after the conv, in_proj_b's beta). Its beta is beta_max sigmoid(b) on
  a token that carries the second step; on a token that does not it is sigmoid(b) (first_beta "carry", the default: that token is
  R1-entry's token exactly, the design's "a head that does not carry it is R1-entry's head exactly") or beta_max sigmoid(b) (first_beta
  "all": step 1 on every token).
- The decay is on the first row only, so the second update is not decayed.
- beta_max is `dprod.beta_max` (the dprodb1 build's key, default 2); the 60M arms of this build keep 2.

CAUSALITY (what a token reads, when an update lands). A token reads S_t: every update of the tokens up to t, and its own second update
iff t is in T_c. E: the second update of token p is a function of x_p, lands between p's first update and p's read, and is in the
state of every later token; the s - 1 tokens before p in its group do not see it. D: the group's update is a function of the group's
tokens up to its last token (the mean never reads a later token), lands between the last token's first update and its read, and no token
of the group before the last sees it. A document start resets the state at the first row of the document: an update never crosses a
document boundary. A group whose tokens belong to several documents summarises only the last token's document (the cache's running sums
reset at a document start for the same reason). No index op and no data-dependent shape: the carrying positions are Python integers
(the packed-row phase), the masks are comparisons.

THE ROW LAYOUT. The kernels see one row per token and one extra row after each carrying token: a carrying token has the rows (k1, v1,
beta1, g, q = 0) then (k2, v2, beta2, g = 0, q); every other token has the one row (k1, v1, beta1, g, q). Rows: L + (the carrying tokens
in the call). A document start is on the first row of the token only. The token's output is read from its last row (take_tokens).
`insert_rows` builds the layout from the stacks in the token layout by views (the token axis is cut into groups of s, the extra rows are
concatenated after each group and the result flattened: one concatenation per group shape, no gather); `rows_cu` maps the varlen
boundaries (device and host copies) to rows.

THE PATHS (each the same function; `sparse_recurrence`, a per-token loop written from the rule above and independent of the layout,
is the ground truth):
- the loop (GDNConfig.kernel "recurrence", and a call of one token);
- the chunked rule on the row layout (kernels "chunked", "fla", "ced" off the fused path);
- the fused path (CUDA, kernel "ced" or "fla", from an empty state with varlen boundaries: the training forward and the scoring
  forward): the layer's own prep (the fused qkv_prep), the extras formed from the carrying tokens, the row layout into the kernel as it
  is (grouped heads kept), the boundaries mapped by rows_cu. Today's 'ced' kernel computes the readout and the u products on every row,
  so the extra rows cost what a token costs (the design's "interleave route"); a rows-pattern kernel mode that skips them is the
  design's kernel item (3) and is NOT built here. K2 (kda.kernel "k2") is refused by name for the same reason.

THE COPIES (GDNConfig.dprod.sparse_path). NOT SET (the default, None): the variant "xmean" runs "fast", the variant "kvmean" runs "fast" (its GPU
values check passed on a GPU; the first order, had it on "rows", which for kvmean is "fast": its preparation is not written in
the rows), every other form runs "plain"; a key that is set is kept as it is (kvmean set to "fast" or "rows" runs the fast copies, as built). "plain"
(set): insert_rows / take_tokens below, as built. "fast": the same function with the copies at their fewest passes
(ced/model/gdn_dprod_sparse_fast.py: insert_rows_fast / take_tokens_fast, one split and one cat each), values and gradients bit for bit the plain
path's. "rows": "fast", and on the fused path the input preparation itself writes q, k, v in the row layout and reads their gradients there
(ced/kernels/gdn_prep_rows.py), for whole groups (L a multiple of the stride) and not for D "kvmean"; otherwise "fast". The three are one function:
the bits are the same (tests/test_gdn_dprod_sparse_fast.py); only the number of passes over q, k, v and o differs.

THE ROUTE (note). GDNConfig.dprod.route "one-readout" (main's key, ced/kernels/gdn_1r.py) is read by E (summary "none") on the fused path: the kernel
is given the Gated DeltaProduct form's two rows a token, the second step's k2, v2, beta2 written into (B, L...) tensors at the carrying positions and
ZERO elsewhere (beta2 = 0 there: R(k, v, 0) is the identity, so the function is E's), the stacks [k1, k2], [v1, v2], [beta1, beta2] of the layer's own
layout, q and g at the token rows, one call of gdn_dprod._one_readout_call. This is the 1r note's form for E: the kernel computes the second row of
every token, so E's saving in the chunk kernel (the inserted-row layout's (1 + 1/s) T rows) is NOT kept on this route; only the matrices, the prep and
the derived passes (on the n_c rows that carry the step) are. A rows-pattern mode of the one-readout kernel would keep both; it is kernel work and not
built. The default "interleave" is the inserted-row layout above, bit for bit. D (a summary) is refused with the route by name (check_keys).

THE CACHE. E: nothing extra (the phase is the absolute index of the first token of the call). D: `SparseState` (GDNState with `acc`
(B, F) and `cnt` (B,): the sums of the summary's features and their count over the current, not yet closed, group, restarted at a group
start and at a document start). The layer's conv history is the plain layer's (the extra channels have no conv). A call that continues a
group adds the carried sums iff the group's last token is in the same document as the carried tokens.

INITIALISATION. The same as the Gated DeltaProduct form's (gdn_dprod.py): the extra tensors are drawn from the run's own generator
(init_own_parameters), in_proj_kv then in_proj_b, N(0, init_std); `beta_const` zeros (nothing drawn). The first draws of a stride arm and
the Gated DeltaProduct form of the same seed are the same two tensors. Optimiser groups by name (ced/train/optim.py):
dprod.in_proj_kv.weight a hidden matrix (Muon under "muon"), dprod.in_proj_b.weight a gate projection (AdamW), dprod.beta_const the
zero-initialised constant (ZERO_GATE_SUFFIXES: no weight decay under optim.zero_gate_decay "none", the standing rule for a new arm).

PARAMETERS (exact): E and D "xmean": d (H_k d_k + H_v d_v) + d H_v a layer: 395,264 at 60M (5,928,960 over 15 GDN layers), 9,881,600 at 4B;
D "kvmean": H_v a layer (60 at 60M). Arm 6's conv weight is not there.

NOT BUILT (refused by name): the decoder's GDN layers, n_h other than 2, the placement keys (dprod.layers /heads / heads_by_layer) together with these
keys, the second-step construction keys (a derived dprod.key, or dprod.value / beta / rank / perm off their defaults), dprod.beta_init (checked in
check_keys), dprod.route "one-readout" with a summary (E is served, see THE ROUTE), K2, edges other than "plain", chunk_backend other than "triton",
and everything the Gated DeltaProduct form refuses (the GDN-side arms, padding, keep, a TST bag phase, CUDA graphs' fixed-capacity metadata, the MTP
drafter, FP8 / NVFP4 linears: gdn_dprod._refuse_call and check_config). Python 3.11 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ced.config import GDNConfig
from ced.model import gdn as G
from ced.model import gdn_dprod as D
from ced.model import gdn_dprod_sparse_fast as FP
from ced.model.gdn_chunked import chunked_delta_rule

SUMMARIES = ("none", "xmean", "kvmean")
FIRST_BETAS = ("carry", "all")
DEFAULTS = {"stride": 1, "summary": "none", "conv": True, "first_beta": "carry"}
PATH_DEFAULT = "plain"  # GDNConfig.dprod.sparse_path (ced/model/gdn_dprod_sparse_fast.py) not set, every form but the variant: the row layout's copies as built; "fast": the same values, fewest passes
PATH_DEFAULT_D = "fast"
PATH_DEFAULT_KV = "fast"  # not set, variant "kvmean": "fast" (its GPU values check passed: the fast and rows layouts equal "plain" bit for bit, rel max 0.000e+00)
PATH_UNMOVED = (None, PATH_DEFAULT)  # the key as set that changes nothing against main: not set, or "plain" (the path of every run before the key; accepted wherever the key is refused with another value)
PLACEMENT = {"layers": (), "heads": 0, "heads_by_layer": {}}  # these keys are not built together with the sparse variants


# --------------------------------------------------------------------------------------------------------------------------
# the keys


def keys_of(cfg: GDNConfig) -> tuple:
    """(stride, summary, conv, first_beta) of GDNConfig.dprod, checked: stride an int >= 1 (not a bool), summary in SUMMARIES, conv a
    bool, first_beta in FIRST_BETAS; ValueError naming the key. A section without a `dprod` attribute reads as the defaults."""
    d = getattr(cfg, "dprod", None)
    s, summary, conv, fb = (getattr(d, k, v) for k, v in DEFAULTS.items())
    if isinstance(s, bool) or not isinstance(s, int) or s < 1:
        raise ValueError(f"GDNConfig.dprod.stride {s!r}: an integer >= 1 (1: the second step on every token, the Gated DeltaProduct form; the strides: 2, 4, 8, 16, 64)")
    if summary not in SUMMARIES:
        raise ValueError(f"GDNConfig.dprod.summary {summary!r}: one of {SUMMARIES}")
    if not isinstance(conv, bool):
        raise ValueError(f"GDNConfig.dprod.conv {conv!r}: a bool")
    if fb not in FIRST_BETAS:
        raise ValueError(f"GDNConfig.dprod.first_beta {fb!r}: one of {FIRST_BETAS}")
    return s, summary, conv, fb


def path_set(cfg: GDNConfig) -> str | None:
    """GDNConfig.dprod.sparse_path as it is set, checked: None (not set, also a section without the key) or one of FP.PATHS; ValueError naming the
    key. The refusals of the key read this (an explicit "plain" is accepted where the key is refused with another value; not set is not "set")."""
    p = getattr(getattr(cfg, "dprod", None), "sparse_path", None)
    if p is not None and p not in FP.PATHS:
        raise ValueError(f"GDNConfig.dprod.sparse_path {p!r}: None (not set) or one of {FP.PATHS} ('plain': the row layout's copies as built; 'fast': the same "
                         "values and gradients with the copies at their fewest passes, ced/model/gdn_dprod_sparse_fast.py)")
    return p


def path_of(cfg: GDNConfig) -> str:
    """The path the layer runs, one of FP.PATHS: the key as set (path_set), and when it is not set "fast" for the variant "xmean" (PATH_DEFAULT_D) and
    for the variant "kvmean" (PATH_DEFAULT_KV; its GPU values check passed), and "plain" for every other form
    (PATH_DEFAULT: E, a derived key, n_h 1). kvmean set to "fast" or "rows" runs the fast copies (its preparation is not written in the rows)."""
    p = path_set(cfg)
    if p is not None:
        return p
    summary = keys_of(cfg)[1] if sparse_on(cfg) else "none"
    return {"xmean": PATH_DEFAULT_D, "kvmean": PATH_DEFAULT_KV}.get(summary, PATH_DEFAULT)


def derived_key(cfg: GDNConfig) -> bool:
    """True iff dprod.key (ced/model/gdn_dprod_c.py) names a second step made from the first (anything but "proj"): that layer is
    gdn_dprod_c's (its conv is False too), never this module's."""
    return getattr(getattr(cfg, "dprod", None), "key", "proj") != "proj"


def sparse_on(cfg: GDNConfig) -> bool:
    """True iff the layer runs this module: stride other than 1, a summary, or conv False, with key proj (a derived key is
    gdn_dprod_c's). All defaults: the Gated DeltaProduct form's layer (or no DeltaProduct at all). first_beta alone changes nothing
    (at stride 1 every token carries)."""
    s, summary, conv, _ = keys_of(cfg)
    if derived_key(cfg):
        return False
    return s != 1 or summary != "none" or conv is False


def check_keys(cfg: GDNConfig, n: int) -> None:
    """The key-level refusals of one GDN section (n = dprod.n_h): a non-default key with n_h 1 (it would change nothing); n_h other
    than 2; conv True with a stride above 1 or a summary; the placement keys with these keys; K2."""
    s, summary, conv, fb = keys_of(cfg)
    path = path_set(cfg)  # as set (None: not set): the variant's "fast" when not set is the default, not a setting to refuse
    used = [k for k, v in zip(DEFAULTS, (s, summary, conv, fb)) if v != DEFAULTS[k]]
    if path not in PATH_UNMOVED:
        used.append("sparse_path")
    if n == 1:
        c_keys = ("key", "value", "beta", "rank", "perm")  # the keys: when one is set too, its check (gdn_dprod_c.check_second) names every second-step key
        if used and not any(getattr(cfg.dprod, k, v) != v for k, v in zip(c_keys, ("proj", "proj", "proj", 16, "roll"))):
            raise ValueError(f"GDNConfig.dprod.{', '.join(used)} set with dprod.n_h 1: these keys place the SECOND Householder step and are "
                             "read only with n_h 2 (Gated DeltaNet, n_h 1, has none; at n_h 1 they would change nothing)")
        return
    if derived_key(cfg):  # a second step made from the first: gdn_dprod_c's layer; its keys are checked there
        moved = [k for k, v in (("stride", s), ("summary", summary), ("first_beta", fb)) if v != DEFAULTS[k]]
        if path not in PATH_UNMOVED:
            moved.append("sparse_path")
        if moved:
            raise ValueError(f"GDNConfig.dprod.{', '.join(moved)} with dprod.key {cfg.dprod.key!r}: the strided and summarised second step (this "
                             "module) builds the second step from the carrying token's x or the group's summary; a derived key (ced/model/gdn_dprod_c.py) "
                             "builds it from the first step's key; not built together")
        return
    if not sparse_on(cfg):
        if path not in PATH_UNMOVED:
            raise ValueError(f"GDNConfig.dprod.sparse_path {path!r} with stride {s} / summary {summary!r} / conv {conv}: the keys are at their defaults, so the "
                             "layer is the plain Gated DeltaProduct form's (ced/model/gdn_dprod.py), which has no inserted-row layout of this module's; the key would change nothing")
        return
    if n != 2:
        raise ValueError(f"GDNConfig.dprod.n_h {n} with stride {s} / summary {summary!r} / conv {conv}: the strided and summarised forms are the "
                         "SECOND step on a subset of the tokens: n_h 2 only")
    if conv and (s > 1 or summary != "none"):
        raise ValueError(f"GDNConfig.dprod.conv True with stride {s} / summary {summary!r}: the design's strided and summarised arms have no short "
                         "conv on the second step's channels (a conv over the carrying tokens would read tokens s apart; over every token it would "
                         "cost the full-T pass the stride avoids): set dprod.conv=false")
    d = cfg.dprod
    stray = [k for k, v in PLACEMENT.items() if getattr(d, k, v) != v]
    if stray:
        raise ValueError(f"GDNConfig.dprod.{', '.join(stray)} (the per-layer / per-head placement) with stride {s} / "
                         f"summary {summary!r} / conv {conv}: not built together (the placement rebuilds each layer's DeltaProductConfig and "
                         "would drop these keys)")
    second = {"value": "proj", "beta": "proj", "rank": 16, "perm": "roll"}  # the keys of the second step's construction
    c_stray = [k for k, v in second.items() if getattr(d, k, v) != v]
    if c_stray:
        raise ValueError(f"GDNConfig.dprod.{', '.join(c_stray)} (the second step's construction) with stride {s} / "
                         f"summary {summary!r} / conv {conv}: not built together (this module forms the second step from the carrying token's x or "
                         "the group's summary through its own matrices and reads none of these keys)")
    route = D.route_of(cfg)  # main's key, checked
    if route != "interleave" and path not in PATH_UNMOVED:
        raise ValueError(f"GDNConfig.dprod.sparse_path {path!r} with route {route!r}: the sparse_path keys are the inserted-row layout's copies and the "
                         "interleave route's preparation; the one-readout route has no such layout (it reads q, k, v in the token layout), so the key would "
                         "change nothing on the fused path")
    if route != "interleave":
        # E (summary "none") is served by the route as the Gated DeltaProduct form's two rows a token with beta2 = 0 off the stride (sparse_forward,
        # the module docstring, THE ROUTE); D is not (one write per token between the boundaries: the route serves the forms that write twice per
        # token)
        if summary != "none":
            raise ValueError(f"GDNConfig.dprod.route {route!r} with summary {summary!r}: not built (arm D writes once per token between the group boundaries; "
                             "the one-readout kernel serves the forms that write twice per token. D trains on the interleave route, on the inserted-row layout)")
        bad_route = D.one_readout_refusals(cfg)
        if bad_route:
            raise ValueError(f"GDNConfig.dprod.route {route!r} (ced/kernels/gdn_1r.py) with stride {s} is not built with: {'; '.join(bad_route)}")
    start = getattr(d, "beta_init", None)  # main's key: read here, so the key is refused rather than ignored
    if start is not None:
        raise ValueError(f"GDNConfig.dprod.beta_init {start!r} with stride {s} / summary {summary!r} / conv {conv}: not built together (the start-beta "
                         "offset is applied in gdn_dprod.dprod_forward; this module forms step 1's beta from layer.gates and the second step's from "
                         "dprod.in_proj_b without it, so the key would be ignored)")
    if cfg.kda.kernel == "k2":
        raise ValueError("GDNConfig.kda.kernel 'k2' with the strided or summarised second step: K2 (ced/kernels/gdn_k2.py) has the same r rows a "
                         "token; a rows pattern needs its own mode, not built")


def params_per_layer(d_model: int, cfg: GDNConfig) -> int:
    """The tensors one GDN layer adds (exact): the Gated DeltaProduct form's count when the module is not on; kvmean H_v; else d c + d H_v,
c = H_k d_k + H_v d_v."""
    if not sparse_on(cfg):
        return D.params_per_layer(d_model, cfg)
    if keys_of(cfg)[1] == "kvmean":
        return cfg.n_v_heads
    return d_model * D.step_channels(cfg) + d_model * cfg.n_v_heads


# --------------------------------------------------------------------------------------------------------------------------
# the row layout: pure functions of Python integers and views


def first_carry(pos0: int, s: int) -> int:
    """The index, in a call whose first token is at absolute position pos0, of the first carrying token (p mod s = s - 1)."""
    return (s - 1 - pos0) % s


def carry_flags(L: int, s: int, pos0: int = 0) -> list:
    """Whether each of the L tokens of a call carries the second step (a Python list of bools: the phase is static)."""
    return [(pos0 + i) % s == s - 1 for i in range(L)]


def groups_of(L: int, s: int, off: int) -> tuple:
    """The groups of one call as runs: ([(a, ng, sz), ...], a_end). A run is ng consecutive groups of sz tokens from token a, the last
    token of each carrying; the head run is the one partial group (off + 1 < s tokens) that closes at the first carrying token, then the
    full groups of s; the tail [a_end, L) is the tokens after the last carrying token (fewer than s, no carry)."""
    out, a = [], 0
    if off < L:
        h = off + 1
        if h < s:
            out.append((0, 1, h))
            a = h
        ng = (L - a) // s
        if ng:
            out.append((a, ng, s))
            a += ng * s
    return out, a


def n_carrying(L: int, s: int, pos0: int = 0) -> int:
    off = first_carry(pos0, s)
    return 0 if off >= L else (L - 1 - off) // s + 1


def carrying(t: torch.Tensor, s: int, off: int) -> torch.Tensor:
    """t at the carrying tokens: (B, L, ...) -> (B, n_c, ...), a strided slice (positions off, off + s, ...)."""
    return t[:, off::s]


def insert_rows(t: torch.Tensor, extra, s: int, off: int, kind: str = "plain") -> torch.Tensor:
    """The row layout of a stack t (B, L, ...): the extra row of each carrying token after its first row. extra (B, n_c, ...) holds the
    extra rows in carrying order (kind "plain"); kind "q": the extra row is the carrying token's own q and its first row is zero (q is read
    on the token's last row only). Returns (B, L + n_c, ...); views, concatenations and one reshape a group shape (fixed shapes)."""
    B, L = t.shape[:2]
    rs, a_end = groups_of(L, s, off)
    if not rs:
        return t
    rest = t.shape[2:]
    parts, c = [], 0
    for a, ng, sz in rs:
        seg = t[:, a:a + ng * sz].reshape(B, ng, sz, *rest)
        if kind == "q":
            first = torch.cat([seg[:, :, :sz - 1], torch.zeros_like(seg[:, :, sz - 1:])], dim=2)
            ex = seg[:, :, sz - 1:]
        else:
            first, ex = seg, extra[:, c:c + ng].unsqueeze(2)
        c += ng
        parts.append(torch.cat([first, ex], dim=2).reshape(B, ng * (sz + 1), *rest))
    if a_end < L:
        parts.append(t[:, a_end:])
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


def take_tokens(o: torch.Tensor, L: int, s: int, off: int) -> torch.Tensor:
    """The token outputs of the rows' outputs o (B, L + n_c, ...): a carrying token's output is read on its extra row (its first row has
    q = 0); every other token's on its own row. Returns (B, L, ...)."""
    B = o.shape[0]
    rs, a_end = groups_of(L, s, off)
    if not rs:
        return o
    rest = o.shape[2:]
    parts, r = [], 0
    for a, ng, sz in rs:
        seg = o[:, r:r + ng * (sz + 1)].reshape(B, ng, sz + 1, *rest)
        r += ng * (sz + 1)
        parts.append(torch.cat([seg[:, :, :sz - 1], seg[:, :, sz:]], dim=2).reshape(B, ng * sz, *rest))
    if a_end < L:
        parts.append(o[:, r:])
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


def rows_cu(cu: torch.Tensor, L: int, s: int) -> torch.Tensor:
    """The varlen boundaries of the row layout (a call from position 0, B rows of L tokens flattened; every row start is a boundary): a
    boundary at token c = b L + p (b the row, p the position in it) lies at row c + b (L // s) + p // s (the carrying tokens before it).
    For s dividing L that is c + c // s (the design's cu + cu // s). Integer arithmetic on the boundaries' tensor (device or host)."""
    return cu + (cu // L) * (L // s) + (cu % L) // s


def group_sums(u: torch.Tensor, doc: torch.Tensor, s: int, off: int, carry_in=None):
    """The summary's sums over each closed group of a call: u (B, L, F) features, doc (B, L) document ids. Returns (sums (B, n_c, F),
    counts (B, n_c)): for group i, the sum and the number of the tokens of the group that are in the document of its last (carrying)
    token. carry_in = (acc, cnt, ok): the sums of a group that began in an earlier call (acc (B, F), cnt (B,)) and ok (B,) 1 where the
    document of the head group's last token is the carried tokens' document; added to the head group (the first run when its size is
    below s)."""
    B = u.shape[0]
    rs, _ = groups_of(u.shape[1], s, off)
    sums, counts = [], []
    for j, (a, ng, sz) in enumerate(rs):
        seg_u = u[:, a:a + ng * sz].reshape(B, ng, sz, -1)
        seg_d = doc[:, a:a + ng * sz].reshape(B, ng, sz)
        m = (seg_d == seg_d[:, :, -1:]).to(u.dtype)
        su, sc = (m[..., None] * seg_u).sum(2), m.sum(2)
        if carry_in is not None and j == 0 and sz < s:
            acc, cnt, ok = carry_in
            su = su + (ok[:, None] * acc)[:, None]
            sc = sc + (ok * cnt)[:, None]
        sums.append(su)
        counts.append(sc)
    return torch.cat(sums, dim=1), torch.cat(counts, dim=1)


def tail_sums(u: torch.Tensor, doc: torch.Tensor, s: int, off: int, carry_in=None):
    """The sums over the group still open at the end of a call (the tokens after its last carrying token, or the whole call when it has
    none): (acc (B, F), cnt (B,)), restarted at a document start (only the last token's document counts), plus the carried sums
    carry_in = (acc, cnt, ok) when the call has no closed group (the open group continues one that began earlier; ok (B,): no document
    start in the call). Zeros when the call ends on a group boundary."""
    B, L = u.shape[:2]
    rs, a_end = groups_of(L, s, off)
    if a_end >= L:
        return u.new_zeros(B, u.shape[-1]), u.new_zeros(B)
    seg_u, seg_d = u[:, a_end:], doc[:, a_end:]
    m = (seg_d == seg_d[:, -1:]).to(u.dtype)
    acc, cnt = (m[..., None] * seg_u).sum(1), m.sum(1)
    if carry_in is not None and not rs:
        c_acc, c_cnt, ok = carry_in
        acc, cnt = acc + ok[:, None] * c_acc, cnt + ok * c_cnt
    return acc, cnt


# --------------------------------------------------------------------------------------------------------------------------
# the rule: the reference loop


def sparse_recurrence(q: torch.Tensor, k1: torch.Tensor, v1: torch.Tensor, g: torch.Tensor, beta1: torch.Tensor, is_doc_start: torch.Tensor,
                      S: torch.Tensor, carry, k2: torch.Tensor, v2: torch.Tensor, beta2: torch.Tensor, all_states: bool = False):
    """The rule above, one token at a time (the ground truth of these arms; written from the rule, not through the row layout).

    q, k1: (B, L, H, d_k) at the value heads (repeated), q scaled; v1: (B, L, H, d_v); g, beta1: (B, L, H); is_doc_start: (B, L) bool; S:
    (B, H, d_k, d_v) the state before the first token; carry: a list of L bools (T_c); k2: (B, n_c, H, d_k), v2: (B, n_c, H, d_v), beta2:
    (B, n_c, H): the second steps of the carrying tokens in order (n_c = sum(carry)). Per token: reset at a document start, decay by exp(g)
    once, step 1, step 2 if the token carries, read S^T q. Returns the outputs (B, L, H, d_v) and the final state; all_states: also the
    state after every token (B, L, H, d_k, d_v)."""
    assert len(carry) == q.shape[1] and sum(carry) == k2.shape[1] == v2.shape[1] == beta2.shape[1]
    outputs, states, j = [], [], 0
    rows = zip(q.unbind(1), k1.unbind(1), v1.unbind(1), g.unbind(1), beta1.unbind(1), is_doc_start.unbind(1), carry)
    for q_t, k_t, v_t, g_t, b_t, start_t, carries in rows:
        S = S * (~start_t).to(S.dtype)[:, None, None, None]
        S = S * torch.exp(g_t)[:, :, None, None]
        steps = [(k_t, v_t, b_t)]
        if carries:
            steps.append((k2[:, j], v2[:, j], beta2[:, j]))
            j += 1
        for kj, vj, bj in steps:
            predicted = torch.einsum("bhkv,bhk->bhv", S, kj)
            S = S + torch.einsum("bhk,bhv->bhkv", kj, bj[:, :, None] * (vj - predicted))
        outputs.append(torch.einsum("bhkv,bhk->bhv", S, q_t))
        if all_states:
            states.append(S)
    if all_states:
        return torch.stack(outputs, dim=1), S, torch.stack(states, dim=1)
    return torch.stack(outputs, dim=1), S


# --------------------------------------------------------------------------------------------------------------------------
# the layer's tensors and the cache


class SparseSteps(D.DeltaProductSteps):
    """The tensors of the second step of a strided or summarised layer (GatedDeltaNet.dprod): in_proj_kv (d -> c, c = H_k d_k + H_v d_v, [k_2
    | v_2]) and in_proj_b (d -> H_v) for E and D "xmean"; beta_const (H_v,) for D "kvmean". A DeltaProductSteps (the model's
    init_own_parameters draws every one in module order), built on the meta device and moved to `device` without a draw, then drawn from a
    fixed generator (reset_own); the model's build draws them again from the run's own stream. `forward_fn` is the layer's forward
    (GatedDeltaNet.forward calls it in place of dprod_forward)."""

    sparse = True

    def __init__(self, d_model: int, cfg: GDNConfig, device=None, dtype=None):
        nn.Module.__init__(self)
        self.n_extra = 1
        self.stride, self.summary, self.conv, self.first_beta = keys_of(cfg)
        self.sparse_path = path_of(cfg)
        self.channels = D.step_channels(cfg)
        dev = torch.device("cpu") if device is None else torch.device(device)
        if self.summary == "kvmean":
            self.in_proj_kv = None
            self.in_proj_b = None
            self.beta_const = nn.Parameter(torch.zeros(cfg.n_v_heads, device=dev, dtype=dtype))
        else:
            self.in_proj_kv = nn.Linear(d_model, self.channels, bias=False, device="meta", dtype=dtype).to_empty(device=dev)
            self.in_proj_b = nn.Linear(d_model, cfg.n_v_heads, bias=False, device="meta", dtype=dtype).to_empty(device=dev)
            self.in_proj_kv.own_init = True  # init_weights draws every other linear from the global stream; these are drawn here
            self.in_proj_b.own_init = True
        self.reset_own(torch.Generator().manual_seed(D.STANDALONE_SEED), 0.02)

    @torch.no_grad()
    def reset_own(self, g: torch.Generator, std: float) -> None:
        """in_proj_kv, then in_proj_b: N(0, std) from the CPU generator g in float32 and copied in; beta_const: zeros, nothing
        drawn. Tensors on the meta device are skipped."""
        if self.summary == "kvmean":
            self.beta_const.zero_()
            return
        for p in (self.in_proj_kv.weight, self.in_proj_b.weight):
            if p.is_meta:
                continue
            p.copy_((torch.randn(p.shape, generator=g, dtype=torch.float32) * std).to(device=p.device, dtype=p.dtype))

    @staticmethod
    def forward_fn(layer, *args, **kw):
        return sparse_forward(layer, *args, **kw)


@dataclass
class SparseState(G.GDNState):
    """GDNState plus the running sums of D's summary (the open group): acc (B, F), cnt (B,) float. None in E's cache (the phase is the
    absolute index) and in the training forward."""

    acc: torch.Tensor | None = None
    cnt: torch.Tensor | None = None


# --------------------------------------------------------------------------------------------------------------------------
# the layer's forward


def _unit(x: torch.Tensor) -> torch.Tensor:
    return D._unit(x)


def _rows_call(fused, q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu, L: int, s: int, **kw):
    """The fused delta rule on the row layout: the boundaries (device and host copies) mapped by rows_cu."""
    cu = rows_cu(cu_seqlens, L, s)
    cu_cpu = rows_cu(cu_seqlens_cpu, L, s) if cu_seqlens_cpu is not None else None
    return fused(q, k, v, g, beta, cu, cu_cpu, **kw)


_rows_call = torch.compiler.disable(_rows_call)  # never traced, as the kernels' own wrappers: the boundaries' host copy is a CPU tensor


def sparse_forward(layer, x: torch.Tensor, ctx, state=None, valid=None, keep: bool = False, arms=None, x_write=None, ccr=None,
                   x_read=None, lerped: bool = False):
    """GatedDeltaNet.forward for a layer with the second step on a subset of the tokens (layer.dprod a SparseSteps): the same inputs and
    result, (y, GDNState) (a SparseState for D). Module docstring: the rule, the layout, the paths, the cache."""
    D._refuse_call(layer, ctx, valid, keep, arms, x_write, ccr, x_read)
    B, L, _ = x.shape
    cfg, steps = layer.cfg, layer.dprod
    hk, hv, dk, dv, K = cfg.n_k_heads, cfg.n_v_heads, cfg.k_head_dim, cfg.v_head_dim, cfg.conv_kernel
    s, summary, fb = steps.stride, steps.summary, steps.first_beta
    if cfg.kda.kernel == "k2":
        raise ValueError("kda.kernel 'k2' with the strided or summarised second step: not built (ced/model/gdn_dprod_sparse.py)")
    u_pre = layer.in_proj_qkv(x)
    if state is not None:
        if state.conv_input.shape[-1] != u_pre.shape[-1]:
            raise ValueError(f"this GDN cache holds {state.conv_input.shape[-1]} conv channels; a layer with the strided second step carries "
                             f"{u_pre.shape[-1]} (its own: the second step's channels have no conv)")
        if summary != "none" and not isinstance(state, SparseState):
            raise ValueError("this GDN cache has no running sums (SparseState): a layer with a summarised second step (dprod.summary "
                             f"{summary!r}) reads and writes them")
    use_fused = cfg.kernel in ("fla", "ced") and state is None and ctx.cu_seqlens is not None and G.fla_supported(x)
    kernel_dtype = (torch.bfloat16 if use_fused and cfg.input_dtype == "bfloat16" and u_pre.dtype == torch.bfloat16 else None)
    fused_prep = layer._fused_prep and state is None and ctx.cu_seqlens is not None and G.fla_supported(x) and L >= K - 1
    # sparse_path "rows": the preparation writes the row layout itself (ced/kernels/gdn_prep_rows.py), so q, k1, v1 are not formed in the token layout;
    # from position 0 with whole groups (L a multiple of the stride), and not for D "kvmean" (its summary pools k1 and v1)
    rows_prep = fused_prep and steps.sparse_path == "rows" and summary != "kvmean" and L % s == 0 and D.route_of(cfg) == "interleave"  # (the one-readout route reads q, k, v in the token layout)
    if rows_prep:
        q = k1 = v1 = None
        conv_input, conv_doc = u_pre[:, L - (K - 1):], ctx.doc[:, L - (K - 1):]
    elif fused_prep:
        q, k1, v1 = G.qkv_prep(u_pre, ctx.doc, layer.conv_weight, hk, hv, dk, out_dtype=kernel_dtype)
        conv_input, conv_doc = u_pre[:, L - (K - 1):], ctx.doc[:, L - (K - 1):]
    else:
        u, conv_input, conv_doc = layer.short_conv(u_pre, ctx, state, None)
        u = F.silu(u)
        q, k1, v1 = torch.split(u, [hk * dk, hk * dk, hv * dv], dim=-1)
        q = G.l2_normalize(q.reshape(B, L, hk, dk)) * dk**-0.5
        k1 = G.l2_normalize(k1.reshape(B, L, hk, dk))
        v1 = v1.reshape(B, L, hv, dv)
    pos0 = 0 if state is None else int(ctx.idx[0, 0])  # the phase of the stride: the absolute index of the call's first token
    off = first_carry(pos0, s)
    n_c = n_carrying(L, s, pos0)
    bmax = D.beta_multiplier(cfg)
    g, beta1 = layer.gates(x)

    # ---- step 1's beta: bmax sigmoid(b) on a carrying token; on the others sigmoid(b) (first_beta "carry") or bmax sigmoid(b) ("all")
    if fb == "all" or s == 1 or bmax == 1:
        beta1 = bmax * beta1
    else:
        mask = ((torch.arange(L, device=x.device) + pos0) % s == s - 1).to(beta1.dtype)[None, :, None]
        beta1 = beta1 * (1 + (bmax - 1) * mask)

    # ---- D's features, the carried sums of a group that began in an earlier call, and the group sums
    feats = carry_in = mean = None
    if summary == "xmean":
        feats = x.to(torch.promote_types(x.dtype, torch.float32))
    elif summary == "kvmean":
        feats = torch.cat([k1.reshape(B, L, hk * dk), v1.reshape(B, L, hv * dv)], dim=-1)
        feats = feats.to(torch.promote_types(feats.dtype, torch.float32))
    if summary != "none":
        if state is not None and pos0 % s != 0:  # the call's first group began in an earlier call
            if state.acc is None:
                raise ValueError("this GDN cache has no running sums (a SparseState written by a layer with a summarised second step)")
            ok = ~ctx.is_doc_start[:, :min(off + 1, L)].any(dim=1)  # the carried tokens are in the closing token's document
            carry_in = (state.acc.to(feats.dtype), state.cnt.to(feats.dtype), ok.to(feats.dtype))
        if n_c:
            sums, counts = group_sums(feats, ctx.doc, s, off, carry_in)
            mean = sums / counts[..., None]

    # ---- the second step's k2, v2, beta2 on the carrying tokens, (B, n_c, ...)
    k2 = v2 = beta2 = None
    if n_c and summary == "kvmean":
        k_bar, v_bar = torch.split(mean, [hk * dk, hv * dv], dim=-1)
        k2 = _unit(k_bar.reshape(B, n_c, hk, dk).to(k1.dtype))
        v2 = v_bar.reshape(B, n_c, hv, dv).to(v1.dtype)
        beta2 = (bmax * torch.sigmoid(steps.beta_const))[None, None, :].expand(B, n_c, hv).to(beta1.dtype)
    elif n_c:
        src = carrying(x, s, off) if summary == "none" else mean.to(x.dtype)
        k_part, v_part = torch.split(F.silu(steps.in_proj_kv(src)), [hk * dk, hv * dv], dim=-1)
        k2 = _unit(k_part.reshape(B, n_c, hk, dk))
        v2 = v_part.reshape(B, n_c, hv, dv)
        beta2 = bmax * torch.sigmoid(steps.in_proj_b(src))
    acc_out = cnt_out = None
    if summary != "none" and not use_fused:  # the cache's open group (the training and scoring forwards build no cache)
        acc_out, cnt_out = tail_sums(feats, ctx.doc, s, off, carry_in)

    if kernel_dtype is not None:
        if not rows_prep:
            q, k1, v1 = q.to(kernel_dtype), k1.to(kernel_dtype), v1.to(kernel_dtype)
        beta1 = beta1.to(kernel_dtype)
        if n_c:
            k2, v2, beta2 = k2.to(kernel_dtype), v2.to(kernel_dtype), beta2.to(kernel_dtype)

    # ---- the rule (sparse_path "plain": the layout's copies as built; "fast" and "rows": the same values and gradients, the fewest passes)
    if steps.sparse_path in ("fast", "rows"):
        rs, a_end = groups_of(L, s, off)
        ins = lambda t, extra, kind="plain": FP.insert_rows_fast(t, extra, rs, a_end, kind)  # noqa: E731
        take = lambda o_: FP.take_tokens_fast(o_, L, rs, a_end)  # noqa: E731
    else:
        ins = lambda t, extra, kind="plain": insert_rows(t, extra, s, off, kind)  # noqa: E731
        take = lambda o_: take_tokens(o_, L, s, off)  # noqa: E731
    if use_fused and D.route_of(cfg) == "one-readout":
        # THE ROUTE (module docstring): the Gated DeltaProduct form's two rows a token, the second step at the carrying positions and zero (beta2 = 0,
        # the identity) elsewhere
        def spread(t, base):  # (B, n_c, ...) -> (B, L, ...): the carrying positions off, off + s, ... hold t, the others zero (a strided write: fixed shapes)
            full = torch.zeros(B, L, *base.shape[2:], dtype=base.dtype, device=base.device)
            return full.slice_scatter(t.to(base.dtype), dim=1, start=off, step=s) if n_c else full

        k = torch.stack([k1, spread(k2, k1)], dim=2)  # (B, L, 2, H_k, d_k)
        v = torch.stack([v1, spread(v2, v1)], dim=2)  # (B, L, 2, H_v, d_v)
        beta = torch.stack([beta1, spread(beta2, beta1)], dim=2)  # (B, L, 2, H_v)
        o = D._one_readout_call(q, k, v, g, beta, ctx.cu_seqlens, ctx.cu_seqlens_cpu, **layer._ced_kw)
        S = None  # training / scoring forward: the final state is not needed
    elif use_fused:
        zeros = lambda t: torch.zeros(B, n_c, *t.shape[2:], dtype=t.dtype, device=t.device)  # noqa: E731
        if rows_prep:  # the preparation writes q, k, v in the row layout, the extra rows' k and v from the group summaries (k2, v2)
            qr, kr, vr = FP.prep_rows(u_pre, ctx.doc, layer.conv_weight, hk, hv, dk, k2, v2, s, out_dtype=kernel_dtype)
        else:
            v1_ = v1.contiguous()
            qr = ins(q, None, "q")
            kr = ins(k1, k2)
            vr = ins(v1_, v2)
        gr = ins(g, zeros(g))
        br = ins(beta1, beta2)
        fused = G.fla_delta_rule if cfg.kernel == "fla" else G.ced_delta_rule
        o = take(_rows_call(fused, qr, kr, vr, gr, br, ctx.cu_seqlens, ctx.cu_seqlens_cpu, L, s, **layer._ced_kw))
        S = None  # training / scoring forward: the final state is not needed
    else:
        rep = hv // hk
        q_r, k1_r = q.repeat_interleave(rep, dim=2), k1.repeat_interleave(rep, dim=2)
        k2_r = k2.repeat_interleave(rep, dim=2) if n_c else k1_r[:, :0]
        v2_ = v2 if n_c else v1[:, :0]
        b2_ = beta2 if n_c else beta1[:, :0]
        S0 = x.new_zeros(B, hv, dk, dv) if state is None else state.S
        if cfg.kernel in ("chunked", "fla", "ced") and L > 1:
            zeros = lambda t: torch.zeros(B, n_c, *t.shape[2:], dtype=t.dtype, device=t.device)  # noqa: E731
            qr = ins(q_r, None, "q")
            kr = ins(k1_r, k2_r)
            vr = ins(v1, v2_)
            gr = ins(g, zeros(g))
            br = ins(beta1, b2_)
            sr = ins(ctx.is_doc_start, torch.zeros(B, n_c, dtype=torch.bool, device=x.device))
            o_rows, S = chunked_delta_rule(qr, kr, vr, gr, br, sr, S0, cfg.chunk_size)
            o = take(o_rows)
        else:
            o, S = sparse_recurrence(q_r, k1_r, v1, g, beta1, ctx.is_doc_start, S0, carry_flags(L, s, pos0), k2_r, v2_, b2_)
    z = layer.in_proj_z(x).view(B, L, hv, dv)
    y = layer.out_proj(layer.norm(o, z).flatten(2))
    if summary != "none" and not use_fused:
        return y, SparseState(conv_input=conv_input, conv_doc=conv_doc, S=S, acc=acc_out, cnt=cnt_out)
    return y, G.GDNState(conv_input=conv_input, conv_doc=conv_doc, S=S)


# --------------------------------------------------------------------------------------------------------------------------
# the hooks (appended to gdn_dprod.py as two rebinding lines: its statements keep their lines)


def build_steps(layer: nn.Module, d_model: int, cfg: GDNConfig, *args, **kw):
    """gdn_dprod.build for a layer: the key refusals; with the module off, the Gated DeltaProduct form's build (None for n_h 1,
    DeltaProductSteps), else the layer-level refusals of the Gated DeltaProduct form's build (kernel, edges, chunk backend) and
    SparseSteps."""
    check_keys(cfg, D.n_steps(cfg))
    if not sparse_on(cfg):
        return D._arm6_build(layer, d_model, cfg, *args, **kw)
    refused = [(cfg.kernel not in D.KERNELS, f"kernel {cfg.kernel!r} (built for {D.KERNELS})"),
               (path_of(cfg) == "rows" and cfg.kernel != "ced", f"sparse_path 'rows' with kernel {cfg.kernel!r} (the preparation that writes the rows is our kernel 'ced' fused prep's)"),
               (cfg.edges != "plain", f"edges {cfg.edges!r} (the fused edges ops form q, k, v of one step in our chunk kernels)"),
               (cfg.chunk_backend != "triton", f"chunk_backend {cfg.chunk_backend!r} (built on our Triton kernel's call)")]
    bad = [why for cond, why in refused if cond]
    if bad:
        raise ValueError(f"GDNConfig.dprod.stride / summary / conv (ced/model/gdn_dprod_sparse.py) is not built with: {'; '.join(bad)}")
    heads = args[0] if args else kw.get("heads")  # the carrying value heads (None: every head); a proper subset is not built here
    if heads is not None and tuple(heads) != tuple(range(cfg.n_v_heads)):
        raise ValueError(f"GDNConfig.dprod.stride / summary / conv with carrying value heads {tuple(heads)!r} of {cfg.n_v_heads}: the per-head placement "
                         "is not built together with the strided and summarised second step")
    w = layer.in_proj_qkv.weight
    return SparseSteps(d_model, cfg, device=w.device, dtype=w.dtype)


def check_model(cfg) -> None:
    """gdn_dprod.check_config for a model: the decoder's GDN section keeps the keys at their defaults; the encoder's are
    checked (check_keys); then the Gated DeltaProduct form's model-level refusals (arch, the GDN-side arms, the second
    pass, the drafter, FP8 / NVFP4)."""
    for side in ("encoder", "decoder"):
        sec = getattr(cfg, side, None)
        gdn = getattr(sec, "gdn", None)
        if gdn is None:
            continue
        if side == "decoder":
            moved = [k for k, v in zip(DEFAULTS, keys_of(gdn)) if v != DEFAULTS[k]]
            if path_set(gdn) not in PATH_UNMOVED:
                moved.append("sparse_path")
            if moved:
                raise ValueError(f"model.decoder.gdn.dprod.{', '.join(moved)}: the strided and summarised second step is built for the encoder's "
                                 "GDN layers only")
        check_keys(gdn, D.n_steps(gdn))
    D._arm6_check_config(cfg)


