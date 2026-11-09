"""Block Attention Residuals: the encoder stack's residual accumulation replaced by softmax attention over depth
(EncoderConfig.attn_res "block").

The source: arXiv 2603.15031v1, "Attention Residuals" (Kimi Team, 16 Mar 2026), section 3.2 "Block Attention
Residuals", equations 2, 3, 5 and 6, the pseudocode of its Figure 2, and section 5 "Architecture Details" for the
initialisation. The repository github.com/MoonshotAI/Attention-Residuals holds the paper and that pseudocode in its
README and no other code (read 2026-10-13).

What is computed. A "layer" of the paper is one SUBLAYER here: an encoder layer's mixer (GDN or global attention) and
its MLP are two of them ("we treat each self-attention or MLP as an individual layer", section 2). A sublayer's output
f is what the base adds to the stream: residual_scale * out_norm(mixer(in_norm(h))), and the same for the MLP: our
sublayers stay as they are (pre-norm, mixer or MLP, output norm). Only where h comes from changes.

    b_0 = the embedding output (after embed_norm): always a source                    ("defining b_0 = h_1", 3.2)
    b_n = the sum of the outputs f of the sublayers of block n, n = 1 .. N            (Eq. 5)
    b_n^i = the sum of the first i of them (the partial sum)

    the input of sublayer i of block n (i = 1 .. S):                                   (Eq. 6)
        V = [b_0, b_1, ..., b_{n-1}]              i = 1
        V = [b_0, b_1, ..., b_{n-1}, b_n^{i-1}]   i >= 2
        alpha = softmax over the sources of  w^T RMSNorm(v),  v in V                   (Eq. 2 and 3: key = value)
        h = sum of alpha_v * v over V                                                  (Eq. 4)
    the output after the last sublayer: the same attention over [b_0, b_1, ..., b_N], with a query and a key norm of
    its own ("The final output layer aggregates all N block representations"; Figure 1 (c) draws it with its own w
    and with the embedding among its sources). The model's final norm and head read it. Its two tensors are held by
    the last layer (layer.attn_res.out), whose call ends with this read, so that it runs inside that layer's compiled
    graph and nothing of a forward's arithmetic runs outside the compiled layers.

    w: one (d,) pseudo-query per reader, ZEROS at the start ("all pseudo-query vectors must be initialized to zero",
    section 5), so every alpha starts uniform over the reader's sources. RMSNorm: one per reader, on the keys only
    (the values are the sources themselves). The very first sublayer has one source, b_0: its weight is 1 whatever w
    is, so no query and no norm is built for it.

Blocks here are the hybrid groups: full_attn_interval layers each, [GDN, GDN, GDN, global] at 60M, so S = 8 sublayers
and N = n_layers / full_attn_interval = 5 blocks; a layout that does not divide is refused.

The depth attention is per position: nothing in it mixes positions, so causality and the cache of a chunk are the
layers' own (the per-layer GDN states and K/V; a cached chunk carries nothing of the depth attention).

Precision. The softmax over depth, its logits, the key norm and the weighted sum run in at least float32 with
autocast off, whatever the autocast dtype around them; block sums accumulate in the stream's dtype (float32 under
bf16 autocast, as the base's stream does).

Two implementations of the same rule:
- reference_stack: the paper's pseudocode line for line (stack the sources, RMSNorm, einsum, softmax, einsum), eager,
  one sublayer after the other; it can record every sublayer's output and weights (depth_terms). The ground truth.
- the fast path, which every forward runs (training under compile "all", evaluation, scoring, prefill / extend):
  EncoderLayer.forward with `depth` runs layer_forward, the layer's two sublayers (and in the last layer the
  output's read) with the logits taken source by source (no stacked copy of the sources, no normalised copy of
  them); the stack is BlockAttnRes.stack. Accepted against the reference in tests/test_attn_res.py.

R1-entry (EncoderConfig.attn_res "block_entry"): the depth attention only where a block begins
--------------------------------------------------------------------------------------------- This is NOT a form of the
paper: the paper's Block formulation reads before every sublayer (Eq. 6, i = 1 .. S); here only its i = 1 read is kept.
Everything not named below is R1's, above.

    b_0 = the embedding output (after embed_norm): always a source                    (as R1)
    b_n = the sum of the outputs f of the sublayers of block n, n = 1 .. N            (Eq. 5, unchanged)

    where block n begins, its input is R1's first read of the block:                   (Eq. 6 at i = 1)
        V = [b_0, b_1, ..., b_{n-1}]
        alpha = softmax over the sources of  w_n^T RMSNorm_n(v),  v in V               (Eq. 2 and 3: key = value)
        x_n^0 = sum of alpha_v * v over V                                              (Eq. 4)
        for n = 1 there is one source, b_0: x_1^0 = b_0, and no query and no norm is built (as R1's first sublayer).
    inside the block, the plain residual add of the base, sublayer after sublayer (i = 1 .. S; no reader):
        x_n^i = x_n^{i-1} + f_i(x_n^{i-1})
    so  b_n = sum_i f_i(x_n^{i-1}) = x_n^S - x_n^0 : what the block added to its own input.
    the output after the last block: R1's own read, unchanged: the same attention over [b_0, b_1, ..., b_N] with a
    query and a key norm of its own. The model's final norm and head read it.

    Readers: the entries of blocks 2 .. N and the output's read: N in all (5 at 60M, 9 at 9 blocks), against R1's 2 per
    layer (8 N). Each is a DepthQuery with R1's defaults: a (d,) pseudo-query of ZEROS at the start (uniform weights),
    an RMSNorm on its keys, the embedding always among its sources. The entry reader of block n is held by the
    block's first layer (layer.attn_res.entry), the output's by the last layer (layer.attn_res.out): the state-dict
    names are encoder.layers.<(n-1) S>.attn_res.entry.{query, key_norm.weight} and
    encoder.layers.<L-1>.attn_res.out.{query, key_norm.weight}. A layer that neither opens nor closes a block holds
    nothing and runs the base's own layer call.

    What does not carry over from the base: a block's input is the MIX of the earlier blocks, not the stream where the
    block before ended. At the start (queries at zero) block n reads the plain mean (b_0 + ... + b_{n-1}) / n where
    the base reads the sum b_0 + ... + b_{n-1}; only block 1 is the base's block 1. So, as R1, this arm does not
    start as the base.

Two implementations, as for R1:
- entry_reference_stack: the form above line for line, eager, one sublayer after the other: the stream x and the
  block's sum (Eq. 5) are both accumulated, and a read is block_attn_res (the paper's pseudocode). The ground truth.
- the fast path, which every forward runs: EntryWalk and entry_layer_forward. A layer inside a block is the base's own
  EncoderLayer.forward (no `depth`: the base's compiled trace). A block's first layer forms the entry read with
  depth_mix inside its own graph and returns the block's input beside the stream; the block's last layer takes that
  input and returns b_n = (the stream where the block ends) - (the block's input) in place of the stream, which
  nothing reads after the block; the stack's last layer ends with the output's read. So, as for R1, nothing of a
  forward's arithmetic runs outside the compiled layers. The difference of two float32 streams is the block's sum up
  to rounding, not bit for bit (tests/test_attn_res_entry.py holds the two implementations together).

The architecture arms on R1-entry
-----------------------------------------------------------------------------------------------------------------
Two options of R1-entry's readers, each off by default (and then every line below is R1-entry's, bit for bit):

- EncoderConfig.attn_res_heads H ("Grouped depth mixing on R1-entry, H=4 and H=8 (MHAR, arXiv 2607.27230)"): the
  reader's query w and every source's channels are split into H contiguous slices of d / H, and each slice has its
  own softmax over the sources (the paper's Eq. 3): alpha^(h)_v = softmax over v of w_[h]^T RMSNorm(v)_[h] h_[h] =
  sum_v alpha^(h)_v v_[h] with the key norm's RMS taken over all d channels and then sliced (Eq. 3, which slices
  RMSNorm(s_i), and the fused kernel of the paper's Appendix E, "the RMS statistic over the full d-row"; its Figure
  3 pseudocode calls norm(Vh) after the reshape). No tensor is added: the H queries are the (d,) query's slices. At
  H = 1 it is R1-entry's read.
- EncoderConfig.attn_res_query "token" (the intermediate arm, "R1-entry with token-dependent queries (LDM found
  static queries cost +0.19 ppl)"): a reader's query is w(x) = W2 SiLU(W1 RMSNorm(x) + b1) + b2, LDM's address network
  (arXiv 2610.18809, section 3.2; ced/model/ldm.py AddressNet) with a d-wide head and no gate, per position, where x is
  the stream at the read: for the entry of block n the stream where block n - 1 ended (x_{n-1}^S, the stream the
  base would hand to block n), for the output's read the stream after the last layer. W2 and b2 start at zero, so
  every reader starts uniform and the model starts as R1-entry; b2 is the static part (R1-entry's w), W1 is drawn
  N(0, init_std) from the depth mixers' own generator (ldm.init_own_parameters). The logits keep R1-entry's key norm:
  w(x)^T RMSNorm(v). The two options compose (a token query split into heads).
"""


from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ced.config import ModelConfig, check_residual_dtype as _check_residual_dtype_value, check_write_mix_storage
from ced.model import act_checkpoint, ldm
from ced.model.norms import RMSNorm, compute_dtype, round_stream

ATTN_RES = ("none", "block", "block_entry", "ldm")
ENTRY = "block_entry"  # R1-entry: the depth attention at each block's entry and at the output only (the docstring above)
LDM = ldm.MODE  # the variant, Linear Depth Memory with the delta-rule write (ced/model/ldm.py)
QUERIES = ("static", "token")  # EncoderConfig.attn_res_query: R1-entry's pseudo-queries, or token-dependent ones
# Compiled layers (nn.Module.compile; the trainer's compile "all", the scoring scripts): Dynamo keeps one compiled
# entry of EncoderLayer.forward per distinct trace and row shape, and stops (an error under the trainer's settings,
# ced/train/trainer.py configure_compile) at torch._dynamo.config.recompile_limit entries, 32 there. Without the key
# the layers of one kind share an entry (three per row shape at 60M: GDN, capped attention, NoPE attention). Under
# "block" the trace also depends on how many sources a layer reads, so each block has three of its own (its first
# layer, its other GDN layers, its global layer): 3 x n_blocks per row shape, 15 at 60M. The limit is raised by the
# same factor whenever a walk of a stack with compiled layers starts (BlockAttnRes.walk: every forward through stack,
# and the GDN diagnostics' own walk), so a process has room for as many row shapes as before; reaching it stays an error.
RECOMPILE_LIMIT_PER_BLOCK = 32


def check_attn_res(cfg: ModelConfig) -> None:
    """Refuse EncoderConfig.attn_res wherever it is not built, so that no model ignores the key, and refuse a layout
    its blocks do not fit. ValueError, by name. The same for both values, "block" (R1) and "block_entry" (R1-entry).
    Built: arch "control", one pass through the stack, blocks of full_attn_interval layers. Refused, each because another code path walks or changes the layers' residual add and
    would skip the depth attention, or because nothing tests the pair: any other arch; the residual scalars
    (encoder.layer_scalars); the gated second pass (loop.pass2_gate) and the loop in the published form
    (loop.recur_layers); a looped encoder; Canon layers; NVFP4 fake quantisation. The architecture arms' options
    (new_options: "ldm", attn_res_heads, attn_res_query) are also refused with FP8, the scalar block-norm gain, the MTP
    head and the GDN-side arms (gdn_write_mix, gdn_ccr_spacing; ced/model/gdn_arms.py)."""
    mode = cfg.encoder.attn_res
    if mode not in ATTN_RES:
        raise ValueError(f"model.encoder.attn_res {mode!r}: one of {ATTN_RES}")
    check_residual_dtype(cfg)  # EncoderConfig.residual_dtype: its value, and refused where another path would skip the rounding
    check_depth_options(cfg)
    if mode == "none":
        return
    e = cfg.encoder
    if cfg.arch != "control":
        raise ValueError(f"model.encoder.attn_res {mode!r} is built for arch 'control', not {cfg.arch!r}")
    if e.full_attn_interval <= 0 or e.n_layers % e.full_attn_interval:
        raise ValueError(f"model.encoder.attn_res {mode!r}: a block is one hybrid group of full_attn_interval layers ending "
                         f"with its global layer; n_layers {e.n_layers} is not a whole number of groups of "
                         f"full_attn_interval {e.full_attn_interval}")
    if e.layer_scalars != "none":
        raise ValueError(f"model.encoder.attn_res {mode!r} is not built together with model.encoder.layer_scalars "
                         f"({e.layer_scalars!r}): both replace the layers' residual add. Set one")
    if cfg.loop.pass2_gate != "none":
        raise ValueError(f"model.encoder.attn_res {mode!r} is not built together with the control's second pass "
                         f"(model.loop.pass2_gate {cfg.loop.pass2_gate!r}): pass 2's zero-write layer calls add to a "
                         "stream that this model does not have")
    if cfg.loop.recur_layers:
        raise ValueError(f"model.encoder.attn_res {mode!r} is not built together with the loop in the published form "
                         f"(model.loop.recur_layers {list(cfg.loop.recur_layers)!r}): a later pass would need block "
                         "sources of its own, and nothing defines them")
    if e.loops != 1:
        raise ValueError(f"model.encoder.attn_res {mode!r} is not built for a looped encoder (encoder.loops {e.loops})")
    if e.canon_kernel:
        raise ValueError(f"model.encoder.attn_res {mode!r} is not built together with model.encoder.canon_kernel "
                         f"({e.canon_kernel!r}): the layer call under attn_res does not run a Canon layer")
    if cfg.nvfp4.mode not in ("none", "mlp_only"):
        # "mlp_only" (the NVFP4 MLP arm, 2026-10-27) turns the FFN's linears only and leaves the mixers' alone; the walk calls
        # layer.mlp(...) as a module, so the pair is built and tested (tests/test_nvfp4_mlp.py)
        raise ValueError(f"model.encoder.attn_res {mode!r} is not built together with model.nvfp4 (mode "
                         f"{cfg.nvfp4.mode!r}): nothing tests the pair")
    if new_options(cfg):
        # the architecture arms (2026-10-16): built and tested on the plain trunk only
        what = ", ".join(new_options(cfg))
        if cfg.fp8.mode != "none":
            raise ValueError(f"{what} is not built together with model.fp8 (mode {cfg.fp8.mode!r}): nothing tests the pair")
        if e.block_norm_gain != "channel":
            raise ValueError(f"{what} is not built together with model.encoder.block_norm_gain {e.block_norm_gain!r}: "
                             "nothing tests the pair")
        if cfg.mtp.enabled:
            raise ValueError(f"{what} is not built together with model.mtp: nothing tests the pair")
        if e.gdn_write_mix != "none" or e.gdn_ccr_spacing:
            # the GDN-side arms (ced/model/gdn_arms.py; merged beside these options, 2026-10-16): their layer calls take
            # R1-entry's static readers and walk (A reads the entry's blocks, CCR hands records between GDN layers);
            # no run combines an arm of each kind, and nothing tests the pair
            raise ValueError(f"{what} is not built together with the GDN-side arms (model.encoder.gdn_write_mix "
                             f"{e.gdn_write_mix!r}, gdn_ccr_spacing {e.gdn_ccr_spacing!r}): nothing tests the pair")


def check_residual_dtype(cfg: ModelConfig) -> None:
    """EncoderConfig.residual_dtype (the stream held at bfloat16 precision; ced/config.py, ced/model/norms.py round_stream): its value, and each pairing
    refused by name where another code path forms the stream and would skip the rounding silently, so that no run says "bf16" and trains float32. "float32"
    (the default) checks nothing more. "bf16" is built for arch "control" with attn_res "none" or "block_entry" (static pseudo-queries) on the plain layer
    call; refused with: the fused entry and write mixes (attn_res_fused: their kernels read the stream in float32), mix_fold, model.fp8 chains and folds
    (the fused norm chains and the G0 folds replace the layer's residual adds), the residual scalars, the loops, the gated second pass and the loop in the
    published form (their later passes add to a stream of their own), the CP head (closed, and it reads the sources)."""
    e = cfg.encoder
    _check_residual_dtype_value(e.residual_dtype)
    if e.residual_dtype == "float32":
        return
    why = "model.encoder.residual_dtype 'bf16' is built for arch 'control' with model.encoder.attn_res 'none' or 'block_entry' (static pseudo-queries) on the plain layer call, not together with "
    refused = {
        "an arch other than 'control'": cfg.arch != "control",
        f"model.encoder.attn_res {e.attn_res!r}": e.attn_res not in ("none", ENTRY),
        f"model.encoder.attn_res_query {e.attn_res_query!r}": e.attn_res_query != "static",
        "model.encoder.attn_res_fused": bool(e.attn_res_fused),
        f"model.encoder.mix_fold {e.mix_fold!r}": bool(e.mix_fold),
        f"model.encoder.layer_scalars {e.layer_scalars!r}": e.layer_scalars != "none",
        f"model.encoder.loops {e.loops}": e.loops != 1,
        f"model.loop.pass2_gate {cfg.loop.pass2_gate!r}": cfg.loop.pass2_gate != "none",
        f"model.loop.recur_layers {list(cfg.loop.recur_layers)!r}": bool(cfg.loop.recur_layers),
        f"model.fp8.chains {list(cfg.fp8.chains)!r}": bool(cfg.fp8.chains),
        f"model.fp8.folds {list(cfg.fp8.folds)!r}": bool(cfg.fp8.folds),
        "model.cp_head": bool(cfg.cp_head.enabled),
    }
    for what, bad in refused.items():
        if bad:
            raise ValueError(why + what)


def new_options(cfg: ModelConfig) -> list[str]:
    """The architecture arms' options that are on (2026-10-16; the module docstring and ced/model/ldm.py), by name:
    [] for every model before them."""
    e = cfg.encoder
    out = [f"model.encoder.attn_res {e.attn_res!r}"] if e.attn_res == LDM else []
    if e.attn_res_heads != 1:
        out.append(f"model.encoder.attn_res_heads {e.attn_res_heads!r}")
    if e.attn_res_query != "static":
        out.append(f"model.encoder.attn_res_query {e.attn_res_query!r}")
    return out


def check_depth_options(cfg: ModelConfig) -> None:
    """The keys of the architecture arms (ced/config.py EncoderConfig: attn_res_heads, attn_res_query,
    attn_res_query_hidden, ldm_groups, ldm_key_dim, ldm_hidden): their values, and each refused by name where it
    would do nothing or is not built, so that no config names a model that was not built as named."""
    e, d = cfg.encoder, cfg.d_model
    H = e.attn_res_heads
    if type(H) is not int or H < 1 or d % H:
        raise ValueError(f"model.encoder.attn_res_heads {H!r}: a whole number of heads that divides d_model {d}")
    if H != 1 and e.attn_res != ENTRY:
        raise ValueError(f"model.encoder.attn_res_heads {H!r} is built for model.encoder.attn_res {ENTRY!r} (R1-entry's "
                         f"readers), not {e.attn_res!r}")
    if e.attn_res_query not in QUERIES:
        raise ValueError(f"model.encoder.attn_res_query {e.attn_res_query!r}: one of {QUERIES}")
    if e.attn_res_query != "static" and e.attn_res != ENTRY:
        raise ValueError(f"model.encoder.attn_res_query {e.attn_res_query!r} is built for model.encoder.attn_res "
                         f"{ENTRY!r} (R1-entry's readers), not {e.attn_res!r}")
    check_fused(cfg)
    hq = e.attn_res_query_hidden
    if type(hq) is not int or hq < 0:
        raise ValueError(f"model.encoder.attn_res_query_hidden {hq!r}: 0 (d_model // 32) or a width")
    if hq and e.attn_res_query != "token":
        raise ValueError(f"model.encoder.attn_res_query_hidden {hq!r} is read with model.encoder.attn_res_query 'token' "
                         "only: leave it 0")
    shape = {"ldm_groups": (e.ldm_groups, 8), "ldm_key_dim": (e.ldm_key_dim, 8), "ldm_hidden": (e.ldm_hidden, 0)}
    if e.attn_res != LDM:
        for key, (value, default) in shape.items():
            if value != default:
                raise ValueError(f"model.encoder.{key} {value!r} is read with model.encoder.attn_res {LDM!r} only: "
                                 f"leave it {default!r}")
        return
    G, K, h = e.ldm_groups, e.ldm_key_dim, e.ldm_hidden
    if type(G) is not int or G < 1 or d % G:
        raise ValueError(f"model.encoder.ldm_groups {G!r}: a whole number of groups that divides d_model {d}")
    if type(K) is not int or K < 1:
        raise ValueError(f"model.encoder.ldm_key_dim {K!r}: a key width of at least 1")
    if type(h) is not int or h < 0:
        raise ValueError(f"model.encoder.ldm_hidden {h!r}: 0 (d_model // 32) or a width")


def check_fused(cfg: ModelConfig) -> None:
    """EncoderConfig.attn_res_fused (ced/kernels/entry_mix.py): a bool; built for R1-entry's own readers ("block_entry", one
    static pseudo-query and one softmax per reader). Refused by name, because the kernel is not that rule or nothing tests
    the pair: any other attn_res; the arms' readers (attn_res_heads, attn_res_query "token"); the write mix with a lambda a
    head (gdn_write_mix "head"); CCR (it hands records between GDN layers); FP8 (its quantiser would read the kernel's bf16
    output, not today's float32 norm output; except with the fused write mix of wired forms, gdn_arms.MX_VARIANTS, whose
    quantised projections read the group read's outputs: gdn_arms "On MXFP8"). Where it runs (CUDA, or the Triton
    interpreter) is checked by the op at its first call. With the write mix's "layer" form (gdn_write_mix "layer", wsl):
    the key also fuses the write mixes (ced/kernels/write_mix.py): in block n >= 2 the entry read and the block's GDN
    layers' write mixes m' run as one kernel each way (EntryWalk._run_arms), so a block may hold at most
    write_mix.MAX_QUERIES - 1 GDN layers (blocks of four layers at 60M and 4B). Under the wsl variant qmix
    (gdn_write_variant "qmix") the same read also forms each of those layers' query mix m'_q: at most
    (write_mix.QMIX_QUERIES - 1) / 2 = 3 GDN layers a block."""
    e = cfg.encoder
    if type(e.attn_res_fused) is not bool:
        raise ValueError(f"model.encoder.attn_res_fused {e.attn_res_fused!r}: true or false")
    check_write_mix_storage(e.write_mix_storage)  # EncoderConfig.write_mix_storage: "float32" (the default) or "bf16", whatever else is on
    if e.write_mix_storage != "float32":  # "bf16" is the explicit opt-in: refused where no extra mix is stored (it would silently do nothing)
        from ced.model.gdn_arms import mix_rc  # noqa: PLC0415

        if not (e.attn_res_fused and e.attn_res == ENTRY and e.gdn_write_mix == "layer"):
            raise ValueError(f"model.encoder.write_mix_storage {e.write_mix_storage!r} is the storage of the fused write mix's extra mixes: it needs "
                             f"model.encoder.attn_res {ENTRY!r} with attn_res_fused true and gdn_write_mix 'layer' (got attn_res {e.attn_res!r}, "
                             f"attn_res_fused {e.attn_res_fused!r}, gdn_write_mix {e.gdn_write_mix!r}); every other path stores no extra mix in this "
                             "dtype. Only 'float32' (the default) is accepted there")
        if mix_rc(e):
            raise ValueError(f"model.encoder.write_mix_storage {e.write_mix_storage!r} with model.encoder.mix_fold {e.mix_fold!r}: the recompute "
                             "form stores no extra mix, so it has no storage to choose. Only 'float32' (the default) is accepted there")
    if not e.attn_res_fused:
        return
    if e.attn_res != ENTRY:
        raise ValueError(f"model.encoder.attn_res_fused is built for model.encoder.attn_res {ENTRY!r} (R1-entry's entry "
                         f"and output reads), not {e.attn_res!r}")
    from ced.model.gdn_arms import MX_VARIANTS

    # FP8 is allowed with the fused WRITE mix of the wired wsl forms only (gdn_arms.check_fp8_forms says the rest): the arms' quantised
    # projections read the group read's bf16 x and m' widened to float32 and the lerps formed from them (gdn_arms "On MXFP8"). E alone
    # (gdn_write_mix none) stays refused with FP8: its quantiser would read the kernel's bf16 output, not today's float32 norm output
    fp8_wired = e.gdn_write_mix == "layer" and e.gdn_write_variant in MX_VARIANTS
    others = {"model.encoder.attn_res_heads": (e.attn_res_heads, 1), "model.encoder.attn_res_query": (e.attn_res_query, "static"),
              "model.encoder.gdn_ccr_spacing": (e.gdn_ccr_spacing, 0), "model.fp8.mode": (cfg.fp8.mode, "none")}
    if fp8_wired:
        del others["model.fp8.mode"]
    for key, (value, default) in others.items():
        if value != default:
            raise ValueError(f"model.encoder.attn_res_fused is not built together with {key} {value!r}: the fused kernel "
                             "is R1-entry's own read (one static query, one softmax), and nothing tests the pair")
    if e.gdn_write_mix not in ("none", "layer"):
        raise ValueError(f"model.encoder.attn_res_fused is not built together with model.encoder.gdn_write_mix "
                         f"{e.gdn_write_mix!r}: the fused write mix (ced/kernels/write_mix.py) is built for 'layer' (one "
                         "lambda a layer, wsl), and nothing tests the pair")
    if e.gdn_write_mix == "layer":
        from ced.kernels.write_mix import MAX_QUERIES, QMIX_QUERIES

        per_block = e.full_attn_interval - 1 if e.full_attn_interval > 0 else 0  # a block's GDN layers (layer_kinds)
        if e.gdn_write_variant == "qmix":  # the entry, each layer's write mix and its query mix (write_mix: QMIX)
            if 2 * per_block + 1 > QMIX_QUERIES:
                raise ValueError(f"model.encoder.attn_res_fused with model.encoder.gdn_write_variant 'qmix': a block of "
                                 f"full_attn_interval {e.full_attn_interval} layers has {per_block} GDN layers, and the fused "
                                 f"write mix reads the entry and at most {(QMIX_QUERIES - 1) // 2} write mixes and their query "
                                 "mixes in one pass")
        elif per_block + 1 > MAX_QUERIES:
            raise ValueError(f"model.encoder.attn_res_fused with model.encoder.gdn_write_mix 'layer': a block of "
                             f"full_attn_interval {e.full_attn_interval} layers has {per_block} GDN layers, and the fused "
                             f"write mix reads the entry and at most {MAX_QUERIES - 1} write mixes in one pass")


def fused_here(res, x: torch.Tensor) -> bool:
    """Whether this layer's reads take the fused kernel (EncoderConfig.attn_res_fused): the key on, and a forward with
    gradients on (training, and activation checkpointing's recomputation); evaluation, scoring, prefill / extend and the
    diagnostics (no_grad) keep today's path."""
    return bool(getattr(res, "fused", False)) and torch.is_grad_enabled()


LDM_CHECKPOINT_MODES = ("none", "layer", "block")  # "ldm": the memory M stays on the device (check_checkpoint_mode)


def checkpoint_modes(cfg: ModelConfig) -> tuple:
    """The activation_checkpoint values built for this model's depth-mix options (check_checkpoint_mode)."""
    return LDM_CHECKPOINT_MODES if cfg.encoder.attn_res == LDM else act_checkpoint.MODES


def check_checkpoint_mode(cfg: ModelConfig, mode: str) -> None:
    """TrainConfig.activation_checkpoint with the architecture arms' options:
    - "layer": each layer under its own checkpoint; LDM keeps the stream and the memory a call takes, the token query's
      opening layer also the stream where the block before ended (`end`);
    - "block": one checkpoint per hybrid group (walk_block): the walk where the block begins (LDM's stream and memory,
      R1-entry's finished blocks and the token query's `end`) is held by reference, the rest recomputed;
    - "host" / "host_overlap": the heads and the token query (whose opening layer moves `end` with the kept streams,
      as R1-entry's closing layer moves its block input); refused for LDM by name: its call keeps the memory M (G K
      floats a token, 8 stream vectors at G = K = 8) and the stream by reference, and moving them is not built."""
    if not new_options(cfg) or mode in checkpoint_modes(cfg):
        return
    raise ValueError(f"activation_checkpoint {mode!r} is not built together with {', '.join(new_options(cfg))}: the "
                     "memory M a layer call keeps would stay on the device and moving it to the host is not built; "
                     f"{', '.join(repr(m) for m in checkpoint_modes(cfg))} are")


def refuse_attn_res(cfg: ModelConfig, what: str) -> None:
    """For code that walks the encoder's layers itself (the decode engine, a cost harness, a hook that reads a layer's
    first argument as the stream): under Block AttnRes a layer's first argument is its block's partial sum and what a
    sublayer reads is formed inside the layer call, so such code would compute or report another model under this
    model's name. ValueError, by name."""
    if cfg.encoder.attn_res != "none":
        raise ValueError(f"{what} walks the encoder's layers itself and does not run model.encoder.attn_res "
                         f"({cfg.encoder.attn_res!r}: Block Attention Residuals, ced/model/attn_res.py): use the model's "
                         "own forward or prefill / extend")
    if cfg.encoder.residual_dtype != "float32":  # the rounding of the stream is in the layer calls (EncoderConfig.residual_dtype), which such code replaces
        raise ValueError(f"{what} walks the encoder's layers itself and does not round the residual stream (model.encoder.residual_dtype "
                         f"{cfg.encoder.residual_dtype!r}, ced/model/norms.py round_stream): use the model's own forward or prefill / extend")


def refuse_new_options(cfg: ModelConfig, what: str) -> None:
    """For the residual diagnostics' term lists and depth weights, written for R1's and R1-entry's single softmax per
    reader: ValueError by name when an architecture arm's option is on (new_options)."""
    if new_options(cfg):
        raise ValueError(f"{what} is built for R1 and R1-entry, not together with {', '.join(new_options(cfg))}")


class DepthQuery(nn.Module):
    """One reader's depth attention: the pseudo-query w (d,), zeros at the start, and the RMSNorm of its keys (the
    paper: "only one RMSNorm and one pseudo-query vector w_l in R^d per layer"; Figure 2's attn_res_proj /
    attn_res_norm and mlp_res_proj / mlp_res_norm). The query is a plain parameter, not an nn.Linear, so the model's
    init_weights (Linear and Embedding weights) neither draws for it nor moves it off zero. The norm is this
    codebase's RMSNorm: gain 1 + g, g zeros at the start and trained."""

    def __init__(self, d: int, eps: float, heads: int = 1):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d))
        self.key_norm = RMSNorm(d, eps)
        # EncoderConfig.attn_res_heads: the softmaxes over the sources, one per contiguous slice of d /heads
        # channels. A plain attribute (no tensor; never in the state dict): 1 is R1's and R1-entry's read.
        self.heads = heads


class TokenQuery(nn.Module):
    """A reader of R1-entry with a token-dependent query (EncoderConfig.attn_res_query "token"; the module docstring):
    the key norm as DepthQuery's, and in place of the (d,) pseudo-query the address network `net` (ldm.AddressNet: an
    RMSNorm, inp Linear(d, hidden), SiLU, head Linear(hidden, d); no gate head), whose head starts at zero weights and
    bias, so that the query is 0 everywhere at the start, as R1-entry's. heads: as DepthQuery's."""

    def __init__(self, d: int, eps: float, hidden: int, heads: int = 1):
        super().__init__()
        self.key_norm = RMSNorm(d, eps)
        self.net = ldm.AddressNet(d, hidden, d, 0, eps, zero_head=True)
        self.heads = heads

    def query_of(self, src: torch.Tensor) -> torch.Tensor:
        """w(src) (B, L, d), under the caller's autocast (as LDM's address networks run in the working dtype)."""
        return self.net(src)[0]


class LayerQueries(nn.Module):
    """The depth attention of one encoder layer's two sublayers (attached to the layer as `attn_res`): pre_mixer
    (None in layer 0, whose mixer reads the embedding alone) and pre_mlp. first: whether the layer opens a block.
    out: None but in the stack's last layer, which also holds the read of the output (over the embedding and every
    block) and ends its call with it."""

    def __init__(self, d: int, eps: float, first: bool, has_mixer_query: bool, last: bool):
        super().__init__()
        self.first = first
        self.pre_mixer = DepthQuery(d, eps) if has_mixer_query else None
        self.pre_mlp = DepthQuery(d, eps)
        self.out = DepthQuery(d, eps) if last else None


def _no_autocast(x: torch.Tensor):
    return torch.autocast(x.device.type, enabled=False)


def depth_mix(sources: list, q: DepthQuery, src: torch.Tensor | None = None) -> torch.Tensor:
    """The fast path's depth attention: h = sum_i alpha_i v_i with alpha = softmax_i(w^T RMSNorm(v_i)) over the
    sources v_i ((B, L, d) each), per position. The logit of a source is taken without a normalised copy of it:

        w^T RMSNorm(v) = sum_d w_d (1 + g_d) v_d rsqrt(mean(v^2) + eps) = (v @ (w (1 + g))) * rsqrt(mean(v^2) + eps)

    which is the key norm's own formula (ced/model/norms.py) with the sum over d taken first. In at least float32,
    autocast off; returned in the sources' dtype. With heads (q.heads > 1) or a token query (TokenQuery: src, the
    stream at the read, is its input; the intermediate arm), depth_mix_general below; R1's and R1-entry's readers
    never reach it."""
    if getattr(q, "heads", 1) != 1 or isinstance(q, TokenQuery):
        return depth_mix_general(sources, q, src)
    v0 = sources[0]
    dtype = compute_dtype(v0)
    with _no_autocast(v0):
        u = q.query.to(dtype) * (1.0 + q.key_norm.weight.to(dtype))
        eps = q.key_norm.eps
        logits = []
        for v in sources:
            vf = v.to(dtype)
            logits.append((vf @ u) * torch.rsqrt(vf.pow(2).mean(-1) + eps))
        alpha = torch.softmax(torch.stack(logits, dim=-1), dim=-1)  # (B, L, n)
        h = alpha[..., 0:1] * v0.to(dtype)
        for i in range(1, len(sources)):
            h = h + alpha[..., i : i + 1] * sources[i].to(dtype)
    return h.to(v0.dtype)


def depth_logits(sources: list, q: DepthQuery) -> torch.Tensor:
    """(B, L, n): the logit of every source as depth_mix forms it, (v @ (w (1 + g))) rsqrt(mean(v^2) + eps), in at least float32,
    autocast off. depth_mix itself is not changed (off is main bit for bit); this is the same arithmetic for the monitor below.
    A query with heads or a token query is not read here."""
    assert getattr(q, "heads", 1) == 1 and not isinstance(q, TokenQuery), "the monitor reads a pseudo-query with one head"
    v0 = sources[0]
    dtype = compute_dtype(v0)
    with _no_autocast(v0):
        u = q.query.to(dtype) * (1.0 + q.key_norm.weight.to(dtype))
        eps = q.key_norm.eps
        out = []
        for v in sources:
            vf = v.to(dtype)
            out.append((vf @ u) * torch.rsqrt(vf.pow(2).mean(-1) + eps))
        return torch.stack(out, dim=-1)


def depth_mix_pair(sources: list, q: DepthQuery, keep: torch.Tensor) -> torch.Tensor:
    """depth_mix over the 2 sources `keep` names (a (2,) int64 index tensor, ascending): h = alpha_a v_a + alpha_b v_b with alpha the
    softmax of the two sources' logits alone (qmix's pseudo-query and key norm); the other sources are neither read nor given
    weight, and the gradient reaches the 2 chosen sources only (the index_select's backward writes zeros to the rest). The static
    restricted query mix of EncoderConfig.gdn_qmix_select_step (ced/model/gdn_arms.py read_source). The sources are stacked (in the
    promoted dtype, a no-op when they share one) and the pair gathered by the index buffer: a fixed shape, no boolean mask, no
    nonzero. The reference form: the fast kernel that would read only the 2 rows is not built. Returned in the first source's
    dtype, as depth_mix."""
    assert getattr(q, "heads", 1) == 1 and not isinstance(q, TokenQuery), "the static pair is built for a pseudo-query with one head"
    dt = sources[0].dtype
    for s in sources[1:]:
        dt = torch.promote_types(dt, s.dtype)
    pair = torch.stack([s.to(dt) for s in sources]).index_select(0, keep)
    return depth_mix([pair[0], pair[1]], q).to(sources[0].dtype)


@torch.no_grad()
def depth_weight_sums(sources: list, q: DepthQuery, keep: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """(the sum over the tokens of a layer's query-mix weights, (n,) float64; the number of tokens B L), for the monitor and the
    selection of EncoderConfig.gdn_qmix_select_step. The weights are the layer's own: the softmax of depth_logits over all n sources
    (keep None), or over the 2 sources `keep` names with zero on the others (the static form: the other sources are not read). Each
    token's weights are formed in at least float32 and summed in float64; the token-mean weight is sums / tokens
    (tools/analysis/qmix_mix_read.py forms its token-mean the same way: the float32 weights, cast to float64, averaged over the tokens).
    Under no_grad on detached sources."""
    srcs = [s.detach() for s in sources]
    n = len(srcs)
    if keep is None:
        alpha = torch.softmax(depth_logits(srcs, q), dim=-1)
        sums = alpha.double().sum(dim=(0, 1))
    else:
        dt = srcs[0].dtype
        for s in srcs[1:]:
            dt = torch.promote_types(dt, s.dtype)
        pair = torch.stack([s.to(dt) for s in srcs]).index_select(0, keep)
        alpha = torch.softmax(depth_logits([pair[0], pair[1]], q), dim=-1)
        sums = torch.zeros(n, dtype=torch.float64, device=alpha.device).index_copy(0, keep, alpha.double().sum(dim=(0, 1)))
    return sums, int(srcs[0].shape[0] * srcs[0].shape[1])


def _query_vector(q, src: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
    """u = w (1 + g) in `dtype`: (d,) for a pseudo-query, (B, L, d) for a token query (from src, the stream at the
    read). g: the key norm's gain. The token query's network runs under the caller's autocast; the product does not."""
    if isinstance(q, TokenQuery):
        assert src is not None, "a token-dependent query needs the stream at the read (src)"
        w = q.query_of(src)
    else:
        w = q.query
    with torch.autocast(w.device.type, enabled=False):
        return w.to(dtype) * (1.0 + q.key_norm.weight.to(dtype))


def depth_mix_general(sources: list, q, src: torch.Tensor | None = None) -> torch.Tensor:
    """depth_mix with H = q.heads softmaxes (one per contiguous slice of d / H channels; MHAR's Eq. 3) and/or a
    token query (the intermediate arm): for slice h,

        logit^(h)_v = sum_{c in h} u_c v_c * rsqrt(mean over ALL d of v^2 + eps),   u = w (1 + g)

    (the key norm over the full row, then sliced), alpha^(h) = softmax over the sources, h_[h] = sum_v alpha^(h)_v v_[h].
    At H = 1 with a pseudo-query it is depth_mix's arithmetic up to the order of the sum over d. In at least float32,
    autocast off; returned in the sources' dtype."""
    v0 = sources[0]
    dtype = compute_dtype(v0)
    H = q.heads
    u = _query_vector(q, src, dtype)
    with _no_autocast(v0):
        eps = q.key_norm.eps
        logits = []
        for v in sources:
            vf = v.to(dtype)
            r = torch.rsqrt(vf.pow(2).mean(-1) + eps)  # (B, L)
            logits.append((vf * u).unflatten(-1, (H, -1)).sum(-1) * r[..., None])  # (B, L, H)
        alpha = torch.softmax(torch.stack(logits, dim=-1), dim=-1)  # (B, L, H, n)
        h = alpha[..., 0:1] * v0.to(dtype).unflatten(-1, (H, -1))
        for i in range(1, len(sources)):
            h = h + alpha[..., i : i + 1] * sources[i].to(dtype).unflatten(-1, (H, -1))
    return h.flatten(-2).to(v0.dtype)


def layer_forward(layer, x: torch.Tensor, ctx, cache, depth: list, keep: bool = False, stream: torch.Tensor | None = None,
                  ccr=None, entry: tuple | None = None):
    """EncoderLayer.forward under Block AttnRes (the fast path; what a compiled layer traces). The paper's Figure 2
    `forward`, for one encoder layer (its mixer sublayer, then its MLP sublayer):

    x: the last source of the mixer's read: the partial sum of the layer's block so far, or, in a block's first layer,
       the finished sum of the block before (in layer 0 the embedding output b_0).
    depth: the sources before it, bottom first: the finished blocks b_0 .. (a list of (B, L, d) tensors).
    Returns (the block's partial sum after this layer, the layer's new cache entry); the stack's last layer returns
    the output's read in its place: the attention over b_0 and every block, the last one just finished.
    entry (R1-entry with the fused write mix only; EntryWalk._run_arms): the opening GDN layer's (block input, normed
    input, write mix m') formed by the block's one group read (ced/kernels/write_mix.py); None in every other call."""
    assert not keep, "a second pass (keep) is not built together with encoder.attn_res (attn_res.check_attn_res)"
    res = getattr(layer, "attn_res", None)
    assert res is not None, "this layer has no depth attention (model.encoder.attn_res 'none'): it takes no `depth`"
    if isinstance(res, EntryLayer):  # model.encoder.attn_res "block_entry": another rule, a function of its own
        if getattr(layer, "gdn_arms", None) is not None:  # the GDN-side arms (ced/model/gdn_arms.py)
            assert stream is None, "the GDN-side arms are not built with token-dependent queries (attn_res.check_attn_res)"
            if entry is not None:  # the fused write mix (attn_res_fused with gdn_write_mix "layer"; EntryWalk._run_arms)
                return entry_layer_forward(layer, x, ctx, cache, depth, ccr=ccr, entry=entry)
            return entry_layer_forward(layer, x, ctx, cache, depth, ccr=ccr)
        assert entry is None, "only a GDN layer with arm A's write mix takes the fused group read"
        return entry_layer_forward(layer, x, ctx, cache, depth, stream)
    assert entry is None, "the fused group read is R1-entry's (attn_res.check_fused)"
    assert ccr is None, "the GDN-side arms are not built with attn_res 'block' or 'ldm' (gdn_arms.check_gdn_arms)"
    assert stream is None, "only R1-entry's readers with token-dependent queries take the stream at the read"
    if isinstance(res, ldm.LdmLayer):  # model.encoder.attn_res "ldm": Linear Depth Memory (ced/model/ldm.py)
        return ldm.layer_forward(layer, x, ctx, cache, depth)
    sources = [*depth, x]
    if res.pre_mixer is None:  # the network's first sublayer: b_0 is its only source, with weight 1
        assert not depth, "layer 0 reads the embedding alone"
        h = x
    else:
        h = depth_mix(sources, res.pre_mixer)
    mixed, new_cache = layer.mixer(layer.mixer_in_norm(h), ctx, cache)
    f = layer.residual_scale * layer.mixer_out_norm(mixed)
    if res.first:  # the block's sum starts with this output; every source so far is a finished block
        done, partial = sources, f.to(x.dtype)
    else:
        done, partial = depth, x + f
    h = depth_mix([*done, partial], res.pre_mlp)
    partial = partial + layer.residual_scale * layer.mlp_out_norm(layer.mlp(layer.mlp_in_norm(h)))
    if res.out is not None:  # the last layer: the final output layer aggregates b_0 and all N block representations
        return depth_mix([*done, partial], res.out), new_cache
    return partial, new_cache


class Walk:
    """One forward of a Block AttnRes stack, layer by layer, on the fast path: for BlockAttnRes.stack and for code
    that must see what each layer's mixer reads (the GDN diagnostics). Holds the finished blocks and the current
    block's partial sum; run(i, ...) calls layer i (compiled if the layers are), mixer_input(i) is the depth
    attention's output that layer i's mixer reads (formed here, eagerly and under no_grad: a second copy of what the
    layer forms inside its call). After the last layer's run, x is the stream after the last sublayer, which the
    final norm reads (the last layer's call ends with the output's read)."""

    def __init__(self, enc, h: torch.Tensor):
        self.enc = enc
        self.done: list = []  # the finished blocks before x: b_0 ..
        self.x = h  # the last source: b_0 at the start, then the partial sum (the finished block at a boundary)

    @torch.no_grad()
    def mixer_input(self, i: int) -> torch.Tensor:
        q = self.enc.layers[i].attn_res.pre_mixer
        return self.x.detach() if q is None else depth_mix([*(v.detach() for v in self.done), self.x.detach()], q)

    def run(self, i: int, ctx, cache, ckpt: bool = False):
        """Layer i, compiled if the layers are. ckpt (activation_checkpoint "layer" / "host" / "host_overlap" in a
        training forward; C2L, ced/model/act_checkpoint.py): the same call under its own checkpoint, keeping the partial
        sum x (moved to the host under "host" and "host_overlap") and references to the finished blocks; in a block's
        first layer x is the finished block b_{n-1}, which the walk holds anyway: kept by reference, never moved (the
        call moves nothing)."""
        layer = self.enc.layers[i]
        if ckpt:
            moved = () if layer.attn_res.first else (self.x,)
            partial, new_cache = act_checkpoint.run_kept(layer, moved, self.x, ctx, cache, depth=self.done)
        else:
            partial, new_cache = layer(self.x, ctx, cache, depth=self.done)
        if layer.attn_res.first:  # x was a finished block (b_0 in layer 0): from here on one of the earlier sources
            self.done = self.done + [self.x]
        self.x = partial
        return new_cache

    def copy(self):
        """A walk in this walk's state that runs on without changing it: the same tensors (references), a list of
        its own (run never changes a list in place either). walk_block runs a checkpointed block on a copy."""
        w = copy.copy(self)
        w.done = list(self.done)
        return w


class EntryLayer(nn.Module):
    """R1-entry ("block_entry"): what one encoder layer holds of the depth attention (attached as `attn_res` to the
    layers that open or close a block; a layer inside a block has no such attribute and runs the base's layer call).
    entry: the reader of its block's input, in the first layer of blocks 2 .. N (None elsewhere: block 1 reads the
    embedding alone). closes: whether the layer is its block's last (its call then returns the block's sum).
    out: the output's read, in the stack's last layer (None elsewhere)."""

    def __init__(self, d: int, eps: float, entry: bool, closes: bool, last: bool, heads: int = 1, query_hidden: int = 0):
        """heads: EncoderConfig.attn_res_heads. query_hidden: 0 for R1-entry's pseudo-queries (DepthQuery); a
        width for token-dependent queries (TokenQuery; EncoderConfig.attn_res_query "token")."""
        super().__init__()

        def reader():
            return TokenQuery(d, eps, query_hidden, heads) if query_hidden else DepthQuery(d, eps, heads)

        self.entry = reader() if entry else None
        self.closes = closes
        self.out = reader() if last else None
        # whether the readers take the stream at the read (a token query): the walk then hands it to them
        self.token = bool(query_hidden)
        # EncoderConfig.attn_res_fused (ced/kernels/entry_mix.py): a plain attribute (no tensor; never in the state dict),
        # set by build; read by entry_layer_forward through fused_here
        self.fused = False
        # The Copy-Paste pointer's readers (model.cp_head.pointer_input "mix"; ced/model/cp_head.py attach_cp_head sets it
        # on the stack's last layer): the layer's call then also returns the last block b_N beside the output's read, so
        # that the stack can hand the read's sources [b_0 .. b_N] to the pointer. A plain attribute (no tensor): False in
        # every other model, and the call is the call it was.
        self.keep_sources = False


def entry_layer_forward(layer, x: torch.Tensor, ctx, cache, depth: list, stream: torch.Tensor | None = None, ccr=None,
                        entry: tuple | None = None):
    """EncoderLayer.forward under R1-entry for a layer that opens or closes a block (the fast path; what a compiled
    layer traces). The layer body is the base's: the plain residual add, mixer then MLP.

    A layer that opens block n >= 2 (layer.attn_res.entry):
        x: the finished block before, b_{n-1};  depth: the ones before it, [b_0 .. b_{n-2}].
        The block's input is the read over [*depth, x] (Eq. 6 at i = 1).
        Returns ((the block's input, the stream after this layer), the layer's new cache entry).
    A layer that closes a block without opening it (layer.attn_res.closes, no entry):
        x: the stream entering the layer;  depth[-1]: the block's input (b_0 for block 1);
        depth[:-1]: in the stack's last layer the finished blocks b_0 .. b_{N-1}, elsewhere nothing.
    A layer that closes its block returns (b_n, new cache entry) with b_n = (the stream after this layer) - (the
    block's input): the sum of the block's sublayer outputs (Eq. 5). The stack's last layer returns the output's read
    in its place: the attention over b_0 and every block. (With blocks of one layer a layer opens and closes its
    block: x and depth as for an opening layer, the return as for a closing one.)
    With token-dependent queries (res.token; EncoderConfig.attn_res_query "token"): an opening layer also takes
    `stream`, the stream where the block before ended, from which its reader's query is formed; a closing layer that
    is not the stack's last returns ((b_n, the stream after this layer), cache): the next opening layer's `stream`.
    The output's read forms its query from the stream after the last layer.
    With res.keep_sources (the stack's last layer under model.cp_head.pointer_input "mix") the last layer returns
    ((the output's read, b_N), cache): the walk keeps [b_0 .. b_N] for the pointer's readers.
    entry (the fused write mix: attn_res_fused with gdn_write_mix "layer", a block's opening GDN layer in a forward with
    gradients; EntryWalk._run_arms): (the block's input, the layer's normed input, the layer's write mix m'), formed with
    the block's other write mixes by one group read (ced/kernels/write_mix.py); the layer reads them in place of its entry
    read, its input norm and its own write mix."""
    res = layer.attn_res
    fused = fused_here(res, x)  # EncoderConfig.attn_res_fused in a forward with gradients (ced/kernels/entry_mix.py)
    if entry is not None:  # the fused write mix: the group read happened in the walk (one kernel each way)
        assert res.entry is not None and not res.closes and getattr(layer, "gdn_arms", None) is not None, \
            "the fused group read serves a block's opening GDN layer with arm A"
        # (qmix: a fourth, the layer's query mix m'_q from the same read; the second round: a fifth, True, when the group read
        # formed the layer's lerps, so the third and fourth are x_w and x_q: write_mix.group_read)
        start, normed, x_write, *x_read = entry
        h = start
        if len(x_read) == 2 and x_read[1] is True:
            mixed, new_cache = layer.gdn_arms_mixer(normed, ctx, cache, None, ccr, x_write=x_write, x_read=x_read[0],
                                                    lerped=True)
        elif x_read:
            mixed, new_cache = layer.gdn_arms_mixer(normed, ctx, cache, None, ccr, x_write=x_write, x_read=x_read[0])
        else:
            mixed, new_cache = layer.gdn_arms_mixer(normed, ctx, cache, None, ccr, x_write=x_write)
    elif fused and res.entry is not None:  # the entry read and the layer's input norm in one kernel each way
        from ced.kernels import entry_mix

        blocks = [*depth, x]
        assert stream is None and not res.token, "the fused entry read is R1-entry's static one (attn_res.check_fused)"
        start, normed = entry_mix.entry_read(blocks, res.entry, layer.mixer_in_norm)
        h = start
        assert getattr(layer, "gdn_arms", None) is None, \
            "a GDN layer with arm A takes the fused group read from the walk (EntryWalk._run_arms; CCR is refused by check_fused)"
        mixed, new_cache = layer.mixer(normed, ctx, cache)
    else:
        if res.entry is not None:  # the block's input: the softmax mix over the embedding and the finished blocks
            blocks = [*depth, x]
            assert (stream is not None) == res.token, "a token-dependent query takes the stream at the read, a pseudo-query none"
            start = depth_mix(blocks, res.entry, stream)
            if layer.stream_bf16:  # residual_dtype "bf16": the block's input x_n^0 is the stream's stored copy of the mix
                start = round_stream(start)
            h = start
        else:  # a block's last layer: the stream comes in as it is, and the block's input with it
            blocks, start, h = depth[:-1], depth[-1], x
        if getattr(layer, "mx_wire", None) is not None and layer.training:  # model.fp8.chains (ced/model/mx_wire.py):
            # the layer's statements below with in_norm / exit_entry on h (in an opening layer the entry read x_n^0)
            arms = getattr(layer, "gdn_arms", None)  # (only as the producer form of the arms' mixes: mx_wire.mix_mixer; check_chains)
            assert arms is None or (layer.mx_wire.mix and ccr is None and res.entry is not None), \
                "the chains are built with the GDN-side arms only as the producer form, in an opening layer (check_chains)"
            mixed = None
            from ced.model import mx_wire

            y, new_cache = mx_wire.layer_forward(layer, h, ctx, cache, write=None if arms is None else blocks)
        elif getattr(layer, "gdn_arms", None) is None:
            mixed, new_cache = layer.mixer(layer.mixer_in_norm(h), ctx, cache)
        else:  # the GDN-side arms (ced/model/gdn_arms.py): a block's opening GDN layer; the write mix's write mix reads the same blocks
            assert res.entry is not None, "a GDN layer with the GDN-side arms opens its block or sits inside it"
            mixed, new_cache = layer.gdn_arms_mixer(layer.mixer_in_norm(h), ctx, cache, blocks, ccr)
    if mixed is not None:
        y = h + layer.residual_scale * layer.mixer_out_norm(mixed)
        if layer.stream_bf16:  # residual_dtype "bf16": the stream between the two sublayers is stored at bfloat16
            y = round_stream(y)
        y = y + layer.residual_scale * layer.mlp_out_norm(layer.mlp(layer.mlp_in_norm(y)))
        if layer.stream_bf16 and not res.closes:  # (a block's last sum is not stored: the block sum below is formed from it in the same epilogue)
            y = round_stream(y)
    if not res.closes:
        return (start, y), new_cache
    block = y - start  # b_n: what the block's sublayers added to its input
    if res.out is not None:  # the last layer: the final output layer aggregates b_0 and all N block representations
        if fused:  # the output read as the same kernel without a norm (the model's final norm stays outside)
            from ced.kernels import entry_mix

            return entry_mix.output_read([*blocks, block], res.out), new_cache
        out = depth_mix([*blocks, block], res.out, y if res.token else None)
        if res.keep_sources:  # the CP pointer's readers read the same sources (model.cp_head.pointer_input "mix")
            return (out, block), new_cache
        return out, new_cache
    if res.token:  # the stream where this block ends: the query source of the next block's entry read
        return (block, y), new_cache
    return block, new_cache


class EntryWalk:
    """Walk for R1-entry: one forward of the stack, layer by layer, on the fast path (BlockAttnRes.stack, the GDN
    diagnostics). done: the finished blocks, b_0 first. x: the stream inside a block (b_0 before layer 0); between a
    block's end and the next block's first layer it is the block just finished, and after the last layer the output's
    read, which the final norm reads. start: the current block's input (b_0 in block 1), which the block's last layer
    takes. mixer_input(i): what layer i's mixer reads: the entry read in a block's first layer (a second, eager copy
    under no_grad of what the layer forms inside its call), else the stream."""

    def __init__(self, enc, h: torch.Tensor):
        self.enc = enc
        self.done: list = [h]  # b_0, the source, as the embedding stage formed it (float32 under residual_dtype "bf16" too)
        if enc.stream_bf16:  # residual_dtype "bf16": the stream's stored copy of b_0, block 1's input x_1^0 (b_1 is formed from this copy)
            h = round_stream(h)
        self.x = h
        self.start = h
        # the GDN-side arms (ced/model/gdn_arms.py): the last GDN layer's CCR record (None before the first, and in every
        # model without the arms, where nothing reads it)
        self.ccr = None
        # with token-dependent queries (EncoderConfig.attn_res_query "token"): the stream where the last finished block
        # ended, the query source of the next block's entry read (None before the first block ends, and always None
        # with pseudo-queries)
        self.end = None
        # the output read's sources [b_0 .. b_N] after the last layer, when that layer keeps them (EntryLayer.keep_sources;
        # the CP pointer's readers); None otherwise
        self.sources = None
        # the fused write mix (attn_res_fused with gdn_write_mix "layer"; _run_arms): {layer index: its write mix m'}, the
        # inner GDN layers' share of the block's group read, formed at the block's opening layer and handed to each layer
        # once. Always replaced, never changed in place (copy() shares it). Empty in every model without the pair.
        self.writes: dict = {}
        # the wsl variant qmix on the fused write mix: {layer index: its query mix m'_q} from the same group read, handed
        # with its m' and replaced the same way (never changed in place). Empty in every other model.
        self.reads: dict = {}

    @torch.no_grad()
    def mixer_input(self, i: int) -> torch.Tensor:
        res = getattr(self.enc.layers[i], "attn_res", None)
        if res is None or res.entry is None:
            return self.x.detach()
        read = depth_mix([v.detach() for v in self.done], res.entry, None if self.end is None else self.end.detach())
        return round_stream(read) if self.enc.stream_bf16 else read

    def _moved(self, *ts) -> tuple:
        """C2L under "host" and "host_overlap": the kept inputs of a layer call that go to the host: those of ts that
        are not a finished block (b_0 as block 1's stream and input), which the walk holds on the device anyway."""
        return tuple(t for t in ts if not any(t is b for b in self.done))

    def run(self, i: int, ctx, cache, ckpt: bool = False):
        """Layer i, compiled if the layers are. ckpt (activation_checkpoint "layer" / "host" / "host_overlap" in a
        training forward; C2L, ced/model/act_checkpoint.py): the same call under its own checkpoint. It keeps the
        stream x of a layer inside a block and of a block's closing layer, and the closing layer's block input `start`
        (both moved to the host under "host" and "host_overlap": one call, two kept inputs; _Closing makes start a
        top-level argument), and references to the finished blocks; the input of a block's opening layer is a finished
        block: kept by reference, never moved."""
        layer = self.enc.layers[i]
        res = getattr(layer, "attn_res", None)
        if getattr(layer, "gdn_arms", None) is not None:  # the GDN-side arms (ced/model/gdn_arms.py): a GDN layer
            return self._run_arms(layer, res, ctx, cache, ckpt, i)
        if res is None:  # inside a block: the base's own layer call, x <- x + f(x)
            if ckpt:
                self.x, new_cache = act_checkpoint.run_kept(layer, self._moved(self.x), self.x, ctx, cache)
            else:
                self.x, new_cache = layer(self.x, ctx, cache)
            return new_cache
        if res.entry is not None:  # opens a block: reads the finished blocks (the last of them as x)
            if res.token:  # a token-dependent query: the stream where the block before ended goes in too
                if ckpt:  # `end` is kept by the checkpoint alone (the walk lets go of it below): moved under the host modes
                    out, new_cache = act_checkpoint.run_kept(_Opening(layer), self._moved(self.end), self.done[-1],
                                                             self.end, ctx, cache, self.done[:-1])
                else:
                    out, new_cache = layer(self.done[-1], ctx, cache, depth=self.done[:-1], stream=self.end)
            elif ckpt:
                out, new_cache = act_checkpoint.run_kept(layer, (), self.done[-1], ctx, cache, depth=self.done[:-1])
            else:
                out, new_cache = layer(self.done[-1], ctx, cache, depth=self.done[:-1])
            self.end = None
            if not res.closes:
                self.start, self.x = out
                return new_cache
        else:  # closes a block it did not open: the stream, the block's input, and for the output's read the blocks
            blocks = self.done if res.out is not None else []
            if ckpt:
                out, new_cache = act_checkpoint.run_kept(_Closing(layer), self._moved(self.x, self.start), self.x,
                                                         self.start, ctx, cache, blocks)
            else:
                out, new_cache = layer(self.x, ctx, cache, depth=[*blocks, self.start])
        if res.out is not None and res.keep_sources:  # (the output's read, b_N): the read's sources for the CP pointer
            out, last = out
            self.sources = [*self.done, last]
        if res.out is None and res.token:  # (the finished block, the stream where it ended)
            out, self.end = out
        if res.out is None:  # `out` is the finished block: from here on one of the sources
            self.done = self.done + [out]
        self.x, self.start = out, None
        return new_cache

    def _run_arms(self, layer, res, ctx, cache, ckpt: bool, i: int):
        """run() for a GDN layer with the GDN-side arms (layer.gdn_arms; ced/model/gdn_arms.py): the same calls, with the
        write mix's sources (the finished blocks b_0 .. which a block's opening layer reads as its entry's) and arm
        CCR's record of the previous GDN layer; the layer's own record is taken out of its cache entry for the next GDN
        layer (CCR v2: the walk holds the last record written and hands it only to a reader, gdn_arms.record_in
        /hand_record; under "previous" that is v1's chain). A GDN layer opens its block or sits inside it (it never
        closes one: a block ends with its global layer). ckpt (activation_checkpoint "layer" / "host" / "host_overlap"
        in a training forward; gdn_arms' docstring): the same call under its own checkpoint. It keeps the stream of a
        layer inside a block (moved to the host under "host" / "host_overlap", as the base's), and references to the
        finished blocks and to the previous GDN layer's record; an opening layer's input is a finished block (kept by
        reference, nothing moved). The layer's own record is in its call's output and is taken out here, after the call:
        the recomputation in the backward forms it again for its own saved tensors and hands it to no one. The fused
        write mix (model.encoder.attn_res_fused with gdn_write_mix "layer"; ced/kernels/write_mix.py), in a forward with
        gradients: at a block's opening GDN layer i the walk forms the block's reads in ONE group read over the finished
        blocks: the entry read with layer i's input norm (the block's input and layer i's normed input), layer i's write
        mix m' and the write mixes m' of the block's other GDN layers, each through its layer's input norm. Layer i
        takes its three as `entry`; each inner layer takes its m' as `x_write` in place of the sources (self.writes,
        handed once). The group read runs outside the layers' own checkpoints (under "layer" and the host modes its
        outputs are kept as the calls' inputs; under "block" it is inside the block's checkpoint and is recomputed with
        it)."""
        from ced.model.gdn_arms import fused_write_here, hand_record, record_in

        arms = layer.gdn_arms
        rec = record_in(arms, self.ccr)  # CCR v2: a layer that reads none is called without the walk's record
        if res is None:  # inside a block: x <- x + f(x), the write mix over the finished blocks
            if i in self.writes:  # its m' from the block's group read (the fused write mix)
                kw = {"x_write": self.writes[i], "ccr": rec}
                self.writes = {k: v for k, v in self.writes.items() if k != i}
                if i in self.reads:  # qmix: its m'_q from the same group read
                    kw["x_read"] = self.reads[i]
                    self.reads = {k: v for k, v in self.reads.items() if k != i}
            else:
                assert not fused_write_here(arms), "the fused write mix: the block's opening layer forms this layer's m'"
                kw = {"write": self.done, "ccr": rec}
            if ckpt:
                self.x, new_cache = act_checkpoint.run_kept(layer, self._moved(self.x), self.x, ctx, cache, **kw)
            else:
                self.x, new_cache = layer(self.x, ctx, cache, **kw)
        else:  # opens block n >= 2: its entry read and its write mix over the same finished blocks
            assert res.entry is not None and not res.closes, "a GDN layer with the arms opens its block or sits inside it"
            assert not res.token, "the GDN-side arms are not built with token-dependent queries (attn_res.check_attn_res)"
            kw = {"ccr": rec}
            if fused_write_here(arms):  # the block's group read: one kernel each way
                kw["entry"] = self._group_read(layer, res, i)
            if ckpt:
                (self.start, self.x), new_cache = act_checkpoint.run_kept(layer, (), self.done[-1], ctx, cache,
                                                                          depth=self.done[:-1], **kw)
            else:
                (self.start, self.x), new_cache = layer(self.done[-1], ctx, cache, depth=self.done[:-1], **kw)
        self.ccr, new_cache = hand_record(arms, self.ccr, new_cache)
        return new_cache

    def _group_read(self, layer, res, i: int) -> tuple:
        """The fused write mix's group read at block-opening layer i (_run_arms): sets self.writes for the block's other GDN
        layers with the fused write mix and returns layer i's (block input, normed input, write mix m'). The NQ normed outputs
        leave the op as one (NQ, B, L, D) tensor and are handed out through ONE unbind: its backward stacks the layers'
        gradients once (one read and one write of the tensor). Indexing it per layer (y[j]) would give each layer a
        SelectBackward that zero-fills a whole (NQ, B, L, D) gradient, then three full-size accumulations: 15 tensor-sized
        passes a block where the unbind's stack is 2. Values and gradients are the same. Under the wsl
        variant qmix the same read also forms each layer's query mix m'_q: queries [entry, write i, write inner. query i,
        query inner..] (NQ 1 + 2 G), the query mixes through the same layers' input norms; layer i's comes back as the fourth
        element (block input, normed input, m', m'_q), and the inner layers' through self.reads beside self.writes. The second
        round (qmix with a forward launch r2; write_mix.group_read): the outputs come as NQ tensors (no unbind, no stack in
        the backward), and the group read also takes layer i's write_lambda and qmix_lambda and forms its x_w and x_q (slots 1
        and G + 1), so layer i's entry is (block input, normed input, x_w, x_q, True) and it forms no addcmul
        (gdn_arms.write_inputs's `lerped`); the inner layers are as before."""
        from ced.kernels import write_mix

        S = self.enc.attn_res.block_layers
        inner = [k for k in range(i + 1, i + S) if getattr(getattr(self.enc.layers[k], "gdn_arms", None), "fused_write", False)]
        layers = [layer] + [self.enc.layers[k] for k in inner]
        readers = [res.entry] + [m.gdn_arms.write_query for m in layers]
        norms = [layer.mixer_in_norm] + [m.mixer_in_norm for m in layers]
        qmix = layer.gdn_arms.variant == "qmix"
        if qmix:
            assert all(m.gdn_arms.variant == "qmix" and m.gdn_arms.qmix_query is not None for m in layers)
            readers += [m.gdn_arms.qmix_query for m in layers]
            norms += [m.mixer_in_norm for m in layers]
        x0 = self.done[0]
        # the extra mixes' storage: EncoderConfig.write_mix_storage (set on the layer's arms by gdn_arms.build; "float32" unless the key says "bf16")
        hx = write_mix.hx_dtype_for(layer.gdn_arms.write_mix_storage, x0)
        if getattr(layer.gdn_arms, "hx_source", False):
            # EncoderConfig.write_mix_row_term "pdp": the extra queries' mixes in the sources' dtype (write_mix.group_read's hx_dtype). That is what write_mix_storage
            # "float32" (the default since 2026-10-26) stores as well; the two disagree only with write_mix_storage "bf16", which gdn_arms.check_write_mix_row_term refuses.
            hx = x0.dtype
        if getattr(layer.gdn_arms, "mix_rc", False):  # the recompute group read (gdn_arms.mix_rc): the extra mixes are recomputed in the backward
            from ced.kernels import write_mix_rc

            if qmix:  # the opening layer's lerps in the kernel, as the second round's group read has them
                start, y, lerped = write_mix_rc.group_read_rc(self.done, readers, norms, lerp=(layer.gdn_arms.write_lambda, layer.gdn_arms.qmix_lambda))
            else:  # (wsl: its opening layer forms its lerp in write_inputs)
                start, y = write_mix_rc.group_read_rc(self.done, readers, norms)  # (y: NQ tensors)
                lerped = False
        elif qmix and write_mix.is_r2(x0.shape[-1], len(self.done), len(readers), x0.device):
            arms = layer.gdn_arms
            start, y, lerped = write_mix.group_read(self.done, readers, norms, hx_dtype=hx, lerp=(arms.write_lambda, arms.qmix_lambda))
        else:  # (the call as before the second round)
            start, y = write_mix.group_read(self.done, readers, norms, hx_dtype=hx)
            lerped = False
        ys = write_mix.split(y)
        G = len(layers)
        self.writes = {k: ys[2 + n] for n, k in enumerate(inner)}
        if qmix:
            self.reads = {k: ys[G + 2 + n] for n, k in enumerate(inner)}
            if lerped:
                return start, ys[0], ys[1], ys[G + 1], True
            return start, ys[0], ys[1], ys[G + 1]
        return start, ys[0], ys[1]

    copy = Walk.copy  # x, start, end, ccr and the finished blocks: the same tensors; the list a list of its own


class _Opening:
    """R1-entry's opening layer with a token-dependent query as one checkpointed call (EntryWalk.run with ckpt): calls
    layer(x, ctx, cache, depth=blocks, stream=end), the call the walk makes without checkpointing, with `end` (the
    stream where the block before ended) a top-level argument of the checkpoint. ckpt_layer: the layer."""

    __slots__ = ("ckpt_layer",)

    def __init__(self, layer):
        self.ckpt_layer = layer

    def __call__(self, x, end, ctx, cache, blocks):
        return self.ckpt_layer(x, ctx, cache, depth=blocks, stream=end)


class _Closing:
    """R1-entry's closing layer as one checkpointed call under C2L (EntryWalk.run with ckpt): calls
    layer(x, ctx, cache, depth=[*blocks, start]), the call the walk makes without checkpointing, with the block's input
    `start` a top-level argument of the checkpoint (a list argument is held by reference, so start inside `depth`
    could not move to the host under "host"). ckpt_layer: the layer (the memory trace of a training step books the
    call under the layer's name)."""

    __slots__ = ("ckpt_layer",)

    def __init__(self, layer):
        self.ckpt_layer = layer

    def __call__(self, x, start, ctx, cache, blocks):
        return self.ckpt_layer(x, ctx, cache, depth=[*blocks, start])


def _monitor(walk, i: int, rms: list | None, stats: dict | None) -> None:
    """The stack's monitoring before layer i (BlockAttnRes.stack): the RMS of what the layer's mixer reads and, at a
    global layer, its max |activation|, from Walk.mixer_input (a second, eager copy of the read under no_grad)."""
    from ced.model.encoder import record_activation_max, record_stream_rms  # (encoder.py imports this module)

    layer = walk.enc.layers[i]
    if rms is not None or (stats is not None and layer.kind == "attention"):
        read = walk.mixer_input(i)
        if stats is not None and layer.kind == "attention":
            record_activation_max(stats, f"enc{i}", read)
        if rms is not None:
            record_stream_rms(rms, read)
    rec = getattr(walk.enc, "qmix_record", None)
    if rec is not None:  # EncoderConfig.gdn_qmix_select_step: the token-sums of each query mix's weights, on a monitored forward only
        from ced.model.gdn_arms import record_query_weights

        record_query_weights(rec, i, layer, walk.done)


def walk_block(walk, ctx, i0: int, rms: list | None, stats: dict | None, first):
    """One block of the fast path under activation_checkpoint "block" (ced/model/act_checkpoint.py): layers
    i0 .. i0 + block_layers - 1 run on a copy of `walk`, which is in the state where the block begins, with the stack's
    own calls (Walk.run / EntryWalk.run, compiled if the layers are) and its monitoring (while first.on). The
    function a checkpointed block runs, in the forward and again in the backward: `walk` itself is never changed, so
    the recomputation starts from the same state. What the checkpoint holds through `walk`: references to the finished
    blocks b_0 .. and to the block's first source (R1: b_{n-1}; R1-entry: the same, and b_0 as block 1's input), the
    tensors the walk holds anyway; nothing of the block's partial sums, its entry read or its inner stream.
    Returns (the walk after the block, the layers' new cache entries)."""
    w = walk.copy()
    caches = []
    for i in range(i0, i0 + w.enc.attn_res.block_layers):
        if first.on:
            _monitor(w, i, rms, stats)
        caches.append(w.run(i, ctx, None))
    return w, caches


def block_attn_res(blocks: list, partial_block: torch.Tensor | None, q: DepthQuery, src: torch.Tensor | None = None):
    """The paper's Figure 2, `block_attn_res`, line for line (its proj.weight.squeeze() is q.query, its norm
    q.key_norm), in at least float32 with autocast off. blocks: the finished block representations, b_0 first;
    partial_block: the current block's partial sum, or None before the block's first sublayer (Eq. 6, i = 1).
    Returns (h (B, L, d) in the sources' dtype, alpha (n, B, L)).
    With heads or a token query (the architecture arms on R1-entry): block_attn_res_heads, MHAR's Figure 3 route with
    the key norm over the full row (its Eq. 3); src is the token query's input."""
    if getattr(q, "heads", 1) != 1 or isinstance(q, TokenQuery):
        return block_attn_res_heads(blocks, partial_block, q, src)
    sources = blocks + ([] if partial_block is None else [partial_block])
    dtype = compute_dtype(sources[0])
    with _no_autocast(sources[0]):
        V = torch.stack([v.to(dtype) for v in sources])  # [N+1, B, T, D]
        K = q.key_norm(V)
        logits = torch.einsum("d, n b t d -> n b t", q.query.to(dtype), K)
        alpha = logits.softmax(0)
        h = torch.einsum("n b t, n b t d -> b t d", alpha, V)
    return h.to(sources[0].dtype), alpha


def block_attn_res_heads(blocks: list, partial_block: torch.Tensor | None, q, src: torch.Tensor | None = None):
    """The reference of the multi-head read and of the token query: MHAR's Figure 3 `route` line for line (arXiv
    2607.27230: "q = proj.weight.view(self.H, D // self.H)", "Vh = V.view(N, B, T, self.H, D // self.H)", logits by
    einsum, "alpha = logits.softmax(0)", the mix by einsum), with the key norm applied to the full row V and then sliced
    (its Eq. 3: RMSNorm(s_i)_[h]); a token query's w(src) (B, T, D) in place of the (D,) query. In at least float32,
    autocast off. Returns (h (B, L, d) in the sources' dtype, alpha (n, B, L, H))."""
    sources = blocks + ([] if partial_block is None else [partial_block])
    dtype = compute_dtype(sources[0])
    H = q.heads
    w = q.query_of(src) if isinstance(q, TokenQuery) else q.query
    with _no_autocast(sources[0]):
        V = torch.stack([v.to(dtype) for v in sources])  # [N, B, T, D]
        N, B, T, D = V.shape
        K = q.key_norm(V).view(N, B, T, H, D // H)
        Vh = V.view(N, B, T, H, D // H)
        if isinstance(q, TokenQuery):
            qh = w.to(dtype).view(B, T, H, D // H)
            logits = torch.einsum("b t h e, n b t h e -> n b t h", qh, K)
        else:
            qh = w.to(dtype).view(H, D // H)
            logits = torch.einsum("h e, n b t h e -> n b t h", qh, K)
        alpha = logits.softmax(0)
        mixed = torch.einsum("n b t h, n b t h e -> b t h e", alpha, Vh)
    return mixed.reshape(B, T, D).to(sources[0].dtype), alpha


def reference_stack(enc, h: torch.Tensor, ctx, caches: list | None, record: list | None = None, skip=()):
    """The reference: the stack under Block AttnRes written from the paper's section 3.2, one sublayer after the
    other, eager, with block_attn_res above. Returns (the stream after the last sublayer, the layers' new caches).
    record (a list): receives one dict per sublayer, in order, {"reader": "L<i>.mixer" | "L<i>.mlp", "layer": i,
    "block": n (1-based), "sources": the labels of its sources, "alpha": (n, B, L) or None for the first sublayer,
    "input": h, "output": f}, and a last one for the output's read ({"reader": "out", ..., "output": None}).
    skip (the diagnostics' "skip one layer"; layer indices, or (layer, sublayer) pairs with sublayer 0 the mixer and 1
    the MLP): a skipped sublayer is taken out of the stack: it reads nothing and adds nothing to its block's sum, and
    every other sublayer reads what Eq. 6 gives it in the stack without it (after a skipped first layer of a block,
    the next sublayer reads the finished blocks alone). A layer index skips both sublayers. The cache entry of a layer
    whose mixer is skipped is None. A block needs one sublayer that runs.
    R1-entry ("block_entry") has a reference of its own, entry_reference_stack below: this function hands over."""
    if enc.attn_res.mode == ENTRY:
        return entry_reference_stack(enc, h, ctx, caches, record, skip)
    if enc.attn_res.mode == LDM:  # Linear Depth Memory: the released code's reference (ced/model/ldm.py)
        assert not skip, "sublayers are not taken out of the stack under model.encoder.attn_res 'ldm'"
        return ldm.reference_stack(enc, h, ctx, caches, record)
    S = enc.attn_res.block_layers
    caches = caches if caches is not None else [None] * len(enc.layers)
    gone = set()
    for x in skip:
        gone |= {(x, 0), (x, 1)} if isinstance(x, int) else {(int(x[0]), int(x[1]))}
    blocks = [h]  # b_0 = h_1, the token embedding: always a source
    partial = None  # b_n^i: the current block's partial sum (None before its first sublayer)
    new_caches = []

    def labels():
        return ["embed"] + [f"block{k}" for k in range(1, len(blocks))] + ([] if partial is None else ["partial"])

    for i, (layer, cache) in enumerate(zip(enc.layers, caches)):
        if i and i % S == 0:  # a block is finished: it joins the sources, and a new partial sum starts
            assert partial is not None, "every sublayer of a block was skipped"
            blocks.append(partial)
            partial = None
        res, n = layer.attn_res, i // S + 1
        new_cache = None
        if (i, 0) not in gone:
            # the mixer sublayer (the paper's attention layer)
            srcs = labels()
            if res.pre_mixer is None:  # the first sublayer of the network: its input is the token embedding
                hin, alpha = blocks[0], None
            else:
                hin, alpha = block_attn_res(blocks, partial, res.pre_mixer)  # (one source: its weight is 1)
            mixed, new_cache = layer.mixer(layer.mixer_in_norm(hin), ctx, cache)
            f = layer.residual_scale * layer.mixer_out_norm(mixed)
            if record is not None:
                record.append({"reader": f"L{i}.mixer", "layer": i, "block": n, "sources": srcs, "alpha": alpha, "input": hin, "output": f})
            partial = f.to(h.dtype) if partial is None else partial + f
        if (i, 1) not in gone:
            # the MLP sublayer
            srcs = labels()
            hin, alpha = block_attn_res(blocks, partial, res.pre_mlp)
            f = layer.residual_scale * layer.mlp_out_norm(layer.mlp(layer.mlp_in_norm(hin)))
            if record is not None:
                record.append({"reader": f"L{i}.mlp", "layer": i, "block": n, "sources": srcs, "alpha": alpha, "input": hin, "output": f})
            partial = f.to(h.dtype) if partial is None else partial + f  # (None: the block's mixer before it was skipped)
        new_caches.append(new_cache)
    assert partial is not None, "every sublayer of a block was skipped"
    blocks.append(partial)
    partial = None
    srcs = labels()
    out, alpha = block_attn_res(blocks, None, enc.layers[-1].attn_res.out)  # the final output layer: all N blocks and b_0
    if record is not None:
        record.append({"reader": "out", "layer": None, "block": None, "sources": srcs, "alpha": alpha, "input": out, "output": None})
    return out, new_caches


def entry_reference_stack(enc, h: torch.Tensor, ctx, caches: list | None, record: list | None = None, skip=()):
    """The reference of R1-entry ("block_entry"): the form of the module docstring line for line, one sublayer after
    the other, eager, with block_attn_res (the paper's pseudocode) for the two kinds of read. Returns (the stream the
    final norm reads, the layers' new caches).
    record (a list): one dict per sublayer, in order, with reference_stack's keys: "reader" ("L<i>.mixer" |
    "L<i>.mlp"), "layer", "block", "sources", "alpha", "input" (what the sublayer reads), "output" (its f), and a last
    one for the output's read ("out"). "alpha" is (n, B, L) for the sublayer that reads a block's entry read (the
    first sublayer of blocks 2 .. N; its "sources" are "embed", "block<k>") and None for every other: the first
    sublayer of the network reads the embedding alone (sources ["embed"]), a sublayer inside a block reads the running
    stream (sources ["stream"]).
    skip (reference_stack's): a skipped sublayer reads nothing and adds nothing, neither to the stream nor to its
    block's sum. The block's entry read is formed all the same (it is the block's input) and the block's first
    sublayer that runs reads it. A block needs one sublayer that runs."""
    S = enc.attn_res.block_layers
    caches = caches if caches is not None else [None] * len(enc.layers)
    gone = set()
    for x in skip:
        gone |= {(x, 0), (x, 1)} if isinstance(x, int) else {(int(x[0]), int(x[1]))}
    blocks = [h]  # b_0 = the token embedding: always a source
    partial = None  # the current block's sum so far (Eq. 5); None before its first sublayer
    stream = None  # x_n^i: the plain residual stream inside the current block
    read = None  # (sources, alpha) of the block's entry read, until a sublayer has read it
    # EncoderConfig.residual_dtype "bf16": the stream is stored at bfloat16 wherever the fused design stores it (round_stream): the block's input x_n^0 (the
    # stored copy of the entry mix; of b_0 in block 1) and the stream after each sublayer's add, except the block's last add (its sum goes straight into the
    # block sum). The block's sum is then the stream difference b_n = x_n^S - x_n^0 (Eq. 5's sum of the outputs is not what the fused epilogue forms: it
    # forms the difference from the stored x_n^0), in float32 like every source; `start` is x_n^0. Off: the sum of the outputs, as before.
    bf = enc.stream_bf16
    start = None
    new_caches = []
    for i, (layer, cache) in enumerate(zip(enc.layers, caches)):
        n = i // S + 1
        if i % S == 0:  # block n begins
            if i:  # the block before is finished: it joins the sources
                assert partial is not None, "every sublayer of a block was skipped"
                blocks.append(stream - start if bf else partial)
                partial = None
            srcs = ["embed"] + [f"block{k}" for k in range(1, len(blocks))]
            if n == 1:  # the first block reads the token embedding alone: weight 1, no query
                stream, read = blocks[0], (srcs, None)
            else:  # Eq. 6 at i = 1: the softmax mix over b_0 .. b_{n-1}, with the block's own query and key norm
                # (a token query is formed from the stream where block n - 1 ended: `stream` before it is replaced)
                stream, alpha = block_attn_res(blocks, None, layer.attn_res.entry, stream)
                read = (srcs, alpha)
            if bf:
                stream = start = round_stream(stream)
        new_cache = None
        if (i, 0) not in gone:
            # the mixer sublayer: the plain residual add
            mixed, new_cache = layer.mixer(layer.mixer_in_norm(stream), ctx, cache)
            f = layer.residual_scale * layer.mixer_out_norm(mixed)
            if record is not None:
                srcs, alpha = read if read is not None else (["stream"], None)
                record.append({"reader": f"L{i}.mixer", "layer": i, "block": n, "sources": srcs, "alpha": alpha, "input": stream, "output": f})
            read = None
            stream = stream + f
            if bf:
                stream = round_stream(stream)
            partial = f.to(h.dtype) if partial is None else partial + f
        if (i, 1) not in gone:
            # the MLP sublayer: the plain residual add
            f = layer.residual_scale * layer.mlp_out_norm(layer.mlp(layer.mlp_in_norm(stream)))
            if record is not None:
                srcs, alpha = read if read is not None else (["stream"], None)
                record.append({"reader": f"L{i}.mlp", "layer": i, "block": n, "sources": srcs, "alpha": alpha, "input": stream, "output": f})
            read = None
            stream = stream + f
            if bf and i % S != S - 1:  # (the block's last add is not stored)
                stream = round_stream(stream)
            partial = f.to(h.dtype) if partial is None else partial + f
        new_caches.append(new_cache)
    assert partial is not None, "every sublayer of a block was skipped"
    blocks.append(stream - start if bf else partial)
    srcs = ["embed"] + [f"block{k}" for k in range(1, len(blocks))]
    out, alpha = block_attn_res(blocks, None, enc.layers[-1].attn_res.out, stream)  # the output's read: b_0 and all N blocks
    if record is not None:
        record.append({"reader": "out", "layer": None, "block": None, "sources": srcs, "alpha": alpha, "input": out, "output": None})
    if enc.layers[-1].attn_res.keep_sources:  # the CP pointer's readers read the same sources (model.cp_head.pointer_input "mix")
        from ced.model.cp_head import SOURCES_KEY

        ctx.memo[SOURCES_KEY] = list(blocks)
    return out, new_caches


class BlockAttnRes(nn.Module):
    """The encoder's Block AttnRes (Encoder.attn_res; None with EncoderConfig.attn_res "none"): the block layout and
    the stack. It holds no tensor: every depth attention is on a layer (layer.attn_res, a LayerQueries; the output's
    on the last layer), where a compiled layer finds it as its own parameters. mode: the config's value, "block" (R1)
    or "block_entry" (R1-entry: an EntryLayer on the layers that open or close a block, and an EntryWalk)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.mode = cfg.encoder.attn_res  # "block", "block_entry" or "ldm": a plain attribute, read from the config at build
        self.block_layers = cfg.encoder.full_attn_interval  # layers per block: one hybrid group
        self.n_blocks = cfg.encoder.n_layers // self.block_layers
        # the architecture arms' options that are on (new_options; [] for R1 and R1-entry): a plain attribute
        self.new_options = new_options(cfg)
        # the activation_checkpoint values built for these options and for the GDN-side arms (check_checkpoint_mode,
        # gdn_arms.check_checkpoint_mode): plain tuples, read by stack's guards
        self.checkpoint_modes = checkpoint_modes(cfg)
        from ced.model import gdn_arms

        self.arms_checkpoint_modes = (gdn_arms.CCR_CHECKPOINT_MODES if gdn_arms.ccr_on(cfg)
                                      else gdn_arms.CHECKPOINT_MODES)
        # "fast" (every forward) or "reference" (reference_stack): a plain attribute, never in the state dict or the
        # config; tests and diagnostics set it.
        self.impl = "fast"
        # sublayers taken out of the stack (reference_stack's skip), for the diagnostics' "skip one layer": a plain
        # attribute like impl, set through Skipped below. Only the reference stack reads it; the fast path refuses it.
        self.skip = ()

    def stack(self, enc, h: torch.Tensor, ctx, caches: list | None, stats: dict | None = None):
        """Encoder._stack under Block AttnRes: (the stream after the last sublayer, the layers' new caches).
        Monitoring, as in _stack: with enc.stream_rms a list, the RMS of what each layer's mixer reads (entry 0: the
        embedding output, as in the base) and of the stream after the last sublayer (the output's read, which the
        last layer's call returns): n_layers + 1 values; with stats,
        the max |activation| of what each global layer's mixer reads, "act_max/enc<i>". Both read Walk.mixer_input,
        a second, eager copy of the read under no_grad, on monitored forwards only.
        Activation checkpointing (ced/model/act_checkpoint.py): "block" runs each block of the fast path as one
        checkpointed call (walk_block) in a training forward; "layer", "host" and "host_overlap" run each layer of
        each block as its own checkpointed call (C2L: Walk.run / EntryWalk.run with ckpt; "host" and "host_overlap"
        move the kept partial sums, streams and R1-entry's block inputs to the host, "host_overlap" on a side stream:
        C1b). The reference stack is never checkpointed. Every other mode asserts (the tested forms only: a mode added
        later must not run here unchecked or without checkpointing)."""
        assert enc.activation_checkpoint in ("none", "block", "layer", "host", "host_overlap"), \
            f"activation checkpointing is not built together with encoder.attn_res in the form " \
            f"{enc.activation_checkpoint!r} (the trainer refuses it); 'block', 'layer', 'host' and 'host_overlap' are"
        assert not self.new_options or enc.activation_checkpoint in self.checkpoint_modes, \
            f"activation checkpointing {enc.activation_checkpoint!r} is not built together with {self.new_options} " \
            f"(check_checkpoint_mode; the trainer refuses it): {self.checkpoint_modes} are"
        assert enc.activation_checkpoint in self.arms_checkpoint_modes, \
            f"activation checkpointing {enc.activation_checkpoint!r} is not built together with the GDN-side arms' CCR " \
            f"(gdn_arms.check_checkpoint_mode; the trainer refuses it): {self.arms_checkpoint_modes} are"
        if self.impl == "reference":
            return reference_stack(enc, h, ctx, caches, skip=self.skip)
        assert self.impl == "fast", self.impl
        assert not self.skip, "sublayers are taken out of the stack by the reference stack only (attn_res.Skipped)"
        from ced.model.encoder import record_stream_rms  # (encoder.py imports this module)

        rms = enc.stream_rms
        walk = self.walk(enc, h)
        new_caches = []
        if caches is None and act_checkpoint.block_active(enc.activation_checkpoint, h):
            for i0 in range(0, len(enc.layers), self.block_layers):  # one checkpoint per block: the same calls
                first = act_checkpoint.FirstForward()
                walk, block_caches = act_checkpoint.run_block(walk_block, walk, ctx, i0, rms, stats, first)
                first.on = False  # the recomputation in the backward records nothing
                new_caches += block_caches
        else:
            # C2L: each layer under its own checkpoint in a training forward ("layer", "host", "host_overlap"; C1a's
            # and C1b's pool guard runs here)
            ckpt = caches is None and act_checkpoint.active(enc.activation_checkpoint, h)
            caches = caches if caches is not None else [None] * len(enc.layers)
            for i, (_, cache) in enumerate(zip(enc.layers, caches)):
                _monitor(walk, i, rms, stats)
                new_caches.append(walk.run(i, ctx, cache, ckpt))
        out = walk.x  # the last layer's call ended with the output's read
        if rms is not None:
            record_stream_rms(rms, out)
        if getattr(walk, "sources", None) is not None:  # the CP pointer's readers (model.cp_head.pointer_input "mix")
            from ced.model.cp_head import SOURCES_KEY

            ctx.memo[SOURCES_KEY] = walk.sources
        return out, new_caches

    def raise_recompile_limit(self) -> int:
        """Room in Dynamo's cache for this stack's compiled layers (RECOMPILE_LIMIT_PER_BLOCK above): the process's
        recompile limit is raised to 32 x n_blocks if it is lower; never lowered. Returns the limit in force."""
        import torch._dynamo.config as dynamo_config

        want = RECOMPILE_LIMIT_PER_BLOCK * self.n_blocks
        if dynamo_config.recompile_limit < want:
            dynamo_config.recompile_limit = want
        return dynamo_config.recompile_limit

    def walk(self, enc, h: torch.Tensor) -> Walk:
        """A Walk of this stack from the embedding output h (stack below; the GDN diagnostics: ced/eval/gdn_diag.py).
        With compiled layers it first makes room in Dynamo's cache (raise_recompile_limit), whoever starts the walk."""
        assert self.impl == "fast", "a Walk runs the fast path"
        if enc._layers_compiled():  # as stack does: a walk may be a process's first compiled run of a row shape
            self.raise_recompile_limit()
        if self.mode == LDM:
            return ldm.LdmWalk(enc, h)
        return EntryWalk(enc, h) if self.mode == ENTRY else Walk(enc, h)

    def readers(self, enc) -> list:
        """(label, DepthQuery) of every reader in running order: "L<i>.mixer" (not layer 0's), "L<i>.mlp", "out".
        R1-entry: "L<i>.mixer" for the first layer i of blocks 2 .. N (the block's entry read, which that mixer
        reads), then "out": n_blocks readers."""
        if self.mode == LDM:  # the memory's reads (ldm.LdmRead), in running order
            items = []
            for i, layer in enumerate(enc.layers):
                items += [(f"L{i}.mixer", layer.attn_res.mixer_read), (f"L{i}.mlp", layer.attn_res.mlp_read)]
            return items + [("out", enc.layers[-1].attn_res.out_read)]
        if self.mode == ENTRY:
            firsts = range(self.block_layers, len(enc.layers), self.block_layers)
            return [(f"L{i}.mixer", enc.layers[i].attn_res.entry) for i in firsts] + [("out", enc.layers[-1].attn_res.out)]
        items = []
        for i, layer in enumerate(enc.layers):
            if layer.attn_res.pre_mixer is not None:
                items.append((f"L{i}.mixer", layer.attn_res.pre_mixer))
            items.append((f"L{i}.mlp", layer.attn_res.pre_mlp))
        return items + [("out", enc.layers[-1].attn_res.out)]

    @torch.no_grad()
    def summary(self, enc) -> dict:
        """The pseudo-queries as they are now, for the logs (the trainer, every log_every steps) and checks:
        {"readers": [labels], "query_norm": [||w||], "key_gain_dev": [max |g|]} per reader in running order (the
        output's last). A query's norm is how far it has moved from its start, where every weight is equal; the
        weights themselves need a forward (depth_terms). Python floats; two host reads in all."""
        items = self.readers(enc)
        if self.mode == LDM:
            return ldm_summary(enc, items)
        if isinstance(items[0][1], TokenQuery):
            # token-dependent queries: "query_norm" is the norm of the static part (the head's bias b2, R1-entry's w),
            # "query_proj_norm" the Frobenius norm of the head's weight W2 (zero at the start: the token part)
            norms = torch.stack([torch.linalg.vector_norm(q.net.head.bias.double()) for _, q in items])
            gains = torch.stack([q.key_norm.weight.double().abs().max() for _, q in items])
            proj = torch.stack([torch.linalg.vector_norm(q.net.head.weight.double()) for _, q in items])
            return {"readers": [name for name, _ in items], "query_norm": norms.cpu().tolist(),
                    "key_gain_dev": gains.cpu().tolist(), "query_proj_norm": proj.cpu().tolist()}
        norms = torch.stack([torch.linalg.vector_norm(q.query.double()) for _, q in items])
        gains = torch.stack([q.key_norm.weight.double().abs().max() for _, q in items])
        return {"readers": [name for name, _ in items], "query_norm": norms.cpu().tolist(), "key_gain_dev": gains.cpu().tolist()}


class Skipped:
    """Context manager, the residual diagnostics' skip for a model with the key (ced/eval/resid_diag.py SKIP_BUILDERS):
    inside it the model's own forward runs the REFERENCE stack with the given sublayers taken out (reference_stack's
    skip). targets: [(layer, sublayer)], sublayer 0 the mixer and 1 the MLP; [] is the whole stack through the reference
    (the paired baseline of a skip: the same implementation on both sides of the difference).

    Why not zeros in place of a branch output, as for the scalar forms: a zero is still a SOURCE here. A block's first
    sublayer that adds zeros leaves a partial sum of zeros, which the next reads take as a source with logit 0: it takes
    weight from the real sources, the read shrinks, and the pre-norm undoes that only up to its eps. Measured (the
    20-layer layout in small, float64, pseudo-queries of order 0.4 a coordinate): zeroing layer 0 gives a loss change of
    -0.0219 nats per token where the stack without layer 0 gives -0.0162; with the norms' eps at 1e-12 the two agree to
    3e-8, and at every layer that is not a block's first they agree to 1e-14 at any eps.

    R1-entry ("block_entry") runs its skips through the same path (entry_reference_stack's skip). There nothing reads a
    partial sum, so zeros in place of a sublayer's output ARE the stack without it while another sublayer of its block
    runs (tests/test_attn_res_entry.py); the reference path is kept so that both arms' skips are one kind of
    measurement, paired inside one implementation."""

    def __init__(self, model, targets):
        self.res = model.encoder.attn_res
        assert self.res is not None, "Skipped is for a model with encoder.attn_res 'block'"
        refuse_new_options(model.cfg, "the residual diagnostics' skip (attn_res.Skipped)")
        self.targets = frozenset((int(i), int(j)) for i, j in targets)

    def __enter__(self):
        self.kept = (self.res.impl, self.res.skip)
        self.res.impl, self.res.skip = "reference", self.targets
        return self

    def __exit__(self, *exc):
        self.res.impl, self.res.skip = self.kept


def build(cfg: ModelConfig, layers) -> BlockAttnRes | None:
    """Encoder.__init__'s one call. "none": None, and nothing is touched: no tensor, no module, no attribute on a
    layer, no random number. "block": every layer gets its LayerQueries (as its last submodule; the other parameters
    keep their names, shapes and order) and the encoder its BlockAttnRes. Zeros only: no random number is drawn, and
    init_weights touches Linear and Embedding weights only, so every other tensor starts at the value it has in the
    model without the key."""
    check_attn_res(cfg)
    if cfg.encoder.attn_res == "none":
        return None
    S, n = cfg.encoder.full_attn_interval, cfg.encoder.n_layers
    if cfg.encoder.attn_res == LDM:
        # Linear Depth Memory (ced/model/ldm.py): every layer gets its two sites (the first also the embedding's write,
        # the last also the output's read), as its last submodule. No random number is drawn here: the model's build
        # starts the address networks from a generator of their own after init_weights (ldm.init_own_parameters).
        ldm.build(cfg, layers)
        return BlockAttnRes(cfg)
    if cfg.encoder.attn_res == ENTRY:
        # R1-entry: only a layer that opens a block (from block 2 on) or closes one gets a module, as its last
        # submodule; a layer inside a block is left as it is and runs the base's call. Zeros only, as below.
        # attn_res_heads and attn_res_query (the architecture arms): the readers' kind; at their defaults R1-entry's.
        e = cfg.encoder
        hidden = ldm.address_hidden(cfg.d_model, e.attn_res_query_hidden) if e.attn_res_query == "token" else 0
        for i, layer in enumerate(layers):
            opens, closes = i > 0 and i % S == 0, i % S == S - 1
            if opens or closes:
                layer.attn_res = EntryLayer(cfg.d_model, cfg.norm_eps, entry=opens, closes=closes, last=i == n - 1,
                                            heads=e.attn_res_heads, query_hidden=hidden)
                layer.attn_res.fused = bool(e.attn_res_fused)  # E: a plain attribute, no tensor
        return BlockAttnRes(cfg)
    for i, layer in enumerate(layers):
        layer.attn_res = LayerQueries(cfg.d_model, cfg.norm_eps, first=i % S == 0, has_mixer_query=i > 0, last=i == n - 1)
    return BlockAttnRes(cfg)


def parameter_names(cfg: ModelConfig) -> list[str]:
    """The names, in running order, of the tensors the key adds to arch "control": (2 * n_layers - 1) + 1 readers, a
    (d,) query and a (d,) key-norm gain each; the output's are the last layer's `out`. [] for "none".
    "block_entry": n_blocks readers, the `entry` of the first layer of blocks 2 .. N and the last layer's `out`."""
    if cfg.encoder.attn_res == "none":
        return []
    names, n = [], cfg.encoder.n_layers
    if cfg.encoder.attn_res == LDM:  # every LdmLayer's tensors, in module order (ced/model/ldm.py)
        net = ["net.norm.weight", "net.inp.weight", "net.inp.bias", "net.head.weight", "net.head.bias", "net.gate.weight",
               "net.gate.bias"]
        for i in range(n):
            sites = (["embed_write"] if i == 0 else []) + ["mixer_read", "mixer_write", "mlp_read", "mlp_write"] + (
                ["out_read"] if i == n - 1 else [])
            for site in sites:
                own = ["read_gain"] if site.endswith("_read") else ([] if site == "embed_write" else ["lam_logit"])
                names += [f"encoder.layers.{i}.attn_res.{site}.{t}" for t in own + net]
        return names
    if cfg.encoder.attn_res == ENTRY:
        S = cfg.encoder.full_attn_interval
        if cfg.encoder.attn_res_query == "token":  # TokenQuery: the key norm, then the query network
            parts = ["key_norm.weight", "net.norm.weight", "net.inp.weight", "net.inp.bias", "net.head.weight", "net.head.bias"]
        else:
            parts = ["query", "key_norm.weight"]
        for i in range(S, n, S):
            names += [f"encoder.layers.{i}.attn_res.entry.{t}" for t in parts]
        return names + [f"encoder.layers.{n - 1}.attn_res.out.{t}" for t in parts]
    for i in range(n):
        for part in (("pre_mixer",) if i else ()) + ("pre_mlp",) + (("out",) if i == n - 1 else ()):
            names += [f"encoder.layers.{i}.attn_res.{part}.query", f"encoder.layers.{i}.attn_res.{part}.key_norm.weight"]
    return names


@torch.no_grad()
def capture_terms(model, cap) -> list:
    """The term list of the residual diagnostics (ced/eval/resid_diag.py, TERM_BUILDERS: the builder of a model with
    Block AttnRes) from that tool's capture of the model's own last forward: cap.x[0] the embedding output (what layer
    0's first pre-norm reads), cap.u[i][j] what sublayer j (0 mixer, 1 MLP) of layer i adds (residual_scale times its
    output norm's output). Returns [(info, contribution (B, L, d) float64)] summing to the stream the final norm reads:
    "x0", then "L<i>.mixer" and "L<i>.mlp". A contribution is the sublayer's output times the output read's weight of
    its block, per position; the weights are formed here from the captured outputs (the blocks' sums) and the last
    layer's `out` query and key norm, as the model forms them. info: {"name", "kind" (embedding / gdn / attention /
    mlp), "layer", "sub", "coef"}; "coef" is the weight's mean over positions (the scalar forms have one coefficient
    per term; here it differs by position)."""
    enc = model.encoder
    assert getattr(enc, "attn_res", None) is not None, "capture_terms is for a model with encoder.attn_res 'block'"
    refuse_new_options(model.cfg, "attn_res.capture_terms (the residual diagnostics)")
    S, b0 = enc.attn_res.block_layers, cap.x[0]
    blocks, partial = [b0], None
    for i in range(len(enc.layers)):
        if i and i % S == 0:
            blocks.append(partial)
            partial = None
        for u in cap.u[i]:
            partial = u.to(b0.dtype) if partial is None else partial + u
    blocks.append(partial)
    _, alpha = block_attn_res(blocks, None, enc.layers[-1].attn_res.out)  # (N + 1, B, L)
    alpha = alpha.double()
    out = [({"name": "x0", "kind": "embedding", "layer": None, "sub": None, "coef": float(alpha[0].mean())},
            alpha[0][..., None] * b0.double())]
    for i, layer in enumerate(enc.layers):
        w = alpha[i // S + 1]
        for j, sub in enumerate(("mixer", "mlp")):
            info = {"name": f"L{i}.{sub}", "kind": layer.kind if sub == "mixer" else "mlp", "layer": i, "sub": sub, "coef": float(w.mean())}
            out.append((info, w[..., None] * cap.u[i][j].double()))
    return out


@torch.no_grad()
def capture_weights(model, cap) -> dict:
    """Every reader's depth-attention weights for the residual diagnostics (ced/eval/resid_diag.py, WEIGHT_BUILDERS), from
    that tool's capture of the model's own last forward (capture_terms' cap). The sources of every read are rebuilt from
    the captured outputs by Eq. 5 and 6 and the weights taken with the reader's own query and key norm (block_attn_res):

        {"readers": [{"reader": "L<i>.mixer" | "L<i>.mlp" | "out", "layer": i or None, "sources": [labels]}],  running order
         "sum":     [(n,) float64: the reader's weight of each source, summed over the forward's positions],
         "positions": B * L,
         "read_err": the largest |rebuilt read - captured read| / rms(captured read) over the mixers' reads and the output's
                     (cap.x[i]: what layer i's first pre-norm and the final norm read). It says the weights are the ones
                     the forward used; the MLPs' reads are not captured, so they are not compared}

    Sources are "embed", "block<k>" (a finished block) and "partial" (the reader's own block so far).
    R1-entry ("block_entry"): the readers are "L<i>.mixer" for the first layer i of blocks 2 .. N (the block's entry
    read; no "partial" among its sources) and "out"; every one of them is compared with its hooked read."""
    enc = model.encoder
    assert getattr(enc, "attn_res", None) is not None, "capture_weights is for a model with encoder.attn_res 'block'"
    refuse_new_options(model.cfg, "attn_res.capture_weights (the residual diagnostics)")
    S, b0 = enc.attn_res.block_layers, cap.x[0]
    blocks, partial = [b0], None
    readers, sums, err = [], [], 0.0

    def labels():
        return ["embed"] + [f"block{k}" for k in range(1, len(blocks))] + ([] if partial is None else ["partial"])

    def read(name, layer, q, seen):
        nonlocal err
        h, alpha = block_attn_res(blocks, partial, q)
        readers.append({"reader": name, "layer": layer, "sources": labels()})
        sums.append(alpha.double().flatten(1).sum(1).cpu())
        if seen is not None:
            rms = seen.double().pow(2).mean().sqrt().clamp_min(1e-300)
            err = _max_nan((err, float((h.double() - seen.double()).abs().max() / rms)))

    entry = enc.attn_res.mode == ENTRY  # R1-entry: the readers are the blocks' entries (their first mixers' reads) and `out`
    for i, layer in enumerate(enc.layers):
        if i and i % S == 0:
            blocks.append(partial)
            partial = None
        res = getattr(layer, "attn_res", None)
        if entry:
            if res is not None and res.entry is not None:
                read(f"L{i}.mixer", i, res.entry, cap.x[i])
        elif res.pre_mixer is not None:
            read(f"L{i}.mixer", i, res.pre_mixer, cap.x[i])
        partial = cap.u[i][0].to(b0.dtype) if partial is None else partial + cap.u[i][0]
        if not entry:
            read(f"L{i}.mlp", i, res.pre_mlp, None)
        partial = partial + cap.u[i][1]
    blocks.append(partial)
    partial = None
    read("out", None, enc.layers[-1].attn_res.out, cap.x[len(enc.layers)])
    head = getattr(model, "cp_head", None)
    if head is not None and head.cfg.pointer_input == "mix":
        # cp-mix (model.cp_head.pointer_input "mix"; ced/model/cp_head.py PointerMix): the CP pointer's query and key readers
        # over the output read's sources, "cp.q" and "cp.k" (their reads are not hooked: not compared)
        if hasattr(head.mix, "read_q"):
            read("cp.q", None, head.mix.read_q, None)
            read("cp.k", None, head.mix.read_k, None)
        else:  # the pointer's heads (model.cp_head.n_heads > 1; cp_head.HeadsMix): "cp.q<h>" and "cp.k<h>", head by head
            for i, m in enumerate(head.mix.heads):
                read(f"cp.q{i}", None, m.read_q, None)
                read(f"cp.k{i}", None, m.read_k, None)
    return {"readers": readers, "sum": sums, "positions": b0.shape[0] * b0.shape[1], "read_err": err}


def add_weights(acc: dict | None, w: dict) -> dict:
    """capture_weights of one more forward added to a running total (None: the first)."""
    if acc is None:
        return {"readers": w["readers"], "sum": [t.clone() for t in w["sum"]], "positions": w["positions"], "read_err": w["read_err"]}
    assert acc["readers"] == w["readers"]
    return {"readers": acc["readers"], "sum": [a + b for a, b in zip(acc["sum"], w["sum"])],
            "positions": acc["positions"] + w["positions"], "read_err": _max_nan((acc["read_err"], w["read_err"]))}


def weights_result(per_set: dict) -> dict:
    """The diagnostics' `depth_weights` from the running totals of each row set ({name: add_weights' total}):
    {"readers": [{"reader", "layer", "sources", "mean": [the weight of each source, mean over positions, then the mean
    over the row sets with equal weight per set], "per_set": {name: [...]}}], "read_max_rel": the largest read_err}."""
    names = [n for n in per_set if per_set[n] is not None]
    if not names:
        return {"readers": [], "read_max_rel": None}
    out = []
    for r, reader in enumerate(per_set[names[0]]["readers"]):
        each = {n: (per_set[n]["sum"][r] / per_set[n]["positions"]).tolist() for n in names}
        mean = [sum(each[n][k] for n in names) / len(names) for k in range(len(reader["sources"]))]
        out.append({**reader, "mean": mean, "per_set": each})
    return {"readers": out, "read_max_rel": _max_nan(per_set[n]["read_err"] for n in names)}


def _max_nan(xs) -> float:
    """max() that keeps a NaN (ced/utils/criterion.py nan_max): Python's max() drops a NaN that is not first, so a NaN read error
    hid behind an earlier finite one."""
    xs = [float(x) for x in xs]
    return float("nan") if any(x != x for x in xs) else max(xs)


@torch.no_grad()
def depth_terms(model, tokens: torch.Tensor, doc_ids: torch.Tensor, valid: torch.Tensor | None = None, skip=()) -> dict:
    """For the diagnostics (per-layer share of the final stream, depth-attention weights, what each sublayer reads,
    the loss with one layer skipped): one forward of a control with Block AttnRes through the REFERENCE stack, eager,
    no cache, under the caller's autocast. skip: layer indices taken out of the stack (reference_stack; "terms",
    "raw" and "reads" then have no entry for them); the loss with a layer skipped is the head's loss on "hidden".
    Returns

        "terms"    [(label, tensor (B, L, d))]: "embed", then "L<i>.mixer" and "L<i>.mlp" for every layer, bottom
                   first. Their SUM is "final". Each is the sublayer's output f (the embedding output for "embed")
                   times the output read's weight of its block, alpha_out[block] (B, L, 1): the final stream is
                   sum_n alpha_out[n] b_n and b_n is the sum of its sublayers' outputs.
        "final"    (B, L, d): the stream after the last sublayer, before the model's final norm.
        "hidden"   (B, L, d): final_norm(final), the head's input.
        "weights"  one entry per reader in running order (every sublayer but the first, then "out"):
                   {"reader", "layer", "block", "sources": [labels], "mean": [the reader's weight of each source, the
                   mean over positions (over `valid` (B, L) bool if given)]}. Sources are "embed", "block<k>" (a
                   finished block) and "partial" (the reader's own block so far).
        "raw"      the sublayers' own outputs f, unweighted: [(label, tensor)], for norms and skip tests.
        "reads"    [(label, tensor (B, L, d))]: what every sublayer reads (the input of its pre-norm: the embedding
                   output for "L0.mixer", else its depth attention's output), then ("out", final): the arm's
                   counterpart of the base's stream at each layer boundary (max-abs, kurtosis, RMS).
        "out_weights"  (N + 1, B, L): the output read's weight of the embedding and of every block, per position.
        "block_of" {label: the index into out_weights of the term's block} (0 for "embed").
    R1-entry ("block_entry"): the same keys from entry_reference_stack. "weights" then has n_blocks entries: the
    entry read of blocks 2 .. N under the label of the sublayer that reads it ("L<i>.mixer", i the block's first
    layer) and "out"; "reads" holds the running stream for a sublayer inside a block.
    """
    cfg = model.cfg
    assert cfg.arch == "control" and cfg.encoder.attn_res != "none", "depth_terms is for a control with encoder.attn_res 'block' or 'block_entry'"
    refuse_new_options(cfg, "attn_res.depth_terms (the residual diagnostics)")
    from ced.model.sequence import sequence_context

    enc = model.encoder
    ctx, _ = sequence_context(doc_ids, None)
    b0 = model.embed_norm(model.embed(tokens))
    record: list = []
    final, _ = reference_stack(enc, b0, ctx, None, record, skip=set(skip))
    alpha_out = record[-1]["alpha"]  # (N + 1, B, L): the output read's weights of b_0 .. b_N
    dtype = alpha_out.dtype
    terms = [("embed", alpha_out[0][..., None] * b0.to(dtype))]
    raw = [("embed", b0)]
    block_of = {"embed": 0}
    weights = []
    w = None if valid is None else valid.to(dtype) / valid.sum().clamp_min(1).to(dtype)
    for r in record:
        if r["output"] is not None:
            terms.append((r["reader"], alpha_out[r["block"]][..., None] * r["output"].to(dtype)))
            raw.append((r["reader"], r["output"]))
            block_of[r["reader"]] = r["block"]
        if r["alpha"] is not None:
            a = r["alpha"]
            mean = a.flatten(1).mean(1) if w is None else (a * w).flatten(1).sum(1)
            weights.append({"reader": r["reader"], "layer": r["layer"], "block": r["block"], "sources": r["sources"],
                            "mean": [float(m) for m in mean.double().cpu()]})
    return {"terms": terms, "final": final, "hidden": model.final_norm(final), "weights": weights, "raw": raw,
            "reads": [(r["reader"], r["input"]) for r in record], "out_weights": alpha_out, "block_of": block_of}


@torch.no_grad()
def ldm_summary(enc, items) -> dict:
    """BlockAttnRes.summary under "ldm", for the logs: per read in running order, "query_norm" the norm of its read gain
    gamma (G values, sqrt(G) at the start) and "key_gain_dev" the largest |gain - 1| of its address network's input norm
    (0 at the start); per write in running order, "lambda" the mean forgetting factor
    sigmoid(logit) (0.99 at the start; the embedding's write has none: 2 n_layers values) and "write_gain_dev" the
    largest |gain - 1| of its input norm (2 n_layers + 1 values, the embedding's first). Python floats."""
    norms = torch.stack([torch.linalg.vector_norm(q.read_gain.double()) for _, q in items])
    gains = torch.stack([q.net.norm.weight.double().abs().max() for _, q in items])
    writes = [enc.layers[0].attn_res.embed_write]
    for layer in enc.layers:
        writes += [layer.attn_res.mixer_write, layer.attn_res.mlp_write]
    lam = torch.stack([torch.sigmoid(w.lam_logit.double()).mean() for w in writes[1:]])
    wdev = torch.stack([w.net.norm.weight.double().abs().max() for w in writes])
    return {"readers": [name for name, _ in items], "query_norm": norms.cpu().tolist(), "key_gain_dev": gains.cpu().tolist(),
            "lambda": lam.cpu().tolist(), "write_gain_dev": wdev.cpu().tolist()}
