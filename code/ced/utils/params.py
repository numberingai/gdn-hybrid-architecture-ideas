"""Parameter accounting, the matched non-looped control, and FLOPs per token.

FLOP counts are the usual matmul approximation: a weight matrix with N entries costs 2N FLOPs per
token forward and 6N for forward + backward (a depthwise conv weight counts the same way: N
multiply-adds per token). Attention score/value FLOPs are counted separately (they depend on context
length): 4 * H_q * head_dim per query and key (QK^T and AV) forward, 3x that in training. The GDN
recurrence is counted as 6 * H_v * d_k * d_v per token forward (`_gdn_recurrence_flops`). Norms,
gates, elementwise ops and softmax are not counted.

The dense FFN (ModelConfig.ffn_activation) has three d x f matrices for SwiGLU and two for ReLU^2
(`dense_ffn_params`); its activation (SiLU times the up projection, or the square of the ReLU) is
elementwise and not counted. Every count below comes from the built model's matrices, so it follows
the activation without a special case.
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict

import torch

from ced.config import AttentionConfig, DecoderConfig, EncoderConfig, GDNConfig, LoopConfig, ModelConfig
from ced.model.control import build_model


def full_config(gdn_k_heads: int, gdn_v_heads: int, n_enc_layers: int = 36, vocab_size: int = 128000) -> ModelConfig:
    """The ~3.8B shape of architecture-spec-v1 section 4 with a given GDN head configuration."""
    attn = AttentionConfig(n_q_heads=16, n_kv_heads=2, head_dim=256)
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=2560,
        ffn_hidden=7168,
        encoder=EncoderConfig(
            n_layers=n_enc_layers,
            attention=attn,
            gdn=GDNConfig(n_k_heads=gdn_k_heads, n_v_heads=gdn_v_heads, k_head_dim=128, v_head_dim=128),
        ),
        decoder=DecoderConfig(n_layers=4, attention=dataclasses.replace(attn)),
        loop=LoopConfig(),
    )


def build_on_meta(cfg: ModelConfig) -> torch.nn.Module:
    """Build without allocating memory (shapes only)."""
    with torch.device("meta"):
        return build_model(cfg)


def parameter_breakdown(model: torch.nn.Module) -> "OrderedDict[str, int]":
    """Parameters per group. "embedding (tied)" is the input embedding (the LM head too when tied);
    "output embedding" the separate output embedding of an untied model (v2). "non-embedding"
    leaves out both, "embeddings" is their sum. CED v2's decoder layers count as "decoder layers",
    its W_mem as "memory KV"; the second-pass control's pass embedding as "pass embedding". A model with Block
    AttnRes (EncoderConfig.attn_res) has one more group, "attention residuals" (its pseudo-queries and key-norm
    gains); no other model has the key."""
    groups: OrderedDict[str, int] = OrderedDict(
        (k, 0)
        for k in (
            "embedding (tied)", "output embedding", "encoder GDN layers", "encoder global layers",
            "decoder layers", "memory KV", "injection", "loop embedding", "pass embedding",
            "hyper-connections", "other norms",
        )
    )
    kinds = {}
    if hasattr(model, "encoder"):
        for i, layer in enumerate(model.encoder.layers):
            kinds[f"encoder.layers.{i}."] = "encoder GDN layers" if layer.kind == "gdn" else "encoder global layers"
    for name, p in model.named_parameters():
        n = p.numel()
        if name.startswith("embed."):
            groups["embedding (tied)"] += n
        elif name.startswith("out_embed."):
            groups["output embedding"] += n
        elif name.startswith("pass_embedding"):
            groups["pass embedding"] += n
        elif ".attn_res." in name:
            # Block AttnRes (EncoderConfig.attn_res; ced/model/attn_res.py): the readers' pseudo-queries and key-norm
            # gains, on the layers and on the encoder. A group of its own, present only in a model that has them.
            groups["attention residuals"] = groups.get("attention residuals", 0) + n
        elif name.startswith("encoder.layers."):
            prefix = ".".join(name.split(".")[:3]) + "."
            groups[kinds[prefix]] += n
        elif name.startswith("decoder_layers."):
            groups["decoder layers"] += n
        elif name.startswith("memory_kv."):
            groups["memory KV"] += n
        elif name.startswith(("injection.", "encoder.recur_injection.", "encoder.recur_depth.")) or name == "encoder.recur_mix":
            # (encoder.recur_depth.*: the reader of the depth mix at the loop boundary, R3; ced/model/recur_depth.py)
            groups["injection"] += n
        elif name.startswith("loop_embedding"):
            groups["loop embedding"] += n
        elif name.startswith("hyper_connections."):
            groups["hyper-connections"] += n
        else:
            groups["other norms"] += n
    groups["total"] = sum(groups.values())
    groups["embeddings"] = groups["embedding (tied)"] + groups["output embedding"]
    groups["non-embedding"] = groups["total"] - groups["embeddings"]
    return groups


def matched_control(cfg: ModelConfig) -> ModelConfig:
    """The non-looped control: the same encoder blocks, no loop, no encoder/decoder split, with the
    layer count whose non-embedding parameter count is closest to Looped-CED's."""
    target = parameter_breakdown(build_on_meta(cfg))["non-embedding"]

    def control_with(n_layers: int) -> ModelConfig:
        return dataclasses.replace(cfg, arch="control", encoder=dataclasses.replace(cfg.encoder, n_layers=n_layers))

    # Per-layer counts from one small build, then the count for any depth by arithmetic.
    probe_layers = max(1, cfg.encoder.full_attn_interval)
    probe = build_on_meta(control_with(probe_layers))
    b = parameter_breakdown(probe)
    n_gdn_probe = sum(1 for k in control_with(probe_layers).encoder.layer_kinds() if k == "gdn")
    per_gdn = b["encoder GDN layers"] / max(1, n_gdn_probe)
    per_global = b["encoder global layers"] / max(1, probe_layers - n_gdn_probe)
    fixed = b["non-embedding"] - b["encoder GDN layers"] - b["encoder global layers"]

    def non_embedding(n_layers: int) -> float:
        kinds = control_with(n_layers).encoder.layer_kinds()
        return fixed + sum(per_gdn if k == "gdn" else per_global for k in kinds)

    best = min(range(1, 2 * (cfg.encoder.n_layers + cfg.loop.t_max * cfg.decoder.n_layers)),
               key=lambda n: abs(non_embedding(n) - target))
    return control_with(best)


def dense_ffn_params(cfg: ModelConfig) -> int:
    """Matrix entries of one dense FFN from the config alone: 3 d f for SwiGLU (gate, up, down),
    2 d f for ReLU^2 (up, down). The same per token in FLOPs: 2x forward, 6x training."""
    n_matrices = {"swiglu": 3, "relu2": 2}[cfg.ffn_activation]
    return n_matrices * cfg.d_model * cfg.ffn_hidden


def _matrix_params(module: torch.nn.Module | None) -> int:
    """Entries of the matrices that cost FLOPs per token (2-D weights), excluding embeddings."""
    if module is None or not isinstance(module, torch.nn.Module):
        return 0
    return sum(p.numel() for n, p in module.named_parameters() if p.dim() == 2 and not n.startswith("embed."))


def _gdn_recurrence_flops(g) -> int:
    """Forward FLOPs per token of the delta-rule state update and readout (decay, S^T k, outer
    product, S^T q): about 6 * H_v * d_k * d_v."""
    return 6 * g.n_v_heads * g.k_head_dim * g.v_head_dim


def decoder_active_params(layer) -> int:
    """Matrix parameters one decoder layer applies to a token in ONE loop: one mixer copy (untied
    copies are used one per loop), the FFN's top-k experts and router (routed), depth attention."""
    from ced.model.mlp import RoutedSwiGLU

    mixer = _matrix_params(layer.mixers[0])
    if isinstance(layer.mlp, RoutedSwiGLU):
        ffn = layer.mlp.top_k * _matrix_params(layer.mlp.experts[0]) + _matrix_params(layer.mlp.router)
    else:
        ffn = _matrix_params(layer.mlp)
    return mixer + ffn + _matrix_params(getattr(layer, "depth_attention", None))


def second_pass_params(model: torch.nn.Module, first_layer: int = 0) -> int:
    """Matrix entries pass 2 of the second-pass control applies per slot (ced/model/second_pass.py):
    every encoder layer's FFN and output projection; GDN layers only the q rows of in_proj_qkv, the
    q channels of the short conv and in_proj_z (pass 2 forms no k, v, beta or g); global layers the
    query (and gate) projection, no k / v projections.
    first_layer: count the layers from this index on only (the looped control's block: ced/model/control.py,
    "Two passes"; its pass 2 is the same zero-write call per layer)."""
    total = 0
    for layer in list(model.encoder.layers)[first_layer:]:
        m = layer.mixer
        total += _matrix_params(layer.mlp)
        if layer.kind == "gdn":
            nq = m.cfg.n_k_heads * m.cfg.k_head_dim
            total += m.in_proj_qkv.weight[:nq].numel() + m.conv_weight[:nq].numel()
            total += m.in_proj_z.weight.numel() + m.out_proj.weight.numel()
        else:
            total += _matrix_params(m.q_proj) + _matrix_params(m.g_proj) + _matrix_params(m.o_proj)
    return total


def flops_per_token(cfg: ModelConfig, seq_len: int, dec_fraction: float, T: int, aux_fraction: float | None = None,
                    attn_context: float | None = None, context_own: float | None = None) -> dict[str, float]:
    """Training and inference FLOPs per token (matmuls 2N forward / 6N training, plus attention and
    the GDN recurrence; norms, gates and softmax are not counted).

    training: averaged over a packed row of length seq_len; dec_fraction of positions occupy decoder
              slots (the capacity C/L: empty slots cost as much as full ones); the aux head runs at
              aux_fraction of the positions (None: every position, the v1 count); intermediate-loop
              readouts add a head pass when draft_loss_weight > 0 (v1).
    prefill_context: forward FLOPs to read one context token at context length seq_len (no head).
    decode_generated: forward FLOPs to generate one token at context length seq_len (with head).
    Attention terms use attn_context keys per query in training (None: seq_len / 2, causal over the
    whole row, an upper bound when packed documents are masked from each other) and seq_len for
    inference. A looped encoder multiplies all encoder terms by its loop count.
    context_own (CED v2 with decoder self_kv, decode only): the share of the context's positions that carry
    decoder own K/V (None: dec_fraction, the decoder everywhere as in training at capacity L; 0: the decoder
    at the generated positions only, the stage 3 regime).

    With E the encoder's matrix entries, H = V * d the head, a_q = 4 * H_q * head_dim per key and
    attention layer, K the training keys per query, R the encoder's GDN recurrence, p = dec_fraction,
    q = aux_fraction:

    control: 6 (E + H) + 3 (a_enc K + R) With a looped middle (LoopConfig.recur_layers = [first, end];
                 ced/model/control.py, "The loop in the published form") at T passes, every term of the middle is paid T
                 times, at every position, in training, at prefill and at decode alike (each pass writes its own K/V and
                 states): E becomes E_out + T (E_mid + J), a_enc becomes a_q (n_a_out + T n_a_mid) and R becomes R_1
                 (n_g_out + T n_g_mid), with E_mid the middle layers' matrices, J the injection's dense B (d^2; 0 for
                 "diag" or without injection: elementwise work is not counted), n_a / n_g the attention / GDN layers
                 inside and outside the middle. Linear in T. With a gated second pass through its top block
                 (LoopConfig.pass2_gate with pass2_layers; ced/model/control.py, "Two passes") and T = 2: + 6 P2 + 3
                 (a_q n_a K + n_g 2 H_v d_k d_v) in training (pass 2 at every position), + 2 P2 + a_q n_a L + n_g 2 H_v
                 d_k d_v per generated token at decode, and nothing per read token (P2 = second_pass_params of the
                 block; n_a, n_g its attention and GDN layers). At T = 1, and without a gate whatever T is passed, the
                 line above. looped_ced: 6 (E + M + H) + 6 p (T D + (1 + drafts) H) + 3 (a_enc K + R) + 3 p T (a_dec K +
                 R_dec) (M memory KV, D one loop of decoder layers; decoder GDN layers pay a recurrence) ced_v2: 6 (E +
                 M) + 6 q H + 6 p (T D + H) + 3 (a_enc K + R) + 3 p T a_dec K (1 + s p) (every position writes the
                 memory; each decoder layer reads it at every loop; s = 1 with decoder self_kv: each decoder layer also
                 reads the own K/V of the decoder positions among the K keys, p K of them, and D includes k_proj /
                 v_proj) With a read pattern (DecoderConfig.read_pattern), a_dec counts the full layers only, and each
                 local layer reads p min(K, W + S) own keys instead (W the local window, S the sinks; no memory): 3 p T
                 a_q n_local p min(K, W + S) in training, T a_q n_local c min(L, W + S) at decode (c = context_own).
                 min(K, W + S) is the keys a full read would see, capped at the local read's; like K, an upper bound.
                 With self_kv at T = 2, pass 2 reads pass 1's K/V and projects none (ced/model/ced_v2.py, "Two passes
                 with self-KV"): T D becomes D + (T - 1) (D - P), P the k_proj / v_proj entries of the decoder layers;
                 every read term keeps its factor T (pass 2 makes the same reads over the same keys). second_pass: 6 E +
                 6 q H + 6 p (P2 + H) + 3 (a_enc K + R) + 3 p (a_enc K + n_gdn 2 H_v d_k d_v) (P2 = second_pass_params;
                 pass 2 attends to pass-1 K/V of each global layer and reads each GDN layer's state, o = S^T q, without
                 writing)

    The second-pass count is the function's: the training path computes pass 2's GDN read with one
    more delta-rule call per GDN layer over every position (ced/model/gdn.py, zero_write_read), about
    3 R more, which is not counted (the plan defers that speed work).
    """
    model = build_on_meta(cfg)
    V, d = cfg.vocab_size, cfg.d_model
    head = V * d
    K_train = seq_len / 2 if attn_context is None else attn_context
    aux = 1.0 if aux_fraction is None else aux_fraction
    enc_loops = cfg.encoder.loops
    enc = enc_loops * _matrix_params(model.encoder.layers)
    if enc_loops > 1:
        enc += enc_loops * _matrix_params(model.encoder.injection)
    kinds = cfg.encoder.layer_kinds()
    a = cfg.encoder.attention
    enc_attn_per_key = enc_loops * 4 * a.n_q_heads * a.head_dim * kinds.count("attention")
    # each GDN layer's recurrence by its own shape (EncoderConfig.gdn_shape_by_layer; with the table {} every layer is cfg.encoder.gdn
    # and this is the count times the one value, as it was)
    enc_recur = enc_loops * sum(_gdn_recurrence_flops(g) for g in cfg.encoder.gdn_layers())

    if cfg.arch == "control" and cfg.loop.recur_layers:
        # The loop in the published form: the middle's matrices, reads and recurrences T times, at every position.
        first, end = cfg.loop.recur_layers
        layers = list(model.encoder.layers)
        mid = _matrix_params(torch.nn.ModuleList(layers[first:end])) + _matrix_params(getattr(model.encoder, "recur_injection", None))
        outer = _matrix_params(torch.nn.ModuleList(layers[:first] + layers[end:]))
        n_a_mid, n_g_mid = kinds[first:end].count("attention"), kinds[first:end].count("gdn")
        n_a_out, n_g_out = kinds.count("attention") - n_a_mid, kinds.count("gdn") - n_g_mid
        mats = outer + T * mid
        attn_per_key = 4 * a.n_q_heads * a.head_dim * (n_a_out + T * n_a_mid)
        recur = (n_g_out + T * n_g_mid) * _gdn_recurrence_flops(cfg.encoder.gdn)
        return {"training": 6 * (mats + head) + 3 * (attn_per_key * K_train + recur),
                "prefill_context": 2 * mats + attn_per_key * seq_len + recur,
                "decode_generated": 2 * (mats + head) + attn_per_key * seq_len + recur}

    if cfg.arch == "control":
        train = 6 * (enc + head) + 3 * (enc_attn_per_key * K_train + enc_recur)
        prefill = 2 * enc + enc_attn_per_key * seq_len + enc_recur
        decode = 2 * (enc + head) + enc_attn_per_key * seq_len + enc_recur
        if cfg.loop.pass2_gate != "none" and T >= 2:
            # The looped control at T = 2 (ced/model/control.py, "Two passes"): pass 2 through the top block at every
            # position in training and at the generated token at decode; a read token pays nothing.
            g = cfg.encoder.gdn
            first = cfg.encoder.n_layers - cfg.loop.pass2_layers
            pass2 = second_pass_params(model, first)
            attn_per_key = 4 * a.n_q_heads * a.head_dim * kinds[first:].count("attention")
            read = kinds[first:].count("gdn") * 2 * g.n_v_heads * g.k_head_dim * g.v_head_dim  # o = S^T q, forward
            train += 6 * pass2 + 3 * (attn_per_key * K_train + read)
            decode += 2 * pass2 + attn_per_key * seq_len + read
        return {"training": train, "prefill_context": prefill, "decode_generated": decode}

    if cfg.arch == "second_pass":
        g = cfg.encoder.gdn
        pass2 = second_pass_params(model)
        read = kinds.count("gdn") * 2 * g.n_v_heads * g.k_head_dim * g.v_head_dim  # o = S^T q, forward
        train = (
            6 * enc + 6 * aux * head  # pass 1 at every position, aux head
            + 6 * dec_fraction * (pass2 + head)  # pass 2 and the main head at the slots
            + 3 * (enc_attn_per_key * K_train + enc_recur)
            + 3 * dec_fraction * (enc_attn_per_key * K_train + read)
        )
        prefill = 2 * enc + enc_attn_per_key * seq_len + enc_recur
        decode = 2 * (enc + pass2 + head) + 2 * enc_attn_per_key * seq_len + enc_recur + read
        return {"training": train, "prefill_context": prefill, "decode_generated": decode}

    da = cfg.decoder.attention
    if cfg.arch == "ced_v2":
        mem = _matrix_params(model.memory_kv)
        per_loop = _matrix_params(model.decoder_layers)
        # Pass 2 with self-KV writes no K/V: its matrices are the block's without k_proj / v_proj (0 without self-KV,
        # where the layers have none). all_loops = the decoder matrices applied over the T passes of one slot.
        kv_proj = sum(p.numel() for n, p in model.decoder_layers.named_parameters()
                      if p.dim() == 2 and n.endswith(("k_proj.weight", "v_proj.weight"))) if cfg.decoder.self_kv else 0
        all_loops = T * per_loop - (T - 1) * kv_proj
        # Read pattern: the full layers read the memory and all own keys; the local layers only the own keys
        # at the sinks and in the window. Every layer full by default.
        dkinds = cfg.decoder.read_kinds()
        n_full, n_local = dkinds.count("full"), dkinds.count("local")
        dec_attn_per_key = 4 * da.n_q_heads * da.head_dim * n_full  # per loop: every full layer
        local_attn_per_key = 4 * da.n_q_heads * da.head_dim * n_local  # per loop: every local layer
        local_keys = cfg.decoder.local_window + cfg.decoder.n_sinks
        # Decoder self-KV: own keys of the decoder positions as well, dec_fraction of the keys in training
        # and of the context at decode (the decoder positions among them; context_own when given).
        own = dec_fraction if cfg.decoder.self_kv else 0.0
        own_ctx = own if context_own is None or not cfg.decoder.self_kv else context_own
        train = (
            6 * (enc + mem) + 6 * aux * head  # every position: encoder, memory KV; aux head
            + 6 * dec_fraction * (all_loops + head)  # decoder slots, main head
            + 3 * (enc_attn_per_key * K_train + enc_recur)
            + 3 * dec_fraction * T * dec_attn_per_key * K_train
        )
        if own:
            train += 3 * dec_fraction * T * dec_attn_per_key * own * K_train
        if n_local:
            train += 3 * dec_fraction * T * local_attn_per_key * own * min(K_train, local_keys)
        prefill = 2 * (enc + mem) + enc_attn_per_key * seq_len + enc_recur
        decode = (2 * (enc + mem + all_loops + head) + (enc_attn_per_key + T * dec_attn_per_key) * seq_len
                  + enc_recur)
        if own_ctx:
            decode += T * dec_attn_per_key * own_ctx * seq_len
        if n_local:
            decode += T * local_attn_per_key * own_ctx * min(seq_len, local_keys)
        return {"training": train, "prefill_context": prefill, "decode_generated": decode}

    mem = sum(_matrix_params(m) for m in model.memory_kv)
    per_loop = sum(decoder_active_params(layer) for layer in model.decoder_layers) + _matrix_params(model.injection)
    dkinds = cfg.decoder.layer_kinds()
    dec_attn_per_key = 4 * da.n_q_heads * da.head_dim * dkinds.count("memory_attention")  # per loop
    dec_recur = dkinds.count("gdn") * _gdn_recurrence_flops(cfg.decoder.gdn)  # per loop
    draft_heads = (T - 1) if cfg.loop.draft_loss_weight > 0 else 0

    train = (
        6 * (enc + mem + aux * head)  # every position: encoder, memory KV, aux head
        + 6 * dec_fraction * (T * per_loop + (1 + draft_heads) * head)  # decoder slots
        + 3 * (enc_attn_per_key * K_train + enc_recur)
        + 3 * dec_fraction * T * (dec_attn_per_key * K_train + dec_recur)
    )
    prefill = 2 * (enc + mem) + enc_attn_per_key * seq_len + enc_recur
    decode = (2 * (enc + T * per_loop + head) + enc_attn_per_key * seq_len + enc_recur
              + T * (dec_attn_per_key * seq_len + dec_recur))
    return {"training": train, "prefill_context": prefill, "decode_generated": decode}


# --------------------------------------------------------------------------------------------
# segment-based pass counts (LoopConfig.segment_passes; ced/model/pass_schedule.py)
# --------------------------------------------------------------------------------------------


def mean_passes(segment_passes: dict | None, shares: dict[str, float]) -> float:
    """The mean pass count of positions whose segments have the given shares: sum over the segments of
    share x passes. segment_passes: the map of LoopConfig.segment_passes (a key left out counts 2; None: every segment
    2); shares: {segment: share of the positions}, over pass_schedule.SEGMENT_KEYS, summing to 1."""
    from ced.model.pass_schedule import DEFAULT_COUNT, SEGMENT_KEYS

    unknown = sorted(set(shares) - set(SEGMENT_KEYS))
    assert not unknown, f"unknown segment(s) {unknown}; the segments are {SEGMENT_KEYS}"
    total = sum(shares.values())
    assert abs(total - 1.0) < 1e-6, f"the segment shares sum to {total}, not 1"
    passes = segment_passes or {}
    return sum(share * passes.get(key, DEFAULT_COUNT) for key, share in shares.items())


def flops_per_token_at(cfg: ModelConfig, seq_len: int, dec_fraction: float, passes: float, row_reads: float | None = None,
                       **kwargs) -> dict[str, float]:
    """flops_per_token at a mean pass count `passes` that need not be an integer (CED v2 with decoder self-KV): the
    decoder positions (training) or the generated tokens (decode) run `passes` passes on average, pass 1 at each and
    the later ones where the schedule asks for them. A later pass costs the same whichever it is (it applies the
    block without k_proj / v_proj and makes the block's reads over pass 1's keys: flops_per_token's T = 2 minus its
    T = 1), so the cost is linear in the mean:

        f(passes) = f(T = 1) + (passes - 1) * (f(T = 2) - f(T = 1))

    which is flops_per_token itself at passes = 1, 2 (and 3, where a third pass is defined like the second). Reading
    a context token (prefill_context) has no decoder pass and does not depend on it. Padding slots of a pass's fixed
    slot size are not counted: this is the function's cost, an estimate of a step's from below.

    The control with a gated second pass through its top block (ced/model/control.py, "Segment-based pass counts"):
    the same line. In training every position of a row is a unit, so `passes` is the mean over ALL positions; at
    decode it is the mean over the generated tokens; a read token pays nothing. flops_per_token has no T = 3 for this
    arch (its forward takes T <= 2): a third pass is one more f(T = 2) - f(T = 1), as the line says.
    row_reads (the looped control, training only; None: not applied): the number of later passes that RUN in a
    training forward (0, 1 or 2; a mean over the forwards). On the fused GDN kernel a later pass's state read runs
    over the whole row whatever the number of its slots (ced/model/control.py), so with it the block's GDN read
    term r = n_g 2 H_v d_k d_v is counted once per position for each later pass that runs, not once per slot:

        training = f(1) + (passes - 1) * (f(2) - f(1) - 3 r) + row_reads * 3 r

    which is the plain line when every position runs every later pass (row_reads = passes - 1)."""
    control = cfg.arch == "control" and cfg.loop.pass2_gate != "none"
    assert control or (cfg.arch == "ced_v2" and cfg.decoder.self_kv), \
        "a pass count per position is built for CED v2 with decoder self-KV and for the control with a second pass"
    assert passes >= 1.0, passes
    one = flops_per_token(cfg, seq_len, dec_fraction, 1, **kwargs)
    two = flops_per_token(cfg, seq_len, dec_fraction, 2, **kwargs)
    out = {k: one[k] + (passes - 1.0) * (two[k] - one[k]) for k in one}
    if row_reads is not None:
        assert control, "row_reads is the looped control's GDN state read over the whole row"
        g = cfg.encoder.gdn
        n_gdn = cfg.encoder.layer_kinds()[cfg.encoder.n_layers - cfg.loop.pass2_layers:].count("gdn")
        read = n_gdn * 2 * g.n_v_heads * g.k_head_dim * g.v_head_dim
        out["training"] += 3 * read * (row_reads - (passes - 1.0))
    return out


def segment_flops(cfg: ModelConfig, seq_len: int, dec_fraction: float, train_shares: dict[str, float],
                  decode_shares: dict[str, float], segment_passes: dict | None = None, **kwargs) -> dict[str, float]:
    """FLOPs per token under a segment pass map (default: the config's own loop.segment_passes): training at the mean
    pass count of the decoder positions of the training mix (train_shares: the share of decoder positions per
    segment, "other" included), decode at the mean pass count of generated tokens (decode_shares). Returns
    flops_per_token's keys plus "mean_passes_training" and "mean_passes_decode"."""
    passes = cfg.loop.segment_passes if segment_passes is None else segment_passes
    m_train, m_decode = mean_passes(passes, train_shares), mean_passes(passes, decode_shares)
    train = flops_per_token_at(cfg, seq_len, dec_fraction, m_train, **kwargs)
    decode = flops_per_token_at(cfg, seq_len, dec_fraction, m_decode, **kwargs)
    return {"training": train["training"], "prefill_context": train["prefill_context"],
            "decode_generated": decode["decode_generated"], "mean_passes_training": m_train,
            "mean_passes_decode": m_decode}
