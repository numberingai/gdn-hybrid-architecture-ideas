"""Scorer for the v2 ladder probes (uuid_needle, multikey, path_copy, depth; generators in
ced/eval/probe_gen.py, build tool tools/build_probes.py, CLI scripts/eval_probes.py, design and
numbers).

Teacher-forced, answer positions only. An item is ids = [context tokens][answer tokens]; the model
reads ids[:-1] and the positions that PREDICT answer tokens are the scored positions:

    position  ... c_last  a_1  ...  a_{m-1}
    target    ...   a_1   a_2  ...   a_m
    dec_mask  ...  True  True  ...  True        every other position: False

- Arches with a decoder (looped_ced, ced_v2, second_pass) run their decoder / main head exactly at those
  positions (dec_mask = the scored positions, capacity >= their count, keep-first-C; no position may be dropped)
  and are scored from `hidden_dec` at the slots, as they would generate the answer. Controls are scored from
  `hidden_aux` (their single head) at the same positions. This is the convention of ced/eval/qa_probe.py,
  span_bpb.py and multiple_choice.py. Exception, decoder_at="every" (score_sets; scripts/eval_probes.py
  --decoder-at every): the decoder runs at every position of every item (capacity = the row length) and the answer
  positions are read from its slots. For CED v2 without self-KV and control (c) both modes give the same scores
  (their decoder writes nothing). "answers" on a self-KV model (stage 1, stage 2's LLLF): the decoder runs at the
  scored positions alone, so the context positions have their memory entries and no decoder K/V; a full layer
  reads the memory of s <= t plus the own K/V of the answer slots s <= t, a local layer only the answer slots' own
  K/V in its window (no sinks: the document's first positions are context and run no decoder).
  tests/test_stage3_zero.py checks both regimes against a per-position construction. decoder_at="answers_sinks"
  ("Sinks"): "answers" plus each item's first n_sinks positions (document positions < decoder.n_sinks, the local
  layers' sinks; 4 in stage 2). Those run the decoder and write decoder K/V, so the local layers see the sinks as
  in training and the full layers see their own keys too; their outputs are not scored. Capacity: the answers plus
  the sinks, rounded up as for "answers". Without local layers it is "answers" plus those slots (without self-KV:
  equal to "answers"). decoder_at="answers_sinks_replay" ("replay window"; score_sets replay_window W, default
  128): "answers_sinks" plus each item's replay positions, the min(W, available) input positions of the item
  immediately before its first scored position (in row_tensors' layout an item is one document at [start, start +
  n), its scored positions the last m, so the replay is [max(start, start + n - m - W), start + n - m): never a
  position of another item or of the padding document). A bounded replay of the decoder's sliding-window state
  before generation: the replay positions run the decoder and write decoder K/V, so the first answer token's local
  window holds own keys of the question as in training; their outputs are not scored. It is not "every" unless the
  sinks and the replay together cover the item's whole context: a replay position's own window and its full-layer
  own keys start at the replay's first position (plus the sinks), so its K/V at the deeper layers are not those of
  training (tests/test_stage3_zero.py: one position short of the context already differs). Capacity: the answers
  plus the distinct extra positions (min(n - m, n_sinks + W) per item: sinks and replay overlap in a short
  context), rounded up as for "answers". Without self-KV: equal to "answers". Per-item windows (score_sets
  replay_windows; replay of the long-context suite, scripts/eval_long.py --replay-window turn): item i of set s
  replays its own min(replay_windows[s][i], available) positions before its first scored position instead of the
  one W (0: none, then the item is scored as under "answers_sinks"); the capacity counts each item's own window.
- Optional inference-time length scale (`length_scaled_reads`; scripts/eval_probes.py --length-scale):
  every full-attention read's query times max(1, ln(pos + 1) / ln 8192), local decoder layers unscaled.
- Optional inference-time rotary distance clamp (`set_rope_clamp`; scripts/eval_probes.py --rope-clamp C): every
  full-attention read's rotary part computed with min(distance, C); set on the config before the model is built.
- The LM head is applied only at the scored positions, in float32 with TF32 off (`head_weight`:
  model.head_weight() on the v2 models, the tied embedding on v1).
- Long items are one row each (batch 1), padded to a fixed bucket length (8k, 16k, 32k, 64k, then
  multiples of 8k) with an inert padding document, so a compiled kernel sees a handful of shapes.
  The decoder capacity is set per (row length, single / packed), so a long item's score does not
  depend on which short items share its row length.
  Short items (path_copy, depth) are packed whole into 8,192-token rows, one document each; both
  model families isolate documents (attention mask, GDN/conv state reset, RoPE restart), so a packed
  item scores like the item alone (tests/test_probes.py checks this against a direct log-softmax
  of each item alone).

Per item: nats = -sum log p(answer tokens); n_correct = answer tokens whose argmax is the target;
exact = all correct; first_correct = the argmax is right at the first answer token that is not
pure whitespace (the token that decides WHICH value is copied; path_copy: the first choice token,
first_index), with tok_logp / tok_correct for every answer token (a metric can be redefined
without rescoring); for multiple choice (single-token
candidates, read from the same log-softmax at the one answer position): the candidates'
log-probs, pred = their argmax, mc_correct, greedy = full-vocabulary argmax is the gold token,
distractor = pred is one of the item's distractor candidates. Aggregates per (probe, cell) and per
(probe, cell, group) are item means with item-level standard errors (std / sqrt(n), ddof 1).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

DECODER_ARCHES = ("looped_ced", "ced_v2", "second_pass")
PACK_LEN = 8192
PACK_MAX_ITEM = 4096  # items with at most this many inputs are packed
# Resumable scoring (score_sets unit_store): a unit of rows is closed, at a batch boundary, when the next batch would pass
# UNIT_TOKENS input tokens or when UNIT_SECONDS have gone by, whichever comes first. The scorers differ by a factor of 200
# in tokens/s (MEASURED in the delivered logs: 69k to 212k on the usual needle rows, 4k to 17k on the transcode rows;
# ASEntmax's exact reference path about 9 s per 64k row = 7.3k), so tokens alone cannot bound what a stop costs: 4M tokens
# are 19 to 58 s on the usual rows and 9 minutes on ASEntmax's 64k rows; the time bound closes the slower ones after 5
# minutes (plus the batch in flight).
UNIT_TOKENS = 4_000_000
UNIT_SECONDS = 300.0
CAPACITY_MULTIPLE = 128  # decoder slots per row: the answer positions, rounded up to this
DECODER_AT = ("answers", "answers_sinks", "answers_sinks_replay", "every")  # score_sets decoder regimes (module docstring)
REPLAY_WINDOW = 128  # default replay_window of "answers_sinks_replay": stage 2's local window
LN2 = math.log(2.0)


# --------------------------------------------------------------------------------------------
# model helpers
# --------------------------------------------------------------------------------------------


# per-item set fields copied into the results: where the needle sits; path_copy's first choice
# token (distinct tokens there, listed paths sharing the gold token); the item text's hash (pairs
# runs across tokenizers)
# Extension probes (ced/eval/probe_gen_ext.py): the item's floors (a dict), the answer-token indices of
# the tokens a near-duplicate decoy changes (copy), and the question a line_edit / line_edit_foil row
# belongs to (base id, the line number it answers, gold or foil)
# Copying probes (ced/eval/probe_gen_copy.py): the realised distance back to the span (tokens, under the
# set's tokenizer) and the unescaped old text's byte count is answer_bytes (edit_old)
# Multi-fact probes (ced/eval/probe_gen_multifact.py): which answer tokens are fact tokens, so that a scoring output of
# scripts/eval_probes.py (the grid's 128k sets) carries what the fact metrics need (tools/analysis/auto/lc_curves.py)
PASS_THROUGH = ("needle_depth_tokens", "choice_options", "choice_share", "text_sha", "floor", "diff_index", "base", "line", "role",
                "dist_real", "fact_spans")
REST_KEYS = ("nats_per_token_rest", "token_acc_rest", "exact_rest")  # span probes: the metrics on answer tokens 2..n (span_metrics)
TAUS = (0.7,)  # sampling temperatures scored besides 1.0 (tempered: log_softmax(logits / tau)); copy_span and edit_old report them
SPAN_PROBES = ("copy_span", "edit_old", "transcode", "sb_escape")
# Fixed-distance needles (ced/eval/probe_gen_distance.py): the needle's distance from the first answer query in tokens
# under the set's tokenizer (the request is in the item's group). Added here, apart from the tuple above, so that
# branches which extend that tuple merge without a conflict.
PASS_THROUGH += ("needle_distance_tokens",)
# Latent two-hop (ced/eval/probe_gen_twohop.py): the base item a scored item belongs to, its variant (twohop, onehop,
# link, shuffled), its settings and where the two facts sit in tokens: what the paired summary groups and pairs by.
PASS_THROUGH += ("twohop",)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    inner = model
    while hasattr(inner, "module") or hasattr(inner, "_orig_mod"):
        inner = getattr(inner, "module", None) or getattr(inner, "_orig_mod")
    return inner


def has_decoder(model: torch.nn.Module) -> bool:
    arch = unwrap(model).cfg.arch
    return arch in DECODER_ARCHES


def self_kv_decoder(model: torch.nn.Module) -> bool:
    """CED v2 with decoder self-KV: its decoder positions write K/V that later decoder positions read,
    so which positions run the decoder changes the scores."""
    cfg = unwrap(model).cfg
    return cfg.arch == "ced_v2" and bool(cfg.decoder.self_kv)


def decoder_sinks(model: torch.nn.Module) -> int:
    """The decoder's n_sinks (DecoderConfig.n_sinks): a local decoder layer's sinks are the own K/V at the first
    n_sinks positions of the document (decoder_at "answers_sinks" / "answers_sinks_replay" run the decoder there)."""
    return int(unwrap(model).cfg.decoder.n_sinks)


def head_weight(model: torch.nn.Module) -> torch.Tensor:
    """The LM head matrix (V, d): model.head_weight() on the v2 models (untied output embedding),
    the tied input embedding on the v1 models."""
    inner = unwrap(model)
    if hasattr(inner, "head_weight"):
        return inner.head_weight()
    return inner.embed.weight


def compile_layers(model: torch.nn.Module) -> int:
    """torch.compile every encoder / decoder layer in place, as the trainer does for compile "all"
    (CUDA only; static shapes). Returns the number of layers compiled."""
    inner = unwrap(model)
    layers = list(getattr(getattr(inner, "encoder", None), "layers", []) or [])
    layers += list(getattr(inner, "decoder_layers", []) or [])
    layers += list(getattr(getattr(inner, "decoder", None), "layers", []) or [])
    for layer in layers:
        layer.compile(dynamic=False)
    return len(layers)


def model_device(model: torch.nn.Module) -> torch.device:
    for t in (*model.parameters(), *model.buffers()):
        return t.device
    return torch.device("cpu")


# --------------------------------------------------------------------------------------------
# inference-time length scale (scripts/eval_probes.py --length-scale)
# --------------------------------------------------------------------------------------------


LENGTH_SCALE_REF = 8192  # L: the 60M runs' training row length (attn_diag.TRAIN_LEN)


def length_scale_modules(model: torch.nn.Module) -> dict[str, list]:
    """The GatedAttention modules of the inference-time length scale (`length_scaled_reads`), by role:

    encoder        the encoder's global (full-attention) layers: control (a)'s and CED v2's; scaled
    decoder_full   CED v2's decoder layers that read the memory (all of them without a read pattern; the F
                   layers of stage 2's LLLF); scaled, in every pass of the loop (one module per layer)
    decoder_local  stage 2's local decoder layers (n_sinks sinks plus the last local_window own keys); NOT scaled

    Control (a) and CED v2 only (arch "control" / "ced_v2"; v1 and the second-pass arch are refused)."""
    inner = unwrap(model)
    arch = inner.cfg.arch
    if arch not in ("control", "ced_v2"):
        raise ValueError(f"the length scale is built for control (a) and CED v2, not arch {arch!r}")
    out = {"encoder": [layer.mixer for layer in inner.encoder.layers if layer.kind == "attention"],
           "decoder_full": [], "decoder_local": []}
    for layer in getattr(inner, "decoder_layers", None) or []:
        out["decoder_local" if layer.local else "decoder_full"].append(layer.attn)
    return out


@contextlib.contextmanager
def length_scaled_reads(model: torch.nn.Module, ref_len: int = LENGTH_SCALE_REF):
    """        max(1, ln(n_t) / ln(L)),   n_t = pos_t + 1,  L = ref_len (8,192: the training row length)

    through GatedAttention.query_scale (attn_diag.logn_scale, the helper of scripts/diag_length.py test 8), so the
    reference, flex and sdpa paths, compiled or not, get it where the temperature is applied (after QK-norm and RoPE).
    Which reads (length_scale_modules):
    - the encoder's global layers: n_t = pos + 1 is exactly the keys a query sees (its document's positions s <= t);
    - CED v2's decoder FULL layers: n_t = pos + 1 as well, the query's MEMORY entries. Their own keys are not
      counted. In the regime the model trained in (decoder everywhere, every position has one own key per layer) a
      full-layer query sees 2 (pos + 1) keys, and 2 L at the training length, so the exact count would give
      ln(2n) / ln(2L) instead of ln(n) / ln(L): at n = 64k, 17 / 14 = 1.214 against 16 / 13 = 1.231 (16k: 1.071
      against 1.077), a difference of at most 0.017 in the factor over the probes' lengths. With the decoder at
      the answer positions only (stage 3's restricted regime) the own keys are the few answer positions so far
      and n_t = pos + 1 is the count to within those. One count serves both regimes, so their comparison is not
      confounded by a different factor;
    - CED v2's decoder LOCAL layers (stage 2's L layers: sinks plus the last local_window own keys, at most
      n_sinks + local_window keys whatever the length): unscaled.
    The factor is exactly 1 for n_t <= L (computed as 1 + ln(n / L) / ln(L), clamped), so rows up to L tokens score
    bit for bit as without it. Layers trained with a length-dependent scale already (temperature "ssmax", ASEntmax)
    are refused: the factor would come on top of their own. A learned or fixed temperature takes it on top (the base
    recipe of 2026-10-09: a learned per-head scale plus this factor). The modules' query_scale is restored on exit
    (also on an error). Yields what was scaled: {"ref_len", "encoder_layers", "decoder_full_layers",
    "decoder_local_layers_unscaled"}."""
    try:
        from ced.eval.attn_diag import logn_scale
    except ImportError as e:  # the diagnostics harness is not part of the public release
        raise RuntimeError("length-scaled reads need ced/eval/attn_diag.py, which this release does not include") from e

    assert ref_len >= 2, ref_len
    mods = length_scale_modules(model)
    scaled = mods["encoder"] + mods["decoder_full"]
    for m in scaled:
        if m.cfg.temperature == "ssmax" or m.cfg.normalizer == "asentmax":
            raise ValueError(f"the length scale on a layer trained with temperature {m.cfg.temperature!r} / normalizer "
                             f"{m.cfg.normalizer!r}: its query is length-scaled already")
    assert all(m.query_scale is None for m in scaled + mods["decoder_local"]), \
        "length_scaled_reads does not nest over another query_scale"
    f = logn_scale(ref_len)
    try:
        for m in scaled:
            m.query_scale = f
        yield {"ref_len": ref_len, "encoder_layers": len(mods["encoder"]), "decoder_full_layers": len(mods["decoder_full"]),
               "decoder_local_layers_unscaled": len(mods["decoder_local"])}
    finally:
        for m in scaled:
            m.query_scale = None


# --------------------------------------------------------------------------------------------
# inference-time rotary distance clamp (scripts/eval_probes.py / eval_long.py --rope-clamp)
# --------------------------------------------------------------------------------------------


def set_rope_clamp(model_cfg, clamp: int) -> dict:
    """Score with the rotary distance clamped: set AttentionConfig.rope_max_distance = clamp on the model config of a
    loaded checkpoint, BEFORE the model is built, so every full-attention read computes its rotary part with
    min(distance, clamp) and nothing else changes (ced/model/attention.py, "Rotary distance cap"; the option adds no
    parameter, so the checkpoint loads as it is). The same operation as the diagnostic harness's rope_clamp_all
    (ced/eval/attn_diag.py; test 9), through the model's own attention paths, compiled or not.

    Which reads: the encoder's global layers (control (a)'s, CED v2's) and CED v2's decoder memory reads, every pass,
    in every decoder regime (the read is a slot read over the memory wherever the decoder runs). The inference length
    scale (length_scaled_reads) composes: it multiplies both rotations of the query.

    Rules, per role (encoder, decoder):
    - no rotary dimensions (rope_fraction 0.0, a NoPE arm): nothing to clamp; left as it is;
    - trained without a cap: clamped at `clamp`;
    - trained WITH a cap C: a clamp >= C changes nothing (every distance is already read as at most C) and the config
      is left as it is; a clamp below C would merge distances the model was trained to tell apart and is refused.
    - the by-layer mix (EncoderConfig.nope_global_layers, always trained with a cap): the rule above for its capped
      layers; its NoPE layers have nothing to clamp, which a note says.
    Refused: arches other than "control" and "ced_v2"; CED v2 with decoder self-KV (stage 1, stage 2: the own keys of
    a full layer and the local layers' read are not built); the asentmax normaliser.

    Returns the record the scripts keep in their settings: {"clamp", "encoder": the encoder's effective cap or None,
    "decoder": the decoder's (None for the control), "notes": [what was left as it was and why]}."""
    from ced.model.rope import rotary_dims

    if isinstance(clamp, bool) or not isinstance(clamp, int) or clamp < 1:
        raise ValueError(f"--rope-clamp {clamp!r}: an integer >= 1 (a distance in tokens)")
    arch = model_cfg.arch
    if arch not in ("control", "ced_v2"):
        raise ValueError(f"the rotary distance clamp is built for control (a) and CED v2, not arch {arch!r}")
    if arch == "ced_v2" and model_cfg.decoder.self_kv:
        raise ValueError("the rotary distance clamp is not built for CED v2 with decoder self-KV (stage 1, stage 2): its "
                         "decoder reads its own keys, and its local layers their sinks, without the clamp")
    roles = [("encoder", model_cfg.encoder.attention)] + ([("decoder", model_cfg.decoder.attention)] if arch == "ced_v2" else [])
    out = {"clamp": clamp, "encoder": None, "decoder": None, "notes": []}
    for role, a in roles:
        if a.normalizer != "softmax":
            raise ValueError(f"the rotary distance clamp is built for the softmax normaliser, not the {role}'s {a.normalizer!r}")
        if role == "encoder" and model_cfg.encoder.nope_layers():
            out["notes"].append(f"encoder: global layers {list(model_cfg.encoder.nope_layers())} (bottom first) are NoPE "
                                "(nope_global_layers): nothing to clamp there")
        trained = a.rope_max_distance
        if rotary_dims(a.head_dim, a.rope_fraction) == 0:
            out["notes"].append(f"{role}: no rotary dimensions (rope_fraction {a.rope_fraction}): nothing to clamp")
        elif trained is None:
            a.rope_max_distance = out[role] = clamp
        elif clamp >= trained:
            out[role] = trained
            out["notes"].append(f"{role}: trained with a cap of {trained} <= the clamp: unchanged")
        else:
            raise ValueError(f"--rope-clamp {clamp} is below the {role}'s trained cap of {trained}: it would merge distances "
                             "the model was trained to tell apart")
    return out


# --------------------------------------------------------------------------------------------
# encoded probe sets
# --------------------------------------------------------------------------------------------


@dataclass
class EncodedSet:
    """One probe's items under one tokenizer. ids of item i: ids[offsets[i]:offsets[i + 1]]."""

    probe: str
    ids: torch.Tensor  # (sum of item lengths,) int32
    offsets: torch.Tensor  # (n + 1,) int64
    n_answer: torch.Tensor  # (n,) int32
    first_info: torch.Tensor  # (n,) int32: index in the answer of the first non-whitespace token (path_copy: first choice)
    answer_bytes: torch.Tensor  # (n,) int32
    items: list[dict]  # per item: id, cell, group (and for multiple choice: gold, distractors)
    cand_ids: torch.Tensor | None = None  # (n, n_cand) int32 single-token candidates (multiple choice)
    info: dict = field(default_factory=dict)  # tokenizer, counts, ...

    def __len__(self) -> int:
        return len(self.items)

    def item_ids(self, i: int) -> torch.Tensor:
        return self.ids[int(self.offsets[i]) : int(self.offsets[i + 1])]

    def length(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])

    def fingerprint(self, i: int) -> str:
        h = hashlib.blake2b(digest_size=8)
        h.update(self.item_ids(i).numpy().astype(np.int32).tobytes())
        h.update(int(self.n_answer[i]).to_bytes(4, "little"))
        if self.cand_ids is not None:
            h.update(self.cand_ids[i].numpy().astype(np.int32).tobytes())
        return h.hexdigest()

    def save(self, path: str | Path) -> None:
        torch.save({"probe": self.probe, "ids": self.ids, "offsets": self.offsets, "n_answer": self.n_answer,
                    "first_info": self.first_info, "answer_bytes": self.answer_bytes, "items": self.items,
                    "cand_ids": self.cand_ids, "info": self.info}, path)

    @classmethod
    def load(cls, path: str | Path) -> "EncodedSet":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**d)

    def subset(self, keep: list[int]) -> "EncodedSet":
        parts = [self.item_ids(i) for i in keep]
        lengths = torch.tensor([len(p) for p in parts], dtype=torch.int64)
        return EncodedSet(
            probe=self.probe, ids=torch.cat(parts) if parts else self.ids[:0],
            offsets=torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)]),
            n_answer=self.n_answer[keep], first_info=self.first_info[keep], answer_bytes=self.answer_bytes[keep],
            items=[self.items[i] for i in keep], cand_ids=None if self.cand_ids is None else self.cand_ids[keep],
            info=dict(self.info))


def encode_items(items, encode, probe: str | None = None) -> EncodedSet:
    """Encode text items (probe_gen.TextItem) with `encode`: str -> TokenizedDoc (ids, byte_lens).
    Context and answer are tokenized separately and concatenated (as in qa_probe). Multiple-choice
    candidates must be single tokens."""
    ids, lengths, n_answer, first_info, answer_bytes, meta, cands = [], [], [], [], [], [], []
    for it in items:
        ctx, ans = encode(it.context), encode(it.answer)
        assert ans.ids and ctx.ids, it.id
        seq = list(ctx.ids) + list(ans.ids)
        ids.append(torch.tensor(seq, dtype=torch.int32))
        lengths.append(len(seq))
        n_answer.append(len(ans.ids))
        answer_bytes.append(len(it.answer.encode("utf-8")))
        rec = {"id": it.id, "cell": it.cell, "group": it.group, "text_sha": text_sha(it.context, it.answer)}
        if it.probe == "path_copy":
            k, options, share = first_choice_info(list(ans.ids), [list(encode(" " + p).ids) for p in it.meta["paths"]])
            first_info.append(k)
            rec["choice_options"], rec["choice_share"] = options, share
        else:
            first_info.append(_first_informative(it.answer, ans))
        if it.candidates is not None:
            cand = []
            for c in it.candidates:
                t = encode(c).ids
                assert len(t) == 1, f"{it.id}: candidate {c!r} is {len(t)} tokens; only single-token candidates are supported"
                cand.append(int(t[0]))
            assert cand[it.gold] == seq[-1] and len(ans.ids) == 1, it.id
            cands.append(cand)
            rec["gold"] = it.gold
            rec["distractors"] = [it.candidates.index(str(d)) for d in it.meta.get("distractor_digits", [])]
        meta.append(rec)
    probe = probe or (items[0].probe if items else "")
    return EncodedSet(
        probe=probe, ids=torch.cat(ids) if ids else torch.zeros(0, dtype=torch.int32),
        offsets=torch.cat([torch.zeros(1, dtype=torch.int64), torch.tensor(lengths, dtype=torch.int64).cumsum(0)]),
        n_answer=torch.tensor(n_answer, dtype=torch.int32), first_info=torch.tensor(first_info, dtype=torch.int32),
        answer_bytes=torch.tensor(answer_bytes, dtype=torch.int32), items=meta,
        cand_ids=torch.tensor(cands, dtype=torch.int32) if cands else None)


def text_sha(context: str, answer: str) -> str:
    """Tokenizer-independent identity of an item's text (pairs the same item across tokenizers)."""
    h = hashlib.blake2b(digest_size=8)
    h.update(context.encode("utf-8") + b"\x00" + answer.encode("utf-8"))
    return h.hexdigest()


def first_choice(answer_ids: list[int], listed: list[list[int]]) -> tuple[int, int]:
    """path_copy: (index of the first answer token at which the asked path departs from another
    listed path, number of distinct tokens the listed paths have there). Every earlier answer token
    is shared by all listed paths, so under teacher forcing this is the first token that decides
    which path is copied. `listed` are the token sequences of every listed path in answer form
    (" " + path), the asked one included. See first_choice_info for the floor."""
    k, options, _ = first_choice_info(answer_ids, listed)
    return k, options


def first_choice_info(answer_ids: list[int], listed: list[list[int]]) -> tuple[int, int, int]:
    """first_choice plus `share`: how many listed paths (the asked one included) have the asked
    path's token at that index. A copier that copies a random listed path perfectly gets the token
    right share / K of the time (the copy-without-binding floor of first_acc); 1 / options is that
    floor only when no two listed paths share a token there (share = 1). The one-choice cells are
    built so that share = 1 under every tokenizer (probe_gen.GROUP_WORDS; build check)."""
    others = [x for x in listed if x != answer_ids]
    assert len(others) == len(listed) - 1 and others, "the asked path must be listed exactly once"

    def diverge(x: list[int]) -> int:
        for j, (a, b) in enumerate(zip(answer_ids, x)):
            if a != b:
                return j
        return min(len(answer_ids), len(x))

    k = min(diverge(x) for x in others)
    assert k < len(answer_ids), "the asked path is a prefix of another listed path"
    options = len({x[k] for x in listed if len(x) > k})
    share = sum(1 for x in listed if len(x) > k and x[k] == answer_ids[k])
    return k, options, share


def _first_informative(answer: str, tok) -> int:
    """Index of the first answer token that covers a non-whitespace character (from byte_lens)."""
    data = answer.encode("utf-8")
    at = 0
    for k, n in enumerate(tok.byte_lens):
        piece = data[at : at + n]
        at += n
        if piece.strip():
            return k
    return 0


# --------------------------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------------------------


@dataclass
class Row:
    length: int  # L (bucket)
    members: list[tuple[int, int, int, int]] = field(default_factory=list)  # (set index, item index, start, n_inputs)
    used: int = 0
    packed: bool = True  # False: a long item alone in its row

    @property
    def kind(self) -> str:
        return "packed" if self.packed else "single"


def bucket_length(n_inputs: int, base: int = PACK_LEN) -> int:
    """Smallest of base x {1, 2, 4, 8} (8k, 16k, 32k, 64k by default) that holds n_inputs plus one
    padding position, beyond that the next multiple of base."""
    L = base
    while L < n_inputs + 1 and L < 8 * base:
        L *= 2
    if L < n_inputs + 1:
        L = math.ceil((n_inputs + 1) / base) * base
    return L


def plan_rows(sets: list[EncodedSet], pack_len: int = PACK_LEN, pack_max_item: int = PACK_MAX_ITEM) -> list[Row]:
    """Long items one per row (bucketed length); short ones packed first-fit in item order into rows
    of pack_len, each row keeping at least one padding position. Deterministic."""
    rows: list[Row] = []
    open_rows: list[Row] = []
    for s, es in enumerate(sets):
        for i in range(len(es)):
            n = es.length(i) - 1
            if n > pack_max_item:
                rows.append(Row(bucket_length(n, pack_len), [(s, i, 0, n)], n, packed=False))
                continue
            for r in open_rows:
                if r.used + n + 1 <= r.length:
                    r.members.append((s, i, r.used, n))
                    r.used += n
                    break
            else:
                r = Row(pack_len, [(s, i, 0, n)], n)
                open_rows.append(r)
                rows.append(r)
    return rows


def row_tensors(row: Row, sets: list[EncodedSet], pad_id: int = 0):
    """(tokens, targets, doc_ids, dec_mask) of one row, each (L,), and the scored positions per member."""
    L = row.length
    tokens = torch.full((L,), pad_id, dtype=torch.long)
    targets = torch.full((L,), pad_id, dtype=torch.long)
    doc_ids = torch.full((L,), len(row.members), dtype=torch.long)  # padding: a fresh, larger id
    dec = torch.zeros(L, dtype=torch.bool)
    positions = []
    for d, (s, i, start, n) in enumerate(row.members):
        ids = sets[s].item_ids(i).long()
        tokens[start : start + n] = ids[:-1]
        targets[start : start + n] = ids[1:]
        doc_ids[start : start + n] = d
        m = int(sets[s].n_answer[i])
        pos = torch.arange(start + n - m, start + n)
        dec[pos] = True
        positions.append(pos)
    return tokens, targets, doc_ids, dec, positions


def sink_mask(row: Row, n_sinks: int) -> torch.Tensor:
    """(L,) the first n_sinks positions of each member item of the row (document positions < n_sinks; an item
    shorter than that: all of it). The padding document gets none."""
    mask = torch.zeros(row.length, dtype=torch.bool)
    for _, _, start, n in row.members:
        mask[start : start + min(n_sinks, n)] = True
    return mask


def replay_mask(row: Row, sets: list[EncodedSet], window: int, windows=None) -> torch.Tensor:
    """(L,) each member item's replay positions (decoder_at "answers_sinks_replay"): the min(window, available) input
    positions of the item immediately before its first scored position. row_tensors lays a member out as one
    document at [start, start + n) whose scored positions are the last m = n_answer, so the first scored position
    is start + n - m and `available` is n - m (the context but its last token, which is the first scored position).
    Clamped at the item's start: never a position of the member before it, and none of the padding document.
    windows (score_sets replay_windows): windows[s][i] is item i of set s's own window (>= 0) instead of `window`."""
    assert windows is not None or window >= 1, window
    mask = torch.zeros(row.length, dtype=torch.bool)
    for s, i, start, n in row.members:
        first = start + n - int(sets[s].n_answer[i])
        assert first >= start, (s, i)
        w = window if windows is None else int(windows[s][i])
        assert w >= 0, (s, i, w)
        mask[max(start, first - w) : first] = True
    return mask


# --------------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def head_logprobs(h: torch.Tensor, weight: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """log_softmax(h @ weight.T) in float32 or wider (TF32 off) for h (P, d): (P, V)."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out = []
        dtype = torch.promote_types(h.dtype, torch.float32)  # float32, or float64 in the CPU tests
        with torch.autocast(h.device.type, enabled=False):
            W = weight.to(dtype)
            for a in range(0, h.shape[0], chunk):
                out.append(torch.log_softmax(h[a : a + chunk].to(dtype) @ W.T, dim=-1))
        return torch.cat(out) if out else h.new_zeros(0, weight.shape[0], dtype=torch.promote_types(h.dtype, torch.float32))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


@torch.no_grad()
def tempered_target_logp(logp: torch.Tensor, target: torch.Tensor, tau: float, chunk: int = 256) -> torch.Tensor:
    """log softmax(logits / tau) at the target, from logp = log_softmax(logits) (P, V): the shift by the
    log-partition does not change the softmax of logits / tau, so log_softmax(logp / tau) is the tempered
    distribution. (P,) float64 on the CPU. tau = 1 returns logp at the target. No top-p, no top-k."""
    out = []
    for a in range(0, logp.shape[0], chunk):
        x = logp[a : a + chunk]
        if tau != 1.0:
            x = torch.log_softmax(x / tau, dim=-1)
        out.append(x.gather(1, target[a : a + chunk, None]).squeeze(1).double().cpu())
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.float64)


@torch.no_grad()
def _answer_hidden(model, tokens, doc_ids, dec, positions, capacity, T, varlen, decoder: bool,
                   run: torch.Tensor | None = None):
    """Hidden states (B, P, d) feeding the head at the scored positions of each row, P = capacity
    (decoder) or the widest row's count (control), plus the (B, P) validity.

    run (decoder_at "every" / "answers_sinks" / "answers_sinks_replay"): the decoder runs at these positions
    instead (every input position of every item, capacity = L; or the scored positions plus each item's sinks,
    and with the replay its replay positions, capacity >= their count), and the slots of the scored positions
    are read out; P is then the widest row's count of scored positions."""
    B = tokens.shape[0]
    if decoder and run is not None:
        # A model with a pass schedule (loop.unscored_passes; ced/model/pass_schedule.py) is told which of the run
        # positions are scored: the sinks and the replay positions are not, and may run pass 1 only. Any other model
        # gets the call it always got.
        sched = {"scored": dec} if getattr(unwrap(model), "schedule", None) is not None else {}
        out = model(tokens, doc_ids, run, T=T, capacity=capacity, return_hidden=True, varlen=varlen, **sched)
        assert int(out.n_dropped.sum()) == 0 and torch.equal(out.kept, run), "the decoder did not run at exactly the run positions"
        P = max(len(p) for p in positions)
        index = torch.zeros(B, P, dtype=torch.long)
        valid = torch.zeros(B, P, dtype=torch.bool)
        rank = run.long().cumsum(1).cpu() - 1  # keep-first-C: the j-th run position is slot j
        for b, p in enumerate(positions):
            index[b, : len(p)] = rank[b, p]
            valid[b, : len(p)] = True
            assert torch.equal(out.slot_index[b, rank[b, p]].cpu(), p), "the decoder slots do not cover the answer positions"
        index = index.to(tokens.device)
        h = torch.gather(out.hidden_dec, 1, index[..., None].expand(B, P, out.hidden_dec.shape[-1]))
        return h, valid.to(tokens.device)
    if decoder:
        out = model(tokens, doc_ids, dec, T=T, capacity=capacity, return_hidden=True, varlen=varlen)
        assert int(out.n_dropped.sum()) == 0, "answer positions were dropped: raise the capacity"
        idx, valid = out.slot_index, out.slot_valid
        for b in range(B):
            n = len(positions[b])
            assert torch.equal(idx[b, :n].cpu(), positions[b]) and bool(valid[b, :n].all()) and not bool(valid[b, n:].any()), \
                "the decoder slots are not the answer positions"
        return out.hidden_dec, valid
    out = model(tokens, doc_ids, dec, return_hidden=True, varlen=varlen)
    P = max(len(p) for p in positions)
    index = torch.zeros(B, P, dtype=torch.long)
    valid = torch.zeros(B, P, dtype=torch.bool)
    for b, p in enumerate(positions):
        index[b, : len(p)] = p
        valid[b, : len(p)] = True
    index = index.to(tokens.device)
    h = torch.gather(out.hidden_aux, 1, index[..., None].expand(B, P, out.hidden_aux.shape[-1]))
    return h, valid.to(tokens.device)


@torch.no_grad()
def _answer_logprobs_cp(model, tokens, doc_ids, dec, positions, varlen) -> torch.Tensor:
    """The one-head control with a Copy-Paste head (model.cp_head; ced/model/cp_head.py): log p over V of its
    next-token distribution (the mixture, or p_LM alone under score "lm") at the scored positions, (n, V) in the order
    of _answer_hidden's h[valid] (row by row, each row's positions in order), float32 or wider with TF32 off, as
    head_logprobs."""
    from ced.model.cp_head import head_logprobs_at

    out = model(tokens, doc_ids, dec, return_hidden=True, varlen=varlen)
    B = tokens.shape[0]
    P = max(len(p) for p in positions)
    index = torch.zeros(B, P, dtype=torch.long)
    valid = torch.zeros(B, P, dtype=torch.bool)
    for b, p in enumerate(positions):
        index[b, : len(p)] = p
        valid[b, : len(p)] = True
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        lp = head_logprobs_at(model, out.hidden_aux, tokens, doc_ids, index.to(tokens.device), pointer=out.cp_pointer)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    return lp[valid.to(tokens.device)]


@contextlib.contextmanager
def compiled_block_masks(enabled: bool):
    """Make ced.model.fast_attention build flex BlockMasks with create_block_mask(_compile=True).

    Eager create_block_mask materialises the whole (B, Lq, S) mask grid (and an int64 reduction of
    it): 32 GiB at 64k, more than a MIG slice holds. Training never sees this because its layers
    run under torch.compile, which traces the mask creation. The eval runs the model eagerly, so it
    asks for the compiled mask builder instead (the same block mask; one compile per shape)."""
    if not enabled:
        yield
        return
    import functools

    from ced.model import fast_attention

    original = fast_attention.create_block_mask
    compiled = functools.partial(original, _compile=True)

    def create(*args, **kwargs):  # inside a compiled layer the outer compile traces it anyway
        return original(*args, **kwargs) if torch.compiler.is_compiling() else compiled(*args, **kwargs)

    fast_attention.create_block_mask = create
    try:
        yield
    finally:
        fast_attention.create_block_mask = original


@torch.no_grad()
def score_sets(
    model: torch.nn.Module,
    sets: list[EncodedSet],
    T: int | None = None,
    autocast_dtype: torch.dtype | None = None,
    fused_varlen: bool = False,
    batch_rows: int = 1,
    pack_len: int = PACK_LEN,
    pack_max_item: int = PACK_MAX_ITEM,
    progress=None,
    taus: tuple[float, ...] = TAUS,
    decoder_at: str | None = None,
    replay_window: int = REPLAY_WINDOW,
    replay_windows=None,
    unit_store=None,
    unit_tokens: int = UNIT_TOKENS,
    unit_seconds: float = UNIT_SECONDS,
) -> tuple[list[dict], dict]:
    """Score every item of `sets`. Returns (per-item results in set and item order, timing per row shape
    "<length>/<single|packed>"). batch_rows: rows per forward for rows of length <= pack_len (longer rows: 1). pack_len /
    pack_max_item: the packed row length (also the smallest bucket) and the largest item that is packed (defaults 8,192
    and 4,096; the CPU tests use small values). taus: temperatures besides 1.0 at which the per-token log-prob of the
    target is also recorded (`tok_logp_tau`, keys str(tau)); computed on the device from the same float32 log-softmax.
    decoder_at (decoder arches only): "answers" runs the decoder at the scored positions only; "every" runs it at every
    input position of every item (capacity L), the regime of a model trained with the decoder everywhere, and reads the
    scored positions. The two agree for models whose decoder writes nothing (CED v2 without self_kv, control (c)); with
    decoder self-KV or v1 they differ, "answers" being the restricted regime (no own keys in the context).
    "answers_sinks": "answers" plus each item's first decoder.n_sinks positions (the local layers' sinks; their outputs
    are not scored), capacity the answers plus the sinks rounded up as for "answers". "answers_sinks_replay":
    "answers_sinks" plus each item's replay positions, the min(replay_window, available) input positions of the item
    immediately before its first scored position (replay_mask; they run the decoder and write decoder K/V, not scored),
    capacity the answers plus the distinct extra positions rounded up the same way; replay_window < 1 is an error there,
    and the argument is ignored by every other regime. replay_windows ("answers_sinks_replay" only): per set, a sequence
    of each item's own window (integers >= 0) used instead of replay_window (the module docstring, "Per-item windows").
    None (default): "every" for CED v2 with decoder self_kv, its training regime; "answers" otherwise (every earlier
    score unchanged). unit_store (ced/eval/units.py UnitStore; None: off, the pass as it always was): the rows of each
    (length, kind) shape are scored in units. A unit starts at a row of the shape and is closed after a whole batch when
    the next batch would pass `unit_tokens` input tokens or when `unit_seconds` have gone by (always whole batches from
    the shape's first row, so the batches are those of one pass, wherever the units end). A closed unit's records are
    saved under a key of the shape, its first row and the pass's settings (the store's base), with its rows' members,
    item fingerprints and windows. A rerun follows the chain: at each row it reads the unit that starts there, if its
    file holds exactly the rows the plan has there, and scores from the first row without one. The units' ends are not
    part of the key (they depend on the clock), so a rerun with another unit size continues the same chain. The records
    and the capacity are those of the pass without units (the capacity is a function of the whole shape group); only the
    timing is the sum of what the units took when they were scored."""
    inner = unwrap(model)
    decoder = has_decoder(model)
    if decoder_at is None:
        decoder_at = "every" if self_kv_decoder(model) else "answers"
    if decoder_at not in DECODER_AT:
        raise ValueError(f"decoder_at {decoder_at!r}: one of {DECODER_AT}")
    if decoder_at != "answers" and not decoder:
        raise ValueError(f"decoder_at {decoder_at!r} needs a model with a decoder")
    replay = decoder_at == "answers_sinks_replay"
    if replay and (isinstance(replay_window, bool) or not isinstance(replay_window, int) or replay_window < 1):
        raise ValueError(f"replay_window {replay_window!r}: decoder_at 'answers_sinks_replay' needs an integer >= 1")
    n_sinks = decoder_sinks(model) if decoder_at in ("answers_sinks", "answers_sinks_replay") else 0
    window = replay_window if replay else 0  # 0: no replay positions (every other regime)
    if replay_windows is not None:
        if not replay:
            raise ValueError(f"replay_windows is for decoder_at 'answers_sinks_replay', not {decoder_at!r}")
        if len(replay_windows) != len(sets) or any(len(w) != len(es) for w, es in zip(replay_windows, sets)):
            raise ValueError("replay_windows: one window per item of every set")
        if any(int(x) < 0 for w in replay_windows for x in w):
            raise ValueError("replay_windows: every item's window must be >= 0")

    def item_window(s: int, i: int) -> int:
        return window if replay_windows is None else int(replay_windows[s][i])
    vocab = getattr(inner.cfg, "vocab_size", None)
    for es in sets:
        if vocab is not None and len(es.ids):
            assert int(es.ids.max()) < vocab, f"{es.probe}: token id {int(es.ids.max())} outside the vocabulary {vocab}"
    device = model_device(model)
    from ced.model.cp_head import cp_head_of

    cp = cp_head_of(inner) is not None  # a Copy-Paste head: log p of its next-token distribution (_answer_logprobs_cp)
    if cp and decoder:
        raise ValueError("a Copy-Paste head is built for the one-head control, not a model with a decoder")
    W = None if cp else head_weight(model)
    rows = plan_rows(sets, pack_len, pack_max_item)
    by_shape: dict[tuple[int, str], list[Row]] = defaultdict(list)
    for r in rows:
        by_shape[(r.length, r.kind)].append(r)
    results: dict[tuple[int, int], dict] = {}
    timing: dict[str, dict] = {}
    was = model.training
    model.eval()
    done = 0
    masks = compiled_block_masks(device.type == "cuda")
    masks.__enter__()
    try:
        for L, kind in sorted(by_shape):
            group = by_shape[(L, kind)]
            # One capacity per (row length, row kind), so the compiled decoder sees one shape per
            # length and kind, and a long item's capacity (its own answer, rounded up) does not depend
            # on which short items are packed into rows of the same length; a multiple of 128 (at
            # least 128), since flex attention switches to its decoding kernel for short query
            # blocks, and that kernel rejects the 128-token mask blocks.
            # answers_sinks: plus each item's sinks that are not answer positions (the first min(n_sinks, n - m) of
            # its n inputs; the answers are the last m). answers_sinks_replay: plus its replay positions, the
            # min(window, n - m) inputs before the answers; with the sinks that is min(n - m, n_sinks + window)
            # distinct positions (the two overlap when the n - m inputs before the answers are fewer than that).
            # window = 0 in every other regime, where this is the count above; with per-item windows, each item's own
            need = max(sum(int(sets[s].n_answer[i]) + min(n - int(sets[s].n_answer[i]), n_sinks + item_window(s, i))
                           for s, i, _, n in r.members) for r in group)
            capacity = min(L, max(CAPACITY_MULTIPLE, math.ceil(need / CAPACITY_MULTIPLE) * CAPACITY_MULTIPLE))
            if decoder_at == "every":
                capacity = L  # the decoder at every item position (the padding document is never a slot)
            B = batch_rows if L <= pack_len else 1

            def describe(rows_):  # what a unit's records belong to: each row's members, their fingerprints and windows
                return [[[int(s), int(i), int(st), int(n), sets[s].fingerprint(i), int(item_window(s, i))] for s, i, st, n in r.members]
                        for r in rows_]

            parts = []  # per unit: (seconds, tokens, seconds to the first batch)
            u0 = 0  # the unit's first row in the shape group; without a store the group is one unit, the pass as it always was
            while u0 < len(group):
                unit_name = f"{L}-{kind}-r{u0}"
                unit_key = None
                if unit_store is not None:
                    unit_key = {"length": L, "kind": kind, "capacity": capacity, "batch_rows": B, "first_row": u0,
                                "decoder_at": decoder_at, "window": window, "n_sinks": n_sinks}

                    def holds_these_rows(payload, u0=u0):  # whole batches (or the shape's end) of exactly the plan's rows
                        n = len(payload["rows"])
                        return (n >= 1 and u0 + n <= len(group) and (n % B == 0 or u0 + n == len(group))
                                and payload["rows"] == describe(group[u0 : u0 + n])
                                and [[s, i] for s, i, _ in payload["records"]] == [[s, i] for r in payload["rows"] for s, i, *_ in r])

                    cached = unit_store.load(unit_name, unit_key, holds_these_rows)
                    if cached is not None:
                        for s, i, rec in cached["records"]:
                            results[(s, i)] = rec
                        parts.append(tuple(cached["timing"]))
                        unit_store.reused_seconds += cached["timing"][0]
                        unit_store.reused_peak_bytes = max(unit_store.reused_peak_bytes, cached.get("peak_bytes") or 0)
                        u0 += len(cached["rows"])
                        done += len(cached["rows"])
                        if progress:
                            progress(done, len(rows))
                        continue
                t0, n_tok, first_s = time.perf_counter(), 0, None
                a = u0
                while a < len(group):
                    chunk = group[a : a + B]
                    a += len(chunk)
                    built = [row_tensors(r, sets) for r in chunk]
                    tokens = torch.stack([x[0] for x in built])
                    targets = torch.stack([x[1] for x in built])
                    doc_ids = torch.stack([x[2] for x in built])
                    dec = torch.stack([x[3] for x in built])
                    positions = [torch.cat(x[4]) for x in built]
                    # decoder_at "every": every position of a member item (each predicts the item's next token);
                    # the padding document's id is the row's member count. "answers_sinks": the scored positions
                    # plus each member's first n_sinks positions. "answers_sinks_replay": those plus each member's
                    # replay positions (the inputs just before its scored positions, at most `window`)
                    run = None
                    if decoder_at == "every":
                        run = torch.stack([x[2] < len(r.members) for x, r in zip(built, chunk)])
                    elif decoder_at == "answers_sinks":
                        run = dec | torch.stack([sink_mask(r, n_sinks) for r in chunk])
                    elif replay:
                        run = dec | torch.stack([sink_mask(r, n_sinks) | replay_mask(r, sets, window, replay_windows) for r in chunk])
                    varlen = None
                    if fused_varlen:
                        from ced.train.varlen import to_device, varlen_for_batch

                        dec_varlen = dec if run is None else run
                        varlen = to_device(varlen_for_batch(doc_ids, dec_varlen if decoder else None, capacity if decoder else None),
                                           device)
                    with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                        if cp:
                            logp_dev = _answer_logprobs_cp(model, tokens.to(device), doc_ids.to(device), dec.to(device),
                                                           positions, varlen)
                        else:
                            h, valid = _answer_hidden(model, tokens.to(device), doc_ids.to(device), dec.to(device),
                                                      positions, capacity, T, varlen, decoder,
                                                      run=None if run is None else run.to(device))
                    if not cp:
                        flat = h[valid]
                        logp_dev = head_logprobs(flat, W)
                    tgt = torch.cat([targets[b, positions[b]] for b in range(len(chunk))])
                    tempered = {tau: tempered_target_logp(logp_dev, tgt.to(logp_dev.device), tau) for tau in taus}
                    logp = logp_dev.cpu().double()
                    del logp_dev
                    lp_t = logp.gather(1, tgt[:, None]).squeeze(1)
                    top = logp.argmax(1)
                    at = 0
                    for b, r in enumerate(chunk):
                        for s, i, _, _ in r.members:
                            es = sets[s]
                            m = int(es.n_answer[i])
                            sl = slice(at, at + m)
                            at += m
                            correct = (top[sl] == tgt[sl])
                            # per answer token: log-prob of the target and argmax-correct ("0"/"1"), so a
                            # metric can be redefined later without rescoring; first_index: the token
                            # first_correct reads (path_copy: the first choice token)
                            rec = {"nats": float(-lp_t[sl].sum()), "n_correct": int(correct.sum()),
                                   "first_correct": bool(correct[int(es.first_info[i])]),
                                   "first_index": int(es.first_info[i]),
                                   "tok_logp": [round(float(x), 6) for x in lp_t[sl]],
                                   "tok_correct": "".join("1" if c else "0" for c in correct.tolist())}
                            if taus:
                                rec["tok_logp_tau"] = {str(tau): [round(float(x), 6) for x in tempered[tau][sl]] for tau in taus}
                            if es.cand_ids is not None:
                                cl = logp[sl.start, es.cand_ids[i].long()]
                                pred = int(cl.argmax())
                                gold = es.items[i]["gold"]
                                distractors = es.items[i].get("distractors", [])
                                rec.update(cand_logp=[round(float(x), 6) for x in cl], pred=pred, mc_correct=pred == gold,
                                           greedy=bool(correct[0]), distractor=pred in distractors)
                                rec["cand_mass"] = round(float(torch.logsumexp(cl, 0).exp()), 6)  # p on the candidates
                                if distractors:  # the choice among the values assigned in the item only
                                    rec["literal_correct"] = bool(all(cl[gold] > cl[j] for j in distractors))
                                    assigned = cl[[gold, *distractors]]
                                    rec["binding_logp"] = round(float(cl[gold] - torch.logsumexp(assigned, 0)), 6)
                            results[(s, i)] = rec
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    if first_s is None:
                        first_s = time.perf_counter() - t0
                    n_tok += L * len(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, len(rows))
                    # the unit ends here, after a whole batch, when the next batch would pass the token bound or the
                    # time bound has gone by
                    if unit_store is not None and (n_tok + L * B > unit_tokens or time.perf_counter() - t0 >= unit_seconds):
                        break
                ugroup = group[u0:a]
                parts.append((time.perf_counter() - t0, n_tok, first_s))
                if unit_store is not None:  # saved only here, after the unit's last row: never a half unit
                    unit_store.save(unit_name, unit_key, {
                        "rows": describe(ugroup),
                        "records": [[int(s), int(i), results[(s, i)]] for r in ugroup for s, i, _, _ in r.members],
                        "timing": list(parts[-1]),
                        # the scoring process's peak of allocated CUDA memory up to this unit: a pass put together by
                        # a rerun reports the largest one (the rerun's own peak does not hold the rows it only read)
                        "peak_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None})
                u0 = a
            total = sum(x[0] for x in parts)
            n_tok = sum(x[1] for x in parts)
            first_s = parts[0][2]
            # the rate after the first batch: of each unit, since a unit scored by a rerun starts with that process's
            # warm-up (compilation), which the pass's first unit does not hold. One unit (no store): as it always was.
            after_tok = sum(x[1] - L * min(B, x[1] // L) for x in parts)
            after_s = sum(x[0] - x[2] for x in parts)
            timing[f"{L}/{kind}"] = {"length": L, "kind": kind, "rows": len(group), "tokens": n_tok, "seconds": total,
                         "first_batch_seconds": first_s, "capacity": capacity, "batch_rows": B,
                         "decoder_at": decoder_at if decoder else None,
                         # answers_sinks_replay: W, or with per-item windows (replay_windows) the largest window among
                         # this shape's items; None in every other regime
                         "replay_window": (None if not replay else window if replay_windows is None else
                                           max(item_window(s, i) for r in group for s, i, _, _ in r.members)),
                         "replay_per_item": replay_windows is not None,
                         "tokens_per_s_after_first": after_tok / max(1e-9, after_s) if len(group) > B and after_tok > 0 else None}
    finally:
        masks.__exit__(None, None, None)
        model.train(was)
    out = []
    for s, es in enumerate(sets):
        for i in range(len(es)):
            r = results[(s, i)]
            meta = es.items[i]
            out.append({"probe": es.probe, "id": meta["id"], "cell": meta["cell"], "group": meta.get("group", {}),
                        "n_tokens": es.length(i), "n_answer": int(es.n_answer[i]),
                        "answer_bytes": int(es.answer_bytes[i]), "fingerprint": es.fingerprint(i),
                        **{k: meta[k] for k in PASS_THROUGH if k in meta}, **r})
    return out, timing


# --------------------------------------------------------------------------------------------
# aggregates
# --------------------------------------------------------------------------------------------


def mean_se(values) -> tuple[float | None, float | None]:
    x = np.asarray(list(values), dtype=np.float64)
    if len(x) == 0:
        return None, None
    if len(x) == 1:
        return float(x[0]), None
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def _cell_name(cell: dict) -> str:
    return ",".join(f"{k}={cell[k]}" for k in sorted(cell))


def span_metrics(r: dict) -> dict:
    """Per-item metrics of the copying probes (copy_span, edit_old), from the per-token record.

    exact: greedy exact match. Under greedy decoding the model emits the gold span iff, at every answer position, the
      argmax given the GOLD prefix is the gold token (induction on the position: up to the first miss, the generated
      prefix is the gold prefix). Ties between equal logits go to the lower token id, as argmax does. first_err_tok: the
      index (0-based, in answer tokens) of the first argmax miss, = n_answer when there is none; first_err_frac =
      first_err_tok / n_answer (1.0: no error); err: 1.0 if there is an error. The first error of free greedy generation
      is exactly this one (same induction); later errors are not observable. exp_exact_t1 / exp_exact_t07: the
      probability that sampling at temperature tau (softmax(logits / tau), no top-p / top-k) reproduces the span: prod_t
      p_tau(gold_t | gold prefix) over the answer tokens, which is exactly the chance that free sampling of n_answer
      tokens equals the gold (the same induction); the stop token after the span is not included. At tau = 1 it is
      exp(-nats); tau -> 0 gives the greedy exact match. nats_per_token_t07: mean over tokens of -log p_0.7(gold_t | gold
      prefix). nats_first: -log p(first answer token | context): where the reply's format is decided (9.10, 10.6). The
      same on answer tokens 2..n (REST_KEYS; present when n_answer >= 2): nats_per_token_rest = mean of -log p over
      tokens 2..n, token_acc_rest = share of them that are the argmax, exact_rest = 1.0 if all of them are (greedy under
      teacher forcing: the gold first token is given, so this is not a greedy rollout when the first token is missed).
    """
    n = r["n_answer"]
    miss = r["tok_correct"].find("0")
    out = {"first_err_tok": float(n if miss < 0 else miss), "first_err_frac": (n if miss < 0 else miss) / n,
           "err": float(miss >= 0), "exp_exact_t1": math.exp(-r["nats"])}
    for tau, lp in (r.get("tok_logp_tau") or {}).items():
        key = str(tau).replace(".", "")
        out["exp_exact_t" + key] = math.exp(sum(lp))
        out["nats_per_token_t" + key] = -sum(lp) / n
    lp, ok = r["tok_logp"], r["tok_correct"]
    out["nats_first"] = -float(lp[0])
    if n >= 2:
        out["nats_per_token_rest"] = -float(sum(lp[1:])) / (n - 1)
        out["token_acc_rest"] = ok[1:].count("1") / (n - 1)
        out["exact_rest"] = float("0" not in ok[1:])
    return out


def item_metrics(r: dict) -> dict:
    m = {"nats": r["nats"], "nats_per_token": r["nats"] / r["n_answer"], "token_acc": r["n_correct"] / r["n_answer"],
         "exact": float(r["n_correct"] == r["n_answer"]), "first_acc": float(r["first_correct"]),
         "bpb": r["nats"] / (LN2 * r["answer_bytes"])}
    if "mc_correct" in r:
        # gold_logp: log p(gold digit) over the FULL vocabulary (it mixes the format, how much mass
        # goes to digits at all, with the choice); binding_logp: log p(gold) renormalised over the
        # digits assigned in the item (the binding choice alone; floor -ln C); cand_mass: p on "0".."9"
        m.update(mc_acc=float(r["mc_correct"]), greedy_acc=float(r["greedy"]), distractor_rate=float(r["distractor"]),
                 gold_logp=-r["nats"])
        if "literal_correct" in r:
            m["literal_acc"] = float(r["literal_correct"])
        for k in ("binding_logp", "cand_mass"):
            if k in r:
                m[k] = r[k]
    if r["probe"] in SPAN_PROBES:
        m.update(span_metrics(r))
    # extension probes: the item's floors next to its level (item means are the cell's floor), and, for a
    # copy item with a near-duplicate decoy, the accuracy / nats at the differing tokens only
    for k, v in (r.get("floor") or {}).items():
        m["floor_nats_per_token" if k == "nats" else "floor_" + k] = v / r["n_answer"] if k == "nats" else v
    if r.get("diff_index"):
        idx = r["diff_index"]
        m["diff_acc"] = sum(r["tok_correct"][i] == "1" for i in idx) / len(idx)
        m["diff_nats"] = -sum(r["tok_logp"][i] for i in idx) / len(idx)
    return m


def summarize(results: list[dict]) -> dict:
    """{probe: {cell: {n, metric: mean, metric_se: se, ..., groups: {group: {...}}}}}."""
    tree: dict = defaultdict(lambda: defaultdict(list))
    sub: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        c = _cell_name(r["cell"])
        tree[r["probe"]][c].append(r)
        if r.get("group"):
            sub[r["probe"]][c][_cell_name(r["group"])].append(r)

    def agg(rs):
        ms = [item_metrics(r) for r in rs]
        out = {"n": len(rs), "mean_tokens": float(np.mean([r["n_tokens"] for r in rs]))}
        keys = list(dict.fromkeys(k for m in ms for k in m))  # first-seen order; an extension metric (diff_acc) exists on part of an aggregate
        for k in keys:
            out[k], out[k + "_se"] = mean_se(m[k] for m in ms if k in m)
        return out

    summary = {}
    for probe, cells in tree.items():
        summary[probe] = {}
        for c, rs in sorted(cells.items(), key=lambda kv: _sort_key(kv[0])):
            summary[probe][c] = agg(rs)
            if sub[probe][c]:
                summary[probe][c]["groups"] = {g: agg(x) for g, x in sorted(sub[probe][c].items(), key=lambda kv: _sort_key(kv[0]))}
        summary[probe]["all"] = agg([r for rs in cells.values() for r in rs])
    return summary


def _sort_key(name: str):
    parts = []
    for kv in name.split(","):
        k, _, v = kv.partition("=")
        try:
            parts.append((k, 0, float(v), ""))
        except ValueError:
            parts.append((k, 1, 0.0, v))
    return parts


def same_item(r: dict, o: dict) -> bool:
    """Two results are of the same item: identical scored tokens (fingerprint), or, across
    tokenizers, the same item text (text_sha, recorded by sets built since 2026-10-07 r2)."""
    if r["fingerprint"] == o["fingerprint"]:
        return True
    return r.get("text_sha") is not None and r.get("text_sha") == o.get("text_sha")


def paired_difference(a: list[dict], b: list[dict],
                      metrics=("nats", "exact", "token_acc", "first_acc", "mc_acc", "literal_acc")) -> dict:
    """Per (probe, cell): mean of per-item differences a - b with its paired SE, over the items both
    scored. Items are paired by id and must be the same item (`same_item`: identical tokens, or the
    same text under another tokenizer; compare tokenizers on tokenizer-neutral metrics only: exact,
    literal_acc, mc_acc, bpb)."""
    by_b = {r["id"]: r for r in b}
    tree: dict = defaultdict(lambda: defaultdict(list))
    for r in a:
        o = by_b.get(r["id"])
        if o is None:
            continue
        assert same_item(r, o), f"{r['id']}: the two runs scored different items"
        tree[r["probe"]][_cell_name(r["cell"])].append((item_metrics(r), item_metrics(o)))
    out = {}
    for probe, cells in tree.items():
        out[probe] = {}
        for c, pairs in sorted(cells.items(), key=lambda kv: _sort_key(kv[0])):
            rec = {"n": len(pairs)}
            for k in metrics:
                if k in pairs[0][0]:
                    rec[k], rec[k + "_se"] = mean_se(x[k] - y[k] for x, y in pairs)
            out[probe][c] = rec
    return out


def load_sets(directory: str | Path, probes=None) -> list[EncodedSet]:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text()) if (directory / "manifest.json").exists() else None
    names = probes or (list(manifest["files"]) if manifest else sorted(p.stem for p in directory.glob("*.pt")))
    return [EncodedSet.load(directory / f"{n}.pt") for n in names]
