"""Token Superposition Training (TrainConfig.tst; ced/train/tst.py).

 1. The section: defaults, YAML / --set round trip; what is refused, by name.
 2. s = 1 is the base bit for bit (losses, gradient norms, LRs, batch hashes = data positions, final weights and
    optimizer state), on the base and on R1-entry; and off / s = 1 against main's tree (tests/golden/
    make_tst_off_golden.py, written on 5804693).
 3. A bag phase's loss and gradient norm equal a slow reference written from the definition (float64): s = 3
    (uniform weights) and s = 8 (1/i weights), the fused and the chunked head.
 4. The stream: across switches (staged, s up and down) the input tokens of all steps are the stream from 0 in order
    (no token read twice, none skipped); every target is the next bag's token and valid iff in the bag's document.
    Option (d): phase 0 reads only the not-excluded sources; no (document, offset) is read twice across the phases,
    each document's tokens read are a prefix of it, the documents phase 0 began come first in the later stream.
 5. The switch resets every Muon momentum buffer and every AdamW m, v, step and changes nothing else (weights, RNG,
    LR); with (a) A_log moves by ln(s' / s) exactly; the event is in the metrics and in metrics.jsonl. It makes ONE
    call on the optimizer: reset_moments() when the optimizer defines it, else today's emptying; an in-place zeroing
    through that hook is today's emptying bit for bit.
 6. Kill and resume before, at and after a switch is the uninterrupted run bit for bit (the plain arm, every option
    on at once, R1-entry), through scripts/train.py's prepare as a run runs it.
 7. Options against references: (a) the decay per token is unchanged; (b) the token-scale capped read equals the
    definition on the reference, sdpa and flex backends; (e) the multi-hot aux loss equals the definition, its weight
    falls linearly to 0 at the decay start, and from there the step is the base's path.
 8. NVFP4 emulation (all_linears, keep_last_fraction) and FP8 composed with TST: s = 1 is the switch-off run bit for
    bit, and a run with a switch resumes bit for bit.
 9. The multi-target head (fused, chunked) against float64 autograd on full logits; K = 1 is fused_linear_ce.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from ced.config import DataConfig, OptimConfig, TrainConfig, TSTConfig, from_dict, tiny_v2_control_config, to_dict
from ced.data.shard_format import ShardWriter
from ced.model.attention import GatedAttention, KeySet, allowed, attend_capped
from ced.model.gdn import GatedDeltaNet
from ced.model.rope import apply_rope
from ced.model.sequence import sequence_context
from ced.train import tst as T
from ced.train.chunked_loss import token_multi_ce
from ced.train.fused_head import fused_linear_ce, fused_linear_multi_ce
from ced.train.trainer import Trainer, lr_of

from helpers import set_config
from test_fast_attention import flex_cpu_backward  # noqa: F401  (a pytest fixture)
CHUNKED = {"encoder.gdn.kernel": "chunked", "encoder.gdn.chunk_size": 8}  # test_loop_published's, inlined (that module is not in this export)
from pos_arms_helpers import assert_same_state
from pos_arms_helpers import ENC, SS_ENC

# test_pos_mix_layer's MIX (the mxl-a-s1 arm in small), inlined (that module is not in this export)
MIX = {**SS_ENC, "encoder.full_attn_interval": 2, **ENC, "encoder.nope_global_layers": (3, 4)}

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden"
sys.path.insert(0, str(REPO / "scripts"))
from train import apply_override, prepare  # noqa: E402

N = 12  # steps of the tiny runs; decay from int(12 x 0.75) = 9
ENTRY = {"encoder.attn_res": "block_entry"}
QUIET = lambda *_: None  # noqa: E731
TRACE = ("loss", "grad_norm", "lr", "batch_hash", "ce_dec", "ce_ctx", "tst_s")
EXCLUDE = ["code", "agent"]


# ----------------------------------------------------------------------------------------------
# fixtures


def write_shards(path: Path, seed: int = 5, n_docs: int = 360, vocab: int = 256) -> Path:
    """Short and long documents of four sources; the "agent" ones are chat documents (assistant tokens)."""
    rng = np.random.default_rng(seed)
    w = ShardWriter(path, token_dtype="uint16")
    for i in range(n_docs):
        long = i % 5 == 0
        n = int(rng.integers(40, 120)) if long else int(rng.integers(6, 40))
        chat = not long and i % 4 == 1
        gen = np.zeros(n, dtype=np.uint8)
        if chat:
            gen[n // 3:] = 1
        source = "agent" if chat else ("web", "code", "math")[i % 3]
        w.add_arrays(rng.integers(0, vocab, n), np.ones(n, dtype=np.uint8), gen, chat=chat, source=source)
    w.close({"tokenizer_sha256": "abc"})
    return path


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    return write_shards(tmp_path_factory.mktemp("tst_data") / "A")


def cfg_for(shards: Path, schedule=(), model_over: dict | None = None, tst: dict | None = None, **over) -> TrainConfig:
    """The tiny run: the 60M base recipe in small (MIX: Scalable-Softmax, capped rotary, NoPE in the upper global layers;
    the chunked GDN), Muon with weight decay, 4 rows of 32 positions a step in micro-batches of 2, random decoder
    positions, the fused head on the selected positions, float32 under bf16 autocast."""
    model = set_config(tiny_v2_control_config(), {**MIX, **CHUNKED, **(model_over or {})})
    cfg = TrainConfig(model=model,
                      data=DataConfig(kind="shards", shard_dir=str(shards), seq_len=32, batch_size=4, chat_whole=True,
                                      span_median=6.0, span_min=2, span_max=16, decoder_positions="random", seed=1001),
                      optim=OptimConfig(name="muon", lr=0.02, weight_decay=0.165, warmup_steps=3, decay_fraction=0.25),
                      steps=N, seed=1, micro_batch_size=2, log_every=1000, ckpt_every_steps=1000, ckpt_every_minutes=1e9,
                      ckpt_keep=2, metrics_jsonl=True, head_select=True, head_impl="fused", autocast="bfloat16")
    cfg.tst.schedule = [list(e) for e in schedule]
    for k, v in (tst or {}).items():
        setattr(cfg.tst, k, v)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def trace(trainer: Trainer, n: int) -> list[dict]:
    out = []
    for _ in range(n):
        m = trainer.train_step()
        out.append({k: m.get(k) for k in TRACE})
    return out


def stream_tokens(packer, n: int) -> np.ndarray:
    return packer._stream(0, n)["tokens"]


# ----------------------------------------------------------------------------------------------
# 1. the section


def test_defaults_round_trip_and_set():
    t = TSTConfig()
    assert (t.schedule, t.gdn_decay_rescale, t.token_scale_rope, t.phase1_exclude_sources, t.aux_multi_hot, t.aux_weight) == \
        ([], False, False, [], 0, 0.1)
    assert TrainConfig().tst == t
    cfg = TrainConfig()
    apply_override(cfg, "tst.schedule=[[0,6],[1622,1]]")
    apply_override(cfg, "tst.phase1_exclude_sources=[stack-edu-commonpile,toucan]")
    apply_override(cfg, "tst.aux_multi_hot=6")
    assert cfg.tst.schedule == [[0, 6], [1622, 1]] and cfg.tst.phase1_exclude_sources == ["stack-edu-commonpile", "toucan"]
    back = from_dict(TrainConfig, yaml.safe_load(yaml.safe_dump(to_dict(cfg))))
    assert back == cfg
    old = to_dict(TrainConfig())
    del old["tst"]  # a config written before the section
    assert from_dict(TrainConfig, old).tst == TSTConfig()


@pytest.mark.parametrize("change, match", [
    ({"tst": {"gdn_decay_rescale": True}}, "tst options without tst.schedule"),
    ({"schedule": [[1, 3]]}, "starts at step 0"),
    ({"schedule": [[0, 3], [5, 3]]}, "two neighbouring phases"),
    ({"schedule": [[0, 3], [5, 1], [4, 2]]}, "starts must increase"),
    ({"schedule": [[0, 0]]}, "bag sizes s >= 1"),
    ({"schedule": [[0, 3], [N, 1]]}, "at or after the run's end"),
    ({"schedule": [[0, 3.0]]}, "two integers"),
    ({"schedule": [[0, 3]], "tst": {"gdn_decay_rescale": True}}, "act at or after a switch"),
    ({"schedule": [[0, 3], [5, 1]], "tst": {"aux_multi_hot": 1}}, "the horizon K >= 2"),
    ({"schedule": [[0, 1], [5, 3]], "tst": {"aux_multi_hot": 3}}, "next-token phase after a bag phase"),
    ({"schedule": [[0, 3], [5, 1]], "tst": {"aux_multi_hot": 3, "aux_weight": 0.0}}, "a finite weight above 0"),
    ({"schedule": [[0, 3], [10, 1]], "tst": {"aux_multi_hot": 3}}, "nothing to anneal"),
    ({"schedule": [[0, 12], [5, 1]], "tst": {"token_scale_rope": True}}, "no near key"),
    ({"schedule": [[0, 3], [5, 1]], "tst": {"token_scale_rope": True},
      "model": {"encoder.attention.rope_max_distance": None, "encoder.nope_global_layers": ()}}, "no rotary distance cap"),
    ({"schedule": [[0, 3], [5, 1]], "tst": {"phase1_exclude_sources": ["nope"]}}, "not sources of this shard set"),
    ({"schedule": [[0, 3], [5, 1]], "over": {"head_select": False, "head_impl": "chunked"}}, "head_select true"),
    ({"schedule": [[0, 3], [5, 1]], "model": {"encoder.attn_res": "block"}}, "'block_entry'"),
    ({"schedule": [[0, 3], [5, 1]], "over": {"resume_from": "x.pt"}}, "resume_from or init_from"),
    ({"schedule": [[0, 3], [5, 1]], "over": {"init_from": "x.pt"}}, "resume_from or init_from"),
    ({"schedule": [[0, 3], [5, 1]], "over": {"shape_schedule": [[0.5, 2, 32]]}}, "shape_schedule"),
])
def test_refusals(change, match, shards, tmp_path):
    cfg = cfg_for(shards, change.get("schedule", ()), change.get("model"), change.get("tst"), **change.get("over", {}))
    with pytest.raises(ValueError, match=match):
        Trainer(cfg, tmp_path / "r")


def test_another_loader_is_refused_with_bags(shards):
    """TST reads one ShardPacker's stream (its _stream, rows and _epoch): any other loader class is refused by name until
    it is built and tested with TST (review, 2026-10-15)."""
    from types import SimpleNamespace

    from ced.train.shards import ShardPacker, ShardSet

    class Other(ShardPacker):
        pass

    cfg = cfg_for(shards, [[0, 3], [5, 1]])
    data = Other(ShardSet(str(shards)), cfg.data, cfg.data.span_fraction)
    with pytest.raises(ValueError, match="reads one ShardPacker's stream; this run's loader is Other"):
        T.TSTPlan(SimpleNamespace(cfg=cfg, positions=None, data=data, capacity=None))


def test_synthetic_data_is_refused_with_bags(tmp_path):
    cfg = cfg_for(Path("unused"), [[0, 3], [5, 1]])
    cfg.data = dataclasses.replace(cfg.data, kind="synthetic")
    with pytest.raises(ValueError, match="data.kind 'shards'"):
        Trainer(cfg, tmp_path / "r")


# ----------------------------------------------------------------------------------------------
# 2. s = 1 is the base


@pytest.mark.parametrize("form", ["base", "r1e"])
def test_s1_is_the_base_bit_for_bit(form, shards, tmp_path):
    over = ENTRY if form == "r1e" else None
    # one after the other: a trainer seeds the global RNG when it is built, and its steps draw from it (NVFP4 below)
    off = Trainer(cfg_for(shards, model_over=over), tmp_path / "off")
    a = trace(off, 6)
    on = Trainer(cfg_for(shards, [[0, 1]], model_over=over), tmp_path / "on")
    b = trace(on, 6)
    assert [{k: v for k, v in x.items() if k != "tst_s"} for x in b] == [{k: v for k, v in x.items() if k != "tst_s"} for x in a]
    assert all(x["tst_s"] == 1 for x in b)
    assert_same_state(off, on)
    assert off.data_state() == on.data_state()  # the same data identity: a base checkpoint and an s = 1 one agree




# ----------------------------------------------------------------------------------------------
# 3. a bag phase's loss against the definition


def reference_bag_loss(trainer: Trainer, s: int) -> tuple[float, float]:
    """The first step's loss and gradient norm from the definition, float64, on a copy of the trainer's model: the
    stream read from the packer token by token, bags by explicit index arithmetic, the embedding mean by indexing the
    table, the full logits, log-softmax, the weights (1/s, or 1/i normalised for s >= 8), validity per target token."""
    import copy

    cfg, packer = trainer.cfg, trainer.data
    B, L = cfg.data.batch_size, cfg.data.seq_len
    x = packer._stream(0, B * L * s + s)
    tok, doc = torch.from_numpy(x["tokens"]), torch.from_numpy(x["doc"])
    inputs = torch.zeros(B, L, s, dtype=torch.long)
    targets = torch.zeros(B, L, s, dtype=torch.long)
    valid = torch.zeros(B, L, s, dtype=torch.bool)
    bag_doc = torch.zeros(B, L, dtype=torch.long)
    for b in range(B):
        for j in range(L):
            first = b * L * s + j * s  # bag row b starts at base row b s, i.e. token b s L
            inputs[b, j] = tok[first : first + s]
            bag_doc[b, j] = doc[first + s - 1]
            for i in range(s):
                targets[b, j, i] = tok[first + s + i]
                valid[b, j, i] = bool(doc[first + s + i] == doc[first + s - 1])
    if s >= 8:
        w = torch.tensor([1.0 / i for i in range(1, s + 1)], dtype=torch.float64)
        w = w / w.sum()
    else:
        w = torch.full((s,), 1.0 / s, dtype=torch.float64)
    model = copy.deepcopy(trainer.model)
    model.train()
    h0 = model.embed_norm(model.embed.weight[inputs].sum(dim=2) / s)
    ctx, _ = sequence_context(bag_doc, None)
    h, _ = model.encoder(h0, ctx, None)
    logits = torch.nn.functional.linear(model.final_norm(h), model.head_weight())
    lse = torch.logsumexp(logits, -1)
    ce = lse[..., None] - logits.gather(-1, targets)
    vw = valid.double() * w
    loss = (vw * (ce + cfg.loss.z_loss * lse[..., None] ** 2)).sum() / vw.sum()
    loss.backward()
    gn = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
    return float(loss.detach()), float(gn)


@pytest.mark.parametrize("s", [3, 8])
@pytest.mark.parametrize("head", ["fused", "chunked"])
def test_a_bag_phase_loss_is_the_definition(s, head, shards, tmp_path):
    cfg = cfg_for(shards, [[0, s], [5, 1]], dtype="float64", autocast="none", head_impl=head)
    t = Trainer(cfg, tmp_path / "r")
    ref_loss, ref_gn = reference_bag_loss(t, s)
    shares, inner = [], t._micro_loss_tst  # the micro-batches' float64 losses (the step's "loss" metric sums in float32)

    def keep(*args, **kwargs):
        loss, sums = inner(*args, **kwargs)
        shares.append(loss.detach())
        return loss, sums

    t._micro_loss_tst = keep
    m = t.train_step()
    assert m["tst_s"] == s and m["tokens"] == 4 * 32 * s and len(shares) == 2
    assert float(sum(shares)) == pytest.approx(ref_loss, rel=1e-12, abs=1e-12)
    assert m["loss"] == pytest.approx(ref_loss, rel=1e-6)
    assert m["grad_norm"] == pytest.approx(ref_gn, rel=1e-10)
    assert T.bag_weights(s) == ([1 / s] * s if s < 8 else pytest.approx([(1 / i) / sum(1 / k for k in range(1, 9)) for i in range(1, 9)]))
    # the metrics: the weighted bag CE is the loss without the z-term (to the z-term's size)
    assert m["ce_all"] == pytest.approx(ref_loss, rel=1e-3)


# ----------------------------------------------------------------------------------------------
# 4. the stream


def check_batch_against_stream(batch, s: int, stream: dict, start_row: int, L: int):
    """Every input, target and validity of a step's batch from the stream itself (tokens and documents)."""
    tok, doc = stream["tokens"], stream["doc"]
    B = batch.tokens.shape[0]
    for b in range(B):
        base = (start_row + b * s) * L
        if s == 1:
            assert np.array_equal(batch.tokens[b].numpy(), tok[base : base + L])
            assert np.array_equal(batch.targets[b].numpy(), tok[base + 1 : base + L + 1])
            assert np.array_equal(batch.target_valid[b].numpy(), doc[base + 1 : base + L + 1] == doc[base : base + L])
            continue
        assert np.array_equal(batch.tokens[b].reshape(-1).numpy(), tok[base : base + L * s])
        assert np.array_equal(batch.multi_targets[b].reshape(-1).numpy(), tok[base + s : base + L * s + s])
        last = doc[base + s - 1 : base + L * s : s]
        assert np.array_equal(batch.doc_ids[b].numpy(), last)
        assert np.array_equal(batch.multi_valid[b].numpy(), doc[base + s : base + L * s + s].reshape(L, s) == last[:, None])
        assert torch.equal(batch.target_valid[b], batch.multi_valid[b][:, 0])  # the valid targets are a prefix of a bag
        mv = batch.multi_valid[b].numpy()
        assert all(not mv[j, i] or mv[j, i - 1] for j in range(L) for i in range(1, s))


def test_the_stream_across_switches_no_repeat_no_skip(shards, tmp_path):
    sched = [[0, 3], [2, 1], [4, 2], [6, 1], [8, 8], [9, 1]]
    t = Trainer(cfg_for(shards, sched), tmp_path / "r")
    B, L = 4, 32
    total = sum(B * s * L for s in (t.tst.sched.s_at(k) for k in range(N)))
    stream = t.data._stream(0, total + 8)
    read, row = [], 0
    for k in range(N):
        s = t.tst.sched.s_at(k)
        batch = t.batch_at(k)
        assert t.tst.first_row(k) == row
        check_batch_against_stream(batch, s, stream, row, L)
        read.append(batch.tokens.reshape(-1).numpy())
        row += B * s
    assert np.array_equal(np.concatenate(read), stream["tokens"][:total])  # inputs: the stream in order, once each
    # a next-token phase reads the base's own rows (ShardPacker.rows) from where the bags stopped
    k = 6
    assert torch.equal(t.batch_at(k).tokens, t.data.rows(t.tst.first_row(k), B).tokens)


def stream_doc_offsets(packer, start: int, n: int) -> list[tuple[int, int]]:
    """(document, offset in it) of the stream positions start .. start + n - 1 (from the packer's layout)."""
    lay = packer._epoch(0)
    cum = lay["cum"]
    out = []
    for p in range(start, start + n):
        i = int(np.searchsorted(cum, p, side="right") - 1)
        out.append((int(lay["doc"][i]), int(lay["offset"][i] + p - cum[i])))
    return out


def test_phase1_exclude_sources_reads_every_document_once(shards, tmp_path):
    t = Trainer(cfg_for(shards, [[0, 3], [4, 1]], tst={"phase1_exclude_sources": EXCLUDE}), tmp_path / "r")
    sp = t.tst.split
    names = sp.first.shards.meta["sources"]
    excl = {names.index(n) for n in EXCLUDE}
    src = t.data.shards.doc_source
    B, L = 4, 32
    t1, t2 = 4 * B * 3 * L, (N - 4) * B * L
    # what the batches hold is the two streams in order
    for k in range(N):
        s = t.tst.sched.s_at(k)
        stream = (sp.first if k < 4 else sp.rest)._stream(0, (t1 if k < 4 else t2) + 8)
        check_batch_against_stream(t.batch_at(k), s, stream, t.tst.first_row(k), L)
    one = stream_doc_offsets(sp.first, 0, t1)
    two = stream_doc_offsets(sp.rest, 0, t2)
    assert all(int(src[d]) not in excl for d, _ in one)  # phase 0: the not-excluded sources only
    pairs = one + two
    assert len(set(pairs)) == len(pairs)  # no token of any document read twice as an input
    by_doc: dict[int, list[int]] = {}
    for d, o in pairs:
        by_doc.setdefault(d, []).append(o)
    for d, offs in by_doc.items():  # each document's tokens read are a prefix of it, in order
        assert offs == list(range(len(offs))), d
    # none skipped: every document whose reading the stream passed is read whole (only the last one of each phase can
    # be cut, and phase 0's is continued first in the later stream)
    lengths = np.diff(t.data.shards.doc_starts)
    cut = [d for d, offs in by_doc.items() if len(offs) != lengths[d]]
    assert two[-1][0] in cut or two[-1][1] + 1 == lengths[two[-1][0]]
    lay = sp.rest._epoch(0)
    for d in cut:  # a document read in part: every token not read lies in the later stream past what was read
        for j in np.flatnonzero(lay["doc"] == d):
            a, n, at = int(lay["offset"][j]), int(lay["length"][j]), int(lay["cum"][j])
            unread = [o for o in range(a, a + n) if o >= len(by_doc[d])]
            assert all(at + (o - a) >= t2 for o in unread), d
        assert d in lay["doc"], d
    d0 = one[-1][0]
    assert two[0] == (d0, one[-1][1] + 1) or (one[-1][1] + 1 == lengths[d0] and d0 not in cut)
    # the later stream holds both queues in the ladder mix's proportion (loose: a tiny set)
    share = float(np.mean([int(src[d]) in excl for d, _ in two]))
    whole = float(lengths[np.isin(src, list(excl))].sum() / lengths.sum())
    assert abs(share - whole) < 0.25, (share, whole)
    assert t.data_state()["tst_phase1_exclude_sources"] == EXCLUDE


def stream_token_ids(packer, start: int, end: int) -> np.ndarray:
    """The shard set's global token indices (doc_starts[doc] + offset) of the stream positions [start, end)."""
    lay = packer._epoch(0)
    cum, starts = lay["cum"], packer.shards.doc_starts
    out = [np.zeros(0, dtype=np.int64)]
    for i in range(len(lay["doc"])):
        a, b = max(start, int(cum[i])), min(end, int(cum[i + 1]))
        if b > a:
            g = int(starts[lay["doc"][i]]) + int(lay["offset"][i]) + a - int(cum[i])
            out.append(np.arange(g, g + b - a))
    return np.concatenate(out)


def test_phase1_split_reads_every_token_once_at_any_cut(shards):
    """Option (d)'s two streams at any end of phase 1, including an end exactly at the start of a chat_whole fill's tail
    (a document begun before the cut whose last piece starts at the cut): phase 1's tokens and the whole later stream
    hold every token of the shard set exactly once. Only "code" is excluded here, so that stream 1 holds chat documents
    and therefore split fills (with "agent" excluded too, as in the ladder, stream 1 has none). Found by the review
    (2026-10-15): a cut at a tail's start lost that tail."""
    from ced.train.shards import ShardSet

    data_cfg = cfg_for(shards).data
    sset = ShardSet(str(shards))
    for exclude in (["code"], ["web", "math"]):
        probe = T.split_streams(sset, data_cfg, data_cfg.span_fraction, None, exclude, 0, 0)
        lay = probe.first._epoch(0)
        tails = np.flatnonzero(lay["offset"] > 0)  # chat_whole's fill tails: the only pieces that do not start a document
        assert len(tails), "the fixture's stream 1 has no split fill: the edge is not exercised"
        n1 = int(lay["cum"][-1])
        cuts = sorted({int(lay["cum"][j]) for j in tails} | set(range(0, n1, 97)) | {n1 - 1})
        for cut in cuts:
            sp = T.split_streams(sset, data_cfg, data_cfg.span_fraction, None, exclude, cut, 0)
            ids = np.concatenate([stream_token_ids(sp.first, 0, cut),
                                  stream_token_ids(sp.rest, 0, sp.rest.n_stream_tokens)])
            count = np.bincount(ids, minlength=sset.n_tokens)
            assert count.max() == 1 and count.min() == 1, (exclude, cut, int((count == 0).sum()), int((count > 1).sum()))


def test_the_two_queue_merge_is_proportional():
    lengths = np.array([10] * 20)
    a, b = np.arange(0, 12), np.arange(12, 20)
    merged = T.proportional_merge(a, b, lengths, total_a=120, total_b=80)
    assert sorted(merged.tolist()) == list(range(20))
    taken_b = np.cumsum(np.isin(merged, b))
    # after every document the B share is within one document of 80 / 200
    assert all(abs(tb - (i + 1) * 0.4) <= 1.0 for i, tb in enumerate(taken_b))
    assert merged[: 12].tolist() != a.tolist()  # interleaved, not one queue after the other


# ----------------------------------------------------------------------------------------------
# 5. the switch


def test_the_switch_resets_moments_and_nothing_else(shards, tmp_path):
    t = Trainer(cfg_for(shards, [[0, 3], [5, 2], [8, 1]], tst={"gdn_decay_rescale": True}), tmp_path / "r")
    for _ in range(5):
        t.train_step()
    assert any(t.optimizer.state.values())
    params = {n: p.detach().clone() for n, p in t.model.named_parameters()}
    rng = torch.get_rng_state()
    lr = lr_of(t.cfg, 5)
    event = t.tst.switch(t, 5)
    # every Muon momentum buffer and every AdamW m, v and step: zero (MuonWithAdamW.reset_moments zeroes them in place
    # since the merge; that equals emptying bit for bit: test_the_switch_makes_one_call_on_the_optimizer)
    assert all(not v.any() for st in t.optimizer.state.values() for v in st.values() if torch.is_tensor(v))
    assert event["from_s"] == 3 and event["to_s"] == 2 and event["optimizer_states_reset"] > 0
    shift = math.log(2 / 3)
    assert event["gdn_decay_rescale"]["A_log_shift"] == shift
    for n, p in t.model.named_parameters():
        if n.endswith(".A_log") and ".mixer." in n:  # the GDN layers' A_log moved by ln(s' / s), exactly as float32 adds it
            assert torch.equal(p, params[n] + torch.tensor(shift, dtype=p.dtype)), n
        else:
            assert torch.equal(p, params[n]), n  # the embedding, the head and every other weight: untouched
    assert torch.equal(torch.get_rng_state(), rng) and lr_of(t.cfg, 5) == lr
    assert t.tst.switch(t, 6) is None  # not a switch step: nothing happens
    # the optimizer then starts afresh: after the switch step's update every AdamW step count is 1
    t2 = Trainer(cfg_for(shards, [[0, 3], [5, 2], [8, 1]]), tmp_path / "r2")
    for _ in range(6):
        m = t2.train_step()
    assert "tst_switch" in m and m["tst_switch"]["step"] == 5
    steps = [float(st["step"]) for st in t2.optimizer.state.values() if "step" in st]
    assert steps and all(x == 1.0 for x in steps)
    bufs = [st["momentum_buffer"] for st in t2.optimizer.state.values() if "momentum_buffer" in st]
    assert bufs


def zero_every_state_tensor(optimizer) -> int:
    """An in-place reset as the MatrixWithAdamW.reset_moments does it for its Muon and AdamW rules: every
    state tensor zeroed, the entries kept."""
    n = 0
    for st in optimizer.state.values():
        n += bool(st)
        for v in st.values():
            if torch.is_tensor(v):
                v.zero_()
    return n


def test_the_switch_makes_one_call_on_the_optimizer(shards, tmp_path):
    """The switch's one call on the optimizer (tst.reset_optimizer_moments): an optimizer that defines reset_moments()
    gets that call alone and the switch touches nothing else of it; an optimizer without one (torch.optim.AdamW) has its
    whole state emptied (test_the_switch_resets_moments_and_nothing_else). An in-place zeroing of every state tensor through the
    hook (the optimizer branch's form) trains through the switch bit for bit like today's emptying."""
    from ced.train.muon import MuonWithAdamW

    # since the merge (2026-10-16) MuonWithAdamW defines the hook (ced/train/muon.py reset_moments: every state tensor
    # zeroed in place); torch.optim.AdamW does not. The end of this test checks the real hook against emptying.
    assert hasattr(MuonWithAdamW, T.RESET_HOOK) and not hasattr(torch.optim.AdamW, T.RESET_HOOK)
    sched = [[0, 3], [5, 1]]
    t = Trainer(cfg_for(shards, sched), tmp_path / "hook")
    for _ in range(5):
        t.train_step()
    before = {i: {k: v.clone() for k, v in st.items()} for i, st in t.optimizer.state_dict()["state"].items()}
    assert before
    calls = []
    t.optimizer.reset_moments = lambda: calls.append(1) or 7
    event = t.tst.switch(t, 5)
    assert calls == [1] and event["optimizer_states_reset"] == 7
    after = t.optimizer.state_dict()["state"]
    assert sorted(after) == sorted(before)
    assert all(torch.equal(after[i][k], v) for i, st in before.items() for k, v in st.items())  # left to the hook
    assert t.tst.switch(t, 6) is None and calls == [1]  # not a switch step: no call
    # zeroing in place through the hook = emptying, bit for bit (built one after the other: a trainer seeds the RNG)
    ref = Trainer(cfg_for(shards, sched), tmp_path / "ref")
    ref.optimizer.reset_moments = None  # no callable hook: today's emptying (optimizer.state.clear())
    a = trace(ref, 8)
    z = Trainer(cfg_for(shards, sched), tmp_path / "zero")
    z.optimizer.reset_moments = lambda: zero_every_state_tensor(z.optimizer)
    b = trace(z, 8)
    m = Trainer(cfg_for(shards, sched), tmp_path / "muon")  # MuonWithAdamW's own reset_moments
    assert isinstance(m.optimizer, MuonWithAdamW)
    c = trace(m, 8)
    assert a == b == c
    assert_same_state(ref, z)
    assert_same_state(ref, m)


def test_the_switch_event_is_written(shards, tmp_path):
    cfg = cfg_for(shards, [[0, 3], [5, 1]])
    t = Trainer(cfg, tmp_path / "r")
    t.fit(until_step=7, log=QUIET)
    recs = [json.loads(x) for x in (tmp_path / "r" / "metrics.jsonl").read_text().splitlines()]
    ev = [r for r in recs if r.get("event") == T.SWITCH_EVENT]
    assert len(ev) == 1 and ev[0]["step"] == 5 and ev[0]["from_s"] == 3 and ev[0]["to_s"] == 1
    i = recs.index(ev[0])
    assert recs[i + 1].get("step") == 6 and recs[i + 1]["tst_s"] == 1  # the record of the update with index 5 follows


# ----------------------------------------------------------------------------------------------
# 6. kill and resume


VARIANTS = {
    "wave_a": ([[0, 3], [5, 1]], {}, None),
    "all_options": ([[0, 4], [3, 2], [5, 1]], {"gdn_decay_rescale": True, "token_scale_rope": True, "aux_multi_hot": 3,
                                               "phase1_exclude_sources": EXCLUDE}, None),
    "r1e": ([[0, 3], [5, 1]], {"gdn_decay_rescale": True}, ENTRY),
}


def run_through_prepare(cfg: TrainConfig, run_dir: Path, until: int | None = None) -> list[dict]:
    t = prepare(cfg, run_dir, log=QUIET)
    t.fit(until_step=until, log=QUIET)
    return [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]


def steps_of(recs: list[dict]) -> dict:
    return {r["step"]: {k: r.get(k) for k in TRACE + ("aux_lambda", "ce_multi")} for r in recs if "loss" in r}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_kill_and_resume_before_at_and_after_a_switch(variant, shards, tmp_path):
    sched, opts, model_over = VARIANTS[variant]
    cfg = cfg_for(shards, sched, model_over, opts)
    whole = steps_of(run_through_prepare(cfg, tmp_path / "whole"))
    ref = Trainer.from_checkpoint(Trainer(cfg, tmp_path / "whole").latest_checkpoint())
    assert sorted(whole) == list(range(1, N + 1))
    switch = sched[-1][0]
    for stop in (switch - 1, switch, switch + 1):
        d = tmp_path / f"stop{stop}"
        run_through_prepare(cfg, d, until=stop)
        recs = run_through_prepare(cfg, d)
        assert steps_of(recs) == whole, stop
        assert sum(r.get("event") == T.SWITCH_EVENT for r in recs) == len(sched) - 1  # each switch once
        resumed = Trainer.from_checkpoint(Trainer(cfg, d).latest_checkpoint())
        assert_same_state(resumed, ref)


# ----------------------------------------------------------------------------------------------
# 7. the options against references


def test_gdn_decay_rescale_keeps_the_decay_per_token(shards, tmp_path):
    t = Trainer(cfg_for(shards, [[0, 6], [3, 2], [5, 1]], tst={"gdn_decay_rescale": True}, dtype="float64",
                        autocast="none"), tmp_path / "r")
    layers = [m for m in t.model.modules() if isinstance(m, GatedDeltaNet)]
    x = torch.randn(2, 9, t.cfg.model.d_model, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    s_before = 6
    g0 = [m.gates(x)[0] for m in layers]
    for step, s_after in ((3, 2), (5, 1)):
        t.tst.switch(t, step)
        g1 = [m.gates(x)[0] for m in layers]
        for a, b in zip(g0, g1):
            # the decay per token: exp(g / s) for a position of s tokens, the same before and after
            torch.testing.assert_close(b / s_after, a / s_before, rtol=1e-13, atol=0)
        g0, s_before = g1, s_after


def reference_token_scale(layer: GatedAttention, x: torch.Tensor, docs: torch.Tensor, s: int, cap: int) -> torch.Tensor:
    """The capped read under token-scale rotary from its definition: positions of s tokens; a key d positions back is
    near while s d <= cap, the logit then R(s pos_q) q . R(s pos_k) k; else R(cap) q . k; the temperature at the
    positions themselves; one softmax over the query's document at and before it."""
    ctx, _ = sequence_context(docs, None)
    B, L, _ = x.shape
    c = layer.cfg
    qn, gate = layer._normed_queries(x)
    kn = layer.k_norm(layer.k_proj(x).view(B, L, c.n_kv_heads, c.head_dim))
    v = layer.v_proj(x).view(B, L, c.n_kv_heads, c.head_dim)
    f = layer.query_factor(ctx.pos, x.dtype)
    q = apply_rope(qn, ctx.pos * s, c.rope_fraction, c.rope_theta)
    q_far = apply_rope(qn, ctx.pos.new_full((1, 1), cap), c.rope_fraction, c.rope_theta)
    if f is not None:
        q, q_far = q * f, q_far * f
    k = apply_rope(kn, ctx.pos * s, c.rope_fraction, c.rope_theta)
    keys = KeySet(k=k, v=v, idx=ctx.idx, doc=ctx.doc, ok=torch.ones_like(docs, dtype=torch.bool), k_far=kn)
    far = s * (ctx.idx[:, :, None] - ctx.idx[:, None, :]) > cap
    return layer.output(attend_capped(q, q_far, keys, allowed(keys, ctx.idx, ctx.doc), far), gate)


@pytest.mark.parametrize("backend", ["reference", "sdpa", "flex"])
def test_token_scale_rope_is_the_definition(backend, flex_cpu_backward):  # noqa: F811
    cfg = set_config(tiny_v2_control_config(), {**MIX, "encoder.attention.backend": backend})
    acfg = cfg.encoder.attention
    cap = acfg.rope_max_distance
    torch.manual_seed(0)
    layer = GatedAttention(cfg.d_model, acfg, cfg.norm_eps).double()
    with torch.no_grad():
        for p in layer.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x = torch.randn(2, 40, cfg.d_model, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    docs = torch.tensor([[0] * 30 + [1] * 10, [0] * 40])
    for s in (2, 3):
        ref = reference_token_scale(layer, x, docs, s, cap)
        ctx, _ = sequence_context(docs, None)
        ctx.rope_scale = s
        out, _ = layer(x, ctx)
        torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)
        assert (s * 39 > cap) and cap // s < 39  # the rows hold near and far pairs
    # s = 1 is the plain capped read
    ctx, _ = sequence_context(docs, None)
    torch.testing.assert_close(layer(x, ctx)[0], reference_token_scale(layer, x, docs, 1, cap), rtol=1e-12, atol=1e-12)


def test_token_scale_rope_through_the_model_agrees_across_backends(flex_cpu_backward):  # noqa: F811
    """The encoder's shared tables and masks (Encoder.share_capped_rotation / share_capped_masks) under rope_scale: the
    model's logits on the flex and sdpa backends equal the reference backend's."""
    from ced.model.control import build_model

    outs = {}
    tokens = torch.randint(0, 256, (2, 40, 3), generator=torch.Generator().manual_seed(2))
    docs = torch.tensor([[0] * 25 + [1] * 15, [0] * 40])
    for backend in ("reference", "sdpa", "flex"):
        cfg = set_config(tiny_v2_control_config(), {**MIX, **CHUNKED, "encoder.attention.backend": backend})
        torch.manual_seed(4)
        model = build_model(cfg).double()
        outs[backend] = model(tokens, docs, rope_scale=3).logits
    torch.testing.assert_close(outs["sdpa"], outs["reference"], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(outs["flex"], outs["reference"], rtol=1e-10, atol=1e-10)
    plain = model(tokens, docs).logits
    assert not torch.allclose(plain, outs["flex"])  # the scale changes the function


def test_aux_multi_hot_is_the_definition(shards, tmp_path):
    import copy

    cfg = cfg_for(shards, [[0, 3], [3, 1]], tst={"aux_multi_hot": 3}, dtype="float64", autocast="none")
    t = Trainer(cfg, tmp_path / "r")
    assert t.tst.aux_from == 3 and t.tst.aux_until == 9
    assert [t.tst.aux_lambda(k) for k in range(N)] == [0.0] * 3 + [0.1 * (9 - k) / 6 for k in range(3, 9)] + [0.0] * 3
    for _ in range(4):
        t.train_step()
    batch = t.batch_at(4)
    lam = 0.1 * 5 / 6
    model = copy.deepcopy(t.model)
    out = model(batch.tokens, batch.doc_ids, return_hidden=True)
    logits = torch.nn.functional.linear(out.hidden_aux, model.head_weight())
    lse = torch.logsumexp(logits, -1)
    ce = lse[..., None] - logits.gather(-1, batch.multi_targets)
    v = batch.multi_valid.double()
    n_valid = v[..., 0].sum()
    ref = ((ce[..., 0] + cfg.loss.z_loss * lse ** 2) * v[..., 0]).sum() / n_valid \
        + lam * (ce[..., 1:] * v[..., 1:]).sum() / (2 * n_valid)
    # the batch: the next two tokens of the stream after each target, valid iff in the position's document
    stream = t.data._stream(0, 3 * 4 * 3 * 32 + 4 * 32 * 2 + 3)
    r0 = t.tst.first_row(4)
    for b in range(4):
        base = (r0 + b) * 32
        for k in range(3):
            assert np.array_equal(batch.multi_targets[b, :, k].numpy(), stream["tokens"][base + 1 + k : base + 33 + k])
            assert np.array_equal(batch.multi_valid[b, :, k].numpy(),
                                  stream["doc"][base + 1 + k : base + 33 + k] == stream["doc"][base : base + 32])
    shares, inner = [], t._micro_loss_tst  # the micro-batches' float64 losses (the "loss" metric sums in float32)

    def keep(*args, **kwargs):
        loss, sums = inner(*args, **kwargs)
        shares.append(loss.detach())
        return loss, sums

    t._micro_loss_tst = keep
    m = t.train_step()
    assert m["aux_lambda"] == pytest.approx(lam, rel=1e-15) and len(shares) == 2
    assert float(sum(shares)) == pytest.approx(float(ref.detach()), rel=1e-12)
    assert m["loss"] == pytest.approx(float(ref.detach()), rel=1e-6)
    # from the decay start on the batch has one target a position: the base's path
    assert t.batch_at(9).multi_targets is None and t.batch_at(8).multi_targets is not None


# ----------------------------------------------------------------------------------------------
# 8. NVFP4 and FP8 with TST


QUANT = {"nvfp4": {"nvfp4.mode": "all_linears", "nvfp4.keep_last_fraction": 0.15},
         "fp8": {"fp8.mode": "linears"}}


@pytest.mark.parametrize("quant", sorted(QUANT))
def test_quantised_training_with_tst(quant, shards, tmp_path):
    q = QUANT[quant]
    off = Trainer(cfg_for(shards, model_over=q), tmp_path / "off")  # one after the other: NVFP4's stochastic rounding
    a = trace(off, 4)  # draws from the global RNG, which a trainer seeds when it is built
    on = Trainer(cfg_for(shards, [[0, 1]], model_over=q), tmp_path / "on")
    b = trace(on, 4)
    assert [{k: v for k, v in x.items() if k != "tst_s"} for x in a] == [{k: v for k, v in x.items() if k != "tst_s"} for x in b]
    assert_same_state(off, on)
    # a run with a switch: kill and resume at the switch is bit for bit
    cfg = cfg_for(shards, [[0, 3], [5, 1]], q, {"gdn_decay_rescale": True})
    whole = steps_of(run_through_prepare(cfg, tmp_path / "whole", until=8))
    run_through_prepare(cfg, tmp_path / "stop", until=5)
    assert steps_of(run_through_prepare(cfg, tmp_path / "stop", until=8)) == whole
    assert all(math.isfinite(v["loss"]) for v in whole.values())


# ----------------------------------------------------------------------------------------------
# 9. the multi-target head


def test_the_multi_target_head_against_float64_autograd():
    g = torch.Generator().manual_seed(0)
    n, d, V, K = 37, 16, 50, 4
    h = torch.randn(n, d, dtype=torch.float64, generator=g)
    W = torch.randn(V, d, dtype=torch.float64, generator=g) * 0.3
    t = torch.randint(0, V, (n, K), generator=g)
    t[3, 2] = t[3, 0]  # a token twice in a bag
    a = torch.rand(n, K, dtype=torch.float64, generator=g) * (torch.rand(n, K, generator=g) > 0.3)
    b = torch.rand(n, dtype=torch.float64, generator=g) * 1e-3
    hr, Wr = h.clone().requires_grad_(), W.clone().requires_grad_()
    logits = hr @ Wr.T
    lse = torch.logsumexp(logits, -1)
    ce = lse[:, None] - logits.gather(1, t)
    ref = (a * ce).sum() + (b * lse ** 2).sum()
    dh, dW = torch.autograd.grad(ref, (hr, Wr))
    for chunk in (8, 64):
        hf, Wf = h.clone().requires_grad_(), W.clone().requires_grad_()
        loss, ce_f, lse_f = fused_linear_multi_ce(hf, Wf, t, a, b, chunk=chunk)
        loss.backward()
        torch.testing.assert_close(loss, ref.detach(), rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(ce_f, ce.detach(), rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(lse_f, lse.detach(), rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(hf.grad, dh, rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(Wf.grad, dW, rtol=1e-12, atol=1e-13)
    hc, Wc = h.clone().requires_grad_(), W.clone().requires_grad_()
    ce_c, lse_c = token_multi_ce(hc, Wc, t, chunk=8)
    ((a * ce_c).sum() + (b * lse_c ** 2).sum()).backward()
    torch.testing.assert_close(hc.grad, dh, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(Wc.grad, dW, rtol=1e-12, atol=1e-13)
    # one target a row: fused_linear_ce's numbers
    one = fused_linear_ce(h, W, t[:, 0], a[:, 0], b)
    multi = fused_linear_multi_ce(h, W, t[:, :1], a[:, :1], b)
    torch.testing.assert_close(multi[0], one[0], rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(multi[1][:, 0], one[1], rtol=1e-14, atol=1e-14)
