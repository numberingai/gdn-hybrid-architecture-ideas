"""The fast attention path (ced/model/fast_attention.py) against the reference `attend`.

The GPU ladder sweep will run attention through attend_fast, so it has to compute exactly what the
reference computes, for the two key sets the model builds:

- encoder global layer: the layer's own keys (plus cached keys of earlier chunks), causal and
  same-document;
- decoder layer: [memory (ok = context positions); own_past; own_now] plus the self-mask that lets
  every slot, including empty ones (doc = -1, ok = False), attend to itself;
- slot reads (attention.attend_slots, empty_reads_first): slot queries over keys they did not write,
  the v2 decoder's memory and the second pass's pass-1 K/V; an empty slot reads key 0 only.

1. The flex mask_mod and the sdpa dense mask equal the reference mask entry for entry, and the
   BlockMask's tiles cover every allowed entry.
2. Forward outputs and q/k/v gradients equal the reference (float32 1e-5, float64 1e-12), with
   packed documents, empty slots, a cached prefix, and GQA. The GQA head mapping is checked
   directly.
3. Compiled flex (the block-sparse kernel; CPU: forward, float32 only) matches too, and it really
   uses the tiles: a wrong BlockMask with the right mask_mod changes its output. Eager flex is a
   dense fallback that ignores the tiles, so only the compiled test checks them.
4. Plugged into the full models by monkeypatching ced.model.attention.attend and
   ced.model.decoder.attend: logits and every parameter gradient match, including cached
   generation (prefill + extend) against the teacher-forced reference.
5. No data-dependent shapes: create_block_mask, flex (forward and backward) and sdpa record no
   nonzero / masked_select / boolean indexing, and the op trace does not depend on the masks.

CPU limits (torch 2.14.1), see the module docstring: PyTorch refuses flex backward on CPU at its
front end. The higher-order op has a dense math backward; the `flex_cpu_backward` fixture removes
that one device check so gradient plumbing (mask, GQA, scale) is checked here. The Triton forward
and backward kernels only exist on GPU: every device-parametrised test also runs on CUDA when a GPU
is present, where flex is compiled by default and no check is bypassed.
"""

from __future__ import annotations

import importlib

import pytest
import torch
from torch._ops import HigherOrderOperator
from torch.nn.attention.flex_attention import create_block_mask, create_mask

import ced.model.attention as attention_module
import ced.model.decoder as decoder_module
from ced.model.attention import KeySet, allowed, attend, slots_allowed
from ced.model.capacity import gather_slots, select_slots
from ced.model.fast_attention import attend_fast, dense_mask, make_block_mask, mask_mod_for
from ced.train.loss import lm_loss

from helpers import LOOPED_VARIANTS, make_model, random_rows
from test_capacity import OpRecorder

flex_module = importlib.import_module("torch.nn.attention.flex_attention")
flex_hop_module = importlib.import_module("torch._higher_order_ops.flex_attention")

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
BACKENDS = ["flex", "sdpa"]
TOL = {torch.float32: dict(rtol=1e-5, atol=1e-5), torch.float64: dict(rtol=1e-12, atol=1e-12)}


@pytest.fixture
def flex_cpu_backward(monkeypatch):
    """Let flex run backward on CPU through its dense math implementation (see module docstring).
    Other devices keep PyTorch's own check."""
    original = flex_module._validate_device

    def validate(query, key, value):
        if query.device.type != "cpu":
            original(query, key, value)

    monkeypatch.setattr(flex_module, "_validate_device", validate)


@pytest.fixture(autouse=True)
def no_silent_eager_fallback(monkeypatch):
    """Compiled flex specialises on shapes and dtypes. Past dynamo's recompile limit it would
    silently run flex eagerly (the dense fallback), and a compiled test would no longer test the
    kernel. Raise the limit for the many shapes in this file and make hitting it an error."""
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 64)
    monkeypatch.setattr(torch._dynamo.config, "fail_on_recompile_limit_hit", True)


# ------------------------------------------------------------------------------------------
# key sets as the model builds them


def _qkv(B, Lq, S, Hq, Hkv, hd, dtype, device, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, Lq, Hq, hd, generator=g, dtype=torch.float64)
    k = torch.randn(B, S, Hkv, hd, generator=g, dtype=torch.float64)
    v = torch.randn(B, S, Hkv, hd, generator=g, dtype=torch.float64)
    return q.to(device, dtype), k.to(device, dtype), v.to(device, dtype)


def encoder_case(dtype=torch.float32, device="cpu", heads=(4, 2), prefix=0, L=300, B=2, hd=16, seed=0):
    """A global layer's keys: `prefix` cached positions of earlier chunks, then this chunk's L.

    Queries are this chunk's positions. Several packed documents per row.
    """
    S = prefix + L
    _, docs, _ = random_rows(B, S, 256, seed=seed, n_docs=4)
    idx = torch.arange(S).expand(B, S)
    q, k, v = _qkv(B, L, S, *heads, hd, dtype, device, seed)
    keys = KeySet(k=k, v=v, idx=idx.to(device), doc=docs.to(device), ok=torch.ones(B, S, dtype=torch.bool, device=device))
    return q, keys, keys.idx[:, prefix:], keys.doc[:, prefix:], None


def decoder_case(dtype=torch.float32, device="cpu", heads=(4, 2), past=0, L=300, C=96, n_dec=80, B=2, hd=16, seed=0):
    """A decoder layer's keys for a row of L positions whose last chunk starts at `past`.

    memory:   every position of the row, ok = context position (not kept as a decoder slot);
    own_past: the earlier chunk's decoder slots (capacity = chunk length, as in prefill), if past > 0;
    own_now:  this chunk's C slots; rows with fewer decoder positions get empty slots.
    Queries are own_now. self_block_offset = L + len(own_past).
    """
    _, docs, dec = random_rows(B, L, 256, seed=seed, n_dec=n_dec, n_docs=4)
    positions = torch.arange(L).expand(B, L)

    def slots(a, b, capacity):
        s = select_slots(dec[:, a:b], capacity)
        meta = (gather_slots(positions[:, a:b], s.index, fill=-1), gather_slots(docs[:, a:b], s.index, fill=-1), s.valid)
        return s.kept, meta

    kept_now, now = slots(past, L, C)
    kept_past, prev = slots(0, past, past) if past > 0 else (dec[:, :0], None)
    is_context = ~torch.cat([kept_past, kept_now], dim=1)
    meta = [(positions, docs, is_context)] + ([prev] if prev else []) + [now]
    idx, doc, ok = (torch.cat(parts, dim=1) for parts in zip(*meta))
    S = idx.shape[1]
    q, k, v = _qkv(B, C, S, *heads, hd, dtype, device, seed)
    keys = KeySet(k=k, v=v, idx=idx.to(device), doc=doc.to(device), ok=ok.to(device))
    assert not bool(now[2].all()), "the case must contain empty slots"
    return q, keys, now[0].to(device), now[1].to(device), S - C


SLOT_READS = "slot_reads"  # a case's mask rule in place of a self-block offset (see slot_read_case)


def slot_read_case(dtype=torch.float32, device="cpu", heads=(4, 2), L=700, C=200, n_dec=180, B=2, hd=16, seed=0):
    """Slot reads (attention.attend_slots with empty_reads_first): C slot queries over a key set
    they did not write, as the v2 decoder reads the memory and the second pass reads a layer's pass-1
    K/V. Keys: every one of the row's L positions (all ok). Queries: the row's first C decoder
    positions (keep-first-C), then empty slots (index and document -1), which read key 0 only.
    Q_LEN (C) != KV_LEN (L), neither a multiple of the 128 tile."""
    _, docs, dec = random_rows(B, L, 256, seed=seed, n_dec=n_dec, n_docs=4)
    positions = torch.arange(L).expand(B, L)
    slots = select_slots(dec, C)
    q_idx = gather_slots(positions, slots.index, fill=-1)
    q_doc = gather_slots(docs, slots.index, fill=-1)
    q, k, v = _qkv(B, C, L, *heads, hd, dtype, device, seed)
    keys = KeySet(k=k, v=v, idx=positions.to(device), doc=docs.to(device), ok=torch.ones(B, L, dtype=torch.bool, device=device))
    assert bool(slots.valid.any(1).all()) and not bool(slots.valid.all()), "rows must be partly empty"
    return q, keys, q_idx.to(device), q_doc.to(device), SLOT_READS


CASES = {
    "encoder": lambda **kw: encoder_case(**kw),
    "encoder_cached_prefix": lambda **kw: encoder_case(prefix=137, **kw),
    "encoder_mha": lambda **kw: encoder_case(heads=(4, 4), **kw),
    "decoder_training": lambda **kw: decoder_case(**kw),
    "decoder_cached": lambda **kw: decoder_case(past=110, C=120, n_dec=90, **kw),
    "decoder_gqa3": lambda **kw: decoder_case(heads=(6, 2), **kw),
    "slot_reads": lambda **kw: slot_read_case(**kw),
}


def mask_kwargs(rule) -> dict:
    """attend_fast / mask_mod_for / dense_mask / make_block_mask keyword arguments for a case's mask
    rule: None (encoder), an int (the decoder's self-block offset) or SLOT_READS."""
    if rule == SLOT_READS:
        return {"empty_reads_first": True}
    return {"self_block_offset": rule}


def reference_allow(keys, q_idx, q_doc, self_block_offset):
    """The mask exactly as the reference model builds it (attention.py; decoder.py's eye block;
    attention.slots_allowed for slot reads)."""
    if self_block_offset == SLOT_READS:
        return slots_allowed(keys, q_idx, q_doc)
    allow = allowed(keys, q_idx, q_doc)
    if self_block_offset is not None:
        B, C, S = allow.shape
        assert self_block_offset == S - C  # in the model own_now is always the last block
        self_mask = torch.eye(C, dtype=torch.bool, device=allow.device).expand(B, C, C)
        allow = torch.cat([allow[:, :, : S - C], allow[:, :, S - C :] | self_mask], dim=2)
    return allow


# ------------------------------------------------------------------------------------------
# 1. masks


@pytest.mark.parametrize("case", CASES)
def test_masks_equal_the_reference_mask(case):
    q, keys, q_idx, q_doc, offset = CASES[case]()
    expected = reference_allow(keys, q_idx, q_doc, offset)
    B, Lq, S = expected.shape

    flex_mask = create_mask(mask_mod_for(keys, q_idx, q_doc, **mask_kwargs(offset)), B, None, Lq, S, device="cpu")
    assert torch.equal(flex_mask[:, 0], expected)
    assert torch.equal(dense_mask(keys, q_idx, q_doc, **mask_kwargs(offset)), expected)

    block_mask = make_block_mask(keys, q_idx, q_doc, **mask_kwargs(offset))
    tiles = block_mask.to_dense()[:, 0]  # (B, n_q_tiles, n_kv_tiles), partial or full
    size_q, size_kv = block_mask.BLOCK_SIZE
    covered = tiles.repeat_interleave(size_q, dim=1).repeat_interleave(size_kv, dim=2)[:, :Lq, :S]
    assert not bool((expected & ~covered).any()), "an allowed entry lies in a skipped tile"
    assert bool((~covered).any()), "no tile is skipped, so the coverage check is vacuous"


# ------------------------------------------------------------------------------------------
# 2. forward and gradients


def _compiled_for(device, dtype):
    """Compiled flex on CUDA (the real kernel); eager on CPU and for float64."""
    return device == "cuda" and dtype != torch.float64


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES)
def test_forward_matches_reference(case, backend, dtype, device):
    q, keys, q_idx, q_doc, offset = CASES[case](dtype=dtype, device=device)
    expected = attend(q, keys, reference_allow(keys, q_idx, q_doc, offset))
    out = attend_fast(q, keys, q_idx, q_doc, backend=backend, compiled=_compiled_for(device, dtype), **mask_kwargs(offset))
    assert out.shape == expected.shape and out.dtype == expected.dtype
    torch.testing.assert_close(out, expected, **TOL[dtype])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES)
def test_gradients_match_reference(case, backend, dtype, device, flex_cpu_backward):
    q, keys, q_idx, q_doc, offset = CASES[case](dtype=dtype, device=device)
    g = torch.Generator().manual_seed(1)
    weight = torch.randn(q.shape, generator=g, dtype=torch.float64).to(device, dtype)

    def grads(fn):
        qq, kk, vv = (t.detach().clone().requires_grad_() for t in (q, keys.k, keys.v))
        key_set = KeySet(k=kk, v=vv, idx=keys.idx, doc=keys.doc, ok=keys.ok)
        (fn(qq, key_set) * weight).sum().backward()
        return qq.grad, kk.grad, vv.grad

    expected = grads(lambda qq, ks: attend(qq, ks, reference_allow(ks, q_idx, q_doc, offset)))
    got = grads(
        lambda qq, ks: attend_fast(qq, ks, q_idx, q_doc, backend=backend, compiled=_compiled_for(device, dtype),
                                   **mask_kwargs(offset))
    )
    for name, a, b in zip("qkv", got, expected):
        torch.testing.assert_close(a, b, **TOL[dtype], msg=lambda m: f"d{name}: {m}")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("heads", [(6, 2), (8, 2), (4, 4)])
def test_gqa_head_mapping(backend, heads):
    """kv head j must serve query heads j*group .. (j+1)*group - 1 (repeat_interleave), not j, j+Hkv, ..."""
    hq, hkv = heads
    q, keys, q_idx, q_doc, offset = encoder_case(dtype=torch.float64, heads=heads, L=40)
    v = torch.arange(hkv, dtype=torch.float64).view(1, 1, hkv, 1).expand_as(keys.v).contiguous()
    keys = KeySet(k=keys.k, v=v, idx=keys.idx, doc=keys.doc, ok=keys.ok)
    out = attend_fast(q, keys, q_idx, q_doc, offset, backend=backend)
    # softmax weights sum to one, so query head h returns exactly the value of the kv head it reads
    expected_head = torch.arange(hq, dtype=torch.float64) // (hq // hkv)
    torch.testing.assert_close(out, expected_head.view(1, 1, hq, 1).expand_as(out), rtol=0, atol=1e-12)
    torch.testing.assert_close(out, attend(q, keys, allowed(keys, q_idx, q_doc)), rtol=0, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", ["encoder", "decoder_cached"])
def test_bfloat16_within_bfloat16_rounding(case, backend):
    """bfloat16 is not bit-comparable: the reference upcasts to float32 and rounds once at the end.
    Both paths must stay within bfloat16 rounding of the float64 result, like the reference does."""
    q, keys, q_idx, q_doc, offset = CASES[case](dtype=torch.float64)
    exact = attend(q, keys, reference_allow(keys, q_idx, q_doc, offset))
    to_bf16 = lambda t: t.to(torch.bfloat16)
    keys16 = KeySet(k=to_bf16(keys.k), v=to_bf16(keys.v), idx=keys.idx, doc=keys.doc, ok=keys.ok)
    reference = attend(to_bf16(q), keys16, reference_allow(keys16, q_idx, q_doc, offset))
    fast = attend_fast(to_bf16(q), keys16, q_idx, q_doc, backend=backend, **mask_kwargs(offset))
    assert fast.dtype == torch.bfloat16
    ref_err = (reference.double() - exact).abs().max().item()
    fast_err = (fast.double() - exact).abs().max().item()
    assert fast_err <= max(2 * ref_err, 1e-2), (fast_err, ref_err)


# ------------------------------------------------------------------------------------------
# 3. compiled flex (block-sparse kernel)


@pytest.mark.parametrize("case", ["encoder_cached_prefix", "decoder_cached", "slot_reads"])
def test_compiled_flex_forward_matches_and_uses_the_tiles(case):
    q, keys, q_idx, q_doc, offset = CASES[case](dtype=torch.float32)
    expected = attend(q, keys, reference_allow(keys, q_idx, q_doc, offset))
    with torch.no_grad():
        out = attend_fast(q, keys, q_idx, q_doc, backend="flex", compiled=True, **mask_kwargs(offset))
        torch.testing.assert_close(out, expected, **TOL[torch.float32])

        # Same mask_mod, but tiles of a diagonal-only mask: the kernel must skip the tiles that the
        # real mask needs, so the output changes. (Eager flex ignores tiles and would not notice.)
        B, Lq = q_idx.shape
        shift = offset if isinstance(offset, int) else len(keys) - Lq
        wrong = create_block_mask(lambda b, h, i, j: j == i + shift, B, None, Lq, len(keys), device="cpu")
        wrong.mask_mod = mask_mod_for(keys, q_idx, q_doc, **mask_kwargs(offset))
        out_wrong = attend_fast(q, keys, q_idx, q_doc, backend="flex", compiled=True, block_mask=wrong, **mask_kwargs(offset))
        assert not torch.allclose(out_wrong, expected, rtol=1e-3, atol=1e-3)


# ------------------------------------------------------------------------------------------
# 4. inside the full models


def use_fast_attention(monkeypatch, backend, compiled=False):
    """Route the models' attention through attend_fast without editing them.

    The models call attend(q, keys, allow) with a materialised mask. The query metadata is the
    last Lq entries of the key set at both call sites (a chunk's own keys in the encoder, own_now
    in the decoder), so the wrappers take it from there, and they check that the mask attend_fast
    implies is exactly the `allow` the reference built before ignoring `allow`.
    """
    calls = {"encoder": 0, "decoder": 0}

    def wrapper(kind):
        def fast(q, keys, allow):
            start = len(keys) - q.shape[1]
            q_idx, q_doc = keys.idx[:, start:], keys.doc[:, start:]
            offset = start if kind == "decoder" else None
            assert torch.equal(dense_mask(keys, q_idx, q_doc, offset), allow)
            calls[kind] += 1
            return attend_fast(q, keys, q_idx, q_doc, offset, backend=backend, compiled=compiled)

        return fast

    monkeypatch.setattr(attention_module, "attend", wrapper("encoder"))
    monkeypatch.setattr(decoder_module, "attend", wrapper("decoder"))
    return calls


def _loss_and_grads(model, tokens, docs, dec, T):
    model.zero_grad(set_to_none=True)
    out = model(tokens[:, :-1], docs[:, :-1], dec[:, :-1], T=T)
    loss, _ = lm_loss(out.logits, tokens[:, 1:], docs[:, 1:] == docs[:, :-1], out.kept, ctx_weight=1.0, z_coef=1e-4)
    loss.backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    return out.logits.detach(), grads


MODEL_VARIANTS = ["default", "T1_no_injection", "memory_kv_per_layer", "dynamic_hc_and_injection", "static_hc_only"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("arch,variant", [("looped_ced", v) for v in MODEL_VARIANTS] + [("control", "default")])
def test_full_model_logits_and_gradients_match(arch, variant, backend, monkeypatch, flex_cpu_backward):
    model = make_model(arch, LOOPED_VARIANTS[variant] if arch == "looped_ced" else None)
    model.train()
    T = model.cfg.loop.t_eval
    # 12 decoder positions per row against a capacity of 16: empty slots in every row.
    tokens, docs, dec = random_rows(2, 49, model.cfg.vocab_size, seed=5, n_dec=12)
    expected_logits, expected_grads = _loss_and_grads(model, tokens, docs, dec, T)

    calls = use_fast_attention(monkeypatch, backend)
    logits, grads = _loss_and_grads(model, tokens, docs, dec, T)
    assert calls["encoder"] > 0 and (calls["decoder"] > 0) == (arch == "looped_ced")
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=1e-10)
    assert grads.keys() == expected_grads.keys()
    for name in expected_grads:
        torch.testing.assert_close(grads[name], expected_grads[name], rtol=0, atol=1e-10, msg=lambda m: f"{name}: {m}")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("variant", ["default", "T2", "dynamic_hc_only"])
def test_cached_generation_with_fast_attention_matches_reference_teacher_forcing(variant, backend, monkeypatch):
    """Prefill + random chunks + single tokens through attend_fast (cached encoder keys, memory,
    own_past) against the reference model's teacher-forced logits."""
    model = make_model("looped_ced", LOOPED_VARIANTS[variant])
    L = 48
    tokens, docs, dec = random_rows(2, L, model.cfg.vocab_size, seed=11, n_dec=model.capacity(L))
    T = model.cfg.loop.t_eval
    with torch.no_grad():
        expected = model(tokens, docs, dec, T=T).logits

    use_fast_attention(monkeypatch, backend)
    edges = [0, 17, 30, *range(31, L + 1)]
    pieces, out = [], None
    for a, b in zip(edges[:-1], edges[1:]):
        if out is None:
            out = model.prefill(tokens[:, a:b], docs[:, a:b], dec[:, a:b], T=T)
        else:
            out = model.extend(tokens[:, a:b], docs[:, a:b], dec[:, a:b], out.cache)
        pieces.append(out.logits)
    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, rtol=0, atol=1e-10)


def test_full_model_with_compiled_flex_matches_in_float32(monkeypatch):
    """The CPU block-sparse kernel inside the real model (forward only: no CPU compiled backward)."""
    model = make_model("looped_ced", {"loop.hyper_connections": "dynamic_diag", "loop.hc_streams": 2}, dtype=torch.float32)
    tokens, docs, dec = random_rows(2, 48, model.cfg.vocab_size, seed=6, n_dec=12)
    with torch.no_grad():
        expected = model(tokens, docs, dec).logits
        use_fast_attention(monkeypatch, "flex", compiled=True)
        out = model(tokens, docs, dec).logits
    torch.testing.assert_close(out, expected, **TOL[torch.float32])


# ------------------------------------------------------------------------------------------
# 5. fixed shapes


class HopRecorder(OpRecorder):
    """OpRecorder that also sees inside flex.

    flex is a higher-order op (HOP): one dispatcher call for the forward (flex_attention) and one
    for the backward (flex_attention_backward). This recorder logs the HOP and then runs PyTorch's
    eager body for it (dense math) under the recorder, so the aten ops inside are checked too. On
    GPU the body is a Triton kernel instead, with no aten ops and fixed output shapes; the only
    data-dependent quantities there are the BlockMask's per-row tile counts, which are loop bounds
    inside the kernel, not tensor shapes.
    """

    supports_higher_order_operators = True
    BODIES = {
        "flex_attention": flex_hop_module.sdpa_dense,
        "flex_attention_backward": flex_hop_module.sdpa_dense_backward,
    }

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if isinstance(func, HigherOrderOperator):
            self.ops.append((f"hop.{func.name()}", ()))
            with self:
                return self.BODIES[func.name()](*args, **(kwargs or {}))
        return super().__torch_dispatch__(func, types, args, kwargs)


@pytest.fixture
def recordable_flex(monkeypatch, flex_cpu_backward):
    """Eager flex wraps the HOP in torch.compile(backend="eager"), which refuses to run under a
    user dispatch mode. This flag calls the HOP directly; the recorder only needs the ops."""
    monkeypatch.setattr(flex_module, "_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG", True)


def record_attention(case, backend, **kwargs):
    q, keys, q_idx, q_doc, offset = CASES[case](**kwargs)
    q.requires_grad_()
    recorder = HopRecorder if backend == "flex" else OpRecorder
    with recorder() as rec:
        out = attend_fast(q, keys, q_idx, q_doc, backend=backend, **mask_kwargs(offset))  # builds the BlockMask too
        out.square().sum().backward()
    return rec


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", ["encoder_cached_prefix", "decoder_cached", "slot_reads"])
def test_no_dynamic_shape_ops(case, backend, recordable_flex):
    rec = record_attention(case, backend)
    names = {name for name, _ in rec.ops}
    assert rec.flagged == [], f"data-dependent shapes: {sorted(set(rec.flagged))}"
    if backend == "flex":
        assert {"hop.flex_attention", "hop.flex_attention_backward", "aten.sort.stable"} <= names
    else:
        assert any("scaled_dot_product" in name for name in names)


@pytest.mark.parametrize("backend", BACKENDS)
def test_op_trace_does_not_depend_on_the_masks(backend, recordable_flex):
    """Different documents, decoder layouts and empty-slot counts: same ops, same shapes."""
    traces = [record_attention("decoder_cached", backend, seed=s).ops for s in (0, 1, 2)]
    assert traces[0] == traces[1] == traces[2]


def test_training_step_with_flex_has_no_dynamic_shape_ops(monkeypatch, recordable_flex):
    model = make_model("looped_ced", dtype=torch.float32)
    model.train()
    use_fast_attention(monkeypatch, "flex")
    tokens, docs, dec = random_rows(2, 49, model.cfg.vocab_size, seed=0, n_dec=30)  # includes drops
    with HopRecorder() as rec:
        _loss_and_grads(model, tokens, docs, dec, T=3)
    names = [name for name, _ in rec.ops]
    assert names.count("hop.flex_attention") == 2 + 3 * 2  # 2 encoder global layers, T x 2 decoder layers
    assert rec.flagged == [], f"data-dependent shapes: {sorted(set(rec.flagged))}"
