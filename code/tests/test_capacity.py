"""Fixed-capacity gather correctness and fixed shapes.

1. Slot selection keeps exactly the first C decoder positions of each row, in order.
2. The fixed-capacity decoder (gathered slots, one masked softmax) equals an independent naive
   decoder that loops over positions and builds each query's key list by hand. With more decoder
   positions than C, the dropped ones behave as context positions in both.
3. The training step has no data-dependent shapes: no nonzero / masked_select / boolean indexing,
   and the exact same op sequence and tensor shapes whatever the decoder mask.
4. The gathers with an inverse map (gather_rows, gather_slots_inv, scatter_slots_inv; control (c)'s
   pass 2) give bitwise the values and gradients of torch.gather / gather_slots / scatter_slots,
   including empty slots and shifted reads (the q-conv taps), and their backward has no scatter-add
   or index_put (the sort-based deterministic kernel they replace).
"""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from ced.config import TrainConfig, DataConfig, tiny_model_config
from ced.model.capacity import (
    decoder_capacity,
    gather_rows,
    gather_slots,
    gather_slots_inv,
    scatter_slots,
    scatter_slots_inv,
    select_slots,
    slot_inverse,
)
from ced.model.looped_ced import tied_head
from ced.model.sequence import sequence_context
from ced.train.loss import lm_loss
from ced.train.trainer import Trainer

from helpers import make_model, random_rows


def test_decoder_capacity_formula():
    assert decoder_capacity(4096, 0.25, 128) == 1024
    assert decoder_capacity(8192, 0.25, 128) == 2048
    assert decoder_capacity(1000, 0.25, 128) == 256
    assert decoder_capacity(64, 0.25, 128) == 64  # never more than the row
    assert decoder_capacity(64, 0.25, 8) == 16


@pytest.mark.parametrize("density,capacity", [(0.0, 8), (0.2, 8), (0.5, 8), (0.9, 8), (0.5, 40), (1.0, 40)])
def test_select_slots_keeps_first_C_in_order(density, capacity):
    B, L = 4, 40
    g = torch.Generator().manual_seed(int(density * 10) + capacity)
    dec = torch.rand(B, L, generator=g) < density
    slots = select_slots(dec, capacity)
    assert slots.index.shape == (B, capacity)
    for b in range(B):
        positions = [t for t in range(L) if dec[b, t]]
        kept = positions[:capacity]
        expected = kept + [L] * (capacity - len(kept))
        assert slots.index[b].tolist() == expected
        assert slots.valid[b].tolist() == [i < len(kept) for i in range(capacity)]
        assert [t for t in range(L) if slots.kept[b, t]] == kept
        assert int(slots.n_dropped[b]) == len(positions) - len(kept)


def test_gather_scatter_round_trip():
    B, L, C = 2, 12, 5
    g = torch.Generator().manual_seed(0)
    dec = torch.rand(B, L, generator=g) < 0.3
    slots = select_slots(dec, C)
    x = torch.randn(B, L, 3, generator=g)
    gathered = gather_slots(x, slots.index)
    assert torch.equal(gathered[slots.valid], x[slots.kept])
    assert torch.count_nonzero(gathered[~slots.valid]) == 0  # empty slots read zeros
    base = torch.randn(B, L, 3, generator=g)
    values = torch.randn(B, C, 3, generator=g)
    out = scatter_slots(base, slots.index, values)
    assert torch.equal(out[slots.kept], values[slots.valid])
    assert torch.equal(out[~slots.kept], base[~slots.kept])  # empty slots write nowhere


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("density", [0.05, 0.3, 0.9])
def test_inverse_gathers_equal_torch_gather_bitwise(density, dtype):
    """gather_slots_inv / scatter_slots_inv against gather_slots / scatter_slots (into zeros): the same
    values and the same gradients, bitwise, with empty slots (low density) and drops (high density)."""
    B, L, C = 3, 40, 12
    g = torch.Generator().manual_seed(int(density * 100))
    dec = torch.rand(B, L, generator=g) < density
    slots = select_slots(dec, C)
    inv = slot_inverse(slots.index, L)
    # the inverse: slot c reads position index[c] <=> inv[index[c]] = c; positions without a slot map to C
    for b in range(B):
        for c in range(C):
            if bool(slots.valid[b, c]):
                assert int(inv[b, slots.index[b, c]]) == c
        assert int((inv[b] < C).sum()) == int(slots.valid[b].sum())
    x = torch.randn(B, L, 2, 5, generator=g, dtype=torch.float64).to(dtype)
    w = torch.randn(B, C, 2, 5, generator=g, dtype=torch.float64).to(dtype)
    res = {}
    for name in ("plain", "inverse"):
        xi = x.clone().requires_grad_(True)
        out = gather_slots(xi, slots.index) if name == "plain" else gather_slots_inv(xi, slots.index, inv)
        (out * w).sum().backward()
        res[name] = (out.detach(), xi.grad)
    assert torch.equal(res["plain"][0], res["inverse"][0]) and torch.equal(res["plain"][1], res["inverse"][1])
    v = torch.randn(B, C, 2, 5, generator=g, dtype=torch.float64).to(dtype)
    wl = torch.randn(B, L, 2, 5, generator=g, dtype=torch.float64).to(dtype)
    res = {}
    for name in ("plain", "inverse"):
        vi = v.clone().requires_grad_(True)
        out = (scatter_slots(torch.zeros(B, L, 2, 5, dtype=dtype), slots.index, vi) if name == "plain"
               else scatter_slots_inv(vi, slots.index, inv))
        (out * wl).sum().backward()
        res[name] = (out.detach(), vi.grad)
    assert torch.equal(res["plain"][0], res["inverse"][0]) and torch.equal(res["plain"][1], res["inverse"][1])


@pytest.mark.parametrize("shift", [0, 1, 2])
def test_inverse_gather_of_shifted_rows_equals_torch_gather(shift):
    """The q-conv tap read of control (c)'s pass 2 (GatedDeltaNet.zero_write_read): row index + shift of a
    (B, K-1+L, ...) tensor for valid slots, the fill row for empty slots, with the shifted inverse for the
    backward; against torch.gather of index + shift (empty slots masked out after the read, as there)."""
    B, L, C, K = 2, 30, 9, 4
    g = torch.Generator().manual_seed(shift)
    dec = torch.rand(B, L, generator=g) < 0.2
    slots = select_slots(dec, C)
    assert not bool(slots.valid.all())  # empty slots present
    inv = slot_inverse(slots.index, L)
    N = K - 1 + L
    x = torch.randn(B, N, 6, generator=g, dtype=torch.float64)
    w = torch.randn(B, C, 6, generator=g, dtype=torch.float64)
    take = slots.index + shift
    mask = slots.valid[..., None].to(x.dtype)
    res = {}
    for name in ("plain", "inverse"):
        xi = x.clone().requires_grad_(True)
        if name == "plain":
            out = torch.gather(xi, 1, take[..., None].expand(B, C, 6))
        else:
            pad = lambda n: inv.new_full((B, n), C)  # noqa: E731
            bwd = torch.cat([pad(shift), inv, pad(K - 1 - shift)], dim=1)
            out = gather_rows(xi, torch.where(slots.valid, take, N), bwd)
        (out * mask * w).sum().backward()
        res[name] = ((out * mask).detach(), xi.grad)
    assert torch.equal(res["plain"][0], res["inverse"][0]) and torch.equal(res["plain"][1], res["inverse"][1])


def test_inverse_gather_backward_has_no_scatter_add():
    """The point of the inverse map: the backward is a gather, not a scatter-add / index_put (which
    PyTorch's deterministic mode runs as a sort over every element of the expanded index)."""
    B, L, C = 2, 64, 16
    dec = torch.rand(B, L, generator=torch.Generator().manual_seed(3)) < 0.3
    slots = select_slots(dec, C)
    inv = slot_inverse(slots.index, L)
    x = torch.randn(B, L, 8, requires_grad=True)
    v = torch.randn(B, C, 8, requires_grad=True)
    with OpRecorder() as rec:
        (gather_slots_inv(x, slots.index, inv).sum() + scatter_slots_inv(v, slots.index, inv).sum()).backward()
    names = {n for n, _ in rec.ops}
    assert not any("scatter" in n or "index_put" in n or "index_add" in n for n in names), names
    with OpRecorder() as rec:
        (gather_slots(x, slots.index).sum() + scatter_slots(torch.zeros(B, L, 8), slots.index, v).sum()).backward()
    assert any("scatter" in n or "index_put" in n for n, _ in rec.ops)  # not vacuous: the plain ones do


# ------------------------------------------------------------------------------------------
# naive decoder: per position, keys collected by hand (boolean logic is fine here; it is a test)


def naive_logits(model, tokens, docs, dec, T, capacity):
    cfg = model.cfg
    B, L = tokens.shape
    ctx, _ = sequence_context(docs, None)
    h_enc, _ = model.encoder(model.embed_norm(model.embed(tokens)), ctx)
    logits = tied_head(model.aux_norm(h_enc), model.embed).clone()

    for b in range(B):
        D = [t for t in range(L) if dec[b, t]][:capacity]  # first C decoder positions
        context = [s for s in range(L) if s not in D]
        memory = [m(h_enc[b : b + 1], _slice(ctx, b), ok=torch.ones(1, L, dtype=torch.bool)) for m in model.memory_kv]

        e = {t: model.inject_norm(h_enc[b, t]) for t in D}
        Y = {t: model.hyper_connections.start(e[t][None, None])[0, 0] for t in D}
        for k in range(T):
            z_in = {t: model.hyper_connections.read(Y[t][None, None], k)[0, 0] for t in D}
            u = {t: model.injection(z_in[t], e[t]) for t in D}
            if model.loop_embedding is not None:
                u = {t: u[t] + model.loop_embedding[k] for t in D}
            for l, layer in enumerate(model.decoder_layers):
                mem = memory[l if len(memory) > 1 else 0]
                h = {t: layer.attn_in_norm(u[t]) for t in D}
                own = {}
                for t in D:
                    pos = ctx.pos[b, t].view(1, 1)
                    q, gate = layer.attn.queries(h[t][None, None], pos)
                    kk, vv = layer.attn.keys_values(h[t][None, None], pos)
                    own[t] = (q[0, 0], None if gate is None else gate[0, 0], kk[0, 0], vv[0, 0])
                new_u = {}
                for t in D:
                    q, gate, _, _ = own[t]
                    keys, values = [], []
                    for s in context:
                        if s <= t and docs[b, s] == docs[b, t]:
                            keys.append(mem.k[0, s])
                            values.append(mem.v[0, s])
                    for s in D:
                        if s <= t and docs[b, s] == docs[b, t]:
                            keys.append(own[s][2])
                            values.append(own[s][3])
                    K, V = torch.stack(keys), torch.stack(values)  # (S, Hkv, hd)
                    group = q.shape[0] // K.shape[1]
                    heads = []
                    for i in range(q.shape[0]):
                        scores = (K[:, i // group] @ q[i]) / q.shape[-1] ** 0.5
                        heads.append(torch.softmax(scores, 0) @ V[:, i // group])
                    o = torch.stack(heads)[None, None]
                    attn_out = layer.attn.output(o, None if gate is None else gate[None, None])[0, 0]
                    v_t = u[t] + layer.residual_scale * layer.attn_out_norm(attn_out)
                    v_t = v_t + layer.residual_scale * layer.mlp_out_norm(layer.mlp(layer.mlp_in_norm(v_t)))
                    new_u[t] = v_t
                u = new_u
            Y = {t: model.hyper_connections.write(Y[t][None, None], z_in[t][None, None], u[t][None, None], k)[0, 0] for t in D}
        for t in D:
            z = model.hyper_connections.finish(Y[t][None, None])
            logits[b, t] = tied_head(model.decoder_norm(z), model.embed)[0, 0]
    return logits


def _slice(ctx, b):
    return type(ctx)(**{name: getattr(ctx, name)[b : b + 1] for name in ("idx", "doc", "pos", "is_doc_start")})


@pytest.mark.parametrize(
    "overrides,n_dec",
    [
        ({}, 12),  # under capacity: some empty slots
        ({}, 16),  # exactly at capacity
        ({}, 26),  # over capacity: 10 drops per row become context positions
        ({"decoder.memory_kv_per_layer": True, "loop.t_eval": 2}, 20),
        ({"loop.injection": "none", "loop.loop_embedding": False, "loop.t_eval": 1}, 16),
        ({"loop.hyper_connections": "static", "loop.injection": "none"}, 26),
        ({"loop.hyper_connections": "dynamic_diag", "loop.hc_streams": 4}, 20),
        ({"residual_scale.mode": "loop_factored"}, 16),
    ],
)
def test_fixed_capacity_decoder_matches_naive_decoder(overrides, n_dec):
    model = make_model("looped_ced", overrides)
    L = 48
    tokens, docs, dec = random_rows(2, L, model.cfg.vocab_size, seed=n_dec, n_dec=n_dec)
    capacity = model.capacity(L)
    T = model.cfg.loop.t_eval
    with torch.no_grad():
        out = model(tokens, docs, dec, T=T)
        expected = naive_logits(model, tokens, docs, dec, T, capacity)
    assert int(out.n_dropped.sum()) == 2 * max(0, n_dec - capacity)
    torch.testing.assert_close(out.logits, expected, rtol=0, atol=1e-10)


# ------------------------------------------------------------------------------------------
# fixed shapes in the training step

DYNAMIC_SHAPE_OPS = {
    "aten.nonzero.default",
    "aten.masked_select.default",
    "aten.masked_scatter.default",
    "aten.argwhere.default",
    "aten._unique2.default",
    "aten.unique_dim.default",
    "aten.unique_consecutive.default",
    "aten.repeat_interleave.Tensor",
}
INDEX_OPS = {"aten.index.Tensor", "aten.index_put.default", "aten.index_put_.default", "aten._index_put_impl_.default"}


class OpRecorder(TorchDispatchMode):
    """Records every aten op with its output shapes, and flags data-dependent-shape ops."""

    def __init__(self):
        super().__init__()
        self.ops = []
        self.flagged = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func)
        shapes = tuple(tuple(x.shape) for x in tree_leaves(out) if isinstance(x, torch.Tensor))
        self.ops.append((name, shapes))
        bool_index = name in INDEX_OPS and any(
            isinstance(i, torch.Tensor) and i.dtype == torch.bool for i in tree_leaves(args[1])
        )
        if name in DYNAMIC_SHAPE_OPS or bool_index:
            self.flagged.append(name)
        return out


def record_train_step(model, batch_tokens, docs, dec, targets, valid, T):
    model.zero_grad(set_to_none=True)
    with OpRecorder() as rec:
        out = model(batch_tokens, docs, dec, T=T)
        loss, _ = lm_loss(out.logits, targets, valid, out.kept, ctx_weight=1.0, z_coef=1e-4)
        loss.backward()
    return rec


@pytest.mark.parametrize("arch", ["looped_ced", "control"])
def test_training_step_has_no_dynamic_shape_ops(arch):
    model = make_model(arch, dtype=torch.float32)
    model.train()
    L = 48
    tokens, docs, dec = random_rows(2, L, model.cfg.vocab_size, seed=0, n_dec=30)  # includes drops
    rec = record_train_step(model, tokens[:, :-1], docs[:, :-1], dec[:, :-1], tokens[:, 1:], docs[:, 1:] == docs[:, :-1], T=3)
    assert len(rec.ops) > 100
    assert rec.flagged == [], f"data-dependent shapes in the training step: {sorted(set(rec.flagged))}"


def test_training_step_shapes_do_not_depend_on_the_decoder_mask():
    """Same ops, same shapes, whether a row has no decoder positions, exactly C, or more than C."""
    model = make_model("looped_ced", dtype=torch.float32)
    model.train()
    L = 47
    tokens, docs, _ = random_rows(2, L + 1, model.cfg.vocab_size, seed=1)
    x, y, valid, d = tokens[:, :-1], tokens[:, 1:], docs[:, 1:] == docs[:, :-1], docs[:, :-1]
    g = torch.Generator().manual_seed(0)
    masks = [
        torch.zeros(2, L, dtype=torch.bool),
        random_rows(2, L, model.cfg.vocab_size, seed=2, n_dec=model.capacity(L))[2],
        torch.rand(2, L, generator=g) < 0.8,
    ]
    traces = [record_train_step(model, x, d, m, y, valid, T=2).ops for m in masks]
    assert traces[0] == traces[1] == traces[2]


def test_trainer_step_records_no_dynamic_shape_ops():
    """The real Trainer.train_step (data, forward, loss, backward, optimizer) under the recorder."""
    cfg = TrainConfig(model=tiny_model_config(), data=DataConfig(seq_len=32, batch_size=2), steps=4)
    trainer = Trainer(cfg, run_dir="/nonexistent-not-written")
    with OpRecorder() as rec:
        trainer.train_step()
    assert any("adamw" in name.lower() or "_foreach" in name or "addcdiv" in name for name, _ in rec.ops)
    assert rec.flagged == []
