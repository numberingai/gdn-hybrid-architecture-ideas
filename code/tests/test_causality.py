"""Causality: changing anything at position j never changes any output before j.

Runs the training forward (fixed decoder capacity, gathered slots). Outputs before j must be
bit-identical: masked attention gives exactly-zero weight to future keys, and the GDN recurrence
never reads ahead, so there is no tolerance to hide behind.
"""

import pytest
import torch

from helpers import LOOPED_VARIANTS, make_model, random_rows

L = 48


@pytest.mark.parametrize("arch,variant", [("looped_ced", v) for v in LOOPED_VARIANTS] + [("control", "default")])
def test_future_token_does_not_change_past_outputs(arch, variant):
    model = make_model(arch, LOOPED_VARIANTS[variant] if arch == "looped_ced" else None)
    tokens, docs, dec = random_rows(2, L, model.cfg.vocab_size, seed=1)
    T = model.cfg.loop.t_eval
    base = model(tokens, docs, dec, T=T).logits
    for j in [1, 7, 20, 33, L - 1]:
        changed = tokens.clone()
        changed[:, j] = (changed[:, j] + 1 + j) % model.cfg.vocab_size
        out = model(changed, docs, dec, T=T).logits
        assert torch.equal(out[:, :j], base[:, :j]), f"outputs before {j} changed"
        assert not torch.equal(out[:, j:], base[:, j:]), "the change had no effect at all (test is vacuous)"


@pytest.mark.parametrize("variant", ["default", "memory_kv_per_layer"])
def test_future_decoder_mask_does_not_change_past_outputs(variant):
    """Whether a later position is a decoder position must not affect earlier outputs, including
    when the row exceeds the decoder capacity (keep-first-C is prefix-stable)."""
    model = make_model("looped_ced", LOOPED_VARIANTS[variant])
    tokens, docs, dec = random_rows(2, L, model.cfg.vocab_size, seed=2)
    capacity = model.capacity(L)
    base = model(tokens, docs, dec).logits
    g = torch.Generator().manual_seed(0)
    for j in [5, 16, 30]:
        changed = dec.clone()
        changed[:, j:] = torch.rand(changed[:, j:].shape, generator=g) < 0.7  # many more decoder positions
        out = model(tokens, docs, changed)
        assert int(out.n_dropped.sum()) > 0 or int(changed.sum(1).max()) <= capacity
        assert torch.equal(out.logits[:, :j], base[:, :j])


@pytest.mark.parametrize("arch", ["looped_ced", "control"])
def test_gradients_do_not_flow_from_future_inputs(arch):
    """d logits[t] / d (input embedding at s) is exactly zero for every s > t. This catches leaks
    through gather/scatter or masks that a forward-only check could miss."""
    model = make_model(arch)
    tokens, docs, dec = random_rows(1, L, model.cfg.vocab_size, seed=3)
    captured = {}

    def keep_embedding_output(module, inputs, output):
        output.retain_grad()
        captured["h"] = output

    hook = model.embed.register_forward_hook(keep_embedding_output)
    try:
        for t in [0, 9, 24, 40]:
            model.zero_grad(set_to_none=True)
            logits = model(tokens, docs, dec).logits
            logits[:, t].sum().backward()
            grad = captured["h"].grad
            assert torch.count_nonzero(grad[:, t + 1 :]) == 0, f"future inputs influence position {t}"
            assert torch.count_nonzero(grad[:, : t + 1]) > 0
    finally:
        hook.remove()


@pytest.mark.parametrize("arch", ["looped_ced", "control"])
def test_packed_documents_are_isolated(arch):
    """A document packed after another one gives the same outputs as the document on its own:
    attention masks, GDN state and conv window resets, and document-relative RoPE positions."""
    model = make_model(arch, {"encoder.gdn.conv_kernel": 4} if arch == "looped_ced" else None)
    tokens, _, dec = random_rows(1, L, model.cfg.vocab_size, seed=4)
    split = 19
    docs = torch.zeros(1, L, dtype=torch.long)
    docs[:, split:] = 1
    packed = model(tokens, docs, dec, capacity=L).logits
    alone = model(tokens[:, split:], torch.zeros(1, L - split, dtype=torch.long), dec[:, split:], capacity=L - split).logits
    torch.testing.assert_close(packed[:, split:], alone, rtol=0, atol=1e-12)
    # And the earlier document does not see the later one (causality covers this, checked anyway).
    first = model(tokens[:, :split], torch.zeros(1, split, dtype=torch.long), dec[:, :split], capacity=split).logits
    torch.testing.assert_close(packed[:, :split], first, rtol=0, atol=1e-12)
