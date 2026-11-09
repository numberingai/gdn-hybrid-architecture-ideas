"""Exact-zero weight gradients by linear: the NVFP4 tail arms' counter (TrainConfig.log_zero_wgrad_every; off by default).

Why: a published large-scale FP4 training report saw zero-valued weight gradients reach 7% of the parameters late in NVFP4 training (FC2
Dgrad underflow leading to FC1 Wgrad zeros; section 1); the emulation reproduces the mechanism (a block scale that
underflows decodes its block to zeros: ced/model/nvfp4.py nvfp4_encode).

What is counted. For every linear of the layers' mixers and FFNs (wgrad_modules: encoder.layers.<i>.<mixer|mlp>.<name>,
in the model's order; an NVFP4Linear or FP8Linear is an nn.Linear), from the gradient of its weight (out_features,
in_features) as the step's micro-batches accumulated it into .grad, before clipping (clipping multiplies by one positive
number and the optimizer reads the clipped gradient; the zeros are the backward's):
  zeros      the number of elements equal to 0.0 exactly;
  zero_rows  the number of output rows (output features) whose every element is 0.0 (that report's FC1 pattern: a feature
             whose upstream gradient underflowed to zero in every token block of every micro-batch).
A weight that received no gradient at all this step (.grad None) gives -1 for both (none of the linears counted here is
ever skipped in a training forward of the control; the reader treats -1 as "no gradient").

Read only. counts() makes boolean temporaries from the gradients and reduces them on the device; it writes to no tensor
the step uses, draws no random number and runs outside any compiled region, so the losses, the gradients, the
optimizer's state, the RNG state and the checkpoints' tensors are those of the run without it, bit for bit
(tests/test_nvfp4_arms.py). The host reads the counts once, after the optimizer step, with the step's other metrics.

Fixed shapes: every tensor's shape follows from the weights' (comparisons, sums over fixed dimensions, a stack).
"""

from __future__ import annotations

import torch
import torch.nn as nn

PARTS = ("mixer", "mlp")


def wgrad_modules(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """(name, module) for every nn.Linear (subclasses included) of the layers' mixers and FFNs, in layer order: the walk of
    ced/model/nvfp4.py apply_nvfp4. Refused (ValueError) for a model without encoder.layers holding mixer and mlp."""
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise ValueError("the zero-wgrad counter walks model.encoder.layers (arch 'control'): this model has none")
    out = []
    for i, layer in enumerate(layers):
        for part in PARTS:
            sub = getattr(layer, part, None)
            if sub is None:
                raise ValueError(f"encoder.layers.{i} has no {part!r}: the zero-wgrad counter is built for the control's layers")
            for name, m in sub.named_modules():
                if isinstance(m, nn.Linear):
                    out.append((f"encoder.layers.{i}.{part}.{name}", m))
    return out


def describe(modules: list[tuple[str, nn.Linear]]) -> dict:
    """What the start record holds, so that a step's two lists can be read by name: the names, each weight's number of
    elements and of output rows."""
    return {"modules": [n for n, _ in modules], "numel": [int(m.weight.numel()) for _, m in modules],
            "rows": [int(m.weight.shape[0]) for _, m in modules]}


def counts(grads: list[torch.Tensor | None]) -> torch.Tensor:
    """(len(grads), 2) int64 on the gradients' device: [zeros, zero_rows] per gradient (module docstring); [-1, -1] where
    a gradient is None. Nothing is read to the host here."""
    rows = []
    device = next((g.device for g in grads if g is not None), torch.device("cpu"))
    none = torch.full((2,), -1, dtype=torch.int64, device=device)
    for g in grads:
        if g is None:
            rows.append(none)
            continue
        z = g == 0
        rows.append(torch.stack([z.sum(), z.reshape(z.shape[0], -1).all(dim=1).sum()]).to(torch.int64))
    return torch.stack(rows) if rows else torch.zeros((0, 2), dtype=torch.int64, device=device)


def read(t: torch.Tensor) -> dict:
    """The metrics entry of one logged step: {"zeros": [...], "zero_rows": [...]} (one host read)."""
    v = t.cpu().tolist()
    return {"zeros": [int(r[0]) for r in v], "zero_rows": [int(r[1]) for r in v]}


def fractions(entry: dict, desc: dict) -> dict[str, tuple[float | None, float | None]]:
    """{module: (zero fraction, zero-row fraction)} of one step's entry against the start record's description; None where
    the weight had no gradient (-1)."""
    out = {}
    for name, n, r, z, zr in zip(desc["modules"], desc["numel"], desc["rows"], entry["zeros"], entry["zero_rows"]):
        out[name] = (None, None) if z < 0 else (z / max(1, n), zr / max(1, r))
    return out
