"""Shared helpers for the CPU test suite."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import torch

from ced.config import (
    ModelConfig,
    tiny_control_config,
    tiny_model_config,
    tiny_second_pass_config,
    tiny_v2_config,
    tiny_v2_control_config,
)
from ced.model.control import build_model
from ced.train.data import counter_generator, sample_spans


def set_config(cfg: ModelConfig, overrides: dict) -> ModelConfig:
    """Apply dotted-key overrides, e.g. {"loop.injection": "none", "decoder.attention.gate": "headwise"}."""
    cfg = dataclasses.replace(cfg)
    for key, value in overrides.items():
        obj = cfg
        *path, last = key.split(".")
        for part in path:
            obj = getattr(obj, part)
        assert hasattr(obj, last), key
        setattr(obj, last, value)
    return cfg


def dump_env(tree, pythonpath: str) -> dict:
    """The environment of an "off is main" dump process (tests/*_dump.py run by the code tree `tree` with PYTHONPATH `pythonpath`).

    Since 2026-10-26 the fused write mix stores its extra mixes in float32 by default, where the older main trees stored them in bf16 under
    autocast unless CED_WRITE_MIX_STORAGE=source. So a tree other than this repository (an old main) runs with that setting, which is this tree's
    default, bit for bit (a model without the fused write mix reads no such variable), and this repository's own process must not have the variable
    (it is refused by name)."""
    repo = Path(__file__).resolve().parent.parent
    env = {k: v for k, v in os.environ.items() if k != "CED_WRITE_MIX_STORAGE"}
    env["PYTHONPATH"] = pythonpath
    if Path(tree).resolve() != repo.resolve():
        env["CED_WRITE_MIX_STORAGE"] = "source"
    return env


@torch.no_grad()
def perturb_parameters(model: torch.nn.Module, scale: float = 0.05, seed: int = 0) -> None:
    """Add small noise to every parameter so that zero-initialised parts (norm gains, loop
    embeddings, identity-initialised matrices) actually take part in the computation. Otherwise
    a test could pass only because some component is still the identity."""
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.add_(scale * torch.randn(p.shape, generator=g, dtype=torch.float64).to(p.dtype))


# Tiny base configs by name: the four arches, plus "v2_control", the control with the v2 layer
# settings (untied output embedding, sigmoid GDN gate): controls (a) and (b).
TINY_CONFIGS = {
    "looped_ced": tiny_model_config,
    "control": tiny_control_config,
    "ced_v2": tiny_v2_config,
    "second_pass": tiny_second_pass_config,
    "v2_control": tiny_v2_control_config,
}


def make_model(arch: str = "looped_ced", overrides: dict | None = None, dtype=torch.float64, seed: int = 0):
    cfg = set_config(TINY_CONFIGS[arch](), overrides or {})
    torch.manual_seed(seed)
    model = build_model(cfg).to(dtype).eval()
    perturb_parameters(model, seed=seed)
    return model


def random_rows(
    batch: int,
    length: int,
    vocab: int,
    seed: int,
    n_dec: int | None = None,
    n_docs: int = 3,
    span_min: int = 2,
    span_max: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random tokens, packed documents and decoder spans (n_dec decoder positions per row)."""
    tokens, docs, decs = [], [], []
    for b in range(batch):
        g = counter_generator(seed, b, stream=7)
        tokens.append(torch.randint(0, vocab, (length,), generator=g))
        cuts = sorted(torch.randperm(length - 1, generator=g)[: n_docs - 1].add(1).tolist())
        doc = torch.zeros(length, dtype=torch.long)
        for c in cuts:
            doc[c:] += 1
        docs.append(doc)
        k = length // 4 if n_dec is None else n_dec
        decs.append(sample_spans(length, k, g, distribution="uniform", span_min=span_min, span_max=span_max))
    return torch.stack(tokens), torch.stack(docs), torch.stack(decs)


# Configuration variants every equivalence test runs over.
LOOPED_VARIANTS = {
    "default": {},
    "T1_no_injection": {"loop.injection": "none", "loop.t_eval": 1},
    "T2": {"loop.t_eval": 2},
    "no_loop_embedding": {"loop.loop_embedding": False},
    "nope": {"encoder.attention.rope_fraction": 0.0, "decoder.attention.rope_fraction": 0.0},
    "memory_kv_per_layer": {"decoder.memory_kv_per_layer": True},
    "headwise_gate": {"encoder.attention.gate": "headwise", "decoder.attention.gate": "headwise"},
    "completep_scaling": {"residual_scale.mode": "completep_effective_depth"},
    "loop_factored_scaling": {"residual_scale.mode": "loop_factored"},
    "gdn_hf_init": {"encoder.gdn.init": "hf"},
    # Phase 2 arms: injection-only is "default"; HC-only and both:
    "static_hc_only": {"loop.hyper_connections": "static", "loop.injection": "none"},
    "static_hc_and_injection": {"loop.hyper_connections": "static", "loop.hc_write": "full"},
    "dynamic_hc_only": {"loop.hyper_connections": "dynamic_diag", "loop.injection": "none", "loop.hc_streams": 4},
    "dynamic_hc_and_injection": {"loop.hyper_connections": "dynamic_diag", "loop.hc_streams": 2},
    # 2026-10-03 primary-candidate features
    "hybrid_decoder": {"decoder.memory_attn_interval": 2},
    "hybrid_decoder_4_layers": {"decoder.n_layers": 4, "decoder.memory_attn_interval": 4, "loop.t_eval": 2},
    "inverse_loops_scaling": {"residual_scale.mode": "inverse_loops"},
    "routed_ffn": {"decoder.ffn_experts": 4, "decoder.ffn_top_k": 2, "decoder.memory_attn_interval": 2},
    "untied_mixers": {"decoder.untie_mixers": "all", "decoder.memory_attn_interval": 2},
    "untie_attention_routed_ffn": {"decoder.untie_mixers": "attention", "decoder.ffn_experts": 4,
                                   "decoder.ffn_expert_hidden": 48, "decoder.memory_attn_interval": 2,
                                   "decoder.router_z_weight": 1e-4},
    "looped_encoder_yocou": {"encoder.loops": 2, "loop.t_eval": 1, "decoder.memory_attn_interval": 2,
                             "loop.hyper_connections": "static", "residual_scale.mode": "inverse_loops"},
    "gdn_chunked_kernel": {"encoder.gdn.kernel": "chunked", "encoder.gdn.chunk_size": 4,
                           "decoder.gdn.kernel": "chunked", "decoder.gdn.chunk_size": 4, "decoder.memory_attn_interval": 2},
    "depth_attention": {"loop.depth_attention": "layer", "loop.depth_attention_head_dim": 8,
                        "decoder.memory_attn_interval": 2, "loop.t_eval": 2},
}
