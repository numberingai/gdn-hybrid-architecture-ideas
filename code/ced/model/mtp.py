"""A DeepSeek-V3-style multi-token-prediction (MTP) drafter of depth 1 on a FROZEN trunk (ModelConfig.mtp).

The form (DeepSeek-V3, arXiv 2412.19437 section 2.2, with k = 1):

    h'_i  = M [ RMSNorm_h(h_i) ; RMSNorm_e(Emb(t_{i+1})) ]      M: (d, 2d), no bias ("eh_proj")
    y_i   = Block(h'_1 .. h'_i)                                one causal block over the head's own positions
    p_i   = softmax(OutHead(RMSNorm_o(y_i)))                   the distribution of t_{i+2}

- h_i is the trunk's final hidden state AFTER its final norm: the LM head's input (HybridLM.forward with return_hidden,
  `hidden_aux`), the state vLLM passes to DeepSeek's and Qwen3-Next's MTP layers (their model forward returns the
  normed state). The paper says only "the representation given by the main model"; hnorm normalises it again, so the
  two readings differ by the trunk's final per-channel gain only.
- Emb is the trunk's own input embedding (`model.embed`, its raw output, without the trunk's embed_norm) and OutHead
  the trunk's own output head (`model.head_weight()`): shared, never copied (DeepSeek: "its embedding layer is shared
  with the main model", "its output head is shared with the main model"). Both stay frozen with the trunk.
- The concatenation is [hidden ; embedding] as the paper writes it (vLLM's DeepSeek and Qwen3-Next code put the
  embedding first; for a head trained from scratch the order is a relabelling of M's columns).
- RMSNorm_o is the head's own final norm (DeepSeek's `shared_head.norm`), zero-centred like every norm here.
- Block (MTPConfig.block "attention"): one of the trunk's own global-attention layers (EncoderLayer "attention": the
  Peri-LN sandwich, gated GQA attention with the trunk's model.encoder.attention settings, SwiGLU FFN of
  model.ffn_hidden), at the trunk's residual scale. An attention block, as DeepSeek's (the main model's block) and
  Qwen3-Next's (vLLM: `layer_type="full_attention"`) MTP layers: its own state across positions is a key/value cache,
  which a rejected draft truncates; no recurrent state of its own to roll back.
- Drafts of depth k = 2, 3 apply the same module again on its own output: position i + 1 reads (y_i, d_1) in place of
  (h_{i+1}, t_{i+2}), and so on (ced/infer/mtp_spec.py MTPDrafter). Training here is depth 1 only (scripts/train_mtp.py).

Off by default (MTPConfig.enabled False): build_model builds nothing of this, so the model, its state dict, its
parameter order and its RNG use are the model without the key (tests/test_mtp.py). On: build_model builds the trunk
exactly as without the key (every trunk tensor first, from the same RNG draws), then this head as `model.mtp`, drawn
after it (attach_mtp). HybridLM.forward never calls the head: the trunk's outputs are the same with it or without.
Built for arch "control" with one pass; refused elsewhere (check_mtp). The trainer refuses the key: the head is trained
by scripts/train_mtp.py with the trunk frozen (freeze_trunk).

The trunks it is built on (MTP_TRUNKS; check_mtp, check_mtp_on_cp):
- the base (encoder.attn_res "none"), as built on 2026-10-15;
- R1-entry's own form (attn_res "block_entry": one static reader a block entry, the unfused path, no GDN-side arm). Its
  depth attention is per position, so its cache is the layers' own (the GDN states and the key/value sets) and the
  speculative decoder's rollback is unchanged (tests/test_mtp_on_cp.py);
- the Copy-Paste trunk on R1-entry (model.cp_head with pointer_input "final" (cp-r1e) or "mix" (cp-mix); every other CP
  key at its default). The drafter is the same head on the same h (the final norm's output, which is also what the <copy>
  row reads), trained exactly as on a plain trunk: it does NOT get the pointer, and its output head is the LM head's V
  rows (lm_head_weight), never the <copy> row. The verifier in speculative decoding is the trunk's own output, the mixture
  p = (1 - lambda) p_LM + lambda p_CP with the pointer over the copy cache, whose entries a rejected draft cuts off as the
  attention keys (ced/infer/mtp_spec.py). The CP head on the base (attn_res "none"), "shared", the fused kernel, the band,
  the length rule, force_lambda_zero, score "lm" and any later CP key stay refused: nothing tests the pair.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ced.config import MTPConfig, ModelConfig
from ced.model.encoder import EncoderLayer
from ced.model.norms import RMSNorm
from ced.model.sequence import SeqContext

MTP_BLOCKS = ("attention",)
PREFIX = "mtp."  # the head's parameter names in the model's state dict
# model.encoder.attn_res values the drafter is built on (module docstring): the base and R1-entry's own form
MTP_TRUNKS = ("none", "block_entry")
# model.cp_head.pointer_input values the drafter is built on, on R1-entry only (cp-r1e, cp-mix; module docstring)
MTP_CP_POINTERS = ("final", "mix")
# model.cp_head keys that may differ from their defaults under the drafter: they change the pointer's tensors or how the
# same numbers are computed, never what the verifier is (every other CP key must be at its default: check_mtp_on_cp)
MTP_CP_FREE = ("enabled", "d_c", "query_chunk", "qk_norm", "init", "pointer_input")


def check_mtp(cfg: ModelConfig) -> None:
    """Refuse the MTP keys wherever the drafter is not built, and their bad values. ValueError."""
    m = cfg.mtp
    if type(m.enabled) is not bool:
        raise ValueError(f"model.mtp.enabled {m.enabled!r}: true or false")
    if m.block not in MTP_BLOCKS:
        raise ValueError(f"model.mtp.block {m.block!r}: one of {MTP_BLOCKS}")
    if not m.enabled:
        if m.block != MTPConfig().block:
            raise ValueError(f"model.mtp.block {m.block!r} without model.mtp.enabled: the key would be ignored")
        return
    from ced.model.control import has_loop

    if cfg.arch != "control":
        raise ValueError(f"model.mtp is built for arch 'control' (one pass), not for arch {cfg.arch!r}")
    if has_loop(cfg) or cfg.encoder.loops != 1:
        raise ValueError("model.mtp is built for the one-pass control: no loop.pass2_gate, loop.recur_layers or looped "
                         "encoder (the trunk's hidden state and its cache would be another model's)")
    if cfg.encoder.canon_kernel:
        raise ValueError("model.mtp is not built together with model.encoder.canon_kernel: a Canon layer's history is "
                         "cache state the speculative decoder's rollback does not handle")
    e = cfg.encoder
    if e.attn_res not in MTP_TRUNKS:
        raise ValueError(f"model.mtp is not built together with model.encoder.attn_res {e.attn_res!r}: nothing "
                         f"tests the depth-mix stack's cache under the speculative decoder's rollback (built: {MTP_TRUNKS}, "
                         "the base and R1-entry's own form)")
    if e.attn_res == "block_entry":
        r1e = {"model.encoder.attn_res_heads": (e.attn_res_heads, 1),
               "model.encoder.attn_res_query": (e.attn_res_query, "static"),
               "model.encoder.attn_res_fused": (e.attn_res_fused, False),
               "model.encoder.gdn_write_mix": (e.gdn_write_mix, "none"),
               "model.encoder.gdn_ccr_spacing": (e.gdn_ccr_spacing, 0)}
        for key, (value, off) in r1e.items():
            if value != off:
                raise ValueError(f"model.mtp on R1-entry (model.encoder.attn_res 'block_entry') is built for R1-entry's own "
                                 f"form only: not together with {key} {value!r} (nothing tests the pair)")
    if cfg.cp_head.enabled:
        check_mtp_on_cp(cfg)
    if cfg.encoder.layer_scalars != "none":
        raise ValueError(f"model.mtp is not built together with model.encoder.layer_scalars "
                         f"{cfg.encoder.layer_scalars!r}: nothing tests the pair")
    if cfg.nvfp4.mode != "none" or cfg.fp8.mode != "none":
        raise ValueError("model.mtp is not built together with model.nvfp4 / model.fp8 (the head trains on a frozen BF16 "
                         "trunk; nothing tests the pair)")
    if cfg.encoder.attention.normalizer != "softmax":
        raise ValueError(f"model.mtp's attention block is built for the softmax normaliser, not "
                         f"{cfg.encoder.attention.normalizer!r}")


def check_mtp_on_cp(cfg: ModelConfig) -> None:
    """The drafter on a Copy-Paste trunk (module docstring): built on R1-entry with pointer_input "final" or "mix" and
    every CP key outside MTP_CP_FREE at its default (a key added to CPHeadConfig later is refused until a build tests
    it). ValueError, by name, with "model.cp_head" and "model.mtp" in the message. ced/model/cp_head.py check_cp_head
    calls this for a CP model with model.mtp.enabled (and check_mtp for every model with it)."""
    import dataclasses

    from ced.config import CPHeadConfig

    c, d = cfg.cp_head, CPHeadConfig()
    if cfg.encoder.attn_res != "block_entry":
        raise ValueError(f"model.cp_head is not built together with model.mtp.enabled on model.encoder.attn_res "
                         f"{cfg.encoder.attn_res!r}: the drafter on a CP trunk is built on R1-entry ('block_entry', the "
                         "cp-r1e / cp-mix trunks) only; nothing tests the pair")
    if c.pointer_input not in MTP_CP_POINTERS:
        raise ValueError(f"model.cp_head.pointer_input {c.pointer_input!r} is not built together with model.mtp.enabled: "
                         f"nothing tests the pair (built: {MTP_CP_POINTERS})")
    for f in dataclasses.fields(CPHeadConfig):
        if f.name not in MTP_CP_FREE and getattr(c, f.name) != getattr(d, f.name):
            raise ValueError(f"model.cp_head.{f.name} {getattr(c, f.name)!r} is not built together with model.mtp.enabled: "
                             "nothing tests the pair (the drafter on a CP trunk: every CP key but "
                             f"{', '.join(k for k in MTP_CP_FREE if k != 'enabled')} at its default)")


class MTPHead(nn.Module):
    """The head's own tensors (module docstring): hnorm, enorm, eh_proj, block, out_norm."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from ced.model.scaling import residual_scales

        d, eps = cfg.d_model, cfg.norm_eps
        self.hnorm = RMSNorm(d, eps)
        self.enorm = RMSNorm(d, eps)
        self.eh_proj = nn.Linear(2 * d, d, bias=False)
        # one of the trunk's global-attention layers, at the trunk's residual scale (1.0 for the one-pass control)
        self.block = EncoderLayer("attention", cfg, residual_scales(cfg)[0], cfg.encoder.attention)
        self.out_norm = RMSNorm(d, eps)

    def forward(self, hidden: torch.Tensor, next_tokens: torch.Tensor, embed: nn.Embedding, ctx: SeqContext,
                cache=None) -> tuple[torch.Tensor, object]:
        """hidden (B, L, d): the trunk's head input h_i at the head's positions i (or, at a draft of depth >= 2, the
        head's own output y of the position before); next_tokens (B, L): t_{i+1} (or the drafted token); embed: the
        trunk's input embedding; ctx: the positions' sequence context (the trunk's positions); cache: the block's key
        set of earlier positions (None: none). Returns (y (B, L, d), the block's key set including these positions).
        The distribution of t_{i+2} is softmax(lm_head(out_norm(y_i), the trunk's head_weight()))."""
        e = embed(next_tokens)
        # the block's residual stream in the trunk's stream dtype (float32 under bf16 autocast, as the trunk's own:
        # its embedding output is float32), not in the projection's autocast dtype
        u = self.eh_proj(torch.cat([self.hnorm(hidden), self.enorm(e)], dim=-1)).to(hidden.dtype)
        return self.block(u, ctx, cache)


def attach_mtp(model: nn.Module) -> nn.Module:
    """model.mtp = a new MTPHead, Normal(0, init_std) on its linear weights (init_weights, the trunk's rule), drawn
    from the global RNG AFTER every trunk tensor: the trunk starts as the model without the key, bit for bit."""
    from ced.model.looped_ced import init_weights

    head = MTPHead(model.cfg)
    init_weights(head, model.cfg.init_std)
    model.mtp = head
    return model


def mtp_parameter_names(model: nn.Module) -> list[str]:
    return [n for n, _ in model.named_parameters() if n.startswith(PREFIX)]


def freeze_trunk(model: nn.Module) -> list[nn.Parameter]:
    """requires_grad False on every trunk tensor (every parameter not under `mtp.`), True on the head's; returns the
    head's parameters (the only ones an optimizer may get). AssertionError without a head."""
    assert getattr(model, "mtp", None) is not None, "the model has no MTP head (model.mtp.enabled)"
    head = []
    for name, p in model.named_parameters():
        own = name.startswith(PREFIX)
        p.requires_grad_(own)
        if own:
            head.append(p)
    return head


def assert_trunk_frozen(model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> None:
    """The frozen-trunk invariant: no trunk tensor requires a gradient or holds one, and an optimizer holds the head's
    parameters only. AssertionError otherwise."""
    head = {id(p) for n, p in model.named_parameters() if n.startswith(PREFIX)}
    for name, p in model.named_parameters():
        if id(p) in head:
            continue
        assert not p.requires_grad, f"trunk tensor {name} requires a gradient"
        assert p.grad is None, f"trunk tensor {name} holds a gradient"
    if optimizer is not None:
        held = {id(p) for g in optimizer.param_groups for p in g["params"]}
        assert held <= head, "the optimizer holds a trunk tensor"
        assert held == head, "the optimizer misses a head tensor"


def mtp_targets(targets: torch.Tensor, target_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, L) data targets t_{i+1} and their validity -> the head's targets t_{i+2} and validity, fixed shapes, no
    boolean indexing: target2[i] = targets[i + 1], valid2[i] = target_valid[i] and target_valid[i + 1] (t_{i+1} and
    t_{i+2} both continue position i's document); the last position of a row has no t_{i+2} in the row (False,
    target 0)."""
    t2 = torch.cat([targets[:, 1:], torch.zeros_like(targets[:, :1])], dim=1)
    v2 = torch.cat([target_valid[:, :-1] & target_valid[:, 1:], torch.zeros_like(target_valid[:, :1])], dim=1)
    return t2, v2


@torch.no_grad()
def trunk_hidden(model: nn.Module, tokens: torch.Tensor, doc_ids: torch.Tensor, varlen: dict | None = None,
                 cache=None, return_cache: bool = False):
    """The trunk's head input at every position (B, L, d) (HybridLM.forward with return_hidden), without a gradient.
    With return_cache: (hidden, the trunk's cache)."""
    out = model(tokens, doc_ids, varlen=varlen, cache=cache, return_cache=return_cache, return_hidden=True)
    return (out.hidden_aux, out.cache) if return_cache else out.hidden_aux
