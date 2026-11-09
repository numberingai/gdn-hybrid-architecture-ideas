"""The looped decoder's layers and the memory KV that context positions offer it.

Keys and values for decoder self-attention. Let D be the decoder positions that run the decoder. A
query at position t, loop k, decoder layer l attends, in ONE softmax, to

    { memory K/V of s           : s <= t, same document, s not in D }
  u { own loop-k, layer-l K/V of s : s <= t, same document, s in D   }   (including t itself)

So every position contributes exactly one kind of key. A decoder position never offers a memory
KV, and earlier assistant turns are seen by later turns only through their own per-loop KVs.

This is causal-consistent: loop k at position t needs only loop-k KVs of earlier positions, so a
teacher-forced pass (all positions of loop k in parallel) and cached decoding (each position runs
all its loops before the next position starts) compute the same function.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ced.config import ModelConfig
from ced.model.attention import GatedAttention, KeySet, allowed, attend, rope_cap
from ced.model.gdn import GatedDeltaNet, GDNState
from ced.model.mlp import RoutedSwiGLU, dense_ffn
from ced.model.norms import RMSNorm
from ced.model.rope import apply_rope
from ced.model.sequence import SeqContext


@dataclass
class SlotContext:
    """Bookkeeping for the gathered decoder slots of a chunk (see ced/model/capacity.py)."""

    idx: torch.Tensor  # (B, C) absolute index of each slot's position
    doc: torch.Tensor  # (B, C) document id; -1 for empty slots, which matches no real key
    pos: torch.Tensor  # (B, C) document-relative position
    valid: torch.Tensor  # (B, C) bool
    # Training only, optional: segment boundaries of the flattened (B * C) slot sequence for the
    # fused GDN kernel: row starts, document starts and the start of each row's empty-slot suffix.
    cu_seqlens: torch.Tensor | None = None
    cu_seqlens_cpu: torch.Tensor | None = None
    memo: dict = field(default_factory=dict)  # per-forward BlockMask memo (see SeqContext)
    # Optional (control (c)'s pass 2): (B, L) the slot of each position of the chunk, C where none
    # (capacity.slot_inverse), for gathers whose backward is another gather (capacity.gather_rows).
    inverse: torch.Tensor | None = None


class MemoryKV(nn.Module):
    """K, V = W_mem RMSNorm(h_enc), with a dedicated k-norm and the decoder's RoPE, applied once.

    The memory KV of a position is computed once and reused by every loop (and, unless
    memory_kv_per_layer, by every decoder layer), like cross-attention keys.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        attn = cfg.decoder.attention
        self.attn_cfg = attn
        self.in_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.proj = nn.Linear(cfg.d_model, 2 * attn.n_kv_heads * attn.head_dim, bias=False)
        self.k_norm = RMSNorm(attn.head_dim, cfg.norm_eps) if attn.qk_norm else nn.Identity()

    def forward(self, h_enc: torch.Tensor, ctx: SeqContext, ok: torch.Tensor) -> KeySet:
        if rope_cap(self.attn_cfg) is not None:
            # The rotary distance cap of the decoder's attention (AttentionConfig.rope_max_distance; attention.py,
            # "Rotary distance cap"): the memory also carries the keys before rotation, for the reads beyond the cap.
            B, L, _ = h_enc.shape
            a = self.attn_cfg
            k, v = self.proj(self.in_norm(h_enc)).view(B, L, 2, a.n_kv_heads, a.head_dim).unbind(dim=2)
            k_far = self.k_norm(k)
            k = apply_rope(k_far, ctx.pos, a.rope_fraction, a.rope_theta)
            return KeySet(k=k, v=v, idx=ctx.idx, doc=ctx.doc, ok=ok, k_far=k_far)
        k, v = self.keys_values(h_enc, ctx.pos)
        return KeySet(k=k, v=v, idx=ctx.idx, doc=ctx.doc, ok=ok)

    def keys_values(self, h_enc: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(k, v), each (B, L, H_kv, head_dim), at document positions pos (B, L): the tensors of forward's KeySet
        (CED v2's decoder fast path compiles this function on its own: ced_v2.CEDv2.compile_decoder_glue)."""
        assert rope_cap(self.attn_cfg) is None, \
            "the decoder's attention has a rotary distance cap (rope_max_distance): its memory keys are kept in two " \
            "forms (MemoryKV.forward); a caller of keys_values() would write the memory without the cap"
        B, L, _ = h_enc.shape
        a = self.attn_cfg
        k, v = self.proj(self.in_norm(h_enc)).view(B, L, 2, a.n_kv_heads, a.head_dim).unbind(dim=2)
        k = apply_rope(self.k_norm(k), pos, a.rope_fraction, a.rope_theta)
        return k, v


def slot_sequence_context(slots: SlotContext, past: GDNState | None) -> SeqContext:
    """The decoder slots seen as a sequence, for a GDN layer that runs over decoder positions only.

    A document starts at a valid slot whose document differs from the previous valid slot's (from
    earlier chunks: the last entry of the GDN conv history). Empty slots are never starts.
    """
    B = slots.doc.shape[0]
    last = past.conv_doc[:, -1] if past is not None else torch.full((B,), -1, dtype=torch.long, device=slots.doc.device)
    prev = torch.cat([last[:, None], slots.doc[:, :-1]], dim=1)
    return SeqContext(idx=slots.idx, doc=slots.doc, pos=slots.pos, is_doc_start=slots.valid & (slots.doc != prev),
                      cu_seqlens=slots.cu_seqlens if past is None else None,
                      cu_seqlens_cpu=slots.cu_seqlens_cpu if past is None else None)


class DecoderLayer(nn.Module):
    """One decoder layer, run at decoder slots only, with Peri-LN:

        u = u + eps * Norm(mixer(Norm(u)));   u = u + eps * Norm(FFN(Norm(u)))

    kind "memory_attention": gated softmax attention over the memory KV of context positions and
        the own loop-k KV of decoder positions (the key-set rule in the module docstring).
    kind "gdn": Gated DeltaNet over the sequence of decoder positions of the row, with its own
        recurrent state per loop (the 2026-10-03 default puts these in 3 of every 4 layers, so
        decoding reads the memory KV in only one layer in four).
    The mixer can be untied per loop (one copy per loop index); the FFN can be routed.
    """

    def __init__(self, cfg: ModelConfig, kind: str, residual_scale: float):
        super().__init__()
        d, eps = cfg.d_model, cfg.norm_eps
        self.kind = kind
        self.residual_scale = residual_scale
        self.attn_in_norm = RMSNorm(d, eps)
        self.attn_out_norm = RMSNorm(d, eps)
        self.mlp_in_norm = RMSNorm(d, eps)
        self.mlp_out_norm = RMSNorm(d, eps)
        untie = cfg.decoder.untie_mixers
        assert untie in ("none", "attention", "all"), untie
        n_mixers = cfg.loop.t_max if untie == "all" or (untie == "attention" and kind == "memory_attention") else 1
        if kind == "memory_attention":
            # v1's decoder builds its own mask (the self-mask over own keys) and calls attend / attend_fast itself,
            # so ASEntmax (AttentionConfig.normalizer) is not built here: refused, not silently run as softmax.
            assert cfg.decoder.attention.normalizer == "softmax", \
                "decoder.attention.normalizer asentmax is not built for v1's (looped_ced) decoder"
            self.mixers = nn.ModuleList(GatedAttention(d, cfg.decoder.attention, eps) for _ in range(n_mixers))
        elif kind == "gdn":
            self.mixers = nn.ModuleList(GatedDeltaNet(d, cfg.decoder.gdn, eps) for _ in range(n_mixers))
        else:
            raise ValueError(kind)
        lc = cfg.loop
        self.depth_attention = (
            DepthAttention(d, lc.depth_attention_head_dim, lc.depth_attention_rope_theta, eps)
            if lc.depth_attention == "layer" else None
        )
        dc = cfg.decoder
        if dc.ffn_experts > 0:
            if cfg.ffn_activation != "swiglu":
                raise ValueError("the routed decoder FFN is SwiGLU only; ffn_activation must be 'swiglu' with ffn_experts > 0")
            hidden = dc.ffn_expert_hidden or cfg.ffn_hidden // dc.ffn_top_k
            self.mlp = RoutedSwiGLU(d, dc.ffn_experts, dc.ffn_top_k, hidden)
        else:
            self.mlp = dense_ffn(d, cfg.ffn_hidden, cfg.ffn_activation)

    @property
    def attn(self) -> nn.Module:
        """The (first) mixer; the shared one when mixers are tied."""
        return self.mixers[0]

    def forward(
        self, u: torch.Tensor, slots: SlotContext, memory: KeySet | None, own_past, k: int,
        depth: int = 0, depth_stack: list | None = None,
    ) -> tuple[torch.Tensor, KeySet | GDNState]:
        """u: (B, C, d) slot states at loop k. own_past: this (loop, layer)'s cache from earlier
        chunks (KeySet for attention, GDNState for GDN). depth / depth_stack: this layer
        application's depth index and the position's depth stack (depth attention only).
        Returns the new slot states and the updated cache entry for this (loop, layer)."""
        mixer = self.mixers[k if len(self.mixers) > 1 else 0]
        h = self.attn_in_norm(u)
        if self.kind == "memory_attention":
            q, gate = mixer.queries(h, slots.pos)
            kk, vv = mixer.keys_values(h, slots.pos)
            own_now = KeySet(k=kk, v=vv, idx=slots.idx, doc=slots.doc, ok=slots.valid)
            keys = KeySet.cat(memory, own_past, own_now)
            C = u.shape[1]
            if mixer.cfg.backend != "reference":
                from ced.model.fast_attention import attend_fast

                # The same mask, built from the metadata; own_now is the last block of keys.
                o = attend_fast(q, keys, slots.idx, slots.doc, len(keys) - C, backend=mixer.cfg.backend)
            else:
                allow = allowed(keys, slots.idx, slots.doc)
                # Every slot may attend to itself. For valid slots this is already allowed; for empty
                # slots it is the only allowed key, which keeps their softmax rows finite. Empty slots'
                # outputs are never used and never seen by valid slots (their ok flag is False).
                self_mask = torch.eye(C, dtype=torch.bool, device=u.device).expand(u.shape[0], C, C)
                allow = torch.cat([allow[:, :, : len(keys) - C], allow[:, :, len(keys) - C :] | self_mask], dim=2)
                o = attend(q, keys, allow)
            mixed = mixer.output(o, gate)
            new_cache = KeySet.cat(own_past, own_now)
        else:
            mixed, new_cache = mixer(h, slot_sequence_context(slots, own_past), own_past, valid=slots.valid)
        if self.depth_attention is not None:
            # Parallel residual branch next to the mixer (Dreamer Eq. 5), from the same normed input.
            u = u + self.residual_scale * self.attn_out_norm(mixed) + self.residual_scale * self.depth_attention(h, depth, depth_stack)
        else:
            u = u + self.residual_scale * self.attn_out_norm(mixed)
        x = self.mlp_in_norm(u)
        ffn = self.mlp(x, valid=slots.valid) if isinstance(self.mlp, RoutedSwiGLU) else self.mlp(x)
        u = u + self.residual_scale * self.mlp_out_norm(ffn)
        return u, new_cache


class DepthAttention(nn.Module):
    """Depth attention (Dreamer, arXiv 2601.21582, section 3.4), one head, per position.

    Every decoder layer application j = k * L + l (loop k, layer l) projects its pre-normed input to
    a query, key and value, appends the key and value to the position's depth stack, and attends
    over the stack (entries 0..j, the current one included). RoPE runs over the depth index. The
    result is added as a parallel residual branch next to the token mixer (the paper's Eq. 5).

    Weights are shared by all loops (the paper uses depth-routed experts; with 2 loops we tie them,
    which also keeps LoopSpec's batched verification simple). The output projection starts at zero,
    so the arm equals the base model at initialisation.
    The stack lives only for the current step of a position: no cache across positions.
    """

    def __init__(self, d_model: int, head_dim: int, rope_theta: float, norm_eps: float):
        super().__init__()
        self.head_dim, self.rope_theta = head_dim, rope_theta
        self.qkv = nn.Linear(d_model, 3 * head_dim, bias=False)
        self.q_norm = RMSNorm(head_dim, norm_eps)
        self.k_norm = RMSNorm(head_dim, norm_eps)
        self.out = nn.Linear(head_dim, d_model, bias=False)
        self.zero_output = True  # init_weights re-initialises Linear layers; the model zeroes `out` after

    def forward(self, h: torch.Tensor, depth: int, stack: list) -> torch.Tensor:
        """h: (B, C, d) the layer's pre-normed input. stack: list of (k, v), each (B, C, head_dim),
        for earlier depths of this step; it is extended in place with this depth's entry."""
        B, C, _ = h.shape
        q, k, v = self.qkv(h).split(self.head_dim, dim=-1)
        pos = torch.full((B, C), depth, dtype=torch.long, device=h.device)
        q = apply_rope(self.q_norm(q)[:, :, None], pos, 1.0, self.rope_theta)[:, :, 0]
        k = apply_rope(self.k_norm(k)[:, :, None], pos, 1.0, self.rope_theta)[:, :, 0]
        stack.append((k, v))
        keys = torch.stack([kv[0] for kv in stack], dim=2)  # (B, C, D, hd)
        values = torch.stack([kv[1] for kv in stack], dim=2)
        dtype = torch.promote_types(q.dtype, torch.float32)
        scores = torch.einsum("bcx,bcdx->bcd", q.to(dtype), keys.to(dtype)) * self.head_dim**-0.5
        mixed = torch.einsum("bcd,bcdx->bcx", torch.softmax(scores, dim=-1), values.to(dtype)).to(h.dtype)
        return self.out(mixed)
