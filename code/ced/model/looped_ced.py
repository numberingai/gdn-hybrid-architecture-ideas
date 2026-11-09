"""Looped-CED: a causal encoder that every token passes through once, and a shared decoder block
that loops T times only at decoder positions (positions whose next token the model generates).

Data flow:

    h_enc = Encoder(embed(x))                          every position
    aux logits = head(RMSNorm_aux(h_enc))              every position (loss only at context positions)
    memory KV  = MemoryKV(h_enc)                       offered by context positions
    e = RMSNorm_inj(h_enc[slots]);  Y = HC.start(e)    decoder slots only
    for k in 1..T:
        z_in = HC.read(Y, k)
        u = inject(z_in, e) + loop_emb[k]
        s = DecBlock(u)        (4 layers; loop-k own KVs of decoder positions)
        Y = HC.write(Y, z_in, s, k)
    decoder logits = head(RMSNorm_dec(HC.finish(Y)))   decoder slots, scattered back

The same `forward` serves training (whole rows, fixed decoder capacity), prefill (returns the
caches) and cached generation (one token at a time with the caches).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ced.config import ModelConfig
from ced.model.attention import KeySet
from ced.model.capacity import Slots, decoder_capacity, gather_slots, scatter_slots, select_slots
from ced.model.decoder import DecoderLayer, MemoryKV, SlotContext
from ced.model.encoder import Encoder
from ced.model.loop import build_hyper_connections, build_injection
from ced.model.mlp import RoutedSwiGLU
from ced.model.norms import RMSNorm, compute_dtype
from ced.model.scaling import residual_scales
from ced.model.sequence import SeqContext, SeqState, sequence_context


@dataclass
class LoopedCEDCache:
    """Everything later chunks of the same rows need.

    seq:     index / document bookkeeping
    encoder: per encoder layer, a GDNState or the global layer's KeySet
    memory:  per memory projection, memory KVs of all past positions (entries of decoder positions
             have ok=False)
    own:     own[k][l] = loop k, decoder layer l: a KeySet of past decoder slots' own KVs
             (attention layers) or the GDNState over past decoder slots (GDN layers)
    T:       loop count used to build the cache; later chunks must use the same T
    """

    seq: SeqState
    encoder: list
    memory: list[KeySet]
    own: list[list]
    T: int


@dataclass
class ModelOutput:
    logits: torch.Tensor | None  # (B, L, V): decoder head at decoder positions, aux head elsewhere
    kept: torch.Tensor  # (B, L) bool: positions that ran the decoder
    n_dropped: torch.Tensor  # (B,) decoder positions dropped at capacity
    cache: LoopedCEDCache | None = None
    # Readouts of intermediate loops at the decoder slots: draft_logits[k-1] = after loop k < T.
    draft_logits: list | None = None
    slot_index: torch.Tensor | None = None  # (B, C) position of each slot (L for empty)
    slot_valid: torch.Tensor | None = None  # (B, C)
    aux_loss: torch.Tensor | None = None  # routed-FFN load balance, already weighted
    # With return_hidden=True, logits (and draft_logits) are None and the normed pre-head states
    # are returned instead; the loss then computes the tied head chunk by chunk
    # (ced/train/chunked_loss.py). hidden_aux: (B, L, d) at every position (the control's only
    # head); hidden_dec: (B, C, d) at the slots; hidden_drafts[k-1]: (B, C, d) after loop k < T.
    hidden_aux: torch.Tensor | None = None
    hidden_dec: torch.Tensor | None = None
    hidden_drafts: list | None = None
    # The Copy-Paste pointer's own inputs with return_hidden (model.cp_head.pointer_input "mix" / "shared";
    # ced/model/cp_head.py PointerInputs): every CP scoring function takes them as `pointer`. None in every other model.
    cp_pointer: object | None = None


@dataclass
class EncodedChunk:
    """Encoder-side results of a chunk, everything the decoder loops need."""

    ctx: SeqContext
    seq_state: SeqState
    encoder_caches: list
    aux_hidden: torch.Tensor  # (B, L, d): RMSNorm_aux(h_enc), the aux head's input
    slots: Slots
    slot_ctx: SlotContext
    memory: list[KeySet]  # memory KVs of earlier chunks and this one
    e: torch.Tensor  # (B, C, d) RMSNorm(h_enc) at the slots: z_0 and the injection input


@dataclass
class LoopState:
    """Decoder state of a chunk's slots between loops."""

    Y: torch.Tensor  # hyper-connection streams (B, C, n, d), or the single state (B, C, d)
    loops_done: int
    depth_stack: list | None = None  # per decoder layer: list of (k, v) of earlier depths (depth attention)


def lm_head(h: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Logits h @ weight^T with weight (V, d) the model's head_weight(), computed in at least
    float32 (FP32 head)."""
    dtype = compute_dtype(h)
    return F.linear(h.to(dtype), weight.to(dtype))


def tied_head(h: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
    """LM head tied to the embedding, computed in at least float32 (FP32 head)."""
    return lm_head(h, embedding.weight)


def output_embedding(cfg: ModelConfig) -> nn.Embedding | None:
    """The separate output embedding of an untied model (vocab x d_model), or None when tied. It
    is an nn.Embedding so that it gets the embedding's initialisation and optimizer treatment
    (its parameter name ends in "embed.weight")."""
    return None if cfg.tie_embeddings else nn.Embedding(cfg.vocab_size, cfg.d_model)


def init_weights(model: nn.Module, std: float) -> None:
    """Normal(0, std) for every Linear and Embedding weight. Norm gains, GDN gates, the GDN conv,
    Parcae, loop embeddings and hyper-connection parameters keep their own initialisation."""
    # ASEntmax's scale heads (AttentionConfig.normalizer "asentmax") are skipped here and drawn afterwards from a
    # generator of their own (init_asentmax_parameters): the global stream is used exactly as by the softmax model
    # of the same seed, so every shared parameter starts equal to the softmax baseline's.
    entmax_mods = [m for m in model.modules()
                   if getattr(m, "beta_proj", None) is not None and hasattr(m, "reset_asentmax_parameters")]
    skip = {id(lin) for m in entmax_mods for lin in (m.beta_proj, m.gamma_proj)}
    # The depth mixers' address networks (the architecture arms of 2026-10-16; ced/model/ldm.py OwnLinear, own_init)
    # are left alone here too and started by the model's build from a generator of their own
    # (ldm.init_own_parameters), so the global stream is used exactly as by the model without them.
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)) and id(module) not in skip and not getattr(module, "own_init", False):
            nn.init.normal_(module.weight, mean=0.0, std=std)
    if entmax_mods:
        init_asentmax_parameters(entmax_mods)


ASENTMAX_SEED_OFFSET = 0xA5E_2506_16640  # the scale heads' own stream: seed = torch.initial_seed() + this


def init_asentmax_parameters(modules: list) -> None:
    """beta_proj / gamma_proj of every asentmax GatedAttention in `modules` (model order): N(0, 0.02) and bias 0
    (GatedAttention.reset_asentmax_parameters), from one CPU generator seeded with torch.initial_seed() (the run's
    seed: the trainer calls torch.manual_seed(seed) before building) plus ASENTMAX_SEED_OFFSET. The global RNG is
    not used."""
    g = torch.Generator().manual_seed((torch.initial_seed() + ASENTMAX_SEED_OFFSET) % (1 << 63))
    for m in modules:
        m.reset_asentmax_parameters(g)


class LoopedCED(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.arch == "looped_ced"
        self.cfg = cfg
        d = cfg.d_model
        eps_enc, eps_dec = residual_scales(cfg)

        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.embed_norm = RMSNorm(d, cfg.norm_eps) if cfg.embed_norm else nn.Identity()
        self.encoder = Encoder(cfg, eps_enc)
        self.aux_norm = RMSNorm(d, cfg.norm_eps)

        kinds = cfg.decoder.layer_kinds()
        memory_layers = [l for l, kind in enumerate(kinds) if kind == "memory_attention"]
        assert memory_layers, "the decoder needs at least one memory-attention layer to read the context"
        # Memory KV: one projection shared by all memory-attention layers, or one per such layer.
        self.memory_slot = {l: (i if cfg.decoder.memory_kv_per_layer else 0) for i, l in enumerate(memory_layers)}
        n_memory = len(memory_layers) if cfg.decoder.memory_kv_per_layer else 1
        self.memory_kv = nn.ModuleList(MemoryKV(cfg) for _ in range(n_memory))
        self.inject_norm = RMSNorm(d, cfg.norm_eps)
        self.injection = build_injection(d, cfg.loop)
        self.loop_embedding = nn.Parameter(torch.zeros(cfg.loop.t_max, d)) if cfg.loop.loop_embedding else None
        self.hyper_connections = build_hyper_connections(d, cfg.loop)
        self.decoder_layers = nn.ModuleList(DecoderLayer(cfg, kind, eps_dec) for kind in kinds)
        self.decoder_norm = RMSNorm(d, cfg.norm_eps)
        # Readouts after intermediate loops (loop k < T) through the same tied head: the LoopSpec draft.
        self.readout_norms = nn.ModuleList(RMSNorm(d, cfg.norm_eps) for _ in range(cfg.loop.t_max - 1))
        self.out_embed = output_embedding(cfg)

        init_weights(self, cfg.init_std)
        for layer in self.decoder_layers:
            if layer.depth_attention is not None:  # the arm starts equal to the base model
                nn.init.zeros_(layer.depth_attention.out.weight)

    # ----------------------------------------------------------------------------------------

    def head_weight(self) -> torch.Tensor:
        """The (V, d) matrix of every LM head: the input embedding when tied, else the output embedding."""
        return self.embed.weight if self.out_embed is None else self.out_embed.weight

    def capacity(self, length: int) -> int:
        return decoder_capacity(length, self.cfg.decoder.capacity_fraction, self.cfg.decoder.capacity_multiple)

    def encode(
        self, tokens, doc_ids, dec_mask, capacity: int, cache: LoopedCEDCache | None, varlen: dict | None = None
    ) -> EncodedChunk:
        """Encoder over every position of the chunk, aux head input, slot selection, memory KV.

        varlen (training only, optional): precomputed segment boundaries for fused kernels,
        {"enc": cu_seqlens over the flattened (B * L) positions, "dec": over the (B * C) slots};
        see ced/train/varlen.py. They are computed on the host by the data pipeline, so the
        training step itself needs no nonzero / boolean indexing.
        """
        ctx, seq_state = sequence_context(doc_ids, cache.seq if cache else None)
        if varlen is not None:
            ctx.cu_seqlens, ctx.cu_seqlens_cpu = varlen["enc"], varlen.get("enc_cpu")
        h = self.embed_norm(self.embed(tokens))
        h_enc, encoder_caches = self.encoder(h, ctx, cache.encoder if cache else None)
        aux_hidden = self.aux_norm(h_enc)

        slots = select_slots(dec_mask, capacity)
        is_context = ~slots.kept
        memory = []
        for i, mem in enumerate(self.memory_kv):
            now = mem(h_enc, ctx, ok=is_context)
            memory.append(KeySet.cat(cache.memory[i] if cache else None, now))
        slot_ctx = SlotContext(
            idx=gather_slots(ctx.idx, slots.index, fill=-1),
            doc=gather_slots(ctx.doc, slots.index, fill=-1),
            pos=gather_slots(ctx.pos, slots.index, fill=0),
            valid=slots.valid,
            cu_seqlens=varlen["dec"] if varlen is not None else None,
            cu_seqlens_cpu=varlen.get("dec_cpu") if varlen is not None else None,
        )
        e = self.inject_norm(gather_slots(h_enc, slots.index))  # zeros at empty slots
        return EncodedChunk(ctx, seq_state, encoder_caches, aux_hidden, slots, slot_ctx, memory, e)

    def loop_init(self, chunk: EncodedChunk) -> LoopState:
        stack = [[] for _ in self.decoder_layers] if self.cfg.loop.depth_attention == "layer" else None
        return LoopState(Y=self.hyper_connections.start(chunk.e), loops_done=0, depth_stack=stack)

    def run_loop(self, chunk: EncodedChunk, state: LoopState, own_past: list | None) -> tuple[LoopState, list]:
        """One pass of the shared decoder block (loop index k = state.loops_done) over the slots.

        own_past[l] is this loop's cache entry of layer l from earlier chunks (or None). Returns the
        new state and the updated cache entries of this loop.
        """
        k, hc = state.loops_done, self.hyper_connections
        z_in = hc.read(state.Y, k)
        u = self.injection(z_in, chunk.e)
        if self.loop_embedding is not None:
            u = u + self.loop_embedding[k]
        own_new, n_layers = [], len(self.decoder_layers)
        # Depth stacks are copied, never mutated: a LoopState can be reused (LoopSpec verification).
        stacks = [list(s) for s in state.depth_stack] if state.depth_stack is not None else None
        for l, layer in enumerate(self.decoder_layers):
            memory = chunk.memory[self.memory_slot[l]] if layer.kind == "memory_attention" else None
            u, entry = layer(u, chunk.slot_ctx, memory, own_past[l] if own_past else None, k,
                             depth=k * n_layers + l, depth_stack=stacks[l] if stacks is not None else None)
            own_new.append(entry)
        return LoopState(Y=hc.write(state.Y, z_in, u, k), loops_done=k + 1, depth_stack=stacks), own_new

    def readout_hidden(self, state: LoopState, T: int) -> torch.Tensor:
        """The head input (B, C, d) after state.loops_done loops: the final readout's when all T
        loops are done, otherwise the intermediate (draft) readout's."""
        z = self.hyper_connections.finish(state.Y)
        norm = self.decoder_norm if state.loops_done == T else self.readout_norms[state.loops_done - 1]
        return norm(z)

    def readout(self, state: LoopState, T: int) -> torch.Tensor:
        """Slot logits (B, C, V) from the state after state.loops_done loops (see readout_hidden)."""
        return lm_head(self.readout_hidden(state, T), self.head_weight())

    def forward(
        self,
        tokens: torch.Tensor,
        doc_ids: torch.Tensor,
        dec_mask: torch.Tensor,
        T: int | None = None,
        capacity: int | None = None,
        cache: LoopedCEDCache | None = None,
        return_cache: bool = False,
        return_drafts: bool = False,
        return_hidden: bool = False,
        varlen: dict | None = None,
    ) -> ModelOutput:
        """tokens, doc_ids: (B, L) long; dec_mask: (B, L) bool, True where the position's next
        token is generated by the model (a decoder position).

        capacity defaults to the training capacity C(L). Prefill and cached steps pass capacity=L
        so that no decoder position is dropped. return_drafts adds the intermediate-loop readouts.
        return_hidden returns head inputs instead of logits (training and evaluation at large vocab).
        varlen: optional precomputed segment boundaries for the fused GDN kernel (see encode).
        """
        B, L = tokens.shape
        T = self.cfg.loop.t_eval if T is None else T
        assert 1 <= T <= self.cfg.loop.t_max, f"T={T} outside 1..t_max={self.cfg.loop.t_max}"
        if cache is not None:
            assert cache.T == T, "a cache is only valid for the T it was built with"
        capacity = self.capacity(L) if capacity is None else capacity

        routed = [m.mlp for m in self.decoder_layers if isinstance(m.mlp, RoutedSwiGLU)]
        for mlp in routed:
            mlp.reset_stats()
        chunk = self.encode(tokens, doc_ids, dec_mask, capacity, cache, varlen)
        state = self.loop_init(chunk)
        own_caches, drafts = [], []
        head = self.readout_hidden if return_hidden else self.readout
        for k in range(T):
            state, own_k = self.run_loop(chunk, state, cache.own[k] if cache else None)
            own_caches.append(own_k)
            if return_drafts and k < T - 1:
                drafts.append(head(state, T))
        final = head(state, T)
        if return_hidden:
            logits = None
        else:
            logits = scatter_slots(lm_head(chunk.aux_hidden, self.head_weight()), chunk.slots.index, final)

        aux_loss = None
        if routed:  # pooled over all loops of this forward
            dc = self.cfg.decoder
            aux_loss = dc.moe_balance_weight * torch.stack([m.balance_loss() for m in routed]).mean()
            if dc.router_z_weight > 0:
                aux_loss = aux_loss + dc.router_z_weight * torch.stack([m.z_loss() for m in routed]).mean()
        new_cache = None
        if return_cache:
            new_cache = LoopedCEDCache(seq=chunk.seq_state, encoder=chunk.encoder_caches, memory=chunk.memory,
                                       own=own_caches, T=T)
        return ModelOutput(
            logits=logits, kept=chunk.slots.kept, n_dropped=chunk.slots.n_dropped, cache=new_cache,
            draft_logits=drafts if return_drafts and not return_hidden else None, slot_index=chunk.slots.index,
            slot_valid=chunk.slots.valid, aux_loss=aux_loss,
            hidden_aux=chunk.aux_hidden if return_hidden else None,
            hidden_dec=final if return_hidden else None,
            hidden_drafts=drafts if return_drafts and return_hidden else None,
        )

    # ----------------------------------------------------------------------------------------
    # inference helpers

    @torch.no_grad()
    def prefill(self, tokens, doc_ids, dec_mask, T=None) -> ModelOutput:
        """Teacher-forced pass over a prompt that returns the caches. Earlier assistant turns in the
        prompt are decoder positions, so they run the decoder again and their own per-loop KVs are
        in the cache."""
        return self.forward(tokens, doc_ids, dec_mask, T=T, capacity=tokens.shape[1], return_cache=True)

    @torch.no_grad()
    def extend(self, tokens, doc_ids, dec_mask, cache: LoopedCEDCache) -> ModelOutput:
        """Process the next chunk (often one token) of the same rows, using and extending the cache."""
        return self.forward(
            tokens, doc_ids, dec_mask, T=cache.T, capacity=tokens.shape[1], cache=cache, return_cache=True
        )

    @torch.no_grad()
    def generate(self, prompt, prompt_doc_ids, prompt_dec_mask, max_new_tokens: int, T=None, greedy=True, generator=None):
        """Generate continuation tokens for every row. Each fed-back generated token is a decoder
        position (its next token is generated too). The prompt's last position must be a decoder
        position (the assistant header).

        Returns (tokens (B, N), logits (B, N, V)) where logits[:, i] is the distribution tokens[:, i]
        was drawn from.
        """
        assert bool(prompt_dec_mask[:, -1].all()), "the prompt must end at a decoder position"
        out = self.prefill(prompt, prompt_doc_ids, prompt_dec_mask, T=T)
        if max_new_tokens == 0:
            B, V = prompt.shape[0], out.logits.shape[-1]
            return prompt.new_empty(B, 0), out.logits.new_empty(B, 0, V)
        doc_last = prompt_doc_ids[:, -1:]
        new_tokens, step_logits = [], []
        for _ in range(max_new_tokens):
            next_logits = out.logits[:, -1]
            if greedy:
                tok = next_logits.argmax(-1, keepdim=True)
            else:
                tok = torch.multinomial(torch.softmax(next_logits, -1), 1, generator=generator)
            new_tokens.append(tok)
            step_logits.append(next_logits)
            out = self.extend(tok, doc_last, torch.ones_like(tok, dtype=torch.bool), out.cache)
        return torch.cat(new_tokens, dim=1), torch.stack(step_logits, dim=1)
