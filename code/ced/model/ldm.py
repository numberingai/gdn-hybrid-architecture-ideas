"""Linear Depth Memory with the delta-rule write: the variant of the architecture arms (EncoderConfig.attn_res
"ldm").

The sources (read 2026-10-16): the paper, arXiv 2610.18809v1 ("Depth Memory", Gonzalez-Martinez, Nazari, Rusch, Liu),
section 3.1 (the Read / Write framework), Eq. 3 and section 3.2 ("Efficient parameterization": the address networks,
the gates, the read gain); and its released code, github.com/davidgonmar/depth-memory at 9a6865feeb43 (MIT):
pretrain/OLMo-core/src/olmo_core/nn/ldm/backends/reference.py (write, read, run_block, run_segment, _initial_state,
_weighted_observation), components.py (AddressProjector, ReadParameters, WriteParameters, the gates, the forgetting
factor and its initialisation) and paper-experiments/ablations/configs/400m/delta_rule.json (the paper's "(b) Delta"
row: recurrence delta_rule, everything else the main LDM configuration). The code's delta rule is the rule built here;
the paper's text gives no equation for it.

What is computed. A "layer" of the paper is one SUBLAYER here (a layer's mixer and its MLP are two). The base's
residual stream is kept as it is: x <- x + delta. Beside it each position carries a memory M of G groups, each a
(K, D) matrix (K = ldm_key_dim, D = d_model / G), as the code's memory "[..., K, D]" per group:

    the embedding output h0 (after embed_norm) is the first write:  M_0 = write(0, k(h0), v = h0)
    before every sublayer f (the mixer, then the MLP, of every layer):
        q, g = the read's address network of x       (q (G, K) L2-normalised per group, g = 2 sigmoid(.) (G,))
        r    = per group  M^T (gamma_g g_g q_g)       (gamma: the read gain, (G,), a parameter)
        delta = f(x + r)                              (our sublayer as it is: residual_scale out_norm(sub(in_norm(.))))
        x     = x + delta                             ("keep the separate residual stream": the read is not added to x)
        k, w = the write's address network of x       (the updated stream; k (G, K) L2-normalised, w = 2 sigmoid(.))
        M     = write(M, k, v = delta)                ("store delta_l")
    the final norm reads x + r_out, r_out the output's own read of M (query from x).

    write (the gated delta rule; reference.py `write`, recurrence "delta_rule"; flexible/README.md "Delta rule"):
        k_w = sqrt(w) k,  v_w = sqrt(w) v              (_weighted_observation; w floored at eps before the root)
        A   = lam M                                     (lam = sigmoid(logit) floored at eps; one per group and write)
        M'  = A + k_w (v_w - A^T k_w)^T                 (per group: the error of the decayed memory's prediction)
    so M' = lam (I - w k k^T) M + w k v^T, the delta rule with step w in (0, 2) and decay lam.

Address networks (components.py AddressProjector; section 3.2): RMSNorm (float32, a gain) -> Linear(d, hidden) ->
SiLU -> a head Linear(hidden, G K) (the address) and a gate head Linear(hidden, G). hidden = ldm_hidden, or
d_model // 32 when 0 (the paper's configs: 32 at d 1,024, 64 at d 2,048). The read's query comes from the stream
before the sublayer ("current_stream"), the write's key from the stream after it ("updated_stream").

Initialisation (components.py reset_parameters; the paper's Appendix E): the address networks' first Linear and
their address heads "in the same way as the backbone linear layers" (here N(0, init_std), the backbone's draw), biases
zero; the gate heads with zero weights and zero biases, so every gate is 2 sigmoid(0) = 1 at the start; the read
gains 1; the forgetting logits logit(0.99). Drawn from a generator of their own (init_own_parameters), never from the
global one, so every other tensor of the model starts at the value it has in the base of the same seed.

Precision: the memory, its writes and reads, the L2 norms, the gates and the forgetting factors in at least float32
with autocast off (reference.py: "The recurrence state, weighted observations, gains and forgetting factors are FP32");
the address networks run under the caller's autocast (the code: "the address networks run in the model's working
dtype"); the read is returned in the stream's dtype.

Not built: RLS ("port RLS only if the delta version wins"), the outer-product rule, static addresses, the windowed
writes and the state unwinding of the paper's fast backends (an execution choice; here activation checkpointing per
layer, TrainConfig.activation_checkpoint "layer", keeps the memory a layer's call takes instead).

Two implementations of the same rule:
- reference_stack: the code's reference.py line for line (_initial_state, run_block's read, sublayer, write, read,
  sublayer, write, run_segment's final read; `write` and `read` with einsum, every M kept), eager. The ground truth.
- the fast path, which every forward runs (training, evaluation, scoring, prefill / extend): EncoderLayer.forward with
  `depth` [M] runs layer_forward, the layer's two sites inside the layer's own call (so a compiled layer compiles them),
  the first layer also the embedding write, the last also the output's read; LdmWalk is the stack.
The memory is per position: nothing in it mixes positions, so causality and the cache of a chunk are the layers' own;
a cached chunk carries nothing of M (each position's memory is built anew through the stack).
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ced.model import act_checkpoint
from ced.model.norms import RMSNorm, compute_dtype

MODE = "ldm"
LAMBDA_INIT = 0.99  # the forgetting factors' start (components.py forgetting_logit_init; the configs' rls.lambda_init)
EPS = 1e-6  # the floor of the L2 norms, the write weights and the forgetting factors (the configs' quality.eps)
# The generator of the depth mixers' own tensors (init_own_parameters): seeded with torch.initial_seed() + this, the
# pattern of ASEntmax's scale heads (ced/model/looped_ced.py init_asentmax_parameters).
OWN_SEED_OFFSET = 0x1D3_2610_18809


def address_hidden(d: int, value: int) -> int:
    """The hidden width of an address network: the configured value, or d // 32 (the paper's ratio) when 0."""
    return value if value else max(1, d // 32)


class OwnLinear(nn.Linear):
    """An nn.Linear the model's init_weights leaves alone (ced/model/looped_ced.py: own_init): a depth mixer's address
    network, drawn by init_own_parameters from a generator of its own."""

    own_init = True

    def reset_parameters(self) -> None:
        """At construction: zeros, and no draw from the global generator (nn.Linear's own reset draws a Kaiming
        uniform, which would move the global stream and with it every tensor the model's init_weights draws after)."""
        with torch.no_grad():
            self.weight.zero_()
            if self.bias is not None:
                self.bias.zero_()


class AddressNet(nn.Module):
    """One address network (components.py AddressProjector): RMSNorm -> inp: Linear(d, hidden) -> SiLU, then a head
    Linear(hidden, out) (None when out is 0) and a gate head Linear(hidden, gates) (None when gates is 0). Every Linear
    with a bias. zero_head: the head starts at zero weights and bias (the token-dependent query of R1-entry's readers,
    which must start as R1-entry's zero pseudo-query); otherwise it is drawn as the backbone's Linears are."""

    def __init__(self, d: int, hidden: int, out: int, gates: int, eps: float, zero_head: bool = False):
        super().__init__()
        self.norm = RMSNorm(d, eps)
        self.inp = OwnLinear(d, hidden)
        self.head = OwnLinear(hidden, out) if out else None
        self.gate = OwnLinear(hidden, gates) if gates else None
        self.zero_head = zero_head

    @torch.no_grad()
    def reset_own(self, g: torch.Generator, std: float) -> None:
        """The start (module docstring): inp and (unless zero_head) head N(0, std) from g, biases 0; the gate head 0."""
        def draw(t):
            t.copy_(torch.randn(t.shape, generator=g, dtype=torch.float32, device="cpu") * std)

        draw(self.inp.weight)
        self.inp.bias.zero_()
        if self.head is not None:
            if self.zero_head:
                self.head.weight.zero_()
            else:
                draw(self.head.weight)
            self.head.bias.zero_()
        if self.gate is not None:
            self.gate.weight.zero_()
            self.gate.bias.zero_()  # logit(initial_gain / 2) with initial_gain 1: gate 2 sigmoid(0) = 1

    def forward(self, x: torch.Tensor):
        hid = F.silu(self.inp(self.norm(x)))
        return (None if self.head is None else self.head(hid)), (None if self.gate is None else self.gate(hid))


def _no_autocast(x: torch.Tensor):
    return torch.autocast(x.device.type, enabled=False)


def state_dtype(x: torch.Tensor) -> torch.dtype:
    """The memory's dtype: float32, or float64 for a float64 model (reference.py _state_dtype)."""
    return compute_dtype(x)


class LdmRead(nn.Module):
    """One read of the memory (components.py ReadParameters): the address network (a (G K) address head and a (G,) gate
    head) and the read gain gamma (G,), ones at the start."""

    def __init__(self, d: int, groups: int, key_dim: int, hidden: int, eps: float):
        super().__init__()
        self.groups, self.key_dim = groups, key_dim
        self.net = AddressNet(d, hidden, groups * key_dim, groups, eps)
        self.read_gain = nn.Parameter(torch.ones(groups))

    def query(self, x: torch.Tensor) -> torch.Tensor:
        """gamma g q (B, L, G, K) in the memory's dtype: q L2-normalised per group, g = 2 sigmoid(gate logit)."""
        addr, gate = self.net(x)
        dtype = state_dtype(x)
        with _no_autocast(x):
            q = F.normalize(addr.to(dtype).unflatten(-1, (self.groups, self.key_dim)), dim=-1, eps=EPS)
            g = 2.0 * torch.sigmoid(gate.to(dtype))
            return q * (g * self.read_gain.to(dtype))[..., None]

    def read(self, M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """The read r (B, L, d) in x's dtype: per group M^T (gamma g q), M (B, L, G, K, D)."""
        q = self.query(x)
        with _no_autocast(x):
            r = (M * q[..., None]).sum(-2)  # (B, L, G, D)
        return r.flatten(-2).to(x.dtype)


class LdmWrite(nn.Module):
    """One write into the memory (components.py WriteParameters / RLSWriteSite): the address network (a (G K) key head
    and a (G,) write-gate head) and the forgetting logit (G,), logit(0.99) at the start (ForgettingParameters).
    first: the embedding's write into the empty memory, which has no forgetting logit: under the delta rule the factor
    multiplies a memory of zeros, so its value never reaches the model (the code's root_logit, ForgettingParameters,
    which RLS reads through the covariance, would be a tensor without a gradient here; not built)."""

    def __init__(self, d: int, groups: int, key_dim: int, hidden: int, eps: float, first: bool = False):
        super().__init__()
        self.groups, self.key_dim = groups, key_dim
        self.net = AddressNet(d, hidden, groups * key_dim, groups, eps)
        self.lam_logit = None if first else nn.Parameter(torch.full((groups,), math.log(LAMBDA_INIT / (1.0 - LAMBDA_INIT))))

    def observation(self, x: torch.Tensor, v: torch.Tensor):
        """(k_w (B, L, G, K), v_w (B, L, G, D), lam (G,)) in the memory's dtype: k from x (the updated stream) L2-
        normalised per group, w = 2 sigmoid(gate logit) floored at eps, k_w = sqrt(w) k, v_w = sqrt(w) v
        (_weighted_observation), lam = sigmoid(logit) floored at eps (forgetting_from_logit)."""
        addr, gate = self.net(x)
        dtype = state_dtype(x)
        with _no_autocast(x):
            k = F.normalize(addr.to(dtype).unflatten(-1, (self.groups, self.key_dim)), dim=-1, eps=EPS)
            s = (2.0 * torch.sigmoid(gate.to(dtype))).clamp_min(EPS).sqrt()
            kw = k * s[..., None]
            vw = v.to(dtype).unflatten(-1, (self.groups, -1)) * s[..., None]
            lam = None if self.lam_logit is None else torch.sigmoid(self.lam_logit.to(dtype)).clamp_min(EPS)
        return kw, vw, lam

    def first(self, h: torch.Tensor) -> torch.Tensor:
        """The embedding write into the empty memory (reference.py _initial_state): M_0 = k_w v_w^T per group (the
        rule on M = 0: A = 0, A^T k_w = 0), (B, L, G, K, D), key and value both the embedding output h."""
        kw, vw, _ = self.observation(h, h)
        with _no_autocast(h):
            return kw[..., None] * vw[..., None, :]

    def write(self, M: torch.Tensor, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """M' = A + k_w (v_w - A^T k_w)^T, A = lam M (module docstring), per group."""
        kw, vw, lam = self.observation(x, v)
        with _no_autocast(x):
            A = M * lam[:, None, None]
            p = (A * kw[..., None]).sum(-2)  # A^T k_w: (B, L, G, D)
            return A + kw[..., None] * (vw - p)[..., None, :]


class LdmLayer(nn.Module):
    """What one encoder layer holds of the memory (attached as `attn_res`): the read and the write of its mixer's site
    and of its MLP's site; in layer 0 also the embedding write (embed_write), in the last layer also the output's
    read (out_read). The state-dict names: encoder.layers.<i>.attn_res.{embed_write, mixer_read, mixer_write,
    mlp_read, mlp_write, out_read}.*"""

    def __init__(self, d: int, groups: int, key_dim: int, hidden: int, eps: float, first: bool, last: bool):
        super().__init__()
        args = (d, groups, key_dim, hidden, eps)
        self.embed_write = LdmWrite(*args, first=True) if first else None
        self.mixer_read = LdmRead(*args)
        self.mixer_write = LdmWrite(*args)
        self.mlp_read = LdmRead(*args)
        self.mlp_write = LdmWrite(*args)
        self.out_read = LdmRead(*args) if last else None


def layer_forward(layer, x: torch.Tensor, ctx, cache, depth: list):
    """EncoderLayer.forward under "ldm" (the fast path; what a compiled layer traces). x: the residual stream entering
    the layer (in layer 0 the embedding output); depth: [M], the memory after the writes before this layer ([] in
    layer 0, which first writes the embedding). Returns ((the stream, the memory) after the layer's two sites, the
    layer's new cache entry); the stack's last layer returns the stream plus the output's read in their place, which
    the final norm reads."""
    res = layer.attn_res
    if res.embed_write is not None:
        assert not depth, "layer 0 writes the embedding into the empty memory: it takes no memory"
        M = res.embed_write.first(x)
    else:
        (M,) = depth
    h = x + res.mixer_read.read(M, x)
    mixed, new_cache = layer.mixer(layer.mixer_in_norm(h), ctx, cache)
    delta = layer.residual_scale * layer.mixer_out_norm(mixed)
    x = x + delta
    M = res.mixer_write.write(M, x, delta)
    h = x + res.mlp_read.read(M, x)
    delta = layer.residual_scale * layer.mlp_out_norm(layer.mlp(layer.mlp_in_norm(h)))
    x = x + delta
    M = res.mlp_write.write(M, x, delta)
    if res.out_read is not None:
        return x + res.out_read.read(M, x), new_cache
    return (x, M), new_cache


class _Call:
    """One checkpointed layer call under activation_checkpoint "layer" (LdmWalk.run with ckpt): the memory M a top-level
    argument of the checkpoint beside the stream (None in layer 0). ckpt_layer: the layer (the memory trace of a
    training step books the call under the layer's name)."""

    __slots__ = ("ckpt_layer",)

    def __init__(self, layer):
        self.ckpt_layer = layer

    def __call__(self, x, M, ctx, cache):
        return self.ckpt_layer(x, ctx, cache, depth=[] if M is None else [M])


class LdmWalk:
    """One forward of the stack under "ldm", layer by layer, on the fast path (BlockAttnRes.stack, the GDN
    diagnostics). x: the residual stream (the embedding output before layer 0; after the last layer the stream plus
    the output's read, which the final norm reads). M: the memory after the writes so far (None before layer 0).
    mixer_input(i): what layer i's mixer sublayer reads, x + its read of M (a second, eager copy under no_grad)."""

    def __init__(self, enc, h: torch.Tensor):
        self.enc = enc
        self.x = h
        self.M = None

    @torch.no_grad()
    def mixer_input(self, i: int) -> torch.Tensor:
        res = self.enc.layers[i].attn_res
        x = self.x.detach()
        M = res.embed_write.first(x) if self.M is None else self.M.detach()
        return x + res.mixer_read.read(M, x)

    def run(self, i: int, ctx, cache, ckpt: bool = False):
        """Layer i, compiled if the layers are. ckpt (activation_checkpoint "layer" in a training forward): the same
        call under its own checkpoint, which keeps the stream and the memory the call takes (one (B, L, d) and one
        (B, L, G, K, D) tensor a layer; a later layer's recomputation needs nothing else)."""
        layer = self.enc.layers[i]
        if ckpt:
            out, new_cache = act_checkpoint.run_kept(_Call(layer), (), self.x, self.M, ctx, cache)
        else:
            out, new_cache = layer(self.x, ctx, cache, depth=[] if self.M is None else [self.M])
        if layer.attn_res.out_read is not None:
            self.x, self.M = out, None
        else:
            self.x, self.M = out
        return new_cache

    def copy(self):
        return copy.copy(self)


# ------------------------------------------------------------------------------------------
# the reference: the released code's reference.py, line for line


def ref_write(memory: torch.Tensor, key: torch.Tensor, value: torch.Tensor, forgetting_factor: torch.Tensor | None) -> torch.Tensor:
    """reference.py `write` with recurrence "delta_rule": memory [..., K, D] per group, key [..., K], value [..., D],
    forgetting_factor [G] (broadcast over the groups' axis); None for the embedding's write into the empty memory,
    whose factor multiplies zeros (LdmWrite: not built)."""
    gain = key
    if forgetting_factor is not None:
        memory = memory * forgetting_factor[..., None, None]
    prediction = torch.einsum("...kd,...k->...d", memory, key)
    residual = value - prediction
    return memory + gain.unsqueeze(-1) * residual.unsqueeze(-2)


def ref_read(memory: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """reference.py `read`: memory^T query per group."""
    return torch.einsum("...kd,...k->...d", memory, query)


def _ref_project(net: AddressNet, source: torch.Tensor, groups: int, key_dim: int):
    """WriteParameters.project / ReadParameters.project's address and gate (normalised in float32, components.py)."""
    addr, gate = net(source)
    dtype = state_dtype(source)
    with _no_autocast(source):
        address = F.normalize(addr.to(dtype).view(*source.shape[:-1], groups, key_dim), dim=-1, eps=EPS)
        g = 2.0 * torch.sigmoid(gate.to(dtype))
    return address, g


def _ref_segment_write(memory, site: LdmWrite, updated_stream, branch_delta):
    """reference.py _segment_write (key source "updated_stream", value source "branch_delta") and
    _weighted_observation."""
    G, K = site.groups, site.key_dim
    key, gate = _ref_project(site.net, updated_stream, G, K)
    dtype = state_dtype(updated_stream)
    with _no_autocast(updated_stream):
        value = branch_delta.to(dtype).view(*branch_delta.shape[:-1], G, branch_delta.shape[-1] // G)
        scale = gate.clamp_min(EPS).sqrt()
        key, value = key * scale.unsqueeze(-1), value * scale.unsqueeze(-1)
        lam = None if site.lam_logit is None else torch.sigmoid(site.lam_logit.to(dtype)).clamp_min(EPS)
        return ref_write(memory, key, value, lam)


def _ref_segment_read(memory, site: LdmRead, current_stream):
    """reference.py _segment_read (query source "current_stream", add_to the stream) with ReadParameters.project's
    gate and read gain."""
    G, K = site.groups, site.key_dim
    query, gate = _ref_project(site.net, current_stream, G, K)
    dtype = state_dtype(current_stream)
    with _no_autocast(current_stream):
        query = query * gate.unsqueeze(-1)
        query = query * site.read_gain.to(dtype).unsqueeze(-1)
        value = ref_read(memory, query).reshape(*current_stream.shape[:-1], -1)
    return current_stream + value.to(current_stream.dtype)


def reference_stack(enc, h: torch.Tensor, ctx, caches: list | None, record: list | None = None):
    """The reference under "ldm": run_segment of reference.py (the embedding write, every block's read, sublayer, write,
    read, sublayer, write, the final read), one sublayer after the other, eager, every memory state kept. Returns (the
    stream the final norm reads, the layers' new caches). record (a list): receives one dict per sublayer, in order,
    {"reader": "L<i>.mixer" | "L<i>.mlp", "layer": i, "input": what the sublayer's pre-norm reads, "output": delta,
    "memory": the memory the read used}, and a last one {"reader": "out", "input": the final stream, "memory": ...}."""
    caches = caches if caches is not None else [None] * len(enc.layers)
    layers = enc.layers
    first = layers[0].attn_res.embed_write
    dtype = state_dtype(h)
    G, K, D = first.groups, first.key_dim, h.shape[-1] // first.groups
    memory = torch.zeros(*h.shape[:-1], G, K, D, dtype=dtype, device=h.device)
    memory = _ref_segment_write(memory, first, h, h)  # _initial_state: the embedding is key source and value
    hidden, new_caches = h, []
    for i, (layer, cache) in enumerate(zip(layers, caches)):
        res = layer.attn_res
        attention_input = _ref_segment_read(memory, res.mixer_read, hidden)
        if record is not None:
            record.append({"reader": f"L{i}.mixer", "layer": i, "input": attention_input, "memory": memory})
        mixed, new_cache = layer.mixer(layer.mixer_in_norm(attention_input), ctx, cache)
        attention_update = layer.residual_scale * layer.mixer_out_norm(mixed)
        attention_stream = hidden + attention_update
        memory = _ref_segment_write(memory, res.mixer_write, attention_stream, attention_update)
        mlp_input = _ref_segment_read(memory, res.mlp_read, attention_stream)
        mlp_update = layer.residual_scale * layer.mlp_out_norm(layer.mlp(layer.mlp_in_norm(mlp_input)))
        if record is not None:
            record[-1]["output"] = attention_update
            record.append({"reader": f"L{i}.mlp", "layer": i, "input": mlp_input, "memory": memory, "output": mlp_update})
        output = attention_stream + mlp_update
        memory = _ref_segment_write(memory, res.mlp_write, output, mlp_update)
        hidden = output
        new_caches.append(new_cache)
    out = _ref_segment_read(memory, layers[-1].attn_res.out_read, hidden)
    if record is not None:
        record.append({"reader": "out", "layer": None, "input": hidden, "memory": memory, "output": None})
    return out, new_caches


# ------------------------------------------------------------------------------------------
# building, the start, the counts


def build(cfg, layers) -> None:
    """Attach an LdmLayer to every layer (as its last submodule; every other parameter keeps its name, shape and order).
    Draws no random number here: init_own_parameters starts the tensors after the model's init_weights."""
    e, d = cfg.encoder, cfg.d_model
    hidden = address_hidden(d, e.ldm_hidden)
    n = len(layers)
    for i, layer in enumerate(layers):
        layer.attn_res = LdmLayer(d, e.ldm_groups, e.ldm_key_dim, hidden, cfg.norm_eps, first=i == 0, last=i == n - 1)


def own_generator() -> torch.Generator:
    """The generator of the depth mixers' own tensors: CPU, seeded with torch.initial_seed() + OWN_SEED_OFFSET (the run's
    seed: the trainer seeds the global generator before it builds the model). The global generator is never drawn."""
    return torch.Generator().manual_seed((torch.initial_seed() + OWN_SEED_OFFSET) % (1 << 63))


def init_own_parameters(model: nn.Module, std: float) -> int:
    """Start every AddressNet of `model`, in module order, from own_generator() (AddressNet.reset_own). Returns how
    many were started. Called by the model's build after init_weights, and only for a model that has one."""
    nets = [m for m in model.modules() if isinstance(m, AddressNet)]
    if not nets:
        return 0
    g = own_generator()
    for m in nets:
        m.reset_own(g, std)
    return len(nets)


def sites(n_layers: int) -> tuple[int, int]:
    """(reads, writes) of the memory in a stack of n_layers: 2 n + 1 each (every sublayer's read and write, the
    output's read, the embedding's write)."""
    return 2 * n_layers + 1, 2 * n_layers + 1


def parameter_count(d: int, n_layers: int, groups: int, key_dim: int, hidden: int) -> dict:
    """The tensors "ldm" adds, exact from the shapes: per address network d (norm gain) + (d + 1) hidden + (hidden + 1)
    G K + (hidden + 1) G; per read the read gain G; per write but the embedding's the forgetting logit G."""
    net = d + (d + 1) * hidden + (hidden + 1) * groups * key_dim + (hidden + 1) * groups
    reads, writes = sites(n_layers)
    return {"address_network": net, "read": net + groups, "write": net + groups, "reads": reads, "writes": writes,
            "total": reads * (net + groups) + writes * net + (writes - 1) * groups}
