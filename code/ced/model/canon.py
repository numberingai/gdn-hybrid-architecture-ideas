"""Canon layers (arXiv 2512.17351, Allen-Zhu, "Physics of Language Models: Part 4.1, Architecture Design and the
Magic of Canon Layers").

The source. "Canon layers compute weighted sums of nearby token representations" (abstract), implemented "as 1-d causal
convolution with kernel size 4", with a residual connection ("Canon(res)"):

    h'_t = h_t + conv1d([h_t, h_{t-1}, h_{t-2}, h_{t-3}])

Its released code (github.com/facebookresearch/PhysicsLM4, huggingface/canon_helper.py, create_canon / apply_canon)
builds the convolution depthwise (nn.Conv1d with groups = the hidden size: one weight per channel and tap, K * d in
all), without a bias and without an activation, the residual on, and keeps the last inputs as the generation cache. It
names four places for it; "Canon-A" is "Before the attention block [...] after RMSnorm".

What is built here: Canon-A on the input of the encoder's GLOBAL attention layers, EncoderConfig.canon_kernel = K
(ced/model/encoder.py EncoderLayer):

    h     = mixer_in_norm(x)                               the layer's normed input, (B, L, d)
    h'_t  = h_t + sum_{j=0..K-1} w[:, K-1-j] * h_{t-j}     per channel; only inputs of t's own document
    mixed = attention(h')                                   queries, keys, values and the output gate all read h'

Where it differs from the source, each on purpose:
- Documents. A packed row holds several documents, and a document's first tokens must not read the document before
  it. The taps follow the rule of the GDN layers' short convolution and are that code (gdn.causal_doc_conv): a tap
  whose input lies in another document than t, or before the row, contributes nothing. For a row that is one document
  this is the source's convolution with K - 1 zeros of left padding.
- The start. The taps are zeros at the start, so h' = h and the model starts as the model without the key; the
  source's code leaves PyTorch's Conv1d default (uniform in +-1/sqrt(K) per tap, +-0.5 at K = 4), under which the arm
  would not start as its baseline. That start is EncoderConfig.canon_init = "uniform" ("The source's start" below);
  the default is "zeros". Zeros draw no random number: every other parameter starts at the baseline's value.
  Bit for bit on the CPU, eager and compiled (tests/test_canon.py). Not bit for bit in compiled training on the GPU
  (measured on a GPU, 2026-10-12): the arm's loss at step 1 was 4.8e-6 nats from the base's and its
  gradient norm 0.12% away; a layer with a Canon layer compiles to another graph than the layer without one. The
  eager comparison on the GPU is tests/gpu/test_ladder_c_canon_gpu.py, section 3.
- Where. Canon-A only, and in the global layers only (5 of the 20 at 60M). The GDN layers have a short convolution
  of their own on q, k and v (what the source calls a partial Canon-B in such layers); Canon-C and Canon-D (around
  and inside the FFN) and the decoder's Canon layers are not built.

Parameters: K * d_model per Canon layer (4 x 512 = 2,048 at 60M; five layers: 10,240). The parameter is named
`conv_weight`, as the GDN layers' depthwise convolution, and ced/train/optim.py treats it the same way: AdamW (not
Muon: it is not a hidden-state matrix), with the run's weight decay, which pulls the taps to zero, their start.

Cached generation: the layer's cache for the next chunk is the last K - 1 inputs h of the Canon layer (before the
convolution) and their document ids, carried next to the layer's keys (attention.KeySet.canon_input / canon_doc), so
teacher forcing equals cached generation for any chunking. The decode engine keeps the same K - 1 inputs per Canon
layer (ced/infer/engine.py).

Built for arch "control" without a second pass; refused elsewhere at model build (ced/model/control.py check_canon).

The source's start (EncoderConfig.canon_init = "uniform").
- What the source does (read on 2026-10-12 in facebookresearch/PhysicsLM4 at commit 924678b). Its Canon layer is a
  subclass of nn.Conv1d(d, d, K, groups=d, bias=False, padding=K-1) (huggingface/canon_helper.py, ShortConvolution,
  create_canon) and nothing sets its weight: both of its model codes call the module's own reset_parameters() at the
  model's initialisation (huggingface/modeling_llama_canon.py _init_weights, "Use Kaiming initialization";
  lingua_modified/lingua/transformer.py init_weights). That is PyTorch's Conv1d default,
  kaiming_uniform_(weight, a=sqrt(5)): uniform in +-1/sqrt(fan_in), and fan_in = (d / groups) * K = K for a depthwise
  convolution, so +-1/sqrt(K): +-0.5 at K = 4 (on this torch, 512 channels: largest |w| 0.4998, SD 0.284 against
  1/sqrt(12) = 0.289; measured). Its recipes page names "the default PyTorch initialization (rather than alternatives
  like `std=0.02`)" as part of what keeps training stable (lingua_recipes/README.md). Its optimizer is one AdamW
  group over every parameter with one weight decay (lingua_modified/lingua/optim.py, default 0.1): the source decays
  its Canon weights too.
- What is built. "uniform" draws every tap uniform in [-1/sqrt(K), 1/sqrt(K)) (canon_bound). The rule of the draw
  (init_canon_taps, canon_generator): one CPU generator per Canon layer, seeded with the first 8 bytes (little-endian,
  63 bits) of BLAKE2b("canon-taps:<seed>:<i>"), where <seed> is torch.initial_seed(), the seed the global generator
  was last seeded with (the run's seed: the trainer calls torch.manual_seed(cfg.seed) right before it builds the
  model), and <i> is the layer's index in the encoder stack (3, 7, 11, 15, 19 at 60M). The taps are drawn in float32
  on the CPU in the tensor's own order and copied into the parameter. Nothing is drawn from the global generator
  (only the seed it was given is read), so every other parameter of the model starts at exactly the value it has
  with "zeros", which is the baseline's (tests/test_canon.py proves it parameter by parameter), and the same seed
  gives the same taps on any device. A resumed run does not keep a fresh draw: the build draws the same taps again
  and the checkpoint's state dict then replaces them with the trained ones, as it replaces every weight.
- How the model differs at step 0. h' = h + conv(h) is no longer h: per channel the current input is scaled by
  1 + w_0 (between 0.5 and 1.5) and the three inputs before it are mixed in with weights in +-0.5. For taps
  independent of h the added term has sqrt(K / 12) = 0.58 of h's RMS at K = 4 (modelled; 0.29 at a document's first
  token, where one tap reads). The model is not the baseline's function at step 0, on purpose: that is the arm.
- Weight decay. The taps stay on AdamW with the run's weight decay (ced/train/optim.py), which pulls them toward
  zero: with "zeros" toward their start, with "uniform" away from it. The size of the pull is in ced/train/optim.py.
"""

from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn

from ced.model.gdn import causal_doc_conv

CANON_INITS = ("zeros", "uniform")  # EncoderConfig.canon_init: the start of the taps (module docstring)


def canon_bound(kernel: int) -> float:
    """The half-width of the source's uniform start: 1/sqrt(K), PyTorch's default for a depthwise Conv1d of kernel K
    (kaiming_uniform_ with a = sqrt(5) and fan_in = K). 0.5 at K = 4."""
    return 1.0 / math.sqrt(kernel)


def canon_generator(seed: int, layer: int) -> torch.Generator:
    """The generator of one Canon layer's uniform start: a CPU generator whose state depends only on (seed, layer), the
    run's seed and the layer's index in the encoder stack. The two are hashed together, so no two layers and no two
    seeds share a stream, and none is the global generator's."""
    digest = hashlib.blake2b(f"canon-taps:{seed}:{layer}".encode(), digest_size=8).digest()
    return torch.Generator(device="cpu").manual_seed(int.from_bytes(digest, "little") & (2**63 - 1))


def init_canon_taps(layers, init: str, seed: int | None = None) -> None:
    """Start the taps of every Canon layer among `layers` (the encoder's stack, in order) as EncoderConfig.canon_init
    says. "zeros": nothing is done, read or drawn (the taps are built as zeros). "uniform": each layer's taps from
    canon_generator(seed, the layer's index in `layers`), seed = torch.initial_seed() unless given (module docstring,
    "The source's start"). Nothing is drawn from the global generator: only the seed it was given is read."""
    assert init in CANON_INITS, f"encoder.canon_init {init!r}: one of {CANON_INITS}"
    if init == "zeros":
        return
    seed = torch.initial_seed() if seed is None else seed
    for i, layer in enumerate(layers):
        canon = getattr(layer, "canon", None)
        if canon is not None:
            canon.init_uniform(canon_generator(seed, i))


class CanonLayer(nn.Module):
    """h' = h + (a causal depthwise convolution of kernel K over h, inside each document): the module docstring.

    conv_weight (d_model, K), zeros as built (init_uniform gives it the source's start): conv_weight[:, K-1] multiplies
    the current input and conv_weight[:, K-1-j] the input j steps back (the layout of GatedDeltaNet.conv_weight)."""

    def __init__(self, d_model: int, kernel: int):
        super().__init__()
        assert isinstance(kernel, int) and not isinstance(kernel, bool) and kernel >= 2, \
            f"a Canon layer's kernel is an integer >= 2, not {kernel!r}"
        self.kernel = kernel
        self.conv_weight = nn.Parameter(torch.zeros(d_model, kernel))

    @torch.no_grad()
    def init_uniform(self, generator: torch.Generator) -> None:
        """The source's start: every tap uniform in [-1/sqrt(K), 1/sqrt(K)), drawn from `generator` (a CPU generator) in
        float32 on the CPU and copied into the parameter, whatever its device and dtype (a parameter on the meta device
        keeps its shape and holds no value)."""
        w = self.conv_weight
        bound = canon_bound(self.kernel)
        drawn = torch.empty(tuple(w.shape), dtype=torch.float32, device="cpu").uniform_(-bound, bound, generator=generator)
        if not w.is_meta:
            w.copy_(drawn)

    def forward(
        self, h: torch.Tensor, doc: torch.Tensor, hist: torch.Tensor | None = None, hist_doc: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h (B, L, d): the attention layer's normed input; doc (B, L): the positions' document ids; hist (B, K-1, d) /
        hist_doc (B, K-1): the last K - 1 inputs of earlier chunks and their document ids (None: no earlier chunk).
        Returns (h', the new history, its document ids); h' in h's dtype."""
        out, hist, hist_doc = causal_doc_conv(h, doc, self.conv_weight, hist, hist_doc)
        return (h + out).to(h.dtype), hist, hist_doc
