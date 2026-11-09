"""NVFP4 fake quantisation of linear layers (opt-in, off by default: ModelConfig.nvfp4, NVFP4Config).

Built 2026-10-07/08 for the GDN study's part (e) and brought to main for the 60M NVFP4 screen (which states what
"emulated" means and what it cannot show).

Emulates training with NVFP4 GEMMs on H100, which has no FP4 tensor cores: every GEMM of a quantised
linear layer runs on operands rounded to NVFP4 and dequantised, in the bf16 GEMM the layer uses today.
It measures quality only; it says nothing about speed.

The recipe followed is NVIDIA's "Pretraining Large Language Models with NVFP4" (arXiv 2509.25149v2;
section numbers below are that paper's):

Format (section 2, appendix B): E2M1 elements (0, 0.5, 1, 1.5, 2, 3, 4, 6 and negatives) in blocks of
16 elements, one FP8 E4M3 scale per block, one FP32 scale per tensor:

    s_enc        = 6 * 448 / amax(tensor)              global encode scale (B.1)
    s_dec_b      = amax(block) / 6                     block decode scale (B.2)
    s_dec_b_e4m3 = e4m3(s_dec_b * s_enc)               stored block scale, round to nearest even (B.2)
    s_enc_b      = 1 / (fp32(s_dec_b_e4m3) * s_dec)    with s_dec = 1 / s_enc (B.2)
    x_hat        = q(x * s_enc_b)                      E2M1, saturating at +-6 (B.3)
    x           ~= x_hat * s_dec_b_e4m3 * s_dec

GEMMs (section 3, Fig. 5): Fprop y = x W^T, Dgrad dx = dy W and Wgrad dW = dy^T x each quantise both
inputs.
- Weights: 16x16 blocks (section 4.3, "2D scaling"), so Fprop and Dgrad see the same quantised W;
  round to nearest even.
- Activations x: 1x16 blocks along the dot-product dimension; round to nearest even. Wgrad
  re-quantises x from the high-precision tensor, blocked along the tokens (appendix E.5: activations
  tolerate the different Fprop/Wgrad representations).
- Gradients dy: 1x16 blocks along the dot-product dimension, stochastic rounding, in both Dgrad and
  Wgrad (section 4.4, appendix E.3; stochastic rounding of weights or activations is worse).
  Stochastic rounding is unbiased only below saturation. The block scale is rounded to E4M3 (B.2), and
  when it rounds down (about half the blocks) the block's largest elements land above 6 after scaling
  and saturate there (B.3): E[q] = 6 < a, a bias toward zero of at most 5.9% of such an element (1 -
  1 / 1.0625; more for E4M3-subnormal block scales, blocks below ~3.5e-5 of the tensor amax). Measured
  on a Gaussian 4096 x 1024 tensor: 3.5% of the elements saturate, by 2.2% on average, so the bias
  E[q] - x has about 1% of the tensor's RMS (1.4% for Student-t with 3 df), against 13.7% for one
  draw's error. It is always toward zero, so unlike the draws' noise it does not average out.
- Wgrad inputs only (section 4.2, appendix C): a 16x16 random Hadamard transform over each 16-token
  segment, x' = x H (H = diag(s) Hadamard_16 / 4), applied to both operands, so dy'^T x' = dy^T x in
  exact arithmetic; one random sign vector s for every layer and step; the tensor scale is taken after
  the transform. Fprop and Dgrad inputs are not transformed (transforming them hurt). The transform
  runs in float32 with autocast off, also when torch.compile traces the backward under the forward's
  autocast (rht_tokens).
- Kept in high precision (section 4.1): embeddings, the output head, norms, nonlinearities, attention's
  softmax and its QK^T / PV products, master weights, weight gradients and optimizer state. Here that
  also covers the GDN delta rule, its short conv and gates (not linear layers). The recipe also keeps
  the linear layers of a few sensitive blocks in BF16, mostly the last ones (12B: the first 2 and last
  8 of 62 blocks, 16% of the linear layers; 8B: the last 8 of 52; 1.2B: the last 4 of 20; "about 15%
  of the network"): NVFP4Config.keep_last_fraction / keep_first_layers.
- Evaluations in the paper are run in BF16 (Table 2): the eval forward here is BF16 too. The quantised
  path runs only in training mode, so every eval-mode forward (in-training evals, the re-score, the
  probes, cached generation) is exactly the model without the switch.

Positions and the tensor scale: an activation's blocks are 16 features of ONE token and the weights do
not depend on the input, so Fprop is per token except for one number per quantised tensor, its largest
magnitude over the micro-batch (s_enc, B.1), which every block scale is rounded against. A later token
that moves that maximum therefore changes the rounding at earlier positions: the training forward is
causal given the tensor scales, not bit for bit (tests/test_nvfp4_screen.py measures it; real NVFP4
tensors have the same per-tensor scale). No gradient flows through a scale (the backward treats it as a
constant), so d logits[t] / d inputs[s > t] is exactly zero, and the eval forward is exactly causal.

Emulation: the GEMM operands are v = x_hat * s_dec_b_e4m3, which bf16 represents exactly (at most 2 + 4
significant bits, magnitudes 2^-10 .. 2688). The GEMM runs in the autocast dtype (bf16; fp32
accumulation), then both tensors' s_dec are applied to its output. Up to the accumulation order and
one extra rounding of the output, this is the NVFP4 GEMM's result. Weight gradients come out of a
bf16 GEMM and are accumulated into the fp32 .grad, as for nn.Linear under autocast.
Sources of the quantisation: Fprop quantises the activation as it arrives (float32 from the norms) and
the float32 master weight; Wgrad re-quantises the activation from the bf16 copy saved for the backward
(what nn.Linear saves under autocast); gradients arrive in bf16. Transformer Engine in a BF16 model
quantises bf16 tensors; the difference is below one bf16 ulp before a 4-bit rounding.
Compiled: Inductor (the trainer's backend) does not round fused fp32 -> bf16 -> fp32 round trips
(torch._inductor.config.emulate_precision_casts is off by default), so in compiled code dx and dW may
skip the bf16 rounding of their GEMM output and Wgrad may quantise the float32 activation instead of its
bf16 copy (the same sub-ulp difference). With emulate_precision_casts on, and with the aot_eager
backend, the compiled layer under bf16 autocast is bitwise the eager one (tests/test_nvfp4.py).

Not emulated: the speed of FP4 GEMMs; the paper's late switch to BF16 (appendix D, optional); Transformer
Engine's exact kernels (fused cast + transpose, the order of accumulation inside a block).

Fixed shapes: every tensor here has a shape that follows from its input's shape alone (padding to a
multiple of 16, reshape, where, clamp); no boolean-mask indexing, nonzero or masked_select.

- NVFP4Config.keep_names: linears kept in BF16 by name in every layer (e.g. GDN's gate projections in_proj_a and
  in_proj_b: "GDN gate projections in BF16 vs NVFP4"), on top of the mode and the kept layers (apply_nvfp4).
- NVFP4Config.recipe "quartet2": the Quartet II scheme (Panferov, Schultheis, Tabesh, Alistarh, arXiv 2601.22813;
  read from the paper's Algorithm 1 and the authors' pseudo-quantisation reference, repository IST-DASLab/Quartet-II,
  kernels/python/quartet2/pseudoquant.py and linear.py), in the same emulation as the nvidia recipe (GEMM operands
  exact in bf16, the two tensor scales applied to the GEMM's output). Quartet2LinearFunction:
  * Fprop: activations AND weights in 1x16 blocks along the dot product, round to nearest with ties AWAY from zero
    (the source's rtn_fp4 thresholds), and "four over six": every block is quantised twice, with its largest
    magnitude mapped to 6 and to 4 (block scale e4m3(amax_b / 6 / s_dec) and e4m3(1.5 amax_b / 6 / s_dec)), and keeps
    the one with the smaller squared error (6 on a tie). The tensor decode scale is s_dec = amax / 256 / 6 (256, not
    448, leaves the 4-scale room below E4M3's 448). A block scale that rounds to 0 is replaced by 1 (the source's
    rule), which leaves its elements at 0 here as there.
  * Dgrad dx = dy W and Wgrad dW = dy^T x: each operand is re-quantised with MS-EDEN along the GEMM's dot-product
    dimension (output features for Dgrad, tokens for Wgrad). The operands are the forward's QUANTISED weight and
    activation (dequantised), not the high-precision tensors, and the gradient. MS-EDEN (quartet2_eden_quantize): a
    random Hadamard transform over each 128-element segment, x' = x diag(s) H_128 / sqrt(128), with ONE fresh random
    sign vector s per backward call shared by both GEMMs' four operands (so the transform cancels in each product);
    the tensor scale s_dec = amax(x') / 255.99 / g with the grid's maximum g = 6 / (17/16 * 0.93); block scales
    e4m3(amax_b / g / s_dec) (0 replaced by 1); round to nearest (ties away); per 16-block the correction S = <x',
    x'> / <x', q> (in scaled units; 1 where the denominator is 0); the corrected scale S * scale rounded
    STOCHASTICALLY to one of its two E4M3 neighbours, so that E[scale] = S * scale. The dot-product dimension is
    padded with zeros to a multiple of 128 (the source asserts it is one; zeros change no product). Deliberately
    different from the source: the source rounds every dequantised operand to bf16 and rotates in bf16; here the
    operands are exact and the transform runs in float32 (as the nvidia recipe's). The source's training code
    (src/models/quantization/schemes/quartet_2.py) computes S over each 128-segment instead of each 16-block; the
    paper's Algorithm 1 and the kernels' reference compute it per 16-block, as here. Delayed amax (an option of the
    training code, off in its sweep) is not built: every tensor scale is the current tensor's. The kept-layers,
    kept-names and mode rules are the nvidia recipe's; stochastic_rounding, rht and weight_2d are the nvidia
    recipe's switches and must keep their defaults under "quartet2" (check_nvfp4).

One addition of the NVFP4 tail arms, off by default (every run before it, bit for bit):

- NVFP4Config.fp8_names: linears that run FP8 by name, where the mode and the kept layers would quantise them (a published large-scale FP4 training
  report keeps its Mamba output projection in MXFP8). Such a linear stays an NVFP4Linear (one structure per layer kind for
  torch.compile) with nvfp4_active True and nvfp4_fp8 True, and its training forward is ced/model/fp8.py's
  FP8LinearFunction with the FP8 control's "rowwise" recipe (E4M3; one power-of-two scale per row of each GEMM operand
  along its dot product; Fprop, Dgrad and Wgrad), under either recipe. With the key set, every NVFP4Linear carries
  nvfp4_fp8 (False where it does not apply); with it at () no module gets the attribute. The summary lists such linears
  under "fp8" (not under "quantised" or "high_precision") with their weights in "fp8_params"; quantised_param_share stays
  the NVFP4 share.

One addition of the NVFP4 MLP arm, off unless selected (every run before it, bit for bit):

- mode "mlp_only": the three linears of the FFN (w_gate, w_up, w_down) of every layer that is not kept are NVFP4; no linear of any
  mixer (GDN or attention) is. keep_last_fraction picks the layers as before (0.2 of 20 layers at 60M = layers 0-15 quantised, 16-19
  kept: the floor's "first 32 of 40" as a fraction). The mixers' linears are left as the plain nn.Linear they are (apply_nvfp4 does not
  change their class), which is what lets the mode run beside R1-entry's walk, the GDN write mixes and the attention variants: they
  see the modules they always saw (ced/model/attn_res.py check_attn_res and ced/model/gdn_arms.py check_gdn_arms accept this mode and
  refuse the others as before). The summary counts the mixers' linears in "linear_params" and lists them under "high_precision",
  so quantised_param_share is the share of ALL the layers' linear weights.
- NVFP4Config.until_step N > 0: the quantised linears are NVFP4 for updates 0 .. N - 1 and compute F.linear from update N on (NVFP4 for 66.8
  % of the steps, inside one run). set_phase flips nvfp4_active in memory, which is what the stage "continue" with precision_switch does
  between processes: bit for bit (tests/test_nvfp4_mlp.py). Each module that the plan makes active carries nvfp4_planned
  (True) so that a linear kept in BF16 stays so when the phase comes back on (a resume before N); the attribute exists only with the key set.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ced.model import fp8 as _fp8

E2M1_MAX = 6.0
E4M3_MAX = 448.0
BLOCK = 16
RHT_SEED = 2509  # the random sign vector of the Hadamard transform: one, fixed, for every layer and step
MODES = ("none", "all_linears", "all_but_gdn", "all_but_gdn_mixer", "mlp_only")
RECIPES = ("nvidia", "quartet2")
# Quartet II (recipe "quartet2"; the module docstring): the forward's E4M3 ceiling for the block scales under "four
# over six", the backward's (MS-EDEN leaves room for the correction), the backward grid's maximum and the size of
# MS-EDEN's Hadamard transform. The source's values (kernels/python/quartet2/pseudoquant.py, linear.py).
Q2_FWD_SCALES_MAX = 256.0
Q2_BWD_SCALES_MAX = 255.99
Q2_BWD_GRID_MAX = 6.0 / ((17.0 / 16.0) * 0.93)
Q2_RHT = 128


# ----------------------------------------------------------------------------------------------
# the quantiser


def e2m1_round(a: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
    """Round magnitudes a >= 0 (float32) to the E2M1 grid {0, 0.5, 1, 1.5, 2, 3, 4, 6}, saturating at 6.

    The grid is uniform on [0, 2) (step 0.5), [2, 4) (step 1) and [4, 6] (step 2), and the parity of
    the index a / step is the parity of the E2M1 mantissa bit, so round-half-to-even on the index is
    round-to-nearest-even on the format. u None: round to nearest, ties to even. u ~ U[0, 1) (same
    shape): stochastic rounding, up with probability (a - lo) / (hi - lo), so E[q] = a for a <= 6 and
    E[q] = 6 above: unbiased only below saturation (for the bias that saturation leaves in a whole
    NVFP4 tensor, about 1% of its RMS, see the module docstring)."""
    step = torch.where(a < 2.0, 0.5, torch.where(a < 4.0, 1.0, 2.0)).to(a.dtype)
    t = a / step  # exact: step is a power of two
    n = torch.round(t) if u is None else torch.floor(t + u)
    return torch.clamp(n * step, max=E2M1_MAX)


def global_encode_scale(amax: torch.Tensor) -> torch.Tensor:
    """s_enc = 6 * 448 / amax (float32, 0-dim); 1 for an all-zero tensor, capped at float32's max."""
    amax = amax.float()
    s = (E2M1_MAX * E4M3_MAX) / amax
    s = torch.clamp(s, max=torch.finfo(torch.float32).max)
    return torch.where(amax > 0, s, torch.ones_like(s))


def nvfp4_encode(
    x: torch.Tensor, *, stochastic: bool = False, block_2d: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4 encoding of x with the paper's two-level scaling: what an NVFP4 tensor stores.

    Blocks: 16 consecutive elements of the last dimension (1x16), or with block_2d 16x16 tiles of the
    last two dimensions (weights). Incomplete blocks are padded with zeros (which change no amax).
    Rounding of the elements: to nearest even, or stochastic (gradients); of the block scales: to
    nearest even.

    Returns (x_hat, scales, s_dec):
    - x_hat: float32 E2M1 values, blocked: (..., n/16, 16), or (..., m/16, 16, n/16, 16) for 2D;
    - scales: the block decode scales s_dec_b_e4m3 as torch.float8_e4m3fn, (..., n/16, 1) or
      (..., m/16, 1, n/16, 1);
    - s_dec: 1 / s_enc, the float32 per-tensor decode scale (0-dim)."""
    x32 = x.float()
    shape = x32.shape
    pad_n = (-shape[-1]) % BLOCK
    if block_2d:
        pad_m = (-shape[-2]) % BLOCK
        xp = F.pad(x32, (0, pad_n, 0, pad_m)) if pad_n or pad_m else x32
        M, N = xp.shape[-2:]
        blocks = xp.reshape(*xp.shape[:-2], M // BLOCK, BLOCK, N // BLOCK, BLOCK)
        amax_b = blocks.abs().amax(dim=(-3, -1), keepdim=True)
    else:
        xp = F.pad(x32, (0, pad_n)) if pad_n else x32
        blocks = xp.reshape(*xp.shape[:-1], xp.shape[-1] // BLOCK, BLOCK)
        amax_b = blocks.abs().amax(dim=-1, keepdim=True)
    s_enc = global_encode_scale(x32.abs().amax())
    s_dec = 1.0 / s_enc
    s_dec_b = amax_b / E2M1_MAX
    # The product is <= 448 up to float32 rounding; clamp so that the cast cannot overflow (e4m3fn has no inf).
    scales = torch.clamp(s_dec_b * s_enc, max=E4M3_MAX).to(torch.float8_e4m3fn)
    s32 = scales.float()
    has_scale = s32 > 0  # a block whose stored scale is 0 (all zeros, or underflow) decodes to zeros
    s_enc_b = torch.where(has_scale, 1.0 / (torch.where(has_scale, s32, 1.0) * s_dec), 0.0)
    scaled = blocks * s_enc_b
    mag = scaled.abs()
    u = torch.rand_like(mag) if stochastic else None
    return torch.copysign(e2m1_round(mag, u), scaled), scales, s_dec


def nvfp4_quantize(
    x: torch.Tensor, *, stochastic: bool = False, block_2d: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """x quantised to NVFP4 (nvfp4_encode), as GEMM operands: (v, s_dec) with v = x_hat * s_dec_b_e4m3
    (float32, x's shape; every value exact in bf16) and s_dec the per-tensor decode scale, so that the
    dequantised tensor is v * s_dec."""
    x_hat, scales, s_dec = nvfp4_encode(x, stochastic=stochastic, block_2d=block_2d)
    v = x_hat * scales.float()
    shape = x.shape
    if block_2d:
        v = v.reshape(*v.shape[:-4], v.shape[-4] * BLOCK, v.shape[-2] * BLOCK)[..., : shape[-2], : shape[-1]]
    else:
        v = v.reshape(*v.shape[:-2], v.shape[-2] * BLOCK)[..., : shape[-1]]
    return v, s_dec


def fake_quantize(x: torch.Tensor, *, stochastic: bool = False, block_2d: bool = False) -> torch.Tensor:
    """The dequantised NVFP4 tensor (float32): what a GEMM on NVFP4 operands multiplies."""
    v, s_dec = nvfp4_quantize(x, stochastic=stochastic, block_2d=block_2d)
    return v * s_dec


def hadamard_matrix(n: int = BLOCK) -> torch.Tensor:
    """The n x n Sylvester Hadamard matrix (entries +-1), n a power of two."""
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def rht_matrix(seed: int = RHT_SEED) -> torch.Tensor:
    """The 16x16 random Hadamard transform T = diag(s) H_16 / 4 (orthogonal: T T^T = I), s a random sign
    vector drawn once from `seed` (the paper shares one sign vector across layers and steps)."""
    g = torch.Generator().manual_seed(seed)
    s = torch.randint(0, 2, (BLOCK,), generator=g).float() * 2 - 1
    return s[:, None] * hadamard_matrix(BLOCK) / math.sqrt(BLOCK)


def rht_tokens(t2d: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """(M, n) -> (n, M'): the transpose, with every 16-token segment of each row multiplied by the
    transform (x' = x T, appendix C), M padded with zeros to a multiple of 16. Float32.

    Autocast is off inside: the trainer's eager backward runs outside autocast, but torch.compile
    traces the custom Function's backward under the forward's autocast, where this matmul would run in
    bf16 (rounding the transformed operands before their 4-bit quantisation; the paper keeps it fp32)."""
    with torch.autocast(t2d.device.type, enabled=False):
        t = t2d.float().t()
        pad = (-t.shape[-1]) % BLOCK
        if pad:
            t = F.pad(t, (0, pad))
        n, m = t.shape
        return torch.matmul(t.reshape(n, m // BLOCK, BLOCK), transform.float()).reshape(n, m)


def tokens_blocked(t2d: torch.Tensor) -> torch.Tensor:
    """(M, n) -> (n, M') transposed and padded like rht_tokens, without the transform."""
    t = t2d.float().t()
    pad = (-t.shape[-1]) % BLOCK
    return F.pad(t, (0, pad)) if pad else t


# ----------------------------------------------------------------------------------------------
# the Quartet II quantisers (recipe "quartet2"; the module docstring)


def e2m1_round_away(a: torch.Tensor) -> torch.Tensor:
    """Round magnitudes a >= 0 (float32) to the E2M1 grid, ties AWAY from zero, saturating at 6: the source's rtn_fp4
    (thresholds 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, each one rounding up). Exact: the index t = a / step is exact
    (a power-of-two step) and so is t - floor(t), so no sum can round across a threshold."""
    step = torch.where(a < 2.0, 0.5, torch.where(a < 4.0, 1.0, 2.0)).to(a.dtype)
    t = a / step
    n = torch.floor(t)
    n = n + (t - n >= 0.5).to(a.dtype)
    return torch.clamp(n * step, max=E2M1_MAX)


def _blocks16(x32: torch.Tensor) -> torch.Tensor:
    """(..., n) -> (..., n'/16, 16), the last dimension padded with zeros to a multiple of 16."""
    pad = (-x32.shape[-1]) % BLOCK
    xp = F.pad(x32, (0, pad)) if pad else x32
    return xp.reshape(*xp.shape[:-1], xp.shape[-1] // BLOCK, BLOCK)


def _e4m3_or_one(s: torch.Tensor) -> torch.Tensor:
    """e4m3(s) as float32 (round to nearest even; s >= 0, clamped to 448), a zero replaced by 1 (the source's
    get_scales: a block whose scale underflows is divided by 1 * s_dec, so its elements quantise to 0)."""
    q = torch.clamp(s, max=E4M3_MAX).to(torch.float8_e4m3fn).float()
    return torch.where(q == 0, torch.ones_like(q), q)


def quartet2_rtn_quantize(x: torch.Tensor, four_over_six: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Quartet II's forward quantiser (the source's rtn_1x16s_fp4_kernel): x (..., n) in 1x16 blocks along its last
    dimension, round to nearest with ties away from zero, and with four_over_six the better of the block maxima 6
    and 4 by the block's squared error (6 on a tie). Returns (v, s_dec) as nvfp4_quantize does: v = x_hat * block
    scale (float32, x's shape, exact in bf16), and the dequantised tensor is v * s_dec."""
    x32 = x.float()
    shape = x32.shape
    blocks = _blocks16(x32)
    amax = x32.abs().amax()
    ceiling = Q2_FWD_SCALES_MAX if four_over_six else E4M3_MAX
    s_dec = torch.where(amax > 0, amax / ceiling / E2M1_MAX, torch.ones_like(amax))
    amax_b = blocks.abs().amax(dim=-1, keepdim=True)

    def candidate(scale):
        unit = scale * s_dec
        scaled = blocks / unit
        q = torch.copysign(e2m1_round_away(scaled.abs()), scaled)
        return q, ((blocks - q * unit) ** 2).sum(-1, keepdim=True)

    scale = _e4m3_or_one(amax_b / E2M1_MAX / s_dec)
    q, err = candidate(scale)
    if four_over_six:
        scale4 = _e4m3_or_one(amax_b / E2M1_MAX * 1.5 / s_dec)
        q4, err4 = candidate(scale4)
        six = err <= err4
        q, scale = torch.where(six, q, q4), torch.where(six, scale, scale4)
    v = (q * scale).reshape(*blocks.shape[:-2], blocks.shape[-2] * BLOCK)[..., : shape[-1]]
    return v, s_dec


def hadamard_rotation(signs: torch.Tensor) -> torch.Tensor:
    """The transform diag(s) H_n / sqrt(n) (orthogonal; n = len(signs)), float32, on the signs' device: the source's
    `had * signs[None, :]` as its eden wrapper applies it (x @ had.T; Sylvester's H is symmetric)."""
    n = signs.shape[0]
    return signs.float()[:, None] * hadamard_matrix(n).to(signs.device) / math.sqrt(n)


def quartet2_rotate(t2d: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """(R, K) -> (R, K'): every n-segment of each row multiplied by the (n x n) transform, K padded with zeros to a
    multiple of n. Float32 with autocast off (as rht_tokens)."""
    n = transform.shape[0]
    with torch.autocast(t2d.device.type, enabled=False):
        t = t2d.float()
        pad = (-t.shape[-1]) % n
        if pad:
            t = F.pad(t, (0, pad))
        r, k = t.shape
        return torch.matmul(t.reshape(r, k // n, n), transform.float()).reshape(r, k)


def e4m3_neighbours(c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(down, up): the E4M3 values next to c >= 0 (float32), down <= c <= up, both equal to c when c is an E4M3 value
    (the source's bit +-1 around the nearest value). c is clamped to 448 first (unreachable here: corrected scales
    stay below 256 x 1.1)."""
    c = torch.clamp(c, max=E4M3_MAX)
    near = c.to(torch.float8_e4m3fn)
    bits = near.view(torch.uint8).to(torch.int16)
    r = near.float()

    def at(b):
        return b.clamp(0, 0x7E).to(torch.uint8).view(torch.float8_e4m3fn).float()

    above = r > c
    down = torch.where(above, at(bits - 1), r)
    up = torch.where(above, r, torch.where(r == c, r, at(bits + 1)))
    return down, up


def quartet2_eden_quantize(t2d: torch.Tensor, transform: torch.Tensor, u: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """MS-EDEN (the module docstring; the paper's Algorithm 1, the source's eden_1x16s_fp4_kernel) of t2d (R, K) along
    K, the dot-product dimension of its GEMM: rotated (quartet2_rotate; K padded to a multiple of 128), blocks of 16,
    round to nearest (ties away), the per-block correction S = <x', x'> / <x', q> in scaled units, the corrected scale
    rounded stochastically to an E4M3 neighbour with u (R, K'/16) ~ U[0, 1) (None: drawn here with torch.rand): up
    with probability (S scale - down) / (up - down). Returns (v, s_dec): v (R, K') float32 = x_hat * stored scale
    (exact in bf16), so that the dequantised ROTATED tensor is v * s_dec."""
    xr = quartet2_rotate(t2d, transform)
    r, k = xr.shape
    blocks = xr.reshape(r, k // BLOCK, BLOCK)
    amax = xr.abs().amax()
    s_dec = torch.where(amax > 0, amax / Q2_BWD_SCALES_MAX / Q2_BWD_GRID_MAX, torch.ones_like(amax))
    scale = _e4m3_or_one(blocks.abs().amax(dim=-1, keepdim=True) / Q2_BWD_GRID_MAX / s_dec)
    scaled = blocks / (scale * s_dec)
    q = torch.copysign(e2m1_round_away(scaled.abs()), scaled)
    num = (scaled * scaled).sum(-1, keepdim=True)
    den = (scaled * q).sum(-1, keepdim=True)
    zero = den == 0
    corr = torch.where(zero, torch.ones_like(den), num / torch.where(zero, torch.ones_like(den), den))
    c = scale * corr
    down, up = e4m3_neighbours(c)
    gap = up - down
    p_up = torch.where(gap > 0, (c - down) / torch.where(gap > 0, gap, torch.ones_like(gap)), torch.zeros_like(gap))
    if u is None:  # rand_like of a tensor computed from t2d: see quartet2_signs for why the draw must depend on the data
        u = torch.rand_like(num[..., 0])
    stored = torch.where(u[..., None] < p_up, up, down)
    return (q * stored).reshape(r, k), s_dec


@torch.library.custom_op("ced::quartet2_signs", mutates_args=())
def _quartet2_signs_op(like: torch.Tensor, n: int) -> torch.Tensor:
    """n random signs (+-1, float32) on like's device from its global generator; like's values are not read."""
    return torch.where(torch.rand(n, device=like.device) < 0.5, 1.0, -1.0)


@_quartet2_signs_op.register_fake
def _(like: torch.Tensor, n: int) -> torch.Tensor:
    return like.new_empty(n, dtype=torch.float32)


def quartet2_signs(like: torch.Tensor, n: int = Q2_RHT) -> torch.Tensor:
    """A fresh random sign vector (n,), +-1 float32, from the global generator of like's device (the source's
    rerotate_hadamard: one per backward call), each sign -1 with probability 1/2. In eager code: torch.rand(n).

    An opaque custom op (ced::quartet2_signs) that takes the backward's gradient `like` as its input (its values are not
    read), because ONE draw must serve all four rotations of a backward call, and compiled code broke that twice
    (measured on this layer, tests/test_nvfp4_structure.py checks compiled against eager): a free-standing
    torch.randint is a "forward node" to AOT autograd's min-cut partitioner (aot_eager and Inductor), which computed it
    a second time in the FORWARD graph for the rotations of W^T and x^T, so they no longer cancelled against dy's
    (relative gradient error 1.37 against 0.16 eager); a draw that depends on the gradient stays in the backward, but
    Inductor then inlined the random op into each consumer kernel with its own random numbers (error 1.53). The custom
    op is one kernel, run once, whose output every rotation reads."""
    return _quartet2_signs_op(like, n)


# ----------------------------------------------------------------------------------------------
# the linear layer


def _gemm(a: torch.Tensor, b_t: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(a @ b_t^T) * scale with both operands in `dtype` (exact for NVFP4 operands), output in `dtype`;
    the scale (the product of the two float32 tensor decode scales) is applied in at least float32."""
    out = torch.matmul(a.to(dtype), b_t.to(dtype).t())
    return (out.to(torch.promote_types(dtype, torch.float32)) * scale).to(dtype)


class NVFP4LinearFunction(torch.autograd.Function):
    """y = x W^T with NVFP4 operands in Fprop, Dgrad and Wgrad (see the module docstring)."""

    @staticmethod
    def forward(ctx, x, weight, transform, gemm_dtype, stochastic, rht, weight_2d):
        k = x.shape[-1]
        # Fprop quantises the tensors as they arrive: the activation in its own precision (float32 from
        # the norms here) and the float32 master weight. (Quantising a bf16 copy first would round twice;
        # torch.compile may also drop such an intermediate cast, so eager and compiled code would differ.)
        xv, xs = nvfp4_quantize(x.reshape(-1, k))
        wv, ws = nvfp4_quantize(weight, block_2d=weight_2d)
        y = _gemm(xv, wv, xs * ws, gemm_dtype)
        # Saved for Wgrad: the activation in the GEMM dtype, as nn.Linear saves it under autocast (same memory).
        x_saved = x.to(gemm_dtype)
        if weight_2d:  # the same quantised W serves Dgrad
            ctx.save_for_backward(x_saved, wv.to(gemm_dtype), ws, transform)
        else:
            ctx.save_for_backward(x_saved, weight, ws, transform)
        ctx.gemm_dtype, ctx.stochastic, ctx.rht, ctx.weight_2d = gemm_dtype, stochastic, rht, weight_2d
        ctx.x_dtype, ctx.w_dtype = x.dtype, weight.dtype
        return y.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x_hp, w_saved, ws, transform = ctx.saved_tensors  # x_hp: the activation in the GEMM dtype
        gd = ctx.gemm_dtype
        n, k = w_saved.shape
        dy2 = dy.to(gd).reshape(-1, n)
        dx = dw = None
        if ctx.needs_input_grad[0]:
            # Dgrad: dx = dy W, the dot product over the output features n.
            gv, gs = nvfp4_quantize(dy2, stochastic=ctx.stochastic)
            if ctx.weight_2d:
                w_t = w_saved.t()  # (k, n): the forward's quantised W, transposed
            else:  # 1x16 blocks along n: W^T quantised on its own (inconsistent with Fprop)
                w_t, ws = nvfp4_quantize(w_saved.t())
            dx = _gemm(gv, w_t, gs * ws, gd).reshape(*x_hp.shape[:-1], k).to(ctx.x_dtype)
        if ctx.needs_input_grad[1]:
            # Wgrad: dW = dy^T x, the dot product over the tokens: both operands blocked along the tokens,
            # after the random Hadamard transform.
            x2 = x_hp.reshape(-1, k)
            if ctx.rht:
                g_t, x_t = rht_tokens(dy2, transform), rht_tokens(x2, transform)
            else:
                g_t, x_t = tokens_blocked(dy2), tokens_blocked(x2)
            gv, gs = nvfp4_quantize(g_t, stochastic=ctx.stochastic)
            xv, xs = nvfp4_quantize(x_t)
            dw = _gemm(gv, xv, gs * xs, gd).to(ctx.w_dtype)
        return dx, dw, None, None, None, None, None


class Quartet2LinearFunction(torch.autograd.Function):
    """y = x W^T with the Quartet II scheme (NVFP4Config.recipe "quartet2"; the module docstring): Fprop on the
    four-over-six RTN quantisation of x and W (1x16 blocks along the dot product); Dgrad and Wgrad on the MS-EDEN
    re-quantisation of the gradient and of the forward's quantised W and x, along each GEMM's dot product, after one
    random Hadamard transform per backward call (fresh signs from the global generator, then the stochastic scale
    roundings, in this order: the signs, dy and W^T for Dgrad, dy^T and x^T for Wgrad).

    hadamard: the (128 x 128) normalised Sylvester matrix H / sqrt(128) (the module's buffer q2_hadamard); the
    transform of a call is diag(s) hadamard."""

    @staticmethod
    def forward(ctx, x, weight, hadamard, gemm_dtype):
        k = x.shape[-1]
        xv, xs = quartet2_rtn_quantize(x.reshape(-1, k))
        wv, ws = quartet2_rtn_quantize(weight)
        y = _gemm(xv, wv, xs * ws, gemm_dtype)
        # Saved: the forward's quantised operands (exact in the GEMM dtype; the same memory as nn.Linear's bf16 copy
        # of x) and their tensor decode scales. The backward re-quantises THESE (the source saves input_fp4 and
        # weight_fp4): v is rotated and quantised, and its tensor scale joins the GEMM's output scale (the rotation is
        # linear, so this is the dequantised operand's re-quantisation without one more rounding).
        ctx.save_for_backward(xv.to(gemm_dtype), xs, wv.to(gemm_dtype), ws, hadamard)
        ctx.gemm_dtype, ctx.x_dtype, ctx.w_dtype, ctx.x_shape = gemm_dtype, x.dtype, weight.dtype, x.shape
        return y.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, dy):
        xv, xs, wv, ws, hadamard = ctx.saved_tensors
        gd = ctx.gemm_dtype
        n, k = wv.shape
        dy2 = dy.to(gd).reshape(-1, n)
        transform = quartet2_signs(dy2, hadamard.shape[0])[:, None] * hadamard.float()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            # Dgrad: dx = dy W, the dot product over the output features n: dy (M, n) and W^T (k, n) along n.
            gv, gs = quartet2_eden_quantize(dy2, transform)
            wtv, wts = quartet2_eden_quantize(wv.t(), transform)
            dx = _gemm(gv, wtv, gs * wts * ws, gd).reshape(*ctx.x_shape[:-1], k).to(ctx.x_dtype)
        if ctx.needs_input_grad[1]:
            # Wgrad: dW = dy^T x, the dot product over the tokens: dy^T (n, M) and x^T (k, M) along the tokens.
            gtv, gts = quartet2_eden_quantize(dy2.t(), transform)
            xtv, xts = quartet2_eden_quantize(xv.reshape(-1, k).t(), transform)
            dw = _gemm(gtv, xtv, gts * xts * xs, gd).to(ctx.w_dtype)
        return dx, dw, None, None


class NVFP4Linear(nn.Linear):
    """nn.Linear with the same parameters and state dict whose GEMMs run on NVFP4 operands while
    `nvfp4_active` and the module is in training mode. Made by `apply_nvfp4`, which turns every linear
    layer of the chosen layers into this class (inactive ones compute F.linear), so in eval mode every
    layer of a kind has the same structure for torch.compile. In eval mode it is F.linear: the plain
    layer, bit for bit.

    Attributes: nvfp4_active, nvfp4_stochastic, nvfp4_rht, nvfp4_weight_2d (bools), nvfp4_recipe ("nvidia",
    the attribute's absence too, or "quartet2") and the non-persistent buffer rht (the 16x16 transform); under
    "quartet2" also the non-persistent buffer q2_hadamard (H_128 / sqrt(128)); with NVFP4Config.fp8_names set,
    nvfp4_fp8 (bool; absent otherwise, which is False): an active linear named there runs the FP8 control's rowwise
    recipe (ced/model/fp8.py FP8LinearFunction) instead of NVFP4."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.nvfp4_active:
            return F.linear(x, self.weight, self.bias)
        dev = x.device.type
        dtype = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else x.dtype
        if getattr(self, "nvfp4_fp8", False):  # NVFP4Config.fp8_names: the FP8 control's GEMMs, either recipe
            y = _fp8.FP8LinearFunction.apply(x, self.weight, "rowwise", dtype)
        elif getattr(self, "nvfp4_recipe", "nvidia") == "quartet2":
            y = Quartet2LinearFunction.apply(x, self.weight, self.q2_hadamard, dtype)
        else:
            y = NVFP4LinearFunction.apply(x, self.weight, self.rht, dtype, self.nvfp4_stochastic, self.nvfp4_rht,
                                          self.nvfp4_weight_2d)
        return y if self.bias is None else y + self.bias.to(y.dtype)


# ----------------------------------------------------------------------------------------------
# which layers


def kept_layers(n_layers: int, keep_last_fraction: float, keep_first_layers: int) -> set[int]:
    """Indices of the layers whose linears stay in high precision: the first keep_first_layers and the
    last round(keep_last_fraction * n_layers) (half up)."""
    n_last = int(math.floor(keep_last_fraction * n_layers + 0.5))
    return set(range(min(keep_first_layers, n_layers))) | set(range(n_layers - min(n_last, n_layers), n_layers))


def quantised_in_layer(mode: str, kind: str, part: str) -> bool:
    """Whether the linears of `part` ("mixer" or "mlp") of a layer of `kind` ("gdn" / "attention") are
    quantised in `mode`:
    - "all_linears": every linear layer (the recipe's high-precision parts aside);
    - "all_but_gdn": every linear except those inside GDN layers, read literally: a GDN layer's mixer
      projections (in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj) and its FFN stay in BF16;
      only the global-attention layers (q/k/v/o projections and their FFN) are quantised;
    - "all_but_gdn_mixer": every linear except a GDN layer's mixer projections (its FFN is quantised);"""
    if mode == "all_linears":
        return True
    if mode == "mlp_only":
        return part == "mlp"
    if mode == "all_but_gdn":
        return kind != "gdn"
    if mode == "all_but_gdn_mixer":
        return not (kind == "gdn" and part == "mixer")
    raise ValueError(f"unknown nvfp4 mode {mode!r}; one of {MODES}")


def check_nvfp4(cfg, q=None) -> None:
    """Refuse the switch (cfg: a ModelConfig; q: the NVFP4Config, default cfg.nvfp4, with a mode other than "none")
    wherever it is not built, so that no model ignores or half-applies one it was given:
    - any arch but the non-looped control (CED v2's decoder, its memory projections and its glue were never
      quantised or tested; a looped encoder likewise);
    - the asentmax normaliser: its two scale heads are nn.Linear modules that the attention applies with F.linear on
      their weights (ced/model/attention.py asentmax_scale), so they would be listed as quantised and not be;
    - values outside their range."""
    q = cfg.nvfp4 if q is None else q
    if q.mode not in MODES:
        raise ValueError(f"model.nvfp4.mode {q.mode!r}: one of {MODES}")
    if q.recipe not in RECIPES:
        raise ValueError(f"model.nvfp4.recipe {q.recipe!r}: one of {RECIPES}")
    names = q.keep_names
    if not isinstance(names, tuple) or not all(isinstance(n, str) and n and not n.startswith(".") and not n.endswith(".")
                                               for n in names) or len(set(names)) != len(names):
        raise ValueError(f"model.nvfp4.keep_names {names!r}: distinct module-name suffixes (e.g. [in_proj_a, in_proj_b])")
    fp8 = getattr(q, "fp8_names", ())
    if not isinstance(fp8, tuple) or not all(isinstance(n, str) and n and not n.startswith(".") and not n.endswith(".")
                                             for n in fp8) or len(set(fp8)) != len(fp8):
        raise ValueError(f"model.nvfp4.fp8_names {fp8!r}: distinct module-name suffixes (e.g. [out_proj])")
    both = sorted(set(fp8) & set(names))
    if both:
        raise ValueError(f"model.nvfp4.fp8_names and keep_names both name {both!r}: a linear is kept in BF16 or runs FP8, "
                         "not both")
    hp = [n for n in fp8 if n.split(".")[-1] in _fp8.HIGH_PRECISION]
    if hp:
        raise ValueError(f"model.nvfp4.fp8_names {hp!r}: GDN's per-head projections {list(_fp8.HIGH_PRECISION)} are not "
                         "run in FP8 (ced/model/fp8.py keeps them in high precision); keep them by name instead")
    if q.recipe == "quartet2":
        changed = [k for k in ("stochastic_rounding", "rht", "weight_2d") if getattr(q, k) is not True]
        if changed:
            raise ValueError(f"model.nvfp4.recipe 'quartet2' has its own rounding, transform and blocks; "
                             f"{', '.join('model.nvfp4.' + k for k in changed)} are the nvidia recipe's switches: leave "
                             "them at their defaults")
    until = getattr(q, "until_step", 0)
    if type(until) is not int or until < 0:
        raise ValueError(f"model.nvfp4.until_step {until!r}: an integer >= 0 (0: NVFP4 for every update)")
    if not 0.0 <= q.keep_last_fraction <= 1.0:
        raise ValueError(f"model.nvfp4.keep_last_fraction {q.keep_last_fraction!r} outside [0, 1]")
    if type(q.keep_first_layers) is not int or q.keep_first_layers < 0:
        raise ValueError(f"model.nvfp4.keep_first_layers {q.keep_first_layers!r}: an integer >= 0")
    if cfg.arch != "control" or cfg.encoder.loops != 1:
        raise NotImplementedError("model.nvfp4 (NVFP4 fake quantisation) is built for the non-looped control "
                                  f"(arch 'control', encoder.loops 1), not arch {cfg.arch!r} with encoder.loops "
                                  f"{cfg.encoder.loops}")
    if cfg.encoder.attention.normalizer != "softmax":
        raise NotImplementedError("model.nvfp4 is built for the softmax normaliser, not "
                                  f"{cfg.encoder.attention.normalizer!r}")


def apply_nvfp4(model: nn.Module, cfg) -> dict:
    """Turn the linear layers of the model's layers into NVFP4Linear per cfg (an NVFP4Config), in place,
    before the optimizer is built. Parameters, their names and the state dict do not change. The token
    embedding, the output head (model.head_weight(), an embedding matrix) and everything that is not
    an nn.Linear inside a layer keep their precision. Returns a summary (also stored as
    model.nvfp4_summary). Only the non-looped control (arch "control") is supported (check_nvfp4)."""
    if cfg.mode == "none":
        return {}
    check_nvfp4(model.cfg, cfg)
    layers = list(model.encoder.layers)
    kept = kept_layers(len(layers), cfg.keep_last_fraction, cfg.keep_first_layers)
    transform = rht_matrix()
    q2 = cfg.recipe == "quartet2"
    q2_hadamard = hadamard_matrix(Q2_RHT) / math.sqrt(Q2_RHT) if q2 else None
    summary = {"mode": cfg.mode, "kept_layers": sorted(kept), "quantised": [], "high_precision": [],
               "quantised_params": 0, "linear_params": 0}
    mlp_only = cfg.mode == "mlp_only"
    until_step = getattr(cfg, "until_step", 0)
    by_name, matched = [], set()  # linears kept in BF16 by name that the mode and the layers would quantise; matches
    fp8_names = tuple(getattr(cfg, "fp8_names", ()))
    fp8_on, fp8_matched, fp8_params = [], set(), 0  # linears that run FP8 by name (NVFP4Config.fp8_names); matches
    for i, layer in enumerate(layers):
        for part in ("mixer", "mlp"):
            for name, m in getattr(layer, part).named_modules():
                if type(m) is not nn.Linear:
                    continue
                full = f"encoder.layers.{i}.{part}.{name}"
                if mlp_only and part != "mlp":
                    # mode "mlp_only": the mixers' linears stay the plain nn.Linear they are (no class change, no attribute): the
                    # mixer code of every arm (GDN write mixes, R1-entry's walk, the attention kernels) sees the module it always
                    # saw. keep_names / fp8_names match the FFN's linears only (a name that matches none is refused below).
                    summary["linear_params"] += m.weight.numel()
                    summary["high_precision"].append(full)
                    continue
                hits = {k for k in cfg.keep_names if full == k or full.endswith("." + k)}
                matched |= hits
                fp8_hits = {k for k in fp8_names if full == k or full.endswith("." + k)}
                fp8_matched |= fp8_hits
                if hits and fp8_hits:
                    raise ValueError(f"{full}: kept in BF16 by model.nvfp4.keep_names {sorted(hits)!r} and run in FP8 by "
                                     f"model.nvfp4.fp8_names {sorted(fp8_hits)!r}: one or the other")
                active = i not in kept and quantised_in_layer(cfg.mode, layer.kind, part)
                if active and hits:
                    by_name.append(full)
                    active = False
                in_fp8 = active and bool(fp8_hits)
                if in_fp8 and (m.in_features % _fp8.ALIGN or m.out_features % _fp8.ALIGN):
                    raise ValueError(f"model.nvfp4.fp8_names: {full} is {m.in_features} -> {m.out_features}, not a multiple "
                                     f"of {_fp8.ALIGN} (an FP8 GEMM's alignment)")
                m.__class__ = NVFP4Linear
                m.nvfp4_active = active
                m.nvfp4_stochastic = bool(cfg.stochastic_rounding)
                m.nvfp4_rht = bool(cfg.rht)
                m.nvfp4_weight_2d = bool(cfg.weight_2d)
                if fp8_names:  # absent without the key: the modules of every run before it, attribute for attribute
                    m.nvfp4_fp8 = in_fp8
                if until_step:  # absent without the key: whether the plan makes this linear active while the phase is on (set_phase)
                    m.nvfp4_planned = active
                m.register_buffer("rht", transform.clone(), persistent=False)
                if q2:
                    m.nvfp4_recipe = "quartet2"
                    m.register_buffer("q2_hadamard", q2_hadamard.clone(), persistent=False)
                summary["linear_params"] += m.weight.numel()
                if in_fp8:
                    fp8_on.append(full)
                    fp8_params += m.weight.numel()
                    continue
                summary["quantised" if active else "high_precision"].append(full)
                summary["quantised_params"] += m.weight.numel() if active else 0
    unmatched = [k for k in cfg.keep_names if k not in matched]
    if unmatched:
        raise ValueError(f"model.nvfp4.keep_names {unmatched!r} name no linear of the layers' mixers and FFNs (a suffix "
                         "at a dot boundary, e.g. in_proj_a or mixer.in_proj_a)")
    unmatched = [k for k in fp8_names if k not in fp8_matched]
    if unmatched:
        raise ValueError(f"model.nvfp4.fp8_names {unmatched!r} name no linear of the layers' mixers and FFNs (a suffix "
                         "at a dot boundary, e.g. out_proj or mixer.out_proj)")
    summary["quantised_param_share"] = summary["quantised_params"] / max(1, summary["linear_params"])
    if cfg.keep_names:  # absent without the key: the summary of every run before it, key for key
        summary["keep_names"] = list(cfg.keep_names)
        summary["kept_by_name"] = by_name
    if q2:  # likewise absent under the default recipe
        summary["recipe"] = cfg.recipe
    if fp8_names:  # likewise absent without fp8_names: the linears that run FP8 (rowwise) and their weights
        summary["fp8_names"] = list(fp8_names)
        summary["fp8"] = fp8_on
        summary["fp8_params"] = fp8_params
        summary["fp8_param_share"] = fp8_params / max(1, summary["linear_params"])
    if until_step:  # likewise absent without until_step: the update at which the quantised path ends
        summary["until_step"] = until_step
    model.nvfp4_summary = summary
    return summary


def set_phase(model: nn.Module, on: bool) -> int:
    """NVFP4Config.until_step: turn the quantised path of the whole model on or off in memory, and return the number of linears that changed.
    Off: every NVFP4Linear computes F.linear (what the stage "continue" with precision_switch builds, with no module class change). On: the
    linears the plan made active (nvfp4_planned) are active again, the others (kept layers, kept names) stay as they are. A model built without
    until_step has no nvfp4_planned and is left alone (0)."""
    changed = 0
    for m in model.modules():
        planned = getattr(m, "nvfp4_planned", None)
        if planned is None:
            continue
        want = bool(planned and on)
        if m.nvfp4_active != want:
            m.nvfp4_active = want
            changed += 1
    return changed
