"""Fixed residual-branch multipliers (loop residual scaling). Phase 1 decides which rule to use.

The multipliers are plain Python floats computed from the config. They are not parameters and are
not in the state dict, so they cannot drift and checkpoint resume is exact. They use a fixed
reference loop count t_ref, never the sampled T, so the shared block computes the same function
whatever T a training step draws.
"""

from __future__ import annotations

import math

from ced.config import ModelConfig


def residual_scales(cfg: ModelConfig) -> tuple[float, float]:
    """Return (encoder multiplier, decoder multiplier)."""
    rs = cfg.residual_scale
    l_enc = cfg.encoder.n_layers
    # Only Looped-CED (v1) and CED v2 have a looped decoder; the control and the second-pass control
    # are plain encoder stacks (the second pass reuses the encoder layers with their own scale).
    l_dec = cfg.decoder.n_layers if cfg.arch in ("looped_ced", "ced_v2") else 0
    t_ref = cfg.t_ref()
    enc_loops = cfg.encoder.loops
    if rs.mode == "none":
        return 1.0, 1.0
    if enc_loops > 1:
        # The encoder is the looped block (YOCO-U arm). 1/r applies to it; the decoder is unscaled.
        assert rs.mode == "inverse_loops", "only inverse_loops is defined for a looped encoder"
        return rs.lam / enc_loops, 1.0
    if rs.mode == "completep_effective_depth":
        # CompleteP (alpha = 1) applied to effective depth L_enc + t_ref * L_dec. Note that it also moves
        # the encoder multiplier when t_ref changes.
        eps = rs.lam * rs.depth_base / (l_enc + t_ref * l_dec)
        return eps, eps
    if rs.mode == "loop_factored":
        # arXiv 2606.18524: looped block branches get lam / (N * sqrt(L)), N = loop count,
        # L = unique layers in the looped block. The unlooped encoder keeps CompleteP on its own depth.
        eps_enc = rs.lam * rs.depth_base / l_enc
        eps_dec = rs.lam / (t_ref * math.sqrt(l_dec / rs.l_ref)) if l_dec else 1.0
        return eps_enc, eps_dec
    if rs.mode == "inverse_loops":
        # 1/r on the looped block (decided 2026-10-03 for the primary candidate).
        return 1.0, (rs.lam / t_ref if l_dec else 1.0)
    raise ValueError(f"unknown residual scale mode {rs.mode!r}")
