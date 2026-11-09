"""Our gated delta rule for the training step (GDNConfig.kernel = "ced").

The function is fla's `chunk_gated_delta_rule` (fla-core 0.5.2, MIT licence; Songlin Yang, Yu Zhang,
Zhiyuan Li and contributors) for our call: no initial or final state, scale 1, q and k already
L2-normalised, varlen boundaries in `cu_seqlens`, chunks of 64, grouped value heads, K = V = 128.

Two implementations (`impl`, default DEFAULT_IMPL; the environment variable CED_GDN_IMPL overrides
it per process, for A/B runs):

"fla-kernels" (step 1): fla's own Triton kernels, orchestrated differently. fla's backward
  recomputes the WY factors (w, u) and the whole chunk-state recurrence (h, v_new) from the
  forward's inputs; here the forward keeps them. Outputs and gradients are bitwise fla's.

"triton" (steps 2+, the default): our kernels (ced/kernels/gdn_triton.py): fla's per-value-head
  arithmetic with the same dot precisions, organised per key head where key-side work is shared
  (K K^T, q k^T, k q^T; dq and dk summed over the group in registers), with the reverse cumsum of
  dg fused into the WY backward. Not bitwise fla (summation order); checked against the float64
  reference and against fla's own error at production precision (tests/gpu/test_gdn_kernel_gpu.py).

`keep` (both implementations; default DEFAULT_KEEP, CED_GDN_KEEP overrides it per process) trades
memory for backward time:
  "all"    keep what the backward reads (w, h, v_new; step 1 also u);
  "states" keep h and v_new; recompute w (and u) from the stored inverse;
  "none"   keep nothing extra; recompute w, u and the state recurrence (fla's backward).

Control (c)'s second-pass read (impl "triton"): `ced_delta_rule_states` is the same call that also
returns pass 1's chunk states (h, v_new) as differentiable outputs, and `ced_read` computes S1_t^T
q2_t from them with the output kernel (q = q2). The read's backward hands dL/dh and dL/dv_new to pass
1's backward, which folds them into its one state-gradient recurrence: one state recursion forward and
one backward per layer for both passes, instead of a second full delta-rule call.

Also here: `qkv_prep`, GatedDeltaNet's training-path input preparation (short conv within documents,
SiLU, split, L2 norm, q scale) fused into one kernel forward and two backward.

F3 (`f3`, impl "triton" only; GDNConfig.decay_cumsum "compensated"): the within-chunk cumulative log decay is kept
as a float pair (gc, gl) = (hi, lo) from a TwoSum scan and every decay is formed from the pair
(ced/kernels/gdn_triton.py, docstring). One more
(1, T, H_v) array in g's dtype a call (saved for the backward with gc); no host sync, no new shape. f3 False (the
default): every launch passes gl None and F3 False, so the kernels compile to the code they always were.
"""

from __future__ import annotations

import math
import os

import torch

CHUNK = 64
KEEP_CHOICES = ("all", "states", "none")
DEFAULT_KEEP = "all"  # what the model's call uses (benchmarks set it to compare the modes)
IMPL_CHOICES = ("fla-kernels", "triton")
DEFAULT_IMPL = "triton"

# Launch configurations of our kernels (fixed: no run-time autotuning, so every process runs the same
# kernels), per input dtype (CONFIGS; float64 uses the float32 one).
CONFIG = {
    "cumsum": {"num_warps": 1},
    "intra": {"BK": 64, "GP": 2, "num_warps": 1, "num_stages": 1},
    "wu": {"GP": 1, "num_warps": 8, "num_stages": 1},
    "state_fwd": {"BV": 32, "num_warps": 2, "num_stages": 2},
    "out": {"BV": 32, "num_warps": 4, "num_stages": 1},
    "dv_local": {"BV": 64, "num_warps": 4, "num_stages": 2},
    "state_bwd": {"BV": 64, "num_warps": 4, "num_stages": 1},
    "dqkwg": {"BK": 64, "BV": 64, "num_warps": 8, "num_stages": 1},
    "wy": {"GP": 1, "num_warps": 4, "num_stages": 1},
    "dk_sum": {"BT": 16, "num_warps": 8},
    # control (c)'s second-pass read (its dL/dv_new runs dv_local_kernel with dv_local's options): swept
    # 2026-10-07 (3 rows, real 64k boundaries): 1.63 -> 1.33 ms
    "read_dqkg": {"BK": 128, "BV": 32, "num_warps": 8, "num_stages": 1},
    # pass 1's state backward with the read's gradients (EXT): swept, 1.70 -> 1.60 ms
    "state_bwd_ext": {"BV": 32, "num_warps": 4, "num_stages": 1},
}
# bf16 inputs: swept 2026-10-07 (3 rows of 8,192 with real 64k-shard
# boundaries); per kernel the fastest option, the float32 one kept where it was fastest (wy, dk_sum) or tied (cumsum).
CONFIG_BF16 = {
    "cumsum": {"num_warps": 1},
    "intra": {"BK": 128, "GP": 1, "num_warps": 1, "num_stages": 1},
    "wu": {"GP": 1, "num_warps": 4, "num_stages": 1},
    "state_fwd": {"BV": 64, "num_warps": 4, "num_stages": 3},
    "out": {"BV": 128, "num_warps": 4, "num_stages": 1},
    "dv_local": {"BV": 128, "num_warps": 2, "num_stages": 1},
    "state_bwd": {"BV": 32, "num_warps": 4, "num_stages": 2},
    "dqkwg": {"BK": 128, "BV": 64, "num_warps": 8, "num_stages": 2},
    "wy": {"GP": 1, "num_warps": 4, "num_stages": 1},
    "dk_sum": {"BT": 16, "num_warps": 8},
    "read_dqkg": {"BK": 128, "BV": 64, "num_warps": 8, "num_stages": 2},  # swept with the float32 one: best
    "state_bwd_ext": {"BV": 32, "num_warps": 4, "num_stages": 2},  # swept: state_bwd's option is the best
}
CONFIGS = {"float32": CONFIG, "bfloat16": CONFIG_BF16}


def config_for(dtype: torch.dtype) -> dict:
    """The launch configurations for inputs of this dtype (float64: the float32 ones)."""
    return CONFIGS["bfloat16"] if dtype == torch.bfloat16 else CONFIGS["float32"]


def _impl(impl: str | None) -> str:
    impl = impl or os.environ.get("CED_GDN_IMPL") or DEFAULT_IMPL
    assert impl in IMPL_CHOICES, impl
    return impl


# ------------------------------------------------------------------------------------------------
# step 1: fla's kernels, the forward's w, u, h, v_new kept


def _fla():
    """fla's sub-kernels (imported lazily: fla needs CUDA/Triton, the CPU suite does not)."""
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.index import prepare_chunk_indices

    return locals()


def _fla_function():
    from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

    f = _fla()

    class GatedDeltaRule(torch.autograd.Function):
        @staticmethod
        @input_guard
        @autocast_custom_fwd
        def forward(ctx, q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu, keep):
            chunk_indices = f["prepare_chunk_indices"](cu_seqlens, CHUNK, cu_seqlens_cpu=cu_seqlens_cpu)
            g = f["chunk_local_cumsum"](g, chunk_size=CHUNK, scale=f["RCP_LN2"], cu_seqlens=cu_seqlens,
                                        chunk_indices=chunk_indices)
            w, u, A = f["chunk_gated_delta_rule_fwd_intra"](k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens,
                                                            chunk_indices=chunk_indices, chunk_size=CHUNK)
            h, v_new, _ = f["chunk_gated_delta_rule_fwd_h"](k=k, w=w, u=u, g=g, initial_state=None,
                                                            output_final_state=False, cu_seqlens=cu_seqlens,
                                                            chunk_indices=chunk_indices, chunk_size=CHUNK)
            o = f["chunk_fwd_o"](q=q, k=k, v=v_new, h=h, g=g, scale=1.0, cu_seqlens=cu_seqlens,
                                 chunk_indices=chunk_indices, chunk_size=CHUNK)
            kept = {"all": (w, u, h, v_new), "states": (None, None, h, v_new), "none": (None, None, None, None)}[keep]
            ctx.save_for_backward(q, k, v, g, beta, A, cu_seqlens, chunk_indices, *kept)
            return o.to(q.dtype)

        @staticmethod
        @input_guard
        @autocast_custom_bwd
        def backward(ctx, do):
            q, k, v, g, beta, A, cu_seqlens, chunk_indices, w, u, h, v_new = ctx.saved_tensors
            kw = {"cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices}
            if w is None:
                w, u = f["recompute_w_u_fwd"](k=k, v=v, beta=beta, A=A, g=g, **kw)
            if h is None:
                h, v_new, _ = f["chunk_gated_delta_rule_fwd_h"](k=k, w=w, u=u, g=g, initial_state=None,
                                                                output_final_state=False, chunk_size=CHUNK, **kw)
            dv = f["chunk_bwd_dv_local"](q=q, k=k, g=g, do=do, scale=1.0, chunk_size=CHUNK, **kw)
            dh, _, dv = f["chunk_gated_delta_rule_bwd_dhu"](q=q, k=k, w=w, g=g, h0=None, dht=None, do=do, dv=dv,
                                                            scale=1.0, chunk_size=CHUNK, **kw)
            dq, dk, dw, dg = f["chunk_bwd_dqkwg"](q=q, k=k, v=v_new, w=w, g=g, h=h, dv=dv, do=do, dh=dh, scale=1.0,
                                                  chunk_size=CHUNK, **kw)
            dk2, dv, db, dg2 = f["prepare_wy_repr_bwd"](k=k, v=v, beta=beta, g=g, A=A, dw=dw, du=dv, **kw)
            dk.add_(dk2)
            dg.add_(dg2)
            dg = f["chunk_local_cumsum"](dg, chunk_size=CHUNK, reverse=True, **kw)
            return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, None, None

    return GatedDeltaRule


# ------------------------------------------------------------------------------------------------
# steps 2+: our kernels


def chunk_index(cu_seqlens: torch.Tensor, cu_seqlens_cpu: torch.Tensor | None, chunk: int = CHUNK):
    """Host side, from the CPU copy of the boundaries: chunk_indices (NT, 2) = (segment, chunk in
    segment) per global chunk and chunk_offsets (N + 1) = each segment's first global chunk, moved to
    the device in one copy. Returns (chunk_indices, chunk_offsets, NT)."""
    cpu = (cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.cpu()).long()
    counts = (cpu[1:] - cpu[:-1] + chunk - 1) // chunk
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    nt = int(offsets[-1])
    seg = torch.repeat_interleave(torch.arange(len(counts)), counts)
    idx = torch.arange(nt) - offsets[:-1][seg]
    host = torch.cat([torch.stack([seg, idx], 1).reshape(-1), offsets]).to(torch.int32)
    dev = host.to(cu_seqlens.device, non_blocking=True)
    return dev[: 2 * nt].view(nt, 2), dev[2 * nt:], nt


def fixed_chunk_index(cu_full, nt_max: int, chunk: int = CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """Host side, CUDA graphs (TrainConfig.cuda_graphs; ced/train/cuda_graphs.py): chunk_index at a FIXED size.

    cu_full (S + 2,) int: a micro-batch's boundaries at a fixed capacity (ced/train/varlen.py fixed_varlen_host): its
    real segments, then empty ones (every entry the token count T) up to S, then one more empty segment, the sentinel,
    whose index is S. Returns (chunk_indices (nt_max, 2), chunk_offsets (S + 2,)), int32 host tensors: the first nt
    rows of chunk_indices are chunk_index's rows exactly (an empty segment has no chunk), the rows after them are
    (S, 0), the sentinel's chunk 0; chunk_offsets is chunk_index's over cu_full. With these and grids of nt_max chunks
    (S + 1 segments) every kernel's program of a real chunk or segment computes what it computes over the exact
    boundaries, and every other program reads the sentinel (bos = T, length 0): its loads and stores are all masked
    (ced/kernels/gdn_triton.py: every per-token access is masked by o_t < T; the chunk-state loops run 0 times), so the
    outputs and gradients are bit for bit the exact call's (a GPU run checks it).
    In numpy (integers, the same values as chunk_index's torch ops): a few small torch ops on a busy host's thread pool
    cost milliseconds (measured on the control VM: 9 ms at 4 threads, 0.1 ms at 1), and this runs once a micro-batch."""
    import numpy as np

    cpu = np.asarray(cu_full, dtype=np.int64)
    counts = (cpu[1:] - cpu[:-1] + chunk - 1) // chunk
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)])
    nt = int(offsets[-1])
    if nt > nt_max:
        raise ValueError(f"{nt} chunks of {chunk} in this micro-batch, more than the fixed capacity {nt_max}")
    seg = np.repeat(np.arange(len(counts)), counts)
    idx = np.arange(nt) - offsets[:-1][seg]
    ci = np.zeros((nt_max, 2), dtype=np.int32)
    ci[:, 0] = len(counts) - 1  # the sentinel segment (the last of cu_full's segments)
    ci[:nt, 0], ci[:nt, 1] = seg, idx
    return torch.from_numpy(ci), torch.from_numpy(offsets.astype(np.int32))


def _gp(cfgs, name, G: int) -> int:
    """The launch option GP of kernel `name` (value heads per program) for G value heads per key head.
    A program's GP value heads share one key head (the kernels index it as (i_p * GP) // G), so GP is
    cut to a divisor of G: gcd(GP, G). With G = 2 every option (GP 1 or 2) is used as given; with G = 1
    (HV == H) a GP of 2 would pair two value heads of different key heads (review 2026-10-07)."""
    return math.gcd(cfgs[name]["GP"], G)


def _launch(kernel, grid, cfgs, name, *args, **kwargs):
    cfg = {k: v for k, v in cfgs[name].items() if k != "u_prec"}
    if "GP" in cfg:  # the grid's program count per chunk must use the same value (callers: _gp)
        cfg["GP"] = _gp(cfgs, name, kwargs["G"])
    kernel[grid](*args, **kwargs, **cfg)


def _triton_function():
    import triton.language as tl

    from ced.kernels import gdn_triton as K_

    def acc_dtypes(x: torch.Tensor):
        return (tl.float64, torch.float64) if x.dtype == torch.float64 else (tl.float32, torch.float32)

    def u_prec(x: torch.Tensor) -> str:
        # fla: recompute_w_u's u dot is allow_tf32=False (IEEE float32); bf16 / float64 ignore it
        return CONFIG["wu"].get("u_prec", "ieee") if x.dtype == torch.float32 else "tf32"

    def recompute_wu(k, v, gc, gl, beta, A, cu, ci, nt, H, G, need_u=True):
        cf = config_for(k.dtype)
        w = torch.empty(*v.shape[:3], k.shape[-1], dtype=k.dtype, device=k.device)
        u = torch.empty_like(v)
        _launch(K_.wu_fwd_kernel, (nt, H * G // _gp(cf, "wu", G)), cf, "wu", k, v, gc, gl, beta, A, w, u, cu, ci, v.shape[1],
                H=H, G=G, K=k.shape[-1], V=v.shape[-1], BT=CHUNK, U_PREC=u_prec(k), F3=gl is not None)
        return w, u

    def state_fwd(k, w, u, gc, gl, cu, co, nt, H, G, ACC):
        cf = config_for(k.dtype)
        T, HV, K, V = k.shape[1], u.shape[2], k.shape[-1], u.shape[-1]
        h = torch.empty(1, nt, HV, K, V, dtype=k.dtype, device=k.device)
        v_new = torch.empty_like(u)
        BV = cf["state_fwd"]["BV"]
        _launch(K_.state_fwd_kernel, (V // BV, (len(cu) - 1) * HV), cf, "state_fwd", k, u, w, v_new, gc, gl, h, cu, co, T,
                H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=gl is not None)
        return h, v_new

    def decay_cumsum(g, cu, ci, nt, T, HV, cf, ACC, f3):
        """(gc, gl): the within-chunk cumulative log decay; gl None unless f3 (then (gc, gl) is F3's pair (hi, lo) in
        natural units, see gdn_triton.cumsum_fwd_kernel). The one place both GatedDeltaRule and GatedDeltaRuleRead
        form it, so the read's pair is bitwise pass 1's. cf: the launch options of the call (k's dtype)."""
        gc = torch.empty_like(g)
        gl = torch.empty_like(g) if f3 else None
        _launch(K_.cumsum_fwd_kernel, (nt, HV), cf, "cumsum", g, gc, gl, cu, ci, T, HV=HV, BT=CHUNK, ACC=ACC, F3=bool(f3))
        return gc, gl

    class GatedDeltaRule(torch.autograd.Function):
        """Returns (o, h, v_new): the output, and the chunk states (h: the state at every chunk's start,
        (1, NT, HV, K, V); v_new (1, T, HV, V)) as differentiable outputs, so that control (c)'s
        second-pass read (GatedDeltaRuleRead) can reuse them; their gradients (from the read) join this
        backward's state-gradient recurrence (state_bwd_kernel EXT). Callers that only need o drop h
        and v_new (ced_delta_rule): their gradients are then None and the backward is the plain one."""

        @staticmethod
        def forward(ctx, q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu, keep, f3, chunks=None):
            ctx.set_materialize_grads(False)
            q, k, v, beta = (x.contiguous() for x in (q, k, v, beta))
            B, T, H, K = k.shape
            HV, V = v.shape[2], v.shape[3]
            assert B == 1 and K == 128 and V == 128 and HV in (H, 2 * H) and q.dtype == k.dtype == v.dtype, (k.shape, v.shape)
            G = HV // H
            ACC, acc_t = acc_dtypes(k)
            cu = cu_seqlens.to(torch.int32) if cu_seqlens.dtype != torch.int32 else cu_seqlens
            # chunks: CUDA graphs' fixed-capacity index (chunk_indices, chunk_offsets, nt_max) on the device, made by
            # fixed_chunk_index (no host read, no host-dependent grid); None: the exact index from the host boundaries
            ci, co, nt = chunk_index(cu_seqlens, cu_seqlens_cpu) if chunks is None else chunks
            g_dtype = g.dtype
            g = g.to(acc_t).contiguous()
            cf = config_for(k.dtype)
            gc, gl = decay_cumsum(g, cu, ci, nt, T, HV, cf, ACC, f3)  # gl None unless F3
            F3 = gl is not None
            A = torch.empty(1, T, HV, CHUNK, dtype=k.dtype, device=k.device)
            _launch(K_.intra_fwd_kernel, (nt, HV // _gp(cf, "intra", G)), cf, "intra", k, gc, gl, beta, A, cu, ci, T,
                    H=H, G=G, K=K, BT=CHUNK, BC=16, ACC=ACC, SOLVE_PREC=K_.SOLVE_PRECISION, F3=F3)
            w, u = recompute_wu(k, v, gc, gl, beta, A, cu, ci, nt, H, G)
            h, v_new = state_fwd(k, w, u, gc, gl, cu, co, nt, H, G, ACC)
            o = torch.empty_like(v_new)
            _launch(K_.out_fwd_kernel, (nt, H), cf, "out", q, k, v_new, h, gc, gl, o, cu, ci, T,
                    H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=F3)
            kept = {"all": (w, h, v_new), "states": (None, h, v_new), "none": (None, None, None)}[keep]
            ctx.save_for_backward(q, k, v, gc, beta, A, cu, ci, co, *kept, gl)
            ctx.nt, ctx.g_dtype = nt, g_dtype
            return o.to(q.dtype), h, v_new

        @staticmethod
        def backward(ctx, do, dh_ext, dv_ext):
            q, k, v, gc, beta, A, cu, ci, co, w, h, v_new, gl = ctx.saved_tensors
            F3 = gl is not None
            nt = ctx.nt
            B, T, H, K = k.shape
            HV, V = v.shape[2], v.shape[3]
            G = HV // H
            ACC, acc_t = acc_dtypes(k)
            cf = config_for(k.dtype)
            do = torch.zeros(1, T, HV, V, dtype=q.dtype, device=q.device) if do is None else do.contiguous()
            if w is None or h is None:
                w, u = recompute_wu(k, v, gc, gl, beta, A, cu, ci, nt, H, G)
                if h is None:
                    h, v_new = state_fwd(k, w, u, gc, gl, cu, co, nt, H, G, ACC)
            ext = dh_ext is not None or dv_ext is not None
            if ext:  # the second-pass read's gradients wrt h and v_new (GatedDeltaRuleRead)
                dh_ext = torch.zeros_like(h) if dh_ext is None else dh_ext.contiguous()
                dv_ext = torch.zeros_like(v_new) if dv_ext is None else dv_ext.contiguous()
            dv_local = torch.empty_like(do)
            _launch(K_.dv_local_kernel, (nt, H), cf, "dv_local", q, k, gc, gl, do, dv_local, cu, ci, T,
                    H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=F3)
            dh = torch.empty(1, nt, HV, K, V, dtype=q.dtype, device=q.device)
            dv2 = torch.empty_like(dv_local)
            sb = "state_bwd_ext" if ext else "state_bwd"
            BV = cf[sb]["BV"]
            _launch(K_.state_bwd_kernel, (V // BV, (len(cu) - 1) * HV), cf, sb, q, k, w, gc, gl, do, dv_local, dv2, dh,
                    dh_ext if ext else dh, dv_ext if ext else dv_local, cu, co, T, H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC,
                    EXT=ext, F3=F3)
            dq = torch.empty_like(q)
            dk_part = torch.empty(1, T, H, K, dtype=acc_t, device=k.device)
            dw = torch.empty_like(w)
            dg_part = torch.empty(1, T, HV, dtype=acc_t, device=k.device)
            _launch(K_.dqkwg_kernel, (nt, H), cf, "dqkwg", q, k, v_new, gc, gl, h, do, dh, dv2, dq, dk_part, dw, dg_part,
                    cu, ci, T, H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=F3)
            dk_wy = torch.empty(1, T, HV, K, dtype=acc_t, device=k.device)
            dv = torch.empty_like(v)
            db = torch.empty_like(beta)
            dg = torch.empty(1, T, HV, dtype=acc_t, device=k.device)
            _launch(K_.wy_bwd_kernel, (nt, HV // _gp(cf, "wy", G)), cf, "wy", k, v, beta, gc, gl, A, dw, dv2, dg_part, dk_wy,
                    dv, db, dg, cu, ci, T, H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=F3)
            dk = torch.empty_like(k)
            BTs = cf["dk_sum"]["BT"]
            _launch(K_.dk_sum_kernel, ((T + BTs - 1) // BTs, H), cf, "dk_sum", dk_part, dk_wy, dk, T, H=H, G=G, K=K, ACC=ACC)
            return dq, dk, dv, dg.to(ctx.g_dtype), db, None, None, None, None, None

    class GatedDeltaRuleRead(torch.autograd.Function):
        """Control (c)'s second-pass read over pass 1's chunk states: o2 = (q2 h) e^g + (q2 k^T * decay,
        causal) v_new, i.e. at every position t the state after t applied to q2_t, S1_t^T q2_t; nothing
        is written. The forward is the output kernel with q = q2; the backward gives dq2, the read's own
        dk and dg (the intra-chunk term and the decays), and the gradients wrt h and v_new, which pass
        1's backward folds into its one state-gradient recurrence."""

        @staticmethod
        def forward(ctx, q2, k, g, h, v_new, cu_seqlens, cu_seqlens_cpu, f3):
            q2, k = q2.contiguous(), k.contiguous()
            B, T, H, K = k.shape
            HV, V = v_new.shape[2], v_new.shape[3]
            assert B == 1 and q2.shape == k.shape and q2.dtype == k.dtype == h.dtype == v_new.dtype, (q2.shape, k.shape, q2.dtype, h.dtype)
            G = HV // H
            ACC, acc_t = acc_dtypes(k)
            cu = cu_seqlens.to(torch.int32) if cu_seqlens.dtype != torch.int32 else cu_seqlens
            ci, co, nt = chunk_index(cu_seqlens, cu_seqlens_cpu)
            assert h.shape == (1, nt, HV, K, V) and v_new.shape == (1, T, HV, V), (h.shape, v_new.shape, nt)
            cf = config_for(k.dtype)
            g_dtype = g.dtype
            g = g.to(acc_t).contiguous()
            # the same kernel on the same g as pass 1's: bitwise pass 1's gc (and, with F3, its gl); F3 must be pass 1's
            gc, gl = decay_cumsum(g, cu, ci, nt, T, HV, cf, ACC, f3)
            F3 = gl is not None
            o = torch.empty_like(v_new)
            _launch(K_.out_fwd_kernel, (nt, H), cf, "out", q2, k, v_new, h, gc, gl, o, cu, ci, T,
                    H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=F3)
            ctx.save_for_backward(q2, k, gc, h, v_new, cu, ci, gl)
            ctx.nt, ctx.g_dtype = nt, g_dtype
            return o.to(q2.dtype)

        @staticmethod
        def backward(ctx, do):
            q2, k, gc, h, v_new, cu, ci, gl = ctx.saved_tensors
            F3 = gl is not None
            nt = ctx.nt
            do = do.contiguous()
            B, T, H, K = k.shape
            HV, V = v_new.shape[2], v_new.shape[3]
            G = HV // H
            ACC, acc_t = acc_dtypes(k)
            cf = config_for(k.dtype)
            # dL/dv_new: pass 1's dv_local arithmetic with q = q2; stored in v_new's dtype (autograd casts a
            # gradient to its tensor's dtype anyway), added in float32 inside pass 1's state backward
            dv_ext = torch.empty_like(v_new)
            _launch(K_.dv_local_kernel, (nt, H), cf, "dv_local", q2, k, gc, gl, do, dv_ext, cu, ci, T,
                    H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=F3)
            dq2 = torch.empty_like(q2)
            dk = torch.empty_like(k)
            dg = torch.empty(1, T, HV, dtype=acc_t, device=k.device)
            dh_ext = torch.empty_like(h)
            _launch(K_.read_dqkg_kernel, (nt, H), cf, "read_dqkg", q2, k, v_new, gc, gl, h, do, dq2, dk, dg, dh_ext, cu, ci, T,
                    H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=F3)
            return dq2, dk, dg.to(ctx.g_dtype), dh_ext, dv_ext, None, None, None

    return GatedDeltaRule, GatedDeltaRuleRead


_FN: dict = {}


def _functions(impl: str):
    """(the delta-rule Function, the read Function or None) of an implementation."""
    if impl not in _FN:
        _FN[impl] = (_fla_function(), None) if impl == "fla-kernels" else _triton_function()
    return _FN[impl]


def _check_f3(f3, impl: str) -> bool:
    """F3 is an option of our kernels (impl "triton") only; fla's kernels have no compensated sum."""
    if not isinstance(f3, bool):
        raise TypeError(f"f3 must be a bool, not {f3!r}")
    if f3 and impl != "triton":
        raise ValueError(f"F3 (the compensated cumulative sum, GDNConfig.decay_cumsum 'compensated') is an option of impl "
                         f"'triton' only; this call runs impl {impl!r} (CED_GDN_IMPL)")
    return f3


def ced_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                   cu_seqlens: torch.Tensor, cu_seqlens_cpu: torch.Tensor | None = None, keep: str | None = None,
                   impl: str | None = None, f3: bool = False, chunks: tuple | None = None) -> torch.Tensor:
    """Gated delta rule from an empty state over the flattened varlen sequence.

    q, k: (1, T, H_k, d_k), q L2-normalised and scaled, k L2-normalised; v: (1, T, H_v, d_v);
    g: (1, T, H_v) float32 log decay (float64 with float64 q); beta: (1, T, H_v) in q's dtype;
    cu_seqlens (long). Returns
    o (1, T, H_v, d_v) in q's dtype: fla's `chunk_gated_delta_rule(..., scale=1.0)[0]`.
    f3: F3, the compensated cumulative sum of the decays (impl "triton" only; module docstring).
    chunks: CUDA graphs' fixed-capacity metadata (impl "triton" only): (chunk_indices (nt_max, 2) int32, chunk_offsets
    (S + 2,) int32, nt_max) on the device, from fixed_chunk_index over the fixed-capacity cu_seqlens given here; the
    kernels then launch over nt_max chunks and S + 1 segments, the same values as the exact call (fixed_chunk_index).
    None (every call before CUDA graphs): the index is built from the host boundaries, as always."""
    keep = keep or os.environ.get("CED_GDN_KEEP") or DEFAULT_KEEP  # the environment variable: A/B runs
    assert keep in KEEP_CHOICES, keep
    impl = _impl(impl)
    f3 = _check_f3(f3, impl)
    if impl == "fla-kernels":
        if chunks is not None:
            raise ValueError("the fixed-capacity chunk index (CUDA graphs) is built for impl 'triton', not 'fla-kernels' "
                             "(fla's own index is made on the host)")
        return _functions(impl)[0].apply(q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu, keep)
    if chunks is None:  # the call as it always was
        return _functions(impl)[0].apply(q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu, keep, f3)[0]
    return _functions(impl)[0].apply(q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu, keep, f3, chunks)[0]


def read_supported(impl: str | None = None) -> bool:
    """Whether this implementation has the second-pass read over pass 1's chunk states (ced_delta_rule_states
    + ced_read): impl "triton" (the default). "fla-kernels" does not; control (c) then makes one more full
    delta-rule call per GDN layer instead (ced/model/gdn.py)."""
    return _impl(impl) == "triton"


def ced_delta_rule_states(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                          cu_seqlens: torch.Tensor, cu_seqlens_cpu: torch.Tensor | None = None, keep: str | None = None,
                          f3: bool = False):
    """ced_delta_rule (impl "triton") that also returns pass 1's chunk states for control (c)'s
    second-pass read: (o, h, v_new), h (1, NT, H_v, d_k, d_v) the state at the start of every chunk of 64
    (chunks never cross a boundary of cu_seqlens), v_new (1, T, H_v, d_v), both in q's dtype and
    differentiable (their gradients join this call's backward). keep "none" is promoted to "states":
    the read holds h and v_new anyway, so recomputing them in the backward would only cost time.
    f3: F3 (module docstring); the read over these states (ced_read) must be given the same f3."""
    keep = keep or os.environ.get("CED_GDN_KEEP") or DEFAULT_KEEP
    assert keep in KEEP_CHOICES, keep
    assert read_supported(), "the second-pass read needs impl 'triton'"
    f3 = _check_f3(f3, "triton")
    return _functions("triton")[0].apply(q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu, "states" if keep == "none" else keep, f3)


def ced_read(q2: torch.Tensor, k: torch.Tensor, g: torch.Tensor, h: torch.Tensor, v_new: torch.Tensor,
             cu_seqlens: torch.Tensor, cu_seqlens_cpu: torch.Tensor | None = None, f3: bool = False) -> torch.Tensor:
    """Control (c)'s second-pass read: at every position t, S1_t^T q2_t, where S1_t is the state after t of
    the delta rule that ced_delta_rule_states ran (same k, g, boundaries; its h and v_new). q2 (1, T, H_k, d_k)
    in k's dtype (zero where nothing reads); g as given to ced_delta_rule_states. Returns (1, T, H_v, d_v).
    f3: as given to ced_delta_rule_states (F3: the read forms its decays from the same compensated pair)."""
    f3 = _check_f3(f3, "triton")
    return _functions("triton")[1].apply(q2, k, g, h, v_new, cu_seqlens, cu_seqlens_cpu, f3)


# ------------------------------------------------------------------------------------------------
# The GDN input preparation (short conv within documents, SiLU, split, L2 norm of q and k, q scale),
# fused: one kernel forward, one backward (ced/kernels/gdn_triton.py prep_*_kernel).

# Launch configurations per output dtype of q, k, v (float64: the float32 ones); bench_gdn_layer.py --sweep-prep.
PREP_CONFIG = {"fwd": {"BT": 16, "num_warps": 2}, "da": {"BT": 32, "num_warps": 2}, "dx": {"BT": 32, "num_warps": 4}}
# bf16 output (the same sweep): only the forward changes (da and dx were fastest at the float32 options)
PREP_CONFIG_BF16 = {"fwd": {"BT": 32, "num_warps": 4}, "da": {"BT": 32, "num_warps": 2}, "dx": {"BT": 32, "num_warps": 4}}
PREP_CONFIGS = {"float32": PREP_CONFIG, "bfloat16": PREP_CONFIG_BF16}


def prep_config_for(dtype: torch.dtype) -> dict:
    return PREP_CONFIGS["bfloat16"] if dtype == torch.bfloat16 else PREP_CONFIGS["float32"]


class _QKVPrep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, doc, w, hk, hv, d, out_dtype):
        import triton.language as tl

        from ced.kernels import gdn_triton as K_

        B, L, C = x.shape
        assert C == (2 * hk + hv) * d and d == 128, (x.shape, hk, hv, d)
        x, doc = x.contiguous(), doc.contiguous()
        comp = torch.promote_types(x.dtype, w.dtype)  # the computation's dtype (the PyTorch code's result)
        acc = tl.float64 if comp == torch.float64 else tl.float32
        out = out_dtype or comp  # stored in out_dtype: rounded once, from the float32 result
        q = torch.empty(B, L, hk, d, dtype=out, device=x.device)
        k = torch.empty(B, L, hk, d, dtype=out, device=x.device)
        v = torch.empty(B, L, hv, d, dtype=out, device=x.device)
        cfgs = prep_config_for(out)
        cfg = cfgs["fwd"]
        grid = ((B * L + cfg["BT"] - 1) // cfg["BT"], 2 * hk + hv)
        K_.prep_fwd_kernel[grid](x, doc, w, q, k, v, B * L, L, C=C, HK=hk, HV=hv, D=d, KC=w.shape[1], ACC=acc,
                                 Q_SCALE=d**-0.5, EPS=1e-6, **cfg)
        ctx.save_for_backward(x, doc, w)
        ctx.dims = (hk, hv, d)
        ctx.cfgs = cfgs
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        import triton.language as tl

        from ced.kernels import gdn_triton as K_

        x, doc, w = ctx.saved_tensors
        hk, hv, d = ctx.dims
        B, L, C = x.shape
        comp = torch.promote_types(x.dtype, w.dtype)
        acc_t = torch.float64 if comp == torch.float64 else torch.float32
        acc = tl.float64 if comp == torch.float64 else tl.float32
        dq, dk, dv = (t.contiguous() for t in (dq, dk, dv))  # in the outputs' dtype; read into float32
        cfg = ctx.cfgs["da"]
        nb = (B * L + cfg["BT"] - 1) // cfg["BT"]
        da = torch.empty(B, L, C, dtype=acc_t, device=x.device)
        dw_part = torch.empty(nb, w.shape[1], C, dtype=acc_t, device=x.device)
        K_.prep_da_kernel[(nb, 2 * hk + hv)](x, doc, w, dq, dk, dv, da, dw_part, B * L, L, C=C, HK=hk, HV=hv, D=d,
                                             KC=w.shape[1], ACC=acc, Q_SCALE=d**-0.5, EPS=1e-6, **cfg)
        dx = torch.empty_like(x)
        cfg = ctx.cfgs["dx"]
        K_.prep_dx_kernel[((B * L + cfg["BT"] - 1) // cfg["BT"], C // d)](da, doc, w, dx, B * L, L, C=C, D=d,
                                                                          KC=w.shape[1], ACC=acc, **cfg)
        return dx, None, dw_part.sum(0).t().to(w.dtype), None, None, None, None


def qkv_prep(x: torch.Tensor, doc: torch.Tensor, conv_weight: torch.Tensor, hk: int, hv: int, d: int,
             out_dtype: torch.dtype | None = None):
    """GatedDeltaNet's training-path input preparation, fused (state None, no padding): x (B, L, C) the
    in_proj_qkv output, doc (B, L) document ids, conv_weight (C, K). Returns q (scaled), k, v as
    contiguous (B, L, heads, d) tensors, computed in promote(x, conv_weight)'s dtype as the PyTorch code
    and stored in out_dtype (None: that dtype; bf16 for GDNConfig.input_dtype "bfloat16"): the same
    values as the PyTorch code followed by a cast to out_dtype."""
    return _QKVPrep.apply(x, doc, conv_weight, hk, hv, d, out_dtype)
