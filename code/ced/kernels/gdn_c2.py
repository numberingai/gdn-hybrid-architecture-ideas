"""C2: the GDN layer's input preparation and delta rule with q, k and v formed once per chunk
(GDNConfig.edges "once").

What "once" computes: the function of kernel "ced" with its fused input preparation (ced.kernels.gdn.qkv_prep followed by
ced.kernels.gdn.ced_delta_rule), forward and backward, with today's statements, in fewer kernels:

- forward: today's cumsum_fwd_kernel; then front_fwd_kernel (ced/kernels/gdn_c2_triton.py), the first kernel that reads
  a chunk: per (chunk, key head) it forms q, k and v from the in_proj_qkv output with qkv_prep's statements and stores
  them once (they are what the state recurrence, the output kernel and the backward read), then does
  intra_fwd_kernel's and wu_fwd_kernel's work on them (K K^T once per key head, the solve, A, w, u); then today's
  state_fwd_kernel and out_fwd_kernel. Three launches fewer than today (prep_fwd, intra, wu -> front), and no HBM
  round trip of k and v between the preparation and their first readers.
- backward: today's dv_local, state_bwd and dqkwg kernels; then wy_dk_kernel (the WY backward with dk_sum folded in:
  the float32 WY part of dk is never written); then qkv_prep's backward (prep_da_kernel with da in float32,
  prep_dx_kernel, the conv weight's gradient by the same dw_part.sum(0)). One launch fewer than today.
- saved for the backward: what today's two functions save (u, doc, the conv weight, q, k, v, the cumulative decay,
  beta, A, w, h, v_new); h at the chunk capacity.
- the varlen metadata on the device at a capacity fixed by the shapes (ced.kernels.gdn_edges.chunk_meta; no host read),
  one torch.library custom op each way (ced::gdn_c2_fwd, ced::gdn_c2_bwd), called where today's two opaque calls are,
  with today's graph breaks (ced/model/gdn.py), so the compiled graphs around it are today's.

Numerics: every value is today's statement on today's operands (the CPU emulation and the interpreter give today's
path bit for bit); a GPU can round the solve's sums and products, the L2 norm's row sum and, with float32 kernel inputs,
w / u's products differently, where Triton lays those tiles out for a program of another size or warp count
(ced/kernels/gdn_c2_triton.py, its docstring). So "once" is judged as a
numerics change: against the float64 reference by its GPU check (scripts/gdn_c2_check.py) and by the 60M band check.

Scope: the one-pass training path (the "fused" form: no keep, valid, state or GDN-side arms), impl "triton", keep
"all", K = V = 128, G = H_v / H_k of 1 or 2. On a CPU without the Triton interpreter the ops run the PyTorch
reference (ced.kernels.gdn_edges._reference), with placeholder saved tensors: it serves the CPU tests of the ops'
registration, autograd and tracing, not training.
"""

from __future__ import annotations

import os

import torch

from ced.kernels import gdn as G_
from ced.kernels import gdn_edges as E_

CHUNK = G_.CHUNK
D_HEAD = E_.D_HEAD
# Launch options of the two new kernels (fixed: no run-time autotuning). front: BK as today's intra option for the input
# dtype (bf16 128, float32 64), 4 warps (bf16 wu's; float32 wu runs 8, so there w / u's products are laid out differently
# too; 4 is the warp count wgmma needs for its dots); wy_dk: wy's 4 warps. Not swept (the GPU check times them, and
# prints both at 8 warps; a sweep is the follow-up if the kernel pays).
CONFIG = {"front": {"BK": 64, "num_warps": 4, "num_stages": 1}, "wy_dk": {"num_warps": 4, "num_stages": 1}}
CONFIG_BF16 = {"front": {"BK": 128, "num_warps": 4, "num_stages": 1}, "wy_dk": {"num_warps": 4, "num_stages": 1}}
# How often the Triton implementation ran in this process, each way; with CED_C2_REPORT=1 the first call each way prints
# one "C2-RAN" line (the GPU check's training leg reads them: positive evidence that "once" ran these kernels).
RAN = {"forward": 0, "backward": 0}


def config_for(dtype: torch.dtype) -> dict:
    return CONFIG_BF16 if dtype == torch.bfloat16 else CONFIG


def _ran(way: str, u: torch.Tensor, hk: int, hv: int) -> None:
    RAN[way] += 1
    if RAN[way] == 1 and os.environ.get("CED_C2_REPORT") == "1":
        print(f"C2-RAN {way} (first call: u {tuple(u.shape)} {u.dtype}, {hk} key / {hv} value heads, {u.device})",
              flush=True)


def _launch(kernel, grid, cfg: dict, *args, **kwargs):
    kernel[grid](*args, **kwargs, **cfg)


# ------------------------------------------------------------------------------------------------
# The Triton implementation (a GPU, or the interpreter)


def _fwd_triton(u, doc, cw, g, beta, cu, ci, co, hk: int, hv: int, out_bf16: bool, f3: bool):
    import triton.language as tl

    from ced.kernels import gdn_c2_triton as C2
    from ced.kernels import gdn_triton as K_

    B, L, C = u.shape
    T, d = B * L, C // (2 * hk + hv)
    H, HV, K, V = hk, hv, d, d
    assert d == D_HEAD and C == (2 * hk + hv) * d and HV in (H, 2 * H), (u.shape, hk, hv)
    assert g.shape == beta.shape == (1, T, HV) and doc.shape == (B, L), (g.shape, beta.shape, doc.shape)
    G = HV // H
    comp, out, prep_acc, acc = E_._dtypes(u.dtype, cw.dtype, out_bf16)
    assert beta.dtype == out, (beta.dtype, out)
    PACC = tl.float64 if prep_acc == torch.float64 else tl.float32
    ACC = tl.float64 if acc == torch.float64 else tl.float32
    u, doc, cw = u.contiguous(), doc.contiguous(), cw.contiguous()
    _ran("forward", u, hk, hv)
    dev = u.device
    cap, N = ci.shape[0], co.shape[0] - 1
    g = g.to(acc).contiguous()
    beta = beta.contiguous()
    cf = G_.config_for(out)
    # today's decay cumsum (GatedDeltaRule.forward's first launch), over the capacity
    gc = torch.empty_like(g)
    gl = torch.empty_like(g) if f3 else None
    G_._launch(K_.cumsum_fwd_kernel, (cap, HV), cf, "cumsum", g, gc, gl, cu, ci, T, HV=HV, BT=CHUNK, ACC=ACC, F3=bool(f3))
    F3 = gl is not None
    # the front kernel: q, k, v formed once a chunk, then A, w, u
    q = torch.empty(1, T, H, K, dtype=out, device=dev)
    k = torch.empty(1, T, H, K, dtype=out, device=dev)
    v = torch.empty(1, T, HV, V, dtype=out, device=dev)
    A = torch.empty(1, T, HV, CHUNK, dtype=out, device=dev)
    w = torch.empty(1, T, HV, K, dtype=out, device=dev)
    u_ = torch.empty(1, T, HV, V, dtype=out, device=dev)
    u_prec = G_.CONFIG["wu"].get("u_prec", "ieee") if out == torch.float32 else "tf32"
    _launch(C2.front_fwd_kernel, (cap, H), config_for(out)["front"], u, doc, cw, q, k, v, gc, gl, beta, A, w, u_, cu, ci,
            T, L, C=C, H=H, G=G, K=K, V=V, BT=CHUNK, BC=16, KC=cw.shape[1], ACC=ACC, PACC=PACC,
            SOLVE_PREC=K_.SOLVE_PRECISION, U_PREC=u_prec, Q_SCALE=d**-0.5, EPS=1e-6, F3=F3)
    # today's state recurrence and output kernel, on the stored q, k
    h = torch.empty(1, cap, HV, K, V, dtype=out, device=dev)
    v_new = torch.empty_like(u_)
    BV = cf["state_fwd"]["BV"]
    G_._launch(K_.state_fwd_kernel, (V // BV, N * HV), cf, "state_fwd", k, u_, w, v_new, gc, gl, h, cu, co, T,
               H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=F3)
    o = torch.empty_like(v_new)
    G_._launch(K_.out_fwd_kernel, (cap, H), cf, "out", q, k, v_new, h, gc, gl, o, cu, ci, T,
               H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=F3)
    return o, q, k, v, gc, (gl if F3 else g.new_empty(0)), A, w, h, v_new


def _bwd_triton(do, u, doc, cw, q, k, v, beta, gc, gl, A, w, h, v_new, cu, ci, co, hk: int, hv: int, out_bf16: bool):
    import triton.language as tl

    from ced.kernels import gdn_c2_triton as C2
    from ced.kernels import gdn_triton as K_

    B, L, C = u.shape
    T, d = B * L, C // (2 * hk + hv)
    H, HV, K, V = hk, hv, d, d
    G = HV // H
    comp, out, prep_acc, acc = E_._dtypes(u.dtype, cw.dtype, out_bf16)
    PACC = tl.float64 if prep_acc == torch.float64 else tl.float32
    ACC = tl.float64 if acc == torch.float64 else tl.float32
    gl = gl if gl.numel() else None
    F3 = gl is not None
    cap, N = ci.shape[0], co.shape[0] - 1
    # the saved inputs are the op's inputs as given, not the forward's contiguous copies
    u, doc, cw = u.contiguous(), doc.contiguous(), cw.contiguous()
    _ran("backward", u, hk, hv)
    dev = u.device
    cf = G_.config_for(out)
    # today's GatedDeltaRule.backward (keep "all", no second-pass read: EXT False) up to dqkwg, launch for launch
    do = torch.zeros(1, T, HV, V, dtype=out, device=dev) if do is None else do.contiguous()
    dv_local = torch.empty_like(do)
    G_._launch(K_.dv_local_kernel, (cap, H), cf, "dv_local", q, k, gc, gl, do, dv_local, cu, ci, T,
               H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=F3)
    dh = torch.empty(1, cap, HV, K, V, dtype=out, device=dev)
    dv2 = torch.empty_like(dv_local)
    BV = cf["state_bwd"]["BV"]
    G_._launch(K_.state_bwd_kernel, (V // BV, N * HV), cf, "state_bwd", q, k, w, gc, gl, do, dv_local, dv2, dh,
               dh, dv_local, cu, co, T, H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, EXT=False, F3=F3)
    dq = torch.empty(1, T, H, K, dtype=out, device=dev)
    dk_part = torch.empty(1, T, H, K, dtype=acc, device=dev)
    dw = torch.empty_like(w)
    dg_part = torch.empty(1, T, HV, dtype=acc, device=dev)
    G_._launch(K_.dqkwg_kernel, (cap, H), cf, "dqkwg", q, k, v_new, gc, gl, h, do, dh, dv2, dq, dk_part, dw, dg_part,
               cu, ci, T, H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=F3)
    # the WY backward with the dk sum folded in (wy_bwd_kernel + dk_sum_kernel)
    dv = torch.empty(1, T, HV, V, dtype=out, device=dev)
    db = torch.empty_like(beta)
    dg = torch.empty(1, T, HV, dtype=acc, device=dev)
    dk = torch.empty(1, T, H, K, dtype=out, device=dev)
    _launch(C2.wy_dk_kernel, (cap, H), config_for(out)["wy_dk"], k, v, beta, gc, gl, A, dw, dv2, dg_part, dk_part, dv, db,
            dg, dk, cu, ci, T, H=H, G=G, K=K, V=V, BT=CHUNK, ACC=ACC, F3=F3)
    # today's _QKVPrep.backward, launch for launch (dq, dk, dv in the input dtype, read into float32; da in float32)
    cfgs = G_.prep_config_for(out)
    cfg = cfgs["da"]
    nb = (T + cfg["BT"] - 1) // cfg["BT"]
    da = torch.empty(B, L, C, dtype=prep_acc, device=dev)
    dw_part = torch.empty(nb, cw.shape[1], C, dtype=prep_acc, device=dev)
    K_.prep_da_kernel[(nb, 2 * hk + hv)](u, doc, cw, dq, dk, dv, da, dw_part, T, L, C=C, HK=hk, HV=hv, D=d,
                                         KC=cw.shape[1], ACC=PACC, Q_SCALE=d**-0.5, EPS=1e-6, **cfg)
    du = torch.empty_like(u)
    cfg = cfgs["dx"]
    K_.prep_dx_kernel[((T + cfg["BT"] - 1) // cfg["BT"], C // d)](da, doc, cw, du, T, L, C=C, D=d, KC=cw.shape[1],
                                                                   ACC=PACC, **cfg)
    dcw = dw_part.sum(0).t().to(cw.dtype).contiguous()
    return du, dcw, dg.to(acc), db


# ------------------------------------------------------------------------------------------------
# The custom ops


@torch.library.custom_op("ced::gdn_c2_fwd", mutates_args=())
def gdn_c2_fwd(u: torch.Tensor, doc: torch.Tensor, cw: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
               cu: torch.Tensor, ci: torch.Tensor, co: torch.Tensor, hk: int, hv: int, out_bf16: bool, f3: bool
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                          torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(o, q, k, v, gc, gl, A, w, h, v_new): o (1, B*L, hv, 128) the delta rule's output in the input dtype; the rest is what
    the backward reads (gl empty without F3). u (B, L, C) the in_proj_qkv output, doc (B, L) int64, cw (C, KC), g
    (1, B*L, hv) float32 (float64 with float64 inputs), beta (1, B*L, hv) in the input dtype; cu, ci, co the chunk
    metadata of ced.kernels.gdn_edges.chunk_meta."""
    if E_._use_triton(u):
        return _fwd_triton(u, doc, cw, g, beta, cu, ci, co, hk, hv, out_bf16, f3)
    B, L, C = u.shape
    T, d = B * L, C // (2 * hk + hv)
    comp, out, prep_acc, acc = E_._dtypes(u.dtype, cw.dtype, out_bf16)
    o = E_._reference(u, doc, cw, g, beta, cu[:-1], hk, hv, out)
    z = lambda *s, dt=out: torch.zeros(*s, dtype=dt, device=u.device)  # noqa: E731
    return (o, z(1, T, hk, d), z(1, T, hk, d), z(1, T, hv, d), g.to(acc).clone(),
            (z(1, T, hv, dt=acc) if f3 else z(0, dt=acc)), z(1, T, hv, CHUNK), z(1, T, hv, d), z(1, ci.shape[0], hv, d, d),
            z(1, T, hv, d))


@gdn_c2_fwd.register_fake
def _(u, doc, cw, g, beta, cu, ci, co, hk, hv, out_bf16, f3):
    B, L, C = u.shape
    T, d = B * L, C // (2 * hk + hv)
    comp, out, prep_acc, acc = E_._dtypes(u.dtype, cw.dtype, out_bf16)
    e = lambda *s, dt=out: u.new_empty(*s, dtype=dt)  # noqa: E731
    return (e(1, T, hv, d), e(1, T, hk, d), e(1, T, hk, d), e(1, T, hv, d), e(1, T, hv, dt=acc),
            (e(1, T, hv, dt=acc) if f3 else e(0, dt=acc)), e(1, T, hv, CHUNK), e(1, T, hv, d), e(1, ci.shape[0], hv, d, d),
            e(1, T, hv, d))


@torch.library.custom_op("ced::gdn_c2_bwd", mutates_args=())
def gdn_c2_bwd(do: torch.Tensor, u: torch.Tensor, doc: torch.Tensor, cw: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
               v: torch.Tensor, beta: torch.Tensor, gc: torch.Tensor, gl: torch.Tensor, A: torch.Tensor, w: torch.Tensor,
               h: torch.Tensor, v_new: torch.Tensor, cu: torch.Tensor, ci: torch.Tensor, co: torch.Tensor, hk: int,
               hv: int, out_bf16: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(du, dcw, dg, dbeta): the gradients wrt u (B, L, C), cw, g (1, B*L, hv) and beta, as today's path gives them."""
    if E_._use_triton(u):
        return _bwd_triton(do, u, doc, cw, q, k, v, beta, gc, gl, A, w, h, v_new, cu, ci, co, hk, hv, out_bf16)
    # the CPU reference: the vector-Jacobian product of the forward again (gc holds g itself there)
    comp, out, prep_acc, acc = E_._dtypes(u.dtype, cw.dtype, out_bf16)
    fn = lambda u_, cw_, g_, b_: E_._reference(u_, doc, cw_, g_, b_, cu[:-1], hk, hv, out)  # noqa: E731
    o, vjp = torch.func.vjp(fn, u, cw, gc, beta)
    return tuple(t.contiguous() for t in vjp(do.to(o.dtype)))


@gdn_c2_bwd.register_fake
def _(do, u, doc, cw, q, k, v, beta, gc, gl, A, w, h, v_new, cu, ci, co, hk, hv, out_bf16):
    comp, out, prep_acc, acc = E_._dtypes(u.dtype, cw.dtype, out_bf16)
    return u.new_empty(u.shape), cw.new_empty(cw.shape), gc.new_empty(gc.shape, dtype=acc), beta.new_empty(beta.shape)


def _setup_context(ctx, inputs, output):
    u, doc, cw, g, beta, cu, ci, co, hk, hv, out_bf16, f3 = inputs
    o, q, k, v, gc, gl, A, w, h, v_new = output
    ctx.save_for_backward(u, doc, cw, q, k, v, beta, gc, gl, A, w, h, v_new, cu, ci, co)
    ctx.meta = (hk, hv, out_bf16)
    ctx.mark_non_differentiable(q, k, v, gc, gl, A, w, h, v_new)


def _backward(ctx, do, *unused):
    hk, hv, out_bf16 = ctx.meta
    du, dcw, dg, db = torch.ops.ced.gdn_c2_bwd(do, *ctx.saved_tensors, hk, hv, out_bf16)
    return du, None, dcw, dg, db, None, None, None, None, None, None, None


gdn_c2_fwd.register_autograd(_backward, setup_context=_setup_context)


def once_delta_rule(u: torch.Tensor, doc: torch.Tensor, conv_weight: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                    cu_seqlens: torch.Tensor, hk: int, hv: int, d: int, out_dtype: torch.dtype | None = None,
                    f3: bool = False) -> torch.Tensor:
    """qkv_prep(u, doc, conv_weight, hk, hv, d, out_dtype) followed by the delta rule of kernel "ced" (the model's
    ced_delta_rule wrapper with these q, k, v), with q, k, v formed once a chunk in the first kernel that reads it. The
    arguments and the result are ced.kernels.gdn_edges.fused_delta_rule's: u (B, L, C) the in_proj_qkv output; doc (B, L)
    document ids; g, beta (B, L, hv); cu_seqlens on u's device (no host copy is read). Returns o (B, L, hv, d) in the
    kernel's input dtype. Impl "triton" and keep "all" only (CED_GDN_IMPL / CED_GDN_KEEP)."""
    if not torch.compiler.is_compiling():
        E_.check_process()
    if not isinstance(f3, bool):
        raise TypeError(f"f3 must be a bool, not {f3!r}")
    B, L, C = u.shape
    assert d == D_HEAD and C == (2 * hk + hv) * d and hv in (hk, 2 * hk), (u.shape, hk, hv, d)
    assert out_dtype in (None, torch.bfloat16), out_dtype
    out_bf16 = out_dtype == torch.bfloat16
    comp, out, prep_acc, acc = E_._dtypes(u.dtype, conv_weight.dtype, out_bf16)
    flat = lambda t: t.reshape(1, B * L, *t.shape[2:])  # noqa: E731
    meta = E_.chunk_meta(cu_seqlens, B * L)[:3] if torch.compiler.is_compiling() else E_.cached_chunk_meta(cu_seqlens, B * L)
    o = gdn_c2_fwd(u, doc, conv_weight, flat(g).to(acc), flat(beta).to(out), *meta, hk, hv, out_bf16, f3)[0]
    return o.reshape(B, L, hv, d)
