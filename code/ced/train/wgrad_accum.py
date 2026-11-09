"""Weight gradients accumulated in fp32 by the wgrad GEMM itself (TrainConfig.wgrad_accum).

TODAY ("autograd", the default: every run before the key, bit for bit). Under bf16 autocast a linear y = x W^T casts its
float32 master weight to bf16 (autocast's cached cast), and its backward forms dW = dy^T x in ONE GEMM whose float32
accumulator is rounded to a bf16 output; the cast's backward turns that bf16 tensor into a new float32 tensor, and the
autograd engine's AccumulateGrad takes it as .grad at the step's first micro-batch and adds it into .grad
(read 4 + read 4 + write 4 bytes a parameter) at every later one. Per parameter and micro-batch after the first: the GEMM
writes 2 bytes, the cast reads 2 and writes 4, the add moves 12: 20 bytes, and each micro-batch's gradient is rounded once
to bf16 (8 significant bits) before the sum.

"fp32_fused". Every eligible linear's weight gradient is accumulated by the wgrad GEMM straight into a persistent float32
buffer, which IS the parameter's .grad: acc = 1 x acc + dy^T x (beta = 1), the products of the bf16 operands summed in the
GEMM's float32 accumulator and added to acc in float32, written once. Per parameter and micro-batch: read 4 + write 4 = 8
bytes, no bf16 gradient is materialised, no cast, no separate add. The buffers are zeroed in place at the start of every
step (torch._foreach_zero_: 4 bytes a parameter once an optimizer step). Numerics: one rounding fewer (the bf16 rounding of
each micro-batch's gradient); not bit for bit, so the band check (scripts/band_check_60m.py) is its gate.

Eligible linears (apply_wgrad_accum): every bias-free plain nn.Linear of the layers' mixers and FFNs (the walk of
ced/train/zero_wgrad.py wgrad_modules: encoder.layers.<i>.<mixer|mlp>.<name>): GDN in_proj_qkv, in_proj_z, in_proj_a,
in_proj_b, out_proj; attention q_proj (with the elementwise gate's half), g_proj (the headwise gate), k_proj, v_proj,
o_proj; SwiGLU w_gate, w_up, w_down. Their forward under autocast is today's F.linear on the same bf16 casts, bit for
bit, and so are dx and the loss; only the weight gradient changes. Not eligible, and unchanged:
- the output head: the fused head (ced/train/fused_head.py) forms its dW in float32 already (TF32 or, with G, bf16 operands
  with float32 output) and returns it through autograd; its add (12 bytes x 163.8M at 4B, about 2 ms a micro-batch on the
  slice) stays. Accumulating in place there is a change to the fused head, not to a linear;
- the token embedding (an index-add, not a GEMM), norms, gates, the GDN conv weights, R1-entry's queries: autograd;
- the emulated FP8 / NVFP4 linears (model.fp8, model.nvfp4): refused by name. Their Wgrad is a GEMM on
  fake-quantised operands rounded to the GEMM dtype as the real FP8 GEMM's bf16 output is; MXFP8 through Transformer
  Engine has its own fused accumulation (fuse_wgrad_accumulation into a float32 main_grad), which B's wiring should
  point at these buffers;
- a linear with a bias (the asentmax scale heads), or any subclass of nn.Linear: kept, and named in the summary.

The optimizer, the clip and the logs. The buffer is p.grad, float32, the parameter's shape, contiguous: MuonWithAdamW (Muon
and AdamW groups alike), torch.nn.utils.clip_grad_norm_, the gradient-norm metrics and the zero-wgrad counter read it as
they read .grad today. Nothing else changes in the optimizer step.

CUDA graphs (TrainConfig.cuda_graphs, N1; ced/train/cuda_graphs.py): composes. N1 keeps the gradients the process's first
step made and zeroes them in place; with F those include these buffers (attached before the first forward), so a replay
accumulates into the same addresses, and the step's zeroing is N1's (start_step runs only on the eager path). The
accumulation is a cuBLAS GEMM on the replay's stream, capturable; the route is chosen (a probe with a host read) when the
trainer is built, never inside a capture.

Activation checkpointing: "layer" and "block" recompute the forward (the same casts) and run this backward once per
micro-batch: the same accumulation (tests). "host" / "host_overlap": refused by name (not tested here).

Resume. Nothing is saved: the buffers are zeroed at every step's start, so a resumed step is the uninterrupted one bit for
bit (tests). Determinism: one accumulation per eligible weight per micro-batch, in the micro-batches' order; inside it
cuBLAS's fixed reduction order for the shape (the trainer's deterministic mode and CUBLAS_WORKSPACE_CONFIG as today).

The GEMM (the custom op ced::wgrad_accum_fp32, opaque to torch.compile, so Inductor neither decomposes it into a GEMM and a
separate add nor clones the buffer; tests/test_wgrad_accum.py reads the compiled code). On CUDA with bf16 / fp16 operands
and a float32 buffer, one of three routes, chosen once a process by select_route (the first that passes an exact probe):
  addmm_out  torch.addmm(acc, dy^T, x, out_dtype=torch.float32, out=acc): one cuBLAS GEMM, bf16 inputs, float32 C with
             beta = 1 (whether torch 2.14.1 accepts a float32 `input` aliasing `out` here is unverified on the GPU: a
             GPU check covers it, its result and its kernel count);
  cublas     cublasGemmEx on PyTorch's own handle and stream (ctypes; CUDA_R_16BF inputs, CUDA_R_32F C, CUBLAS_COMPUTE_32F,
             beta = 1): the same GEMM without the torch wrapper;
  mm_add     torch.mm(dy^T, x, out_dtype=torch.float32) then acc.add_: the float32 GEMM output and an add (16 bytes a
             parameter, not 8): the fallback, same numerics up to the order of one add.
Operands of the model's own dtype (float32 or float64 without autocast): acc.addmm_(dy^T, x) (beta = 1; TF32 per the
trainer's setting, as F.linear's GEMMs). On the CPU (the tests): acc.addmm_ on the operands converted to the buffer's
dtype (bf16 products are exact in float32), the same numerics as the CUDA GEMM up to the order of summation.

Fixed shapes: every tensor's shape follows from the layer's (reshape, transpose, GEMM); no boolean-mask indexing or nonzero.
"""

from __future__ import annotations

import ctypes

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("autograd", "fp32_fused")
ROUTES = ("addmm_out", "cublas", "mm_add")
CHECKPOINT_MODES = ("none", "layer", "block")  # activation_checkpoint modes built with fp32_fused (module docstring)

_route: str | None = None  # the CUDA route of this process (select_route); None until chosen
_route_failures: dict[str, str] = {}  # the routes that failed their probe, with the reason


# ------------------------------------------------------------------------------------------------
# the GEMM: acc += dy^T x


def _addmm_out(acc: torch.Tensor, dy: torch.Tensor, x: torch.Tensor) -> None:
    torch.addmm(acc, dy.t(), x, out_dtype=torch.float32, out=acc)


def _mm_add(acc: torch.Tensor, dy: torch.Tensor, x: torch.Tensor) -> None:
    acc.add_(torch.mm(dy.t(), x, out_dtype=torch.float32))


_CUBLAS: dict = {}
_CUDA_DTYPE = {torch.bfloat16: 14, torch.float16: 2}  # cudaDataType_t: CUDA_R_16BF, CUDA_R_16F
_CUDA_R_32F, _CUBLAS_COMPUTE_32F, _CUBLAS_GEMM_DEFAULT, _OP_N, _OP_T = 0, 68, -1, 0, 1


def _cublas_gemm_ex():
    """cublasGemmEx of the cuBLAS library PyTorch loaded (found in /proc/self/maps after PyTorch made its handle)."""
    if "fn" in _CUBLAS:
        return _CUBLAS["fn"]
    torch.cuda.current_blas_handle()  # loads cuBLAS and creates this thread's handle
    paths: list[str] = []
    with open("/proc/self/maps") as f:
        for line in f:
            name = line.split()[-1] if line.split() else ""
            if name.rsplit("/", 1)[-1].startswith("libcublas.so") and name not in paths:
                paths.append(name)
    if not paths:
        raise RuntimeError("wgrad_accum route 'cublas': libcublas.so is not mapped in this process")
    if len(paths) > 1:
        raise RuntimeError(f"wgrad_accum route 'cublas': several cuBLAS libraries are mapped ({paths}); not knowing which one "
                           "made PyTorch's handle, the route is not used")
    path = paths[0]
    lib = ctypes.CDLL(path)
    fn = lib.cublasGemmEx
    fn.restype = ctypes.c_int
    vp, ci = ctypes.c_void_p, ctypes.c_int
    fn.argtypes = [vp, ci, ci, ci, ci, ci, vp, vp, ci, ci, vp, ci, ci, vp, vp, ci, ci, ci, ci]
    _CUBLAS["fn"], _CUBLAS["path"] = fn, path
    return fn


def _cublas(acc: torch.Tensor, dy: torch.Tensor, x: torch.Tensor) -> None:
    """acc (N, K) float32 row-major += dy^T (N, M) x (M, K): in cuBLAS's column-major terms acc^T (K, N) = x^T (K, M) dy (M, N),
    A = x (column-major K x M, lda K, no transpose), B = dy (column-major N x M, ldb N, transposed), beta = 1."""
    if not (acc.is_contiguous() and acc.dtype == torch.float32 and dy.dtype == x.dtype and dy.dtype in _CUDA_DTYPE):
        raise RuntimeError("wgrad_accum route 'cublas': a contiguous float32 buffer and bf16 / fp16 operands")
    dy, x = dy.contiguous(), x.contiguous()
    M, N = dy.shape
    K = x.shape[1]
    fn = _cublas_gemm_ex()
    one = ctypes.c_float(1.0)
    handle = torch.cuda.current_blas_handle()  # sets the handle's stream (and workspace) to the current stream
    dt = _CUDA_DTYPE[dy.dtype]
    status = fn(handle, _OP_N, _OP_T, K, N, M, ctypes.byref(one), x.data_ptr(), dt, K, dy.data_ptr(), dt, N,
                ctypes.byref(one), acc.data_ptr(), _CUDA_R_32F, K, _CUBLAS_COMPUTE_32F, _CUBLAS_GEMM_DEFAULT)
    if status != 0:
        raise RuntimeError(f"wgrad_accum route 'cublas': cublasGemmEx returned status {status}")


ROUTE_FN = {"addmm_out": _addmm_out, "cublas": _cublas, "mm_add": _mm_add}


def probe(route: str, device) -> str | None:
    """None when `route` computes acc += dy^T x exactly and in place on `device` (twice, so the second call accumulates
    onto the first: small integers, exact in bf16 and float32); else the reason. Reads the device (host sync): never call
    it inside a capture."""
    g = torch.Generator().manual_seed(1)
    M, N, K = 96, 40, 24
    dy = torch.randint(-4, 5, (M, N), generator=g).to(torch.bfloat16)
    x = torch.randint(-4, 5, (M, K), generator=g).to(torch.bfloat16)
    acc0 = torch.randint(-64, 65, (N, K), generator=g).float()
    want = acc0.double() + 2 * (dy.double().t() @ x.double())
    try:
        acc = acc0.to(device)
        ptr = acc.data_ptr()
        d_dy, d_x = dy.to(device), x.to(device)
        ROUTE_FN[route](acc, d_dy, d_x)
        ROUTE_FN[route](acc, d_dy, d_x)
        got = acc.cpu().double()
    except Exception as e:  # noqa: BLE001 - a route that does not run is a reason, not an error
        return f"{type(e).__name__}: {str(e)[:300]}"
    if acc.data_ptr() != ptr:
        return "the buffer was reallocated (not in place)"
    if not torch.equal(got, want):
        return f"wrong result: max |got - want| {float((got - want).abs().max())}"
    return None


def select_route(device, prefer: str | None = None) -> str:
    """The process's CUDA route: `prefer` if given and it passes its probe, else the first of ROUTES that passes. Recorded
    in _route (and the failures in _route_failures). On the CPU: 'cpu'."""
    global _route
    device = torch.device(device)
    if device.type != "cuda":
        return "cpu"
    if _route is not None and prefer in (None, _route):
        return _route
    order = ([prefer] if prefer else []) + [r for r in ROUTES if r != prefer]
    for r in order:
        why = probe(r, device)
        if why is None:
            _route = r
            return r
        _route_failures[r] = why
    raise RuntimeError(f"wgrad_accum fp32_fused: no route passed its probe on {device}: {_route_failures}")


def route() -> str | None:
    return _route


def accumulate(acc: torch.Tensor, dy: torch.Tensor, x: torch.Tensor) -> None:
    """acc += dy^T x in place: acc (N, K) in the weight's dtype, dy (M, N) and x (M, K) in the GEMM's dtype (module docstring,
    "The GEMM")."""
    if dy.dtype == acc.dtype and x.dtype == acc.dtype:
        acc.addmm_(dy.t(), x)
    elif acc.device.type == "cuda":
        if _route is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("wgrad_accum fp32_fused: no route chosen before a CUDA graph capture (select_route)")
            select_route(acc.device)
        ROUTE_FN[_route](acc, dy, x)
    else:  # the CPU: the bf16 products are exact in the buffer's dtype
        acc.addmm_(dy.t().to(acc.dtype), x.to(acc.dtype))


@torch.library.custom_op("ced::wgrad_accum_fp32", mutates_args=("acc",))
def wgrad_accum_fp32(acc: torch.Tensor, dy: torch.Tensor, x: torch.Tensor) -> None:
    """The wgrad GEMM with beta = 1 into the persistent buffer (accumulate); opaque to torch.compile."""
    accumulate(acc, dy, x)


@wgrad_accum_fp32.register_fake
def _wgrad_accum_fp32_fake(acc, dy, x) -> None:
    return None


# ------------------------------------------------------------------------------------------------
# the linear


class _FusedWgradLinearFunction(torch.autograd.Function):
    """y = x W^T in the GEMM dtype as F.linear under autocast computes it; the backward returns dx as today and accumulates
    dW into acc (the weight's .grad) instead of returning it."""

    @staticmethod
    def forward(ctx, x, weight, acc, dtype):
        xc = x if x.dtype == dtype else x.to(dtype)  # autocast's casts: the same _to_copy, the same bits
        wc = weight if weight.dtype == dtype else weight.to(dtype)
        with torch.autocast(x.device.type, enabled=False):
            y = F.linear(xc, wc)
        # saved: the GEMM-dtype activation and weight (what the mm under autocast saves: the same memory) and the buffer
        ctx.save_for_backward(xc, wc, acc)
        ctx.x_dtype = x.dtype
        return y

    @staticmethod
    def backward(ctx, dy):
        xc, wc, acc = ctx.saved_tensors
        n, k = wc.shape
        dy2 = dy.reshape(-1, n)
        dx = None
        if ctx.needs_input_grad[0]:  # today's MmBackward: dy W in the GEMM dtype, then the cast's backward to x's dtype
            dx = dy2.mm(wc).reshape(xc.shape).to(ctx.x_dtype)
        torch.ops.ced.wgrad_accum_fp32(acc, dy2, xc.reshape(-1, k))
        return dx, None, None, None


class FusedWgradLinear(nn.Linear):
    """nn.Linear with the same parameter and state dict whose training backward accumulates the weight's gradient into its
    persistent float32 .grad (module docstring). Made by apply_wgrad_accum (a class swap, as apply_fp8 makes FP8Linear).
    In eval mode, without grad mode or for a weight that does not require grad it is F.linear: the plain layer, bit for bit."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if not (self.training and torch.is_grad_enabled() and w.requires_grad):
            return F.linear(x, w, self.bias)
        acc = w.grad
        if acc is None:
            raise RuntimeError("wgrad_accum fp32_fused: a training forward of a fused linear whose weight has no float32 "
                               "gradient buffer (the trainer attaches them: WgradAccum.start_step)")
        dev = x.device.type
        dtype = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else x.dtype
        return _FusedWgradLinearFunction.apply(x, w, acc, dtype)


# ------------------------------------------------------------------------------------------------
# the config, the wiring


def check_config(cfg) -> None:
    """ValueError, by name, for a wgrad_accum value or a config fp32_fused is not built for. "autograd": only the value."""
    mode = cfg.wgrad_accum
    if mode not in MODES:
        raise ValueError(f"wgrad_accum {mode!r}: one of {MODES}")
    if mode == "autograd":
        return
    m, e = cfg.model, cfg.model.encoder
    refused = [
        (m.arch != "control", f"arch {m.arch!r} (built for arch 'control': its layers' mixers and FFNs)"),
        (m.loop.pass2_gate != "none", "model.loop.pass2_gate (the second pass applies part of in_proj_qkv outside the module)"),
        (bool(m.loop.recur_layers), "model.loop.recur_layers (a looped middle calls its layers more than once a forward)"),
        (e.loops != 1, f"model.encoder.loops {e.loops}"),
        (e.gdn_write_mix != "none" or bool(e.gdn_ccr_spacing),
         "the GDN-side arms (gdn_write_mix, gdn_ccr_spacing: arm A applies in_proj weights outside the module)"),
        (e.attention.normalizer != "softmax", f"the {e.attention.normalizer!r} normaliser (nothing tests the pair)"),
        # the message below names the attn_res forms it is built for; "ldm" was not refused (2026-10-17)
        (e.attn_res not in ("none", "block", "block_entry"), f"model.encoder.attn_res {e.attn_res!r} (nothing tests the pair)"),
        (e.layer_scalars != "none", f"model.encoder.layer_scalars {e.layer_scalars!r} (nothing tests the pair)"),
        (bool(e.canon_kernel), "model.encoder.canon_kernel (nothing tests the pair)"),
        (m.fp8.mode != "none", "model.fp8 (its emulated Wgrad; MXFP8 through Transformer Engine has its own fused accumulation)"),
        (m.nvfp4.mode != "none", "model.nvfp4 (its emulated Wgrad)"),
        (bool(getattr(cfg, "precision_switch", False)), "precision_switch"),
        # the buffers take the parameters' dtype (zeros_like): float32 master weights make them float32; with bfloat16
        # parameters they would be bf16, the accumulation this mode removes (2026-10-17)
        (cfg.dtype != "float32", f"dtype {cfg.dtype!r} (the buffers are the float32 master weights' .grad)"),
        (cfg.activation_checkpoint not in CHECKPOINT_MODES,
         f"activation_checkpoint {cfg.activation_checkpoint!r} (built with {CHECKPOINT_MODES})"),
    ]
    bad = [why for cond, why in refused if cond]
    if bad:
        raise ValueError(f"wgrad_accum {mode!r} is built for the plain control (arch 'control', one pass, attn_res none / block / "
                         f"block_entry, bf16 linears); not for: {'; '.join(bad)}. Leave it 'autograd'")


def apply_wgrad_accum(model: nn.Module, mode: str) -> dict:
    """Turn every eligible linear (module docstring) into FusedWgradLinear in place, before the layers are compiled.
    Parameters, their names, the state dict and the RNG use do not change. Returns the summary (also model.wgrad_summary):
    the mode, the fused linears, the kept ones with the reason, and their parameter counts."""
    if mode == "autograd":
        return {}
    from ced.train.zero_wgrad import wgrad_modules

    summary = {"mode": mode, "fused": [], "kept": {}, "fused_params": 0, "kept_params": 0}
    for name, m in wgrad_modules(model):
        why = None
        if type(m) is not nn.Linear:
            why = f"a {type(m).__name__}, not a plain nn.Linear"
        elif m.bias is not None:
            why = "has a bias"
        if why is None:
            m.__class__ = FusedWgradLinear
            summary["fused"].append(name)
            summary["fused_params"] += m.weight.numel()
        else:
            summary["kept"][name] = why
            summary["kept_params"] += m.weight.numel()
    model.wgrad_summary = summary
    return summary


class WgradAccum:
    """The trainer's side of fp32_fused: the class swap, the persistent float32 buffers (each the weight's .grad) and the
    route. start_step re-attaches and zeroes them (the eager path: Trainer.train_step after optimizer.zero_grad)."""

    def __init__(self, model: nn.Module, mode: str, device, grad_of=None):
        """grad_of: under data parallelism (ced/train/data_parallel.py), the parameter's view of the flat gradient buffer,
        which becomes its persistent buffer (zeroed, float32, contiguous); None (one process): a zeros_like of its own."""
        self.mode = mode
        self.summary = apply_wgrad_accum(model, mode)
        weights, seen = [], set()
        for m in model.modules():
            if isinstance(m, FusedWgradLinear) and id(m.weight) not in seen:
                seen.add(id(m.weight))
                weights.append(m.weight)
        if grad_of is None:
            self.pairs = [(p, torch.zeros_like(p, memory_format=torch.contiguous_format)) for p in weights]
        else:
            self.pairs = [(p, grad_of(p)) for p in weights]
        for p, g in self.pairs:
            p.grad = g
        self.route = select_route(device)

    def start_step(self) -> None:
        """Every buffer attached as its weight's .grad (zero_grad(set_to_none=True) dropped them) and zeroed in place."""
        for p, g in self.pairs:
            if p.grad is not g:
                p.grad = g
        torch._foreach_zero_([g for _, g in self.pairs])

    def describe(self) -> dict:
        """The start record's entry: the mode, the route, the fused linears' count and parameters, the kept ones."""
        s = self.summary
        return {"mode": self.mode, "route": self.route, "fused_linears": len(s["fused"]), "fused_params": s["fused_params"],
                "kept": s["kept"], "kept_params": s["kept_params"]}
