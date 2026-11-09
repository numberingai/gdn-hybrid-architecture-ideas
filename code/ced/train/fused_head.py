"""The LM head and its loss in one pass, with the gradients formed in the forward pass
(TrainConfig.head_impl = "fused").

The loss of one head call is

    sum_i  a_i * ce_i + b_i * lse_i^2,      ce_i = lse_i - logit_i[target_i],

with per-position weights a, b fixed before the call. The trainer knows them from the batch: the
normalisers, the aux and draft weights and the z-loss coefficient, and zero where a position does
not count. The gradient with respect to position i's logits is

    softmax_i * (a_i + 2 b_i lse_i) - a_i * onehot(target_i),

so a chunk's logits can be overwritten by their own gradient G as soon as lse is known. Then
dh = G @ W and dW += G^T @ h. Per chunk that is three matmuls (the chunked path does four: forward,
recompute, and two in backward), and the logits are never stored or recomputed. The matmuls are the
same cuBLAS calls as the chunked path (float32 inputs; TF32 when the global setting allows it). The
softmax and gradient are float32 (float64 for float64 inputs), computed by a Triton kernel on CUDA
and by plain PyTorch elsewhere.

The function returns the loss, a 0-dim tensor whose backward multiplies the stored gradients by the
incoming scalar, plus per-position ce and lse for metrics (no gradient). This is correct only when
the returned loss enters the training loss linearly, which is how the trainer uses it.

TrainConfig.head_precision "bf16" (ced/train/head_precision.py; the section "bf16 operands" at the end of this file):
the same three matmuls a chunk with bf16 operands and float32 outputs, the softmax and gradient in float32 as above,
the gradient written as bf16 (the operand of the two backward matmuls) by a second kernel that leaves the logits in
place. The default "fp32" runs the code above, unchanged.

TrainConfig.loss.ce_form "fixed" (ced/train/ce_fix.py, ced/kernels/ce_fix_triton.py): the `fixed` argument of fused_linear_ce and
fused_linear_multi_ce (the `ce_form` argument of their public functions; "standard", the default, is this file's code above,
unchanged). The target's column is left out of the log-sum-exp, the cross-entropy is softplus(LSE_{j != t}(z_j) - z_t) and the
target's logit gradient 2 b lse softmax_t - a sigmoid(LSE_{j != t}(z_j)
- z_t) instead of (a + 2 b lse) softmax_t - a; the other elements are the base kernel's. The Copy-Paste mixture is not built with it.
"""

from __future__ import annotations

import torch

from ced.model.norms import compute_dtype
from ced.train import ce_fix
from ced.train.head_precision import OPERAND, addmm_acc_, check_head_precision, mm_acc

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only environments: the PyTorch version below is used
    triton = None

CHUNK = 2048  # positions per chunk: 1 GiB of float32 logits; 12% faster than 1024 on a slice (bench_head_gemm)
# (TrainConfig.head_chunk's default: the trainer passes its head_chunk as `chunk` to every call below)
BLOCK_V = 2048


def _ce_grad_torch(logits, targets, a, b, ce, lse) -> None:
    """In place: logits -> their gradient; fills ce and lse. Reference version (any device)."""
    l = torch.logsumexp(logits, dim=-1)
    ce.copy_(l - logits.gather(1, targets[:, None]).squeeze(1))
    lse.copy_(l)
    logits.sub_(l[:, None]).exp_().mul_((a + 2 * b * l)[:, None])
    logits.scatter_add_(1, targets[:, None], -a[:, None])


if triton is not None:

    @triton.jit
    def _ce_grad_kernel(logits_ptr, stride, targets_ptr, a_ptr, b_ptr, ce_ptr, lse_ptr, n, V, BLOCK: tl.constexpr):
        """Rows row = program_id, program_id + num_programs, ... (one row per program when the grid is
        n; a smaller, persistent grid keeps fewer rows in flight, so the second pass over a row can
        hit the L2 cache). Per row: an online logsumexp over the vocabulary in BLOCK-wide tiles (each
        lane keeps its own running max and sum; the lanes are combined at the end), then a second pass
        that overwrites the logits with their gradient. The arithmetic of a row does not depend on the
        grid.

        The target's logit is taken from the tiles of the first pass, never re-read from memory: the
        second pass overwrites the row in place, and the warps of a program are not ordered with
        respect to global memory, so a separate load could see the gradient instead (a race found by
        the kill-and-resume test, 2026-10-05: it changed reported ce, never the gradient)."""
        cols = tl.arange(0, BLOCK)
        dtype = logits_ptr.dtype.element_ty
        for row in range(tl.program_id(0), n, tl.num_programs(0)):
            # tl.cast(row...), not row.to(...): compiled, the two are one op (Triton's tensor.to is an alias of cast, and
            # cast passes a tensor through unchanged); under TRITON_INTERPRET=1 the loop index is a Python int, which has
            # no .to (the CP interpreter run, 2026-10-18)
            base = logits_ptr + tl.cast(row, tl.int64) * stride
            target = tl.load(targets_ptr + row)
            m = tl.full([BLOCK], -1e30, dtype)
            s = tl.zeros([BLOCK], dtype)
            xt = tl.zeros([BLOCK], dtype)
            for start in range(0, V, BLOCK):
                x = tl.load(base + start + cols, mask=start + cols < V, other=float("-inf"))
                m_new = tl.maximum(m, x)
                s = s * tl.exp(m - m_new) + tl.exp(x - m_new)
                m = m_new
                xt += tl.where(start + cols == target, x, 0.0)
            mx = tl.max(m, axis=0)
            lse = mx + tl.log(tl.sum(s * tl.exp(m - mx), axis=0))
            a = tl.load(a_ptr + row)
            b = tl.load(b_ptr + row)
            tl.store(ce_ptr + row, lse - tl.sum(xt, axis=0))
            tl.store(lse_ptr + row, lse)
            coef = a + 2 * b * lse
            for start in range(0, V, BLOCK):
                idx = start + cols
                x = tl.load(base + idx, mask=idx < V, other=0.0)
                g = tl.exp(x - lse) * coef - tl.where(idx == target, a, 0.0)
                tl.store(base + idx, g, mask=idx < V)


# Launch of the softmax-gradient kernel: programs per SM (None: one program per row, the original
# grid), tile width, warps. Fixed (no autotuning): every process runs the same kernel.
CE_CONFIG = {"programs_per_sm": None, "block": BLOCK_V, "num_warps": 8}

# The logits dtypes the Triton kernel is built and tested for. It keeps its running max, running sum and target logit in
# the logits' dtype, so bfloat16 logits would mean bfloat16 accumulators: a GPU run of the kernel gates
# fed bfloat16 logits straight to it at V = 131,072 and the launch failed to compile. Production
# never does (the head computes in compute_dtype: float32, float64 for float64 inputs), and float64 is the GPU test's
# reference dtype (tests/gpu/test_fused_head_gpu.py). Any other dtype is refused at the call site, before the launch;
# the kernel is unchanged.
KERNEL_LOGITS_DTYPES = (torch.float32, torch.float64)


def check_kernel_logits(logits) -> None:
    """TypeError unless the logits' dtype is one the Triton kernel takes (KERNEL_LOGITS_DTYPES). Host metadata only."""
    if logits.dtype not in KERNEL_LOGITS_DTYPES:
        raise TypeError(f"the head's Triton kernel (_ce_grad_kernel) takes float32 logits (float64 for a reference), not "
                        f"{logits.dtype}: it accumulates in the logits' dtype. Upcast first, as fused_linear_ce does")


def _ce_grad(logits, targets, a, b, ce, lse, config: dict | None = None, fixed: bool = False) -> None:
    if fixed:  # loss.ce_form "fixed" (ced/train/ce_fix.py): the target column out of the log-sum-exp; the kernel's one pass
        if logits.is_cuda:
            check_kernel_logits(logits)
        ce_fix.ce_grad_fixed(logits, targets, a, b, ce, lse, CE_CONFIG if config is None else config)
        return
    if logits.is_cuda and triton is not None:
        check_kernel_logits(logits)
        cfg = CE_CONFIG if config is None else config
        n, V = logits.shape
        grid = n
        if cfg["programs_per_sm"]:
            sms = torch.cuda.get_device_properties(logits.device).multi_processor_count
            grid = min(n, cfg["programs_per_sm"] * sms)
        _ce_grad_kernel[(grid,)](logits, logits.stride(0), targets, a, b, ce, lse, n, V, BLOCK=cfg["block"],
                                 num_warps=cfg["num_warps"])
    else:
        _ce_grad_torch(logits, targets, a, b, ce, lse)


def _dW_update(dW, g, h_chunk) -> None:
    # h^T as its own contiguous (d, C) copy: cuBLAS then picks a TF32 kernel 2.2x faster than with h
    # itself, bitwise-identical results (scripts/bench_dw_layouts.py).
    dW.addmm_(g.t(), h_chunk.t().contiguous().t())


def _chunks_one_stream(h, W, t, a, b, ce, lse, dh, dW, chunk, fixed: bool = False) -> None:
    n, V = h.shape[0], W.shape[0]
    buf = torch.empty(min(chunk, n), V, dtype=h.dtype, device=h.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        g = buf[: e - s]
        torch.mm(h[s:e], W.t(), out=g)
        _ce_grad(g, t[s:e], a[s:e], b[s:e], ce[s:e], lse[s:e], fixed=fixed)
        if dh is not None:
            torch.mm(g, W, out=dh[s:e])
        if dW is not None:
            _dW_update(dW, g, h[s:e])


_SIDE_STREAMS: dict = {}


def _chunks_two_streams(h, W, t, a, b, ce, lse, dh, dW, chunk, fixed: bool = False) -> None:
    """The same chunk loop with the softmax-gradient kernel on a second CUDA stream and two logits
    buffers: chunk i's gradient pass runs while the current stream computes chunk i + 1's logits or
    chunk i - 1's dh and dW. Every kernel and its inputs are the ones of the one-stream loop, and the
    matmuls (dW's accumulation order included) stay in order on the current stream, so the results
    are bitwise the same."""
    n, V = h.shape[0], W.shape[0]
    main = torch.cuda.current_stream(h.device)
    side = _SIDE_STREAMS.get(h.device)
    if side is None:
        side = _SIDE_STREAMS[h.device] = torch.cuda.Stream(h.device)
    rows = min(chunk, n)
    bufs = [torch.empty(rows, V, dtype=h.dtype, device=h.device) for _ in range(2)]
    spans = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]
    done = []

    def logits_then_grad(i):
        s, e = spans[i]
        g = bufs[i % 2][: e - s]
        torch.mm(h[s:e], W.t(), out=g)  # waits (stream order) for the dh / dW of chunk i - 2, the buffer's last reader
        ready = torch.cuda.Event()
        ready.record(main)
        side.wait_event(ready)
        with torch.cuda.stream(side):
            _ce_grad(g, t[s:e], a[s:e], b[s:e], ce[s:e], lse[s:e], fixed=fixed)
            ev = torch.cuda.Event()
            ev.record(side)
        done.append(ev)

    logits_then_grad(0)
    for i, (s, e) in enumerate(spans):
        if i + 1 < len(spans):
            logits_then_grad(i + 1)
        main.wait_event(done[i])  # also orders every later use of ce, lse and the buffers after the side stream
        g = bufs[i % 2][: e - s]
        if dh is not None:
            torch.mm(g, W, out=dh[s:e])
        if dW is not None:
            _dW_update(dW, g, h[s:e])


STREAMS = 1  # 2: _chunks_two_streams on CUDA (same results)


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, a, b, chunk, streams, fixed=False):
        dtype = compute_dtype(hidden)
        d = hidden.shape[-1]
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            h = hidden.reshape(-1, d).to(dtype)
            W = weight.to(dtype)
            t = targets.reshape(-1)
            a, b = a.reshape(-1).to(dtype), b.reshape(-1).to(dtype)
            n = h.shape[0]
            ce = torch.empty(n, dtype=dtype, device=h.device)
            lse = torch.empty_like(ce)
            dh = torch.empty_like(h) if ctx.needs_input_grad[0] else None
            dW = torch.zeros_like(W) if ctx.needs_input_grad[1] else None
            two = streams == 2 and h.is_cuda and triton is not None and n > chunk
            if fixed:  # loss.ce_form "fixed": the same loops, the fixed kernel (the standard call below is untouched)
                (_chunks_two_streams if two else _chunks_one_stream)(h, W, t, a, b, ce, lse, dh, dW, chunk, fixed=True)
            else:
                (_chunks_two_streams if two else _chunks_one_stream)(h, W, t, a, b, ce, lse, dh, dW, chunk)
            loss = (a * ce + b * lse * lse).sum()
        ctx.save_for_backward(dh, dW)
        ctx.hidden_meta = (hidden.dtype, hidden.shape)
        ctx.weight_dtype = weight.dtype
        ctx.mark_non_differentiable(ce, lse)
        return loss, ce.reshape(targets.shape), lse.reshape(targets.shape)

    @staticmethod
    def backward(ctx, g_loss, g_ce, g_lse):
        dh, dW = ctx.saved_tensors
        h_dtype, h_shape = ctx.hidden_meta
        gh = None if dh is None else (dh * g_loss).to(h_dtype).reshape(h_shape)
        gW = None if dW is None else (dW * g_loss).to(ctx.weight_dtype)
        return gh, gW, None, None, None, None, None, None


def fused_linear_ce(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, a: torch.Tensor,
                    b: torch.Tensor, chunk: int | None = None, streams: int | None = None, precision: str = "fp32",
                    ce_form: str = "standard") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """hidden (..., d), weight (V, d), targets (...), a and b (...) -> (loss, ce, lse).

    loss = sum(a * ce + b * lse^2) with ce, lse as in ced.train.chunked_loss.token_ce. The
    gradients of loss with respect to hidden and weight are computed here, chunk by chunk, and
    returned by backward scaled by the incoming gradient. ce and lse carry no gradient.
    chunk: positions per chunk (default CHUNK); streams: 1 or 2 (default STREAMS; see
    _chunks_two_streams). precision (TrainConfig.head_precision): "fp32", this function as it always was; "bf16",
    _FusedLinearCEBf16 (bf16 operands, float32 accumulation; one stream: streams 2 is refused). ce_form
    (TrainConfig.loss.ce_form; ced/train/ce_fix.py): "standard", this function as it always was; "fixed", the fixed
    hard-target cross-entropy (ce = softplus(LSE_{j != t}(z_j) - z_t); the target's gradient element
    -a sigmoid(LSE_{j != t}(z_j) - z_t) + 2 b lse softmax_t), in the same loops and the same one pass over the vocabulary."""
    fixed = ce_fix.check_ce_form(ce_form) == "fixed"
    if precision != "fp32":
        check_head_precision(precision)
        if (STREAMS if streams is None else streams) != 1:  # the argument, or the module default when it is left out
            raise ValueError(f"fused_linear_ce: streams {STREAMS if streams is None else streams} is built for "
                             "head_precision 'fp32'; 'bf16' runs one stream")
        return _FusedLinearCEBf16.apply(hidden, weight, targets, a, b, CHUNK if chunk is None else chunk, fixed)
    return _FusedLinearCE.apply(hidden, weight, targets, a, b, CHUNK if chunk is None else chunk,
                                STREAMS if streams is None else streams, fixed)


# ----------------------------------------------------------------------------------------------
# Several targets per row (Token Superposition Training, ced/train/tst.py): one logit row scored against K targets.
#
# The loss of one call is
#
#     sum_i  sum_k a_ik * ce_ik  +  b_i * lse_i^2,      ce_ik = lse_i - logit_i[target_ik],
#
# with weights a (n, K) and b (n,) fixed before the call (a bag phase: the bag weights over the valid targets and the
# normaliser; the multi-hot aux: the next token's weight and the aux weight). The gradient with respect to row i's logits
#
#     softmax_i * (A_i + 2 b_i lse_i) - sum_k a_ik * onehot(target_ik),        A_i = sum_k a_ik,
#
# is formed in place as for one target: the softmax part by the same kernel (on CUDA the Triton kernel of fused_linear_ce
# with the row weight A_i and the target -1, which no column matches, so it subtracts nothing; elsewhere PyTorch), then
# the K one-hot parts by K scatter-adds of one column each (one index per row and call: no two writes meet, so the
# order of the sum is fixed: column 0, 1, ..., K - 1; a token twice in a bag gets both). The target logits are read
# before the kernel overwrites the row. Nothing of fused_linear_ce changes: a call with K = 1 and target -1 never
# happens there, and the one-target path is not touched.


def _ce_grad_multi_torch(logits, targets, a, b, ce, lse) -> None:
    """In place: logits (n, V) -> their gradient; fills ce (n, K) and lse (n,). Reference version (any device)."""
    l = torch.logsumexp(logits, dim=-1)
    ce.copy_(l[:, None] - logits.gather(1, targets))
    lse.copy_(l)
    logits.sub_(l[:, None]).exp_().mul_((a.sum(1) + 2 * b * l)[:, None])
    for k in range(targets.shape[1]):
        logits.scatter_add_(1, targets[:, k : k + 1], -a[:, k : k + 1])


def _multi_softmax_pass(logits, a, b, lse, cfg: dict) -> None:
    """The fixed multi-target form's softmax part: logits (n, V) -> (A + 2 b lse) softmax in place, lse filled; the Triton
    kernel with the target -1 (no column matches, as _ce_grad_multi's) on CUDA, PyTorch elsewhere."""
    if logits.is_cuda and triton is not None:
        n, V = logits.shape
        none = torch.full((n,), -1, dtype=torch.long, device=logits.device)
        scratch = torch.empty_like(lse)
        grid = n
        if cfg["programs_per_sm"]:
            sms = torch.cuda.get_device_properties(logits.device).multi_processor_count
            grid = min(n, cfg["programs_per_sm"] * sms)
        _ce_grad_kernel[(grid,)](logits, logits.stride(0), none, a.sum(1).contiguous(), b, scratch, lse, n, V,
                                 BLOCK=cfg["block"], num_warps=cfg["num_warps"])
    else:
        l = torch.logsumexp(logits, dim=-1)
        lse.copy_(l)
        logits.sub_(l[:, None]).exp_().mul_((a.sum(1) + 2 * b * l)[:, None])


def _ce_grad_multi_fixed(logits, targets, a, b, ce, lse, config: dict | None = None) -> None:
    """_ce_grad_multi with loss.ce_form "fixed" (ced/train/ce_fix.py): each of the K terms is a hard-target cross-entropy of the same
    row and gets the fix. Before the softmax pass: each term's z_t and L'_k = LSE_{j != t_k}(z_j) - z_{t_k} (ce_fix.multi_prep:
    one argmax and one masked log-sum-exp of the chunk, the logits restored, whatever K is; the one-target kernel does this inside
    its one pass); the softmax pass as the standard form's; then ce[:, k] = softplus(L'_k) and the K target columns rewritten ((A
    - A_j + 2 b lse) softmax_j - A_j sigmoid(L'_j)) instead of the K scatter-adds of -a_k. Where the Triton kernels are routed
    (CUDA, or TRITON_INTERPRET=1: ce_fix.use_triton) the whole of it is ONE kernel
    (ced/kernels/ce_fix_triton.ce_grad_multi_fixed_kernel: the same two passes over the chunk as the standard multi form's, no
    extra pass), between one gather of the K target logits and one scatter of the K new column values
    (ce_fix.ce_grad_multi_fixed). Elsewhere the PyTorch passes below, the reference (two extra passes over the chunk; the speed
    check of 2026-10-26 measured them at ratio 1.61 on CUDA, which the kernel replaced)."""
    if logits.is_cuda:
        check_kernel_logits(logits)
    if ce_fix.use_triton(logits):
        ce_fix.ce_grad_multi_fixed(logits, targets, a, b, ce, lse, CE_CONFIG if config is None else config)
        return
    xt, lp = ce_fix.multi_prep(logits, targets)
    _multi_softmax_pass(logits, a, b, lse, CE_CONFIG if config is None else config)
    ce.copy_(ce_fix.multi_ce(lp))
    ce_fix.multi_fix_targets(logits, targets, a, b, lse, xt, lp)


def _ce_grad_multi(logits, targets, a, b, ce, lse, config: dict | None = None, fixed: bool = False) -> None:
    if fixed:
        _ce_grad_multi_fixed(logits, targets, a, b, ce, lse, config)
        return
    if logits.is_cuda and triton is not None:
        check_kernel_logits(logits)
        cfg = CE_CONFIG if config is None else config
        n, V = logits.shape
        xt = logits.gather(1, targets)  # the target logits, before the kernel overwrites the row
        none = torch.full((n,), -1, dtype=targets.dtype, device=targets.device)  # no column matches: no one-hot part
        scratch = torch.empty_like(lse)  # the kernel's ce of target -1 (lse itself): not used
        grid = n
        if cfg["programs_per_sm"]:
            sms = torch.cuda.get_device_properties(logits.device).multi_processor_count
            grid = min(n, cfg["programs_per_sm"] * sms)
        _ce_grad_kernel[(grid,)](logits, logits.stride(0), none, a.sum(1).contiguous(), b, scratch, lse, n, V,
                                 BLOCK=cfg["block"], num_warps=cfg["num_warps"])
        ce.copy_(lse[:, None] - xt)
        for k in range(targets.shape[1]):
            logits.scatter_add_(1, targets[:, k : k + 1], -a[:, k : k + 1])
    else:
        _ce_grad_multi_torch(logits, targets, a, b, ce, lse)


def _chunks_multi(h, W, t, a, b, ce, lse, dh, dW, chunk, fixed: bool = False) -> None:
    """_chunks_one_stream with K targets per row."""
    n, V = h.shape[0], W.shape[0]
    buf = torch.empty(min(chunk, n), V, dtype=h.dtype, device=h.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        g = buf[: e - s]
        torch.mm(h[s:e], W.t(), out=g)
        _ce_grad_multi(g, t[s:e], a[s:e], b[s:e], ce[s:e], lse[s:e], fixed=fixed)
        if dh is not None:
            torch.mm(g, W, out=dh[s:e])
        if dW is not None:
            _dW_update(dW, g, h[s:e])


class _FusedLinearMultiCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, a, b, chunk, fixed=False):
        dtype = compute_dtype(hidden)
        d = hidden.shape[-1]
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            h = hidden.reshape(-1, d).to(dtype)
            W = weight.to(dtype)
            K = targets.shape[-1]
            t = targets.reshape(-1, K)
            a, b = a.reshape(-1, K).to(dtype), b.reshape(-1).to(dtype)
            n = h.shape[0]
            ce = torch.empty(n, K, dtype=dtype, device=h.device)
            lse = torch.empty(n, dtype=dtype, device=h.device)
            dh = torch.empty_like(h) if ctx.needs_input_grad[0] else None
            dW = torch.zeros_like(W) if ctx.needs_input_grad[1] else None
            if fixed:
                _chunks_multi(h, W, t, a, b, ce, lse, dh, dW, chunk, fixed=True)
            else:
                _chunks_multi(h, W, t, a, b, ce, lse, dh, dW, chunk)
            loss = (a * ce).sum() + (b * lse * lse).sum()
        ctx.save_for_backward(dh, dW)
        ctx.hidden_meta = (hidden.dtype, hidden.shape)
        ctx.weight_dtype = weight.dtype
        ctx.mark_non_differentiable(ce, lse)
        return loss, ce.reshape(targets.shape), lse.reshape(targets.shape[:-1])

    @staticmethod
    def backward(ctx, g_loss, g_ce, g_lse):
        dh, dW = ctx.saved_tensors
        h_dtype, h_shape = ctx.hidden_meta
        gh = None if dh is None else (dh * g_loss).to(h_dtype).reshape(h_shape)
        gW = None if dW is None else (dW * g_loss).to(ctx.weight_dtype)
        return gh, gW, None, None, None, None, None


def fused_linear_multi_ce(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, a: torch.Tensor,
                          b: torch.Tensor, chunk: int | None = None, precision: str = "fp32", ce_form: str = "standard",
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """hidden (..., d), weight (V, d), targets (..., K), a (..., K), b (...) -> (loss, ce (..., K), lse (...)).

    loss = sum(a * ce) + sum(b * lse^2), ce[..., k] = lse - logit[targets[..., k]] (the comment above). The gradients of
    loss with respect to hidden and weight are computed here, chunk by chunk, and returned by backward scaled by the
    incoming gradient (the returned loss must enter the training loss linearly, as fused_linear_ce's). ce and lse carry
    no gradient. chunk: positions per chunk (default CHUNK). precision (TrainConfig.head_precision): "fp32", this
    function as it always was; "bf16", _FusedLinearMultiCEBf16. ce_form (TrainConfig.loss.ce_form): "standard", as it
    always was; "fixed", the fixed cross-entropy on each of the K hard-target terms (_ce_grad_multi_fixed)."""
    fixed = ce_fix.check_ce_form(ce_form) == "fixed"
    if precision != "fp32":
        check_head_precision(precision)
        return _FusedLinearMultiCEBf16.apply(hidden, weight, targets, a, b, CHUNK if chunk is None else chunk, fixed)
    return _FusedLinearMultiCE.apply(hidden, weight, targets, a, b, CHUNK if chunk is None else chunk, fixed)


# ----------------------------------------------------------------------------------------------
# The Copy-Paste head's mixture (ModelConfig.cp_head; ced/model/cp_head.py, whose docstring states the form).
#
# The loss of one call is
#
#     sum_i  a_i * ce_i + b_i * lse_i^2,      ce_i = -log p(y_i) of the mixture,   lse_i = LSE_V of row i,
#
# the base's form with the mixture's CE in place of the LM head's (the z-loss stays the base's term on the LM head's
# V rows: with lambda forced to 0 the call is the base's call bit for bit). The mixture reads the V logits only through
# LSE_V and z_y, so the gradient with respect to row i's V logits is
#
#     softmax_i * (g_lse_i + 2 b_i lse_i) - g_zy_i * onehot(y_i)
#
# (cp_head.mixture_grads gives g_lse and g_zy in closed form). Per chunk: the logits (the base's GEMM), z_y gathered
# before the softmax-gradient kernel overwrites them, then THE BASE'S KERNEL with the base's weights (a, b): the row
# becomes G0 = (a + 2 b lse) softmax - a onehot, and lse comes back. Then, from lse and the CP terms, the mixture's
# coefficients: G = rho G0 + delta onehot with rho = (g_lse + 2 b lse) / (a + 2 b lse) and delta = rho a - g_zy. The
# row scale is never applied to the (chunk, V) buffer: G = rho (G0 + (delta / rho) onehot), so one element per row is
# added to G0 (a scatter of one column per row) and rho scales the (chunk, d) side of the two GEMMs instead:
# dh = rho * (G0' W) and dW += G0'^T (rho * h). At lambda = 0 (forced, or exactly 0) g_lse = a and g_zy = a bit for
# bit, so rho = 1.0, the added element is 0 and both scalings multiply by 1.0: the base's gradients, bit for bit, from
# the base's own kernel and GEMMs on every device. A row whose rho is 0 (c = 1 and no LM share: G = 0) adds nothing.
# The gradients of z_copy, log s, log(1 - s) and log p_CP(y) are per row and are returned by backward for autograd to
# carry into the CP head.


class _FusedLinearCPCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, z_c, log_s, log_1ms, log_pcp, targets, a, b, chunk, force_zero):
        from ced.model.cp_head import mixture_grads

        dtype = compute_dtype(hidden)
        d = hidden.shape[-1]
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            h = hidden.reshape(-1, d).to(dtype)
            W = weight.to(dtype)
            t = targets.reshape(-1)
            a, b = a.reshape(-1).to(dtype), b.reshape(-1).to(dtype)
            n, V = h.shape[0], W.shape[0]
            ce_lm = torch.empty(n, dtype=dtype, device=h.device)
            lse = torch.empty_like(ce_lm)
            ce = torch.empty_like(ce_lm)
            lam = torch.zeros_like(ce_lm)
            cp = None if force_zero else [x.reshape(-1).to(dtype) for x in (z_c, log_s, log_1ms, log_pcp)]
            g_cp = None if force_zero else torch.zeros(4, n, dtype=dtype, device=h.device)
            dh = torch.empty_like(h) if ctx.needs_input_grad[0] else None
            dW = torch.zeros_like(W) if ctx.needs_input_grad[1] else None
            buf = torch.empty(min(chunk, n), V, dtype=h.dtype, device=h.device)
            one = torch.ones((), dtype=dtype, device=h.device)
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                g = buf[: e - s]
                torch.mm(h[s:e], W.t(), out=g)
                zy = g.gather(1, t[s:e, None]).squeeze(1)  # before the kernel overwrites the row
                _ce_grad(g, t[s:e], a[s:e], b[s:e], ce_lm[s:e], lse[s:e])
                l, aa, bb = lse[s:e], a[s:e], b[s:e]
                if force_zero:
                    g_lse, g_zy = aa, aa
                    ce[s:e] = ce_lm[s:e]
                else:
                    m = mixture_grads(l, zy, *(x[s:e] for x in cp), aa)
                    g_lse, g_zy = m["g_lse"], m["g_zy"]
                    ce[s:e] = m["ce"]
                    lam[s:e] = torch.exp(m["log_lam"])
                    for i, key in enumerate(("g_zc", "g_ls", "g_l1s", "g_lp")):
                        g_cp[i, s:e] = m[key]
                num = g_lse + 2 * bb * l
                den = aa + 2 * bb * l
                rho = torch.where(den == 0, one, num / den)
                shift = torch.where(rho == 0, torch.zeros_like(rho), (rho * aa - g_zy) / rho)
                g.scatter_add_(1, t[s:e, None], shift[:, None])
                if dh is not None:
                    torch.mm(g, W, out=dh[s:e])
                    dh[s:e].mul_(rho[:, None])
                if dW is not None:
                    _dW_update(dW, g, h[s:e] * rho[:, None])
            loss = (a * ce + b * lse * lse).sum()
        ctx.save_for_backward(dh, dW, g_cp)
        ctx.hidden_meta = (hidden.dtype, hidden.shape)
        ctx.weight_dtype = weight.dtype
        ctx.cp_meta = None if force_zero else [(x.dtype, x.shape) for x in (z_c, log_s, log_1ms, log_pcp)]
        ctx.mark_non_differentiable(ce, lse, ce_lm, lam)
        shape = targets.shape
        return loss, ce.reshape(shape), lse.reshape(shape), ce_lm.reshape(shape), lam.reshape(shape)

    @staticmethod
    def backward(ctx, g_loss, g_ce, g_lse, g_ce_lm, g_lam):
        dh, dW, g_cp = ctx.saved_tensors
        h_dtype, h_shape = ctx.hidden_meta
        gh = None if dh is None else (dh * g_loss).to(h_dtype).reshape(h_shape)
        gW = None if dW is None else (dW * g_loss).to(ctx.weight_dtype)
        cp = [None] * 4
        if ctx.cp_meta is not None:
            cp = [(g_cp[i] * g_loss).to(dt).reshape(sh) for i, (dt, sh) in enumerate(ctx.cp_meta)]
        return (gh, gW, *cp, None, None, None, None, None)


CP_MIXES = ("post", "kernel")  # fused_linear_cp_ce's mix: the as-built correction after the base's kernel, or in the kernel


def fused_linear_cp_ce(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, a: torch.Tensor,
                       b: torch.Tensor, z_c: torch.Tensor | None = None, log_s: torch.Tensor | None = None,
                       log_1ms: torch.Tensor | None = None, log_pcp: torch.Tensor | None = None,
                       force_zero: bool = False, chunk: int | None = None, precision: str = "fp32", mix: str = "post"):
    """The Copy-Paste head's mixture loss (the comment above): hidden (n, d), weight (V, d) the LM head's V rows,
    targets, a, b (n,), and per row z_copy, log s, log(1 - s), log p_CP(y) (n,) with their autograd graphs (None each
    with force_zero: lambda = 0, p = p_LM). Returns (loss, ce, lse, ce_lm, lam): loss = sum(a * ce + b * lse^2) with ce
    the mixture's -log p(y); lse = LSE_V; ce_lm = LSE_V - z_y (p_LM's CE); lam = lambda (0 with force_zero). The
    gradients of loss with respect to hidden, weight and the four per-row inputs are formed here and returned by
    backward scaled by the incoming gradient (the loss must enter the training loss linearly, as fused_linear_ce's).
    ce, lse, ce_lm and lam carry no gradient. chunk: positions per chunk (default CHUNK). precision
    (TrainConfig.head_precision): "fp32", this function as it always was; "bf16", _FusedLinearCPCEBf16 (the LM head's
    V rows in bf16 with float32 accumulation, as fused_linear_ce's; the CP terms are the caller's, unchanged). mix: "post"
    (this function as it was: the base's kernel, then the per-row correction) or "kernel" (model.cp_head.kernel "fused":
    the mixture inside the CE kernel, _FusedLinearCPMix; with force_zero the "post" path runs, the base's bit for bit)."""
    if mix not in CP_MIXES:
        raise ValueError(f"fused_linear_cp_ce: mix {mix!r}: one of {CP_MIXES}")
    if force_zero:
        assert z_c is None and log_s is None and log_1ms is None and log_pcp is None, \
            "lambda forced to 0: the CP head is not evaluated"
    else:
        assert all(x is not None for x in (z_c, log_s, log_1ms, log_pcp)), "the mixture needs the four CP inputs"
    if precision != "fp32":
        check_head_precision(precision)
    if mix == "kernel" and not force_zero:
        return _FusedLinearCPMix.apply(hidden, weight, z_c, log_s, log_1ms, log_pcp, targets, a, b,
                                       CHUNK if chunk is None else chunk, precision)
    if precision != "fp32":
        return _FusedLinearCPCEBf16.apply(hidden, weight, z_c, log_s, log_1ms, log_pcp, targets, a, b,
                                          CHUNK if chunk is None else chunk, force_zero)
    return _FusedLinearCPCE.apply(hidden, weight, z_c, log_s, log_1ms, log_pcp, targets, a, b,
                                  CHUNK if chunk is None else chunk, force_zero)


# ---------------------------------------------------------------------------------------------- bf16 operands
# (TrainConfig.head_precision "bf16"; ced/train/head_precision.py).
#
# Per chunk, with hb = bf16(h) and Wb = bf16(W) formed once a call (h and W first in the compute dtype, so each is
# rounded once):
#
#     x  = hb[s:e] @ Wb^T            bf16 operands, float32 accumulation, float32 logits          (mm_acc)
#     ce, lse, G = softmax(x) (a + 2 b lse) - a onehot   in float32, from x                (_ce_grad_out)
#     Gb = bf16(G)                   written by the kernel's store, round to nearest even; x is left as it is
#     dh[s:e] = Gb @ Wb              bf16 operands, float32 accumulation and output              (mm_acc)
#     dW     += Gb^T @ hb[s:e]       the same, accumulated into the float32 dW (beta = 1 on CUDA) (addmm_acc_)
#
# The compute dtype is compute_dtype(hidden): float32 (float64 for float64 inputs, the tests' exact form: the same
# roundings of the operands, everything else in float64). The multi-target and Copy-Paste forms change only what the
# fp32 path changes between the three functions: the multi-target form keeps today's kernel and K scatters in float32
# and rounds the finished gradient once (one more pass over the chunk); the Copy-Paste form takes the base's bf16
# gradient from the same kernel and, in a row whose shift is not 0, replaces its target element by bf16(G0_y + shift)
# with G0_y recomputed in float32 from z_y and LSE_V (each element rounded once; a row with shift 0, every row at
# lambda = 0, keeps the kernel's element, so that call is the base's bit for bit), and scales the (chunk, d) side of
# the two backward matmuls by rho, as the fp32 form does.


if triton is not None:

    @triton.jit
    def _ce_grad_out_kernel(logits_ptr, stride, out_ptr, out_stride, targets_ptr, a_ptr, b_ptr, ce_ptr, lse_ptr, n, V,
                            BLOCK: tl.constexpr):
        """_ce_grad_kernel with the gradient stored into a second buffer (out_ptr, bfloat16) instead of over the logits:
        the same arithmetic, in the logits' dtype (float32), statement for statement; the store rounds g to out's dtype
        (round to nearest even). The logits are only read, so the in-place race of _ce_grad_kernel cannot arise; the
        target's logit is still taken from the first pass's tiles, as there."""
        cols = tl.arange(0, BLOCK)
        dtype = logits_ptr.dtype.element_ty
        for row in range(tl.program_id(0), n, tl.num_programs(0)):
            base = logits_ptr + row.to(tl.int64) * stride
            obase = out_ptr + row.to(tl.int64) * out_stride
            target = tl.load(targets_ptr + row)
            m = tl.full([BLOCK], -1e30, dtype)
            s = tl.zeros([BLOCK], dtype)
            xt = tl.zeros([BLOCK], dtype)
            for start in range(0, V, BLOCK):
                x = tl.load(base + start + cols, mask=start + cols < V, other=float("-inf"))
                m_new = tl.maximum(m, x)
                s = s * tl.exp(m - m_new) + tl.exp(x - m_new)
                m = m_new
                xt += tl.where(start + cols == target, x, 0.0)
            mx = tl.max(m, axis=0)
            lse = mx + tl.log(tl.sum(s * tl.exp(m - mx), axis=0))
            a = tl.load(a_ptr + row)
            b = tl.load(b_ptr + row)
            tl.store(ce_ptr + row, lse - tl.sum(xt, axis=0))
            tl.store(lse_ptr + row, lse)
            coef = a + 2 * b * lse
            for start in range(0, V, BLOCK):
                idx = start + cols
                x = tl.load(base + idx, mask=idx < V, other=0.0)
                g = tl.exp(x - lse) * coef - tl.where(idx == target, a, 0.0)
                tl.store(obase + idx, g.to(out_ptr.dtype.element_ty, fp_downcast_rounding="rtne"), mask=idx < V)


def _ce_grad_out(logits, out, targets, a, b, ce, lse, config: dict | None = None, fixed: bool = False) -> None:
    """Fills ce and lse and writes the logit gradient, rounded to out's dtype (bf16), into out. On CUDA with float32
    logits: the Triton kernel above, which only reads the logits. Elsewhere (the CPU, float64 logits on any device):
    the reference version in place over the logits, then one cast into out (the same values). fixed (loss.ce_form "fixed",
    ced/train/ce_fix.py): the fixed cross-entropy's kernel and reference in the same places."""
    if fixed:
        assert out.dtype == OPERAND and out.shape == logits.shape, (out.dtype, out.shape, logits.shape)
        ce_fix.ce_grad_out_fixed(logits, out, targets, a, b, ce, lse, CE_CONFIG if config is None else config)
        return
    if logits.is_cuda and triton is not None and logits.dtype == torch.float32:
        assert out.dtype == OPERAND and out.shape == logits.shape, (out.dtype, out.shape, logits.shape)
        cfg = CE_CONFIG if config is None else config
        n, V = logits.shape
        grid = n
        if cfg["programs_per_sm"]:
            sms = torch.cuda.get_device_properties(logits.device).multi_processor_count
            grid = min(n, cfg["programs_per_sm"] * sms)
        _ce_grad_out_kernel[(grid,)](logits, logits.stride(0), out, out.stride(0), targets, a, b, ce, lse, n, V,
                                     BLOCK=cfg["block"], num_warps=cfg["num_warps"])
    else:
        _ce_grad_torch(logits, targets, a, b, ce, lse)
        out.copy_(logits)


def _bf16_operands(hidden: torch.Tensor, weight: torch.Tensor):
    """(compute dtype, h in it (n, d), bf16(h), bf16(W)): each operand rounded once, from the compute dtype."""
    dtype = compute_dtype(hidden)
    h = hidden.reshape(-1, hidden.shape[-1]).to(dtype)
    return dtype, h, h.to(OPERAND), weight.to(dtype).to(OPERAND)


def _chunks_bf16(hb, wb, t, a, b, ce, lse, dh, dW, chunk, fixed: bool = False) -> None:
    """The chunk loop of the comment above: hb (n, d), wb (V, d) bf16; ce, lse, dh, dW in the compute dtype."""
    n, V = hb.shape[0], wb.shape[0]
    acc = ce.dtype
    rows = min(chunk, n)
    xbuf = torch.empty(rows, V, dtype=acc, device=hb.device)
    gbuf = torch.empty(rows, V, dtype=OPERAND, device=hb.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        x, g = xbuf[: e - s], gbuf[: e - s]
        mm_acc(hb[s:e], wb.t(), acc, out=x)
        _ce_grad_out(x, g, t[s:e], a[s:e], b[s:e], ce[s:e], lse[s:e], fixed=fixed)
        if dh is not None:
            mm_acc(g, wb, acc, out=dh[s:e])
        if dW is not None:
            addmm_acc_(dW, g.t(), hb[s:e])


class _FusedLinearCEBf16(torch.autograd.Function):
    """fused_linear_ce with head_precision "bf16" (the comment above); backward as _FusedLinearCE's."""

    @staticmethod
    def forward(ctx, hidden, weight, targets, a, b, chunk, fixed=False):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            dtype, h, hb, wb = _bf16_operands(hidden, weight)
            t = targets.reshape(-1)
            a, b = a.reshape(-1).to(dtype), b.reshape(-1).to(dtype)
            n = h.shape[0]
            ce = torch.empty(n, dtype=dtype, device=h.device)
            lse = torch.empty_like(ce)
            dh = torch.empty_like(h) if ctx.needs_input_grad[0] else None
            dW = torch.zeros(wb.shape, dtype=dtype, device=h.device) if ctx.needs_input_grad[1] else None
            if fixed:
                _chunks_bf16(hb, wb, t, a, b, ce, lse, dh, dW, chunk, fixed=True)
            else:
                _chunks_bf16(hb, wb, t, a, b, ce, lse, dh, dW, chunk)
            loss = (a * ce + b * lse * lse).sum()
        ctx.save_for_backward(dh, dW)
        ctx.hidden_meta = (hidden.dtype, hidden.shape)
        ctx.weight_dtype = weight.dtype
        ctx.mark_non_differentiable(ce, lse)
        return loss, ce.reshape(targets.shape), lse.reshape(targets.shape)

    @staticmethod
    def backward(ctx, g_loss, g_ce, g_lse):
        dh, dW = ctx.saved_tensors
        h_dtype, h_shape = ctx.hidden_meta
        gh = None if dh is None else (dh * g_loss).to(h_dtype).reshape(h_shape)
        gW = None if dW is None else (dW * g_loss).to(ctx.weight_dtype)
        return gh, gW, None, None, None, None, None


def _chunks_multi_bf16(hb, wb, t, a, b, ce, lse, dh, dW, chunk, fixed: bool = False) -> None:
    """_chunks_multi with bf16 operands: the gradient formed in float32 by today's multi-target code (_ce_grad_multi:
    the kernel and the K scatters), then rounded once to bf16 (one pass over the chunk), then the two bf16 matmuls."""
    n, V = hb.shape[0], wb.shape[0]
    acc = lse.dtype
    rows = min(chunk, n)
    xbuf = torch.empty(rows, V, dtype=acc, device=hb.device)
    gbuf = torch.empty(rows, V, dtype=OPERAND, device=hb.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        x, g = xbuf[: e - s], gbuf[: e - s]
        mm_acc(hb[s:e], wb.t(), acc, out=x)
        _ce_grad_multi(x, t[s:e], a[s:e], b[s:e], ce[s:e], lse[s:e], fixed=fixed)
        g.copy_(x)
        if dh is not None:
            mm_acc(g, wb, acc, out=dh[s:e])
        if dW is not None:
            addmm_acc_(dW, g.t(), hb[s:e])


class _FusedLinearMultiCEBf16(torch.autograd.Function):
    """fused_linear_multi_ce with head_precision "bf16"; backward as _FusedLinearMultiCE's."""

    @staticmethod
    def forward(ctx, hidden, weight, targets, a, b, chunk, fixed=False):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            dtype, h, hb, wb = _bf16_operands(hidden, weight)
            K = targets.shape[-1]
            t = targets.reshape(-1, K)
            a, b = a.reshape(-1, K).to(dtype), b.reshape(-1).to(dtype)
            n = h.shape[0]
            ce = torch.empty(n, K, dtype=dtype, device=h.device)
            lse = torch.empty(n, dtype=dtype, device=h.device)
            dh = torch.empty_like(h) if ctx.needs_input_grad[0] else None
            dW = torch.zeros(wb.shape, dtype=dtype, device=h.device) if ctx.needs_input_grad[1] else None
            if fixed:
                _chunks_multi_bf16(hb, wb, t, a, b, ce, lse, dh, dW, chunk, fixed=True)
            else:
                _chunks_multi_bf16(hb, wb, t, a, b, ce, lse, dh, dW, chunk)
            loss = (a * ce).sum() + (b * lse * lse).sum()
        ctx.save_for_backward(dh, dW)
        ctx.hidden_meta = (hidden.dtype, hidden.shape)
        ctx.weight_dtype = weight.dtype
        ctx.mark_non_differentiable(ce, lse)
        return loss, ce.reshape(targets.shape), lse.reshape(targets.shape[:-1])

    @staticmethod
    def backward(ctx, g_loss, g_ce, g_lse):
        dh, dW = ctx.saved_tensors
        h_dtype, h_shape = ctx.hidden_meta
        gh = None if dh is None else (dh * g_loss).to(h_dtype).reshape(h_shape)
        gW = None if dW is None else (dW * g_loss).to(ctx.weight_dtype)
        return gh, gW, None, None, None, None, None


class _FusedLinearCPCEBf16(torch.autograd.Function):
    """fused_linear_cp_ce with head_precision "bf16": _FusedLinearCPCE's chunk loop with the base's bf16 matmuls and
    gradient kernel (_ce_grad_out). What the mixture sees: LSE_V and z_y from the bf16-operand float32 logits (z_y
    gathered from them before the kernel); z_copy, log s, log(1 - s) and log p_CP(y) as the caller computed them (the
    CP head's own terms: float32, autograd, unchanged). The matmuls' V-column operand is the fp32 form's G0 + shift
    onehot (G = rho times it), each element rounded once to bf16: the kernel rounds G0, and in a row whose shift is not
    0 the target element is replaced by bf16(G0_y + shift), G0_y = (a + 2 b LSE_V) exp(z_y - LSE_V) - a recomputed in
    float32; rho scales dh after its matmul, and h before it is rounded to bf16 for dW. At rho = 1 and shift 0 (lambda = 0, forced or exactly
    0) every operand and every call is the base's (_FusedLinearCEBf16): its loss and gradients bit for bit."""

    @staticmethod
    def forward(ctx, hidden, weight, z_c, log_s, log_1ms, log_pcp, targets, a, b, chunk, force_zero):
        from ced.model.cp_head import mixture_grads

        with torch.autocast(device_type=hidden.device.type, enabled=False):
            dtype, h, hb, wb = _bf16_operands(hidden, weight)
            t = targets.reshape(-1)
            a, b = a.reshape(-1).to(dtype), b.reshape(-1).to(dtype)
            n, V = h.shape[0], wb.shape[0]
            ce_lm = torch.empty(n, dtype=dtype, device=h.device)
            lse = torch.empty_like(ce_lm)
            ce = torch.empty_like(ce_lm)
            lam = torch.zeros_like(ce_lm)
            cp = None if force_zero else [x.reshape(-1).to(dtype) for x in (z_c, log_s, log_1ms, log_pcp)]
            g_cp = None if force_zero else torch.zeros(4, n, dtype=dtype, device=h.device)
            dh = torch.empty_like(h) if ctx.needs_input_grad[0] else None
            dW = torch.zeros(wb.shape, dtype=dtype, device=h.device) if ctx.needs_input_grad[1] else None
            rows = min(chunk, n)
            xbuf = torch.empty(rows, V, dtype=dtype, device=h.device)
            gbuf = torch.empty(rows, V, dtype=OPERAND, device=h.device)
            one = torch.ones((), dtype=dtype, device=h.device)
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                x, g = xbuf[: e - s], gbuf[: e - s]
                mm_acc(hb[s:e], wb.t(), dtype, out=x)
                zy = x.gather(1, t[s:e, None]).squeeze(1)  # before the CPU path's kernel overwrites the row
                _ce_grad_out(x, g, t[s:e], a[s:e], b[s:e], ce_lm[s:e], lse[s:e])
                l, aa, bb = lse[s:e], a[s:e], b[s:e]
                if force_zero:
                    g_lse, g_zy = aa, aa
                    ce[s:e] = ce_lm[s:e]
                else:
                    m = mixture_grads(l, zy, *(c[s:e] for c in cp), aa)
                    g_lse, g_zy = m["g_lse"], m["g_zy"]
                    ce[s:e] = m["ce"]
                    lam[s:e] = torch.exp(m["log_lam"])
                    for i, key in enumerate(("g_zc", "g_ls", "g_l1s", "g_lp")):
                        g_cp[i, s:e] = m[key]
                num = g_lse + 2 * bb * l
                den = aa + 2 * bb * l
                rho = torch.where(den == 0, one, num / den)
                shift = torch.where(rho == 0, torch.zeros_like(rho), (rho * aa - g_zy) / rho)
                g0y = (aa + 2 * bb * l) * torch.exp(zy - l) - aa  # G0 at the target, as the kernel forms it
                ty = t[s:e, None]
                g.scatter_(1, ty, torch.where(shift == 0, g.gather(1, ty).squeeze(1), (g0y + shift).to(OPERAND))[:, None])
                if dh is not None:
                    mm_acc(g, wb, dtype, out=dh[s:e])
                    dh[s:e].mul_(rho[:, None])
                if dW is not None:
                    addmm_acc_(dW, g.t(), (h[s:e] * rho[:, None]).to(OPERAND))
            loss = (a * ce + b * lse * lse).sum()
        ctx.save_for_backward(dh, dW, g_cp)
        ctx.hidden_meta = (hidden.dtype, hidden.shape)
        ctx.weight_dtype = weight.dtype
        ctx.cp_meta = None if force_zero else [(x.dtype, x.shape) for x in (z_c, log_s, log_1ms, log_pcp)]
        ctx.mark_non_differentiable(ce, lse, ce_lm, lam)
        shape = targets.shape
        return loss, ce.reshape(shape), lse.reshape(shape), ce_lm.reshape(shape), lam.reshape(shape)

    @staticmethod
    def backward(ctx, g_loss, g_ce, g_lse, g_ce_lm, g_lam):
        return _FusedLinearCPCE.backward(ctx, g_loss, g_ce, g_lse, g_ce_lm, g_lam)


# ---------------------------------------------------------------------------------------------- The Copy-Paste head's
# mixture inside the CE kernel (fused_linear_cp_ce with mix "kernel"; model.cp_head.kernel "fused";
# ced/kernels/cp_pointer.py ce_mix).
#
# Per chunk: the logits (the base's GEMM: float32, or G's bf16 operands with float32 accumulation), then ONE kernel that
# makes the base kernel's two passes over a row with the mixture's per-row arithmetic between them and writes
# G = softmax (g_lse + 2 b LSE_V) - g_zy onehot directly (in place in float32, or rounded once into a bf16 buffer for
# G's bf16 matmuls); then the base's two GEMMs, dh = G W and dW += G^T h, unscaled. Against _FusedLinearCPCE: no
# scatter of a per-row shift, no dh *= rho pass, no (h * rho) temporary for dW, and the closed form's ~60 small kernels
# a chunk become per-row arithmetic in the kernel. The z-loss is the base's term (2 b LSE_V in the coefficient), as G
# has it. Not bit for bit with mix "post" (G = rho (G0 + shift onehot) there, G here: the same number rounded once).


class _FusedLinearCPMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, z_c, log_s, log_1ms, log_pcp, targets, a, b, chunk, precision):
        from ced.kernels.cp_pointer import ce_mix

        bf16 = precision != "fp32"
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            if bf16:
                dtype, h, hb, wb = _bf16_operands(hidden, weight)
                V = wb.shape[0]
            else:
                dtype = compute_dtype(hidden)
                h = hidden.reshape(-1, hidden.shape[-1]).to(dtype)
                W = weight.to(dtype)
                V = W.shape[0]
            t = targets.reshape(-1)
            a, b = a.reshape(-1).to(dtype).contiguous(), b.reshape(-1).to(dtype).contiguous()
            cp = [x.reshape(-1).to(dtype).contiguous() for x in (z_c, log_s, log_1ms, log_pcp)]
            n = h.shape[0]
            ce = torch.empty(n, dtype=dtype, device=h.device)
            lse, ce_lm, lam = torch.empty_like(ce), torch.empty_like(ce), torch.empty_like(ce)
            g_cp = torch.zeros(n, 4, dtype=dtype, device=h.device)  # a row's four CP gradients side by side
            dh = torch.empty_like(h) if ctx.needs_input_grad[0] else None
            dW = torch.zeros(V, h.shape[1], dtype=dtype, device=h.device) if ctx.needs_input_grad[1] else None
            rows = min(chunk, n)
            xbuf = torch.empty(rows, V, dtype=dtype, device=h.device)
            gbuf = torch.empty(rows, V, dtype=OPERAND, device=h.device) if bf16 else None
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                x = xbuf[: e - s]
                if bf16:
                    mm_acc(hb[s:e], wb.t(), dtype, out=x)
                    g = gbuf[: e - s]
                else:
                    torch.mm(h[s:e], W.t(), out=x)
                    g = x
                ce_mix(x, g, t[s:e], a[s:e], b[s:e], *(c[s:e] for c in cp), ce[s:e], lse[s:e], ce_lm[s:e], lam[s:e],
                       g_cp[s:e], CE_CONFIG)
                if bf16:
                    if dh is not None:
                        mm_acc(g, wb, dtype, out=dh[s:e])
                    if dW is not None:
                        addmm_acc_(dW, g.t(), hb[s:e])
                else:
                    if dh is not None:
                        torch.mm(g, W, out=dh[s:e])
                    if dW is not None:
                        _dW_update(dW, g, h[s:e])
            loss = (a * ce + b * lse * lse).sum()
        ctx.save_for_backward(dh, dW, g_cp)
        ctx.hidden_meta = (hidden.dtype, hidden.shape)
        ctx.weight_dtype = weight.dtype
        ctx.cp_meta = [(x.dtype, x.shape) for x in (z_c, log_s, log_1ms, log_pcp)]
        ctx.mark_non_differentiable(ce, lse, ce_lm, lam)
        shape = targets.shape
        return loss, ce.reshape(shape), lse.reshape(shape), ce_lm.reshape(shape), lam.reshape(shape)

    @staticmethod
    def backward(ctx, g_loss, g_ce, g_lse, g_ce_lm, g_lam):
        dh, dW, g_cp = ctx.saved_tensors
        h_dtype, h_shape = ctx.hidden_meta
        gh = None if dh is None else (dh * g_loss).to(h_dtype).reshape(h_shape)
        gW = None if dW is None else (dW * g_loss).to(ctx.weight_dtype)
        cp = [(g_cp[:, i] * g_loss).to(dt).reshape(sh) for i, (dt, sh) in enumerate(ctx.cp_meta)]
        return (gh, gW, *cp, None, None, None, None, None)
