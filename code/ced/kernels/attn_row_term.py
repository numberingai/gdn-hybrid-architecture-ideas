"""The attention backward's row term from P. dP (AttentionConfig.row_term "pdp"). The comparison (tools/analysis/croissattn_grad_compare.py) read
MATTERS on form `attn` (the global attention).

THE DEFECT. A softmax read o = sum_j P_j v_j has the score gradient dz_j = P_j (dP_j - delta), dP_j = dO . v_j, and the exact row term is
delta = sum_j P_j dP_j (= dO . o for the exact o). The compiled flex kernel (torch/_inductor/kernel/flex/flex_attention.py:953-956)
forms delta = sum(out * grad_out) from the STORED output, which is in the query's dtype (bf16): delta_S = delta + dO . (o_stored - o), a
bias that is the same for every key of the row. The true gradient of a peaky row is tiny (P_max ~ 1, dP_max ~ delta), so the bias is
not small against it.

THE FIX, ON A KERNEL WE CANNOT EDIT. The kernel's backward adds the gradient of the log-sum-exp as a shift of delta
(delta_eff = delta_S - g_lse; flex_attention.py:957-960: dz_j = P_j (dP_j - delta_S + g_lse), the term P_j g_lse being the lse's own
gradient). So the corrected row term needs no new backward kernel: an identity node on the pair (o, lse) of each flex call receives
(dO, g_lse) in the backward, forms b = delta_S - delta_F with delta_F = sum_j P_j dP_j from a pre-pass over q, k, v, lse and the same
dO, and hands the kernel g_lse + b: then delta_eff = delta_S - (g_lse + b) = delta_F - g_lse. The forward values are untouched
(the node returns copies of o and lse); the parameters and the kernel's own backward are untouched. `tie_read` is that node (a
torch.library custom op with its autograd registered, like ced::gdn_c2_fwd: opaque to the compiled layer it sits in).

THE PRE-PASS (`row_terms`): per query row, delta_F = sum_j P_j dP_j / sum_j P_j (the row sum of P in the divisor: it is 1 up to the rounding of lse, a
coherent error eta that scales every P of the row alike and would otherwise leave an error eta x delta in the row term, against a true gap
dP_max - delta of (1 - p_max) |dP| on a peaky row, where eta ~ 1e-6 is 5 % of it at 1 - p_max = 2e-5) and delta_S = sum_d dO_d o_d, both float32, one program per block of
BLOCK_M queries (attn_row_term_triton.py): two products a tile (q.k^T, dO.v^T), one exp a pair, no V product into an output, no
stored P. The reference of the arithmetic is `_torch_row_terms` (chunks of TORCH_CHUNK queries, the same statements). The read it
covers is the encoder's own read of a row (a key set with one entry per position of the row, keys.idx = 0 .. S - 1, every key valid,
documents contiguous): the plain read (KIND_PLAIN), the rotary cap's near call (KIND_NEAR: distance <= cap) and far call (KIND_FAR:
distance > cap, the far logits); each flex call is tied separately (its softmax is over its own keys, its dO is the join's
(1 - w) dO or w dO). The far call's stand-in key (a row with no key beyond the cap reads key 0 there) is left out: such a row has
dO = 0 exactly (the join's weight is 0), so b = 0 for it.

WHERE IT RUNS. `applies(q, k, v)`: a gradient needed and a 16-bit query dtype, i.e. where the stored output carries a rounding that
matters. Flex's backward forms its row term from the stored output in both of its forms: the compiled kernel (above) and the dense
math fallback of the CPU and of the uncompiled CUDA call (torch/_higher_order_ops/flex_attention.py sdpa_dense_backward: `sum_scores =
sum(out * grad_out)`, then `grad_softmax_scores - sum_scores + grad_logsumexp`: the same statements, the lse's gradient entering
with the same sign and unit), so the tie is the same node on both and the CPU tests run it against torch's own backward (bf16 leaves,
tests/test_attn_row_term.py). At float32 and float64 the stored output has no rounding that matters and the key is inert. The pre-pass
kernel runs on CUDA (and under TRITON_INTERPRET=1 / tests/tl_emu.py); anywhere else the PyTorch route runs (the CPU tests, the GPU
check's reference).

COST (judgement): two matmuls a tile of the read, i.e. the forward's QK^T and a dO.V^T: about 2 of the backward's 5 matmul passes
(+40 % of the read's backward MMA work) on the flex path. On an optimized baseline (FlashAttention-4's O residual: the forward
keeps o_lo = fp32(o) - bf16(o), the backward's preprocess reads it beside o and dO) the idea's cost is one O-sized bf16 tensor
written and read a layer, no extra matmul: the pre-pass is implementation debt of the flex path.
"""

from __future__ import annotations

import importlib
import os

import torch

ROW_TERMS = ("stored_o", "pdp")
KIND_PLAIN, KIND_NEAR, KIND_FAR = 0, 1, 2
BLOCK_M = 64  # queries a program
BLOCK_N = 64  # keys a tile
NUM_WARPS = 4
NUM_STAGES = 2
TORCH_CHUNK = 256  # queries a chunk of the PyTorch route
REPORT_ENV = "CED_ROW_TERM_REPORT"  # set to 1: the first call prints one "ROW-TERM-RAN" line (the GPU check's proof that the kernel ran)
RAN = {"kernel": 0, "torch": 0}
FORCE_STORED_O = [False]  # the CPU tests: a stand-in flex whose backward forms delta from the stored output is installed (tests/attn_row_term_double.py)
_HAVE_TRITON: list = []


def check_row_term(value) -> str:
    """The key's value, or ValueError by name."""
    if value not in ROW_TERMS:
        raise ValueError(f"attention.row_term {value!r}: one of {ROW_TERMS} ('stored_o': delta = rowsum(dO . O) from the stored output, "
                         "as always; 'pdp': delta = rowsum(P . dP), the alternative row term of the attention backward)")
    return value


def have_triton() -> bool:
    if not _HAVE_TRITON:
        try:
            importlib.import_module("triton")
            _HAVE_TRITON.append(True)
        except ImportError:
            _HAVE_TRITON.append(False)
    return _HAVE_TRITON[0]


ROUTE_ENV = "CED_ROW_TERM_ROUTE"  # "torch": the PyTorch route even where the Triton kernel could run (the GPU check's fallback leg: the tie and the compiled
# flex without the kernel); unset: the kernel where it can run


def use_triton(x: torch.Tensor) -> bool:
    """The Triton kernel for x's device: CUDA, or any device under TRITON_INTERPRET=1 (the interpreter, or the tests' emulation), when Triton
    imports (and CED_ROW_TERM_ROUTE is not "torch")."""
    return (x.is_cuda or os.environ.get("TRITON_INTERPRET") == "1") and os.environ.get(ROUTE_ENV) != "torch" and have_triton()


def kernels():
    """The kernel module, imported now (never cached here: the tests' emulation swaps it in sys.modules)."""
    return importlib.import_module("ced.kernels.attn_row_term_triton")


def report(way: str, what: str) -> None:
    RAN[way] += 1
    if RAN[way] == 1 and os.environ.get(REPORT_ENV) == "1":
        print(f"ROW-TERM-RAN {way} (first call: {what})", flush=True)


def acc_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _prec(x: torch.Tensor) -> str:
    if x.is_cuda and x.dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32:
        return "tf32"
    return "ieee"


def _tl_acc(x: torch.Tensor):
    tl = importlib.import_module("triton.language")
    return tl.float64 if x.dtype == torch.float64 else tl.float32


def interpreting() -> bool:
    """Whether Triton runs as its interpreter (TRITON_INTERPRET=1): the launch's INTERP (attn_row_term_triton's docstring)."""
    return os.environ.get("TRITON_INTERPRET") == "1"


def kernel_head_dim_ok(d: int) -> bool:
    """The kernel's head dimensions: a power of two from 16 to 256 (tl.arange and tl.dot's K)."""
    return 16 <= d <= 256 and (d & (d - 1)) == 0


STORED_LOW = (torch.bfloat16, torch.float16)  # the dtypes whose rounding of the stored output is the defect


def applies(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Whether a flex call on q, k, v needs the tie: a gradient is needed and the output the kernel stores (in the query's dtype) is a 16-bit
    float. Flex's backward forms its row term from that stored output on every device and in both of its forms (the compiled kernel,
    flex_attention.py:953-956, and the dense math fallback of the CPU and the uncompiled CUDA call, torch/_higher_order_ops/flex_attention.py
    sdpa_dense_backward: sum_scores = sum(out * grad_out), the lse's gradient added: the same statements); at float32 and float64 the stored
    output carries no rounding that matters (the key is inert: the tie would add a correction of float round-off). The CPU tests set
    FORCE_STORED_O with their stand-in flex installed (a float64 call whose stored output is rounded to bf16)."""
    needs_grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
    if not needs_grad:
        return False
    return bool(FORCE_STORED_O[0] or q.dtype in STORED_LOW)


# ---------------------------------------------------------------------------------------------------------------------
# the pre-pass and its PyTorch route


def doc_start(doc: torch.Tensor) -> torch.Tensor:
    """(B, L) int32: the first position of each position's document in its row (cp_pointer.doc_bounds: fixed-shape ops, the packing
    contract that document ids increase along a row and are never reused in it)."""
    from ced.kernels.cp_pointer import doc_bounds

    return doc_bounds(doc)[0].to(torch.int32).contiguous()


def _allow(doc: torch.Tensor, a: int, e: int, kind: int, cap: int) -> torch.Tensor:
    """(B, e - a, S) bool: the pairs the call's softmax covers, the queries a .. e - 1 over every key of the row (the same document, not in the
    future; KIND_NEAR: at most `cap` positions back, KIND_FAR: more than `cap`)."""
    L = doc.shape[1]
    qi = torch.arange(a, e, device=doc.device)[:, None]
    ki = torch.arange(L, device=doc.device)[None, :]
    dist = qi - ki
    ok = (doc[:, a:e, None] == doc[:, None, :]) & (dist >= 0)[None]
    if kind == KIND_NEAR:
        ok = ok & (dist <= cap)[None]
    elif kind == KIND_FAR:
        ok = ok & (dist > cap)[None]
    return ok


def _torch_row_terms(o, g_o, q, k, v, lse, doc, scale: float, kind: int, cap: int):
    """(delta_S, delta_F), each (B, Hq, L) in the accumulation dtype: the pre-pass kernel's arithmetic in PyTorch, TORCH_CHUNK queries at a time.
    delta_F = (sum_j P_j dP_j) / (sum_j P_j): the row term of the softmax whose weights are the P_j, the row sum of P in the divisor (1 up to the rounding of
    lse; see row_terms). o, g_o, q (B, Hq, L, d); k, v (B, Hkv, L, d); lse (B, Hq, L); doc (B, L)."""
    acc = acc_dtype(q)
    B, H, L, d = q.shape
    group = H // k.shape[1]
    kk = k.to(acc).repeat_interleave(group, dim=1)
    vv = v.to(acc).repeat_interleave(group, dim=1)
    lse_f = lse.to(acc)
    lse_f = torch.where(torch.isinf(lse_f), torch.zeros((), dtype=acc, device=lse.device), lse_f)
    zero = torch.zeros((), dtype=acc, device=q.device)
    delta_f = torch.empty(B, H, L, dtype=acc, device=q.device)
    for a in range(0, L, TORCH_CHUNK):
        e = min(a + TORCH_CHUNK, L)
        s = torch.matmul(q[:, :, a:e].to(acc), kk.transpose(-1, -2)) * scale
        dp = torch.matmul(g_o[:, :, a:e].to(acc), vv.transpose(-1, -2))
        allow = _allow(doc, a, e, kind, cap)[:, None]
        p = torch.where(allow, torch.exp(s - lse_f[:, :, a:e, None]), zero)
        mass = p.sum(-1)
        delta_f[:, :, a:e] = (p * dp).sum(-1) / torch.where(mass > 0, mass, torch.ones_like(mass))  # (a row with no pair: 0 / 1)
    delta_s = (o.to(acc) * g_o.to(acc)).sum(-1)
    return delta_s, delta_f


def row_terms(o, g_o, q, k, v, lse, doc, scale: float, kind: int, cap: int):
    """(delta_S, delta_F), each (B, Hq, L) in the accumulation dtype: delta_S = rowsum(dO . o) as the flex kernel forms it from the stored
    output, delta_F = rowsum(P . dP) / rowsum(P) of this call's softmax (P = exp(scale q.k - lse) over the pairs of `kind`, dP = dO . v). The Triton
    kernel on CUDA (or under the interpreter / the tests' emulation), else the PyTorch route."""
    B, H, L, d = q.shape
    assert o.shape == g_o.shape == q.shape and k.shape == v.shape and k.shape[2] == L and lse.shape == (B, H, L), \
        (o.shape, g_o.shape, q.shape, k.shape, v.shape, lse.shape)
    assert H % k.shape[1] == 0 and doc.shape == (B, L), (q.shape, k.shape, doc.shape)
    if use_triton(q) and kernel_head_dim_ok(d):
        acc = acc_dtype(q)
        # the kernel's d stride is 1; off the GPU (the interpreter, the tests' emulation) the arguments are contiguous
        q, k, v, o, g_o = (t if t.is_cuda and t.stride(-1) == 1 else t.contiguous() for t in (q, k, v, o, g_o))
        report("kernel", f"{B} x {H} x {L} x {d}, {q.dtype}, kind {kind}, cap {cap}, {q.device}")
        out = torch.empty(2, B, H, L, dtype=acc, device=q.device)
        start = doc_start(doc)
        grid = (-(-L // BLOCK_M), B * H)
        kernels().row_terms_kernel[grid](
            q, k, v, o, g_o, lse.to(acc).contiguous(), doc.to(torch.int64).contiguous(), start, out[0], out[1],
            q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2), g_o.stride(0), g_o.stride(1), g_o.stride(2), L, H, H // k.shape[1], scale, cap,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=d, KIND=kind, PREC=_prec(q), ACC=_tl_acc(q), INTERP=interpreting(), num_warps=NUM_WARPS,
            num_stages=NUM_STAGES)
        return out[1], out[0]
    report("torch", f"{B} x {H} x {L} x {d}, {q.dtype}, kind {kind}, cap {cap}, {q.device}")
    return _torch_row_terms(o, g_o, q, k, v, lse, doc, scale, kind, cap)


# ---------------------------------------------------------------------------------------------------------------------
# the custom ops: the correction, and the identity node that applies it


@torch.library.custom_op("ced::attn_row_corr", mutates_args=())
def attn_row_corr(o: torch.Tensor, g_o: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lse: torch.Tensor,
                  doc: torch.Tensor, scale: float, kind: int, cap: int) -> torch.Tensor:
    """b = delta_S - delta_F, (B, Hq, L) float32 (float64 for float64 inputs): the shift the flex kernel's lse gradient must carry so that
    its backward's row term is rowsum(P . dP). g_o is the gradient of the call's output o (the kernel's own dO)."""
    ds, df = row_terms(o, g_o, q, k, v, lse, doc, scale, kind, cap)
    return ds - df


@attn_row_corr.register_fake
def _(o, g_o, q, k, v, lse, doc, scale, kind, cap):
    return lse.new_empty(lse.shape, dtype=acc_dtype(q))


@torch.library.custom_op("ced::attn_row_tie", mutates_args=())
def attn_row_tie(o: torch.Tensor, lse: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, doc: torch.Tensor,
                 scale: float, kind: int, cap: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The identity on (o, lse), the output and log-sum-exp of one flex call (copies: a custom op does not alias its inputs). Its backward
    (below) is the fix: it adds attn_row_corr to the lse's gradient. q, k, v, doc are what the call read (never differentiated here)."""
    return o.clone(), lse.clone()


@attn_row_tie.register_fake
def _(o, lse, q, k, v, doc, scale, kind, cap):
    return torch.empty_like(o), torch.empty_like(lse)


def _tie_setup(ctx, inputs, output):
    o, lse, q, k, v, doc, scale, kind, cap = inputs
    ctx.save_for_backward(o, lse, q, k, v, doc)
    ctx.meta = (scale, kind, cap)


def _tie_backward(ctx, g_o, g_lse):
    o, lse, q, k, v, doc = ctx.saved_tensors
    scale, kind, cap = ctx.meta
    b = torch.ops.ced.attn_row_corr(o, g_o, q, k, v, lse, doc, scale, kind, cap)
    return g_o, g_lse + b.to(g_lse.dtype), None, None, None, None, None, None, None


attn_row_tie.register_autograd(_tie_backward, setup_context=_tie_setup)


def tie_read(o: torch.Tensor, lse: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, doc: torch.Tensor,
             scale: float, kind: int, cap: int = 0):
    """(o, lse) of a flex call, tied (the module docstring): the same values, a backward whose row term is rowsum(P . dP). o (B, Hq, L, d)
    and lse (B, Hq, L) as the call returned them; q, k, v (B, H, L, d) the tensors it read (the transposed views the call took); doc (B, L)
    the row's document ids; kind KIND_PLAIN / KIND_NEAR / KIND_FAR with `cap` for the last two. The read must be the encoder's own read of a
    row (the module docstring); anything else is refused by name."""
    if kind not in (KIND_PLAIN, KIND_NEAR, KIND_FAR):
        raise ValueError(f"unknown read kind {kind!r}")
    B, H, L, d = q.shape
    if k.shape[2] != L or doc.shape != (B, L):
        raise ValueError("attention.row_term 'pdp' is built for the encoder's own read of a row (one key a position, no cache, no "
                         f"slot read); got q {tuple(q.shape)}, k {tuple(k.shape)}, doc {tuple(doc.shape)}")
    if q.is_cuda and not kernel_head_dim_ok(d):
        raise ValueError(f"attention.row_term 'pdp': head dimension {d} is not one the pre-pass kernel takes (a power of two, 16 to 256)")
    return torch.ops.ced.attn_row_tie(o, lse, q, k, v, doc, float(scale), int(kind), int(cap))
