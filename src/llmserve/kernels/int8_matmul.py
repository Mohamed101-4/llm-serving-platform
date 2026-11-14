"""W8A16 matmul in Triton: int8 weights, fp16/fp32 activations, per-output-channel scales.

Weight-only INT8 quantization halves weight memory vs fp16, which matters because LLM decode is
memory-bandwidth bound: each step reads every weight once for a handful of tokens. The kernel
loads int8 weight tiles, converts them to the activation dtype in registers, accumulates in fp32
and applies the per-channel scale once per output tile (scale[n] factors out of the K-sum), so the
full-precision weight matrix is never materialised in GPU memory.

    w_q, scale = quantize_per_channel(w)      # w: [K, N] float -> int8 [K, N], scale [N]
    y = int8_matmul(x, w_q, scale)            # x: [M, K] -> y: [M, N] ~= x @ w

`reference_int8_matmul` is the PyTorch reference the kernel is validated against.
On machines without an NVIDIA GPU, run with TRITON_INTERPRET=1 (CPU interpreter, correctness only).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def quantize_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel INT8: w[:, n] ~= w_q[:, n] * scale[n]."""
    amax = w.abs().amax(dim=0).clamp(min=1e-8)
    scale = (amax / 127.0).to(torch.float32)
    w_q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return w_q, scale


def reference_int8_matmul(x: torch.Tensor, w_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: dequantize the whole matrix, then matmul in fp32."""
    return (x.float() @ (w_q.float() * scale[None, :])).to(x.dtype)


@triton.jit
def _w8a16_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K,
                  stride_xm, stride_xk, stride_wk, stride_wn, stride_ym, stride_yn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        x = tl.load(x_ptr + rm[:, None] * stride_xm + kk[None, :] * stride_xk,
                    mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        w = tl.load(w_ptr + kk[:, None] * stride_wk + rn[None, :] * stride_wn,
                    mask=(kk[:, None] < K) & (rn[None, :] < N), other=0)
        acc += tl.dot(x, w.to(x.dtype))  # int8 -> activation dtype in registers
    s = tl.load(s_ptr + rn, mask=rn < N, other=0.0)
    acc = acc * s[None, :]  # per-channel scale applied once, after the K reduction
    tl.store(y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn,
             acc.to(y_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


def int8_matmul(x: torch.Tensor, w_q: torch.Tensor, scale: torch.Tensor,
                block_m: int = 32, block_n: int = 64, block_k: int = 32) -> torch.Tensor:
    assert x.ndim == 2 and w_q.ndim == 2 and x.shape[1] == w_q.shape[0], (x.shape, w_q.shape)
    assert w_q.dtype == torch.int8 and scale.shape == (w_q.shape[1],)
    M, K = x.shape
    N = w_q.shape[1]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _w8a16_kernel[grid](x, w_q, scale, y, M, N, K,
                        x.stride(0), x.stride(1), w_q.stride(0), w_q.stride(1), y.stride(0), y.stride(1),
                        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
    return y
