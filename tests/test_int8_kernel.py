"""Phase 7: Triton W8A16 kernel vs the PyTorch reference.

Skipped unless torch and triton are installed. Without an NVIDIA GPU, run under the Triton
CPU interpreter: ./scripts/check_int8_kernel.sh (Docker) or TRITON_INTERPRET=1 pytest ...
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from llmserve.kernels.int8_matmul import int8_matmul, quantize_per_channel, reference_int8_matmul  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("m,k,n", [(1, 64, 64), (7, 96, 130), (33, 256, 192), (64, 128, 64)])
def test_matches_pytorch_reference(m, k, n):
    g = torch.Generator().manual_seed(m * 1000 + k + n)
    x = torch.randn(m, k, generator=g).to(DEVICE)
    w = torch.randn(k, n, generator=g).to(DEVICE)
    w_q, scale = quantize_per_channel(w)
    out = int8_matmul(x, w_q, scale)
    ref = reference_int8_matmul(x, w_q, scale)
    assert out.shape == (m, n)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_quantization_error_is_small():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(16, 256, generator=g).to(DEVICE)
    w = torch.randn(256, 128, generator=g).to(DEVICE)
    w_q, scale = quantize_per_channel(w)
    out = int8_matmul(x, w_q, scale)
    full = x @ w
    rel = (out - full).norm() / full.norm()
    assert rel < 0.01  # per-channel INT8 on Gaussian weights: ~0.5% relative error


def test_quantize_roundtrip_bounds():
    w = torch.tensor([[1.0, -0.5], [-2.0, 0.25], [0.5, 0.0]])
    w_q, scale = quantize_per_channel(w)
    assert w_q.dtype == torch.int8
    assert w_q.abs().max() == 127
    assert torch.all((w_q.float() * scale - w).abs() <= scale / 2 + 1e-7)
