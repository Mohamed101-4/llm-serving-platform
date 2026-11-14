"""Time the Triton W8A16 kernel vs PyTorch on an NVIDIA GPU (NOT yet run; needs CUDA).

    pip install torch triton && python scripts/gpu/bench_int8_kernel.py

Compares, for decode-like (small M) and prefill-like (large M) shapes of a 1.5B-class model:
  fp16 torch.matmul with full-precision weights, dequantize-then-matmul in PyTorch, and the
fused Triton kernel. Saves results/phase7-int8-bench-<gpu>-<timestamp>.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import triton.testing

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from llmserve.kernels.int8_matmul import int8_matmul, quantize_per_channel  # noqa: E402

# (M tokens, K in, N out): Qwen2.5-1.5B hidden 1536, MLP 8960
SHAPES = [(1, 1536, 8960), (16, 1536, 8960), (1, 8960, 1536), (512, 1536, 8960), (2048, 1536, 1536)]


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit("needs an NVIDIA GPU; use scripts/check_int8_kernel.sh for CPU correctness checks")
    rows = []
    for m, k, n in SHAPES:
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        w = torch.randn(k, n, device="cuda", dtype=torch.float16)
        w_q, scale = quantize_per_channel(w.float())
        torch.testing.assert_close(int8_matmul(x, w_q, scale).float(),
                                   (x.float() @ (w_q.float() * scale)), rtol=2e-2, atol=2e-1)
        t_fp16 = triton.testing.do_bench(lambda: x @ w)
        t_deq = triton.testing.do_bench(lambda: x @ (w_q.half() * scale.half()))
        t_tri = triton.testing.do_bench(lambda: int8_matmul(x, w_q, scale))
        rows.append({"m": m, "k": k, "n": n, "fp16_matmul_ms": t_fp16, "torch_dequant_matmul_ms": t_deq,
                     "triton_w8a16_ms": t_tri, "speedup_vs_fp16": t_fp16 / t_tri,
                     "speedup_vs_torch_dequant": t_deq / t_tri, "weight_bytes_fp16": k * n * 2, "weight_bytes_int8": k * n})
        print(rows[-1], flush=True)
    gpu = torch.cuda.get_device_name()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    path = Path("results") / f"phase7-int8-bench-{gpu.replace(' ', '_')}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"hardware": gpu, "commit": commit, "torch": torch.__version__, "rows": rows}, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
