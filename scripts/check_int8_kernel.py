"""Validate the Triton W8A16 kernel against PyTorch and save the errors to results/.

On an NVIDIA GPU this runs the compiled kernel; elsewhere set TRITON_INTERPRET=1 (CPU interpreter).
Correctness only: interpreter timings are meaningless, so no speed is reported here
(see scripts/gpu/bench_int8_kernel.py for that).
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from llmserve.kernels.int8_matmul import int8_matmul, quantize_per_channel, reference_int8_matmul  # noqa: E402

SHAPES = [(1, 512, 512), (8, 512, 1024), (17, 384, 640), (64, 1024, 256), (128, 256, 512)]


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    interp = os.environ.get("TRITON_INTERPRET") == "1"
    rows = []
    for m, k, n in SHAPES:
        g = torch.Generator().manual_seed(m + k + n)
        x = torch.randn(m, k, generator=g).to(dev)
        w = torch.randn(k, n, generator=g).to(dev)
        w_q, scale = quantize_per_channel(w)
        out = int8_matmul(x, w_q, scale)
        ref = reference_int8_matmul(x, w_q, scale)
        full = x @ w
        row = {"m": m, "k": k, "n": n, "dtype": "float32",
               "max_abs_err_vs_reference": float((out - ref).abs().max()),
               "max_rel_err_vs_reference": float(((out - ref).abs() / ref.abs().clamp(min=1e-3)).max()),
               "rel_fro_err_vs_fp32_unquantized": float((out - full).norm() / full.norm()),
               "allclose_1e-4": bool(torch.allclose(out, ref, rtol=1e-4, atol=1e-4))}
        rows.append(row)
        print(row, flush=True)
    commit = os.environ.get("GIT_COMMIT", "unknown")
    hw = (f"{torch.cuda.get_device_name()}" if dev == "cuda" else
          f"CPU via TRITON_INTERPRET=1, {platform.system()} {platform.machine()}"
          + (f" ({os.environ['HOST_DESC']})" if os.environ.get("HOST_DESC") else ""))
    out = {"label": "phase7-int8-kernel-correctness", "commit": commit, "hardware": hw,
           "interpreter": interp, "torch": torch.__version__, "triton": triton.__version__,
           "all_pass": all(r["allclose_1e-4"] for r in rows), "shapes": rows,
           "note": "correctness only; no timing measured" if dev == "cpu" else ""}
    path = Path("results") / f"phase7-int8-correctness-{'interpret' if interp else dev}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"saved {path}  all_pass={out['all_pass']}")
    sys.exit(0 if out["all_pass"] else 1)


if __name__ == "__main__":
    main()
