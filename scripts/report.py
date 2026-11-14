"""Compare saved runs against a baseline run and print a Markdown table.

    uv run python scripts/report.py results/phase2-8prefix-*.json --baseline round_robin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = [  # (label, path in summary, higher_is_better)
    ("Throughput (req/s)", ("throughput_req_s",), True),
    ("Output tokens/s", ("output_tokens_per_s",), True),
    ("Latency p50 (ms)", ("latency_ms", "p50"), False),
    ("Latency p95 (ms)", ("latency_ms", "p95"), False),
    ("Latency p99 (ms)", ("latency_ms", "p99"), False),
    ("TTFT p50 (ms)", ("ttft_ms", "p50"), False),
    ("TTFT p95 (ms)", ("ttft_ms", "p95"), False),
    ("TTFT p99 (ms)", ("ttft_ms", "p99"), False),
    ("Prefix-cache hit rate", ("prefix_cache_hit_rate",), True),
]


def get(d: dict, path: tuple) -> float | None:
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def delta(base: float | None, new: float | None, higher_better: bool) -> str:
    if base in (None, 0) or new is None:
        return "n/a"
    change = (new - base) / base * 100
    better = change > 0 if higher_better else change < 0
    return f"{change:+.1f}% ({'better' if better else 'worse'})"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--baseline", default="round_robin")
    args = p.parse_args()
    runs = [json.loads(Path(f).read_text()) | {"_file": Path(f).name} for f in args.files]
    base = next(r for r in runs if r["policy"] == args.baseline)
    others = [r for r in runs if r is not base]
    for r in [base, *others]:
        s = r["summary"]
        print(f"- `{r['policy']}`: {r['_file']} (commit {r['commit']}, {s['requests_ok']}/{s['requests_sent']} ok)")
    print()
    header = ["Metric", f"{args.baseline}"] + [f"{r['policy']}" for r in others] + [f"Δ {r['policy']}" for r in others]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for label, path, hb in METRICS:
        b = get(base["summary"], path)
        vals = [get(r["summary"], path) for r in others]
        row = [label, str(b)] + [str(v) for v in vals] + [delta(b, v, hb) for v in vals]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
