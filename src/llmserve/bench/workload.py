"""Synthetic chat workload with shared system-prompt prefixes.

Real traffic often reuses a few long system prompts (one per app/tenant) with short,
varied user messages. That is what makes prefix-aware routing worth doing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

WORDS = (
    "the model request token cache batch latency worker router queue memory prompt user system "
    "assistant answer question data serve stream decode prefill kernel scale cluster node policy"
).split()


@dataclass
class WorkloadConfig:
    num_prefixes: int = 8          # distinct system prompts
    prefix_tokens_mean: int = 600  # system prompt length
    zipf_a: float = 1.2            # popularity skew across system prompts
    user_tokens_median: int = 120  # lognormal user message length
    user_tokens_sigma: float = 0.6
    output_tokens_median: int = 96
    output_tokens_sigma: float = 0.5
    max_output_tokens: int = 512
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChatRequest:
    prefix_id: int
    messages: list[dict]
    max_tokens: int


def _text(rng: np.random.Generator, n_tokens: int) -> str:
    # ~4 characters per token keeps the mock worker's token count close to real tokenizers
    n_words = max(1, int(n_tokens * 4 / 6))
    return " ".join(rng.choice(WORDS, size=n_words))


class Workload:
    def __init__(self, cfg: WorkloadConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        prefix_rng = np.random.default_rng(cfg.seed + 1)
        self.system_prompts = [
            f"[app-{i}] You are assistant #{i}. " + _text(prefix_rng, int(prefix_rng.normal(cfg.prefix_tokens_mean, cfg.prefix_tokens_mean * 0.2)))
            for i in range(cfg.num_prefixes)
        ]
        ranks = np.arange(1, cfg.num_prefixes + 1)
        weights = 1.0 / ranks**cfg.zipf_a
        self.prefix_probs = weights / weights.sum()

    def sample(self) -> ChatRequest:
        c, r = self.cfg, self.rng
        pid = int(r.choice(c.num_prefixes, p=self.prefix_probs))
        user_tokens = int(r.lognormal(np.log(c.user_tokens_median), c.user_tokens_sigma))
        out_tokens = int(min(c.max_output_tokens, max(1, r.lognormal(np.log(c.output_tokens_median), c.output_tokens_sigma))))
        return ChatRequest(
            prefix_id=pid,
            messages=[
                {"role": "system", "content": self.system_prompts[pid]},
                {"role": "user", "content": _text(r, user_tokens)},
            ],
            max_tokens=out_tokens,
        )
