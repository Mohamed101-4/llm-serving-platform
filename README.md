# llmserve

A distributed LLM serving platform: an OpenAI-compatible router in front of
multiple vLLM workers, with smart routing, overload protection and a
benchmark harness that compares each improvement against a naive
round-robin baseline.

Work in progress. See [docs/PLAN.md](docs/PLAN.md) for the phase plan and
which parts need a real NVIDIA GPU.
