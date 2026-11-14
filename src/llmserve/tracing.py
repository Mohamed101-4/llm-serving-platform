"""OpenTelemetry setup shared by the router and the mock worker.

Spans are exported over OTLP/HTTP when OTEL_EXPORTER_OTLP_ENDPOINT is set (e.g. to Jaeger or
an OTel collector), and dropped otherwise. The router propagates W3C `traceparent` to workers,
so a request's router spans and worker spans land in one trace:

    llmserve.request                (router, SERVER)
      route                         policy + admission decision
      upstream                      (router, CLIENT) connect -> last byte
        wait_first_token            queueing + prefill, as the router sees it
        stream                      decode, as the router sees it
        worker.request              (worker, SERVER)  mock worker; vLLM emits its own
          worker.queue              waiting for a batch slot / KV space
          worker.prefill
          worker.decode
"""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter

_configured = False


def setup_tracing(service_name: str, exporter: SpanExporter | None = None) -> None:
    """Install a tracer provider once per process. Tests pass an in-memory exporter."""
    global _configured
    if _configured:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter is not None:  # tests: export synchronously so spans are visible immediately
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    elif os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True


def tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)
