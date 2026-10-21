"""Short-lived OTLP tracing helpers for CLI workflows."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Tracer, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult, SpanExporter


class _ResultTrackingExporter(SpanExporter):
    """Record exporter results so CLI commands never claim false success."""

    def __init__(self, delegate: SpanExporter) -> None:
        self._delegate = delegate
        self.last_export_succeeded: bool | None = None

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            result = self._delegate.export(spans)
        except Exception:
            self.last_export_succeeded = False
            return SpanExportResult.FAILURE
        self.last_export_succeeded = result is SpanExportResult.SUCCESS
        return result

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)


@dataclass
class TracingSession:
    """Own a tracer provider and expose explicit flush and shutdown."""

    provider: TracerProvider
    tracer: Tracer
    _exporter: _ResultTrackingExporter

    def flush(self) -> bool:
        """Flush finished spans and return whether export succeeded."""
        return (
            self.provider.force_flush(timeout_millis=7000)
            and self._exporter.last_export_succeeded is not False
        )

    def shutdown(self) -> None:
        """Shut down the provider and its exporter."""
        self.provider.shutdown()


def new_run_id() -> str:
    """Return a non-sensitive identifier for one workflow invocation."""
    return uuid.uuid4().hex


def create_tracing_session(
    endpoint: str,
    *,
    exporter: SpanExporter | None = None,
) -> TracingSession:
    """Create a service-scoped tracer with OTLP HTTP export."""
    delegate = exporter if exporter is not None else OTLPSpanExporter(endpoint=endpoint, timeout=5)
    tracking_exporter = _ResultTrackingExporter(delegate)
    provider = TracerProvider(
        resource=Resource(
            attributes={"service.name": "agentic-dev-workflow", "service.version": "0.1.0"}
        )
    )
    provider.add_span_processor(SimpleSpanProcessor(tracking_exporter))
    return TracingSession(provider, provider.get_tracer("agentic_dev_workflow"), tracking_exporter)
