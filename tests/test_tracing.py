import unittest

from opentelemetry.sdk.trace.export import SpanExportResult, SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentic_dev_workflow.tracing import create_tracing_session, new_run_id


class TracingTests(unittest.TestCase):
    def test_run_ids_are_unique_and_opaque(self) -> None:
        first = new_run_id()
        second = new_run_id()
        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 32)

    def test_session_exports_parent_and_child_without_jaeger(self) -> None:
        exporter = InMemorySpanExporter()
        session = create_tracing_session(
            "http://localhost:4318/v1/traces", exporter=exporter
        )
        with session.tracer.start_as_current_span("workflow.test") as parent:
            parent.set_attribute("workflow.run_id", "test-run")
            with session.tracer.start_as_current_span("workflow.child"):
                pass
        self.assertTrue(session.flush())
        spans = exporter.get_finished_spans()
        self.assertEqual({span.name for span in spans}, {"workflow.test", "workflow.child"})
        self.assertEqual(
            {span.context.trace_id for span in spans},
            {spans[0].context.trace_id},
        )
        self.assertEqual(
            spans[0].resource.attributes,
            {"service.name": "agentic-dev-workflow", "service.version": "0.1.0"},
        )
        self.assertTrue(all(span.parent is not None or span.name == "workflow.test" for span in spans))
        session.shutdown()

    def test_export_failure_is_reported(self) -> None:
        class FailingExporter(SpanExporter):
            def export(self, spans):
                return SpanExportResult.FAILURE

            def shutdown(self) -> None:
                pass

        session = create_tracing_session(
            "http://localhost:4318/v1/traces", exporter=FailingExporter()
        )
        with session.tracer.start_as_current_span("workflow.failure-check"):
            pass
        self.assertFalse(session.flush())
        session.shutdown()

    def test_exporter_exception_does_not_escape_span_end(self) -> None:
        class RaisingExporter(SpanExporter):
            def export(self, spans):
                raise OSError("unavailable")

            def shutdown(self) -> None:
                pass

        session = create_tracing_session(
            "http://localhost:4318/v1/traces", exporter=RaisingExporter()
        )
        with session.tracer.start_as_current_span("workflow.failure-check"):
            pass
        self.assertFalse(session.flush())
        session.shutdown()


if __name__ == "__main__":
    unittest.main()
