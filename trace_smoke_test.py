from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

resource = Resource.create({
    "service.name": "agentic-dev-workflow",
    "service.version": "0.1.0",
})

provider = TracerProvider(resource=resource)
provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint="http://localhost:4318/v1/traces"
        )
    )
)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("agentic-dev-workflow.smoke-test")

with tracer.start_as_current_span("workflow.test") as parent:
    parent.set_attribute("workflow.run_id", "smoke-test-001")
    parent.set_attribute("workflow.outcome", "success")

    with tracer.start_as_current_span("workflow.child") as child:
        child.set_attribute("agent.role", "test")
        child.set_attribute("test.result", "passed")

provider.force_flush()
provider.shutdown()

print("Trace exported. Search for agentic-dev-workflow in Jaeger.")
