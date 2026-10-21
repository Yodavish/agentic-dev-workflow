"""Preflight-only workflow entry points for this implementation phase."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

from opentelemetry.trace import Status, StatusCode

from agentic_dev_workflow.config import Configuration
from agentic_dev_workflow.git_safety import RepositoryInspection, inspect_repository
from agentic_dev_workflow.tracing import TracingSession, create_tracing_session, new_run_id


@dataclass(frozen=True)
class WorkflowResult:
    """Result data from one dry-run or preflight operation."""

    run_id: str
    operation: str
    outcome: str
    repository: RepositoryInspection
    trace_exported: bool
    telemetry_error: str | None = None
    dry_run_only: bool = False


def run_preflight(
    config: Configuration,
    *,
    operation: str = "preflight",
    dry_run_only: bool = False,
    tracing_session: TracingSession | None = None,
) -> WorkflowResult:
    """Inspect the configured checkout and emit a workflow trace."""
    run_id = new_run_id()
    session = tracing_session
    telemetry_error: str | None = None
    if session is None:
        try:
            session = create_tracing_session(config.otlp_endpoint)
        except Exception as error:
            telemetry_error = f"tracing setup failed ({type(error).__name__})"

    manager = (
        session.tracer.start_as_current_span(f"workflow.{operation}")
        if session is not None else nullcontext(None)
    )
    with manager as span:
        if span is not None:
            span.set_attribute("workflow.run_id", run_id)
            span.set_attribute("workflow.operation", operation)
            span.set_attribute("workflow.phase", "preflight")
        repository = inspect_repository(config.repository, expected_path=config.repository)
        outcome = "success" if not repository.issues else "blocked"
        if span is not None:
            span.set_attribute("workflow.outcome", outcome)
            span.set_attribute("repository.clean", repository.is_clean)
            if outcome != "success":
                span.set_status(Status(StatusCode.ERROR, "repository preflight blocked"))

    exported = False
    if session is not None:
        try:
            exported = session.flush()
            if not exported:
                telemetry_error = "trace export failed or could not be confirmed"
        except Exception as error:
            telemetry_error = f"trace export failed ({type(error).__name__})"
        finally:
            try:
                session.shutdown()
            except Exception as error:
                telemetry_error = f"trace shutdown failed ({type(error).__name__})"

    return WorkflowResult(
        run_id=run_id, operation=operation, outcome=outcome, repository=repository,
        trace_exported=exported, telemetry_error=telemetry_error,
        dry_run_only=dry_run_only,
    )
