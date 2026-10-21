"""Deterministic LangGraph workflow used to exercise orchestration safely.

Every planner, executor, and verifier here is a simulation. No repository,
filesystem, shell, coding agent, or model is accessed.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any, Callable, Literal, NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from opentelemetry.trace import Status, StatusCode, Tracer

from agentic_dev_workflow.tracing import TracingSession, new_run_id

MAX_RETRIES = 2


class MockScenario(StrEnum):
    """Explicit deterministic outcomes supported by the mock graph."""

    SUCCESS = "success"
    RETRY_SUCCESS = "retry-success"
    EXHAUSTED_RETRIES = "exhausted-retries"


class WorkflowState(TypedDict):
    """Serializable workflow state; it contains no credentials or environment."""

    run_id: str
    spec_text: str
    spec_hash: str
    scenario: str
    plan: dict[str, Any] | None
    execution_attempt: int
    max_retries: int
    repair_retries_used: int
    execution_status: Literal["not_started", "simulated_success", "simulated_failure"]
    verification_status: Literal["not_run", "simulated_pass", "simulated_fail"]
    verification_summary: str
    retry_reason: str | None
    final_status: Literal["running", "success", "failed"]
    errors: list[str]
    attempt_history: list[dict[str, Any]]
    simulation_mode: bool
    _next_action: NotRequired[Literal["retry", "stop"]]


def create_initial_state(
    spec_text: str,
    *,
    scenario: MockScenario | str = MockScenario.SUCCESS,
    run_id: str | None = None,
    max_retries: int = 1,
) -> WorkflowState:
    """Validate and build a fresh state; retries count repairs after attempt 1."""
    if not isinstance(max_retries, int) or isinstance(max_retries, bool):
        raise ValueError("max_retries must be an integer from 0 to 2")
    if not 0 <= max_retries <= MAX_RETRIES:
        raise ValueError("max_retries must be between 0 and 2")
    try:
        selected = MockScenario(scenario)
    except ValueError as error:
        raise ValueError("unknown mock scenario") from error
    if not isinstance(spec_text, str) or not spec_text.strip():
        raise ValueError("spec_text must not be empty")
    identifier = run_id or new_run_id()
    if not identifier.strip():
        raise ValueError("run_id must not be empty")
    return {
        "run_id": identifier,
        "spec_text": spec_text,
        "spec_hash": hashlib.sha256(spec_text.encode("utf-8")).hexdigest(),
        "scenario": selected.value,
        "plan": None,
        "execution_attempt": 0,
        "max_retries": max_retries,
        "repair_retries_used": 0,
        "execution_status": "not_started",
        "verification_status": "not_run",
        "verification_summary": "",
        "retry_reason": None,
        "final_status": "running",
        "errors": [],
        "attempt_history": [],
        "simulation_mode": True,
    }


def validate_resume_identity(
    state: dict[str, Any], *, run_id: str, spec_text: str
) -> None:
    """Reject resuming a checkpoint with a different run or specification."""
    requested_hash = hashlib.sha256(spec_text.encode("utf-8")).hexdigest()
    if state.get("run_id") != run_id:
        raise ValueError("checkpoint run_id does not match requested run")
    if state.get("spec_hash") != requested_hash:
        raise ValueError("checkpoint specification hash does not match requested specification")


def resume_mock_workflow(
    graph: Any,
    config: dict[str, Any],
    *,
    run_id: str,
    spec_text: str,
) -> WorkflowState:
    """Validate a stored checkpoint's identity before continuing its thread."""
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id != f"agentic-dev-workflow:{run_id}":
        raise ValueError("checkpoint thread_id does not match requested run")
    checkpoint = graph.get_state(config)
    if not checkpoint.values:
        raise ValueError("no checkpoint is available for this workflow thread")
    validate_resume_identity(checkpoint.values, run_id=run_id, spec_text=spec_text)
    return graph.invoke(None, config)


def build_mock_graph(
    *,
    checkpointer: Any | None = None,
    tracing_session: TracingSession | None = None,
    interrupt_before: list[str] | None = None,
) -> Any:
    """Build the deterministic StateGraph with optional local checkpointing."""
    tracer = tracing_session.tracer if tracing_session else None

    def instrument(
        node_name: str,
        operation: Callable[[WorkflowState], dict[str, Any]],
    ) -> Callable[[WorkflowState], dict[str, Any]]:
        def call(state: WorkflowState) -> dict[str, Any]:
            if tracer is None:
                return operation(state)
            with tracer.start_as_current_span(f"workflow.{node_name}") as span:
                span.set_attribute("workflow.run_id", state["run_id"])
                span.set_attribute("workflow.phase", "mock-orchestration")
                span.set_attribute("workflow.node", node_name)
                attempt = state["execution_attempt"] + (1 if node_name == "mock_executor" else 0)
                span.set_attribute("workflow.attempt", attempt)
                span.set_attribute("workflow.mock_mode", True)
                try:
                    result = operation(state)
                    outcome = result.get(
                        "final_status",
                        result.get(
                            "verification_status",
                            result.get("_next_action", result.get("execution_status", "completed")),
                        ),
                    )
                    span.set_attribute("workflow.outcome", str(outcome))
                    if outcome in {"failed", "simulated_fail"}:
                        span.set_status(Status(StatusCode.ERROR, "simulated workflow failure"))
                    return result
                except Exception as error:
                    span.set_attribute("workflow.outcome", "error")
                    span.set_status(Status(StatusCode.ERROR, type(error).__name__))
                    raise

        return call

    def planner(state: WorkflowState) -> dict[str, Any]:
        return {
            "plan": {
                "requested_change": state["spec_text"].strip(),
                "steps": [
                    "identify the feature behavior",
                    "implement the smallest change",
                    "verify the behavior",
                ],
            }
        }

    def executor(state: WorkflowState) -> dict[str, Any]:
        attempt = state["execution_attempt"] + 1
        failed = state["scenario"] == MockScenario.EXHAUSTED_RETRIES.value or (
            state["scenario"] == MockScenario.RETRY_SUCCESS.value and attempt == 1
        )
        return {
            "execution_attempt": attempt,
            "execution_status": "simulated_failure" if failed else "simulated_success",
        }

    def verifier(state: WorkflowState) -> dict[str, Any]:
        passed = state["execution_status"] == "simulated_success"
        summary = (
            "SIMULATED verification passed; no real tests or checks were run."
            if passed
            else "SIMULATED verification failed; no real tests or checks were run."
        )
        history_entry = {
            "attempt": state["execution_attempt"],
            "verification": "passed" if passed else "failed",
        }
        return {
            "verification_status": "simulated_pass" if passed else "simulated_fail",
            "verification_summary": summary,
            "attempt_history": [*state["attempt_history"], history_entry],
            "errors": state["errors"]
            if passed
            else [
                *state["errors"],
                f"Simulated verification failed on attempt {state['execution_attempt']}",
            ],
        }

    def retry_decision(state: WorkflowState) -> dict[str, Any]:
        if state["verification_status"] == "simulated_pass":
            return {"final_status": "success", "_next_action": "stop"}
        if state["repair_retries_used"] < state["max_retries"]:
            return {
                "repair_retries_used": state["repair_retries_used"] + 1,
                "retry_reason": "Simulated verification failed; retrying mock execution.",
                "_next_action": "retry",
            }
        return {"final_status": "failed", "_next_action": "stop", "retry_reason": None}

    builder = StateGraph(WorkflowState)
    builder.add_node("planner", instrument("planner", planner))
    builder.add_node("mock_executor", instrument("mock_executor", executor))
    builder.add_node("verifier", instrument("verifier", verifier))
    builder.add_node("retry_decision", instrument("retry_decision", retry_decision))
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "mock_executor")
    builder.add_edge("mock_executor", "verifier")
    builder.add_edge("verifier", "retry_decision")
    builder.add_conditional_edges(
        "retry_decision",
        lambda state: state["_next_action"],
        {"retry": "mock_executor", "stop": END},
    )
    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def run_mock_workflow(
    spec_text: str,
    *,
    scenario: MockScenario | str = MockScenario.SUCCESS,
    run_id: str | None = None,
    max_retries: int = 1,
    tracing_session: TracingSession | None = None,
    checkpointer: Any | None = None,
) -> WorkflowState:
    """Run a fresh mock graph with a unique explicit LangGraph thread ID."""
    state = create_initial_state(spec_text, scenario=scenario, run_id=run_id, max_retries=max_retries)
    graph = build_mock_graph(
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
        tracing_session=tracing_session,
    )
    config = {"configurable": {"thread_id": f"agentic-dev-workflow:{state['run_id']}"}}
    tracer: Tracer | None = tracing_session.tracer if tracing_session else None
    if tracer is None:
        return graph.invoke(state, config)
    with tracer.start_as_current_span("workflow.mock-run") as span:
        span.set_attribute("workflow.run_id", state["run_id"])
        span.set_attribute("workflow.phase", "mock-orchestration")
        span.set_attribute("workflow.node", "graph")
        span.set_attribute("workflow.mock_mode", True)
        try:
            result = graph.invoke(state, config)
            span.set_attribute("workflow.outcome", result["final_status"])
            if result["final_status"] == "failed":
                span.set_status(Status(StatusCode.ERROR, "simulated workflow failure"))
            return result
        except Exception as error:
            span.set_attribute("workflow.outcome", "error")
            span.set_status(Status(StatusCode.ERROR, type(error).__name__))
            raise
