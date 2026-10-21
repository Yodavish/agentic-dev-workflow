"""Deterministic LangGraph workflow used to exercise orchestration safely.

Every planner, executor, and verifier here is a simulation. No repository,
filesystem, shell, coding agent, or model is accessed.
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from typing import Any, Callable, Literal, NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from opentelemetry.trace import Status, StatusCode, Tracer

from agentic_dev_workflow.tracing import TracingSession, new_run_id

MAX_RETRIES = 2
CHECKPOINT_FORMAT_VERSION = 1
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class MockScenario(StrEnum):
    """Explicit deterministic outcomes supported by the mock graph."""

    SUCCESS = "success"
    RETRY_SUCCESS = "retry-success"
    EXHAUSTED_RETRIES = "exhausted-retries"


class WorkflowState(TypedDict):
    """Serializable workflow state; it contains no credentials or environment."""

    run_id: str
    checkpoint_format_version: int
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
    identifier = new_run_id() if run_id is None else run_id
    _validate_run_id(identifier)
    return {
        "run_id": identifier,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
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
    state: dict[str, Any],
    *,
    run_id: str,
    spec_text: str,
    scenario: MockScenario | str = MockScenario.SUCCESS,
    max_retries: int = 1,
) -> None:
    """Reject a checkpoint whose immutable run inputs do not match."""
    if not isinstance(state, dict):
        raise ValueError("checkpoint workflow state is invalid")
    _validate_run_id(run_id)
    try:
        selected_scenario = MockScenario(scenario)
    except ValueError as error:
        raise ValueError("unknown mock scenario") from error
    if not isinstance(max_retries, int) or isinstance(max_retries, bool):
        raise ValueError("max_retries must be an integer from 0 to 2")
    if not 0 <= max_retries <= MAX_RETRIES:
        raise ValueError("max_retries must be between 0 and 2")
    if not isinstance(spec_text, str) or not spec_text.strip():
        raise ValueError("spec_text must not be empty")
    requested_hash = hashlib.sha256(spec_text.encode("utf-8")).hexdigest()
    if state.get("run_id") != run_id:
        raise ValueError("checkpoint run_id does not match requested run")
    if state.get("spec_hash") != requested_hash:
        raise ValueError("checkpoint specification hash does not match requested specification")
    if state.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("checkpoint format version is incompatible")
    if state.get("scenario") != selected_scenario.value:
        raise ValueError("checkpoint scenario does not match requested scenario")
    if state.get("max_retries") != max_retries:
        raise ValueError("checkpoint retry limit does not match requested retry limit")
    stored_spec = state.get("spec_text")
    if (
        not isinstance(stored_spec, str)
        or not stored_spec.strip()
        or hashlib.sha256(stored_spec.encode("utf-8")).hexdigest() != state.get("spec_hash")
    ):
        raise ValueError("checkpoint specification state is invalid")


def _validate_resume_state(state: dict[str, Any], next_nodes: tuple[str, ...]) -> None:
    """Reject malformed state or continuations before any graph node runs."""
    if not isinstance(state, dict):
        raise ValueError("checkpoint workflow state is invalid")
    required = {
        "run_id",
        "checkpoint_format_version",
        "spec_text",
        "spec_hash",
        "scenario",
        "plan",
        "execution_attempt",
        "max_retries",
        "repair_retries_used",
        "execution_status",
        "verification_status",
        "verification_summary",
        "retry_reason",
        "final_status",
        "errors",
        "attempt_history",
        "simulation_mode",
    }
    if not required.issubset(state):
        raise ValueError("checkpoint workflow state is incomplete")
    if not isinstance(state["run_id"], str):
        raise ValueError("checkpoint workflow state is invalid")
    if (
        not isinstance(state["checkpoint_format_version"], int)
        or isinstance(state["checkpoint_format_version"], bool)
        or state["checkpoint_format_version"] != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError("checkpoint format version is incompatible")
    if not isinstance(state["spec_text"], str) or not isinstance(state["spec_hash"], str):
        raise ValueError("checkpoint workflow state is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", state["spec_hash"]):
        raise ValueError("checkpoint workflow state is invalid")
    if not isinstance(state["scenario"], str):
        raise ValueError("checkpoint workflow state is invalid")
    if not isinstance(state["plan"], dict):
        raise ValueError("checkpoint planner output is missing or invalid")
    plan = state["plan"]
    if (
        not isinstance(plan.get("requested_change"), str)
        or plan["requested_change"] != state["spec_text"].strip()
        or not isinstance(plan.get("steps"), list)
        or not plan["steps"]
        or any(not isinstance(step, str) or not step.strip() for step in plan["steps"])
    ):
        raise ValueError("checkpoint planner output is missing or invalid")
    if (
        not isinstance(state["execution_attempt"], int)
        or isinstance(state["execution_attempt"], bool)
        or not isinstance(state["max_retries"], int)
        or isinstance(state["max_retries"], bool)
        or not isinstance(state["repair_retries_used"], int)
        or isinstance(state["repair_retries_used"], bool)
    ):
        raise ValueError("checkpoint retry state is invalid")
    attempt = state["execution_attempt"]
    retries = state["repair_retries_used"]
    max_retries = state["max_retries"]
    if (
        not 0 <= max_retries <= MAX_RETRIES
        or not 0 <= retries <= max_retries
        or not 0 <= attempt <= max_retries + 1
    ):
        raise ValueError("checkpoint retry state is outside configured limits")
    if (
        not isinstance(state["execution_status"], str)
        or state["execution_status"]
        not in ("not_started", "simulated_success", "simulated_failure")
        or not isinstance(state["verification_status"], str)
        or state["verification_status"] not in ("not_run", "simulated_pass", "simulated_fail")
    ):
        raise ValueError("checkpoint workflow status is invalid")
    if not isinstance(state["verification_summary"], str):
        raise ValueError("checkpoint workflow state is invalid")
    retry_reason = state["retry_reason"]
    if retry_reason is not None and not isinstance(retry_reason, str):
        raise ValueError("checkpoint workflow state is invalid")
    if state["final_status"] != "running" or state["simulation_mode"] is not True:
        raise ValueError("checkpoint continuation/status combination is invalid")
    errors = state["errors"]
    history = state["attempt_history"]
    if not isinstance(errors, list) or any(not isinstance(error, str) for error in errors):
        raise ValueError("checkpoint error history is invalid")
    if not isinstance(history, list):
        raise ValueError("checkpoint attempt history is invalid")
    for index, entry in enumerate(history, start=1):
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("attempt"), int)
            or isinstance(entry.get("attempt"), bool)
            or entry.get("attempt") != index
            or not isinstance(entry.get("verification"), str)
            or entry.get("verification") not in ("passed", "failed")
        ):
            raise ValueError("checkpoint attempt history is invalid")

    if len(next_nodes) != 1 or next_nodes[0] not in {
        "mock_executor",
        "verifier",
        "retry_decision",
    }:
        raise ValueError("checkpoint continuation state is invalid")
    next_node = next_nodes[0]
    expected_history_length = attempt - 1 if next_node == "verifier" else attempt
    if len(history) != expected_history_length:
        raise ValueError("checkpoint attempt count does not match attempt history")

    try:
        scenario = MockScenario(state["scenario"])
    except ValueError:
        raise ValueError("checkpoint scenario is invalid") from None

    def expected_result(number: int) -> str:
        failed = scenario == MockScenario.EXHAUSTED_RETRIES or (
            scenario == MockScenario.RETRY_SUCCESS and number == 1
        )
        return "failed" if failed else "passed"

    if any(entry["verification"] != expected_result(index) for index, entry in enumerate(history, 1)):
        raise ValueError("checkpoint attempt history conflicts with the mock scenario")
    failures = sum(entry["verification"] == "failed" for entry in history)
    if len(errors) != failures:
        raise ValueError("checkpoint error history does not match attempt history")
    expected_retries = failures
    if next_node in {"verifier", "retry_decision"}:
        if next_node == "retry_decision":
            expected_retries = sum(
                entry["verification"] == "failed" for entry in history[:-1]
            )
    if retries != expected_retries:
        raise ValueError("checkpoint retry count does not match attempt history")

    if next_node == "mock_executor":
        if attempt == 0:
            if (
                state["execution_status"] != "not_started"
                or state["verification_status"] != "not_run"
                or history
                or errors
                or retries != 0
                or retry_reason is not None
                or state.get("_next_action") is not None
            ):
                raise ValueError("checkpoint planner continuation state is invalid")
        elif (
            attempt != retries
            or history[-1]["verification"] != "failed"
            or state["execution_status"] != "simulated_failure"
            or state["verification_status"] != "simulated_fail"
            or not retry_reason
            or state.get("_next_action") != "retry"
        ):
            raise ValueError("checkpoint retry continuation state is invalid")
    else:
        if attempt < 1:
            raise ValueError("checkpoint continuation has no execution attempt")
        current_result = expected_result(attempt)
        current_execution = "simulated_failure" if current_result == "failed" else "simulated_success"
        if state["execution_status"] != current_execution:
            raise ValueError("checkpoint execution status conflicts with the mock scenario")
        if next_node == "verifier":
            if state["verification_status"] != "not_run":
                raise ValueError("checkpoint verifier continuation state is invalid")
        elif (
            state["verification_status"]
            != ("simulated_fail" if current_result == "failed" else "simulated_pass")
        ):
            raise ValueError("checkpoint verification status conflicts with attempt history")
        if (attempt == 1 and retry_reason is not None) or (attempt > 1 and not retry_reason):
            raise ValueError("checkpoint retry reason is inconsistent with attempt history")
        action = state.get("_next_action")
        if action not in ((None,) if attempt == 1 else ("retry",)):
            raise ValueError("checkpoint continuation action is invalid")


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must contain 1-128 safe ASCII letters, digits, dots, underscores, or hyphens"
        )


def resume_mock_workflow(
    graph: Any,
    config: dict[str, Any],
    *,
    run_id: str,
    spec_text: str,
    scenario: MockScenario | str = MockScenario.SUCCESS,
    max_retries: int = 1,
) -> WorkflowState:
    """Validate a paused checkpoint before continuing its thread."""
    _validate_run_id(run_id)
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id != f"agentic-dev-workflow:{run_id}":
        raise ValueError("checkpoint thread_id does not match requested run")
    try:
        checkpoint = graph.get_state(config)
    except Exception:
        raise ValueError("checkpoint could not be read; stored data was preserved") from None
    if not checkpoint.values:
        raise ValueError("no checkpoint is available for this workflow thread")
    if not isinstance(checkpoint.values, dict):
        raise ValueError("checkpoint workflow state is invalid")
    validate_resume_identity(
        checkpoint.values,
        run_id=run_id,
        spec_text=spec_text,
        scenario=scenario,
        max_retries=max_retries,
    )
    if checkpoint.values.get("final_status") in ("success", "failed") and not checkpoint.next:
        raise ValueError("checkpoint is terminal and cannot be resumed")
    _validate_resume_state(checkpoint.values, checkpoint.next)
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
    interrupt_before: list[str] | None = None,
    resume: bool = False,
) -> WorkflowState:
    """Run or resume a deterministic mock graph using an injected saver."""
    if resume and run_id is None:
        raise ValueError("resume requires an explicit run_id")
    if resume:
        _validate_run_id(run_id or "")
        state = None
        selected_run_id = run_id
    else:
        state = create_initial_state(
            spec_text,
            scenario=scenario,
            run_id=run_id,
            max_retries=max_retries,
        )
        selected_run_id = state["run_id"]
    graph = build_mock_graph(
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
        tracing_session=tracing_session,
        interrupt_before=interrupt_before,
    )
    config = {"configurable": {"thread_id": f"agentic-dev-workflow:{selected_run_id}"}}

    def invoke_graph() -> WorkflowState:
        if resume:
            return resume_mock_workflow(
                graph,
                config,
                run_id=selected_run_id,
                spec_text=spec_text,
                scenario=scenario,
                max_retries=max_retries,
            )
        if graph.get_state(config).values:
            raise ValueError("run_id already has a checkpoint; use resume or choose a new run_id")
        return graph.invoke(state, config)

    tracer: Tracer | None = tracing_session.tracer if tracing_session else None

    def invoke() -> WorkflowState:
        if tracer is None:
            return invoke_graph()
        with tracer.start_as_current_span("workflow.checkpoint-session") as checkpoint_span:
            checkpoint_span.set_attribute("workflow.run_id", selected_run_id)
            checkpoint_span.set_attribute("workflow.phase", "checkpoint")
            checkpoint_span.set_attribute(
                "workflow.operation", "resume" if resume else "start"
            )
            checkpoint_span.set_attribute("workflow.mock_mode", True)
            checkpoint_span.set_attribute(
                "workflow.backend", "sqlite" if checkpointer is not None else "in-memory"
            )
            try:
                result = invoke_graph()
                outcome = (
                    "paused"
                    if result["final_status"] == "running"
                    else result["final_status"]
                )
                checkpoint_span.set_attribute("workflow.outcome", outcome)
                if outcome == "failed":
                    checkpoint_span.set_status(
                        Status(StatusCode.ERROR, "simulated workflow failure")
                    )
                return result
            except Exception as error:
                checkpoint_span.set_attribute("workflow.outcome", "error")
                checkpoint_span.set_status(Status(StatusCode.ERROR, type(error).__name__))
                raise

    if tracer is None:
        return invoke()
    with tracer.start_as_current_span("workflow.mock-run") as span:
        span.set_attribute("workflow.run_id", selected_run_id)
        span.set_attribute("workflow.phase", "mock-orchestration")
        span.set_attribute("workflow.node", "graph")
        span.set_attribute("workflow.mock_mode", True)
        try:
            result = invoke()
            outcome = (
                "paused"
                if result["final_status"] == "running"
                else result["final_status"]
            )
            span.set_attribute("workflow.outcome", outcome)
            if outcome == "failed":
                span.set_status(Status(StatusCode.ERROR, "simulated workflow failure"))
            return result
        except Exception as error:
            span.set_attribute("workflow.outcome", "error")
            span.set_status(Status(StatusCode.ERROR, type(error).__name__))
            raise
