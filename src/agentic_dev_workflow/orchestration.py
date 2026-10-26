"""Smallest end-to-end orchestration over the existing safety contracts.

Fake agents and an injected sandbox runner are composed in this order:
plan, explicit human approval, executor, Git inventory, exact approved-scope
validation, registered checks, independent review, and a bounded repair.
The same canonical repository root is used for inventory, scope, and the
lock-check applicability decision. A Git inventory is not approval. Nothing
is executed before approval is present and exactly matches the planned file
set. Verification errors are not successes and are not retried. Repair stops
at the configured limit. Resume accepts a saved ``verify`` checkpoint only
when a process-local HMAC shows that the executor completed that attempt.
Resume accepts a saved ``review`` or ``decide`` checkpoint only when its
verification and review evidence still matches a process-local HMAC written
by those nodes, and the same kind of HMAC covers every attempt-history entry.
Those bindings are not a defense against unrestricted code in this process.

This module does not call a model, a coding agent, or a real sandbox backend.
Checkpointing uses the injected LangGraph saver, including the existing
SQLite helper when the caller opens one.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agentic_dev_workflow.adapters import (
    MAX_AGENT_TIMEOUT_SECONDS,
    AdapterError,
    AgentAdapter,
    AgentRequest,
    AgentRole,
    AgentStatus,
    invoke_adapter,
)
from agentic_dev_workflow.agent_outputs import (
    MAX_CHECK_DURATION_MS,
    AgentOutputValidationError,
    CheckStatus,
    ExecutorOutcome,
    ReviewDecision,
    validate_executor_output,
    validate_planner_output,
    validate_relative_repository_path,
    validate_reviewer_output,
    validate_verifier_output,
)
from agentic_dev_workflow.git_inventory import (
    ApprovedScopeError,
    GitInventoryError,
    read_git_change_inventory,
    validate_approved_scope,
)
from agentic_dev_workflow.run_summary import build_run_summary
from agentic_dev_workflow.sandbox import (
    MAX_RUN_ID_ATTEMPT,
    SandboxRunner,
    SandboxRunnerError,
    SandboxStatus,
    VerifiedWorktree,
    invoke_sandbox_runner,
)
from agentic_dev_workflow.verification_registry import (
    CheckApplicability,
    IncompletePathInventoryError,
    VerificationRegistryError,
    get_check_definition,
    lock_check_applies,
)

MAX_RETRIES = 2
CHECKPOINT_FORMAT_VERSION = 2
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_EVIDENCE_KEY = secrets.token_bytes(32)
_EVIDENCE_DOMAIN = b"agentic-dev-workflow/orchestration-evidence/v1\x00"
_INTERRUPT_NODES = {"execute", "verify", "review", "decide"}
_CHECK_FIELDS = (
    "check_id",
    "status",
    "exit_code",
    "duration_ms",
    "evidence_ref",
    "summary",
)
_HISTORY_FIELDS = (
    "attempt",
    "verification_outcome",
    "evidence_refs",
    "review_decision",
    "checks",
    "terminal_error",
)


class OrchestrationState(TypedDict):
    """Serializable run state. Sandbox evidence and process configuration stay out."""

    run_id: str
    checkpoint_format_version: int
    spec_text: str
    spec_hash: str
    repository_root: str
    max_retries: int
    approval_granted: bool
    approved_paths: list[str]
    proposed_paths: list[str]
    plan: dict[str, Any] | None
    execution_attempt: int
    executor_invocations: int
    repair_retries_used: int
    execution_binding: str | None
    verification: dict[str, Any] | None
    verification_binding: str | None
    checks: list[dict[str, Any]]
    review_decision: str | None
    review_binding: str | None
    attempt_history: list[dict[str, Any]]
    history_binding: str | None
    final_status: Literal["running", "success", "failed", "awaiting_approval", "error"]
    errors: list[str]
    _next_action: NotRequired[Literal["continue", "execute", "verify", "retry", "stop"]]


@dataclass(frozen=True)
class ScopeApproval:
    """Explicit human approval of one exact relative-path set."""

    approved_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.approved_paths) is not tuple or any(
            type(path) is not str for path in self.approved_paths
        ):
            raise ValueError("approved_paths must be a tuple of strings")


@dataclass(frozen=True)
class OrchestrationReport:
    """Bounded terminal or paused report. It does not carry sandbox bytes."""

    run_id: str
    outcome: str
    attempts: int
    executor_invocations: int
    repair_retries_used: int
    max_retries: int
    checks: tuple[dict[str, Any], ...]
    review_decision: str | None
    evidence_refs: tuple[str, ...]
    errors: tuple[str, ...]
    proposed_paths: tuple[str, ...]
    attempt_history: tuple[dict[str, Any], ...]
    elapsed_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "outcome": self.outcome,
            "attempts": self.attempts,
            "executor_invocations": self.executor_invocations,
            "repair_retries_used": self.repair_retries_used,
            "max_retries": self.max_retries,
            "checks": [_public_check(check) for check in self.checks],
            "review_decision": self.review_decision,
            "evidence_refs": list(self.evidence_refs),
            "errors": list(self.errors),
            "proposed_paths": list(self.proposed_paths),
            "attempt_history": [_public_history(entry) for entry in self.attempt_history],
            "elapsed_seconds": self.elapsed_seconds,
        }
        payload["summary"] = build_run_summary(
            run_id=self.run_id,
            outcome=self.outcome,
            elapsed_seconds=self.elapsed_seconds,
            errors=list(self.errors),
            checks=payload["checks"],
            attempt_history=payload["attempt_history"],
            repair_retries_used=self.repair_retries_used,
            max_retries=self.max_retries,
            executor_invocations=self.executor_invocations,
        )
        return payload


def load_task_specification(
    *,
    inline: str | None = None,
    spec_path: Path | None = None,
) -> str:
    """Return one task specification.

    An inline string and a spec file may both be supplied only when their
    contents are exactly equal. Different texts, a missing source, and
    whitespace-only text are errors. The returned text is not logged.
    """
    if inline is not None and not isinstance(inline, str):
        raise ValueError("inline task must be a string")
    file_text: str | None = None
    if spec_path is not None:
        path = spec_path if isinstance(spec_path, Path) else Path(spec_path)
        if not path.is_file():
            raise ValueError("spec file does not exist")
        file_text = path.read_text(encoding="utf-8")
        if not file_text.strip():
            raise ValueError("spec file must not be empty")
    if inline is not None and not inline.strip():
        raise ValueError("inline task must not be empty")
    if inline is None and file_text is None:
        raise ValueError("task specification is missing")
    if inline is not None and file_text is not None and inline != file_text:
        raise ValueError("inline task and spec file disagree")
    chosen = inline if inline is not None else file_text
    if not isinstance(chosen, str) or not chosen.strip():
        raise ValueError("task specification is missing")
    return chosen


def run_orchestration(
    spec_text: str,
    *,
    worktree: VerifiedWorktree,
    planner: AgentAdapter,
    executor: AgentAdapter,
    reviewer: AgentAdapter,
    sandbox: SandboxRunner,
    approval: ScopeApproval | None,
    max_retries: int = 1,
    run_id: str | None = None,
    checkpointer: Any | None = None,
    interrupt_before: list[str] | None = None,
    resume: bool = False,
    agent_timeout_seconds: float = 30,
) -> OrchestrationReport:
    """Run or resume one controlled workflow on a verified worktree handle."""
    repository_root = _canonical_worktree_root(worktree)
    selected_run_id = worktree.run_id if run_id is None else run_id
    _validate_run_id(selected_run_id)
    if selected_run_id != worktree.run_id:
        raise ValueError("worktree run_id does not match the workflow run")
    _validate_max_retries(max_retries)
    _validate_agent_timeout(agent_timeout_seconds)
    if not isinstance(spec_text, str) or not spec_text.strip():
        raise ValueError("spec_text must not be empty")
    if approval is not None and type(approval) is not ScopeApproval:
        raise ValueError("approval must be an explicit scope approval")
    interrupt_nodes = _validate_interrupt_before(interrupt_before)
    if resume and interrupt_nodes:
        raise ValueError("resume cannot introduce a new interrupt")

    saver = checkpointer if checkpointer is not None else InMemorySaver()
    graph = _build_graph(
        repository_root=repository_root,
        worktree=worktree,
        planner=planner,
        executor=executor,
        reviewer=reviewer,
        sandbox=sandbox,
        checkpointer=saver,
        interrupt_before=interrupt_nodes,
        agent_timeout_seconds=float(agent_timeout_seconds),
    )
    config = {
        "configurable": {
            "thread_id": f"agentic-dev-workflow:orchestration:{selected_run_id}"
        }
    }
    started = time.monotonic()
    if resume:
        state = _resume(
            graph,
            config,
            run_id=selected_run_id,
            spec_text=spec_text,
            max_retries=max_retries,
            repository_root=repository_root.as_posix(),
            approval=approval,
        )
    else:
        if _checkpoint_exists(graph, config):
            raise ValueError(
                "run_id already has a checkpoint; use resume or choose a new run_id"
            )
        state = graph.invoke(
            _initial_state(
                spec_text,
                run_id=selected_run_id,
                repository_root=repository_root.as_posix(),
                max_retries=max_retries,
                approval=approval,
            ),
            config,
        )
    snapshot = graph.get_state(config)
    outcome = "paused" if snapshot.next else state["final_status"]
    if outcome == "running":
        raise ValueError("workflow stopped without a terminal outcome")
    return _report(state, outcome=outcome, elapsed_seconds=time.monotonic() - started)


def _build_graph(
    *,
    repository_root: Path,
    worktree: VerifiedWorktree,
    planner: AgentAdapter,
    executor: AgentAdapter,
    reviewer: AgentAdapter,
    sandbox: SandboxRunner,
    checkpointer: Any,
    interrupt_before: list[str] | None,
    agent_timeout_seconds: float = 30,
) -> Any:
    def plan(state: OrchestrationState) -> dict[str, Any]:
        try:
            result = invoke_adapter(
                planner,
                _request(
                    state["run_id"],
                    AgentRole.PLANNER,
                    state["spec_text"].strip(),
                    {"phase": "plan"},
                    timeout_seconds=agent_timeout_seconds,
                    invocation=1,
                ),
            )
        except AdapterError as error:
            return _terminal("error", error.diagnostic_code)
        if result.status != AgentStatus.SUCCEEDED:
            return _terminal("error", result.diagnostic_code or "planner_failed")
        try:
            output = validate_planner_output(dict(result.structured_output))
        except AgentOutputValidationError as error:
            return _terminal("error", error.diagnostic_code)
        proposed: list[str] = []
        for step in output.steps:
            for path in step.expected_files:
                if path not in proposed:
                    proposed.append(path)
        return {
            "plan": output.to_state(),
            "proposed_paths": proposed,
            "_next_action": "continue",
        }

    def approval(state: OrchestrationState) -> dict[str, Any]:
        if not state["approval_granted"]:
            return _terminal("awaiting_approval", "approval_required")
        try:
            approved = [
                validate_relative_repository_path(path)
                for path in state["approved_paths"]
            ]
        except AgentOutputValidationError:
            return _terminal("failed", "invalid_approved_path")
        if len(set(approved)) != len(approved):
            return _terminal("failed", "ambiguous_scope")
        proposed = state["proposed_paths"]
        if set(approved) != set(proposed) or len(approved) != len(proposed):
            return _terminal("failed", "scope_mismatch")
        return {"approved_paths": approved, "_next_action": "execute"}

    def execute(state: OrchestrationState) -> dict[str, Any]:
        if not state["approval_granted"]:
            return _terminal("awaiting_approval", "approval_required")
        if set(state["approved_paths"]) != set(state["proposed_paths"]):
            return _terminal("failed", "scope_mismatch")
        attempt = state["execution_attempt"] + 1
        if attempt > state["max_retries"] + 1 or attempt > MAX_RUN_ID_ATTEMPT:
            return _terminal("failed", "retries_exhausted")
        invocations = state["executor_invocations"] + 1
        try:
            result = invoke_adapter(
                executor,
                _request(
                    state["run_id"],
                    AgentRole.EXECUTOR,
                    "Apply the approved file scope",
                    {
                        "approved_paths": list(state["approved_paths"]),
                        "attempt": attempt,
                        "plan": state["plan"],
                        "specification": state["spec_text"],
                    },
                    timeout_seconds=agent_timeout_seconds,
                    invocation=invocations,
                    attempt=attempt,
                ),
            )
        except AdapterError as error:
            return _invoked(_terminal("error", error.diagnostic_code), invocations)
        if result.status != AgentStatus.SUCCEEDED:
            return _invoked(
                _terminal("error", result.diagnostic_code or "executor_failed"),
                invocations,
            )
        try:
            output = validate_executor_output(dict(result.structured_output))
        except AgentOutputValidationError as error:
            return _invoked(_terminal("error", error.diagnostic_code), invocations)
        if output.status != ExecutorOutcome.COMPLETED:
            # The validated diagnostic_code is model text. Keep the workflow
            # error on this control-flow outcome instead of publishing it.
            return _invoked(_terminal("error", "executor_failed"), invocations)
        return {
            "execution_attempt": attempt,
            "executor_invocations": invocations,
            "execution_binding": _sign(
                "execute",
                _execution_material(state, attempt=attempt, invocations=invocations),
            ),
            "_next_action": "verify",
        }

    def verify(state: OrchestrationState) -> dict[str, Any]:
        if not _execution_authentic(state) or not _history_authentic(state):
            return _terminal("error", "fabricated_evidence")
        if (
            state["repository_root"] != repository_root.as_posix()
            or worktree.path != repository_root.as_posix()
            or worktree.run_id != state["run_id"]
        ):
            return _close_attempt(state, "error", "repository_root_mismatch")
        try:
            inventory = read_git_change_inventory(repository_root)
            validate_approved_scope(inventory, tuple(state["approved_paths"]))
        except GitInventoryError as error:
            return _close_attempt(state, "error", error.diagnostic_code.value)
        except ApprovedScopeError as error:
            return _close_attempt(state, "failed", error.diagnostic_code.value)
        plan = state["plan"] or {}
        try:
            definitions = [
                get_check_definition(check_id)
                for check_id in plan.get("verification_check_ids", ())
            ]
        except VerificationRegistryError:
            return _close_attempt(state, "error", "verification_error")
        complete = None
        if any(
            definition.applicability == CheckApplicability.DEPENDENCY_FILES_CHANGED
            for definition in definitions
        ):
            try:
                complete = inventory.to_complete_path_inventory(
                    repository_root=repository_root
                )
            except GitInventoryError:
                return _close_attempt(state, "error", "verification_error")
        report_checks: list[dict[str, Any]] = []
        executed: list[dict[str, Any]] = []
        for definition in definitions:
            if definition.applicability == CheckApplicability.DEPENDENCY_FILES_CHANGED:
                try:
                    applies = lock_check_applies(complete, repository_root=repository_root)
                except IncompletePathInventoryError:
                    return _close_attempt(state, "error", "verification_error")
                if not applies:
                    report_checks.append(
                        _check_record(
                            definition.check_id.value,
                            CheckStatus.SKIPPED.value,
                            exit_code=None,
                            duration_ms=0,
                            evidence_ref=None,
                            summary="dependency files were not changed",
                        )
                    )
                    continue
            try:
                outcome = invoke_sandbox_runner(
                    sandbox,
                    definition,
                    worktree,
                    run_id=state["run_id"],
                    attempt=state["execution_attempt"],
                )
            except SandboxRunnerError:
                record = _error_check(definition.check_id.value)
                report_checks.append(record)
                executed.append(record)
                continue
            record = _sandbox_check(outcome)
            report_checks.append(record)
            if record["status"] != CheckStatus.SKIPPED.value:
                executed.append(record)
        if not executed:
            return _close_attempt(
                state, "error", "verification_error", checks=report_checks
            )
        verifier_outcome = _verifier_outcome(executed)
        payload = {
            "outcome": verifier_outcome,
            "checks": [_verifier_check(check) for check in executed],
        }
        try:
            verified = validate_verifier_output(payload)
        except AgentOutputValidationError:
            return _close_attempt(state, "error", "verification_error", checks=report_checks)
        verification = verified.to_state()
        update: dict[str, Any] = {
            "checks": report_checks,
            "verification": verification,
            "verification_binding": _sign(
                "verify",
                _verify_material(state, verification=verification, checks=report_checks),
            ),
            "review_binding": None,
            "final_status": "running",
            "_next_action": "stop",
        }
        if verifier_outcome == "error":
            update["errors"] = ["verification_error"]
        return update

    def review(state: OrchestrationState) -> dict[str, Any]:
        if not _execution_authentic(state) or not _history_authentic(state):
            return _terminal("error", "fabricated_evidence")
        if _attempt_closed(state) or state["final_status"] != "running":
            return {"review_decision": None}
        verification = state.get("verification") or {}
        outcome = verification.get("outcome")
        if outcome not in ("pass", "fail"):
            return _reviewed_attempt(state, None)
        evidence_refs = [
            check["evidence_ref"]
            for check in state["checks"]
            if isinstance(check.get("evidence_ref"), str)
        ]
        try:
            result = invoke_adapter(
                reviewer,
                _request(
                    state["run_id"],
                    AgentRole.REVIEWER,
                    "Review the verification result",
                    {
                        "verification_outcome": outcome,
                        "evidence_refs": evidence_refs,
                        "attempt": state["execution_attempt"],
                        "specification": state["spec_text"],
                    },
                    timeout_seconds=agent_timeout_seconds,
                    invocation=state["execution_attempt"],
                    attempt=state["execution_attempt"],
                ),
            )
        except AdapterError as error:
            return _reviewed_failure(state, error.diagnostic_code)
        if result.status != AgentStatus.SUCCEEDED:
            return _reviewed_failure(state, result.diagnostic_code or "reviewer_failed")
        try:
            output = validate_reviewer_output(dict(result.structured_output))
        except AgentOutputValidationError as error:
            return _reviewed_failure(state, error.diagnostic_code)
        decision = output.decision.value
        return _reviewed_attempt(state, decision)

    def decide(state: OrchestrationState) -> dict[str, Any]:
        if state["final_status"] != "running":
            return {"_next_action": "stop"}
        if not _execution_authentic(state) or not _history_authentic(state):
            return _terminal("error", "fabricated_evidence")
        verification = state.get("verification") or {}
        outcome = verification.get("outcome")
        if outcome == "error" or outcome is None:
            errors = list(state["errors"])
            if "verification_error" not in errors:
                errors.append("verification_error")
            return {
                "final_status": "error",
                "errors": errors,
                "_next_action": "stop",
            }
        if outcome == "pass" and state.get("review_decision") == ReviewDecision.APPROVE.value:
            if not _success_evidence_ok(state):
                return _terminal("error", "fabricated_evidence")
            return {"final_status": "success", "errors": [], "_next_action": "stop"}
        reason = (
            "review_rejected" if outcome == "pass" else "verification_failed"
        )
        if state["repair_retries_used"] < state["max_retries"]:
            return {
                "repair_retries_used": state["repair_retries_used"] + 1,
                "final_status": "running",
                "review_decision": None,
                "_next_action": "retry",
            }
        return {
            "final_status": "failed",
            "errors": [reason, "retries_exhausted"],
            "_next_action": "stop",
        }

    builder = StateGraph(OrchestrationState)
    builder.add_node("plan", plan)
    builder.add_node("approval", approval)
    builder.add_node("execute", execute)
    builder.add_node("verify", verify)
    builder.add_node("review", review)
    builder.add_node("decide", decide)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan",
        lambda state: state.get("_next_action", "stop"),
        {"continue": "approval", "stop": END},
    )
    builder.add_conditional_edges(
        "approval",
        lambda state: state.get("_next_action", "stop"),
        {"execute": "execute", "stop": END},
    )
    builder.add_conditional_edges(
        "execute",
        lambda state: state.get("_next_action", "stop"),
        {"verify": "verify", "stop": END},
    )
    builder.add_edge("verify", "review")
    builder.add_edge("review", "decide")
    builder.add_conditional_edges(
        "decide",
        lambda state: state.get("_next_action", "stop"),
        {"retry": "execute", "stop": END},
    )
    return builder.compile(
        checkpointer=checkpointer, interrupt_before=interrupt_before or None
    )


def _resume(
    graph: Any,
    config: dict[str, Any],
    *,
    run_id: str,
    spec_text: str,
    max_retries: int,
    repository_root: str,
    approval: ScopeApproval | None,
) -> OrchestrationState:
    if approval is None:
        raise ValueError("resume requires the original scope approval")
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id != f"agentic-dev-workflow:orchestration:{run_id}":
        raise ValueError("checkpoint thread_id does not match requested run")
    try:
        checkpoint = graph.get_state(config)
    except Exception:
        raise ValueError("checkpoint could not be read; stored data was preserved") from None
    state = checkpoint.values
    if not state:
        raise ValueError("no checkpoint is available for this workflow thread")
    if not isinstance(state, dict):
        raise ValueError("checkpoint workflow state is invalid")
    _validate_resume_identity(
        state,
        run_id=run_id,
        spec_text=spec_text,
        max_retries=max_retries,
        repository_root=repository_root,
        approval=approval,
        next_nodes=tuple(checkpoint.next),
    )
    return graph.invoke(None, config)


def _validate_resume_identity(
    state: dict[str, Any],
    *,
    run_id: str,
    spec_text: str,
    max_retries: int,
    repository_root: str,
    approval: ScopeApproval,
    next_nodes: tuple[str, ...],
) -> None:
    required = {
        "run_id",
        "checkpoint_format_version",
        "spec_text",
        "spec_hash",
        "repository_root",
        "max_retries",
        "approval_granted",
        "approved_paths",
        "proposed_paths",
        "plan",
        "execution_attempt",
        "repair_retries_used",
        "executor_invocations",
        "execution_binding",
        "verification",
        "verification_binding",
        "checks",
        "review_decision",
        "review_binding",
        "attempt_history",
        "history_binding",
        "final_status",
        "errors",
    }
    if not required.issubset(state):
        raise ValueError("checkpoint workflow state is incomplete")
    if state.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("checkpoint format version is incompatible")
    if state.get("run_id") != run_id:
        raise ValueError("checkpoint run_id does not match requested run")
    if state.get("repository_root") != repository_root:
        raise ValueError("checkpoint repository root does not match the worktree")
    if state.get("max_retries") != max_retries:
        raise ValueError("checkpoint retry limit does not match requested retry limit")
    requested_hash = hashlib.sha256(spec_text.encode("utf-8")).hexdigest()
    if state.get("spec_hash") != requested_hash:
        raise ValueError(
            "checkpoint specification hash does not match requested specification"
        )
    stored_spec = state.get("spec_text")
    if (
        not isinstance(stored_spec, str)
        or hashlib.sha256(stored_spec.encode("utf-8")).hexdigest()
        != state.get("spec_hash")
    ):
        raise ValueError("checkpoint specification state is invalid")
    if state.get("approval_granted") is not True:
        raise ValueError("checkpoint does not contain explicit scope approval")
    try:
        approved = [
            validate_relative_repository_path(path) for path in approval.approved_paths
        ]
    except AgentOutputValidationError as error:
        raise ValueError("resume approval does not match the checkpoint") from error
    if approved != state.get("approved_paths"):
        raise ValueError("checkpoint approved scope does not match the requested approval")
    if state.get("final_status") != "running":
        raise ValueError("checkpoint continuation/status combination is invalid")
    if len(next_nodes) != 1 or next_nodes[0] not in _INTERRUPT_NODES:
        raise ValueError("checkpoint continuation state is invalid")
    attempt = state.get("execution_attempt")
    retries = state.get("repair_retries_used")
    if (
        not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or not isinstance(retries, int)
        or isinstance(retries, bool)
        or not 0 <= retries <= max_retries
        or not 0 <= attempt <= max_retries + 1
    ):
        raise ValueError("checkpoint retry state is outside configured limits")
    if not isinstance(state.get("plan"), dict):
        raise ValueError("checkpoint planner output is missing or invalid")
    try:
        planned = validate_planner_output(state["plan"])
    except AgentOutputValidationError as error:
        raise ValueError("checkpoint planner output is missing or invalid") from error
    proposed: list[str] = []
    for step in planned.steps:
        for path in step.expected_files:
            if path not in proposed:
                proposed.append(path)
    if proposed != state.get("proposed_paths"):
        raise ValueError("checkpoint planner output is missing or invalid")
    if set(state.get("approved_paths", ())) != set(proposed):
        raise ValueError("checkpoint approved scope does not match the planned file scope")
    if not isinstance(state.get("checks"), list) or not isinstance(
        state.get("attempt_history"), list
    ):
        raise ValueError("checkpoint workflow state is invalid")
    if not isinstance(state.get("errors"), list) or any(
        type(item) is not str for item in state["errors"]
    ):
        raise ValueError("checkpoint error history is invalid")
    _validate_saved_transition(state, next_nodes[0])


def _initial_state(
    spec_text: str,
    *,
    run_id: str,
    repository_root: str,
    max_retries: int,
    approval: ScopeApproval | None,
) -> OrchestrationState:
    return {
        "run_id": run_id,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "spec_text": spec_text,
        "spec_hash": hashlib.sha256(spec_text.encode("utf-8")).hexdigest(),
        "repository_root": repository_root,
        "max_retries": max_retries,
        "approval_granted": approval is not None,
        "approved_paths": [] if approval is None else list(approval.approved_paths),
        "proposed_paths": [],
        "plan": None,
        "execution_attempt": 0,
        "executor_invocations": 0,
        "repair_retries_used": 0,
        "execution_binding": None,
        "verification": None,
        "verification_binding": None,
        "checks": [],
        "review_decision": None,
        "review_binding": None,
        "attempt_history": [],
        "history_binding": None,
        "final_status": "running",
        "errors": [],
        "_next_action": "stop",
    }


def _canonical_worktree_root(worktree: VerifiedWorktree) -> Path:
    if type(worktree) is not VerifiedWorktree:
        raise ValueError("worktree must be a verified worktree handle")
    try:
        root = Path(worktree.path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("worktree path is not the canonical repository root") from error
    if root.as_posix() != worktree.path or not root.is_dir():
        raise ValueError("worktree path is not the canonical repository root")
    return root


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must be 1-64 safe ASCII letters, digits, '_' or '-'")


def _validate_agent_timeout(agent_timeout_seconds: float) -> None:
    if (
        not isinstance(agent_timeout_seconds, (int, float))
        or isinstance(agent_timeout_seconds, bool)
        or not math.isfinite(float(agent_timeout_seconds))
        or agent_timeout_seconds <= 0
        or agent_timeout_seconds > MAX_AGENT_TIMEOUT_SECONDS
    ):
        raise ValueError(
            f"agent_timeout_seconds must be positive and at most {MAX_AGENT_TIMEOUT_SECONDS}"
        )


def _validate_max_retries(max_retries: int) -> None:
    if (
        not isinstance(max_retries, int)
        or isinstance(max_retries, bool)
        or not 0 <= max_retries <= MAX_RETRIES
        or max_retries + 1 > MAX_RUN_ID_ATTEMPT
    ):
        raise ValueError("max_retries must be an integer from 0 to 2")


def _validate_interrupt_before(interrupt_before: list[str] | None) -> list[str] | None:
    if interrupt_before is None:
        return None
    if type(interrupt_before) is not list or any(
        type(node) is not str or node not in _INTERRUPT_NODES for node in interrupt_before
    ):
        raise ValueError("interrupt_before names an unknown workflow node")
    return list(interrupt_before)


def _checkpoint_exists(graph: Any, config: dict[str, Any]) -> bool:
    try:
        checkpoint = graph.get_state(config)
    except Exception:
        return False
    return bool(checkpoint.values)


def _request(
    run_id: str,
    role: AgentRole,
    task: str,
    context: dict[str, Any],
    *,
    timeout_seconds: float,
    invocation: int,
    attempt: int | None = None,
) -> AgentRequest:
    if type(invocation) is not int or invocation < 1:
        raise ValueError("invocation must be a positive integer")
    metadata = {
        "role": role.value,
        "invocation": str(invocation),
        "invocation_id": f"{role.value}:{invocation}",
    }
    if attempt is not None:
        if type(attempt) is not int or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        metadata["attempt"] = str(attempt)
    return AgentRequest(
        run_id=run_id,
        role=role,
        task=task,
        structured_context=context,
        timeout_seconds=timeout_seconds,
        correlation_metadata=metadata,
    )


def _terminal(status: str, code: str) -> dict[str, Any]:
    return {"final_status": status, "errors": [code], "_next_action": "stop"}


def _invoked(update: dict[str, Any], invocations: int) -> dict[str, Any]:
    return {**update, "executor_invocations": invocations}


def _close_attempt(
    state: OrchestrationState,
    status: str,
    code: str,
    *,
    checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Close the current attempt without leaving an earlier pass as its result."""
    entry = _history_entry(
        state,
        None,
        outcome="not_run",
        checks=[] if checks is None else checks,
        terminal_error=code,
    )
    return {
        "final_status": status,
        "errors": [code],
        "verification": None,
        "verification_binding": None,
        "checks": [],
        "review_decision": None,
        "review_binding": None,
        **_recorded_history(state, entry),
        "_next_action": "stop",
    }


def _evidence_ref(check_id: str) -> str:
    return f"ev_{check_id.replace('.', '_')}"


def _check_record(
    check_id: str,
    status: str,
    *,
    exit_code: int | None,
    duration_ms: int,
    evidence_ref: str | None,
    summary: str,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": status,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "evidence_ref": evidence_ref,
        "summary": summary,
    }


def _error_check(check_id: str) -> dict[str, Any]:
    return _check_record(
        check_id,
        CheckStatus.ERROR.value,
        exit_code=None,
        duration_ms=0,
        evidence_ref=_evidence_ref(check_id),
        summary="registered check could not run",
    )


def _sandbox_check(outcome: Any) -> dict[str, Any]:
    check_id = outcome.check_id.value
    if outcome.status == SandboxStatus.PASSED:
        status = CheckStatus.PASSED.value
        exit_code = 0
        summary = "registered check passed"
    elif outcome.status == SandboxStatus.FAILED:
        status = CheckStatus.FAILED.value
        exit_code = outcome.exit_code
        summary = "registered check failed"
    else:
        status = CheckStatus.ERROR.value
        exit_code = None
        summary = "registered check could not run"
    try:
        duration_ms = int(round(float(outcome.duration_seconds) * 1000))
    except (TypeError, ValueError, OverflowError):
        duration_ms = -1
    if not 0 <= duration_ms <= MAX_CHECK_DURATION_MS:
        return _error_check(check_id)
    return _check_record(
        check_id,
        status,
        exit_code=exit_code,
        duration_ms=duration_ms,
        evidence_ref=_evidence_ref(check_id),
        summary=summary,
    )


def _verifier_check(check: dict[str, Any]) -> dict[str, Any]:
    return {
        "check_id": check["check_id"],
        "status": check["status"],
        "exit_code": check["exit_code"],
        "duration_ms": check["duration_ms"],
        "evidence_ref": check["evidence_ref"],
        "summary": check["summary"],
    }


def _verifier_outcome(checks: list[dict[str, Any]]) -> str:
    statuses = {check["status"] for check in checks}
    if CheckStatus.ERROR.value in statuses:
        return "error"
    if CheckStatus.FAILED.value in statuses:
        return "fail"
    if statuses == {CheckStatus.PASSED.value}:
        return "pass"
    return "error"


def _history_entry(
    state: OrchestrationState,
    review_decision: str | None,
    *,
    outcome: str | None = None,
    checks: list[dict[str, Any]] | None = None,
    terminal_error: str | None = None,
) -> dict[str, Any]:
    verification = state.get("verification") or {}
    recorded = outcome if outcome is not None else verification.get("outcome") or "not_run"
    if not isinstance(recorded, str):
        recorded = "not_run"
    source = state["checks"] if checks is None else checks
    public_checks = [_public_check(check) for check in source]
    return {
        "attempt": state["execution_attempt"],
        "verification_outcome": recorded,
        "evidence_refs": [
            check["evidence_ref"]
            for check in public_checks
            if isinstance(check.get("evidence_ref"), str)
        ],
        "review_decision": review_decision,
        "checks": public_checks,
        "terminal_error": terminal_error,
    }


def _reviewed_attempt(state: OrchestrationState, decision: str | None) -> dict[str, Any]:
    entry = _history_entry(state, decision)
    return {
        "review_decision": decision,
        "review_binding": _sign("review", _review_material(state, entry)),
        **_recorded_history(state, entry),
    }


def _reviewed_failure(state: OrchestrationState, code: str) -> dict[str, Any]:
    recorded = _reviewed_attempt(state, None)
    return {
        **recorded,
        "final_status": "error",
        "errors": [code],
        "_next_action": "stop",
    }


def _attempt_closed(state: OrchestrationState) -> bool:
    history = state["attempt_history"]
    return bool(history) and history[-1].get("attempt") == state["execution_attempt"] and bool(
        history[-1].get("terminal_error")
    )


def _recorded_history(state: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    history = [*state["attempt_history"], entry]
    return {
        "attempt_history": history,
        "history_binding": _sign("history", _history_material(history)),
    }


def _execution_material(
    state: dict[str, Any], *, attempt: int, invocations: int
) -> dict[str, Any]:
    return {
        "run_id": state["run_id"],
        "repository_root": state["repository_root"],
        "execution_attempt": attempt,
        "executor_invocations": invocations,
        "approved_paths": list(state["approved_paths"]),
    }


def _history_material(history: list[Any]) -> dict[str, Any]:
    return {"entries": [_public_history(entry) for entry in history]}


def _execution_authentic(state: dict[str, Any]) -> bool:
    attempt = state.get("execution_attempt")
    invocations = state.get("executor_invocations")
    if type(attempt) is not int or type(invocations) is not int:
        return False
    return _matches(
        state.get("execution_binding"),
        _sign(
            "execute",
            _execution_material(state, attempt=attempt, invocations=invocations),
        ),
    )


def _history_authentic(state: dict[str, Any]) -> bool:
    history = state.get("attempt_history")
    if not isinstance(history, list) or any(not isinstance(entry, dict) for entry in history):
        return False
    if not history:
        return state.get("history_binding") is None
    return _matches(state.get("history_binding"), _sign("history", _history_material(history)))


def _require_execution_binding(state: dict[str, Any]) -> None:
    if not _execution_authentic(state):
        _reject_evidence()


def _require_history_binding(state: dict[str, Any]) -> None:
    if not _history_authentic(state):
        _reject_evidence()


def _sign(kind: str, material: dict[str, Any]) -> str:
    body = json.dumps(
        {"kind": kind, **material},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hmac.new(_EVIDENCE_KEY, _EVIDENCE_DOMAIN + body, hashlib.sha256).hexdigest()


def _matches(actual: object, expected: str) -> bool:
    return (
        type(actual) is str
        and len(actual) == len(expected)
        and hmac.compare_digest(actual, expected)
    )


def _verify_material(
    state: dict[str, Any],
    *,
    verification: object | None = None,
    checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected_checks = state["checks"] if checks is None else checks
    return {
        "run_id": state["run_id"],
        "repository_root": state["repository_root"],
        "execution_attempt": state["execution_attempt"],
        "approved_paths": list(state["approved_paths"]),
        "verification": state["verification"] if verification is None else verification,
        "checks": [_public_check(check) for check in selected_checks],
    }


def _review_material(state: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    material = _verify_material(state)
    material["history_entry"] = _public_history(entry)
    return material


def _reject_evidence() -> None:
    raise ValueError("checkpoint evidence is missing or inconsistent")


def _binding_matches(state: dict[str, Any], kind: str, material: dict[str, Any]) -> bool:
    field = "verification_binding" if kind == "verify" else "review_binding"
    return _matches(state.get(field), _sign(kind, material))


def _require_verifier_binding(state: dict[str, Any]) -> None:
    verification = state.get("verification")
    if not isinstance(verification, dict):
        _reject_evidence()
    try:
        validate_verifier_output(verification)
    except AgentOutputValidationError:
        _reject_evidence()
    if not _binding_matches(state, "verify", _verify_material(state)):
        _reject_evidence()


def _require_review_binding(state: dict[str, Any]) -> None:
    history = state.get("attempt_history")
    if not isinstance(history, list) or not history or not isinstance(history[-1], dict):
        _reject_evidence()
    if not _binding_matches(state, "review", _review_material(state, history[-1])):
        _reject_evidence()


def _success_evidence_ok(state: dict[str, Any]) -> bool:
    history = state.get("attempt_history") or []
    verification = state.get("verification") or {}
    if not history or not isinstance(history[-1], dict):
        return False
    last = history[-1]
    if last.get("verification_outcome") != "pass" or last.get("review_decision") != "approve":
        return False
    if verification.get("outcome") != "pass" or state.get("review_decision") != "approve":
        return False
    if not _history_authentic(state) or not _execution_authentic(state):
        return False
    try:
        validate_verifier_output(verification)
    except (AgentOutputValidationError, TypeError, ValueError):
        return False
    return _binding_matches(
        state, "verify", _verify_material(state)
    ) and _binding_matches(state, "review", _review_material(state, last))


def _validate_saved_transition(state: dict[str, Any], next_node: str) -> None:
    attempt = state["execution_attempt"]
    invocations = state.get("executor_invocations")
    history = state.get("attempt_history")
    if (
        type(invocations) is not int
        or isinstance(invocations, bool)
        or invocations != attempt
        or not isinstance(history, list)
    ):
        _reject_evidence()
    if next_node == "execute" and attempt == 0:
        if (
            history
            or state.get("verification") is not None
            or state.get("checks")
            or state.get("review_decision") is not None
            or state.get("execution_binding") is not None
            or state.get("verification_binding") is not None
            or state.get("review_binding") is not None
            or state.get("history_binding") is not None
            or state.get("repair_retries_used") != 0
        ):
            _reject_evidence()
        return
    _require_history_binding(state)
    if next_node == "execute":
        if len(history) != attempt:
            _reject_evidence()
        _require_execution_binding(state)
        _require_verifier_binding(state)
        _require_review_binding(state)
        last = history[-1]
        if not isinstance(last, dict) or (
            last.get("verification_outcome") == "pass"
            and last.get("review_decision") == "approve"
        ):
            _reject_evidence()
        return
    if next_node == "verify":
        if attempt < 1 or len(history) != attempt - 1:
            _reject_evidence()
        _require_execution_binding(state)
        return
    if next_node == "review":
        if (
            attempt < 1
            or len(history) != attempt - 1
            or state.get("review_decision") is not None
            or state.get("review_binding") is not None
        ):
            _reject_evidence()
        _require_execution_binding(state)
        _require_verifier_binding(state)
        return
    if next_node == "decide":
        if attempt < 1 or len(history) != attempt:
            _reject_evidence()
        _require_execution_binding(state)
        _require_verifier_binding(state)
        _require_review_binding(state)
        last = history[-1]
        verification = state.get("verification") or {}
        if (
            not isinstance(last, dict)
            or last.get("verification_outcome") != verification.get("outcome")
            or last.get("review_decision") != state.get("review_decision")
        ):
            _reject_evidence()
        return
    _reject_evidence()


def _public_check(check: dict[str, Any]) -> dict[str, Any]:
    return {field: check.get(field) for field in _CHECK_FIELDS}


def _public_history(entry: dict[str, Any]) -> dict[str, Any]:
    published = {field: entry.get(field) for field in _HISTORY_FIELDS}
    refs = published["evidence_refs"]
    published["evidence_refs"] = list(refs) if isinstance(refs, (list, tuple)) else []
    checks = published["checks"]
    published["checks"] = (
        [_public_check(check) for check in checks if isinstance(check, dict)]
        if isinstance(checks, list)
        else []
    )
    return published


def _report(
    state: OrchestrationState, *, outcome: str, elapsed_seconds: float | None = None
) -> OrchestrationReport:
    checks = tuple(_public_check(check) for check in state["checks"])
    return OrchestrationReport(
        run_id=state["run_id"],
        outcome=outcome,
        attempts=state["execution_attempt"],
        executor_invocations=state["executor_invocations"],
        repair_retries_used=state["repair_retries_used"],
        max_retries=state["max_retries"],
        checks=checks,
        review_decision=state["review_decision"],
        evidence_refs=tuple(
            check["evidence_ref"]
            for check in checks
            if isinstance(check.get("evidence_ref"), str)
        ),
        errors=tuple(state["errors"]),
        proposed_paths=tuple(state["proposed_paths"]),
        attempt_history=tuple(_public_history(entry) for entry in state["attempt_history"]),
        elapsed_seconds=elapsed_seconds,
    )
