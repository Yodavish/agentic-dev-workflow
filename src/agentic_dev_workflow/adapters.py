"""Typed agent contracts and deterministic adapters; no process or model calls."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping, Protocol, runtime_checkable

MAX_AGENT_TIMEOUT_SECONDS = 600
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SAFE_CODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ADAPTER_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_FORBIDDEN_CONTEXT_KEYS = {
    "args",
    "arguments",
    "argv",
    "command",
    "commands",
    "cwd",
    "env",
    "environment",
    "environmentdump",
    "environmentvariables",
    "environmentvars",
    "executable",
    "executablepath",
    "processenvironment",
    "shell",
    "shellcommand",
    "workdir",
    "workingdirectory",
}


class AgentRole(StrEnum):
    """Agent responsibilities recognized by the runner contract."""

    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"


class AgentStatus(StrEnum):
    """Terminal status returned by an adapter invocation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentFailureCategory(StrEnum):
    """Stable failure classes safe to use in workflow control and telemetry."""

    TIMEOUT = "timeout"
    UNAVAILABLE_ADAPTER = "unavailable_adapter"
    INVALID_OUTPUT = "invalid_output"
    EXECUTION_FAILURE = "execution_failure"
    POLICY_REJECTION = "policy_rejection"
    CANCELLATION = "cancellation"


@dataclass(frozen=True)
class AgentRequest:
    """Provider-neutral task request. Process configuration is intentionally absent."""

    run_id: str
    role: AgentRole
    task: str
    structured_context: Mapping[str, object]
    timeout_seconds: float
    correlation_metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not _RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError("run_id must be 1-64 safe ASCII letters, digits, '_' or '-'")
        try:
            object.__setattr__(self, "role", AgentRole(self.role))
        except (TypeError, ValueError) as error:
            raise ValueError("role must be planner, executor, or reviewer") from error
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must not be empty")
        if not isinstance(self.structured_context, Mapping):
            raise ValueError("structured_context must be a mapping")
        _validate_json_mapping(self.structured_context, "structured_context")
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
            or not _is_finite_number(self.timeout_seconds)
            or self.timeout_seconds <= 0
            or self.timeout_seconds > MAX_AGENT_TIMEOUT_SECONDS
        ):
            raise ValueError(
                f"timeout_seconds must be positive and at most {MAX_AGENT_TIMEOUT_SECONDS}"
            )
        if not isinstance(self.correlation_metadata, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.correlation_metadata.items()
        ):
            raise ValueError("correlation_metadata must map strings to strings")


@dataclass(frozen=True)
class AgentResult:
    """Structured result with bounded diagnostics and no raw process output field."""

    run_id: str
    role: AgentRole
    status: AgentStatus
    structured_output: Mapping[str, object] = field(default_factory=dict)
    exit_code: int | None = None
    duration_seconds: float = 0.0
    failure_category: AgentFailureCategory | None = None
    diagnostic_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not _RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError("run_id is invalid")
        try:
            object.__setattr__(self, "role", AgentRole(self.role))
            object.__setattr__(self, "status", AgentStatus(self.status))
            if self.failure_category is not None:
                object.__setattr__(
                    self, "failure_category", AgentFailureCategory(self.failure_category)
                )
        except (TypeError, ValueError) as error:
            raise ValueError("result contains an invalid role, status, or failure category") from error
        if not isinstance(self.structured_output, Mapping):
            raise ValueError("structured_output must be a mapping")
        _validate_json_mapping(self.structured_output, "structured_output")
        if self.exit_code is not None and (
            not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool)
        ):
            raise ValueError("exit_code must be an integer or None")
        if (
            not isinstance(self.duration_seconds, (int, float))
            or isinstance(self.duration_seconds, bool)
            or not _is_finite_number(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise ValueError("duration_seconds must be a finite non-negative number")
        if self.status == AgentStatus.SUCCEEDED and self.failure_category is not None:
            raise ValueError("successful results cannot have a failure category")
        if self.status == AgentStatus.FAILED and self.failure_category is None:
            raise ValueError("failed results require a failure category")
        if self.status == AgentStatus.CANCELLED and self.failure_category != AgentFailureCategory.CANCELLATION:
            raise ValueError("cancelled results require the cancellation category")
        if self.diagnostic_code is not None and (
            not isinstance(self.diagnostic_code, str)
            or not _SAFE_CODE_PATTERN.fullmatch(self.diagnostic_code)
        ):
            raise ValueError("diagnostic_code must be a short safe identifier")


class AdapterError(RuntimeError):
    """Adapter failure carrying a stable category and safe diagnostic code."""

    def __init__(
        self,
        category: AgentFailureCategory,
        diagnostic_code: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.category = AgentFailureCategory(category)
        if not isinstance(diagnostic_code, str) or not _SAFE_CODE_PATTERN.fullmatch(
            diagnostic_code
        ):
            raise ValueError("diagnostic_code must be a short safe identifier")
        self.diagnostic_code = diagnostic_code


@runtime_checkable
class AgentAdapter(Protocol):
    """Synchronous provider-neutral adapter contract."""

    adapter_type: str

    def invoke(self, request: AgentRequest) -> AgentResult:
        """Handle one request and return a result correlated to the same run."""


class MockAgentAdapter:
    """Deterministic in-memory adapter; performs no I/O or external inference."""

    adapter_type = "mock"

    def __init__(
        self,
        *,
        status: AgentStatus = AgentStatus.SUCCEEDED,
        structured_output: Mapping[str, object] | None = None,
        failure_category: AgentFailureCategory | None = None,
        diagnostic_code: str | None = None,
    ) -> None:
        self.status = AgentStatus(status)
        self.structured_output = dict(structured_output or {})
        self.failure_category = failure_category
        self.diagnostic_code = diagnostic_code

    def invoke(self, request: AgentRequest) -> AgentResult:
        """Return the configured response without touching external resources."""
        category = self.failure_category
        if self.status == AgentStatus.FAILED and category is None:
            category = AgentFailureCategory.EXECUTION_FAILURE
        if self.status == AgentStatus.CANCELLED and category is None:
            category = AgentFailureCategory.CANCELLATION
        return AgentResult(
            run_id=request.run_id,
            role=request.role,
            status=self.status,
            structured_output=self.structured_output,
            duration_seconds=0.0,
            failure_category=category,
            diagnostic_code=self.diagnostic_code,
        )


class DisabledCodexAdapter:
    """Explicitly disabled placeholder; it cannot launch Codex or a process."""

    adapter_type = "codex-disabled"

    def invoke(self, request: AgentRequest) -> AgentResult:
        """Reject every request until a later phase defines an approved adapter."""
        raise AdapterError(
            AgentFailureCategory.POLICY_REJECTION,
            "codex_adapter_disabled",
            "Codex adapter is disabled in this phase",
        )


def invoke_adapter(adapter: AgentAdapter, request: AgentRequest) -> AgentResult:
    """Invoke an adapter and reject malformed or mismatched results safely."""
    adapter_type = getattr(adapter, "adapter_type", "")
    if not isinstance(adapter_type, str) or not _ADAPTER_TYPE_PATTERN.fullmatch(adapter_type):
        raise AdapterError(
            AgentFailureCategory.UNAVAILABLE_ADAPTER,
            "invalid_adapter_type",
            "adapter type is missing or invalid",
        )
    try:
        result = adapter.invoke(request)
    except AdapterError:
        raise
    except Exception as error:
        raise AdapterError(
            AgentFailureCategory.EXECUTION_FAILURE,
            "adapter_invocation_failed",
            f"adapter invocation failed ({type(error).__name__})",
        ) from error
    if not isinstance(result, AgentResult):
        raise AdapterError(
            AgentFailureCategory.INVALID_OUTPUT,
            "result_type_invalid",
            "adapter returned a value that is not an AgentResult",
        )
    if result.run_id != request.run_id or result.role != request.role:
        raise AdapterError(
            AgentFailureCategory.INVALID_OUTPUT,
            "result_correlation_mismatch",
            "adapter result correlation does not match its request",
        )
    return result


def safe_telemetry_attributes(
    request: AgentRequest,
    result: AgentResult,
    *,
    attempt: int,
    adapter_type: str,
) -> dict[str, str | int | float]:
    """Return an allowlisted metadata set; never include task or payload content."""
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ValueError("attempt must be a positive integer")
    if not isinstance(adapter_type, str) or not _ADAPTER_TYPE_PATTERN.fullmatch(adapter_type):
        raise ValueError("adapter_type must be a short safe identifier")
    if result.run_id != request.run_id or result.role != request.role:
        raise ValueError("result correlation does not match the request")
    attributes: dict[str, str | int | float] = {
        "workflow.run_id": request.run_id,
        "agent.role": request.role.value,
        "agent.attempt": attempt,
        "agent.adapter_type": adapter_type,
        "agent.status": result.status.value,
        "agent.duration_seconds": result.duration_seconds,
    }
    if result.failure_category is not None:
        attributes["agent.failure_category"] = result.failure_category.value
    return attributes


def _validate_json_mapping(value: Mapping[str, object], field_name: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must contain JSON-compatible values") from error
    _validate_nested_mappings(value, reject_process_configuration=field_name == "structured_context")


def _validate_nested_mappings(
    value: object, *, reject_process_configuration: bool
) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError("structured mappings must use string keys")
            normalized_key = "".join(character for character in key.lower() if character.isalnum())
            if reject_process_configuration and normalized_key in _FORBIDDEN_CONTEXT_KEYS:
                raise ValueError("structured_context must not contain process configuration")
            _validate_nested_mappings(
                nested,
                reject_process_configuration=reject_process_configuration,
            )
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_nested_mappings(
                nested,
                reject_process_configuration=reject_process_configuration,
            )


def _is_finite_number(value: int | float) -> bool:
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False
