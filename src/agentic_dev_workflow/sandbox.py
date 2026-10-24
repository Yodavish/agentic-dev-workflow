"""Pure sandbox contracts; this module does not launch commands or models.

The only implementations here are fail-closed unavailability and a
deterministic in-memory fake. A future reviewed backend must enforce the
registered runtime profile and a real OS sandbox before running any argv.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Protocol

from agentic_dev_workflow.verification_registry import (
    RUNNER_RUNTIME_PROFILE_ID,
    CheckId,
    VerificationCheckDefinition,
    VerificationRegistryError,
    get_check_definition,
    validate_registered_check,
)
from agentic_dev_workflow.worktrees import WorktreeOperationResult

MAX_IN_MEMORY_EVIDENCE_BYTES = 4096
MAX_PROCESS_OUTPUT_CHUNK_BYTES = 16 * 1024
CLEANUP_GRACE_SECONDS = 10
MAX_RUN_ID_ATTEMPT = 3
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_BACKEND_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


class SandboxStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class SandboxFailureCategory(StrEnum):
    NONZERO_EXIT = "nonzero_exit"
    TIMEOUT = "timeout"
    OUTPUT_LIMIT = "output_limit"
    SANDBOX_UNAVAILABLE = "sandbox_unavailable"
    SANDBOX_SETUP_FAILURE = "sandbox_setup_failure"
    LAUNCH_FAILURE = "launch_failure"
    CLEANUP_FAILURE = "cleanup_failure"
    CANCELLED = "cancelled"
    POLICY_REJECTION = "policy_rejection"
    INVALID_RESULT = "invalid_result"


class SandboxDiagnosticCode(StrEnum):
    SANDBOX_NOT_CONFIGURED = "sandbox_not_configured"
    CHECK_NOT_REGISTERED = "check_not_registered"
    RUNTIME_LAUNCH_FAILED = "runtime_launch_failed"
    SANDBOX_SETUP_FAILED = "sandbox_setup_failed"
    CHECK_TIMED_OUT = "check_timed_out"
    OUTPUT_LIMIT_EXCEEDED = "output_limit_exceeded"
    CLEANUP_UNCONFIRMED = "cleanup_unconfirmed"
    CHECK_CANCELLED = "check_cancelled"
    RESULT_INVALID = "result_invalid"


class SandboxCleanupStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NOT_REQUIRED = "not_required"


class SandboxTermination(StrEnum):
    NATURAL_EXIT = "natural_exit"
    TIMEOUT = "timeout"
    OUTPUT_LIMIT = "output_limit"
    LAUNCH_FAILURE = "launch_failure"
    CANCELLED = "cancelled"
    NOT_STARTED = "not_started"


_FAILURE_DIAGNOSTICS = {
    SandboxFailureCategory.TIMEOUT: SandboxDiagnosticCode.CHECK_TIMED_OUT,
    SandboxFailureCategory.OUTPUT_LIMIT: SandboxDiagnosticCode.OUTPUT_LIMIT_EXCEEDED,
    SandboxFailureCategory.SANDBOX_UNAVAILABLE: SandboxDiagnosticCode.SANDBOX_NOT_CONFIGURED,
    SandboxFailureCategory.SANDBOX_SETUP_FAILURE: SandboxDiagnosticCode.SANDBOX_SETUP_FAILED,
    SandboxFailureCategory.LAUNCH_FAILURE: SandboxDiagnosticCode.RUNTIME_LAUNCH_FAILED,
    SandboxFailureCategory.CLEANUP_FAILURE: SandboxDiagnosticCode.CLEANUP_UNCONFIRMED,
    SandboxFailureCategory.CANCELLED: SandboxDiagnosticCode.CHECK_CANCELLED,
    SandboxFailureCategory.POLICY_REJECTION: SandboxDiagnosticCode.CHECK_NOT_REGISTERED,
    SandboxFailureCategory.INVALID_RESULT: SandboxDiagnosticCode.RESULT_INVALID,
}


class SandboxRunnerError(RuntimeError):
    """Fixed, safe error for invalid runner inputs or malformed results."""

    def __init__(self, diagnostic_code: SandboxDiagnosticCode) -> None:
        self.diagnostic_code = SandboxDiagnosticCode(diagnostic_code)
        super().__init__("sandbox runner rejected the request or returned an invalid result")


@dataclass(frozen=True, init=False)
class VerifiedWorktree:
    """Typed handle derived from a successful manager verification result.

    This is an in-process contract, not an unforgeable capability or a defense
    against another same-user process changing the filesystem after validation.
    A future backend must revalidate the manager's worktree immediately before
    mounting it.
    """

    run_id: str
    source_repository: str
    path: str
    base_commit: str
    _manager_result: WorktreeOperationResult = field(repr=False, compare=False)

    @classmethod
    def from_manager_verification(
        cls, result: WorktreeOperationResult
    ) -> VerifiedWorktree:
        if (
            type(result) is not WorktreeOperationResult
            or result.operation != "verify"
            or result.success is not True
            or result.partial_state is not False
            or result.issues != ()
            or not isinstance(result.run_id, str)
            or not _RUN_ID_PATTERN.fullmatch(result.run_id)
            or not _is_canonical_absolute(result.source_repository)
            or not _is_canonical_absolute(result.worktree_path)
            or result.source_repository == result.worktree_path
            or PurePosixPath(result.source_repository)
            in PurePosixPath(result.worktree_path).parents
            or PurePosixPath(result.worktree_path)
            in PurePosixPath(result.source_repository).parents
            or PurePosixPath(result.worktree_path).name
            != f"agentic-workflow-{result.run_id}"
            or not isinstance(result.base_commit, str)
            or not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", result.base_commit)
        ):
            raise SandboxRunnerError(SandboxDiagnosticCode.RESULT_INVALID)
        handle = object.__new__(cls)
        object.__setattr__(handle, "run_id", result.run_id)
        object.__setattr__(handle, "source_repository", result.source_repository)
        object.__setattr__(handle, "path", result.worktree_path)
        object.__setattr__(handle, "base_commit", result.base_commit)
        object.__setattr__(handle, "_manager_result", result)
        return handle


@dataclass(frozen=True)
class SandboxCheckResult:
    """Bounded result contract; evidence is in-memory only and hidden from repr.

    The applicable output cap is resolved from the code-owned verification
    registry by ``check_id``. A terminal ``passed`` or ``failed`` result can
    never report more output than that registered cap. Every other result
    state follows the output-limit category invariants: an over-cap
    observation and an output-limit failure must agree, so over-cap output is
    reported only as an output-limit failure, and an output-limit failure
    always reports an over-cap observation. The runner boundary repeats the
    same rules for results that arrive from a backend.
    """

    run_id: str
    check_id: CheckId
    attempt: int
    status: SandboxStatus
    exit_code: int | None
    duration_seconds: float
    output_bytes_observed: int
    termination: SandboxTermination = SandboxTermination.NATURAL_EXIT
    stdout_evidence: bytes = field(default=b"", repr=False)
    stderr_evidence: bytes = field(default=b"", repr=False)
    cleanup_status: SandboxCleanupStatus = SandboxCleanupStatus.NOT_REQUIRED
    failure_category: SandboxFailureCategory | None = None
    diagnostic_code: SandboxDiagnosticCode | None = None
    backend_type: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not _RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError("sandbox result is invalid")
        try:
            object.__setattr__(self, "check_id", CheckId(self.check_id))
            object.__setattr__(self, "status", SandboxStatus(self.status))
            object.__setattr__(self, "cleanup_status", SandboxCleanupStatus(self.cleanup_status))
            object.__setattr__(self, "termination", SandboxTermination(self.termination))
            if self.failure_category is not None:
                object.__setattr__(
                    self,
                    "failure_category",
                    SandboxFailureCategory(self.failure_category),
                )
            if self.diagnostic_code is not None:
                object.__setattr__(
                    self,
                    "diagnostic_code",
                    SandboxDiagnosticCode(self.diagnostic_code),
                )
        except (TypeError, ValueError) as error:
            raise ValueError("sandbox result is invalid") from error
        try:
            output_cap = get_check_definition(self.check_id).max_output_bytes
        except (VerificationRegistryError, KeyError) as error:
            raise ValueError("sandbox result is invalid") from error
        if (
            not isinstance(self.attempt, int)
            or isinstance(self.attempt, bool)
            or not 1 <= self.attempt <= MAX_RUN_ID_ATTEMPT
        ):
            raise ValueError("sandbox result is invalid")
        if self.exit_code is not None and (
            not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool)
        ):
            raise ValueError("sandbox result is invalid")
        if (
            not isinstance(self.duration_seconds, (int, float))
            or isinstance(self.duration_seconds, bool)
            or not _is_finite_number(self.duration_seconds)
            or self.duration_seconds < 0
            or self.duration_seconds > 3610
        ):
            raise ValueError("sandbox result is invalid")
        if (
            not isinstance(self.output_bytes_observed, int)
            or isinstance(self.output_bytes_observed, bool)
            or self.output_bytes_observed < 0
            or self.output_bytes_observed > output_cap + MAX_PROCESS_OUTPUT_CHUNK_BYTES
        ):
            raise ValueError("sandbox result is invalid")
        if type(self.stdout_evidence) is not bytes or type(self.stderr_evidence) is not bytes:
            raise ValueError("sandbox result is invalid")
        if len(self.stdout_evidence) + len(self.stderr_evidence) > MAX_IN_MEMORY_EVIDENCE_BYTES:
            raise ValueError("sandbox result is invalid")
        if not isinstance(self.backend_type, str) or not _BACKEND_TYPE_PATTERN.fullmatch(
            self.backend_type
        ):
            raise ValueError("sandbox result is invalid")
        self._validate_status_invariants(output_cap)

    def _validate_status_invariants(self, output_cap: int) -> None:
        # A terminal check status never reports an over-cap observation.
        if (
            self.status in (SandboxStatus.PASSED, SandboxStatus.FAILED)
            and self.output_bytes_observed > output_cap
        ):
            raise ValueError("sandbox result is invalid")
        # An over-cap observation and an output-limit failure must agree: only
        # an output-limit failure may report output above the registered cap,
        # and an output-limit failure must report it. The runner boundary
        # repeats this rule for results that arrive from a backend.
        if (self.output_bytes_observed > output_cap) != (
            self.failure_category == SandboxFailureCategory.OUTPUT_LIMIT
        ):
            raise ValueError("sandbox result is invalid")
        if self.status == SandboxStatus.PASSED:
            if (
                self.exit_code != 0
                or self.termination != SandboxTermination.NATURAL_EXIT
                or self.failure_category is not None
                or self.diagnostic_code is not None
                or self.cleanup_status != SandboxCleanupStatus.SUCCEEDED
            ):
                raise ValueError("sandbox result is invalid")
        elif self.status == SandboxStatus.FAILED:
            if (
                self.exit_code in (None, 0)
                or self.termination != SandboxTermination.NATURAL_EXIT
                or self.failure_category != SandboxFailureCategory.NONZERO_EXIT
                or self.diagnostic_code is not None
                or self.cleanup_status != SandboxCleanupStatus.SUCCEEDED
            ):
                raise ValueError("sandbox result is invalid")
        elif (
            self.failure_category is None
            or self.failure_category == SandboxFailureCategory.NONZERO_EXIT
            or self.diagnostic_code is None
        ):
            raise ValueError("sandbox result is invalid")
        if self.status == SandboxStatus.ERROR and self.diagnostic_code != _FAILURE_DIAGNOSTICS.get(
            self.failure_category
        ):
            raise ValueError("sandbox result is invalid")
        expected_termination = {
            SandboxFailureCategory.TIMEOUT: SandboxTermination.TIMEOUT,
            SandboxFailureCategory.OUTPUT_LIMIT: SandboxTermination.OUTPUT_LIMIT,
            SandboxFailureCategory.LAUNCH_FAILURE: SandboxTermination.LAUNCH_FAILURE,
            SandboxFailureCategory.CANCELLED: SandboxTermination.CANCELLED,
            SandboxFailureCategory.SANDBOX_UNAVAILABLE: SandboxTermination.NOT_STARTED,
            SandboxFailureCategory.SANDBOX_SETUP_FAILURE: SandboxTermination.NOT_STARTED,
            SandboxFailureCategory.POLICY_REJECTION: SandboxTermination.NOT_STARTED,
            SandboxFailureCategory.INVALID_RESULT: SandboxTermination.NOT_STARTED,
        }
        if (
            self.status == SandboxStatus.ERROR
            and self.failure_category in expected_termination
            and self.termination != expected_termination[self.failure_category]
        ):
            raise ValueError("sandbox result is invalid")
        if (
            self.failure_category == SandboxFailureCategory.CLEANUP_FAILURE
            and self.cleanup_status != SandboxCleanupStatus.FAILED
        ):
            raise ValueError("sandbox result is invalid")
        if (
            self.cleanup_status == SandboxCleanupStatus.FAILED
            and self.failure_category != SandboxFailureCategory.CLEANUP_FAILURE
        ):
            raise ValueError("sandbox result is invalid")


_SANDBOX_RESULT_FIELDS = tuple(item.name for item in fields(SandboxCheckResult))


def _revalidate_result(outcome: SandboxCheckResult) -> SandboxCheckResult:
    """Rebuild a backend result so normal construction invariants run again.

    A forged, deserialized, or otherwise partially initialized object can hold
    fields that never passed ``__post_init__``. Reconstructing from the declared
    field list makes the model the single validation authority at the boundary
    and returns a canonical object rather than the backend's instance.
    """
    try:
        return SandboxCheckResult(
            **{name: getattr(outcome, name) for name in _SANDBOX_RESULT_FIELDS}
        )
    except (AttributeError, TypeError, ValueError):
        raise SandboxRunnerError(SandboxDiagnosticCode.RESULT_INVALID) from None


class SandboxRunner(Protocol):
    """Injected runner contract; implementations must enforce an OS sandbox."""

    runtime_profile_id: str

    def run(
        self,
        check: VerificationCheckDefinition,
        worktree: VerifiedWorktree,
        *,
        run_id: str,
        attempt: int,
    ) -> SandboxCheckResult:
        """Run a registry-resolved check for a verified manager-owned worktree."""


class UnavailableSandboxRunner:
    """Fail closed until a separately reviewed OS sandbox backend exists."""

    runtime_profile_id = RUNNER_RUNTIME_PROFILE_ID

    def run(
        self,
        check: VerificationCheckDefinition,
        worktree: VerifiedWorktree,
        *,
        run_id: str,
        attempt: int,
    ) -> SandboxCheckResult:
        _validate_invocation(check, worktree, run_id, attempt, self.runtime_profile_id)
        return SandboxCheckResult(
            run_id=run_id,
            check_id=check.check_id,
            attempt=attempt,
            status=SandboxStatus.ERROR,
            exit_code=None,
            duration_seconds=0.0,
            output_bytes_observed=0,
            termination=SandboxTermination.NOT_STARTED,
            cleanup_status=SandboxCleanupStatus.NOT_REQUIRED,
            failure_category=SandboxFailureCategory.SANDBOX_UNAVAILABLE,
            diagnostic_code=SandboxDiagnosticCode.SANDBOX_NOT_CONFIGURED,
            backend_type="unavailable",
        )


class DeterministicFakeSandboxRunner:
    """In-memory test runner with predeclared outcomes and no external effects."""

    runtime_profile_id = RUNNER_RUNTIME_PROFILE_ID

    def __init__(self, outcomes: Mapping[CheckId, SandboxCheckResult]) -> None:
        if not isinstance(outcomes, Mapping):
            raise ValueError("fake outcomes are invalid")
        copied: dict[CheckId, SandboxCheckResult] = {}
        for check_id, outcome in outcomes.items():
            try:
                selected = CheckId(check_id)
            except (TypeError, ValueError) as error:
                raise ValueError("fake outcomes are invalid") from error
            if type(outcome) is not SandboxCheckResult or outcome.check_id != selected:
                raise ValueError("fake outcomes are invalid")
            copied[selected] = outcome
        self._outcomes = MappingProxyType(copied)

    def run(
        self,
        check: VerificationCheckDefinition,
        worktree: VerifiedWorktree,
        *,
        run_id: str,
        attempt: int,
    ) -> SandboxCheckResult:
        _validate_invocation(check, worktree, run_id, attempt, self.runtime_profile_id)
        outcome = self._outcomes.get(check.check_id)
        if outcome is None:
            raise SandboxRunnerError(SandboxDiagnosticCode.RESULT_INVALID)
        return replace(outcome, run_id=run_id, attempt=attempt)


def invoke_sandbox_runner(
    runner: SandboxRunner,
    check: VerificationCheckDefinition,
    worktree: VerifiedWorktree,
    *,
    run_id: str,
    attempt: int,
) -> SandboxCheckResult:
    """Validate both sides of the injected runner boundary with safe errors."""
    try:
        runtime_profile_id = getattr(runner, "runtime_profile_id", None)
    except Exception:
        raise SandboxRunnerError(SandboxDiagnosticCode.RESULT_INVALID) from None
    _validate_invocation(
        check, worktree, run_id, attempt, runtime_profile_id
    )
    try:
        outcome = runner.run(check, worktree, run_id=run_id, attempt=attempt)
    except SandboxRunnerError:
        raise
    except ValueError:
        # A backend that cannot build a result for this check reported an
        # invalid result; the model rejects over-cap output at construction.
        raise SandboxRunnerError(SandboxDiagnosticCode.RESULT_INVALID) from None
    except Exception:
        raise SandboxRunnerError(SandboxDiagnosticCode.RUNTIME_LAUNCH_FAILED) from None
    if type(outcome) is not SandboxCheckResult:
        raise SandboxRunnerError(SandboxDiagnosticCode.RESULT_INVALID)
    # Never trust the backend's instance: rebuild it through the model so a
    # forged or deserialized object cannot smuggle fields past __post_init__.
    canonical = _revalidate_result(outcome)
    _validate_result_for_check(canonical, check)
    if canonical.run_id != run_id or canonical.attempt != attempt:
        raise SandboxRunnerError(SandboxDiagnosticCode.RESULT_INVALID)
    return canonical


def safe_sandbox_telemetry_attributes(
    result: SandboxCheckResult,
) -> dict[str, str | int | float]:
    """Allowlist check metadata; evidence and diagnostics are omitted."""
    if type(result) is not SandboxCheckResult:
        raise ValueError("sandbox result is invalid")
    attributes: dict[str, str | int | float] = {
        "workflow.run_id": result.run_id,
        "verification.check_id": result.check_id.value,
        "verification.attempt": result.attempt,
        "verification.status": result.status.value,
        "verification.backend_type": result.backend_type,
        "verification.duration_seconds": result.duration_seconds,
        "verification.output_bytes_observed": result.output_bytes_observed,
    }
    if result.exit_code is not None:
        attributes["verification.exit_code"] = result.exit_code
    if result.failure_category is not None:
        attributes["verification.failure_category"] = result.failure_category.value
    return attributes


def _validate_invocation(
    check: VerificationCheckDefinition,
    worktree: VerifiedWorktree,
    run_id: str,
    attempt: int,
    runtime_profile_id: object,
) -> None:
    try:
        validate_registered_check(check)
    except VerificationRegistryError:
        raise SandboxRunnerError(SandboxDiagnosticCode.CHECK_NOT_REGISTERED) from None
    if (
        type(worktree) is not VerifiedWorktree
        or worktree.run_id != run_id
        or not isinstance(run_id, str)
        or not _RUN_ID_PATTERN.fullmatch(run_id)
        or not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or not 1 <= attempt <= MAX_RUN_ID_ATTEMPT
        or type(runtime_profile_id) is not str
        or runtime_profile_id != check.runtime_profile_id
    ):
        raise SandboxRunnerError(SandboxDiagnosticCode.RESULT_INVALID)


def _validate_result_for_check(
    result: SandboxCheckResult,
    check: VerificationCheckDefinition,
) -> None:
    if (
        result.check_id != check.check_id
        or result.duration_seconds > check.timeout_seconds + CLEANUP_GRACE_SECONDS
        or result.output_bytes_observed
        > check.max_output_bytes + MAX_PROCESS_OUTPUT_CHUNK_BYTES
    ):
        raise SandboxRunnerError(SandboxDiagnosticCode.RESULT_INVALID)
    overflow = result.output_bytes_observed > check.max_output_bytes
    if overflow != (result.failure_category == SandboxFailureCategory.OUTPUT_LIMIT):
        raise SandboxRunnerError(SandboxDiagnosticCode.RESULT_INVALID)
    if result.status in (SandboxStatus.PASSED, SandboxStatus.FAILED) and overflow:
        raise SandboxRunnerError(SandboxDiagnosticCode.RESULT_INVALID)


def _is_canonical_absolute(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = PurePosixPath(value)
    return path.is_absolute() and str(path) == value and ".." not in path.parts


def _is_finite_number(value: int | float) -> bool:
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False
