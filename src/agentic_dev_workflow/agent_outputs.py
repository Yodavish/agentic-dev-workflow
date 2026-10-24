"""Strict role-specific output contracts for untrusted agent responses.

Validated values remain untrusted and may contain sensitive content. Callers
must keep them out of telemetry and persist only their bounded ``to_state``
representations.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from agentic_dev_workflow.adapters import AgentFailureCategory

MAX_PLAN_OUTPUT_BYTES = 64 * 1024
MAX_EXECUTOR_OUTPUT_BYTES = 64 * 1024
MAX_VERIFIER_OUTPUT_BYTES = 16 * 1024
MAX_REVIEWER_OUTPUT_BYTES = 128 * 1024

MAX_PLAN_SUMMARY_CHARS = 2048
MAX_ASSUMPTIONS = 20
MAX_ASSUMPTION_CHARS = 512
MAX_PLAN_STEPS = 24
MAX_STEP_DESCRIPTION_CHARS = 512
MAX_PLAN_FILES = 64
MAX_CHANGED_FILES = 64
MAX_CHECK_IDS = 8
MAX_VERIFIER_SUMMARY_CHARS = 512
MAX_REVIEW_FINDINGS = 32
MAX_REVIEW_EVIDENCE_REFS = 8
MAX_CHECK_DURATION_MS = 600_000

# These IDs name runner-specific checks. Stage 1 validates references only;
# Stage 2 will define and execute the fixed commands behind this registry.
RUNNER_VERIFICATION_CHECK_IDS = frozenset(
    {
        "runner.tests",
        "runner.compile",
        "runner.diff-check",
        "runner.lock-check",
    }
)

_CHECK_ID_PATTERN = re.compile(r"^runner\.[a-z][a-z0-9-]{0,31}$")
_STEP_ID_PATTERN = re.compile(r"^step-[1-9][0-9]{0,2}$")
_FINDING_ID_PATTERN = re.compile(r"^finding-[1-9][0-9]{0,2}$")
_DIAGNOSTIC_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_EVIDENCE_REF_PATTERN = re.compile(r"^ev_[A-Za-z0-9_-]{1,60}$")
_FUNCTION_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:<>,\[\]-]{0,127}$")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
_WINDOWS_RESERVED_DEVICE_PATTERN = re.compile(
    r"^(?:con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])$", re.IGNORECASE
)
_WINDOWS_INVALID_PATH_CHARACTERS = frozenset('<>:"|?*')


class AgentOutputValidationError(ValueError):
    """Safe validation failure that never echoes model-produced values."""

    def __init__(self, diagnostic_code: str) -> None:
        if not _DIAGNOSTIC_PATTERN.fullmatch(diagnostic_code):
            diagnostic_code = "output_invalid"
        self.diagnostic_code = diagnostic_code
        super().__init__("agent output failed bounded schema validation")


class ExecutorOutcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class VerificationOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


class CheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class FindingSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FindingCategory(StrEnum):
    SPECIFICATION = "specification"
    CORRECTNESS = "correctness"
    SCOPE = "scope"
    SAFETY = "safety"
    MAINTAINABILITY = "maintainability"
    STYLE = "style"
    OTHER = "other"


class ReviewDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


@dataclass(frozen=True)
class PlanStep:
    """One validated implementation step and its expected relative paths."""

    step_id: str
    description: str
    expected_files: tuple[str, ...]

    def to_state(self) -> dict[str, object]:
        return {
            "id": self.step_id,
            "description": self.description,
            "expected_files": list(self.expected_files),
        }


@dataclass(frozen=True)
class PlannerOutput:
    """Bounded implementation plan; content remains untrusted and sensitive."""

    summary: str
    assumptions: tuple[str, ...]
    steps: tuple[PlanStep, ...]
    verification_check_ids: tuple[str, ...]

    def to_state(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "assumptions": list(self.assumptions),
            "steps": [step.to_state() for step in self.steps],
            "verification_check_ids": list(self.verification_check_ids),
        }


@dataclass(frozen=True)
class ExecutorOutput:
    """Bounded executor self-report; it is not verification evidence."""

    status: ExecutorOutcome
    summary: str
    files_changed: tuple[str, ...]
    checks_claimed: tuple[str, ...]
    failure_category: AgentFailureCategory | None
    diagnostic_code: str | None

    def to_state(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "summary": self.summary,
            "files_changed": list(self.files_changed),
            "checks_claimed": list(self.checks_claimed),
            "failure_category": (
                self.failure_category.value if self.failure_category is not None else None
            ),
            "diagnostic_code": self.diagnostic_code,
        }


@dataclass(frozen=True)
class VerificationCheck:
    """Runner-produced result for one registered check."""

    check_id: str
    status: CheckStatus
    exit_code: int | None
    duration_ms: int
    evidence_ref: str | None
    summary: str

    def to_state(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "evidence_ref": self.evidence_ref,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class VerifierOutput:
    """Bounded objective verification evidence produced by the runner."""

    outcome: VerificationOutcome
    checks: tuple[VerificationCheck, ...]

    def to_state(self) -> dict[str, object]:
        return {
            "outcome": self.outcome.value,
            "checks": [check.to_state() for check in self.checks],
        }


@dataclass(frozen=True)
class ReviewFinding:
    """Actionable finding with a repository-relative location and evidence."""

    finding_id: str
    severity: FindingSeverity
    category: FindingCategory
    blocks_approval: bool
    file: str
    function: str | None
    evidence_refs: tuple[str, ...]
    explanation: str
    remediation: str

    def to_state(self) -> dict[str, object]:
        return {
            "finding_id": self.finding_id,
            "severity": self.severity.value,
            "category": self.category.value,
            "blocks_approval": self.blocks_approval,
            "file": self.file,
            "function": self.function,
            "evidence_refs": list(self.evidence_refs),
            "explanation": self.explanation,
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class ReviewerOutput:
    """Independent reviewer decision; schema validity does not establish truth."""

    decision: ReviewDecision
    findings: tuple[ReviewFinding, ...]

    def to_state(self) -> dict[str, object]:
        return {
            "decision": self.decision.value,
            "findings": [finding.to_state() for finding in self.findings],
        }


def validate_planner_output(value: object) -> PlannerOutput:
    """Validate a planner mapping and reject unknown or oversized fields."""
    fields = _object_fields(
        value, {"summary", "assumptions", "steps", "verification_check_ids"}
    )
    summary = _string(fields["summary"], MAX_PLAN_SUMMARY_CHARS)
    assumptions = _string_list(
        fields["assumptions"], MAX_ASSUMPTIONS, MAX_ASSUMPTION_CHARS
    )
    raw_steps = _list(fields["steps"], 1, MAX_PLAN_STEPS)
    steps: list[PlanStep] = []
    step_ids: set[str] = set()
    all_file_keys: set[str] = set()
    for raw_step in raw_steps:
        step = _object_fields(raw_step, {"id", "description", "expected_files"})
        step_id = _string(step["id"], 32)
        if not _STEP_ID_PATTERN.fullmatch(step_id) or step_id in step_ids:
            _invalid("output_field_invalid")
        step_ids.add(step_id)
        description = _string(step["description"], MAX_STEP_DESCRIPTION_CHARS)
        paths = _path_list(step["expected_files"], 16)
        path_keys = {_portable_path_collision_key(path) for path in paths}
        combined_path_keys = all_file_keys | path_keys
        if all_file_keys.intersection(path_keys) or _has_file_directory_prefix_conflict(
            combined_path_keys
        ):
            _invalid("output_path_invalid")
        all_file_keys.update(path_keys)
        if len(all_file_keys) > MAX_PLAN_FILES:
            _invalid("output_limit_exceeded")
        steps.append(PlanStep(step_id, description, tuple(paths)))
    check_ids = _check_id_list(fields["verification_check_ids"], allow_empty=False)
    output = PlannerOutput(summary, tuple(assumptions), tuple(steps), tuple(check_ids))
    _enforce_serialized_size(output.to_state(), MAX_PLAN_OUTPUT_BYTES)
    return output


def validate_executor_output(value: object) -> ExecutorOutput:
    """Validate the executor's bounded self-report without trusting its claims."""
    fields = _object_fields(
        value,
        {
            "status",
            "summary",
            "files_changed",
            "checks_claimed",
            "failure_category",
            "diagnostic_code",
        },
    )
    status = _enum(ExecutorOutcome, fields["status"])
    summary = _string(fields["summary"], 2048)
    files = _path_list(fields["files_changed"], MAX_CHANGED_FILES, allow_empty=True)
    checks = _check_id_list(fields["checks_claimed"], allow_empty=True)
    category_value = fields["failure_category"]
    if category_value is None:
        category = None
    else:
        category = _enum(AgentFailureCategory, category_value)
    diagnostic = fields["diagnostic_code"]
    if diagnostic is not None:
        diagnostic = _string(diagnostic, 64)
        if not _DIAGNOSTIC_PATTERN.fullmatch(diagnostic):
            _invalid("output_field_invalid")
    if status == ExecutorOutcome.COMPLETED and category is not None:
        _invalid("output_inconsistent")
    if status != ExecutorOutcome.COMPLETED and category is None:
        _invalid("output_inconsistent")
    output = ExecutorOutput(status, summary, tuple(files), tuple(checks), category, diagnostic)
    _enforce_serialized_size(output.to_state(), MAX_EXECUTOR_OUTPUT_BYTES)
    return output


def validate_verifier_output(value: object) -> VerifierOutput:
    """Validate runner-produced check results and their outcome consistency."""
    fields = _object_fields(value, {"outcome", "checks"})
    outcome = _enum(VerificationOutcome, fields["outcome"])
    raw_checks = _list(fields["checks"], 1, MAX_CHECK_IDS)
    checks: list[VerificationCheck] = []
    check_ids: set[str] = set()
    for raw_check in raw_checks:
        check = _object_fields(
            raw_check,
            {"check_id", "status", "exit_code", "duration_ms", "evidence_ref", "summary"},
        )
        check_id = _registered_check_id(check["check_id"])
        if check_id in check_ids:
            _invalid("output_field_invalid")
        check_ids.add(check_id)
        status = _enum(CheckStatus, check["status"])
        exit_code = check["exit_code"]
        if exit_code is not None and type(exit_code) is not int:
            _invalid("output_field_invalid")
        duration = check["duration_ms"]
        if type(duration) is not int or not 0 <= duration <= MAX_CHECK_DURATION_MS:
            _invalid("output_limit_exceeded")
        evidence = check["evidence_ref"]
        if status == CheckStatus.SKIPPED:
            if evidence is not None or exit_code is not None:
                _invalid("output_inconsistent")
        else:
            evidence = _string(evidence, 64)
            if not _EVIDENCE_REF_PATTERN.fullmatch(evidence):
                _invalid("output_field_invalid")
            if status == CheckStatus.PASSED and exit_code != 0:
                _invalid("output_inconsistent")
            if status == CheckStatus.FAILED and (
                exit_code is None or exit_code == 0
            ):
                _invalid("output_inconsistent")
            if status == CheckStatus.ERROR and exit_code is not None:
                _invalid("output_inconsistent")
        summary = _string(check["summary"], MAX_VERIFIER_SUMMARY_CHARS)
        checks.append(
            VerificationCheck(check_id, status, exit_code, duration, evidence, summary)
        )
    statuses = {check.status for check in checks}
    # A check infrastructure error takes precedence over ordinary failures:
    # callers must not treat a partially unverified run as repairable failure.
    if CheckStatus.ERROR in statuses and outcome != VerificationOutcome.ERROR:
        _invalid("output_inconsistent")
    if outcome == VerificationOutcome.PASS and statuses != {CheckStatus.PASSED}:
        _invalid("output_inconsistent")
    if outcome == VerificationOutcome.FAIL and not statuses.intersection(
        {CheckStatus.FAILED, CheckStatus.SKIPPED}
    ):
        _invalid("output_inconsistent")
    if outcome == VerificationOutcome.ERROR and CheckStatus.ERROR not in statuses:
        _invalid("output_inconsistent")
    result = VerifierOutput(outcome, tuple(checks))
    _enforce_serialized_size(result.to_state(), MAX_VERIFIER_OUTPUT_BYTES)
    return result


def validate_reviewer_output(value: object) -> ReviewerOutput:
    """Validate bounded reviewer findings; valid shape does not prove correctness."""
    fields = _object_fields(value, {"decision", "findings"})
    decision = _enum(ReviewDecision, fields["decision"])
    raw_findings = _list(fields["findings"], 0, MAX_REVIEW_FINDINGS)
    findings: list[ReviewFinding] = []
    finding_ids: set[str] = set()
    for raw_finding in raw_findings:
        finding = _object_fields(
            raw_finding,
            {
                "finding_id",
                "severity",
                "category",
                "blocks_approval",
                "file",
                "function",
                "evidence_refs",
                "explanation",
                "remediation",
            },
        )
        finding_id = _string(finding["finding_id"], 32)
        if not _FINDING_ID_PATTERN.fullmatch(finding_id) or finding_id in finding_ids:
            _invalid("output_field_invalid")
        finding_ids.add(finding_id)
        severity = _enum(FindingSeverity, finding["severity"])
        category = _enum(FindingCategory, finding["category"])
        blocks = finding["blocks_approval"]
        if type(blocks) is not bool:
            _invalid("output_field_invalid")
        if severity in {FindingSeverity.CRITICAL, FindingSeverity.HIGH} and not blocks:
            _invalid("output_inconsistent")
        if severity == FindingSeverity.LOW and blocks:
            _invalid("output_inconsistent")
        if severity == FindingSeverity.MEDIUM and category in {
            FindingCategory.SPECIFICATION,
            FindingCategory.CORRECTNESS,
            FindingCategory.SCOPE,
            FindingCategory.SAFETY,
        } and not blocks:
            _invalid("output_inconsistent")
        path = _relative_repository_path(finding["file"])
        function = finding["function"]
        if function is not None:
            function = _string(function, 128)
            if not _FUNCTION_PATTERN.fullmatch(function):
                _invalid("output_field_invalid")
        refs = _evidence_ref_list(finding["evidence_refs"])
        explanation = _string(finding["explanation"], 2048)
        remediation = _string(finding["remediation"], 2048)
        findings.append(
            ReviewFinding(
                finding_id,
                severity,
                category,
                blocks,
                path,
                function,
                tuple(refs),
                explanation,
                remediation,
            )
        )
    has_blocking = any(finding.blocks_approval for finding in findings)
    if (decision == ReviewDecision.APPROVE and has_blocking) or (
        decision == ReviewDecision.REJECT and not has_blocking
    ):
        _invalid("output_inconsistent")
    output = ReviewerOutput(decision, tuple(findings))
    _enforce_serialized_size(output.to_state(), MAX_REVIEWER_OUTPUT_BYTES)
    return output


def _object_fields(value: object, expected: set[str]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _invalid("output_type_invalid")
    fields: dict[str, object] = {}
    try:
        for key in value:
            if type(key) is not str or key not in expected or len(fields) >= len(expected):
                _invalid("output_fields_invalid")
            fields[key] = value[key]
    except AgentOutputValidationError:
        raise
    except Exception:
        _invalid("output_type_invalid")
    if fields.keys() != expected:
        _invalid("output_fields_invalid")
    return fields


def _string(value: object, maximum: int) -> str:
    if type(value) is not str or not value or not value.strip() or len(value) > maximum:
        _invalid("output_limit_exceeded" if isinstance(value, str) else "output_field_invalid")
    if any(unicodedata.category(character) == "Cc" for character in value):
        _invalid("output_field_invalid")
    return value


def _list(value: object, minimum: int, maximum: int) -> list[object]:
    if type(value) is not list:
        _invalid("output_field_invalid")
    if not minimum <= len(value) <= maximum:
        _invalid("output_limit_exceeded")
    return value


def _string_list(
    value: object,
    maximum_count: int,
    maximum_length: int,
    *,
    allow_empty: bool = True,
) -> list[str]:
    values = _list(value, 0 if allow_empty else 1, maximum_count)
    return [_string(item, maximum_length) for item in values]


def _path_list(
    value: object,
    maximum_count: int,
    *,
    allow_empty: bool = False,
) -> list[str]:
    values = _list(value, 0 if allow_empty else 1, maximum_count)
    paths = [_relative_repository_path(path) for path in values]
    collision_keys = {_portable_path_collision_key(path) for path in paths}
    if len(collision_keys) != len(paths):
        _invalid("output_path_invalid")
    return paths


def _portable_path_collision_key(path: str) -> str:
    """Return a comparison-only key for portable path collision checks.

    NFKC case folding treats case-insensitive and compatibility-normalized
    spellings as equivalent. It is used only to detect collisions; accepted
    paths retain their original spelling and are never rewritten.
    """
    compatibility_normalized = unicodedata.normalize("NFKC", path)
    return unicodedata.normalize("NFKC", compatibility_normalized.casefold())


def _has_file_directory_prefix_conflict(path_keys: set[str]) -> bool:
    """Return whether two declared files occupy an ancestor and descendant path."""
    paths = tuple(path_keys)
    return any(
        first.startswith(second + "/") or second.startswith(first + "/")
        for index, first in enumerate(paths)
        for second in paths[index + 1 :]
    )


def _relative_repository_path(value: object) -> str:
    """Accept only normalized, portable repository-relative POSIX paths.

    Ambiguous Windows aliases and reserved device components are rejected so
    path scope comparisons have deterministic meaning across platforms. Unicode
    control, formatting, private-use, unassigned, surrogate, and line or
    paragraph separator characters are rejected because they can disrupt path
    display or interpretation. Non-ASCII space separators are also rejected;
    ordinary internal ASCII spaces remain allowed.
    """
    path = _string(value, 512)
    if (
        path.startswith("/")
        or "\\" in path
        or _WINDOWS_DRIVE_PATTERN.match(path)
        or path.endswith("/")
    ):
        _invalid("output_path_invalid")
    parts = path.split("/")
    if any(part in {"", ".", ".."} or part.casefold() == ".git" for part in parts):
        _invalid("output_path_invalid")
    if unicodedata.normalize("NFC", path) != path:
        _invalid("output_path_invalid")
    if any(
        unicodedata.category(character).startswith("C")
        or unicodedata.category(character) in {"Zl", "Zp"}
        or (unicodedata.category(character) == "Zs" and character != " ")
        for character in path
    ):
        _invalid("output_path_invalid")
    for part in parts:
        if (
            len(part) > 255
            or part.endswith((".", " "))
            or any(character in _WINDOWS_INVALID_PATH_CHARACTERS for character in part)
        ):
            _invalid("output_path_invalid")
        device_name = part.split(".", 1)[0].rstrip(" .")
        if _WINDOWS_RESERVED_DEVICE_PATTERN.fullmatch(device_name):
            _invalid("output_path_invalid")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or parsed.as_posix() != path:
        _invalid("output_path_invalid")
    return path


def validate_relative_repository_path(value: object) -> str:
    """Public entry point for the Stage 1 repository-relative path contract."""
    return _relative_repository_path(value)


def portable_path_collision_key(path: str) -> str:
    """Public entry point for portable path collision detection."""
    return _portable_path_collision_key(path)


def has_file_directory_prefix_conflict(path_keys: set[str]) -> bool:
    """Public entry point for declared file/ancestor conflict detection."""
    return _has_file_directory_prefix_conflict(path_keys)


def _check_id_list(value: object, *, allow_empty: bool) -> list[str]:
    values = _string_list(value, MAX_CHECK_IDS, 48, allow_empty=allow_empty)
    check_ids = [_registered_check_id(check_id) for check_id in values]
    if len(set(check_ids)) != len(check_ids):
        _invalid("output_field_invalid")
    return check_ids


def _registered_check_id(value: object) -> str:
    check_id = _string(value, 48)
    if (
        not _CHECK_ID_PATTERN.fullmatch(check_id)
        or check_id not in RUNNER_VERIFICATION_CHECK_IDS
    ):
        _invalid("output_check_id_invalid")
    return check_id


def _evidence_ref_list(value: object) -> list[str]:
    refs = _string_list(value, MAX_REVIEW_EVIDENCE_REFS, 64, allow_empty=False)
    if any(not _EVIDENCE_REF_PATTERN.fullmatch(ref) for ref in refs):
        _invalid("output_field_invalid")
    if len(set(refs)) != len(refs):
        _invalid("output_field_invalid")
    return refs


def _enum(enum_type: type[StrEnum], value: object):
    if type(value) is not str:
        _invalid("output_field_invalid")
    try:
        return enum_type(value)
    except ValueError:
        _invalid("output_field_invalid")


def _enforce_serialized_size(value: dict[str, object], maximum: int) -> None:
    try:
        size = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    except (TypeError, ValueError, UnicodeError):
        _invalid("output_type_invalid")
    if size > maximum:
        _invalid("output_limit_exceeded")


def _invalid(diagnostic_code: str) -> None:
    raise AgentOutputValidationError(diagnostic_code)
