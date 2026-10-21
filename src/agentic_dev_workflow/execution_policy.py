"""Pure declaration checks for a future isolated execution request.

This module validates strings and flags only. It does not inspect paths,
resolve symlinks, verify Git commits, or establish actual isolation.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

MAX_EXECUTION_TIMEOUT_SECONDS = 3600
MAX_POLICY_RETRIES = 2
_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class ExecutionPolicyError(ValueError):
    """Raised when a declared execution request violates policy."""


@dataclass(frozen=True)
class ExecutionPolicy:
    """Declared execution boundaries for a future run; not verified machine state."""

    run_id: str
    base_commit: str
    repository_path: str
    worktree_path: str
    clean_baseline_required: bool
    workspace_write_authorized: bool
    network_access: bool
    timeout_seconds: float
    max_retries: int
    auto_commit: bool
    auto_merge: bool
    deployment_allowed: bool
    production_database_access: bool
    destructive_cleanup_allowed: bool


def validate_execution_policy(policy: ExecutionPolicy) -> ExecutionPolicy:
    """Validate safety declarations without filesystem or Git access."""
    if not isinstance(policy, ExecutionPolicy):
        raise ExecutionPolicyError("policy must be an ExecutionPolicy")
    if not isinstance(policy.run_id, str) or not _RUN_ID_PATTERN.fullmatch(policy.run_id):
        raise ExecutionPolicyError("run_id is invalid")
    if not isinstance(policy.base_commit, str) or not _COMMIT_PATTERN.fullmatch(
        policy.base_commit
    ):
        raise ExecutionPolicyError("base_commit must be a full 40- or 64-character hexadecimal ID")
    if policy.clean_baseline_required is not True:
        raise ExecutionPolicyError("a clean baseline must be explicitly required")
    if policy.workspace_write_authorized is not True:
        raise ExecutionPolicyError("workspace-write permission must be explicitly authorized")
    if policy.network_access is not False:
        raise ExecutionPolicyError("network access must remain disabled")
    if (
        not isinstance(policy.timeout_seconds, (int, float))
        or isinstance(policy.timeout_seconds, bool)
        or not _is_finite_number(policy.timeout_seconds)
        or policy.timeout_seconds <= 0
        or policy.timeout_seconds > MAX_EXECUTION_TIMEOUT_SECONDS
    ):
        raise ExecutionPolicyError(
            f"timeout_seconds must be positive and at most {MAX_EXECUTION_TIMEOUT_SECONDS}"
        )
    if (
        not isinstance(policy.max_retries, int)
        or isinstance(policy.max_retries, bool)
        or not 0 <= policy.max_retries <= MAX_POLICY_RETRIES
    ):
        raise ExecutionPolicyError(f"max_retries must be an integer from 0 to {MAX_POLICY_RETRIES}")

    repository = _canonical_absolute_path(policy.repository_path, "repository_path")
    worktree = _canonical_absolute_path(policy.worktree_path, "worktree_path")
    if repository == worktree:
        raise ExecutionPolicyError("repository and worktree paths must be distinct")
    try:
        common = os.path.commonpath((repository, worktree))
    except ValueError as error:
        raise ExecutionPolicyError("repository and worktree paths must use the same path namespace") from error
    if common in {repository, worktree}:
        raise ExecutionPolicyError("repository and worktree declarations must not contain one another")
    expected_directory = f"agentic-workflow-{policy.run_id}"
    if os.path.basename(worktree) != expected_directory:
        raise ExecutionPolicyError(
            f"worktree_path must designate the specific run directory {expected_directory}"
        )

    forbidden = {
        "auto_commit": policy.auto_commit,
        "auto_merge": policy.auto_merge,
        "deployment_allowed": policy.deployment_allowed,
        "production_database_access": policy.production_database_access,
        "destructive_cleanup_allowed": policy.destructive_cleanup_allowed,
    }
    for name, enabled in forbidden.items():
        if enabled is not False:
            raise ExecutionPolicyError(f"{name} must remain disabled")
    return policy


def _canonical_absolute_path(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise ExecutionPolicyError(f"{field_name} must be an absolute path")
    normalized = os.path.normpath(value)
    if normalized != value.rstrip(os.sep) and not (
        value == os.sep and normalized == os.sep
    ):
        raise ExecutionPolicyError(f"{field_name} must be lexically normalized")
    return os.path.normcase(normalized)


def _is_finite_number(value: int | float) -> bool:
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False
