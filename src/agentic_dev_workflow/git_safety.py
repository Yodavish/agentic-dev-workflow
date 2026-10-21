"""Bounded, read-only Git inspection and explicit baseline verification."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GIT_TIMEOUT_SECONDS = 10
MAX_GIT_OUTPUT_CHARS = 64 * 1024
MAX_REPORTED_PATHS = 200
_HEX_COMMIT = re.compile(r"^[0-9a-fA-F]+$")


@dataclass(frozen=True)
class GitCommandResult:
    """Bounded result of one Git invocation; output is never logged by this module."""

    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    error_type: str | None = None
    output_truncated: bool = False

    @property
    def succeeded(self) -> bool:
        return (
            self.returncode == 0
            and not self.timed_out
            and self.error_type is None
            and not self.output_truncated
        )


@dataclass(frozen=True)
class RepositoryInspection:
    """Summary of repository state, without file contents."""

    configured_path: str
    resolved_path: str | None
    is_git_repository: bool
    branch: str | None
    head_commit: str | None
    is_clean: bool
    changed_files: tuple[str, ...]
    untracked_files: tuple[str, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class BaselineVerification:
    """Structured result for one explicit source-repository base commit."""

    configured_path: str
    source_path: str | None
    git_root: str | None
    requested_commit: str
    resolved_commit: str | None
    object_format: str | None
    head_commit: str | None
    is_clean: bool
    staged_changes: tuple[str, ...]
    unstaged_changes: tuple[str, ...]
    untracked_files: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def success(self) -> bool:
        return not self.issues and self.resolved_commit is not None


def inspect_repository(
    repository: Path,
    *,
    expected_path: Path | None = None,
) -> RepositoryInspection:
    """Inspect Git metadata and status without mutating the repository."""
    configured = repository.expanduser()
    configured_path = _lexical_path(configured)
    try:
        resolved = configured.resolve(strict=True)
    except (OSError, RuntimeError):
        return RepositoryInspection(
            configured_path=configured_path,
            resolved_path=None,
            is_git_repository=False,
            branch=None,
            head_commit=None,
            is_clean=False,
            changed_files=(),
            untracked_files=(),
            issues=("configured repository path does not exist or cannot be resolved",),
        )

    issues: list[str] = []
    if not resolved.is_dir():
        issues.append("configured repository path is not a directory")
        return _inspection(configured_path, resolved, issues)
    git = shutil.which("git")
    if git is None:
        issues.append("Git executable is unavailable")
        return _inspection(configured_path, resolved, issues)

    root_result = _run_git(git, resolved, "rev-parse", "--show-toplevel")
    if not root_result.succeeded:
        issues.append(_command_issue("configured path is not a Git repository", root_result))
        return _inspection(configured_path, resolved, issues)
    repository_root = Path(root_result.stdout).resolve()
    intended_path = (expected_path or configured).expanduser().resolve()
    if repository_root != intended_path:
        issues.append("repository does not resolve to the configured repository path")

    branch_result = _run_git(git, repository_root, "branch", "--show-current")
    branch = branch_result.stdout if branch_result.succeeded else None
    if branch is None:
        issues.append(_command_issue("current branch could not be determined", branch_result))
    elif not branch:
        branch = "(detached HEAD)"

    head_result = _run_git(git, repository_root, "rev-parse", "HEAD")
    head_commit = head_result.stdout if head_result.succeeded else None
    if head_commit is None:
        issues.append(_command_issue("HEAD commit could not be determined", head_result))

    status_result = _run_git(
        git,
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--no-renames",
    )
    if not status_result.succeeded or status_result.output_truncated:
        issues.append(_command_issue("worktree status could not be determined", status_result))
        changed: tuple[str, ...] = ()
        untracked: tuple[str, ...] = ()
        clean = False
    else:
        rows = tuple(line for line in status_result.stdout.splitlines() if line)
        untracked = _bounded_paths(line[3:] for line in rows if line.startswith("?? "))
        changed = _bounded_paths(line[3:] for line in rows if not line.startswith("?? "))
        clean = not rows
        if changed:
            issues.append("worktree has tracked changes")
        if untracked:
            issues.append("worktree has untracked files")

    return RepositoryInspection(
        configured_path=configured_path,
        resolved_path=str(repository_root),
        is_git_repository=True,
        branch=branch,
        head_commit=head_commit,
        is_clean=clean,
        changed_files=changed,
        untracked_files=untracked,
        issues=tuple(issues),
    )


def verify_git_baseline(
    source_repository: str | Path,
    base_commit: str,
    *,
    require_clean: bool = True,
    timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
) -> BaselineVerification:
    """Verify a full explicit commit ID and source worktree without changing it."""
    configured = Path(source_repository).expanduser()
    configured_path = _lexical_path(configured)
    requested_commit = base_commit if isinstance(base_commit, str) else ""
    issues: list[str] = []
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not _is_finite_number(timeout_seconds)
        or timeout_seconds <= 0
    ):
        return _baseline_failure(configured_path, requested_commit, "Git timeout must be positive")

    try:
        source_path = configured.resolve(strict=True)
    except (OSError, RuntimeError):
        return _baseline_failure(
            configured_path,
            requested_commit,
            "source repository path does not exist or cannot be resolved",
        )
    if not source_path.is_dir():
        return _baseline_failure(configured_path, requested_commit, "source path is not a directory")
    git = shutil.which("git")
    if git is None:
        return _baseline_failure(configured_path, requested_commit, "Git executable is unavailable")

    top_result = _run_git(git, source_path, "rev-parse", "--show-toplevel", timeout=timeout_seconds)
    if not top_result.succeeded:
        return _baseline_failure(
            configured_path,
            requested_commit,
            _command_issue("source path is not a Git worktree", top_result),
            source_path=str(source_path),
        )
    git_root = Path(top_result.stdout).resolve()
    if git_root != source_path:
        issues.append("source path must be the Git worktree root")

    format_result = _run_git(
        git, git_root, "rev-parse", "--show-object-format", timeout=timeout_seconds
    )
    object_format = format_result.stdout if format_result.succeeded else None
    expected_length = {"sha1": 40, "sha256": 64}.get(object_format or "")
    if expected_length is None:
        issues.append(_command_issue("repository object format could not be determined", format_result))
    if not isinstance(base_commit, str) or expected_length is None or (
        len(base_commit) != expected_length or not _HEX_COMMIT.fullmatch(base_commit)
    ):
        if expected_length is not None:
            issues.append(
                f"base commit must be a full {expected_length}-character {object_format} identifier"
            )
        elif isinstance(base_commit, str) and not base_commit:
            issues.append("base commit must be explicitly selected")
        else:
            issues.append("base commit identifier is malformed")
        return _baseline_result(
            configured_path,
            str(source_path),
            str(git_root),
            requested_commit,
            None,
            object_format,
            None,
            False,
            (),
            (),
            (),
            issues,
        )

    commit_result = _run_git(
        git,
        git_root,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{base_commit}^{{commit}}",
        timeout=timeout_seconds,
    )
    resolved_commit = commit_result.stdout if commit_result.succeeded else None
    if resolved_commit is None:
        issues.append(_command_issue("selected base commit does not exist in this repository", commit_result))

    head_result = _run_git(git, git_root, "rev-parse", "HEAD", timeout=timeout_seconds)
    head_commit = head_result.stdout if head_result.succeeded else None
    if head_commit is None:
        issues.append(_command_issue("HEAD commit could not be determined", head_result))

    status_result = _run_git(
        git,
        git_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--no-renames",
        timeout=timeout_seconds,
    )
    if not status_result.succeeded or status_result.output_truncated:
        issues.append(_command_issue("worktree status could not be determined", status_result))
        is_clean = False
        staged: tuple[str, ...] = ()
        unstaged: tuple[str, ...] = ()
        untracked: tuple[str, ...] = ()
    else:
        rows = tuple(line for line in status_result.stdout.splitlines() if line)
        staged = _bounded_paths(
            line[3:] for line in rows if not line.startswith("?? ") and line[0] != " "
        )
        unstaged = _bounded_paths(
            line[3:] for line in rows if not line.startswith("?? ") and line[1] != " "
        )
        untracked = _bounded_paths(line[3:] for line in rows if line.startswith("?? "))
        is_clean = not rows
        if require_clean and not is_clean:
            issues.append("source worktree is dirty; a clean baseline is required")

    return _baseline_result(
        configured_path,
        str(source_path),
        str(git_root),
        requested_commit,
        resolved_commit,
        object_format,
        head_commit,
        is_clean,
        staged,
        unstaged,
        untracked,
        issues,
    )


def _baseline_failure(
    configured_path: str,
    requested_commit: str,
    issue: str,
    *,
    source_path: str | None = None,
) -> BaselineVerification:
    return _baseline_result(
        configured_path,
        source_path,
        None,
        requested_commit,
        None,
        None,
        None,
        False,
        (),
        (),
        (),
        [issue],
    )


def _baseline_result(
    configured_path: str,
    source_path: str | None,
    git_root: str | None,
    requested_commit: str,
    resolved_commit: str | None,
    object_format: str | None,
    head_commit: str | None,
    is_clean: bool,
    staged: tuple[str, ...],
    unstaged: tuple[str, ...],
    untracked: tuple[str, ...],
    issues: list[str],
) -> BaselineVerification:
    return BaselineVerification(
        configured_path=configured_path,
        source_path=source_path,
        git_root=git_root,
        requested_commit=requested_commit,
        resolved_commit=resolved_commit,
        object_format=object_format,
        head_commit=head_commit,
        is_clean=is_clean,
        staged_changes=staged,
        unstaged_changes=unstaged,
        untracked_files=untracked,
        issues=tuple(issues),
    )


def _inspection(
    configured_path: str, resolved: Path, issues: list[str]
) -> RepositoryInspection:
    return RepositoryInspection(
        configured_path=configured_path,
        resolved_path=str(resolved),
        is_git_repository=False,
        branch=None,
        head_commit=None,
        is_clean=False,
        changed_files=(),
        untracked_files=(),
        issues=tuple(issues),
    )


def _run_git(
    git: str,
    cwd: Path,
    *arguments: str,
    timeout: float = DEFAULT_GIT_TIMEOUT_SECONDS,
) -> GitCommandResult:
    """Run Git with explicit cwd, bounded timeout, and bounded captured text."""
    try:
        result = subprocess.run(
            [git, "-c", "core.fsmonitor=false", *arguments],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            env=_git_environment(),
        )
    except subprocess.TimeoutExpired:
        return GitCommandResult(None, "", "", timed_out=True)
    except OSError as error:
        return GitCommandResult(None, "", "", error_type=type(error).__name__)
    stdout, stdout_truncated = _truncate(result.stdout)
    stderr, stderr_truncated = _truncate(result.stderr)
    return GitCommandResult(
        result.returncode,
        stdout.rstrip("\n"),
        stderr.rstrip("\n"),
        output_truncated=stdout_truncated or stderr_truncated,
    )


def run_git_command(
    cwd: str | Path,
    *arguments: str,
    timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
) -> GitCommandResult:
    """Run a bounded Git command with an explicit working directory."""
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not _is_finite_number(timeout_seconds)
        or timeout_seconds <= 0
    ):
        return GitCommandResult(None, "", "", error_type="InvalidTimeout")
    git = shutil.which("git")
    if git is None:
        return GitCommandResult(None, "", "", error_type="GitUnavailable")
    return _run_git(
        git,
        Path(cwd),
        *arguments,
        timeout=timeout_seconds,
    )


def _truncate(value: str, limit: int = MAX_GIT_OUTPUT_CHARS) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[:limit], True


def _command_issue(context: str, result: GitCommandResult) -> str:
    if result.timed_out:
        return f"{context} (Git command timed out)"
    if result.error_type is not None:
        return f"{context} (Git command failed: {result.error_type})"
    if result.output_truncated:
        return f"{context} (Git output exceeded the safe limit)"
    return context


def _bounded_paths(values) -> tuple[str, ...]:
    return tuple(value[:1024] for value in list(values)[:MAX_REPORTED_PATHS])


def _lexical_path(path: Path) -> str:
    try:
        return str(path.resolve(strict=False))
    except (OSError, RuntimeError):
        return str(path)


def _is_finite_number(value: int | float) -> bool:
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _git_environment() -> dict[str, str]:
    """Prevent inherited repository overrides and optional index writes."""
    environment = os.environ.copy()
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_NAMESPACE",
        "GIT_PREFIX",
        "GIT_CEILING_DIRECTORIES",
        "GIT_CONFIG_PARAMETERS",
    ):
        environment.pop(name, None)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return environment
