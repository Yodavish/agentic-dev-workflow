"""Read-only Git repository preflight checks."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


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


def inspect_repository(
    repository: Path,
    *,
    expected_path: Path | None = None,
) -> RepositoryInspection:
    """Inspect Git metadata and status without mutating the repository."""
    configured = repository.expanduser()
    try:
        resolved = configured.resolve(strict=True)
    except OSError:
        return RepositoryInspection(
            configured_path=str(configured.resolve()), resolved_path=None,
            is_git_repository=False, branch=None, head_commit=None,
            is_clean=False, changed_files=(), untracked_files=(),
            issues=("configured repository path does not exist",),
        )

    issues: list[str] = []
    if not resolved.is_dir():
        issues.append("configured repository path is not a directory")
    git = shutil.which("git")
    if git is None:
        issues.append("Git executable is unavailable")
        return _inspection(configured, resolved, issues)
    if not resolved.is_dir():
        return _inspection(configured, resolved, issues)

    root = _git(git, resolved, "rev-parse", "--show-toplevel")
    if root is None:
        issues.append("configured path is not a Git repository")
        return _inspection(configured, resolved, issues)
    repository_root = Path(root).resolve()
    intended_path = (expected_path or configured).expanduser().resolve()
    if repository_root != intended_path:
        issues.append("repository does not resolve to the configured repository path")

    branch = _git(git, repository_root, "branch", "--show-current")
    if branch is None:
        issues.append("current branch could not be determined")
    elif not branch:
        branch = "(detached HEAD)"
    head_commit = _git(git, repository_root, "rev-parse", "HEAD")
    if head_commit is None:
        issues.append("HEAD commit could not be determined")

    status = _git(git, repository_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status is None:
        issues.append("worktree status could not be determined")
        changed: tuple[str, ...] = ()
        untracked: tuple[str, ...] = ()
    else:
        rows = tuple(line for line in status.splitlines() if line)
        untracked = tuple(line[3:] for line in rows if line.startswith("?? "))
        changed = tuple(line[3:] for line in rows if not line.startswith("?? "))
        if changed:
            issues.append("worktree has tracked changes")
        if untracked:
            issues.append("worktree has untracked files")

    return RepositoryInspection(
        configured_path=str(configured.resolve()), resolved_path=str(repository_root),
        is_git_repository=True, branch=branch, head_commit=head_commit,
        is_clean=status == "", changed_files=changed, untracked_files=untracked,
        issues=tuple(issues),
    )


def _inspection(configured: Path, resolved: Path, issues: list[str]) -> RepositoryInspection:
    return RepositoryInspection(
        configured_path=str(configured.resolve()), resolved_path=str(resolved),
        is_git_repository=False, branch=None, head_commit=None, is_clean=False,
        changed_files=(), untracked_files=(), issues=tuple(issues),
    )


def _git(git: str, repository: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            [git, "-C", str(repository), *arguments], check=False,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.rstrip("\n") if result.returncode == 0 else None
