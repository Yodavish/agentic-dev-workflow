"""Explicit manager-owned Git worktree creation and conservative cleanup."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_dev_workflow.git_safety import (
    DEFAULT_GIT_TIMEOUT_SECONDS,
    GitCommandResult,
    run_git_command,
    verify_git_baseline,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_ROOT_MARKER = ".agentic-dev-workflow-root.json"
_ROOT_OWNER = {"format": 1, "manager": "agentic-dev-workflow"}
_PROTECTED_ROOTS = tuple(
    Path(path)
    for path in (
        "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/var",
        "/proc", "/sys", "/dev", "/boot", "/root", "/opt",
    )
)


@dataclass(frozen=True)
class WorktreeOperationResult:
    """Outcome of one explicit worktree lifecycle operation."""

    operation: str
    success: bool
    run_id: str
    source_repository: str
    worktree_path: str
    base_commit: str
    partial_state: bool
    issues: tuple[str, ...]


class WorktreeManager:
    """Manage run-specific worktrees below an explicitly selected owned root."""

    def __init__(
        self,
        worktree_root: str | Path,
        *,
        timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
    ) -> None:
        self.configured_root = Path(worktree_root).expanduser()
        self.timeout_seconds = timeout_seconds

    def create_worktree(
        self,
        source_repository: str | Path,
        base_commit: str,
        *,
        run_id: str,
    ) -> WorktreeOperationResult:
        """Verify a clean base then create and verify a detached run worktree."""
        if not self._valid_run_id(run_id):
            return self._failure("create", source_repository, run_id, base_commit, "run_id is invalid")
        source_value = Path(source_repository).expanduser()
        baseline = verify_git_baseline(
            source_value,
            base_commit,
            require_clean=True,
            timeout_seconds=self.timeout_seconds,
        )
        if not baseline.success:
            return self._result(
                "create",
                False,
                run_id,
                baseline.source_path or str(source_value),
                "",
                base_commit,
                False,
                baseline.issues,
            )
        source = Path(baseline.source_path)
        root, root_issue = self._prepare_owned_root(source)
        destination = root / f"agentic-workflow-{run_id}" if root else Path()
        if root_issue:
            return self._result(
                "create", False, run_id, str(source), str(destination), base_commit, False, (root_issue,)
            )
        assert root is not None
        if self._is_protected_destination(destination, source):
            return self._result(
                "create",
                False,
                run_id,
                str(source),
                str(destination),
                base_commit,
                False,
                ("generated worktree destination is protected or overlaps the source repository",),
            )
        if os.path.lexists(destination):
            return self._result(
                "create",
                False,
                run_id,
                str(source),
                str(destination),
                base_commit,
                False,
                ("worktree destination already exists; it will not be reused or overwritten",),
            )

        manifest = self._manifest_path(root, run_id)
        manifest_data = {
            "format": 1,
            "manager": "agentic-dev-workflow",
            "run_id": run_id,
            "source_repository": str(source),
            "worktree_path": str(destination),
            "base_commit": baseline.resolved_commit,
        }
        try:
            self._write_exclusive_json(manifest, manifest_data)
        except OSError as error:
            return self._result(
                "create", False, run_id, str(source), str(destination), base_commit, False,
                (f"could not reserve manager ownership record ({type(error).__name__})",),
            )

        git = shutil.which("git")
        if git is None:
            self._remove_manifest_if_no_partial_state(manifest, destination, source)
            return self._result(
                "create", False, run_id, str(source), str(destination), base_commit, False,
                ("Git executable is unavailable",),
            )
        result = self._run_git(
            source,
            "-c",
            "core.hooksPath=/dev/null",
            "worktree",
            "add",
            "--detach",
            str(destination),
            baseline.resolved_commit or "",
        )
        if not result.succeeded:
            partial = self._has_partial_state(destination, source)
            if not partial:
                self._remove_manifest_if_no_partial_state(manifest, destination, source)
            return self._result(
                "create",
                False,
                run_id,
                str(source),
                str(destination),
                baseline.resolved_commit or base_commit,
                partial,
                (self._git_failure("git worktree add failed", result),),
            )

        verify_issues = self._verify_worktree(source, destination, baseline.resolved_commit or "")
        source_after = verify_git_baseline(
            source,
            baseline.resolved_commit or "",
            require_clean=True,
            timeout_seconds=self.timeout_seconds,
        )
        if not source_after.success or source_after.head_commit != baseline.head_commit:
            verify_issues.extend(source_after.issues or ("source HEAD changed during worktree creation",))
        if verify_issues:
            return self._result(
                "create", False, run_id, str(source), str(destination),
                baseline.resolved_commit or base_commit, True, tuple(verify_issues),
            )
        return self._result(
            "create", True, run_id, str(source), str(destination),
            baseline.resolved_commit or base_commit, False, (),
        )

    def cleanup_worktree(
        self,
        source_repository: str | Path,
        base_commit: str,
        *,
        run_id: str,
    ) -> WorktreeOperationResult:
        """Remove only a clean worktree with matching manager and Git ownership."""
        if not self._valid_run_id(run_id):
            return self._failure("cleanup", source_repository, run_id, base_commit, "run_id is invalid")
        baseline = verify_git_baseline(
            source_repository,
            base_commit,
            require_clean=True,
            timeout_seconds=self.timeout_seconds,
        )
        if not baseline.success:
            return self._result(
                "cleanup", False, run_id, baseline.source_path or str(source_repository), "",
                base_commit, False, baseline.issues,
            )
        source = Path(baseline.source_path)
        root, root_issue = self._existing_owned_root(source)
        destination = root / f"agentic-workflow-{run_id}" if root else Path()
        if root_issue:
            return self._result(
                "cleanup", False, run_id, str(source), str(destination), base_commit, False,
                (root_issue,),
            )
        assert root is not None
        manifest = self._manifest_path(root, run_id)
        owner, owner_issue = self._read_manifest(manifest)
        if owner_issue:
            return self._result(
                "cleanup", False, run_id, str(source), str(destination), base_commit,
                os.path.lexists(destination), (owner_issue,),
            )
        expected = {
            "run_id": run_id,
            "source_repository": str(source),
            "worktree_path": str(destination),
            "base_commit": baseline.resolved_commit,
        }
        if any(owner.get(key) != value for key, value in expected.items()):
            return self._result(
                "cleanup", False, run_id, str(source), str(destination), base_commit, True,
                ("manager ownership record does not match the requested run and source",),
            )

        verify_issues = self._verify_worktree(source, destination, baseline.resolved_commit or "")
        if verify_issues:
            return self._result(
                "cleanup", False, run_id, str(source), str(destination), base_commit, True,
                tuple(verify_issues),
            )
        if not self._worktree_is_clean(destination):
            return self._result(
                "cleanup", False, run_id, str(source), str(destination), base_commit, True,
                ("worktree has changes; cleanup will not discard them",),
            )

        result = self._run_git(source, "worktree", "remove", str(destination))
        if not result.succeeded:
            return self._result(
                "cleanup", False, run_id, str(source), str(destination), base_commit, True,
                (self._git_failure("git worktree remove failed", result),),
            )
        if os.path.lexists(destination):
            return self._result(
                "cleanup", False, run_id, str(source), str(destination), base_commit, True,
                ("Git reported success but the worktree destination remains",),
            )
        try:
            manifest.unlink()
        except OSError as error:
            return self._result(
                "cleanup", False, run_id, str(source), str(destination), base_commit, True,
                (f"worktree was removed but ownership record remains ({type(error).__name__})",),
            )
        return self._result(
            "cleanup", True, run_id, str(source), str(destination), base_commit, False, ()
        )

    def verify_managed_worktree(
        self,
        source_repository: str | Path,
        base_commit: str,
        *,
        run_id: str,
        require_clean: bool = True,
    ) -> WorktreeOperationResult:
        """Verify ownership and registration, optionally requiring a clean worktree."""
        if not self._valid_run_id(run_id):
            return self._failure(
                "verify", source_repository, run_id, base_commit, "run_id is invalid"
            )
        baseline = verify_git_baseline(
            source_repository,
            base_commit,
            require_clean=True,
            timeout_seconds=self.timeout_seconds,
        )
        if not baseline.success:
            return self._result(
                "verify", False, run_id, baseline.source_path or str(source_repository), "",
                base_commit, False, baseline.issues,
            )
        source = Path(baseline.source_path)
        root, root_issue = self._existing_owned_root(source)
        destination = root / f"agentic-workflow-{run_id}" if root else Path()
        if root_issue:
            return self._result(
                "verify", False, run_id, str(source), str(destination), base_commit,
                os.path.lexists(destination), (root_issue,),
            )
        assert root is not None
        owner, owner_issue = self._read_manifest(self._manifest_path(root, run_id))
        if owner_issue:
            return self._result(
                "verify", False, run_id, str(source), str(destination), base_commit,
                os.path.lexists(destination), (owner_issue,),
            )
        expected = {
            "run_id": run_id,
            "source_repository": str(source),
            "worktree_path": str(destination),
            "base_commit": baseline.resolved_commit,
        }
        if any(owner.get(key) != value for key, value in expected.items()):
            return self._result(
                "verify", False, run_id, str(source), str(destination), base_commit,
                True, ("manager ownership record does not match the requested run and source",),
            )
        issues = self._verify_worktree(
            source,
            destination,
            baseline.resolved_commit or "",
            require_clean=require_clean,
        )
        return self._result(
            "verify", not issues, run_id, str(source), str(destination),
            baseline.resolved_commit or base_commit, bool(issues), tuple(issues),
        )

    def _prepare_owned_root(self, source: Path) -> tuple[Path | None, str | None]:
        root, issue = self._canonical_root(source)
        if issue:
            return None, issue
        assert root is not None
        if not root.exists():
            parent = root.parent
            if not parent.is_dir():
                return None, "worktree root parent must already exist"
            try:
                root.mkdir(mode=0o700)
                self._write_exclusive_json(root / _ROOT_MARKER, _ROOT_OWNER)
            except OSError as error:
                return None, f"could not initialize runner-owned worktree root ({type(error).__name__})"
            return root.resolve(strict=True), None
        return self._existing_owned_root(source)

    def _existing_owned_root(self, source: Path) -> tuple[Path | None, str | None]:
        root, issue = self._canonical_root(source)
        if issue:
            return None, issue
        assert root is not None
        if root.is_symlink() or not root.is_dir():
            return None, "worktree root must be a real directory, not a symlink"
        marker = root / _ROOT_MARKER
        try:
            if marker.is_symlink() or not marker.is_file():
                return None, "existing worktree root has no valid runner ownership marker"
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None, "existing worktree root ownership marker is unreadable"
        if data != _ROOT_OWNER:
            return None, "existing worktree root is not owned by this runner"
        return root.resolve(strict=True), None

    def _canonical_root(self, source: Path) -> tuple[Path | None, str | None]:
        configured = self.configured_root
        if not configured.is_absolute():
            return None, "worktree root must be an explicit absolute path"
        if os.path.normpath(str(configured)) != str(configured).rstrip(os.sep):
            return None, "worktree root must be lexically normalized"
        try:
            parent = configured.parent.resolve(strict=True)
            root = parent / configured.name
        except (OSError, RuntimeError):
            return None, "worktree root parent does not exist or cannot be resolved"
        runner_root = Path(__file__).resolve().parents[2]
        if self._paths_overlap(root, source):
            return None, "worktree root must not equal, contain, or be inside the source repository"
        if self._paths_overlap(root, runner_root):
            return None, "worktree root overlaps the runner repository"
        if self._is_protected_path(root):
            return None, "worktree root is a protected system path"
        return root, None

    def _is_protected_destination(self, destination: Path, source: Path) -> bool:
        runner_root = Path(__file__).resolve().parents[2]
        if self._paths_overlap(destination, source) or self._paths_overlap(destination, runner_root):
            return True
        return self._is_protected_path(destination)

    @classmethod
    def _is_protected_path(cls, path: Path) -> bool:
        return path == Path(path.anchor) or any(
            cls._paths_overlap(path, protected) for protected in _PROTECTED_ROOTS
        )

    @staticmethod
    def _paths_overlap(first: Path, second: Path) -> bool:
        try:
            common = Path(os.path.commonpath((str(first), str(second))))
        except ValueError:
            return False
        return common == first or common == second

    def _verify_worktree(
        self,
        source: Path,
        destination: Path,
        expected_commit: str,
        *,
        require_clean: bool = True,
    ) -> list[str]:
        issues: list[str] = []
        if destination.is_symlink() or not destination.is_dir():
            return ["worktree destination is missing or is a symlink"]
        try:
            resolved_destination = destination.resolve(strict=True)
        except (OSError, RuntimeError):
            return ["worktree destination cannot be resolved"]
        top = self._run_git(resolved_destination, "rev-parse", "--show-toplevel")
        if not top.succeeded or Path(top.stdout).resolve() != resolved_destination:
            issues.append(self._git_failure("created path is not the expected Git worktree", top))
        head = self._run_git(resolved_destination, "rev-parse", "HEAD")
        if not head.succeeded or head.stdout != expected_commit:
            issues.append("worktree HEAD does not match the selected base commit")
        source_common = self._common_git_dir(source)
        worktree_common = self._common_git_dir(resolved_destination)
        if source_common is None or worktree_common is None or source_common != worktree_common:
            issues.append("worktree is not associated with the selected source repository")
        if not self._listed_worktree(source, resolved_destination, expected_commit):
            issues.append("source repository does not list this run worktree and base commit")
        if require_clean and not self._worktree_is_clean(resolved_destination):
            issues.append("worktree is not clean")
        return issues

    def _common_git_dir(self, repository: Path) -> Path | None:
        result = self._run_git(repository, "rev-parse", "--git-common-dir")
        if not result.succeeded:
            return None
        path = Path(result.stdout)
        if not path.is_absolute():
            path = repository / path
        try:
            return path.resolve(strict=True)
        except (OSError, RuntimeError):
            return None

    def _listed_worktree(
        self, source: Path, destination: Path, expected_commit: str | None
    ) -> bool:
        result = self._run_git(source, "worktree", "list", "--porcelain")
        if not result.succeeded or result.output_truncated:
            return False
        for block in result.stdout.split("\n\n"):
            fields = dict(
                line.split(" ", 1) for line in block.splitlines() if " " in line
            )
            listed_path = fields.get("worktree")
            if listed_path:
                path_value = Path(listed_path)
                if not path_value.is_absolute():
                    path_value = source / path_value
                if os.path.normcase(os.path.normpath(path_value)) == os.path.normcase(
                    os.path.normpath(destination)
                ):
                    return expected_commit is None or fields.get("HEAD") == expected_commit
        return False

    def _worktree_is_clean(self, destination: Path) -> bool:
        result = self._run_git(
            destination,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--no-renames",
        )
        return result.succeeded and not result.output_truncated and not result.stdout

    def _read_manifest(self, manifest: Path) -> tuple[dict[str, Any], str | None]:
        try:
            if manifest.is_symlink() or not manifest.is_file():
                return {}, "manager ownership record is missing or is a symlink"
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}, "manager ownership record is unreadable"
        if not isinstance(data, dict) or data.get("format") != 1 or data.get("manager") != "agentic-dev-workflow":
            return {}, "manager ownership record is invalid"
        return data, None

    def _has_partial_state(self, destination: Path, source: Path) -> bool:
        return os.path.lexists(destination) or self._listed_worktree(source, destination, None)

    def _remove_manifest_if_no_partial_state(
        self, manifest: Path, destination: Path, source: Path
    ) -> None:
        if not self._has_partial_state(destination, source):
            try:
                manifest.unlink(missing_ok=True)
            except OSError:
                pass

    def _run_git(self, cwd: Path, *arguments: str) -> GitCommandResult:
        return run_git_command(cwd, *arguments, timeout_seconds=self.timeout_seconds)

    @staticmethod
    def _failed_git_result(message: str) -> GitCommandResult:
        return GitCommandResult(1, "", message)

    @staticmethod
    def _git_failure(context: str, result: GitCommandResult) -> str:
        if result.timed_out:
            return f"{context} (Git command timed out)"
        if result.error_type:
            return f"{context} (Git command failed: {result.error_type})"
        if result.output_truncated:
            return f"{context} (Git output exceeded the safe limit)"
        return context

    @staticmethod
    def _write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")

    @staticmethod
    def _manifest_path(root: Path, run_id: str) -> Path:
        return root / f"run-{run_id}.json"

    @staticmethod
    def _valid_run_id(run_id: str) -> bool:
        return isinstance(run_id, str) and _RUN_ID.fullmatch(run_id) is not None

    def _failure(
        self,
        operation: str,
        source_repository: str | Path,
        run_id: str,
        base_commit: str,
        issue: str,
    ) -> WorktreeOperationResult:
        return self._result(
            operation, False, run_id, str(source_repository), "", base_commit, False, (issue,)
        )

    @staticmethod
    def _result(
        operation: str,
        success: bool,
        run_id: str,
        source_repository: str,
        worktree_path: str,
        base_commit: str,
        partial_state: bool,
        issues: tuple[str, ...],
    ) -> WorktreeOperationResult:
        return WorktreeOperationResult(
            operation=operation,
            success=success,
            run_id=run_id,
            source_repository=source_repository,
            worktree_path=worktree_path,
            base_commit=base_commit,
            partial_state=partial_state,
            issues=issues,
        )
