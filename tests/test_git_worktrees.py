import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_dev_workflow.git_safety import verify_git_baseline
from agentic_dev_workflow.worktrees import WorktreeManager


def git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def init_repository(repository: Path) -> str:
    repository.mkdir()
    git(repository.parent, "init", "-q", str(repository))
    (repository / "tracked.txt").write_text("base\n", encoding="utf-8")
    git(repository, "add", "tracked.txt")
    git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "base",
    )
    return git(repository, "rev-parse", "HEAD")


class GitBaselineTests(unittest.TestCase):
    def test_resolves_full_commit_and_reports_clean_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "source"
            commit = init_repository(repository)
            result = verify_git_baseline(repository, commit)
        self.assertTrue(result.success, result.issues)
        self.assertEqual(result.resolved_commit, commit)
        self.assertTrue(result.is_clean)
        self.assertEqual(result.staged_changes, ())
        self.assertEqual(result.unstaged_changes, ())
        self.assertEqual(result.untracked_files, ())
        self.assertEqual(Path(result.git_root), repository.resolve())

    def test_rejects_malformed_and_nonexistent_commit_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "source"
            commit = init_repository(repository)
            malformed = verify_git_baseline(repository, commit[:8])
            missing = verify_git_baseline(repository, "f" * 40)
        self.assertFalse(malformed.success)
        self.assertIn("full", " ".join(malformed.issues))
        self.assertFalse(missing.success)
        self.assertIn("does not exist", " ".join(missing.issues))

    def test_rejects_non_git_source_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = verify_git_baseline(Path(directory), "a" * 40)
        self.assertFalse(result.success)
        self.assertIn("Git", " ".join(result.issues))

    def test_reports_staged_unstaged_and_untracked_paths_and_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "source"
            commit = init_repository(repository)
            (repository / "tracked.txt").write_text("unstaged\n", encoding="utf-8")
            (repository / "staged.txt").write_text("staged\n", encoding="utf-8")
            git(repository, "add", "staged.txt")
            (repository / "untracked.txt").write_text("new\n", encoding="utf-8")
            before_head = git(repository, "rev-parse", "HEAD")
            before_status = git(repository, "status", "--porcelain=v1", "--untracked-files=all")
            result = verify_git_baseline(repository, commit)
            after_head = git(repository, "rev-parse", "HEAD")
            after_status = git(repository, "status", "--porcelain=v1", "--untracked-files=all")
        self.assertFalse(result.success)
        self.assertFalse(result.is_clean)
        self.assertEqual(before_head, after_head)
        self.assertEqual(before_status, after_status)
        self.assertTrue(result.staged_changes)
        self.assertIn("tracked.txt", result.unstaged_changes)
        self.assertIn("untracked.txt", result.untracked_files)
        self.assertIn("staged.txt", after_status)

    def test_baseline_git_timeout_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "source"
            commit = init_repository(repository)
            with patch(
                "agentic_dev_workflow.git_safety.subprocess.run",
                side_effect=subprocess.TimeoutExpired("git", 1),
            ):
                result = verify_git_baseline(repository, commit)
        self.assertFalse(result.success)
        self.assertTrue(any("timed out" in issue for issue in result.issues))

    def test_git_invocations_have_explicit_cwd_timeout_and_no_shell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "source"
            commit = init_repository(repository)
            original_run = subprocess.run
            observed = []

            def checked_run(*args, **kwargs):
                observed.append(kwargs)
                return original_run(*args, **kwargs)

            with patch("agentic_dev_workflow.git_safety.subprocess.run", side_effect=checked_run):
                result = verify_git_baseline(repository, commit)
        self.assertTrue(result.success, result.issues)
        self.assertTrue(observed)
        self.assertTrue(all(item.get("cwd") for item in observed))
        self.assertTrue(all(item.get("timeout", 0) > 0 for item in observed))
        self.assertTrue(all(item.get("shell") is False for item in observed))
        self.assertTrue(all(item.get("env", {}).get("GIT_OPTIONAL_LOCKS") == "0" for item in observed))

    def test_ignores_inherited_git_directory_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "source"
            commit = init_repository(repository)
            with patch.dict("os.environ", {"GIT_DIR": str(Path(directory) / "wrong-git-dir")}):
                result = verify_git_baseline(repository, commit)
        self.assertTrue(result.success, result.issues)


class WorktreeManagerTests(unittest.TestCase):
    def _setup(self, directory: str):
        base = Path(directory)
        source = base / "source"
        commit = init_repository(source)
        root = base / "managed-worktrees"
        return source, commit, root

    def test_creates_expected_detached_worktree_and_cleans_only_owned_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, commit, root = self._setup(directory)
            manager = WorktreeManager(root)
            created = manager.create_worktree(source, commit, run_id="run-123")
            destination = root / "agentic-workflow-run-123"
            self.assertTrue(created.success, created.issues)
            self.assertEqual(Path(created.worktree_path), destination.resolve())
            self.assertEqual(git(destination, "rev-parse", "HEAD"), commit)
            self.assertEqual(git(source, "rev-parse", "HEAD"), commit)
            self.assertTrue(verify_git_baseline(source, commit).success)
            cleaned = manager.cleanup_worktree(source, commit, run_id="run-123")
            self.assertTrue(cleaned.success, cleaned.issues)
            self.assertFalse(destination.exists())

    def test_rejects_invalid_run_id_and_does_not_create_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, commit, root = self._setup(directory)
            result = WorktreeManager(root).create_worktree(source, commit, run_id="../escape")
        self.assertFalse(result.success)
        self.assertFalse(result.partial_state)
        self.assertFalse(root.exists())

    def test_rejects_destination_that_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, commit, root = self._setup(directory)
            manager = WorktreeManager(root)
            first = manager.create_worktree(source, commit, run_id="prior")
            self.assertTrue(first.success, first.issues)
            destination = root / "agentic-workflow-run-123"
            destination.mkdir()
            result = manager.create_worktree(source, commit, run_id="run-123")
            self.assertFalse(result.success)
            self.assertTrue(destination.is_dir())
            self.assertTrue(first.worktree_path)

    def test_rejects_root_colliding_with_source_or_its_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, commit, _root = self._setup(directory)
            same = WorktreeManager(source).create_worktree(source, commit, run_id="run-123")
            parent = WorktreeManager(source.parent).create_worktree(source, commit, run_id="run-456")
        self.assertFalse(same.success)
        self.assertFalse(parent.success)

    def test_refuses_to_adopt_existing_unowned_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, commit, root = self._setup(directory)
            root.mkdir()
            (root / "keep.txt").write_text("unrelated", encoding="utf-8")
            result = WorktreeManager(root).create_worktree(source, commit, run_id="run-123")
            self.assertFalse(result.success)
            self.assertTrue((root / "keep.txt").exists())

    def test_refuses_protected_system_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, commit, _root = self._setup(directory)
            result = WorktreeManager(Path("/usr/local/agentic-workflow-test")).create_worktree(
                source, commit, run_id="run-123"
            )
        self.assertFalse(result.success)
        self.assertIn("protected", " ".join(result.issues))

    def test_cleanup_refuses_changed_ownership_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, commit, root = self._setup(directory)
            manager = WorktreeManager(root)
            created = manager.create_worktree(source, commit, run_id="run-123")
            self.assertTrue(created.success, created.issues)
            manifest = root / "run-run-123.json"
            manifest.write_text('{"format": 1, "manager": "agentic-dev-workflow", "run_id": "other"}', encoding="utf-8")
            result = manager.cleanup_worktree(source, commit, run_id="run-123")
            self.assertFalse(result.success)
            self.assertTrue(Path(created.worktree_path).is_dir())

    def test_cleanup_refuses_dirty_worktree_without_discarding_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, commit, root = self._setup(directory)
            manager = WorktreeManager(root)
            created = manager.create_worktree(source, commit, run_id="run-123")
            self.assertTrue(created.success, created.issues)
            dirty_file = Path(created.worktree_path) / "keep.txt"
            dirty_file.write_text("preserve me", encoding="utf-8")
            result = manager.cleanup_worktree(source, commit, run_id="run-123")
            self.assertFalse(result.success)
            self.assertTrue(dirty_file.exists())

    def test_cleanup_refuses_missing_or_mismatched_ownership_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, commit, root = self._setup(directory)
            manager = WorktreeManager(root)
            created = manager.create_worktree(source, commit, run_id="run-123")
            self.assertTrue(created.success, created.issues)
            (root / "run-run-123.json").unlink()
            result = manager.cleanup_worktree(source, commit, run_id="run-123")
            self.assertFalse(result.success)
            self.assertTrue(Path(created.worktree_path).exists())

    def test_worktree_creation_failure_reports_partial_state_without_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, commit, root = self._setup(directory)
            manager = WorktreeManager(root)
            original_run_git = manager._run_git

            def fail_add(cwd, *args):
                if "worktree" in args and "add" in args:
                    destination = Path(args[-2])
                    destination.mkdir()
                    return manager._failed_git_result("simulated add failure")
                return original_run_git(cwd, *args)

            with patch.object(manager, "_run_git", side_effect=fail_add):
                result = manager.create_worktree(source, commit, run_id="run-123")
                self.assertFalse(result.success)
                self.assertTrue(result.partial_state)
                self.assertTrue(Path(result.worktree_path).is_dir())


if __name__ == "__main__":
    unittest.main()
