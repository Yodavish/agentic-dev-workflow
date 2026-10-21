import subprocess
import tempfile
import unittest
from pathlib import Path

from agentic_dev_workflow.git_safety import inspect_repository


def init_repository(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "tracked.txt").write_text("base", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=Test", "-c",
         "user.email=test@example.invalid", "commit", "--allow-empty", "-qm", "base"],
        check=True,
    )


class PreflightTests(unittest.TestCase):
    def test_reports_nonexistent_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = inspect_repository(Path(directory) / "missing")
        self.assertFalse(result.is_git_repository)
        self.assertIn("does not exist", " ".join(result.issues))

    def test_reports_non_repository_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = inspect_repository(Path(directory))
        self.assertFalse(result.is_git_repository)
        self.assertIn("not a Git repository", " ".join(result.issues))

    def test_reports_clean_repository_and_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            repository.mkdir()
            init_repository(repository)
            result = inspect_repository(repository)
        self.assertTrue(result.is_git_repository)
        self.assertTrue(result.is_clean)
        self.assertFalse(result.untracked_files)
        self.assertTrue(result.branch)
        self.assertEqual(len(result.head_commit or ""), 40)

    def test_reports_tracked_changes_as_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            repository.mkdir()
            init_repository(repository)
            (repository / "tracked.txt").write_text("changed", encoding="utf-8")
            result = inspect_repository(repository)
        self.assertFalse(result.is_clean)
        self.assertIn("tracked.txt", result.changed_files)

    def test_reports_untracked_files_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            repository.mkdir()
            init_repository(repository)
            (repository / "new.txt").write_text("new", encoding="utf-8")
            result = inspect_repository(repository)
        self.assertFalse(result.is_clean)
        self.assertEqual(result.untracked_files, ("new.txt",))

    def test_rejects_path_inside_repository_when_root_differs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repo"
            repository.mkdir()
            init_repository(repository)
            subdirectory = repository / "subdirectory"
            subdirectory.mkdir()
            result = inspect_repository(subdirectory, expected_path=subdirectory)
        self.assertIn("does not resolve to the configured repository path", " ".join(result.issues))


if __name__ == "__main__":
    unittest.main()
