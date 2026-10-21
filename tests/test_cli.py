import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agentic_dev_workflow.cli import main
from agentic_dev_workflow.tracing import create_tracing_session
from opentelemetry.sdk.trace.export import SpanExportResult, SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def memory_session():
    return create_tracing_session(
        "http://localhost:4318/v1/traces", exporter=InMemorySpanExporter()
    )


class CliTests(unittest.TestCase):
    def test_preflight_returns_zero_for_clean_git_repository(self) -> None:
        from test_preflight import init_repository

        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            repository.mkdir()
            init_repository(repository)
            output = io.StringIO()
            with patch("agentic_dev_workflow.workflow.create_tracing_session", return_value=memory_session()):
                with redirect_stdout(output):
                    status = main(["preflight", "--repository", str(repository)])
        self.assertEqual(status, 0)
        self.assertIn('"is_clean": true', output.getvalue())

    def test_preflight_returns_nonzero_for_dirty_repository(self) -> None:
        from test_preflight import init_repository

        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            repository.mkdir()
            init_repository(repository)
            (repository / "untracked.txt").write_text("work", encoding="utf-8")
            output = io.StringIO()
            with patch("agentic_dev_workflow.workflow.create_tracing_session", return_value=memory_session()):
                with redirect_stdout(output):
                    status = main(["preflight", "--repository", str(repository)])
        self.assertEqual(status, 1)
        self.assertIn('"is_clean": false', output.getvalue())

    def test_run_is_preflight_only_and_succeeds_for_clean_repo(self) -> None:
        from test_preflight import init_repository

        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            repository.mkdir()
            init_repository(repository)
            output = io.StringIO()
            with patch("agentic_dev_workflow.workflow.create_tracing_session", return_value=memory_session()):
                with redirect_stdout(output):
                    status = main(["run", "--repository", str(repository)])
        self.assertEqual(status, 0)
        self.assertIn('"dry_run_only": true', output.getvalue())

    def test_invalid_configuration_returns_nonzero_without_secret(self) -> None:
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = main(["trace-smoke", "--otlp-endpoint", "http://user:secret@localhost/v1/traces"])
        self.assertEqual(status, 2)
        self.assertNotIn("secret", errors.getvalue())

    def test_trace_smoke_returns_nonzero_when_export_fails(self) -> None:
        class FailingExporter(SpanExporter):
            def export(self, spans):
                return SpanExportResult.FAILURE

            def shutdown(self) -> None:
                pass

        session = create_tracing_session(
            "http://localhost:4318/v1/traces", exporter=FailingExporter()
        )
        output = io.StringIO()
        errors = io.StringIO()
        with patch("agentic_dev_workflow.cli.create_tracing_session", return_value=session):
            with redirect_stdout(output), redirect_stderr(errors):
                status = main(["trace-smoke"])
        self.assertEqual(status, 1)
        self.assertIn('"exported": false', output.getvalue())
        self.assertIn("trace export failed", errors.getvalue())

    def test_doctor_returns_nonzero_when_local_jaeger_is_unreachable(self) -> None:
        output = io.StringIO()
        session = memory_session()
        with patch("agentic_dev_workflow.cli._endpoint_reachable", return_value=False):
            with patch("agentic_dev_workflow.cli.create_tracing_session", return_value=session):
                with redirect_stdout(output):
                    status = main(["doctor"])
        self.assertEqual(status, 1)
        self.assertIn('"otlp_endpoint_reachable": false', output.getvalue())

    def test_help_returns_zero(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["--help"])
        self.assertEqual(status, 0)
        self.assertIn("trace-smoke", output.getvalue())

    def test_workflow_demo_succeeds_without_repository_configuration(self) -> None:
        output = io.StringIO()
        with patch("agentic_dev_workflow.config.Path.resolve", side_effect=AssertionError("repository accessed")):
            with redirect_stdout(output):
                status = main(["workflow-demo", "--scenario", "retry-success", "--no-tracing"])
        self.assertEqual(status, 0)
        self.assertIn('"final_status": "success"', output.getvalue())
        self.assertIn('"execution_attempts": 2', output.getvalue())
        self.assertIn('"simulated": true', output.getvalue())

    def test_workflow_demo_returns_failure_for_exhausted_scenario(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["workflow-demo", "--scenario", "exhausted-retries", "--max-retries", "0", "--no-tracing"])
        self.assertEqual(status, 1)
        self.assertIn('"final_status": "failed"', output.getvalue())
        self.assertIn('"repair_retries_used": 0', output.getvalue())

    def test_workflow_demo_data_directory_selects_durable_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "runner-data"
            with redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "workflow-demo",
                        "--run-id",
                        "durable-cli-case",
                        "--data-dir",
                        str(data_dir),
                        "--no-tracing",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertTrue((data_dir / "checkpoints.sqlite3").is_file())

    def test_baseline_check_command_requires_and_verifies_temporary_repository(self) -> None:
        from test_git_worktrees import init_repository

        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "source"
            commit = init_repository(repository)
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(
                    ["baseline-check", "--repository", str(repository), "--base-commit", commit]
                )
        self.assertEqual(status, 0)
        self.assertIn('"resolved_commit"', output.getvalue())
        self.assertIn('"success": true', output.getvalue())

    def test_worktree_cli_requires_explicit_action_and_operates_on_temporary_repo(self) -> None:
        from test_git_worktrees import init_repository

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "source"
            commit = init_repository(repository)
            manager_root = root / "managed-worktrees"
            output = io.StringIO()
            with redirect_stdout(output):
                create_status = main(
                    [
                        "worktree",
                        "--action",
                        "create",
                        "--source-repository",
                        str(repository),
                        "--base-commit",
                        commit,
                        "--run-id",
                        "cli-run",
                        "--worktree-root",
                        str(manager_root),
                    ]
                )
            self.assertEqual(create_status, 0)
            self.assertTrue((manager_root / "agentic-workflow-cli-run").is_dir())
            output = io.StringIO()
            with redirect_stdout(output):
                cleanup_status = main(
                    [
                        "worktree",
                        "--action",
                        "cleanup",
                        "--source-repository",
                        str(repository),
                        "--base-commit",
                        commit,
                        "--run-id",
                        "cli-run",
                        "--worktree-root",
                        str(manager_root),
                    ]
                )
        self.assertEqual(cleanup_status, 0)
        self.assertIn('"success": true', output.getvalue())
        self.assertFalse((manager_root / "agentic-workflow-cli-run").exists())

    def test_worktree_cli_rejects_omitted_action(self) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors):
            status = main(["worktree", "--repository", "/tmp/source"])
        self.assertEqual(status, 2)
        self.assertIn("--action", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
