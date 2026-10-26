"""Host check runner tests execute the registered unittest command."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_dev_workflow.host_check_runner import HostProcessCheckRunner
from agentic_dev_workflow.sandbox import SandboxStatus, VerifiedWorktree, invoke_sandbox_runner
from agentic_dev_workflow.verification_registry import get_check_definition
from agentic_dev_workflow.worktrees import WorktreeOperationResult


def handle(source: Path, worktree: Path, run_id: str) -> VerifiedWorktree:
    return VerifiedWorktree.from_manager_verification(
        WorktreeOperationResult(
            operation="verify",
            success=True,
            run_id=run_id,
            source_repository=str(source),
            worktree_path=str(worktree),
            base_commit="a" * 40,
            partial_state=False,
            issues=(),
        )
    )


class HostProcessCheckRunnerTests(unittest.TestCase):
    def test_runner_tests_uses_the_real_unittest_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source"
            worktree = root / "agentic-workflow-host-pass"
            source.mkdir()
            tests = worktree / "tests"
            tests.mkdir(parents=True)
            (tests / "test_sample.py").write_text(
                "import unittest\n"
                "class Sample(unittest.TestCase):\n"
                "    def test_sum(self):\n"
                "        self.assertEqual(2 + 2, 4)\n",
                encoding="utf-8",
            )
            runner = HostProcessCheckRunner()
            outcome = invoke_sandbox_runner(
                runner,
                get_check_definition("runner.tests"),
                handle(source, worktree, "host-pass"),
                run_id="host-pass",
                attempt=1,
            )
        self.assertEqual(outcome.status, SandboxStatus.PASSED)
        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(outcome.backend_type, "host-process")
        # unittest writes verbose results to stderr when stdout is a pipe.
        self.assertIn(b"ok", outcome.stdout_evidence + outcome.stderr_evidence)
        record = runner.invocations[0]
        self.assertEqual(record.registry_command[0], "/usr/local/bin/python")
        self.assertEqual(record.registry_command[1:], ("-m", "unittest", "discover", "-s", "tests", "-v"))
        self.assertNotEqual(record.executed_command[0], "/usr/local/bin/python")
        self.assertIn("unittest", record.executed_command)

    def test_root_level_transaction_test_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source"
            worktree = root / "agentic-workflow-host-root"
            source.mkdir()
            worktree.mkdir()
            (worktree / "test_transaction_actions.py").write_text(
                "import unittest\n"
                "class TransactionActions(unittest.TestCase):\n"
                "    def test_purchase_label(self):\n"
                "        self.assertEqual('Purchase', 'Purchase')\n",
                encoding="utf-8",
            )
            runner = HostProcessCheckRunner()
            outcome = invoke_sandbox_runner(
                runner,
                get_check_definition("runner.tests"),
                handle(source, worktree, "host-root"),
                run_id="host-root",
                attempt=1,
            )
        self.assertEqual(outcome.status, SandboxStatus.PASSED, outcome.stderr_evidence)
        self.assertEqual(outcome.exit_code, 0)
        record = runner.invocations[0]
        self.assertEqual(
            record.registry_command[1:],
            ("-m", "unittest", "discover", "-s", "tests", "-v"),
        )
        self.assertEqual(record.executed_command[-3:], ("-s", ".", "-v"))
        self.assertIn(b"test_purchase_label", outcome.stdout_evidence + outcome.stderr_evidence)

    def test_failed_unittest_stays_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source"
            worktree = root / "agentic-workflow-host-fail"
            source.mkdir()
            tests = worktree / "tests"
            tests.mkdir(parents=True)
            (tests / "test_sample.py").write_text(
                "import unittest\n"
                "class Sample(unittest.TestCase):\n"
                "    def test_sum(self):\n"
                "        self.assertEqual(2 + 2, 5)\n",
                encoding="utf-8",
            )
            runner = HostProcessCheckRunner()
            outcome = invoke_sandbox_runner(
                runner,
                get_check_definition("runner.tests"),
                handle(source, worktree, "host-fail"),
                run_id="host-fail",
                attempt=1,
            )
        self.assertEqual(outcome.status, SandboxStatus.FAILED)
        self.assertEqual(outcome.exit_code, 1)
        self.assertIn(b"AssertionError", outcome.stdout_evidence + outcome.stderr_evidence)
