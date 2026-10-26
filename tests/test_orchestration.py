"""End-to-end orchestration MVP: fake agents, fake sandbox, real Git inventory."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agentic_dev_workflow.adapters import (
    AgentFailureCategory,
    AgentResult,
    AgentStatus,
    MockAgentAdapter,
)
from agentic_dev_workflow.checkpointing import open_sqlite_checkpointer
from agentic_dev_workflow.sandbox import (
    RUNNER_RUNTIME_PROFILE_ID,
    SandboxCheckResult,
    SandboxCleanupStatus,
    SandboxFailureCategory,
    SandboxStatus,
    SandboxTermination,
    UnavailableSandboxRunner,
    VerifiedWorktree,
)
from agentic_dev_workflow.verification_registry import CheckId
from agentic_dev_workflow.worktrees import WorktreeOperationResult


PLAN = {
    "summary": "Add a note",
    "assumptions": [],
    "steps": [
        {
            "id": "step-1",
            "description": "Write the note",
            "expected_files": ["notes.txt"],
        }
    ],
    "verification_check_ids": ["runner.tests", "runner.lock-check"],
}
EXECUTOR = {
    "status": "completed",
    "summary": "Wrote the note",
    "files_changed": ["notes.txt"],
    "checks_claimed": [],
    "failure_category": None,
    "diagnostic_code": None,
}
REVIEW = {"decision": "approve", "findings": []}


def git_environment() -> dict[str, str]:
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
    ):
        environment.pop(name, None)
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
        env=git_environment(),
    )
    return result.stdout.strip()


class CountingAdapter:
    """Counts invoke calls and delegates to the deterministic mock adapter."""

    adapter_type = "mock"

    def __init__(self, structured_output: dict[str, object]) -> None:
        self.inner = MockAgentAdapter(structured_output=structured_output)
        self.calls = 0
        self.requests = []

    def invoke(self, request: object) -> AgentResult:
        self.calls += 1
        self.requests.append(request)
        return self.inner.invoke(request)


class ScriptedSandbox:
    """Attempt-indexed fake. Outcomes still pass through the sandbox boundary."""

    runtime_profile_id = RUNNER_RUNTIME_PROFILE_ID

    def __init__(
        self, outcomes_by_attempt: dict[int, dict[CheckId, SandboxCheckResult]]
    ) -> None:
        self.outcomes_by_attempt = outcomes_by_attempt
        self.calls: list[tuple[CheckId, int, str]] = []

    def run(self, check, worktree, *, run_id: str, attempt: int) -> SandboxCheckResult:
        self.calls.append((check.check_id, attempt, worktree.path))
        outcome = self.outcomes_by_attempt[attempt][check.check_id]
        return replace(outcome, run_id=run_id, attempt=attempt)


def sandbox_result(
    *,
    check_id: CheckId = CheckId.RUNNER_TESTS,
    status: SandboxStatus = SandboxStatus.PASSED,
    run_id: str = "demo-success",
) -> SandboxCheckResult:
    failed = status == SandboxStatus.FAILED
    return SandboxCheckResult(
        run_id=run_id,
        check_id=check_id,
        attempt=1,
        status=status,
        exit_code=1 if failed else 0,
        duration_seconds=0.25,
        output_bytes_observed=0,
        termination=SandboxTermination.NATURAL_EXIT,
        stdout_evidence=b"SECRET-STDOUT",
        stderr_evidence=b"SECRET-STDERR",
        cleanup_status=SandboxCleanupStatus.SUCCEEDED,
        failure_category=SandboxFailureCategory.NONZERO_EXIT if failed else None,
        diagnostic_code=None,
        backend_type="fake",
    )


def prepare_repository(parent: Path, run_id: str, *, extra_file: str | None = None):
    source = parent / "source"
    source.mkdir()
    worktree = parent / f"agentic-workflow-{run_id}"
    worktree.mkdir()
    git(parent, "init", "-q", str(worktree))
    git(worktree, "config", "user.name", "Test")
    git(worktree, "config", "user.email", "test@example.invalid")
    (worktree / "tracked.txt").write_text("base\n", encoding="utf-8")
    git(worktree, "add", "tracked.txt")
    git(worktree, "commit", "-qm", "base")
    (worktree / "notes.txt").write_text("note\n", encoding="utf-8")
    if extra_file is not None:
        (worktree / extra_file).write_text("extra\n", encoding="utf-8")
    canonical = worktree.resolve()
    handle = VerifiedWorktree.from_manager_verification(
        WorktreeOperationResult(
            operation="verify",
            success=True,
            run_id=run_id,
            source_repository=source.resolve().as_posix(),
            worktree_path=canonical.as_posix(),
            base_commit=git(worktree, "rev-parse", "HEAD"),
            partial_state=False,
            issues=(),
        )
    )
    return canonical, handle


def run_case(
    *,
    run_id: str,
    sandbox,
    approval,
    max_retries: int = 1,
    extra_file: str | None = None,
    planner_output: dict[str, object] | None = None,
    checkpointer=None,
    interrupt_before=None,
    resume: bool = False,
    parent: Path | None = None,
    spec_text: str = "Add a note",
):
    from agentic_dev_workflow.orchestration import ScopeApproval, run_orchestration

    def invoke(directory: Path):
        _root, handle = prepare_repository(directory, run_id, extra_file=extra_file)
        planner = CountingAdapter(PLAN if planner_output is None else planner_output)
        executor = CountingAdapter(EXECUTOR)
        reviewer = CountingAdapter(REVIEW)
        granted = None if approval is None else ScopeApproval(approval)
        report = run_orchestration(
            spec_text=spec_text,
            worktree=handle,
            planner=planner,
            executor=executor,
            reviewer=reviewer,
            sandbox=sandbox,
            approval=granted,
            max_retries=max_retries,
            run_id=run_id,
            checkpointer=checkpointer,
            interrupt_before=interrupt_before,
            resume=resume,
        )
        return report, planner, executor, reviewer

    if parent is not None:
        return invoke(parent)
    with tempfile.TemporaryDirectory() as directory:
        return invoke(Path(directory))


class OrchestrationMvpTests(unittest.TestCase):
    def test_successful_run_reports_passed_checks_and_skips_lock_check(self) -> None:
        sandbox = ScriptedSandbox({1: {CheckId.RUNNER_TESTS: sandbox_result()}})
        report, planner, executor, reviewer = run_case(
            run_id="demo-success",
            sandbox=sandbox,
            approval=("notes.txt",),
        )
        self.assertEqual(report.outcome, "success")
        self.assertEqual(report.attempts, 1)
        self.assertEqual(report.repair_retries_used, 0)
        self.assertEqual(report.review_decision, "approve")
        self.assertEqual(report.errors, ())
        self.assertEqual(report.evidence_refs, ("ev_runner_tests",))
        self.assertEqual(report.proposed_paths, ("notes.txt",))
        self.assertEqual(
            [(check["check_id"], check["status"]) for check in report.checks],
            [("runner.tests", "passed"), ("runner.lock-check", "skipped")],
        )
        self.assertIsNone(report.checks[1]["evidence_ref"])
        self.assertIsNone(report.checks[1]["exit_code"])
        self.assertEqual(report.attempt_history[0]["verification_outcome"], "pass")
        self.assertEqual(planner.calls, 1)
        self.assertEqual(executor.calls, 1)
        self.assertEqual(reviewer.calls, 1)
        self.assertEqual(
            [(check_id.value, attempt) for check_id, attempt, _path in sandbox.calls],
            [("runner.tests", 1)],
        )
        encoded = json.dumps(report.to_dict())
        self.assertNotIn("SECRET", encoded)

    def test_one_bounded_repair_then_successful_reverification(self) -> None:
        sandbox = ScriptedSandbox(
            {
                1: {
                    CheckId.RUNNER_TESTS: sandbox_result(
                        status=SandboxStatus.FAILED, run_id="demo-repair"
                    )
                },
                2: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-repair")},
            }
        )
        report, _planner, executor, reviewer = run_case(
            run_id="demo-repair",
            sandbox=sandbox,
            approval=("notes.txt",),
            max_retries=1,
        )
        self.assertEqual(report.outcome, "success")
        self.assertEqual(report.attempts, 2)
        self.assertEqual(report.repair_retries_used, 1)
        self.assertEqual(report.max_retries, 1)
        self.assertEqual(
            [entry["verification_outcome"] for entry in report.attempt_history],
            ["fail", "pass"],
        )
        self.assertEqual(report.checks[0]["status"], "passed")
        self.assertEqual(executor.calls, 2)
        self.assertEqual(reviewer.calls, 2)
        self.assertEqual([attempt for _check, attempt, _path in sandbox.calls], [1, 2])
        self.assertNotIn(3, [attempt for _check, attempt, _path in sandbox.calls])

    def test_persistent_failure_exhausts_one_repair_and_stops(self) -> None:
        failed = sandbox_result(status=SandboxStatus.FAILED, run_id="demo-exhausted")
        sandbox = ScriptedSandbox(
            {
                1: {CheckId.RUNNER_TESTS: failed},
                2: {CheckId.RUNNER_TESTS: failed},
                3: {CheckId.RUNNER_TESTS: failed},
            }
        )
        report, _planner, executor, reviewer = run_case(
            run_id="demo-exhausted",
            sandbox=sandbox,
            approval=("notes.txt",),
            max_retries=1,
        )
        self.assertEqual(report.outcome, "failed")
        self.assertEqual(report.attempts, 2)
        self.assertEqual(report.repair_retries_used, 1)
        self.assertEqual(report.review_decision, "approve")
        self.assertEqual(report.errors, ("verification_failed", "retries_exhausted"))
        self.assertEqual(
            [entry["verification_outcome"] for entry in report.attempt_history],
            ["fail", "fail"],
        )
        self.assertEqual(executor.calls, 2)
        self.assertEqual(reviewer.calls, 2)
        self.assertEqual([attempt for _check, attempt, _path in sandbox.calls], [1, 2])

    def test_retry_limit_stops_at_zero_and_rejects_a_third_repair(self) -> None:
        from agentic_dev_workflow.orchestration import run_orchestration

        failed = sandbox_result(status=SandboxStatus.FAILED, run_id="demo-boundary")
        sandbox = ScriptedSandbox({1: {CheckId.RUNNER_TESTS: failed}})
        report, _planner, executor, _reviewer = run_case(
            run_id="demo-boundary",
            sandbox=sandbox,
            approval=("notes.txt",),
            max_retries=0,
        )
        self.assertEqual(report.outcome, "failed")
        self.assertEqual(report.attempts, 1)
        self.assertEqual(report.repair_retries_used, 0)
        self.assertEqual(executor.calls, 1)
        self.assertEqual(sandbox.calls[0][1], 1)
        with tempfile.TemporaryDirectory() as directory:
            _root, handle = prepare_repository(Path(directory), "demo-boundary")
            with self.assertRaises(ValueError):
                run_orchestration(
                    spec_text="Add a note",
                    worktree=handle,
                    planner=CountingAdapter(PLAN),
                    executor=CountingAdapter(EXECUTOR),
                    reviewer=CountingAdapter(REVIEW),
                    sandbox=sandbox,
                    approval=None,
                    max_retries=3,
                    run_id="demo-boundary",
                )
            with self.assertRaises(ValueError):
                run_orchestration(
                    spec_text="Add a note",
                    worktree=handle,
                    planner=CountingAdapter(PLAN),
                    executor=CountingAdapter(EXECUTOR),
                    reviewer=CountingAdapter(REVIEW),
                    sandbox=sandbox,
                    approval=None,
                    max_retries=True,
                    run_id="demo-boundary",
                )

    def test_missing_or_wider_approval_does_not_execute(self) -> None:
        sandbox = ScriptedSandbox({1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-approval")}})
        missing, _planner, executor, reviewer = run_case(
            run_id="demo-approval",
            sandbox=sandbox,
            approval=None,
        )
        self.assertEqual(missing.outcome, "awaiting_approval")
        self.assertEqual(missing.attempts, 0)
        self.assertEqual(missing.errors, ("approval_required",))
        self.assertEqual(executor.calls, 0)
        self.assertEqual(reviewer.calls, 0)
        self.assertEqual(sandbox.calls, [])

        sandbox = ScriptedSandbox(
            {1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-expand")}}
        )
        expanded, _planner, executor, reviewer = run_case(
            run_id="demo-expand",
            sandbox=sandbox,
            approval=("notes.txt", "other.txt"),
        )
        self.assertEqual(expanded.outcome, "failed")
        self.assertEqual(expanded.errors, ("scope_mismatch",))
        self.assertEqual(expanded.attempts, 0)
        self.assertEqual(executor.calls, 0)
        self.assertEqual(sandbox.calls, [])

    def test_change_outside_approved_scope_is_not_success(self) -> None:
        sandbox = ScriptedSandbox({1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-scope")}})
        report, _planner, executor, reviewer = run_case(
            run_id="demo-scope",
            sandbox=sandbox,
            approval=("notes.txt",),
            extra_file="extra.txt",
        )
        self.assertEqual(report.outcome, "failed")
        self.assertEqual(report.errors, ("out_of_scope",))
        self.assertEqual(report.attempts, 1)
        self.assertEqual(executor.calls, 1)
        self.assertEqual(reviewer.calls, 0)
        self.assertEqual(sandbox.calls, [])

    def test_unavailable_sandbox_is_not_success_and_does_not_retry(self) -> None:
        sandbox = UnavailableSandboxRunner()
        report, _planner, executor, reviewer = run_case(
            run_id="demo-unavailable",
            sandbox=sandbox,
            approval=("notes.txt",),
            max_retries=1,
        )
        self.assertEqual(report.outcome, "error")
        self.assertEqual(report.attempts, 1)
        self.assertEqual(report.repair_retries_used, 0)
        self.assertIsNone(report.review_decision)
        self.assertEqual(report.checks[0]["status"], "error")
        self.assertIn("verification_error", report.errors)
        self.assertEqual(executor.calls, 1)
        self.assertEqual(reviewer.calls, 0)

    def test_invalid_planner_output_does_not_execute(self) -> None:
        sandbox = ScriptedSandbox({1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-planner")}})
        report, planner, executor, _reviewer = run_case(
            run_id="demo-planner",
            sandbox=sandbox,
            approval=("notes.txt",),
            planner_output={},
        )
        self.assertEqual(report.outcome, "error")
        self.assertEqual(report.errors, ("output_fields_invalid",))
        self.assertEqual(planner.calls, 1)
        self.assertEqual(executor.calls, 0)
        self.assertEqual(sandbox.calls, [])

    def test_checkpoint_resume_executes_only_after_resume(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from agentic_dev_workflow.orchestration import ScopeApproval, run_orchestration

        sandbox = ScriptedSandbox(
            {1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-checkpoint")}}
        )
        saver = InMemorySaver()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root, handle = prepare_repository(parent, "demo-checkpoint")
            planner = CountingAdapter(PLAN)
            executor = CountingAdapter(EXECUTOR)
            reviewer = CountingAdapter(REVIEW)
            approval = ScopeApproval(("notes.txt",))
            paused = run_orchestration(
                spec_text="Add a note",
                worktree=handle,
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                approval=approval,
                max_retries=1,
                run_id="demo-checkpoint",
                checkpointer=saver,
                interrupt_before=["execute"],
            )
            self.assertEqual(paused.outcome, "paused")
            self.assertEqual(paused.attempts, 0)
            self.assertEqual(executor.calls, 0)
            self.assertEqual(sandbox.calls, [])
            self.assertEqual(paused.proposed_paths, ("notes.txt",))
            finished = run_orchestration(
                spec_text="Add a note",
                worktree=handle,
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                approval=approval,
                max_retries=1,
                run_id="demo-checkpoint",
                checkpointer=saver,
                resume=True,
            )
            self.assertEqual(finished.outcome, "success")
            self.assertEqual(executor.calls, 1)
            self.assertEqual(root.as_posix(), handle.path)
            with self.assertRaises(ValueError):
                run_orchestration(
                    spec_text="A different specification",
                    worktree=handle,
                    planner=planner,
                    executor=executor,
                    reviewer=reviewer,
                    sandbox=sandbox,
                    approval=approval,
                    max_retries=1,
                    run_id="demo-checkpoint",
                    checkpointer=saver,
                    resume=True,
                )
            self.assertEqual(executor.calls, 1)

    def test_sqlite_checkpoint_survives_reopen(self) -> None:
        from agentic_dev_workflow.orchestration import ScopeApproval, run_orchestration

        sandbox = ScriptedSandbox(
            {1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-sqlite")}}
        )
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            _root, handle = prepare_repository(parent, "demo-sqlite")
            planner = CountingAdapter(PLAN)
            executor = CountingAdapter(EXECUTOR)
            reviewer = CountingAdapter(REVIEW)
            approval = ScopeApproval(("notes.txt",))
            data_dir = parent / "checkpoints"
            with open_sqlite_checkpointer(data_dir) as saver:
                paused = run_orchestration(
                    spec_text="Add a note",
                    worktree=handle,
                    planner=planner,
                    executor=executor,
                    reviewer=reviewer,
                    sandbox=sandbox,
                    approval=approval,
                    run_id="demo-sqlite",
                    checkpointer=saver,
                    interrupt_before=["execute"],
                )
            self.assertEqual(paused.outcome, "paused")
            self.assertEqual(executor.calls, 0)
            with open_sqlite_checkpointer(data_dir, require_existing=True) as saver:
                finished = run_orchestration(
                    spec_text="Add a note",
                    worktree=handle,
                    planner=planner,
                    executor=executor,
                    reviewer=reviewer,
                    sandbox=sandbox,
                    approval=approval,
                    run_id="demo-sqlite",
                    checkpointer=saver,
                    resume=True,
                )
        self.assertEqual(finished.outcome, "success")
        self.assertEqual(executor.calls, 1)
        self.assertEqual(finished.attempts, 1)


REJECT = {
    "decision": "reject",
    "findings": [
        {
            "finding_id": "finding-1",
            "severity": "high",
            "category": "safety",
            "blocks_approval": True,
            "file": "notes.txt",
            "function": None,
            "evidence_refs": ["ev_runner_tests"],
            "explanation": "The note is outside the requested behavior.",
            "remediation": "Remove the extra change.",
        }
    ],
}
PASSED_CHECK = {
    "check_id": "runner.tests",
    "status": "passed",
    "exit_code": 0,
    "duration_ms": 1,
    "evidence_ref": "ev_runner_tests",
    "summary": "registered check passed",
}


class WritingExecutor:
    """Writes an unapproved file on the second invocation."""

    adapter_type = "mock"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.inner = MockAgentAdapter(structured_output=EXECUTOR)
        self.calls = 0

    def invoke(self, request: object) -> AgentResult:
        self.calls += 1
        if self.calls >= 2:
            (self.root / "extra.txt").write_text("extra\n", encoding="utf-8")
        return self.inner.invoke(request)


class RejectOnce:
    """Rejects the first review and approves any later one."""

    adapter_type = "mock"

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, request: object) -> AgentResult:
        self.calls += 1
        payload = REJECT if self.calls == 1 else {"decision": "approve", "findings": []}
        return MockAgentAdapter(structured_output=payload).invoke(request)


class PassingSandbox:
    runtime_profile_id = RUNNER_RUNTIME_PROFILE_ID
    calls: list[tuple[str, int]]

    def __init__(self) -> None:
        self.calls = []

    def run(self, check, worktree, *, run_id: str, attempt: int):
        self.calls.append((check.check_id.value, attempt))
        return sandbox_result(check_id=check.check_id, run_id=run_id)


class OrchestrationIntegrityTests(unittest.TestCase):
    def test_forged_review_checkpoint_cannot_resume_as_success(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from agentic_dev_workflow.orchestration import (
            ScopeApproval,
            _build_graph,
            run_orchestration,
        )

        sandbox = ScriptedSandbox(
            {1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-forge")}}
        )
        saver = InMemorySaver()
        with tempfile.TemporaryDirectory() as directory:
            root, handle = prepare_repository(Path(directory), "demo-forge")
            planner = CountingAdapter(PLAN)
            executor = CountingAdapter(EXECUTOR)
            reviewer = CountingAdapter(REVIEW)
            approval = ScopeApproval(("notes.txt",))
            paused = run_orchestration(
                spec_text="Add a note",
                worktree=handle,
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                approval=approval,
                max_retries=0,
                run_id="demo-forge",
                checkpointer=saver,
                interrupt_before=["execute"],
            )
            self.assertEqual(paused.outcome, "paused")
            graph = _build_graph(
                repository_root=root,
                worktree=handle,
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                checkpointer=saver,
                interrupt_before=None,
            )
            graph.update_state(
                {"configurable": {"thread_id": "agentic-dev-workflow:orchestration:demo-forge"}},
                {
                    "verification": {"outcome": "pass", "checks": []},
                    "review_decision": "approve",
                    "execution_attempt": 1,
                    "repair_retries_used": 0,
                    "final_status": "running",
                    "checks": [dict(PASSED_CHECK)],
                    "errors": [],
                },
                as_node="review",
            )
            with self.assertRaisesRegex(ValueError, "evidence"):
                run_orchestration(
                    spec_text="Add a note",
                    worktree=handle,
                    planner=planner,
                    executor=executor,
                    reviewer=reviewer,
                    sandbox=sandbox,
                    approval=approval,
                    max_retries=0,
                    run_id="demo-forge",
                    checkpointer=saver,
                    resume=True,
                )
            self.assertEqual(executor.calls, 0)
            self.assertEqual(reviewer.calls, 0)
            self.assertEqual(sandbox.calls, [])

    def test_consistent_fabricated_pass_cannot_resume_as_success(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from agentic_dev_workflow.orchestration import (
            ScopeApproval,
            _build_graph,
            run_orchestration,
        )

        sandbox = ScriptedSandbox(
            {1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-consistent-forge")}}
        )
        saver = InMemorySaver()
        with tempfile.TemporaryDirectory() as directory:
            root, handle = prepare_repository(Path(directory), "demo-consistent-forge")
            planner = CountingAdapter(PLAN)
            executor = CountingAdapter(EXECUTOR)
            reviewer = CountingAdapter(REVIEW)
            approval = ScopeApproval(("notes.txt",))
            run_orchestration(
                spec_text="Add a note",
                worktree=handle,
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                approval=approval,
                max_retries=0,
                run_id="demo-consistent-forge",
                checkpointer=saver,
                interrupt_before=["execute"],
            )
            history_entry = {
                "attempt": 1,
                "verification_outcome": "pass",
                "evidence_refs": ["ev_runner_tests"],
                "review_decision": "approve",
                "checks": [dict(PASSED_CHECK)],
                "terminal_error": None,
            }
            graph = _build_graph(
                repository_root=root,
                worktree=handle,
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                checkpointer=saver,
                interrupt_before=None,
            )
            graph.update_state(
                {
                    "configurable": {
                        "thread_id": "agentic-dev-workflow:orchestration:demo-consistent-forge"
                    }
                },
                {
                    "verification": {"outcome": "pass", "checks": [dict(PASSED_CHECK)]},
                    "review_decision": "approve",
                    "execution_attempt": 1,
                    "executor_invocations": 1,
                    "repair_retries_used": 0,
                    "final_status": "running",
                    "checks": [dict(PASSED_CHECK)],
                    "attempt_history": [history_entry],
                    "verification_binding": "a" * 64,
                    "review_binding": "b" * 64,
                    "errors": [],
                },
                as_node="review",
            )
            with self.assertRaisesRegex(ValueError, "evidence"):
                run_orchestration(
                    spec_text="Add a note",
                    worktree=handle,
                    planner=planner,
                    executor=executor,
                    reviewer=reviewer,
                    sandbox=sandbox,
                    approval=approval,
                    max_retries=0,
                    run_id="demo-consistent-forge",
                    checkpointer=saver,
                    resume=True,
                )
            self.assertEqual(executor.calls, 0)
            self.assertEqual(reviewer.calls, 0)
            self.assertEqual(sandbox.calls, [])

    def test_real_pause_before_decide_can_resume(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from agentic_dev_workflow.orchestration import ScopeApproval, run_orchestration

        sandbox = PassingSandbox()
        saver = InMemorySaver()
        with tempfile.TemporaryDirectory() as directory:
            _root, handle = prepare_repository(Path(directory), "demo-real-decide")
            planner = CountingAdapter(PLAN)
            executor = CountingAdapter(EXECUTOR)
            reviewer = CountingAdapter(REVIEW)
            approval = ScopeApproval(("notes.txt",))
            paused = run_orchestration(
                spec_text="Add a note",
                worktree=handle,
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                approval=approval,
                max_retries=0,
                run_id="demo-real-decide",
                checkpointer=saver,
                interrupt_before=["decide"],
            )
            self.assertEqual(paused.outcome, "paused")
            self.assertEqual(executor.calls, 1)
            self.assertEqual(reviewer.calls, 1)
            self.assertEqual(sandbox.calls, [("runner.tests", 1)])
            finished = run_orchestration(
                spec_text="Add a note",
                worktree=handle,
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                approval=approval,
                max_retries=0,
                run_id="demo-real-decide",
                checkpointer=saver,
                resume=True,
            )
            self.assertEqual(finished.outcome, "success")
            self.assertEqual(executor.calls, 1)
            self.assertEqual(reviewer.calls, 1)
            self.assertEqual(sandbox.calls, [("runner.tests", 1)])

    def test_scope_failure_on_repair_does_not_keep_stale_pass(self) -> None:
        from agentic_dev_workflow.orchestration import ScopeApproval, run_orchestration

        with tempfile.TemporaryDirectory() as directory:
            root, handle = prepare_repository(Path(directory), "demo-stale")
            sandbox = PassingSandbox()
            executor = WritingExecutor(root)
            reviewer = RejectOnce()
            report = run_orchestration(
                spec_text="Add a note",
                worktree=handle,
                planner=CountingAdapter(PLAN),
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                approval=ScopeApproval(("notes.txt",)),
                max_retries=1,
                run_id="demo-stale",
            )
        self.assertEqual(report.outcome, "failed")
        self.assertEqual(report.errors, ("out_of_scope",))
        self.assertEqual(report.checks, ())
        self.assertEqual(report.evidence_refs, ())
        self.assertEqual(report.review_decision, None)
        self.assertEqual(
            [entry["attempt"] for entry in report.attempt_history],
            [1, 2],
        )
        self.assertEqual(report.attempt_history[0]["verification_outcome"], "pass")
        self.assertEqual(report.attempt_history[0]["review_decision"], "reject")
        self.assertEqual(report.attempt_history[0]["checks"][0]["status"], "passed")
        self.assertEqual(report.attempt_history[0]["checks"][0]["check_id"], "runner.tests")
        self.assertEqual(report.attempt_history[1]["verification_outcome"], "not_run")
        self.assertEqual(report.attempt_history[1]["evidence_refs"], [])
        self.assertIsNone(report.attempt_history[1]["review_decision"])
        self.assertEqual(report.attempt_history[1]["checks"], [])
        self.assertEqual(report.attempt_history[1]["terminal_error"], "out_of_scope")
        self.assertEqual(executor.calls, 2)
        self.assertEqual(reviewer.calls, 1)
        self.assertEqual(sandbox.calls, [("runner.tests", 1)])

    def test_invalid_executor_output_counts_the_invocation(self) -> None:
        from agentic_dev_workflow.orchestration import ScopeApproval, run_orchestration

        sandbox = ScriptedSandbox(
            {1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-executor")}}
        )
        with tempfile.TemporaryDirectory() as directory:
            _root, handle = prepare_repository(Path(directory), "demo-executor")
            executor = CountingAdapter({})
            report = run_orchestration(
                spec_text="Add a note",
                worktree=handle,
                planner=CountingAdapter(PLAN),
                executor=executor,
                reviewer=CountingAdapter(REVIEW),
                sandbox=sandbox,
                approval=ScopeApproval(("notes.txt",)),
                run_id="demo-executor",
            )
        self.assertEqual(report.outcome, "error")
        self.assertEqual(report.errors, ("output_fields_invalid",))
        self.assertEqual(report.attempts, 0)
        self.assertEqual(report.executor_invocations, 1)
        self.assertEqual(executor.calls, 1)
        self.assertEqual(sandbox.calls, [])

    def test_lock_check_runs_when_a_dependency_file_changes(self) -> None:
        from agentic_dev_workflow.orchestration import ScopeApproval, run_orchestration

        plan = {
            "summary": "Update the lockfile",
            "assumptions": [],
            "steps": [
                {
                    "id": "step-1",
                    "description": "Record the lock and note",
                    "expected_files": ["notes.txt", "uv.lock"],
                }
            ],
            "verification_check_ids": ["runner.tests", "runner.lock-check"],
        }
        sandbox = ScriptedSandbox(
            {
                1: {
                    CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-lock"),
                    CheckId.RUNNER_LOCK_CHECK: sandbox_result(
                        check_id=CheckId.RUNNER_LOCK_CHECK, run_id="demo-lock"
                    ),
                }
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            _root, handle = prepare_repository(parent, "demo-lock")
            (parent / "agentic-workflow-demo-lock" / "uv.lock").write_text(
                "lock\n", encoding="utf-8"
            )
            report = run_orchestration(
                spec_text="Update the lockfile",
                worktree=handle,
                planner=CountingAdapter(plan),
                executor=CountingAdapter(EXECUTOR),
                reviewer=CountingAdapter(REVIEW),
                sandbox=sandbox,
                approval=ScopeApproval(("notes.txt", "uv.lock")),
                run_id="demo-lock",
            )
        self.assertEqual(report.outcome, "success")
        self.assertEqual(
            [(check["check_id"], check["status"]) for check in report.checks],
            [("runner.tests", "passed"), ("runner.lock-check", "passed")],
        )
        self.assertEqual(
            [(check_id.value, attempt) for check_id, attempt, _path in sandbox.calls],
            [("runner.tests", 1), ("runner.lock-check", 1)],
        )

    def test_rename_endpoint_outside_approval_stops_before_sandbox(self) -> None:
        from agentic_dev_workflow.orchestration import ScopeApproval, run_orchestration

        plan = {
            "summary": "Rename a tracked file",
            "assumptions": [],
            "steps": [
                {
                    "id": "step-1",
                    "description": "Move the tracked file",
                    "expected_files": ["tracked.txt"],
                }
            ],
            "verification_check_ids": ["runner.tests"],
        }
        sandbox = ScriptedSandbox(
            {1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-rename")}}
        )
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "source"
            source.mkdir()
            worktree = parent / "agentic-workflow-demo-rename"
            worktree.mkdir()
            git(parent, "init", "-q", str(worktree))
            git(worktree, "config", "user.name", "Test")
            git(worktree, "config", "user.email", "test@example.invalid")
            (worktree / "tracked.txt").write_text("base\n", encoding="utf-8")
            git(worktree, "add", "tracked.txt")
            git(worktree, "commit", "-qm", "base")
            git(worktree, "mv", "tracked.txt", "moved.txt")
            canonical = worktree.resolve()
            handle = VerifiedWorktree.from_manager_verification(
                WorktreeOperationResult(
                    operation="verify",
                    success=True,
                    run_id="demo-rename",
                    source_repository=source.resolve().as_posix(),
                    worktree_path=canonical.as_posix(),
                    base_commit=git(worktree, "rev-parse", "HEAD"),
                    partial_state=False,
                    issues=(),
                )
            )
            executor = CountingAdapter(EXECUTOR)
            report = run_orchestration(
                spec_text="Rename a tracked file",
                worktree=handle,
                planner=CountingAdapter(plan),
                executor=executor,
                reviewer=CountingAdapter(REVIEW),
                sandbox=sandbox,
                approval=ScopeApproval(("tracked.txt",)),
                run_id="demo-rename",
            )
        self.assertEqual(report.outcome, "failed")
        self.assertEqual(report.errors, ("out_of_scope",))
        self.assertEqual(executor.calls, 1)
        self.assertEqual(sandbox.calls, [])

    def test_paused_resume_rejects_changed_retry_approval_and_root(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from agentic_dev_workflow.orchestration import ScopeApproval, run_orchestration

        sandbox = ScriptedSandbox(
            {1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-mismatch")}}
        )
        saver = InMemorySaver()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            _root, handle = prepare_repository(parent, "demo-mismatch")
            other = parent / "other"
            other.mkdir()
            _other_root, other_handle = prepare_repository(other, "demo-mismatch")
            planner = CountingAdapter(PLAN)
            executor = CountingAdapter(EXECUTOR)
            reviewer = CountingAdapter(REVIEW)
            approval = ScopeApproval(("notes.txt",))
            paused = run_orchestration(
                spec_text="Add a note",
                worktree=handle,
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                approval=approval,
                max_retries=1,
                run_id="demo-mismatch",
                checkpointer=saver,
                interrupt_before=["execute"],
            )
            self.assertEqual(paused.outcome, "paused")
            common = dict(
                spec_text="Add a note",
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                run_id="demo-mismatch",
                checkpointer=saver,
                resume=True,
            )
            with self.assertRaises(ValueError):
                run_orchestration(
                    worktree=handle,
                    approval=approval,
                    max_retries=0,
                    **common,
                )
            with self.assertRaises(ValueError):
                run_orchestration(
                    worktree=handle,
                    approval=ScopeApproval(("other.txt",)),
                    max_retries=1,
                    **common,
                )
            with self.assertRaises(ValueError):
                run_orchestration(
                    worktree=other_handle,
                    approval=approval,
                    max_retries=1,
                    **common,
                )
            self.assertEqual(executor.calls, 0)
            self.assertEqual(sandbox.calls, [])
            finished = run_orchestration(
                worktree=handle,
                approval=approval,
                max_retries=1,
                **common,
            )
        self.assertEqual(finished.outcome, "success")
        self.assertEqual(executor.calls, 1)

    def test_resume_at_verify_requires_executor_completion_evidence(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from agentic_dev_workflow.orchestration import (
            ScopeApproval,
            _build_graph,
            run_orchestration,
        )

        sandbox = ScriptedSandbox(
            {1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-skip-exec")}}
        )
        saver = InMemorySaver()
        with tempfile.TemporaryDirectory() as directory:
            root, handle = prepare_repository(Path(directory), "demo-skip-exec")
            planner = CountingAdapter(PLAN)
            executor = CountingAdapter(EXECUTOR)
            reviewer = CountingAdapter(REVIEW)
            approval = ScopeApproval(("notes.txt",))
            paused = run_orchestration(
                spec_text="Add a note",
                worktree=handle,
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                approval=approval,
                max_retries=0,
                run_id="demo-skip-exec",
                checkpointer=saver,
                interrupt_before=["execute"],
            )
            self.assertEqual(paused.outcome, "paused")
            graph = _build_graph(
                repository_root=root,
                worktree=handle,
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                checkpointer=saver,
                interrupt_before=None,
            )
            graph.update_state(
                {
                    "configurable": {
                        "thread_id": "agentic-dev-workflow:orchestration:demo-skip-exec"
                    }
                },
                {
                    "execution_attempt": 1,
                    "executor_invocations": 1,
                    "_next_action": "verify",
                    "final_status": "running",
                },
                as_node="execute",
            )
            with self.assertRaisesRegex(ValueError, "evidence"):
                run_orchestration(
                    spec_text="Add a note",
                    worktree=handle,
                    planner=planner,
                    executor=executor,
                    reviewer=reviewer,
                    sandbox=sandbox,
                    approval=approval,
                    max_retries=0,
                    run_id="demo-skip-exec",
                    checkpointer=saver,
                    resume=True,
                )
            self.assertEqual(executor.calls, 0)
            self.assertEqual(reviewer.calls, 0)
            self.assertEqual(sandbox.calls, [])

    def test_real_pause_before_verify_can_resume(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from agentic_dev_workflow.orchestration import ScopeApproval, run_orchestration

        sandbox = PassingSandbox()
        saver = InMemorySaver()
        with tempfile.TemporaryDirectory() as directory:
            _root, handle = prepare_repository(Path(directory), "demo-real-verify")
            planner = CountingAdapter(PLAN)
            executor = CountingAdapter(EXECUTOR)
            reviewer = CountingAdapter(REVIEW)
            approval = ScopeApproval(("notes.txt",))
            paused = run_orchestration(
                spec_text="Add a note",
                worktree=handle,
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                approval=approval,
                max_retries=0,
                run_id="demo-real-verify",
                checkpointer=saver,
                interrupt_before=["verify"],
            )
            self.assertEqual(paused.outcome, "paused")
            self.assertEqual(executor.calls, 1)
            self.assertEqual(sandbox.calls, [])
            finished = run_orchestration(
                spec_text="Add a note",
                worktree=handle,
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                approval=approval,
                max_retries=0,
                run_id="demo-real-verify",
                checkpointer=saver,
                resume=True,
            )
            self.assertEqual(finished.outcome, "success")
            self.assertEqual(finished.attempts, 1)
            self.assertEqual(finished.executor_invocations, 1)
            self.assertEqual(executor.calls, 1)
            self.assertEqual(reviewer.calls, 1)
            self.assertEqual(sandbox.calls, [("runner.tests", 1)])

    def test_rewritten_earlier_attempt_history_is_rejected(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from agentic_dev_workflow.orchestration import (
            ScopeApproval,
            _build_graph,
            _initial_state,
            run_orchestration,
        )

        run_id = "demo-history"
        sandbox = ScriptedSandbox(
            {
                1: {
                    CheckId.RUNNER_TESTS: sandbox_result(
                        status=SandboxStatus.FAILED, run_id=run_id
                    )
                },
                2: {CheckId.RUNNER_TESTS: sandbox_result(run_id=run_id)},
            }
        )
        saver = InMemorySaver()
        with tempfile.TemporaryDirectory() as directory:
            root, handle = prepare_repository(Path(directory), run_id)
            planner = CountingAdapter(PLAN)
            executor = CountingAdapter(EXECUTOR)
            reviewer = CountingAdapter(REVIEW)
            approval = ScopeApproval(("notes.txt",))
            graph = _build_graph(
                repository_root=root,
                worktree=handle,
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                checkpointer=saver,
                interrupt_before=["decide"],
            )
            config = {
                "configurable": {
                    "thread_id": f"agentic-dev-workflow:orchestration:{run_id}"
                }
            }
            graph.invoke(
                _initial_state(
                    "Add a note",
                    run_id=run_id,
                    repository_root=root.as_posix(),
                    max_retries=1,
                    approval=approval,
                ),
                config,
            )
            graph.invoke(None, config)
            snapshot = graph.get_state(config)
            self.assertEqual(snapshot.next, ("decide",))
            recorded = [
                entry["verification_outcome"]
                for entry in snapshot.values["attempt_history"]
            ]
            self.assertEqual(recorded, ["fail", "pass"])
            rewritten = [dict(entry) for entry in snapshot.values["attempt_history"]]
            rewritten[0]["verification_outcome"] = "pass"
            rewritten[0]["review_decision"] = "approve"
            graph.update_state(
                config, {"attempt_history": rewritten}, as_node="review"
            )
            executor_calls = executor.calls
            sandbox_calls = list(sandbox.calls)
            with self.assertRaisesRegex(ValueError, "evidence"):
                run_orchestration(
                    spec_text="Add a note",
                    worktree=handle,
                    planner=planner,
                    executor=executor,
                    reviewer=reviewer,
                    sandbox=sandbox,
                    approval=approval,
                    max_retries=1,
                    run_id=run_id,
                    checkpointer=saver,
                    resume=True,
                )
            self.assertEqual(executor.calls, executor_calls)
            self.assertEqual(sandbox.calls, sandbox_calls)
            self.assertEqual(reviewer.calls, 2)

    def test_requests_carry_distinct_invocation_identity_outside_the_prompt(self) -> None:
        sandbox = ScriptedSandbox(
            {
                1: {
                    CheckId.RUNNER_TESTS: sandbox_result(
                        status=SandboxStatus.FAILED, run_id="demo-identity"
                    )
                },
                2: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-identity")},
            }
        )
        _report, planner, executor, reviewer = run_case(
            run_id="demo-identity",
            sandbox=sandbox,
            approval=("notes.txt",),
            max_retries=1,
        )
        self.assertEqual(
            planner.requests[0].correlation_metadata,
            {"role": "planner", "invocation": "1", "invocation_id": "planner:1"},
        )
        self.assertEqual(planner.requests[0].structured_context, {"phase": "plan"})
        self.assertEqual(
            [request.correlation_metadata for request in executor.requests],
            [
                {
                    "role": "executor",
                    "attempt": "1",
                    "invocation": "1",
                    "invocation_id": "executor:1",
                },
                {
                    "role": "executor",
                    "attempt": "2",
                    "invocation": "2",
                    "invocation_id": "executor:2",
                },
            ],
        )
        self.assertEqual(
            [request.correlation_metadata for request in reviewer.requests],
            [
                {
                    "role": "reviewer",
                    "attempt": "1",
                    "invocation": "1",
                    "invocation_id": "reviewer:1",
                },
                {
                    "role": "reviewer",
                    "attempt": "2",
                    "invocation": "2",
                    "invocation_id": "reviewer:2",
                },
            ],
        )
        for agent_request in (*planner.requests, *executor.requests, *reviewer.requests):
            self.assertNotIn("invocation_id", agent_request.structured_context)
            self.assertNotIn("correlation_metadata", agent_request.structured_context)
        self.assertEqual(executor.requests[0].structured_context["attempt"], 1)
        self.assertEqual(executor.requests[1].structured_context["attempt"], 2)
        self.assertIsInstance(executor.requests[0].structured_context["attempt"], int)

    def test_model_executor_diagnostics_do_not_become_workflow_errors(self) -> None:
        from agentic_dev_workflow.orchestration import ScopeApproval, run_orchestration

        for code in ("cursor_timeout", "output_field_invalid", "model_supplied_reason"):
            with self.subTest(code=code):
                sandbox = ScriptedSandbox(
                    {1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-model-diag")}}
                )
                payload = dict(EXECUTOR)
                payload["status"] = "failed"
                payload["failure_category"] = "execution_failure"
                payload["diagnostic_code"] = code
                with tempfile.TemporaryDirectory() as directory:
                    _root, handle = prepare_repository(Path(directory), "demo-model-diag")
                    report = run_orchestration(
                        spec_text="Add a note",
                        worktree=handle,
                        planner=CountingAdapter(PLAN),
                        executor=MockAgentAdapter(structured_output=payload),
                        reviewer=CountingAdapter(REVIEW),
                        sandbox=sandbox,
                        approval=ScopeApproval(("notes.txt",)),
                        run_id="demo-model-diag",
                    )
                self.assertEqual(report.outcome, "error")
                self.assertEqual(report.errors, ("executor_failed",))
                self.assertNotIn(code, json.dumps(report.to_dict()))
                self.assertEqual(sandbox.calls, [])

    def test_adapter_timeout_remains_a_workflow_error(self) -> None:
        from agentic_dev_workflow.orchestration import ScopeApproval, run_orchestration

        sandbox = ScriptedSandbox(
            {1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-adapter-timeout")}}
        )
        with tempfile.TemporaryDirectory() as directory:
            _root, handle = prepare_repository(Path(directory), "demo-adapter-timeout")
            report = run_orchestration(
                spec_text="Add a note",
                worktree=handle,
                planner=CountingAdapter(PLAN),
                executor=MockAgentAdapter(
                    status=AgentStatus.FAILED,
                    failure_category=AgentFailureCategory.TIMEOUT,
                    diagnostic_code="cursor_timeout",
                ),
                reviewer=CountingAdapter(REVIEW),
                sandbox=sandbox,
                approval=ScopeApproval(("notes.txt",)),
                run_id="demo-adapter-timeout",
            )
        self.assertEqual(report.outcome, "error")
        self.assertEqual(report.errors, ("cursor_timeout",))
        self.assertEqual(report.executor_invocations, 1)
        self.assertEqual(sandbox.calls, [])

    def test_invalid_executor_diagnostic_stays_a_validation_error(self) -> None:
        from agentic_dev_workflow.orchestration import ScopeApproval, run_orchestration

        sandbox = ScriptedSandbox(
            {1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="demo-invalid-diag")}}
        )
        payload = dict(EXECUTOR)
        payload["status"] = "failed"
        payload["failure_category"] = "execution_failure"
        payload["diagnostic_code"] = "Not a workflow code"
        with tempfile.TemporaryDirectory() as directory:
            _root, handle = prepare_repository(Path(directory), "demo-invalid-diag")
            report = run_orchestration(
                spec_text="Add a note",
                worktree=handle,
                planner=CountingAdapter(PLAN),
                executor=MockAgentAdapter(structured_output=payload),
                reviewer=CountingAdapter(REVIEW),
                sandbox=sandbox,
                approval=ScopeApproval(("notes.txt",)),
                run_id="demo-invalid-diag",
            )
        self.assertEqual(report.outcome, "error")
        self.assertEqual(report.errors, ("output_field_invalid",))
        self.assertNotIn("Not a workflow code", json.dumps(report.to_dict()))
        self.assertEqual(sandbox.calls, [])


class InlineTaskTests(unittest.TestCase):
    SPEC = "Create greeting.py.\nCover an empty name.\n"

    def test_inline_task_reaches_planning_execution_review_and_retry(self) -> None:
        sandbox = ScriptedSandbox(
            {
                1: {
                    CheckId.RUNNER_TESTS: sandbox_result(
                        status=SandboxStatus.FAILED, run_id="inline-task"
                    )
                },
                2: {CheckId.RUNNER_TESTS: sandbox_result(run_id="inline-task")},
            }
        )
        report, planner, executor, reviewer = run_case(
            run_id="inline-task",
            sandbox=sandbox,
            approval=("notes.txt",),
            max_retries=1,
            spec_text=self.SPEC,
        )
        self.assertEqual(report.outcome, "success")
        self.assertEqual(planner.requests[0].task, self.SPEC.strip())
        self.assertNotIn("SPEC.md", planner.requests[0].task)
        for request in (*executor.requests, *reviewer.requests):
            self.assertEqual(request.structured_context["specification"], self.SPEC)
        self.assertEqual(len(executor.requests), 2)
        self.assertGreaterEqual(len(reviewer.requests), 1)
        self.assertNotIn(self.SPEC, json.dumps(report.to_dict()))

    def test_spec_file_still_supplies_the_task(self) -> None:
        from agentic_dev_workflow.orchestration import load_task_specification

        with tempfile.TemporaryDirectory() as directory:
            spec_path = Path(directory) / "SPEC.md"
            spec_path.write_text(self.SPEC, encoding="utf-8")
            loaded = load_task_specification(spec_path=spec_path)
        self.assertEqual(loaded, self.SPEC)
        sandbox = ScriptedSandbox(
            {1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="spec-file")}}
        )
        _report, planner, executor, _reviewer = run_case(
            run_id="spec-file",
            sandbox=sandbox,
            approval=("notes.txt",),
            spec_text=loaded,
        )
        self.assertEqual(planner.requests[0].task, self.SPEC.strip())
        self.assertEqual(executor.requests[0].structured_context["specification"], self.SPEC)

    def test_missing_and_whitespace_tasks_fail(self) -> None:
        from agentic_dev_workflow.orchestration import load_task_specification

        with self.assertRaises(ValueError):
            load_task_specification()
        with self.assertRaises(ValueError):
            load_task_specification(inline=" \n\t")
        with tempfile.TemporaryDirectory() as directory:
            spec_path = Path(directory) / "SPEC.md"
            spec_path.write_text(" \n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_task_specification(spec_path=spec_path)

    def test_conflicting_inline_and_file_tasks_are_rejected(self) -> None:
        from agentic_dev_workflow.orchestration import load_task_specification

        with tempfile.TemporaryDirectory() as directory:
            spec_path = Path(directory) / "SPEC.md"
            spec_path.write_text(self.SPEC, encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_task_specification(inline="A different task.\n", spec_path=spec_path)
            self.assertNotIn("different task", str(caught.exception))
            self.assertEqual(
                load_task_specification(inline=self.SPEC, spec_path=spec_path),
                self.SPEC,
            )

    def test_inline_task_does_not_widen_approved_scope(self) -> None:
        sandbox = ScriptedSandbox(
            {1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="inline-scope")}}
        )
        report, _planner, executor, reviewer = run_case(
            run_id="inline-scope",
            sandbox=sandbox,
            approval=("notes.txt",),
            extra_file="extra.txt",
            spec_text=self.SPEC,
        )
        self.assertEqual(report.outcome, "failed")
        self.assertEqual(report.errors, ("out_of_scope",))
        self.assertEqual(executor.calls, 1)
        self.assertEqual(reviewer.calls, 0)
        self.assertEqual(sandbox.calls, [])
        summary = report.to_dict()["summary"]
        self.assertEqual(summary["verification_status"], "not_run")
        self.assertEqual(summary["stop_reason"], "out_of_scope")
        self.assertEqual(summary["scope_violations"], ["out_of_scope"])
        self.assertEqual(summary["elapsed_seconds"]["status"], "reported")
        self.assertIsInstance(summary["elapsed_seconds"]["value"], float)
        self.assertEqual(summary["usage"]["workflow"]["counters"]["input_tokens"]["status"], "unavailable")
        self.assertIsNone(summary["usage"]["workflow"]["counters"]["input_tokens"]["value"])
        self.assertNotIn(self.SPEC, json.dumps(summary))

    def test_checkpoint_keeps_the_inline_specification(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from agentic_dev_workflow.orchestration import ScopeApproval, run_orchestration

        sandbox = ScriptedSandbox(
            {1: {CheckId.RUNNER_TESTS: sandbox_result(run_id="inline-checkpoint")}}
        )
        saver = InMemorySaver()
        with tempfile.TemporaryDirectory() as directory:
            _root, handle = prepare_repository(Path(directory), "inline-checkpoint")
            planner = CountingAdapter(PLAN)
            executor = CountingAdapter(EXECUTOR)
            reviewer = CountingAdapter(REVIEW)
            approval = ScopeApproval(("notes.txt",))
            paused = run_orchestration(
                spec_text=self.SPEC,
                worktree=handle,
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                approval=approval,
                run_id="inline-checkpoint",
                checkpointer=saver,
                interrupt_before=["execute"],
            )
            self.assertEqual(paused.outcome, "paused")
            with self.assertRaises(ValueError) as caught:
                run_orchestration(
                    spec_text="A different task.\n",
                    worktree=handle,
                    planner=planner,
                    executor=executor,
                    reviewer=reviewer,
                    sandbox=sandbox,
                    approval=approval,
                    run_id="inline-checkpoint",
                    checkpointer=saver,
                    resume=True,
                )
            self.assertIn("specification", str(caught.exception))
            finished = run_orchestration(
                spec_text=self.SPEC,
                worktree=handle,
                planner=planner,
                executor=executor,
                reviewer=reviewer,
                sandbox=sandbox,
                approval=approval,
                run_id="inline-checkpoint",
                checkpointer=saver,
                resume=True,
            )
        self.assertEqual(finished.outcome, "success")
        self.assertEqual(executor.requests[0].structured_context["specification"], self.SPEC)
        self.assertNotIn(self.SPEC, json.dumps(finished.to_dict()))


if __name__ == "__main__":
    unittest.main()
