"""Mocked orchestration demonstration.

Calls ``run_orchestration`` with deterministic mock agents and a scripted
in-memory sandbox against temporary Git repositories. This is not production
execution, not a model call, and not OS-level sandbox isolation.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from agentic_dev_workflow.adapters import AgentResult, MockAgentAdapter
from agentic_dev_workflow.orchestration import ScopeApproval, run_orchestration
from agentic_dev_workflow.sandbox import (
    RUNNER_RUNTIME_PROFILE_ID,
    SandboxCheckResult,
    SandboxCleanupStatus,
    SandboxFailureCategory,
    SandboxStatus,
    SandboxTermination,
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
SPEC_TEXT = "Add a note"


class CountingAdapter:
    """Counts invoke calls and returns one fixed mock response."""

    adapter_type = "mock"

    def __init__(self, structured_output: dict[str, object]) -> None:
        self.inner = MockAgentAdapter(structured_output=structured_output)
        self.calls = 0

    def invoke(self, request: object) -> AgentResult:
        self.calls += 1
        return self.inner.invoke(request)


class ScriptedSandbox:
    """Returns a prepared sandbox result for each attempt. No process is launched."""

    runtime_profile_id = RUNNER_RUNTIME_PROFILE_ID

    def __init__(self, outcomes_by_attempt: dict[int, SandboxStatus]) -> None:
        self.outcomes_by_attempt = outcomes_by_attempt
        self.calls: list[tuple[str, int]] = []

    def run(self, check, worktree, *, run_id: str, attempt: int) -> SandboxCheckResult:
        del worktree
        self.calls.append((check.check_id.value, attempt))
        status = self.outcomes_by_attempt[attempt]
        failed = status == SandboxStatus.FAILED
        outcome = SandboxCheckResult(
            run_id=run_id,
            check_id=check.check_id,
            attempt=attempt,
            status=status,
            exit_code=1 if failed else 0,
            duration_seconds=0.25,
            output_bytes_observed=0,
            termination=SandboxTermination.NATURAL_EXIT,
            stdout_evidence=b"",
            stderr_evidence=b"",
            cleanup_status=SandboxCleanupStatus.SUCCEEDED,
            failure_category=SandboxFailureCategory.NONZERO_EXIT if failed else None,
            diagnostic_code=None,
            backend_type="fake",
        )
        return replace(outcome, run_id=run_id, attempt=attempt)


def git(repository: Path, *args: str) -> str:
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
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
        env=environment,
    )
    return result.stdout.strip()


def prepare_worktree(parent: Path, run_id: str) -> VerifiedWorktree:
    source = parent / "source"
    source.mkdir(exist_ok=True)
    worktree = parent / f"agentic-workflow-{run_id}"
    worktree.mkdir()
    git(parent, "init", "-q", str(worktree))
    git(worktree, "config", "user.name", "Demo")
    git(worktree, "config", "user.email", "demo@example.invalid")
    (worktree / "tracked.txt").write_text("base\n", encoding="utf-8")
    git(worktree, "add", "tracked.txt")
    git(worktree, "commit", "-qm", "base")
    (worktree / "notes.txt").write_text("note\n", encoding="utf-8")
    canonical = worktree.resolve()
    return VerifiedWorktree.from_manager_verification(
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


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def present(scenario: str, report) -> dict[str, object]:
    payload = report.to_dict()
    return {
        "scenario": scenario,
        "outcome": payload["outcome"],
        "attempts": payload["attempts"],
        "executor_invocations": payload["executor_invocations"],
        "repair_retries_used": payload["repair_retries_used"],
        "review_decision": payload["review_decision"],
        "verification_results": payload["checks"],
        "errors": payload["errors"],
        "attempt_history": payload["attempt_history"],
    }


def run_case(
    parent: Path,
    *,
    scenario: str,
    run_id: str,
    outcomes: dict[int, SandboxStatus],
    max_retries: int,
) -> dict[str, object]:
    handle = prepare_worktree(parent, run_id)
    planner = CountingAdapter(PLAN)
    executor = CountingAdapter(EXECUTOR)
    reviewer = CountingAdapter(REVIEW)
    sandbox = ScriptedSandbox(outcomes)
    report = run_orchestration(
        SPEC_TEXT,
        worktree=handle,
        planner=planner,
        executor=executor,
        reviewer=reviewer,
        sandbox=sandbox,
        approval=ScopeApproval(("notes.txt",)),
        max_retries=max_retries,
        run_id=run_id,
    )
    expect(planner.calls == 1, f"{scenario}: planner ran {planner.calls} times")
    expect(
        executor.calls == report.executor_invocations,
        f"{scenario}: executor calls {executor.calls} "
        f"do not match reported invocations {report.executor_invocations}",
    )
    expect(
        [check_id for check_id, _attempt in sandbox.calls] == [CheckId.RUNNER_TESTS.value]
        * len(sandbox.calls),
        f"{scenario}: unexpected sandbox checks {sandbox.calls}",
    )
    expect(
        [attempt for _check_id, attempt in sandbox.calls]
        == list(range(1, len(sandbox.calls) + 1)),
        f"{scenario}: unexpected sandbox attempts {sandbox.calls}",
    )
    return present(scenario, report)


def run_scenarios() -> list[dict[str, object]]:
    with tempfile.TemporaryDirectory() as directory:
        parent = Path(directory)
        success = run_case(
            parent,
            scenario="success",
            run_id="demo-success",
            outcomes={1: SandboxStatus.PASSED},
            max_retries=0,
        )
        expect(success["outcome"] == "success", "success scenario did not pass")
        expect(success["attempts"] == 1, "success scenario did not verify once")
        expect(
            success["executor_invocations"] == 1,
            "success scenario did not invoke the executor once",
        )
        expect(success["repair_retries_used"] == 0, "success scenario used a repair")
        expect(success["review_decision"] == "approve", "success scenario was not approved")
        expect(success["errors"] == [], "success scenario reported errors")

        repair = run_case(
            parent,
            scenario="repair_then_success",
            run_id="demo-repair",
            outcomes={1: SandboxStatus.FAILED, 2: SandboxStatus.PASSED},
            max_retries=1,
        )
        expect(repair["outcome"] == "success", "repair scenario did not pass")
        expect(repair["attempts"] == 2, "repair scenario did not verify twice")
        expect(
            repair["executor_invocations"] == 2,
            "repair scenario did not invoke the executor twice",
        )
        expect(repair["repair_retries_used"] == 1, "repair scenario did not use one repair")
        expect(
            [entry["verification_outcome"] for entry in repair["attempt_history"]]
            == ["fail", "pass"],
            "repair scenario history is not fail then pass",
        )

        exhausted = run_case(
            parent,
            scenario="retry_exhaustion",
            run_id="demo-exhausted",
            outcomes={1: SandboxStatus.FAILED, 2: SandboxStatus.FAILED},
            max_retries=1,
        )
        expect(exhausted["outcome"] == "failed", "exhaustion scenario did not fail")
        expect(exhausted["attempts"] == 2, "exhaustion scenario did not stop after two attempts")
        expect(
            exhausted["executor_invocations"] == 2,
            "exhaustion scenario invoked the executor more than twice",
        )
        expect(
            exhausted["review_decision"] == "approve",
            "exhaustion scenario should still show the reviewer approval",
        )
        expect(
            exhausted["errors"] == ["verification_failed", "retries_exhausted"],
            "exhaustion scenario did not report retry exhaustion",
        )
        expect(
            [entry["verification_outcome"] for entry in exhausted["attempt_history"]]
            == ["fail", "fail"],
            "exhaustion scenario history is not two failures",
        )
        return [success, repair, exhausted]


def main() -> int:
    for report in run_scenarios():
        print(json.dumps(report, indent=2))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
