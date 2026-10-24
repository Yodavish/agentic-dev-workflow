import pickle
import subprocess
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from agentic_dev_workflow.verification_registry import CheckId, get_check_definition
from agentic_dev_workflow.worktrees import WorktreeOperationResult
from agentic_dev_workflow.sandbox import (
    CLEANUP_GRACE_SECONDS,
    MAX_IN_MEMORY_EVIDENCE_BYTES,
    MAX_PROCESS_OUTPUT_CHUNK_BYTES,
    SandboxCheckResult,
    SandboxCleanupStatus,
    SandboxDiagnosticCode,
    SandboxFailureCategory,
    SandboxRunnerError,
    SandboxStatus,
    SandboxTermination,
    UnavailableSandboxRunner,
    VerifiedWorktree,
    DeterministicFakeSandboxRunner,
    invoke_sandbox_runner,
    safe_sandbox_telemetry_attributes,
)


def worktree_result(**changes):
    values = {
        "operation": "verify",
        "success": True,
        "run_id": "run-123",
        "source_repository": "/tmp/fixtures/source",
        "worktree_path": "/tmp/fixtures/worktrees/agentic-workflow-run-123",
        "base_commit": "a" * 40,
        "partial_state": False,
        "issues": (),
    }
    values.update(changes)
    return WorktreeOperationResult(**values)


def verified_worktree(**changes):
    return VerifiedWorktree.from_manager_verification(worktree_result(**changes))


def result(
    *,
    check_id=CheckId.RUNNER_TESTS,
    run_id="run-123",
    attempt=1,
    status=SandboxStatus.PASSED,
    exit_code=0,
    failure_category=None,
    diagnostic_code=None,
    cleanup_status=SandboxCleanupStatus.SUCCEEDED,
    termination=SandboxTermination.NATURAL_EXIT,
    duration_seconds=0.25,
    output_bytes_observed=0,
    stdout_evidence=b"",
    stderr_evidence=b"",
):
    return SandboxCheckResult(
        run_id=run_id,
        check_id=check_id,
        attempt=attempt,
        status=status,
        exit_code=exit_code,
        duration_seconds=duration_seconds,
        output_bytes_observed=output_bytes_observed,
        termination=termination,
        stdout_evidence=stdout_evidence,
        stderr_evidence=stderr_evidence,
        cleanup_status=cleanup_status,
        failure_category=failure_category,
        diagnostic_code=diagnostic_code,
        backend_type="fake",
    )


class SandboxContractTests(unittest.TestCase):
    def test_verified_worktree_handle_requires_successful_manager_verification(self) -> None:
        handle = verified_worktree()
        self.assertEqual(handle.run_id, "run-123")
        self.assertEqual(handle.path, "/tmp/fixtures/worktrees/agentic-workflow-run-123")
        with self.assertRaises(SandboxRunnerError):
            verified_worktree(success=False)
        with self.assertRaises(SandboxRunnerError):
            verified_worktree(operation="create")
        with self.assertRaises(SandboxRunnerError):
            verified_worktree(run_id="another-run")

    def test_result_only_passes_for_clean_natural_zero_exit(self) -> None:
        self.assertEqual(result().status, SandboxStatus.PASSED)
        with self.assertRaises(ValueError):
            result(termination=SandboxTermination.TIMEOUT)
        with self.assertRaises(ValueError):
            result(exit_code=1)
        with self.assertRaises(ValueError):
            result(status=SandboxStatus.FAILED, exit_code=0,
                   failure_category=SandboxFailureCategory.NONZERO_EXIT)
        with self.assertRaises(ValueError):
            result(status=SandboxStatus.FAILED, exit_code=1,
                   failure_category=SandboxFailureCategory.TIMEOUT)
        with self.assertRaises(ValueError):
            result(status=SandboxStatus.ERROR, exit_code=0)

    def test_fake_runner_represents_nonzero_timeout_overflow_launch_and_cleanup_failures(self) -> None:
        check = get_check_definition(CheckId.RUNNER_TESTS)
        outcomes = {
            CheckId.RUNNER_TESTS: result(),
            CheckId.RUNNER_COMPILE: result(
                check_id=CheckId.RUNNER_COMPILE,
                status=SandboxStatus.FAILED,
                exit_code=1,
                failure_category=SandboxFailureCategory.NONZERO_EXIT,
            ),
            CheckId.RUNNER_DIFF_CHECK: result(
                check_id=CheckId.RUNNER_DIFF_CHECK,
                status=SandboxStatus.ERROR,
                exit_code=None,
                failure_category=SandboxFailureCategory.TIMEOUT,
                diagnostic_code=SandboxDiagnosticCode.CHECK_TIMED_OUT,
                termination=SandboxTermination.TIMEOUT,
            ),
            CheckId.RUNNER_LOCK_CHECK: result(
                check_id=CheckId.RUNNER_LOCK_CHECK,
                status=SandboxStatus.ERROR,
                exit_code=None,
                failure_category=SandboxFailureCategory.OUTPUT_LIMIT,
                diagnostic_code=SandboxDiagnosticCode.OUTPUT_LIMIT_EXCEEDED,
                termination=SandboxTermination.OUTPUT_LIMIT,
                output_bytes_observed=(
                    get_check_definition(CheckId.RUNNER_LOCK_CHECK).max_output_bytes + 1
                ),
            ),
        }
        fake = DeterministicFakeSandboxRunner(outcomes)
        for check_id, expected_status in (
            (CheckId.RUNNER_TESTS, SandboxStatus.PASSED),
            (CheckId.RUNNER_COMPILE, SandboxStatus.FAILED),
            (CheckId.RUNNER_DIFF_CHECK, SandboxStatus.ERROR),
            (CheckId.RUNNER_LOCK_CHECK, SandboxStatus.ERROR),
        ):
            actual = invoke_sandbox_runner(
                fake, get_check_definition(check_id), verified_worktree(), run_id="run-123", attempt=1
            )
            self.assertEqual(actual.status, expected_status)

        launch_failure = result(
            status=SandboxStatus.ERROR,
            exit_code=None,
            failure_category=SandboxFailureCategory.LAUNCH_FAILURE,
            diagnostic_code=SandboxDiagnosticCode.RUNTIME_LAUNCH_FAILED,
            cleanup_status=SandboxCleanupStatus.NOT_REQUIRED,
            termination=SandboxTermination.LAUNCH_FAILURE,
        )
        cleanup_failure = result(
            status=SandboxStatus.ERROR,
            exit_code=0,
            failure_category=SandboxFailureCategory.CLEANUP_FAILURE,
            diagnostic_code=SandboxDiagnosticCode.CLEANUP_UNCONFIRMED,
            cleanup_status=SandboxCleanupStatus.FAILED,
        )
        for failure_result in (launch_failure, cleanup_failure):
            self.assertEqual(
                invoke_sandbox_runner(
                    DeterministicFakeSandboxRunner({CheckId.RUNNER_TESTS: failure_result}),
                    get_check_definition(CheckId.RUNNER_TESTS),
                    verified_worktree(),
                    run_id="run-123",
                    attempt=1,
                ).status,
                SandboxStatus.ERROR,
            )

    def test_unavailable_runner_returns_fixed_error_without_launching_processes(self) -> None:
        with patch("subprocess.Popen") as popen, patch("subprocess.run") as run:
            outcome = invoke_sandbox_runner(
                UnavailableSandboxRunner(),
                get_check_definition(CheckId.RUNNER_TESTS),
                verified_worktree(),
                run_id="run-123",
                attempt=1,
            )
        popen.assert_not_called()
        run.assert_not_called()
        self.assertEqual(outcome.status, SandboxStatus.ERROR)
        self.assertEqual(outcome.failure_category, SandboxFailureCategory.SANDBOX_UNAVAILABLE)
        self.assertEqual(outcome.diagnostic_code, SandboxDiagnosticCode.SANDBOX_NOT_CONFIGURED)

    def test_unregistered_or_modified_checks_are_rejected_before_runner_call(self) -> None:
        class RecordingRunner:
            runtime_profile_id = "runner-verification-v1"

            def __init__(self):
                self.called = False

            def run(self, check, worktree, *, run_id, attempt):
                self.called = True
                return result(check_id=check.check_id, run_id=run_id, attempt=attempt)

        runner = RecordingRunner()

        changed = replace(
            get_check_definition(CheckId.RUNNER_TESTS),
            commands=(("/tmp/untrusted-python", "-V"),),
        )
        with self.assertRaises(SandboxRunnerError):
            invoke_sandbox_runner(
                runner, changed, verified_worktree(), run_id="run-123", attempt=1
            )
        self.assertFalse(runner.called)

    def test_result_correlation_and_bad_runner_output_fail_with_safe_diagnostics(self) -> None:
        class BadRunner:
            runtime_profile_id = "runner-verification-v1"

            def __init__(self, returned):
                self.returned = returned

            def run(self, check, worktree, *, run_id, attempt):
                return self.returned

        for returned in (
            object(),
            result(run_id="other-run"),
            result(check_id=CheckId.RUNNER_COMPILE),
            result(attempt=2),
        ):
            with self.subTest(returned=returned), self.assertRaises(SandboxRunnerError) as caught:
                invoke_sandbox_runner(
                    BadRunner(returned), get_check_definition(CheckId.RUNNER_TESTS),
                    verified_worktree(), run_id="run-123", attempt=1,
                )
            self.assertNotIn("other-run", str(caught.exception))

    def test_runner_exception_text_is_not_returned(self) -> None:
        class RaisingRunner:
            runtime_profile_id = "runner-verification-v1"

            def run(self, check, worktree, *, run_id, attempt):
                raise RuntimeError("synthetic-command-output-secret")

        with self.assertRaises(SandboxRunnerError) as caught:
            invoke_sandbox_runner(
                RaisingRunner(), get_check_definition(CheckId.RUNNER_TESTS),
                verified_worktree(), run_id="run-123", attempt=1,
            )
        self.assertNotIn("synthetic-command-output-secret", str(caught.exception))
        self.assertEqual(
            caught.exception.diagnostic_code,
            SandboxDiagnosticCode.RUNTIME_LAUNCH_FAILED,
        )

    def test_failures_reject_arbitrary_diagnostic_text_and_bound_evidence(self) -> None:
        with self.assertRaises(ValueError):
            result(
                status=SandboxStatus.ERROR,
                exit_code=None,
                failure_category=SandboxFailureCategory.LAUNCH_FAILURE,
                diagnostic_code="secret-command-output",
                cleanup_status=SandboxCleanupStatus.NOT_REQUIRED,
                termination=SandboxTermination.LAUNCH_FAILURE,
            )
        with self.assertRaises(ValueError):
            result(stdout_evidence=b"x" * (MAX_IN_MEMORY_EVIDENCE_BYTES + 1))
        oversized_observation = get_check_definition(CheckId.RUNNER_TESTS).max_output_bytes + 65_537
        with self.assertRaises(ValueError):
            result(
                status=SandboxStatus.ERROR,
                exit_code=None,
                failure_category=SandboxFailureCategory.OUTPUT_LIMIT,
                diagnostic_code=SandboxDiagnosticCode.OUTPUT_LIMIT_EXCEEDED,
                termination=SandboxTermination.OUTPUT_LIMIT,
                output_bytes_observed=oversized_observation,
            )
        too_slow = result(
            status=SandboxStatus.ERROR,
            exit_code=None,
            failure_category=SandboxFailureCategory.TIMEOUT,
            diagnostic_code=SandboxDiagnosticCode.CHECK_TIMED_OUT,
            termination=SandboxTermination.TIMEOUT,
            duration_seconds=301 + CLEANUP_GRACE_SECONDS,
        )
        with self.assertRaises(SandboxRunnerError):
            invoke_sandbox_runner(
                DeterministicFakeSandboxRunner({
                    CheckId.RUNNER_TESTS: result(
                        status=SandboxStatus.ERROR,
                        exit_code=None,
                        failure_category=SandboxFailureCategory.TIMEOUT,
                        diagnostic_code=SandboxDiagnosticCode.CHECK_TIMED_OUT,
                        termination=SandboxTermination.TIMEOUT,
                        duration_seconds=too_slow.duration_seconds,
                    )
                }),
                get_check_definition(CheckId.RUNNER_TESTS), verified_worktree(),
                run_id="run-123", attempt=1,
            )

    def test_command_evidence_is_not_in_repr_or_telemetry(self) -> None:
        secret_output = b"synthetic-secret-and-command-output"
        outcome = result(stdout_evidence=secret_output, output_bytes_observed=len(secret_output))
        self.assertNotIn(secret_output.decode(), repr(outcome))
        attributes = safe_sandbox_telemetry_attributes(outcome)
        serialized = repr(attributes)
        self.assertNotIn(secret_output.decode(), serialized)
        self.assertEqual(attributes["verification.check_id"], "runner.tests")
        self.assertEqual(attributes["verification.status"], "passed")

    def test_result_is_immutable_and_duration_is_bounded(self) -> None:
        outcome = result()
        with self.assertRaises(FrozenInstanceError):
            outcome.exit_code = 2
        with self.assertRaises(ValueError):
            result(duration_seconds=-1)


def forged_result(
    *,
    check_id=CheckId.RUNNER_TESTS,
    run_id="run-123",
    attempt=1,
    status=SandboxStatus.PASSED,
    exit_code=0,
    failure_category=None,
    diagnostic_code=None,
    cleanup_status=SandboxCleanupStatus.SUCCEEDED,
    termination=SandboxTermination.NATURAL_EXIT,
    duration_seconds=0.25,
    output_bytes_observed=0,
    stdout_evidence=b"",
    stderr_evidence=b"",
    backend_type="fake",
):
    """Build a result without validation, simulating a defective backend."""
    forged = object.__new__(SandboxCheckResult)
    for name, value in (
        ("run_id", run_id),
        ("check_id", check_id),
        ("attempt", attempt),
        ("status", status),
        ("exit_code", exit_code),
        ("duration_seconds", duration_seconds),
        ("output_bytes_observed", output_bytes_observed),
        ("termination", termination),
        ("stdout_evidence", stdout_evidence),
        ("stderr_evidence", stderr_evidence),
        ("cleanup_status", cleanup_status),
        ("failure_category", failure_category),
        ("diagnostic_code", diagnostic_code),
        ("backend_type", backend_type),
    ):
        object.__setattr__(forged, name, value)
    return forged


class SandboxOutputCapTests(unittest.TestCase):
    """The registered output cap is a property of the result type itself."""

    def test_passed_result_at_the_registered_cap_is_constructible(self) -> None:
        for check_id in CheckId:
            with self.subTest(check_id=check_id):
                cap = get_check_definition(check_id).max_output_bytes
                outcome = result(check_id=check_id, output_bytes_observed=cap)
                self.assertEqual(outcome.status, SandboxStatus.PASSED)
                self.assertEqual(outcome.output_bytes_observed, cap)

    def test_direct_construction_rejects_passed_result_above_the_registered_cap(
        self,
    ) -> None:
        for check_id in CheckId:
            with self.subTest(check_id=check_id), self.assertRaises(ValueError):
                cap = get_check_definition(check_id).max_output_bytes
                result(check_id=check_id, output_bytes_observed=cap + 1)

    def test_output_cap_is_check_specific_rather_than_global(self) -> None:
        compile_cap = get_check_definition(CheckId.RUNNER_COMPILE).max_output_bytes
        tests_cap = get_check_definition(CheckId.RUNNER_TESTS).max_output_bytes
        self.assertLess(compile_cap + 1, tests_cap)
        with self.assertRaises(ValueError):
            result(check_id=CheckId.RUNNER_COMPILE, output_bytes_observed=compile_cap + 1)
        accepted = result(
            check_id=CheckId.RUNNER_TESTS, output_bytes_observed=compile_cap + 1
        )
        self.assertEqual(accepted.status, SandboxStatus.PASSED)
        self.assertEqual(accepted.output_bytes_observed, compile_cap + 1)

    def test_failed_result_above_the_registered_cap_is_rejected(self) -> None:
        cap = get_check_definition(CheckId.RUNNER_COMPILE).max_output_bytes
        with self.assertRaises(ValueError):
            result(
                check_id=CheckId.RUNNER_COMPILE,
                status=SandboxStatus.FAILED,
                exit_code=1,
                failure_category=SandboxFailureCategory.NONZERO_EXIT,
                output_bytes_observed=cap + 1,
            )

    def test_replacing_a_result_cannot_bypass_the_registered_cap(self) -> None:
        cap = get_check_definition(CheckId.RUNNER_COMPILE).max_output_bytes
        outcome = result(check_id=CheckId.RUNNER_COMPILE)
        with self.assertRaises(ValueError):
            replace(outcome, output_bytes_observed=cap + 1)

    def test_output_limit_error_result_may_report_bytes_above_the_cap(self) -> None:
        cap = get_check_definition(CheckId.RUNNER_LOCK_CHECK).max_output_bytes
        outcome = result(
            check_id=CheckId.RUNNER_LOCK_CHECK,
            status=SandboxStatus.ERROR,
            exit_code=None,
            failure_category=SandboxFailureCategory.OUTPUT_LIMIT,
            diagnostic_code=SandboxDiagnosticCode.OUTPUT_LIMIT_EXCEEDED,
            termination=SandboxTermination.OUTPUT_LIMIT,
            output_bytes_observed=cap + 1,
        )
        self.assertEqual(outcome.output_bytes_observed, cap + 1)
        self.assertEqual(outcome.failure_category, SandboxFailureCategory.OUTPUT_LIMIT)

    def test_observed_output_is_bounded_by_the_cap_plus_one_output_chunk(self) -> None:
        cap = get_check_definition(CheckId.RUNNER_COMPILE).max_output_bytes
        at_limit = result(
            check_id=CheckId.RUNNER_COMPILE,
            status=SandboxStatus.ERROR,
            exit_code=None,
            failure_category=SandboxFailureCategory.OUTPUT_LIMIT,
            diagnostic_code=SandboxDiagnosticCode.OUTPUT_LIMIT_EXCEEDED,
            termination=SandboxTermination.OUTPUT_LIMIT,
            output_bytes_observed=cap + MAX_PROCESS_OUTPUT_CHUNK_BYTES,
        )
        self.assertEqual(
            at_limit.output_bytes_observed, cap + MAX_PROCESS_OUTPUT_CHUNK_BYTES
        )
        with self.assertRaises(ValueError):
            result(
                check_id=CheckId.RUNNER_COMPILE,
                status=SandboxStatus.ERROR,
                exit_code=None,
                failure_category=SandboxFailureCategory.OUTPUT_LIMIT,
                diagnostic_code=SandboxDiagnosticCode.OUTPUT_LIMIT_EXCEEDED,
                termination=SandboxTermination.OUTPUT_LIMIT,
                output_bytes_observed=cap + MAX_PROCESS_OUTPUT_CHUNK_BYTES + 1,
            )

    def test_runner_invocation_accepts_passed_result_at_the_registered_cap(self) -> None:
        cap = get_check_definition(CheckId.RUNNER_COMPILE).max_output_bytes
        outcome = invoke_sandbox_runner(
            DeterministicFakeSandboxRunner(
                {
                    CheckId.RUNNER_COMPILE: result(
                        check_id=CheckId.RUNNER_COMPILE, output_bytes_observed=cap
                    )
                }
            ),
            get_check_definition(CheckId.RUNNER_COMPILE),
            verified_worktree(),
            run_id="run-123",
            attempt=1,
        )
        self.assertEqual(outcome.status, SandboxStatus.PASSED)
        self.assertEqual(outcome.output_bytes_observed, cap)

    def test_runner_invocation_rejects_a_passed_result_above_the_registered_cap(
        self,
    ) -> None:
        cap = get_check_definition(CheckId.RUNNER_COMPILE).max_output_bytes

        class ForgedRunner:
            runtime_profile_id = "runner-verification-v1"

            def __init__(self, returned):
                self.returned = returned

            def run(self, check, worktree, *, run_id, attempt):
                return self.returned

        forged = forged_result(
            check_id=CheckId.RUNNER_COMPILE, output_bytes_observed=cap + 1
        )
        with self.assertRaises(SandboxRunnerError) as caught:
            invoke_sandbox_runner(
                ForgedRunner(forged),
                get_check_definition(CheckId.RUNNER_COMPILE),
                verified_worktree(),
                run_id="run-123",
                attempt=1,
            )
        self.assertEqual(
            caught.exception.diagnostic_code, SandboxDiagnosticCode.RESULT_INVALID
        )

    def test_runner_invocation_reports_over_cap_output_as_infrastructure_error(
        self,
    ) -> None:
        cap = get_check_definition(CheckId.RUNNER_COMPILE).max_output_bytes
        outcome = invoke_sandbox_runner(
            DeterministicFakeSandboxRunner(
                {
                    CheckId.RUNNER_COMPILE: result(
                        check_id=CheckId.RUNNER_COMPILE,
                        status=SandboxStatus.ERROR,
                        exit_code=None,
                        failure_category=SandboxFailureCategory.OUTPUT_LIMIT,
                        diagnostic_code=SandboxDiagnosticCode.OUTPUT_LIMIT_EXCEEDED,
                        termination=SandboxTermination.OUTPUT_LIMIT,
                        output_bytes_observed=cap + 1,
                    )
                }
            ),
            get_check_definition(CheckId.RUNNER_COMPILE),
            verified_worktree(),
            run_id="run-123",
            attempt=1,
        )
        self.assertEqual(outcome.status, SandboxStatus.ERROR)
        self.assertEqual(outcome.failure_category, SandboxFailureCategory.OUTPUT_LIMIT)
        self.assertEqual(outcome.output_bytes_observed, cap + 1)

    def test_runner_invocation_rejects_a_backend_that_reports_an_over_cap_pass(
        self,
    ) -> None:
        cap = get_check_definition(CheckId.RUNNER_COMPILE).max_output_bytes

        class OverCapRunner:
            runtime_profile_id = "runner-verification-v1"

            def run(self, check, worktree, *, run_id, attempt):
                return result(
                    check_id=check.check_id,
                    run_id=run_id,
                    attempt=attempt,
                    output_bytes_observed=cap + 1,
                )

        with self.assertRaises(SandboxRunnerError) as caught:
            invoke_sandbox_runner(
                OverCapRunner(),
                get_check_definition(CheckId.RUNNER_COMPILE),
                verified_worktree(),
                run_id="run-123",
                attempt=1,
            )
        self.assertEqual(
            caught.exception.diagnostic_code, SandboxDiagnosticCode.RESULT_INVALID
        )


def error_result(
    *,
    check_id=CheckId.RUNNER_COMPILE,
    failure_category,
    diagnostic_code,
    termination=SandboxTermination.NOT_STARTED,
    exit_code=None,
    cleanup_status=SandboxCleanupStatus.NOT_REQUIRED,
    output_bytes_observed=0,
):
    """Build an error result for one failure category."""
    return result(
        check_id=check_id,
        status=SandboxStatus.ERROR,
        exit_code=exit_code,
        failure_category=failure_category,
        diagnostic_code=diagnostic_code,
        termination=termination,
        cleanup_status=cleanup_status,
        output_bytes_observed=output_bytes_observed,
    )


class SandboxOutputCategoryTests(unittest.TestCase):
    """Over-cap observations and output-limit failures must agree."""

    def boundary_accepts(self, outcome) -> bool:
        class FixedRunner:
            runtime_profile_id = "runner-verification-v1"

            def run(self, check, worktree, *, run_id, attempt):
                return outcome

        try:
            invoke_sandbox_runner(
                FixedRunner(),
                get_check_definition(CheckId.RUNNER_COMPILE),
                verified_worktree(),
                run_id="run-123",
                attempt=1,
            )
            return True
        except SandboxRunnerError:
            return False

    def test_over_cap_observation_requires_the_output_limit_category(self) -> None:
        cap = get_check_definition(CheckId.RUNNER_COMPILE).max_output_bytes
        with self.assertRaises(ValueError):
            error_result(
                failure_category=SandboxFailureCategory.TIMEOUT,
                diagnostic_code=SandboxDiagnosticCode.CHECK_TIMED_OUT,
                termination=SandboxTermination.TIMEOUT,
                output_bytes_observed=cap + 1,
            )
        with self.assertRaises(ValueError):
            error_result(
                failure_category=SandboxFailureCategory.CLEANUP_FAILURE,
                diagnostic_code=SandboxDiagnosticCode.CLEANUP_UNCONFIRMED,
                exit_code=0,
                termination=SandboxTermination.NATURAL_EXIT,
                cleanup_status=SandboxCleanupStatus.FAILED,
                output_bytes_observed=cap + 1,
            )
        accepted = error_result(
            failure_category=SandboxFailureCategory.OUTPUT_LIMIT,
            diagnostic_code=SandboxDiagnosticCode.OUTPUT_LIMIT_EXCEEDED,
            termination=SandboxTermination.OUTPUT_LIMIT,
            output_bytes_observed=cap + 1,
        )
        self.assertEqual(accepted.failure_category, SandboxFailureCategory.OUTPUT_LIMIT)
        self.assertEqual(accepted.output_bytes_observed, cap + 1)

    def test_output_limit_category_requires_an_over_cap_observation(self) -> None:
        cap = get_check_definition(CheckId.RUNNER_COMPILE).max_output_bytes
        for observed in (0, cap - 1, cap):
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                error_result(
                    failure_category=SandboxFailureCategory.OUTPUT_LIMIT,
                    diagnostic_code=SandboxDiagnosticCode.OUTPUT_LIMIT_EXCEEDED,
                    termination=SandboxTermination.OUTPUT_LIMIT,
                    output_bytes_observed=observed,
                )

    def test_at_cap_observations_with_other_failure_categories_remain_valid(self) -> None:
        cap = get_check_definition(CheckId.RUNNER_COMPILE).max_output_bytes
        timeout = error_result(
            failure_category=SandboxFailureCategory.TIMEOUT,
            diagnostic_code=SandboxDiagnosticCode.CHECK_TIMED_OUT,
            termination=SandboxTermination.TIMEOUT,
            output_bytes_observed=cap,
        )
        cleanup = error_result(
            failure_category=SandboxFailureCategory.CLEANUP_FAILURE,
            diagnostic_code=SandboxDiagnosticCode.CLEANUP_UNCONFIRMED,
            exit_code=0,
            termination=SandboxTermination.NATURAL_EXIT,
            cleanup_status=SandboxCleanupStatus.FAILED,
            output_bytes_observed=cap,
        )
        self.assertEqual(timeout.output_bytes_observed, cap)
        self.assertEqual(cleanup.output_bytes_observed, cap)

    def test_replacing_a_result_cannot_bypass_output_limit_consistency(self) -> None:
        cap = get_check_definition(CheckId.RUNNER_COMPILE).max_output_bytes
        timeout = error_result(
            failure_category=SandboxFailureCategory.TIMEOUT,
            diagnostic_code=SandboxDiagnosticCode.CHECK_TIMED_OUT,
            termination=SandboxTermination.TIMEOUT,
            output_bytes_observed=cap,
        )
        with self.assertRaises(ValueError):
            replace(timeout, output_bytes_observed=cap + 1)
        overflow = error_result(
            failure_category=SandboxFailureCategory.OUTPUT_LIMIT,
            diagnostic_code=SandboxDiagnosticCode.OUTPUT_LIMIT_EXCEEDED,
            termination=SandboxTermination.OUTPUT_LIMIT,
            output_bytes_observed=cap + 1,
        )
        with self.assertRaises(ValueError):
            replace(overflow, output_bytes_observed=cap)
        consistent = replace(overflow, run_id="run-456")
        self.assertEqual(consistent.output_bytes_observed, cap + 1)

    def test_model_and_runner_boundary_agree_on_output_limit_consistency(self) -> None:
        cap = get_check_definition(CheckId.RUNNER_COMPILE).max_output_bytes
        cases = (
            (
                SandboxFailureCategory.TIMEOUT,
                SandboxDiagnosticCode.CHECK_TIMED_OUT,
                SandboxTermination.TIMEOUT,
                None,
                SandboxCleanupStatus.NOT_REQUIRED,
                cap,
                True,
            ),
            (
                SandboxFailureCategory.TIMEOUT,
                SandboxDiagnosticCode.CHECK_TIMED_OUT,
                SandboxTermination.TIMEOUT,
                None,
                SandboxCleanupStatus.NOT_REQUIRED,
                cap + 1,
                False,
            ),
            (
                SandboxFailureCategory.CLEANUP_FAILURE,
                SandboxDiagnosticCode.CLEANUP_UNCONFIRMED,
                SandboxTermination.NATURAL_EXIT,
                0,
                SandboxCleanupStatus.FAILED,
                cap,
                True,
            ),
            (
                SandboxFailureCategory.CLEANUP_FAILURE,
                SandboxDiagnosticCode.CLEANUP_UNCONFIRMED,
                SandboxTermination.NATURAL_EXIT,
                0,
                SandboxCleanupStatus.FAILED,
                cap + 1,
                False,
            ),
            (
                SandboxFailureCategory.OUTPUT_LIMIT,
                SandboxDiagnosticCode.OUTPUT_LIMIT_EXCEEDED,
                SandboxTermination.OUTPUT_LIMIT,
                None,
                SandboxCleanupStatus.NOT_REQUIRED,
                cap,
                False,
            ),
            (
                SandboxFailureCategory.OUTPUT_LIMIT,
                SandboxDiagnosticCode.OUTPUT_LIMIT_EXCEEDED,
                SandboxTermination.OUTPUT_LIMIT,
                None,
                SandboxCleanupStatus.NOT_REQUIRED,
                cap + 1,
                True,
            ),
        )
        for (
            category,
            diagnostic,
            termination,
            exit_code,
            cleanup_status,
            observed,
            accepted,
        ) in cases:
            with self.subTest(category=category, observed=observed):
                built = None
                try:
                    built = error_result(
                        failure_category=category,
                        diagnostic_code=diagnostic,
                        termination=termination,
                        exit_code=exit_code,
                        cleanup_status=cleanup_status,
                        output_bytes_observed=observed,
                    )
                except ValueError:
                    built = None
                self.assertEqual(built is not None, accepted)
                forged = forged_result(
                    check_id=CheckId.RUNNER_COMPILE,
                    status=SandboxStatus.ERROR,
                    exit_code=exit_code,
                    failure_category=category,
                    diagnostic_code=diagnostic,
                    termination=termination,
                    cleanup_status=cleanup_status,
                    output_bytes_observed=observed,
                )
                self.assertEqual(self.boundary_accepts(forged), accepted)


class SandboxBoundaryRevalidationTests(unittest.TestCase):
    """The runner boundary revalidates every backend result before returning it."""

    def forged(self, **changes):
        values = dict(check_id=CheckId.RUNNER_COMPILE)
        values.update(changes)
        return forged_result(**values)

    def assert_boundary_rejects(self, outcome) -> None:
        class FixedRunner:
            runtime_profile_id = "runner-verification-v1"

            def run(self, check, worktree, *, run_id, attempt):
                return outcome

        with self.assertRaises(SandboxRunnerError) as caught:
            invoke_sandbox_runner(
                FixedRunner(),
                get_check_definition(CheckId.RUNNER_COMPILE),
                verified_worktree(),
                run_id="run-123",
                attempt=1,
            )
        self.assertEqual(
            caught.exception.diagnostic_code, SandboxDiagnosticCode.RESULT_INVALID
        )

    def test_forged_passed_with_nonzero_exit_is_rejected(self) -> None:
        self.assert_boundary_rejects(
            self.forged(status=SandboxStatus.PASSED, exit_code=1)
        )

    def test_forged_passed_with_timeout_termination_is_rejected(self) -> None:
        self.assert_boundary_rejects(
            self.forged(
                status=SandboxStatus.PASSED,
                exit_code=0,
                termination=SandboxTermination.TIMEOUT,
            )
        )

    def test_forged_passed_with_failed_cleanup_is_rejected(self) -> None:
        self.assert_boundary_rejects(
            self.forged(
                status=SandboxStatus.PASSED,
                exit_code=0,
                cleanup_status=SandboxCleanupStatus.FAILED,
            )
        )

    def test_forged_passed_with_diagnostic_code_is_rejected(self) -> None:
        self.assert_boundary_rejects(
            self.forged(
                status=SandboxStatus.PASSED,
                exit_code=0,
                diagnostic_code=SandboxDiagnosticCode.CHECK_TIMED_OUT,
            )
        )

    def test_forged_negative_output_count_is_rejected(self) -> None:
        self.assert_boundary_rejects(
            self.forged(status=SandboxStatus.PASSED, exit_code=0, output_bytes_observed=-1)
        )

    def test_forged_boolean_output_count_is_rejected(self) -> None:
        self.assert_boundary_rejects(
            self.forged(status=SandboxStatus.PASSED, exit_code=0, output_bytes_observed=True)
        )

    def test_forged_non_finite_duration_is_rejected(self) -> None:
        for duration in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(duration=duration):
                self.assert_boundary_rejects(
                    self.forged(
                        status=SandboxStatus.PASSED,
                        exit_code=0,
                        duration_seconds=duration,
                    )
                )

    def test_forged_wrongly_typed_numeric_fields_raise_safe_error(self) -> None:
        cases = (
            {"exit_code": 0, "output_bytes_observed": "0"},
            {"exit_code": 0, "duration_seconds": "0.1"},
            {"exit_code": 0, "attempt": "1"},
            {"exit_code": "0"},
        )
        for changes in cases:
            with self.subTest(**{k: repr(v) for k, v in changes.items()}):
                self.assert_boundary_rejects(
                    self.forged(status=SandboxStatus.PASSED, **changes)
                )

    def test_forged_oversized_evidence_is_rejected(self) -> None:
        self.assert_boundary_rejects(
            self.forged(
                status=SandboxStatus.PASSED,
                exit_code=0,
                stdout_evidence=b"x" * (MAX_IN_MEMORY_EVIDENCE_BYTES + 1),
            )
        )

    def test_forged_invalid_backend_metadata_is_rejected(self) -> None:
        self.assert_boundary_rejects(
            self.forged(status=SandboxStatus.PASSED, exit_code=0, backend_type="NOT VALID!")
        )

    def test_forged_passed_surviving_a_pickle_round_trip_is_rejected(self) -> None:
        forged = self.forged(status=SandboxStatus.PASSED, exit_code=1)
        self.assert_boundary_rejects(pickle.loads(pickle.dumps(forged)))

    def test_forged_failed_with_invalid_status_category_pairings_is_rejected(
        self,
    ) -> None:
        cases = (
            {"status": SandboxStatus.FAILED, "exit_code": 1, "failure_category": None},
            {
                "status": SandboxStatus.FAILED,
                "exit_code": 0,
                "failure_category": SandboxFailureCategory.NONZERO_EXIT,
            },
        )
        for changes in cases:
            with self.subTest(**{k: getattr(v, "value", v) for k, v in changes.items()}):
                self.assert_boundary_rejects(self.forged(**changes))

    def test_forged_error_with_invalid_status_category_pairings_is_rejected(
        self,
    ) -> None:
        cases = (
            {
                "status": SandboxStatus.ERROR,
                "exit_code": None,
                "failure_category": SandboxFailureCategory.NONZERO_EXIT,
                "diagnostic_code": None,
                "termination": SandboxTermination.NATURAL_EXIT,
                "cleanup_status": SandboxCleanupStatus.NOT_REQUIRED,
            },
            {
                "status": SandboxStatus.ERROR,
                "exit_code": None,
                "failure_category": SandboxFailureCategory.CLEANUP_FAILURE,
                "diagnostic_code": SandboxDiagnosticCode.CLEANUP_UNCONFIRMED,
                "termination": SandboxTermination.NATURAL_EXIT,
                "cleanup_status": SandboxCleanupStatus.NOT_REQUIRED,
            },
        )
        for changes in cases:
            with self.subTest(category=getattr(changes["failure_category"], "value", None)):
                self.assert_boundary_rejects(self.forged(**changes))

    def test_forged_object_missing_a_field_is_rejected(self) -> None:
        forged = object.__new__(SandboxCheckResult)
        for name, value in (
            ("run_id", "run-123"),
            ("check_id", CheckId.RUNNER_COMPILE),
            ("attempt", 1),
            ("status", SandboxStatus.PASSED),
            ("duration_seconds", 0.25),
            ("output_bytes_observed", 0),
            ("termination", SandboxTermination.NATURAL_EXIT),
            ("stdout_evidence", b""),
            ("stderr_evidence", b""),
            ("cleanup_status", SandboxCleanupStatus.SUCCEEDED),
            ("failure_category", None),
            ("diagnostic_code", None),
            ("backend_type", "fake"),
        ):
            object.__setattr__(forged, name, value)
        self.assert_boundary_rejects(forged)

    def test_fully_valid_forged_result_is_returned_as_canonical_object(self) -> None:
        forged = self.forged(status=SandboxStatus.PASSED, exit_code=0, output_bytes_observed=0)

        class FixedRunner:
            runtime_profile_id = "runner-verification-v1"

            def run(self, check, worktree, *, run_id, attempt):
                return forged

        returned = invoke_sandbox_runner(
            FixedRunner(),
            get_check_definition(CheckId.RUNNER_COMPILE),
            verified_worktree(),
            run_id="run-123",
            attempt=1,
        )
        self.assertIsNot(returned, forged)
        self.assertIs(type(returned), SandboxCheckResult)
        self.assertEqual(returned.status, SandboxStatus.PASSED)
        self.assertEqual(returned.check_id, CheckId.RUNNER_COMPILE)

    def test_subclass_with_misleading_property_is_rejected(self) -> None:
        class LyingResult(SandboxCheckResult):
            @property
            def exit_code(self) -> int:
                return 0

            @property
            def status(self) -> SandboxStatus:
                return SandboxStatus.PASSED

        lying = object.__new__(LyingResult)
        lying.__dict__.update(
            {
                "run_id": "run-123",
                "check_id": CheckId.RUNNER_COMPILE,
                "attempt": 1,
                "status": SandboxStatus.FAILED,
                "exit_code": 1,
                "duration_seconds": 0.25,
                "output_bytes_observed": 0,
                "termination": SandboxTermination.NATURAL_EXIT,
                "stdout_evidence": b"",
                "stderr_evidence": b"",
                "cleanup_status": SandboxCleanupStatus.SUCCEEDED,
                "failure_category": SandboxFailureCategory.NONZERO_EXIT,
                "diagnostic_code": None,
                "backend_type": "fake",
            }
        )
        self.assert_boundary_rejects(lying)


if __name__ == "__main__":
    unittest.main()
