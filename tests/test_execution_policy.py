import unittest

from agentic_dev_workflow.execution_policy import (
    MAX_EXECUTION_TIMEOUT_SECONDS,
    MAX_POLICY_RETRIES,
    ExecutionPolicy,
    ExecutionPolicyError,
    validate_execution_policy,
)


BASE_COMMIT = "a" * 40


def policy(**changes):
    values = {
        "run_id": "run-123",
        "base_commit": BASE_COMMIT,
        "repository_path": "/tmp/projects/source",
        "worktree_path": "/tmp/worktrees/agentic-workflow-run-123",
        "clean_baseline_required": True,
        "workspace_write_authorized": True,
        "network_access": False,
        "timeout_seconds": 600,
        "max_retries": 2,
        "auto_commit": False,
        "auto_merge": False,
        "deployment_allowed": False,
        "production_database_access": False,
        "destructive_cleanup_allowed": False,
    }
    values.update(changes)
    return ExecutionPolicy(**values)


class ExecutionPolicyTests(unittest.TestCase):
    def test_valid_declaration_passes_pure_validation(self) -> None:
        self.assertEqual(validate_execution_policy(policy()), policy())

    def test_base_commit_must_be_full_sha1_or_sha256_hex(self) -> None:
        for candidate in (None, "", "abc", "g" * 40, "a" * 39, "a" * 63):
            with self.subTest(candidate=candidate), self.assertRaises(ExecutionPolicyError):
                validate_execution_policy(policy(base_commit=candidate))
        self.assertEqual(len(validate_execution_policy(policy(base_commit="b" * 64)).base_commit), 64)

    def test_repository_and_worktree_must_be_distinct_and_unambiguous(self) -> None:
        for source, worktree in (
            ("/tmp/same", "/tmp/same"),
            ("relative/source", "/tmp/worktrees/agentic-workflow-run-123"),
            ("/tmp/projects/source", "relative/worktree"),
            ("/tmp/projects/source", "/tmp/worktrees/agentic-workflow-other"),
        ):
            with self.subTest(source=source, worktree=worktree), self.assertRaises(ExecutionPolicyError):
                validate_execution_policy(
                    policy(repository_path=source, worktree_path=worktree)
                )

    def test_clean_baseline_and_explicit_workspace_write_are_required(self) -> None:
        for changes in (
            {"clean_baseline_required": False},
            {"workspace_write_authorized": False},
        ):
            with self.subTest(changes=changes), self.assertRaises(ExecutionPolicyError):
                validate_execution_policy(policy(**changes))

    def test_network_and_forbidden_operations_are_rejected(self) -> None:
        for changes in (
            {"network_access": True},
            {"auto_commit": True},
            {"auto_merge": True},
            {"deployment_allowed": True},
            {"production_database_access": True},
            {"destructive_cleanup_allowed": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ExecutionPolicyError):
                validate_execution_policy(policy(**changes))

    def test_timeout_and_retry_limits_are_bounded(self) -> None:
        self.assertEqual(MAX_POLICY_RETRIES, 2)
        self.assertEqual(
            validate_execution_policy(
                policy(timeout_seconds=MAX_EXECUTION_TIMEOUT_SECONDS)
            ).timeout_seconds,
            MAX_EXECUTION_TIMEOUT_SECONDS,
        )
        for changes in (
            {"timeout_seconds": 0},
            {"timeout_seconds": MAX_EXECUTION_TIMEOUT_SECONDS + 1},
            {"max_retries": -1},
            {"max_retries": MAX_POLICY_RETRIES + 1},
            {"max_retries": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ExecutionPolicyError):
                validate_execution_policy(policy(**changes))

    def test_invalid_run_id_is_rejected(self) -> None:
        with self.assertRaises(ExecutionPolicyError):
            validate_execution_policy(policy(run_id="run id"))


if __name__ == "__main__":
    unittest.main()
