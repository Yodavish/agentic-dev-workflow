from __future__ import annotations

import json
import unittest

from agentic_dev_workflow.agent_outputs import (
    AgentOutputValidationError,
    validate_executor_output,
    validate_planner_output,
    validate_reviewer_output,
    validate_verifier_output,
)
from agentic_dev_workflow.adapters import (
    AgentRequest,
    AgentResult,
    AgentRole,
    AgentStatus,
    safe_telemetry_attributes,
)


def planner_payload() -> dict[str, object]:
    return {
        "summary": "Add a small status indicator.",
        "assumptions": ["The existing settings page is the intended location."],
        "steps": [
            {
                "id": "step-1",
                "description": "Add the status indicator to the settings page.",
                "expected_files": ["src/settings/status.py"],
            }
        ],
        "verification_check_ids": ["runner.tests", "runner.compile"],
    }


def executor_payload() -> dict[str, object]:
    return {
        "status": "completed",
        "summary": "The planned change was applied.",
        "files_changed": ["src/settings/status.py"],
        "checks_claimed": ["runner.tests"],
        "failure_category": None,
        "diagnostic_code": None,
    }


def verifier_payload() -> dict[str, object]:
    return {
        "outcome": "pass",
        "checks": [
            {
                "check_id": "runner.tests",
                "status": "passed",
                "exit_code": 0,
                "duration_ms": 1250,
                "evidence_ref": "ev_test-run-1",
                "summary": "The registered test check passed.",
            }
        ],
    }


def reviewer_payload() -> dict[str, object]:
    return {"decision": "approve", "findings": []}


class AgentOutputValidationTests(unittest.TestCase):
    def test_valid_role_outputs_convert_to_json_state(self) -> None:
        outputs = (
            validate_planner_output(planner_payload()),
            validate_executor_output(executor_payload()),
            validate_verifier_output(verifier_payload()),
            validate_reviewer_output(reviewer_payload()),
        )
        for output in outputs:
            with self.subTest(output=type(output).__name__):
                self.assertIsInstance(json.loads(json.dumps(output.to_state())), dict)

    def test_unknown_fields_and_malformed_types_are_rejected(self) -> None:
        cases = (
            (validate_planner_output, planner_payload()),
            (validate_executor_output, executor_payload()),
            (validate_verifier_output, verifier_payload()),
            (validate_reviewer_output, reviewer_payload()),
        )
        for validator, malformed in cases:
            malformed["unreviewed_property"] = "private model content"
            with self.subTest(validator=validator.__name__), self.assertRaises(
                AgentOutputValidationError
            ) as raised:
                validator(malformed)
            self.assertNotIn("private model content", str(raised.exception))

        malformed = executor_payload()
        malformed["files_changed"] = "src/settings/status.py"
        with self.assertRaises(AgentOutputValidationError):
            validate_executor_output(malformed)

        malformed = verifier_payload()
        malformed["checks"] = [{"check_id": "runner.tests", "status": True}]
        with self.assertRaises(AgentOutputValidationError):
            validate_verifier_output(malformed)

    def test_planner_rejects_invalid_paths(self) -> None:
        invalid_paths = (
            "../outside.py",
            "/absolute/path.py",
            "src/../outside.py",
            "src/.git/config",
            "C:/outside.py",
            r"src\outside.py",
            "src//duplicate.py",
            "src/line\nbreak.py",
            "src/file.txt:stream",
            "src/name.",
            "src/name ",
            "src/CON",
            "src/NUL",
            *(f"src/name{character}bad.txt" for character in '<>"|?*'),
        )
        reserved_names = ("CON", "nul", "Aux", "pRn") + tuple(
            f"{prefix}{number}"
            for prefix in ("COM", "lpt")
            for number in range(1, 10)
        )
        invalid_paths += tuple(
            f"src/{name}{extension}"
            for name in reserved_names
            for extension in ("", ".log")
        )
        for path in invalid_paths:
            payload = planner_payload()
            payload["steps"][0]["expected_files"] = [path]  # type: ignore[index]
            with self.subTest(path=path), self.assertRaises(AgentOutputValidationError):
                validate_planner_output(payload)

    def test_planner_rejects_case_and_unicode_normalization_path_collisions(self) -> None:
        payload = planner_payload()
        payload["steps"][0]["expected_files"] = [  # type: ignore[index]
            "src/Foo.py",
            "src/foo.py",
        ]
        with self.assertRaises(AgentOutputValidationError):
            validate_planner_output(payload)

        payload = planner_payload()
        payload["steps"][0]["expected_files"] = [  # type: ignore[index]
            "src/Ａ.py",
            "src/A.py",
        ]
        with self.assertRaises(AgentOutputValidationError):
            validate_planner_output(payload)

        payload = planner_payload()
        payload["steps"][0]["expected_files"] = [  # type: ignore[index]
            "src/cafe\u0301.py",
        ]
        with self.assertRaises(AgentOutputValidationError):
            validate_planner_output(payload)

        payload["steps"][0]["expected_files"] = [  # type: ignore[index]
            "src/Café.py",
            "src/other.py",
        ]
        accepted = validate_planner_output(payload)
        self.assertEqual(
            accepted.steps[0].expected_files,
            ("src/Café.py", "src/other.py"),
        )

    def test_planner_rejects_portable_path_collisions_across_steps(self) -> None:
        payload = planner_payload()
        payload["steps"] = [
            {
                "id": "step-1",
                "description": "Declare one case spelling.",
                "expected_files": ["src/Foo.py"],
            },
            {
                "id": "step-2",
                "description": "Declare another case spelling.",
                "expected_files": ["src/foo.py"],
            },
        ]
        with self.assertRaises(AgentOutputValidationError):
            validate_planner_output(payload)

    def test_planner_rejects_file_directory_prefix_conflicts_in_both_orders(self) -> None:
        paths = ("src/foo", "src/foo/bar.py")
        for expected_files in (paths, tuple(reversed(paths))):
            payload = planner_payload()
            payload["steps"][0]["expected_files"] = list(expected_files)  # type: ignore[index]
            with self.subTest(expected_files=expected_files), self.assertRaises(
                AgentOutputValidationError
            ):
                validate_planner_output(payload)

        for expected_files in (paths, tuple(reversed(paths))):
            payload = planner_payload()
            payload["steps"] = [
                {
                    "id": "step-1",
                    "description": "Declare one end of a path conflict.",
                    "expected_files": [expected_files[0]],
                },
                {
                    "id": "step-2",
                    "description": "Declare the conflicting descendant.",
                    "expected_files": [expected_files[1]],
                },
            ]
            with self.subTest(steps=expected_files), self.assertRaises(
                AgentOutputValidationError
            ):
                validate_planner_output(payload)

    def test_planner_accepts_paths_with_only_a_shared_textual_prefix(self) -> None:
        payload = planner_payload()
        payload["steps"][0]["expected_files"] = [  # type: ignore[index]
            "src/foo",
            "src/foobar.py",
        ]
        accepted = validate_planner_output(payload)
        self.assertEqual(
            accepted.steps[0].expected_files,
            ("src/foo", "src/foobar.py"),
        )

    def test_paths_reject_unicode_format_and_review_disrupting_categories(self) -> None:
        for character in (
            "\u202e",
            "\u2066",
            "\u2028",
            "\u2029",
            "\u00a0",
            "\ud800",
            "\ue000",
            "\u0378",
        ):
            path = f"src/report{character}name.py"
            payload = planner_payload()
            payload["steps"][0]["expected_files"] = [path]  # type: ignore[index]
            with self.subTest(category=ord(character)):
                with self.assertRaises(AgentOutputValidationError) as raised:
                    validate_planner_output(payload)
                self.assertNotIn(character, str(raised.exception))

    def test_duplicate_planner_entries_are_rejected(self) -> None:
        payload = planner_payload()
        payload["steps"] = [
            payload["steps"][0],
            {
                "id": "step-1",
                "description": "A second step with a duplicate ID.",
                "expected_files": ["src/settings/other.py"],
            },
        ]
        with self.assertRaises(AgentOutputValidationError):
            validate_planner_output(payload)

        payload = planner_payload()
        payload["steps"][0]["expected_files"] = [  # type: ignore[index]
            "src/settings/status.py",
            "src/settings/status.py",
        ]
        with self.assertRaises(AgentOutputValidationError):
            validate_planner_output(payload)

        payload = planner_payload()
        payload["verification_check_ids"] = ["runner.tests", "runner.tests"]
        with self.assertRaises(AgentOutputValidationError):
            validate_planner_output(payload)

    def test_duplicate_executor_check_claims_are_rejected(self) -> None:
        payload = executor_payload()
        payload["checks_claimed"] = ["runner.tests", "runner.tests"]
        with self.assertRaises(AgentOutputValidationError):
            validate_executor_output(payload)

    def test_duplicate_reviewer_finding_ids_are_rejected(self) -> None:
        finding = {
            "finding_id": "finding-1",
            "severity": "high",
            "category": "safety",
            "blocks_approval": True,
            "file": "src/settings/status.py",
            "function": None,
            "evidence_refs": ["ev_test-run-1"],
            "explanation": "A safety issue needs correction.",
            "remediation": "Correct the safety issue.",
        }
        duplicate = dict(finding)
        duplicate["file"] = "src/settings/other.py"
        with self.assertRaises(AgentOutputValidationError):
            validate_reviewer_output(
                {"decision": "reject", "findings": [finding, duplicate]}
            )

    def test_executor_rejects_invalid_paths_and_inconsistent_failure(self) -> None:
        for path in ("../outside.py", ".git/config", "src\\outside.py"):
            payload = executor_payload()
            payload["files_changed"] = [path]
            with self.subTest(path=path), self.assertRaises(AgentOutputValidationError):
                validate_executor_output(payload)

        payload = executor_payload()
        payload["status"] = "failed"
        payload["failure_category"] = None
        with self.assertRaises(AgentOutputValidationError):
            validate_executor_output(payload)

    def test_unregistered_verification_ids_are_rejected_for_all_relevant_roles(self) -> None:
        planner = planner_payload()
        planner["verification_check_ids"] = ["runner.run-arbitrary-command"]
        with self.assertRaises(AgentOutputValidationError):
            validate_planner_output(planner)

        executor = executor_payload()
        executor["checks_claimed"] = ["runner.run-arbitrary-command"]
        with self.assertRaises(AgentOutputValidationError):
            validate_executor_output(executor)

        verifier = verifier_payload()
        verifier["checks"][0]["check_id"] = "runner.run-arbitrary-command"  # type: ignore[index]
        with self.assertRaises(AgentOutputValidationError):
            validate_verifier_output(verifier)

    def test_output_counts_lengths_and_encoded_size_are_bounded(self) -> None:
        boundary = planner_payload()
        boundary["summary"] = "x" * 2048
        boundary["assumptions"] = ["y" * 512] * 20
        boundary["steps"] = [
            {
                "id": "step-1",
                "description": "z" * 512,
                "expected_files": ["a" * 255 + "/" + "b" * 254 + "/c"],
            }
        ]
        accepted = validate_planner_output(boundary)
        self.assertEqual(len(accepted.summary), 2048)
        self.assertEqual(len(accepted.assumptions), 20)
        self.assertEqual(len(accepted.steps[0].description), 512)
        self.assertEqual(len(accepted.steps[0].expected_files[0]), 512)

        payload = planner_payload()
        payload["assumptions"] = ["assumption"] * 21
        with self.assertRaises(AgentOutputValidationError):
            validate_planner_output(payload)

        payload = planner_payload()
        payload["summary"] = "x" * 2049
        with self.assertRaises(AgentOutputValidationError):
            validate_planner_output(payload)

        payload = planner_payload()
        paths = [f'src/{index}/' + '"' * 480 + ".py" for index in range(64)]
        payload["steps"] = [
            {
                "id": f"step-{index + 1}",
                "description": "x" * 512,
                "expected_files": paths[index * 3 : min((index + 1) * 3, 64)],
            }
            for index in range(21)
        ] + [
            {
                "id": "step-22",
                "description": "x" * 512,
                "expected_files": paths[63:],
            }
        ]
        payload["assumptions"] = ["y" * 512] * 20
        with self.assertRaises(AgentOutputValidationError):
            validate_planner_output(payload)

    def test_verifier_requires_consistent_evidence_and_outcome(self) -> None:
        payload = verifier_payload()
        payload["checks"][0]["evidence_ref"] = None  # type: ignore[index]
        with self.assertRaises(AgentOutputValidationError):
            validate_verifier_output(payload)

    def test_verifier_error_outcome_precedes_other_check_statuses(self) -> None:
        failed = {
            "check_id": "runner.tests",
            "status": "failed",
            "exit_code": 1,
            "duration_ms": 1,
            "evidence_ref": "ev_failed-check",
            "summary": "A check failed.",
        }
        errored = {
            "check_id": "runner.compile",
            "status": "error",
            "exit_code": None,
            "duration_ms": 1,
            "evidence_ref": "ev_error-check",
            "summary": "A check could not run.",
        }
        passed = {
            "check_id": "runner.diff-check",
            "status": "passed",
            "exit_code": 0,
            "duration_ms": 1,
            "evidence_ref": "ev_passed-check",
            "summary": "A check passed.",
        }
        skipped = {
            "check_id": "runner.lock-check",
            "status": "skipped",
            "exit_code": None,
            "duration_ms": 0,
            "evidence_ref": None,
            "summary": "A check was skipped.",
        }

        for other_check in (failed, passed, skipped):
            with self.subTest(other_status=other_check["status"]):
                checks = [dict(errored), dict(other_check)]
                accepted = validate_verifier_output(
                    {"outcome": "error", "checks": checks}
                )
                self.assertEqual(accepted.outcome.value, "error")

                for incorrect_outcome in ("pass", "fail"):
                    with self.subTest(outcome=incorrect_outcome), self.assertRaises(
                        AgentOutputValidationError
                    ):
                        validate_verifier_output(
                            {"outcome": incorrect_outcome, "checks": checks}
                        )

        payload = verifier_payload()
        payload["checks"][0]["exit_code"] = 1  # type: ignore[index]
        with self.assertRaises(AgentOutputValidationError):
            validate_verifier_output(payload)

        payload = verifier_payload()
        payload["outcome"] = "fail"
        with self.assertRaises(AgentOutputValidationError):
            validate_verifier_output(payload)

    def test_reviewer_requires_actionable_evidence_and_consistent_decision(self) -> None:
        payload = {
            "decision": "approve",
            "findings": [
                {
                    "finding_id": "finding-1",
                    "severity": "high",
                    "category": "safety",
                    "blocks_approval": True,
                    "file": "src/settings/status.py",
                    "function": "render_status",
                    "evidence_refs": ["ev_test-run-1"],
                    "explanation": "A required authorization check is missing.",
                    "remediation": "Add the authorization check before rendering.",
                }
            ],
        }
        with self.assertRaises(AgentOutputValidationError):
            validate_reviewer_output(payload)

        payload["decision"] = "reject"
        payload["findings"][0]["evidence_refs"] = []  # type: ignore[index]
        with self.assertRaises(AgentOutputValidationError):
            validate_reviewer_output(payload)

    def test_reviewer_severity_and_category_rules_are_enforced(self) -> None:
        finding = {
            "finding_id": "finding-1",
            "severity": "medium",
            "category": "correctness",
            "blocks_approval": False,
            "file": "src/settings/status.py",
            "function": None,
            "evidence_refs": ["ev_test-run-1"],
            "explanation": "The specified behavior is incomplete.",
            "remediation": "Handle the missing state.",
        }
        with self.assertRaises(AgentOutputValidationError):
            validate_reviewer_output({"decision": "approve", "findings": [finding]})

        finding["severity"] = "low"
        finding["blocks_approval"] = True
        with self.assertRaises(AgentOutputValidationError):
            validate_reviewer_output({"decision": "reject", "findings": [finding]})

    def test_failure_diagnostics_never_echo_agent_content(self) -> None:
        secret = "SYNTHETIC_PROMPT_OR_SECRET_7f2d"
        payload = planner_payload()
        payload["summary"] = secret
        payload["unexpected"] = secret
        with self.assertRaises(AgentOutputValidationError) as raised:
            validate_planner_output(payload)
        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn(secret, raised.exception.diagnostic_code)

        request = AgentRequest(
            run_id="run-safe-output",
            role=AgentRole.PLANNER,
            task=secret,
            structured_context={},
            timeout_seconds=30,
        )
        result = AgentResult(
            run_id=request.run_id,
            role=request.role,
            status=AgentStatus.SUCCEEDED,
            structured_output=payload,
        )
        attributes = safe_telemetry_attributes(
            request, result, attempt=1, adapter_type="mock"
        )
        self.assertNotIn(secret, repr(attributes))


if __name__ == "__main__":
    unittest.main()
