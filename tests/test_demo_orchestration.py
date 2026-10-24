"""Focused checks for the standalone orchestration demo harness."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def load_demo():
    path = Path(__file__).resolve().parents[1] / "scripts" / "demo_orchestration.py"
    spec = importlib.util.spec_from_file_location("demo_orchestration", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("orchestration demo script could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DemoOrchestrationTests(unittest.TestCase):
    def test_three_scenarios_match_the_presentation_contract(self) -> None:
        reports = {item["scenario"]: item for item in load_demo().run_scenarios()}

        success = reports["success"]
        self.assertEqual(success["outcome"], "success")
        self.assertEqual(success["attempts"], 1)
        self.assertEqual(success["executor_invocations"], 1)
        self.assertEqual(success["repair_retries_used"], 0)
        self.assertEqual(success["review_decision"], "approve")
        self.assertEqual(success["errors"], [])
        self.assertEqual(
            [entry["verification_outcome"] for entry in success["attempt_history"]],
            ["pass"],
        )

        repair = reports["repair_then_success"]
        self.assertEqual(repair["outcome"], "success")
        self.assertEqual(repair["attempts"], 2)
        self.assertEqual(repair["executor_invocations"], 2)
        self.assertEqual(repair["repair_retries_used"], 1)
        self.assertEqual(repair["review_decision"], "approve")
        self.assertEqual(repair["errors"], [])
        self.assertEqual(
            [entry["verification_outcome"] for entry in repair["attempt_history"]],
            ["fail", "pass"],
        )

        exhausted = reports["retry_exhaustion"]
        self.assertEqual(exhausted["outcome"], "failed")
        self.assertEqual(exhausted["attempts"], 2)
        self.assertEqual(exhausted["executor_invocations"], 2)
        self.assertEqual(exhausted["repair_retries_used"], 1)
        self.assertEqual(exhausted["review_decision"], "approve")
        self.assertEqual(
            exhausted["errors"], ["verification_failed", "retries_exhausted"]
        )
        self.assertEqual(
            [entry["verification_outcome"] for entry in exhausted["attempt_history"]],
            ["fail", "fail"],
        )
        self.assertNotIn("stdout_evidence", str(reports))
        self.assertNotIn("stderr_evidence", str(reports))


if __name__ == "__main__":
    unittest.main()
