"""Run-summary aggregation. Provider fixtures are data, not live model calls."""

from __future__ import annotations

import json
import unittest

from agentic_dev_workflow.run_summary import build_run_summary


FULL = {
    "inputTokens": 10,
    "outputTokens": 4,
    "cacheReadTokens": 40,
    "cacheWriteTokens": 2,
}


def call(role: str, attempt: str | None, usage: dict | None, **extra) -> dict:
    row = {
        "run_id": "run-a",
        "role": role,
        "invocation_id": f"{role}:{attempt or '1'}",
        "attempt": attempt,
        "status": "succeeded",
        "elapsed_seconds": 1.5,
        "cli_duration_ms": 1200,
        "usage": usage,
        "usage_status": "reported" if usage and len(usage) == 4 else "partial" if usage else "unavailable",
        "model_identity": "unknown",
        "model": "Auto",
        "skill_status": "unavailable",
    }
    row.update(extra)
    return row


class RunSummaryTests(unittest.TestCase):
    def test_repeated_attempts_are_rows_and_are_summed_once(self) -> None:
        second = dict(FULL)
        second["inputTokens"] = 5
        second["outputTokens"] = 1
        second["cacheReadTokens"] = 0
        second["cacheWriteTokens"] = 0
        summary = build_run_summary(
            run_id="run-a",
            outcome="success",
            elapsed_seconds=3.25,
            errors=[],
            checks=[{"check_id": "runner.tests", "status": "passed", "exit_code": 0}],
            attempt_history=[{"verification_outcome": "pass", "terminal_error": None}],
            repair_retries_used=1,
            max_retries=1,
            executor_invocations=2,
            agent_calls=[
                call("executor", "1", FULL),
                call("executor", "2", second, elapsed_seconds=2.0, cli_duration_ms=800),
            ],
            trace_id="ab" * 16,
            trace_scope="single_run",
            jaeger_ui="http://127.0.0.1:16686",
        )
        self.assertEqual(summary["outcome"], "success")
        self.assertEqual(summary["run_id"], "run-a")
        self.assertEqual(summary["elapsed_seconds"]["value"], 3.25)
        self.assertEqual(summary["elapsed_seconds"]["status"], "reported")
        self.assertEqual(summary["verification_status"], "pass")
        self.assertEqual([row["attempt"] for row in summary["agents"]], ["1", "2"])
        self.assertEqual(summary["agents"][0]["wall_duration_seconds"]["value"], 1.5)
        self.assertEqual(summary["agents"][1]["cli_duration_ms"]["value"], 800)
        self.assertNotEqual(
            summary["agents"][0]["wall_duration_seconds"]["value"],
            summary["agents"][0]["cli_duration_ms"]["value"],
        )
        workflow = summary["usage"]["workflow"]
        self.assertEqual(workflow["counters"]["input_tokens"]["value"], 15)
        self.assertEqual(workflow["counters"]["cache_read_tokens"]["value"], 40)
        self.assertEqual(workflow["token_total"]["value"], 10 + 4 + 40 + 2 + 5 + 1 + 0 + 0)
        self.assertEqual(summary["usage"]["by_run"]["run-a"]["token_total"]["value"], workflow["token_total"]["value"])
        self.assertNotEqual(workflow["counters"]["input_tokens"]["value"], 30)
        self.assertEqual(summary["trace"]["trace_id"], "ab" * 16)
        self.assertEqual(summary["trace"]["status"], "reported")
        self.assertIn("cache counters are not added to input tokens", summary["definitions"]["token_total"])

    def test_missing_and_partial_metrics_are_not_zero(self) -> None:
        summary = build_run_summary(
            run_id="run-a",
            outcome="failed",
            elapsed_seconds=None,
            agent_calls=[
                call(
                    "planner",
                    None,
                    {"inputTokens": 5},
                    elapsed_seconds=None,
                    cli_duration_ms=None,
                    usage_status="partial",
                )
            ],
        )
        agent = summary["agents"][0]
        self.assertIsNone(summary["elapsed_seconds"]["value"])
        self.assertEqual(summary["elapsed_seconds"]["status"], "unavailable")
        self.assertIsNone(agent["wall_duration_seconds"]["value"])
        self.assertEqual(agent["wall_duration_seconds"]["status"], "unavailable")
        self.assertEqual(agent["cli_duration_ms"]["status"], "unavailable")
        self.assertIsNone(agent["output_tokens"]["value"])
        self.assertEqual(agent["output_tokens"]["status"], "unavailable")
        self.assertEqual(agent["input_tokens"], {"value": 5, "status": "reported"})
        self.assertEqual(agent["reported_token_field_count"], 1)
        self.assertIsNone(agent["token_total"]["value"])
        self.assertEqual(agent["token_total"]["status"], "partial")
        self.assertEqual(agent["model_identity"]["status"], "unavailable")
        self.assertIsNone(agent["model_identity"]["value"])
        self.assertEqual(agent["cost"]["status"], "unavailable")
        self.assertIsNone(agent["cost"]["value"])
        self.assertEqual(agent["skill"]["status"], "unavailable")
        workflow = summary["usage"]["workflow"]
        self.assertEqual(workflow["counters"]["output_tokens"]["status"], "unavailable")
        self.assertIsNone(workflow["counters"]["output_tokens"]["value"])
        self.assertEqual(workflow["counters"]["input_tokens"], {"value": 5, "status": "partial"})
        self.assertNotIn("0", json.dumps(agent["output_tokens"]))
        self.assertEqual(summary["trace"]["status"], "unavailable")
        self.assertIsNone(summary["trace"]["trace_id"])

    def test_estimated_cost_is_dropped_and_actual_cost_is_labeled(self) -> None:
        summary = build_run_summary(
            run_id="run-a",
            outcome="success",
            elapsed_seconds=1,
            agent_calls=[
                call("planner", None, FULL, cost=4.5, cost_kind="estimated"),
                call("executor", "1", FULL, cost=2.0, cost_kind="actual"),
            ],
        )
        rendered = json.dumps(summary)
        self.assertNotIn("4.5", rendered)
        self.assertEqual(summary["agents"][0]["cost"], {"value": None, "status": "unavailable", "kind": None})
        self.assertEqual(summary["agents"][1]["cost"], {"value": 2.0, "status": "reported", "kind": "actual"})
        self.assertEqual(summary["usage"]["workflow"]["cost"]["status"], "unavailable")
        self.assertIsNone(summary["usage"]["workflow"]["cost"]["value"])
        complete = build_run_summary(
            run_id="run-a",
            outcome="success",
            elapsed_seconds=1,
            agent_calls=[
                call("planner", None, FULL, cost=1.5, cost_kind="actual"),
                call("executor", "1", FULL, cost=2.5, cost_kind="actual"),
            ],
        )
        self.assertEqual(complete["usage"]["workflow"]["cost"], {"value": 4.0, "status": "reported", "kind": "actual"})
        self.assertNotIn("estimated", json.dumps(complete["usage"]["workflow"]["cost"]))

    def test_failed_run_that_never_reaches_verification(self) -> None:
        summary = build_run_summary(
            run_id="run-a",
            outcome="failed",
            elapsed_seconds=0.4,
            errors=["out_of_scope"],
            checks=[],
            attempt_history=[{"verification_outcome": "not_run", "terminal_error": "out_of_scope"}],
            repair_retries_used=0,
            max_retries=1,
            executor_invocations=1,
            agent_calls=[call("executor", "1", None, usage_status="unavailable")],
        )
        self.assertEqual(summary["verification_status"], "not_run")
        self.assertEqual(summary["checks"], [])
        self.assertEqual(summary["approval_status"], "granted")
        self.assertEqual(summary["scope_violations"], ["out_of_scope"])
        self.assertEqual(summary["stop_reason"], "out_of_scope")
        self.assertEqual(summary["retries_used"], 0)
        self.assertEqual(summary["max_retries"], 1)
        self.assertEqual(summary["agents"][0]["usage_status"], "unavailable")
        self.assertIsNone(summary["agents"][0]["token_total"]["value"])

    def test_distinct_runs_stay_separate_and_the_workflow_total_sums_them_once(self) -> None:
        other = call("planner", None, FULL)
        other["run_id"] = "run-b"
        summary = build_run_summary(
            run_id="run-a",
            outcome="failed",
            elapsed_seconds=2,
            agent_calls=[call("planner", None, FULL), other],
            trace_scope="multi_run",
        )
        self.assertEqual(summary["usage"]["by_run"]["run-a"]["counters"]["input_tokens"]["value"], 10)
        self.assertEqual(summary["usage"]["by_run"]["run-b"]["counters"]["input_tokens"]["value"], 10)
        self.assertEqual(summary["usage"]["workflow"]["counters"]["input_tokens"]["value"], 20)
        self.assertEqual(summary["usage"]["workflow"]["run_count"], 2)
        self.assertIn("once", summary["definitions"]["workflow_tokens"])


class LiveRunSummaryTests(unittest.TestCase):
    def test_published_report_rebuilds_the_summary_and_drops_an_injected_price(self) -> None:
        import importlib.util
        from pathlib import Path

        script = Path(__file__).resolve().parents[1] / "scripts" / "live_cursor_feature.py"
        spec = importlib.util.spec_from_file_location("live_cursor_feature_summary_test", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        stored = module.redact_live_report(
            {
                "planning_report": {
                    "run_id": "run-a",
                    "outcome": "failed",
                    "attempts": 1,
                    "executor_invocations": 1,
                    "repair_retries_used": 0,
                    "max_retries": 1,
                    "checks": [],
                    "errors": ["out_of_scope"],
                    "proposed_paths": ["greeting.py"],
                    "attempt_history": [
                        {"attempt": 1, "verification_outcome": "not_run", "terminal_error": "out_of_scope", "checks": []}
                    ],
                    "elapsed_seconds": 1.25,
                },
                "agent_calls": [
                    {
                        "run_id": "run-a",
                        "role": "executor",
                        "invocation_id": "executor:1",
                        "attempt": "1",
                        "status": "succeeded",
                        "elapsed_seconds": 1.0,
                        "cli_duration_ms": 900,
                        "usage": FULL,
                        "model": "Auto",
                        "cost": 9.99,
                        "cost_kind": "estimated",
                    }
                ],
                "jaeger": {
                    "reachable": True,
                    "ui": "http://127.0.0.1:16686",
                    "spans": [
                        {
                            "operation": "workflow.live-run",
                            "trace_id": "cd" * 16,
                            "run_id": "run-a",
                            "trace_scope": "single_run",
                        }
                    ],
                },
                "run_summary": {"usage": {"workflow": {"cost": {"value": 9.99, "kind": "estimated"}}}},
            }
        )
        summary = stored["run_summary"]
        self.assertEqual(module.REPORT_SCHEMA_VERSION, 6)
        self.assertEqual(summary["verification_status"], "not_run")
        self.assertEqual(summary["stop_reason"], "out_of_scope")
        self.assertEqual(summary["trace"]["trace_id"], "cd" * 16)
        self.assertEqual(summary["agents"][0]["cost"]["status"], "unavailable")
        self.assertNotIn("9.99", json.dumps(stored))
        self.assertEqual(stored["usage_by_run"]["run-a"]["cost_status"], "unavailable")
        self.assertIn("model_identity_unavailable", summary["limitations"])
        self.assertIn("cost_unavailable", summary["limitations"])
        self.assertIn("skill_observation_unavailable", summary["limitations"])
