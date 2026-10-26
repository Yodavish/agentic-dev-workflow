"""Provider-neutral summary of one workflow invocation.

Token totals sum CLI result snapshots once. Cache counters stay separate
from input tokens. A missing metric stays unavailable. A price is published
only when a caller supplies an actual provider cost; token counts are not
priced here.
"""

from __future__ import annotations

import math
from typing import Any

_COUNTERS = (
    ("input_tokens", "inputTokens"),
    ("output_tokens", "outputTokens"),
    ("cache_read_tokens", "cacheReadTokens"),
    ("cache_write_tokens", "cacheWriteTokens"),
)
_VERIFICATION = frozenset({"pass", "fail", "not_run", "error"})
_DISPLAY_MODELS = frozenset({"Auto", "auto"})


def build_run_summary(
    *,
    run_id: str,
    outcome: str,
    elapsed_seconds: float | None = None,
    errors: list[str] | None = None,
    checks: list[dict] | None = None,
    attempt_history: list[dict] | None = None,
    repair_retries_used: int | None = None,
    max_retries: int | None = None,
    executor_invocations: int | None = None,
    agent_calls: list[dict] | None = None,
    trace_id: str | None = None,
    trace_scope: str | None = None,
    jaeger_ui: str | None = None,
) -> dict[str, Any]:
    """Build one summary. Rows stay separate from the single workflow total."""
    error_list = [item for item in errors or [] if isinstance(item, str)]
    check_list = [_public_check(item) for item in checks or [] if isinstance(item, dict)]
    history = [item for item in attempt_history or [] if isinstance(item, dict)]
    calls = [item for item in agent_calls or [] if isinstance(item, dict)]
    agents = [_agent_row(item) for item in calls]
    by_run = {}
    for item in calls:
        run = item.get("run_id")
        if not isinstance(run, str) or not run:
            continue
        by_run.setdefault(run, []).append(item)
    verification_status = _verification_status(history, check_list)
    workflow_cost = _workflow_cost(agents)
    workflow_counters = _combine_counters(calls)
    summary = {
        "run_id": run_id,
        "outcome": outcome,
        "elapsed_seconds": _metric(_nonnegative_number(elapsed_seconds)),
        "verification_status": verification_status,
        "approval_status": _approval_status(
            outcome, error_list, executor_invocations, check_list
        ),
        "retries_used": repair_retries_used,
        "max_retries": max_retries,
        "stop_reason": _stop_reason(outcome, error_list),
        "scope_violations": ["out_of_scope"] if "out_of_scope" in error_list else [],
        "checks": check_list,
        "agents": agents,
        "usage": {
            "by_run": {
                run: _usage_bucket(rows) for run, rows in by_run.items()
            },
            "workflow": {
                "run_count": len(by_run),
                "counters": workflow_counters,
                "token_total": _token_total(calls),
                "cost": workflow_cost,
            },
        },
        "trace": {
            "trace_id": trace_id,
            "run_id": run_id,
            "trace_scope": trace_scope if trace_scope in {"single_run", "multi_run"} else None,
            "jaeger_ui": jaeger_ui if jaeger_ui == "http://127.0.0.1:16686" else None,
            "status": "reported" if isinstance(trace_id, str) and trace_id else "unavailable",
        },
        "definitions": {
            "wall_duration": "Monotonic time around one adapter invocation.",
            "cli_duration": "duration_ms on the CLI result. It is not the wall-clock measurement.",
            "workflow_elapsed": "Monotonic time for this orchestration invocation.",
            "token_total": (
                "Sum of the four CLI counters only when every counter is present. "
                "Cache counters stay separate: cache counters are not added to input tokens."
            ),
            "workflow_tokens": "Sum of per-run totals. Each invocation is counted once.",
            "cost": "An actual provider cost only. Token counts are not priced.",
        },
    }
    summary["limitations"] = _limitations(summary)
    return summary


def _agent_row(call: dict) -> dict[str, Any]:
    usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
    counters = {name: _counter(usage, source) for name, source in _COUNTERS}
    present = [item["value"] for item in counters.values() if item["status"] == "reported"]
    complete = all(item["status"] == "reported" for item in counters.values())
    model = call.get("model")
    return {
        "role": call.get("role") if call.get("role") in {"planner", "executor", "reviewer"} else None,
        "invocation_id": call.get("invocation_id") if isinstance(call.get("invocation_id"), str) else None,
        "attempt": call.get("attempt") if isinstance(call.get("attempt"), str) else None,
        "status": call.get("status") if call.get("status") in {"succeeded", "failed", "cancelled"} else None,
        "model_label": _metric(model if isinstance(model, str) and model in _DISPLAY_MODELS else None),
        "model_identity": {"value": None, "status": "unavailable"},
        "wall_duration_seconds": _metric(_nonnegative_number(call.get("elapsed_seconds"))),
        "cli_duration_ms": _metric(_nonnegative_int(call.get("cli_duration_ms"))),
        **counters,
        "reported_token_field_count": len(present),
        "usage_status": "reported" if complete else "partial" if present else "unavailable",
        "token_total": (
            {"value": sum(present), "status": "reported"}
            if complete
            else {"value": None, "status": "partial" if present else "unavailable"}
        ),
        "cost": _actual_cost(call),
        "skill": {
            "status": call.get("skill_status")
            if call.get("skill_status") in {"reported", "partial", "unavailable"}
            else "unavailable"
        },
    }


def _usage_bucket(calls: list[dict]) -> dict[str, Any]:
    return {"counters": _combine_counters(calls), "token_total": _token_total(calls)}


def _combine_counters(calls: list[dict]) -> dict[str, dict]:
    combined = {}
    rows = [_agent_row(call) for call in calls]
    complete = bool(rows) and all(row["usage_status"] == "reported" for row in rows)
    for name, _source in _COUNTERS:
        present = [row[name]["value"] for row in rows if row[name]["status"] == "reported"]
        if not present:
            combined[name] = {"value": None, "status": "unavailable"}
        elif complete and len(present) == len(rows):
            combined[name] = {"value": sum(present), "status": "reported"}
        else:
            combined[name] = {"value": sum(present), "status": "partial"}
    return combined


def _token_total(calls: list[dict]) -> dict[str, Any]:
    rows = [_agent_row(call) for call in calls]
    if rows and all(row["token_total"]["status"] == "reported" for row in rows):
        return {"value": sum(row["token_total"]["value"] for row in rows), "status": "reported"}
    if any(row["reported_token_field_count"] for row in rows):
        return {"value": None, "status": "partial"}
    return {"value": None, "status": "unavailable"}


def _workflow_cost(agents: list[dict]) -> dict[str, Any]:
    if agents and all(item["cost"]["status"] == "reported" and item["cost"]["kind"] == "actual" for item in agents):
        return {
            "value": sum(item["cost"]["value"] for item in agents),
            "status": "reported",
            "kind": "actual",
        }
    return {"value": None, "status": "unavailable", "kind": None}


def _actual_cost(call: dict) -> dict[str, Any]:
    amount = _nonnegative_number(call.get("cost"))
    if call.get("cost_kind") == "actual" and amount is not None:
        return {"value": amount, "status": "reported", "kind": "actual"}
    return {"value": None, "status": "unavailable", "kind": None}


def _counter(usage: dict, source: str) -> dict[str, Any]:
    if source not in usage:
        return {"value": None, "status": "unavailable"}
    amount = _nonnegative_int(usage.get(source))
    if amount is None:
        return {"value": None, "status": "unavailable"}
    return {"value": amount, "status": "reported"}


def _metric(value: Any) -> dict[str, Any]:
    if value is None:
        return {"value": None, "status": "unavailable"}
    return {"value": value, "status": "reported"}


def _verification_status(history: list[dict], checks: list[dict]) -> str:
    if history:
        recorded = history[-1].get("verification_outcome")
        if recorded in _VERIFICATION:
            return recorded
    statuses = {check.get("status") for check in checks}
    if not statuses:
        return "not_run"
    if statuses == {"passed"}:
        return "pass"
    if "error" in statuses:
        return "error"
    if "failed" in statuses:
        return "fail"
    return "not_run"


def _approval_status(
    outcome: str,
    errors: list[str],
    executor_invocations: int | None,
    checks: list[dict],
) -> str:
    if outcome == "awaiting_approval" or "approval_required" in errors:
        return "awaiting"
    if "scope_mismatch" in errors:
        return "mismatch"
    invocations = executor_invocations if isinstance(executor_invocations, int) else 0
    if outcome == "success" or "out_of_scope" in errors or invocations > 0 or checks:
        return "granted"
    return "not_reached"


def _stop_reason(outcome: str, errors: list[str]) -> str:
    if errors:
        return errors[0]
    if outcome == "success":
        return "completed"
    return outcome


def _limitations(summary: dict) -> list[str]:
    limits = ["model_identity_unavailable"]
    if summary["usage"]["workflow"]["cost"]["status"] != "reported":
        limits.append("cost_unavailable")
    skills = [row["skill"]["status"] for row in summary["agents"]]
    if not skills or any(status != "reported" for status in skills):
        limits.append("skill_observation_unavailable")
    counters = summary["usage"]["workflow"]["counters"].values()
    if any(item["status"] == "partial" for item in counters) or summary["usage"]["workflow"]["token_total"]["status"] == "partial":
        limits.append("usage_partial")
    if all(item["status"] == "unavailable" for item in counters):
        limits.append("usage_unavailable")
    if summary["trace"]["status"] != "reported":
        limits.append("trace_unavailable")
    if summary["verification_status"] == "not_run":
        limits.append("verification_not_run")
    return limits


def _nonnegative_number(value: object) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or value < 0:
        return None
    return value


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _public_check(check: dict) -> dict[str, Any]:
    return {
        "check_id": check.get("check_id"),
        "status": check.get("status"),
        "exit_code": check.get("exit_code"),
    }
