"""Run the transaction-action feature through the orchestration graph.

The planner, executor, and reviewer are the installed Cursor Agent CLI.

Pass ``--task`` to supply the specification inline. When ``--task`` is omitted
and ``SPEC.md`` exists in the feature repository, that file remains the
specification. Passing both is accepted only when the texts are identical.
``--approve`` is the caller's explicit file list. The runner does not approve
the planner's proposed paths on its own. Without ``--approve``, a plan stops
at ``awaiting_approval``.
Verification runs the registered commands in a host process and keeps the
real exit code. This script does not call the mock workflow demo.

Live-run report schema version 6
--------------------------------
Schema 6 adds ``run_summary``. Existing fields stay. The summary is rebuilt
from the sanitized planning report, agent calls, and Jaeger identifiers. An
injected summary is ignored. Wall-clock duration and ``cli_duration_ms`` stay
separate. Token totals sum each invocation once and do not add cache counters
to input tokens. Missing and partial metrics stay null with a status. The
installed CLI prints a display name and four token counters. It does not print
a provider model id, a price, or a skill event, so those summary fields stay
unavailable. A caller-supplied estimate is not published.

Live-run report schema version 5
--------------------------------
``publish_live_report`` writes ``live-run-report.json`` only when that file
does not already exist. Set ``AGENTIC_REPORT_PATH`` to publish to another
file instead. An existing destination is left in place and is not deleted.
The same JSON is printed to stdout. The document is an allowlist. A field is
kept only when its source is one of the fixed sets below. Regexes here describe
shape; they are not a secret detector. Unrecognized values are JSON null or
dropped from lists.

Schema 5 records skill observation status. The installed CLI stream has no
structured skill event, so each call is ``skill_status: unavailable``.
``skill_by_run`` is recomputed from the sanitized ``agent_calls`` and is keyed
by ``run_id``. Each run has ``skill_status`` and
``invocations_without_skill_observation``. A run whose calls are all
unavailable stays ``unavailable``. ``reported`` and ``partial`` are reserved
for a future structured event. Skill ids, skill counts, tool names, and prompt
text are not written.

Schema 4 adds usage completeness. It does not add prices.

- ``usage_status`` is ``reported`` when all four counters are present,
  ``partial`` when at least one is present, and ``unavailable`` when no valid
  usage object exists. Zero is a present value. Missing counters are omitted.
- ``usage_scope`` is the fixed value ``cli_result_snapshot``. ``reported``
  means those counters were present on the last complete CLI result. It does
  not mean every underlying model request was counted.
- ``missing_invocations`` counts calls that omitted each counter. A run total
  can exist beside a non-zero missing count.
- ``model_identity`` is ``unknown`` for ``Auto``, ``auto``, any other display
  name, and missing model metadata. ``mixed`` is kept only when aggregated
  identities disagree. ``known`` is not emitted: this report has no provider
  model id. ``cost_status`` is always ``unavailable``. No USD field is written.
- Per-run totals are keyed by ``workflow.run_id``. Runs that share a trace
  stay in separate buckets. ``run_id`` plus ``invocation_id`` identifies a call.
  ``usage_by_run`` is recomputed from the sanitized ``agent_calls``. A
  caller-supplied total is not published.
- A parent span that contains one run keeps ``workflow.run_id``. A parent
  that contains more than one run omits that singular id and sets
  ``workflow.trace_scope`` to ``multi_run``. Usage is not summed across runs.

- ``model`` is published only for the known display names ``Auto`` and
  ``auto``. Any other CLI model string is omitted.
- ``cursor_agent_version`` is published only when the first line is a numeric
  version with an optional lowercase hex build suffix, such as
  ``2026.10.22-27dcf5f``. Other ``--version`` text is omitted.
- Relative paths (``proposed_paths``, ``approved_paths``, ``execution.approval``)
  must be repository files: each segment is 1-32 characters of letters,
  digits, ``_``, or ``-``; a segment may not contain both ``_`` and ``-``;
  the final segment must use an extension in ``.py``, ``.md``, ``.toml``,
  ``.json``, ``.txt``, ``.yml``, ``.yaml``, ``.ini``, ``.cfg``, or ``.lock``.
- ``diagnostic_code`` and error lists publish only codes defined by this
  workflow's enums and fixed orchestration literals. Model-supplied codes and
  exception text are omitted.
- Tool names, subtypes, and paths are not published. ``tool_event_count`` is
  the number of tool events, capped at 40.
- Roles, statuses, failure categories, reviewer decisions, check ids, and
  outcomes are fixed enumerations.
- ``base_commit`` must be hexadecimal. Local path fields must be absolute.
  Jaeger span names and the UI URL are fixed. Trace ids are hexadecimal.

Schema 3 closed model strings, version tokens, tool names, and generic
diagnostic patterns. Schema 2 had published those channels. They stay closed.

Removed earlier and still absent: ``stdout_excerpt``, ``stderr_excerpt``,
``cursor_agent_status``, raw ``structured_output``, tool paths and arguments,
check and verification stdout/stderr, command argv, and ``plan_summary``.
Prompts, source, credentials, environment variables, and raw CLI output are
not stored.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_dev_workflow.adapters import AgentFailureCategory
from agentic_dev_workflow.checkpointing import open_sqlite_checkpointer
from agentic_dev_workflow.git_inventory import (
    ApprovedScopeDiagnosticCode,
    GitInventoryDiagnosticCode,
)
from agentic_dev_workflow.cursor_adapter import (
    CursorCLIAdapter,
    UUID_TEXT,
    aggregate_skill_by_run,
    aggregate_usage_by_run,
    apply_invocation_attributes,
    apply_parent_trace_identity,
    apply_run_skill_attributes,
    apply_run_usage_attributes,
    classify_model_identity,
    resolve_cursor_agent,
    sanitized_model,
    sanitized_uuid,
    usage_completeness,
)
from agentic_dev_workflow.host_check_runner import HostProcessCheckRunner
from agentic_dev_workflow.run_summary import build_run_summary
from agentic_dev_workflow.orchestration import (
    ScopeApproval,
    load_task_specification,
    run_orchestration,
)
from agentic_dev_workflow.sandbox import SandboxDiagnosticCode, VerifiedWorktree
from agentic_dev_workflow.tracing import create_tracing_session
from agentic_dev_workflow.worktrees import WorktreeManager

FEATURE_REPOSITORY = Path("/tmp/agentic-feature-test-qQFVY6/feature")
DISPOSABLE_ROOT = FEATURE_REPOSITORY.parent
WORKTREE_ROOT = DISPOSABLE_ROOT / "worktrees"
CHECKPOINT_DIR = DISPOSABLE_ROOT / "checkpoints"
OTLP_ENDPOINT = "http://127.0.0.1:4318/v1/traces"
JAEGER_TRACES = "http://127.0.0.1:16686/api/v3/traces"
RUN_ID = os.environ.get("AGENTIC_RUN_ID") or time.strftime("tx%H%M%S")
AGENT_TIMEOUT_SECONDS = 300
REPORT_SCHEMA_VERSION = 6
_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_UUID = UUID_TEXT
_INVOCATION_ID = re.compile(r"^(planner|executor|reviewer):[1-9][0-9]{0,2}$")
_ATTEMPT = re.compile(r"^[1-9][0-9]{0,2}$")
_VERSION = re.compile(r"^[0-9]{1,4}(?:\.[0-9]{1,4}){1,3}(?:-[0-9a-f]{6,40})?$")
_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_PATH_EXTENSIONS = frozenset(
    {".py", ".md", ".toml", ".json", ".txt", ".yml", ".yaml", ".ini", ".cfg", ".lock"}
)
_HEX_COMMIT = re.compile(r"^[0-9a-fA-F]{4,64}$")
_HEX_TRACE = re.compile(r"^[0-9a-fA-F]{16,32}$")
_CHECKPOINT_ID = re.compile(r"^cp-[A-Za-z0-9]{1,16}$")
_ABSOLUTE_PATH = re.compile(r"^/[A-Za-z0-9_./-]{1,4095}$")
_SPAN_NAMES = frozenset(
    {"workflow.live-run", "agent.planner", "agent.executor", "agent.reviewer"}
)
_NETWORK_ERRORS = frozenset(
    {"URLError", "TimeoutError", "HTTPError", "OSError", "ConnectionError", "gaierror"}
)
_OUTCOMES = frozenset(
    {
        "success",
        "failed",
        "awaiting_approval",
        "error",
        "running",
        "pass",
        "fail",
        "not_run",
        "loop",
        "input",
        "update",
    }
)
_ORCHESTRATION_CODES = frozenset(
    {
        "approval_required",
        "invalid_approved_path",
        "ambiguous_scope",
        "scope_mismatch",
        "retries_exhausted",
        "planner_failed",
        "executor_failed",
        "reviewer_failed",
        "fabricated_evidence",
        "repository_root_mismatch",
        "verification_error",
        "verification_failed",
        "review_rejected",
        "codex_adapter_disabled",
        "invalid_adapter_type",
        "adapter_invocation_failed",
        "result_type_invalid",
        "result_correlation_mismatch",
        "cursor_agent_not_found",
        "cursor_workspace_missing",
        "cursor_launch_failed",
        "cursor_timeout",
        "cursor_not_authenticated",
        "cursor_exit_nonzero",
        "cursor_run_error",
        "cursor_output_invalid",
        "output_invalid",
        "output_field_invalid",
        "output_path_invalid",
        "output_limit_exceeded",
        "output_inconsistent",
        "output_type_invalid",
        "output_fields_invalid",
        "output_check_id_invalid",
    }
)
_REPORT_CODES = (
    frozenset(item.value for item in AgentFailureCategory)
    | frozenset(item.value for item in GitInventoryDiagnosticCode)
    | frozenset(item.value for item in ApprovedScopeDiagnosticCode)
    | frozenset(item.value for item in SandboxDiagnosticCode)
    | _ORCHESTRATION_CODES
)
_RUNNER_CHECK_IDS = frozenset(
    {"runner.tests", "runner.compile", "runner.diff-check", "runner.lock-check"}
)
_EXECUTOR_STATUS = frozenset({"completed", "failed", "blocked"})
_REVIEW_DECISION = frozenset({"approve", "reject"})
_CHECK_STATUS = frozenset({"passed", "failed", "skipped", "error"})
_AGENT_STATUS = frozenset({"succeeded", "failed", "cancelled"})
_USAGE_COUNTERS = ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens")
_USAGE_STATUS = frozenset({"reported", "partial", "unavailable"})
_MODEL_IDENTITY = frozenset({"unknown", "mixed"})


class RecordingAdapter:
    """Record one span and one observation per adapter call."""

    def __init__(self, inner, tracer, records: list[dict], *, replayed: bool = False) -> None:
        self._inner = inner
        self.adapter_type = inner.adapter_type
        self._tracer = tracer
        self._records = records
        self._replayed = replayed

    def invoke(self, request):
        with self._tracer.start_as_current_span(f"agent.{request.role.value}") as span:
            span.set_attribute("workflow.run_id", request.run_id)
            span.set_attribute("agent.role", request.role.value)
            span.set_attribute("agent.adapter_type", self.adapter_type)
            span.set_attribute("agent.replayed", self._replayed)
            started = time.monotonic()
            result = self._inner.invoke(request)
            elapsed = time.monotonic() - started
            span.set_attribute("agent.status", result.status.value)
            span.set_attribute("agent.duration_seconds", result.duration_seconds)
            span.set_attribute("agent.elapsed_seconds", elapsed)
            if (
                isinstance(result.diagnostic_code, str)
                and result.diagnostic_code in _REPORT_CODES
            ):
                span.set_attribute("agent.diagnostic_code", result.diagnostic_code)
            observation = getattr(self._inner, "last_observation", None)
            apply_invocation_attributes(span, observation, request)
            usage = getattr(observation, "usage", None)
            cli_duration = getattr(observation, "cli_duration_ms", None)
            if isinstance(cli_duration, int):
                span.set_attribute("agent.cli_duration_ms", cli_duration)
            self._records.append(
                agent_call_record(
                    request,
                    result,
                    observation,
                    elapsed=elapsed,
                    replayed=self._replayed,
                    cli_duration_ms=cli_duration,
                    usage=usage,
                )
            )
            return result


def _parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Run one disposable feature through the workflow.")
    parser.add_argument("--repository", type=Path, default=FEATURE_REPOSITORY)
    parser.add_argument("--task", default=None, help="Inline task specification. Multiline text is preserved.")
    parser.add_argument("--spec", type=Path, default=None, help="Spec file. Defaults to SPEC.md when that file exists.")
    parser.add_argument(
        "--approve",
        action="append",
        default=None,
        help="Explicit repository-relative path to approve. Repeat for each path. Omit to stop after planning.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(argv)
    feature = arguments.repository.resolve()
    if "politician-dashboard" in feature.parts or not str(feature).startswith("/tmp/agentic-feature-test-"):
        print(json.dumps({"error": "feature repository is not the disposable checkout"}))
        return 2
    try:
        spec_text = resolve_live_task(feature, inline=arguments.task, spec=arguments.spec)
    except ValueError as error:
        print(json.dumps({"error": "task_specification", "message": str(error)}))
        return 2
    approval = explicit_approval(arguments.approve)
    executable = resolve_cursor_agent()
    version = _command_output([executable, "--version"]) if executable else None
    base_commit = _command_output(["git", "-C", str(feature), "rev-parse", "HEAD"]).strip()
    disposable_root = feature.parent
    worktree_root = disposable_root / "worktrees"
    checkpoint_dir = disposable_root / "checkpoints"
    manager = WorktreeManager(worktree_root)
    created = manager.create_worktree(feature, base_commit, run_id=RUN_ID)
    if not created.success:
        print(json.dumps({"error": "worktree_create_failed", "issues": list(created.issues)}))
        return 2
    verified = manager.verify_managed_worktree(feature, base_commit, run_id=RUN_ID, require_clean=True)
    if not verified.success:
        print(json.dumps({"error": "worktree_verify_failed", "issues": list(verified.issues)}))
        return 2
    worktree = VerifiedWorktree.from_manager_verification(verified)
    runner = HostProcessCheckRunner()
    records: list[dict] = []
    usage_by_run: dict = {}
    session = create_tracing_session(OTLP_ENDPOINT)
    cursor = CursorCLIAdapter(workspace=worktree.path, executable=executable)
    report = None
    checkpoints: list[dict] = []
    execution = None
    try:
        with session.tracer.start_as_current_span("workflow.live-run") as span:
            span.set_attribute("workflow.mock_mode", False)
            span.set_attribute("workflow.phase", "orchestration")
            try:
                with open_sqlite_checkpointer(checkpoint_dir) as saver:
                    report = run_orchestration(
                        spec_text,
                        worktree=worktree,
                        planner=RecordingAdapter(cursor, session.tracer, records),
                        executor=RecordingAdapter(cursor, session.tracer, records),
                        reviewer=RecordingAdapter(cursor, session.tracer, records),
                        sandbox=runner,
                        approval=approval,
                        max_retries=1,
                        run_id=RUN_ID,
                        checkpointer=saver,
                        agent_timeout_seconds=AGENT_TIMEOUT_SECONDS,
                    )
                    span.set_attribute("workflow.outcome", report.outcome)
                    checkpoints = _checkpoints(saver, RUN_ID)
                    execution = None
            finally:
                usage_by_run = aggregate_usage_by_run(
                    [
                        (
                            record.get("run_id"),
                            record.get("usage"),
                            record.get("model_identity") or "unknown",
                        )
                        for record in records
                        if record.get("run_id")
                    ]
                )
                apply_run_usage_attributes(span, usage_by_run)
                apply_run_skill_attributes(
                    span,
                    aggregate_skill_by_run(
                        [
                            (record.get("run_id"), record.get("skill_status"))
                            for record in records
                            if record.get("run_id")
                        ]
                    ),
                )
                apply_parent_trace_identity(
                    span,
                    [record.get("run_id") for record in records],
                    fallback_run_id=RUN_ID,
                )
    finally:
        flushed = session.flush()
        session.shutdown()
    payload = {
        "cursor_agent": executable,
        "cursor_agent_version": version,
        "feature_repository": str(feature),
        "base_commit": base_commit,
        "worktree": worktree.path,
        "agent_timeout_seconds": AGENT_TIMEOUT_SECONDS,
        "trace_flushed": flushed,
        "otlp_endpoint": OTLP_ENDPOINT,
        "planning_report": None if report is None else report.to_dict(),
        "agent_calls": records,
        "checkpoints": checkpoints,
        "check_commands": [_command_view(record) for record in runner.invocations],
        "execution": execution,
        "jaeger": _jaeger_summary(RUN_ID),
        "token_usage": "not exposed"
        if not any(record.get("usage") for record in records)
        else "exposed",
        "usage_by_run": usage_by_run,
    }
    destination = report_destination(disposable_root)
    try:
        publish_live_report(destination, payload)
    except FileExistsError:
        print(json.dumps({"error": "live_report_exists", "path": str(destination)}))
        return 2
    planning_outcome = None if report is None else report.outcome
    execution_outcome = (execution or {}).get("report", {}).get("outcome")
    return 0 if planning_outcome == "success" or execution_outcome == "success" else 1


def report_destination(root: Path | None = None) -> Path:
    """Return the report path. A blank override keeps the disposable default."""
    override = os.environ.get("AGENTIC_REPORT_PATH")
    if isinstance(override, str) and override.strip():
        return Path(override)
    base = DISPOSABLE_ROOT if root is None else root
    return base / "live-run-report.json"


def resolve_live_task(repository: Path, *, inline: str | None, spec: Path | None) -> str:
    """Load inline text, or SPEC.md when no inline text is provided."""
    spec_path = spec
    if spec_path is None and (repository / "SPEC.md").is_file():
        spec_path = repository / "SPEC.md"
    return load_task_specification(inline=inline, spec_path=spec_path)


def explicit_approval(paths: list[str] | None) -> ScopeApproval | None:
    """Return the caller's approved paths. An omitted list grants nothing."""
    if not paths:
        return None
    return ScopeApproval(tuple(paths))


def publish_live_report(destination: Path, payload: dict) -> str:
    """Write a new report and print it. An existing file is left untouched."""
    rendered = json.dumps(redact_live_report(payload), indent=2, sort_keys=True) + "\n"
    fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, rendered.encode("utf-8"))
    finally:
        os.close(fd)
    print(rendered, end="")
    return rendered


def redact_live_report(payload: object) -> dict:
    """Rebuild the report from an allowlist. Unknown and raw CLI fields are dropped."""
    source = payload if isinstance(payload, dict) else {}
    agent_calls = _redacted_agent_calls(source.get("agent_calls"))
    planning_report = _redact_planning_report(source.get("planning_report"))
    jaeger = _redact_jaeger(source.get("jaeger"))
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "cursor_agent": _absolute_path(source.get("cursor_agent")),
        "cursor_agent_version": _version(source.get("cursor_agent_version")),
        "feature_repository": _absolute_path(source.get("feature_repository")),
        "base_commit": _match(_HEX_COMMIT, source.get("base_commit")),
        "worktree": _absolute_path(source.get("worktree")),
        "agent_timeout_seconds": _nonnegative_int(source.get("agent_timeout_seconds")),
        "trace_flushed": source.get("trace_flushed") is True,
        "otlp_endpoint": source.get("otlp_endpoint") if source.get("otlp_endpoint") == OTLP_ENDPOINT else None,
        "planning_report": planning_report,
        "agent_calls": agent_calls,
        "checkpoints": [
            _redact_checkpoint(item) for item in _as_list(source.get("checkpoints"))
        ],
        "check_commands": [
            _redact_check_command(item) for item in _as_list(source.get("check_commands"))
        ],
        "execution": _redact_execution(source.get("execution")),
        "jaeger": jaeger,
        "run_summary": _run_summary(planning_report, agent_calls, jaeger),
        "token_usage": source.get("token_usage") if source.get("token_usage") in {"exposed", "not exposed"} else "not exposed",
        "usage_by_run": _usage_from_agent_calls(agent_calls),
        "skill_by_run": _skill_from_agent_calls(agent_calls),
    }


def _redacted_agent_calls(value: object) -> list[dict]:
    return [_redact_agent_call(item) for item in _as_list(value)]


def _run_summary(planning: dict | None, agent_calls: list[dict], jaeger: dict) -> dict:
    """Rebuild the operator summary. Caller-supplied prices and summaries are ignored."""
    source = planning or {}
    spans = jaeger.get("spans") if isinstance(jaeger.get("spans"), list) else []
    trace_id = None
    trace_scope = None
    for span in spans:
        if not isinstance(span, dict):
            continue
        if trace_id is None and isinstance(span.get("trace_id"), str):
            trace_id = span["trace_id"]
        if span.get("trace_scope") in {"single_run", "multi_run"}:
            trace_scope = span["trace_scope"]
    return build_run_summary(
        run_id=source.get("run_id") if isinstance(source.get("run_id"), str) else "",
        outcome=source.get("outcome") if isinstance(source.get("outcome"), str) else "error",
        elapsed_seconds=source.get("elapsed_seconds"),
        errors=source.get("errors") if isinstance(source.get("errors"), list) else [],
        checks=source.get("checks") if isinstance(source.get("checks"), list) else [],
        attempt_history=source.get("attempt_history") if isinstance(source.get("attempt_history"), list) else [],
        repair_retries_used=source.get("repair_retries_used") if isinstance(source.get("repair_retries_used"), int) else None,
        max_retries=source.get("max_retries") if isinstance(source.get("max_retries"), int) else None,
        executor_invocations=source.get("executor_invocations") if isinstance(source.get("executor_invocations"), int) else None,
        agent_calls=agent_calls,
        trace_id=trace_id,
        trace_scope=trace_scope,
        jaeger_ui=jaeger.get("ui") if isinstance(jaeger.get("ui"), str) else None,
    )


def _skill_from_agent_calls(agent_calls: list[dict]) -> dict:
    """Derive skill observation per run. Ignore any supplied skill summary."""
    return aggregate_skill_by_run(
        [
            (call.get("run_id"), call.get("skill_status"))
            for call in agent_calls
            if call.get("run_id")
        ]
    )


def _usage_from_agent_calls(agent_calls: list[dict]) -> dict:
    """Derive run totals from sanitized calls. Ignore any supplied aggregate."""
    return _redact_usage_by_run(
        aggregate_usage_by_run(
            [
                (call.get("run_id"), call.get("usage"), call.get("model_identity") or "unknown")
                for call in agent_calls
                if call.get("run_id")
            ]
        )
    )


def agent_call_record(
    request,
    result,
    observation,
    *,
    elapsed: float,
    replayed: bool,
    cli_duration_ms: int | None,
    usage,
) -> dict:
    """Structured invocation metadata. The prompt and raw CLI text stay out."""
    metadata = getattr(request, "correlation_metadata", {}) or {}
    failure = getattr(result, "failure_category", None)
    return _redact_agent_call(
        {
            "run_id": request.run_id,
            "role": request.role.value,
            "invocation_id": metadata.get("invocation_id") if isinstance(metadata, dict) else None,
            "attempt": metadata.get("attempt") if isinstance(metadata, dict) else None,
            "status": result.status.value,
            "failure_category": None if failure is None else failure.value,
            "diagnostic_code": result.diagnostic_code,
            "duration_seconds": result.duration_seconds,
            "elapsed_seconds": elapsed,
            "replayed": replayed,
            "exit_code": result.exit_code,
            "model": None if observation is None else getattr(observation, "model", None),
            "session_id": None if observation is None else getattr(observation, "session_id", None),
            "request_id": None if observation is None else getattr(observation, "request_id", None),
            "cli_duration_ms": cli_duration_ms,
            "cli_duration_api_ms": None
            if observation is None
            else getattr(observation, "cli_duration_api_ms", None),
            "usage": usage,
            "structured_output": dict(getattr(result, "structured_output", {}) or {}),
            "tool_events": () if observation is None else list(getattr(observation, "tool_events", ()) or ()),
        }
    )


def _redact_agent_call(item: object) -> dict:
    source = item if isinstance(item, dict) else {}
    role = source.get("role") if source.get("role") in {"planner", "executor", "reviewer"} else None
    structured = source.get("structured_output")
    summary_source = structured if isinstance(structured, dict) else source.get("output_summary")
    usage = _safe_usage(source.get("usage"))
    return {
        "run_id": _match(_RUN_ID, source.get("run_id")),
        "role": role,
        "invocation_id": _match(_INVOCATION_ID, source.get("invocation_id")),
        "attempt": _match(_ATTEMPT, source.get("attempt")),
        "status": source.get("status") if source.get("status") in _AGENT_STATUS else None,
        "failure_category": _enum(
            frozenset(item.value for item in AgentFailureCategory),
            source.get("failure_category"),
        ),
        "diagnostic_code": _enum(_REPORT_CODES, source.get("diagnostic_code")),
        "duration_seconds": _nonnegative_number(source.get("duration_seconds")),
        "elapsed_seconds": _nonnegative_number(source.get("elapsed_seconds")),
        "replayed": source.get("replayed") is True,
        "exit_code": _integer(source.get("exit_code")),
        "model": sanitized_model(source.get("model")),
        "session_id": sanitized_uuid(source.get("session_id")),
        "request_id": sanitized_uuid(source.get("request_id")),
        "cli_duration_ms": _nonnegative_int(source.get("cli_duration_ms")),
        "cli_duration_api_ms": _nonnegative_int(source.get("cli_duration_api_ms")),
        "usage_status": usage_completeness(usage),
        "usage_scope": "cli_result_snapshot",
        "usage": usage,
        "model_identity": classify_model_identity(source.get("model")),
        "cost_status": "unavailable",
        "skill_status": "unavailable",
        "output_summary": _output_summary(role, summary_source),
        "tool_event_count": _tool_event_count(source),
    }


def _output_summary(role: object, payload: object) -> dict:
    source = payload if isinstance(payload, dict) else {}
    if role == "planner":
        if "steps" in source:
            steps = source.get("steps")
            step_count = len(steps) if isinstance(steps, list) else 0
        else:
            step_count = _nonnegative_int(source.get("step_count")) or 0
        checks = source.get("verification_check_ids")
        return {
            "step_count": step_count,
            "verification_check_ids": [
                item for item in checks if item in _RUNNER_CHECK_IDS
            ]
            if isinstance(checks, list)
            else [],
        }
    if role == "executor":
        files = source.get("files_changed")
        if isinstance(files, list):
            file_count = len(files)
        else:
            file_count = _nonnegative_int(source.get("file_count")) or 0
        status = source.get("status")
        return {
            "status": status if status in _EXECUTOR_STATUS else None,
            "file_count": file_count,
        }
    if role == "reviewer":
        findings = source.get("findings")
        if isinstance(findings, list):
            finding_count = len(findings)
        else:
            finding_count = _nonnegative_int(source.get("finding_count")) or 0
        decision = source.get("decision")
        return {
            "decision": decision if decision in _REVIEW_DECISION else None,
            "finding_count": finding_count,
        }
    return {}


def _tool_event_count(source: dict) -> int:
    if "tool_events" in source:
        count = sum(1 for event in _as_list(source.get("tool_events")) if isinstance(event, dict))
        return min(count, 40)
    existing = source.get("tool_event_count")
    if isinstance(existing, int) and not isinstance(existing, bool) and 0 <= existing <= 40:
        return existing
    return 0


def _redact_check_command(item: object) -> dict:
    source = item if isinstance(item, dict) else {}
    return {
        "check_id": source.get("check_id") if source.get("check_id") in _RUNNER_CHECK_IDS else None,
        "attempt": _nonnegative_int(source.get("attempt")),
        "exit_code": _integer(source.get("exit_code")),
        "duration_seconds": _nonnegative_number(source.get("duration_seconds")),
    }


def _redact_checkpoint(item: object) -> dict:
    source = item if isinstance(item, dict) else {}
    return {
        "checkpoint_id": _checkpoint_id(source.get("checkpoint_id")),
        "step": _integer(source.get("step")),
        "source": _enum(_OUTCOMES, source.get("source")),
        "final_status": _enum(_OUTCOMES, source.get("final_status")),
        "proposed_paths": _relative_paths(source.get("proposed_paths")),
        "approved_paths": _relative_paths(source.get("approved_paths")),
        "execution_attempt": _nonnegative_int(source.get("execution_attempt")),
        "executor_invocations": _nonnegative_int(source.get("executor_invocations")),
        "repair_retries_used": _nonnegative_int(source.get("repair_retries_used")),
        "review_decision": source.get("review_decision")
        if source.get("review_decision") in _REVIEW_DECISION
        else None,
        "errors": _safe_codes(source.get("errors")),
        "verification_outcome": _enum(_OUTCOMES, source.get("verification_outcome")),
        "checks": [_redact_public_check(check) for check in _as_list(source.get("checks"))],
    }


def _redact_public_check(item: object) -> dict:
    source = item if isinstance(item, dict) else {}
    return {
        "check_id": source.get("check_id") if source.get("check_id") in _RUNNER_CHECK_IDS else None,
        "status": source.get("status") if source.get("status") in _CHECK_STATUS else None,
        "exit_code": _integer(source.get("exit_code")),
        "duration_ms": _nonnegative_int(source.get("duration_ms")),
    }


def _redact_planning_report(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    return {
        "run_id": _match(_RUN_ID, value.get("run_id")),
        "outcome": _enum(_OUTCOMES, value.get("outcome")),
        "attempts": _nonnegative_int(value.get("attempts")),
        "executor_invocations": _nonnegative_int(value.get("executor_invocations")),
        "repair_retries_used": _nonnegative_int(value.get("repair_retries_used")),
        "max_retries": _nonnegative_int(value.get("max_retries")),
        "checks": [_redact_public_check(check) for check in _as_list(value.get("checks"))],
        "review_decision": value.get("review_decision")
        if value.get("review_decision") in _REVIEW_DECISION
        else None,
        "evidence_refs": [
            ref for ref in _as_list(value.get("evidence_refs")) if _match(re.compile(r"^ev_[A-Za-z0-9_-]{1,60}$"), ref)
        ],
        "errors": _safe_codes(value.get("errors")),
        "proposed_paths": _relative_paths(value.get("proposed_paths")),
        "attempt_history": [_redact_history(item) for item in _as_list(value.get("attempt_history"))],
        "elapsed_seconds": _nonnegative_number(value.get("elapsed_seconds")),
    }


def _redact_history(item: object) -> dict:
    source = item if isinstance(item, dict) else {}
    return {
        "attempt": _nonnegative_int(source.get("attempt")),
        "verification_outcome": _enum(_OUTCOMES, source.get("verification_outcome")),
        "evidence_refs": [
            ref for ref in _as_list(source.get("evidence_refs")) if _match(re.compile(r"^ev_[A-Za-z0-9_-]{1,60}$"), ref)
        ],
        "review_decision": source.get("review_decision")
        if source.get("review_decision") in _REVIEW_DECISION
        else None,
        "checks": [_redact_public_check(check) for check in _as_list(source.get("checks"))],
        "terminal_error": _enum(_REPORT_CODES, source.get("terminal_error")),
    }


def _redact_execution(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    return {
        "run_id": _match(_RUN_ID, value.get("run_id")),
        "worktree": _absolute_path(value.get("worktree")),
        "approval": _relative_paths(value.get("approval")),
        "report": _redact_planning_report(value.get("report")),
        "checkpoints": [_redact_checkpoint(item) for item in _as_list(value.get("checkpoints"))],
        "error": _enum(_REPORT_CODES, value.get("error")) or _enum(_OUTCOMES, value.get("error")),
    }


def _redact_jaeger(value: object) -> dict:
    source = value if isinstance(value, dict) else {}
    spans = []
    for item in _as_list(source.get("spans")):
        if not isinstance(item, dict):
            continue
        spans.append(
            {
                "operation": _enum(_SPAN_NAMES, item.get("operation")),
                "trace_id": _match(_HEX_TRACE, item.get("trace_id")),
                "duration_ms": _nonnegative_number(item.get("duration_ms")),
                "run_id": _match(_RUN_ID, item.get("run_id")),
                "role": item.get("role") if item.get("role") in {"planner", "executor", "reviewer"} else None,
                "status": item.get("status") if item.get("status") in _AGENT_STATUS else None,
                "elapsed_seconds": _nonnegative_number(item.get("elapsed_seconds")),
                "cli_duration_ms": _nonnegative_int(item.get("cli_duration_ms")),
                "usage_reported": item.get("usage_reported") if isinstance(item.get("usage_reported"), bool) else None,
                "usage_status": _enum(_USAGE_STATUS, item.get("usage_status")),
                "usage_scope": item.get("usage_scope") if item.get("usage_scope") == "cli_result_snapshot" else None,
                "model_identity": _enum(_MODEL_IDENTITY, item.get("model_identity")),
                "cost_status": "unavailable" if item.get("cost_status") == "unavailable" else None,
                "model": sanitized_model(item.get("model")),
                "session_id": sanitized_uuid(item.get("session_id")),
                "request_id": sanitized_uuid(item.get("request_id")),
                "trace_scope": item.get("trace_scope")
                if item.get("trace_scope") in {"single_run", "multi_run"}
                else None,
                "trace_run_count": _nonnegative_int(item.get("trace_run_count")),
                "invocation_id": _match(_INVOCATION_ID, item.get("invocation_id")),
                "diagnostic_code": _enum(_REPORT_CODES, item.get("diagnostic_code")),
                "skill_status": item.get("skill_status")
                if item.get("skill_status") in _USAGE_STATUS
                else None,
                "outcome": _enum(_OUTCOMES, item.get("outcome")),
            }
        )
    return {
        "reachable": source.get("reachable") is True,
        "error": _enum(_NETWORK_ERRORS, source.get("error")),
        "query": _jaeger_query(source.get("query")),
        "ui": source.get("ui") if source.get("ui") == "http://127.0.0.1:16686" else None,
        "span_count": len(spans) if spans else _nonnegative_int(source.get("span_count")),
        "spans": spans,
    }


def _redact_usage_by_run(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    cleaned = {}
    for run_id, totals in value.items():
        if not _match(_RUN_ID, run_id) or not isinstance(totals, dict):
            continue
        usage = _safe_usage(totals.get("usage"))
        reported = _nonnegative_int(totals.get("reported_invocations")) or 0
        partial = _nonnegative_int(totals.get("partial_invocations")) or 0
        unavailable = _nonnegative_int(totals.get("unavailable_invocations")) or 0
        missing_source = totals.get("missing_invocations")
        missing = {}
        if isinstance(missing_source, dict):
            for key in _USAGE_COUNTERS:
                count = _nonnegative_int(missing_source.get(key))
                if count is not None:
                    missing[key] = count
        identity = totals.get("model_identity")
        if identity not in _MODEL_IDENTITY:
            identity = "unknown"
        complete = usage is not None and all(key in usage for key in _USAGE_COUNTERS)
        if reported == 0 and partial == 0:
            status = "unavailable"
        elif partial == 0 and unavailable == 0 and reported > 0 and complete:
            status = "reported"
        else:
            status = "partial"
        cleaned[run_id] = {
            "reported_invocations": reported,
            "partial_invocations": partial,
            "unavailable_invocations": unavailable,
            "usage_status": status,
            "usage": usage,
            "missing_invocations": missing,
            "model_identity": identity,
            "cost_status": "unavailable",
        }
    return cleaned


def _safe_usage(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    kept = {}
    for key in _USAGE_COUNTERS:
        if key not in value:
            continue
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return None
        kept[key] = item
    return kept or None


def _safe_codes(value: object) -> list[str]:
    return [item for item in _as_list(value) if isinstance(item, str) and item in _REPORT_CODES]


def _relative_paths(value: object) -> list[str]:
    paths = []
    for item in _as_list(value):
        path = _relative_path(item)
        if path is not None:
            paths.append(path)
    return paths


def _relative_path(value: object) -> str | None:
    """Keep a repository file path. This is a path-shape rule, not secret detection."""
    if not isinstance(value, str) or len(value) > 180 or "\\" in value or value.startswith("/"):
        return None
    parts = value.split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    for index, part in enumerate(parts):
        stem = part
        if index == len(parts) - 1:
            stem, dot, extension = part.rpartition(".")
            if dot != "." or f".{extension}" not in _PATH_EXTENSIONS:
                return None
        if not stem or not _PATH_SEGMENT.fullmatch(stem) or ("_" in stem and "-" in stem):
            return None
    return value


def _version(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    line = value.strip().splitlines()[0].strip() if value.strip() else ""
    return _match(_VERSION, line)


def _enum(allowed: frozenset[str], value: object) -> str | None:
    if isinstance(value, str) and value in allowed:
        return value
    return None


def _absolute_path(value: object) -> str | None:
    if not isinstance(value, str) or not _ABSOLUTE_PATH.fullmatch(value):
        return None
    parts = value.split("/")
    if not parts or parts[0] != "" or any(part in {"", ".", ".."} for part in parts[1:]):
        return None
    return value


def _checkpoint_id(value: object) -> str | None:
    if _match(_UUID, value) or _match(_CHECKPOINT_ID, value):
        return value if isinstance(value, str) else None
    return None


def _jaeger_query(value: object) -> str | None:
    if not isinstance(value, str) or not value.startswith(f"{JAEGER_TRACES}?"):
        return None
    remainder = value[len(JAEGER_TRACES) + 1 :]
    if not re.fullmatch(r"[A-Za-z0-9_.=&:%+-]{1,400}", remainder):
        return None
    if "T" not in remainder or "service_name=agentic-dev-workflow" not in remainder:
        return None
    return value


def _match(pattern: re.Pattern[str], value: object) -> str | None:
    if isinstance(value, str) and pattern.fullmatch(value):
        return value
    return None


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _jaeger_attributes(items: object) -> dict:
    """Map OTLP attributes. Jaeger v3 sends intValue as a decimal string."""
    attributes = {}
    for item in _as_list(items):
        if not isinstance(item, dict):
            continue
        attributes[item.get("key")] = _jaeger_attribute_value(item.get("value"))
    return attributes


def _jaeger_attribute_value(value: object) -> object:
    if not isinstance(value, dict) or not value:
        return None
    if "intValue" in value:
        return _decimal_nonnegative_int(value.get("intValue"))
    return next(iter(value.values()), None)


def _decimal_nonnegative_int(value: object) -> int | None:
    """Keep a nonnegative integer. Reject bools, floats, negatives, and other text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
        return int(value)
    return None


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _nonnegative_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value < 0 or value == float("inf"):
        return None
    return value


def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _checkpoints(saver, run_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": f"agentic-dev-workflow:orchestration:{run_id}"}}
    rows = []
    for item in saver.list(config):
        values = item.checkpoint.get("channel_values", {})
        metadata = dict(item.metadata or {})
        rows.append(
            {
                "checkpoint_id": item.config["configurable"].get("checkpoint_id"),
                "step": metadata.get("step"),
                "source": metadata.get("source"),
                "final_status": values.get("final_status"),
                "proposed_paths": values.get("proposed_paths"),
                "approved_paths": values.get("approved_paths"),
                "execution_attempt": values.get("execution_attempt"),
                "executor_invocations": values.get("executor_invocations"),
                "repair_retries_used": values.get("repair_retries_used"),
                "review_decision": values.get("review_decision"),
                "errors": _safe_codes(values.get("errors")),
                "verification_outcome": (values.get("verification") or {}).get("outcome")
                if isinstance(values.get("verification"), dict)
                else None,
                "checks": [
                    {
                        "check_id": check.get("check_id"),
                        "status": check.get("status"),
                        "exit_code": check.get("exit_code"),
                        "duration_ms": check.get("duration_ms"),
                    }
                    for check in values.get("checks") or []
                    if isinstance(check, dict)
                ],
            }
        )
    rows.reverse()
    return rows


def _command_view(record) -> dict:
    return {
        "check_id": record.check_id,
        "attempt": record.attempt,
        "exit_code": record.exit_code,
        "duration_seconds": record.duration_seconds,
    }


def _command_output(argv: list[str]) -> str:
    completed = subprocess.run(
        argv,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=20,
        env={**os.environ, "NO_OPEN_BROWSER": "1"},
    )
    return completed.stdout


def _jaeger_summary(run_id: str) -> dict:
    time.sleep(2)
    end = datetime.now(timezone.utc) + timedelta(minutes=5)
    start = end - timedelta(hours=6)
    params = urllib.parse.urlencode(
        {
            "query.service_name": "agentic-dev-workflow",
            "query.start_time_min": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "query.start_time_max": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "query.search_depth": "20",
        }
    )
    try:
        with urllib.request.urlopen(f"{JAEGER_TRACES}?{params}", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except OSError as error:
        return {"reachable": False, "error": type(error).__name__}
    spans = []
    for resource in payload.get("result", {}).get("resourceSpans", []):
        for scope in resource.get("scopeSpans", []):
            for span in scope.get("spans", []):
                attributes = _jaeger_attributes(span.get("attributes", []))
                related_runs = {run_id, f"{run_id}b"}
                owns_run = attributes.get("workflow.run_id") in related_runs
                multi_run = attributes.get("workflow.trace_scope") == "multi_run" and any(
                    isinstance(key, str) and any(f".{item}." in f".{key}." for item in related_runs)
                    for key in attributes
                )
                if not owns_run and not multi_run:
                    continue
                start = int(span.get("startTimeUnixNano", 0))
                end = int(span.get("endTimeUnixNano", 0))
                spans.append(
                    {
                        "operation": span.get("name"),
                        "trace_id": span.get("traceId"),
                        "duration_ms": (end - start) / 1_000_000,
                        "run_id": attributes.get("workflow.run_id"),
                        "trace_scope": attributes.get("workflow.trace_scope"),
                        "trace_run_count": attributes.get("workflow.trace_run_count"),
                        "role": attributes.get("agent.role"),
                        "status": attributes.get("agent.status"),
                        "elapsed_seconds": attributes.get("agent.elapsed_seconds"),
                        "cli_duration_ms": attributes.get("agent.cli_duration_ms"),
                        "usage_reported": attributes.get("agent.usage_reported"),
                        "usage_status": attributes.get("agent.usage_status"),
                        "usage_scope": attributes.get("agent.usage_scope"),
                        "model_identity": attributes.get("agent.model_identity"),
                        "cost_status": attributes.get("agent.cost_status"),
                        "model": attributes.get("agent.model"),
                        "session_id": attributes.get("agent.session_id"),
                        "request_id": attributes.get("agent.request_id"),
                        "invocation_id": attributes.get("agent.invocation_id"),
                        "diagnostic_code": attributes.get("agent.diagnostic_code"),
                        "skill_status": attributes.get("agent.skill_status"),
                        "outcome": attributes.get("workflow.outcome"),
                    }
                )
    return {
        "reachable": True,
        "query": f"{JAEGER_TRACES}?{params}",
        "ui": "http://127.0.0.1:16686",
        "span_count": len(spans),
        "spans": spans,
    }


if __name__ == "__main__":
    raise SystemExit(main())
