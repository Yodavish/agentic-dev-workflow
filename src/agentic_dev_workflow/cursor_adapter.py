"""Cursor Agent CLI adapter. It launches ``cursor-agent --print`` and nothing else.

The CLI is the programmatic interface already installed in this environment.
A missing binary or a missing login is a failed adapter result. This module
does not invent planner, executor, or reviewer JSON.
"""

from __future__ import annotations

import json
import os
import re
import select
import signal
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from agentic_dev_workflow.adapters import (
    AgentFailureCategory,
    AgentRequest,
    AgentResult,
    AgentRole,
    AgentStatus,
)
from agentic_dev_workflow.agent_outputs import (
    AgentOutputValidationError,
    validate_executor_output,
    validate_planner_output,
    validate_reviewer_output,
)

_MAX_STDOUT_BYTES = 4 * 1024 * 1024
_MAX_STDERR_BYTES = 4 * 1024 * 1024
_PIPE_READ_CHUNK = 64 * 1024
_CLEANUP_DEADLINE_SECONDS = 1.0
_MAX_TOOL_EVENTS = 40
_USAGE_COUNTERS = (
    "inputTokens",
    "outputTokens",
    "cacheReadTokens",
    "cacheWriteTokens",
)
KNOWN_MODEL_NAMES = frozenset({"Auto", "auto"})
UUID_TEXT = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
RUN_ID_TEXT = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SKILL_STATUSES = frozenset({"reported", "partial", "unavailable"})
_SPAN_CORRELATION = {
    "invocation_id": "agent.invocation_id",
    "attempt": "agent.attempt",
}
_VALIDATORS = {
    AgentRole.PLANNER: validate_planner_output,
    AgentRole.EXECUTOR: validate_executor_output,
    AgentRole.REVIEWER: validate_reviewer_output,
}


@dataclass(frozen=True)
class CursorObservation:
    """Bounded CLI metadata. Token fields stay absent unless the CLI emitted them."""

    role: str
    duration_seconds: float
    exit_code: int | None
    diagnostic_code: str | None
    session_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    cli_duration_ms: int | None = None
    cli_duration_api_ms: int | None = None
    usage: dict[str, int] | None = None
    tool_events: tuple[dict[str, str], ...] = field(default_factory=tuple)
    stdout_excerpt: str = ""
    stderr_excerpt: str = ""


class CursorCLIAdapter:
    """Invoke the local Cursor Agent CLI in the approved workspace."""

    adapter_type = "cursor-cli"

    def __init__(
        self,
        *,
        workspace: str | Path,
        executable: str | None = None,
        model: str | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.executable = executable
        self.model = model
        self.last_observation: CursorObservation | None = None
        self.last_structured_output: dict[str, object] | None = None

    def invoke(self, request: AgentRequest) -> AgentResult:
        """Run one print-mode agent turn and map its JSON onto the role schema."""
        started = time.monotonic()
        executable = self.executable or resolve_cursor_agent()
        if executable is None:
            return self._failed(
                request,
                AgentFailureCategory.UNAVAILABLE_ADAPTER,
                "cursor_agent_not_found",
                started,
                exit_code=None,
            )
        if not self.workspace.is_dir():
            return self._failed(
                request,
                AgentFailureCategory.EXECUTION_FAILURE,
                "cursor_workspace_missing",
                started,
                exit_code=None,
            )
        try:
            completed = _run_cli(
                _argv(executable, request, workspace=self.workspace, model=self.model),
                cwd=self.workspace,
                timeout_seconds=float(request.timeout_seconds),
            )
        except OSError:
            return self._failed(
                request,
                AgentFailureCategory.UNAVAILABLE_ADAPTER,
                "cursor_launch_failed",
                started,
                exit_code=None,
            )
        duration = time.monotonic() - started
        parsed = _parse_stream(
            completed.stdout,
            drop_partial_tail=completed.timed_out or completed.stdout_truncated,
        )
        observation = _observation(request, completed, parsed, duration)
        self.last_observation = observation
        if completed.timed_out:
            return self._result(
                request,
                AgentStatus.FAILED,
                AgentFailureCategory.TIMEOUT,
                "cursor_timeout",
                duration,
                completed.exit_code,
            )
        if _authentication_rejected(completed.stderr, completed.exit_code, parsed.result):
            return self._result(
                request,
                AgentStatus.FAILED,
                AgentFailureCategory.UNAVAILABLE_ADAPTER,
                "cursor_not_authenticated",
                duration,
                completed.exit_code,
            )
        if completed.exit_code != 0 or parsed.result is None:
            return self._result(
                request,
                AgentStatus.FAILED,
                AgentFailureCategory.EXECUTION_FAILURE,
                "cursor_exit_nonzero",
                duration,
                completed.exit_code,
            )
        if parsed.result.get("is_error") is True:
            return self._result(
                request,
                AgentStatus.FAILED,
                AgentFailureCategory.EXECUTION_FAILURE,
                "cursor_run_error",
                duration,
                completed.exit_code,
            )
        text = parsed.result.get("result")
        if not isinstance(text, str):
            return self._result(
                request,
                AgentStatus.FAILED,
                AgentFailureCategory.INVALID_OUTPUT,
                "cursor_output_invalid",
                duration,
                completed.exit_code,
            )
        payload = _parse_model_json(text)
        if payload is None:
            return self._result(
                request,
                AgentStatus.FAILED,
                AgentFailureCategory.INVALID_OUTPUT,
                "cursor_output_invalid",
                duration,
                completed.exit_code,
            )
        try:
            validated = _VALIDATORS[request.role](payload)
        except AgentOutputValidationError as error:
            return self._result(
                request,
                AgentStatus.FAILED,
                AgentFailureCategory.INVALID_OUTPUT,
                error.diagnostic_code,
                duration,
                completed.exit_code,
            )
        structured = validated.to_state()
        self.last_structured_output = structured
        return AgentResult(
            run_id=request.run_id,
            role=request.role,
            status=AgentStatus.SUCCEEDED,
            structured_output=structured,
            exit_code=completed.exit_code,
            duration_seconds=duration,
        )

    def _failed(
        self,
        request: AgentRequest,
        category: AgentFailureCategory,
        diagnostic_code: str,
        started: float,
        *,
        exit_code: int | None,
    ) -> AgentResult:
        duration = time.monotonic() - started
        self.last_observation = CursorObservation(
            role=request.role.value,
            duration_seconds=duration,
            exit_code=exit_code,
            diagnostic_code=diagnostic_code,
        )
        self.last_structured_output = None
        return self._result(
            request,
            AgentStatus.FAILED,
            category,
            diagnostic_code,
            duration,
            exit_code,
        )

    def _result(
        self,
        request: AgentRequest,
        status: AgentStatus,
        category: AgentFailureCategory,
        diagnostic_code: str,
        duration: float,
        exit_code: int | None,
    ) -> AgentResult:
        if self.last_observation is not None and status != AgentStatus.SUCCEEDED:
            self.last_observation = replace(
                self.last_observation, diagnostic_code=diagnostic_code, exit_code=exit_code
            )
        return AgentResult(
            run_id=request.run_id,
            role=request.role,
            status=status,
            structured_output={},
            exit_code=exit_code,
            duration_seconds=duration,
            failure_category=category,
            diagnostic_code=diagnostic_code,
        )


def resolve_cursor_agent() -> str | None:
    """Return the installed Cursor Agent executable, if one is on this machine."""
    override = os.environ.get("CURSOR_AGENT_BIN")
    if override:
        candidate = Path(override)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        return None
    for name in ("cursor-agent", "agent"):
        found = _which(name)
        if found is not None:
            return found
    installed = (
        Path.home()
        / ".cursor-server/data/User/globalStorage/anysphere.cursor-agent-worker"
        / "agent-cli/.local/bin/cursor-agent"
    )
    if installed.is_file() and os.access(installed, os.X_OK):
        return str(installed)
    return None


@dataclass(frozen=True)
class _Completed:
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool
    stdout_truncated: bool = False


@dataclass(frozen=True)
class _ParsedStream:
    result: dict | None
    model: str | None
    tool_events: tuple[dict[str, str], ...]
    session_id: str | None = None


def _which(name: str) -> str | None:
    from shutil import which

    found = which(name)
    if found and os.access(found, os.X_OK):
        return found
    return None


def _argv(
    executable: str,
    request: AgentRequest,
    *,
    workspace: Path,
    model: str | None,
) -> list[str]:
    argv = [
        executable,
        "--trust",
        "--print",
        "--output-format",
        "stream-json",
        "--workspace",
        str(workspace),
        "--single-turn",
    ]
    if request.role == AgentRole.EXECUTOR:
        argv.extend(["--force", "--sandbox", "disabled"])
    elif request.role == AgentRole.REVIEWER:
        argv.extend(["--mode", "ask", "--sandbox", "enabled"])
    else:
        argv.extend(["--mode", "plan", "--sandbox", "enabled"])
    if model:
        argv.extend(["--model", model])
    argv.append(_prompt(request))
    return argv


def _prompt(request: AgentRequest) -> str:
    context = json.dumps(request.structured_context, sort_keys=True)
    schema = _SCHEMA[request.role]
    return (
        "You are one role inside a local orchestration graph. "
        "Return only a JSON object matching the schema. "
        "Do not wrap it in markdown.\n"
        f"Role: {request.role.value}\n"
        f"Schema:\n{schema}\n"
        f"Task:\n{request.task}\n"
        f"Context:\n{context}\n"
        f"{_ROLE_NOTE[request.role]}"
    )


_SCHEMA = {
    AgentRole.PLANNER: (
        '{"summary":"string","assumptions":["string"],'
        '"steps":[{"id":"step-1","description":"string","expected_files":["relative/path.py"]}],'
        '"verification_check_ids":["runner.tests"]}'
    ),
    AgentRole.EXECUTOR: (
        '{"status":"completed","summary":"string","files_changed":["relative/path.py"],'
        '"checks_claimed":[],"failure_category":null,"diagnostic_code":null}'
    ),
    AgentRole.REVIEWER: '{"decision":"approve","findings":[]}',
}

_ROLE_NOTE = {
    AgentRole.PLANNER: (
        "Do not edit files. verification_check_ids must be a non-empty subset of "
        "runner.tests, runner.compile, runner.diff-check, and runner.lock-check. "
        "runner.tests discovers test modules under tests/. "
        "runner.compile runs compileall on src and tests. "
        "The task text is the requirements. A SPEC.md file is not required."
    ),
    AgentRole.EXECUTOR: (
        "Edit only the approved relative paths in this workspace. "
        "Follow the specification in the structured context. A SPEC.md file is not required. "
        "Do not create, rename, or delete any other path. "
        "A completed status requires failure_category and diagnostic_code to be null."
    ),
    AgentRole.REVIEWER: (
        "Do not edit files. Approve only when the verification outcome is pass and "
        "you have no blocking finding. A rejection requires one blocking finding. "
        "Use only evidence_refs supplied in the context. "
        "finding_id values look like finding-1. "
        "file must be a repository-relative path."
    ),
}


def _run_cli(argv: list[str], *, cwd: Path, timeout_seconds: float) -> _Completed:
    environment = os.environ.copy()
    environment["NO_OPEN_BROWSER"] = "1"
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    try:
        stdout, stderr, timed_out, stdout_truncated = _collect_process_output(process, deadline)
    finally:
        _close_pipes(process)
        if process.poll() is None:
            _terminate(process)
            _wait_bounded(process, time.monotonic() + _CLEANUP_DEADLINE_SECONDS)
    return _Completed(
        process.returncode,
        stdout,
        stderr,
        timed_out,
        stdout_truncated,
    )


def _collect_process_output(
    process: subprocess.Popen,
    deadline: float,
) -> tuple[bytes, bytes, bool, bool]:
    """Read both pipes up to a fixed cap. Excess bytes are discarded, not stored."""
    stdout_buf = bytearray()
    stderr_buf = bytearray()
    stdout_truncated = False
    stderr_truncated = False
    stdout_open = process.stdout is not None
    stderr_open = process.stderr is not None
    for pipe in (process.stdout, process.stderr):
        if pipe is not None:
            os.set_blocking(pipe.fileno(), False)

    def pump(until: float) -> bool:
        nonlocal stdout_open, stderr_open, stdout_truncated, stderr_truncated
        hit_deadline = False
        while stdout_open or stderr_open:
            now = time.monotonic()
            if now >= until:
                hit_deadline = True
                break
            readable = []
            if stdout_open and process.stdout is not None:
                readable.append(process.stdout)
            if stderr_open and process.stderr is not None:
                readable.append(process.stderr)
            if not readable:
                break
            try:
                ready, _, _ = select.select(readable, [], [], min(0.2, until - now))
            except (ValueError, OSError):
                break
            if process.stdout in ready:
                stdout_open, stdout_truncated = _absorb(
                    process.stdout, stdout_buf, _MAX_STDOUT_BYTES, stdout_truncated
                )
            if process.stderr in ready:
                stderr_open, stderr_truncated = _absorb(
                    process.stderr, stderr_buf, _MAX_STDERR_BYTES, stderr_truncated
                )
        if process.poll() is None and not hit_deadline:
            if not _wait_bounded(process, until):
                hit_deadline = True
        return hit_deadline

    timed_out = pump(deadline)
    if timed_out or process.poll() is None:
        timed_out = True
        _terminate(process)
        pump(time.monotonic() + _CLEANUP_DEADLINE_SECONDS)
        if process.poll() is None:
            _terminate(process)
            _wait_bounded(process, time.monotonic() + _CLEANUP_DEADLINE_SECONDS)
    else:
        _wait_bounded(process, time.monotonic() + _CLEANUP_DEADLINE_SECONDS)
    stdout = bytes(stdout_buf)
    del stdout_buf
    stderr = bytes(stderr_buf)
    del stderr_buf
    return stdout, stderr, timed_out, stdout_truncated


def _absorb(
    pipe,
    bucket: bytearray,
    limit: int,
    truncated: bool,
) -> tuple[bool, bool]:
    try:
        chunk = os.read(pipe.fileno(), _PIPE_READ_CHUNK)
    except BlockingIOError:
        return True, truncated
    except OSError:
        return False, truncated
    if not chunk:
        return False, truncated
    room = limit - len(bucket)
    if room > 0:
        bucket.extend(chunk[:room])
    if len(chunk) > max(room, 0):
        truncated = True
    return True, truncated


def _wait_bounded(process: subprocess.Popen, until: float) -> bool:
    remaining = until - time.monotonic()
    if remaining < 0:
        remaining = 0
    try:
        process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        return False
    return True


def _close_pipes(process: subprocess.Popen) -> None:
    for pipe in (process.stdout, process.stderr):
        if pipe is None or pipe.closed:
            continue
        try:
            pipe.close()
        except OSError:
            pass


def _terminate(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            pass


def _parse_stream(stdout: bytes, *, drop_partial_tail: bool = False) -> _ParsedStream:
    if drop_partial_tail and not stdout.endswith(b"\n"):
        newline = stdout.rfind(b"\n")
        stdout = b"" if newline < 0 else stdout[:newline]
    result = None
    model = None
    session_id = None
    tools: list[dict[str, str]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "system" and event.get("subtype") == "init":
            if model is None and isinstance(event.get("model"), str):
                model = event["model"]
            if session_id is None:
                session_id = _optional_str(event.get("session_id"))
        elif event.get("type") == "tool_call" and len(tools) < _MAX_TOOL_EVENTS:
            summary = _tool_summary(event)
            if summary is not None:
                tools.append(summary)
        elif event.get("type") == "result":
            result = event
    return _ParsedStream(result, model, tuple(tools), session_id)


def _tool_summary(event: dict) -> dict[str, str] | None:
    tool_call = event.get("tool_call")
    name = "tool_call"
    path = ""
    if isinstance(tool_call, dict):
        for key, value in tool_call.items():
            if not isinstance(key, str):
                continue
            name = key
            if isinstance(value, dict):
                args = value.get("args")
                if isinstance(args, dict):
                    for path_key in ("path", "file", "file_path", "target_file"):
                        candidate = args.get(path_key)
                        if isinstance(candidate, str):
                            path = candidate[:200]
                            break
            break
    subtype = event.get("subtype")
    return {
        "subtype": subtype if isinstance(subtype, str) else "",
        "tool": name[:80],
        "path": path,
    }


def _observation(
    request: AgentRequest,
    completed: _Completed,
    parsed: _ParsedStream,
    duration: float,
) -> CursorObservation:
    result = parsed.result or {}
    return CursorObservation(
        role=request.role.value,
        duration_seconds=duration,
        exit_code=completed.exit_code,
        diagnostic_code=None,
        session_id=_optional_str(result.get("session_id")) or parsed.session_id,
        request_id=_optional_str(result.get("request_id")),
        model=parsed.model,
        cli_duration_ms=_optional_int(result.get("duration_ms")),
        cli_duration_api_ms=_optional_int(result.get("duration_api_ms")),
        usage=_usage(result.get("usage")),
        tool_events=parsed.tool_events,
    )


def _usage(value: object) -> dict[str, int] | None:
    """Keep the emitted counters. Missing fields stay absent; invalid ones discard usage."""
    if not isinstance(value, dict):
        return None
    kept: dict[str, int] = {}
    for key in _USAGE_COUNTERS:
        if key not in value:
            continue
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return None
        kept[key] = item
    return kept or None


def usage_completeness(usage: dict[str, int] | None) -> str:
    """Classify which counters are present. This is not a billing-coverage claim."""
    if not isinstance(usage, dict):
        return "unavailable"
    present = [key for key in _USAGE_COUNTERS if key in usage]
    if len(present) == len(_USAGE_COUNTERS):
        return "reported"
    if present:
        return "partial"
    return "unavailable"


def classify_model_identity(model: object) -> str:
    """Display names and missing metadata are not provider model ids."""
    return "unknown"


def sanitized_model(value: object) -> str | None:
    """Keep the approved display labels. Every other model string is omitted."""
    if isinstance(value, str) and value in KNOWN_MODEL_NAMES:
        return value
    return None


def sanitized_uuid(value: object) -> str | None:
    """Keep a canonical UUID. Other session or request text is omitted."""
    if isinstance(value, str) and UUID_TEXT.fullmatch(value):
        return value
    return None


def aggregate_usage_by_run(
    invocations: list[tuple],
) -> dict[str, dict]:
    """Sum counters per workflow run. Missing counters stay missing."""
    grouped: dict[str, dict] = {}
    for item in invocations:
        run_id = item[0]
        usage = item[1] if len(item) > 1 else None
        identity = item[2] if len(item) > 2 else "unknown"
        if identity not in {"known", "unknown", "mixed"}:
            identity = "unknown"
        bucket = grouped.setdefault(
            run_id,
            {
                "reported_invocations": 0,
                "partial_invocations": 0,
                "unavailable_invocations": 0,
                "usage": {},
                "missing_invocations": {key: 0 for key in _USAGE_COUNTERS},
                "identities": [],
            },
        )
        bucket["identities"].append(identity)
        parsed = _usage(usage)
        status = usage_completeness(parsed)
        if status == "unavailable":
            bucket["unavailable_invocations"] += 1
            for key in _USAGE_COUNTERS:
                bucket["missing_invocations"][key] += 1
            continue
        bucket["reported_invocations" if status == "reported" else "partial_invocations"] += 1
        for key in _USAGE_COUNTERS:
            if parsed is None or key not in parsed:
                bucket["missing_invocations"][key] += 1
                continue
            bucket["usage"][key] = bucket["usage"].get(key, 0) + parsed[key]
    summarized = {}
    for run_id, bucket in grouped.items():
        usage = bucket["usage"] or None
        summarized[run_id] = {
            "reported_invocations": bucket["reported_invocations"],
            "partial_invocations": bucket["partial_invocations"],
            "unavailable_invocations": bucket["unavailable_invocations"],
            "usage_status": _run_usage_status(
                bucket["reported_invocations"],
                bucket["partial_invocations"],
                bucket["unavailable_invocations"],
                usage,
            ),
            "usage": usage,
            "missing_invocations": bucket["missing_invocations"],
            "model_identity": _combine_model_identity(bucket["identities"]),
            "cost_status": "unavailable",
        }
    return summarized


def _run_usage_status(
    reported: int, partial: int, unavailable: int, usage: dict | None
) -> str:
    if reported == 0 and partial == 0:
        return "unavailable"
    complete = isinstance(usage, dict) and all(key in usage for key in _USAGE_COUNTERS)
    if partial == 0 and unavailable == 0 and reported > 0 and complete:
        return "reported"
    return "partial"


def _combine_model_identity(identities: list[str]) -> str:
    unique = {item for item in identities if item in {"known", "unknown", "mixed"}}
    if len(unique) == 1:
        return unique.pop()
    if len(unique) > 1:
        return "mixed"
    return "unknown"


def aggregate_skill_by_run(invocations: list[tuple]) -> dict[str, dict]:
    """Count skill observation per run. This does not count skills used.

    The current CLI stream has no skill event. A call whose status is missing
    or unrecognized is unavailable. A run is unavailable when every call is
    unavailable, reported when every call is reported, and partial when the
    calls disagree. No skill count is emitted.
    """
    grouped: dict[str, dict[str, int]] = {}
    for item in invocations:
        run_id = item[0]
        status = item[1] if len(item) > 1 else "unavailable"
        if not isinstance(run_id, str) or RUN_ID_TEXT.fullmatch(run_id) is None:
            continue
        if status not in _SKILL_STATUSES:
            status = "unavailable"
        bucket = grouped.setdefault(run_id, {"reported": 0, "partial": 0, "unavailable": 0})
        bucket[status] += 1
    summarized = {}
    for run_id, bucket in grouped.items():
        reported = bucket["reported"]
        partial = bucket["partial"]
        unavailable = bucket["unavailable"]
        if reported == 0 and partial == 0:
            skill_status = "unavailable"
        elif unavailable == 0 and partial == 0 and reported > 0:
            skill_status = "reported"
        else:
            skill_status = "partial"
        summarized[run_id] = {
            "skill_status": skill_status,
            "invocations_without_skill_observation": unavailable,
        }
    return summarized


def apply_invocation_attributes(span, observation: CursorObservation | None, request=None) -> None:
    """Record one invocation on a span without copying the prompt or context."""
    usage = None if observation is None else observation.usage
    span.set_attribute("agent.usage_reported", usage is not None)
    span.set_attribute("agent.usage_status", usage_completeness(usage))
    span.set_attribute("agent.usage_scope", "cli_result_snapshot")
    span.set_attribute("agent.model_identity", classify_model_identity(None if observation is None else observation.model))
    span.set_attribute("agent.cost_status", "unavailable")
    span.set_attribute("agent.skill_status", "unavailable")
    if usage:
        for key, value in usage.items():
            span.set_attribute(f"agent.usage.{key}", value)
    if observation is not None:
        model = sanitized_model(observation.model)
        if model is not None:
            span.set_attribute("agent.model", model)
        session_id = sanitized_uuid(observation.session_id)
        if session_id is not None:
            span.set_attribute("agent.session_id", session_id)
        request_id = sanitized_uuid(observation.request_id)
        if request_id is not None:
            span.set_attribute("agent.request_id", request_id)
    metadata = getattr(request, "correlation_metadata", None)
    if isinstance(metadata, dict):
        for source, attribute in _SPAN_CORRELATION.items():
            _set_text(span, attribute, metadata.get(source))


def apply_parent_trace_identity(span, run_ids, *, fallback_run_id: str | None = None) -> None:
    """Label a parent span without treating several runs as one run id."""
    unique: list[str] = []
    for run_id in run_ids:
        if isinstance(run_id, str) and run_id and run_id not in unique:
            unique.append(run_id)
    if len(unique) > 1:
        _clear_span_attribute(span, "workflow.run_id")
        span.set_attribute("workflow.trace_scope", "multi_run")
        span.set_attribute("workflow.trace_run_count", len(unique))
        return
    span.set_attribute("workflow.trace_scope", "single_run")
    chosen = unique[0] if unique else fallback_run_id
    if isinstance(chosen, str) and chosen:
        span.set_attribute("workflow.run_id", chosen)


def _clear_span_attribute(span, key: str) -> None:
    attributes = getattr(span, "_attributes", None)
    if attributes is not None and key in attributes:
        del attributes[key]


def apply_run_skill_attributes(span, skill_by_run: dict[str, dict]) -> None:
    """Attach per-run skill observation. Attribute values stay in the fixed status set."""
    for run_id, totals in skill_by_run.items():
        if not isinstance(run_id, str) or RUN_ID_TEXT.fullmatch(run_id) is None or not isinstance(totals, dict):
            continue
        prefix = f"workflow.skill.{run_id}"
        status = totals.get("skill_status")
        if status in _SKILL_STATUSES:
            span.set_attribute(f"{prefix}.skill_status", status)
        count = totals.get("invocations_without_skill_observation")
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            span.set_attribute(f"{prefix}.invocations_without_skill_observation", count)


def apply_run_usage_attributes(span, usage_by_run: dict[str, dict]) -> None:
    """Attach per-run totals. A shared trace must not collapse those runs together."""
    for run_id, totals in usage_by_run.items():
        if not isinstance(run_id, str) or not run_id or not isinstance(totals, dict):
            continue
        prefix = f"workflow.usage.{run_id}"
        for key in ("usage_status", "model_identity", "cost_status"):
            value = totals.get(key)
            if isinstance(value, str) and value:
                span.set_attribute(f"{prefix}.{key}", value)
        for key in ("reported_invocations", "partial_invocations", "unavailable_invocations"):
            value = totals.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                span.set_attribute(f"{prefix}.{key}", value)
        usage = totals.get("usage")
        if isinstance(usage, dict):
            for key in _USAGE_COUNTERS:
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    span.set_attribute(f"{prefix}.{key}", value)
        missing = totals.get("missing_invocations")
        if isinstance(missing, dict):
            for key in _USAGE_COUNTERS:
                value = missing.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    span.set_attribute(f"{prefix}.missing.{key}", value)


def _set_text(span, key: str, value: object) -> None:
    if isinstance(value, str) and value:
        span.set_attribute(key, value[:128])


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value[:128]
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _authentication_rejected(
    stderr: bytes, exit_code: int | None, result: dict | None
) -> bool:
    if result is not None and result.get("is_error") is not True and exit_code == 0:
        return False
    text = stderr.decode("utf-8", "replace").lower()
    markers = (
        "authentication required",
        "not logged in",
        "cursor_api_key",
        "cursor_auth_token",
    )
    return any(marker in text for marker in markers)


def _parse_model_json(text: str) -> dict | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
