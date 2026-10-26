"""Cursor CLI adapter tests. The fixture executable is not a live model."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from agentic_dev_workflow.adapters import (
    AgentFailureCategory,
    AgentRequest,
    AgentRole,
    AgentStatus,
)
from agentic_dev_workflow.cursor_adapter import CursorCLIAdapter, CursorObservation, _argv


PLAN = {
    "summary": "Add the validator",
    "assumptions": [],
    "steps": [
        {
            "id": "step-1",
            "description": "Add the module and its tests",
            "expected_files": ["transaction_actions.py", "test_transaction_actions.py"],
        }
    ],
    "verification_check_ids": ["runner.tests"],
}


def request() -> AgentRequest:
    return AgentRequest(
        run_id="cursor-run-1",
        role=AgentRole.PLANNER,
        task="Implement the feature in SPEC.md",
        structured_context={"phase": "plan"},
        timeout_seconds=15,
    )


def write_executable(directory: Path, source: str) -> Path:
    path = directory / "cursor-agent"
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


_OMIT = object()
REPORTED_USAGE = {
    "inputTokens": 10,
    "outputTokens": 4,
    "cacheReadTokens": 40,
    "cacheWriteTokens": 2,
}


def init_event() -> dict:
    return {
        "type": "system",
        "subtype": "init",
        "model": "Auto",
        "session_id": "sess-init",
        "apiKeySource": "login",
    }


def result_event(usage: object = _OMIT, *, text: str | None = None) -> dict:
    event = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 7,
        "duration_api_ms": 7,
        "result": json.dumps(PLAN) if text is None else text,
        "session_id": "sess-result",
        "request_id": "req-result",
    }
    if usage is not _OMIT:
        event["usage"] = usage
    return event


def stream_script(events: list[dict], *, exit_code: int = 0, sleep_seconds: float = 0) -> str:
    return (
        "#!/usr/bin/env python3\n"
        "import json, sys, time\n"
        f"events = json.loads({json.dumps(json.dumps(events))})\n"
        "for event in events:\n"
        "    sys.stdout.write(json.dumps(event) + '\\n')\n"
        "    sys.stdout.flush()\n"
        f"time.sleep({float(sleep_seconds)})\n"
        f"raise SystemExit({int(exit_code)})\n"
    )


class CursorCLIAdapterTests(unittest.TestCase):
    def test_authentication_failure_does_not_invent_output_or_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = write_executable(
                root,
                "#!/bin/sh\n"
                "echo 'Error: Authentication required. Set CURSOR_API_KEY.' 1>&2\n"
                "exit 1\n",
            )
            adapter = CursorCLIAdapter(workspace=root, executable=str(executable))
            result = adapter.invoke(request())
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.diagnostic_code, "cursor_not_authenticated")
        self.assertEqual(result.structured_output, {})
        self.assertIsNotNone(adapter.last_observation)
        self.assertIsNone(adapter.last_observation.usage)
        self.assertIsNone(adapter.last_structured_output)

    def test_stream_json_plan_is_validated_and_usage_stays_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [
                {
                    "type": "system",
                    "subtype": "init",
                    "model": "auto",
                    "session_id": "sess-1",
                },
                {
                    "type": "tool_call",
                    "subtype": "completed",
                    "tool_call": {
                        "readToolCall": {
                            "args": {"path": "SPEC.md", "contents": "x" * 5000},
                        }
                    },
                },
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "duration_ms": 50,
                    "duration_api_ms": 40,
                    "result": json.dumps(PLAN),
                    "session_id": "sess-1",
                    "request_id": "req-1",
                },
            ]
            script = (
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                f"events = json.loads({json.dumps(json.dumps(events))})\n"
                "for event in events:\n"
                "    sys.stdout.write(json.dumps(event) + '\\n')\n"
            )
            executable = write_executable(root, script)
            adapter = CursorCLIAdapter(workspace=root, executable=str(executable))
            result = adapter.invoke(request())
        self.assertEqual(result.status, AgentStatus.SUCCEEDED)
        self.assertEqual(result.structured_output["verification_check_ids"], ["runner.tests"])
        self.assertEqual(
            result.structured_output["steps"][0]["expected_files"],
            ["transaction_actions.py", "test_transaction_actions.py"],
        )
        observation = adapter.last_observation
        self.assertIsNotNone(observation)
        self.assertIsNone(observation.usage)
        self.assertEqual(observation.cli_duration_ms, 50)
        self.assertEqual(observation.model, "auto")
        self.assertEqual(observation.session_id, "sess-1")
        self.assertEqual(observation.tool_events, ({"subtype": "completed", "tool": "readToolCall", "path": "SPEC.md"},))
        self.assertNotIn("contents", json.dumps(observation.tool_events))

    def test_trust_is_the_first_flag_and_cwd_is_the_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            argv = _argv("cursor-agent", request(), workspace=root, model=None)
            self.assertEqual(argv[0], "cursor-agent")
            self.assertEqual(argv[1:5], ["--trust", "--print", "--output-format", "stream-json"])
            workspace_flag = argv.index("--workspace")
            self.assertEqual(argv[workspace_flag + 1], str(root))
            self.assertLess(argv.index("--trust"), len(argv) - 1)
            script = (
                "#!/usr/bin/env python3\n"
                "import os, sys\n"
                "sys.stderr.write('Workspace Trust Required\\n')\n"
                "open('seen-cwd.txt','w',encoding='utf-8').write(os.getcwd())\n"
                "sys.stdout.write("
                + json.dumps(
                    json.dumps(
                        {
                            "type": "result",
                            "subtype": "success",
                            "is_error": False,
                            "duration_ms": 5,
                            "result": json.dumps(PLAN),
                        }
                    )
                    + "\n"
                )
                + ")\n"
            )
            executable = write_executable(root, script)
            adapter = CursorCLIAdapter(workspace=root, executable=str(executable))
            result = adapter.invoke(request())
            seen_cwd = (root / "seen-cwd.txt").read_text(encoding="utf-8")
            observation = adapter.last_observation
        self.assertEqual(result.status, AgentStatus.SUCCEEDED)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(seen_cwd, str(root))
        self.assertEqual(observation.stdout_excerpt, "")
        self.assertEqual(observation.stderr_excerpt, "")

    def invoke_events(
        self,
        events: list[dict],
        *,
        exit_code: int = 0,
        sleep_seconds: float = 0,
        timeout_seconds: float = 15,
        correlation_metadata: dict[str, str] | None = None,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = write_executable(
                root,
                stream_script(events, exit_code=exit_code, sleep_seconds=sleep_seconds),
            )
            adapter = CursorCLIAdapter(workspace=root, executable=str(executable))
            agent_request = AgentRequest(
                run_id="cursor-run-1",
                role=AgentRole.PLANNER,
                task="Implement the feature in SPEC.md",
                structured_context={"phase": "plan"},
                timeout_seconds=timeout_seconds,
                correlation_metadata=correlation_metadata or {},
            )
            result = adapter.invoke(agent_request)
            return result, adapter.last_observation

    def test_successful_result_preserves_four_usage_counters(self) -> None:
        result, observation = self.invoke_events([init_event(), result_event(REPORTED_USAGE)])
        self.assertEqual(result.status, AgentStatus.SUCCEEDED)
        self.assertEqual(observation.usage, REPORTED_USAGE)
        self.assertEqual(observation.model, "Auto")
        self.assertEqual(observation.session_id, "sess-result")
        self.assertEqual(observation.request_id, "req-result")

    def test_missing_usage_stays_absent_and_zero_usage_is_preserved(self) -> None:
        _missing_result, missing = self.invoke_events([init_event(), result_event(_OMIT)])
        zeros = {
            "inputTokens": 0,
            "outputTokens": 0,
            "cacheReadTokens": 0,
            "cacheWriteTokens": 0,
        }
        _zero_result, reported_zero = self.invoke_events([result_event(zeros)])
        self.assertIsNone(missing.usage)
        self.assertEqual(reported_zero.usage, zeros)

    def test_cache_tokens_are_not_added_back_to_input_tokens(self) -> None:
        _result, observation = self.invoke_events([result_event(REPORTED_USAGE)])
        self.assertEqual(observation.usage["inputTokens"], 10)
        self.assertEqual(observation.usage["cacheReadTokens"], 40)
        self.assertEqual(observation.usage["cacheWriteTokens"], 2)
        self.assertNotEqual(
            observation.usage["inputTokens"],
            REPORTED_USAGE["inputTokens"]
            + REPORTED_USAGE["cacheReadTokens"]
            + REPORTED_USAGE["cacheWriteTokens"],
        )

    def test_usage_is_retained_when_role_output_validation_fails(self) -> None:
        result, observation = self.invoke_events(
            [init_event(), result_event(REPORTED_USAGE, text='{"summary": "missing steps"}')]
        )
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_category, AgentFailureCategory.INVALID_OUTPUT)
        self.assertEqual(result.structured_output, {})
        self.assertEqual(observation.usage, REPORTED_USAGE)

    def test_nonzero_exit_with_complete_result_preserves_usage(self) -> None:
        result, observation = self.invoke_events(
            [init_event(), result_event(REPORTED_USAGE)],
            exit_code=1,
        )
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.diagnostic_code, "cursor_exit_nonzero")
        self.assertEqual(observation.usage, REPORTED_USAGE)
        self.assertEqual(observation.exit_code, 1)
        from agentic_dev_workflow.cursor_adapter import usage_completeness

        self.assertEqual(usage_completeness(observation.usage), "reported")

    def test_timeout_parses_already_captured_usage_without_waiting(self) -> None:
        started = __import__("time").monotonic()
        result, observation = self.invoke_events(
            [init_event(), result_event(REPORTED_USAGE)],
            sleep_seconds=30,
            timeout_seconds=0.6,
        )
        elapsed = __import__("time").monotonic() - started
        self.assertLess(elapsed, 5)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.diagnostic_code, "cursor_timeout")
        self.assertEqual(observation.usage, REPORTED_USAGE)
        self.assertEqual(observation.model, "Auto")
        self.assertEqual(observation.session_id, "sess-result")
        from agentic_dev_workflow.cursor_adapter import usage_completeness

        self.assertEqual(usage_completeness(observation.usage), "reported")

    def test_timeout_without_result_does_not_invent_usage(self) -> None:
        result, observation = self.invoke_events(
            [init_event(), {"type": "thinking", "subtype": "delta", "text": "partial"}],
            sleep_seconds=30,
            timeout_seconds=0.6,
        )
        self.assertEqual(result.diagnostic_code, "cursor_timeout")
        self.assertIsNone(observation.usage)
        self.assertEqual(observation.model, "Auto")
        self.assertEqual(observation.session_id, "sess-init")
        self.assertIsNone(observation.request_id)
        from agentic_dev_workflow.cursor_adapter import usage_completeness

        self.assertEqual(usage_completeness(observation.usage), "unavailable")

    def test_init_metadata_is_kept_when_result_event_is_missing(self) -> None:
        result, observation = self.invoke_events([init_event()], exit_code=1)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIsNone(observation.usage)
        self.assertEqual(observation.model, "Auto")
        self.assertEqual(observation.session_id, "sess-init")
        self.assertIsNone(observation.request_id)

    def test_invalid_counters_leave_usage_unavailable(self) -> None:
        cases = [
            {"inputTokens": -1, "outputTokens": 4, "cacheReadTokens": 0, "cacheWriteTokens": 0},
            {"inputTokens": True, "outputTokens": 4, "cacheReadTokens": 0, "cacheWriteTokens": 0},
            {"inputTokens": 1.5, "outputTokens": 4, "cacheReadTokens": 0, "cacheWriteTokens": 0},
            {"inputTokens": "10", "outputTokens": 4, "cacheReadTokens": 0, "cacheWriteTokens": 0},
            ["inputTokens", 10],
        ]
        for usage in cases:
            with self.subTest(usage=usage):
                _result, observation = self.invoke_events([result_event(usage)])
                self.assertIsNone(observation.usage)

    def test_two_complete_result_events_keep_only_the_last_usage(self) -> None:
        later = {
            "inputTokens": 99,
            "outputTokens": 1,
            "cacheReadTokens": 0,
            "cacheWriteTokens": 0,
        }
        _result, observation = self.invoke_events(
            [result_event(REPORTED_USAGE), result_event(later)]
        )
        self.assertEqual(observation.usage, later)
        self.assertNotEqual(observation.usage["inputTokens"], REPORTED_USAGE["inputTokens"])

    def test_correlation_metadata_is_not_copied_into_the_prompt(self) -> None:
        sentinel = "CORR_SENTINEL_do_not_prompt"
        agent_request = AgentRequest(
            run_id="cursor-run-1",
            role=AgentRole.PLANNER,
            task="Implement the feature in SPEC.md",
            structured_context={"phase": "plan"},
            timeout_seconds=15,
            correlation_metadata={
                "role": "planner",
                "invocation": "1",
                "invocation_id": "planner:1",
                "sentinel": sentinel,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            argv = _argv("cursor-agent", agent_request, workspace=root, model=None)
        self.assertNotIn(sentinel, "\n".join(argv))
        self.assertNotIn("planner:1", argv[-1])
        self.assertNotIn("invocation_id", argv[-1])


class UsageTelemetryTests(unittest.TestCase):
    def test_run_totals_keep_each_run_and_failed_attempts(self) -> None:
        from agentic_dev_workflow.cursor_adapter import aggregate_usage_by_run

        grouped = aggregate_usage_by_run(
            [
                (
                    "run-a",
                    {
                        "inputTokens": 10,
                        "outputTokens": 1,
                        "cacheReadTokens": 4,
                        "cacheWriteTokens": 0,
                    },
                ),
                ("run-a", None),
                (
                    "run-a",
                    {
                        "inputTokens": 3,
                        "outputTokens": 2,
                        "cacheReadTokens": 0,
                        "cacheWriteTokens": 1,
                    },
                ),
                (
                    "run-b",
                    {
                        "inputTokens": 7,
                        "outputTokens": 0,
                        "cacheReadTokens": 0,
                        "cacheWriteTokens": 0,
                    },
                ),
            ]
        )
        run_a = grouped["run-a"]
        self.assertEqual(run_a["usage"]["inputTokens"], 13)
        self.assertEqual(run_a["usage"]["outputTokens"], 3)
        self.assertEqual(run_a["usage"]["cacheReadTokens"], 4)
        self.assertEqual(run_a["usage"]["cacheWriteTokens"], 1)
        self.assertEqual(run_a["reported_invocations"], 2)
        self.assertEqual(run_a["partial_invocations"], 0)
        self.assertEqual(run_a["unavailable_invocations"], 1)
        self.assertEqual(run_a["usage_status"], "partial")
        self.assertEqual(run_a["missing_invocations"]["inputTokens"], 1)
        self.assertEqual(run_a["model_identity"], "unknown")
        self.assertEqual(run_a["cost_status"], "unavailable")
        self.assertEqual(grouped["run-b"]["usage"]["inputTokens"], 7)
        self.assertEqual(grouped["run-b"]["usage_status"], "reported")
        self.assertEqual(grouped["run-b"]["missing_invocations"]["outputTokens"], 0)
        self.assertEqual(set(grouped), {"run-a", "run-b"})

    def test_partial_usage_does_not_invent_missing_counters(self) -> None:
        from agentic_dev_workflow.cursor_adapter import aggregate_usage_by_run

        grouped = aggregate_usage_by_run([("run-a", {"inputTokens": 5})])
        bucket = grouped["run-a"]
        self.assertEqual(bucket["usage"], {"inputTokens": 5})
        self.assertEqual(bucket["usage_status"], "partial")
        self.assertEqual(bucket["reported_invocations"], 0)
        self.assertEqual(bucket["partial_invocations"], 1)
        self.assertEqual(bucket["unavailable_invocations"], 0)
        self.assertEqual(bucket["missing_invocations"]["inputTokens"], 0)
        self.assertEqual(bucket["missing_invocations"]["outputTokens"], 1)
        self.assertEqual(bucket["missing_invocations"]["cacheReadTokens"], 1)
        self.assertEqual(bucket["missing_invocations"]["cacheWriteTokens"], 1)

    def test_mixed_calls_record_missing_counters_without_mixing_runs(self) -> None:
        from agentic_dev_workflow.cursor_adapter import (
            aggregate_usage_by_run,
            apply_run_usage_attributes,
            classify_model_identity,
            usage_completeness,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        zeros = {
            "inputTokens": 0,
            "outputTokens": 0,
            "cacheReadTokens": 0,
            "cacheWriteTokens": 0,
        }
        self.assertEqual(usage_completeness(zeros), "reported")
        self.assertEqual(usage_completeness({"outputTokens": 3}), "partial")
        self.assertEqual(usage_completeness(None), "unavailable")
        for model in ("Auto", "auto", "gpt-5", None, ""):
            self.assertEqual(classify_model_identity(model), "unknown")

        grouped = aggregate_usage_by_run(
            [
                ("run-a", REPORTED_USAGE, "unknown"),
                ("run-a", {"inputTokens": 5}, "unknown"),
                ("run-a", None, "unknown"),
                ("run-b", zeros, "unknown"),
                ("run-mix", REPORTED_USAGE, "known"),
                ("run-mix", REPORTED_USAGE, "unknown"),
            ]
        )
        mixed = grouped["run-a"]
        self.assertEqual(mixed["reported_invocations"], 1)
        self.assertEqual(mixed["partial_invocations"], 1)
        self.assertEqual(mixed["unavailable_invocations"], 1)
        self.assertEqual(mixed["usage"]["inputTokens"], 15)
        self.assertEqual(mixed["usage"]["outputTokens"], 4)
        self.assertEqual(mixed["missing_invocations"]["inputTokens"], 1)
        self.assertEqual(mixed["missing_invocations"]["outputTokens"], 2)
        self.assertEqual(mixed["usage_status"], "partial")
        self.assertEqual(mixed["model_identity"], "unknown")
        self.assertEqual(mixed["cost_status"], "unavailable")
        self.assertEqual(grouped["run-b"]["usage"], zeros)
        self.assertEqual(grouped["run-b"]["usage_status"], "reported")
        self.assertEqual(grouped["run-b"]["missing_invocations"]["inputTokens"], 0)
        self.assertEqual(grouped["run-mix"]["model_identity"], "mixed")

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("usage-test")
        with tracer.start_as_current_span("workflow.live-run") as span:
            apply_run_usage_attributes(span, {"run-a": grouped["run-a"], "run-b": grouped["run-b"]})
        provider.force_flush()
        attributes = exporter.get_finished_spans()[0].attributes
        self.assertEqual(attributes["workflow.usage.run-a.inputTokens"], 15)
        self.assertEqual(attributes["workflow.usage.run-b.inputTokens"], 0)
        self.assertEqual(attributes["workflow.usage.run-a.missing.outputTokens"], 2)
        self.assertEqual(attributes["workflow.usage.run-a.usage_status"], "partial")
        self.assertEqual(attributes["workflow.usage.run-a.cost_status"], "unavailable")
        self.assertNotIn("workflow.usage.inputTokens", attributes)

    def test_span_records_invocation_identity_and_failed_calls_without_usage(self) -> None:
        from agentic_dev_workflow.cursor_adapter import apply_invocation_attributes
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("usage-test")
        observation = CursorObservation(
            role="executor",
            duration_seconds=1.0,
            exit_code=0,
            diagnostic_code=None,
            session_id="12345678-1234-5678-9abc-def012345678",
            request_id="abcdefab-cdef-1234-abcd-1234567890ab",
            model="Auto",
            usage=REPORTED_USAGE,
        )
        failed = CursorObservation(
            role="executor",
            duration_seconds=0.2,
            exit_code=None,
            diagnostic_code="cursor_timeout",
            model="Auto",
            session_id="sess-timeout",
        )
        request_ok = AgentRequest(
            run_id="run-a",
            role=AgentRole.EXECUTOR,
            task="PRIVATE_PROMPT_SENTINEL",
            structured_context={"attempt": 2, "secret_context": "PRIVATE_CONTEXT_SENTINEL"},
            timeout_seconds=15,
            correlation_metadata={"invocation_id": "executor:2", "attempt": "2", "role": "executor"},
        )
        request_failed = AgentRequest(
            run_id="run-a",
            role=AgentRole.EXECUTOR,
            task="PRIVATE_PROMPT_SENTINEL",
            structured_context={"phase": "plan"},
            timeout_seconds=15,
            correlation_metadata={"invocation_id": "executor:3", "attempt": "3"},
        )
        with tracer.start_as_current_span("agent.executor") as span:
            apply_invocation_attributes(span, observation, request_ok)
        with tracer.start_as_current_span("agent.executor") as span:
            apply_invocation_attributes(span, failed, request_failed)
        provider.force_flush()
        spans = exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        success, timeout = spans
        self.assertEqual(success.attributes["agent.usage_reported"], True)
        self.assertEqual(success.attributes["agent.usage_status"], "reported")
        self.assertEqual(success.attributes["agent.usage_scope"], "cli_result_snapshot")
        self.assertEqual(success.attributes["agent.model_identity"], "unknown")
        self.assertEqual(success.attributes["agent.cost_status"], "unavailable")
        self.assertEqual(success.attributes["agent.model"], "Auto")
        self.assertEqual(success.attributes["agent.session_id"], "12345678-1234-5678-9abc-def012345678")
        self.assertEqual(success.attributes["agent.request_id"], "abcdefab-cdef-1234-abcd-1234567890ab")
        self.assertEqual(success.attributes["agent.usage.inputTokens"], 10)
        self.assertEqual(success.attributes["agent.usage.cacheReadTokens"], 40)
        self.assertEqual(success.attributes["agent.invocation_id"], "executor:2")
        self.assertEqual(success.attributes["agent.attempt"], "2")
        self.assertEqual(timeout.attributes["agent.usage_reported"], False)
        self.assertEqual(timeout.attributes["agent.usage_status"], "unavailable")
        self.assertEqual(timeout.attributes["agent.usage_scope"], "cli_result_snapshot")
        self.assertEqual(timeout.attributes["agent.model_identity"], "unknown")
        self.assertEqual(timeout.attributes["agent.cost_status"], "unavailable")
        self.assertEqual(timeout.attributes["agent.model"], "Auto")
        self.assertNotIn("agent.session_id", timeout.attributes)
        self.assertNotIn("agent.request_id", timeout.attributes)
        self.assertNotIn("agent.usage.inputTokens", timeout.attributes)
        rendered = json.dumps({key: value for key, value in success.attributes.items()})
        self.assertNotIn("PRIVATE_PROMPT_SENTINEL", rendered)
        self.assertNotIn("PRIVATE_CONTEXT_SENTINEL", rendered)

    def test_span_attributes_keep_only_allowlisted_model_and_ids(self) -> None:
        from agentic_dev_workflow.cursor_adapter import apply_invocation_attributes
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        session_id = "12345678-1234-5678-9abc-def012345678"
        request_id = "abcdefab-cdef-1234-abcd-1234567890ab"
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("usage-test")
        unsafe = CursorObservation(
            role="planner",
            duration_seconds=0.1,
            exit_code=0,
            diagnostic_code=None,
            model=SECRET_SENTINEL,
            session_id=SECRET_SENTINEL,
            request_id="req-span",
        )
        approved = CursorObservation(
            role="planner",
            duration_seconds=0.1,
            exit_code=0,
            diagnostic_code=None,
            model="auto",
            session_id=session_id,
            request_id=request_id,
        )
        request = AgentRequest(
            run_id="run-a",
            role=AgentRole.PLANNER,
            task=PROMPT_SENTINEL,
            structured_context={"phase": "plan"},
            timeout_seconds=15,
            correlation_metadata={"invocation_id": "planner:1", "role": "planner", "invocation": "1"},
        )
        with tracer.start_as_current_span("agent.planner") as span:
            apply_invocation_attributes(span, unsafe, request)
        with tracer.start_as_current_span("agent.planner") as span:
            apply_invocation_attributes(span, approved, request)
        provider.force_flush()
        rejected, kept = exporter.get_finished_spans()
        rendered = json.dumps(dict(rejected.attributes))
        self.assertNotIn(SECRET_SENTINEL, rendered)
        self.assertNotIn("req-span", rendered)
        self.assertNotIn("agent.model", rejected.attributes)
        self.assertNotIn("agent.session_id", rejected.attributes)
        self.assertNotIn("agent.request_id", rejected.attributes)
        self.assertEqual(rejected.attributes["agent.model_identity"], "unknown")
        self.assertEqual(rejected.attributes["agent.cost_status"], "unavailable")
        self.assertEqual(kept.attributes["agent.model"], "auto")
        self.assertEqual(kept.attributes["agent.session_id"], session_id)
        self.assertEqual(kept.attributes["agent.request_id"], request_id)

    def test_run_usage_attributes_are_not_summed_across_runs(self) -> None:
        from agentic_dev_workflow.cursor_adapter import (
            aggregate_usage_by_run,
            apply_run_usage_attributes,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("usage-test")
        grouped = aggregate_usage_by_run(
            [
                ("run-a", {"inputTokens": 10, "outputTokens": 1, "cacheReadTokens": 0, "cacheWriteTokens": 0}),
                ("run-b", {"inputTokens": 7, "outputTokens": 2, "cacheReadTokens": 1, "cacheWriteTokens": 0}),
            ]
        )
        with tracer.start_as_current_span("workflow.live-run") as span:
            apply_run_usage_attributes(span, grouped)
        provider.force_flush()
        attributes = exporter.get_finished_spans()[0].attributes
        self.assertEqual(attributes["workflow.usage.run-a.inputTokens"], 10)
        self.assertEqual(attributes["workflow.usage.run-b.inputTokens"], 7)
        self.assertEqual(attributes["workflow.usage.run-a.usage_status"], "reported")
        self.assertEqual(attributes["workflow.usage.run-b.usage_status"], "reported")
        self.assertEqual(attributes["workflow.usage.run-a.cost_status"], "unavailable")
        self.assertEqual(attributes["workflow.usage.run-a.missing.inputTokens"], 0)
        self.assertNotIn("workflow.usage.inputTokens", attributes)
        self.assertNotIn("workflow.usage.usage_status", attributes)

    def test_parent_span_keeps_two_runs_without_a_single_run_identity(self) -> None:
        from agentic_dev_workflow.cursor_adapter import (
            aggregate_usage_by_run,
            apply_invocation_attributes,
            apply_parent_trace_identity,
            apply_run_usage_attributes,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("usage-test")
        grouped = aggregate_usage_by_run(
            [
                ("run-a", {"inputTokens": 10, "outputTokens": 1, "cacheReadTokens": 0, "cacheWriteTokens": 0}),
                ("run-b", {"inputTokens": 4, "outputTokens": 2, "cacheReadTokens": 0, "cacheWriteTokens": 0}),
            ]
        )
        with tracer.start_as_current_span("workflow.live-run") as span:
            span.set_attribute("workflow.run_id", "run-a")
            apply_run_usage_attributes(span, grouped)
            apply_parent_trace_identity(span, ["run-a", "run-b"])
        for run_id in ("run-a", "run-b"):
            request = AgentRequest(
                run_id=run_id,
                role=AgentRole.PLANNER,
                task="Implement the feature in SPEC.md",
                structured_context={"phase": "plan"},
                timeout_seconds=15,
                correlation_metadata={"invocation_id": "planner:1", "role": "planner", "invocation": "1"},
            )
            observation = CursorObservation(
                role="planner",
                duration_seconds=0.2,
                exit_code=0,
                diagnostic_code=None,
                model="Auto",
            )
            with tracer.start_as_current_span("agent.planner") as span:
                span.set_attribute("workflow.run_id", run_id)
                apply_invocation_attributes(span, observation, request)
        provider.force_flush()
        parent, first, second = exporter.get_finished_spans()
        self.assertNotIn("workflow.run_id", parent.attributes)
        self.assertEqual(parent.attributes["workflow.trace_scope"], "multi_run")
        self.assertEqual(parent.attributes["workflow.trace_run_count"], 2)
        self.assertEqual(parent.attributes["workflow.usage.run-a.inputTokens"], 10)
        self.assertEqual(parent.attributes["workflow.usage.run-b.inputTokens"], 4)
        self.assertNotIn("workflow.usage.inputTokens", parent.attributes)
        self.assertEqual(first.attributes["agent.invocation_id"], "planner:1")
        self.assertEqual(second.attributes["agent.invocation_id"], "planner:1")
        self.assertEqual(first.attributes["workflow.run_id"], "run-a")
        self.assertEqual(second.attributes["workflow.run_id"], "run-b")

    def test_live_recorder_keeps_failed_spans_and_leaves_the_prompt_off_the_span(self) -> None:
        import importlib.util

        from agentic_dev_workflow.adapters import AgentFailureCategory, AgentResult
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        script = Path(__file__).resolve().parents[1] / "scripts" / "live_cursor_feature.py"
        spec = importlib.util.spec_from_file_location("live_cursor_feature_under_test", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("live-recorder-test")
        records: list[dict] = []

        class Inner:
            adapter_type = "cursor-cli"

            def __init__(self, result: AgentResult, observation: CursorObservation | None) -> None:
                self._result = result
                self.last_observation = observation

            def invoke(self, request: AgentRequest) -> AgentResult:
                return self._result

        failed = AgentResult(
            run_id="run-a",
            role=AgentRole.PLANNER,
            status=AgentStatus.FAILED,
            failure_category=AgentFailureCategory.TIMEOUT,
            diagnostic_code="cursor_timeout",
            duration_seconds=0.4,
        )
        observation = CursorObservation(
            role="planner",
            duration_seconds=0.4,
            exit_code=None,
            diagnostic_code="cursor_timeout",
            model="Auto",
            session_id="sess-timeout",
        )
        request = AgentRequest(
            run_id="run-a",
            role=AgentRole.PLANNER,
            task="PRIVATE_PROMPT_SENTINEL",
            structured_context={"phase": "plan"},
            timeout_seconds=15,
            correlation_metadata={"invocation_id": "planner:1", "role": "planner", "invocation": "1"},
        )
        module.RecordingAdapter(Inner(failed, observation), tracer, records).invoke(request)
        provider.force_flush()
        span = exporter.get_finished_spans()[0]
        self.assertEqual(span.name, "agent.planner")
        self.assertEqual(span.attributes["agent.usage_reported"], False)
        self.assertEqual(span.attributes["agent.usage_status"], "unavailable")
        self.assertEqual(span.attributes["agent.cost_status"], "unavailable")
        self.assertEqual(span.attributes["agent.model_identity"], "unknown")
        self.assertEqual(span.attributes["agent.model"], "Auto")
        self.assertEqual(span.attributes["agent.diagnostic_code"], "cursor_timeout")
        self.assertNotIn("agent.session_id", span.attributes)
        self.assertEqual(span.attributes["agent.invocation_id"], "planner:1")
        self.assertNotIn("agent.request_id", span.attributes)
        self.assertNotIn("agent.usage.inputTokens", span.attributes)
        rendered = json.dumps(dict(span.attributes))
        self.assertNotIn("PRIVATE_PROMPT_SENTINEL", rendered)
        self.assertEqual(records[0]["run_id"], "run-a")
        self.assertIsNone(records[0]["usage"])
        self.assertEqual(records[0]["usage_status"], "unavailable")
        self.assertEqual(records[0]["usage_scope"], "cli_result_snapshot")
        self.assertEqual(records[0]["model_identity"], "unknown")
        self.assertEqual(records[0]["cost_status"], "unavailable")

    def test_span_diagnostic_code_must_be_a_trusted_workflow_code(self) -> None:
        import importlib.util

        from agentic_dev_workflow.adapters import AgentFailureCategory, AgentResult
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        script = Path(__file__).resolve().parents[1] / "scripts" / "live_cursor_feature.py"
        spec = importlib.util.spec_from_file_location("live_cursor_feature_span_codes", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("span-code-test")

        class Inner:
            adapter_type = "cursor-cli"

            def __init__(self, result: AgentResult) -> None:
                self._result = result
                self.last_observation = CursorObservation(
                    role="executor",
                    duration_seconds=0.2,
                    exit_code=1,
                    diagnostic_code=None,
                    model="Auto",
                )

            def invoke(self, request: AgentRequest) -> AgentResult:
                return self._result

        def invoke(code: str) -> object:
            result = AgentResult(
                run_id="run-a",
                role=AgentRole.EXECUTOR,
                status=AgentStatus.FAILED,
                failure_category=AgentFailureCategory.EXECUTION_FAILURE,
                diagnostic_code=code,
                duration_seconds=0.2,
            )
            request = AgentRequest(
                run_id="run-a",
                role=AgentRole.EXECUTOR,
                task="Apply the approved file scope",
                structured_context={"approved_paths": ["notes.txt"]},
                timeout_seconds=15,
                correlation_metadata={"invocation_id": "executor:1", "attempt": "1", "role": "executor"},
            )
            module.RecordingAdapter(Inner(result), tracer, []).invoke(request)
            provider.force_flush()
            return exporter.get_finished_spans()[-1]

        rejected = invoke("model_supplied_reason")
        trusted = invoke("cursor_timeout")
        self.assertNotIn("agent.diagnostic_code", rejected.attributes)
        self.assertNotIn("model_supplied_reason", json.dumps(dict(rejected.attributes)))
        self.assertEqual(trusted.attributes["agent.diagnostic_code"], "cursor_timeout")

    def test_tool_and_prompt_text_do_not_become_skill_identifiers(self) -> None:
        import io
        from contextlib import redirect_stdout

        from agentic_dev_workflow.cursor_adapter import aggregate_skill_by_run, apply_parent_trace_identity, apply_run_skill_attributes
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        skill_path = ".cursor/skills/some-skill/SKILL.md"
        events = [
            {
                "type": "user",
                "message": {"role": "user", "content": [{"type": "text", "text": f"{PROMPT_SENTINEL} /some-skill"}]},
            },
            {
                "type": "tool_call",
                "subtype": "completed",
                "tool_call": {"readToolCall": {"args": {"path": skill_path}}},
            },
            {
                "type": "tool_call",
                "subtype": "completed",
                "tool_call": {"shellToolCall": {"args": {"command": "echo /some-skill"}}},
            },
            {
                "type": "tool_call",
                "subtype": "completed",
                "tool_call": {"skill": {"args": {"path": skill_path, "name": "some-skill"}}},
            },
            result_event(REPORTED_USAGE),
        ]
        module = _load_live_module()
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("skill-status-test")
        records: list[dict] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = CursorCLIAdapter(
                workspace=root,
                executable=str(write_executable(root, stream_script(events))),
            )
            recorder = module.RecordingAdapter(adapter, tracer, records)
            with tracer.start_as_current_span("workflow.live-run") as span:
                span.set_attribute("workflow.run_id", "run-a")
                for run_id in ("run-a", "run-b"):
                    recorder.invoke(
                        AgentRequest(
                            run_id=run_id,
                            role=AgentRole.PLANNER,
                            task=f"{PROMPT_SENTINEL} /some-skill",
                            structured_context={"phase": "plan", "note": skill_path},
                            timeout_seconds=15,
                            correlation_metadata={
                                "invocation_id": "planner:1",
                                "role": "planner",
                                "invocation": "1",
                            },
                        )
                    )
                skill_by_run = aggregate_skill_by_run(
                    [(record["run_id"], record["skill_status"]) for record in records]
                )
                apply_run_skill_attributes(span, skill_by_run)
                apply_parent_trace_identity(span, ["run-a", "run-b"])
            payload = {
                "agent_calls": records,
                "skill_by_run": {
                    "run-a": {
                        "skill_status": "reported",
                        "skill_count": 0,
                        "skill_ids": ["some-skill"],
                        "invocations_without_skill_observation": 99,
                    }
                },
            }
            with tempfile.TemporaryDirectory() as report_dir:
                destination = Path(report_dir) / "live-run-report.json"
                console = io.StringIO()
                with redirect_stdout(console):
                    module.publish_live_report(destination, payload)
                rendered = destination.read_text(encoding="utf-8")
                printed = console.getvalue()
        provider.force_flush()
        spans = exporter.get_finished_spans()
        parent = next(span for span in spans if span.name == "workflow.live-run")
        first, second = [span for span in spans if span.name == "agent.planner"]
        self.assertEqual(len(records), 2)
        self.assertIsNotNone(adapter.last_observation)
        self.assertEqual([record["run_id"] for record in records], ["run-a", "run-b"])
        self.assertEqual([record["invocation_id"] for record in records], ["planner:1", "planner:1"])
        for record in records:
            self.assertEqual(record["skill_status"], "unavailable")
            self.assertNotIn("skill_ids", record)
            self.assertNotIn("skill_count", record)
            self.assertGreater(record["tool_event_count"], 0)
        stored = json.loads(rendered)
        self.assertEqual(stored["report_schema_version"], 6)
        self.assertEqual(stored["skill_by_run"]["run-a"]["skill_status"], "unavailable")
        self.assertEqual(stored["skill_by_run"]["run-a"]["invocations_without_skill_observation"], 1)
        self.assertEqual(stored["skill_by_run"]["run-b"]["skill_status"], "unavailable")
        self.assertEqual(stored["skill_by_run"]["run-b"]["invocations_without_skill_observation"], 1)
        self.assertNotIn("workflow.run_id", parent.attributes)
        self.assertEqual(parent.attributes["workflow.trace_scope"], "multi_run")
        self.assertEqual(parent.attributes["workflow.skill.run-a.skill_status"], "unavailable")
        self.assertEqual(parent.attributes["workflow.skill.run-a.invocations_without_skill_observation"], 1)
        self.assertEqual(parent.attributes["workflow.skill.run-b.skill_status"], "unavailable")
        self.assertEqual(parent.attributes["workflow.skill.run-b.invocations_without_skill_observation"], 1)
        self.assertEqual(first.attributes["agent.skill_status"], "unavailable")
        self.assertEqual(second.attributes["agent.skill_status"], "unavailable")
        self.assertEqual(first.attributes["workflow.run_id"], "run-a")
        self.assertEqual(second.attributes["workflow.run_id"], "run-b")
        forbidden = (
            PROMPT_SENTINEL,
            "/some-skill",
            "some-skill",
            "readToolCall",
            "shellToolCall",
            "SKILL.md",
            "skill_count",
            "skill_ids",
        )
        bodies = [
            rendered,
            printed,
            json.dumps(dict(parent.attributes)),
            json.dumps(dict(first.attributes)),
            json.dumps(dict(second.attributes)),
        ]
        for body in bodies:
            for secret in forbidden:
                self.assertNotIn(secret, body)

    def test_all_unavailable_skill_observation_is_not_partial(self) -> None:
        from agentic_dev_workflow.cursor_adapter import aggregate_skill_by_run

        grouped = aggregate_skill_by_run(
            [
                ("run-a", "unavailable"),
                ("run-a", "unavailable"),
                ("run-b", "reported"),
                ("run-b", "unavailable"),
            ]
        )
        self.assertEqual(grouped["run-a"]["skill_status"], "unavailable")
        self.assertEqual(grouped["run-a"]["invocations_without_skill_observation"], 2)
        self.assertEqual(grouped["run-b"]["skill_status"], "partial")
        self.assertEqual(grouped["run-b"]["invocations_without_skill_observation"], 1)
        self.assertNotIn("skill_count", json.dumps(grouped))
        self.assertNotIn("skill_ids", json.dumps(grouped))


PROMPT_SENTINEL = "PRIVATE_PROMPT_SENTINEL"
SECRET_SENTINEL = "SECRET_CURSOR_API_KEY_sk-live-fixture"
STATUS_SENTINEL = "STATUS_ACCOUNT_SENTINEL"
VERIFY_STDOUT_SENTINEL = "VERIFY_STDOUT_SENTINEL"
VERIFY_STDERR_SENTINEL = "VERIFY_STDERR_SENTINEL"


def _load_live_module():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "live_cursor_feature.py"
    spec = importlib.util.spec_from_file_location("live_cursor_feature_report_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LiveReportRedactionTests(unittest.TestCase):
    def test_persisted_and_printed_report_omit_prompts_secrets_and_raw_output(self) -> None:
        import io
        from contextlib import redirect_stdout
        from types import SimpleNamespace

        from agentic_dev_workflow.adapters import AgentFailureCategory, AgentResult

        module = _load_live_module()
        observation = CursorObservation(
            role="planner",
            duration_seconds=1.25,
            exit_code=0,
            diagnostic_code=None,
            session_id="11111111-1111-1111-1111-111111111111",
            request_id="22222222-2222-2222-2222-222222222222",
            model="Auto",
            cli_duration_ms=1200,
            usage=REPORTED_USAGE,
            tool_events=({"subtype": "completed", "tool": "readToolCall", "path": SECRET_SENTINEL},),
            stdout_excerpt=PROMPT_SENTINEL + "\n" + SECRET_SENTINEL,
            stderr_excerpt=SECRET_SENTINEL,
        )
        result = AgentResult(
            run_id="run-a",
            role=AgentRole.PLANNER,
            status=AgentStatus.SUCCEEDED,
            structured_output={
                "summary": PROMPT_SENTINEL,
                "assumptions": [SECRET_SENTINEL],
                "steps": [{"id": "step-1", "description": SECRET_SENTINEL, "expected_files": []}],
                "verification_check_ids": ["runner.tests", SECRET_SENTINEL],
            },
            exit_code=0,
            duration_seconds=1.25,
        )
        request = AgentRequest(
            run_id="run-a",
            role=AgentRole.PLANNER,
            task=PROMPT_SENTINEL,
            structured_context={"phase": "plan", "note": SECRET_SENTINEL},
            timeout_seconds=15,
            correlation_metadata={"invocation_id": "planner:1", "role": "planner", "invocation": "1"},
        )
        records: list[dict] = []
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
        tracer = provider.get_tracer("report-redaction")
        records.clear()
        inner = SimpleNamespace(adapter_type="cursor-cli", last_observation=observation)

        def invoke(agent_request: AgentRequest) -> AgentResult:
            return result

        inner.invoke = invoke
        module.RecordingAdapter(inner, tracer, records).invoke(request)
        command = SimpleNamespace(
            check_id="runner.tests",
            attempt=1,
            registry_command=("python", "-m", "unittest"),
            executed_command=("python", "-m", "unittest", SECRET_SENTINEL),
            exit_code=1,
            duration_seconds=0.4,
            stdout=f"{VERIFY_STDOUT_SENTINEL}\n{PROMPT_SENTINEL}".encode(),
            stderr=VERIFY_STDERR_SENTINEL.encode(),
        )

        class _Checkpoint:
            config = {"configurable": {"checkpoint_id": "cp-1"}}
            metadata = {"step": 2, "source": "loop"}
            checkpoint = {
                "channel_values": {
                    "final_status": "failed",
                    "proposed_paths": ["transaction_actions.py"],
                    "approved_paths": [],
                    "execution_attempt": 1,
                    "executor_invocations": 1,
                    "repair_retries_used": 0,
                    "review_decision": "reject",
                    "errors": ["cursor_timeout", SECRET_SENTINEL],
                    "plan": {"summary": PROMPT_SENTINEL},
                    "verification": {"outcome": "fail"},
                    "checks": [
                        {
                            "check_id": "runner.tests",
                            "status": "failed",
                            "exit_code": 1,
                            "duration_ms": 400,
                            "stdout": VERIFY_STDOUT_SENTINEL,
                            "stderr": VERIFY_STDERR_SENTINEL,
                        }
                    ],
                }
            }

        class _Saver:
            def list(self, _config):
                return [_Checkpoint()]

        payload = {
            "cursor_agent": "/tmp/cursor-agent",
            "cursor_agent_version": "2026.10.22-27dcf5f",
            "cursor_agent_status": STATUS_SENTINEL,
            "feature_repository": "/tmp/agentic-feature-test-redaction/feature",
            "base_commit": "abc123",
            "worktree": "/tmp/agentic-feature-test-redaction/worktrees/run-a",
            "agent_timeout_seconds": 300,
            "trace_flushed": True,
            "otlp_endpoint": "http://127.0.0.1:4318/v1/traces",
            "planning_report": {
                "run_id": "run-a",
                "outcome": "failed",
                "attempts": 1,
                "executor_invocations": 0,
                "repair_retries_used": 0,
                "max_retries": 1,
                "checks": [
                    {
                        "check_id": "runner.tests",
                        "status": "failed",
                        "exit_code": 1,
                        "duration_ms": 400,
                        "summary": VERIFY_STDOUT_SENTINEL,
                    }
                ],
                "review_decision": None,
                "evidence_refs": [],
                "errors": ["cursor_timeout"],
                "proposed_paths": [],
                "attempt_history": [],
            },
            "agent_calls": records,
            "checkpoints": module._checkpoints(_Saver(), "run-a"),
            "check_commands": [module._command_view(command)],
            "execution": None,
            "jaeger": {"reachable": False, "error": "URLError"},
            "token_usage": "exposed",
            "usage_by_run": {"run-a": {"inputTokens": 10, "reported_invocations": 1, "unreported_invocations": 0}},
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "live-run-report.json"
            console = io.StringIO()
            with redirect_stdout(console):
                module.publish_live_report(destination, payload)
            persisted = destination.read_text(encoding="utf-8")
            printed = console.getvalue()
        for rendered in (persisted, printed):
            self.assertNotIn(PROMPT_SENTINEL, rendered)
            self.assertNotIn(SECRET_SENTINEL, rendered)
            self.assertNotIn(STATUS_SENTINEL, rendered)
            self.assertNotIn(VERIFY_STDOUT_SENTINEL, rendered)
            self.assertNotIn(VERIFY_STDERR_SENTINEL, rendered)
            self.assertNotIn("stdout_excerpt", rendered)
            self.assertNotIn("stderr_excerpt", rendered)
            self.assertNotIn("cursor_agent_status", rendered)
        stored = json.loads(persisted)
        self.assertEqual(stored["agent_calls"][0]["invocation_id"], "planner:1")
        self.assertEqual(stored["agent_calls"][0]["usage"], REPORTED_USAGE)
        self.assertEqual(stored["agent_calls"][0]["model"], "Auto")
        self.assertEqual(stored["agent_calls"][0]["output_summary"]["step_count"], 1)
        self.assertEqual(stored["agent_calls"][0]["output_summary"]["verification_check_ids"], ["runner.tests"])
        self.assertEqual(stored["check_commands"][0]["exit_code"], 1)
        self.assertEqual(stored["check_commands"][0]["check_id"], "runner.tests")
        self.assertNotIn("stdout", stored["check_commands"][0])
        self.assertNotIn("stderr", stored["check_commands"][0])
        self.assertNotIn("plan_summary", stored["checkpoints"][0])
        self.assertEqual(stored["checkpoints"][0]["checks"][0]["status"], "failed")
        self.assertEqual(stored["cursor_agent_version"], "2026.10.22-27dcf5f")

    def test_token_shaped_metadata_is_omitted_from_persisted_and_printed_json(self) -> None:
        import io
        from contextlib import redirect_stdout

        from agentic_dev_workflow.adapters import AgentResult

        module = _load_live_module()
        observation = CursorObservation(
            role="planner",
            duration_seconds=1.0,
            exit_code=0,
            diagnostic_code=SECRET_SENTINEL,
            model=SECRET_SENTINEL,
            tool_events=(
                {"subtype": "completed", "tool": SECRET_SENTINEL, "path": "SPEC.md"},
                {"subtype": "completed", "tool": "readToolCall", "path": PROMPT_SENTINEL},
            ),
            usage=REPORTED_USAGE,
        )
        result = AgentResult(
            run_id="run-a",
            role=AgentRole.PLANNER,
            status=AgentStatus.SUCCEEDED,
            structured_output={
                "summary": SECRET_SENTINEL,
                "assumptions": [],
                "steps": [],
                "verification_check_ids": ["runner.tests"],
            },
            exit_code=0,
            duration_seconds=1.0,
            diagnostic_code=SECRET_SENTINEL,
        )
        request = AgentRequest(
            run_id="run-a",
            role=AgentRole.PLANNER,
            task="Implement the feature in SPEC.md",
            structured_context={"phase": "plan"},
            timeout_seconds=15,
            correlation_metadata={"invocation_id": "planner:1", "invocation": "1"},
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
        records: list[dict] = []

        class Inner:
            adapter_type = "cursor-cli"
            last_observation = observation

            def invoke(self, _request: AgentRequest) -> AgentResult:
                return result

        module.RecordingAdapter(Inner(), provider.get_tracer("report-metadata"), records).invoke(request)
        payload = {
            "cursor_agent_version": SECRET_SENTINEL,
            "agent_calls": records,
            "planning_report": {
                "run_id": "run-a",
                "outcome": "failed",
                "errors": ["cursor_timeout", SECRET_SENTINEL, "sk-live-fixture"],
                "proposed_paths": [
                    "transaction_actions.py",
                    SECRET_SENTINEL,
                    PROMPT_SENTINEL,
                ],
                "attempt_history": [],
                "checks": [],
                "evidence_refs": [],
            },
            "checkpoints": [
                {
                    "checkpoint_id": "cp-1",
                    "source": "loop",
                    "final_status": "failed",
                    "proposed_paths": [PROMPT_SENTINEL, "src/cursor_adapter.py"],
                    "approved_paths": [SECRET_SENTINEL, "tests/test_cursor_adapter.py"],
                    "errors": [PROMPT_SENTINEL, "cursor_timeout"],
                    "verification_outcome": "fail",
                    "checks": [],
                }
            ],
            "execution": {
                "run_id": "run-ab",
                "approval": [SECRET_SENTINEL, "SPEC.md"],
                "error": PROMPT_SENTINEL,
                "report": None,
                "checkpoints": [],
            },
            "check_commands": [],
            "jaeger": {"reachable": False, "error": "URLError"},
            "token_usage": "exposed",
            "usage_by_run": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "live-run-report.json"
            console = io.StringIO()
            with redirect_stdout(console):
                module.publish_live_report(destination, payload)
            rendered = destination.read_text(encoding="utf-8")
            printed = console.getvalue()
        for body in (rendered, printed):
            self.assertNotIn(SECRET_SENTINEL, body)
            self.assertNotIn(PROMPT_SENTINEL, body)
            self.assertNotIn("sk-live-fixture", body)
        stored = json.loads(rendered)
        call = stored["agent_calls"][0]
        self.assertIsNone(call["model"])
        self.assertIsNone(stored["cursor_agent_version"])
        self.assertIsNone(call["diagnostic_code"])
        self.assertNotIn("tool_events", call)
        self.assertNotIn("readToolCall", json.dumps(call))
        self.assertEqual(call["tool_event_count"], 2)
        self.assertEqual(
            stored["planning_report"]["proposed_paths"],
            ["transaction_actions.py"],
        )
        self.assertEqual(stored["planning_report"]["errors"], ["cursor_timeout"])
        self.assertEqual(
            stored["checkpoints"][0]["proposed_paths"],
            ["src/cursor_adapter.py"],
        )
        self.assertEqual(
            stored["checkpoints"][0]["approved_paths"],
            ["tests/test_cursor_adapter.py"],
        )
        self.assertEqual(stored["checkpoints"][0]["errors"], ["cursor_timeout"])
        self.assertEqual(stored["execution"]["approval"], ["SPEC.md"])
        self.assertIsNone(stored["execution"]["error"])

    def test_schema_v4_reports_completeness_without_price_or_model_text(self) -> None:
        import io
        from contextlib import redirect_stdout

        from agentic_dev_workflow.adapters import AgentResult

        module = _load_live_module()
        full = {
            "inputTokens": 0,
            "outputTokens": 0,
            "cacheReadTokens": 0,
            "cacheWriteTokens": 0,
        }
        calls = [
            module.agent_call_record(
                AgentRequest(
                    run_id="run-a",
                    role=AgentRole.PLANNER,
                    task=PROMPT_SENTINEL,
                    structured_context={"phase": "plan"},
                    timeout_seconds=15,
                    correlation_metadata={"invocation_id": "planner:1", "role": "planner", "invocation": "1"},
                ),
                AgentResult(
                    run_id="run-a",
                    role=AgentRole.PLANNER,
                    status=AgentStatus.SUCCEEDED,
                    duration_seconds=1,
                ),
                CursorObservation(
                    role="planner",
                    duration_seconds=1,
                    exit_code=0,
                    diagnostic_code=None,
                    model="gpt-5",
                    usage=full,
                ),
                elapsed=1,
                replayed=False,
                cli_duration_ms=1,
                usage=full,
            ),
            module.agent_call_record(
                AgentRequest(
                    run_id="run-b",
                    role=AgentRole.PLANNER,
                    task=PROMPT_SENTINEL,
                    structured_context={"phase": "plan"},
                    timeout_seconds=15,
                    correlation_metadata={"invocation_id": "planner:1", "role": "planner", "invocation": "1"},
                ),
                AgentResult(
                    run_id="run-b",
                    role=AgentRole.PLANNER,
                    status=AgentStatus.SUCCEEDED,
                    duration_seconds=1,
                ),
                CursorObservation(
                    role="planner",
                    duration_seconds=1,
                    exit_code=0,
                    diagnostic_code=None,
                    model="Auto",
                    usage={"inputTokens": 4},
                ),
                elapsed=1,
                replayed=False,
                cli_duration_ms=1,
                usage={"inputTokens": 4},
            ),
        ]
        payload = {
            "cursor_agent": "/tmp/cursor-agent",
            "cursor_agent_version": "2026.10.22-27dcf5f",
            "feature_repository": "/tmp/agentic-feature-test-redaction/feature",
            "base_commit": "abc123",
            "worktree": "/tmp/agentic-feature-test-redaction/worktrees/run-a",
            "agent_timeout_seconds": 300,
            "trace_flushed": True,
            "otlp_endpoint": "http://127.0.0.1:4318/v1/traces",
            "planning_report": {
                "run_id": "run-a",
                "outcome": "error",
                "attempts": 0,
                "executor_invocations": 1,
                "repair_retries_used": 0,
                "max_retries": 1,
                "checks": [],
                "review_decision": None,
                "evidence_refs": [],
                "errors": ["executor_failed"],
                "proposed_paths": [],
                "attempt_history": [],
            },
            "agent_calls": calls,
            "checkpoints": [],
            "check_commands": [],
            "execution": None,
            "jaeger": {"reachable": False, "error": "URLError"},
            "token_usage": "exposed",
            "usage_by_run": {
                "run-a": {
                    "reported_invocations": 1,
                    "partial_invocations": 0,
                    "unavailable_invocations": 0,
                    "usage_status": "reported",
                    "usage": full,
                    "missing_invocations": {key: 0 for key in full},
                    "model_identity": "gpt-5",
                    "cost_status": "calculated",
                    "usd": 12.5,
                },
                "run-b": {
                    "reported_invocations": 0,
                    "partial_invocations": 1,
                    "unavailable_invocations": 0,
                    "usage_status": "partial",
                    "usage": {"inputTokens": 4},
                    "missing_invocations": {
                        "inputTokens": 0,
                        "outputTokens": 1,
                        "cacheReadTokens": 1,
                        "cacheWriteTokens": 1,
                    },
                    "model_identity": "unknown",
                    "cost_status": "unavailable",
                },
            },
            "usd": 12.5,
            "cost_usd": SECRET_SENTINEL,
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "live-run-report.json"
            console = io.StringIO()
            with redirect_stdout(console):
                module.publish_live_report(destination, payload)
            rendered = destination.read_text(encoding="utf-8")
            printed = console.getvalue()
        for body in (rendered, printed):
            self.assertNotIn(SECRET_SENTINEL, body)
            self.assertNotIn(PROMPT_SENTINEL, body)
            self.assertNotIn("gpt-5", body)
            self.assertNotIn("usd", body)
            self.assertNotIn("calculated", body)
        stored = json.loads(rendered)
        self.assertEqual(stored["report_schema_version"], 6)
        identities = [(call["run_id"], call["invocation_id"]) for call in stored["agent_calls"]]
        self.assertEqual(identities, [("run-a", "planner:1"), ("run-b", "planner:1")])
        self.assertEqual(len(set(identities)), 2)
        first, second = stored["agent_calls"]
        self.assertEqual(first["skill_status"], "unavailable")
        self.assertEqual(second["skill_status"], "unavailable")
        self.assertEqual(stored["skill_by_run"]["run-a"]["skill_status"], "unavailable")
        self.assertEqual(stored["skill_by_run"]["run-a"]["invocations_without_skill_observation"], 1)
        self.assertEqual(stored["skill_by_run"]["run-b"]["skill_status"], "unavailable")
        self.assertEqual(stored["skill_by_run"]["run-b"]["invocations_without_skill_observation"], 1)
        self.assertNotIn("skill_count", json.dumps(stored))
        self.assertEqual(first["usage"], full)
        self.assertEqual(first["usage_status"], "reported")
        self.assertEqual(first["usage_scope"], "cli_result_snapshot")
        self.assertEqual(first["model_identity"], "unknown")
        self.assertIsNone(first["model"])
        self.assertEqual(first["cost_status"], "unavailable")
        self.assertEqual(second["usage"], {"inputTokens": 4})
        self.assertEqual(second["usage_status"], "partial")
        self.assertEqual(second["model"], "Auto")
        self.assertEqual(second["model_identity"], "unknown")
        self.assertEqual(stored["usage_by_run"]["run-a"]["usage"], full)
        self.assertEqual(stored["usage_by_run"]["run-a"]["missing_invocations"]["inputTokens"], 0)
        self.assertEqual(stored["usage_by_run"]["run-a"]["cost_status"], "unavailable")
        self.assertEqual(stored["usage_by_run"]["run-a"]["model_identity"], "unknown")
        self.assertEqual(stored["usage_by_run"]["run-b"]["usage_status"], "partial")
        self.assertEqual(stored["usage_by_run"]["run-b"]["missing_invocations"]["outputTokens"], 1)
        self.assertEqual(stored["planning_report"]["errors"], ["executor_failed"])
        self.assertNotIn("usd", stored)
        self.assertNotIn("cost_usd", stored)

    def test_published_usage_totals_follow_the_agent_calls(self) -> None:
        module = _load_live_module()
        stored = module.redact_live_report(
            {
                "agent_calls": [
                    {
                        "run_id": "run-a",
                        "role": "planner",
                        "invocation_id": "planner:1",
                        "usage": {"inputTokens": 4},
                        "model": "gpt-5",
                    },
                    {
                        "run_id": "run-a",
                        "role": "executor",
                        "invocation_id": "executor:1",
                        "attempt": "1",
                        "usage": None,
                    },
                ],
                "usage_by_run": {
                    "run-a": {
                        "reported_invocations": 9,
                        "partial_invocations": 0,
                        "unavailable_invocations": 0,
                        "usage_status": "reported",
                        "usage": {
                            "inputTokens": 99,
                            "outputTokens": 99,
                            "cacheReadTokens": 99,
                            "cacheWriteTokens": 99,
                        },
                        "missing_invocations": {
                            "inputTokens": 0,
                            "outputTokens": 0,
                            "cacheReadTokens": 0,
                            "cacheWriteTokens": 0,
                        },
                        "model_identity": "known",
                        "cost_status": "calculated",
                        "usd": 12.5,
                    }
                },
            }
        )
        bucket = stored["usage_by_run"]["run-a"]
        self.assertEqual(bucket["usage"], {"inputTokens": 4})
        self.assertEqual(bucket["usage_status"], "partial")
        self.assertEqual(bucket["reported_invocations"], 0)
        self.assertEqual(bucket["partial_invocations"], 1)
        self.assertEqual(bucket["unavailable_invocations"], 1)
        self.assertEqual(bucket["missing_invocations"]["inputTokens"], 1)
        self.assertEqual(bucket["missing_invocations"]["outputTokens"], 2)
        self.assertEqual(bucket["model_identity"], "unknown")
        self.assertEqual(bucket["cost_status"], "unavailable")
        self.assertNotIn("usd", json.dumps(stored["usage_by_run"]))

    def test_published_skill_summary_follows_the_agent_calls(self) -> None:
        module = _load_live_module()
        stored = module.redact_live_report(
            {
                "agent_calls": [
                    {
                        "run_id": "run-a",
                        "role": "planner",
                        "invocation_id": "planner:1",
                        "diagnostic_code": "cursor_timeout",
                        "skill_status": "reported",
                        "skill_ids": ["some-skill"],
                        "skill_count": 0,
                        "tool_events": [
                            {"subtype": "completed", "tool": "skill", "path": ".cursor/skills/some-skill/SKILL.md"}
                        ],
                    },
                    {
                        "run_id": "run-a",
                        "role": "executor",
                        "invocation_id": "executor:1",
                        "attempt": "1",
                        "diagnostic_code": "model_supplied_reason",
                        "skill_status": "reported",
                        "skill_count": 0,
                    },
                    {
                        "run_id": "run-b",
                        "role": "planner",
                        "invocation_id": "planner:1",
                        "skill_status": "partial",
                    },
                ],
                "skill_by_run": {
                    "trace": {
                        "skill_status": "reported",
                        "skill_count": 0,
                        "skill_ids": ["some-skill"],
                        "invocations_without_skill_observation": 0,
                    },
                    "run-a": {
                        "skill_status": "partial",
                        "skill_count": 0,
                        "skill_ids": ["cursor_timeout", "model_supplied_reason"],
                        "invocations_without_skill_observation": 99,
                    },
                },
            }
        )
        body = json.dumps(stored)
        self.assertNotIn("some-skill", body)
        self.assertNotIn("model_supplied_reason", body)
        self.assertNotIn("SKILL.md", body)
        self.assertNotIn("skill_count", body)
        self.assertNotIn("skill_ids", body)
        self.assertNotIn('"tool": "skill"', body)
        self.assertEqual(stored["agent_calls"][0]["diagnostic_code"], "cursor_timeout")
        self.assertEqual(stored["agent_calls"][0]["skill_status"], "unavailable")
        self.assertIsNone(stored["agent_calls"][1]["diagnostic_code"])
        self.assertEqual(stored["agent_calls"][1]["skill_status"], "unavailable")
        self.assertEqual(stored["agent_calls"][2]["skill_status"], "unavailable")
        self.assertEqual(
            [(call["run_id"], call["invocation_id"]) for call in stored["agent_calls"]],
            [("run-a", "planner:1"), ("run-a", "executor:1"), ("run-b", "planner:1")],
        )
        self.assertEqual(set(stored["skill_by_run"]), {"run-a", "run-b"})
        self.assertEqual(stored["skill_by_run"]["run-a"]["skill_status"], "unavailable")
        self.assertEqual(stored["skill_by_run"]["run-a"]["invocations_without_skill_observation"], 2)
        self.assertEqual(stored["skill_by_run"]["run-b"]["skill_status"], "unavailable")
        self.assertEqual(stored["skill_by_run"]["run-b"]["invocations_without_skill_observation"], 1)
        self.assertNotEqual(stored["skill_by_run"]["run-a"]["skill_status"], "partial")
        self.assertNotIn("gpt-5", json.dumps(stored))

    def test_existing_report_is_left_in_place(self) -> None:
        module = _load_live_module()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "live-run-report.json"
            destination.write_text("KEEP_EXISTING_REPORT\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                module.publish_live_report(destination, {"cursor_agent_status": STATUS_SENTINEL})
            self.assertEqual(destination.read_text(encoding="utf-8"), "KEEP_EXISTING_REPORT\n")
            self.assertFalse(any(path.name != "live-run-report.json" for path in Path(directory).iterdir()))

    def test_default_report_path_is_unchanged_when_unset(self) -> None:
        module = _load_live_module()
        previous = os.environ.pop("AGENTIC_REPORT_PATH", None)
        try:
            self.assertEqual(
                module.report_destination(),
                module.DISPOSABLE_ROOT / "live-run-report.json",
            )
            self.assertEqual(
                module.FEATURE_REPOSITORY,
                Path("/tmp/agentic-feature-test-qQFVY6/feature"),
            )
        finally:
            if previous is not None:
                os.environ["AGENTIC_REPORT_PATH"] = previous

    def test_report_path_override_selects_and_publishes_the_explicit_destination(self) -> None:
        import io
        from contextlib import redirect_stdout

        module = _load_live_module()
        previous = os.environ.get("AGENTIC_REPORT_PATH")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "schema-v5-report.json"
            os.environ["AGENTIC_REPORT_PATH"] = str(destination)
            try:
                self.assertEqual(module.report_destination(), destination)
                self.assertEqual(
                    module.FEATURE_REPOSITORY,
                    Path("/tmp/agentic-feature-test-qQFVY6/feature"),
                )
                console = io.StringIO()
                with redirect_stdout(console):
                    module.publish_live_report(
                        module.report_destination(),
                        {
                            "agent_calls": [
                                {
                                    "run_id": "run-a",
                                    "role": "planner",
                                    "invocation_id": "planner:1",
                                    "stdout_excerpt": PROMPT_SENTINEL,
                                }
                            ],
                            "stdout_excerpt": PROMPT_SENTINEL,
                        },
                    )
                persisted = destination.read_text(encoding="utf-8")
                printed = console.getvalue()
                self.assertEqual(list(destination.parent.iterdir()), [destination])
            finally:
                if previous is None:
                    os.environ.pop("AGENTIC_REPORT_PATH", None)
                else:
                    os.environ["AGENTIC_REPORT_PATH"] = previous
        self.assertEqual(persisted, printed)
        stored = json.loads(persisted)
        self.assertEqual(stored["report_schema_version"], 6)
        self.assertEqual(stored["agent_calls"][0]["skill_status"], "unavailable")
        self.assertNotIn(PROMPT_SENTINEL, persisted)
        self.assertNotIn("stdout_excerpt", persisted)

    def test_existing_explicit_destination_is_not_overwritten(self) -> None:
        module = _load_live_module()
        previous = os.environ.get("AGENTIC_REPORT_PATH")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "schema-v5-report.json"
            destination.write_text("KEEP_EXISTING_REPORT\n", encoding="utf-8")
            os.environ["AGENTIC_REPORT_PATH"] = str(destination)
            try:
                chosen = module.report_destination()
                self.assertEqual(chosen, destination)
                with self.assertRaises(FileExistsError):
                    module.publish_live_report(
                        chosen,
                        {"stdout_excerpt": PROMPT_SENTINEL, "cursor_agent_status": STATUS_SENTINEL},
                    )
                self.assertEqual(destination.read_text(encoding="utf-8"), "KEEP_EXISTING_REPORT\n")
                self.assertEqual(list(Path(directory).iterdir()), [destination])
            finally:
                if previous is None:
                    os.environ.pop("AGENTIC_REPORT_PATH", None)
                else:
                    os.environ["AGENTIC_REPORT_PATH"] = previous

    def test_jaeger_int_value_decimal_strings_stay_nonnegative_integers(self) -> None:
        module = _load_live_module()
        attributes = module._jaeger_attributes(
            [
                {"key": "workflow.trace_run_count", "value": {"intValue": "2"}},
                {"key": "agent.cli_duration_ms", "value": {"intValue": "59927"}},
                {"key": "workflow.trace_scope", "value": {"stringValue": "multi_run"}},
                {"key": "agent.usage_reported", "value": {"boolValue": True}},
                {"key": "agent.elapsed_seconds", "value": {"doubleValue": 1.5}},
                {"key": "negative", "value": {"intValue": "-1"}},
                {"key": "negative_int", "value": {"intValue": -3}},
                {"key": "malformed", "value": {"intValue": "2.5"}},
                {"key": "padded", "value": {"intValue": "02"}},
                {"key": "blank", "value": {"intValue": " 2"}},
                {"key": "bool_as_int", "value": {"intValue": True}},
                {"key": "float_as_int", "value": {"intValue": 2.0}},
            ]
        )
        self.assertEqual(attributes["workflow.trace_run_count"], 2)
        self.assertIsInstance(attributes["workflow.trace_run_count"], int)
        self.assertEqual(attributes["agent.cli_duration_ms"], 59927)
        self.assertIsInstance(attributes["agent.cli_duration_ms"], int)
        self.assertEqual(attributes["workflow.trace_scope"], "multi_run")
        self.assertIs(attributes["agent.usage_reported"], True)
        self.assertEqual(attributes["agent.elapsed_seconds"], 1.5)
        self.assertIsInstance(attributes["agent.elapsed_seconds"], float)
        for key in ("negative", "negative_int", "malformed", "padded", "blank", "bool_as_int", "float_as_int"):
            self.assertIsNone(attributes[key])
        self.assertIsNone(module._nonnegative_int("2"))
        self.assertIsNone(module._nonnegative_int(True))
        self.assertIsNone(module._nonnegative_int(2.0))
        self.assertIsNone(module._nonnegative_int(-1))
        self.assertEqual(module._nonnegative_int(2), 2)
        stored = module._redact_jaeger(
            {
                "reachable": True,
                "spans": [
                    {
                        "operation": "workflow.live-run",
                        "trace_id": "42c4592f666328cc531bfb26e5a2793f",
                        "trace_scope": attributes["workflow.trace_scope"],
                        "trace_run_count": attributes["workflow.trace_run_count"],
                        "cli_duration_ms": attributes["agent.cli_duration_ms"],
                        "elapsed_seconds": attributes["agent.elapsed_seconds"],
                        "usage_reported": attributes["agent.usage_reported"],
                    }
                ],
            }
        )
        span = stored["spans"][0]
        self.assertEqual(span["trace_run_count"], 2)
        self.assertEqual(span["cli_duration_ms"], 59927)
        self.assertEqual(span["trace_scope"], "multi_run")
        self.assertEqual(span["elapsed_seconds"], 1.5)
        self.assertIs(span["usage_reported"], True)
        self.assertEqual(module.REPORT_SCHEMA_VERSION, 6)

    def test_live_runner_requires_explicit_approval_for_an_inline_task(self) -> None:
        module = _load_live_module()
        self.assertIsNone(module.explicit_approval(None))
        self.assertIsNone(module.explicit_approval([]))
        granted = module.explicit_approval(["greeting.py", "tests/test_greeting.py"])
        self.assertEqual(granted.approved_paths, ("greeting.py", "tests/test_greeting.py"))
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            task = "Create greeting.py.\nCover an empty name.\n"
            self.assertEqual(module.resolve_live_task(repository, inline=task, spec=None), task)
            spec_path = repository / "SPEC.md"
            spec_path.write_text("File task.\n", encoding="utf-8")
            self.assertEqual(
                module.resolve_live_task(repository, inline=None, spec=None),
                "File task.\n",
            )
            with self.assertRaises(ValueError):
                module.resolve_live_task(repository, inline=task, spec=None)


class TimeoutCollectionTests(unittest.TestCase):
    def test_timeout_cleanup_is_bounded_when_a_descendant_keeps_the_pipe(self) -> None:
        import signal
        import time

        partial = '{"type":"result","usage":{"inputTokens":10,"outputTokens":4,"cacheReadTokens":40,"cacheWriteTokens":2}'
        script = (
            "#!/usr/bin/env python3\n"
            "import os, sys, time\n"
            f"sys.stdout.write({json.dumps(partial)})\n"
            "sys.stdout.flush()\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    os.setsid()\n"
            "    time.sleep(8)\n"
            "    os._exit(0)\n"
            "os._exit(0)\n"
        )

        def invoke():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                executable = write_executable(root, script)
                adapter = CursorCLIAdapter(workspace=root, executable=str(executable))
                agent_request = AgentRequest(
                    run_id="cursor-run-1",
                    role=AgentRole.PLANNER,
                    task=PROMPT_SENTINEL,
                    structured_context={"phase": "plan"},
                    timeout_seconds=0.4,
                )
                return adapter.invoke(agent_request), adapter.last_observation

        signal.signal(signal.SIGALRM, lambda *_args: (_ for _ in ()).throw(TimeoutError("cleanup unbounded")))
        signal.alarm(3)
        started = time.monotonic()
        try:
            result, observation = invoke()
        finally:
            signal.alarm(0)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.5)
        self.assertEqual(result.diagnostic_code, "cursor_timeout")
        self.assertEqual(result.failure_category, AgentFailureCategory.TIMEOUT)
        self.assertIsNone(observation.usage)
        self.assertEqual(observation.stdout_excerpt, "")
        self.assertEqual(observation.stderr_excerpt, "")

    def test_oversized_streams_stay_bounded_and_a_later_result_is_not_invented(self) -> None:
        import tracemalloc

        blob = 12 * 1024 * 1024
        secret_prefix = SECRET_SENTINEL
        result_line = json.dumps(result_event(REPORTED_USAGE))
        script = (
            "#!/usr/bin/env python3\n"
            "import sys\n"
            f"sys.stdout.write({json.dumps(secret_prefix)})\n"
            f"sys.stdout.write('x' * {blob})\n"
            f"sys.stdout.write({json.dumps(chr(10) + result_line + chr(10))})\n"
            "sys.stdout.flush()\n"
            f"sys.stderr.write({json.dumps(secret_prefix)})\n"
            f"sys.stderr.write('y' * {blob})\n"
            "sys.stderr.flush()\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = write_executable(root, script)
            adapter = CursorCLIAdapter(workspace=root, executable=str(executable))
            tracemalloc.start()
            result = adapter.invoke(request())
            observation = adapter.last_observation
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        # 12 MiB is written on each pipe. The cap is 4 MiB per pipe; the peak allows
        # those caps, their one-time copy into bytes, and allocator overhead.
        self.assertLess(peak, 18 * 1024 * 1024)
        self.assertIsNone(observation.usage)
        self.assertNotIn(SECRET_SENTINEL, observation.stdout_excerpt)
        self.assertNotIn(SECRET_SENTINEL, observation.stderr_excerpt)
        self.assertEqual(result.status, AgentStatus.FAILED)

    def test_complete_result_before_overflow_keeps_usage_without_buffering_the_flood(self) -> None:
        import tracemalloc

        blob = 12 * 1024 * 1024
        result_line = json.dumps(result_event(REPORTED_USAGE))
        script = (
            "#!/usr/bin/env python3\n"
            "import sys\n"
            f"sys.stdout.write({json.dumps(result_line + chr(10))})\n"
            "sys.stdout.flush()\n"
            f"sys.stdout.write('x' * {blob})\n"
            "sys.stdout.flush()\n"
            f"sys.stderr.write('y' * {blob})\n"
            "sys.stderr.flush()\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = write_executable(root, script)
            adapter = CursorCLIAdapter(workspace=root, executable=str(executable))
            agent_request = request()
            tracemalloc.start()
            started = __import__("time").monotonic()
            result = adapter.invoke(agent_request)
            elapsed = __import__("time").monotonic() - started
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            observation = adapter.last_observation
        self.assertLess(elapsed, 5)
        self.assertLess(peak, 18 * 1024 * 1024)
        self.assertEqual(result.status, AgentStatus.SUCCEEDED)
        self.assertEqual(observation.usage, REPORTED_USAGE)

    def test_timeout_without_a_terminated_result_line_does_not_invent_usage(self) -> None:
        import time

        event = result_event(REPORTED_USAGE)
        script = (
            "#!/usr/bin/env python3\n"
            "import json, sys, time\n"
            f"sys.stdout.write(json.dumps(json.loads({json.dumps(json.dumps(event))})))\n"
            "sys.stdout.flush()\n"
            "time.sleep(30)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = write_executable(root, script)
            adapter = CursorCLIAdapter(workspace=root, executable=str(executable))
            agent_request = AgentRequest(
                run_id="cursor-run-1",
                role=AgentRole.PLANNER,
                task="Implement the feature in SPEC.md",
                structured_context={"phase": "plan"},
                timeout_seconds=0.4,
            )
            started = time.monotonic()
            result = adapter.invoke(agent_request)
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.5)
        self.assertEqual(result.diagnostic_code, "cursor_timeout")
        self.assertIsNone(adapter.last_observation.usage)

    def test_setsid_descendant_keeps_a_finished_result_without_extending_the_wait(self) -> None:
        import os
        import signal
        import time

        if not hasattr(os, "fork") or not hasattr(os, "setsid"):
            self.skipTest("process-group isolation requires fork and setsid")
        result_line = json.dumps(result_event(REPORTED_USAGE))
        script = (
            "#!/usr/bin/env python3\n"
            "import os, sys, time\n"
            f"sys.stdout.write({json.dumps(result_line + chr(10))})\n"
            "sys.stdout.flush()\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    os.setsid()\n"
            "    time.sleep(8)\n"
            "    os._exit(0)\n"
            "os._exit(0)\n"
        )

        def invoke():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                executable = write_executable(root, script)
                adapter = CursorCLIAdapter(workspace=root, executable=str(executable))
                agent_request = AgentRequest(
                    run_id="cursor-run-1",
                    role=AgentRole.PLANNER,
                    task="Implement the feature in SPEC.md",
                    structured_context={"phase": "plan"},
                    timeout_seconds=0.4,
                )
                return adapter.invoke(agent_request), adapter.last_observation

        signal.signal(signal.SIGALRM, lambda *_args: (_ for _ in ()).throw(TimeoutError("cleanup unbounded")))
        signal.alarm(3)
        started = time.monotonic()
        try:
            result, observation = invoke()
        finally:
            signal.alarm(0)
        self.assertLess(time.monotonic() - started, 2.5)
        self.assertEqual(result.diagnostic_code, "cursor_timeout")
        self.assertEqual(result.failure_category, AgentFailureCategory.TIMEOUT)
        self.assertEqual(observation.usage, REPORTED_USAGE)

    def test_continuous_flood_stays_inside_the_capture_caps(self) -> None:
        import signal
        import time

        from agentic_dev_workflow.cursor_adapter import (
            _MAX_STDERR_BYTES,
            _MAX_STDOUT_BYTES,
            _run_cli,
        )

        script = (
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "chunk = b'x' * 65536\n"
            "err = b'y' * 65536\n"
            "while True:\n"
            "    sys.stdout.buffer.write(chunk)\n"
            "    sys.stderr.buffer.write(err)\n"
            "    sys.stdout.flush()\n"
            "    sys.stderr.flush()\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = write_executable(root, script)
            signal.signal(signal.SIGALRM, lambda *_args: (_ for _ in ()).throw(TimeoutError("flood unbounded")))
            signal.alarm(4)
            started = time.monotonic()
            try:
                completed = _run_cli([str(executable)], cwd=root, timeout_seconds=0.4)
            finally:
                signal.alarm(0)
        self.assertLess(time.monotonic() - started, 2.5)
        self.assertTrue(completed.timed_out)
        self.assertLessEqual(len(completed.stdout), _MAX_STDOUT_BYTES)
        self.assertLessEqual(len(completed.stderr), _MAX_STDERR_BYTES)
        self.assertGreater(len(completed.stdout), 0)
        self.assertGreater(len(completed.stderr), 0)

    def test_truncated_second_result_does_not_replace_the_complete_usage(self) -> None:
        import time

        first = result_event(REPORTED_USAGE)
        partial = '{"type":"result","usage":{"inputTokens":99,"outputTokens":99'
        script = (
            "#!/usr/bin/env python3\n"
            "import json, sys, time\n"
            f"first = json.loads({json.dumps(json.dumps(first))})\n"
            "sys.stdout.write(json.dumps(first) + '\\n')\n"
            f"sys.stdout.write({json.dumps(partial)})\n"
            "sys.stdout.flush()\n"
            "time.sleep(30)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = write_executable(root, script)
            adapter = CursorCLIAdapter(workspace=root, executable=str(executable))
            agent_request = AgentRequest(
                run_id="cursor-run-1",
                role=AgentRole.PLANNER,
                task="Implement the feature in SPEC.md",
                structured_context={"phase": "plan"},
                timeout_seconds=0.4,
            )
            started = time.monotonic()
            result = adapter.invoke(agent_request)
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.5)
        self.assertEqual(result.diagnostic_code, "cursor_timeout")
        self.assertEqual(adapter.last_observation.usage, REPORTED_USAGE)
        self.assertNotEqual(adapter.last_observation.usage["inputTokens"], 99)

