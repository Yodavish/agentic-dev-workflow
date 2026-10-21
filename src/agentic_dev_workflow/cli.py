"""Command-line interface for local workflow diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from opentelemetry.trace import Status, StatusCode

from agentic_dev_workflow.checkpointing import (
    CheckpointStorageError,
    open_sqlite_checkpointer,
)
from agentic_dev_workflow.config import Configuration, ConfigurationError, resolve_data_dir
from agentic_dev_workflow.git_safety import verify_git_baseline
from agentic_dev_workflow.mock_workflow import MockScenario, run_mock_workflow
from agentic_dev_workflow.tracing import create_tracing_session, new_run_id
from agentic_dev_workflow.workflow import WorkflowResult, run_preflight
from agentic_dev_workflow.worktrees import WorktreeManager


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-dev-workflow",
        description="Safety-first local development workflow diagnostics.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "preflight", "trace-smoke", "run"):
        command = commands.add_parser(name)
        command.add_argument("--repository", help="Target Git repository path")
        command.add_argument("--otlp-endpoint", help="OTLP HTTP trace endpoint")
    demo = commands.add_parser("workflow-demo", help="Run deterministic mocked LangGraph scenarios")
    demo.add_argument("--scenario", choices=[scenario.value for scenario in MockScenario], default="success")
    demo.add_argument("--run-id", help="Explicit workflow run ID (generated when omitted)")
    demo.add_argument("--max-retries", type=int, default=1, help="Repair retries after the first attempt (0-2)")
    demo.add_argument("--no-tracing", action="store_true", help="Run without exporting telemetry")
    demo.add_argument("--otlp-endpoint", help="OTLP HTTP trace endpoint")
    demo.add_argument(
        "--durable",
        action="store_true",
        help="Persist this mock run in runner-owned SQLite storage",
    )
    demo.add_argument(
        "--data-dir",
        help="Runner-owned data directory (or AGENTIC_DEV_WORKFLOW_DATA_DIR)",
    )
    demo.add_argument(
        "--pause-after-planning",
        action="store_true",
        help="Persist planner output and exit before mock execution",
    )
    demo.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching paused SQLite-backed mock run",
    )

    baseline = commands.add_parser("baseline-check", help="Verify an explicit Git baseline")
    baseline.add_argument("--repository", required=True, help="Explicit source Git worktree root")
    baseline.add_argument("--base-commit", required=True, help="Full 40- or 64-character commit ID")

    worktree = commands.add_parser(
        "worktree", help="Explicitly create or clean up a manager-owned temporary worktree"
    )
    worktree.add_argument("--action", choices=("create", "cleanup"), required=True)
    worktree.add_argument("--source-repository", required=True, help="Explicit source Git worktree root")
    worktree.add_argument("--base-commit", required=True, help="Full 40- or 64-character commit ID")
    worktree.add_argument("--run-id", required=True, help="Safe dedicated workflow run ID")
    worktree.add_argument("--worktree-root", required=True, help="Explicit runner-owned worktree root")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a CLI command and return its process exit code."""
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    if arguments.command == "workflow-demo":
        return _workflow_demo(arguments)
    if arguments.command == "baseline-check":
        result = verify_git_baseline(arguments.repository, arguments.base_commit)
        report = asdict(result)
        report["success"] = result.success
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if result.success else 1
    if arguments.command == "worktree":
        manager = WorktreeManager(arguments.worktree_root)
        if arguments.action == "create":
            result = manager.create_worktree(
                arguments.source_repository,
                arguments.base_commit,
                run_id=arguments.run_id,
            )
        else:
            result = manager.cleanup_worktree(
                arguments.source_repository,
                arguments.base_commit,
                run_id=arguments.run_id,
            )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return 0 if result.success else 1
    try:
        config = Configuration.from_values(arguments.repository, arguments.otlp_endpoint)
    except ConfigurationError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    if arguments.command == "doctor":
        return _doctor(config)
    if arguments.command == "trace-smoke":
        return _trace_smoke(config.otlp_endpoint)

    is_run = arguments.command == "run"
    result = run_preflight(
        config, operation="run" if is_run else "preflight", dry_run_only=is_run
    )
    print(json.dumps(_workflow_json(result), indent=2, sort_keys=True))
    if result.telemetry_error:
        print(f"observability warning: {result.telemetry_error}", file=sys.stderr)
    return 0 if result.outcome == "success" else 1


def _workflow_demo(arguments: argparse.Namespace) -> int:
    """Run only deterministic mock nodes; this command never loads repository config."""
    if arguments.pause_after_planning and arguments.resume:
        print(
            "workflow configuration error: pause and resume cannot be combined",
            file=sys.stderr,
        )
        return 2
    if (arguments.pause_after_planning or arguments.resume) and not arguments.run_id:
        print(
            "workflow configuration error: pause and resume require an explicit --run-id",
            file=sys.stderr,
        )
        return 2
    durable = (
        arguments.durable
        or arguments.data_dir is not None
        or arguments.pause_after_planning
        or arguments.resume
    )
    session = None
    telemetry_error = None
    config = None
    data_dir = None
    if not arguments.no_tracing:
        try:
            config = Configuration.from_values(
                repository=".",
                otlp_endpoint=arguments.otlp_endpoint,
            )
        except ConfigurationError as error:
            print(f"configuration error: {error}", file=sys.stderr)
            return 2
    if durable:
        try:
            data_dir = resolve_data_dir(arguments.data_dir)
        except ConfigurationError as error:
            print(f"configuration error: {error}", file=sys.stderr)
            return 2
    if not arguments.no_tracing:
        try:
            assert config is not None
            session = create_tracing_session(config.otlp_endpoint)
        except Exception as error:
            telemetry_error = f"tracing setup failed ({type(error).__name__})"
    run_id = arguments.run_id or new_run_id()
    try:
        run_options = {
            "scenario": arguments.scenario,
            "run_id": run_id,
            "max_retries": arguments.max_retries,
            "tracing_session": session,
            "resume": arguments.resume,
        }
        if arguments.pause_after_planning:
            run_options["interrupt_before"] = ["mock_executor"]
        if durable:
            assert data_dir is not None
            with open_sqlite_checkpointer(
                data_dir,
                require_existing=arguments.resume,
            ) as saver:
                result = run_mock_workflow(
                    "Synthetic demo specification for a small status indicator.",
                    checkpointer=saver,
                    **run_options,
                )
        else:
            result = run_mock_workflow(
                "Synthetic demo specification for a small status indicator.",
                **run_options,
            )
        if session is not None and not session.flush():
            telemetry_error = "trace export failed or could not be confirmed"
        report = {
            "run_id": result["run_id"],
            "scenario": result["scenario"],
            "final_status": (
                "paused"
                if result["final_status"] == "running"
                else result["final_status"]
            ),
            "execution_attempts": result["execution_attempt"],
            "repair_retries_used": result["repair_retries_used"],
            "attempt_history_entries": len(result["attempt_history"]),
            "simulated": True,
            "verification_summary": result["verification_summary"],
            "telemetry_error": telemetry_error,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if telemetry_error:
            print(f"observability warning: {telemetry_error}", file=sys.stderr)
        return 0 if result["final_status"] in {"success", "running"} else 1
    except ValueError as error:
        print(f"workflow configuration error: {error}", file=sys.stderr)
        return 2
    except CheckpointStorageError as error:
        print(f"checkpoint error: {error}", file=sys.stderr)
        return 1
    finally:
        if session is not None:
            try:
                session.shutdown()
            except Exception as error:
                print(f"observability warning: trace shutdown failed ({type(error).__name__})", file=sys.stderr)


def _doctor(config: Configuration) -> int:
    parsed = urlsplit(config.otlp_endpoint)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    tools = {
        "python": {"available": True, "version": sys.version.split()[0]},
        "git": _tool_version("git", "--version"),
        "uv": _tool_version("uv", "--version"),
        "opentelemetry_api": _module_available("opentelemetry.trace"),
        "opentelemetry_sdk": _module_available("opentelemetry.sdk.trace"),
        "otlp_http_exporter": _module_available(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        ),
    }
    endpoint_reachable = _endpoint_reachable(parsed.hostname or "", port)
    run_id = new_run_id()
    telemetry_error = None
    session = None
    try:
        session = create_tracing_session(config.otlp_endpoint)
    except Exception as error:
        telemetry_error = f"tracing setup failed ({type(error).__name__})"

    ready = all(
        tools[name]["available"]
        for name in ("git", "opentelemetry_api", "opentelemetry_sdk", "otlp_http_exporter")
    )
    if session is not None:
        try:
            with session.tracer.start_as_current_span("workflow.doctor") as span:
                span.set_attribute("workflow.run_id", run_id)
                span.set_attribute("workflow.operation", "doctor")
                span.set_attribute("workflow.phase", "diagnostics")
                span.set_attribute(
                    "workflow.outcome",
                    "success" if ready and endpoint_reachable else "blocked",
                )
                if not ready or not endpoint_reachable:
                    span.set_status(Status(StatusCode.ERROR, "required tool unavailable"))
            if not session.flush():
                telemetry_error = "trace export failed or could not be confirmed"
        except Exception as error:
            telemetry_error = f"trace export failed ({type(error).__name__})"
        finally:
            try:
                session.shutdown()
            except Exception as error:
                telemetry_error = f"trace shutdown failed ({type(error).__name__})"

    report = {
        "run_id": run_id,
        "repository": str(config.repository),
        "otlp_endpoint_configured": True,
        "otlp_endpoint_reachable": endpoint_reachable,
        "tools": tools,
        "trace_exported": session is not None and telemetry_error is None,
        "telemetry_error": telemetry_error,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ready and endpoint_reachable else 1


def _trace_smoke(endpoint: str) -> int:
    run_id = new_run_id()
    try:
        session = create_tracing_session(endpoint)
    except Exception as error:
        print(json.dumps({"run_id": run_id, "exported": False}, indent=2))
        print(f"trace setup failed ({type(error).__name__})", file=sys.stderr)
        return 1

    try:
        with session.tracer.start_as_current_span("workflow.trace-smoke") as parent:
            parent.set_attribute("workflow.run_id", run_id)
            parent.set_attribute("workflow.operation", "trace-smoke")
            parent.set_attribute("workflow.phase", "observability")
            parent.set_attribute("workflow.outcome", "success")
            with session.tracer.start_as_current_span("workflow.trace-smoke.child") as child:
                child.set_attribute("workflow.run_id", run_id)
                child.set_attribute("workflow.phase", "observability")
                child.set_attribute("workflow.outcome", "success")
        exported = session.flush()
        error_message = None if exported else "trace export failed or could not be confirmed"
    except Exception as error:
        exported = False
        error_message = f"trace export failed ({type(error).__name__})"
    finally:
        try:
            session.shutdown()
        except Exception as error:
            exported = False
            error_message = f"trace shutdown failed ({type(error).__name__})"

    print(json.dumps({"run_id": run_id, "exported": exported}, indent=2, sort_keys=True))
    if error_message:
        print(error_message, file=sys.stderr)
    return 0 if exported else 1


def _workflow_json(result: WorkflowResult) -> dict:
    data = asdict(result)
    data["repository"] = asdict(result.repository)
    return data


def _tool_version(executable: str, argument: str) -> dict[str, object]:
    path = shutil.which(executable)
    if path is None:
        return {"available": False}
    try:
        result = subprocess.run(
            [path, argument],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False}
    return {
        "available": result.returncode == 0,
        "version": result.stdout.strip() if result.returncode == 0 else None,
    }


def _module_available(name: str) -> dict[str, bool]:
    try:
        return {"available": importlib.util.find_spec(name) is not None}
    except (ImportError, ModuleNotFoundError, ValueError):
        return {"available": False}


def _endpoint_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
