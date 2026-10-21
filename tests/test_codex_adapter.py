import json
import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_dev_workflow.adapters import (
    AdapterError,
    AgentFailureCategory,
    AgentRequest,
    AgentRole,
    AgentStatus,
    invoke_adapter,
)
from agentic_dev_workflow.config import Configuration
from agentic_dev_workflow.execution_policy import ExecutionPolicy
from agentic_dev_workflow.worktrees import WorktreeManager


def git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        shell=False,
    ).stdout.strip()


def make_repository(root: Path) -> tuple[Path, str]:
    repository = root / "source"
    repository.mkdir()
    git(root, "init", "-q", str(repository))
    (repository / "tracked.txt").write_text("fixture\n", encoding="utf-8")
    git(repository, "add", "tracked.txt")
    git(
        repository,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return repository, git(repository, "rev-parse", "HEAD")


class CodexCLIAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repository, self.commit = make_repository(self.root)
        self.worktree_root = self.root / "worktrees"
        self.manager = WorktreeManager(self.worktree_root)
        self.worktree = self.manager.create_worktree(
            self.repository, self.commit, run_id="codex-run-1"
        )
        self.assertTrue(self.worktree.success, self.worktree.issues)
        self.policy = ExecutionPolicy(
            run_id="codex-run-1",
            base_commit=self.commit,
            repository_path=str(self.repository),
            worktree_path=self.worktree.worktree_path,
            clean_baseline_required=True,
            workspace_write_authorized=True,
            network_access=False,
            timeout_seconds=40,
            max_retries=1,
            auto_commit=False,
            auto_merge=False,
            deployment_allowed=False,
            production_database_access=False,
            destructive_cleanup_allowed=False,
        )
        self.enabled_configuration = Configuration.from_values(
            repository=str(self.repository),
            codex_executable=str(self.root / "codex-fixture"),
            environ={"AGENTIC_DEV_WORKFLOW_CODEX_ENABLED": "true"},
        )
        self.disabled_configuration = Configuration.from_values(
            repository=str(self.repository), environ={}
        )

    def request(self, *, role=AgentRole.EXECUTOR, task="Do not log this private prompt"):
        return AgentRequest(
            run_id="codex-run-1",
            role=role,
            task=task,
            structured_context={"feature": "synthetic fixture"},
            timeout_seconds=15,
            correlation_metadata={"secret-correlation": "hidden"},
        )

    def make_fake_codex(
        self,
        directory: Path,
        *,
        event: dict | None = None,
        exit_code: int = 0,
        version_delay: float = 0,
        agent_delay: float = 0,
    ) -> Path:
        executable = directory / "codex-fixture"
        event_data = event or {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": json.dumps(
                    {"output": {"summary": "fixture updated", "items": ["fixture.txt"]}}
                ),
            },
        }
        script = "\n".join(
            (
                f"#!{os.sys.executable}",
                "import json, sys, time",
                "if '--version' in sys.argv:",
                f"    time.sleep({version_delay})",
                "    print('codex-cli 0.163.0')",
                "else:",
                "    sys.stdin.read()",
                f"    time.sleep({agent_delay})",
                f"    print({json.dumps(json.dumps(event_data))})",
                f"    raise SystemExit({exit_code})",
            )
        ) + "\n"
        executable.write_text(script, encoding="utf-8")
        executable.chmod(0o700)
        return executable

    def adapter(self, *, configuration=None, policy=None, max_output_bytes=1_048_576):
        from agentic_dev_workflow.codex_adapter import CodexCLIAdapter

        return CodexCLIAdapter(
            configuration=configuration or self.enabled_configuration,
            execution_policy=policy or self.policy,
            worktree_manager=self.manager,
            max_output_bytes=max_output_bytes,
        )

    def test_disabled_by_default_does_not_resolve_or_launch_codex(self) -> None:
        from agentic_dev_workflow.codex_adapter import CodexCLIAdapter

        adapter = CodexCLIAdapter(
            configuration=self.disabled_configuration,
            execution_policy=self.policy,
            worktree_manager=self.manager,
        )
        with patch("agentic_dev_workflow.codex_adapter._resolve_codex") as which:
            with self.assertRaises(AdapterError) as caught:
                adapter.invoke(self.request())
        which.assert_not_called()
        self.assertEqual(caught.exception.category, AgentFailureCategory.POLICY_REJECTION)
        self.assertEqual(caught.exception.diagnostic_code, "codex_disabled")

    def test_invokes_verified_cli_with_restricted_flags_and_structured_result(self) -> None:
        executable = self.make_fake_codex(self.root)
        from agentic_dev_workflow import codex_adapter

        original_popen = subprocess.Popen
        observed = []

        def recording_popen(*args, **kwargs):
            observed.append((args, kwargs))
            return original_popen(*args, **kwargs)

        with patch.object(codex_adapter, "_resolve_codex", return_value=str(executable)):
            with patch.dict(os.environ, {"API_KEY": "SENTINEL_ENV_SECRET"}):
                with patch.object(codex_adapter.subprocess, "Popen", side_effect=recording_popen):
                    result = invoke_adapter(self.adapter(), self.request())

        self.assertEqual(result.status, AgentStatus.SUCCEEDED)
        self.assertEqual(
            result.structured_output,
            {"summary": "fixture updated", "items": ["fixture.txt"]},
        )
        self.assertEqual(result.exit_code, 0)
        invocations = [entry for entry in observed if "exec" in entry[0][0]]
        self.assertEqual(len(invocations), 1)
        command = invocations[0][0][0]
        kwargs = invocations[0][1]
        self.assertEqual(command[0], str(executable))
        self.assertIn("--strict-config", command)
        self.assertIn("check_for_update_on_startup=false", command)
        self.assertIn("analytics.enabled=false", command)
        self.assertIn("--json", command)
        self.assertIn("--output-schema", command)
        self.assertIn("--sandbox", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "workspace-write")
        self.assertEqual(
            command[command.index("--ask-for-approval") + 1], "on-request"
        )
        self.assertIn("--oss", command)
        self.assertEqual(command[command.index("--local-provider") + 1], "ollama")
        self.assertIn("sandbox_workspace_write.network_access=false", command)
        self.assertIn("sandbox_workspace_write.writable_roots=[]", command)
        self.assertIn("sandbox_workspace_write.exclude_slash_tmp=true", command)
        self.assertIn("sandbox_workspace_write.exclude_tmpdir_env_var=true", command)
        self.assertIn("--ephemeral", command)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", command)
        self.assertNotIn("--approve-for-me", command)
        self.assertNotIn("--add-dir", command)
        self.assertNotIn("--search", command)
        self.assertEqual(command[command.index("--cd") + 1], self.worktree.worktree_path)
        self.assertIs(kwargs["shell"], False)
        self.assertEqual(kwargs["cwd"], self.worktree.worktree_path)
        self.assertNotIn("secret-correlation", kwargs["env"])
        self.assertNotIn("API_KEY", kwargs["env"])
        self.assertNotIn("SENTINEL_ENV_SECRET", repr(kwargs["env"]))

    def test_configured_executable_is_resolved_without_path_search(self) -> None:
        executable = self.make_fake_codex(self.root)
        from agentic_dev_workflow import codex_adapter

        self.assertEqual(
            codex_adapter._resolve_codex(self.enabled_configuration.codex_executable),
            str(executable.resolve()),
        )

    def test_adapter_fails_closed_when_process_tree_cleanup_is_unsupported(self) -> None:
        from agentic_dev_workflow import codex_adapter

        with patch.object(codex_adapter, "_supports_process_tree_cleanup", return_value=False):
            with patch.object(codex_adapter, "_resolve_codex") as resolver:
                result = self.adapter().invoke(self.request())
        resolver.assert_not_called()
        self.assertEqual(result.diagnostic_code, "codex_process_tree_cleanup_unsupported")

    def test_overall_deadline_is_shared_by_version_probe_and_agent(self) -> None:
        executable = self.make_fake_codex(
            self.root, version_delay=0.15, agent_delay=2
        )
        from agentic_dev_workflow import codex_adapter

        request = self.request()
        request = AgentRequest(
            run_id=request.run_id,
            role=request.role,
            task=request.task,
            structured_context=request.structured_context,
            timeout_seconds=0.4,
            correlation_metadata=request.correlation_metadata,
        )
        original = codex_adapter._run_bounded_process
        deadlines = []

        def record_deadline(*args, **kwargs):
            deadlines.append(kwargs["deadline"])
            return original(*args, **kwargs)

        with patch.object(codex_adapter, "_resolve_codex", return_value=str(executable)):
            with patch.object(codex_adapter, "_run_bounded_process", side_effect=record_deadline):
                result = self.adapter().invoke(request)
        self.assertEqual(result.failure_category, AgentFailureCategory.TIMEOUT)
        self.assertEqual(len(deadlines), 2)
        self.assertEqual(deadlines[0], deadlines[1])

    def test_non_executor_roles_use_read_only_sandbox(self) -> None:
        executable = self.make_fake_codex(self.root)
        from agentic_dev_workflow import codex_adapter

        commands = []
        original_popen = subprocess.Popen

        def recording_popen(args, **kwargs):
            commands.append(args)
            return original_popen(args, **kwargs)

        with patch.object(codex_adapter, "_resolve_codex", return_value=str(executable)):
            with patch.object(codex_adapter.subprocess, "Popen", side_effect=recording_popen):
                result = invoke_adapter(self.adapter(), self.request(role=AgentRole.REVIEWER))
        self.assertEqual(result.status, AgentStatus.SUCCEEDED)
        command = next(args for args in commands if "exec" in args)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")

    def test_wrong_run_policy_rejects_before_process_launch(self) -> None:
        from dataclasses import replace

        with patch("agentic_dev_workflow.codex_adapter._resolve_codex") as which:
            with self.assertRaises(AdapterError) as caught:
                self.adapter(policy=replace(self.policy, run_id="another-run")).invoke(
                    self.request()
                )
        which.assert_not_called()
        self.assertEqual(caught.exception.category, AgentFailureCategory.POLICY_REJECTION)

    def test_network_enabled_policy_rejects_before_process_launch(self) -> None:
        from dataclasses import replace

        with patch("agentic_dev_workflow.codex_adapter._resolve_codex") as which:
            with self.assertRaises(AdapterError) as caught:
                self.adapter(policy=replace(self.policy, network_access=True)).invoke(
                    self.request()
                )
        which.assert_not_called()
        self.assertEqual(caught.exception.category, AgentFailureCategory.POLICY_REJECTION)

    def test_owned_worktree_changes_do_not_break_a_later_adapter_attempt(self) -> None:
        (Path(self.worktree.worktree_path) / "untracked.txt").write_text("prior attempt", encoding="utf-8")
        executable = self.make_fake_codex(self.root)
        from agentic_dev_workflow import codex_adapter

        with patch.object(codex_adapter, "_resolve_codex", return_value=str(executable)):
            result = self.adapter().invoke(self.request())
        self.assertEqual(result.status, AgentStatus.SUCCEEDED)

    def test_nonzero_exit_is_mapped_without_exposing_output(self) -> None:
        executable = self.make_fake_codex(
            self.root,
            event={"type": "item.completed", "item": {"type": "agent_message", "text": "private"}},
            exit_code=7,
        )
        from agentic_dev_workflow import codex_adapter

        with patch.object(codex_adapter, "_resolve_codex", return_value=str(executable)):
            result = self.adapter().invoke(self.request())
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.exit_code, 7)
        self.assertEqual(result.failure_category, AgentFailureCategory.EXECUTION_FAILURE)
        self.assertEqual(result.diagnostic_code, "codex_exit_nonzero")
        self.assertNotIn("private", repr(result))
        self.assertNotIn("Do not log this private prompt", repr(result))

    def test_successful_output_is_bounded_and_marked_untrusted(self) -> None:
        prompt_fragment = "synthetic prompt fragment"
        secret = "SYNTHETIC_SECRET_7F3A"
        executable = self.make_fake_codex(
            self.root,
            event={
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": json.dumps(
                        {"output": {"summary": prompt_fragment, "items": [secret]}}
                    ),
                },
            },
        )
        from agentic_dev_workflow import codex_adapter
        from agentic_dev_workflow.adapters import safe_telemetry_attributes

        with patch.object(codex_adapter, "_resolve_codex", return_value=str(executable)):
            result = self.adapter().invoke(self.request(task=prompt_fragment))
        self.assertEqual(result.status, AgentStatus.SUCCEEDED)
        self.assertEqual(result.structured_output["summary"], prompt_fragment)
        self.assertEqual(result.structured_output["items"], [secret])
        attributes = safe_telemetry_attributes(
            self.request(task=prompt_fragment), result, attempt=1, adapter_type="codex"
        )
        self.assertNotIn(prompt_fragment, repr(attributes))
        self.assertNotIn(secret, repr(attributes))

    def test_unexpected_output_properties_are_rejected_without_echo(self) -> None:
        executable = self.make_fake_codex(
            self.root,
            event={
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": json.dumps(
                        {"output": {"summary": "ok", "items": [], "token": "HIDDEN"}}
                    ),
                },
            },
        )
        from agentic_dev_workflow import codex_adapter

        with patch.object(codex_adapter, "_resolve_codex", return_value=str(executable)):
            result = self.adapter().invoke(self.request())
        self.assertEqual(result.failure_category, AgentFailureCategory.INVALID_OUTPUT)
        self.assertNotIn("HIDDEN", repr(result))

    def test_oversized_structured_message_is_rejected_before_json_decode(self) -> None:
        from agentic_dev_workflow import codex_adapter

        oversized = b" " * (codex_adapter.MAX_STRUCTURED_OUTPUT_BYTES + 1)
        with patch.object(codex_adapter.json, "loads") as parser:
            parsed = codex_adapter._validate_structured_payload(oversized)
        self.assertIsNone(parsed)
        parser.assert_not_called()

        oversized_event = self.make_fake_codex(
            self.root,
            event={
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": json.dumps(
                        {
                            "output": {
                                "summary": "x" * (codex_adapter.MAX_STRUCTURED_OUTPUT_BYTES + 1),
                                "items": [],
                            }
                        }
                    ),
                },
            },
        )
        with patch.object(
            codex_adapter, "_resolve_codex", return_value=str(oversized_event)
        ):
            result = self.adapter().invoke(self.request())
        self.assertEqual(result.failure_category, AgentFailureCategory.INVALID_OUTPUT)
        self.assertNotIn("x" * 100, repr(result))

    def test_output_list_and_string_limits_are_enforced(self) -> None:
        from agentic_dev_workflow import codex_adapter

        too_many = json.dumps(
            {"output": {"summary": "ok", "items": ["x"] * (codex_adapter.MAX_OUTPUT_ITEMS + 1)}}
        ).encode()
        too_long = json.dumps(
            {"output": {"summary": "x" * (codex_adapter.MAX_SUMMARY_CHARS + 1), "items": []}}
        ).encode()
        long_item = json.dumps(
            {
                "output": {
                    "summary": "ok",
                    "items": ["x" * (codex_adapter.MAX_OUTPUT_ITEM_CHARS + 1)],
                }
            }
        ).encode()
        for payload in (too_many, too_long, long_item):
            self.assertIsNone(codex_adapter._validate_structured_payload(payload))

    def test_malformed_jsonl_and_malformed_structured_output_fail_safely(self) -> None:
        from agentic_dev_workflow import codex_adapter

        for stdout in (
            b"not-json\n",
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "not-json"}}).encode() + b"\n",
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "[]"}}).encode() + b"\n",
            b'{"type":"item.completed","item":{"type":"agent_message","text":"{\\"output\\":{\\"bad\\":NaN}}"}}\n',
        ):
            with self.subTest(stdout=stdout):
                with patch.object(codex_adapter, "_resolve_codex", return_value="/fake/codex"):
                    with patch.object(
                        codex_adapter,
                        "_run_bounded_process",
                        side_effect=[
                            codex_adapter._ProcessOutcome(0, b"codex-cli 0.163.0", b"", False, False),
                            codex_adapter._ProcessOutcome(0, stdout, b"", False, False),
                        ],
                    ):
                        result = self.adapter().invoke(self.request())
                self.assertEqual(result.failure_category, AgentFailureCategory.INVALID_OUTPUT)
                self.assertEqual(result.diagnostic_code, "codex_output_invalid")
                self.assertEqual(result.structured_output, {})

    def test_timeout_and_output_limit_are_mapped(self) -> None:
        from agentic_dev_workflow import codex_adapter

        with patch.object(codex_adapter, "_resolve_codex", return_value="/fake/codex"):
            with patch.object(
                codex_adapter,
                "_run_bounded_process",
                side_effect=[
                    codex_adapter._ProcessOutcome(0, b"codex-cli 0.163.0", b"", False, False),
                    codex_adapter._ProcessOutcome(None, b"", b"", True, False),
                ],
            ):
                timeout_result = self.adapter().invoke(self.request())
        self.assertEqual(timeout_result.failure_category, AgentFailureCategory.TIMEOUT)
        self.assertEqual(timeout_result.diagnostic_code, "codex_timeout")

        large_event = b"x" * 2048
        with patch.object(codex_adapter, "_resolve_codex", return_value="/fake/codex"):
            with patch.object(
                codex_adapter,
                "_run_bounded_process",
                side_effect=[
                    codex_adapter._ProcessOutcome(0, b"codex-cli 0.163.0", b"", False, False),
                    codex_adapter._ProcessOutcome(None, large_event, b"", False, True),
                ],
            ):
                limit_result = self.adapter(max_output_bytes=1024).invoke(self.request())
        self.assertEqual(limit_result.failure_category, AgentFailureCategory.INVALID_OUTPUT)
        self.assertEqual(limit_result.diagnostic_code, "codex_output_limit")
        self.assertEqual(limit_result.structured_output, {})

    def test_cleanup_failure_diagnostic_does_not_expose_captured_values(self) -> None:
        from agentic_dev_workflow import codex_adapter

        with patch.object(codex_adapter, "_resolve_codex", return_value="/fake/codex"):
            with patch.object(
                codex_adapter,
                "_run_bounded_process",
                side_effect=[
                    codex_adapter._ProcessOutcome(0, b"codex-cli 0.163.0", b"", False, False),
                    codex_adapter._ProcessOutcome(
                        None,
                        b"PROMPT_OR_MODEL_SECRET",
                        b"STDERR_SECRET",
                        True,
                        False,
                        cleanup_incomplete=True,
                    ),
                ],
            ):
                result = self.adapter().invoke(self.request(task="PROMPT_OR_MODEL_SECRET"))
        self.assertEqual(result.diagnostic_code, "codex_cleanup_incomplete")
        self.assertNotIn("PROMPT_OR_MODEL_SECRET", repr(result))
        self.assertNotIn("STDERR_SECRET", repr(result))

    def test_process_timeout_and_capture_limit_are_enforced(self) -> None:
        from agentic_dev_workflow.codex_adapter import _run_bounded_process

        environment = {"PATH": os.defpath}
        timeout = _run_bounded_process(
            [os.sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=self.root,
            environment=environment,
            timeout_seconds=0.1,
            input_text="",
            max_output_bytes=1024,
        )
        self.assertTrue(timeout.timed_out)

        overflow = _run_bounded_process(
            [os.sys.executable, "-c", "import sys; sys.stdout.write('x' * 1000000)"],
            cwd=self.root,
            environment=environment,
            timeout_seconds=5,
            input_text="",
            max_output_bytes=1024,
        )
        self.assertTrue(overflow.output_limit_exceeded)
        self.assertLessEqual(len(overflow.stdout) + len(overflow.stderr), 1024)

    @unittest.skipUnless(os.name == "posix", "process-group cleanup is POSIX-specific")
    def test_timeout_kills_child_that_ignores_termination(self) -> None:
        from agentic_dev_workflow.codex_adapter import _run_bounded_process

        pid_file = self.root / "child.pid"
        parent_script = (
            "import pathlib, signal, subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])\n"
            f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
            "time.sleep(30)\n"
        )
        try:
            outcome = _run_bounded_process(
                [os.sys.executable, "-c", parent_script],
                cwd=self.root,
                environment={"PATH": os.defpath},
                timeout_seconds=0.3,
                input_text="",
                max_output_bytes=1024,
            )
            self.assertTrue(outcome.timed_out)
            child_pid = int(pid_file.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.02)
            else:
                self.fail("child process remained after process-group cleanup")
        finally:
            if pid_file.exists():
                try:
                    os.kill(int(pid_file.read_text(encoding="utf-8")), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass

    def test_launch_failure_and_unverified_version_fail_closed(self) -> None:
        from agentic_dev_workflow import codex_adapter

        with patch.object(codex_adapter, "_resolve_codex", return_value=None):
            unavailable = self.adapter().invoke(self.request())
        self.assertEqual(unavailable.failure_category, AgentFailureCategory.UNAVAILABLE_ADAPTER)
        self.assertEqual(unavailable.diagnostic_code, "codex_not_found")

        with patch.object(codex_adapter, "_resolve_codex", return_value="/fake/codex"):
            with patch.object(
                codex_adapter,
                "_run_bounded_process",
                return_value=codex_adapter._ProcessOutcome(None, b"", b"", False, False, "OSError"),
            ):
                launch_failure = self.adapter().invoke(self.request())
        self.assertEqual(launch_failure.failure_category, AgentFailureCategory.UNAVAILABLE_ADAPTER)
        self.assertEqual(launch_failure.diagnostic_code, "codex_launch_failed")

        with patch.object(codex_adapter, "_resolve_codex", return_value="/fake/codex"):
            with patch.object(
                codex_adapter,
                "_run_bounded_process",
                return_value=codex_adapter._ProcessOutcome(0, b"codex-cli 99.0.0", b"", False, False),
            ):
                unsupported = self.adapter().invoke(self.request())
        self.assertEqual(unsupported.failure_category, AgentFailureCategory.UNAVAILABLE_ADAPTER)
        self.assertEqual(unsupported.diagnostic_code, "codex_version_unsupported")

    def test_explicit_executable_resolution_rejects_missing_or_nonexecutable(self) -> None:
        from agentic_dev_workflow import codex_adapter

        self.assertIsNone(codex_adapter._resolve_codex(None))
        self.assertIsNone(codex_adapter._resolve_codex(self.root / "missing-codex"))
        directory = self.root / "not-an-executable"
        directory.mkdir()
        self.assertIsNone(codex_adapter._resolve_codex(directory))
        file = self.root / "not-executable-file"
        file.write_text("not executable", encoding="utf-8")
        file.chmod(0o600)
        self.assertIsNone(codex_adapter._resolve_codex(file))
        self.assertIsNone(codex_adapter._resolve_codex(Path("relative/codex")))

    def test_missing_configured_executable_returns_safe_failure(self) -> None:
        result = self.adapter().invoke(self.request())
        self.assertEqual(result.failure_category, AgentFailureCategory.UNAVAILABLE_ADAPTER)
        self.assertEqual(result.diagnostic_code, "codex_not_found")

    def test_worktree_symlink_is_rejected_before_executable_resolution(self) -> None:
        destination = Path(self.worktree.worktree_path)
        moved = self.root / "moved-worktree"
        destination.rename(moved)
        destination.symlink_to(moved, target_is_directory=True)
        from agentic_dev_workflow import codex_adapter

        with patch.object(codex_adapter, "_resolve_codex") as resolver:
            with self.assertRaises(AdapterError):
                self.adapter().invoke(self.request())
        resolver.assert_not_called()

    def test_inconsistent_manifest_is_rejected_before_executable_resolution(self) -> None:
        manifest = self.worktree_root / "run-codex-run-1.json"
        value = json.loads(manifest.read_text(encoding="utf-8"))
        value["base_commit"] = "0" * 40
        manifest.write_text(json.dumps(value), encoding="utf-8")
        from agentic_dev_workflow import codex_adapter

        with patch.object(codex_adapter, "_resolve_codex") as resolver:
            with self.assertRaises(AdapterError):
                self.adapter().invoke(self.request())
        resolver.assert_not_called()

    def test_worktree_changed_during_version_probe_is_rechecked_before_agent_launch(self) -> None:
        executable = self.make_fake_codex(self.root)
        from agentic_dev_workflow import codex_adapter

        original = codex_adapter._run_bounded_process
        calls = []

        def swap_after_version(*args, **kwargs):
            outcome = original(*args, **kwargs)
            calls.append(args[0])
            if "--version" in args[0]:
                destination = Path(self.worktree.worktree_path)
                moved = self.root / "moved-during-probe"
                destination.rename(moved)
                destination.symlink_to(moved, target_is_directory=True)
            return outcome

        with patch.object(codex_adapter, "_resolve_codex", return_value=str(executable)):
            with patch.object(codex_adapter, "_run_bounded_process", side_effect=swap_after_version):
                with self.assertRaises(AdapterError):
                    self.adapter().invoke(self.request())
        self.assertEqual(len(calls), 1)
        self.assertIn("--version", calls[0])

    def test_sensitive_prompt_is_not_in_diagnostics_or_telemetry(self) -> None:
        from agentic_dev_workflow import codex_adapter
        from agentic_dev_workflow.adapters import safe_telemetry_attributes

        task = "SENTINEL_PROMPT_SECRET"
        with patch.object(codex_adapter, "_resolve_codex", return_value="/fake/codex"):
            with patch.dict(os.environ, {"API_KEY": "SENTINEL_ENV_SECRET"}):
                with patch.object(
                    codex_adapter,
                    "_run_bounded_process",
                    return_value=codex_adapter._ProcessOutcome(
                        1, b"", b"SENTINEL_STDERR_SECRET", False, False
                    ),
                ):
                    result = self.adapter().invoke(self.request(task=task))
        attrs = safe_telemetry_attributes(
            self.request(task=task), result, attempt=1, adapter_type="codex"
        )
        self.assertNotIn(task, repr(result))
        self.assertNotIn("SENTINEL_STDERR_SECRET", repr(result))
        self.assertNotIn(task, repr(attrs))


if __name__ == "__main__":
    unittest.main()
