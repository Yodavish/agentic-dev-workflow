from __future__ import annotations

import json
import os
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import textwrap
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.sqlite import SqliteSaver

from agentic_dev_workflow.checkpointing import (
    CheckpointStorageError,
    open_sqlite_checkpointer,
)
from agentic_dev_workflow import checkpointing
from agentic_dev_workflow.mock_workflow import build_mock_graph


class CheckpointStorageTests(unittest.TestCase):
    def test_saver_uses_wal_and_closes_connection_on_context_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "checkpoints.sqlite3"
            with open_sqlite_checkpointer(Path(directory)) as saver:
                saver.setup()
                connection = saver.conn
                self.assertEqual(
                    connection.execute("PRAGMA journal_mode").fetchone()[0].lower(),
                    "wal",
                )
                self.assertEqual(
                    connection.execute("PRAGMA busy_timeout").fetchone()[0],
                    5000,
                )
                table_names = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                self.assertTrue({"checkpoints", "writes"}.issubset(table_names))
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")
            self.assertTrue(database.is_file())

    def test_new_storage_has_restrictive_permissions_where_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "private-data"
            with open_sqlite_checkpointer(data_dir) as saver:
                saver.setup()
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(data_dir.stat().st_mode), 0o700)
                self.assertEqual(
                    stat.S_IMODE((data_dir / "checkpoints.sqlite3").stat().st_mode),
                    0o600,
                )

    def test_existing_permissive_database_is_rejected_without_chmod(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            database = data_dir / "checkpoints.sqlite3"
            database.touch(mode=0o644)
            if os.name == "posix":
                database.chmod(0o644)
                with self.assertRaises(CheckpointStorageError):
                    with open_sqlite_checkpointer(data_dir):
                        self.fail("unsafe existing database was opened")
                self.assertEqual(stat.S_IMODE(database.stat().st_mode), 0o644)

    def test_existing_permissive_directory_is_rejected_without_chmod(self) -> None:
        if os.name != "posix":
            self.skipTest("POSIX directory mode assertions are unavailable")
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "public-data"
            data_dir.mkdir(mode=0o755)
            data_dir.chmod(0o755)
            with self.assertRaises(CheckpointStorageError):
                with open_sqlite_checkpointer(data_dir):
                    self.fail("unsafe existing directory was opened")
            self.assertEqual(stat.S_IMODE(data_dir.stat().st_mode), 0o755)

    def test_empty_or_unrelated_existing_database_is_not_initialized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            database = data_dir / "checkpoints.sqlite3"
            database.touch(mode=0o600)
            if os.name == "posix":
                database.chmod(0o600)
            with self.assertRaises(CheckpointStorageError):
                with open_sqlite_checkpointer(data_dir):
                    self.fail("empty existing database was initialized")
            self.assertEqual(database.stat().st_size, 0)

            database.unlink()
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE unrelated (id INTEGER)")
            connection.commit()
            connection.close()
            if os.name == "posix":
                database.chmod(0o600)
            original = database.read_bytes()
            with self.assertRaises(CheckpointStorageError):
                with open_sqlite_checkpointer(data_dir):
                    self.fail("unrelated SQLite database was adopted")
            self.assertEqual(database.read_bytes(), original)

    def test_malformed_checkpoint_payload_is_reported_and_preserved(self) -> None:
        from agentic_dev_workflow.mock_workflow import (
            create_initial_state,
            resume_mock_workflow,
        )

        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "checkpoints"
            run_id = "corrupt-payload-case"
            thread_id = f"agentic-dev-workflow:{run_id}"
            config = {"configurable": {"thread_id": thread_id}}
            with open_sqlite_checkpointer(data_dir) as saver:
                graph = build_mock_graph(
                    checkpointer=saver,
                    interrupt_before=["mock_executor"],
                )
                graph.invoke(create_initial_state("Private test spec", run_id=run_id), config)

            database = data_dir / "checkpoints.sqlite3"
            with closing(sqlite3.connect(database)) as connection:
                with connection:
                    connection.execute(
                        "UPDATE checkpoints SET checkpoint = ? WHERE thread_id = ?",
                        (b"malformed checkpoint bytes", thread_id),
                    )
            with open_sqlite_checkpointer(data_dir, require_existing=True) as saver:
                graph = build_mock_graph(checkpointer=saver)
                with self.assertRaisesRegex(ValueError, "could not be read") as raised:
                    resume_mock_workflow(
                        graph,
                        config,
                        run_id=run_id,
                        spec_text="Private test spec",
                    )
                self.assertNotIn("Private test spec", str(raised.exception))
            with closing(sqlite3.connect(database)) as connection:
                with connection:
                    payloads = connection.execute(
                        "SELECT checkpoint FROM checkpoints WHERE thread_id = ?",
                        (thread_id,),
                    ).fetchall()
            self.assertTrue(payloads)
            self.assertTrue(all(row[0] == b"malformed checkpoint bytes" for row in payloads))

    def test_corrupt_database_is_preserved_and_reported_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            database = data_dir / "checkpoints.sqlite3"
            original = b"not a sqlite database\n"
            database.write_bytes(original)
            if os.name == "posix":
                database.chmod(0o600)
            graph = build_mock_graph()
            with self.assertRaises(CheckpointStorageError):
                with open_sqlite_checkpointer(data_dir) as saver:
                    graph = build_mock_graph(
                        checkpointer=saver,
                        interrupt_before=["mock_executor"],
                    )
                    graph.invoke(
                        {"run_id": "corrupt-case"},
                        {"configurable": {"thread_id": "corrupt-case"}},
                    )
            self.assertEqual(database.read_bytes(), original)

    def test_sqlite_lock_wait_is_bounded_and_reported_without_raw_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "checkpoints"
            with open_sqlite_checkpointer(data_dir) as saver:
                saver.setup()
            database = data_dir / "checkpoints.sqlite3"
            blocker = sqlite3.connect(database, timeout=0.1)
            blocker.execute("BEGIN IMMEDIATE")
            try:
                start = time.monotonic()
                with patch.object(checkpointing, "SQLITE_BUSY_TIMEOUT_MS", 100):
                    with self.assertRaisesRegex(CheckpointStorageError, "storage failed"):
                        with open_sqlite_checkpointer(data_dir) as saver:
                            saver.conn.execute("CREATE TABLE lock_probe (id INTEGER)")
                elapsed = time.monotonic() - start
            finally:
                blocker.rollback()
                blocker.close()
            self.assertGreaterEqual(elapsed, 0.08)
            self.assertLess(elapsed, 1.0)

    def test_factory_context_matches_verified_sqlite_saver_lifecycle(self) -> None:
        self.assertTrue(callable(SqliteSaver.from_conn_string))
        with tempfile.TemporaryDirectory() as directory:
            context = SqliteSaver.from_conn_string(
                str(Path(directory) / "factory.sqlite3")
            )
            with context as saver:
                connection = saver.conn
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")


class PersistentWorkflowProcessTests(unittest.TestCase):
    def _invoke_cli(self, data_dir: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            "-c",
            "import sys; from agentic_dev_workflow.cli import main; raise SystemExit(main(sys.argv[1:]))",
            "workflow-demo",
            "--no-tracing",
            "--data-dir",
            str(data_dir),
            *arguments,
        ]
        return subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
            shell=False,
        )

    def test_pause_then_resume_in_separate_processes_without_replanning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "checkpoints"
            run_id = "process-resume-case"
            paused = self._invoke_cli(
                data_dir,
                "--scenario",
                "retry-success",
                "--max-retries",
                "1",
                "--run-id",
                run_id,
                "--pause-after-planning",
            )
            self.assertEqual(paused.returncode, 0, paused.stderr)
            self.assertEqual(json.loads(paused.stdout)["final_status"], "paused")
            self.assertEqual(json.loads(paused.stdout)["execution_attempts"], 0)
            with open_sqlite_checkpointer(data_dir, require_existing=True) as saver:
                graph = build_mock_graph(checkpointer=saver)
                checkpoint = graph.get_state(
                    {"configurable": {"thread_id": f"agentic-dev-workflow:{run_id}"}}
                )
                self.assertEqual(checkpoint.next, ("mock_executor",))
                self.assertIsNotNone(checkpoint.values["plan"])

            resumed = self._invoke_cli(
                data_dir,
                "--scenario",
                "retry-success",
                "--max-retries",
                "1",
                "--run-id",
                run_id,
                "--resume",
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            report = json.loads(resumed.stdout)
            self.assertEqual(report["final_status"], "success")
            self.assertEqual(report["execution_attempts"], 2)
            self.assertEqual(report["repair_retries_used"], 1)
            self.assertEqual(report["attempt_history_entries"], 2)

            # LangGraph's persisted history records the planner completion as
            # the checkpoint whose next node is mock_executor. It must occur once.
            with open_sqlite_checkpointer(data_dir) as saver:
                graph = build_mock_graph(checkpointer=saver)
                config = {
                    "configurable": {
                        "thread_id": f"agentic-dev-workflow:{run_id}"
                    }
                }
                planner_checkpoints = sum(
                    snapshot.next == ("mock_executor",)
                    and snapshot.values.get("execution_attempt") == 0
                    for snapshot in graph.get_state_history(config)
                )
            self.assertEqual(planner_checkpoints, 1)

    def test_resume_with_missing_database_does_not_create_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "missing-checkpoints"
            result = self._invoke_cli(
                data_dir,
                "--run-id",
                "missing-process-case",
                "--resume",
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("checkpoint database does not exist", result.stderr)
            self.assertFalse(data_dir.exists())

    def test_retry_history_survives_saver_reopen_and_resume(self) -> None:
        from agentic_dev_workflow.mock_workflow import (
            MockScenario,
            create_initial_state,
            resume_mock_workflow,
        )

        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "checkpoints"
            run_id = "retry-resume-case"
            config = {"configurable": {"thread_id": f"agentic-dev-workflow:{run_id}"}}
            with open_sqlite_checkpointer(data_dir) as saver:
                graph = build_mock_graph(
                    checkpointer=saver,
                    interrupt_before=["retry_decision"],
                )
                paused = graph.invoke(
                    create_initial_state(
                        "Synthetic retry-persistence case",
                        scenario=MockScenario.EXHAUSTED_RETRIES,
                        run_id=run_id,
                        max_retries=1,
                    ),
                    config,
                )
                self.assertEqual(paused["execution_attempt"], 1)
                self.assertEqual(len(paused["attempt_history"]), 1)

            with open_sqlite_checkpointer(data_dir, require_existing=True) as saver:
                graph = build_mock_graph(checkpointer=saver)
                result = resume_mock_workflow(
                    graph,
                    config,
                    run_id=run_id,
                    spec_text="Synthetic retry-persistence case",
                    scenario=MockScenario.EXHAUSTED_RETRIES,
                    max_retries=1,
                )
            self.assertEqual(result["final_status"], "failed")
            self.assertEqual(result["execution_attempt"], 2)
            self.assertEqual(result["repair_retries_used"], 1)
            self.assertEqual(len(result["attempt_history"]), 2)
            self.assertEqual(len(result["errors"]), 2)

    def test_storage_paths_isolate_identical_thread_ids(self) -> None:
        from agentic_dev_workflow.mock_workflow import (
            MockScenario,
            create_initial_state,
        )

        with tempfile.TemporaryDirectory() as directory:
            first_dir = Path(directory) / "first"
            second_dir = Path(directory) / "second"
            config = {"configurable": {"thread_id": "agentic-dev-workflow:same-id"}}
            first_state = create_initial_state("first", run_id="same-id")
            second_state = create_initial_state("second", run_id="same-id")
            second_state["scenario"] = MockScenario.RETRY_SUCCESS.value
            for data_dir, state in ((first_dir, first_state), (second_dir, second_state)):
                with open_sqlite_checkpointer(data_dir) as saver:
                    graph = build_mock_graph(
                        checkpointer=saver,
                        interrupt_before=["mock_executor"],
                    )
                    graph.invoke(state, config)
                    self.assertEqual(graph.get_state(config).values["spec_text"], state["spec_text"])
            with open_sqlite_checkpointer(first_dir, require_existing=True) as saver:
                graph = build_mock_graph(checkpointer=saver)
                self.assertEqual(graph.get_state(config).values["scenario"], "success")
            with open_sqlite_checkpointer(second_dir, require_existing=True) as saver:
                graph = build_mock_graph(checkpointer=saver)
                self.assertEqual(graph.get_state(config).values["scenario"], "retry-success")

    def test_resume_mismatches_are_rejected_before_execution(self) -> None:
        from agentic_dev_workflow.mock_workflow import (
            CHECKPOINT_FORMAT_VERSION,
            MockScenario,
            create_initial_state,
            resume_mock_workflow,
        )

        mutations = [
            ("spec", "Changed specification", MockScenario.SUCCESS, 1),
            ("scenario", "Synthetic identity case", MockScenario.RETRY_SUCCESS, 1),
            ("retry", "Synthetic identity case", MockScenario.SUCCESS, 0),
            ("format", "Synthetic identity case", MockScenario.SUCCESS, 1),
        ]
        for label, spec, scenario, retries in mutations:
            with self.subTest(mismatch=label), tempfile.TemporaryDirectory() as directory:
                data_dir = Path(directory) / "checkpoints"
                run_id = "identity-case"
                config = {"configurable": {"thread_id": f"agentic-dev-workflow:{run_id}"}}
                state = create_initial_state(
                    "Synthetic identity case", run_id=run_id, max_retries=1
                )
                if label == "format":
                    state["checkpoint_format_version"] = CHECKPOINT_FORMAT_VERSION + 1
                with open_sqlite_checkpointer(data_dir) as saver:
                    graph = build_mock_graph(
                        checkpointer=saver,
                        interrupt_before=["mock_executor"],
                    )
                    graph.invoke(state, config)
                with open_sqlite_checkpointer(data_dir, require_existing=True) as saver:
                    graph = build_mock_graph(checkpointer=saver)
                    with self.assertRaises(ValueError):
                        resume_mock_workflow(
                            graph,
                            config,
                            run_id=run_id,
                            spec_text=spec,
                            scenario=scenario,
                            max_retries=retries,
                        )
                    self.assertEqual(graph.get_state(config).values["execution_attempt"], 0)

    def test_terminal_sqlite_run_is_not_resumable(self) -> None:
        from agentic_dev_workflow.mock_workflow import (
            build_mock_graph,
            create_initial_state,
            resume_mock_workflow,
        )

        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "checkpoints"
            run_id = "terminal-sqlite"
            config = {"configurable": {"thread_id": f"agentic-dev-workflow:{run_id}"}}
            with open_sqlite_checkpointer(data_dir) as saver:
                graph = build_mock_graph(checkpointer=saver)
                graph.invoke(create_initial_state("Terminal fixture", run_id=run_id), config)
            with open_sqlite_checkpointer(data_dir, require_existing=True) as saver:
                graph = build_mock_graph(checkpointer=saver)
                with self.assertRaisesRegex(ValueError, "terminal"):
                    resume_mock_workflow(
                        graph,
                        config,
                        run_id=run_id,
                        spec_text="Terminal fixture",
                    )

    def test_process_restart_preserves_failed_attempt_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "checkpoints"
            run_id = "retry-process-case"
            setup = textwrap.dedent(
                """
                import json, sys
                from pathlib import Path
                from agentic_dev_workflow.checkpointing import open_sqlite_checkpointer
                from agentic_dev_workflow.mock_workflow import MockScenario, build_mock_graph, create_initial_state
                run_id, data_dir = sys.argv[1], Path(sys.argv[2])
                config = {"configurable": {"thread_id": f"agentic-dev-workflow:{run_id}"}}
                with open_sqlite_checkpointer(data_dir) as saver:
                    graph = build_mock_graph(checkpointer=saver, interrupt_before=["retry_decision"])
                    result = graph.invoke(create_initial_state("Process retry fixture", scenario=MockScenario.EXHAUSTED_RETRIES, run_id=run_id, max_retries=1), config)
                    print(json.dumps({"attempt": result["execution_attempt"], "history": len(result["attempt_history"])}))
                """
            )
            resume = textwrap.dedent(
                """
                import json, sys
                from pathlib import Path
                from agentic_dev_workflow.checkpointing import open_sqlite_checkpointer
                from agentic_dev_workflow.mock_workflow import MockScenario, build_mock_graph, resume_mock_workflow
                run_id, data_dir = sys.argv[1], Path(sys.argv[2])
                config = {"configurable": {"thread_id": f"agentic-dev-workflow:{run_id}"}}
                with open_sqlite_checkpointer(data_dir, require_existing=True) as saver:
                    result = resume_mock_workflow(build_mock_graph(checkpointer=saver), config, run_id=run_id, spec_text="Process retry fixture", scenario=MockScenario.EXHAUSTED_RETRIES, max_retries=1)
                    print(json.dumps({"status": result["final_status"], "attempt": result["execution_attempt"], "retries": result["repair_retries_used"], "history": len(result["attempt_history"])}))
                """
            )
            first = subprocess.run(
                [sys.executable, "-c", setup, run_id, str(data_dir)],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
                shell=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(json.loads(first.stdout), {"attempt": 1, "history": 1})
            second = subprocess.run(
                [sys.executable, "-c", resume, run_id, str(data_dir)],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
                shell=False,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(
                json.loads(second.stdout),
                {"status": "failed", "attempt": 2, "retries": 1, "history": 2},
            )


if __name__ == "__main__":
    unittest.main()
