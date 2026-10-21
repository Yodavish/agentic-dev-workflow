import unittest
from copy import deepcopy
from types import SimpleNamespace

from agentic_dev_workflow.mock_workflow import (
    CHECKPOINT_FORMAT_VERSION,
    MAX_RETRIES,
    MockScenario,
    build_mock_graph,
    create_initial_state,
    resume_mock_workflow,
    run_mock_workflow,
    validate_resume_identity,
)


SPEC = "Add a small synthetic status badge."


class MockWorkflowTests(unittest.TestCase):
    def _paused_in_memory_graph(self):
        from langgraph.checkpoint.memory import InMemorySaver

        saver = InMemorySaver()
        run_id = "validation-case"
        config = {"configurable": {"thread_id": f"agentic-dev-workflow:{run_id}"}}
        graph = build_mock_graph(checkpointer=saver, interrupt_before=["mock_executor"])
        graph.invoke(create_initial_state(SPEC, run_id=run_id, max_retries=1), config)
        return graph, config, run_id

    def test_immediate_success_is_simulated_and_attempts_once(self) -> None:
        result = run_mock_workflow(SPEC, scenario=MockScenario.SUCCESS, max_retries=2)
        self.assertEqual(result["final_status"], "success")
        self.assertEqual(result["execution_attempt"], 1)
        self.assertEqual(result["repair_retries_used"], 0)
        self.assertTrue(result["simulation_mode"])

    def test_retry_success_records_reason_and_two_attempts(self) -> None:
        result = run_mock_workflow(SPEC, scenario=MockScenario.RETRY_SUCCESS, max_retries=1)
        self.assertEqual(result["final_status"], "success")
        self.assertEqual(result["execution_attempt"], 2)
        self.assertEqual(result["repair_retries_used"], 1)
        self.assertTrue(result["retry_reason"])
        self.assertEqual(len(result["attempt_history"]), 2)

    def test_exhausted_retries_stops_at_initial_plus_retries(self) -> None:
        result = run_mock_workflow(
            SPEC, scenario=MockScenario.EXHAUSTED_RETRIES, max_retries=2
        )
        self.assertEqual(result["final_status"], "failed")
        self.assertEqual(result["execution_attempt"], 3)
        self.assertEqual(result["repair_retries_used"], 2)
        self.assertEqual(len(result["errors"]), 3)

    def test_zero_retries_allows_only_initial_attempt(self) -> None:
        result = run_mock_workflow(
            SPEC, scenario=MockScenario.RETRY_SUCCESS, max_retries=0
        )
        self.assertEqual(result["final_status"], "failed")
        self.assertEqual(result["execution_attempt"], 1)
        self.assertEqual(result["repair_retries_used"], 0)

    def test_retry_limit_is_two_and_rejects_higher_values(self) -> None:
        self.assertEqual(MAX_RETRIES, 2)
        with self.assertRaises(ValueError):
            create_initial_state(SPEC, max_retries=3)
        result = run_mock_workflow(
            SPEC, scenario=MockScenario.EXHAUSTED_RETRIES, max_retries=MAX_RETRIES
        )
        self.assertEqual(result["execution_attempt"], 3)

    def test_checkpoint_can_resume_before_executor_without_replanning(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        state = create_initial_state(SPEC, run_id="resume-case", max_retries=1)
        config = {"configurable": {"thread_id": "agentic-dev-workflow:resume-case"}}
        graph = build_mock_graph(checkpointer=InMemorySaver(), interrupt_before=["mock_executor"])
        paused = graph.invoke(state, config)
        checkpoint = graph.get_state(config)
        self.assertEqual(checkpoint.values["plan"], paused["plan"])
        self.assertEqual(checkpoint.next, ("mock_executor",))
        validate_resume_identity(checkpoint.values, run_id="resume-case", spec_text=SPEC)
        resumed = resume_mock_workflow(graph, config, run_id="resume-case", spec_text=SPEC)
        self.assertEqual(resumed["final_status"], "success")
        self.assertEqual(resumed["execution_attempt"], 1)
        self.assertEqual(resumed["plan"], paused["plan"])

    def test_resume_rejects_checkpoint_run_id_or_specification_mismatch(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        graph = build_mock_graph(checkpointer=InMemorySaver(), interrupt_before=["mock_executor"])
        config = {"configurable": {"thread_id": "agentic-dev-workflow:original"}}
        graph.invoke(create_initial_state(SPEC, run_id="original"), config)
        with self.assertRaises(ValueError):
            resume_mock_workflow(graph, config, run_id="different", spec_text=SPEC)
        with self.assertRaises(ValueError):
            resume_mock_workflow(graph, config, run_id="original", spec_text="changed")

    def test_resume_rejects_scenario_retry_limit_and_format_mismatches(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        for mutate, resume_values in (
            (lambda state: None, {"scenario": MockScenario.RETRY_SUCCESS, "max_retries": 1}),
            (lambda state: None, {"scenario": MockScenario.SUCCESS, "max_retries": 0}),
            (
                lambda state: state.__setitem__("checkpoint_format_version", CHECKPOINT_FORMAT_VERSION + 1),
                {"scenario": MockScenario.SUCCESS, "max_retries": 1},
            ),
        ):
            with self.subTest(resume_values=resume_values):
                saver = InMemorySaver()
                state = create_initial_state(SPEC, run_id="mismatch-case", max_retries=1)
                mutate(state)
                config = {
                    "configurable": {"thread_id": "agentic-dev-workflow:mismatch-case"}
                }
                graph = build_mock_graph(checkpointer=saver, interrupt_before=["mock_executor"])
                graph.invoke(state, config)
                with self.assertRaises(ValueError):
                    resume_mock_workflow(
                        graph,
                        config,
                        run_id="mismatch-case",
                        spec_text=SPEC,
                        **resume_values,
                    )

    def test_resume_rejects_invalid_persisted_state_before_running_nodes(self) -> None:
        mutations = {
            "inconsistent attempt history": {"execution_attempt": 1},
            "invalid attempt type": {"execution_attempt": True},
            "missing planner output": {"plan": None},
            "invalid continuation status": {"final_status": "success"},
            "invalid status type": {"final_status": []},
            "invalid execution status type": {"execution_status": []},
            "invalid history item": {"attempt_history": [{"attempt": 1}]},
        }
        for label, update in mutations.items():
            with self.subTest(case=label):
                graph, config, run_id = self._paused_in_memory_graph()
                graph.update_state(config, update)
                before = graph.get_state(config)
                saved_values = deepcopy(before.values)
                saved_next = before.next

                with self.assertRaises(ValueError):
                    resume_mock_workflow(
                        graph,
                        config,
                        run_id=run_id,
                        spec_text=SPEC,
                        max_retries=1,
                    )

                after = graph.get_state(config)
                self.assertEqual(after.values, saved_values)
                self.assertEqual(after.next, saved_next)

    def test_attempt_history_index_requires_a_non_boolean_integer(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        for invalid_attempt in (1.0, True):
            with self.subTest(attempt=invalid_attempt):
                run_id = f"history-index-{type(invalid_attempt).__name__}"
                config = {
                    "configurable": {"thread_id": f"agentic-dev-workflow:{run_id}"}
                }
                graph = build_mock_graph(
                    checkpointer=InMemorySaver(),
                    interrupt_before=["retry_decision"],
                )
                graph.invoke(
                    create_initial_state(
                        SPEC,
                        scenario=MockScenario.EXHAUSTED_RETRIES,
                        run_id=run_id,
                        max_retries=1,
                    ),
                    config,
                )
                graph.update_state(
                    config,
                    {
                        "attempt_history": [
                            {"attempt": invalid_attempt, "verification": "failed"}
                        ]
                    },
                )
                before = graph.get_state(config)
                saved_values = deepcopy(before.values)
                saved_next = before.next

                with self.assertRaises(ValueError):
                    resume_mock_workflow(
                        graph,
                        config,
                        run_id=run_id,
                        spec_text=SPEC,
                        scenario=MockScenario.EXHAUSTED_RETRIES,
                        max_retries=1,
                    )

                after = graph.get_state(config)
                self.assertEqual(after.values, saved_values)
                self.assertEqual(after.next, saved_next)

        run_id = "history-index-valid-int"
        config = {"configurable": {"thread_id": f"agentic-dev-workflow:{run_id}"}}
        saver = InMemorySaver()
        graph = build_mock_graph(checkpointer=saver, interrupt_before=["retry_decision"])
        graph.invoke(
            create_initial_state(
                SPEC,
                scenario=MockScenario.EXHAUSTED_RETRIES,
                run_id=run_id,
                max_retries=1,
            ),
            config,
        )
        result = resume_mock_workflow(
            build_mock_graph(checkpointer=saver),
            config,
            run_id=run_id,
            spec_text=SPEC,
            scenario=MockScenario.EXHAUSTED_RETRIES,
            max_retries=1,
        )
        self.assertEqual(result["final_status"], "failed")
        self.assertEqual(result["execution_attempt"], 2)

    def test_resume_rejects_missing_field_and_incompatible_continuation(self) -> None:
        graph, config, run_id = self._paused_in_memory_graph()
        saved = graph.get_state(config)

        class AlteredCheckpointView:
            def __init__(self, values, next_nodes):
                self.values = values
                self.next = next_nodes

            def get_state(self, _config):
                return self

            def invoke(self, *_args, **_kwargs):
                raise AssertionError("resume invoked a node before validation")

        missing_values = dict(saved.values)
        del missing_values["spec_hash"]
        for values, next_nodes in (
            (missing_values, saved.next),
            (dict(saved.values), ("verifier",)),
        ):
            with self.subTest(next_nodes=next_nodes, missing_hash="spec_hash" not in values):
                with self.assertRaises(ValueError):
                    resume_mock_workflow(
                        AlteredCheckpointView(values, next_nodes),
                        config,
                        run_id=run_id,
                        spec_text=SPEC,
                        max_retries=1,
                    )
        after = graph.get_state(config)
        self.assertEqual(after.values, saved.values)
        self.assertEqual(after.next, saved.next)

    def test_resume_rejects_unsupported_checkpoint_format_without_override(self) -> None:
        graph, config, run_id = self._paused_in_memory_graph()
        graph.update_state(
            config,
            {"checkpoint_format_version": CHECKPOINT_FORMAT_VERSION + 1},
        )
        with self.assertRaisesRegex(ValueError, "format version"):
            resume_mock_workflow(
                graph,
                config,
                run_id=run_id,
                spec_text=SPEC,
                max_retries=1,
            )

        with self.assertRaises(TypeError):
            resume_mock_workflow(
                graph,
                config,
                run_id=run_id,
                spec_text=SPEC,
                max_retries=1,
                checkpoint_format_version=CHECKPOINT_FORMAT_VERSION + 1,
            )

    def test_resume_rejects_missing_and_terminal_checkpoints(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        saver = InMemorySaver()
        config = {"configurable": {"thread_id": "agentic-dev-workflow:missing"}}
        graph = build_mock_graph(checkpointer=saver)
        with self.assertRaises(ValueError):
            resume_mock_workflow(graph, config, run_id="missing", spec_text=SPEC)

        terminal_config = {
            "configurable": {"thread_id": "agentic-dev-workflow:terminal"}
        }
        graph.invoke(create_initial_state(SPEC, run_id="terminal"), terminal_config)
        with self.assertRaises(ValueError):
            resume_mock_workflow(
                graph,
                terminal_config,
                run_id="terminal",
                spec_text=SPEC,
            )

    def test_spans_record_safe_attributes_without_specification(self) -> None:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from agentic_dev_workflow.tracing import create_tracing_session

        exporter = InMemorySpanExporter()
        session = create_tracing_session("http://localhost:4318/v1/traces", exporter=exporter)
        secret_spec = "Synthetic requirement with distinctive private text"
        try:
            run_mock_workflow(secret_spec, run_id="trace-test", tracing_session=session)
            self.assertTrue(session.flush())
            spans = exporter.get_finished_spans()
        finally:
            session.shutdown()
        names = {span.name for span in spans}
        self.assertTrue({"workflow.mock-run", "workflow.planner", "workflow.mock_executor", "workflow.verifier", "workflow.retry_decision"}.issubset(names))
        for span in spans:
            self.assertEqual(span.attributes.get("workflow.run_id"), "trace-test")
            self.assertTrue(span.attributes.get("workflow.mock_mode"))
            self.assertNotIn(secret_spec, str(span.attributes))
            self.assertNotIn("spec_hash", str(span.attributes))


if __name__ == "__main__":
    unittest.main()
