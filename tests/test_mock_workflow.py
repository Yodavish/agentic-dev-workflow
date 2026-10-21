import unittest

from agentic_dev_workflow.mock_workflow import (
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


if __name__ == "__main__":
    unittest.main()
