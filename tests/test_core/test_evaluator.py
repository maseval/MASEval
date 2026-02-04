"""Test Evaluator functionality.

These tests verify that Evaluator receives correct inputs and produces results.
"""

import pytest
from maseval import TaskQueue


@pytest.mark.core
class TestEvaluator:
    """Tests for Evaluator integration."""

    def test_evaluator_receives_message_history(self):
        """Test that evaluator receives message history in traces."""
        from conftest import DummyBenchmark
        from maseval import Evaluator

        received_traces = []
        received_final_answers = []

        class TracingEvaluator(Evaluator):
            def __init__(self, task, environment, user=None):
                super().__init__(task, environment, user)

            def filter_traces(self, traces):
                return traces

            def __call__(self, traces, final_answer=None):
                received_traces.append(traces)
                received_final_answers.append(final_answer)
                return {"score": 1.0}

        class TestBenchmark(DummyBenchmark):
            def setup_evaluators(self, environment, task, agents, user, seed_generator):
                return [TracingEvaluator(task, environment, user)]

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = TestBenchmark()

        benchmark.run(tasks, agent_data={"model": "test"})

        assert len(received_traces) == 1
        assert isinstance(received_traces[0], dict)
        assert "agents" in received_traces[0]
        # Verify message history is accessible within traces
        assert len(received_final_answers) == 1

    def test_evaluator_receives_agents_dict(self):
        """Test that evaluate() receives agents dictionary."""
        from conftest import DummyBenchmark

        class TestBenchmark(DummyBenchmark):
            def evaluate(self, evaluators, agents, final_answer, traces):
                # Verify agents is a dict
                assert isinstance(agents, dict)
                assert "test_agent" in agents
                return super().evaluate(evaluators, agents, final_answer, traces)

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = TestBenchmark()

        benchmark.run(tasks, agent_data={"model": "test"})

    def test_evaluator_receives_final_answer(self):
        """Test that evaluate() receives the final answer from agents."""
        from conftest import DummyBenchmark

        received_answers = []

        class TestBenchmark(DummyBenchmark):
            def evaluate(self, evaluators, agents, final_answer, traces):
                received_answers.append(final_answer)
                return super().evaluate(evaluators, agents, final_answer, traces)

        tasks = TaskQueue.from_list([{"query": "My test query", "environment_data": {}}])
        benchmark = TestBenchmark()

        benchmark.run(tasks, agent_data={"model": "test"})

        assert len(received_answers) == 1
        assert "Response to: My test query" in received_answers[0]

    def test_evaluator_receives_traces(self):
        """Test that evaluate() receives execution traces."""
        from conftest import DummyBenchmark

        received_traces = []

        class TestBenchmark(DummyBenchmark):
            def evaluate(self, evaluators, agents, final_answer, traces):
                received_traces.append(traces)
                return super().evaluate(evaluators, agents, final_answer, traces)

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = TestBenchmark()

        benchmark.run(tasks, agent_data={"model": "test"})

        assert len(received_traces) == 1
        traces = received_traces[0]
        assert "agents" in traces
        assert "environment" in traces

    def test_multiple_evaluators_all_run(self):
        """Test that multiple evaluators all execute."""
        from conftest import DummyBenchmark
        from maseval import Evaluator

        call_counts = {"eval1": 0, "eval2": 0}

        class Evaluator1(Evaluator):
            def __init__(self, task, environment, user=None):
                super().__init__(task, environment, user)

            def filter_traces(self, traces):
                return traces

            def __call__(self, traces, final_answer=None):
                call_counts["eval1"] += 1
                return {"score": 1.0, "evaluator": "eval1"}

        class Evaluator2(Evaluator):
            def __init__(self, task, environment, user=None):
                super().__init__(task, environment, user)

            def filter_traces(self, traces):
                return traces

            def __call__(self, traces, final_answer=None):
                call_counts["eval2"] += 1
                return {"score": 0.8, "evaluator": "eval2"}

        class TestBenchmark(DummyBenchmark):
            def setup_evaluators(self, environment, task, agents, user, seed_generator):
                return [
                    Evaluator1(task, environment, user),
                    Evaluator2(task, environment, user),
                ]

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = TestBenchmark()

        benchmark.run(tasks, agent_data={"model": "test"})

        assert call_counts["eval1"] == 1
        assert call_counts["eval2"] == 1

    def test_evaluator_results_in_report(self):
        """Test that evaluator results appear in the final report."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        assert len(reports) == 1
        report = reports[0]

        assert "eval" in report
        assert isinstance(report["eval"], list)
        assert len(report["eval"]) > 0
        assert report["eval"][0]["score"] == 1.0
