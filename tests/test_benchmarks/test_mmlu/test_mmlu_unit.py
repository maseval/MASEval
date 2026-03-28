"""Unit tests for MMLU benchmark components (Tier 1: offline, no real data).

Tests MMLUEnvironment, MMLUEvaluator, _ScorerBackedAdapter, load_tasks,
compute_benchmark_metrics, MMLUBenchmark, and DefaultMMLUBenchmark.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from maseval.core.history import MessageHistory
from maseval.core.seeding import DefaultSeedGenerator
from maseval.core.task import DISCOQueue, SequentialTaskQueue

from .conftest import make_mmlu_task

pytestmark = pytest.mark.benchmark


# ---------------------------------------------------------------------------
# MMLUEnvironment
# ---------------------------------------------------------------------------


class TestMMLUEnvironment:
    def _make_env(self, *, use_full_prompt: bool = True):
        from maseval.benchmark.mmlu.mmlu import MMLUEnvironment

        task = make_mmlu_task(full_prompt="Full prompt here")
        task_data = {
            "query": task.query,
            "environment_data": {**task.environment_data, "use_full_prompt": use_full_prompt},
        }
        return MMLUEnvironment(task_data)

    def test_setup_state_extracts_fields(self):
        env = self._make_env()
        for key in ("query", "choices", "full_prompt", "use_full_prompt"):
            assert key in env.state

    def test_create_tools_returns_empty_dict(self):
        env = self._make_env()
        assert env.tools == {}

    def test_get_prompt_full_prompt_true(self):
        env = self._make_env(use_full_prompt=True)
        assert env.get_prompt() == "Full prompt here"

    def test_get_prompt_full_prompt_false(self):
        env = self._make_env(use_full_prompt=False)
        assert env.get_prompt() == "What is 2+2?"


# ---------------------------------------------------------------------------
# MMLUEvaluator._parse_answer
# ---------------------------------------------------------------------------


class TestMMLUEvaluatorParseAnswer:
    @pytest.fixture
    def evaluator(self, sample_mmlu_task):
        from maseval.benchmark.mmlu.mmlu import MMLUEnvironment, MMLUEvaluator

        task_data = {
            "query": sample_mmlu_task.query,
            "environment_data": {**sample_mmlu_task.environment_data, "use_full_prompt": False},
        }
        env = MMLUEnvironment(task_data)
        return MMLUEvaluator(sample_mmlu_task, env)

    @pytest.mark.parametrize(
        "response, expected",
        [
            # Direct letter
            ("A", 0),
            ("B", 1),
            ("C", 2),
            ("D", 3),
            # With period
            ("A.", 0),
            ("B.", 1),
            # Sentence patterns
            ("The answer is A", 0),
            ("ANSWER IS C", 2),
            ("ANSWER: D", 3),
            # Last character
            ("I think it's B", 1),
            # Empty / unparseable
            ("", None),
            ("random text", None),
        ],
    )
    def test_parse_answer(self, evaluator, response, expected):
        assert evaluator._parse_answer(response) == expected


# ---------------------------------------------------------------------------
# MMLUEvaluator.__call__
# ---------------------------------------------------------------------------


class TestMMLUEvaluatorCall:
    @pytest.fixture
    def evaluator(self, sample_mmlu_task):
        from maseval.benchmark.mmlu.mmlu import MMLUEnvironment, MMLUEvaluator

        task_data = {
            "query": sample_mmlu_task.query,
            "environment_data": {**sample_mmlu_task.environment_data, "use_full_prompt": False},
        }
        env = MMLUEnvironment(task_data)
        return MMLUEvaluator(sample_mmlu_task, env)

    def test_correct_answer_scores_1(self, evaluator):
        result = evaluator({"messages": []}, final_answer="A")
        assert result["acc"] == 1.0
        assert result["correct"] is True
        assert result["predicted"] == 0

    def test_incorrect_answer_scores_0(self, evaluator):
        result = evaluator({"messages": []}, final_answer="B")
        assert result["acc"] == 0.0
        assert result["correct"] is False

    def test_extracts_logprobs_from_traces(self, evaluator):
        traces = {"messages": [{"role": "assistant", "content": "A", "logprobs": [-1.0, -2.0, -3.0, -4.0]}]}
        result = evaluator(traces, final_answer="A")
        assert result["logprobs"] == [-1.0, -2.0, -3.0, -4.0]

    def test_filter_traces_with_agents(self, evaluator):
        traces = {"agents": {"agent1": {"messages": [{"role": "user", "content": "hi"}]}}}
        filtered = evaluator.filter_traces(traces)
        assert filtered["messages"] == [{"role": "user", "content": "hi"}]

    def test_filter_traces_empty(self, evaluator):
        assert evaluator.filter_traces({})["messages"] == []


# ---------------------------------------------------------------------------
# _ScorerBackedAdapter
# ---------------------------------------------------------------------------


class TestScorerBackedAdapter:
    def test_record_and_get_messages(self):
        from maseval.benchmark.mmlu.mmlu import _ScorerBackedAdapter

        adapter = _ScorerBackedAdapter(scorer=MagicMock(), name="test")
        adapter.record_message({"role": "user", "content": "hello"})
        adapter.record_message({"role": "assistant", "content": "world"})
        history = adapter.get_messages()
        assert isinstance(history, MessageHistory)
        assert len(history) == 2

    def test_run_agent_raises(self):
        from maseval.benchmark.mmlu.mmlu import _ScorerBackedAdapter

        adapter = _ScorerBackedAdapter(scorer=MagicMock(), name="test")
        with pytest.raises(NotImplementedError):
            adapter._run_agent("query")


# ---------------------------------------------------------------------------
# load_tasks
# ---------------------------------------------------------------------------


class TestLoadTasks:
    def test_basic_load(self, mmlu_json_path):
        from maseval.benchmark.mmlu.mmlu import load_tasks

        tasks = load_tasks(data_path=mmlu_json_path)
        assert isinstance(tasks, SequentialTaskQueue)
        assert len(tasks) == 5

    def test_with_limit(self, mmlu_json_path):
        from maseval.benchmark.mmlu.mmlu import load_tasks

        tasks = load_tasks(data_path=mmlu_json_path, limit=2)
        assert len(tasks) == 2

    def test_uses_example_field(self, tmp_path):
        from maseval.benchmark.mmlu.mmlu import load_tasks

        item = {"example": "Example question", "gold": 0, "choices": ["A", "B", "C", "D"]}
        path = tmp_path / "data.json"
        path.write_text(json.dumps([item]))
        tasks = load_tasks(data_path=path)
        assert tasks[0].query == "Example question"

    def test_missing_gold_raises(self, tmp_path):
        from maseval.benchmark.mmlu.mmlu import load_tasks

        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"query": "Q", "choices": ["A", "B", "C", "D"]}]))
        with pytest.raises(ValueError, match="gold"):
            load_tasks(data_path=path)

    def test_missing_choices_raises(self, tmp_path):
        from maseval.benchmark.mmlu.mmlu import load_tasks

        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"query": "Q", "gold": 0}]))
        with pytest.raises(ValueError, match="choices"):
            load_tasks(data_path=path)

    def test_missing_query_and_example_raises(self, tmp_path):
        from maseval.benchmark.mmlu.mmlu import load_tasks

        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"gold": 0, "choices": ["A", "B", "C", "D"]}]))
        with pytest.raises(ValueError, match="neither"):
            load_tasks(data_path=path)

    def test_with_anchor_points_path(self, mmlu_json_path, tmp_path):
        from maseval.benchmark.mmlu.mmlu import load_tasks

        # Write a JSON anchor points file with indices [0, 2]
        anchor_path = tmp_path / "anchors.json"
        anchor_path.write_text(json.dumps([0, 2]))
        tasks = load_tasks(data_path=mmlu_json_path, anchor_points_path=anchor_path)
        assert isinstance(tasks, DISCOQueue)
        assert len(tasks) == 2


# ---------------------------------------------------------------------------
# compute_benchmark_metrics
# ---------------------------------------------------------------------------


def _success_result(acc: float = 1.0, acc_norm: float = 1.0, correct: bool = True):
    return {
        "status": "success",
        "eval": [{"acc": acc, "acc_norm": acc_norm, "correct": correct}],
    }


def _error_result():
    return {"status": "error", "eval": None}


class TestComputeBenchmarkMetrics:
    @pytest.mark.parametrize(
        "results, expected_acc, expected_correct",
        [
            ([], 0.0, 0),
            ([_success_result(acc=1.0)], 1.0, 1),
            ([_success_result(acc=1.0), _success_result(acc=0.0, correct=False)], 0.5, 1),
            ([_error_result()], 0.0, 0),
        ],
        ids=["empty", "all_correct", "mixed", "error_skipped"],
    )
    def test_compute_metrics(self, results, expected_acc, expected_correct):
        from maseval.benchmark.mmlu.mmlu import compute_benchmark_metrics

        metrics = compute_benchmark_metrics(results)
        assert metrics["acc"] == pytest.approx(expected_acc)
        if results:
            assert metrics["correct_count"] == expected_correct


# ---------------------------------------------------------------------------
# MMLUBenchmark (base class)
# ---------------------------------------------------------------------------


class TestMMLUBenchmarkBase:
    @pytest.fixture
    def benchmark(self):
        from maseval.benchmark.mmlu.mmlu import MMLUBenchmark

        class ConcreteMMLU(MMLUBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                adapter = MagicMock()
                adapter.name = "mock_agent"
                adapter.run.return_value = "A"
                adapter.get_messages.return_value = MessageHistory([])
                adapter.gather_traces.return_value = {}
                adapter.gather_config.return_value = {}
                adapter.callbacks = []
                return [adapter], {"mock_agent": adapter}

            def get_model_adapter(self, model_id, **kwargs):
                return MagicMock()

        return ConcreteMMLU(use_full_prompt=False)

    def test_setup_environment_returns_mmlu_env(self, benchmark, sample_mmlu_task):
        from maseval.benchmark.mmlu.mmlu import MMLUEnvironment

        env = benchmark.setup_environment({}, sample_mmlu_task, DefaultSeedGenerator())
        assert isinstance(env, MMLUEnvironment)
        assert "choices" in env.state

    def test_setup_evaluators_returns_mmlu_evaluator(self, benchmark, sample_mmlu_task):
        from maseval.benchmark.mmlu.mmlu import MMLUEnvironment, MMLUEvaluator

        task_data = {
            "query": sample_mmlu_task.query,
            "environment_data": {**sample_mmlu_task.environment_data, "use_full_prompt": False},
        }
        env = MMLUEnvironment(task_data)
        evaluators = benchmark.setup_evaluators(env, sample_mmlu_task, [], None, DefaultSeedGenerator())
        assert len(evaluators) == 1
        assert isinstance(evaluators[0], MMLUEvaluator)

    def test_run_agents_calls_agent_run(self, benchmark, sample_mmlu_task):
        from maseval.benchmark.mmlu.mmlu import MMLUEnvironment

        task_data = {
            "query": sample_mmlu_task.query,
            "environment_data": {**sample_mmlu_task.environment_data, "use_full_prompt": False},
        }
        env = MMLUEnvironment(task_data)
        agents, _ = benchmark.setup_agents({}, env, sample_mmlu_task, None, DefaultSeedGenerator())
        result = benchmark.run_agents(agents, sample_mmlu_task, env, query="")
        assert result == "A"
        agents[0].run.assert_called_once()

    def test_evaluate_delegates_to_evaluators(self, benchmark, sample_mmlu_task):
        from maseval.benchmark.mmlu.mmlu import MMLUEnvironment

        task_data = {
            "query": sample_mmlu_task.query,
            "environment_data": {**sample_mmlu_task.environment_data, "use_full_prompt": False},
        }
        env = MMLUEnvironment(task_data)
        evaluators = benchmark.setup_evaluators(env, sample_mmlu_task, [], None, DefaultSeedGenerator())
        results = benchmark.evaluate(evaluators, {}, "A", {"agents": {}})
        assert len(results) == 1
        assert results[0]["correct"] is True


# ---------------------------------------------------------------------------
# DefaultMMLUBenchmark
# ---------------------------------------------------------------------------


class TestDefaultMMLUBenchmark:
    @pytest.fixture
    def mock_scorer(self):
        scorer = MagicMock()
        scorer.model_id = "test-model"
        scorer.loglikelihood_choices.return_value = [-1.0, -2.0, -3.0, -4.0]
        scorer.gather_traces.return_value = {}
        scorer.gather_config.return_value = {"model_id": "test-model"}
        return scorer

    @pytest.fixture
    def benchmark(self, mock_scorer):
        with patch(
            "maseval.interface.inference.huggingface_scorer.HuggingFaceModelScorer",
            return_value=mock_scorer,
        ):
            from maseval.benchmark.mmlu.mmlu import DefaultMMLUBenchmark

            return DefaultMMLUBenchmark(model_id="test-model", device="cpu")

    def test_setup_agents_returns_scorer_backed_adapter(self, benchmark, sample_mmlu_task):
        from maseval.benchmark.mmlu.mmlu import MMLUEnvironment, _ScorerBackedAdapter

        task_data = {
            "query": sample_mmlu_task.query,
            "environment_data": {**sample_mmlu_task.environment_data, "use_full_prompt": True},
        }
        env = MMLUEnvironment(task_data)
        agents, agents_dict = benchmark.setup_agents({}, env, sample_mmlu_task, None, DefaultSeedGenerator())
        assert len(agents) == 1
        assert isinstance(agents[0], _ScorerBackedAdapter)

    def test_run_agents_with_precomputed_logprobs(self, benchmark, sample_mmlu_task, mock_scorer):
        from maseval.benchmark.mmlu.mmlu import MMLUEnvironment, _ScorerBackedAdapter

        task_data = {
            "query": sample_mmlu_task.query,
            "environment_data": {**sample_mmlu_task.environment_data, "use_full_prompt": True},
        }
        env = MMLUEnvironment(task_data)
        adapter = _ScorerBackedAdapter(mock_scorer, "agent")

        benchmark._precomputed_logprobs = {0: [-0.5, -1.5, -2.5, -3.5]}
        answer = benchmark.run_agents([adapter], sample_mmlu_task, env)
        assert answer == "A"  # index 0 has highest logprob
        mock_scorer.loglikelihood_choices.assert_not_called()

    def test_run_agents_without_precomputed(self, benchmark, sample_mmlu_task, mock_scorer):
        from maseval.benchmark.mmlu.mmlu import MMLUEnvironment, _ScorerBackedAdapter

        task_data = {
            "query": sample_mmlu_task.query,
            "environment_data": {**sample_mmlu_task.environment_data, "use_full_prompt": True},
        }
        env = MMLUEnvironment(task_data)
        adapter = _ScorerBackedAdapter(mock_scorer, "agent")

        benchmark._precomputed_logprobs = None
        answer = benchmark.run_agents([adapter], sample_mmlu_task, env)
        assert answer == "A"  # index 0 has highest logprob (-1.0)
        mock_scorer.loglikelihood_choices.assert_called_once()

    def test_get_model_adapter_raises(self, benchmark):
        with pytest.raises(NotImplementedError):
            benchmark.get_model_adapter("test-model")

    def test_precompute_all_logprobs_lmeval(self, benchmark, sample_mmlu_tasks, mock_scorer):
        """Test precompute_all_logprobs_lmeval with mocked lm-evaluation-harness."""
        import sys
        from types import ModuleType

        # Build mock lm-evaluation-harness modules
        mock_hflm_mod = ModuleType("lm_eval.models.huggingface")
        mock_instance_mod = ModuleType("lm_eval.api.instance")
        mock_lm_top = ModuleType("lm_eval")
        mock_lm_models = ModuleType("lm_eval.models")
        mock_lm_api = ModuleType("lm_eval.api")

        # FakeInstance stores keyword args as attributes
        class FakeInstance:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        setattr(mock_instance_mod, "Instance", FakeInstance)

        # FakeHFLM returns (logprob, is_greedy) tuples
        class FakeHFLM:
            def __init__(self, **kwargs):
                pass

            def loglikelihood(self, instances):
                return [(-float(i), True) for i in range(len(instances))]

        setattr(mock_hflm_mod, "HFLM", FakeHFLM)

        tasks = sample_mmlu_tasks
        with patch.dict(
            sys.modules,
            {
                "lm_eval": mock_lm_top,
                "lm_eval.models": mock_lm_models,
                "lm_eval.models.huggingface": mock_hflm_mod,
                "lm_eval.api": mock_lm_api,
                "lm_eval.api.instance": mock_instance_mod,
            },
        ):
            doc_logprobs = benchmark.precompute_all_logprobs_lmeval(tasks)

        # 3 tasks, each with 4 choices
        assert len(doc_logprobs) == 3
        for doc_id in [0, 1, 2]:
            assert doc_id in doc_logprobs
            assert len(doc_logprobs[doc_id]) == 4

        # Verify stored for later use
        assert benchmark._precomputed_logprobs is doc_logprobs
