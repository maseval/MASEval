"""Tier 2 tests for MMLU benchmark — real data from HuggingFace.

Downloads the real MMLU dataset and validates data integrity, load_tasks
pipeline, and the evaluation pipeline against real task structure.

Run with::

    pytest -m "mmlu and live" tests/test_benchmarks/test_mmlu/test_data_integrity.py -v
"""

from unittest.mock import MagicMock

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.slow,
    pytest.mark.benchmark,
    pytest.mark.mmlu,
]


# ---------------------------------------------------------------------------
# Session-scoped fixtures — download once, reuse across tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ensure_mmlu_data():
    """Download real MMLU data from HuggingFace once per session.

    Uses hf_hub_download with built-in caching — skips download when cached.
    Returns the local path to the downloaded JSON file.
    """
    hf_hub = pytest.importorskip("huggingface_hub")
    local_path = hf_hub.hf_hub_download(
        repo_id="arubique/flattened-MMLU",
        filename="mmlu_prompts_examples.json",
        repo_type="dataset",
    )
    return local_path


@pytest.fixture(scope="session")
def real_mmlu_tasks(ensure_mmlu_data):
    """Load all real MMLU tasks from the downloaded dataset."""
    from maseval.benchmark.mmlu.mmlu import load_tasks

    return load_tasks(data_path=ensure_mmlu_data)


@pytest.fixture(scope="session")
def real_mmlu_tasks_small(ensure_mmlu_data):
    """Load a small subset (50 tasks) for faster integration tests."""
    from maseval.benchmark.mmlu.mmlu import load_tasks

    return load_tasks(data_path=ensure_mmlu_data, limit=50)


# ---------------------------------------------------------------------------
# Data integrity — validate the real dataset
# ---------------------------------------------------------------------------


class TestMMLUDataIntegrity:
    def test_dataset_loads_from_huggingface(self, ensure_mmlu_data):
        """Downloaded file exists and is reachable."""
        from pathlib import Path

        assert Path(ensure_mmlu_data).exists()

    def test_total_task_count(self, real_mmlu_tasks):
        """MMLU full dataset has >14 000 tasks."""
        assert len(real_mmlu_tasks) > 14_000

    def test_task_schema(self, real_mmlu_tasks):
        """Every task has required fields with correct types."""
        for task in real_mmlu_tasks:
            assert isinstance(task.query, str) and len(task.query) > 0
            assert isinstance(task.evaluation_data["gold"], int)
            assert 0 <= task.evaluation_data["gold"] <= 3
            choices = task.environment_data["choices"]
            assert isinstance(choices, list) and len(choices) == 4
            assert isinstance(task.metadata["doc_id"], int)

    def test_gold_answer_distribution(self, real_mmlu_tasks):
        """All four answer indices (0-3) appear as gold answers."""
        golds = {task.evaluation_data["gold"] for task in real_mmlu_tasks}
        assert golds == {0, 1, 2, 3}

    def test_choices_are_abcd(self, real_mmlu_tasks):
        """Every task's choices are [A, B, C, D]."""
        for task in real_mmlu_tasks:
            assert task.environment_data["choices"] == ["A", "B", "C", "D"]

    def test_full_prompt_present(self, real_mmlu_tasks):
        """Every task has a non-empty full_prompt."""
        for task in real_mmlu_tasks:
            assert len(task.environment_data.get("full_prompt", "")) > 0

    def test_doc_ids_unique(self, real_mmlu_tasks):
        """No duplicate doc_id values."""
        doc_ids = [task.metadata["doc_id"] for task in real_mmlu_tasks]
        assert len(doc_ids) == len(set(doc_ids))


# ---------------------------------------------------------------------------
# load_tasks with real data
# ---------------------------------------------------------------------------


class TestMMLULoadTasksWithRealData:
    def test_load_with_limit(self, ensure_mmlu_data):
        from maseval.benchmark.mmlu.mmlu import load_tasks

        tasks = load_tasks(data_path=ensure_mmlu_data, limit=10)
        assert len(tasks) == 10

    def test_load_returns_sequential_queue(self, ensure_mmlu_data):
        from maseval.benchmark.mmlu.mmlu import load_tasks
        from maseval.core.task import SequentialTaskQueue

        tasks = load_tasks(data_path=ensure_mmlu_data, limit=5)
        assert isinstance(tasks, SequentialTaskQueue)

    def test_tasks_have_correct_id_format(self, real_mmlu_tasks_small):
        for i, task in enumerate(real_mmlu_tasks_small):
            assert task.id == f"mmlu_{i}"


# ---------------------------------------------------------------------------
# Real data pipeline — real tasks, no GPU
# ---------------------------------------------------------------------------


class TestMMLURealDataPipeline:
    def test_environment_setup_with_real_task(self, real_mmlu_tasks_small):
        from maseval.benchmark.mmlu.mmlu import MMLUEnvironment

        task = real_mmlu_tasks_small[0]
        task_data = {
            "query": task.query,
            "environment_data": {**task.environment_data, "use_full_prompt": True},
        }
        env = MMLUEnvironment(task_data)
        assert "choices" in env.state
        assert "full_prompt" in env.state
        assert env.get_prompt() == task.environment_data["full_prompt"]

    def test_evaluator_with_real_task(self, real_mmlu_tasks_small):
        from maseval.benchmark.mmlu.mmlu import MMLUEnvironment, MMLUEvaluator

        task = real_mmlu_tasks_small[0]
        task_data = {
            "query": task.query,
            "environment_data": {**task.environment_data, "use_full_prompt": False},
        }
        env = MMLUEnvironment(task_data)
        evaluator = MMLUEvaluator(task, env)
        assert 0 <= evaluator.gold <= 3
        assert evaluator.choices == ["A", "B", "C", "D"]

    def test_evaluate_real_task_correct(self, real_mmlu_tasks_small):
        from maseval.benchmark.mmlu.mmlu import MMLUEnvironment, MMLUEvaluator

        task = real_mmlu_tasks_small[0]
        task_data = {
            "query": task.query,
            "environment_data": {**task.environment_data, "use_full_prompt": False},
        }
        env = MMLUEnvironment(task_data)
        evaluator = MMLUEvaluator(task, env)
        gold_letter = ["A", "B", "C", "D"][evaluator.gold]
        result = evaluator({"messages": []}, final_answer=gold_letter)
        assert result["acc"] == 1.0
        assert result["correct"] is True

    def test_evaluate_real_task_incorrect(self, real_mmlu_tasks_small):
        from maseval.benchmark.mmlu.mmlu import MMLUEnvironment, MMLUEvaluator

        task = real_mmlu_tasks_small[0]
        task_data = {
            "query": task.query,
            "environment_data": {**task.environment_data, "use_full_prompt": False},
        }
        env = MMLUEnvironment(task_data)
        evaluator = MMLUEvaluator(task, env)
        wrong_letter = ["A", "B", "C", "D"][(evaluator.gold + 1) % 4]
        result = evaluator({"messages": []}, final_answer=wrong_letter)
        assert result["acc"] == 0.0
        assert result["correct"] is False

    def test_full_pipeline_single_task_with_mock_scorer(self, real_mmlu_tasks_small):
        """Run the full MMLU pipeline on a real task with a stub scorer (no GPU)."""
        from maseval.benchmark.mmlu.mmlu import MMLUBenchmark, MMLUEnvironment, _ScorerBackedAdapter
        from maseval.core.seeding import DefaultSeedGenerator

        task = real_mmlu_tasks_small[0]
        gold = task.evaluation_data["gold"]

        # Build a concrete benchmark subclass with a stub scorer
        class StubMMLUBenchmark(MMLUBenchmark):
            def __init__(self):
                super().__init__(use_full_prompt=True)
                self._stub_scorer = MagicMock()

            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                adapter = _ScorerBackedAdapter(self._stub_scorer, "stub_agent")
                return [adapter], {"stub_agent": adapter}

            def get_model_adapter(self, model_id, **kwargs):
                raise NotImplementedError

            def run_agents(self, agents, task, environment, query=""):
                env = environment
                prompt = env.get_prompt()
                choices = env.state["choices"]
                # Return fixed logprobs that pick the gold answer
                logprobs = [-5.0] * 4
                logprobs[gold] = -0.1
                agent = agents[0]
                answer = choices[logprobs.index(max(logprobs))]
                agent.record_message({"role": "user", "content": prompt})
                agent.record_message({"role": "assistant", "content": answer, "logprobs": logprobs})
                return answer

        benchmark = StubMMLUBenchmark()

        # Run the pipeline components manually
        seed_gen = DefaultSeedGenerator()
        env = benchmark.setup_environment({}, task, seed_gen)
        assert isinstance(env, MMLUEnvironment)

        agents_list, agents_dict = benchmark.setup_agents({}, env, task, None, seed_gen)
        answer = benchmark.run_agents(agents_list, task, env)
        assert answer == ["A", "B", "C", "D"][gold]

        evaluators = benchmark.setup_evaluators(env, task, agents_list, None, seed_gen)
        traces = {"agents": {name: {"messages": list(a.get_messages())} for name, a in agents_dict.items()}}
        results = benchmark.evaluate(evaluators, agents_dict, answer, traces)
        assert len(results) == 1
        assert results[0]["correct"] is True
        assert results[0]["doc_id"] == task.metadata["doc_id"]
