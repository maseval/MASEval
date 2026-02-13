"""Tests for GAIA2 data loading functions.

Tests load_tasks and configure_model_ids functions.
Note: load_tasks requires ARE package which may not be installed.
"""

import pytest

from maseval import Task, TaskQueue


# =============================================================================
# Test Constants
# =============================================================================


@pytest.mark.benchmark
class TestDataLoaderConstants:
    """Tests for data loader constants."""

    def test_hf_dataset_id_is_correct(self):
        """Test HF_DATASET_ID matches expected value."""
        from maseval.benchmark.gaia2.data_loader import HF_DATASET_ID

        assert HF_DATASET_ID == "meta-agents-research-environments/gaia2"

    def test_valid_capabilities_includes_expected(self):
        """Test VALID_CAPABILITIES includes capabilities that exist on HuggingFace."""
        from maseval.benchmark.gaia2.data_loader import VALID_CAPABILITIES

        expected = ["execution", "search", "adaptability", "time", "ambiguity"]
        for cap in expected:
            assert cap in VALID_CAPABILITIES

    def test_valid_capabilities_excludes_nonexistent(self):
        """Test VALID_CAPABILITIES does not include configs absent from HuggingFace."""
        from maseval.benchmark.gaia2.data_loader import VALID_CAPABILITIES

        for cap in ("agent2agent", "noise"):
            assert cap not in VALID_CAPABILITIES

    def test_hf_dataset_revision_is_pinned(self):
        """Test HF_DATASET_REVISION is set for reproducibility."""
        from maseval.benchmark.gaia2.data_loader import HF_DATASET_REVISION

        assert HF_DATASET_REVISION, "HF_DATASET_REVISION must be set for reproducibility"
        assert isinstance(HF_DATASET_REVISION, str)

    def test_valid_splits_includes_validation(self):
        """Test VALID_SPLITS includes validation."""
        from maseval.benchmark.gaia2.data_loader import VALID_SPLITS

        assert "validation" in VALID_SPLITS


# =============================================================================
# Test load_tasks validation
# =============================================================================


@pytest.mark.benchmark
class TestLoadTasksValidation:
    """Tests for load_tasks parameter validation (no ARE needed)."""

    def test_validates_capability(self):
        """Test load_tasks validates capability parameter."""
        from maseval.benchmark.gaia2.data_loader import load_tasks

        with pytest.raises(ValueError, match="Invalid capability"):
            load_tasks(capability="invalid_capability")

    def test_validates_split(self):
        """Test load_tasks validates split parameter."""
        from maseval.benchmark.gaia2.data_loader import load_tasks

        with pytest.raises(ValueError, match="Invalid split"):
            load_tasks(split="invalid_split")


# =============================================================================
# Test configure_model_ids
# =============================================================================


@pytest.mark.benchmark
class TestConfigureModelIds:
    """Tests for configure_model_ids function."""

    def test_sets_evaluator_model_id(self, sample_gaia2_task_queue):
        """Test configure_model_ids sets evaluator model_id."""
        from maseval.benchmark.gaia2.data_loader import configure_model_ids

        configure_model_ids(sample_gaia2_task_queue, evaluator_model_id="gpt-4")

        for task in sample_gaia2_task_queue:
            assert task.evaluation_data.get("model_id") == "gpt-4"

    def test_handles_none_values(self, sample_gaia2_task_queue):
        """Test configure_model_ids handles None values."""
        from maseval.benchmark.gaia2.data_loader import configure_model_ids

        # Should not raise
        configure_model_ids(sample_gaia2_task_queue)

        # evaluation_data should not have model_id if not set
        for task in sample_gaia2_task_queue:
            # Either not set or set to None
            pass  # Just verify no exception

    def test_modifies_tasks_in_place(self, sample_gaia2_task_queue):
        """Test configure_model_ids modifies tasks in place."""
        from maseval.benchmark.gaia2.data_loader import configure_model_ids

        original_tasks = list(sample_gaia2_task_queue)

        configure_model_ids(sample_gaia2_task_queue, evaluator_model_id="test-model")

        # Tasks should be the same objects, modified
        for original, current in zip(original_tasks, sample_gaia2_task_queue):
            assert original is current
            assert current.evaluation_data.get("model_id") == "test-model"

    def test_raises_on_conflict(self):
        """Test configure_model_ids raises on conflicting model_id."""
        from maseval.benchmark.gaia2.data_loader import configure_model_ids

        task = Task(
            id="test",
            query="test",
            evaluation_data={"model_id": "existing-model"},
        )
        tasks = TaskQueue([task])

        with pytest.raises(ValueError, match="already has evaluator"):
            configure_model_ids(tasks, evaluator_model_id="new-model")

    def test_works_with_list(self):
        """Test configure_model_ids works with plain list."""
        from maseval.benchmark.gaia2.data_loader import configure_model_ids

        tasks = [
            Task(id="1", query="test1"),
            Task(id="2", query="test2"),
        ]

        result = configure_model_ids(tasks, evaluator_model_id="test-model")

        assert result is tasks
        assert all(t.evaluation_data.get("model_id") == "test-model" for t in tasks)

    def test_sets_judge_engine_config(self, sample_gaia2_task_queue):
        """Test configure_model_ids stores judge_engine_config in evaluation_data."""
        from maseval.benchmark.gaia2.data_loader import Gaia2JudgeEngineConfig, configure_model_ids

        config = Gaia2JudgeEngineConfig(provider="openrouter")
        configure_model_ids(sample_gaia2_task_queue, judge_engine_config=config)

        for task in sample_gaia2_task_queue:
            assert task.evaluation_data.get("judge_engine_config") is config

    def test_judge_engine_config_none_does_not_set(self, sample_gaia2_task_queue):
        """Test configure_model_ids with None judge_engine_config does not modify evaluation_data."""
        from maseval.benchmark.gaia2.data_loader import configure_model_ids

        configure_model_ids(sample_gaia2_task_queue)

        for task in sample_gaia2_task_queue:
            assert "judge_engine_config" not in task.evaluation_data

    def test_both_evaluator_and_judge_config(self, sample_gaia2_task_queue):
        """Test configure_model_ids sets both evaluator model_id and judge_engine_config."""
        from maseval.benchmark.gaia2.data_loader import Gaia2JudgeEngineConfig, configure_model_ids

        config = Gaia2JudgeEngineConfig(model_name="gpt-4o", provider="openai")
        configure_model_ids(
            sample_gaia2_task_queue,
            evaluator_model_id="gpt-4o",
            judge_engine_config=config,
        )

        for task in sample_gaia2_task_queue:
            assert task.evaluation_data.get("model_id") == "gpt-4o"
            assert task.evaluation_data.get("judge_engine_config") is config


# =============================================================================
# Test Gaia2JudgeEngineConfig
# =============================================================================


@pytest.mark.benchmark
class TestGaia2JudgeEngineConfig:
    """Tests for Gaia2JudgeEngineConfig dataclass."""

    def test_default_values_match_are(self):
        """Test defaults match ARE's validation/configs.py:28-29."""
        from maseval.benchmark.gaia2.data_loader import Gaia2JudgeEngineConfig

        config = Gaia2JudgeEngineConfig()
        assert config.model_name == "meta-llama/Meta-Llama-3.3-70B-Instruct"
        assert config.provider == "huggingface"
        assert config.endpoint is None

    def test_custom_provider(self):
        """Test custom provider can be set."""
        from maseval.benchmark.gaia2.data_loader import Gaia2JudgeEngineConfig

        config = Gaia2JudgeEngineConfig(provider="openrouter")
        assert config.provider == "openrouter"
        assert config.model_name == "meta-llama/Meta-Llama-3.3-70B-Instruct"

    def test_custom_model_and_provider(self):
        """Test custom model and provider can be set together."""
        from maseval.benchmark.gaia2.data_loader import Gaia2JudgeEngineConfig

        config = Gaia2JudgeEngineConfig(
            model_name="openai/gpt-4o",
            provider="openrouter",
            endpoint="https://openrouter.ai/api/v1",
        )
        assert config.model_name == "openai/gpt-4o"
        assert config.provider == "openrouter"
        assert config.endpoint == "https://openrouter.ai/api/v1"

    def test_importable_from_package(self):
        """Test Gaia2JudgeEngineConfig is importable from the gaia2 package."""
        from maseval.benchmark.gaia2 import Gaia2JudgeEngineConfig

        config = Gaia2JudgeEngineConfig()
        assert config is not None
