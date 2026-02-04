"""Tests for GAIA2 data loading functions.

Tests load_tasks and configure_model_ids functions.
Note: load_tasks requires ARE package which may not be installed.
"""

import pytest
from unittest.mock import MagicMock

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
        """Test VALID_CAPABILITIES includes expected capabilities."""
        from maseval.benchmark.gaia2.data_loader import VALID_CAPABILITIES

        expected = ["execution", "search", "adaptability", "time", "ambiguity", "agent2agent", "noise"]
        for cap in expected:
            assert cap in VALID_CAPABILITIES

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


# =============================================================================
# Test _get_scenario_metadata helper
# =============================================================================


@pytest.mark.benchmark
class TestGetScenarioMetadata:
    """Tests for _get_scenario_metadata helper function."""

    def test_extracts_from_dict_metadata(self):
        """Test extraction from dict-style metadata."""
        from maseval.benchmark.gaia2.data_loader import _get_scenario_metadata

        scenario = MagicMock()
        scenario.metadata = {"capability": "execution", "universe_id": "test"}

        assert _get_scenario_metadata(scenario, "capability") == "execution"
        assert _get_scenario_metadata(scenario, "universe_id") == "test"

    def test_returns_default_for_missing_key(self):
        """Test returns default when key not found."""
        from maseval.benchmark.gaia2.data_loader import _get_scenario_metadata

        scenario = MagicMock()
        scenario.metadata = {"other": "value"}

        assert _get_scenario_metadata(scenario, "missing") is None
        assert _get_scenario_metadata(scenario, "missing", "default") == "default"

    def test_handles_none_metadata(self):
        """Test handles None metadata attribute."""
        from maseval.benchmark.gaia2.data_loader import _get_scenario_metadata

        scenario = MagicMock()
        scenario.metadata = None

        assert _get_scenario_metadata(scenario, "any_key") is None
        assert _get_scenario_metadata(scenario, "any_key", "default") == "default"

    def test_handles_object_metadata(self):
        """Test handles object-style metadata with attributes."""
        from maseval.benchmark.gaia2.data_loader import _get_scenario_metadata

        metadata = MagicMock()
        metadata.capability = "search"
        scenario = MagicMock()
        scenario.metadata = metadata

        # Note: dict access will fail, falls back to attribute
        assert _get_scenario_metadata(scenario, "capability") == "search"
