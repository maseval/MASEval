"""Data integrity tests for GAIA2 benchmark.

These tests download real data from HuggingFace and validate that the
downloaded tasks are structurally sound.  They are marked ``live`` (network
required) and ``slow`` (HuggingFace download takes time) so that they are
excluded from the default fast test suite.

Run with::

    pytest -m "live and slow" tests/test_benchmarks/test_gaia2/test_data_integrity.py -v
"""

import pytest

from maseval.benchmark.gaia2.data_loader import VALID_CAPABILITIES

pytestmark = [pytest.mark.live, pytest.mark.slow, pytest.mark.benchmark, pytest.mark.gaia2]

# Minimum expected task count across all capabilities.
# The GAIA2 dataset organizes tasks by capability (HuggingFace config name).
# Each capability typically has ~160 tasks in the validation split.
MIN_TOTAL_TASKS = 50


# =============================================================================
# Fixture: load data once for the whole module
# =============================================================================


@pytest.fixture(scope="module")
def gaia2_tasks():
    """Load GAIA2 validation tasks from HuggingFace across all capabilities.

    The HuggingFace dataset uses capability names as config names.
    This fixture loads all tasks via load_tasks(capability=None).
    Requires ``datasets`` and ``are`` packages.
    """
    from maseval.benchmark.gaia2.data_loader import load_tasks

    tasks = load_tasks(split="validation")
    all_tasks = list(tasks)

    assert len(all_tasks) > 0, (
        "No GAIA2 tasks loaded from any capability. Check that the HuggingFace dataset is accessible and has validation data."
    )
    return all_tasks


# =============================================================================
# Dataset Structure
# =============================================================================


class TestGaia2DatasetIntegrity:
    """Validate that the HuggingFace dataset loads and has expected structure."""

    def test_validation_split_loads(self, gaia2_tasks):
        """load_tasks('validation') returns a non-empty collection."""
        assert len(gaia2_tasks) > 0, "GAIA2 validation split returned 0 tasks"

    def test_minimum_task_count(self, gaia2_tasks):
        """Dataset has at least the expected number of tasks."""
        assert len(gaia2_tasks) >= MIN_TOTAL_TASKS, (
            f"GAIA2 validation has {len(gaia2_tasks)} tasks across all capabilities, "
            f"expected >= {MIN_TOTAL_TASKS}. "
            "This may indicate an upstream dataset change."
        )

    def test_required_environment_fields(self, gaia2_tasks):
        """Every task has required fields in environment_data."""
        for task in gaia2_tasks:
            assert "scenario" in task.environment_data, f"Task {task.id} missing 'scenario' in environment_data"
            assert "capability" in task.environment_data, f"Task {task.id} missing 'capability' in environment_data"

    def test_required_evaluation_fields(self, gaia2_tasks):
        """Every task has judge_type in evaluation_data."""
        for task in gaia2_tasks:
            assert "judge_type" in task.evaluation_data, f"Task {task.id} missing 'judge_type' in evaluation_data"

    def test_tasks_have_ids(self, gaia2_tasks):
        """Every task has a non-empty id."""
        for task in gaia2_tasks:
            assert task.id, "Found task with empty/None id"

    def test_scenario_objects_exist(self, gaia2_tasks):
        """Every task's scenario is not None (ARE deserialized it)."""
        for task in gaia2_tasks:
            scenario = task.environment_data.get("scenario")
            assert scenario is not None, f"Task {task.id} has None scenario. ARE's JsonScenarioImporter may have failed to deserialize."

    def test_scenarios_have_serialized_events(self, gaia2_tasks):
        """Every scenario has serialized_events (populated from HF JSON)."""
        for task in gaia2_tasks:
            scenario = task.environment_data.get("scenario")
            serialized_events = getattr(scenario, "serialized_events", None)
            assert serialized_events, f"Task {task.id} has empty serialized_events. The HF JSON data column should contain events."

    def test_scenarios_have_serialized_apps(self, gaia2_tasks):
        """Every scenario has serialized_apps (the universe state)."""
        for task in gaia2_tasks:
            scenario = task.environment_data.get("scenario")
            serialized_apps = getattr(scenario, "serialized_apps", None)
            assert serialized_apps, f"Task {task.id} has empty serialized_apps. The HF JSON data column should contain app definitions."


# =============================================================================
# Capability Coverage
# =============================================================================


class TestGaia2CapabilityCoverage:
    """Validate that all declared capabilities can be loaded from HuggingFace."""

    @pytest.mark.parametrize("capability", list(VALID_CAPABILITIES))
    def test_capability_has_tasks(self, capability):
        """Each VALID_CAPABILITY can be loaded and has tasks on HuggingFace."""
        from maseval.benchmark.gaia2.data_loader import load_tasks

        try:
            tasks = load_tasks(capability=capability, split="validation", limit=3)
        except (ValueError, Exception) as e:
            pytest.fail(
                f"Capability '{capability}' failed to load from HuggingFace: {e}. This capability may have been removed or renamed upstream."
            )

        assert len(tasks) > 0, (
            f"Capability '{capability}' returned 0 tasks from HuggingFace validation split. "
            "This may indicate a dataset regression or schema change."
        )
