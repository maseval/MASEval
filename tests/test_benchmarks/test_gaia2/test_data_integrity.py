"""Data integrity tests for GAIA2 benchmark.

These tests download real data from HuggingFace and validate that the
downloaded tasks are structurally sound.  They are marked ``live`` (network
required) and ``slow`` (HuggingFace download takes time) so that they are
excluded from the default fast test suite.

Run with::

    pytest -m "live and slow" tests/test_benchmarks/test_gaia2/test_data_integrity.py -v
"""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.slow, pytest.mark.benchmark, pytest.mark.gaia2]

# Minimum expected task count across all capabilities.
# The GAIA2 dataset organizes tasks by capability (HuggingFace config name).
# Each capability typically has 20-30+ tasks in the validation split.
MIN_TOTAL_TASKS = 50


# =============================================================================
# Fixture: load data once for the whole module
# =============================================================================


@pytest.fixture(scope="module")
def gaia2_tasks():
    """Load GAIA2 validation tasks from HuggingFace across all capabilities.

    The HuggingFace dataset uses capability names as config names (not "validation").
    This fixture loads tasks from each capability individually and combines them.
    Requires ``datasets`` and ``are`` packages.
    """
    pytest.importorskip("datasets", reason="HuggingFace datasets library required")
    pytest.importorskip("are", reason="ARE (meta-agents-research-environments) required")

    from maseval.benchmark.gaia2.data_loader import VALID_CAPABILITIES, load_tasks

    all_tasks = []
    for cap in VALID_CAPABILITIES:
        try:
            tasks = load_tasks(capability=cap, split="validation")
            all_tasks.extend(list(tasks))
        except (ValueError, Exception):
            # Capability may not be available on HuggingFace yet
            pass

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
        """Every task has oracle_events in evaluation_data."""
        for task in gaia2_tasks:
            assert "oracle_events" in task.evaluation_data, f"Task {task.id} missing 'oracle_events' in evaluation_data"

    def test_oracle_events_non_empty(self, gaia2_tasks):
        """Every task has at least one oracle event."""
        for task in gaia2_tasks:
            oracle_events = task.evaluation_data.get("oracle_events", [])
            assert len(oracle_events) > 0, f"Task {task.id} has 0 oracle events. Oracle events are required for GAIA2 evaluation."

    def test_tasks_have_queries(self, gaia2_tasks):
        """Every task has a non-empty query (task instruction)."""
        for task in gaia2_tasks:
            assert task.query, f"Task {task.id} has empty query"

    def test_tasks_have_ids(self, gaia2_tasks):
        """Every task has a non-empty id."""
        for task in gaia2_tasks:
            assert task.id, "Found task with empty/None id"

    def test_scenario_objects_exist(self, gaia2_tasks):
        """Every task's scenario is not None (ARE deserialized it)."""
        for task in gaia2_tasks:
            scenario = task.environment_data.get("scenario")
            assert scenario is not None, f"Task {task.id} has None scenario. ARE's JsonScenarioImporter may have failed to deserialize."


# =============================================================================
# Capability Coverage
# =============================================================================


class TestGaia2CapabilityCoverage:
    """Validate that all declared capabilities can be loaded from HuggingFace."""

    @pytest.mark.parametrize(
        "capability",
        [
            "execution",
            "search",
            "adaptability",
            "time",
            "ambiguity",
            "agent2agent",
            "noise",
        ],
    )
    def test_capability_has_tasks(self, capability):
        """Each VALID_CAPABILITY can be loaded and has tasks on HuggingFace."""
        pytest.importorskip("datasets", reason="HuggingFace datasets library required")
        pytest.importorskip("are", reason="ARE (meta-agents-research-environments) required")

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
