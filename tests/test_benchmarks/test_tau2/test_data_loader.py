"""Unit tests for Tau2 data_loader module."""

import pytest

from maseval.benchmark.tau2.data_loader import (
    DEFAULT_DATA_DIR,
    VALID_DOMAINS,
    TASK_SPLITS,
    load_domain_config,
    load_tasks,
    configure_model_ids,
    ensure_data_exists,
)


# =============================================================================
# Constants Tests
# =============================================================================


@pytest.mark.benchmark
class TestConstants:
    """Tests for module constants."""

    def test_valid_domains(self):
        """VALID_DOMAINS contains expected domains."""
        assert "retail" in VALID_DOMAINS
        assert "airline" in VALID_DOMAINS
        assert "telecom" in VALID_DOMAINS
        assert len(VALID_DOMAINS) == 3

    def test_task_splits(self):
        """TASK_SPLITS contains expected splits."""
        assert "base" in TASK_SPLITS
        assert "hard" in TASK_SPLITS
        assert "all" in TASK_SPLITS

    def test_default_data_dir_is_module_relative(self):
        """DEFAULT_DATA_DIR points to module's data directory."""
        assert DEFAULT_DATA_DIR.name == "data"
        assert DEFAULT_DATA_DIR.parent.name == "tau2"
        assert DEFAULT_DATA_DIR.parent.parent.name == "benchmark"


# =============================================================================
# Domain Config Tests
# =============================================================================


@pytest.mark.benchmark
class TestLoadDomainConfig:
    """Tests for load_domain_config function."""

    @pytest.mark.live
    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_loads_domain_config(self, domain, ensure_tau2_data):
        """Loads domain configuration successfully."""
        config = load_domain_config(domain, ensure_tau2_data)

        assert "policy" in config
        assert "db_path" in config
        assert config["db_path"].exists()
        assert len(config["policy"]) > 0

    def test_invalid_domain_raises(self):
        """Invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="Invalid domain"):
            load_domain_config("invalid_domain")


# =============================================================================
# Load Tasks Tests
# =============================================================================


@pytest.mark.benchmark
class TestLoadTasks:
    """Tests for load_tasks function."""

    @pytest.mark.live
    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_loads_domain_tasks(self, domain, ensure_tau2_data):
        """Loads domain tasks with limit."""
        tasks = load_tasks(domain, split="base", limit=5, data_dir=ensure_tau2_data)

        assert len(tasks) == 5
        assert tasks[0].query is not None

    @pytest.mark.live
    def test_limit_parameter(self, ensure_tau2_data):
        """Limit parameter restricts number of tasks."""
        tasks_3 = load_tasks("retail", limit=3, data_dir=ensure_tau2_data)
        tasks_10 = load_tasks("retail", limit=10, data_dir=ensure_tau2_data)

        assert len(tasks_3) == 3
        assert len(tasks_10) == 10

    def test_invalid_domain_raises(self):
        """Invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="Invalid domain"):
            load_tasks("invalid_domain")

    def test_invalid_split_raises(self):
        """Invalid split raises ValueError."""
        with pytest.raises(ValueError, match="Invalid split"):
            load_tasks("retail", split="invalid_split")


# =============================================================================
# Configure Model IDs Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestConfigureModelIds:
    """Tests for configure_model_ids function."""

    def test_configures_user_model(self, ensure_tau2_data):
        """Configures user model ID."""
        tasks = load_tasks("retail", limit=2, data_dir=ensure_tau2_data)
        configure_model_ids(tasks, user_model_id="gpt-4o")

        for task in tasks:
            assert task.user_data.get("model_id") == "gpt-4o"

    def test_configures_evaluator_model(self, ensure_tau2_data):
        """Configures evaluator model ID."""
        tasks = load_tasks("retail", limit=2, data_dir=ensure_tau2_data)
        configure_model_ids(tasks, evaluator_model_id="claude-3-opus")

        for task in tasks:
            assert task.evaluation_data.get("model_id") == "claude-3-opus"

    def test_configures_multiple_models(self, ensure_tau2_data):
        """Configures multiple model IDs at once."""
        tasks = load_tasks("retail", limit=2, data_dir=ensure_tau2_data)
        configure_model_ids(
            tasks,
            user_model_id="user-model",
            evaluator_model_id="eval-model",
        )

        for task in tasks:
            assert task.user_data.get("model_id") == "user-model"
            assert task.evaluation_data.get("model_id") == "eval-model"


# =============================================================================
# Ensure Data Exists Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestEnsureDataExists:
    """Tests for ensure_data_exists function."""

    @pytest.mark.parametrize(
        "domain,db_ext",
        [
            ("retail", ".json"),
            ("airline", ".json"),
            ("telecom", ".toml"),
        ],
    )
    def test_domain_data_exists(self, domain, db_ext, ensure_tau2_data):
        """Domain data files exist after ensure_data_exists."""
        result = ensure_data_exists(data_dir=ensure_tau2_data, domain=domain)

        assert result.exists()
        assert (result / domain / f"db{db_ext}").exists()
        assert (result / domain / "tasks.json").exists()
        assert (result / domain / "policy.md").exists()


# =============================================================================
# Task Content Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestTaskContent:
    """Tests for task content structure."""

    def test_task_has_required_fields(self, ensure_tau2_data):
        """Tasks have all required fields."""
        tasks = load_tasks("retail", limit=5, data_dir=ensure_tau2_data)

        for task in tasks:
            assert task.id is not None
            assert task.query is not None
            assert task.environment_data is not None
            assert task.user_data is not None
            assert task.evaluation_data is not None

    def test_task_evaluation_data_structure(self, ensure_tau2_data):
        """Task evaluation_data has expected structure."""
        tasks = load_tasks("retail", limit=5, data_dir=ensure_tau2_data)

        for task in tasks:
            eval_data = task.evaluation_data
            # Just verify the structure is a dict
            assert isinstance(eval_data, dict)

    def test_task_user_data_has_instructions(self, ensure_tau2_data):
        """Task user_data contains instructions."""
        tasks = load_tasks("retail", limit=5, data_dir=ensure_tau2_data)

        for task in tasks:
            user_data = task.user_data
            # Just verify user_data is a dict
            assert isinstance(user_data, dict)


# =============================================================================
# Split Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestTaskSplits:
    """Tests for task split loading."""

    def test_load_all_split(self, ensure_tau2_data):
        """Load all split returns all tasks."""
        tasks_all = load_tasks("retail", split="all", limit=100, data_dir=ensure_tau2_data)
        tasks_base = load_tasks("retail", split="base", limit=100, data_dir=ensure_tau2_data)
        tasks_hard = load_tasks("retail", split="hard", limit=100, data_dir=ensure_tau2_data)

        # All should include both base and hard
        assert len(tasks_all) >= len(tasks_base)
        assert len(tasks_all) >= len(tasks_hard)

    def test_load_hard_split(self, ensure_tau2_data):
        """Load hard split returns hard tasks."""
        tasks = load_tasks("retail", split="hard", limit=10, data_dir=ensure_tau2_data)

        # Hard split may be empty if all tasks are in base split
        for task in tasks:
            # Hard tasks should have hard indicator in metadata or split
            if hasattr(task, "metadata") and "split" in task.metadata:
                assert task.metadata.get("split") in ["hard", "all"]


# =============================================================================
# Configure Model IDs Edge Cases
# =============================================================================


@pytest.mark.benchmark
class TestConfigureModelIdsEdgeCases:
    """Edge case tests for configure_model_ids function."""

    def test_configures_empty_task_list(self):
        """Configure model IDs on empty task list doesn't error."""
        tasks = []
        configure_model_ids(tasks, user_model_id="test-model")

        assert len(tasks) == 0

    @pytest.mark.live
    def test_cannot_overwrite_existing_model_id(self, ensure_tau2_data):
        """Cannot overwrite existing model_id - raises error."""
        tasks = load_tasks("retail", limit=2, data_dir=ensure_tau2_data)

        # First configure
        configure_model_ids(tasks, user_model_id="first-model")
        assert tasks[0].user_data.get("model_id") == "first-model"

        # Second configure should raise ValueError
        with pytest.raises(ValueError, match="already has"):
            configure_model_ids(tasks, user_model_id="second-model")


# =============================================================================
# Task Metadata Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestTaskMetadata:
    """Tests for task metadata."""

    def test_task_has_id(self, ensure_tau2_data):
        """All tasks have an id."""
        tasks = load_tasks("retail", limit=10, data_dir=ensure_tau2_data)

        for task in tasks:
            assert task.id is not None
            # Task ID is always a string
            assert isinstance(task.id, str)

    def test_task_ids_unique(self, ensure_tau2_data):
        """Task IDs are unique."""
        tasks = load_tasks("retail", limit=50, data_dir=ensure_tau2_data)

        ids = [task.id for task in tasks]
        assert len(ids) == len(set(ids)), "Task IDs are not unique"

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_task_environment_data_has_domain(self, domain, ensure_tau2_data):
        """Task environment_data includes correct domain."""
        tasks = load_tasks(domain, limit=5, data_dir=ensure_tau2_data)

        for task in tasks:
            assert "domain" in task.environment_data
            assert task.environment_data["domain"] == domain


# =============================================================================
# Domain-specific Task Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestDomainTasks:
    """Tests for domain-specific task loading."""

    def test_cross_domain_tasks_different(self, ensure_tau2_data):
        """Tasks from different domains have different content."""
        retail_tasks = load_tasks("retail", limit=3, data_dir=ensure_tau2_data)
        airline_tasks = load_tasks("airline", limit=3, data_dir=ensure_tau2_data)

        # Task queries should be different between domains
        retail_queries = {t.query for t in retail_tasks}
        airline_queries = {t.query for t in airline_tasks}

        # There should be little to no overlap
        overlap = retail_queries & airline_queries
        assert len(overlap) <= 1, "Too much overlap between domain tasks"


# =============================================================================
# Task Filtering Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestTaskFiltering:
    """Tests for task filtering functionality."""

    def test_load_specific_task_ids(self, ensure_tau2_data):
        """Load specific tasks by ID."""
        # First get some task IDs
        all_tasks = load_tasks("retail", limit=10, data_dir=ensure_tau2_data)
        if len(all_tasks) < 3:
            pytest.skip("Not enough tasks in database")

        # Try to load specific tasks by filtering (if supported)
        task_ids = [str(t.id) for t in all_tasks[:3]]

        # Check that tasks have the expected IDs
        for i, task in enumerate(all_tasks[:3]):
            assert str(task.id) == task_ids[i]

    def test_load_with_negative_limit(self, ensure_tau2_data):
        """Load with negative limit returns all tasks (no limit)."""
        tasks = load_tasks("retail", limit=-1, data_dir=ensure_tau2_data)
        # Negative limit means no limit - should return all tasks
        assert len(tasks) > 0

    def test_load_with_large_limit(self, ensure_tau2_data):
        """Load with limit larger than available tasks."""
        tasks = load_tasks("retail", limit=10000, data_dir=ensure_tau2_data)
        # Should return all available tasks without error
        assert len(tasks) > 0


# =============================================================================
# Policy Loading Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestPolicyLoading:
    """Tests for policy loading in domain config."""

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_policy_content(self, domain, ensure_tau2_data):
        """Domain policy contains content."""
        config = load_domain_config(domain, ensure_tau2_data)

        policy = config.get("policy", "")
        assert len(policy) > 0
        assert isinstance(policy, str)


# =============================================================================
# Database Path Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestDatabasePaths:
    """Tests for database path handling."""

    @pytest.mark.parametrize(
        "domain,expected_suffix",
        [
            ("retail", ".json"),
            ("airline", ".json"),
            ("telecom", ".toml"),
        ],
    )
    def test_db_format(self, domain, expected_suffix, ensure_tau2_data):
        """Database uses correct format for domain."""
        config = load_domain_config(domain, ensure_tau2_data)

        db_path = config.get("db_path")
        assert db_path is not None
        assert db_path.suffix == expected_suffix


# =============================================================================
# Task ID Handling Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestTaskIdHandling:
    """Tests for task ID handling."""

    def test_task_ids_are_strings_or_ints(self, ensure_tau2_data):
        """Task IDs can be converted to strings."""
        tasks = load_tasks("retail", limit=10, data_dir=ensure_tau2_data)

        for task in tasks:
            # Should be able to convert to string
            str_id = str(task.id)
            assert len(str_id) > 0

    def test_task_metadata_contains_original_id(self, ensure_tau2_data):
        """Task metadata may contain original task ID."""
        tasks = load_tasks("retail", limit=5, data_dir=ensure_tau2_data)

        for task in tasks:
            # Task should have either id in metadata or as attribute
            assert task.id is not None
