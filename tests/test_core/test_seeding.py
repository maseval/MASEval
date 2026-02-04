"""Tests for seed generation infrastructure.

These tests verify that the seeding system produces deterministic, reproducible
seeds and correctly handles hierarchical paths, task/repetition scoping, and
the per_repetition flag for selective variance.
"""

import pytest
from maseval.core.seeding import SeedGenerator, DefaultSeedGenerator, SeedingError


# Module-level marker for all tests in this file
pytestmark = pytest.mark.core


# ==================== Fixtures ====================


@pytest.fixture
def seed_generator():
    """Create a basic DefaultSeedGenerator with standard test seed."""
    return DefaultSeedGenerator(global_seed=42)


@pytest.fixture
def scoped_generator(seed_generator):
    """Create a generator scoped to task and repetition, ready for derive_seed()."""
    return seed_generator.for_task("task_1").for_repetition(0)


@pytest.fixture
def task_scoped_generator(seed_generator):
    """Create a generator scoped to task only (no repetition)."""
    return seed_generator.for_task("task_1")


# ==================== DefaultSeedGenerator Tests ====================


class TestDefaultSeedGenerator:
    """Tests for DefaultSeedGenerator implementation."""

    def test_deterministic_derivation(self, seed_generator):
        """Same inputs produce same output."""
        gen1 = seed_generator.for_task("task_1").for_repetition(0)
        gen2 = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)

        assert gen1.derive_seed("agent") == gen2.derive_seed("agent")

    @pytest.mark.parametrize("seed1,seed2", [(42, 43), (0, 1), (100, 200)])
    def test_different_global_seeds_produce_different_results(self, seed1, seed2):
        """Changing global_seed changes all derived seeds."""
        gen1 = DefaultSeedGenerator(global_seed=seed1).for_task("task_1").for_repetition(0)
        gen2 = DefaultSeedGenerator(global_seed=seed2).for_task("task_1").for_repetition(0)

        assert gen1.derive_seed("agent") != gen2.derive_seed("agent")

    @pytest.mark.parametrize("task1,task2", [("task_1", "task_2"), ("a", "b"), ("foo", "bar")])
    def test_different_tasks_produce_different_seeds(self, seed_generator, task1, task2):
        """Different task IDs produce different seeds."""
        gen1 = seed_generator.for_task(task1).for_repetition(0)
        gen2 = seed_generator.for_task(task2).for_repetition(0)

        assert gen1.derive_seed("agent") != gen2.derive_seed("agent")

    @pytest.mark.parametrize("rep1,rep2", [(0, 1), (0, 2), (1, 5)])
    def test_different_repetitions_produce_different_seeds(self, seed_generator, rep1, rep2):
        """Different repetition indices produce different seeds (when per_repetition=True)."""
        gen1 = seed_generator.for_task("task_1").for_repetition(rep1)
        gen2 = seed_generator.for_task("task_1").for_repetition(rep2)

        assert gen1.derive_seed("agent", per_repetition=True) != gen2.derive_seed("agent", per_repetition=True)

    @pytest.mark.parametrize("name1,name2", [("agent_a", "agent_b"), ("x", "y"), ("env/tool", "env/other")])
    def test_different_paths_produce_different_seeds(self, scoped_generator, name1, name2):
        """Different component names produce different seeds."""
        seed1 = scoped_generator.derive_seed(name1)
        # Need fresh generator since seed_log would have name1
        gen2 = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)
        seed2 = gen2.derive_seed(name2)

        assert seed1 != seed2

    @pytest.mark.parametrize("rep_index", [0, 1, 2, 5, 10])
    def test_per_repetition_false_constant_across_reps(self, seed_generator, rep_index):
        """per_repetition=False produces same seed across repetitions."""
        gen = seed_generator.for_task("task_1").for_repetition(rep_index)
        baseline_gen = seed_generator.for_task("task_1").for_repetition(0)

        seed = gen.derive_seed("baseline", per_repetition=False)
        baseline_seed = baseline_gen.derive_seed("baseline", per_repetition=False)

        assert seed == baseline_seed

    def test_mixed_per_repetition_in_same_task(self, scoped_generator):
        """Can mix per_repetition=True and False in same task repetition."""
        varying_seed = scoped_generator.derive_seed("experimental", per_repetition=True)
        constant_seed = scoped_generator.derive_seed("baseline", per_repetition=False)

        # They should be different (different paths)
        assert varying_seed != constant_seed

        # Verify constant is actually constant across reps
        gen_rep1 = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(1)
        constant_seed_rep1 = gen_rep1.derive_seed("baseline", per_repetition=False)
        assert constant_seed == constant_seed_rep1

    def test_global_seed_property(self, seed_generator):
        """global_seed property returns correct value and is preserved through scoping."""
        assert seed_generator.global_seed == 42

        task_gen = seed_generator.for_task("task_1")
        assert task_gen.global_seed == 42

        rep_gen = task_gen.for_repetition(0)
        assert rep_gen.global_seed == 42

    @pytest.mark.parametrize("i", range(20))
    def test_seed_in_valid_range(self, scoped_generator, i):
        """Derived seeds are in valid range [0, 2^31-1]."""
        seed = scoped_generator.derive_seed(f"component_{i}")
        assert 0 <= seed <= 0x7FFFFFFF


class TestDefaultSeedGeneratorChild:
    """Tests for child() method and hierarchical namespacing."""

    def test_child_extends_path(self, scoped_generator):
        """child() creates generator with extended path."""
        env_gen = scoped_generator.child("environment")
        env_gen.derive_seed("tool_weather")

        assert "environment/tool_weather" in scoped_generator.seed_log

    def test_child_shares_log(self, scoped_generator):
        """child() generators share the same seed log."""
        child1 = scoped_generator.child("agents")
        child2 = scoped_generator.child("tools")

        child1.derive_seed("agent_a")
        child2.derive_seed("tool_x")

        # Both should be in the shared log
        assert "agents/agent_a" in scoped_generator.seed_log
        assert "tools/tool_x" in scoped_generator.seed_log

        # Children also see the shared log
        assert "agents/agent_a" in child1.seed_log
        assert "tools/tool_x" in child2.seed_log

    def test_nested_children(self, scoped_generator):
        """Nested child() calls build correct paths."""
        env_gen = scoped_generator.child("environment")
        tools_gen = env_gen.child("tools")
        tools_gen.derive_seed("weather")

        assert "environment/tools/weather" in scoped_generator.seed_log

    def test_child_inherits_context(self, scoped_generator):
        """child() inherits task_id and rep_index."""
        child = scoped_generator.child("agents")

        # Should not raise - context is inherited
        seed = child.derive_seed("agent_a")
        assert isinstance(seed, int)

    def test_flat_paths_equivalent_to_child(self, seed_generator):
        """Flat paths produce same seeds as child() hierarchy."""
        # Using child()
        gen1 = seed_generator.for_task("task_1").for_repetition(0)
        child_gen = gen1.child("environment")
        seed_via_child = child_gen.derive_seed("tool_weather")

        # Using flat path in fresh generator
        gen2 = seed_generator.for_task("task_1").for_repetition(0)
        seed_via_flat = gen2.derive_seed("environment/tool_weather")

        assert seed_via_child == seed_via_flat


class TestDefaultSeedGeneratorSeedLog:
    """Tests for seed_log tracking."""

    def test_seed_log_records_all_derivations(self, scoped_generator):
        """seed_log records all derive_seed() calls."""
        scoped_generator.derive_seed("agent")
        scoped_generator.derive_seed("environment")
        scoped_generator.derive_seed("user")

        log = scoped_generator.seed_log
        assert len(log) == 3
        assert "agent" in log
        assert "environment" in log
        assert "user" in log

    def test_seed_log_returns_copy(self, scoped_generator):
        """seed_log returns a copy, not the internal dict."""
        scoped_generator.derive_seed("agent")

        log = scoped_generator.seed_log
        log["fake_entry"] = 999

        # Internal log should be unchanged
        assert "fake_entry" not in scoped_generator.seed_log

    def test_for_task_creates_fresh_log(self, seed_generator):
        """for_task() creates a fresh seed log."""
        task1_gen = seed_generator.for_task("task_1").for_repetition(0)
        task1_gen.derive_seed("agent")

        task2_gen = seed_generator.for_task("task_2").for_repetition(0)
        task2_gen.derive_seed("agent")

        # Each task has its own log
        assert "agent" in task1_gen.seed_log
        assert "agent" in task2_gen.seed_log

        # Logs are separate (different seeds due to different task_id)
        assert task1_gen.seed_log["agent"] != task2_gen.seed_log["agent"]


class TestDefaultSeedGeneratorErrors:
    """Tests for error handling."""

    def test_derive_seed_without_task_raises(self, seed_generator):
        """derive_seed() raises if task_id not set."""
        with pytest.raises(SeedingError, match="task_id not set"):
            seed_generator.derive_seed("agent")

    def test_derive_seed_per_rep_without_rep_raises(self, task_scoped_generator):
        """derive_seed(per_repetition=True) raises if rep_index not set."""
        with pytest.raises(SeedingError, match="rep_index not set"):
            task_scoped_generator.derive_seed("agent", per_repetition=True)

    def test_derive_seed_per_rep_false_without_rep_ok(self, task_scoped_generator):
        """derive_seed(per_repetition=False) works without rep_index."""
        seed = task_scoped_generator.derive_seed("baseline", per_repetition=False)
        assert isinstance(seed, int)


# ==================== SeedGenerator ABC Tests ====================


class TestSeedGeneratorABC:
    """Tests for SeedGenerator abstract base class."""

    def test_cannot_instantiate_abc(self):
        """SeedGenerator cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            SeedGenerator()  # type: ignore

    def test_subclass_must_implement_abstract_methods(self):
        """Incomplete subclass raises TypeError."""

        class IncompleteSeedGenerator(SeedGenerator):
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteSeedGenerator()  # type: ignore

    def test_custom_subclass_works(self):
        """Custom subclass with all methods implemented works."""

        class SimpleSeedGenerator(SeedGenerator):
            def __init__(self, seed: int = 0):
                super().__init__()
                self._seed = seed
                self._log: dict = {}

            @property
            def global_seed(self) -> int:
                return self._seed

            def derive_seed(self, name: str, per_repetition: bool = True) -> int:
                result = hash(f"{self._seed}:{name}") & 0x7FFFFFFF
                self._log[name] = result
                return result

            def for_task(self, task_id: str) -> "SimpleSeedGenerator":
                return self

            def for_repetition(self, rep_index: int) -> "SimpleSeedGenerator":
                return self

            @property
            def seed_log(self) -> dict:
                return dict(self._log)

        gen = SimpleSeedGenerator(seed=42)
        seed = gen.derive_seed("test")
        assert isinstance(seed, int)
        assert gen.global_seed == 42
        assert "test" in gen.seed_log


class TestCustomHashAlgorithm:
    """Tests for overriding _compute_seed() with custom hash."""

    def test_custom_compute_seed(self):
        """Custom _compute_seed() method is used."""
        from typing import Any, List

        class ConstantSeedGenerator(DefaultSeedGenerator):
            """Always returns 12345 for testing."""

            def _compute_seed(self, full_path: str, components: List[Any]) -> int:
                return 12345

        gen = ConstantSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)

        assert gen.derive_seed("anything") == 12345
        assert gen.derive_seed("something_else") == 12345


# ==================== Configuration Integration Tests ====================


class TestSeedGeneratorGatherConfig:
    """Tests for gather_config() integration."""

    def test_gather_config_includes_seeds(self, scoped_generator):
        """gather_config() includes seed_log."""
        scoped_generator.derive_seed("agent")
        scoped_generator.derive_seed("environment")

        config = scoped_generator.gather_config()

        assert "seeds" in config
        assert "agent" in config["seeds"]
        assert "environment" in config["seeds"]

    def test_gather_config_includes_context(self, seed_generator):
        """gather_config() includes global_seed, task_id, rep_index."""
        gen = seed_generator.for_task("task_1").for_repetition(3)
        config = gen.gather_config()

        assert config["global_seed"] == 42
        assert config["task_id"] == "task_1"
        assert config["rep_index"] == 3

    def test_gather_config_includes_base_fields(self, seed_generator):
        """gather_config() includes ConfigurableMixin base fields."""
        config = seed_generator.gather_config()

        assert "type" in config
        assert config["type"] == "DefaultSeedGenerator"
        assert "gathered_at" in config


class TestSeedingError:
    """Tests for SeedingError exception."""

    def test_seeding_error_message(self):
        """SeedingError has correct message."""
        error = SeedingError("Custom error message")
        assert str(error) == "Custom error message"

    def test_seeding_error_is_exception(self):
        """SeedingError is an Exception subclass."""
        error = SeedingError("test")
        assert isinstance(error, Exception)
