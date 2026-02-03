"""Tests for seed generation infrastructure.

These tests verify that the seeding system produces deterministic, reproducible
seeds and correctly handles hierarchical paths, task/repetition scoping, and
the per_repetition flag for selective variance.
"""

import pytest
from maseval.core.seeding import SeedGenerator, DefaultSeedGenerator, SeedingError


@pytest.mark.core
class TestDefaultSeedGenerator:
    """Tests for DefaultSeedGenerator implementation."""

    def test_deterministic_derivation(self):
        """Same inputs produce same output."""
        gen1 = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)
        gen2 = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)

        seed1 = gen1.derive_seed("agent")
        seed2 = gen2.derive_seed("agent")

        assert seed1 == seed2

    def test_different_global_seeds_produce_different_results(self):
        """Changing global_seed changes all derived seeds."""
        gen1 = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)
        gen2 = DefaultSeedGenerator(global_seed=43).for_task("task_1").for_repetition(0)

        seed1 = gen1.derive_seed("agent")
        seed2 = gen2.derive_seed("agent")

        assert seed1 != seed2

    def test_different_tasks_produce_different_seeds(self):
        """Different task IDs produce different seeds."""
        gen1 = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)
        gen2 = DefaultSeedGenerator(global_seed=42).for_task("task_2").for_repetition(0)

        seed1 = gen1.derive_seed("agent")
        seed2 = gen2.derive_seed("agent")

        assert seed1 != seed2

    def test_different_repetitions_produce_different_seeds(self):
        """Different repetition indices produce different seeds (when per_repetition=True)."""
        gen1 = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)
        gen2 = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(1)

        seed1 = gen1.derive_seed("agent", per_repetition=True)
        seed2 = gen2.derive_seed("agent", per_repetition=True)

        assert seed1 != seed2

    def test_different_paths_produce_different_seeds(self):
        """Different component names produce different seeds."""
        gen = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)

        seed1 = gen.derive_seed("agent_a")
        seed2 = gen.derive_seed("agent_b")

        assert seed1 != seed2

    def test_per_repetition_false_constant_across_reps(self):
        """per_repetition=False produces same seed across repetitions."""
        gen1 = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)
        gen2 = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(1)
        gen3 = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(2)

        seed1 = gen1.derive_seed("baseline", per_repetition=False)
        seed2 = gen2.derive_seed("baseline", per_repetition=False)
        seed3 = gen3.derive_seed("baseline", per_repetition=False)

        assert seed1 == seed2 == seed3

    def test_mixed_per_repetition_in_same_task(self):
        """Can mix per_repetition=True and False in same task repetition."""
        gen = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)

        # One component varies per rep
        varying_seed = gen.derive_seed("experimental", per_repetition=True)

        # Another component is constant
        constant_seed = gen.derive_seed("baseline", per_repetition=False)

        # They should be different (different paths)
        assert varying_seed != constant_seed

        # Verify constant is actually constant
        gen2 = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(1)
        constant_seed_rep1 = gen2.derive_seed("baseline", per_repetition=False)
        assert constant_seed == constant_seed_rep1

    def test_global_seed_property(self):
        """global_seed property returns correct value."""
        gen = DefaultSeedGenerator(global_seed=12345)
        assert gen.global_seed == 12345

        # Preserved through scoping
        task_gen = gen.for_task("task_1")
        assert task_gen.global_seed == 12345

        rep_gen = task_gen.for_repetition(0)
        assert rep_gen.global_seed == 12345

    def test_seed_in_valid_range(self):
        """Derived seeds are in valid range [0, 2^31-1]."""
        gen = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)

        for i in range(100):
            seed = gen.derive_seed(f"component_{i}")
            assert 0 <= seed <= 0x7FFFFFFF


@pytest.mark.core
class TestDefaultSeedGeneratorChild:
    """Tests for child() method and hierarchical namespacing."""

    def test_child_extends_path(self):
        """child() creates generator with extended path."""
        gen = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)
        env_gen = gen.child("environment")

        seed = env_gen.derive_seed("tool_weather")

        # Check seed log has full path
        assert "environment/tool_weather" in gen.seed_log

    def test_child_shares_log(self):
        """child() generators share the same seed log."""
        gen = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)

        child1 = gen.child("agents")
        child2 = gen.child("tools")

        child1.derive_seed("agent_a")
        child2.derive_seed("tool_x")

        # Both should be in the shared log
        assert "agents/agent_a" in gen.seed_log
        assert "tools/tool_x" in gen.seed_log

        # Children also see the shared log
        assert "agents/agent_a" in child1.seed_log
        assert "tools/tool_x" in child2.seed_log

    def test_nested_children(self):
        """Nested child() calls build correct paths."""
        gen = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)

        env_gen = gen.child("environment")
        tools_gen = env_gen.child("tools")
        _ = tools_gen.derive_seed("weather")

        assert "environment/tools/weather" in gen.seed_log

    def test_child_inherits_context(self):
        """child() inherits task_id and rep_index."""
        gen = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)
        child = gen.child("agents")

        # Should not raise - context is inherited
        seed = child.derive_seed("agent_a")
        assert isinstance(seed, int)

    def test_flat_paths_equivalent_to_child(self):
        """Flat paths produce same seeds as child() hierarchy."""
        gen = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)

        # Using child()
        child_gen = gen.child("environment")
        seed_via_child = child_gen.derive_seed("tool_weather")

        # Using flat path in fresh generator
        gen2 = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)
        seed_via_flat = gen2.derive_seed("environment/tool_weather")

        assert seed_via_child == seed_via_flat


@pytest.mark.core
class TestDefaultSeedGeneratorSeedLog:
    """Tests for seed_log tracking."""

    def test_seed_log_records_all_derivations(self):
        """seed_log records all derive_seed() calls."""
        gen = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)

        gen.derive_seed("agent")
        gen.derive_seed("environment")
        gen.derive_seed("user")

        log = gen.seed_log
        assert len(log) == 3
        assert "agent" in log
        assert "environment" in log
        assert "user" in log

    def test_seed_log_returns_copy(self):
        """seed_log returns a copy, not the internal dict."""
        gen = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)
        gen.derive_seed("agent")

        log = gen.seed_log
        log["fake_entry"] = 999

        # Internal log should be unchanged
        assert "fake_entry" not in gen.seed_log

    def test_for_task_creates_fresh_log(self):
        """for_task() creates a fresh seed log."""
        root = DefaultSeedGenerator(global_seed=42)

        task1_gen = root.for_task("task_1").for_repetition(0)
        task1_gen.derive_seed("agent")

        task2_gen = root.for_task("task_2").for_repetition(0)
        task2_gen.derive_seed("agent")

        # Each task has its own log
        assert "agent" in task1_gen.seed_log
        assert "agent" in task2_gen.seed_log

        # Logs are separate
        assert task1_gen.seed_log["agent"] != task2_gen.seed_log["agent"]


@pytest.mark.core
class TestDefaultSeedGeneratorErrors:
    """Tests for error handling."""

    def test_derive_seed_without_task_raises(self):
        """derive_seed() raises if task_id not set."""
        gen = DefaultSeedGenerator(global_seed=42)

        with pytest.raises(SeedingError, match="task_id not set"):
            gen.derive_seed("agent")

    def test_derive_seed_per_rep_without_rep_raises(self):
        """derive_seed(per_repetition=True) raises if rep_index not set."""
        gen = DefaultSeedGenerator(global_seed=42).for_task("task_1")

        with pytest.raises(SeedingError, match="rep_index not set"):
            gen.derive_seed("agent", per_repetition=True)

    def test_derive_seed_per_rep_false_without_rep_ok(self):
        """derive_seed(per_repetition=False) works without rep_index."""
        gen = DefaultSeedGenerator(global_seed=42).for_task("task_1")

        # Should not raise
        seed = gen.derive_seed("baseline", per_repetition=False)
        assert isinstance(seed, int)


@pytest.mark.core
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
        from typing import Self

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

            def for_task(self, task_id: str) -> Self:
                return self

            def for_repetition(self, rep_index: int) -> Self:
                return self

            @property
            def seed_log(self) -> dict:
                return dict(self._log)

        gen = SimpleSeedGenerator(seed=42)
        seed = gen.derive_seed("test")
        assert isinstance(seed, int)
        assert gen.global_seed == 42
        assert "test" in gen.seed_log


@pytest.mark.core
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


@pytest.mark.core
class TestSeedGeneratorGatherConfig:
    """Tests for gather_config() integration."""

    def test_gather_config_includes_seeds(self):
        """gather_config() includes seed_log."""
        gen = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)
        gen.derive_seed("agent")
        gen.derive_seed("environment")

        config = gen.gather_config()

        assert "seeds" in config
        assert "agent" in config["seeds"]
        assert "environment" in config["seeds"]

    def test_gather_config_includes_context(self):
        """gather_config() includes global_seed, task_id, rep_index."""
        gen = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(3)

        config = gen.gather_config()

        assert config["global_seed"] == 42
        assert config["task_id"] == "task_1"
        assert config["rep_index"] == 3

    def test_gather_config_includes_base_fields(self):
        """gather_config() includes ConfigurableMixin base fields."""
        gen = DefaultSeedGenerator(global_seed=42)

        config = gen.gather_config()

        assert "type" in config
        assert config["type"] == "DefaultSeedGenerator"
        assert "gathered_at" in config


@pytest.mark.core
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
