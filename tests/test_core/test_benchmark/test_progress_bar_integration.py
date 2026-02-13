"""Integration test for progress bar with Benchmark.

NOTE: These tests verify that progress bars integrate correctly with the benchmark
execution flow. Visual aesthetics (colors, formatting, layout) are NOT tested and
should be manually verified during development.
"""

import pytest
import sys
from pathlib import Path

# Add tests directory to path to import conftest
tests_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(tests_dir))

from conftest import DummyBenchmark  # noqa: E402

from maseval.core.task import Task, TaskQueue  # noqa: E402
from maseval.core.callbacks.progress_bar import (  # noqa: E402
    TqdmProgressBarCallback,
    RichProgressBarCallback,
)


@pytest.mark.core
def test_benchmark_with_default_progress_bar():
    """Test that benchmark attaches tqdm progress bar by default."""
    tasks = TaskQueue([Task(query="What is 2+2?")])

    # Default should have progress bar
    benchmark = DummyBenchmark()

    # Check that a progress bar callback was added
    progress_bars = [cb for cb in benchmark.callbacks if isinstance(cb, TqdmProgressBarCallback)]
    assert len(progress_bars) == 1

    # Run and verify it works
    reports = benchmark.run(tasks, agent_data={"model": "test"})
    assert len(reports) == 1
    assert reports[0]["status"] == "success"


@pytest.mark.core
def test_benchmark_with_disabled_progress_bar():
    """Test that progress bar can be disabled."""
    tasks = TaskQueue([Task(query="What is 2+2?")])

    benchmark = DummyBenchmark(progress_bar=False)

    # Should have no progress bar callbacks
    progress_bars = [cb for cb in benchmark.callbacks if isinstance(cb, (TqdmProgressBarCallback, RichProgressBarCallback))]
    assert len(progress_bars) == 0

    reports = benchmark.run(tasks, agent_data={"model": "test"})
    assert len(reports) == 1


@pytest.mark.core
def test_benchmark_with_rich_progress_bar():
    """Test that rich progress bar can be specified."""
    tasks = TaskQueue([Task(query="What is 2+2?")])

    benchmark = DummyBenchmark(progress_bar="rich")

    # Should have a rich progress bar
    progress_bars = [cb for cb in benchmark.callbacks if isinstance(cb, RichProgressBarCallback)]
    assert len(progress_bars) == 1

    reports = benchmark.run(tasks, agent_data={"model": "test"})
    assert len(reports) == 1


@pytest.mark.core
def test_benchmark_with_custom_progress_bar():
    """Test that custom progress bar callback prevents default from being added."""
    tasks = TaskQueue([Task(query="What is 2+2?")])

    # User provides their own progress bar
    custom_pbar = TqdmProgressBarCallback(desc="Custom Progress")

    benchmark = DummyBenchmark(
        callbacks=[custom_pbar],
        progress_bar=True,  # Should be ignored
    )

    # Should only have the one custom progress bar
    progress_bars = [cb for cb in benchmark.callbacks if isinstance(cb, TqdmProgressBarCallback)]
    assert len(progress_bars) == 1
    assert progress_bars[0].desc == "Custom Progress"

    reports = benchmark.run(tasks, agent_data={"model": "test"})
    assert len(reports) == 1


@pytest.mark.core
def test_benchmark_with_multiple_tasks_and_repeats():
    """Test progress bar with multiple tasks and repeats."""
    tasks = TaskQueue([Task(query=f"Task {i}") for i in range(3)])

    benchmark = DummyBenchmark(n_task_repeats=2, progress_bar=True)

    # Get the progress bar callback
    progress_bars = [cb for cb in benchmark.callbacks if isinstance(cb, TqdmProgressBarCallback)]
    assert len(progress_bars) == 1
    pbar = progress_bars[0]

    reports = benchmark.run(tasks, agent_data={"model": "test"})

    # Should have 3 tasks * 2 repeats = 6 reports
    assert len(reports) == 6

    # Progress bar should have tracked all iterations
    assert pbar.total == 6
    assert pbar.current == 6


@pytest.mark.core
def test_invalid_progress_bar_value():
    """Test that invalid progress bar value raises error."""
    with pytest.raises(ValueError, match="Invalid progress_bar value"):
        DummyBenchmark(progress_bar="invalid")
