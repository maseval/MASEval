"""Tests for progress bar callbacks.

NOTE: These tests verify functional behavior (metrics tracking, lifecycle hooks,
thread safety) but do NOT test visual aesthetics (colors, formatting, appearance).
Visual output should be manually verified during development.
"""

import pytest
from unittest.mock import MagicMock, patch

from maseval.core.callbacks.progress_bar import (
    ProgressBarCallback,
    TqdmProgressBarCallback,
    RichProgressBarCallback,
)
from maseval.core.task import Task

pytestmark = pytest.mark.core


@pytest.fixture
def mock_benchmark():
    """Create a mock benchmark for testing."""
    benchmark = MagicMock()
    benchmark.tasks = [
        Task(query="Task 1"),
        Task(query="Task 2"),
    ]
    benchmark.n_task_repeats = 2
    benchmark.__class__.__name__ = "MockBenchmark"
    return benchmark


@pytest.fixture
def mock_task():
    """Create a mock task for testing."""
    return Task(query="What is the weather today?")


@pytest.fixture
def mock_report():
    """Create a mock report for testing."""
    return {
        "task_id": "task1",
        "repeat_idx": 0,
        "status": "success",
        "traces": {},
        "config": {},
        "evaluation_result": {},
    }


class TestProgressBarCallback:
    """Test the abstract ProgressBarCallback base class."""

    def test_initialization(self):
        """Test that ProgressBarCallback can be instantiated with proper attributes."""

        class ConcreteProgressBar(ProgressBarCallback):
            def _initialize_progress_bar(self):
                pass

            def _update_description(self, desc):
                pass

            def _update_progress(self, current, status):
                pass

            def _update_metrics(self):
                pass

            def _close_progress_bar(self):
                pass

        pbar = ConcreteProgressBar(desc="Test", show_status=False)
        assert pbar.desc == "Test"
        assert pbar.show_status is False
        assert pbar.total == 0
        assert pbar.current == 0

    def test_set_metrics(self):
        """Test manually setting custom metrics."""

        class ConcreteProgressBar(ProgressBarCallback):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.update_metrics_called = False

            def _initialize_progress_bar(self):
                pass

            def _update_description(self, desc):
                pass

            def _update_progress(self, current, status):
                pass

            def _update_metrics(self):
                self.update_metrics_called = True

            def _close_progress_bar(self):
                pass

        pbar = ConcreteProgressBar()

        # Set metrics manually
        pbar.set_metrics(accuracy="95%", f1="0.87")

        assert pbar._custom_metrics == {"accuracy": "95%", "f1": "0.87"}
        assert pbar.update_metrics_called

    def test_update_metrics_override(self, mock_benchmark, mock_report):
        """Test that update_metrics can be overridden to extract metrics from reports."""

        class CustomProgressBar(ProgressBarCallback):
            def _initialize_progress_bar(self):
                pass

            def _update_description(self, desc):
                pass

            def _update_progress(self, current, status):
                pass

            def _update_metrics(self):
                pass

            def _close_progress_bar(self):
                pass

            def update_metrics(self, report):
                if "evaluation_result" in report and report["evaluation_result"]:
                    return {"test_metric": "extracted"}
                return {}

        pbar = CustomProgressBar()
        pbar.on_run_start(mock_benchmark)

        # Update with report containing evaluation results
        mock_report["evaluation_result"] = {"accuracy": 0.95}
        pbar.on_task_repeat_end(mock_benchmark, mock_report)

        # Metrics should be extracted and stored
        assert "test_metric" in pbar._custom_metrics
        assert pbar._custom_metrics["test_metric"] == "extracted"


class TestTqdmProgressBarCallback:
    """Test the TqdmProgressBarCallback implementation.

    NOTE: Visual appearance (colors, formatting) not tested - verify manually.
    """

    def test_initialization(self):
        """Test TqdmProgressBarCallback initialization."""
        pbar = TqdmProgressBarCallback(desc="Test Benchmark", leave=True, ncols=80)
        assert pbar.desc == "Test Benchmark"
        assert pbar.leave is True
        assert pbar.ncols == 80

    @patch("tqdm.tqdm")
    def test_lifecycle(self, mock_tqdm_class, mock_benchmark, mock_task, mock_report):
        """Test the full lifecycle of TqdmProgressBarCallback."""
        mock_tqdm_instance = MagicMock()
        mock_tqdm_class.return_value = mock_tqdm_instance

        pbar = TqdmProgressBarCallback(desc="Test", leave=False)

        # Start
        pbar.on_run_start(mock_benchmark)
        assert pbar.total == 4  # 2 tasks * 2 repeats
        assert pbar.desc == "Test"  # Custom desc preserved

        # Verify tqdm was initialized
        mock_tqdm_class.assert_called_once_with(total=4, desc="Test", leave=False, ncols=None, bar_format=None, unit="task")

        # Task repeat end
        pbar.on_task_repeat_end(mock_benchmark, mock_report)
        assert pbar.current == 1
        assert pbar.successful == 1  # Status was "success"

        # Verify progress update
        mock_tqdm_instance.update.assert_called_once_with(1)

        # End
        pbar.on_run_end(mock_benchmark, [])
        mock_tqdm_instance.close.assert_called_once()

    @patch("tqdm.tqdm")
    def test_default_description_from_benchmark(self, mock_tqdm_class, mock_benchmark):
        """Test that description defaults to benchmark class name."""
        mock_tqdm_instance = MagicMock()
        mock_tqdm_class.return_value = mock_tqdm_instance

        pbar = TqdmProgressBarCallback()  # No desc provided
        pbar.on_run_start(mock_benchmark)

        # Should use benchmark class name
        assert pbar.desc == "Running MockBenchmark"

    @patch("tqdm.tqdm")
    def test_status_tracking(self, mock_tqdm_class, mock_benchmark):
        """Test that different statuses are tracked correctly."""
        mock_tqdm_instance = MagicMock()
        mock_tqdm_class.return_value = mock_tqdm_instance

        pbar = TqdmProgressBarCallback()
        pbar.on_run_start(mock_benchmark)

        # Success status
        pbar.on_task_repeat_end(mock_benchmark, {"status": "success"})
        assert pbar.successful == 1
        assert pbar.current == 1

        # Failure status
        pbar.on_task_repeat_end(mock_benchmark, {"status": "task_execution_failed"})
        assert pbar.successful == 1  # Should not increment
        assert pbar.current == 2

    @patch("tqdm.tqdm")
    def test_custom_metrics_display(self, mock_tqdm_class, mock_benchmark):
        """Test that custom metrics are displayed correctly."""
        mock_tqdm_instance = MagicMock()
        mock_tqdm_class.return_value = mock_tqdm_instance

        pbar = TqdmProgressBarCallback()
        pbar.on_run_start(mock_benchmark)

        # Set custom metrics
        pbar.set_metrics(accuracy="95%", f1="0.87")

        # Verify set_postfix_str was called with metrics
        mock_tqdm_instance.set_postfix_str.assert_called()
        call_args = mock_tqdm_instance.set_postfix_str.call_args[0][0]
        assert "accuracy=95%" in call_args
        assert "f1=0.87" in call_args

    @patch("tqdm.tqdm")
    def test_update_metrics_extraction(self, mock_tqdm_class, mock_benchmark):
        """Test automatic metric extraction from reports."""

        class CustomTqdmProgress(TqdmProgressBarCallback):
            def update_metrics(self, report):
                if "evaluation_result" in report:
                    result = report["evaluation_result"]
                    return {
                        "accuracy": f"{result.get('accuracy', 0):.1%}",
                    }
                return {}

        mock_tqdm_instance = MagicMock()
        mock_tqdm_class.return_value = mock_tqdm_instance

        pbar = CustomTqdmProgress()
        pbar.on_run_start(mock_benchmark)

        # Process report with evaluation results
        report = {"status": "success", "evaluation_result": {"accuracy": 0.95}}
        pbar.on_task_repeat_end(mock_benchmark, report)

        # Metrics should be extracted and displayed
        assert "accuracy" in pbar._custom_metrics
        assert pbar._custom_metrics["accuracy"] == "95.0%"

    @patch("tqdm.tqdm")
    def test_show_status_flag(self, mock_tqdm_class, mock_benchmark):
        """Test that show_status flag controls status display."""
        mock_tqdm_instance = MagicMock()
        mock_tqdm_class.return_value = mock_tqdm_instance

        pbar = TqdmProgressBarCallback(show_status=False)
        pbar.on_run_start(mock_benchmark)

        pbar.on_task_repeat_end(mock_benchmark, {"status": "success"})

        # With show_status=False, postfix should not include success counter
        # (implementation passes None for status parameter)
        assert pbar.successful == 1  # Still tracked internally
        assert pbar.current == 1


class TestRichProgressBarCallback:
    """Test the RichProgressBarCallback implementation.

    NOTE: Visual appearance (colors, formatting, rich styling) not tested - verify manually.
    """

    def test_initialization(self):
        """Test RichProgressBarCallback initialization."""
        pbar = RichProgressBarCallback(desc="Test Benchmark", transient=True)
        assert pbar.desc == "Test Benchmark"
        assert pbar.transient is True

    @patch("rich.progress.Progress")
    def test_lifecycle(self, mock_progress_class, mock_benchmark, mock_task, mock_report):
        """Test the full lifecycle of RichProgressBarCallback."""
        mock_progress_instance = MagicMock()
        mock_progress_class.return_value = mock_progress_instance
        mock_progress_instance.add_task.return_value = "task_id_123"

        pbar = RichProgressBarCallback(desc="Test")

        # Start
        pbar.on_run_start(mock_benchmark)
        assert pbar.total == 4  # 2 tasks * 2 repeats

        # Verify Progress was created and started
        mock_progress_instance.start.assert_called_once()
        mock_progress_instance.add_task.assert_called_once()

        # Task repeat end
        pbar.on_task_repeat_end(mock_benchmark, mock_report)
        assert pbar.current == 1
        assert pbar.successful == 1

        # Verify progress update
        mock_progress_instance.update.assert_called()

        # End
        pbar.on_run_end(mock_benchmark, [])
        mock_progress_instance.stop.assert_called_once()

    @patch("rich.progress.Progress")
    def test_custom_metrics_with_rich(self, mock_progress_class, mock_benchmark):
        """Test that custom metrics work with rich progress bar."""
        mock_progress_instance = MagicMock()
        mock_progress_class.return_value = mock_progress_instance
        mock_progress_instance.add_task.return_value = "task_id_123"

        pbar = RichProgressBarCallback()
        pbar.on_run_start(mock_benchmark)

        # Set custom metrics
        pbar.set_metrics(avg_score="0.89", correct="42/50")

        # Verify update was called with info parameter
        assert "avg_score" in pbar._custom_metrics
        assert "correct" in pbar._custom_metrics


class TestThreadSafety:
    """Test thread safety of progress bar callbacks.

    Progress bars use threading.Lock to prevent race conditions when multiple
    tasks complete simultaneously (e.g., in parallel benchmark execution).
    Without proper locking, concurrent increments to counters like `current`
    and `successful` could be lost, leading to incorrect progress tracking.
    """

    @patch("tqdm.tqdm")
    def test_concurrent_updates(self, mock_tqdm_class, mock_benchmark):
        """Test that concurrent updates don't cause race conditions.

        Verifies that the internal lock properly protects shared state (current,
        successful) from concurrent modification, ensuring no updates are lost
        when multiple threads call on_task_repeat_end() simultaneously.
        """
        import threading
        import time

        mock_tqdm_instance = MagicMock()
        mock_tqdm_class.return_value = mock_tqdm_instance

        pbar = TqdmProgressBarCallback()
        pbar.on_run_start(mock_benchmark)

        def update_progress(task_idx):
            report = {"status": "success"}
            time.sleep(0.001)  # Simulate work
            pbar.on_task_repeat_end(mock_benchmark, report)

        # Run multiple updates concurrently
        threads = []
        for i in range(4):
            t = threading.Thread(target=update_progress, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify all updates were counted (no race condition lost updates)
        assert pbar.current == 4
        assert pbar.successful == 4

        pbar.on_run_end(mock_benchmark, [])
