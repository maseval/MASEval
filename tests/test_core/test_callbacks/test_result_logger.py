"""Tests for base ResultLogger class.

These tests verify that ResultLogger correctly orchestrates result logging,
validation, and lifecycle management. The ResultLogger is a benchmark callback
that logs task execution results and validates that all expected iterations
complete successfully.
"""

import pytest
from typing import Dict

from maseval.core.callbacks.result_logger import ResultLogger
from maseval.core.task import Task


class MockResultLogger(ResultLogger):
    """Mock implementation for testing base ResultLogger functionality.

    This mock tracks which methods are called and stores logged reports
    to enable verification of the base class orchestration logic.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logged_reports = []
        self.finalize_called = False
        self.validate_called = False

    def log_iteration(self, report: Dict) -> None:
        self.logged_reports.append(report)

    def finalize(self) -> None:
        self.finalize_called = True

    def validate(self) -> bool:
        self.validate_called = True
        # Check all expected iterations were logged
        logged_iterations = {(r["task_id"], r["repeat_idx"]) for r in self.logged_reports}
        return logged_iterations == self._expected_iterations


class MockBenchmark:
    """Mock benchmark for testing ResultLogger with configurable task counts.

    Simulates benchmark structure with tasks and repetition settings to test
    logger's ability to track expected vs actual iterations.
    """

    def __init__(self, n_tasks=2, n_repeats=3):
        self.tasks = [Task(query=f"Query {i}") for i in range(n_tasks)]
        self.n_task_repeats = n_repeats
        # Store string IDs for easier testing
        self.task_ids = [str(task.id) for task in self.tasks]


@pytest.mark.core
class TestResultLogger:
    """Tests for ResultLogger base class."""

    def test_initialization(self):
        """Test logger initialization with default parameters.

        Verifies that ResultLogger starts with expected defaults: all result
        components included, validation enabled, and empty tracking sets.
        """
        logger = MockResultLogger()

        assert logger.include_traces is True
        assert logger.include_config is True
        assert logger.include_eval is True
        assert logger.validate_on_completion is True
        assert len(logger._expected_iterations) == 0
        assert len(logger._logged_iterations) == 0

    def test_initialization_custom_params(self):
        """Test logger initialization with custom filtering parameters.

        Verifies that logger respects custom settings for excluding traces,
        config, eval, or validation from logged results.
        """
        logger = MockResultLogger(
            include_traces=False,
            include_config=False,
            include_eval=False,
            validate_on_completion=False,
        )

        assert logger.include_traces is False
        assert logger.include_config is False
        assert logger.include_eval is False
        assert logger.validate_on_completion is False

    def test_on_run_start_records_expected_iterations(self):
        """Test that on_run_start computes and stores expected iterations.

        Verifies that logger calculates the complete set of (task_id, repeat_idx)
        pairs that should be logged based on benchmark configuration.
        """
        logger = MockResultLogger()
        benchmark = MockBenchmark(n_tasks=2, n_repeats=3)

        logger.on_run_start(benchmark)  # type: ignore[arg-type]

        assert logger._n_tasks == 2
        assert logger._n_repeats == 3
        assert len(logger._expected_iterations) == 6  # 2 tasks * 3 repeats

        # Verify all task_id/repeat_idx combinations are present
        for task_id in benchmark.task_ids:
            for repeat_idx in range(3):
                assert (task_id, repeat_idx) in logger._expected_iterations

    def test_filter_report_all_included(self):
        """Test report filtering when all components are included.

        Verifies that when include_traces, include_config, and include_eval
        are all True, the full report is passed through unchanged.
        """
        logger = MockResultLogger(include_traces=True, include_config=True, include_eval=True)

        report = {
            "task_id": "task_0",
            "repeat_idx": 0,
            "traces": {"agent": "trace_data"},
            "config": {"model": "gpt-4"},
            "eval": {"score": 0.95},
        }

        filtered = logger._filter_report(report)

        assert filtered["task_id"] == "task_0"
        assert filtered["repeat_idx"] == 0
        assert "traces" in filtered
        assert "config" in filtered
        assert "eval" in filtered

    def test_filter_report_preserves_status_and_error(self):
        """Test that status and error fields are always included in filtered reports.

        These are core metadata fields (like task_id and repeat_idx) that must
        always be present so persisted results can distinguish successes from failures.
        """
        logger = MockResultLogger(include_traces=False, include_config=False, include_eval=False)

        report = {
            "task_id": "task_0",
            "repeat_idx": 0,
            "status": "agent_error",
            "error": {"type": "AgentError", "message": "Tool call failed"},
            "traces": {"agent": "trace_data"},
            "config": {"model": "gpt-4"},
            "eval": {"score": 0.0},
        }

        filtered = logger._filter_report(report)

        assert filtered["status"] == "agent_error"
        assert filtered["error"] == {"type": "AgentError", "message": "Tool call failed"}
        assert "traces" not in filtered
        assert "config" not in filtered
        assert "eval" not in filtered

    def test_filter_report_status_and_error_absent(self):
        """Test that missing status/error fields result in None values."""
        logger = MockResultLogger()

        report = {
            "task_id": "task_0",
            "repeat_idx": 0,
        }

        filtered = logger._filter_report(report)

        assert filtered["status"] is None
        assert filtered["error"] is None

    def test_filter_report_partial_included(self):
        """Test report filtering with only some fields included."""
        logger = MockResultLogger(include_traces=False, include_config=True, include_eval=False)

        report = {
            "task_id": "task_0",
            "repeat_idx": 0,
            "traces": {"agent": "trace_data"},
            "config": {"model": "gpt-4"},
            "eval": {"score": 0.95},
        }

        filtered = logger._filter_report(report)

        assert filtered["task_id"] == "task_0"
        assert filtered["repeat_idx"] == 0
        assert "traces" not in filtered
        assert "config" in filtered
        assert "eval" not in filtered

    def test_on_task_repeat_end_logs_iteration(self):
        """Test that on_task_repeat_end logs iterations correctly."""
        logger = MockResultLogger()
        benchmark = MockBenchmark(n_tasks=1, n_repeats=2)

        logger.on_run_start(benchmark)  # type: ignore[arg-type]

        report = {
            "task_id": benchmark.task_ids[0],
            "repeat_idx": 0,
            "traces": {},
            "config": {},
            "eval": {},
        }

        logger.on_task_repeat_end(benchmark, report)  # type: ignore[arg-type]

        assert len(logger.logged_reports) == 1
        assert len(logger._logged_iterations) == 1
        assert (benchmark.task_ids[0], 0) in logger._logged_iterations

    def test_on_run_end_calls_finalize_and_validate(self):
        """Test that on_run_end calls finalize and validate."""
        logger = MockResultLogger(validate_on_completion=True)
        benchmark = MockBenchmark(n_tasks=1, n_repeats=1)

        logger.on_run_start(benchmark)  # type: ignore[arg-type]

        # Log one iteration
        report = {
            "task_id": benchmark.task_ids[0],
            "repeat_idx": 0,
            "traces": {},
            "config": {},
            "eval": {},
        }
        logger.on_task_repeat_end(benchmark, report)  # type: ignore[arg-type]

        # End run
        logger.on_run_end(benchmark, [report])  # type: ignore[arg-type]

        assert logger.finalize_called is True
        assert logger.validate_called is True

    def test_on_run_end_skips_validate_if_disabled(self):
        """Test that on_run_end skips validation if disabled."""
        logger = MockResultLogger(validate_on_completion=False)
        benchmark = MockBenchmark(n_tasks=1, n_repeats=1)

        logger.on_run_start(benchmark)  # type: ignore[arg-type]

        report = {
            "task_id": benchmark.task_ids[0],
            "repeat_idx": 0,
            "traces": {},
            "config": {},
            "eval": {},
        }
        logger.on_task_repeat_end(benchmark, report)  # type: ignore[arg-type]

        logger.on_run_end(benchmark, [report])  # type: ignore[arg-type]

        assert logger.finalize_called is True
        assert logger.validate_called is False

    def test_validation_success(self):
        """Test validation succeeds when all iterations logged."""
        logger = MockResultLogger()
        benchmark = MockBenchmark(n_tasks=2, n_repeats=2)

        logger.on_run_start(benchmark)  # type: ignore[arg-type]

        # Log all 4 iterations
        for task_id in benchmark.task_ids:
            for repeat_idx in [0, 1]:
                report = {
                    "task_id": task_id,
                    "repeat_idx": repeat_idx,
                    "traces": {},
                    "config": {},
                    "eval": {},
                }
                logger.on_task_repeat_end(benchmark, report)  # type: ignore[arg-type]

        result = logger.validate()
        assert result is True

    def test_validation_failure_missing_iterations(self):
        """Test validation fails when iterations are missing."""
        logger = MockResultLogger()
        benchmark = MockBenchmark(n_tasks=2, n_repeats=2)

        logger.on_run_start(benchmark)  # type: ignore[arg-type]

        # Log only 2 of 4 iterations
        for repeat_idx in [0, 1]:
            report = {
                "task_id": benchmark.task_ids[0],
                "repeat_idx": repeat_idx,
                "traces": {},
                "config": {},
                "eval": {},
            }
            logger.on_task_repeat_end(benchmark, report)  # type: ignore[arg-type]

        result = logger.validate()
        assert result is False
