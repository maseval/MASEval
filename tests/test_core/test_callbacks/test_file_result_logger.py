"""Tests for FileResultLogger.

These tests verify that FileResultLogger correctly writes benchmark results
to JSONL files, with proper filtering of result components based on logger
configuration. FileResultLogger is the standard way to persist benchmark
execution data to disk for later analysis.
"""

import json
from pathlib import Path

import pytest

from maseval.core.callbacks import FileResultLogger
from maseval.core.task import Task


class MockBenchmark:
    """Minimal benchmark shim for testing FileResultLogger.

    Simulates benchmark structure with tasks and repetitions to test
    file output behavior.
    """

    def __init__(self, n_tasks=1, n_repeats=1):
        self.tasks = [Task(query=f"Query {i}") for i in range(n_tasks)]
        self.n_task_repeats = n_repeats
        self.task_ids = [str(t.id) for t in self.tasks]


@pytest.mark.core
def test_file_result_logger_writes_jsonl(tmp_path):
    """Test that FileResultLogger creates JSONL file with filtered reports.

    Verifies that logger writes one JSON object per line and applies
    configured filters to exclude unwanted result components.
    """
    out_dir = tmp_path / "results"
    out_dir.mkdir()

    logger = FileResultLogger(output_dir=str(out_dir), filename_pattern="test_results.jsonl")

    benchmark = MockBenchmark(n_tasks=1, n_repeats=1)
    logger.on_run_start(benchmark)  # type: ignore[arg-type]

    report = {
        "task_id": benchmark.task_ids[0],
        "repeat_idx": 0,
        "traces": {"agent": "trace"},
        "config": {"model": "gpt"},
        "eval": {"score": 1.0},
    }
    logger.on_task_repeat_end(benchmark, report)  # type: ignore[arg-type]
    logger.on_run_end(benchmark, [report])  # type: ignore[arg-type]

    # Ensure file created and contains one JSON object per iteration
    out_file = out_dir / "test_results.jsonl"
    assert out_file.exists()

    lines = out_file.read_text().strip().splitlines()
    assert len(lines) == 1

    obj = json.loads(lines[0])
    assert obj["task_id"] == report["task_id"]
    assert obj["repeat_idx"] == report["repeat_idx"]
    assert "traces" in obj and "config" in obj and "eval" in obj


@pytest.mark.core
def test_file_result_logger_accepts_pathlib_path(tmp_path):
    """Test that FileResultLogger accepts pathlib.Path for output_dir.

    Verifies that the logger works correctly when output_dir is specified
    as a Path object instead of a string.
    """
    out_dir = tmp_path / "results"
    out_dir.mkdir()

    # Pass Path object directly instead of string
    logger = FileResultLogger(output_dir=out_dir, filename_pattern="test_results.jsonl")

    benchmark = MockBenchmark(n_tasks=1, n_repeats=1)
    logger.on_run_start(benchmark)  # type: ignore[arg-type]

    report = {
        "task_id": benchmark.task_ids[0],
        "repeat_idx": 0,
        "traces": {"agent": "trace"},
        "config": {"model": "gpt"},
        "eval": {"score": 1.0},
    }
    logger.on_task_repeat_end(benchmark, report)  # type: ignore[arg-type]
    logger.on_run_end(benchmark, [report])  # type: ignore[arg-type]

    # Verify file was created
    out_file = out_dir / "test_results.jsonl"
    assert out_file.exists()
    assert isinstance(logger.output_dir, Path)

    lines = out_file.read_text().strip().splitlines()
    assert len(lines) == 1


@pytest.mark.core
def test_file_result_logger_overwrite_false_prevents_overwriting(tmp_path):
    """Test that FileResultLogger raises error when file exists and overwrite=False.

    Verifies that when overwrite is False (default), attempting to write to
    an existing file raises FileExistsError.
    """
    out_dir = tmp_path / "results"
    out_dir.mkdir()

    # Create an existing file
    existing_file = out_dir / "test_results.jsonl"
    existing_file.write_text("existing content\n")

    # Try to create logger with overwrite=False (default)
    logger = FileResultLogger(output_dir=out_dir, filename_pattern="test_results.jsonl", overwrite=False)

    benchmark = MockBenchmark(n_tasks=1, n_repeats=1)
    logger.on_run_start(benchmark)  # type: ignore[arg-type]

    report = {
        "task_id": benchmark.task_ids[0],
        "repeat_idx": 0,
        "traces": {"agent": "trace"},
        "config": {"model": "gpt"},
        "eval": {"score": 1.0},
    }

    # Should raise FileExistsError when trying to log first iteration
    with pytest.raises(FileExistsError, match="Output file already exists.*Set overwrite=True"):
        logger.on_task_repeat_end(benchmark, report)  # type: ignore[arg-type]

    # Verify original file is unchanged
    assert existing_file.read_text() == "existing content\n"


@pytest.mark.core
def test_file_result_logger_overwrite_true_allows_overwriting(tmp_path):
    """Test that FileResultLogger overwrites existing file when overwrite=True.

    Verifies that when overwrite is True, the logger successfully overwrites
    an existing file with the same name.
    """
    out_dir = tmp_path / "results"
    out_dir.mkdir()

    # Create an existing file
    existing_file = out_dir / "test_results.jsonl"
    existing_file.write_text("existing content\n")

    # Create logger with overwrite=True
    logger = FileResultLogger(output_dir=out_dir, filename_pattern="test_results.jsonl", overwrite=True)

    benchmark = MockBenchmark(n_tasks=1, n_repeats=1)
    logger.on_run_start(benchmark)  # type: ignore[arg-type]

    report = {
        "task_id": benchmark.task_ids[0],
        "repeat_idx": 0,
        "traces": {"agent": "trace"},
        "config": {"model": "gpt"},
        "eval": {"score": 1.0},
    }
    logger.on_task_repeat_end(benchmark, report)  # type: ignore[arg-type]
    logger.on_run_end(benchmark, [report])  # type: ignore[arg-type]

    # Verify file was overwritten with new content
    lines = existing_file.read_text().strip().splitlines()
    assert len(lines) == 1
    assert "existing content" not in existing_file.read_text()

    obj = json.loads(lines[0])
    assert obj["task_id"] == report["task_id"]
    assert obj["repeat_idx"] == report["repeat_idx"]
