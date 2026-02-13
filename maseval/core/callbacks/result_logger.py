"""Abstract base class for logging benchmark results to various backends.

This module provides a common interface for result logging implementations that write
benchmark results incrementally during execution and validate completeness at the end.
"""

import json
import os
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from ..callback import BenchmarkCallback

if TYPE_CHECKING:
    from ..benchmark import Benchmark


class ResultLogger(BenchmarkCallback, ABC):
    """Abstract base class for logging benchmark results to various backends.

    This class provides a framework for implementing result loggers that:
    - Write results incrementally after each task iteration (repeat)
    - Track expected vs actual logged iterations
    - Validate completeness at benchmark end
    - Support selective logging of traces, config, and eval results

    Subclasses implement specific backends (file, wandb, opentelemetry, etc.) by
    overriding the abstract methods.

    Attributes:
        include_traces: Whether to include execution traces in logged results
        include_config: Whether to include configuration in logged results
        include_eval: Whether to include evaluation results in logged results
        validate_on_completion: Whether to validate all iterations were logged

    Example:
        ```python
        class MyLogger(ResultLogger):
            def log_iteration(self, report: Dict) -> None:
                # Write report to backend
                pass

            def finalize(self) -> None:
                # Close connections, flush buffers
                pass

            def validate(self) -> bool:
                # Check all iterations present
                return True

        logger = MyLogger(include_traces=True)
        benchmark = MyBenchmark(tasks, agent_data, callbacks=[logger])
        benchmark.run()
        ```
    """

    def __init__(
        self,
        include_traces: bool = True,
        include_config: bool = True,
        include_eval: bool = True,
        validate_on_completion: bool = True,
    ):
        """Initialize the result logger.

        Args:
            include_traces: If True, include execution traces in logged results
            include_config: If True, include configuration in logged results
            include_eval: If True, include evaluation results in logged results
            validate_on_completion: If True, validate all iterations were logged at end
        """
        super().__init__()
        self.include_traces = include_traces
        self.include_config = include_config
        self.include_eval = include_eval
        self.validate_on_completion = validate_on_completion

        # Tracking for validation
        self._expected_iterations: Set[Tuple[str, int]] = set()
        self._logged_iterations: Set[Tuple[str, int]] = set()
        self._n_tasks: Optional[int] = None
        self._n_repeats: Optional[int] = None

    def on_run_start(self, benchmark: "Benchmark") -> None:
        """Called when benchmark execution starts.

        Records the expected number of tasks and repeats for validation.

        Args:
            benchmark: The benchmark instance
        """
        self._n_tasks = len(list(benchmark.tasks))
        self._n_repeats = benchmark.n_task_repeats

        # Calculate all expected iterations
        for task in benchmark.tasks:
            for repeat_idx in range(benchmark.n_task_repeats):
                self._expected_iterations.add((task.id, repeat_idx))

    def on_task_repeat_end(self, benchmark: "Benchmark", report: Dict) -> None:
        """Called after each task iteration completes.

        Filters the report based on include flags, logs it, and tracks the iteration.

        Args:
            benchmark: The benchmark instance
            report: The complete report dict with task_id, repeat_idx, traces, config, eval
        """
        # Filter report based on include flags
        filtered_report = self._filter_report(report)

        # Log the iteration to backend
        try:
            self.log_iteration(filtered_report)

            # Track successful log
            task_id = report.get("task_id")
            repeat_idx = report.get("repeat_idx")
            if task_id is not None and repeat_idx is not None:
                self._logged_iterations.add((task_id, repeat_idx))

        except Exception as e:
            print(f"[ResultLogger] Error logging iteration: {e}")
            raise

    def on_run_end(self, benchmark: "Benchmark", results: List[Dict]) -> None:
        """Called when benchmark execution completes.

        Finalizes logging and optionally validates completeness.

        Args:
            benchmark: The benchmark instance
            results: List of all result reports from the benchmark
        """
        # Finalize backend (close files, flush buffers, etc.)
        try:
            self.finalize()
        except Exception as e:
            print(f"[ResultLogger] Error during finalization: {e}")
            raise

        # Validate if enabled
        if self.validate_on_completion:
            is_valid = self.validate()
            if not is_valid:
                self._report_validation_errors()

    def _filter_report(self, report: Dict) -> Dict:
        """Filter report based on include flags.

        Args:
            report: The complete report dict

        Returns:
            Filtered report dict containing only requested fields
        """
        filtered = {
            "task_id": report.get("task_id"),
            "repeat_idx": report.get("repeat_idx"),
        }

        if self.include_traces and "traces" in report:
            filtered["traces"] = report["traces"]

        if self.include_config and "config" in report:
            filtered["config"] = report["config"]

        if self.include_eval and "eval" in report:
            filtered["eval"] = report["eval"]

        return filtered

    def _report_validation_errors(self) -> None:
        """Report validation errors to user."""
        missing = self._expected_iterations - self._logged_iterations
        extra = self._logged_iterations - self._expected_iterations

        error_parts = []
        error_parts.append("[ResultLogger] Validation failed!")
        error_parts.append(f"  Expected iterations: {len(self._expected_iterations)}")
        error_parts.append(f"  Logged iterations: {len(self._logged_iterations)}")

        if missing:
            error_parts.append(f"  Missing {len(missing)} iterations:")
            for task_id, repeat_idx in sorted(missing)[:5]:  # Show first 5
                error_parts.append(f"    - task_id={task_id}, repeat_idx={repeat_idx}")
            if len(missing) > 5:
                error_parts.append(f"    ... and {len(missing) - 5} more")

        if extra:
            error_parts.append(f"  Unexpected {len(extra)} iterations:")
            for task_id, repeat_idx in sorted(extra)[:5]:  # Show first 5
                error_parts.append(f"    - task_id={task_id}, repeat_idx={repeat_idx}")
            if len(extra) > 5:
                error_parts.append(f"    ... and {len(extra) - 5} more")

        print("\n".join(error_parts))

    @abstractmethod
    def log_iteration(self, report: Dict) -> None:
        """Log a single task iteration to the backend.

        This method is called after each task repeat completes. Implementations
        should write the report to their specific backend (file, API, etc.).

        Args:
            report: Filtered report dict containing task_id, repeat_idx, and
                optionally traces, config, and eval based on include flags

        Raises:
            Exception: If logging fails (will be caught and re-raised by base class)
        """
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Finalize logging operations.

        Called at benchmark end. Implementations should:
        - Close file handles
        - Flush buffers
        - Close network connections
        - Write metadata files
        - Perform any cleanup operations

        Raises:
            Exception: If finalization fails (will be caught and re-raised by base class)
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate that all expected iterations were logged correctly.

        Called at benchmark end if validate_on_completion is True. Implementations
        should verify:
        - All expected iterations are present
        - No duplicate iterations exist
        - Data integrity is maintained

        Returns:
            True if validation passes, False otherwise
        """
        pass


class FileResultLogger(ResultLogger):
    """Logger that writes benchmark results incrementally to JSONL files.

    This logger writes each task iteration to a JSONL file (one JSON object per line)
    as soon as it completes. This provides:
    - Recovery from crashes: partial results are preserved
    - Streaming analysis: results can be read while benchmark is running
    - Safe concurrent reads: JSONL format is line-atomic
    - Validation: ensures all expected iterations were written

    The logger uses atomic writes (write to temp file, then rename) to prevent
    file corruption from crashes or interruptions.

    Attributes:
        output_dir: Directory where result files will be written
        filename_pattern: Pattern for result filename (supports {timestamp})
        write_metadata: Whether to write a metadata file with benchmark info
        atomic_writes: Whether to use atomic writes (recommended)

    Example:
        ```python
        from maseval.core.callbacks.result_logger import FileResultLogger

        # Basic usage
        logger = FileResultLogger(output_dir="./results")

        # Custom configuration
        logger = FileResultLogger(
            output_dir="./results",
            filename_pattern="benchmark_{timestamp}.jsonl",
            include_traces=True,
            include_config=True,
            validate_on_completion=True
        )

        # Use with benchmark
        benchmark = MyBenchmark(
            tasks=tasks,
            agent_data=agent_data,
            callbacks=[logger]
        )
        results = benchmark.run()

        # Results are written to: ./results/benchmark_20251028_143022.jsonl
        ```
    """

    def __init__(
        self,
        output_dir: Path | str = "./results",
        filename_pattern: str = "benchmark_{timestamp}.jsonl",
        write_metadata: bool = True,
        atomic_writes: bool = True,
        overwrite: bool = False,
        include_traces: bool = True,
        include_config: bool = True,
        include_eval: bool = True,
        validate_on_completion: bool = True,
    ):
        """Initialize the file logger.

        Args:
            output_dir: Directory where result files will be written (created if needed).
                Accepts either a Path object or a string path.
            filename_pattern: Pattern for result filename. Use {timestamp} for
                automatic timestamp insertion (format: YYYYMMDD_HHMMSS)
            write_metadata: If True, write a metadata file alongside results
            atomic_writes: If True, use atomic writes (write to temp, then rename)
            overwrite: If True, overwrite existing files. If False, raise an error
                when the output file already exists.
            include_traces: If True, include execution traces in logged results
            include_config: If True, include configuration in logged results
            include_eval: If True, include evaluation results in logged results
            validate_on_completion: If True, validate all iterations were logged
        """
        super().__init__(
            include_traces=include_traces,
            include_config=include_config,
            include_eval=include_eval,
            validate_on_completion=validate_on_completion,
        )

        self.output_dir = Path(output_dir)
        self.filename_pattern = filename_pattern
        self.write_metadata = write_metadata
        self.atomic_writes = atomic_writes
        self.overwrite = overwrite

        # Runtime state
        self._output_path: Optional[Path] = None
        self._file_handle = None  # type: ignore[assignment]
        self._timestamp: Optional[str] = None
        self._lines_written: int = 0

    def log_iteration(self, report: Dict) -> None:
        """Log a single task iteration to the JSONL file.

        Args:
            report: Filtered report dict to write

        Raises:
            IOError: If writing to file fails
        """
        # Lazy initialization of file on first write
        if self._file_handle is None:
            self._initialize_output_file()

        # Serialize to JSON
        json_line = json.dumps(report, default=str) + "\n"

        # Write to file
        if self.atomic_writes:
            self._write_atomic(json_line)
        else:
            self._file_handle.write(json_line)  # type: ignore[union-attr]
            self._file_handle.flush()  # type: ignore[union-attr]

        self._lines_written += 1

    def finalize(self) -> None:
        """Finalize logging by closing files and writing metadata.

        Raises:
            IOError: If file operations fail
        """
        # Close file handle
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

        # Write metadata if enabled
        if self.write_metadata and self._output_path is not None:
            self._write_metadata()

    def validate(self) -> bool:
        """Validate that all expected iterations were written to file.

        Checks:
        1. Number of lines matches number of logged iterations
        2. All expected iterations are present
        3. No duplicate iterations exist

        Returns:
            True if validation passes, False otherwise
        """
        # Check file exists
        if self._output_path is None or not self._output_path.exists():
            print("[FileResultLogger] Validation failed: output file not found")
            return False

        # Read all lines from file
        try:
            with open(self._output_path, "r") as f:
                lines = f.readlines()
        except IOError as e:
            print(f"[FileResultLogger] Validation failed: could not read file: {e}")
            return False

        # Check line count matches
        if len(lines) != self._lines_written:
            print(f"[FileResultLogger] Validation warning: expected {self._lines_written} lines, found {len(lines)}")

        # Parse and check iterations
        file_iterations = set()
        for line_num, line in enumerate(lines, 1):
            try:
                data = json.loads(line)
                task_id = data.get("task_id")
                repeat_idx = data.get("repeat_idx")

                if task_id is None or repeat_idx is None:
                    print(f"[FileResultLogger] Validation failed: line {line_num} missing task_id or repeat_idx")
                    return False

                iteration = (task_id, repeat_idx)
                if iteration in file_iterations:
                    print(f"[FileResultLogger] Validation failed: duplicate iteration {iteration} at line {line_num}")
                    return False

                file_iterations.add(iteration)

            except json.JSONDecodeError as e:
                print(f"[FileResultLogger] Validation failed: invalid JSON at line {line_num}: {e}")
                return False

        # Check all expected iterations present
        missing = self._expected_iterations - file_iterations
        if missing:
            print(f"[FileResultLogger] Validation failed: {len(missing)} iterations missing from file")
            return False

        # Check no extra iterations
        extra = file_iterations - self._expected_iterations
        if extra:
            print(f"[FileResultLogger] Validation failed: {len(extra)} unexpected iterations in file")
            return False

        return True

    def _initialize_output_file(self) -> None:
        """Initialize the output file and directory structure.

        Raises:
            IOError: If file or directory creation fails
            FileExistsError: If output file exists and overwrite is False
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate filename
        filename = self.filename_pattern.replace("{timestamp}", self._timestamp)
        self._output_path = self.output_dir / filename

        # Check if file exists and handle overwrite
        if self._output_path.exists() and not self.overwrite:
            raise FileExistsError(f"Output file already exists: {self._output_path}. Set overwrite=True to allow overwriting existing files.")

        # Open file for writing
        self._file_handle = open(self._output_path, "w")

    def _write_atomic(self, content: str) -> None:
        """Write content atomically by writing to temp file then appending.

        Args:
            content: Content to write

        Raises:
            IOError: If write operations fail
        """
        # For append operations, we write to temp then append to main file
        # This ensures we never leave partial lines in the main file
        with tempfile.NamedTemporaryFile(mode="w", dir=self.output_dir, delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name

        try:
            # Read temp file and append to main file
            with open(tmp_path, "r") as tmp_read:
                self._file_handle.write(tmp_read.read())  # type: ignore[union-attr]
                self._file_handle.flush()  # type: ignore[union-attr]
                os.fsync(self._file_handle.fileno())  # type: ignore[union-attr]
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    def _write_metadata(self) -> None:
        """Write metadata file with benchmark information.

        Raises:
            IOError: If metadata file write fails
        """
        if self._output_path is None:
            return

        metadata = {
            "output_file": str(self._output_path.name),
            "timestamp": self._timestamp,
            "n_tasks": self._n_tasks,
            "n_repeats": self._n_repeats,
            "total_iterations": len(self._expected_iterations),
            "lines_written": self._lines_written,
            "include_traces": self.include_traces,
            "include_config": self.include_config,
            "include_eval": self.include_eval,
            "validation_enabled": self.validate_on_completion,
        }

        # Write metadata to .meta.json file
        meta_path = self._output_path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
