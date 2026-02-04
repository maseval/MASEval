#!/usr/bin/env python3
"""Generate test coverage report organized by feature area.

Automatically discovers benchmarks and integrations from the codebase structure.
Provides a high-level view of coverage by logical component rather than by file.

Usage:
    uv run python scripts/coverage_by_feature.py
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set


def discover_benchmarks(maseval_dir: Path) -> List[str]:
    """Auto-discover benchmark implementations."""
    benchmark_dir = maseval_dir / "benchmark"
    if not benchmark_dir.exists():
        return []

    benchmarks = []
    for item in benchmark_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            benchmarks.append(item.name)

    return sorted(benchmarks)


def discover_integrations(maseval_dir: Path) -> Dict[str, Dict[str, List[str]]]:
    """Auto-discover interface integrations (agents, inference, etc.).

    Groups related files together (e.g., smolagents.py and smolagents_optional.py -> smolagents).

    Returns:
        Dict mapping category -> integration_name -> list of file stems
    """
    interface_dir = maseval_dir / "interface"
    if not interface_dir.exists():
        return {}

    integrations = {}
    for category_dir in interface_dir.iterdir():
        if category_dir.is_dir() and not category_dir.name.startswith("_"):
            category_name = category_dir.name
            files = []
            for item in category_dir.iterdir():
                # Look for .py files (not __init__.py or __pycache__)
                if item.is_file() and item.suffix == ".py" and not item.name.startswith("_"):
                    files.append(item.stem)

            if files:
                # Group related files (e.g., smolagents and smolagents_optional)
                grouped = {}
                for file_stem in files:
                    # Extract base name (before _optional, _extra, etc.)
                    base_name = file_stem.split("_")[0] if "_" in file_stem else file_stem
                    if base_name not in grouped:
                        grouped[base_name] = []
                    grouped[base_name].append(file_stem)

                integrations[category_name] = grouped

    return integrations


def run_coverage() -> bool:
    """Run pytest with coverage collection."""
    print("Running tests with coverage...")
    result = subprocess.run(
        ["pytest", "--cov=maseval", "--cov-report=json", "--quiet"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("\nTests failed:")
        print(result.stdout)
        print(result.stderr)
    return result.returncode == 0


def load_coverage_data() -> Dict:
    """Load coverage data from JSON report."""
    coverage_file = Path("coverage.json")
    if not coverage_file.exists():
        print("Error: coverage.json not found. Run with --cov-report=json")
        sys.exit(1)

    with open(coverage_file) as f:
        return json.load(f)


def calculate_coverage(files: Set[str], coverage_data: Dict) -> Dict[str, float]:
    """Calculate coverage statistics for a set of files.

    Returns:
        Dict with 'covered', 'total', and 'percent' keys
    """
    covered_lines = 0
    total_lines = 0

    for file_path, file_data in coverage_data["files"].items():
        if any(target in file_path for target in files):
            summary = file_data["summary"]
            covered_lines += summary["covered_lines"]
            total_lines += summary["num_statements"]

    percent = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0

    return {
        "covered": covered_lines,
        "total": total_lines,
        "percent": percent,
    }


def format_coverage(label: str, stats: Dict[str, float], indent: int = 0) -> str:
    """Format coverage statistics for display."""
    indent_str = "  " * indent
    percent = stats["percent"]

    # Color coding
    if percent >= 80:
        color = "\033[92m"  # Green
    elif percent >= 60:
        color = "\033[93m"  # Yellow
    else:
        color = "\033[91m"  # Red
    reset = "\033[0m"

    return f"{indent_str}{label:<30} {color}{percent:6.2f}%{reset}  ({stats['covered']}/{stats['total']} lines)"


def main():
    """Generate coverage report by feature area."""
    repo_root = Path(__file__).parent.parent
    maseval_dir = repo_root / "maseval"

    # Run coverage
    if not run_coverage():
        print("\nTests failed. Coverage report may be incomplete.")

    print("\n" + "=" * 80)
    print("COVERAGE BY FEATURE AREA")
    print("=" * 80 + "\n")

    # Load coverage data
    coverage_data = load_coverage_data()

    # Auto-discover components
    benchmarks = discover_benchmarks(maseval_dir)
    integrations = discover_integrations(maseval_dir)

    # Overall coverage
    total_stats = calculate_coverage({"maseval/"}, coverage_data)
    print(format_coverage("OVERALL", total_stats))
    print()

    # Benchmarks
    benchmark_files = {f"maseval/benchmark/{b}/" for b in benchmarks}
    all_benchmark_stats = calculate_coverage(benchmark_files, coverage_data)
    print(format_coverage("Benchmarks", all_benchmark_stats))

    for benchmark in benchmarks:
        benchmark_stats = calculate_coverage({f"maseval/benchmark/{benchmark}/"}, coverage_data)
        print(format_coverage(benchmark.upper(), benchmark_stats, indent=1))
    print()

    # Core
    core_stats = calculate_coverage({"maseval/core/"}, coverage_data)
    print(format_coverage("Core", core_stats))

    # Group related core modules
    core_groups = {
        "Agent": ["agent", "agentic_user"],
        "User": ["user"],
        "Simulation": ["simulator", "environment"],
        "Evaluation": ["evaluator"],
        "Seeding": ["seeding"],
        "Infrastructure": ["benchmark", "callback", "callback_handler", "config", "exceptions", "history", "model", "task", "tracing"],
    }

    # Calculate grouped coverage
    for group_name, modules in core_groups.items():
        module_paths = {f"maseval/core/{m}.py" for m in modules}
        group_stats = calculate_coverage(module_paths, coverage_data)
        if group_stats["total"] > 0:
            print(format_coverage(group_name, group_stats, indent=1))
    print()

    # Interfaces
    if integrations:
        interface_stats = calculate_coverage({"maseval/interface/"}, coverage_data)
        print(format_coverage("Interfaces", interface_stats))

        for category, grouped_items in integrations.items():
            category_stats = calculate_coverage({f"maseval/interface/{category}/"}, coverage_data)
            print(format_coverage(f"{category.title()}", category_stats, indent=1))

            for base_name, file_stems in sorted(grouped_items.items()):
                # Calculate coverage for all files in this group
                file_paths = {f"maseval/interface/{category}/{stem}.py" for stem in file_stems}
                item_stats = calculate_coverage(file_paths, coverage_data)
                if item_stats["total"] > 0:
                    print(format_coverage(base_name, item_stats, indent=2))
        print()

    print("=" * 80)

    # Clean up coverage.json
    coverage_file = Path("coverage.json")
    if coverage_file.exists():
        coverage_file.unlink()


if __name__ == "__main__":
    main()
