"""Post-hoc usage reporting utilities.

This module provides ``UsageReporter`` for slicing and analyzing usage data
from benchmark reports. Unlike the registry's live aggregates (which provide
running totals), the reporter can slice by task since it sees the full report
list with task IDs.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .usage import Usage, TokenUsage


class UsageReporter:
    """Post-hoc utility for analyzing usage across benchmark reports.

    Walks ``report["usage"]`` across all reports to produce breakdowns
    by task, component, model, etc.

    Example:
        ```python
        reporter = UsageReporter.from_reports(benchmark.reports)
        print(reporter.total())
        print(reporter.by_task())
        print(reporter.by_component())
        ```
    """

    def __init__(self, entries: List[Dict[str, Any]]):
        """Initialize with raw entries extracted from reports.

        Args:
            entries: List of dicts, each with ``"task_id"``, ``"repeat_idx"``,
                and ``"usage_items"`` (list of ``(key, usage_dict)`` tuples).
        """
        self._entries = entries

    @staticmethod
    def from_reports(reports: List[Dict[str, Any]]) -> UsageReporter:
        """Create a UsageReporter from benchmark reports.

        Args:
            reports: The ``benchmark.reports`` list.

        Returns:
            A UsageReporter ready for analysis.
        """
        entries = []
        for report in reports:
            usage_data = report.get("usage")
            if not usage_data or "error" in usage_data:
                continue

            usage_items = []
            for category, value in usage_data.items():
                if category == "metadata":
                    continue
                if isinstance(value, dict) and "cost" in value:
                    # Direct value (environment/user) — it's a usage dict
                    usage_items.append((category, value))
                elif isinstance(value, dict):
                    # Category dict with component names as keys
                    for comp_name, comp_usage in value.items():
                        if isinstance(comp_usage, dict) and "error" not in comp_usage:
                            usage_items.append((f"{category}:{comp_name}", comp_usage))

            entries.append(
                {
                    "task_id": report.get("task_id"),
                    "repeat_idx": report.get("repeat_idx"),
                    "usage_items": usage_items,
                }
            )

        return UsageReporter(entries)

    @staticmethod
    def _usage_from_dict(d: Dict[str, Any]) -> Usage:
        """Reconstruct a Usage (or TokenUsage) from a serialized dict."""
        has_tokens = "input_tokens" in d
        if has_tokens:
            return TokenUsage(
                cost=d.get("cost"),
                units=d.get("units", {}),
                provider=d.get("provider"),
                category=d.get("category"),
                component_name=d.get("component_name"),
                kind=d.get("kind"),
                input_tokens=d.get("input_tokens", 0),
                output_tokens=d.get("output_tokens", 0),
                total_tokens=d.get("total_tokens", 0),
                cached_input_tokens=d.get("cached_input_tokens", 0),
                reasoning_tokens=d.get("reasoning_tokens", 0),
                audio_tokens=d.get("audio_tokens", 0),
            )
        return Usage(
            cost=d.get("cost"),
            units=d.get("units", {}),
            provider=d.get("provider"),
            category=d.get("category"),
            component_name=d.get("component_name"),
            kind=d.get("kind"),
        )

    def by_task(self) -> Dict[str, Usage]:
        """Aggregate usage by task_id across all repetitions."""
        result: Dict[str, Usage] = {}
        for entry in self._entries:
            task_id = entry["task_id"]
            for _key, usage_dict in entry["usage_items"]:
                usage = self._usage_from_dict(usage_dict)
                if task_id in result:
                    result[task_id] = result[task_id] + usage
                else:
                    result[task_id] = usage
        return result

    def by_component(self) -> Dict[str, Usage]:
        """Aggregate usage by registry key (e.g., ``"models:main_model"``)."""
        result: Dict[str, Usage] = {}
        for entry in self._entries:
            for key, usage_dict in entry["usage_items"]:
                usage = self._usage_from_dict(usage_dict)
                if key in result:
                    result[key] = result[key] + usage
                else:
                    result[key] = usage
        return result

    def total(self) -> Usage:
        """Grand total across all tasks and components."""
        all_usages = []
        for entry in self._entries:
            for _key, usage_dict in entry["usage_items"]:
                all_usages.append(self._usage_from_dict(usage_dict))
        if not all_usages:
            return Usage()
        return sum(all_usages, Usage())

    def summary(self) -> Dict[str, Any]:
        """Nested dict with all breakdowns."""
        return {
            "total": self.total().to_dict(),
            "by_task": {k: v.to_dict() for k, v in self.by_task().items()},
            "by_component": {k: v.to_dict() for k, v in self.by_component().items()},
        }
