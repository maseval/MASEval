"""Tests for DAF agent adapter integration.

These tests validate the DAFAgentAdapter and related components
without requiring a running DAF backend.
"""

import pytest
from unittest.mock import MagicMock
from typing import Any, Dict


# =============================================================================
# DAF Adapter Tests (Offline - No DAF Backend Required)
# =============================================================================


class TestDAFAgentAdapter:
    """Tests for DAFAgentAdapter with callable agents (no DAF backend)."""

    def test_adapter_with_callable_agent(self):
        """Test adapter works with a simple callable agent."""
        from maseval.interface.agents.daf import DAFAgentAdapter

        # Create a simple callable agent
        def mock_agent(query: str, config: Dict[str, Any]) -> str:
            return f"Response to: {query}"

        adapter = DAFAgentAdapter(
            agent_instance=mock_agent,
            name="test_agent",
        )

        # Run
        result = adapter.run("Hello world")

        assert result == "Response to: Hello world"
        assert adapter.name == "test_agent"

    def test_adapter_with_dict_response(self):
        """Test adapter handles dict responses."""
        from maseval.interface.agents.daf import DAFAgentAdapter

        def mock_agent(query: str, config: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "response": f"Answer to: {query}",
                "tokens": 100,
                "skills": ["skill_a", "skill_b"],
            }

        adapter = DAFAgentAdapter(
            agent_instance=mock_agent,
            name="dict_agent",
        )

        result = adapter.run("Test query")

        assert result == "Answer to: Test query"
        assert adapter._execution_metrics.get("total_tokens") == 100
        assert adapter._execution_metrics.get("skills_invoked") == ["skill_a", "skill_b"]

    def test_adapter_gathers_traces(self):
        """Test adapter gathers traces correctly."""
        from maseval.interface.agents.daf import DAFAgentAdapter

        def mock_agent(query: str, config: Dict[str, Any]) -> str:
            return "Test response"

        adapter = DAFAgentAdapter(
            agent_instance=mock_agent,
            name="trace_agent",
            skill_id="test_skill",
            team_config={"domain": "retail"},
        )

        adapter.run("Test")

        traces = adapter.gather_traces()

        assert "daf_skill_id" in traces
        assert traces["daf_skill_id"] == "test_skill"
        assert "daf_team_config" in traces
        assert traces["daf_team_config"]["domain"] == "retail"
        assert "daf_duration_seconds" in traces

    def test_adapter_gathers_config(self):
        """Test adapter gathers config correctly."""
        from maseval.interface.agents.daf import DAFAgentAdapter

        def mock_agent(query: str, config: Dict[str, Any]) -> str:
            return "Test"

        adapter = DAFAgentAdapter(
            agent_instance=mock_agent,
            name="config_agent",
            skill_id="my_skill",
        )

        config = adapter.gather_config()

        assert config["name"] == "config_agent"
        assert config["daf_skill_id"] == "my_skill"

    def test_adapter_get_messages(self):
        """Test adapter returns message history."""
        from maseval.interface.agents.daf import DAFAgentAdapter

        def mock_agent(query: str, config: Dict[str, Any]) -> str:
            return "Hello from agent"

        adapter = DAFAgentAdapter(
            agent_instance=mock_agent,
            name="msg_agent",
        )

        adapter.run("Query")

        messages = adapter.get_messages()
        message_list = list(messages)

        assert len(message_list) > 0
        # Last message should be the response
        last_msg = message_list[-1]
        assert last_msg["role"] == "assistant"
        assert "Hello from agent" in last_msg["content"]

    def test_adapter_handles_exceptions(self):
        """Test adapter handles agent exceptions gracefully."""
        from maseval.interface.agents.daf import DAFAgentAdapter

        def failing_agent(query: str, config: Dict[str, Any]) -> str:
            raise ValueError("Agent failed")

        adapter = DAFAgentAdapter(
            agent_instance=failing_agent,
            name="failing_agent",
        )

        with pytest.raises(ValueError, match="Agent failed"):
            adapter.run("Test")

        # Check that error was recorded in events
        assert len(adapter._events) > 0
        assert adapter._events[0]["type"] == "error"


# =============================================================================
# Evaluator Tests (inline implementation for test isolation)
# =============================================================================


class InlineAccuracyEvaluator:
    """Inline evaluator for testing (mirrors the one in daf_benchmark.py)."""

    def __init__(self, name: str = "accuracy_evaluator"):
        self.name = name

    def evaluate(self, environment, agents, user=None):
        if not agents:
            return {"score": 0.0, "error": "No agents provided"}

        agent = agents[0]
        messages = agent.get_messages()
        message_list = list(messages)

        if not message_list:
            return {"score": 0.0, "error": "No messages from agent"}

        response = ""
        for msg in reversed(message_list):
            if msg.get("role") == "assistant" and msg.get("content"):
                response = msg["content"]
                break

        expected = environment.task_metadata.get("expected", "")
        category = environment.task_metadata.get("category", "factual")

        response_lower = response.lower().strip()

        if category == "factual" and isinstance(expected, str):
            if expected.lower() in response_lower:
                return {"score": 1.0, "match_type": "exact_match"}
            return {"score": 0.0, "match_type": "no_match"}

        elif category == "reasoning" and isinstance(expected, str):
            import re

            matches = re.findall(r"-?\d+\.?\d*", response.replace(",", ""))
            if expected in matches:
                return {"score": 1.0, "match_type": "numeric_match"}
            return {"score": 0.0, "match_type": "no_numeric_match"}

        elif category == "explanation" and isinstance(expected, list):
            found = sum(1 for kw in expected if kw.lower() in response_lower)
            score = found / len(expected) if expected else 0.0
            return {"score": score, "match_type": "keyword_match"}

        elif category == "coding" and isinstance(expected, str):
            if expected.lower() in response_lower:
                return {"score": 1.0, "match_type": "code_present"}
            return {"score": 0.0, "match_type": "no_code"}

        return {"score": 0.5 if response.strip() else 0.0, "match_type": "default"}


class InlineSimpleEnvironment:
    """Inline environment for testing."""

    def __init__(self, task_metadata):
        self.task_metadata = task_metadata


class TestAccuracyEvaluator:
    """Tests for accuracy evaluation logic."""

    def test_factual_exact_match(self):
        """Test exact match for factual questions."""
        evaluator = InlineAccuracyEvaluator()
        env = InlineSimpleEnvironment({"expected": "Paris", "category": "factual"})

        mock_agent = MagicMock()
        mock_agent.get_messages.return_value = [{"role": "assistant", "content": "The capital of France is Paris."}]

        result = evaluator.evaluate(env, [mock_agent])

        assert result["score"] == 1.0
        assert result["match_type"] == "exact_match"

    def test_factual_no_match(self):
        """Test no match for incorrect factual answer."""
        evaluator = InlineAccuracyEvaluator()
        env = InlineSimpleEnvironment({"expected": "Paris", "category": "factual"})

        mock_agent = MagicMock()
        mock_agent.get_messages.return_value = [{"role": "assistant", "content": "London is the capital."}]

        result = evaluator.evaluate(env, [mock_agent])

        assert result["score"] == 0.0

    def test_reasoning_numeric_match(self):
        """Test numeric match for math problems."""
        evaluator = InlineAccuracyEvaluator()
        env = InlineSimpleEnvironment({"expected": "150", "category": "reasoning"})

        mock_agent = MagicMock()
        mock_agent.get_messages.return_value = [{"role": "assistant", "content": "The distance is 150 miles."}]

        result = evaluator.evaluate(env, [mock_agent])

        assert result["score"] == 1.0
        assert result["match_type"] == "numeric_match"

    def test_explanation_keyword_match(self):
        """Test keyword match for explanations."""
        evaluator = InlineAccuracyEvaluator()
        env = InlineSimpleEnvironment(
            {
                "expected": ["recursion", "function", "itself"],
                "category": "explanation",
            }
        )

        mock_agent = MagicMock()
        mock_agent.get_messages.return_value = [{"role": "assistant", "content": "Recursion is when a function calls itself."}]

        result = evaluator.evaluate(env, [mock_agent])

        assert result["score"] == 1.0  # All 3 keywords found
        assert result["match_type"] == "keyword_match"

    def test_coding_code_present(self):
        """Test code presence for coding tasks."""
        evaluator = InlineAccuracyEvaluator()
        env = InlineSimpleEnvironment({"expected": "def", "category": "coding"})

        mock_agent = MagicMock()
        mock_agent.get_messages.return_value = [{"role": "assistant", "content": "def is_palindrome(s):\n    return s == s[::-1]"}]

        result = evaluator.evaluate(env, [mock_agent])

        assert result["score"] == 1.0
        assert result["match_type"] == "code_present"


# =============================================================================
# LiteLLM Model Adapter Tests (with mocks)
# =============================================================================


class TestLiteLLMModelAdapter:
    """Tests for LiteLLM model adapter integration."""

    def test_adapter_initialization(self):
        """Test LiteLLM adapter initializes correctly."""
        from maseval.interface.inference import LiteLLMModelAdapter

        adapter = LiteLLMModelAdapter(
            model_id="ollama/llama3.2",
            api_base="http://localhost:11434/v1",
            api_key="ollama",
        )

        assert adapter.model_id == "ollama/llama3.2"

    def test_adapter_gather_config(self):
        """Test adapter gathers config correctly."""
        from maseval.interface.inference import LiteLLMModelAdapter

        adapter = LiteLLMModelAdapter(
            model_id="ollama/mistral",
            default_generation_params={"temperature": 0.7},
        )

        config = adapter.gather_config()

        assert config["model_id"] == "ollama/mistral"
        assert config["default_generation_params"]["temperature"] == 0.7


# =============================================================================
# Benchmark Tests (self-contained, no external imports)
# =============================================================================


class TestSimpleBenchmark:
    """Tests for benchmark metrics computation logic."""

    def test_benchmark_task_creation(self):
        """Test Task creation with metadata."""
        from maseval import Task

        task = Task(
            query="What is 2+2?",
            metadata={
                "task_id": "math_001",
                "category": "reasoning",
                "expected": "4",
            },
        )

        assert task.query == "What is 2+2?"
        assert task.metadata["category"] == "reasoning"
        assert task.metadata["expected"] == "4"

    def test_benchmark_metrics_computation(self):
        """Test metrics computation from reports (inline implementation)."""

        # Inline metrics computation (mirrors daf_benchmark.py)
        def compute_metrics(reports):
            if not reports:
                return {"error": "No reports"}

            total_tasks = len(reports)
            total_score = sum(r.get("eval", [{}])[0].get("score", 0) for r in reports)
            avg_score = total_score / total_tasks if total_tasks > 0 else 0.0

            total_duration = sum(r.get("traces", {}).get("agents", {}).get("agent_1", {}).get("duration_seconds", 0) for r in reports)

            total_tokens = sum(
                r.get("traces", {}).get("agents", {}).get("agent_1", {}).get("usage", {}).get("input_tokens", 0)
                + r.get("traces", {}).get("agents", {}).get("agent_1", {}).get("usage", {}).get("output_tokens", 0)
                for r in reports
            )

            categories = {}
            for r in reports:
                cat = r.get("task", {}).get("metadata", {}).get("category", "unknown")
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(r.get("eval", [{}])[0].get("score", 0))

            return {
                "total_tasks": total_tasks,
                "accuracy": avg_score,
                "total_duration_seconds": total_duration,
                "total_tokens": total_tokens,
                "metrics_by_category": {k: {"accuracy": sum(v) / len(v), "count": len(v)} for k, v in categories.items()},
            }

        # Create mock reports
        reports = [
            {
                "task": {"metadata": {"category": "factual"}},
                "eval": [{"score": 1.0}],
                "traces": {
                    "agents": {
                        "agent_1": {
                            "duration_seconds": 2.5,
                            "usage": {"input_tokens": 50, "output_tokens": 20},
                        }
                    }
                },
            },
            {
                "task": {"metadata": {"category": "reasoning"}},
                "eval": [{"score": 0.5}],
                "traces": {
                    "agents": {
                        "agent_1": {
                            "duration_seconds": 5.0,
                            "usage": {"input_tokens": 100, "output_tokens": 50},
                        }
                    }
                },
            },
        ]

        metrics = compute_metrics(reports)

        assert metrics["total_tasks"] == 2
        assert metrics["accuracy"] == 0.75  # (1.0 + 0.5) / 2
        assert metrics["total_duration_seconds"] == 7.5
        assert metrics["total_tokens"] == 220  # 50+20+100+50
        assert "factual" in metrics["metrics_by_category"]
        assert "reasoning" in metrics["metrics_by_category"]


# =============================================================================
# Markers for test categorization
# =============================================================================

pytestmark = [
    pytest.mark.core,  # Core tests (no external dependencies)
]
