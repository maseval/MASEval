"""Tests for MultiAgentBench evaluator."""

import pytest

from conftest import DummyModelAdapter
from maseval.benchmark.multiagentbench.evaluator import (
    DEFAULT_RESULT_TRUNCATION_LENGTH,
    MultiAgentBenchEvaluator,
    MultiAgentBenchMetrics,
)

pytestmark = pytest.mark.benchmark


class TestMultiAgentBenchMetrics:
    """Tests for MultiAgentBenchMetrics dataclass."""

    def test_default_values(self):
        """Metrics should have sensible defaults."""
        metrics = MultiAgentBenchMetrics()

        assert metrics.task_completion is False
        assert metrics.token_consumption == 0
        assert metrics.communication_score is None
        assert metrics.task_evaluation == {}
        assert metrics.agent_kpis == {}
        assert metrics.total_milestones == 0

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        metrics = MultiAgentBenchMetrics(
            task_completion=True,
            token_consumption=1000,
            communication_score=5.0,
        )
        d = metrics.to_dict()

        assert d["task_completion"] is True
        assert d["token_consumption"] == 1000
        assert d["communication_score"] == 5.0


class TestMultiAgentBenchEvaluator:
    """Tests for MultiAgentBenchEvaluator class."""

    @pytest.fixture
    def mock_model_adapter(self):
        """Create a mock model adapter."""
        return DummyModelAdapter(
            model_id="test-model",
            responses=['{"rating": 4}'],
        )

    @pytest.fixture
    def research_evaluator(self, mock_model_adapter):
        """Create a research domain evaluator."""
        return MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=mock_model_adapter,
        )

    @pytest.fixture
    def bargaining_evaluator(self, mock_model_adapter):
        """Create a bargaining domain evaluator."""
        return MultiAgentBenchEvaluator(
            domain="bargaining",
            model_adapter=mock_model_adapter,
        )

    def test_init_normalizes_domain(self, mock_model_adapter):
        """Evaluator should normalize domain to lowercase."""
        evaluator = MultiAgentBenchEvaluator(
            domain="RESEARCH",
            model_adapter=mock_model_adapter,
        )
        assert evaluator.domain == "research"

    def test_filter_traces_extracts_agents(self, research_evaluator):
        """filter_traces should extract agent traces."""
        traces = {
            "agents": {"agent1": {"action_log": [{"task": "test", "result": "done"}]}},
            "environment": {},
        }
        filtered = research_evaluator.filter_traces(traces)

        assert "agents" in filtered
        assert "agent1" in filtered["agents"]

    def test_extract_communications_formats_correctly(self, research_evaluator):
        """_extract_communications should format communication logs."""
        traces = {
            "agents": {
                "agent1": {"communication_log": [{"communication": "Hello from agent1"}]},
                "agent2": {"communication_log": [{"communication": "Hello from agent2"}]},
            }
        }
        comms = research_evaluator._extract_communications(traces)

        assert "[agent1]: Hello from agent1" in comms
        assert "[agent2]: Hello from agent2" in comms

    def test_extract_communications_empty(self, research_evaluator):
        """_extract_communications should handle empty logs."""
        traces = {"agents": {}}
        comms = research_evaluator._extract_communications(traces)

        assert comms == "No communications recorded."

    def test_extract_results_formats_correctly(self, research_evaluator):
        """_extract_results should format action logs."""
        traces = {"agents": {"agent1": {"action_log": [{"result": "Completed task A"}]}}}
        results = research_evaluator._extract_results(traces)

        assert "[agent1]: Completed task A" in results

    def test_calculate_token_consumption(self, research_evaluator):
        """_calculate_token_consumption should sum agent token usage."""
        traces = {
            "agents": {
                "agent1": {"token_usage": 500},
                "agent2": {"token_usage": 300},
            }
        }
        total = research_evaluator._calculate_token_consumption(traces)

        assert total == 800

    def test_parse_score_valid_json(self, research_evaluator):
        """_parse_score should parse valid JSON response."""
        response = '{"rating": 4}'
        score = research_evaluator._parse_score(response)

        assert score == 4.0

    def test_parse_score_with_markdown(self, research_evaluator):
        """_parse_score should handle markdown code blocks."""
        response = '```json\n{"rating": 5}\n```'
        score = research_evaluator._parse_score(response)

        assert score == 5.0

    def test_parse_score_no_json_raises(self, research_evaluator):
        """_parse_score should raise when no valid JSON found."""
        response = "The rating is 4 out of 5"
        with pytest.raises(ValueError, match="No JSON object found"):
            research_evaluator._parse_score(response)

    def test_parse_score_no_rating_key_raises(self, research_evaluator):
        """_parse_score should raise when JSON has no rating key."""
        response = '{"score": 4}'
        with pytest.raises(ValueError, match="Expected"):
            research_evaluator._parse_score(response)

    def test_parse_research_ratings_valid(self, research_evaluator):
        """_parse_research_ratings should parse valid ratings."""
        response = '{"innovation": 4, "safety": 3, "feasibility": 5}'
        ratings = research_evaluator._parse_research_ratings(response)

        assert ratings["innovation"] == 4
        assert ratings["safety"] == 3
        assert ratings["feasibility"] == 5

    def test_parse_research_ratings_invalid(self, research_evaluator):
        """_parse_research_ratings should raise for invalid response."""
        with pytest.raises(ValueError, match="No JSON object found"):
            research_evaluator._parse_research_ratings("Invalid response")

    def test_determine_completion_research_positive(self, research_evaluator):
        """_determine_completion should return True for positive research scores."""
        metrics = MultiAgentBenchMetrics(task_evaluation={"innovation": 4, "safety": 3, "feasibility": 5})
        assert research_evaluator._determine_completion(metrics) is True

    def test_determine_completion_research_negative(self, research_evaluator):
        """_determine_completion should return False for None scores."""
        metrics = MultiAgentBenchMetrics(task_evaluation={"innovation": None, "safety": 3, "feasibility": 5})
        assert research_evaluator._determine_completion(metrics) is False

    def test_call_returns_expected_structure(self, research_evaluator):
        """__call__ should return expected result structure."""
        traces = {
            "agents": {"agent1": {"token_usage": 100, "action_log": [], "communication_log": []}},
            "environment": {},
        }
        final_answer = [{"agent_id": "agent1", "result": "Done"}]

        # Needs two responses: one for LLM summarization, one for research evaluation
        research_evaluator.model_adapter = DummyModelAdapter(
            model_id="test",
            responses=[
                '{"summary": "Agent completed the task"}',
                '{"innovation": 4, "safety": 4, "feasibility": 4}',
            ],
        )

        result = research_evaluator(traces, final_answer)

        assert "passed" in result
        assert "metrics" in result
        assert "domain" in result
        assert result["domain"] == "research"

    def test_format_final_answer_dict(self, research_evaluator):
        """_format_final_answer should format dict input."""
        final_answer = {
            "agent_results": [
                {"agent_id": "agent1", "result": "Result 1"},
                {"agent_id": "agent2", "result": "Result 2"},
            ]
        }
        formatted = research_evaluator._format_final_answer(final_answer)

        assert "[agent1]: Result 1" in formatted
        assert "[agent2]: Result 2" in formatted

    def test_format_final_answer_list(self, research_evaluator):
        """_format_final_answer should format list input."""
        final_answer = [
            {"agent_id": "agent1", "result": "Result 1"},
        ]
        formatted = research_evaluator._format_final_answer(final_answer)

        assert "[agent1]: Result 1" in formatted

    def test_format_final_answer_string(self, research_evaluator):
        """_format_final_answer should pass through string."""
        formatted = research_evaluator._format_final_answer("Simple result")
        assert formatted == "Simple result"

    def test_format_final_answer_none(self, research_evaluator):
        """_format_final_answer should handle None."""
        formatted = research_evaluator._format_final_answer(None)
        assert formatted == ""


class TestBargainingEvaluation:
    """Tests for bargaining-specific evaluation."""

    @pytest.fixture
    def bargaining_evaluator(self):
        """Create evaluator with bargaining-specific responses."""
        adapter = DummyModelAdapter(
            model_id="test",
            responses=[
                '{"effectiveness_of_strategies": 4, "progress_and_outcome": 3, "interaction_dynamics": 5}',
                '{"effectiveness_of_strategies": 3, "progress_and_outcome": 4, "interaction_dynamics": 4}',
            ],
        )
        return MultiAgentBenchEvaluator(
            domain="bargaining",
            model_adapter=adapter,
        )

    def test_evaluate_bargaining_returns_buyer_seller(self, bargaining_evaluator):
        """_evaluate_bargaining should return buyer and seller ratings."""
        ratings = bargaining_evaluator._evaluate_bargaining("Task", "Result")

        assert "buyer" in ratings
        assert "seller" in ratings
        assert "effectiveness_of_strategies" in ratings["buyer"]

    def test_determine_completion_bargaining_positive(self, bargaining_evaluator):
        """_determine_completion should work for bargaining domain."""
        metrics = MultiAgentBenchMetrics(
            task_evaluation={
                "buyer": {
                    "effectiveness_of_strategies": 4,
                    "progress_and_outcome": 3,
                    "interaction_dynamics": 5,
                },
                "seller": {
                    "effectiveness_of_strategies": 3,
                    "progress_and_outcome": 4,
                    "interaction_dynamics": 4,
                },
            }
        )
        assert bargaining_evaluator._determine_completion(metrics) is True


class TestCodingEvaluation:
    """Tests for coding-specific evaluation."""

    @pytest.fixture
    def coding_evaluator(self):
        """Create evaluator with coding-specific responses."""
        adapter = DummyModelAdapter(
            model_id="test",
            responses=[
                '{"instruction_following": 5, "executability": 4, "consistency": 4, "quality": 3}',
            ],
        )
        return MultiAgentBenchEvaluator(
            domain="coding",
            model_adapter=adapter,
        )

    def test_evaluate_coding_returns_all_metrics(self, coding_evaluator):
        """_evaluate_coding should return all coding metrics."""
        ratings = coding_evaluator._evaluate_coding("Task", "Solution")

        assert "instruction_following" in ratings
        assert "executability" in ratings
        assert "consistency" in ratings
        assert "quality" in ratings

    def test_determine_completion_coding_positive(self, coding_evaluator):
        """_determine_completion should work for coding domain."""
        metrics = MultiAgentBenchMetrics(
            task_evaluation={
                "instruction_following": 5,
                "executability": 4,
                "consistency": 4,
                "quality": 3,
            }
        )
        assert coding_evaluator._determine_completion(metrics) is True

    def test_determine_completion_coding_negative(self, coding_evaluator):
        """_determine_completion should return False for None coding scores."""
        metrics = MultiAgentBenchMetrics(
            task_evaluation={
                "instruction_following": 5,
                "executability": None,
                "consistency": 4,
                "quality": 3,
            }
        )
        assert coding_evaluator._determine_completion(metrics) is False

    def test_parse_coding_ratings_invalid(self, coding_evaluator):
        """_parse_coding_ratings should raise for invalid response."""
        with pytest.raises(ValueError, match="No JSON object found"):
            coding_evaluator._parse_coding_ratings("Invalid JSON")


class TestDatabaseEvaluation:
    """Tests for database-specific evaluation."""

    @pytest.fixture
    def database_evaluator(self):
        """Create evaluator for database domain."""
        adapter = DummyModelAdapter(
            model_id="test",
            responses=['{"rating": 4}'],
        )
        return MultiAgentBenchEvaluator(
            domain="database",
            model_adapter=adapter,
        )

    def test_evaluate_database_returns_structure(self, database_evaluator):
        """_evaluate_database should return predicted result."""
        result = database_evaluator._evaluate_database("Query task", "SELECT * FROM users")

        assert "predicted" in result
        assert result["predicted"] == "SELECT * FROM users"
        assert "root_cause" in result
        assert result["root_cause"] == []

    def test_determine_completion_database_with_prediction(self, database_evaluator):
        """_determine_completion should return True for database with prediction."""
        metrics = MultiAgentBenchMetrics(task_evaluation={"predicted": "SELECT * FROM users", "root_cause": []})
        assert database_evaluator._determine_completion(metrics) is True

    def test_determine_completion_database_empty_prediction(self, database_evaluator):
        """_determine_completion should return False for empty prediction."""
        metrics = MultiAgentBenchMetrics(task_evaluation={"predicted": "", "root_cause": []})
        assert database_evaluator._determine_completion(metrics) is False

    def test_call_database_domain(self, database_evaluator):
        """__call__ should work for database domain."""
        traces = {
            "agents": {"agent1": {"token_usage": 100, "action_log": [], "communication_log": []}},
            "environment": {"marble_state": {"task_description": "Find query issue"}},
        }
        final_answer = [{"agent_id": "agent1", "result": "SELECT * FROM orders"}]

        # Needs a response for the LLM summarization call
        database_evaluator.model_adapter = DummyModelAdapter(
            model_id="test",
            responses=["SELECT * FROM orders"],
        )

        result = database_evaluator(traces, final_answer)

        assert result["domain"] == "database"
        assert "passed" in result


class TestWerewolfEvaluation:
    """Tests for werewolf-specific evaluation."""

    @pytest.fixture
    def werewolf_evaluator(self):
        """Create evaluator with werewolf-specific responses."""
        adapter = DummyModelAdapter(
            model_id="test",
            responses=[
                '{"game_outcome": 4, "deception_detection": 3, "voting_strategy": 4, '
                '"role_fulfillment": 5, "information_usage": 3, "collaboration": 4, "survival_rate": 3}',
            ],
        )
        return MultiAgentBenchEvaluator(
            domain="werewolf",
            model_adapter=adapter,
        )

    def test_evaluate_werewolf_returns_all_metrics(self, werewolf_evaluator):
        """_evaluate_werewolf should return all werewolf metrics."""
        ratings = werewolf_evaluator._evaluate_werewolf("Task", "Result")

        assert "game_outcome" in ratings
        assert "deception_detection" in ratings
        assert "voting_strategy" in ratings
        assert "role_fulfillment" in ratings
        assert "information_usage" in ratings
        assert "collaboration" in ratings
        assert "survival_rate" in ratings

    def test_determine_completion_werewolf_positive(self, werewolf_evaluator):
        """_determine_completion should work for werewolf domain."""
        metrics = MultiAgentBenchMetrics(
            task_evaluation={
                "game_outcome": 4,
                "deception_detection": 3,
                "voting_strategy": 4,
                "role_fulfillment": 5,
                "information_usage": 3,
                "collaboration": 4,
                "survival_rate": 3,
            }
        )
        assert werewolf_evaluator._determine_completion(metrics) is True

    def test_determine_completion_werewolf_negative(self, werewolf_evaluator):
        """_determine_completion should return False for None werewolf scores."""
        metrics = MultiAgentBenchMetrics(
            task_evaluation={
                "game_outcome": 4,
                "deception_detection": None,
                "voting_strategy": 4,
                "role_fulfillment": 5,
                "information_usage": 3,
                "collaboration": 4,
                "survival_rate": 3,
            }
        )
        assert werewolf_evaluator._determine_completion(metrics) is False

    def test_call_werewolf_domain(self, werewolf_evaluator):
        """__call__ should work for werewolf domain."""
        traces = {
            "agents": {"agent1": {"token_usage": 100, "action_log": [], "communication_log": []}},
            "environment": {},
        }
        final_answer = "Villagers won the game"

        result = werewolf_evaluator(traces, final_answer)

        assert result["domain"] == "werewolf"
        assert "passed" in result
        assert "game_outcome" in result["metrics"]["task_evaluation"]

    def test_parse_werewolf_ratings_invalid(self, werewolf_evaluator):
        """_parse_werewolf_ratings should raise for invalid response."""
        with pytest.raises(ValueError, match="No JSON object found"):
            werewolf_evaluator._parse_werewolf_ratings("Invalid JSON")


class TestMinecraftEvaluation:
    """Tests for minecraft-specific evaluation."""

    @pytest.fixture
    def minecraft_evaluator(self):
        """Create evaluator with minecraft-specific responses."""
        adapter = DummyModelAdapter(
            model_id="test",
            responses=[
                '{"structural_completeness": 4, "blueprint_accuracy": 3, "coordination": 5, "efficiency": 4}',
            ],
        )
        return MultiAgentBenchEvaluator(
            domain="minecraft",
            model_adapter=adapter,
        )

    def test_evaluate_minecraft_returns_all_metrics(self, minecraft_evaluator):
        """_evaluate_minecraft should return all minecraft metrics."""
        ratings = minecraft_evaluator._evaluate_minecraft("Task", "Result")

        assert "structural_completeness" in ratings
        assert "blueprint_accuracy" in ratings
        assert "coordination" in ratings
        assert "efficiency" in ratings

    def test_determine_completion_minecraft_positive(self, minecraft_evaluator):
        """_determine_completion should work for minecraft domain."""
        metrics = MultiAgentBenchMetrics(
            task_evaluation={
                "structural_completeness": 4,
                "blueprint_accuracy": 3,
                "coordination": 5,
                "efficiency": 4,
            }
        )
        assert minecraft_evaluator._determine_completion(metrics) is True

    def test_determine_completion_minecraft_negative(self, minecraft_evaluator):
        """_determine_completion should return False for None minecraft scores."""
        metrics = MultiAgentBenchMetrics(
            task_evaluation={
                "structural_completeness": 4,
                "blueprint_accuracy": None,
                "coordination": 5,
                "efficiency": 4,
            }
        )
        assert minecraft_evaluator._determine_completion(metrics) is False

    def test_call_minecraft_domain(self, minecraft_evaluator):
        """__call__ should work for minecraft domain."""
        traces = {
            "agents": {"agent1": {"token_usage": 100, "action_log": [], "communication_log": []}},
            "environment": {},
        }
        final_answer = "Structure built successfully"

        result = minecraft_evaluator(traces, final_answer)

        assert result["domain"] == "minecraft"
        assert "passed" in result
        assert "structural_completeness" in result["metrics"]["task_evaluation"]

    def test_parse_minecraft_ratings_invalid(self, minecraft_evaluator):
        """_parse_minecraft_ratings should raise for invalid response."""
        with pytest.raises(ValueError, match="No JSON object found"):
            minecraft_evaluator._parse_minecraft_ratings("Invalid JSON")

    def test_evaluate_minecraft_on_error(self):
        """_evaluate_minecraft should propagate LLM errors."""
        from unittest.mock import MagicMock

        mock_adapter = MagicMock()
        mock_adapter.generate.side_effect = RuntimeError("Model failed")

        evaluator = MultiAgentBenchEvaluator(
            domain="minecraft",
            model_adapter=mock_adapter,
        )

        with pytest.raises(RuntimeError, match="Model failed"):
            evaluator._evaluate_minecraft("Task", "Result")


class TestUnknownDomainEvaluation:
    """Tests for unknown domain handling."""

    @pytest.fixture
    def unknown_evaluator(self):
        """Create evaluator for unknown domain."""
        adapter = DummyModelAdapter(
            model_id="test",
            responses=['{"rating": 4}'],
        )
        return MultiAgentBenchEvaluator(
            domain="unknown_domain",
            model_adapter=adapter,
        )

    def test_unknown_domain_sets_completion_from_result(self, unknown_evaluator):
        """Unknown domain should set completion based on final result presence."""
        traces = {
            "agents": {},
            "environment": {},
        }
        final_answer = "Some result"

        result = unknown_evaluator(traces, final_answer)

        assert result["domain"] == "unknown_domain"
        # Should still have metrics
        assert "metrics" in result

    def test_determine_completion_unknown_domain(self, unknown_evaluator):
        """_determine_completion should return False for unknown domain."""
        metrics = MultiAgentBenchMetrics(task_evaluation={})
        assert unknown_evaluator._determine_completion(metrics) is False


class TestCommunicationEvaluation:
    """Tests for communication evaluation."""

    @pytest.fixture
    def evaluator_with_comm_response(self):
        """Create evaluator with communication rating response."""
        adapter = DummyModelAdapter(
            model_id="test",
            responses=['{"rating": 4}', '{"innovation": 4, "safety": 4, "feasibility": 4}'],
        )
        return MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=adapter,
        )

    def test_evaluate_communication_returns_score(self, evaluator_with_comm_response):
        """_evaluate_communication should return score from LLM."""
        score = evaluator_with_comm_response._evaluate_communication("Research task", "[agent1]: Hello\n[agent2]: Hi")
        assert score == 4.0

    def test_evaluate_communication_on_error(self):
        """_evaluate_communication should propagate LLM errors."""
        from unittest.mock import MagicMock

        # Create mock that raises exception on generate
        mock_adapter = MagicMock()
        mock_adapter.generate.side_effect = RuntimeError("Model failed")

        evaluator = MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=mock_adapter,
        )

        with pytest.raises(RuntimeError, match="Model failed"):
            evaluator._evaluate_communication("Task", "Comms")

    def test_call_with_communications(self, evaluator_with_comm_response):
        """__call__ should evaluate communications when present."""
        traces = {
            "agents": {
                "agent1": {
                    "token_usage": 100,
                    "action_log": [],
                    "communication_log": [{"communication": "Hello from agent1"}],
                }
            },
            "environment": {},
        }
        final_answer = "Result"

        result = evaluator_with_comm_response(traces, final_answer)

        # Communication score should be evaluated (not -1)
        assert result["metrics"]["communication_score"] == 4.0


class TestExceptionHandling:
    """Tests for exception handling in evaluation methods."""

    @pytest.fixture
    def failing_evaluator(self):
        """Create evaluator with adapter that fails."""
        from unittest.mock import MagicMock

        # Create mock that raises exception on generate
        mock_adapter = MagicMock()
        mock_adapter.generate.side_effect = RuntimeError("Model failed")
        return MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=mock_adapter,
        )

    def test_evaluate_research_on_error(self, failing_evaluator):
        """_evaluate_research should propagate LLM errors."""
        with pytest.raises(RuntimeError, match="Model failed"):
            failing_evaluator._evaluate_research("Task", "Result")

    def test_evaluate_bargaining_on_error(self, failing_evaluator):
        """_evaluate_bargaining should propagate LLM errors."""
        failing_evaluator.domain = "bargaining"
        with pytest.raises(RuntimeError, match="Model failed"):
            failing_evaluator._evaluate_bargaining("Task", "Result")

    def test_evaluate_coding_on_error(self, failing_evaluator):
        """_evaluate_coding should propagate LLM errors."""
        failing_evaluator.domain = "coding"
        with pytest.raises(RuntimeError, match="Model failed"):
            failing_evaluator._evaluate_coding("Task", "Result")


class TestParsingEdgeCases:
    """Tests for parsing edge cases."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator for parsing tests."""
        adapter = DummyModelAdapter(model_id="test", responses=['{"rating": 4}'])
        return MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=adapter,
        )

    def test_parse_score_out_of_range(self, evaluator):
        """_parse_score should raise for scores outside 1-5 range."""
        response = '{"rating": 10}'
        with pytest.raises(ValueError, match="out of valid range"):
            evaluator._parse_score(response)

    def test_parse_score_with_just_code_block(self, evaluator):
        """_parse_score should handle code block without json marker."""
        response = '```\n{"rating": 3}\n```'
        score = evaluator._parse_score(response)
        assert score == 3.0  # Valid score parsed from JSON

    def test_parse_score_with_text_before_json(self, evaluator):
        """_parse_score should find JSON in text."""
        response = 'Here is my rating: {"rating": 5}'
        score = evaluator._parse_score(response)
        assert score == 5.0

    def test_parse_bargaining_ratings_missing_keys(self, evaluator):
        """_parse_bargaining_ratings should raise when required keys are missing."""
        response = '{"effectiveness_of_strategies": 4}'  # Missing other fields
        with pytest.raises(KeyError):
            evaluator._parse_bargaining_ratings(response)

    def test_parse_bargaining_ratings_invalid(self, evaluator):
        """_parse_bargaining_ratings should raise for invalid response."""
        with pytest.raises(ValueError, match="No JSON object found"):
            evaluator._parse_bargaining_ratings("Invalid")


class TestFormatFinalAnswerEdgeCases:
    """Tests for _format_final_answer edge cases."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator for format tests."""
        adapter = DummyModelAdapter(model_id="test", responses=[])
        return MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=adapter,
        )

    def test_format_dict_without_agent_results(self, evaluator):
        """_format_final_answer should JSON dump dict without agent_results."""
        final_answer = {"status": "completed", "data": "some data"}
        formatted = evaluator._format_final_answer(final_answer)
        assert "status" in formatted
        assert "completed" in formatted

    def test_format_dict_with_empty_agent_results(self, evaluator):
        """_format_final_answer should handle empty agent_results."""
        final_answer = {"agent_results": []}
        formatted = evaluator._format_final_answer(final_answer)
        # Empty results should JSON dump the dict
        assert formatted == '{"agent_results": []}'

    def test_format_list_with_non_dict_items(self, evaluator):
        """_format_final_answer should skip non-dict items in list."""
        final_answer = [
            {"agent_id": "agent1", "result": "Result 1"},
            "not a dict",
            {"agent_id": "agent2", "result": "Result 2"},
        ]
        formatted = evaluator._format_final_answer(final_answer)
        assert "[agent1]: Result 1" in formatted
        assert "[agent2]: Result 2" in formatted
        assert "not a dict" not in formatted


class TestTokenConsumptionEdgeCases:
    """Tests for token consumption calculation edge cases."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator for token tests."""
        adapter = DummyModelAdapter(model_id="test", responses=[])
        return MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=adapter,
        )

    def test_token_consumption_with_non_int(self, evaluator):
        """_calculate_token_consumption should skip non-int values."""
        traces = {
            "agents": {
                "agent1": {"token_usage": 500},
                "agent2": {"token_usage": "invalid"},  # Non-int
                "agent3": {"token_usage": 300},
            }
        }
        total = evaluator._calculate_token_consumption(traces)
        assert total == 800  # Only int values counted

    def test_token_consumption_empty_agents(self, evaluator):
        """_calculate_token_consumption should return 0 for no agents."""
        traces = {"agents": {}}
        total = evaluator._calculate_token_consumption(traces)
        assert total == 0

    def test_token_consumption_missing_key(self, evaluator):
        """_calculate_token_consumption should handle missing token_usage."""
        traces = {
            "agents": {
                "agent1": {"action_log": []},  # No token_usage
            }
        }
        total = evaluator._calculate_token_consumption(traces)
        assert total == 0


class TestGetTaskDescription:
    """Tests for _get_task_description method."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator for task description tests."""
        adapter = DummyModelAdapter(model_id="test", responses=[])
        return MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=adapter,
        )

    def test_get_task_description_from_traces(self, evaluator):
        """_get_task_description should extract from traces."""
        traces = {"environment": {"marble_state": {"task_description": "Research AI safety"}}}
        desc = evaluator._get_task_description(traces)
        assert desc == "Research AI safety"

    def test_get_task_description_missing_env(self, evaluator):
        """_get_task_description should return empty for missing env."""
        traces = {}
        desc = evaluator._get_task_description(traces)
        assert desc == ""

    def test_get_task_description_missing_state(self, evaluator):
        """_get_task_description should return empty for missing marble_state."""
        traces = {"environment": {}}
        desc = evaluator._get_task_description(traces)
        assert desc == ""


class TestExtractResultsEdgeCases:
    """Tests for _extract_results edge cases."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator for extract results tests."""
        adapter = DummyModelAdapter(model_id="test", responses=[])
        return MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=adapter,
        )

    def test_extract_results_empty_result(self, evaluator):
        """_extract_results should skip empty results."""
        traces = {
            "agents": {
                "agent1": {"action_log": [{"result": ""}]},
            }
        }
        results = evaluator._extract_results(traces)
        assert results == "No results recorded."

    def test_extract_results_multiple_actions(self, evaluator):
        """_extract_results should include all actions."""
        traces = {
            "agents": {
                "agent1": {
                    "action_log": [
                        {"result": "Step 1 done"},
                        {"result": "Step 2 done"},
                    ]
                }
            }
        }
        results = evaluator._extract_results(traces)
        assert "Step 1 done" in results
        assert "Step 2 done" in results


class TestFilterTracesEdgeCases:
    """Tests for filter_traces edge cases."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator for filter traces tests."""
        adapter = DummyModelAdapter(model_id="test", responses=[])
        return MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=adapter,
        )

    def test_filter_traces_empty(self, evaluator):
        """filter_traces should handle empty traces."""
        filtered = evaluator.filter_traces({})
        assert filtered["agents"] == {}
        assert filtered["environment"] == {}
        assert "communications" in filtered

    def test_filter_traces_extracts_all(self, evaluator):
        """filter_traces should extract all relevant data."""
        traces = {
            "agents": {
                "agent1": {
                    "action_log": [{"result": "Done"}],
                    "communication_log": [{"communication": "Hello"}],
                }
            },
            "environment": {"domain": "research"},
        }
        filtered = evaluator.filter_traces(traces)
        assert "[agent1]: Hello" in filtered["communications"]
        assert filtered["agents"] == traces["agents"]
        assert filtered["environment"] == traces["environment"]


class TestDetermineCompletionEdgeCases:
    """Tests for _determine_completion edge cases."""

    def test_completion_empty_eval_data(self):
        """_determine_completion should return False for empty eval data."""
        adapter = DummyModelAdapter(model_id="test", responses=[])
        evaluator = MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=adapter,
        )
        metrics = MultiAgentBenchMetrics(task_evaluation={})
        assert evaluator._determine_completion(metrics) is False

    def test_completion_bargaining_partial_buyer(self):
        """_determine_completion should return False if buyer has None scores."""
        adapter = DummyModelAdapter(model_id="test", responses=[])
        evaluator = MultiAgentBenchEvaluator(
            domain="bargaining",
            model_adapter=adapter,
        )
        metrics = MultiAgentBenchMetrics(
            task_evaluation={
                "buyer": {
                    "effectiveness_of_strategies": None,
                    "progress_and_outcome": 4,
                    "interaction_dynamics": 4,
                },
                "seller": {
                    "effectiveness_of_strategies": 4,
                    "progress_and_outcome": 4,
                    "interaction_dynamics": 4,
                },
            }
        )
        assert evaluator._determine_completion(metrics) is False

    def test_completion_bargaining_partial_seller(self):
        """_determine_completion should return False if seller has None scores."""
        adapter = DummyModelAdapter(model_id="test", responses=[])
        evaluator = MultiAgentBenchEvaluator(
            domain="bargaining",
            model_adapter=adapter,
        )
        metrics = MultiAgentBenchMetrics(
            task_evaluation={
                "buyer": {
                    "effectiveness_of_strategies": 4,
                    "progress_and_outcome": 4,
                    "interaction_dynamics": 4,
                },
                "seller": {
                    "effectiveness_of_strategies": None,
                    "progress_and_outcome": 4,
                    "interaction_dynamics": 4,
                },
            }
        )
        assert evaluator._determine_completion(metrics) is False


class TestResultSummarization:
    """Tests for MARBLE-compatible result truncation and LLM summarization.

    MARBLE's evaluation pipeline truncates each agent result to 1000 chars
    (_summarize_results) then summarizes via an LLM call (summarize_output)
    before passing to the domain evaluator.
    """

    @pytest.fixture
    def evaluator(self):
        """Create evaluator with summarization enabled (default)."""
        adapter = DummyModelAdapter(
            model_id="test",
            responses=["Summarized output"],
        )
        return MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=adapter,
            output_format="Present the idea in 5Q format.",
        )

    def test_summarize_results_truncates_long_results(self, evaluator):
        """Each agent result line should be truncated to result_truncation_length chars."""
        long_result = "x" * 2000
        agent_results = [{"agent_id": "a1", "result": long_result}]
        summary = evaluator._summarize_results(agent_results)

        # Header line + one result line
        lines = summary.strip().split("\n")
        assert lines[0] == "Agents' Results Summary:"
        # "- " prefix + content, truncated to DEFAULT_RESULT_TRUNCATION_LENGTH total
        assert len(lines[1]) == DEFAULT_RESULT_TRUNCATION_LENGTH

    def test_summarize_results_short_results_unchanged(self, evaluator):
        """Short results should not be truncated."""
        agent_results = [{"agent_id": "a1", "result": "short"}]
        summary = evaluator._summarize_results(agent_results)

        lines = summary.strip().split("\n")
        assert lines[1] == "- short"

    def test_summarize_results_multiple_agents(self, evaluator):
        """Multiple agent results should all appear in the summary."""
        agent_results = [
            {"agent_id": "a1", "result": "Result 1"},
            {"agent_id": "a2", "result": "Result 2"},
        ]
        summary = evaluator._summarize_results(agent_results)

        assert "Result 1" in summary
        assert "Result 2" in summary
        assert summary.startswith("Agents' Results Summary:\n")

    def test_summarize_results_empty_list(self, evaluator):
        """Empty results should produce header-only summary."""
        summary = evaluator._summarize_results([])
        assert summary == "Agents' Results Summary:\n"

    def test_summarize_results_custom_truncation_length(self):
        """Custom result_truncation_length should be respected."""
        adapter = DummyModelAdapter(model_id="test", responses=[])
        evaluator = MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=adapter,
            result_truncation_length=50,
        )
        agent_results = [{"agent_id": "a1", "result": "x" * 200}]
        summary = evaluator._summarize_results(agent_results)

        lines = summary.strip().split("\n")
        assert len(lines[1]) == 50

    def test_summarize_output_calls_model(self, evaluator):
        """_summarize_output should call the model adapter and return its response."""
        result = evaluator._summarize_output("truncated summary", "research task", "5Q format")
        assert isinstance(result, str)
        assert result == "Summarized output"

    def test_call_with_summarization_uses_both_steps(self):
        """__call__ with agent results should use truncation + LLM summarization."""
        adapter = DummyModelAdapter(
            model_id="test",
            responses=[
                '{"summary": "Agents collaborated on federated learning"}',
                '{"innovation": 4, "safety": 4, "feasibility": 4}',
            ],
        )
        evaluator = MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=adapter,
            output_format="5Q format",
        )
        traces = {
            "agents": {"a1": {"token_usage": 100, "action_log": [], "communication_log": []}},
            "environment": {"marble_state": {"task_description": "Research task"}},
        }
        final_answer = {
            "agent_results": [{"agent_id": "a1", "result": "Raw paper search JSON " * 100}],
        }

        result = evaluator(traces, final_answer)

        assert result["passed"] is True
        assert result["metrics"]["task_evaluation"]["innovation"] == 4

    def test_call_with_summarization_disabled(self):
        """__call__ with result_truncation_length=None should skip summarization."""
        adapter = DummyModelAdapter(
            model_id="test",
            # Only one response needed: no summarization call, just domain eval
            responses=['{"innovation": 4, "safety": 4, "feasibility": 4}'],
        )
        evaluator = MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=adapter,
            result_truncation_length=None,
        )
        traces = {
            "agents": {"a1": {"token_usage": 100, "action_log": [], "communication_log": []}},
            "environment": {},
        }
        final_answer = {
            "agent_results": [{"agent_id": "a1", "result": "Raw result"}],
        }

        result = evaluator(traces, final_answer)

        assert result["domain"] == "research"
        assert result["passed"] is True

    def test_call_string_final_answer_bypasses_summarization(self):
        """String final_answer should bypass summarization (no agent_results to extract)."""
        adapter = DummyModelAdapter(
            model_id="test",
            # Only one response needed: domain eval only
            responses=['{"innovation": 4, "safety": 4, "feasibility": 4}'],
        )
        evaluator = MultiAgentBenchEvaluator(
            domain="research",
            model_adapter=adapter,
        )
        traces = {
            "agents": {},
            "environment": {},
        }

        result = evaluator(traces, "plain string result")

        assert result["domain"] == "research"
        assert result["passed"] is True
