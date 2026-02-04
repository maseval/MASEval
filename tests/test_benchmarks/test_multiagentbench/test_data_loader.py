"""Tests for MultiAgentBench data loading functionality."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from maseval import Task
from maseval.benchmark.multiagentbench.data_loader import (
    load_tasks,
    configure_model_ids,
    get_domain_info,
    download_marble,
    ensure_marble_exists,
    _get_marble_dir,
    VALID_DOMAINS,
    _parse_task_entry,
    _resolve_data_dir,
)


class TestValidDomains:
    """Tests for domain validation."""

    def test_valid_domains_contains_expected(self):
        """VALID_DOMAINS should contain all expected domains."""
        expected = {"coding", "database", "minecraft", "research", "bargaining", "web", "worldsimulation"}
        assert expected == VALID_DOMAINS

    def test_valid_domains_is_frozen(self):
        """VALID_DOMAINS should be immutable."""
        assert isinstance(VALID_DOMAINS, frozenset)


class TestGetDomainInfo:
    """Tests for get_domain_info function."""

    def test_research_domain_info(self):
        """get_domain_info should return correct info for research."""
        info = get_domain_info("research")
        assert info["requires_infrastructure"] is False
        assert info["coordination_mode"] == "cooperative"
        assert "description" in info

    def test_database_requires_infrastructure(self):
        """Database domain should require infrastructure."""
        info = get_domain_info("database")
        assert info["requires_infrastructure"] is True

    def test_minecraft_requires_infrastructure(self):
        """Minecraft domain should require infrastructure."""
        info = get_domain_info("minecraft")
        assert info["requires_infrastructure"] is True

    def test_invalid_domain_raises(self):
        """get_domain_info should raise for invalid domain."""
        with pytest.raises(ValueError, match="Invalid domain"):
            get_domain_info("invalid_domain")

    def test_case_insensitive(self):
        """get_domain_info should be case-insensitive."""
        info_lower = get_domain_info("research")
        info_upper = get_domain_info("RESEARCH")
        assert info_lower == info_upper


class TestParseTaskEntry:
    """Tests for _parse_task_entry function."""

    def test_parse_minimal_entry(self):
        """_parse_task_entry should parse a minimal valid entry."""
        entry = {
            "scenario": "research",
            "task_id": 1,
            "task": {"content": "Do research", "output_format": "5Q format"},
            "agents": [{"agent_id": "agent1", "profile": "Researcher"}],
            "relationships": [["agent1", "agent1", "self"]],
        }
        task = _parse_task_entry(entry, "research", 0)

        assert isinstance(task, Task)
        assert task.id == "research_1"
        assert task.query == "Do research"
        assert task.environment_data["scenario"] == "research"
        assert len(task.environment_data["agents"]) == 1

    def test_parse_entry_missing_required_field(self):
        """_parse_task_entry should raise for missing required fields."""
        entry = {
            "scenario": "research",
            "task_id": 1,
            # Missing "task", "agents", "relationships"
        }
        with pytest.raises(ValueError, match="missing required fields"):
            _parse_task_entry(entry, "research", 0)

    def test_parse_entry_missing_agent_id(self):
        """_parse_task_entry should raise if agent missing agent_id."""
        entry = {
            "scenario": "research",
            "task_id": 1,
            "task": {"content": "Do research"},
            "agents": [{"profile": "Researcher"}],  # Missing agent_id
            "relationships": [],
        }
        with pytest.raises(ValueError, match="missing 'agent_id'"):
            _parse_task_entry(entry, "research", 0)

    def test_parse_entry_empty_query(self):
        """_parse_task_entry should raise for empty query."""
        entry = {
            "scenario": "research",
            "task_id": 1,
            "task": {"content": "", "output_format": "5Q format"},
            "agents": [{"agent_id": "agent1"}],
            "relationships": [],
        }
        with pytest.raises(ValueError, match="empty query"):
            _parse_task_entry(entry, "research", 0)

    def test_parse_entry_with_string_task(self):
        """_parse_task_entry should handle task as string."""
        entry = {
            "scenario": "research",
            "task_id": 1,
            "task": "Do research task",
            "agents": [{"agent_id": "agent1"}],
            "relationships": [],
        }
        task = _parse_task_entry(entry, "research", 0)
        assert task.query == "Do research task"

    def test_parse_entry_preserves_metadata(self):
        """_parse_task_entry should preserve metadata correctly."""
        entry = {
            "scenario": "bargaining",
            "task_id": 42,
            "task": {"content": "Negotiate"},
            "agents": [{"agent_id": "buyer"}],
            "relationships": [],
            "coordinate_mode": "star",
            "environment": {"max_iterations": 20},
        }
        task = _parse_task_entry(entry, "bargaining", 0)

        assert task.metadata["domain"] == "bargaining"
        assert task.metadata["task_id"] == 42
        assert task.environment_data["coordinate_mode"] == "star"
        assert task.environment_data["max_iterations"] == 20


class TestLoadTasks:
    """Tests for load_tasks function."""

    def test_load_tasks_invalid_domain(self):
        """load_tasks should raise for invalid domain."""
        with pytest.raises(ValueError, match="Invalid domain"):
            load_tasks("invalid_domain")

    def test_load_tasks_missing_data_dir(self):
        """load_tasks should raise if data directory not found."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(FileNotFoundError, match="does not exist"):
                load_tasks("research", data_dir=Path("/nonexistent/path"))

    def test_load_tasks_with_mock_data(self):
        """load_tasks should load tasks from JSONL file."""
        # Create temporary JSONL file
        with tempfile.TemporaryDirectory() as tmpdir:
            research_dir = Path(tmpdir) / "research"
            research_dir.mkdir()
            jsonl_path = research_dir / "research_main.jsonl"

            # Write sample task
            task_data = {
                "scenario": "research",
                "task_id": 1,
                "task": {"content": "Research task", "output_format": "5Q"},
                "agents": [{"agent_id": "agent1", "profile": "Researcher"}],
                "relationships": [],
            }
            with jsonl_path.open("w") as f:
                f.write(json.dumps(task_data) + "\n")

            tasks = load_tasks("research", data_dir=Path(tmpdir))

            assert len(tasks) == 1
            assert tasks[0].query == "Research task"

    def test_load_tasks_with_limit(self):
        """load_tasks should respect limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            research_dir = Path(tmpdir) / "research"
            research_dir.mkdir()
            jsonl_path = research_dir / "research_main.jsonl"

            # Write multiple tasks
            with jsonl_path.open("w") as f:
                for i in range(5):
                    task_data = {
                        "scenario": "research",
                        "task_id": i + 1,
                        "task": {"content": f"Research task {i + 1}"},
                        "agents": [{"agent_id": f"agent{i + 1}"}],
                        "relationships": [],
                    }
                    f.write(json.dumps(task_data) + "\n")

            tasks = load_tasks("research", data_dir=Path(tmpdir), limit=2)

            assert len(tasks) == 2

    def test_load_tasks_case_insensitive_domain(self):
        """load_tasks should handle domain case-insensitively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            research_dir = Path(tmpdir) / "research"
            research_dir.mkdir()
            jsonl_path = research_dir / "research_main.jsonl"

            task_data = {
                "scenario": "research",
                "task_id": 1,
                "task": {"content": "Test"},
                "agents": [{"agent_id": "agent1"}],
                "relationships": [],
            }
            with jsonl_path.open("w") as f:
                f.write(json.dumps(task_data) + "\n")

            tasks_lower = load_tasks("research", data_dir=Path(tmpdir))
            tasks_upper = load_tasks("RESEARCH", data_dir=Path(tmpdir))

            assert len(tasks_lower) == len(tasks_upper) == 1


class TestConfigureModelIds:
    """Tests for configure_model_ids function."""

    def test_configure_model_ids_sets_llm(self):
        """configure_model_ids should set llm in environment_data."""
        task = Task(
            id="test_1",
            query="Test query",
            environment_data={"scenario": "research"},
            evaluation_data={},
            metadata={},
        )
        tasks = [task]

        configure_model_ids(tasks, agent_model_id="gpt-4o")

        assert tasks[0].environment_data["llm"] == "gpt-4o"

    def test_configure_model_ids_sets_evaluator_model(self):
        """configure_model_ids should set evaluator model_id."""
        task = Task(
            id="test_1",
            query="Test query",
            environment_data={},
            evaluation_data={},
            metadata={},
        )
        tasks = [task]

        configure_model_ids(
            tasks,
            agent_model_id="gpt-4o",
            evaluator_model_id="gpt-4o-mini",
        )

        assert tasks[0].evaluation_data["model_id"] == "gpt-4o-mini"

    def test_configure_model_ids_defaults_evaluator_to_agent(self):
        """configure_model_ids should default evaluator model to agent model."""
        task = Task(
            id="test_1",
            query="Test query",
            environment_data={},
            evaluation_data={},
            metadata={},
        )
        tasks = [task]

        configure_model_ids(tasks, agent_model_id="gpt-4o")

        assert tasks[0].evaluation_data["model_id"] == "gpt-4o"

    def test_configure_model_ids_returns_tasks(self):
        """configure_model_ids should return the input tasks."""
        task = Task(
            id="test_1",
            query="Test query",
            environment_data={},
            evaluation_data={},
            metadata={},
        )
        tasks = [task]

        result = configure_model_ids(tasks, agent_model_id="gpt-4o")

        assert result is tasks


class TestResolveDataDir:
    """Tests for _resolve_data_dir function."""

    def test_resolve_explicit_path(self):
        """_resolve_data_dir should use explicit path if provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _resolve_data_dir(Path(tmpdir))
            assert result == Path(tmpdir)

    def test_resolve_nonexistent_explicit_path(self):
        """_resolve_data_dir should raise for nonexistent explicit path."""
        with pytest.raises(FileNotFoundError):
            _resolve_data_dir(Path("/nonexistent/path"))

    def test_resolve_from_env_var(self):
        """_resolve_data_dir should use MARBLE_DATA_DIR env var."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"MARBLE_DATA_DIR": tmpdir}):
                result = _resolve_data_dir()
                assert result == Path(tmpdir)

    def test_resolve_not_found(self):
        """_resolve_data_dir should raise when no directory found."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("pathlib.Path.cwd", return_value=Path("/nonexistent/cwd")):
                # Mock Path.exists to return False for all candidate paths
                with patch.object(Path, "exists", return_value=False):
                    with pytest.raises(FileNotFoundError, match="MARBLE data directory not found"):
                        _resolve_data_dir()


class TestGetMarbleDir:
    """Tests for _get_marble_dir function."""

    def test_returns_path_relative_to_module(self):
        """_get_marble_dir should return path relative to module."""
        result = _get_marble_dir()
        assert result.name == "marble"
        assert "multiagentbench" in str(result.parent)


class TestDownloadMarble:
    """Tests for download_marble function."""

    def test_download_marble_already_exists(self):
        """download_marble should return existing path if not force."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marble_dir = Path(tmpdir) / "marble"
            marble_dir.mkdir()

            result = download_marble(target_dir=marble_dir, force=False)

            assert result == marble_dir

    def test_download_marble_force_removes_existing(self):
        """download_marble should remove existing dir when force=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marble_dir = Path(tmpdir) / "marble"
            marble_dir.mkdir()
            (marble_dir / "test_file.txt").write_text("test")

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                download_marble(target_dir=marble_dir, force=True)

                # Directory should have been removed and git clone called
                mock_run.assert_called()

    def test_download_marble_git_clone_called(self):
        """download_marble should call git clone."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marble_dir = Path(tmpdir) / "marble"

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                download_marble(target_dir=marble_dir)

                # Verify git clone was called
                calls = mock_run.call_args_list
                assert len(calls) >= 1
                clone_call = calls[0]
                assert "git" in clone_call[0][0]
                assert "clone" in clone_call[0][0]

    def test_download_marble_with_commit(self):
        """download_marble should checkout specific commit if provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marble_dir = Path(tmpdir) / "marble"

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                download_marble(target_dir=marble_dir, commit="abc123")

                # Verify git checkout was called
                calls = mock_run.call_args_list
                assert len(calls) >= 2
                checkout_call = calls[1]
                assert "checkout" in checkout_call[0][0]
                assert "abc123" in checkout_call[0][0]

    def test_download_marble_clone_fails(self):
        """download_marble should raise RuntimeError on clone failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marble_dir = Path(tmpdir) / "marble"

            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.CalledProcessError(1, "git clone", stderr="Clone failed")

                with pytest.raises(RuntimeError, match="Failed to clone MARBLE"):
                    download_marble(target_dir=marble_dir)

    def test_download_marble_git_not_found(self):
        """download_marble should raise RuntimeError if git not installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marble_dir = Path(tmpdir) / "marble"

            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError()

                with pytest.raises(RuntimeError, match="git is not installed"):
                    download_marble(target_dir=marble_dir)

    def test_download_marble_checkout_fails(self):
        """download_marble should raise RuntimeError on checkout failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marble_dir = Path(tmpdir) / "marble"

            def mock_run_side_effect(*args, **kwargs):
                cmd = args[0]
                if "checkout" in cmd:
                    raise subprocess.CalledProcessError(1, "git checkout", stderr="Checkout failed")
                return MagicMock(returncode=0)

            with patch("subprocess.run", side_effect=mock_run_side_effect):
                with pytest.raises(RuntimeError, match="Failed to checkout commit"):
                    download_marble(target_dir=marble_dir, commit="invalid")


class TestEnsureMarbleExists:
    """Tests for ensure_marble_exists function."""

    def test_ensure_marble_exists_already_present(self):
        """ensure_marble_exists should return path if MARBLE exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marble_dir = Path(tmpdir) / "marble"
            marble_dir.mkdir()
            (marble_dir / "multiagentbench").mkdir()

            with patch(
                "maseval.benchmark.multiagentbench.data_loader._get_marble_dir",
                return_value=marble_dir,
            ):
                result = ensure_marble_exists(auto_download=False)

            assert result == marble_dir

    def test_ensure_marble_exists_not_present_no_download(self):
        """ensure_marble_exists should raise if not present and auto_download=False."""
        with patch(
            "maseval.benchmark.multiagentbench.data_loader._get_marble_dir",
            return_value=Path("/nonexistent/marble"),
        ):
            with pytest.raises(FileNotFoundError, match="MARBLE not found"):
                ensure_marble_exists(auto_download=False)

    def test_ensure_marble_exists_auto_download(self):
        """ensure_marble_exists should download if not present and auto_download=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marble_dir = Path(tmpdir) / "marble"

            with patch(
                "maseval.benchmark.multiagentbench.data_loader._get_marble_dir",
                return_value=marble_dir,
            ):
                with patch(
                    "maseval.benchmark.multiagentbench.data_loader.download_marble",
                    return_value=marble_dir,
                ) as mock_download:
                    result = ensure_marble_exists(auto_download=True)

                    mock_download.assert_called_once_with(marble_dir)
                    assert result == marble_dir


class TestLoadTasksEdgeCases:
    """Edge case tests for load_tasks."""

    def test_load_tasks_empty_lines(self):
        """load_tasks should skip empty lines in JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            research_dir = Path(tmpdir) / "research"
            research_dir.mkdir()
            jsonl_path = research_dir / "research_main.jsonl"

            # Write tasks with empty lines
            with jsonl_path.open("w") as f:
                task_data: Dict[str, Any] = {
                    "scenario": "research",
                    "task_id": 1,
                    "task": {"content": "Task 1"},
                    "agents": [{"agent_id": "agent1"}],
                    "relationships": [],
                }
                f.write(json.dumps(task_data) + "\n")
                f.write("\n")  # Empty line
                f.write("   \n")  # Whitespace-only line
                task_data["task_id"] = 2
                task_data["task"]["content"] = "Task 2"  # type: ignore[index]
                f.write(json.dumps(task_data) + "\n")

            tasks = load_tasks("research", data_dir=Path(tmpdir))

            assert len(tasks) == 2

    def test_load_tasks_file_not_found(self):
        """load_tasks should raise FileNotFoundError if JSONL file missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory but not the JSONL file
            research_dir = Path(tmpdir) / "research"
            research_dir.mkdir()

            with pytest.raises(FileNotFoundError, match="Task data not found"):
                load_tasks("research", data_dir=Path(tmpdir))


class TestGetDomainInfoAllDomains:
    """Test get_domain_info for all domains."""

    @pytest.mark.parametrize(
        "domain",
        ["coding", "database", "minecraft", "research", "bargaining", "web", "worldsimulation"],
    )
    def test_all_domains_have_info(self, domain):
        """All valid domains should return info."""
        info = get_domain_info(domain)
        assert "requires_infrastructure" in info
        assert "description" in info
        assert "coordination_mode" in info

    def test_coding_domain_info(self):
        """Coding domain should have tree coordination."""
        info = get_domain_info("coding")
        assert info["coordination_mode"] == "tree"
        assert info["requires_infrastructure"] is False

    def test_web_domain_info(self):
        """Web domain should have star coordination."""
        info = get_domain_info("web")
        assert info["coordination_mode"] == "star"
        assert info["requires_infrastructure"] is False

    def test_worldsimulation_domain_info(self):
        """WorldSimulation domain should have cooperative coordination."""
        info = get_domain_info("worldsimulation")
        assert info["coordination_mode"] == "cooperative"
        assert info["requires_infrastructure"] is False
