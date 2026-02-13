"""Unit tests for MACS data_loader module."""

import json
import pytest
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import patch, MagicMock
from tempfile import TemporaryDirectory
from urllib.error import URLError, HTTPError
from email.message import Message

from maseval.benchmark.macs.data_loader import (
    DEFAULT_DATA_DIR,
    VALID_DOMAINS,
    DATA_URLS,
    EVALUATION_URLS,
    download_file,
    download_json,
    download_original_data,
    download_prompt_templates,
    restructure_data,
    ensure_data_exists,
    load_tasks,
    load_agent_config,
    _dedupe_tools_by_name,
    _create_tools_list,
    _create_agents_list,
    _create_tasks_list,
)


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_agents_data() -> Dict[str, Any]:
    """Sample agents.json data matching AWS format."""
    return {
        "agents": [
            {
                "agent_id": "supervisor",
                "agent_name": "Supervisor Agent",
                "agent_instruction": "Coordinate other agents.",
                "reachable_agents": ["worker_1"],
                "tools": [
                    {"tool_name": "route_to_agent", "tool_type": "router"},
                    {"tool_name": "common_tool", "description": "Shared tool"},
                ],
            },
            {
                "agent_id": "worker_1",
                "agent_name": "Worker Agent",
                "agent_instruction": "Do work.",
                "reachable_agents": [],
                "tools": [
                    {"tool_name": "do_work", "tool_type": "action"},
                    {"tool_name": "common_tool", "description": "Shared tool"},  # Duplicate
                ],
            },
        ],
        "primary_agent_id": "supervisor",
        "human_id": "user",
    }


@pytest.fixture
def sample_scenarios_data() -> Dict[str, Any]:
    """Sample scenarios.json data matching AWS format (no IDs - they get generated)."""
    return {
        "scenarios": [
            {
                "scenario": "Bicycle tour planning",
                "input_problem": "Book a flight to New York",
                "assertions": [
                    "user: Flight booked",
                    "agent: Database updated",
                ],
                "category": "travel",
                "complexity": "simple",
            },
            {
                "scenario": "Reservation cancellation",
                "input_problem": "Cancel my reservation",
                "assertions": [
                    "user: Reservation cancelled",
                ],
                "category": "travel",
                "complexity": "simple",
            },
        ]
    }


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Unit Tests: Helper Functions
# =============================================================================


@pytest.mark.benchmark
class TestDedupeToolsByName:
    """Tests for _dedupe_tools_by_name function."""

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert _dedupe_tools_by_name([]) == []

    def test_no_duplicates(self):
        """Tools without duplicates are preserved."""
        tools = [
            {"tool_name": "a", "param": 1},
            {"tool_name": "b", "param": 2},
        ]
        result = _dedupe_tools_by_name(tools)
        assert len(result) == 2
        assert {t["tool_name"] for t in result} == {"a", "b"}

    def test_exact_duplicates_deduped(self):
        """Identical duplicate tools are deduplicated."""
        tools = [
            {"tool_name": "a", "param": 1},
            {"tool_name": "a", "param": 1},  # Exact duplicate
        ]
        result = _dedupe_tools_by_name(tools)
        assert len(result) == 1
        assert result[0]["tool_name"] == "a"

    def test_conflicting_duplicates_raise(self):
        """Conflicting tools with same name raise ValueError."""
        tools = [
            {"tool_name": "a", "param": 1},
            {"tool_name": "a", "param": 2},  # Different params!
        ]
        with pytest.raises(ValueError, match="Conflicting tools"):
            _dedupe_tools_by_name(tools)

    def test_tools_without_name_preserved(self):
        """Tools without tool_name are preserved as-is."""
        tools = [
            {"tool_name": "a"},
            {"no_name": "here"},
            {},
        ]
        result = _dedupe_tools_by_name(tools)
        assert len(result) == 3


@pytest.mark.benchmark
class TestCreateToolsList:
    """Tests for _create_tools_list function."""

    def test_dict_with_agents_key(self, sample_agents_data):
        """Extract tools from dict with 'agents' key."""
        tools = _create_tools_list(sample_agents_data)
        tool_names = {t["tool_name"] for t in tools}
        assert "route_to_agent" in tool_names
        assert "do_work" in tool_names
        assert "common_tool" in tool_names
        # Duplicates should be deduped
        assert sum(1 for t in tools if t.get("tool_name") == "common_tool") == 1

    def test_list_of_agents(self, sample_agents_data):
        """Extract tools from list of agents directly."""
        tools = _create_tools_list(sample_agents_data["agents"])
        assert len(tools) == 3

    def test_empty_input(self):
        """Empty or invalid input returns empty list."""
        assert _create_tools_list({}) == []
        assert _create_tools_list([]) == []


@pytest.mark.benchmark
class TestCreateAgentsList:
    """Tests for _create_agents_list function."""

    def test_basic_conversion(self, sample_agents_data):
        """Converts agents and replaces tool dicts with names."""
        result = _create_agents_list(sample_agents_data)

        assert "agents" in result
        assert result["primary_agent_id"] == "supervisor"
        assert result["human_id"] == "user"

        agents = result["agents"]
        assert len(agents) == 2

        supervisor = next(a for a in agents if a["agent_id"] == "supervisor")
        assert supervisor["agent_name"] == "Supervisor Agent"
        assert supervisor["tools"] == ["route_to_agent", "common_tool"]
        assert "agent_instruction" in supervisor

    def test_empty_input(self):
        """Empty input returns empty dict."""
        assert _create_agents_list({}) == {}
        assert _create_agents_list([]) == {}


@pytest.mark.benchmark
class TestCreateTasksList:
    """Tests for _create_tasks_list function."""

    def test_basic_conversion(self, sample_scenarios_data, sample_agents_data):
        """Converts scenarios to task format with sequential IDs."""
        tools = _create_tools_list(sample_agents_data)
        tasks = _create_tasks_list(sample_scenarios_data, tools)

        assert len(tasks) == 2

        task1 = tasks[0]
        assert task1["id"] == "task-000001"  # Sequential ID generated
        assert task1["query"] == "Book a flight to New York"
        assert "tools" in task1["environment_data"]
        assert "assertions" in task1["evaluation_data"]
        assert task1["metadata"]["category"] == "travel"
        assert task1["metadata"]["scenario"] == "Bicycle tour planning"

    def test_list_of_scenarios(self, sample_scenarios_data, sample_agents_data):
        """Also works with list of scenarios directly."""
        tools = _create_tools_list(sample_agents_data)
        tasks = _create_tasks_list(sample_scenarios_data["scenarios"], tools)
        assert len(tasks) == 2

    def test_empty_input(self):
        """Empty input returns empty list."""
        assert _create_tasks_list({}, []) == []
        assert _create_tasks_list([], []) == []


# =============================================================================
# Unit Tests: Download Functions (with mocking)
# =============================================================================


@pytest.mark.benchmark
class TestDownloadFunctions:
    """Tests for download functions using mocks."""

    def test_download_file_success(self):
        """download_file returns text content."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"key": "value"}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("maseval.benchmark.macs.data_loader.urlopen", return_value=mock_resp):
            result = download_file("http://example.com/test.json")
            assert result == '{"key": "value"}'

    def test_download_json_success(self):
        """download_json returns parsed JSON."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"key": "value"}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("maseval.benchmark.macs.data_loader.urlopen", return_value=mock_resp):
            result = download_json("http://example.com/test.json")
            assert result == {"key": "value"}

    def test_download_json_invalid_json(self):
        """download_json raises on invalid JSON."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not valid json"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("maseval.benchmark.macs.data_loader.urlopen", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="Failed to decode JSON"):
                download_json("http://example.com/test.json")


@pytest.mark.benchmark
class TestDownloadOriginalData:
    """Tests for download_original_data function."""

    def test_downloads_all_domains(self, temp_data_dir, sample_agents_data, sample_scenarios_data):
        """Downloads data for all domains."""

        def mock_download_json(url: str):
            if "agents" in url:
                return sample_agents_data
            if "scenarios" in url:
                return sample_scenarios_data
            raise ValueError(f"Unexpected URL: {url}")

        with patch("maseval.benchmark.macs.data_loader.download_json", side_effect=mock_download_json):
            result = download_original_data(data_dir=temp_data_dir, verbose=0)

        assert result == temp_data_dir / "original"
        for domain in VALID_DOMAINS:
            assert (result / domain / "agents.json").exists()
            assert (result / domain / "scenarios.json").exists()

    def test_downloads_single_domain(self, temp_data_dir, sample_agents_data, sample_scenarios_data):
        """Downloads data for a single domain."""

        def mock_download_json(url: str):
            if "agents" in url:
                return sample_agents_data
            if "scenarios" in url:
                return sample_scenarios_data
            raise ValueError(f"Unexpected URL: {url}")

        with patch("maseval.benchmark.macs.data_loader.download_json", side_effect=mock_download_json):
            result = download_original_data(data_dir=temp_data_dir, domain="travel", verbose=0)

        assert (result / "travel" / "agents.json").exists()
        # Other domains should NOT exist
        assert not (result / "mortgage").exists()
        assert not (result / "software").exists()

    def test_invalid_domain_raises(self, temp_data_dir):
        """Invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="Unknown domain"):
            download_original_data(data_dir=temp_data_dir, domain="invalid", verbose=0)


# =============================================================================
# Unit Tests: Restructure Functions
# =============================================================================


@pytest.mark.benchmark
class TestRestructureData:
    """Tests for restructure_data function."""

    def test_restructures_all_domains(self, temp_data_dir, sample_agents_data, sample_scenarios_data):
        """Restructures data for all domains."""
        # Set up original data
        for domain in VALID_DOMAINS:
            orig_dir = temp_data_dir / "original" / domain
            orig_dir.mkdir(parents=True)
            with (orig_dir / "agents.json").open("w") as f:
                json.dump(sample_agents_data, f)
            with (orig_dir / "scenarios.json").open("w") as f:
                json.dump(sample_scenarios_data, f)

        result = restructure_data(data_dir=temp_data_dir, verbose=0)

        assert result == temp_data_dir / "restructured"
        for domain in VALID_DOMAINS:
            assert (result / domain / "agents.json").exists()
            assert (result / domain / "tasks.json").exists()

            # Verify content
            with (result / domain / "tasks.json").open() as f:
                tasks = json.load(f)
            assert len(tasks) == 2
            assert tasks[0]["query"] == "Book a flight to New York"

            with (result / domain / "agents.json").open() as f:
                agents = json.load(f)
            assert len(agents["agents"]) == 2
            assert agents["primary_agent_id"] == "supervisor"

    def test_missing_original_raises(self, temp_data_dir):
        """Missing original data raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Original data not found"):
            restructure_data(data_dir=temp_data_dir, domain="travel", verbose=0)


@pytest.mark.benchmark
class TestEnsureDataExists:
    """Tests for ensure_data_exists function."""

    def test_skips_download_if_exists(self, temp_data_dir):
        """Skips download if restructured data already exists."""
        # Create fake restructured data
        for domain in VALID_DOMAINS:
            rest_dir = temp_data_dir / "restructured" / domain
            rest_dir.mkdir(parents=True)
            (rest_dir / "tasks.json").write_text("[]")
            (rest_dir / "agents.json").write_text("{}")

        # Should not call download
        with patch("maseval.benchmark.macs.data_loader.download_original_data") as mock_download:
            result = ensure_data_exists(data_dir=temp_data_dir, verbose=0)
            mock_download.assert_not_called()

        assert result == temp_data_dir / "restructured"

    def test_downloads_if_missing(self, temp_data_dir, sample_agents_data, sample_scenarios_data):
        """Downloads if restructured data missing."""
        # No data exists

        def mock_download_json(url: str):
            if "agents" in url:
                return sample_agents_data
            if "scenarios" in url:
                return sample_scenarios_data
            raise ValueError(f"Unexpected URL: {url}")

        def mock_download_file(url: str, timeout=15):
            # Return minimal Python file for prompt templates
            return "USER_GSR_PROMPT = 'test'\nSYSTEM_GSR_PROMPT = 'test'\nISSUES_PROMPT = 'test'"

        with patch("maseval.benchmark.macs.data_loader.download_json", side_effect=mock_download_json):
            with patch("maseval.benchmark.macs.data_loader.download_file", side_effect=mock_download_file):
                result = ensure_data_exists(data_dir=temp_data_dir, verbose=0)

        assert result == temp_data_dir / "restructured"
        for domain in VALID_DOMAINS:
            assert (result / domain / "tasks.json").exists()

    def test_force_download_redownloads(self, temp_data_dir, sample_agents_data, sample_scenarios_data):
        """force_download=True redownloads even if data exists."""
        # Create fake restructured data
        for domain in VALID_DOMAINS:
            rest_dir = temp_data_dir / "restructured" / domain
            rest_dir.mkdir(parents=True)
            (rest_dir / "tasks.json").write_text("[]")
            (rest_dir / "agents.json").write_text("{}")

        def mock_download_json(url: str):
            if "agents" in url:
                return sample_agents_data
            if "scenarios" in url:
                return sample_scenarios_data
            raise ValueError(f"Unexpected URL: {url}")

        def mock_download_file(url: str, timeout=15):
            return "USER_GSR_PROMPT = 'test'\nSYSTEM_GSR_PROMPT = 'test'\nISSUES_PROMPT = 'test'"

        with patch("maseval.benchmark.macs.data_loader.download_json", side_effect=mock_download_json):
            with patch("maseval.benchmark.macs.data_loader.download_file", side_effect=mock_download_file):
                result = ensure_data_exists(data_dir=temp_data_dir, force_download=True, verbose=0)

        # Should have new data with 2 tasks
        with (result / "travel" / "tasks.json").open() as f:
            tasks = json.load(f)
        assert len(tasks) == 2


# =============================================================================
# Unit Tests: Load Functions
# =============================================================================


@pytest.mark.benchmark
class TestLoadTasks:
    """Tests for load_tasks function."""

    def test_loads_tasks(self, temp_data_dir, sample_agents_data, sample_scenarios_data):
        """Loads tasks from restructured data."""
        # Create restructured data
        tools = _create_tools_list(sample_agents_data)
        tasks_list = _create_tasks_list(sample_scenarios_data, tools)

        rest_dir = temp_data_dir / "restructured" / "travel"
        rest_dir.mkdir(parents=True)
        with (rest_dir / "tasks.json").open("w") as f:
            json.dump(tasks_list, f)

        collection = load_tasks("travel", data_dir=temp_data_dir)

        assert len(collection) == 2
        task = collection[0]
        assert task.query == "Book a flight to New York"
        assert "tools" in task.environment_data
        assert "assertions" in task.evaluation_data

    def test_limit_parameter(self, temp_data_dir, sample_agents_data, sample_scenarios_data):
        """Limit parameter restricts number of tasks."""
        tools = _create_tools_list(sample_agents_data)
        tasks_list = _create_tasks_list(sample_scenarios_data, tools)

        rest_dir = temp_data_dir / "restructured" / "travel"
        rest_dir.mkdir(parents=True)
        with (rest_dir / "tasks.json").open("w") as f:
            json.dump(tasks_list, f)

        collection = load_tasks("travel", data_dir=temp_data_dir, limit=1)
        assert len(collection) == 1

    def test_invalid_domain_raises(self, temp_data_dir):
        """Invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="Invalid domain"):
            load_tasks("invalid", data_dir=temp_data_dir)

    def test_missing_file_raises(self, temp_data_dir):
        """Missing tasks.json raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_tasks("travel", data_dir=temp_data_dir)


@pytest.mark.benchmark
class TestLoadAgentConfig:
    """Tests for load_agent_config function."""

    def test_loads_config(self, temp_data_dir, sample_agents_data):
        """Loads agent config from restructured data."""
        agents = _create_agents_list(sample_agents_data)

        rest_dir = temp_data_dir / "restructured" / "travel"
        rest_dir.mkdir(parents=True)
        with (rest_dir / "agents.json").open("w") as f:
            json.dump(agents, f)

        config = load_agent_config("travel", data_dir=temp_data_dir)

        assert "agents" in config
        assert config["primary_agent_id"] == "supervisor"
        assert len(config["agents"]) == 2

        supervisor = next(a for a in config["agents"] if a["agent_id"] == "supervisor")
        assert supervisor["tools"] == ["route_to_agent", "common_tool"]

    def test_invalid_domain_raises(self, temp_data_dir):
        """Invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="Invalid domain"):
            load_agent_config("invalid", data_dir=temp_data_dir)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestDataLoaderIntegration:
    """Integration tests for the full data loading pipeline."""

    def test_full_pipeline(self, temp_data_dir, sample_agents_data, sample_scenarios_data):
        """Test the complete download → restructure → load pipeline."""

        def mock_download_json(url: str):
            if "agents" in url:
                return sample_agents_data
            if "scenarios" in url:
                return sample_scenarios_data
            raise ValueError(f"Unexpected URL: {url}")

        def mock_download_file(url: str, timeout=15):
            return "USER_GSR_PROMPT = 'test'\nSYSTEM_GSR_PROMPT = 'test'\nISSUES_PROMPT = 'test'"

        with patch("maseval.benchmark.macs.data_loader.download_json", side_effect=mock_download_json):
            with patch("maseval.benchmark.macs.data_loader.download_file", side_effect=mock_download_file):
                # Step 1: Download
                download_original_data(data_dir=temp_data_dir, verbose=0)
                download_prompt_templates(data_dir=temp_data_dir, verbose=0)

                # Verify original data exists
                assert (temp_data_dir / "original" / "travel" / "agents.json").exists()
                assert (temp_data_dir / "original" / "travel" / "scenarios.json").exists()

                # Step 2: Restructure
                restructure_data(data_dir=temp_data_dir, verbose=0)

                # Verify restructured data exists
                assert (temp_data_dir / "restructured" / "travel" / "tasks.json").exists()
                assert (temp_data_dir / "restructured" / "travel" / "agents.json").exists()

                # Step 3: Load
                tasks = load_tasks("travel", data_dir=temp_data_dir)
                config = load_agent_config("travel", data_dir=temp_data_dir)

                # Verify loaded data
                assert len(tasks) == 2
                assert tasks[0].query == "Book a flight to New York"
                assert len(config["agents"]) == 2

    def test_urls_structure(self):
        """Verify URL constants have expected structure."""
        for domain in VALID_DOMAINS:
            assert domain in DATA_URLS
            assert "agents" in DATA_URLS[domain]
            assert "scenarios" in DATA_URLS[domain]

        assert "prompt_templates" in EVALUATION_URLS


# =============================================================================
# Connection Error Handling Tests
# =============================================================================


@pytest.mark.benchmark
class TestConnectionErrorHandling:
    """Tests for graceful handling of network errors."""

    def test_download_file_http_error(self):
        """download_file raises RuntimeError on HTTP error."""
        with patch("maseval.benchmark.macs.data_loader.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = HTTPError(
                url="http://example.com/test.json",
                code=404,
                msg="Not Found",
                hdrs=Message(),
                fp=None,
            )
            with pytest.raises(RuntimeError, match="Failed to download"):
                download_file("http://example.com/test.json")

    def test_download_file_url_error(self):
        """download_file raises RuntimeError on URL/network error."""
        with patch("maseval.benchmark.macs.data_loader.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = URLError("Connection refused")
            with pytest.raises(RuntimeError, match="Failed to download"):
                download_file("http://example.com/test.json")

    def test_download_file_timeout(self):
        """download_file raises RuntimeError on timeout."""
        with patch("maseval.benchmark.macs.data_loader.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = URLError("timed out")
            with pytest.raises(RuntimeError, match="Failed to download"):
                download_file("http://example.com/test.json")

    def test_download_original_data_network_error(self, temp_data_dir):
        """download_original_data propagates network errors gracefully."""
        with patch("maseval.benchmark.macs.data_loader.download_json") as mock_download:
            mock_download.side_effect = RuntimeError("Network unreachable")
            with pytest.raises(RuntimeError, match="Network unreachable"):
                download_original_data(data_dir=temp_data_dir, domain="travel", verbose=0)

    def test_ensure_data_exists_network_error(self, temp_data_dir):
        """ensure_data_exists propagates network errors when data missing."""
        with patch("maseval.benchmark.macs.data_loader.download_json") as mock_download:
            mock_download.side_effect = RuntimeError("Connection failed")
            with pytest.raises(RuntimeError, match="Connection failed"):
                ensure_data_exists(data_dir=temp_data_dir, verbose=0)


# =============================================================================
# Data Location Tests
# =============================================================================


@pytest.mark.benchmark
class TestDataLocation:
    """Tests for custom and default data location handling."""

    def test_default_data_dir_is_module_relative(self):
        """DEFAULT_DATA_DIR points to module's data directory."""
        assert DEFAULT_DATA_DIR.name == "data"
        assert DEFAULT_DATA_DIR.parent.name == "macs"
        assert DEFAULT_DATA_DIR.parent.parent.name == "benchmark"

    def test_load_tasks_uses_default_location(self, sample_agents_data, sample_scenarios_data):
        """load_tasks uses DEFAULT_DATA_DIR when data_dir not specified."""
        # Create mock data in default location
        tools = _create_tools_list(sample_agents_data)
        tasks_list = _create_tasks_list(sample_scenarios_data, tools)

        rest_dir = DEFAULT_DATA_DIR / "restructured" / "travel"
        rest_dir.mkdir(parents=True, exist_ok=True)
        tasks_path = rest_dir / "tasks.json"

        # Write test data
        with tasks_path.open("w") as f:
            json.dump(tasks_list, f)

        try:
            # Load without specifying data_dir
            collection = load_tasks("travel")
            assert len(collection) == 2
        finally:
            # Cleanup
            tasks_path.unlink(missing_ok=True)
            # Don't remove dirs as other tests may use them

    def test_load_tasks_custom_location(self, temp_data_dir, sample_agents_data, sample_scenarios_data):
        """load_tasks correctly uses custom data_dir."""
        tools = _create_tools_list(sample_agents_data)
        tasks_list = _create_tasks_list(sample_scenarios_data, tools)

        rest_dir = temp_data_dir / "restructured" / "travel"
        rest_dir.mkdir(parents=True)
        with (rest_dir / "tasks.json").open("w") as f:
            json.dump(tasks_list, f)

        # Load from custom location
        collection = load_tasks("travel", data_dir=temp_data_dir)
        assert len(collection) == 2

        # Verify it didn't touch default location
        default_tasks = DEFAULT_DATA_DIR / "restructured" / "nonexistent_test_domain" / "tasks.json"
        assert not default_tasks.exists()

    def test_load_agent_config_custom_location(self, temp_data_dir, sample_agents_data):
        """load_agent_config correctly uses custom data_dir."""
        agents = _create_agents_list(sample_agents_data)

        rest_dir = temp_data_dir / "restructured" / "mortgage"
        rest_dir.mkdir(parents=True)
        with (rest_dir / "agents.json").open("w") as f:
            json.dump(agents, f)

        config = load_agent_config("mortgage", data_dir=temp_data_dir)
        assert config["primary_agent_id"] == "supervisor"

    def test_download_original_data_custom_location(self, temp_data_dir, sample_agents_data, sample_scenarios_data):
        """download_original_data saves to custom data_dir."""

        def mock_download_json(url: str):
            if "agents" in url:
                return sample_agents_data
            if "scenarios" in url:
                return sample_scenarios_data
            raise ValueError(f"Unexpected URL: {url}")

        with patch("maseval.benchmark.macs.data_loader.download_json", side_effect=mock_download_json):
            result = download_original_data(data_dir=temp_data_dir, domain="software", verbose=0)

        # Verify data is in custom location
        assert result == temp_data_dir / "original"
        assert (temp_data_dir / "original" / "software" / "agents.json").exists()

        # Verify default location is untouched
        assert not (DEFAULT_DATA_DIR / "original" / "software_test_marker").exists()

    def test_restructure_uses_custom_location(self, temp_data_dir, sample_agents_data, sample_scenarios_data):
        """restructure_data reads from and writes to custom data_dir."""
        # Setup original data in custom location
        orig_dir = temp_data_dir / "original" / "travel"
        orig_dir.mkdir(parents=True)
        with (orig_dir / "agents.json").open("w") as f:
            json.dump(sample_agents_data, f)
        with (orig_dir / "scenarios.json").open("w") as f:
            json.dump(sample_scenarios_data, f)

        # Restructure
        result = restructure_data(data_dir=temp_data_dir, domain="travel", verbose=0)

        # Verify output is in custom location
        assert result == temp_data_dir / "restructured"
        assert (temp_data_dir / "restructured" / "travel" / "tasks.json").exists()
        assert (temp_data_dir / "restructured" / "travel" / "agents.json").exists()


# =============================================================================
# Sequential ID Generation Tests
# =============================================================================


@pytest.mark.benchmark
class TestSequentialIdGeneration:
    """Tests for task ID generation."""

    def test_generates_sequential_ids(self):
        """_create_tasks_list always generates sequential IDs."""
        scenarios = {
            "scenarios": [
                {"input_problem": "Task 1", "assertions": []},
                {"input_problem": "Task 2", "assertions": []},
                {"input_problem": "Task 3", "assertions": []},
            ]
        }
        tasks = _create_tasks_list(scenarios, [])

        assert tasks[0]["id"] == "task-000001"
        assert tasks[1]["id"] == "task-000002"
        assert tasks[2]["id"] == "task-000003"

    def test_sequential_ids_ignore_original_ids(self):
        """_create_tasks_list ignores any existing id/uuid fields in scenarios."""
        scenarios = {
            "scenarios": [
                {"id": "should-be-ignored", "input_problem": "Task 1", "assertions": []},
                {"uuid": "also-ignored", "input_problem": "Task 2", "assertions": []},
                {"input_problem": "Task 3", "assertions": []},
            ]
        }
        tasks = _create_tasks_list(scenarios, [])

        # All get sequential IDs regardless of original id/uuid fields
        assert tasks[0]["id"] == "task-000001"
        assert tasks[1]["id"] == "task-000002"
        assert tasks[2]["id"] == "task-000003"

        # Original id/uuid are preserved in metadata
        assert tasks[0]["metadata"].get("id") == "should-be-ignored"
        assert tasks[1]["metadata"].get("uuid") == "also-ignored"
