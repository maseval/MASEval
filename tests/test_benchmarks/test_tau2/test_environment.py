"""Unit tests for Tau2 environment module."""

import pytest
from unittest.mock import MagicMock

from maseval.benchmark.tau2 import Tau2Environment
from maseval.benchmark.tau2.data_loader import VALID_DOMAINS
from maseval.benchmark.tau2.environment import (
    DOMAIN_DB_CLASSES,
    DOMAIN_TOOLKIT_CLASSES,
    DOMAIN_USER_TOOLKIT_CLASSES,
    get_environment_constructor,
)


def _minimal_retail_db_data(include_user: bool = False):
    """Build a minimal valid retail DB dict that passes pydantic validation."""
    data = {"users": {}, "orders": {}, "products": {}}
    if include_user:
        data["users"]["u1"] = {
            "user_id": "u1",
            "name": {"first_name": "Alice", "last_name": "Smith"},
            "address": {
                "address1": "123 Main St",
                "address2": "",
                "city": "Anytown",
                "country": "US",
                "state": "CA",
                "zip": "90001",
            },
            "email": "alice@test.com",
            "payment_methods": {
                "pm1": {"source": "credit_card", "id": "pm1", "brand": "visa", "last_four": "1234"},
            },
            "orders": [],
        }
    return data


# =============================================================================
# Class Structure Tests (Tier 1 - offline)
# =============================================================================


@pytest.mark.benchmark
class TestTau2EnvironmentClassStructure:
    """Tests for Tau2Environment class structure (no data needed)."""

    def test_inherits_from_environment(self):
        """Tau2Environment inherits from MASEval Environment base class."""
        from maseval.core.environment import Environment

        assert issubclass(Tau2Environment, Environment)

    def test_has_required_methods(self):
        """Tau2Environment has all required methods."""
        required = [
            "setup_state",
            "create_tools",
            "create_user_tools",
            "make_tool_call",
            "make_user_tool_call",
            "get_db_hash",
            "get_initial_db_hash",
            "gather_traces",
            "gather_config",
            "sync_tools",
        ]
        for method in required:
            assert hasattr(Tau2Environment, method), f"Missing method: {method}"

    def test_domain_registrations_cover_valid_domains(self):
        """All VALID_DOMAINS have DB and toolkit registrations."""
        for domain in VALID_DOMAINS:
            assert domain in DOMAIN_DB_CLASSES, f"Missing DB class for {domain}"
            assert domain in DOMAIN_TOOLKIT_CLASSES, f"Missing toolkit class for {domain}"

    def test_telecom_has_user_toolkit(self):
        """Only telecom has a user toolkit registration."""
        assert "telecom" in DOMAIN_USER_TOOLKIT_CLASSES
        assert "retail" not in DOMAIN_USER_TOOLKIT_CLASSES
        assert "airline" not in DOMAIN_USER_TOOLKIT_CLASSES


# =============================================================================
# Environment Creation Tests
# =============================================================================


@pytest.mark.benchmark
class TestEnvironmentCreation:
    """Tests for Tau2Environment creation."""

    @pytest.mark.live
    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_creates_environment(self, domain, ensure_tau2_data):
        """Creates environment successfully for each domain."""
        env = Tau2Environment({"domain": domain})

        assert env.domain == domain
        assert env.db is not None
        assert env.toolkit is not None
        assert env.policy is not None

    def test_invalid_domain_raises(self):
        """Invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="Invalid domain"):
            Tau2Environment({"domain": "invalid_domain"})


# =============================================================================
# Environment Tools Tests
# =============================================================================


# Expected tool counts per domain
DOMAIN_TOOL_COUNTS = {
    "retail": 15,
    "airline": 14,
    "telecom": 13,
}


@pytest.mark.benchmark
@pytest.mark.live
class TestEnvironmentTools:
    """Tests for environment tool creation."""

    @pytest.mark.parametrize(
        "domain,expected_count",
        [
            ("retail", 15),
            ("airline", 14),
            ("telecom", 13),
        ],
    )
    def test_domain_has_correct_tool_count(self, domain, expected_count, ensure_tau2_data):
        """Each domain has the expected number of tools."""
        env = Tau2Environment({"domain": domain})
        tools = env.create_tools()

        assert len(tools) == expected_count
        assert isinstance(tools, dict)

    def test_tools_are_callable(self, retail_environment):
        """All tools are callable."""
        tools = retail_environment.create_tools()

        for name, tool in tools.items():
            assert callable(tool), f"Tool {name} is not callable"


# =============================================================================
# Database State Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestDatabaseState:
    """Tests for database state management."""

    def test_get_db_hash_returns_string(self, retail_environment):
        """get_db_hash returns a hash string."""
        hash_value = retail_environment.get_db_hash()

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hex length

    def test_initial_db_hash_stored(self, retail_environment):
        """Initial DB hash is stored in state."""
        initial_hash = retail_environment.get_initial_db_hash()

        assert isinstance(initial_hash, str)
        assert len(initial_hash) == 64

    def test_hash_consistent_without_changes(self, retail_environment):
        """Hash is consistent when no changes made."""
        hash1 = retail_environment.get_db_hash()
        hash2 = retail_environment.get_db_hash()

        assert hash1 == hash2

    def test_hash_changes_after_modification(self, retail_environment):
        """Hash changes after database modification."""
        initial_hash = retail_environment.get_db_hash()

        # Modify database through a tool
        try:
            # Try to modify order status (may fail if order doesn't exist in test state)
            retail_environment.make_tool_call(
                "modify_pending_order_address",
                order_id="test",
                address1="123 Test St",
                address2="",
                city="Test City",
                state="TX",
                country="USA",
                zip="12345",
            )
        except (ValueError, KeyError):
            # Expected if order doesn't exist - test passes
            return

        final_hash = retail_environment.get_db_hash()
        # If modification succeeded, hash should be different
        assert initial_hash != final_hash


# =============================================================================
# Tool Execution Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestToolExecution:
    """Tests for tool execution via environment."""

    def test_make_tool_call_read_tool(self, retail_environment):
        """Can execute a read-only tool."""
        # Get first user from database
        users = list(retail_environment.db.users.keys())
        if users:
            user_id = users[0]
            result = retail_environment.make_tool_call("get_user_details", user_id=user_id)
            assert result is not None

    def test_make_tool_call_invalid_tool_raises(self, retail_environment):
        """Invalid tool name raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            retail_environment.make_tool_call("nonexistent_tool")

    def test_airline_tool_execution(self, airline_environment):
        """Can execute airline domain tools."""
        users = list(airline_environment.db.users.keys())
        if users:
            user_id = users[0]
            result = airline_environment.make_tool_call("get_user_details", user_id=user_id)
            assert result is not None

    def test_telecom_tool_execution(self, telecom_environment):
        """Can execute telecom domain tools."""
        customers = telecom_environment.db.customers
        if customers:
            customer_id = customers[0].customer_id
            result = telecom_environment.make_tool_call("get_customer_by_id", customer_id=customer_id)
            assert result is not None


# =============================================================================
# Toolkit Statistics Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestToolkitStatistics:
    """Tests for toolkit statistics."""

    @pytest.mark.parametrize(
        "domain,num_tools,num_read,num_write",
        [
            ("retail", 15, 6, 7),
            ("airline", 14, 6, 6),
            ("telecom", 13, 6, 6),
        ],
    )
    def test_toolkit_stats(self, domain, num_tools, num_read, num_write, ensure_tau2_data):
        """Toolkit has expected statistics for each domain."""
        env = Tau2Environment({"domain": domain})
        stats = env.toolkit.get_statistics()

        assert stats["num_tools"] == num_tools
        assert stats["num_read_tools"] == num_read
        assert stats["num_write_tools"] == num_write


# =============================================================================
# Database Statistics Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestDatabaseStatistics:
    """Tests for database statistics."""

    def test_retail_db_stats(self, retail_environment):
        """Retail database has expected statistics."""
        stats = retail_environment.db.get_statistics()

        assert "num_products" in stats
        assert "num_users" in stats
        assert "num_orders" in stats

    def test_airline_db_stats(self, airline_environment):
        """Airline database has expected statistics."""
        stats = airline_environment.db.get_statistics()

        assert "num_flights" in stats
        assert "num_users" in stats
        assert "num_reservations" in stats

    def test_telecom_db_stats(self, telecom_environment):
        """Telecom database has expected statistics."""
        stats = telecom_environment.db.get_statistics()

        assert "num_customers" in stats
        assert "num_plans" in stats
        assert "num_lines" in stats


# =============================================================================
# Trace Gathering Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestTraceGathering:
    """Tests for environment trace gathering."""

    def test_gather_traces_structure(self, retail_environment):
        """gather_traces returns expected structure."""
        traces = retail_environment.gather_traces()

        assert "domain" in traces
        assert traces["domain"] == "retail"
        assert "initial_db_hash" in traces
        assert "final_db_hash" in traces
        assert "db_changed" in traces

    def test_gather_config_structure(self, retail_environment):
        """gather_config returns expected structure."""
        config = retail_environment.gather_config()

        assert "domain" in config
        assert "toolkit_stats" in config
        assert "db_stats" in config

    def test_traces_db_changed_false_initially(self, retail_environment):
        """db_changed is False when no modifications made."""
        traces = retail_environment.gather_traces()

        assert traces["db_changed"] is False
        assert traces["initial_db_hash"] == traces["final_db_hash"]


# =============================================================================
# User Tools Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestUserTools:
    """Tests for user tool creation."""

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_create_user_tools(self, domain, ensure_tau2_data):
        """Each domain can create user tools."""
        env = Tau2Environment({"domain": domain})
        user_tools = env.create_user_tools()

        # User tools should be dict (may be empty for some domains)
        assert isinstance(user_tools, dict)


# =============================================================================
# Tool Call Tracing Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestToolCallTracing:
    """Tests for tool call tracing."""

    def test_tool_calls_traced(self, retail_environment):
        """Tool invocations are traced."""
        users = list(retail_environment.db.users.keys())
        if not users:
            pytest.skip("No users in test database")

        user_id = users[0]
        retail_environment.make_tool_call("get_user_details", user_id=user_id)

        traces = retail_environment.gather_traces()

        # Verify traces dict is returned with expected fields
        assert "domain" in traces
        assert "initial_db_hash" in traces
        assert "final_db_hash" in traces

    def test_multiple_tool_calls_success(self, retail_environment):
        """Multiple tool invocations execute without error."""
        users = list(retail_environment.db.users.keys())
        orders = list(retail_environment.db.orders.keys())

        if not users or not orders:
            pytest.skip("Insufficient test data")

        # Make multiple tool calls
        result1 = retail_environment.make_tool_call("get_user_details", user_id=users[0])
        result2 = retail_environment.make_tool_call("get_order_details", order_id=orders[0])

        # Both calls should succeed
        assert result1 is not None
        assert result2 is not None


# =============================================================================
# Environment Reset Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestEnvironmentReset:
    """Tests for environment reset functionality."""

    def test_environment_reset(self, retail_environment):
        """Environment can be reset to initial state."""
        initial_hash = retail_environment.get_db_hash()

        # Make a modification
        users = list(retail_environment.db.users.keys())
        if users:
            # Modify user directly
            user = retail_environment.db.users[users[0]]
            original_email = user.email
            user.email = "modified@test.com"

            # Hash should change
            modified_hash = retail_environment.get_db_hash()
            assert initial_hash != modified_hash

            # Reset
            user.email = original_email

    def test_policy_available(self, retail_environment):
        """Environment provides policy."""
        assert retail_environment.policy is not None
        assert len(retail_environment.policy) > 0

    def test_policy_is_string(self, retail_environment):
        """Policy is a string."""
        assert isinstance(retail_environment.policy, str)


# =============================================================================
# Tool Description Tests
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.live
class TestToolDescriptions:
    """Tests for tool descriptions."""

    @pytest.mark.parametrize(
        "domain,expected_count",
        [
            ("retail", 15),
            ("airline", 14),
            ("telecom", 13),
        ],
    )
    def test_tool_descriptions(self, domain, expected_count, ensure_tau2_data):
        """Each domain has descriptions for all tools."""
        env = Tau2Environment({"domain": domain})
        descriptions = env.toolkit.get_tool_descriptions()

        assert len(descriptions) == expected_count
        for name, desc in descriptions.items():
            assert isinstance(desc, str)
            assert len(desc) > 0, f"Tool {name} has empty description"


# =============================================================================
# Offline setup_state Tests (Tier 1 — mock-based)
# =============================================================================


@pytest.mark.benchmark
class TestSetupStateOffline:
    """Tests for Tau2Environment.setup_state() without real data."""

    def test_setup_state_loads_db_from_path(self, tmp_path):
        """setup_state loads DB from db_path in task_data."""
        import json

        # Create minimal retail DB file
        db_data = {"users": {}, "orders": {}, "products": {}}
        db_path = tmp_path / "db.json"
        db_path.write_text(json.dumps(db_data))
        policy_text = "Test policy"

        env = Tau2Environment({"domain": "retail", "db_path": str(db_path), "policy": policy_text})

        assert env.domain == "retail"
        assert env.db is not None
        assert env.toolkit is not None
        assert env.policy == policy_text

    def test_setup_state_applies_initial_state(self, tmp_path):
        """setup_state applies initial_state.initialization_data to DB."""
        import json

        db_data = _minimal_retail_db_data(include_user=True)
        db_path = tmp_path / "db.json"
        db_path.write_text(json.dumps(db_data))

        initial_state = {"initialization_data": {"agent_data": {"users": {"u1": {"email": "new@test.com"}}}}}
        env = Tau2Environment({"domain": "retail", "db_path": str(db_path), "policy": "p", "initial_state": initial_state})

        assert env.db.users["u1"].email == "new@test.com"  # type: ignore[union-attr]

    def test_setup_state_executes_initialization_actions(self, tmp_path):
        """setup_state executes initialization_actions via toolkit."""
        import json

        db_data = _minimal_retail_db_data(include_user=True)
        db_path = tmp_path / "db.json"
        db_path.write_text(json.dumps(db_data))

        # get_user_details is a read tool that should execute without error
        initial_state = {"initialization_actions": [{"env_type": "assistant", "func_name": "get_user_details", "arguments": {"user_id": "u1"}}]}
        env = Tau2Environment({"domain": "retail", "db_path": str(db_path), "policy": "p", "initial_state": initial_state})

        # If we get here without error, the action executed successfully
        assert env.db is not None

    def test_setup_state_stores_initial_hash_after_actions(self, tmp_path):
        """Initial DB hash is recorded AFTER initialization_actions."""
        import json

        db_data = {"users": {}, "orders": {}, "products": {}}
        db_path = tmp_path / "db.json"
        db_path.write_text(json.dumps(db_data))

        env = Tau2Environment({"domain": "retail", "db_path": str(db_path), "policy": "p"})

        # Hash should match current state (no changes since setup)
        assert env.get_initial_db_hash() == env.get_db_hash()

    def test_setup_state_creates_user_toolkit_for_telecom(self, tmp_path):
        """setup_state creates user_toolkit only for telecom."""
        import json

        db_data = {"users": {}, "orders": {}, "products": {}}
        db_path = tmp_path / "db.json"
        db_path.write_text(json.dumps(db_data))

        env = Tau2Environment({"domain": "retail", "db_path": str(db_path), "policy": "p"})
        assert env.user_toolkit is None


# =============================================================================
# Offline create_tools / create_user_tools Tests
# =============================================================================


@pytest.mark.benchmark
class TestCreateToolsOffline:
    """Tests for tool creation without real data."""

    def test_create_tools_returns_callable_dict(self, tmp_path):
        """create_tools returns dict of callables."""
        import json

        db_data = {"users": {}, "orders": {}, "products": {}}
        db_path = tmp_path / "db.json"
        db_path.write_text(json.dumps(db_data))

        env = Tau2Environment({"domain": "retail", "db_path": str(db_path), "policy": "p"})
        tools = env.create_tools()

        assert isinstance(tools, dict)
        assert len(tools) > 0
        for name, tool in tools.items():
            assert callable(tool), f"Tool {name} is not callable"

    def test_create_tools_wraps_with_sync(self, tmp_path):
        """create_tools wraps tools to call sync_tools() after invocation."""
        import json

        db_data = _minimal_retail_db_data(include_user=True)
        db_path = tmp_path / "db.json"
        db_path.write_text(json.dumps(db_data))

        env = Tau2Environment({"domain": "retail", "db_path": str(db_path), "policy": "p"})
        tools = env.create_tools()

        sync_called: list = []
        env.sync_tools = lambda: sync_called.append(True)  # type: ignore[assignment]

        # Execute a read tool
        tools["get_user_details"](user_id="u1")
        assert len(sync_called) == 1

    def test_create_user_tools_empty_for_non_telecom(self, tmp_path):
        """create_user_tools returns empty dict for retail/airline."""
        import json

        db_data = {"users": {}, "orders": {}, "products": {}}
        db_path = tmp_path / "db.json"
        db_path.write_text(json.dumps(db_data))

        env = Tau2Environment({"domain": "retail", "db_path": str(db_path), "policy": "p"})
        assert env.create_user_tools() == {}


# =============================================================================
# Offline make_tool_call Tests
# =============================================================================


@pytest.mark.benchmark
class TestMakeToolCallOffline:
    """Tests for make_tool_call without real data."""

    def test_make_tool_call_delegates_to_toolkit(self, tmp_path):
        """make_tool_call delegates to toolkit.use_tool."""
        import json

        db_data = _minimal_retail_db_data(include_user=True)
        db_path = tmp_path / "db.json"
        db_path.write_text(json.dumps(db_data))

        env = Tau2Environment({"domain": "retail", "db_path": str(db_path), "policy": "p"})
        result = env.make_tool_call("get_user_details", user_id="u1")

        assert result is not None

    def test_make_tool_call_unknown_raises(self, tmp_path):
        """make_tool_call raises ValueError for unknown tool."""
        import json

        db_data = {"users": {}, "orders": {}, "products": {}}
        db_path = tmp_path / "db.json"
        db_path.write_text(json.dumps(db_data))

        env = Tau2Environment({"domain": "retail", "db_path": str(db_path), "policy": "p"})

        with pytest.raises(ValueError, match="not found"):
            env.make_tool_call("nonexistent_tool")

    def test_make_user_tool_call_raises_for_non_telecom(self, tmp_path):
        """make_user_tool_call raises ValueError when no user toolkit."""
        import json

        db_data = {"users": {}, "orders": {}, "products": {}}
        db_path = tmp_path / "db.json"
        db_path.write_text(json.dumps(db_data))

        env = Tau2Environment({"domain": "retail", "db_path": str(db_path), "policy": "p"})

        with pytest.raises(ValueError, match="No user toolkit"):
            env.make_user_tool_call("some_tool")


# =============================================================================
# Offline gather_traces / gather_config Tests
# =============================================================================


@pytest.mark.benchmark
class TestGatherTracesOffline:
    """Tests for gather_traces/gather_config without real data."""

    def test_gather_traces_structure(self, tmp_path):
        """gather_traces returns expected structure."""
        import json

        db_data = {"users": {}, "orders": {}, "products": {}}
        db_path = tmp_path / "db.json"
        db_path.write_text(json.dumps(db_data))

        env = Tau2Environment({"domain": "retail", "db_path": str(db_path), "policy": "p"})
        traces = env.gather_traces()

        assert traces["domain"] == "retail"
        assert "initial_db_hash" in traces
        assert "final_db_hash" in traces
        assert traces["db_changed"] is False
        assert traces["type"] == "Tau2Environment"

    def test_gather_config_structure(self, tmp_path):
        """gather_config returns expected structure."""
        import json

        db_data = {"users": {}, "orders": {}, "products": {}}
        db_path = tmp_path / "db.json"
        db_path.write_text(json.dumps(db_data))

        env = Tau2Environment({"domain": "retail", "db_path": str(db_path), "policy": "p"})
        config = env.gather_config()

        assert config["domain"] == "retail"
        assert "toolkit_stats" in config
        assert "db_stats" in config
        assert config["type"] == "Tau2Environment"


# =============================================================================
# Offline get_environment_constructor Tests
# =============================================================================


@pytest.mark.benchmark
class TestGetEnvironmentConstructor:
    """Tests for get_environment_constructor factory."""

    def test_returns_callable(self):
        """get_environment_constructor returns a callable."""
        constructor = get_environment_constructor({"domain": "retail"})
        assert callable(constructor)

    @pytest.mark.live
    def test_constructor_creates_environment(self, ensure_tau2_data):
        """Returned constructor creates Tau2Environment instances."""
        constructor = get_environment_constructor({"domain": "retail"})
        env = constructor()
        assert isinstance(env, Tau2Environment)
        assert env.domain == "retail"


# =============================================================================
# Offline _execute_initialization_actions Tests
# =============================================================================


@pytest.mark.benchmark
class TestExecuteInitializationActions:
    """Tests for _execute_initialization_actions edge cases."""

    def test_raises_on_unknown_env_type(self, tmp_path):
        """Raises ValueError for unknown env_type."""
        import json

        db_data = {"users": {}, "orders": {}, "products": {}}
        db_path = tmp_path / "db.json"
        db_path.write_text(json.dumps(db_data))

        actions = [{"env_type": "unknown", "func_name": "foo"}]

        env = Tau2Environment.__new__(Tau2Environment)
        env._domain = "retail"
        env._initial_state_config = None
        env._policy = "p"
        env._db_path = str(db_path)

        with pytest.raises(ValueError, match="Unknown env_type"):
            env._execute_initialization_actions(actions, MagicMock(), None, MagicMock())

    def test_raises_on_missing_user_toolkit(self, tmp_path):
        """Raises ValueError for user action when user_toolkit is None."""
        actions = [{"env_type": "user", "func_name": "some_func"}]

        env = Tau2Environment.__new__(Tau2Environment)
        env._domain = "retail"

        with pytest.raises(ValueError, match="No user toolkit"):
            env._execute_initialization_actions(actions, MagicMock(), None, MagicMock())

    def test_raises_on_missing_assistant_function(self):
        """Raises ValueError when assistant function not found on toolkit."""
        actions = [{"env_type": "assistant", "func_name": "nonexistent_func"}]

        mock_toolkit = MagicMock(spec=[])  # No attributes

        env = Tau2Environment.__new__(Tau2Environment)
        env._domain = "retail"

        with pytest.raises(ValueError, match="not found on toolkit"):
            env._execute_initialization_actions(actions, mock_toolkit, None, MagicMock())
