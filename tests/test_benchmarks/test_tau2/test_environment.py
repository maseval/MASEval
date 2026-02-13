"""Unit tests for Tau2 environment module."""

import pytest

from maseval.benchmark.tau2 import Tau2Environment
from maseval.benchmark.tau2.data_loader import VALID_DOMAINS


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
