"""Unit tests for Tau2 telecom domain tools."""

import pytest

from maseval.benchmark.tau2.domains.base import ToolType
from maseval.benchmark.tau2.domains.telecom.models import LineStatus

pytestmark = [pytest.mark.live]


# =============================================================================
# Toolkit Basic Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomToolkitBasic:
    """Basic tests for TelecomTools."""

    def test_toolkit_has_tools(self, telecom_toolkit):
        """Toolkit has tools available."""
        assert len(telecom_toolkit.tools) > 0

    def test_all_tools_callable(self, telecom_toolkit):
        """All tools are callable methods."""
        for name, tool in telecom_toolkit.tools.items():
            assert callable(tool), f"Tool {name} is not callable"

    def test_toolkit_statistics(self, telecom_toolkit):
        """Toolkit provides statistics."""
        stats = telecom_toolkit.get_statistics()

        assert stats["num_tools"] == 13
        assert stats["num_read_tools"] == 6
        assert stats["num_write_tools"] == 6

    def test_toolkit_descriptions(self, telecom_toolkit):
        """Toolkit provides tool descriptions."""
        descriptions = telecom_toolkit.get_tool_descriptions()

        assert len(descriptions) > 0
        for name, desc in descriptions.items():
            assert isinstance(desc, str)


# =============================================================================
# Read Tool Tests - Customer Lookup
# =============================================================================


@pytest.mark.benchmark
class TestTelecomCustomerLookup:
    """Tests for telecom customer lookup tools."""

    def test_get_customer_by_id(self, telecom_toolkit):
        """get_customer_by_id returns customer information."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        result = telecom_toolkit.use_tool("get_customer_by_id", customer_id=customer.customer_id)

        assert result is not None
        assert result.customer_id == customer.customer_id

    def test_get_customer_by_id_invalid(self, telecom_toolkit):
        """get_customer_by_id raises for invalid customer."""
        with pytest.raises(ValueError, match="not found"):
            telecom_toolkit.use_tool("get_customer_by_id", customer_id="INVALID_ID")

    def test_get_customer_by_phone(self, telecom_toolkit):
        """get_customer_by_phone returns customer information."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        result = telecom_toolkit.use_tool("get_customer_by_phone", phone_number=customer.phone_number)

        assert result is not None
        assert result.customer_id == customer.customer_id

    def test_get_customer_by_phone_invalid(self, telecom_toolkit):
        """get_customer_by_phone raises for invalid phone."""
        with pytest.raises(ValueError, match="not found"):
            telecom_toolkit.use_tool("get_customer_by_phone", phone_number="000-000-0000")

    def test_get_customer_by_name(self, telecom_toolkit):
        """get_customer_by_name returns matching customers."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        result = telecom_toolkit.use_tool(
            "get_customer_by_name",
            full_name=customer.full_name,
            dob=customer.date_of_birth,
        )

        assert result is not None
        assert isinstance(result, list)
        if len(result) > 0:
            assert result[0].customer_id == customer.customer_id


# =============================================================================
# Read Tool Tests - Billing
# =============================================================================


@pytest.mark.benchmark
class TestTelecomBillingRead:
    """Tests for telecom billing read tools."""

    def test_get_bills_for_customer(self, telecom_toolkit):
        """get_bills_for_customer returns bills."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        assert len(customer.bill_ids) > 0, (
            f"Customer {customer.customer_id} should have bills. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customer.bill_ids)} bills."
        )

        result = telecom_toolkit.use_tool("get_bills_for_customer", customer_id=customer.customer_id)

        assert result is not None
        assert isinstance(result, list)

    def test_get_data_usage(self, telecom_toolkit):
        """get_data_usage returns usage information."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        assert len(customer.line_ids) > 0, (
            f"Customer {customer.customer_id} should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customer.line_ids)} lines."
        )

        result = telecom_toolkit.use_tool(
            "get_data_usage",
            customer_id=customer.customer_id,
            line_id=customer.line_ids[0],
        )

        assert result is not None
        assert "data_used_gb" in result
        assert "data_limit_gb" in result


# =============================================================================
# Read Tool Tests - Details by ID
# =============================================================================


@pytest.mark.benchmark
class TestTelecomDetailsById:
    """Tests for get_details_by_id tool."""

    def test_get_details_by_customer_id(self, telecom_toolkit):
        """get_details_by_id returns customer for C-prefixed ID."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        result = telecom_toolkit.use_tool("get_details_by_id", id=customer.customer_id)

        assert result is not None
        assert result.customer_id == customer.customer_id

    def test_get_details_by_line_id(self, telecom_toolkit):
        """get_details_by_id returns line for L-prefixed ID."""
        lines = telecom_toolkit.db.lines
        assert len(lines) > 0, (
            "Real telecom database should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.lines)} lines."
        )

        line = lines[0]
        result = telecom_toolkit.use_tool("get_details_by_id", id=line.line_id)

        assert result is not None
        assert result.line_id == line.line_id

    def test_get_details_by_invalid_id(self, telecom_toolkit):
        """get_details_by_id raises for invalid ID format."""
        with pytest.raises(ValueError, match="Unknown ID format"):
            telecom_toolkit.use_tool("get_details_by_id", id="INVALID_123")


# =============================================================================
# Tool Type Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomToolTypes:
    """Tests for telecom tool type annotations."""

    def test_read_tools_have_correct_type(self, telecom_toolkit):
        """Read tools are marked as READ type."""
        read_tools = ["get_customer_by_id", "get_customer_by_phone", "get_bills_for_customer", "get_data_usage"]

        for tool_name in read_tools:
            if telecom_toolkit.has_tool(tool_name):
                tool_type = telecom_toolkit.tool_type(tool_name)
                assert tool_type == ToolType.READ, f"{tool_name} should be READ type"

    def test_write_tools_have_correct_type(self, telecom_toolkit):
        """Write tools are marked as WRITE type."""
        write_tools = ["suspend_line", "resume_line", "enable_roaming", "disable_roaming", "refuel_data"]

        for tool_name in write_tools:
            if telecom_toolkit.has_tool(tool_name):
                tool_type = telecom_toolkit.tool_type(tool_name)
                assert tool_type == ToolType.WRITE, f"{tool_name} should be WRITE type"


# =============================================================================
# Database Hash Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomDatabaseHash:
    """Tests for telecom database hashing."""

    def test_hash_consistent(self, telecom_toolkit):
        """Database hash is consistent."""
        hash1 = telecom_toolkit.get_db_hash()
        hash2 = telecom_toolkit.get_db_hash()

        assert hash1 == hash2

    def test_hash_is_sha256(self, telecom_toolkit):
        """Database hash is SHA-256 format."""
        hash_value = telecom_toolkit.get_db_hash()

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in hash_value)


# =============================================================================
# Write Tool Tests - Line Management
# =============================================================================


@pytest.mark.benchmark
class TestTelecomLineManagement:
    """Tests for line management tools."""

    def test_suspend_active_line(self, telecom_toolkit):
        """Successfully suspend an active line."""
        # Find an active line
        active_line = None
        customer_id = None
        for customer in telecom_toolkit.db.customers:
            for line_id in customer.line_ids:
                line = telecom_toolkit._get_line_by_id(line_id)
                if line.status == LineStatus.ACTIVE:
                    active_line = line
                    customer_id = customer.customer_id
                    break
            if active_line:
                break

        assert active_line is not None, (
            "Real telecom database should have active lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Checked {len(telecom_toolkit.db.customers)} customers."
        )

        result = telecom_toolkit.use_tool(
            "suspend_line",
            customer_id=customer_id,
            line_id=active_line.line_id,
            reason="Customer requested suspension",
        )

        assert result is not None
        assert result["line"].status == LineStatus.SUSPENDED

    def test_suspend_non_active_line_fails(self, telecom_toolkit):
        """Cannot suspend a non-active line."""
        # Find a suspended line
        suspended_line = None
        customer_id = None
        for customer in telecom_toolkit.db.customers:
            for line_id in customer.line_ids:
                line = telecom_toolkit._get_line_by_id(line_id)
                if line.status == LineStatus.SUSPENDED:
                    suspended_line = line
                    customer_id = customer.customer_id
                    break
            if suspended_line:
                break

        assert suspended_line is not None, (
            "Real telecom database should have suspended lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Checked {len(telecom_toolkit.db.customers)} customers."
        )

        with pytest.raises(ValueError, match="must be active"):
            telecom_toolkit.use_tool(
                "suspend_line",
                customer_id=customer_id,
                line_id=suspended_line.line_id,
                reason="Test",
            )

    def test_resume_suspended_line(self, telecom_toolkit):
        """Successfully resume a suspended line."""
        # Find a suspended line
        suspended_line = None
        customer_id = None
        for customer in telecom_toolkit.db.customers:
            for line_id in customer.line_ids:
                line = telecom_toolkit._get_line_by_id(line_id)
                if line.status == LineStatus.SUSPENDED:
                    suspended_line = line
                    customer_id = customer.customer_id
                    break
            if suspended_line:
                break

        assert suspended_line is not None, (
            "Real telecom database should have suspended lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Checked {len(telecom_toolkit.db.customers)} customers."
        )

        result = telecom_toolkit.use_tool(
            "resume_line",
            customer_id=customer_id,
            line_id=suspended_line.line_id,
        )

        assert result is not None
        assert result["line"].status == LineStatus.ACTIVE


# =============================================================================
# Write Tool Tests - Roaming
# =============================================================================


@pytest.mark.benchmark
class TestTelecomRoaming:
    """Tests for roaming tools."""

    def test_enable_roaming(self, telecom_toolkit):
        """Enable roaming on a line."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        assert len(customer.line_ids) > 0, (
            f"Customer {customer.customer_id} should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customer.line_ids)} lines."
        )

        line_id = customer.line_ids[0]
        line = telecom_toolkit._get_line_by_id(line_id)

        # First disable if enabled
        if line.roaming_enabled:
            telecom_toolkit.use_tool(
                "disable_roaming",
                customer_id=customer.customer_id,
                line_id=line_id,
            )

        result = telecom_toolkit.use_tool(
            "enable_roaming",
            customer_id=customer.customer_id,
            line_id=line_id,
        )

        assert "enabled" in result.lower() or "already" in result.lower()
        assert line.roaming_enabled is True

    def test_disable_roaming(self, telecom_toolkit):
        """Disable roaming on a line."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        assert len(customer.line_ids) > 0, (
            f"Customer {customer.customer_id} should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customer.line_ids)} lines."
        )

        line_id = customer.line_ids[0]
        line = telecom_toolkit._get_line_by_id(line_id)

        # First enable if disabled
        if not line.roaming_enabled:
            telecom_toolkit.use_tool(
                "enable_roaming",
                customer_id=customer.customer_id,
                line_id=line_id,
            )

        result = telecom_toolkit.use_tool(
            "disable_roaming",
            customer_id=customer.customer_id,
            line_id=line_id,
        )

        assert "disabled" in result.lower() or "already" in result.lower()
        assert line.roaming_enabled is False


# =============================================================================
# Write Tool Tests - Data Refueling
# =============================================================================


@pytest.mark.benchmark
class TestTelecomDataRefueling:
    """Tests for data refueling tools."""

    def test_refuel_data(self, telecom_toolkit):
        """Successfully refuel data for a line."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        assert len(customer.line_ids) > 0, (
            f"Customer {customer.customer_id} should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customer.line_ids)} lines."
        )

        line_id = customer.line_ids[0]
        line = telecom_toolkit._get_line_by_id(line_id)
        original_data = line.data_refueling_gb

        result = telecom_toolkit.use_tool(
            "refuel_data",
            customer_id=customer.customer_id,
            line_id=line_id,
            gb_amount=5.0,
        )

        assert result is not None
        assert result["new_data_refueling_gb"] == original_data + 5.0
        assert result["charge"] > 0

    def test_refuel_data_invalid_amount(self, telecom_toolkit):
        """Refuel with invalid amount fails."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        assert len(customer.line_ids) > 0, (
            f"Customer {customer.customer_id} should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customer.line_ids)} lines."
        )

        with pytest.raises(ValueError, match="positive"):
            telecom_toolkit.use_tool(
                "refuel_data",
                customer_id=customer.customer_id,
                line_id=customer.line_ids[0],
                gb_amount=-5.0,
            )


# =============================================================================
# Generic Tool Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomGenericTools:
    """Tests for generic tools."""

    def test_transfer_to_human(self, telecom_toolkit):
        """Transfer to human agent returns success."""
        result = telecom_toolkit.use_tool(
            "transfer_to_human_agents",
            summary="Customer needs technical support",
        )
        assert "Transfer successful" in result


# =============================================================================
# Tool Metadata Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomToolMetadata:
    """Tests for telecom tool metadata."""

    def test_get_tool_metadata(self, telecom_toolkit):
        """get_tool_metadata returns expected structure."""
        meta = telecom_toolkit.get_tool_metadata("get_customer_by_id")

        assert "description" in meta
        assert "inputs" in meta
        assert "tool_type" in meta
        assert meta["tool_type"] == ToolType.READ

    def test_get_tool_metadata_invalid_tool(self, telecom_toolkit):
        """get_tool_metadata raises for invalid tool."""
        with pytest.raises(ValueError, match="not found"):
            telecom_toolkit.get_tool_metadata("nonexistent_tool")

    def test_tool_inputs_extracted(self, telecom_toolkit):
        """Tool inputs are extracted from signature."""
        meta = telecom_toolkit.get_tool_metadata("get_customer_by_id")

        assert "customer_id" in meta["inputs"]
        assert meta["inputs"]["customer_id"]["type"] == "string"


# =============================================================================
# Read Tool Tests - Bill Details
# =============================================================================


@pytest.mark.benchmark
class TestTelecomBillDetails:
    """Tests for bill detail retrieval."""

    def test_get_bill_by_id(self, telecom_toolkit):
        """Get bill by ID returns bill details."""
        bills = telecom_toolkit.db.bills
        assert len(bills) > 0, (
            "Real telecom database should have bills. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.bills)} bills."
        )

        bill = bills[0]
        result = telecom_toolkit.use_tool("get_details_by_id", id=bill.bill_id)

        assert result is not None
        assert result.bill_id == bill.bill_id

    def test_get_plan_by_id(self, telecom_toolkit):
        """Get plan by ID returns plan details."""
        plans = telecom_toolkit.db.plans
        assert len(plans) > 0, (
            "Real telecom database should have plans. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.plans)} plans."
        )

        plan = plans[0]
        result = telecom_toolkit.use_tool("get_details_by_id", id=plan.plan_id)

        assert result is not None
        assert result.plan_id == plan.plan_id


# =============================================================================
# Write Tool Tests - Plan Changes
# =============================================================================


@pytest.mark.benchmark
class TestTelecomPlanChanges:
    """Tests for plan change operations."""

    def test_change_plan_validation(self, telecom_toolkit):
        """Plan change validates plan exists."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        assert len(customer.line_ids) > 0, (
            f"Customer {customer.customer_id} should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customer.line_ids)} lines."
        )

        with pytest.raises(ValueError):
            telecom_toolkit.use_tool(
                "change_plan",
                customer_id=customer.customer_id,
                line_id=customer.line_ids[0],
                new_plan_id="INVALID_PLAN",
            )


# =============================================================================
# Write Tool Tests - Payment Processing
# =============================================================================


@pytest.mark.benchmark
class TestTelecomPayment:
    """Tests for payment processing."""

    def test_process_payment_validation(self, telecom_toolkit):
        """Payment processing validates bill exists."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]

        with pytest.raises(ValueError):
            telecom_toolkit.use_tool(
                "process_payment",
                customer_id=customer.customer_id,
                bill_id="INVALID_BILL",
                amount=100.0,
            )


# =============================================================================
# Write Tool Tests - Line Activation
# =============================================================================


@pytest.mark.benchmark
class TestTelecomLineActivation:
    """Tests for line activation tools."""

    def test_resume_non_suspended_line_fails(self, telecom_toolkit):
        """Cannot resume a line that is not suspended."""
        active_line = None
        customer_id = None
        for customer in telecom_toolkit.db.customers:
            for line_id in customer.line_ids:
                line = telecom_toolkit._get_line_by_id(line_id)
                if line.status == LineStatus.ACTIVE:
                    active_line = line
                    customer_id = customer.customer_id
                    break
            if active_line:
                break

        assert active_line is not None, (
            "Real telecom database should have active lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Checked {len(telecom_toolkit.db.customers)} customers."
        )

        with pytest.raises(ValueError, match="suspended"):
            telecom_toolkit.use_tool(
                "resume_line",
                customer_id=customer_id,
                line_id=active_line.line_id,  # type: ignore[union-attr]
            )


# =============================================================================
# Read Tool Tests - Service Status
# =============================================================================


@pytest.mark.benchmark
class TestTelecomServiceStatus:
    """Tests for service status tools."""

    def test_get_service_outages(self, telecom_toolkit):
        """Get service outages returns outage list."""
        if telecom_toolkit.has_tool("get_service_outages"):
            result = telecom_toolkit.use_tool("get_service_outages")
            assert isinstance(result, list)

    def test_check_network_coverage(self, telecom_toolkit):
        """Check network coverage for area."""
        if telecom_toolkit.has_tool("check_network_coverage"):
            result = telecom_toolkit.use_tool("check_network_coverage", zip_code="94102")
            assert result is not None


# =============================================================================
# Write Tool Tests - Payment Request
# =============================================================================


@pytest.mark.benchmark
class TestTelecomPaymentRequest:
    """Tests for send_payment_request tool."""

    def test_send_payment_request_success(self, telecom_toolkit):
        """Successfully send payment request for a bill."""
        from maseval.benchmark.tau2.domains.telecom.models import BillStatus

        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        # Find a customer with an issued bill (not yet awaiting payment)
        customer = None
        target_bill = None
        for c in customers:
            for bill_id in c.bill_ids:
                bill = telecom_toolkit._get_bill_by_id(bill_id)
                if bill.status == BillStatus.ISSUED:
                    customer = c
                    target_bill = bill
                    break
            if target_bill:
                break

        assert target_bill is not None, (
            "Real telecom database should have issued bills. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.bills)} bills."
        )

        assert customer is not None  # type narrowing
        result = telecom_toolkit.use_tool(
            "send_payment_request",
            customer_id=customer.customer_id,
            bill_id=target_bill.bill_id,
        )

        assert "Payment request sent" in result
        assert target_bill.status == BillStatus.AWAITING_PAYMENT

    def test_send_payment_request_bill_not_found(self, telecom_toolkit):
        """send_payment_request fails for bill not belonging to customer."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]

        with pytest.raises(ValueError, match="not found"):
            telecom_toolkit.use_tool(
                "send_payment_request",
                customer_id=customer.customer_id,
                bill_id="B_NONEXISTENT",
            )

    def test_send_payment_request_already_awaiting(self, telecom_toolkit):
        """send_payment_request fails if another bill already awaiting payment."""
        from maseval.benchmark.tau2.domains.telecom.models import BillStatus

        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        # Find a customer with multiple bills
        customer = None
        for c in customers:
            if len(c.bill_ids) >= 2:
                customer = c
                break

        assert customer is not None, (
            "Real telecom database should have customers with multiple bills. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )
        # Set first bill to awaiting payment
        first_bill = telecom_toolkit._get_bill_by_id(customer.bill_ids[0])
        original_status = first_bill.status
        first_bill.status = BillStatus.AWAITING_PAYMENT

        try:
            with pytest.raises(ValueError, match="already awaiting payment"):
                telecom_toolkit.use_tool(
                    "send_payment_request",
                    customer_id=customer.customer_id,
                    bill_id=customer.bill_ids[1],
                )
        finally:
            # Restore original status
            first_bill.status = original_status


# =============================================================================
# Internal Helper Method Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomHelperMethods:
    """Tests for internal helper methods."""

    def test_get_line_by_phone(self, telecom_toolkit):
        """_get_line_by_phone returns line for valid phone."""
        lines = telecom_toolkit.db.lines
        assert len(lines) > 0, (
            "Real telecom database should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.lines)} lines."
        )

        line = lines[0]
        result = telecom_toolkit._get_line_by_phone(line.phone_number)

        assert result.line_id == line.line_id

    def test_get_line_by_phone_invalid(self, telecom_toolkit):
        """_get_line_by_phone raises for invalid phone."""
        with pytest.raises(ValueError, match="not found"):
            telecom_toolkit._get_line_by_phone("000-000-0000")

    def test_get_device_by_id(self, telecom_toolkit):
        """_get_device_by_id returns device for valid ID."""
        devices = telecom_toolkit.db.devices
        assert len(devices) > 0, (
            "Real telecom database should have devices. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.devices)} devices."
        )

        device = devices[0]
        result = telecom_toolkit._get_device_by_id(device.device_id)

        assert result.device_id == device.device_id

    def test_get_device_by_id_invalid(self, telecom_toolkit):
        """_get_device_by_id raises for invalid ID."""
        with pytest.raises(ValueError, match="not found"):
            telecom_toolkit._get_device_by_id("D_INVALID")

    def test_get_details_by_device_id(self, telecom_toolkit):
        """get_details_by_id returns device for D-prefixed ID."""
        devices = telecom_toolkit.db.devices
        assert len(devices) > 0, (
            "Real telecom database should have devices. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.devices)} devices."
        )

        device = devices[0]
        result = telecom_toolkit.use_tool("get_details_by_id", id=device.device_id)

        assert result.device_id == device.device_id

    def test_get_target_line_invalid_line(self, telecom_toolkit):
        """_get_target_line raises when line doesn't belong to customer."""
        customers = telecom_toolkit.db.customers
        assert len(customers) >= 2, (
            "Real telecom database should have at least 2 customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customers)} customers."
        )

        customer1 = customers[0]
        customer2 = customers[1]

        assert len(customer2.line_ids) > 0, (
            f"Customer {customer2.customer_id} should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customer2.line_ids)} lines."
        )

        # Try to get customer2's line using customer1's ID
        with pytest.raises(ValueError, match="not found"):
            telecom_toolkit._get_target_line(customer1.customer_id, customer2.line_ids[0])

    def test_get_available_plan_ids(self, telecom_toolkit):
        """get_available_plan_ids returns list of plan IDs."""
        result = telecom_toolkit.get_available_plan_ids()

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(pid, str) for pid in result)

    def test_get_bills_awaiting_payment_empty(self, telecom_toolkit):
        """_get_bills_awaiting_payment returns empty list when none awaiting."""
        from maseval.benchmark.tau2.domains.telecom.models import BillStatus

        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        # Find customer without awaiting payment bills
        customer = None
        for c in customers:
            has_awaiting = False
            for bill_id in c.bill_ids:
                bill = telecom_toolkit._get_bill_by_id(bill_id)
                if bill.status == BillStatus.AWAITING_PAYMENT:
                    has_awaiting = True
                    break
            if not has_awaiting:
                customer = c
                break

        assert customer is not None, (
            "Real telecom database should have customers without awaiting payment bills. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        result = telecom_toolkit._get_bills_awaiting_payment(customer)
        assert result == []


# =============================================================================
# Assertion Method Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomAssertions:
    """Tests for assertion methods used in evaluation."""

    def test_assert_data_refueling_amount_correct(self, telecom_toolkit):
        """assert_data_refueling_amount returns True for matching amount."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        assert len(customer.line_ids) > 0, (
            f"Customer {customer.customer_id} should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customer.line_ids)} lines."
        )

        line = telecom_toolkit._get_line_by_id(customer.line_ids[0])
        actual_amount = line.data_refueling_gb

        result = telecom_toolkit.assert_data_refueling_amount(
            customer.customer_id,
            customer.line_ids[0],
            actual_amount,
        )

        assert result is True

    def test_assert_data_refueling_amount_incorrect(self, telecom_toolkit):
        """assert_data_refueling_amount returns False for wrong amount."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        assert len(customer.line_ids) > 0, (
            f"Customer {customer.customer_id} should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customer.line_ids)} lines."
        )

        result = telecom_toolkit.assert_data_refueling_amount(
            customer.customer_id,
            customer.line_ids[0],
            99999.0,  # Wrong amount
        )

        assert result is False

    def test_assert_line_status_correct(self, telecom_toolkit):
        """assert_line_status returns True for matching status."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        assert len(customer.line_ids) > 0, (
            f"Customer {customer.customer_id} should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customer.line_ids)} lines."
        )

        line = telecom_toolkit._get_line_by_id(customer.line_ids[0])

        result = telecom_toolkit.assert_line_status(
            customer.customer_id,
            customer.line_ids[0],
            line.status.value,
        )

        assert result is True

    def test_assert_line_status_with_enum(self, telecom_toolkit):
        """assert_line_status works with enum value."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        assert len(customer.line_ids) > 0, (
            f"Customer {customer.customer_id} should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customer.line_ids)} lines."
        )

        line = telecom_toolkit._get_line_by_id(customer.line_ids[0])

        result = telecom_toolkit.assert_line_status(
            customer.customer_id,
            customer.line_ids[0],
            line.status,  # Pass enum directly
        )

        assert result is True

    def test_assert_line_status_incorrect(self, telecom_toolkit):
        """assert_line_status returns False for wrong status."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        assert len(customer.line_ids) > 0, (
            f"Customer {customer.customer_id} should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customer.line_ids)} lines."
        )

        result = telecom_toolkit.assert_line_status(
            customer.customer_id,
            customer.line_ids[0],
            "NONEXISTENT_STATUS",
        )

        assert result is False

    def test_assert_no_overdue_bill_paid(self, telecom_toolkit):
        """assert_no_overdue_bill returns True when bill is paid."""
        from maseval.benchmark.tau2.domains.telecom.models import BillStatus

        bills = telecom_toolkit.db.bills
        assert len(bills) > 0, (
            "Real telecom database should have bills. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.bills)} bills."
        )

        # Find a paid bill
        paid_bill = None
        for bill in bills:
            if bill.status == BillStatus.PAID:
                paid_bill = bill
                break

        assert paid_bill is not None, (
            "Real telecom database should have paid bills. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(bills)} bills."
        )
        result = telecom_toolkit.assert_no_overdue_bill(paid_bill.bill_id)
        assert result is True

    def test_assert_no_overdue_bill_nonexistent(self, telecom_toolkit):
        """assert_no_overdue_bill returns True when bill doesn't exist."""
        result = telecom_toolkit.assert_no_overdue_bill("B_NONEXISTENT_12345")
        assert result is True

    def test_assert_no_overdue_bill_still_overdue(self, telecom_toolkit):
        """assert_no_overdue_bill returns False when bill is still overdue."""
        from maseval.benchmark.tau2.domains.telecom.models import BillStatus

        bills = telecom_toolkit.db.bills
        assert len(bills) > 0, (
            "Real telecom database should have bills. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.bills)} bills."
        )

        # Find or create an overdue bill
        overdue_bill = None
        for bill in bills:
            if bill.status == BillStatus.OVERDUE:
                overdue_bill = bill
                break

        assert overdue_bill is not None, (
            "Real telecom database should have overdue bills. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(bills)} bills."
        )
        result = telecom_toolkit.assert_no_overdue_bill(overdue_bill.bill_id)
        assert result is False


# =============================================================================
# Internal Data Modification Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomDataModification:
    """Tests for internal data modification methods."""

    def test_set_data_usage(self, telecom_toolkit):
        """set_data_usage modifies line data usage."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        assert len(customer.line_ids) > 0, (
            f"Customer {customer.customer_id} should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customer.line_ids)} lines."
        )

        line_id = customer.line_ids[0]
        new_usage = 15.5

        result = telecom_toolkit.set_data_usage(
            customer.customer_id,
            line_id,
            new_usage,
        )

        assert "set to" in result
        line = telecom_toolkit._get_line_by_id(line_id)
        assert line.data_used_gb == new_usage

    def test_set_bill_to_paid(self, telecom_toolkit):
        """_set_bill_to_paid changes bill status."""
        from maseval.benchmark.tau2.domains.telecom.models import BillStatus

        bills = telecom_toolkit.db.bills
        assert len(bills) > 0, (
            "Real telecom database should have bills. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.bills)} bills."
        )

        # Find an issued bill (or any non-paid bill)
        target_bill = None
        for bill in bills:
            if bill.status == BillStatus.ISSUED:
                target_bill = bill
                break

        assert target_bill is not None, (
            "Real telecom database should have issued bills. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.bills)} bills."
        )
        original_status = target_bill.status
        try:
            result = telecom_toolkit._set_bill_to_paid(target_bill.bill_id)

            assert "set to paid" in result
            assert target_bill.status == BillStatus.PAID
        finally:
            # Restore original status
            target_bill.status = original_status


# =============================================================================
# ID Generator Tests
# =============================================================================


@pytest.mark.benchmark
class TestIDGenerator:
    """Tests for IDGenerator class."""

    def test_id_generator_increments(self):
        """IDGenerator increments counter for same type."""
        from maseval.benchmark.tau2.domains.telecom.tools import IDGenerator

        gen = IDGenerator()

        id1 = gen.get_id("test")
        id2 = gen.get_id("test")

        assert id1 == "test_1"
        assert id2 == "test_2"

    def test_id_generator_separate_types(self):
        """IDGenerator maintains separate counters per type."""
        from maseval.benchmark.tau2.domains.telecom.tools import IDGenerator

        gen = IDGenerator()

        id_a = gen.get_id("typeA")
        id_b = gen.get_id("typeB")
        id_a2 = gen.get_id("typeA")

        assert id_a == "typeA_1"
        assert id_b == "typeB_1"
        assert id_a2 == "typeA_2"

    def test_id_generator_custom_name(self):
        """IDGenerator uses custom name when provided."""
        from maseval.benchmark.tau2.domains.telecom.tools import IDGenerator

        gen = IDGenerator()

        id1 = gen.get_id("bill", "B")

        assert id1 == "B_1"


# =============================================================================
# Roaming Edge Cases
# =============================================================================


@pytest.mark.benchmark
class TestTelecomRoamingEdgeCases:
    """Edge case tests for roaming tools."""

    def test_enable_roaming_already_enabled(self, telecom_toolkit):
        """enable_roaming returns appropriate message when already enabled."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        assert len(customer.line_ids) > 0, (
            f"Customer {customer.customer_id} should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customer.line_ids)} lines."
        )

        line_id = customer.line_ids[0]
        line = telecom_toolkit._get_line_by_id(line_id)

        # Ensure roaming is enabled
        line.roaming_enabled = True

        result = telecom_toolkit.use_tool(
            "enable_roaming",
            customer_id=customer.customer_id,
            line_id=line_id,
        )

        assert "already" in result.lower()

    def test_disable_roaming_already_disabled(self, telecom_toolkit):
        """disable_roaming returns appropriate message when already disabled."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        customer = customers[0]
        assert len(customer.line_ids) > 0, (
            f"Customer {customer.customer_id} should have lines. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(customer.line_ids)} lines."
        )

        line_id = customer.line_ids[0]
        line = telecom_toolkit._get_line_by_id(line_id)

        # Ensure roaming is disabled
        line.roaming_enabled = False

        result = telecom_toolkit.use_tool(
            "disable_roaming",
            customer_id=customer.customer_id,
            line_id=line_id,
        )

        assert "already" in result.lower()


# =============================================================================
# Customer Lookup Edge Cases
# =============================================================================


@pytest.mark.benchmark
class TestTelecomCustomerLookupEdgeCases:
    """Edge case tests for customer lookup."""

    def test_get_customer_by_phone_via_line(self, telecom_toolkit):
        """get_customer_by_phone finds customer via line phone number."""
        customers = telecom_toolkit.db.customers
        assert len(customers) > 0, (
            "Real telecom database should have customers. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(telecom_toolkit.db.customers)} customers."
        )

        # Find a customer with a line that has a different phone than primary
        for customer in customers:
            if customer.line_ids:
                line = telecom_toolkit._get_line_by_id(customer.line_ids[0])
                if line.phone_number != customer.phone_number:
                    result = telecom_toolkit.use_tool(
                        "get_customer_by_phone",
                        phone_number=line.phone_number,
                    )
                    assert result.customer_id == customer.customer_id
                    return

        # If all lines have same phone as primary, just test with primary
        customer = customers[0]
        result = telecom_toolkit.use_tool(
            "get_customer_by_phone",
            phone_number=customer.phone_number,
        )
        assert result.customer_id == customer.customer_id

    def test_get_customer_by_name_no_match(self, telecom_toolkit):
        """get_customer_by_name returns empty list for no match."""
        result = telecom_toolkit.use_tool(
            "get_customer_by_name",
            full_name="Nonexistent Person",
            dob="1900-01-01",
        )

        assert result == []
