"""Unit tests for Tau2 retail domain tools."""

import pytest

from maseval.benchmark.tau2.domains.base import ToolType

pytestmark = [pytest.mark.live]


# =============================================================================
# Toolkit Basic Tests
# =============================================================================


@pytest.mark.benchmark
class TestRetailToolkitBasic:
    """Basic tests for RetailTools."""

    def test_toolkit_has_tools(self, retail_toolkit):
        """Toolkit has tools available."""
        assert len(retail_toolkit.tools) > 0

    def test_all_tools_callable(self, retail_toolkit):
        """All tools are callable methods."""
        for name, tool in retail_toolkit.tools.items():
            assert callable(tool), f"Tool {name} is not callable"

    def test_toolkit_statistics(self, retail_toolkit):
        """Toolkit provides statistics."""
        stats = retail_toolkit.get_statistics()

        assert stats["num_tools"] == 15
        assert stats["num_read_tools"] == 6
        assert stats["num_write_tools"] == 7

    def test_toolkit_descriptions(self, retail_toolkit):
        """Toolkit provides tool descriptions."""
        descriptions = retail_toolkit.get_tool_descriptions()

        assert len(descriptions) > 0
        for name, desc in descriptions.items():
            assert isinstance(desc, str)


# =============================================================================
# Read Tool Tests
# =============================================================================


@pytest.mark.benchmark
class TestRetailReadTools:
    """Tests for retail read-only tools."""

    def test_get_user_details(self, retail_toolkit):
        """get_user_details returns user information."""
        users = list(retail_toolkit.db.users.keys())
        assert len(users) > 0, (
            "Real retail database should have users. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.users)} users."
        )

        user_id = users[0]
        result = retail_toolkit.use_tool("get_user_details", user_id=user_id)

        assert result is not None
        assert hasattr(result, "user_id") or "user_id" in str(result)

    def test_get_user_details_invalid_user(self, retail_toolkit):
        """get_user_details raises for invalid user."""
        with pytest.raises(ValueError, match="not found"):
            retail_toolkit.use_tool("get_user_details", user_id="nonexistent_user")

    def test_get_order_details(self, retail_toolkit):
        """get_order_details returns order information."""
        orders = list(retail_toolkit.db.orders.keys())
        assert len(orders) > 0, (
            "Real retail database should have orders. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} orders."
        )

        order_id = orders[0]
        result = retail_toolkit.use_tool("get_order_details", order_id=order_id)

        assert result is not None

    def test_get_order_details_invalid_order(self, retail_toolkit):
        """get_order_details raises for invalid order."""
        with pytest.raises(ValueError, match="not found"):
            retail_toolkit.use_tool("get_order_details", order_id="nonexistent_order")

    def test_list_all_product_types(self, retail_toolkit):
        """list_all_product_types returns product categories."""
        result = retail_toolkit.use_tool("list_all_product_types")

        assert result is not None
        assert isinstance(result, (list, dict, str))

    def test_get_product_details(self, retail_toolkit):
        """get_product_details returns product information."""
        products = list(retail_toolkit.db.products.keys())
        assert len(products) > 0, (
            "Real retail database should have products. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.products)} products."
        )

        product_id = products[0]
        result = retail_toolkit.use_tool("get_product_details", product_id=product_id)

        assert result is not None


# =============================================================================
# Tool Type Tests
# =============================================================================


@pytest.mark.benchmark
class TestRetailToolTypes:
    """Tests for retail tool type annotations."""

    def test_read_tools_have_correct_type(self, retail_toolkit):
        """Read tools are marked as READ type."""
        read_tools = ["get_user_details", "get_order_details", "list_all_product_types", "get_product_details"]

        for tool_name in read_tools:
            if retail_toolkit.has_tool(tool_name):
                tool_type = retail_toolkit.tool_type(tool_name)
                assert tool_type == ToolType.READ, f"{tool_name} should be READ type"

    def test_write_tools_have_correct_type(self, retail_toolkit):
        """Write tools are marked as WRITE type."""
        write_tools = ["cancel_pending_order", "modify_pending_order_items", "modify_pending_order_address"]

        for tool_name in write_tools:
            if retail_toolkit.has_tool(tool_name):
                tool_type = retail_toolkit.tool_type(tool_name)
                assert tool_type == ToolType.WRITE, f"{tool_name} should be WRITE type"


# =============================================================================
# Database Hash Tests
# =============================================================================


@pytest.mark.benchmark
class TestRetailDatabaseHash:
    """Tests for retail database hashing."""

    def test_hash_consistent(self, retail_toolkit):
        """Database hash is consistent."""
        hash1 = retail_toolkit.get_db_hash()
        hash2 = retail_toolkit.get_db_hash()

        assert hash1 == hash2

    def test_hash_is_sha256(self, retail_toolkit):
        """Database hash is SHA-256 format."""
        hash_value = retail_toolkit.get_db_hash()

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in hash_value)


# =============================================================================
# Tool Metadata Tests
# =============================================================================


@pytest.mark.benchmark
class TestRetailToolMetadata:
    """Tests for retail tool metadata."""

    def test_get_tool_metadata(self, retail_toolkit):
        """get_tool_metadata returns expected structure."""
        meta = retail_toolkit.get_tool_metadata("get_user_details")

        assert "description" in meta
        assert "inputs" in meta
        assert "tool_type" in meta
        assert meta["tool_type"] == ToolType.READ

    def test_get_tool_metadata_invalid_tool(self, retail_toolkit):
        """get_tool_metadata raises for invalid tool."""
        with pytest.raises(ValueError, match="not found"):
            retail_toolkit.get_tool_metadata("nonexistent_tool")

    def test_tool_inputs_extracted(self, retail_toolkit):
        """Tool inputs are extracted from signature."""
        meta = retail_toolkit.get_tool_metadata("get_user_details")

        assert "user_id" in meta["inputs"]
        assert meta["inputs"]["user_id"]["type"] == "string"


# =============================================================================
# Write Tool Tests - Cancel Orders
# =============================================================================


@pytest.mark.benchmark
class TestRetailCancelOrder:
    """Tests for cancel_pending_order tool."""

    def test_cancel_pending_order(self, retail_toolkit):
        """Successfully cancel a pending order."""
        # Find a pending order
        pending_orders = {oid: o for oid, o in retail_toolkit.db.orders.items() if o.status == "pending"}
        assert len(pending_orders) > 0, (
            "Real retail database should have pending orders. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders, 0 pending."
        )

        pending_order_id = next(iter(pending_orders))
        result = retail_toolkit.use_tool(
            "cancel_pending_order",
            order_id=pending_order_id,
            reason="no longer needed",
        )

        assert result is not None
        assert result.status == "cancelled"
        assert result.cancel_reason == "no longer needed"

    def test_cancel_non_pending_order_fails(self, retail_toolkit):
        """Cannot cancel a non-pending order."""
        # Find a non-pending order
        non_pending_orders = {oid: o for oid, o in retail_toolkit.db.orders.items() if o.status != "pending"}
        assert len(non_pending_orders) > 0, (
            "Real retail database should have non-pending orders. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders, 0 non-pending."
        )

        non_pending_order_id = next(iter(non_pending_orders))
        with pytest.raises(ValueError, match="Non-pending"):
            retail_toolkit.use_tool(
                "cancel_pending_order",
                order_id=non_pending_order_id,
                reason="no longer needed",
            )

    def test_cancel_with_invalid_reason(self, retail_toolkit):
        """Cancel with invalid reason fails."""
        pending_orders = {oid: o for oid, o in retail_toolkit.db.orders.items() if o.status == "pending"}
        assert len(pending_orders) > 0, (
            "Real retail database should have pending orders. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders, 0 pending."
        )

        pending_order_id = next(iter(pending_orders))
        with pytest.raises(ValueError, match="Invalid reason"):
            retail_toolkit.use_tool(
                "cancel_pending_order",
                order_id=pending_order_id,
                reason="invalid reason",
            )


# =============================================================================
# Write Tool Tests - Modify Orders
# =============================================================================


@pytest.mark.benchmark
class TestRetailModifyOrder:
    """Tests for order modification tools."""

    def test_modify_pending_order_address(self, retail_toolkit):
        """Successfully modify address of a pending order."""
        pending_orders = {oid: o for oid, o in retail_toolkit.db.orders.items() if o.status == "pending"}
        assert len(pending_orders) > 0, (
            "Real retail database should have pending orders. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders, 0 pending."
        )

        pending_order_id = next(iter(pending_orders))
        result = retail_toolkit.use_tool(
            "modify_pending_order_address",
            order_id=pending_order_id,
            address1="123 New Street",
            address2="Apt 456",
            city="New York",
            state="NY",
            country="USA",
            zip="10001",
        )

        assert result is not None
        assert result.address.address1 == "123 New Street"
        assert result.address.city == "New York"
        assert result.address.zip == "10001"

    def test_modify_non_pending_order_address_fails(self, retail_toolkit):
        """Cannot modify address of a non-pending order."""
        non_pending_order_id = None
        for order_id, order in retail_toolkit.db.orders.items():
            if order.status not in ["pending", "pending (item modified)"]:
                non_pending_order_id = order_id
                break

        assert non_pending_order_id is not None, (
            "Real retail database should have non-pending orders. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders."
        )

        with pytest.raises(ValueError, match="Non-pending"):
            retail_toolkit.use_tool(
                "modify_pending_order_address",
                order_id=non_pending_order_id,
                address1="123 New Street",
                address2="",
                city="New York",
                state="NY",
                country="USA",
                zip="10001",
            )


# =============================================================================
# Generic Tool Tests
# =============================================================================


@pytest.mark.benchmark
class TestRetailGenericTools:
    """Tests for generic tools."""

    def test_calculate_simple(self, retail_toolkit):
        """Calculate simple expression."""
        result = retail_toolkit.use_tool("calculate", expression="2 + 2")
        assert float(result) == 4.0

    def test_calculate_complex(self, retail_toolkit):
        """Calculate complex expression."""
        result = retail_toolkit.use_tool("calculate", expression="(10 + 5) * 2 / 3")
        assert result == "10.0"

    def test_calculate_invalid_expression(self, retail_toolkit):
        """Calculate with invalid characters fails."""
        with pytest.raises(ValueError, match="Invalid characters"):
            retail_toolkit.use_tool("calculate", expression="2 + 2; import os")

    def test_transfer_to_human(self, retail_toolkit):
        """Transfer to human agent returns success."""
        result = retail_toolkit.use_tool(
            "transfer_to_human_agents",
            summary="User needs help with complex issue",
        )
        assert "Transfer successful" in result


# =============================================================================
# User Lookup Tool Tests
# =============================================================================


@pytest.mark.benchmark
class TestRetailUserLookup:
    """Tests for user lookup tools."""

    def test_find_user_by_email(self, retail_toolkit):
        """Find user by email."""
        # Get first user's email
        users = list(retail_toolkit.db.users.values())
        assert len(users) > 0, (
            f"Real retail database should have users. This indicates upstream data issue or data_integrity test gap. Found {len(users)} users."
        )

        user = users[0]
        result = retail_toolkit.use_tool("find_user_id_by_email", email=user.email)

        assert result == user.user_id

    def test_find_user_by_email_not_found(self, retail_toolkit):
        """Find user by email - not found."""
        with pytest.raises(ValueError, match="not found"):
            retail_toolkit.use_tool("find_user_id_by_email", email="nonexistent@example.com")

    def test_find_user_by_name_zip(self, retail_toolkit):
        """Find user by name and zip."""
        users = list(retail_toolkit.db.users.values())
        assert len(users) > 0, (
            f"Real retail database should have users. This indicates upstream data issue or data_integrity test gap. Found {len(users)} users."
        )

        user = users[0]
        result = retail_toolkit.use_tool(
            "find_user_id_by_name_zip",
            first_name=user.name.first_name,
            last_name=user.name.last_name,
            zip=user.address.zip,
        )

        assert result == user.user_id

    def test_find_user_by_name_zip_not_found(self, retail_toolkit):
        """Find user by name and zip - not found."""
        with pytest.raises(ValueError, match="not found"):
            retail_toolkit.use_tool(
                "find_user_id_by_name_zip",
                first_name="NonExistent",
                last_name="Person",
                zip="00000",
            )


# =============================================================================
# Write Tool Tests - User Address Modification
# =============================================================================


@pytest.mark.benchmark
class TestRetailModifyUserAddress:
    """Tests for user address modification."""

    def test_modify_user_address(self, retail_toolkit):
        """Successfully modify user's default address."""
        user_ids = list(retail_toolkit.db.users.keys())
        assert len(user_ids) > 0, (
            "Real retail database should have users. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(user_ids)} users."
        )

        result = retail_toolkit.use_tool(
            "modify_user_address",
            user_id=user_ids[0],
            address1="456 New Ave",
            address2="Suite 100",
            city="San Francisco",
            state="CA",
            country="USA",
            zip="94102",
        )

        assert result.address.address1 == "456 New Ave"
        assert result.address.city == "San Francisco"
        assert result.address.zip == "94102"

    def test_modify_user_address_invalid_user(self, retail_toolkit):
        """Cannot modify address of nonexistent user."""
        with pytest.raises(ValueError, match="not found"):
            retail_toolkit.use_tool(
                "modify_user_address",
                user_id="nonexistent_user",
                address1="123 Street",
                address2="",
                city="City",
                state="ST",
                country="USA",
                zip="12345",
            )


# =============================================================================
# Write Tool Tests - Item Modification
# =============================================================================


@pytest.mark.benchmark
class TestRetailModifyItems:
    """Tests for pending order item modification."""

    def test_modify_pending_order_items_validation(self, retail_toolkit):
        """Item modification validates item existence."""
        pending_order = None
        for order_id, order in retail_toolkit.db.orders.items():
            if order.status == "pending" and order.items:
                pending_order = order
                break

        assert pending_order is not None, (
            "Real retail database should have pending orders with items. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders."
        )

        user = retail_toolkit.db.users.get(pending_order.user_id)  # type: ignore[union-attr]
        assert user is not None and user.payment_methods, (
            "Order user should have payment methods. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"User: {user}, Payment methods: {len(user.payment_methods) if user else 0}"
        )

        payment_id = list(user.payment_methods.keys())[0]

        # Try with non-existent item
        with pytest.raises(ValueError):
            retail_toolkit.use_tool(
                "modify_pending_order_items",
                order_id=pending_order.order_id,  # type: ignore[union-attr]
                item_ids=["NONEXISTENT"],
                new_item_ids=["ALSO_NONEXISTENT"],
                payment_method_id=payment_id,
            )

    def test_modify_non_pending_order_items_fails(self, retail_toolkit):
        """Cannot modify items of non-pending order."""
        non_pending_order = None
        for order_id, order in retail_toolkit.db.orders.items():
            if order.status not in ["pending", "pending (item modified)"] and order.items:
                non_pending_order = order
                break

        assert non_pending_order is not None, (
            "Real retail database should have suitable non-pending orders. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders."
        )

        user = retail_toolkit.db.users.get(non_pending_order.user_id)  # type: ignore[union-attr]
        assert user is not None and user.payment_methods, (
            "Order user should have payment methods. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"User: {user}, Payment methods: {len(user.payment_methods) if user else 0}"
        )

        payment_id = list(user.payment_methods.keys())[0]
        item_id = non_pending_order.items[0].item_id  # type: ignore[union-attr]

        with pytest.raises(ValueError, match="Non-pending"):
            retail_toolkit.use_tool(
                "modify_pending_order_items",
                order_id=non_pending_order.order_id,  # type: ignore[union-attr]
                item_ids=[item_id],
                new_item_ids=[item_id],
                payment_method_id=payment_id,
            )


# =============================================================================
# Write Tool Tests - Payment Modification
# =============================================================================


@pytest.mark.benchmark
class TestRetailModifyPayment:
    """Tests for payment method modification."""

    def test_modify_payment_invalid_method(self, retail_toolkit):
        """Payment modification validates payment method exists."""
        pending_order = None
        for order_id, order in retail_toolkit.db.orders.items():
            if "pending" in order.status and len(order.payment_history) == 1:
                pending_order = order
                break

        assert pending_order is not None, (
            "Real retail database should have suitable pending orders. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders."
        )

        with pytest.raises(ValueError, match="not found"):
            retail_toolkit.use_tool(
                "modify_pending_order_payment",
                order_id=pending_order.order_id,  # type: ignore[union-attr]
                payment_method_id="nonexistent_payment",
            )

    def test_modify_payment_non_pending_fails(self, retail_toolkit):
        """Cannot modify payment of non-pending order."""
        non_pending_order = None
        for order_id, order in retail_toolkit.db.orders.items():
            if order.status not in ["pending", "pending (item modified)"]:
                non_pending_order = order
                break

        assert non_pending_order is not None, (
            "Real retail database should have non-pending orders. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders."
        )

        user = retail_toolkit.db.users.get(non_pending_order.user_id)  # type: ignore[union-attr]
        assert user is not None and user.payment_methods, (
            "Order user should have payment methods. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"User: {user}, Payment methods: {len(user.payment_methods) if user else 0}"
        )

        payment_id = list(user.payment_methods.keys())[0]

        with pytest.raises(ValueError, match="Non-pending"):
            retail_toolkit.use_tool(
                "modify_pending_order_payment",
                order_id=non_pending_order.order_id,  # type: ignore[union-attr]
                payment_method_id=payment_id,
            )


# =============================================================================
# Write Tool Tests - Exchange Operations
# =============================================================================


@pytest.mark.benchmark
class TestRetailExchangeItems:
    """Tests for delivered order item exchange."""

    def test_exchange_non_delivered_fails(self, retail_toolkit):
        """Cannot exchange items of non-delivered order."""
        non_delivered_order = None
        for order_id, order in retail_toolkit.db.orders.items():
            if order.status != "delivered" and order.items:
                non_delivered_order = order
                break

        assert non_delivered_order is not None, (
            "Real retail database should have non-delivered orders with items. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders."
        )

        user = retail_toolkit.db.users.get(non_delivered_order.user_id)  # type: ignore[union-attr]
        assert user is not None and user.payment_methods, (
            "Order user should have payment methods. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"User: {user}, Payment methods: {len(user.payment_methods) if user else 0}"
        )

        payment_id = list(user.payment_methods.keys())[0]
        item_id = non_delivered_order.items[0].item_id  # type: ignore[union-attr]

        with pytest.raises(ValueError, match="Non-delivered"):
            retail_toolkit.use_tool(
                "exchange_delivered_order_items",
                order_id=non_delivered_order.order_id,  # type: ignore[union-attr]
                item_ids=[item_id],
                new_item_ids=[item_id],
                payment_method_id=payment_id,
            )

    def test_exchange_item_count_mismatch(self, retail_toolkit):
        """Exchange fails when item counts don't match."""
        delivered_order = None
        for order_id, order in retail_toolkit.db.orders.items():
            if order.status == "delivered" and order.items:
                delivered_order = order
                break

        assert delivered_order is not None, (
            "Real retail database should have delivered orders with items. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders."
        )

        user = retail_toolkit.db.users.get(delivered_order.user_id)  # type: ignore[union-attr]
        assert user is not None and user.payment_methods, (
            "Order user should have payment methods. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"User: {user}, Payment methods: {len(user.payment_methods) if user else 0}"
        )

        payment_id = list(user.payment_methods.keys())[0]
        item_id = delivered_order.items[0].item_id  # type: ignore[union-attr]

        with pytest.raises(ValueError, match="match"):
            retail_toolkit.use_tool(
                "exchange_delivered_order_items",
                order_id=delivered_order.order_id,  # type: ignore[union-attr]
                item_ids=[item_id],
                new_item_ids=[item_id, item_id],  # Wrong count
                payment_method_id=payment_id,
            )


# =============================================================================
# Write Tool Tests - Return Operations
# =============================================================================


@pytest.mark.benchmark
class TestRetailReturnItems:
    """Tests for delivered order item returns."""

    def test_return_non_delivered_fails(self, retail_toolkit):
        """Cannot return items of non-delivered order."""
        non_delivered_order = None
        for order_id, order in retail_toolkit.db.orders.items():
            if order.status != "delivered" and order.items:
                non_delivered_order = order
                break

        assert non_delivered_order is not None, (
            "Real retail database should have non-delivered orders with items. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders."
        )

        user = retail_toolkit.db.users.get(non_delivered_order.user_id)  # type: ignore[union-attr]
        assert user is not None and user.payment_methods, (
            "Order user should have payment methods. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"User: {user}, Payment methods: {len(user.payment_methods) if user else 0}"
        )

        payment_id = list(user.payment_methods.keys())[0]
        item_id = non_delivered_order.items[0].item_id  # type: ignore[union-attr]

        with pytest.raises(ValueError, match="Non-delivered"):
            retail_toolkit.use_tool(
                "return_delivered_order_items",
                order_id=non_delivered_order.order_id,  # type: ignore[union-attr]
                item_ids=[item_id],
                payment_method_id=payment_id,
            )

    def test_return_delivered_order_success(self, retail_toolkit):
        """Successfully return items from delivered order."""
        delivered_order = None
        for order_id, order in retail_toolkit.db.orders.items():
            if order.status == "delivered" and order.items:
                delivered_order = order
                break

        assert delivered_order is not None, (
            "Real retail database should have delivered orders with items. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders."
        )

        user = retail_toolkit.db.users.get(delivered_order.user_id)  # type: ignore[union-attr]
        assert user is not None and user.payment_methods, (
            "Order user should have payment methods. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"User: {user}, Payment methods: {len(user.payment_methods) if user else 0}"
        )

        # Use original payment method
        if delivered_order.payment_history:  # type: ignore[union-attr]
            payment_id = delivered_order.payment_history[0].payment_method_id  # type: ignore[union-attr]
        else:
            payment_id = list(user.payment_methods.keys())[0]

        item_id = delivered_order.items[0].item_id  # type: ignore[union-attr]

        result = retail_toolkit.use_tool(
            "return_delivered_order_items",
            order_id=delivered_order.order_id,  # type: ignore[union-attr]
            item_ids=[item_id],
            payment_method_id=payment_id,
        )

        assert result.status == "return requested"
        assert item_id in result.return_items


# =============================================================================
# Write Tool Tests - Modify Items Success Path
# =============================================================================


@pytest.mark.benchmark
class TestRetailModifyItemsSuccess:
    """Tests for successful item modification."""

    def test_modify_pending_order_items_success(self, retail_toolkit):
        """Successfully modify items in a pending order."""
        # Find a pending order with items that can be modified
        pending_order = None
        for order_id, order in retail_toolkit.db.orders.items():
            if order.status == "pending" and order.items:
                pending_order = order
                break

        assert pending_order is not None, (
            "Real retail database should have pending orders with items. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders."
        )

        assert pending_order is not None  # type narrowing after skip
        user = retail_toolkit.db.users.get(pending_order.user_id)
        assert user is not None and user.payment_methods, (
            "Order user should have payment methods. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"User: {user}, Payment methods: {len(user.payment_methods) if user else 0}"
        )

        assert user is not None  # type narrowing after skip
        payment_id = list(user.payment_methods.keys())[0]
        item = pending_order.items[0]

        # Find a different variant of the same product
        product = retail_toolkit.db.products.get(item.product_id)
        assert product is not None and len(product.variants) >= 2, (
            f"Product {item.product_id} should have multiple variants for testing. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found product with {len(product.variants) if product else 0} variants."
        )

        # Find a different item_id for the same product
        new_item_id = None
        for variant_id, variant in product.variants.items():
            if variant_id != item.item_id and variant.available:
                new_item_id = variant_id
                break

        assert new_item_id is not None, (
            f"Product {item.product_id} should have alternative variants available. "
            "This indicates upstream data issue or data_integrity test gap."
        )

        result = retail_toolkit.use_tool(
            "modify_pending_order_items",
            order_id=pending_order.order_id,
            item_ids=[item.item_id],
            new_item_ids=[new_item_id],
            payment_method_id=payment_id,
        )

        assert result.status == "pending (item modified)"

    def test_modify_pending_order_items_same_item_fails(self, retail_toolkit):
        """Modification fails when new item is same as old item."""
        pending_order = None
        for order_id, order in retail_toolkit.db.orders.items():
            if order.status == "pending" and order.items:
                pending_order = order
                break

        assert pending_order is not None, (
            "Real retail database should have pending orders with items. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders."
        )

        assert pending_order is not None  # type narrowing after skip
        user = retail_toolkit.db.users.get(pending_order.user_id)
        assert user is not None and user.payment_methods, (
            "Order user should have payment methods. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"User: {user}, Payment methods: {len(user.payment_methods) if user else 0}"
        )

        assert user is not None  # type narrowing after skip
        payment_id = list(user.payment_methods.keys())[0]
        item = pending_order.items[0]

        with pytest.raises(ValueError, match="different"):
            retail_toolkit.use_tool(
                "modify_pending_order_items",
                order_id=pending_order.order_id,
                item_ids=[item.item_id],
                new_item_ids=[item.item_id],  # Same as old
                payment_method_id=payment_id,
            )


# =============================================================================
# Write Tool Tests - Modify Payment Success Path
# =============================================================================


@pytest.mark.benchmark
class TestRetailModifyPaymentSuccess:
    """Tests for successful payment modification."""

    @pytest.mark.xfail(reason="v0.2.0 upstream data has no users with multiple payment methods")
    def test_modify_pending_order_payment_success(self, retail_toolkit):
        """Successfully modify payment method of a pending order."""
        # Find a pending order with a single payment
        pending_order = None
        for order_id, order in retail_toolkit.db.orders.items():
            if order.status == "pending" and len(order.payment_history) == 1 and order.payment_history[0].transaction_type == "payment":
                pending_order = order
                break

        assert pending_order is not None, (
            "Real retail database should have suitable pending orders. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders."
        )

        assert pending_order is not None  # type narrowing
        user = retail_toolkit.db.users.get(pending_order.user_id)
        assert user is not None and len(user.payment_methods) >= 2, (
            "User should have at least 2 payment methods for testing. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(user.payment_methods) if user else 0} payment methods."
        )
        # Find a different payment method
        current_payment_id = pending_order.payment_history[0].payment_method_id
        new_payment_id = None
        for pm_id in user.payment_methods.keys():
            if pm_id != current_payment_id:
                new_payment_id = pm_id
                break

        assert new_payment_id is not None, (
            f"User should have alternative payment methods available. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(user.payment_methods)} payment methods."
        )

        result = retail_toolkit.use_tool(
            "modify_pending_order_payment",
            order_id=pending_order.order_id,
            payment_method_id=new_payment_id,
        )

        assert result is not None
        # Payment history should now have 3 entries (original + new payment + refund)
        assert len(result.payment_history) == 3

    def test_modify_payment_same_method_fails(self, retail_toolkit):
        """Modification fails when using same payment method."""
        pending_order = None
        for order_id, order in retail_toolkit.db.orders.items():
            if order.status == "pending" and len(order.payment_history) == 1 and order.payment_history[0].transaction_type == "payment":
                pending_order = order
                break

        assert pending_order is not None, (
            "Real retail database should have suitable pending orders. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders."
        )

        assert pending_order is not None  # type narrowing after skip
        current_payment_id = pending_order.payment_history[0].payment_method_id

        with pytest.raises(ValueError, match="different"):
            retail_toolkit.use_tool(
                "modify_pending_order_payment",
                order_id=pending_order.order_id,
                payment_method_id=current_payment_id,  # Same payment method
            )


# =============================================================================
# Write Tool Tests - Exchange Success Path
# =============================================================================


@pytest.mark.benchmark
class TestRetailExchangeSuccess:
    """Tests for successful item exchange."""

    def test_exchange_delivered_order_items_success(self, retail_toolkit):
        """Successfully exchange items in a delivered order."""
        # Find a delivered order with items
        delivered_order = None
        for order_id, order in retail_toolkit.db.orders.items():
            if order.status == "delivered" and order.items:
                delivered_order = order
                break

        assert delivered_order is not None, (
            "Real retail database should have delivered orders with items. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found {len(retail_toolkit.db.orders)} total orders."
        )

        assert delivered_order is not None  # type narrowing after skip
        user = retail_toolkit.db.users.get(delivered_order.user_id)
        assert user is not None and user.payment_methods, (
            "Order user should have payment methods. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"User: {user}, Payment methods: {len(user.payment_methods) if user else 0}"
        )

        assert user is not None  # type narrowing after skip
        payment_id = list(user.payment_methods.keys())[0]
        item = delivered_order.items[0]

        # Find a different variant of the same product
        product = retail_toolkit.db.products.get(item.product_id)
        assert product is not None and len(product.variants) >= 2, (
            f"Product should have multiple variants for exchange testing. "
            "This indicates upstream data issue or data_integrity test gap. "
            f"Found product with {len(product.variants) if product else 0} variants."
        )

        # Find a different item_id for the same product
        new_item_id = None
        for variant_id, variant in product.variants.items():
            if variant_id != item.item_id and variant.available:
                new_item_id = variant_id
                break

        assert new_item_id is not None, (
            f"Product {item.product_id} should have alternative variants available. "
            "This indicates upstream data issue or data_integrity test gap."
        )

        result = retail_toolkit.use_tool(
            "exchange_delivered_order_items",
            order_id=delivered_order.order_id,
            item_ids=[item.item_id],
            new_item_ids=[new_item_id],
            payment_method_id=payment_id,
        )

        assert result.status == "exchange requested"
        assert item.item_id in result.exchange_items
        assert new_item_id in result.exchange_new_items


# =============================================================================
# Helper Method Tests
# =============================================================================


@pytest.mark.benchmark
class TestRetailHelperMethods:
    """Tests for internal helper methods."""

    def test_get_product_details_invalid(self, retail_toolkit):
        """get_product_details raises for invalid product."""
        with pytest.raises(ValueError, match="not found"):
            retail_toolkit.use_tool("get_product_details", product_id="INVALID_PRODUCT")

    def test_is_pending_order_check(self, retail_toolkit):
        """_is_pending_order correctly identifies pending orders."""
        pending_order = None
        non_pending_order = None

        for order_id, order in retail_toolkit.db.orders.items():
            if order.status == "pending" and not pending_order:
                pending_order = order
            elif order.status not in ["pending", "pending (item modified)"] and not non_pending_order:
                non_pending_order = order

            if pending_order and non_pending_order:
                break

        if pending_order:
            assert retail_toolkit._is_pending_order(pending_order) is True

        if non_pending_order:
            assert retail_toolkit._is_pending_order(non_pending_order) is False


# =============================================================================
# Calculator Edge Cases
# =============================================================================


@pytest.mark.benchmark
class TestRetailCalculatorEdgeCases:
    """Edge case tests for calculator tool."""

    def test_calculate_division(self, retail_toolkit):
        """Calculate division expression."""
        result = retail_toolkit.use_tool("calculate", expression="100 / 4")
        assert float(result) == 25.0

    def test_calculate_negative(self, retail_toolkit):
        """Calculate with negative numbers."""
        result = retail_toolkit.use_tool("calculate", expression="10 - 15")
        assert float(result) == -5.0

    def test_calculate_decimal(self, retail_toolkit):
        """Calculate with decimal numbers."""
        result = retail_toolkit.use_tool("calculate", expression="3.5 * 2")
        assert float(result) == 7.0
