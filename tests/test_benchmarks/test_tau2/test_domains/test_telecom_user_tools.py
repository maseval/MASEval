"""Unit tests for Tau2 telecom user tools."""

import pytest
from maseval.benchmark.tau2.domains.telecom.user_models import SimStatus, NetworkStatus, NetworkModePreference

pytestmark = [pytest.mark.live]


# =============================================================================
# Toolkit Basic Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomUserToolkitBasic:
    """Basic tests for TelecomUserTools."""

    def test_toolkit_has_tools(self, telecom_user_toolkit):
        """Toolkit has tools available."""
        assert len(telecom_user_toolkit.tools) > 0

    def test_all_tools_callable(self, telecom_user_toolkit):
        """All tools are callable methods."""
        for name, tool in telecom_user_toolkit.tools.items():
            assert callable(tool), f"Tool {name} is not callable"

    def test_toolkit_statistics(self, telecom_user_toolkit):
        """Toolkit provides statistics."""
        stats = telecom_user_toolkit.get_statistics()
        assert stats["num_tools"] == 34
        assert stats["num_read_tools"] == 19
        assert stats["num_write_tools"] == 15

    def test_toolkit_descriptions(self, telecom_user_toolkit):
        """Toolkit provides tool descriptions."""
        descriptions = telecom_user_toolkit.get_tool_descriptions()
        assert len(descriptions) > 0
        for name, desc in descriptions.items():
            assert isinstance(desc, str)

    def test_user_db_initialized(self, telecom_user_toolkit):
        """User DB is initialized when toolkit is created."""
        assert telecom_user_toolkit.db.user_db is not None
        assert telecom_user_toolkit.db.user_db.device is not None
        assert telecom_user_toolkit.db.user_db.surroundings is not None


# =============================================================================
# Read Tool Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomUserReadTools:
    """Tests for telecom user read-only tools."""

    def test_check_status_bar(self, telecom_user_toolkit):
        """check_status_bar returns correct structure."""
        status = telecom_user_toolkit.use_tool("check_status_bar")
        assert "signal_strength" in status
        assert "network_type" in status
        assert "wifi_connected" in status
        assert "airplane_mode" in status
        assert "battery_level" in status

    def test_check_network_status(self, telecom_user_toolkit):
        """check_network_status returns correct structure."""
        status = telecom_user_toolkit.use_tool("check_network_status")
        assert "status" in status
        assert "technology" in status
        assert "roaming" in status
        assert "mobile_data" in status

    def test_check_installed_apps(self, telecom_user_toolkit):
        """check_installed_apps returns list."""
        apps = telecom_user_toolkit.use_tool("check_installed_apps")
        assert isinstance(apps, list)

    def test_check_apn_settings(self, telecom_user_toolkit):
        """check_apn_settings returns dict."""
        apn = telecom_user_toolkit.use_tool("check_apn_settings")
        assert isinstance(apn, dict)
        assert "name" in apn
        assert "apn" in apn


# =============================================================================
# Write Tool Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomUserWriteTools:
    """Tests for telecom user state-modifying tools."""

    def test_toggle_airplane_mode(self, telecom_user_toolkit):
        """toggle_airplane_mode changes state."""
        # Enable
        telecom_user_toolkit.use_tool("toggle_airplane_mode", enable=True)
        assert telecom_user_toolkit.db.user_db.device.airplane_mode is True
        assert telecom_user_toolkit.db.user_db.device.network_connection_status == NetworkStatus.NO_SERVICE

        # Disable
        telecom_user_toolkit.use_tool("toggle_airplane_mode", enable=False)
        assert telecom_user_toolkit.db.user_db.device.airplane_mode is False
        # Should reconnect if SIM is active
        if telecom_user_toolkit.db.user_db.device.sim_status == SimStatus.ACTIVE:
            assert telecom_user_toolkit.db.user_db.device.network_connection_status == NetworkStatus.CONNECTED

    def test_toggle_data(self, telecom_user_toolkit):
        """toggle_data changes state."""
        telecom_user_toolkit.use_tool("toggle_data", enable=False)
        assert telecom_user_toolkit.db.user_db.device.mobile_data_enabled is False

        telecom_user_toolkit.use_tool("toggle_data", enable=True)
        assert telecom_user_toolkit.db.user_db.device.mobile_data_enabled is True

    def test_set_network_mode_preference(self, telecom_user_toolkit):
        """set_network_mode_preference changes state."""
        pref = NetworkModePreference.FOUR_G_ONLY.value
        result = telecom_user_toolkit.use_tool("set_network_mode_preference", preference=pref)

        assert telecom_user_toolkit.db.user_db.device.network_mode_preference == NetworkModePreference.FOUR_G_ONLY
        assert "set to" in result

    def test_set_network_mode_invalid(self, telecom_user_toolkit):
        """set_network_mode_preference handles invalid input."""
        result = telecom_user_toolkit.use_tool("set_network_mode_preference", preference="invalid_mode")
        assert "Invalid preference" in result

    def test_reboot_device(self, telecom_user_toolkit):
        """reboot_device resets ephemeral state."""
        # Set reset_at_reboot = True
        telecom_user_toolkit.db.user_db.device.apn_settings.reset_at_reboot = True
        telecom_user_toolkit.db.user_db.device.apn_settings.apn = "custom_apn"

        telecom_user_toolkit.use_tool("reboot_device")

        # Should have reset to default
        assert telecom_user_toolkit.db.user_db.device.apn_settings.apn == "internet"


# =============================================================================
# Wi-Fi Tests
# =============================================================================


@pytest.mark.benchmark
class TestWiFiOperations:
    """Tests for Wi-Fi operations."""

    def test_check_wifi_status(self, telecom_user_toolkit):
        """Returns Wi-Fi status structure."""
        result = telecom_user_toolkit.use_tool("check_wifi_status")

        assert "enabled" in result
        assert "connected" in result
        assert "available_networks" in result

    def test_toggle_wifi_on(self, telecom_user_toolkit):
        """Enables Wi-Fi."""
        result = telecom_user_toolkit.use_tool("toggle_wifi", enable=True)

        assert "enabled" in result.lower()
        assert telecom_user_toolkit.db.user_db.device.wifi_enabled is True

    def test_toggle_wifi_off(self, telecom_user_toolkit):
        """Disables Wi-Fi and disconnects."""
        result = telecom_user_toolkit.use_tool("toggle_wifi", enable=False)

        assert "disabled" in result.lower()
        assert telecom_user_toolkit.db.user_db.device.wifi_enabled is False
        assert telecom_user_toolkit.db.user_db.device.wifi_connected is False

    def test_check_wifi_calling_status(self, telecom_user_toolkit):
        """Returns Wi-Fi calling status."""
        result = telecom_user_toolkit.use_tool("check_wifi_calling_status")
        assert isinstance(result, bool)

    def test_toggle_wifi_calling(self, telecom_user_toolkit):
        """Toggles Wi-Fi calling."""
        telecom_user_toolkit.use_tool("toggle_wifi_calling", enable=True)
        assert telecom_user_toolkit.db.user_db.device.wifi_calling_enabled is True

        telecom_user_toolkit.use_tool("toggle_wifi_calling", enable=False)
        assert telecom_user_toolkit.db.user_db.device.wifi_calling_enabled is False


# =============================================================================
# SIM Card Tests
# =============================================================================


@pytest.mark.benchmark
class TestSimCardOperations:
    """Tests for SIM card operations."""

    def test_check_sim_status(self, telecom_user_toolkit):
        """Returns SIM status."""
        result = telecom_user_toolkit.use_tool("check_sim_status")
        valid_statuses = [s.value for s in SimStatus]
        assert result in valid_statuses

    def test_reseat_sim_card(self, telecom_user_toolkit):
        """Reseats SIM card."""
        result = telecom_user_toolkit.use_tool("reseat_sim_card")
        assert "re-seated" in result.lower()


# =============================================================================
# Roaming Tests
# =============================================================================


@pytest.mark.benchmark
class TestRoamingOperations:
    """Tests for roaming operations."""

    def test_toggle_roaming_on(self, telecom_user_toolkit):
        """Enables data roaming."""
        result = telecom_user_toolkit.use_tool("toggle_roaming", enable=True)

        assert "enabled" in result.lower()
        assert telecom_user_toolkit.db.user_db.device.roaming_enabled is True

    def test_toggle_roaming_off(self, telecom_user_toolkit):
        """Disables data roaming."""
        result = telecom_user_toolkit.use_tool("toggle_roaming", enable=False)

        assert "disabled" in result.lower()
        assert telecom_user_toolkit.db.user_db.device.roaming_enabled is False


# =============================================================================
# Data Saver Tests
# =============================================================================


@pytest.mark.benchmark
class TestDataSaverOperations:
    """Tests for data saver operations."""

    def test_check_data_restriction_status(self, telecom_user_toolkit):
        """Returns data saver status."""
        result = telecom_user_toolkit.use_tool("check_data_restriction_status")
        assert isinstance(result, bool)

    def test_toggle_data_saver_mode_on(self, telecom_user_toolkit):
        """Enables data saver mode."""
        result = telecom_user_toolkit.use_tool("toggle_data_saver_mode", enable=True)

        assert "enabled" in result.lower()
        assert telecom_user_toolkit.db.user_db.device.data_saver_mode is True

    def test_toggle_data_saver_mode_off(self, telecom_user_toolkit):
        """Disables data saver mode."""
        result = telecom_user_toolkit.use_tool("toggle_data_saver_mode", enable=False)

        assert "disabled" in result.lower()
        assert telecom_user_toolkit.db.user_db.device.data_saver_mode is False


# =============================================================================
# APN Settings Tests
# =============================================================================


@pytest.mark.benchmark
class TestAPNSettingsOperations:
    """Tests for APN settings operations."""

    def test_set_apn_settings(self, telecom_user_toolkit):
        """Updates APN settings."""
        result = telecom_user_toolkit.use_tool(
            "set_apn_settings",
            apn="custom.apn.example",
        )

        assert "updated" in result.lower()
        assert telecom_user_toolkit.db.user_db.device.apn_settings.apn == "custom.apn.example"

    def test_reset_apn_settings(self, telecom_user_toolkit):
        """Resets APN settings to default."""
        # First set to custom
        telecom_user_toolkit.use_tool("set_apn_settings", apn="custom")

        result = telecom_user_toolkit.use_tool("reset_apn_settings")

        assert "reset" in result.lower() or "default" in result.lower()
        assert telecom_user_toolkit.db.user_db.device.apn_settings.apn == "internet"


# =============================================================================
# VPN Tests
# =============================================================================


@pytest.mark.benchmark
class TestVPNOperations:
    """Tests for VPN operations."""

    def test_check_vpn_status(self, telecom_user_toolkit):
        """Returns VPN status."""
        result = telecom_user_toolkit.use_tool("check_vpn_status")

        assert "connected" in result

    def test_connect_vpn_with_internet(self, telecom_user_toolkit):
        """Connects to VPN when internet available."""
        # Ensure internet is available
        telecom_user_toolkit.use_tool("toggle_data", enable=True)
        telecom_user_toolkit.use_tool("toggle_airplane_mode", enable=False)

        result = telecom_user_toolkit.use_tool(
            "connect_vpn",
            server_address="vpn.example.com",
            protocol="IKEv2",
        )

        assert "connected" in result.lower()
        assert telecom_user_toolkit.db.user_db.device.vpn_status is True

    def test_connect_vpn_no_internet(self, telecom_user_toolkit):
        """Cannot connect VPN without internet."""
        telecom_user_toolkit.use_tool("toggle_data", enable=False)
        telecom_user_toolkit.use_tool("toggle_wifi", enable=False)

        result = telecom_user_toolkit.use_tool(
            "connect_vpn",
            server_address="vpn.example.com",
        )

        assert "cannot" in result.lower() or "no" in result.lower()

    def test_disconnect_vpn(self, telecom_user_toolkit):
        """Disconnects from VPN."""
        # First connect
        telecom_user_toolkit.use_tool("toggle_data", enable=True)
        telecom_user_toolkit.use_tool("connect_vpn", server_address="vpn.example.com")

        result = telecom_user_toolkit.use_tool("disconnect_vpn")

        assert "disconnected" in result.lower() or "not connected" in result.lower()
        assert telecom_user_toolkit.db.user_db.device.vpn_status is False

    def test_disconnect_vpn_when_not_connected(self, telecom_user_toolkit):
        """Disconnecting when not connected returns appropriate message."""
        telecom_user_toolkit.db.user_db.device.vpn_status = False

        result = telecom_user_toolkit.use_tool("disconnect_vpn")

        assert "not connected" in result.lower()


# =============================================================================
# Speed Test Tests
# =============================================================================


@pytest.mark.benchmark
class TestSpeedTestOperations:
    """Tests for network speed test."""

    def test_run_speed_test_with_connection(self, telecom_user_toolkit):
        """Speed test returns results with connection."""
        telecom_user_toolkit.use_tool("toggle_data", enable=True)
        telecom_user_toolkit.use_tool("toggle_airplane_mode", enable=False)

        result = telecom_user_toolkit.use_tool("run_speed_test")

        assert isinstance(result, str)

    def test_run_speed_test_airplane_mode(self, telecom_user_toolkit):
        """Speed test fails in airplane mode."""
        telecom_user_toolkit.use_tool("toggle_airplane_mode", enable=True)

        result = telecom_user_toolkit.use_tool("run_speed_test")

        assert "airplane" in result.lower() or "no" in result.lower()

        # Cleanup
        telecom_user_toolkit.use_tool("toggle_airplane_mode", enable=False)

    def test_run_speed_test_no_connection(self, telecom_user_toolkit):
        """Speed test fails without connection."""
        telecom_user_toolkit.use_tool("toggle_data", enable=False)
        telecom_user_toolkit.use_tool("toggle_wifi", enable=False)
        telecom_user_toolkit.use_tool("toggle_airplane_mode", enable=False)

        result = telecom_user_toolkit.use_tool("run_speed_test")

        assert "no" in result.lower()


# =============================================================================
# Application Tests
# =============================================================================


@pytest.mark.benchmark
class TestApplicationOperations:
    """Tests for application operations using default apps (messaging, browser)."""

    def test_check_installed_apps_has_defaults(self, telecom_user_toolkit):
        """Default apps are installed."""
        apps = telecom_user_toolkit.use_tool("check_installed_apps")
        assert "messaging" in apps
        assert "browser" in apps

    def test_check_app_status(self, telecom_user_toolkit):
        """Returns app status for messaging app."""
        result = telecom_user_toolkit.use_tool("check_app_status", app_name="messaging")

        assert "is_running" in result
        assert "data_usage_mb" in result
        assert "permissions" in result

    def test_check_app_status_invalid(self, telecom_user_toolkit):
        """Raises error for invalid app."""
        with pytest.raises(ValueError, match="not installed"):
            telecom_user_toolkit.use_tool("check_app_status", app_name="nonexistent_app_12345")

    def test_check_app_permissions(self, telecom_user_toolkit):
        """Returns app permissions for browser app."""
        result = telecom_user_toolkit.use_tool("check_app_permissions", app_name="browser")
        assert isinstance(result, dict)
        # Browser has network and storage permissions by default
        assert "network" in result
        assert "storage" in result

    def test_grant_app_permission(self, telecom_user_toolkit):
        """Grants permission to messaging app."""
        result = telecom_user_toolkit.use_tool(
            "grant_app_permission",
            app_name="messaging",
            permission="network",
            grant=True,
        )

        assert "granted" in result.lower()

    def test_revoke_app_permission(self, telecom_user_toolkit):
        """Revokes permission from messaging app."""
        # First grant, then revoke
        telecom_user_toolkit.use_tool(
            "grant_app_permission",
            app_name="messaging",
            permission="network",
            grant=True,
        )
        result = telecom_user_toolkit.use_tool(
            "grant_app_permission",
            app_name="messaging",
            permission="network",
            grant=False,
        )

        assert "revoked" in result.lower()

    def test_grant_invalid_permission(self, telecom_user_toolkit):
        """Raises error for invalid permission."""
        with pytest.raises(ValueError, match="Unknown permission"):
            telecom_user_toolkit.use_tool(
                "grant_app_permission",
                app_name="messaging",
                permission="invalid_perm",
                grant=True,
            )


# =============================================================================
# MMS Tests
# =============================================================================


@pytest.mark.benchmark
class TestMMSOperations:
    """Tests for MMS operations."""

    def test_can_send_mms(self, telecom_user_toolkit):
        """Returns MMS capability status."""
        result = telecom_user_toolkit.use_tool("can_send_mms")
        assert isinstance(result, bool)

    def test_can_send_mms_without_data(self, telecom_user_toolkit):
        """MMS unavailable without mobile data."""
        telecom_user_toolkit.use_tool("toggle_data", enable=False)

        result = telecom_user_toolkit.use_tool("can_send_mms")

        assert result is False


# =============================================================================
# Payment Tests
# =============================================================================


@pytest.mark.benchmark
class TestPaymentOperations:
    """Tests for payment operations."""

    def test_check_payment_request(self, telecom_user_toolkit):
        """Returns payment requests."""
        result = telecom_user_toolkit.use_tool("check_payment_request")
        assert isinstance(result, list)

    def test_make_payment(self, telecom_user_toolkit):
        """Makes payment for bill."""
        result = telecom_user_toolkit.use_tool(
            "make_payment",
            bill_id="NONEXISTENT_BILL",
            amount=100.0,
        )

        # Should return not found or success
        assert isinstance(result, str)
