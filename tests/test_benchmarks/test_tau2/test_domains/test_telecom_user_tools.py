"""Unit tests for Tau2 telecom user tools.

Tests match the original tau2-bench tool signatures:
- Toggle tools take NO args (toggle current state)
- All tools return str (not dict/bool)
- connect_vpn takes no args (uses default VPN details)
- make_payment takes no args (pays single pending payment_request)
"""

import pytest
from maseval.benchmark.tau2.domains.telecom.user_models import (
    SimStatus,
    NetworkStatus,
    NetworkModePreference,
    APNNames,
)

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
        """check_status_bar returns a string with Status Bar prefix."""
        status = telecom_user_toolkit.use_tool("check_status_bar")
        assert isinstance(status, str)
        assert "Status Bar:" in status

    def test_check_network_status(self, telecom_user_toolkit):
        """check_network_status returns formatted string."""
        status = telecom_user_toolkit.use_tool("check_network_status")
        assert isinstance(status, str)
        assert "Airplane Mode:" in status
        assert "Mobile Data Enabled:" in status

    def test_check_installed_apps(self, telecom_user_toolkit):
        """check_installed_apps returns string listing apps."""
        apps = telecom_user_toolkit.use_tool("check_installed_apps")
        assert isinstance(apps, str)
        assert "messaging" in apps
        assert "browser" in apps

    def test_check_apn_settings(self, telecom_user_toolkit):
        """check_apn_settings returns formatted string."""
        apn = telecom_user_toolkit.use_tool("check_apn_settings")
        assert isinstance(apn, str)
        assert "APN Name:" in apn or "internet" in apn


# =============================================================================
# Write Tool Tests
# =============================================================================


@pytest.mark.benchmark
class TestTelecomUserWriteTools:
    """Tests for telecom user state-modifying tools."""

    def test_toggle_airplane_mode(self, telecom_user_toolkit):
        """toggle_airplane_mode toggles state (no args)."""
        # First toggle: OFF → ON
        result = telecom_user_toolkit.use_tool("toggle_airplane_mode")
        assert isinstance(result, str)
        assert telecom_user_toolkit.db.user_db.device.airplane_mode is True
        assert telecom_user_toolkit.db.user_db.device.network_connection_status == NetworkStatus.NO_SERVICE

        # Second toggle: ON → OFF
        result = telecom_user_toolkit.use_tool("toggle_airplane_mode")
        assert telecom_user_toolkit.db.user_db.device.airplane_mode is False
        if telecom_user_toolkit.db.user_db.device.sim_card_status == SimStatus.ACTIVE:
            assert telecom_user_toolkit.db.user_db.device.network_connection_status == NetworkStatus.CONNECTED

    def test_toggle_data(self, telecom_user_toolkit):
        """toggle_data toggles mobile data (no args)."""
        initial = telecom_user_toolkit.db.user_db.device.data_enabled
        result = telecom_user_toolkit.use_tool("toggle_data")
        assert isinstance(result, str)
        assert telecom_user_toolkit.db.user_db.device.data_enabled is (not initial)

        result = telecom_user_toolkit.use_tool("toggle_data")
        assert telecom_user_toolkit.db.user_db.device.data_enabled is initial

    def test_set_network_mode_preference(self, telecom_user_toolkit):
        """set_network_mode_preference changes state."""
        pref = NetworkModePreference.FOUR_G_ONLY
        result = telecom_user_toolkit.use_tool("set_network_mode_preference", mode=pref)

        assert telecom_user_toolkit.db.user_db.device.network_mode_preference == NetworkModePreference.FOUR_G_ONLY
        assert isinstance(result, str)

    def test_set_network_mode_invalid(self, telecom_user_toolkit):
        """set_network_mode_preference handles invalid input."""
        result = telecom_user_toolkit.use_tool("set_network_mode_preference", mode="invalid_mode")
        assert "Failed" in result

    def test_reboot_device(self, telecom_user_toolkit):
        """reboot_device resets APN if reset_at_reboot and calls simulate_network_search."""
        # Set reset_at_reboot = True
        telecom_user_toolkit.db.user_db.device.active_apn_settings.reset_at_reboot = True
        telecom_user_toolkit.db.user_db.device.active_apn_settings.apn_name = APNNames.BROKEN

        telecom_user_toolkit.use_tool("reboot_device")

        # Should have reset to default
        assert telecom_user_toolkit.db.user_db.device.active_apn_settings.apn_name == APNNames.INTERNET
        # simulate_network_search should have reconnected
        assert telecom_user_toolkit.db.user_db.device.network_connection_status == NetworkStatus.CONNECTED


# =============================================================================
# Wi-Fi Tests
# =============================================================================


@pytest.mark.benchmark
class TestWiFiOperations:
    """Tests for Wi-Fi operations."""

    def test_check_wifi_status(self, telecom_user_toolkit):
        """Returns Wi-Fi status string."""
        result = telecom_user_toolkit.use_tool("check_wifi_status")
        assert isinstance(result, str)

    def test_toggle_wifi(self, telecom_user_toolkit):
        """Toggles Wi-Fi (no args)."""
        initial = telecom_user_toolkit.db.user_db.device.wifi_enabled
        result = telecom_user_toolkit.use_tool("toggle_wifi")
        assert isinstance(result, str)
        assert telecom_user_toolkit.db.user_db.device.wifi_enabled is (not initial)

    def test_toggle_wifi_off_disconnects(self, telecom_user_toolkit):
        """Toggling Wi-Fi off disconnects."""
        # Ensure wifi is on first
        telecom_user_toolkit.db.user_db.device.wifi_enabled = True
        telecom_user_toolkit.db.user_db.device.wifi_connected = True

        telecom_user_toolkit.use_tool("toggle_wifi")  # toggle off
        assert telecom_user_toolkit.db.user_db.device.wifi_enabled is False
        assert telecom_user_toolkit.db.user_db.device.wifi_connected is False

    def test_check_wifi_calling_status(self, telecom_user_toolkit):
        """Returns Wi-Fi calling status string."""
        result = telecom_user_toolkit.use_tool("check_wifi_calling_status")
        assert isinstance(result, str)

    def test_toggle_wifi_calling(self, telecom_user_toolkit):
        """Toggles Wi-Fi calling (no args)."""
        initial = telecom_user_toolkit.db.user_db.device.wifi_calling_enabled
        telecom_user_toolkit.use_tool("toggle_wifi_calling")
        assert telecom_user_toolkit.db.user_db.device.wifi_calling_enabled is (not initial)


# =============================================================================
# SIM Card Tests
# =============================================================================


@pytest.mark.benchmark
class TestSimCardOperations:
    """Tests for SIM card operations."""

    def test_check_sim_status(self, telecom_user_toolkit):
        """Returns SIM status string."""
        result = telecom_user_toolkit.use_tool("check_sim_status")
        assert isinstance(result, str)
        assert "SIM" in result or "active" in result.lower()

    def test_reseat_sim_card(self, telecom_user_toolkit):
        """Reseats SIM card."""
        result = telecom_user_toolkit.use_tool("reseat_sim_card")
        assert isinstance(result, str)
        assert "re-seated" in result.lower()


# =============================================================================
# Roaming Tests
# =============================================================================


@pytest.mark.benchmark
class TestRoamingOperations:
    """Tests for roaming operations."""

    def test_toggle_roaming(self, telecom_user_toolkit):
        """Toggles data roaming (no args)."""
        initial = telecom_user_toolkit.db.user_db.device.roaming_enabled
        result = telecom_user_toolkit.use_tool("toggle_roaming")
        assert isinstance(result, str)
        assert telecom_user_toolkit.db.user_db.device.roaming_enabled is (not initial)


# =============================================================================
# Data Saver Tests
# =============================================================================


@pytest.mark.benchmark
class TestDataSaverOperations:
    """Tests for data saver operations."""

    def test_check_data_restriction_status(self, telecom_user_toolkit):
        """Returns data saver status string."""
        result = telecom_user_toolkit.use_tool("check_data_restriction_status")
        assert isinstance(result, str)
        assert "Data Saver" in result

    def test_toggle_data_saver_mode(self, telecom_user_toolkit):
        """Toggles data saver mode (no args)."""
        initial = telecom_user_toolkit.db.user_db.device.data_saver_mode
        result = telecom_user_toolkit.use_tool("toggle_data_saver_mode")
        assert isinstance(result, str)
        assert telecom_user_toolkit.db.user_db.device.data_saver_mode is (not initial)


# =============================================================================
# APN Settings Tests
# =============================================================================


@pytest.mark.benchmark
class TestAPNSettingsOperations:
    """Tests for APN settings operations."""

    def test_check_apn_settings(self, telecom_user_toolkit):
        """check_apn_settings returns string with APN info."""
        result = telecom_user_toolkit.use_tool("check_apn_settings")
        assert isinstance(result, str)
        assert "internet" in result.lower()

    def test_reset_apn_settings(self, telecom_user_toolkit):
        """Resets APN settings (deferred - sets reset_at_reboot)."""
        result = telecom_user_toolkit.use_tool("reset_apn_settings")
        assert isinstance(result, str)
        assert "reboot" in result.lower()
        assert telecom_user_toolkit.db.user_db.device.active_apn_settings.reset_at_reboot is True


# =============================================================================
# VPN Tests
# =============================================================================


@pytest.mark.benchmark
class TestVPNOperations:
    """Tests for VPN operations."""

    def test_check_vpn_status(self, telecom_user_toolkit):
        """Returns VPN status string."""
        result = telecom_user_toolkit.use_tool("check_vpn_status")
        assert isinstance(result, str)

    def test_connect_vpn(self, telecom_user_toolkit):
        """Connects to VPN (no args - uses default details)."""
        result = telecom_user_toolkit.use_tool("connect_vpn")
        assert isinstance(result, str)
        assert telecom_user_toolkit.db.user_db.device.vpn_connected is True
        assert telecom_user_toolkit.db.user_db.device.vpn_details is not None

    def test_disconnect_vpn(self, telecom_user_toolkit):
        """Disconnects from VPN."""
        # First connect
        telecom_user_toolkit.use_tool("connect_vpn")

        result = telecom_user_toolkit.use_tool("disconnect_vpn")
        assert isinstance(result, str)
        assert telecom_user_toolkit.db.user_db.device.vpn_connected is False

    def test_disconnect_vpn_when_not_connected(self, telecom_user_toolkit):
        """Disconnecting when not connected returns appropriate message."""
        telecom_user_toolkit.db.user_db.device.vpn_connected = False

        result = telecom_user_toolkit.use_tool("disconnect_vpn")
        assert isinstance(result, str)
        assert "no active" in result.lower()


# =============================================================================
# Speed Test Tests
# =============================================================================


@pytest.mark.benchmark
class TestSpeedTestOperations:
    """Tests for network speed test."""

    def test_run_speed_test_with_connection(self, telecom_user_toolkit):
        """Speed test returns results with connection."""
        result = telecom_user_toolkit.use_tool("run_speed_test")
        assert isinstance(result, str)

    def test_run_speed_test_airplane_mode(self, telecom_user_toolkit):
        """Speed test fails in airplane mode."""
        telecom_user_toolkit.turn_airplane_mode_on()

        result = telecom_user_toolkit.use_tool("run_speed_test")
        assert isinstance(result, str)
        assert "failed" in result.lower() or "no" in result.lower()

        # Cleanup
        telecom_user_toolkit.turn_airplane_mode_off()

    def test_run_speed_test_no_data(self, telecom_user_toolkit):
        """Speed test fails without data."""
        telecom_user_toolkit.turn_data_off()

        result = telecom_user_toolkit.use_tool("run_speed_test")
        assert isinstance(result, str)
        assert "failed" in result.lower() or "no" in result.lower()


# =============================================================================
# Application Tests
# =============================================================================


@pytest.mark.benchmark
class TestApplicationOperations:
    """Tests for application operations using default apps (messaging, browser)."""

    def test_check_installed_apps_has_defaults(self, telecom_user_toolkit):
        """Default apps are installed."""
        apps = telecom_user_toolkit.use_tool("check_installed_apps")
        assert isinstance(apps, str)
        assert "messaging" in apps
        assert "browser" in apps

    def test_check_app_status(self, telecom_user_toolkit):
        """Returns app status for messaging app."""
        result = telecom_user_toolkit.use_tool("check_app_status", app_name="messaging")
        assert isinstance(result, str)
        assert "messaging" in result.lower()

    def test_check_app_status_invalid(self, telecom_user_toolkit):
        """Returns not found for invalid app."""
        result = telecom_user_toolkit.use_tool("check_app_status", app_name="nonexistent_app_12345")
        assert isinstance(result, str)
        assert "not found" in result.lower()

    def test_check_app_permissions(self, telecom_user_toolkit):
        """Returns app permissions for browser app."""
        result = telecom_user_toolkit.use_tool("check_app_permissions", app_name="browser")
        assert isinstance(result, str)
        assert "network" in result.lower() or "storage" in result.lower()

    def test_grant_app_permission(self, telecom_user_toolkit):
        """Grants permission to messaging app."""
        result = telecom_user_toolkit.use_tool(
            "grant_app_permission",
            app_name="messaging",
            permission="network",
        )
        assert isinstance(result, str)
        assert "granted" in result.lower() or "success" in result.lower()


# =============================================================================
# MMS Tests
# =============================================================================


@pytest.mark.benchmark
class TestMMSOperations:
    """Tests for MMS operations."""

    def test_can_send_mms(self, telecom_user_toolkit):
        """Returns MMS capability string."""
        result = telecom_user_toolkit.use_tool("can_send_mms")
        assert isinstance(result, str)

    def test_can_send_mms_without_data(self, telecom_user_toolkit):
        """MMS unavailable without mobile data."""
        telecom_user_toolkit.turn_data_off()

        result = telecom_user_toolkit.use_tool("can_send_mms")
        assert isinstance(result, str)
        assert "cannot" in result.lower()


# =============================================================================
# Payment Tests
# =============================================================================


@pytest.mark.benchmark
class TestPaymentOperations:
    """Tests for payment operations."""

    def test_check_payment_request(self, telecom_user_toolkit):
        """Returns payment request status string."""
        result = telecom_user_toolkit.use_tool("check_payment_request")
        assert isinstance(result, str)

    def test_make_payment_no_request(self, telecom_user_toolkit):
        """Making payment with no pending request returns appropriate message."""
        result = telecom_user_toolkit.use_tool("make_payment")
        assert isinstance(result, str)
        assert "do not have" in result.lower() or "no" in result.lower()
