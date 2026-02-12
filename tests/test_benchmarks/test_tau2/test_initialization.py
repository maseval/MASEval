"""Tests for tau2 telecom initialization_actions execution.

Verifies that initialization_actions from telecom tasks are properly executed
during environment setup, matching tau2-bench behavior.
"""

import pytest

from maseval.benchmark.tau2.domains.telecom.user_models import (
    APNNames,
    NetworkModePreference,
    NetworkStatus,
    NetworkTechnology,
    PerformanceLevel,
    SignalStrength,
    SimStatus,
)

pytestmark = [pytest.mark.live, pytest.mark.benchmark]


# =============================================================================
# Initialization Method Tests
# =============================================================================


class TestInitializationMethods:
    """Tests for individual initialization methods on TelecomUserTools."""

    def test_set_user_info(self, telecom_user_toolkit):
        """set_user_info sets name and phone_number on surroundings."""
        telecom_user_toolkit.set_user_info(name="John Smith", phone_number="555-123-2002")

        assert telecom_user_toolkit.db.user_db.surroundings.name == "John Smith"
        assert telecom_user_toolkit.db.user_db.surroundings.phone_number == "555-123-2002"

    def test_set_user_location(self, telecom_user_toolkit):
        """set_user_location sets is_abroad."""
        telecom_user_toolkit.set_user_location(abroad=True)
        assert telecom_user_toolkit.db.user_db.surroundings.is_abroad is True

        telecom_user_toolkit.set_user_location(abroad=False)
        assert telecom_user_toolkit.db.user_db.surroundings.is_abroad is False

    def test_turn_data_off(self, telecom_user_toolkit):
        """turn_data_off disables mobile data."""
        assert telecom_user_toolkit.db.user_db.device.mobile_data_enabled is True
        telecom_user_toolkit.turn_data_off()
        assert telecom_user_toolkit.db.user_db.device.mobile_data_enabled is False

    def test_turn_airplane_mode_on(self, telecom_user_toolkit):
        """turn_airplane_mode_on enables airplane mode with side effects."""
        telecom_user_toolkit.turn_airplane_mode_on()

        device = telecom_user_toolkit.db.user_db.device
        assert device.airplane_mode is True
        assert device.wifi_connected is False
        assert device.network_connection_status == NetworkStatus.NO_SERVICE
        assert device.network_technology_connected == NetworkTechnology.NONE
        assert device.network_signal_strength == SignalStrength.NONE

    def test_turn_roaming_off(self, telecom_user_toolkit):
        """turn_roaming_off disables roaming."""
        telecom_user_toolkit.db.user_db.device.roaming_enabled = True
        telecom_user_toolkit.turn_roaming_off()
        assert telecom_user_toolkit.db.user_db.device.roaming_enabled is False

    def test_turn_roaming_on(self, telecom_user_toolkit):
        """turn_roaming_on enables roaming."""
        telecom_user_toolkit.turn_roaming_on()
        assert telecom_user_toolkit.db.user_db.device.roaming_enabled is True

    def test_turn_data_saver_mode_on(self, telecom_user_toolkit):
        """turn_data_saver_mode_on enables data saver."""
        telecom_user_toolkit.turn_data_saver_mode_on()
        assert telecom_user_toolkit.db.user_db.device.data_saver_mode is True

    def test_unseat_sim_card(self, telecom_user_toolkit):
        """unseat_sim_card marks SIM as missing and drops network."""
        telecom_user_toolkit.unseat_sim_card()

        device = telecom_user_toolkit.db.user_db.device
        assert device.sim_card_missing is True
        assert device.network_connection_status == NetworkStatus.NO_SERVICE

    def test_lock_sim_card_pin(self, telecom_user_toolkit):
        """lock_sim_card with pin locks SIM and drops network."""
        telecom_user_toolkit.lock_sim_card(mode="pin")

        device = telecom_user_toolkit.db.user_db.device
        assert device.sim_status == SimStatus.LOCKED_PIN
        assert device.network_connection_status == NetworkStatus.NO_SERVICE

    def test_lock_sim_card_puk(self, telecom_user_toolkit):
        """lock_sim_card with puk locks SIM and drops network."""
        telecom_user_toolkit.lock_sim_card(mode="puk")

        device = telecom_user_toolkit.db.user_db.device
        assert device.sim_status == SimStatus.LOCKED_PUK
        assert device.network_connection_status == NetworkStatus.NO_SERVICE

    def test_break_apn_settings(self, telecom_user_toolkit):
        """break_apn_settings sets APN to broken and drops network."""
        telecom_user_toolkit.break_apn_settings()

        device = telecom_user_toolkit.db.user_db.device
        assert device.apn_settings.name == APNNames.BROKEN.value
        assert device.network_connection_status == NetworkStatus.NO_SERVICE

    def test_break_apn_mms_setting(self, telecom_user_toolkit):
        """break_apn_mms_setting clears MMSC URL."""
        telecom_user_toolkit.break_apn_mms_setting()
        assert telecom_user_toolkit.db.user_db.device.apn_settings.mmsc_url == ""

    def test_break_vpn(self, telecom_user_toolkit):
        """break_vpn connects VPN with poor performance."""
        telecom_user_toolkit.break_vpn()

        device = telecom_user_toolkit.db.user_db.device
        assert device.vpn_status is True
        assert device.vpn_details is not None
        assert device.vpn_details.server_performance == PerformanceLevel.POOR

    def test_remove_app_permission(self, telecom_user_toolkit):
        """remove_app_permission revokes a permission."""
        # Default messaging app has sms=True
        assert telecom_user_toolkit.db.user_db.device.installed_apps["messaging"].permissions.sms is True

        telecom_user_toolkit.remove_app_permission("messaging", "sms")
        assert telecom_user_toolkit.db.user_db.device.installed_apps["messaging"].permissions.sms is False

    def test_set_wifi_calling(self, telecom_user_toolkit):
        """set_wifi_calling sets wifi calling and mms_over_wifi."""
        telecom_user_toolkit.set_wifi_calling(enabled=True, mms_over_wifi=True)

        device = telecom_user_toolkit.db.user_db.device
        assert device.wifi_calling_enabled is True
        assert device.wifi_calling_mms_over_wifi is True


# =============================================================================
# simulate_network_search Tests
# =============================================================================


class TestSimulateNetworkSearch:
    """Tests for simulate_network_search behavior."""

    def test_active_sim_default_preference(self, telecom_user_toolkit):
        """With active SIM and default 4G/5G preference, connects to best available."""
        telecom_user_toolkit.simulate_network_search()

        device = telecom_user_toolkit.db.user_db.device
        assert device.network_connection_status == NetworkStatus.CONNECTED
        # Default signal_strength only has 4G and 3G, no 5G → should connect to 4G
        assert device.network_technology_connected == NetworkTechnology.FOUR_G
        assert device.network_signal_strength == SignalStrength.GOOD

    def test_3g_only_preference(self, telecom_user_toolkit):
        """With 3G only preference, connects to 3G."""
        telecom_user_toolkit.db.user_db.device.network_mode_preference = NetworkModePreference.THREE_G_ONLY
        telecom_user_toolkit.simulate_network_search()

        device = telecom_user_toolkit.db.user_db.device
        assert device.network_connection_status == NetworkStatus.CONNECTED
        assert device.network_technology_connected == NetworkTechnology.THREE_G
        assert device.network_signal_strength == SignalStrength.FAIR

    def test_2g_only_no_signal(self, telecom_user_toolkit):
        """With 2G only preference but no 2G signal, gets no signal."""
        telecom_user_toolkit.db.user_db.device.network_mode_preference = NetworkModePreference.TWO_G_ONLY
        # Default signal_strength doesn't include 2G
        telecom_user_toolkit.simulate_network_search()

        device = telecom_user_toolkit.db.user_db.device
        assert device.network_signal_strength == SignalStrength.NONE

    def test_airplane_mode_no_service(self, telecom_user_toolkit):
        """Airplane mode results in NO_SERVICE."""
        telecom_user_toolkit.db.user_db.device.airplane_mode = True
        telecom_user_toolkit.simulate_network_search()

        device = telecom_user_toolkit.db.user_db.device
        assert device.network_connection_status == NetworkStatus.NO_SERVICE
        assert device.network_technology_connected == NetworkTechnology.NONE

    def test_missing_sim_no_service(self, telecom_user_toolkit):
        """Missing SIM card results in NO_SERVICE."""
        telecom_user_toolkit.db.user_db.device.sim_card_missing = True
        telecom_user_toolkit.simulate_network_search()

        device = telecom_user_toolkit.db.user_db.device
        assert device.network_connection_status == NetworkStatus.NO_SERVICE

    def test_locked_sim_no_service(self, telecom_user_toolkit):
        """Locked SIM card results in NO_SERVICE."""
        telecom_user_toolkit.db.user_db.device.sim_status = SimStatus.LOCKED_PIN
        telecom_user_toolkit.simulate_network_search()

        device = telecom_user_toolkit.db.user_db.device
        assert device.network_connection_status == NetworkStatus.NO_SERVICE

    def test_broken_apn_no_service(self, telecom_user_toolkit):
        """Broken APN results in NO_SERVICE."""
        telecom_user_toolkit.db.user_db.device.apn_settings.name = APNNames.BROKEN.value
        telecom_user_toolkit.simulate_network_search()

        device = telecom_user_toolkit.db.user_db.device
        assert device.network_connection_status == NetworkStatus.NO_SERVICE

    def test_inactive_line_no_service(self, telecom_user_toolkit):
        """Inactive line results in NO_SERVICE."""
        telecom_user_toolkit.db.user_db.surroundings.line_active = False
        telecom_user_toolkit.simulate_network_search()

        device = telecom_user_toolkit.db.user_db.device
        assert device.network_connection_status == NetworkStatus.NO_SERVICE


# =============================================================================
# Speed Test After Initialization Tests
# =============================================================================


class TestSpeedTestWithInitialization:
    """Tests verifying speed test behavior matches tau2-bench after init."""

    def test_speed_test_after_data_off(self, telecom_user_toolkit):
        """Speed test returns No Connection after turn_data_off."""
        telecom_user_toolkit.turn_data_off()
        result = telecom_user_toolkit.use_tool("run_speed_test")
        assert "no" in result.lower()

    def test_speed_test_after_airplane_mode(self, telecom_user_toolkit):
        """Speed test returns no connection after turn_airplane_mode_on."""
        telecom_user_toolkit.turn_airplane_mode_on()
        result = telecom_user_toolkit.use_tool("run_speed_test")
        assert "airplane" in result.lower() or "no" in result.lower()

    def test_speed_test_after_break_vpn(self, telecom_user_toolkit):
        """Speed test returns reduced speed after break_vpn (poor VPN)."""
        telecom_user_toolkit.break_vpn()
        speed, desc = telecom_user_toolkit._run_speed_test()
        # With poor VPN (0.1x factor), 4G Good signal:
        # (10+100)/2 * 0.8 * 0.1 = 4.4 Mbps → "Poor"
        assert speed is not None
        assert speed < 5.0
        assert desc == "Poor"


# =============================================================================
# All Init Func Names Callable Test
# =============================================================================


class TestAllFuncNamesCallable:
    """Verify all func_names used in task JSON are callable on the toolkits."""

    USER_FUNC_NAMES = [
        "set_user_info",
        "set_user_location",
        "turn_data_off",
        "turn_airplane_mode_on",
        "turn_roaming_off",
        "turn_roaming_on",
        "turn_data_saver_mode_on",
        "unseat_sim_card",
        "lock_sim_card",
        "break_apn_settings",
        "break_apn_mms_setting",
        "break_vpn",
        "remove_app_permission",
        "set_network_mode_preference",
        "set_wifi_calling",
        "simulate_network_search",
    ]

    ASSISTANT_FUNC_NAMES = [
        "set_data_usage",
        "enable_roaming",
        "disable_roaming",
        "suspend_line_for_overdue_bill",
    ]

    @pytest.mark.parametrize("func_name", USER_FUNC_NAMES)
    def test_user_func_callable(self, telecom_user_toolkit, func_name):
        """Each user-side init func_name is callable on user toolkit."""
        func = getattr(telecom_user_toolkit, func_name, None)
        assert func is not None, f"User function '{func_name}' not found"
        assert callable(func), f"User function '{func_name}' not callable"

    @pytest.mark.parametrize("func_name", ASSISTANT_FUNC_NAMES)
    def test_assistant_func_callable(self, telecom_toolkit, func_name):
        """Each assistant-side init func_name is callable on agent toolkit."""
        func = getattr(telecom_toolkit, func_name, None)
        assert func is not None, f"Assistant function '{func_name}' not found"
        assert callable(func), f"Assistant function '{func_name}' not callable"
