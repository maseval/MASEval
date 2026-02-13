"""Tau 2 Benchmark - Telecom Domain User Tools.

Tools for the user to interact with their simulated phone device.

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Adapted from: src/tau2/domains/telecom/user_tools.py
"""

from typing import Any, Dict, List, Optional, Tuple

from maseval.benchmark.tau2.domains.base import ToolKitBase, ToolType, is_tool
from maseval.benchmark.tau2.domains.telecom.db import TelecomDB
from maseval.benchmark.tau2.domains.telecom.user_models import (
    APNNames,
    APNSettings,
    NetworkModePreference,
    NetworkStatus,
    NetworkTechnology,
    PerformanceLevel,
    SignalStrength,
    SimStatus,
    VpnDetails,
)


class TelecomUserTools(ToolKitBase[TelecomDB]):
    """Tools for the user to interact with their phone device.

    These tools modify the user_db part of the TelecomDB.
    """

    def __init__(self, db: TelecomDB) -> None:
        super().__init__(db)
        if self.db and self.db.user_db is None:
            # Initialize user DB if not present
            from maseval.benchmark.tau2.domains.telecom.user_models import TelecomUserDB

            self.db.user_db = TelecomUserDB()

    @property
    def _device(self):
        if self.db is None or self.db.user_db is None:
            raise ValueError("User database not initialized")
        return self.db.user_db.device

    @property
    def _surroundings(self):
        if self.db is None or self.db.user_db is None:
            raise ValueError("User database not initialized")
        return self.db.user_db.surroundings

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _get_mobile_data_working(self) -> bool:
        """Check if mobile data connection is working.

        Checks all required conditions for mobile data to function:
        - Mobile data enabled
        - Not in airplane mode
        - SIM is active
        - Network is connected
        - Signal is available
        - If abroad: roaming must be enabled and supported

        Returns:
            True if mobile data is working, False otherwise
        """
        device = self._device
        surroundings = self._surroundings

        # Basic requirements
        if not device.mobile_data_enabled:
            return False
        if device.airplane_mode:
            return False
        if device.sim_status != SimStatus.ACTIVE:
            return False
        if device.network_status != NetworkStatus.CONNECTED:
            return False
        if surroundings.signal_strength == SignalStrength.NONE:
            return False

        # Roaming requirements
        if surroundings.is_abroad:
            if not device.roaming_enabled:
                return False
            if not surroundings.roaming_allowed_in_location:
                return False

        return True

    def _run_speed_test(self) -> Tuple[Optional[float], str]:
        """Run speed test and return numeric speed and description.

        Calculates speed based on:
        - Network technology (2G-5G have different speed ranges)
        - Signal strength multiplier
        - VPN impact (90% reduction if poor performance)
        - Data saver mode (80% reduction)

        Returns:
            Tuple of (speed_mbps, description). Speed is None if no connection.
        """
        if not self._get_mobile_data_working():
            return None, "No Connection"

        device = self._device
        surroundings = self._surroundings

        # Base factor starts at 1.0
        base_factor = 1.0

        # VPN impact - poor VPN performance reduces speed by 90%
        if device.vpn_status and device.vpn_details:
            if device.vpn_details.server_performance == PerformanceLevel.POOR:
                base_factor *= 0.1

        # Data saver mode reduces speed by 80%
        if device.data_saver_mode:
            base_factor *= 0.2

        # Network technology speed ranges (Mbps)
        tech_speeds: Dict[NetworkTechnology, Tuple[float, float]] = {
            NetworkTechnology.TWO_G: (0.1, 0.4),
            NetworkTechnology.THREE_G: (1.0, 5.0),
            NetworkTechnology.FOUR_G: (10.0, 100.0),
            NetworkTechnology.FIVE_G: (50.0, 500.0),
        }

        min_speed, max_speed = tech_speeds.get(device.network_technology, (0.0, 0.0))

        # Signal strength factor
        signal_factors: Dict[SignalStrength, float] = {
            SignalStrength.NONE: 0.0,
            SignalStrength.POOR: 0.2,
            SignalStrength.FAIR: 0.5,
            SignalStrength.GOOD: 0.8,
            SignalStrength.EXCELLENT: 1.0,
        }
        signal_factor = signal_factors.get(surroundings.signal_strength, 0.0)

        # Calculate final speed
        speed = (min_speed + max_speed) / 2 * signal_factor * base_factor

        # Determine description based on speed thresholds
        if speed < 1:
            desc = "Very Poor"
        elif speed < 5:
            desc = "Poor"
        elif speed < 25:
            desc = "Fair"
        elif speed < 100:
            desc = "Good"
        else:
            desc = "Excellent"

        return speed, desc

    def _can_send_mms(self) -> bool:
        """Check if MMS can be sent.

        Returns:
            True if MMS sending is possible, False otherwise
        """
        if not self._get_mobile_data_working():
            return False

        # Check APN MMSC URL
        if not self._device.apn_settings.mmsc_url:
            return False

        return True

    # =========================================================================
    # Status Bar
    # =========================================================================

    @is_tool(ToolType.READ)
    def check_status_bar(self) -> Dict[str, Any]:
        """Check the status bar of the phone.

        Returns:
            Dictionary with status indicators:
            - signal_strength: NONE, POOR, FAIR, GOOD, EXCELLENT
            - network_type: 2g, 3g, 4g, 5g, none
            - wifi_connected: True/False
            - airplane_mode: True/False
            - battery_level: 0-100
        """
        return {
            "signal_strength": self._surroundings.signal_strength.value,
            "network_type": self._device.network_technology.value,
            "wifi_connected": self._device.wifi_connected,
            "airplane_mode": self._device.airplane_mode,
            "battery_level": self._device.battery_level,
        }

    # =========================================================================
    # Network & Connectivity
    # =========================================================================

    @is_tool(ToolType.READ)
    def check_network_status(self) -> Dict[str, Any]:
        """Check detailed network status.

        Returns:
            Dictionary with:
            - status: connected, searching, no_service, emergency_only
            - technology: 2g, 3g, 4g, 5g, none
            - roaming: True/False
            - mobile_data: True/False
        """
        return {
            "status": self._device.network_status.value,
            "technology": self._device.network_technology.value,
            "roaming": self._device.roaming_enabled,
            "mobile_data": self._device.mobile_data_enabled,
        }

    @is_tool(ToolType.READ)
    def check_network_mode_preference(self) -> str:
        """Check the preferred network mode setting.

        Returns:
            Current preference (e.g., "4g_5g_preferred")
        """
        return self._device.network_mode_preference.value

    @is_tool(ToolType.WRITE)
    def set_network_mode_preference(self, preference: str) -> str:
        """Set the preferred network mode.

        Args:
            preference: One of: "4g_5g_preferred", "4g_only", "3g_only", "2g_only"

        Returns:
            Success message
        """
        try:
            mode = NetworkModePreference(preference)
            self._device.network_mode_preference = mode
            return f"Network mode preference set to {preference}"
        except ValueError:
            valid_modes = [m.value for m in NetworkModePreference]
            return f"Invalid preference. Must be one of: {', '.join(valid_modes)}"

    @is_tool(ToolType.READ)
    def run_speed_test(self) -> str:
        """Run a network speed test.

        Calculates speed based on network technology, signal strength,
        VPN performance, and data saver mode.

        Returns:
            Description of network speed with download/upload values.
        """
        speed, desc = self._run_speed_test()

        if speed is None:
            # Check specific failure reasons for better error messages
            if self._device.airplane_mode:
                return "Airplane mode is on. No connection."
            if not self._device.mobile_data_enabled and not self._device.wifi_connected:
                return "No internet connection available."
            if self._surroundings.signal_strength == SignalStrength.NONE:
                return "Speed test failed: No signal."
            return "No internet connection available."

        # Format output with download speed and upload (estimated as 1/5 of download)
        upload = speed / 5
        return f"Download: {speed:.1f} Mbps, Upload: {upload:.1f} Mbps ({desc})"

    # =========================================================================
    # Airplane Mode
    # =========================================================================

    @is_tool(ToolType.WRITE)
    def toggle_airplane_mode(self, enable: bool) -> str:
        """Turn airplane mode on or off.

        Args:
            enable: True to turn on, False to turn off

        Returns:
            Success message
        """
        self._device.airplane_mode = enable
        state = "enabled" if enable else "disabled"

        # Side effects
        if enable:
            self._device.network_status = NetworkStatus.NO_SERVICE
            self._device.wifi_connected = False
            # Bluetooth usually turns off too, but we don't model bluetooth
        else:
            # Reconnect logic would be complex, simplified here:
            if self._device.sim_status == SimStatus.ACTIVE:
                self._device.network_status = NetworkStatus.CONNECTED

            if self._surroundings.wifi_networks_available and self._device.wifi_enabled:
                self._device.wifi_connected = True

        return f"Airplane mode {state}"

    # =========================================================================
    # SIM Card
    # =========================================================================

    @is_tool(ToolType.READ)
    def check_sim_status(self) -> str:
        """Check the status of the SIM card.

        Returns:
            Status string (active, missing, locked_pin, locked_puk)
        """
        return self._device.sim_status.value

    @is_tool(ToolType.WRITE)
    def reseat_sim_card(self) -> str:
        """Remove and re-insert the SIM card.

        Useful for troubleshooting connectivity issues.

        Returns:
            Success message
        """
        if not self._device.has_sim_card:
            return "No physical SIM card to reseat."

        # Simulation of reseating
        if self._device.sim_status != SimStatus.MISSING:
            # If it was locked or active, it might reset or stay same
            # Simplified: just say done
            pass

        return "SIM card reseated. Please wait for network registration."

    # =========================================================================
    # Mobile Data & Roaming
    # =========================================================================

    @is_tool(ToolType.WRITE)
    def toggle_data(self, enable: bool) -> str:
        """Turn mobile data on or off.

        Args:
            enable: True to turn on, False to turn off

        Returns:
            Success message
        """
        self._device.mobile_data_enabled = enable
        state = "enabled" if enable else "disabled"
        return f"Mobile data {state}"

    @is_tool(ToolType.WRITE)
    def toggle_roaming(self, enable: bool) -> str:
        """Turn data roaming on or off.

        Args:
            enable: True to turn on, False to turn off

        Returns:
            Success message
        """
        self._device.roaming_enabled = enable
        state = "enabled" if enable else "disabled"
        return f"Data roaming {state}"

    @is_tool(ToolType.READ)
    def check_data_restriction_status(self) -> bool:
        """Check if background data is restricted (Data Saver).

        Returns:
            True if data saver is on, False otherwise
        """
        return self._device.data_saver_mode

    @is_tool(ToolType.WRITE)
    def toggle_data_saver_mode(self, enable: bool) -> str:
        """Turn Data Saver mode on or off.

        Args:
            enable: True to turn on, False to turn off

        Returns:
            Success message
        """
        self._device.data_saver_mode = enable
        state = "enabled" if enable else "disabled"
        return f"Data Saver mode {state}"

    # =========================================================================
    # APN Settings
    # =========================================================================

    @is_tool(ToolType.READ)
    def check_apn_settings(self) -> Dict[str, Any]:
        """Check current Access Point Name (APN) settings.

        Returns:
            Dictionary with APN configuration.
        """
        apn = self._device.apn_settings
        return apn.model_dump()

    @is_tool(ToolType.WRITE)
    def set_apn_settings(self, **kwargs: Any) -> str:
        """Update APN settings.

        Args:
            **kwargs: Fields to update (name, apn, proxy, port, etc.)

        Returns:
            Success message
        """
        current = self._device.apn_settings.model_dump()
        current.update(kwargs)
        try:
            self._device.apn_settings = APNSettings(**current)
            return "APN settings updated."
        except Exception as e:
            return f"Failed to update APN settings: {str(e)}"

    @is_tool(ToolType.WRITE)
    def reset_apn_settings(self) -> str:
        """Reset APN settings to default.

        Returns:
            Success message
        """
        self._device.apn_settings = APNSettings(name=APNNames.INTERNET.value, apn="internet")
        return "APN settings reset to default."

    # =========================================================================
    # Wi-Fi
    # =========================================================================

    @is_tool(ToolType.READ)
    def check_wifi_status(self) -> Dict[str, Any]:
        """Check Wi-Fi status.

        Returns:
            Dictionary with enabled, connected, and available networks.
        """
        return {
            "enabled": self._device.wifi_enabled,
            "connected": self._device.wifi_connected,
            "available_networks": self._surroundings.wifi_networks_available,
        }

    @is_tool(ToolType.WRITE)
    def toggle_wifi(self, enable: bool) -> str:
        """Turn Wi-Fi on or off.

        Args:
            enable: True to turn on, False to turn off

        Returns:
            Success message
        """
        self._device.wifi_enabled = enable
        state = "enabled" if enable else "disabled"
        if not enable:
            self._device.wifi_connected = False
        return f"Wi-Fi {state}"

    @is_tool(ToolType.READ)
    def check_wifi_calling_status(self) -> bool:
        """Check if Wi-Fi calling is enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self._device.wifi_calling_enabled

    @is_tool(ToolType.WRITE)
    def toggle_wifi_calling(self, enable: bool) -> str:
        """Turn Wi-Fi calling on or off.

        Args:
            enable: True to turn on, False to turn off

        Returns:
            Success message
        """
        self._device.wifi_calling_enabled = enable
        state = "enabled" if enable else "disabled"
        return f"Wi-Fi calling {state}"

    # =========================================================================
    # VPN
    # =========================================================================

    @is_tool(ToolType.READ)
    def check_vpn_status(self) -> Dict[str, Any]:
        """Check VPN status.

        Returns:
            Dictionary with status and details if connected.
        """
        status = {
            "connected": self._device.vpn_status,
        }
        if self._device.vpn_status and self._device.vpn_details:
            status["details"] = self._device.vpn_details.model_dump()
        return status

    @is_tool(ToolType.WRITE)
    def connect_vpn(self, server_address: str, protocol: str = "IKEv2") -> str:
        """Connect to a VPN server.

        Args:
            server_address: Address of VPN server
            protocol: VPN protocol

        Returns:
            Success message
        """
        if not self._device.mobile_data_enabled and not self._device.wifi_connected:
            return "Cannot connect to VPN: No internet connection."

        self._device.vpn_status = True
        self._device.vpn_details = VpnDetails(server_address=server_address, protocol=protocol, server_performance=PerformanceLevel.GOOD)
        return f"Connected to VPN at {server_address}"

    @is_tool(ToolType.WRITE)
    def disconnect_vpn(self) -> str:
        """Disconnect from VPN.

        Returns:
            Success message
        """
        if not self._device.vpn_status:
            return "VPN is not connected."

        self._device.vpn_status = False
        self._device.vpn_details = None
        return "VPN disconnected."

    # =========================================================================
    # Applications
    # =========================================================================

    @is_tool(ToolType.READ)
    def check_installed_apps(self) -> List[str]:
        """List all installed applications.

        Returns:
            List of app names
        """
        return list(self._device.installed_apps.keys())

    @is_tool(ToolType.READ)
    def check_app_status(self, app_name: str) -> Dict[str, Any]:
        """Check status of a specific app.

        Args:
            app_name: Name of the app

        Returns:
            Dictionary with is_running, data_usage_mb, permissions
        """
        if app_name not in self._device.installed_apps:
            raise ValueError(f"App '{app_name}' not installed.")

        app = self._device.installed_apps[app_name]
        return {"is_running": app.is_running, "data_usage_mb": app.data_usage_mb, "permissions": app.permissions.model_dump()}

    @is_tool(ToolType.READ)
    def check_app_permissions(self, app_name: str) -> Dict[str, bool]:
        """Check permissions for an app.

        Args:
            app_name: Name of the app

        Returns:
            Dictionary of permissions (sms, storage, phone, network)
        """
        if app_name not in self._device.installed_apps:
            raise ValueError(f"App '{app_name}' not installed.")

        return self._device.installed_apps[app_name].permissions.model_dump()

    @is_tool(ToolType.WRITE)
    def grant_app_permission(self, app_name: str, permission: str, grant: bool) -> str:
        """Grant or revoke a permission for an app.

        Args:
            app_name: Name of the app
            permission: Permission name (sms, storage, phone, network)
            grant: True to grant, False to revoke

        Returns:
            Success message
        """
        if app_name not in self._device.installed_apps:
            raise ValueError(f"App '{app_name}' not installed.")

        perms = self._device.installed_apps[app_name].permissions
        if not hasattr(perms, permission):
            raise ValueError(f"Unknown permission '{permission}'.")

        setattr(perms, permission, grant)
        action = "granted" if grant else "revoked"
        return f"Permission '{permission}' {action} for {app_name}."

    # =========================================================================
    # MMS
    # =========================================================================

    @is_tool(ToolType.READ)
    def can_send_mms(self) -> bool:
        """Check if MMS sending is possible.

        Returns:
            True if possible, False otherwise
        """
        if not self._device.mobile_data_enabled:
            return False

        # Check APN
        if not self._device.apn_settings.mmsc_url:
            return False

        return True

    # =========================================================================
    # Device
    # =========================================================================

    @is_tool(ToolType.WRITE)
    def reboot_device(self) -> str:
        """Reboot the device.

        Returns:
            Success message
        """
        # Reset ephemeral state
        if self._device.apn_settings.reset_at_reboot:
            self._device.apn_settings = APNSettings(name=APNNames.INTERNET.value, apn="internet")

        return "Device rebooted successfully."

    # =========================================================================
    # Payment
    # =========================================================================

    @is_tool(ToolType.READ)
    def check_payment_request(self) -> List[Dict[str, Any]]:
        """Check for pending payment requests.

        Returns:
            List of unpaid requests
        """
        requests = []
        for req in self._surroundings.payment_requests:
            if not req.paid:
                requests.append(req.model_dump())
        return requests

    @is_tool(ToolType.WRITE)
    def make_payment(self, bill_id: str, amount: float) -> str:
        """Make a payment for a bill.

        Args:
            bill_id: ID of the bill
            amount: Amount to pay

        Returns:
            Success message
        """
        found = False
        for req in self._surroundings.payment_requests:
            if req.bill_id == bill_id:
                if req.paid:
                    return "Bill already paid."
                if abs(req.amount_due - amount) > 0.01:
                    return f"Incorrect amount. Expected {req.amount_due}, got {amount}."

                req.paid = True
                found = True
                break

        if not found:
            # Check if bill exists in main DB and pay it there directly?
            # Or is this tool only for responding to requests received on device?
            # For now, let's assume it pays the bill in the main system too if we can access it.
            # But the user tools typically simulate user ACTIONS on the device.
            # The actual payment processing would update the main DB.
            # We can update the main DB bill status here since we have access to self.db

            # Find bill in main DB
            bill = None
            for b in self.db.bills:  # type: ignore[union-attr]
                if b.bill_id == bill_id:
                    bill = b
                    break

            if not bill:
                return f"Bill {bill_id} not found."

            from maseval.benchmark.tau2.domains.telecom.models import BillStatus

            bill.status = BillStatus.PAID
            return f"Payment of {amount} for bill {bill_id} successful."

        return f"Payment of {amount} for bill {bill_id} successful."

    # =========================================================================
    # Assertion Methods (for evaluation)
    # =========================================================================

    @is_tool(ToolType.READ)
    def assert_internet_speed(self, expected_speed: float, expected_desc: Optional[str] = None) -> bool:
        """Assert that internet speed meets expectations.

        Used by evaluator to verify task completion for mobile data issues.

        Args:
            expected_speed: Minimum expected speed in Mbps
            expected_desc: Expected description (e.g., "excellent", "good")

        Returns:
            True if speed meets or exceeds expectations, False otherwise
        """
        speed, desc = self._run_speed_test()

        if speed is None:
            return False

        if expected_desc:
            return speed >= expected_speed and desc.lower() == expected_desc.lower()

        return speed >= expected_speed

    @is_tool(ToolType.READ)
    def assert_mobile_data_status(self, expected_status: bool) -> bool:
        """Assert that mobile data working status matches expected.

        Args:
            expected_status: True if data should be working, False otherwise

        Returns:
            True if actual status matches expected, False otherwise
        """
        actual = self._get_mobile_data_working()
        return actual == expected_status

    @is_tool(ToolType.READ)
    def assert_service_status(self, expected_status: str) -> bool:
        """Assert that network service status matches expected.

        Args:
            expected_status: Expected status (connected, searching, no_service, emergency_only)

        Returns:
            True if actual status matches expected, False otherwise
        """
        actual = self._device.network_status.value
        return actual == expected_status

    @is_tool(ToolType.READ)
    def assert_can_send_mms(self, expected_status: bool) -> bool:
        """Assert that MMS sending capability matches expected.

        Args:
            expected_status: True if MMS should be sendable, False otherwise

        Returns:
            True if actual capability matches expected, False otherwise
        """
        actual = self._can_send_mms()
        return actual == expected_status
