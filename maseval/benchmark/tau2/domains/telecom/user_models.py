"""Tau 2 Benchmark - Telecom Domain User Models.

Pydantic models for the telecom user device and surroundings.

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Adapted from: src/tau2/domains/telecom/user_data_model.py
"""

from enum import Enum
from typing import Any, Dict, Optional, Union

import pydantic
from pydantic import BaseModel, ConfigDict, Field

from maseval.benchmark.tau2.utils import update_pydantic_model_with_dict


class BaseModelNoExtra(BaseModel):
    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Enums
# =============================================================================


class SimStatus(str, Enum):
    ACTIVE = "active"
    MISSING = "missing"
    LOCKED_PIN = "locked_pin"
    LOCKED_PUK = "locked_puk"


class NetworkTechnology(str, Enum):
    NONE = "none"
    TWO_G = "2G"
    THREE_G = "3G"
    FOUR_G = "4G"
    FIVE_G = "5G"


class NetworkModePreference(str, Enum):
    FOUR_G_5G_PREFERRED = "4g_5g_preferred"
    FOUR_G_ONLY = "4g_only"
    THREE_G_ONLY = "3g_only"
    TWO_G_ONLY = "2g_only"


class SignalStrength(str, Enum):
    NONE = "none"
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class PerformanceLevel(str, Enum):
    UNKNOWN = "unknown"
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class NetworkStatus(str, Enum):
    CONNECTED = "connected"
    SEARCHING = "searching"
    NO_SERVICE = "no_service"
    EMERGENCY_ONLY = "emergency_only"


# --- Nested Models for Complex Attributes ---


class APNNames(str, Enum):
    INTERNET = "internet"
    BROKEN = "broken"


class APNSettings(BaseModelNoExtra):
    """Represents the configuration for a single Access Point Name (APN)."""

    apn_name: APNNames = Field(
        APNNames.INTERNET,
        description="The name identifier for the APN connection.",
    )
    reset_at_reboot: bool = Field(
        False,
        description="Whether the APN settings will be reset at the next reboot.",
    )
    mms_apn: Optional[str] = Field(
        "mms",
        description="Specific APN name used for MMS traffic, if different from general data.",
    )
    mmsc_url: Optional[str] = Field(
        "http://mms.carrier.com/mms/wapenc",
        description="The URL of the Multimedia Messaging Service Center (MMSC). Crucial for MMS.",
    )
    mms_proxy: Optional[str] = Field(
        None,
        description="The proxy server address required for MMS traffic on some networks.",
    )
    mms_port: Optional[int] = Field(
        None,
        description="The proxy server port required for MMS traffic on some networks.",
    )

    def is_mms_basic_configured(self) -> bool:
        """Checks if the essential MMSC URL is set."""
        return bool(self.mmsc_url)


class VpnDetails(BaseModelNoExtra):
    """Holds details about the VPN connection if active."""

    server_address: Optional[str] = Field(None, description="Address of the connected VPN server.")
    protocol: Optional[str] = Field(None, description="VPN protocol being used (e.g., WireGuard, OpenVPN).")
    server_performance: PerformanceLevel = Field(
        default=PerformanceLevel.UNKNOWN,
        validate_default=True,
        description="Estimated performance/latency of the VPN connection.",
    )


class AppPermissions(BaseModelNoExtra):
    """Represents the permissions relevant to an application."""

    sms: bool = Field(False, description="Permission to send/read SMS/MMS.")
    storage: bool = Field(False, description="Permission to access device storage.")
    phone: bool = Field(False, description="Permission to make/manage phone calls.")
    network: bool = Field(False, description="Permission to access network state/internet.")


class AppStatus(BaseModelNoExtra):
    """Represents the status of a specific application relevant to issues."""

    app_name: str
    permissions: AppPermissions = Field(
        default_factory=AppPermissions,
        description="Structured permissions relevant to the application.",
    )


class StatusBar(BaseModelNoExtra):
    """Represents the information displayed in the phone's status bar."""

    signal_strength: SignalStrength = Field(
        default=SignalStrength.NONE,
        validate_default=True,
        description="The cellular signal strength shown in the status bar.",
    )
    network_type: NetworkTechnology = Field(
        default=NetworkTechnology.NONE,
        validate_default=True,
        description="The network technology (2G, 3G, 4G, etc.) shown in the status bar.",
    )
    wifi_connected: bool = Field(False, description="Whether WiFi is connected and shown in the status bar.")
    airplane_mode: bool = Field(False, description="Whether airplane mode is on and shown in the status bar.")
    vpn_active: bool = Field(False, description="Whether a VPN is active and shown in the status bar.")
    data_saver_active: bool = Field(
        False,
        description="Whether data saver mode is active and shown in the status bar.",
    )
    battery_level: int = Field(100, description="The battery level (0-100) shown in the status bar.")


# --- Main Device State Model ---


class MockPhoneAttributes(BaseModelNoExtra):
    """Data model representing the state attributes of a mock phone device."""

    # --- SIM and Basic Network ---
    sim_card_status: SimStatus = Field(
        default=SimStatus.ACTIVE,
        validate_default=True,
        description="Current status of the physical or eSIM card.",
    )
    sim_card_missing: bool = Field(
        False,
        description="Whether the SIM card is missing.",
    )
    airplane_mode: bool = Field(
        False,
        description="Whether Airplane Mode, which disables all radios, is currently enabled.",
    )
    network_signal_strength: SignalStrength = Field(
        default=SignalStrength.GOOD,
        validate_default=True,
        description="Current strength of the cellular network signal.",
    )
    network_technology_connected: NetworkTechnology = Field(
        default=NetworkTechnology.FIVE_G,
        validate_default=True,
        description="The type of cellular network technology currently connected (e.g., 5G, 4G).",
    )
    network_connection_status: NetworkStatus = Field(
        default=NetworkStatus.CONNECTED,
        validate_default=True,
        description="High-level network status description (e.g., 'Connected', 'Searching', 'Emergency Calls Only', 'No Service').",
    )

    # --- Battery ---
    battery_level: int = Field(80, description="The current battery level, from 0 to 100 percent.")

    # --- Mobile Data ---
    data_enabled: bool = Field(
        True,
        description="Whether the master switch for Mobile/Cellular Data usage is enabled.",
    )
    roaming_enabled: bool = Field(
        False,
        description="Whether the user setting to allow data usage while roaming is enabled.",
    )
    network_mode_preference: NetworkModePreference = Field(
        default=NetworkModePreference.FOUR_G_5G_PREFERRED,
        validate_default=True,
        description="User's preferred network type (e.g., prefer 4G/5G, use 3G only).",
    )
    active_apn_settings: APNSettings = Field(
        default_factory=APNSettings,
        description="The currently active Access Point Name configuration.",
    )

    # --- Wi-Fi ---
    wifi_enabled: bool = Field(False, description="Whether the Wi-Fi radio is enabled.")
    wifi_connected: bool = Field(
        False,
        description="Whether the device is currently connected to a Wi-Fi network.",
    )
    wifi_ssid: Optional[str] = Field(None, description="The name (SSID) of the connected Wi-Fi network, if any.")
    wifi_signal_strength: SignalStrength = Field(
        default=SignalStrength.NONE,
        validate_default=True,
        description="Strength of the connected Wi-Fi signal.",
    )

    # --- Calling Features ---
    wifi_calling_enabled: bool = Field(False, description="Whether the Wi-Fi Calling feature is enabled.")
    wifi_calling_mms_over_wifi: bool = Field(
        False,
        description="Preference/capability to send/receive MMS over Wi-Fi (depends on carrier and device support).",
    )

    # --- System-Wide Settings ---
    data_saver_mode: bool = Field(
        False,
        description="Whether the system-wide Data Saver mode is enabled to reduce data consumption.",
    )

    # --- VPN ---
    vpn_enabled_setting: bool = Field(
        False,
        description="Whether a VPN profile is configured and potentially set to be 'always on' or manually enabled in settings.",
    )
    vpn_connected: bool = Field(False, description="Whether there currently is an active VPN connection tunnel.")
    vpn_details: Optional[VpnDetails] = Field(None, description="Details about the active VPN connection, if connected.")

    # --- Application State ---
    app_statuses: Dict[str, AppStatus] = Field(
        default_factory=lambda: {
            "messaging": AppStatus(
                app_name="messaging",
                permissions=AppPermissions(sms=True, storage=True, phone=True),
            ),
            "browser": AppStatus(
                app_name="browser",
                permissions=AppPermissions(network=True, storage=True),
            ),
        },
        description="Status of specific applications relevant to troubleshooting (e.g., messaging app, browser).",
    )


def get_device(
    initial_state: Optional[Union[MockPhoneAttributes, Dict[str, Any]]] = None,
):
    """
    Initializes the action handler with a device state.

    Args:
        initial_state: An optional instance of MockPhoneAttributes.
                        If None, a default state is created.
    """

    if initial_state is None:
        return MockPhoneAttributes()
    if isinstance(initial_state, MockPhoneAttributes):
        return initial_state

    # Attempt to load from dict if provided
    device = MockPhoneAttributes()
    try:
        device = update_pydantic_model_with_dict(device, initial_state)
    except pydantic.ValidationError as e:
        print(f"Error validating initial state: {e}")
        print("Initializing with default state instead.")
    return device


class PaymentRequest(BaseModelNoExtra):
    """Represents a payment made by the user."""

    bill_id: str = Field(description="The ID of the bill.")
    amount_due: float = Field(description="The amount of the payment in USD.")
    paid: bool = Field(description="Whether the payment has been made.", default=False)


class UserSurroundings(BaseModelNoExtra):
    """Represents the physical surroundings of the user."""

    name: Optional[str] = Field(None, description="The name of the user.")
    phone_number: Optional[str] = Field(None, description="The phone number of the user.")
    is_abroad: bool = Field(False, description="Whether the user is currently abroad.")
    roaming_allowed: bool = Field(False, description="Whether the user is allowed to roam.")
    signal_strength: dict[NetworkTechnology, SignalStrength] = Field(
        default_factory=lambda: {
            NetworkTechnology.TWO_G: SignalStrength.POOR,
            NetworkTechnology.THREE_G: SignalStrength.FAIR,
            NetworkTechnology.FOUR_G: SignalStrength.GOOD,
            NetworkTechnology.FIVE_G: SignalStrength.EXCELLENT,
        },
        description="Signal strength for each network technology where the user is located.",
    )
    mobile_data_usage_exceeded: bool = Field(False, description="Whether the user has exceeded their data usage limit.")
    line_active: bool = Field(True, description="Whether the user has an active line.")
    payment_request: Optional[PaymentRequest] = Field(None, description="The payment that the agent has requested.")


class TelecomUserDB(BaseModelNoExtra):
    """Database interface for telecom domain."""

    device: MockPhoneAttributes = Field(default_factory=MockPhoneAttributes, description="Mock phone device")
    surroundings: UserSurroundings = Field(default_factory=UserSurroundings, description="User's physical surroundings")

    def update_device(self, update_data: Dict[str, Any]) -> None:
        """Update the mock device state."""
        self.device = update_pydantic_model_with_dict(self.device, update_data)

    def get_hash(self) -> str:
        """Get deterministic hash of user DB."""
        from maseval.benchmark.tau2.utils import get_pydantic_hash

        return get_pydantic_hash(self)
