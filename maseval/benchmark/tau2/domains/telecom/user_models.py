"""Tau 2 Benchmark - Telecom Domain User Models.

Pydantic models for the telecom user device and surroundings.

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Adapted from: src/tau2/domains/telecom/user_data_model.py
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Enums
# =============================================================================


class SimStatus(str, Enum):
    """Status of the SIM card."""

    ACTIVE = "active"
    MISSING = "missing"
    LOCKED_PIN = "locked_pin"
    LOCKED_PUK = "locked_puk"


class NetworkTechnology(str, Enum):
    """Network technology type."""

    NONE = "none"
    TWO_G = "2g"
    THREE_G = "3g"
    FOUR_G = "4g"
    FIVE_G = "5g"


class NetworkModePreference(str, Enum):
    """User preference for network mode."""

    FOUR_G_5G_PREFERRED = "4g_5g_preferred"
    FOUR_G_ONLY = "4g_only"
    THREE_G_ONLY = "3g_only"
    TWO_G_ONLY = "2g_only"


class SignalStrength(str, Enum):
    """Signal strength level."""

    NONE = "none"
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class PerformanceLevel(str, Enum):
    """Performance level of a service."""

    UNKNOWN = "unknown"
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class NetworkStatus(str, Enum):
    """Current network connection status."""

    CONNECTED = "connected"
    SEARCHING = "searching"
    NO_SERVICE = "no_service"
    EMERGENCY_ONLY = "emergency_only"


class APNNames(str, Enum):
    """APN configuration names."""

    INTERNET = "internet"
    BROKEN = "broken"


# =============================================================================
# Configuration Models
# =============================================================================


class APNSettings(BaseModel):
    """Access Point Name (APN) settings."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Name of the APN")
    apn: str = Field(description="APN address")
    proxy: str = Field("", description="Proxy server address")
    port: str = Field("", description="Proxy server port")
    username: str = Field("", description="Username for APN authentication")
    password: str = Field("", description="Password for APN authentication")
    server: str = Field("", description="Server address")
    mmsc_url: str = Field("", description="MMSC URL for MMS")
    mms_proxy: str = Field("", description="MMS proxy address")
    mms_port: str = Field("", description="MMS proxy port")
    mcc: str = Field("", description="Mobile Country Code")
    mnc: str = Field("", description="Mobile Network Code")
    auth_type: str = Field("", description="Authentication type")
    apn_type: str = Field("", description="APN type")
    apn_protocol: str = Field("IPv4", description="APN protocol")
    apn_roaming_protocol: str = Field("IPv4", description="APN roaming protocol")
    bearer: str = Field("", description="Bearer type")
    mvno_type: str = Field("", description="MVNO type")
    reset_at_reboot: bool = Field(False, description="Whether settings reset at reboot")


class VpnDetails(BaseModel):
    """VPN connection details."""

    model_config = ConfigDict(extra="forbid")

    server_address: str = Field(description="VPN server address")
    protocol: str = Field(description="VPN protocol (e.g., IKEv2, OpenVPN)")
    server_performance: PerformanceLevel = Field(PerformanceLevel.UNKNOWN, description="VPN server performance")


class AppPermissions(BaseModel):
    """Permissions granted to an application."""

    model_config = ConfigDict(extra="forbid")

    sms: bool = Field(False, description="Access to SMS")
    storage: bool = Field(False, description="Access to storage")
    phone: bool = Field(False, description="Access to phone calls")
    network: bool = Field(False, description="Access to network")


class AppStatus(BaseModel):
    """Status and configuration of an installed application."""

    model_config = ConfigDict(extra="forbid")

    app_name: str = Field(description="Name of the application")
    permissions: AppPermissions = Field(default_factory=AppPermissions, description="Permissions granted to the app")
    is_running: bool = Field(False, description="Whether the app is currently running")
    data_usage_mb: float = Field(0.0, description="Data used by the app in MB")


# =============================================================================
# Device State
# =============================================================================


class MockPhoneAttributes(BaseModel):
    """State of the user's simulated phone device."""

    model_config = ConfigDict(extra="forbid")

    # Power & System
    is_on: bool = Field(True, description="Whether the device is powered on")
    airplane_mode: bool = Field(False, description="Whether airplane mode is enabled")
    battery_level: int = Field(100, description="Battery level percentage (0-100)")

    # SIM & Network
    sim_status: SimStatus = Field(SimStatus.ACTIVE, description="Status of the SIM card")
    sim_pin: str = Field("1234", description="SIM PIN code")
    sim_puk: str = Field("12345678", description="SIM PUK code")
    sim_attempts_remaining: int = Field(3, description="Remaining SIM PIN attempts")
    network_status: NetworkStatus = Field(NetworkStatus.CONNECTED, description="Network connection status")
    network_technology: NetworkTechnology = Field(NetworkTechnology.FOUR_G, description="Current network technology")
    network_mode_preference: NetworkModePreference = Field(NetworkModePreference.FOUR_G_5G_PREFERRED, description="Preferred network mode")

    # Data & Roaming
    mobile_data_enabled: bool = Field(True, description="Whether mobile data is enabled")
    roaming_enabled: bool = Field(False, description="Whether data roaming is enabled")
    data_saver_mode: bool = Field(False, description="Whether data saver mode is on")

    # WiFi & Calling
    wifi_enabled: bool = Field(True, description="Whether Wi-Fi is enabled")
    wifi_connected: bool = Field(True, description="Whether connected to a Wi-Fi network")
    wifi_calling_enabled: bool = Field(False, description="Whether Wi-Fi calling is enabled")

    # Configuration
    apn_settings: APNSettings = Field(
        default_factory=lambda: APNSettings(name=APNNames.INTERNET.value, apn="internet"),
        description="Current APN settings",
    )
    vpn_status: bool = Field(False, description="Whether VPN is connected")
    vpn_details: Optional[VpnDetails] = Field(None, description="Active VPN connection details")

    # Applications - default apps match original tau2 benchmark
    installed_apps: Dict[str, AppStatus] = Field(
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
        description="Installed applications",
    )

    # Hardware
    has_sim_card: bool = Field(True, description="Whether a physical SIM card is inserted")


# =============================================================================
# User Environment
# =============================================================================


class PaymentRequest(BaseModel):
    """A payment request received by the user."""

    model_config = ConfigDict(extra="forbid")

    bill_id: str = Field(description="ID of the bill to pay")
    amount_due: float = Field(description="Amount to pay")
    paid: bool = Field(False, description="Whether the payment has been made")


class UserSurroundings(BaseModel):
    """The user's physical environment/context."""

    model_config = ConfigDict(extra="forbid")

    is_abroad: bool = Field(False, description="Whether the user is currently abroad")
    roaming_allowed_in_location: bool = Field(True, description="Whether roaming is supported in current location")
    signal_strength: SignalStrength = Field(SignalStrength.GOOD, description="Signal strength in current location")
    available_technologies: List[NetworkTechnology] = Field(
        default_factory=lambda: [NetworkTechnology.FOUR_G, NetworkTechnology.THREE_G],
        description="Network technologies available in current location",
    )
    wifi_networks_available: List[str] = Field(
        default_factory=lambda: ["Home_WiFi", "Starbucks_WiFi"], description="List of available Wi-Fi networks"
    )
    payment_requests: List[PaymentRequest] = Field(default_factory=list, description="Pending payment requests")


class TelecomUserDB(BaseModel):
    """Database for user-side telecom state."""

    model_config = ConfigDict(extra="forbid")

    device: MockPhoneAttributes = Field(default_factory=MockPhoneAttributes, description="User's phone state")
    surroundings: UserSurroundings = Field(default_factory=UserSurroundings, description="User's environment state")

    def get_hash(self) -> str:
        """Get deterministic hash of user DB."""
        # This will be called by parent DB's get_hash if included there
        from maseval.benchmark.tau2.utils import get_pydantic_hash

        return get_pydantic_hash(self)
