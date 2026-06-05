from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Subsystem(str, Enum):
    EPS = "eps"            # Electrical Power System
    TCS = "tcs"            # Thermal Control System
    ADCS = "adcs"          # Attitude Determination & Control
    COMMS = "comms"        # Communications
    PAYLOAD = "payload"
    OBDH = "obdh"          # On-Board Data Handling
    PROP = "prop"          # Propulsion
    GNC = "gnc"            # Guidance, Navigation & Control
    UNKNOWN = "unknown"


class Units(str, Enum):
    VOLT = "V"
    AMP = "A"
    WATT = "W"
    CELSIUS = "degC"
    KELVIN = "K"
    NEWTON_METER = "Nm"
    RADIAN = "rad"
    RADIAN_PER_SEC = "rad/s"
    METER = "m"
    METER_PER_SEC = "m/s"
    PERCENT = "pct"
    COUNT = "count"
    BIT = "bit"
    DIMENSIONLESS = ""


@dataclass(frozen=True, slots=True)
class Channel:
    """Metadata for one telemetry channel."""

    name: str
    units: Units
    subsystem: Subsystem
    sample_rate_hz: float
    description: str = ""
