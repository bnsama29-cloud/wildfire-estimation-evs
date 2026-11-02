"""Simulated animal collars for the demo. No real animal carries this hardware.

The one ESP32 stands in for whichever collar is selected: dashboard sliders send it a
SET command and it prints those values on the serial link as its sensor packet.
"""
from __future__ import annotations

from dataclasses import dataclass

from serial_source import SMOKE_ALERT_RAW, SensorReading

STATUS_COLOURS = {"OK": [64, 214, 140, 255], "DRY": [255, 186, 73, 255], "SMOKE": [255, 45, 85, 255]}


@dataclass(frozen=True)
class Collar:
    name: str
    site: str
    lat: float
    lon: float
    temp_c: float
    rh_percent: float
    smoke_raw: int


COLLARS = [
    Collar("Elephant E1", "Similipal, Odisha", 21.85, 86.35, 36.0, 24.0, 1200),
    Collar("Tiger T4", "Bandipur, Karnataka", 11.66, 76.63, 31.0, 48.0, 1100),
    Collar("Elephant E7", "Corbett, Uttarakhand", 29.53, 78.77, 33.0, 40.0, 1150),
    Collar("Gaur G2", "Gadchiroli, Maharashtra", 19.80, 80.00, 38.0, 22.0, 1250),
    Collar("Clouded leopard L3", "Mamit, Mizoram", 23.93, 92.48, 27.0, 62.0, 1050),
]
BY_NAME = {collar.name: collar for collar in COLLARS}


def set_command(temp_c: float, rh_percent: float, smoke_raw: int) -> str:
    """The line the sketch parses to override its sensors."""
    return f"SET,{temp_c:.1f},{rh_percent:.1f},{smoke_raw:d}"


def matches(reading: SensorReading | None, temp_c: float, rh_percent: float, smoke_raw: int) -> bool:
    """Whether the ESP32 is already reporting these values."""
    return reading is not None and (reading.temp_c, reading.rh_percent, reading.smoke_raw) == (
        round(temp_c, 1), round(rh_percent, 1), smoke_raw)


def collar_status(reading: SensorReading, fire_weather_band: str) -> str:
    """Smoke outranks dry air, because smoke means something may already be burning."""
    if reading.smoke_raw >= SMOKE_ALERT_RAW:
        return "SMOKE"
    return "DRY" if fire_weather_band == "HIGH" else "OK"
