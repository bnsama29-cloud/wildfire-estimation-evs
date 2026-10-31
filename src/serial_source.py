"""Live sensor source: reads TEMP,HUMIDITY,SMOKE lines sent by the ESP32 over serial.

Accepts a real port ("COM3") or a URL such as the Wokwi bridge ("rfc2217://localhost:4000"),
so the dashboard code is identical for hardware and simulation.

The dashboard must work without hardware, so every failure path returns None
and the caller falls back to the sliders.
"""
import time
from dataclasses import dataclass

WOKWI_URL = "rfc2217://localhost:4000"
SMOKE_ALERT_RAW = 2000  # keep equal to SMOKE_ALERT_RAW in hardware/esp32_sensor/esp32_sensor.ino
MAX_READING_AGE_SECONDS = 15.0  # the ESP32 sends every 2 s, but Wokwi can run near 20% speed (about 10 s apart)


@dataclass(frozen=True)
class SensorReading:
    temp_c: float
    rh_percent: float
    smoke_raw: int  # raw 12-bit ADC value (0-4095), not ppm

    @property
    def smoke_alert(self) -> bool:
        return self.smoke_raw >= SMOKE_ALERT_RAW


def parse_packet(line: str) -> SensorReading | None:
    """Parse 'TEMP,HUMIDITY,SMOKE'. Return None for malformed or out-of-range data."""
    parts = line.strip().split(",")
    if len(parts) != 3:
        return None
    try:
        temp, rh, smoke = float(parts[0]), float(parts[1]), int(float(parts[2]))
    except ValueError:
        return None
    # NaN fails every comparison, so a failed DHT22 read is rejected here too
    if not (-40.0 <= temp <= 80.0 and 0.0 <= rh <= 100.0 and 0 <= smoke <= 4095):
        return None
    return SensorReading(temp, rh, smoke)


def available_ports() -> list[str]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [p.device for p in list_ports.comports()]


class SerialSource:
    """Keeps one connection open (opening a real port resets the ESP32) and tracks the newest valid reading."""

    def __init__(self, port: str, baud: int = 115200):
        self.port, self.baud, self._ser = port, baud, None
        self._last, self._last_time = None, 0.0

    def _open(self):
        if self._ser is None:
            import serial  # pyserial, imported lazily so the app runs without it
            self._ser = serial.serial_for_url(self.port, baudrate=self.baud, timeout=0.5)
        return self._ser

    def latest(self, max_age: float = MAX_READING_AGE_SECONDS) -> SensorReading | None:
        """Return the newest reading, or None if nothing valid arrived within max_age seconds."""
        try:
            ser = self._open()
            while ser.in_waiting:
                reading = parse_packet(ser.readline().decode("ascii", errors="ignore"))
                if reading is not None:
                    self._last, self._last_time = reading, time.monotonic()
        except Exception:  # missing pyserial, unplugged board, simulator stopped: fall back to sliders
            self.close()
            return None
        return self._last if time.monotonic() - self._last_time <= max_age else None

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None
