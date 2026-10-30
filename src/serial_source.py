"""Live sensor source: reads TEMP,HUMIDITY,SMOKE lines sent by the ESP32 over USB serial.

The dashboard must work without hardware, so every failure path returns None
and the caller falls back to the sliders.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SensorReading:
    temp_c: float
    rh_percent: float
    smoke_raw: int  # raw 12-bit ADC value (0-4095), not ppm


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
    """Keeps one serial port open (opening it resets the ESP32) and returns the newest valid reading."""

    def __init__(self, port: str, baud: int = 115200):
        self.port, self.baud, self._ser = port, baud, None

    def _open(self):
        if self._ser is None:
            import serial  # pyserial, imported lazily so the app runs without it
            self._ser = serial.Serial(self.port, self.baud, timeout=0.5)
        return self._ser

    def latest(self) -> SensorReading | None:
        try:
            ser = self._open()
            newest = None
            while ser.in_waiting:
                reading = parse_packet(ser.readline().decode("ascii", errors="ignore"))
                if reading is not None:
                    newest = reading
            return newest
        except Exception:  # missing pyserial, unplugged board, port busy: fall back to sliders
            self.close()
            return None

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None
