import sys
from pathlib import Path

import serial

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from serial_source import SMOKE_ALERT_RAW, SensorReading, SerialSource, parse_packet  # noqa: E402


def test_valid_packet():
    assert parse_packet("36.4,24.0,1830\r\n") == SensorReading(36.4, 24.0, 1830)


def test_malformed_packets_return_none():
    for bad in ["", "36.4,24.0", "a,b,c", "1,2,3,4", "36.4,24.0,"]:
        assert parse_packet(bad) is None


def test_out_of_range_and_nan_rejected():
    assert parse_packet("36.4,140,100") is None  # humidity above 100 %
    assert parse_packet("36.4,24,5000") is None  # beyond 12-bit ADC
    assert parse_packet("nan,24,100") is None  # failed DHT22 read


def test_smoke_alert_threshold():
    assert SensorReading(30.0, 40.0, SMOKE_ALERT_RAW).smoke_alert
    assert not SensorReading(30.0, 40.0, SMOKE_ALERT_RAW - 1).smoke_alert


def test_missing_port_falls_back_to_none():
    assert SerialSource("COM_DOES_NOT_EXIST").latest() is None


def loopback_source(monkeypatch):
    """A SerialSource connected to pyserial's in-memory loopback, which echoes what is written."""
    source = SerialSource("loop://")
    port = serial.serial_for_url("loop://", timeout=0.1)
    monkeypatch.setattr(source, "_ser", port)
    return source, port


def test_latest_returns_newest_valid_line_and_skips_garbage(monkeypatch):
    source, port = loopback_source(monkeypatch)
    port.write(b"30.0,40.0,100\nnoise\n31.5,38.0,250\n")
    assert source.latest() == SensorReading(31.5, 38.0, 250)


def test_latest_keeps_recent_reading_then_expires_it(monkeypatch):
    source, port = loopback_source(monkeypatch)
    port.write(b"30.0,40.0,100\n")
    assert source.latest() == SensorReading(30.0, 40.0, 100)
    assert source.latest() == SensorReading(30.0, 40.0, 100)  # no new line yet, still fresh
    assert source.latest(max_age=-1.0) is None  # stale, so the app falls back to sliders
