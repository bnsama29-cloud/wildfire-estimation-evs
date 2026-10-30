import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from serial_source import SensorReading, SerialSource, parse_packet  # noqa: E402


def test_valid_packet():
    assert parse_packet("36.4,24.0,1830\r\n") == SensorReading(36.4, 24.0, 1830)


def test_malformed_packets_return_none():
    for bad in ["", "36.4,24.0", "a,b,c", "1,2,3,4", "36.4,24.0,"]:
        assert parse_packet(bad) is None


def test_out_of_range_and_nan_rejected():
    assert parse_packet("36.4,140,100") is None  # humidity above 100 %
    assert parse_packet("36.4,24,5000") is None  # beyond 12-bit ADC
    assert parse_packet("nan,24,100") is None  # failed DHT22 read


def test_missing_port_falls_back_to_none():
    assert SerialSource("COM_DOES_NOT_EXIST").latest() is None
