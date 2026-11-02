import socket
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "hardware")]

from demo_sensor import SCENARIO_SECONDS, apply_commands, packet_at, serve  # noqa: E402
from collars import matches, set_command  # noqa: E402
from serial_source import SerialSource, parse_packet  # noqa: E402


def test_scenario_starts_calm_with_no_alert():
    assert packet_at(0) == "28.0,65.0,1200"
    assert not parse_packet(packet_at(0)).smoke_alert


def test_scenario_reaches_hot_dry_air_and_then_the_smoke_alert():
    assert packet_at(70) == "44.0,8.0,1500"
    assert not parse_packet(packet_at(70)).smoke_alert
    assert packet_at(100) == "44.0,8.0,2600"
    assert parse_packet(packet_at(100)).smoke_alert


def test_scenario_loops():
    assert packet_at(SCENARIO_SECONDS + 100) == packet_at(100)


def test_every_packet_passes_the_dashboard_parser():
    assert all(parse_packet(packet_at(second)) is not None for second in range(int(SCENARIO_SECONDS)))


def test_set_overrides_the_scenario_and_auto_returns_to_it():
    assert apply_commands(b"SET,44.0,8.0,2600\n", None) == "44.0,8.0,2600"
    assert apply_commands(b"AUTO\n", "44.0,8.0,2600") is None
    assert apply_commands(b"noise\n", "44.0,8.0,2600") == "44.0,8.0,2600"


def test_a_command_sent_by_the_dashboard_comes_back_as_a_sensor_reading():
    server = socket.create_server(("127.0.0.1", 0))
    threading.Thread(target=serve, args=(server, 0.1), daemon=True).start()
    source = SerialSource(f"rfc2217://localhost:{server.getsockname()[1]}")
    command = set_command(44.0, 8.0, 2600)
    deadline = time.monotonic() + 15
    while not matches(source.latest(), 44.0, 8.0, 2600) and time.monotonic() < deadline:
        source.send(command)
        time.sleep(0.2)
    reading = source.latest()
    source.close()
    server.close()
    assert matches(reading, 44.0, 8.0, 2600)
    assert reading.smoke_alert


def test_serial_source_reads_the_bridge_like_it_reads_wokwi():
    server = socket.create_server(("127.0.0.1", 0))
    port = server.getsockname()[1]
    threading.Thread(target=serve, args=(server, 0.1), daemon=True).start()
    source = SerialSource(f"rfc2217://localhost:{port}")
    deadline = time.monotonic() + 15
    reading = None
    while reading is None and time.monotonic() < deadline:
        reading = source.latest()
        time.sleep(0.1)
    source.close()
    server.close()
    assert reading is not None
    assert (reading.temp_c, reading.rh_percent, reading.smoke_raw) == pytest.approx((28.0, 65.0, 1200))
