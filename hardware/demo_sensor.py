"""Scripted stand-in for the ESP32, for demos with no board and no Wokwi.

Serves the same RFC2217 serial bridge as Wokwi on port 4000 and plays a fire-weather
scenario, so the dashboard reads it through its unchanged Wokwi source.
The values are made up: this shows how the system behaves, not a measurement.

Run: python hardware/demo_sensor.py
"""
from __future__ import annotations

import argparse
import itertools
import socket
import threading
import time
from pathlib import Path

import numpy as np
import serial
from serial import rfc2217

PERIOD_SECONDS = 2.0  # same as PERIOD_MS in esp32_sensor.ino
# seconds into the scenario, temp C, humidity %, raw smoke
SCENARIO = np.array([
    (0, 28.0, 65.0, 1200),
    (20, 28.0, 65.0, 1200),   # calm, humid air
    (50, 38.0, 35.0, 1300),   # the day heats up
    (70, 44.0, 8.0, 1500),    # very hot, very dry air
    (90, 44.0, 8.0, 2600),    # smoke reaches the sensor and the alert turns on
    (110, 44.0, 8.0, 2600),
    (140, 28.0, 65.0, 1200),  # rain and recovery
])
SCENARIO_SECONDS = SCENARIO[-1, 0]


def packet_at(seconds: float) -> str:
    """The TEMP,HUMIDITY,SMOKE line the ESP32 would send at this point of the scenario."""
    moment = seconds % SCENARIO_SECONDS
    temp, rh, smoke = (np.interp(moment, SCENARIO[:, 0], SCENARIO[:, column]) for column in (1, 2, 3))
    return f"{temp:.1f},{rh:.1f},{smoke:.0f}"


class _Connection:
    """The small socket interface that pyserial's PortManager writes to."""

    def __init__(self, sock: socket.socket):
        self.sock = sock

    def write(self, data: bytes) -> None:
        self.sock.sendall(data)


def stream(client: socket.socket, period: float = PERIOD_SECONDS, log_path: str | None = None) -> None:
    connection = _Connection(client)
    try:
        manager = rfc2217.PortManager(serial.serial_for_url("loop://", baudrate=115200, timeout=0.1), connection)
    except OSError:  # the client left during the handshake
        return

    def answer_negotiation() -> None:
        try:
            while data := client.recv(1024):
                list(manager.filter(data))
        except OSError:
            return

    threading.Thread(target=answer_negotiation, daemon=True).start()
    started = time.monotonic()
    for _ in itertools.count():
        line = packet_at(time.monotonic() - started)
        try:
            connection.write(b"".join(manager.escape(line.encode("ascii") + b"\r\n")))
        except OSError:
            return
        print(line, flush=True)
        if log_path:
            with Path(log_path).open("a", encoding="ascii") as log:
                log.write(f"{time.time():.3f},{line}\n")
        time.sleep(period)


def serve(server: socket.socket, period: float = PERIOD_SECONDS, log_path: str | None = None) -> None:
    while True:
        client, _ = server.accept()
        threading.Thread(target=stream, args=(client, period, log_path), daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4000)
    parser.add_argument("--log", help="also write each packet and the time it was sent to this file")
    args = parser.parse_args()
    server = socket.create_server(("127.0.0.1", args.port))
    print(f"Demo sensor on port {args.port}. In the dashboard set Live source to Wokwi.", flush=True)
    serve(server, log_path=args.log)


if __name__ == "__main__":
    main()
