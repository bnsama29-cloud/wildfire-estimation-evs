# Hardware and Wokwi guide

One sketch, `esp32_sensor/esp32_sensor.ino`, runs on the real board and in Wokwi. The dashboard reads both through `src/serial_source.py`. Only the source chosen in the dashboard sidebar changes.

**Not a safety device.** The smoke value is a relative reading from a low-cost sensor. Use this on a demo rig only.

## What each part does

| Part | Job |
|---|---|
| DHT22 | Temperature and humidity. These replace two dashboard sliders and feed the model. |
| MQ-135 | Smoke and air quality. This drives the alert only, because the model is not trained on smoke. |
| ESP32 | Sends `TEMP,HUMIDITY,SMOKE` every 2 s and lights the alert LED, even with no PC attached. |

Wind, recent rain, month and location stay as dashboard inputs, because the rig has no sensors for them.

## Option A: Wokwi in VS Code (connects to the dashboard)

The browser version of Wokwi cannot reach a dashboard on your PC. The VS Code extension can, through its serial bridge on port 4000.

1. Install the **Wokwi Simulator** and **PlatformIO IDE** extensions in VS Code. Run *Wokwi: Request a New License* from the command palette.
2. Open the folder `hardware/esp32_sensor` in VS Code.
3. Build the firmware: run `pio run` in that folder, or use the PlatformIO Build button.
4. Open `diagram.json` and press the green play button. Keep the simulator tab visible, or it pauses.
5. In another terminal, from the repo root, run `streamlit run src/app.py`.
6. In the dashboard sidebar, choose **Wokwi simulator**.
7. Click the DHT22 and change its temperature and humidity: the dashboard values follow. Click the gas sensor and raise its ppm: the smoke value rises, then the LED and the dashboard alert turn on.

Wokwi has no MQ-135, so `diagram.json` uses its MQ-2 gas sensor as the stand-in. Both give an analog output. Wokwi does not model the voltage divider, so the sensor connects directly to GPIO34 in the simulation. Always use the divider on the real board.

Measured in Wokwi with the gas sensor on 5 V: 0.2 ppm reads 1063, 1 ppm reads 1656, 400 ppm reads 3628, and 69,183 ppm reads 4032. So `diagram.json` starts at 1 ppm as "clean air", and raising the slider to 400 ppm crosses the 2000 alert threshold. These values apply to the simulator only, so calibrate the real MQ-135 separately.

## Option B: Wokwi in the browser (simulation only)

Create a new ESP32 project at https://wokwi.com, then paste in `esp32_sensor.ino` as `sketch.ino` and `diagram.json`, and add **DHT sensor library** in the Library Manager tab. The Serial Monitor shows the packets, but they cannot reach the dashboard.

## Option C: real hardware

Parts: ESP32 DevKit, DHT22, MQ-135 module, 10 kOhm and 20 kOhm resistors, breadboard, jumper wires, USB data cable.

| From | To | Note |
|---|---|---|
| DHT22 VCC | 3V3 | |
| DHT22 DATA | GPIO4 | Add a 10 kOhm pull-up to 3V3 on a bare 4-pin sensor. Modules already have one. |
| DHT22 GND | GND | |
| MQ-135 VCC | VIN (5 V) | The heater needs 5 V. |
| MQ-135 GND | GND | |
| MQ-135 AO | 10 kOhm to GPIO34, then 20 kOhm from GPIO34 to GND | **Do not skip this.** AO can reach 5 V and the ESP32 pin tolerates 3.3 V. |
| MQ-135 DO | not connected | |

Steps:

1. Flash: run `pio run -t upload` in `hardware/esp32_sensor`, or use the Arduino IDE with board *ESP32 Dev Module*.
2. Check: open the serial monitor at 115200 baud and expect a line like `36.4,24.0,1830` every 2 s. Close it afterwards, because only one program can hold the port.
3. Warm up the MQ-135 for several minutes before trusting readings. Its datasheet asks for a much longer first preheat.
4. Calibrate: note the raw value in clean air and near smoke from a safely extinguished match. Set `SMOKE_ALERT_RAW` between them in both files.
5. Run `streamlit run src/app.py`, choose **ESP32 on USB**, and pick the port.
6. Unplug the board. The dashboard must return to the sliders within about 10 seconds.

Do this safely: use a small amount of smoke in a ventilated area with an adult or lab supervisor present, and keep flames away from the rig.

## Serial packet

`TEMP,HUMIDITY,SMOKE` at 115200 baud, one line per reading: degrees C, percent, and the raw 12-bit ADC value (0-4095). The raw value is a relative index, not ppm.
