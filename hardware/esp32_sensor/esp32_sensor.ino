// ESP32 + DHT22 + MQ-135 -> USB serial packet "TEMP,HUMIDITY,SMOKE"
// The same sketch runs on the real board and in Wokwi (see hardware/README.md).
// Libraries: "DHT sensor library" and "Adafruit Unified Sensor" (both by Adafruit).
// Board: "ESP32 Dev Module", serial 115200 baud.
//
// Real wiring:
//   DHT22 DATA  -> GPIO4   (VCC -> 3V3, GND -> GND, 10k pull-up DATA->3V3 if you use the bare 4-pin sensor)
//   MQ-135 AO   -> 10k -> GPIO34 -> 20k -> GND   (the ESP32 ADC tolerates 3.3 V max; MQ-135 AO can reach 5 V)
//   MQ-135 VCC  -> VIN (5 V), GND -> GND
//   Alert LED   -> GPIO2 (on-board LED on most DevKit boards)

#include <DHT.h>

constexpr uint8_t DHT_PIN = 4;
constexpr uint8_t MQ_PIN = 34;  // ADC1 channel, input only
constexpr uint8_t ALERT_LED_PIN = 2;
constexpr unsigned long PERIOD_MS = 2000;  // DHT22 can be read at most every ~2 s
constexpr int SMOKE_ALERT_RAW = 2000;  // keep equal to SMOKE_ALERT_RAW in src/serial_source.py; calibrate on the rig

DHT dht(DHT_PIN, DHT22);

void setup() {
  Serial.begin(115200);
  pinMode(ALERT_LED_PIN, OUTPUT);
  dht.begin();
  analogReadResolution(12);  // 0..4095
}

int readSmokeRaw() {
  long sum = 0;
  for (int i = 0; i < 16; i++) {
    sum += analogRead(MQ_PIN);
    delay(5);
  }
  return sum / 16;
}

void loop() {
  static unsigned long last = 0;
  if (millis() - last < PERIOD_MS) {
    delay(10);  // idling instead of spinning took Wokwi from about 20% to real-time speed
    return;
  }
  last = millis();

  int smoke = readSmokeRaw();
  digitalWrite(ALERT_LED_PIN, smoke >= SMOKE_ALERT_RAW ? HIGH : LOW);  // works without a PC attached

  float t = dht.readTemperature();
  float h = dht.readHumidity();
  if (isnan(t) || isnan(h)) return;  // skip failed reads; the Python side also rejects bad lines

  Serial.print(t, 1);
  Serial.print(',');
  Serial.print(h, 1);
  Serial.print(',');
  Serial.println(smoke);
}
