// ESP32 + DHT22 + MQ-135 -> USB serial packet "TEMP,HUMIDITY,SMOKE"
// Libraries (Arduino Library Manager): "DHT sensor library" and "Adafruit Unified Sensor" (both by Adafruit).
// Board: "ESP32 Dev Module", serial monitor 115200 baud.
//
// Wiring (see README):
//   DHT22 DATA  -> GPIO4   (VCC -> 3V3, GND -> GND, 10k pull-up DATA->3V3 if you use the bare 4-pin sensor)
//   MQ-135 AO   -> 10k -> GPIO34 -> 20k -> GND   (the ESP32 ADC tolerates 3.3 V max; MQ-135 AO can reach 5 V)
//   MQ-135 VCC  -> VIN (5 V), GND -> GND

#include <DHT.h>

constexpr uint8_t DHT_PIN = 4;
constexpr uint8_t MQ_PIN = 34;  // ADC1 channel, input only
constexpr unsigned long PERIOD_MS = 2000;  // DHT22 can be read at most every ~2 s

DHT dht(DHT_PIN, DHT22);

void setup() {
  Serial.begin(115200);
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
  if (millis() - last < PERIOD_MS) return;
  last = millis();

  float t = dht.readTemperature();
  float h = dht.readHumidity();
  if (isnan(t) || isnan(h)) return;  // skip failed reads; the Python side also rejects bad lines

  Serial.print(t, 1);
  Serial.print(',');
  Serial.print(h, 1);
  Serial.print(',');
  Serial.println(readSmokeRaw());
}
