#include "comms.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>

static const uint16_t LOCAL_PORT = 12345;
static const char* WIFI_SSIDS[]     = {"JulianCar", "wifi_mid", "wifi_low"};
static const char* WIFI_PASSWORDS[] = {"Hund1234",  "pass_mid", "pass_low"};
static const size_t WIFI_COUNT = sizeof(WIFI_SSIDS) / sizeof(WIFI_SSIDS[0]);

static const IPAddress STATIC_IP(10, 42, 0, 99);
static const IPAddress GATEWAY(10, 42, 0, 1);
static const IPAddress SUBNET(255, 255, 255, 0);

static const unsigned long RECONNECT_INTERVAL_MS = 3000;
static const unsigned long CONNECT_TIMEOUT_MS    = 8000;
static const unsigned long STATUS_INTERVAL_MS    = 100;

static WiFiUDP udp;
static bool wasWifiConnected = false;
static bool wifiConnecting = false;
static size_t wifiCandidateIdx = 0;
static unsigned long wifiPhaseStartMs = 0;
static unsigned long lastReconnectAttemptMs = 0;

static IPAddress senderIp;
static uint16_t  senderPort = 0;
static bool      hasSender  = false;
static unsigned long lastStatusMs = 0;

static bool hasPendingCommand = false;
static Command pendingCommand;

static char rxBuf[512];

static void handleWifiReconnect() {
  unsigned long now = millis();

  if (WiFi.status() == WL_CONNECTED) {
    wifiConnecting = false;
    return;
  }

  if (wifiConnecting) {
    if (now - wifiPhaseStartMs >= CONNECT_TIMEOUT_MS) {
      WiFi.disconnect(true);
      wifiConnecting = false;
      wifiCandidateIdx = (wifiCandidateIdx + 1) % WIFI_COUNT;
      lastReconnectAttemptMs = now;
    }
    return;
  }

  if (now - lastReconnectAttemptMs >= RECONNECT_INTERVAL_MS) {
    lastReconnectAttemptMs = now;
    wifiPhaseStartMs = now;
    wifiConnecting = true;
    Serial.print("Trying ");
    Serial.println(WIFI_SSIDS[wifiCandidateIdx]);
    WiFi.begin(WIFI_SSIDS[wifiCandidateIdx], WIFI_PASSWORDS[wifiCandidateIdx]);
  }
}

static bool parseGyroMode(const char* raw, GyroMode& mode) {
  if (strcmp(raw, "hard") == 0) { mode = GyroMode::Hard; return true; }
  if (strcmp(raw, "soft") == 0) { mode = GyroMode::Soft; return true; }
  return false;
}

static void processPacket(const char* data, int len) {
  StaticJsonDocument<256> doc;
  if (deserializeJson(doc, data, len)) {
    Serial.println("JSON parse error.");
    return;
  }

  const char* modeRaw = doc["gyro_mode"] | "soft";
  GyroMode mode;
  if (!parseGyroMode(modeRaw, mode)) {
    Serial.println("Invalid gyro_mode.");
    return;
  }

  pendingCommand.steering = doc["steering"]  | 0.0f;
  pendingCommand.throttle = doc["throttle"]  | 0.0f;
  pendingCommand.gyroMode = mode;
  pendingCommand.gyroGain = doc["gyro_gain"] | 0.0f;
  pendingCommand.seq      = doc["seq"]       | 0u;
  hasPendingCommand = true;
}

void commsBegin() {
  WiFi.mode(WIFI_STA);
  WiFi.config(STATIC_IP, GATEWAY, SUBNET);
  lastReconnectAttemptMs = millis() - RECONNECT_INTERVAL_MS;
}

void commsUpdate() {
  handleWifiReconnect();

  bool isConnected = (WiFi.status() == WL_CONNECTED);

  if (isConnected && !wasWifiConnected) {
    Serial.print("Connected. IP: ");
    Serial.println(WiFi.localIP());
    udp.begin(LOCAL_PORT);
    Serial.print("UDP listening on port ");
    Serial.println(LOCAL_PORT);
  }
  wasWifiConnected = isConnected;

  if (!isConnected) return;

  int packetSize;
  while ((packetSize = udp.parsePacket()) > 0) {
    if (packetSize >= (int)sizeof(rxBuf)) {
      udp.flush();
      Serial.println("Packet too large, discarded.");
      continue;
    }
    int len = udp.read(rxBuf, sizeof(rxBuf) - 1);
    rxBuf[len] = '\0';
    senderIp   = udp.remoteIP();
    senderPort = udp.remotePort();
    hasSender  = true;
    processPacket(rxBuf, len);
  }
}

bool commsWifiConnected() {
  return WiFi.status() == WL_CONNECTED;
}

bool commsTakeCommand(Command& out) {
  if (!hasPendingCommand) return false;
  out = pendingCommand;
  hasPendingCommand = false;
  return true;
}

void commsSendStatus(const char* state, bool armed, uint32_t lastSeq) {
  if (!hasSender || WiFi.status() != WL_CONNECTED) return;
  unsigned long now = millis();
  if (now - lastStatusMs < STATUS_INTERVAL_MS) return;
  lastStatusMs = now;

  StaticJsonDocument<128> doc;
  doc["state"]    = state;
  doc["armed"]    = armed;
  doc["last_seq"] = lastSeq;

  char buf[128];
  serializeJson(doc, buf, sizeof(buf));

  udp.beginPacket(senderIp, senderPort);
  udp.write((const uint8_t*)buf, strlen(buf));
  udp.endPacket();
}
