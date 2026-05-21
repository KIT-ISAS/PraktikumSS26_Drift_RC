#include "comms.h"
#include <WiFi.h>
#include <ArduinoJson.h>

static const uint16_t SERVER_PORT = 12345;
static const char* WIFI_SSIDS[]     = {"JulianCar", "wifi_mid", "wifi_low"};
static const char* WIFI_PASSWORDS[] = {"Hund1234",  "pass_mid", "pass_low"};
static const size_t WIFI_COUNT = sizeof(WIFI_SSIDS) / sizeof(WIFI_SSIDS[0]);

static const IPAddress STATIC_IP(10, 42, 0, 99);
static const IPAddress GATEWAY(10, 42, 0, 1);
static const IPAddress SUBNET(255, 255, 255, 0);

static const unsigned long RECONNECT_INTERVAL_MS = 3000;
static const unsigned long CONNECT_TIMEOUT_MS    = 8000;

static WiFiServer server(SERVER_PORT);
static WiFiClient client;
static String rxBuffer;
static bool hasPendingCommand = false;
static Command pendingCommand;

static bool wifiConnecting = false;
static size_t wifiCandidateIdx = 0;
static unsigned long wifiPhaseStartMs = 0;
static unsigned long lastReconnectAttemptMs = 0;
static bool wasWifiConnected = false;

// Non-blocking: starts a connection attempt or times out and moves to next SSID.
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

static void processLine(const String& line) {
  StaticJsonDocument<256> doc;
  if (deserializeJson(doc, line)) {
    Serial.print("JSON parse error: ");
    Serial.println(line);
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

static void handleServer() {
  if (!client || !client.connected()) {
    client = server.available();
    if (client) {
      rxBuffer = "";
      Serial.println("Peer connected.");
    } else {
      return;
    }
  }

  while (client.available()) {
    char c = client.read();
    if (c == '\r') continue;
    if (c == '\n') {
      String line = rxBuffer;
      rxBuffer = "";
      line.trim();
      if (line.length() > 0) processLine(line);
    } else {
      rxBuffer += c;
      if (rxBuffer.length() > 512) {
        Serial.println("RX overflow, clearing buffer.");
        rxBuffer = "";
      }
    }
  }

  if (!client.connected()) {
    client.stop();
    rxBuffer = "";
    Serial.println("Peer disconnected.");
  }
}

void commsBegin() {
  WiFi.mode(WIFI_STA);
  WiFi.config(STATIC_IP, GATEWAY, SUBNET);
  // Trigger first attempt immediately on the first commsUpdate() call.
  lastReconnectAttemptMs = millis() - RECONNECT_INTERVAL_MS;
}

void commsUpdate() {
  handleWifiReconnect();

  bool isConnected = (WiFi.status() == WL_CONNECTED);

  if (isConnected && !wasWifiConnected) {
    Serial.print("Connected. IP: ");
    Serial.println(WiFi.localIP());
    server.begin();
    Serial.print("TCP server on port ");
    Serial.println(SERVER_PORT);
  }
  wasWifiConnected = isConnected;

  if (isConnected) {
    handleServer();
  } else if (client) {
    client.stop();
    rxBuffer = "";
  }
}

bool commsWifiConnected() {
  return WiFi.status() == WL_CONNECTED;
}

bool commsPeerConnected() {
  return client && client.connected();
}

bool commsTakeCommand(Command& out) {
  if (!hasPendingCommand) return false;
  out = pendingCommand;
  hasPendingCommand = false;
  return true;
}
