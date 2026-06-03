#include "comms.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>

static const uint16_t LOCAL_PORT = 12345;
static const char* WIFI_SSIDS[]     = {"JulianCar", "louis-Inspiron-7400", "wifi_low"};
static const char* WIFI_PASSWORDS[] = {"Hund1234",  "kinect360", "pass_low"};
static const size_t WIFI_COUNT = sizeof(WIFI_SSIDS) / sizeof(WIFI_SSIDS[0]);

static const IPAddress STATIC_IP(10, 42, 0, 99);
static const IPAddress GATEWAY(10, 42, 0, 1);
static const IPAddress SUBNET(255, 255, 255, 0);

static const unsigned long SCAN_START_DELAY_MS  = 3000;
static const unsigned long SCAN_TIMEOUT_MS      = 8000;
static const unsigned long CONNECT_TIMEOUT_MS   = 10000;
static const unsigned long STATUS_INTERVAL_MS   = 100;
static const unsigned long PEER_STALE_MS        = 2000;

enum class WifiPhase { Idle, Scanning, Connecting };

static WiFiUDP udp;
static bool wasWifiConnected = false;
static WifiPhase wifiPhase = WifiPhase::Idle;
static unsigned long wifiPhaseStartMs = 0;
static unsigned long lastReconnectAttemptMs = 0;
static int8_t connectingIdx = -1;

static IPAddress senderIp;
static uint16_t  senderPort = 0;
static unsigned long lastStatusMs = 0;

static bool hasPendingCommand = false;
static Command pendingCommand;
static uint32_t lastAcceptedSeq = 0;
static bool seenAnySeq = false;
static uint32_t currentSessionId = 0;
static bool seenAnySession = false;
static unsigned long lastValidPacketMs = 0;

static char rxBuf[512];

static int findBestKnownNetwork() {
  int n = WiFi.scanComplete();
  if (n < 0) return -1;
  int bestIdx = -1;
  int32_t bestRssi = INT32_MIN;
  for (int i = 0; i < n; i++) {
    for (size_t k = 0; k < WIFI_COUNT; k++) {
      if (WiFi.SSID(i) == WIFI_SSIDS[k] && WiFi.RSSI(i) > bestRssi) {
        bestRssi = WiFi.RSSI(i);
        bestIdx = (int)k;
      }
    }
  }
  return bestIdx;
}

static void handleWifiReconnect(bool isConnected) {
  if (isConnected) {
    if (wifiPhase != WifiPhase::Idle) {
      WiFi.scanDelete();
      wifiPhase = WifiPhase::Idle;
    }
    return;
  }

  unsigned long now = millis();
  switch (wifiPhase) {
    case WifiPhase::Idle:
      if (now - lastReconnectAttemptMs >= SCAN_START_DELAY_MS) {
        WiFi.scanNetworks(/*async=*/true);
        wifiPhaseStartMs = now;
        wifiPhase = WifiPhase::Scanning;
        Serial.println("WiFi scan started.");
      }
      break;

    case WifiPhase::Scanning: {
      int bestIdx = findBestKnownNetwork();
      if (bestIdx >= 0) {
        WiFi.scanDelete();
        connectingIdx = (int8_t)bestIdx;
        Serial.print("Connecting to best network: ");
        Serial.println(WIFI_SSIDS[connectingIdx]);
        WiFi.begin(WIFI_SSIDS[connectingIdx], WIFI_PASSWORDS[connectingIdx]);
        wifiPhaseStartMs = now;
        wifiPhase = WifiPhase::Connecting;
      } else if (WiFi.scanComplete() == WIFI_SCAN_FAILED ||
                 now - wifiPhaseStartMs >= SCAN_TIMEOUT_MS) {
        WiFi.scanDelete();
        lastReconnectAttemptMs = now;
        wifiPhase = WifiPhase::Idle;
        Serial.println("Scan complete — no known networks found.");
      }
      break;
    }

    case WifiPhase::Connecting:
      if (now - wifiPhaseStartMs >= CONNECT_TIMEOUT_MS) {
        WiFi.disconnect(false);
        lastReconnectAttemptMs = now;
        wifiPhase = WifiPhase::Idle;
        Serial.println("Connection timeout, will rescan.");
      }
      break;
  }
}

static bool parseGyroMode(const char* raw, GyroMode& mode) {
  if (strcmp(raw, "hard") == 0) { mode = GyroMode::Hard; return true; }
  if (strcmp(raw, "soft") == 0) { mode = GyroMode::Soft; return true; }
  return false;
}

static bool processPacket(const char* data, int len) {
  StaticJsonDocument<256> doc;
  if (deserializeJson(doc, data, len)) {
    Serial.println("JSON parse error.");
    return false;
  }

  const char* modeRaw = doc["gyro_mode"] | "soft";
  GyroMode mode;
  if (!parseGyroMode(modeRaw, mode)) {
    Serial.println("Invalid gyro_mode.");
    return false;
  }

  uint32_t sessionId = doc["session_id"] | 0u;
  if (!seenAnySession || sessionId != currentSessionId) {
    if (seenAnySession) {
      Serial.print("Session changed, resetting seq window. New session_id: ");
      Serial.println(sessionId);
    }
    currentSessionId = sessionId;
    seenAnySession = true;
    seenAnySeq = false;
    lastAcceptedSeq = 0;
  }

  uint32_t seq = doc["seq"] | 0u;
  if (seenAnySeq && seq <= lastAcceptedSeq) {
    return false;  // stale or duplicate within the current session, ignore
  }

  pendingCommand.steering = doc["steering"]  | 0.0f;
  pendingCommand.throttle = doc["throttle"]  | 0.0f;
  pendingCommand.gyroMode = mode;
  pendingCommand.gyroGain = doc["gyro_gain"] | 0.0f;
  pendingCommand.seq      = seq;
  hasPendingCommand = true;
  lastAcceptedSeq = seq;
  seenAnySeq = true;
  lastValidPacketMs = millis();
  return true;
}

static void expireStalePeer() {
  if (senderPort == 0) return;
  if ((millis() - lastValidPacketMs) < PEER_STALE_MS) return;
  Serial.println("Peer stale, clearing status target.");
  senderPort = 0;
}

void commsBegin() {
  WiFi.mode(WIFI_STA);
  WiFi.config(STATIC_IP, GATEWAY, SUBNET);
  WiFi.setAutoReconnect(false);
  lastReconnectAttemptMs = millis() - SCAN_START_DELAY_MS;
}

void commsUpdate() {
  bool isConnected = (WiFi.status() == WL_CONNECTED);
  handleWifiReconnect(isConnected);

  if (wasWifiConnected && !isConnected) {
    udp.stop();
    senderPort = 0;
    Serial.println("WiFi lost. UDP stopped.");
  }
  if (isConnected && !wasWifiConnected) {
    Serial.print("WiFi connected. IP: ");
    Serial.println(WiFi.localIP());
    udp.begin(LOCAL_PORT);
    Serial.println("UDP listening on port 12345.");
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
    IPAddress fromIp = udp.remoteIP();
    uint16_t  fromPort = udp.remotePort();
    if (processPacket(rxBuf, len)) {
      senderIp   = fromIp;
      senderPort = fromPort;
    }
  }

  expireStalePeer();
}

bool commsWifiConnected() {
  return wasWifiConnected;
}

bool commsTakeCommand(Command& out) {
  if (!hasPendingCommand) return false;
  out = pendingCommand;
  hasPendingCommand = false;
  return true;
}

void commsSendStatus(const char* state, bool armed, uint32_t lastSeq) {
  if (senderPort == 0 || !wasWifiConnected) return;
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
