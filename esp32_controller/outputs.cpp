#include "outputs.h"
#include <ESP32Servo.h>

static Servo steeringServo;
static Servo escServo;
static Servo gyroServo;

static const int STEERING_PIN = 27;
static const int ESC_PIN      = 13;
static const int GYRO_PIN     = 33;

static const int PWM_MIN_US     = 1000;
static const int PWM_NEUTRAL_US = 1500;
static const int PWM_MAX_US     = 2000;

static const int STEERING_LEFT_US  = 1100;
static const int STEERING_RIGHT_US = 1900;
static const int THROTTLE_MIN_US   = 1200;
static const int THROTTLE_MAX_US   = 1800;
static const int GYRO_SOFT_MAX_US  = 1900;
static const int GYRO_HARD_MIN_US  = 1100;

static const unsigned long ESC_ARM_MS = 3000;
static unsigned long escReadyAtMs = 0;

static float clampf(float v, float lo, float hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

static int mapSymmetric(float value, int lowUs, int highUs) {
  float v = clampf(value, -1.0f, 1.0f);
  int span = v >= 0.0f ? (highUs - PWM_NEUTRAL_US) : (PWM_NEUTRAL_US - lowUs);
  return PWM_NEUTRAL_US + (int)(span * v);
}

static int mapGyro(GyroMode mode, float gain) {
  float g = clampf(gain, 0.0f, 1.0f);
  int limit = mode == GyroMode::Soft ? GYRO_SOFT_MAX_US : GYRO_HARD_MIN_US;
  return PWM_NEUTRAL_US + (int)((limit - PWM_NEUTRAL_US) * g);
}

void outputsBegin() {
  steeringServo.setPeriodHertz(50);
  escServo.setPeriodHertz(50);
  gyroServo.setPeriodHertz(50);

  steeringServo.attach(STEERING_PIN, PWM_MIN_US, PWM_MAX_US);
  escServo.attach(ESC_PIN,      PWM_MIN_US, PWM_MAX_US);
  gyroServo.attach(GYRO_PIN,    PWM_MIN_US, PWM_MAX_US);

  outputsFailsafe();
  escReadyAtMs = millis() + ESC_ARM_MS;
}

void outputsApply(const Command& cmd) {
  steeringServo.writeMicroseconds(mapSymmetric(cmd.steering, STEERING_LEFT_US, STEERING_RIGHT_US));
  escServo.writeMicroseconds(mapSymmetric(cmd.throttle, THROTTLE_MIN_US, THROTTLE_MAX_US));
  gyroServo.writeMicroseconds(mapGyro(cmd.gyroMode, cmd.gyroGain));
}

void outputsFailsafe() {
  steeringServo.writeMicroseconds(PWM_NEUTRAL_US);
  escServo.writeMicroseconds(PWM_NEUTRAL_US);
  gyroServo.writeMicroseconds(PWM_NEUTRAL_US);
}

bool outputsReady() {
  return (long)(millis() - escReadyAtMs) >= 0;
}
