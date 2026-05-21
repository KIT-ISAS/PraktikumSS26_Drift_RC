#include "outputs.h"
#include <ESP32Servo.h>

static Servo steeringServo;
static Servo escServo;
static Servo gyroServo;

static const int STEERING_PIN = 9;
static const int ESC_PIN      = 6;
static const int GYRO_PIN     = 10;

static const int PWM_MIN_US     = 1000;
static const int PWM_NEUTRAL_US = 1500;
static const int PWM_MAX_US     = 2000;

static const int STEERING_LEFT_US  = 1100;
static const int STEERING_RIGHT_US = 1900;
static const int THROTTLE_MIN_US   = 1200;
static const int THROTTLE_MAX_US   = 1800;
static const int GYRO_SOFT_MAX_US  = 1900;
static const int GYRO_HARD_MIN_US  = 1100;

static int clampi(int v, int lo, int hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

static int mapSteering(float steering) {
  int pulse = steering >= 0.0f
    ? PWM_NEUTRAL_US + (int)((STEERING_RIGHT_US - PWM_NEUTRAL_US) * steering)
    : PWM_NEUTRAL_US + (int)((PWM_NEUTRAL_US - STEERING_LEFT_US)  * steering);
  return clampi(pulse, STEERING_LEFT_US, STEERING_RIGHT_US);
}

static int mapThrottle(float throttle) {
  int pulse = throttle >= 0.0f
    ? PWM_NEUTRAL_US + (int)((THROTTLE_MAX_US - PWM_NEUTRAL_US) * throttle)
    : PWM_NEUTRAL_US + (int)((PWM_NEUTRAL_US - THROTTLE_MIN_US) * throttle);
  return clampi(pulse, THROTTLE_MIN_US, THROTTLE_MAX_US);
}

static int mapGyro(GyroMode mode, float gain) {
  float g = gain < 0.0f ? 0.0f : gain > 1.0f ? 1.0f : gain;
  if (mode == GyroMode::Soft)
    return PWM_NEUTRAL_US + (int)((GYRO_SOFT_MAX_US - PWM_NEUTRAL_US) * g);
  return PWM_NEUTRAL_US - (int)((PWM_NEUTRAL_US - GYRO_HARD_MIN_US) * g);
}

void outputsBegin() {
  steeringServo.setPeriodHertz(50);
  escServo.setPeriodHertz(50);
  gyroServo.setPeriodHertz(50);

  steeringServo.attach(STEERING_PIN, PWM_MIN_US, PWM_MAX_US);
  escServo.attach(ESC_PIN,      PWM_MIN_US, PWM_MAX_US);
  gyroServo.attach(GYRO_PIN,    PWM_MIN_US, PWM_MAX_US);

  outputsFailsafe();
  delay(3000);
}

void outputsApply(const Command& cmd) {
  steeringServo.writeMicroseconds(mapSteering(cmd.steering));
  escServo.writeMicroseconds(mapThrottle(cmd.throttle));
  gyroServo.writeMicroseconds(mapGyro(cmd.gyroMode, cmd.gyroGain));
}

void outputsFailsafe() {
  steeringServo.writeMicroseconds(PWM_NEUTRAL_US);
  escServo.writeMicroseconds(PWM_NEUTRAL_US);
  gyroServo.writeMicroseconds(PWM_NEUTRAL_US);
}
