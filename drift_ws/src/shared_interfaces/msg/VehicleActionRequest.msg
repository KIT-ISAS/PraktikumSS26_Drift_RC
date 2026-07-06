# Steering, ISO 8855: range [-1, 1], positive = left (CCW / +yaw rate),
# negative = right. Matches the vehicle model and the /car/odom yaw_rate sign.
# The physical servo direction is handled at the actuation edge (ESP32 firmware).
float32 steering
# Throttle: range [-1, 1], positive = forward, negative = brake/reverse.
float32 throttle
string gyro_mode "soft"
float32 gyro_gain 0.0