"""The arm/stop/brake state machine: start/stop/brake/neutral/release/override paths."""

from agent_pkg.lib.control_gate import ControlGate, InputFreshness

RATE_HZ = 50.0
INPUT_TIMEOUT_S = 0.3
DISARM_TIMEOUT_S = 2.0
BRAKE_THROTTLE = -0.4
SPEED_THRESHOLD = 0.2
STOP_TIMEOUT_S = 2.0
# Neutral settle window: max(1, round(rate_hz * 0.3)) ticks before release.
NEUTRAL_TICKS = 15


def make_gate() -> ControlGate:
    return ControlGate(
        rate_hz=RATE_HZ,
        input_timeout_s=INPUT_TIMEOUT_S,
        disarm_timeout_s=DISARM_TIMEOUT_S,
        stop_brake_throttle=BRAKE_THROTTLE,
        stop_speed_threshold=SPEED_THRESHOLD,
        stop_timeout_s=STOP_TIMEOUT_S,
    )


def test_start_arms_once():
    gate = make_gate()
    assert not gate.running
    assert gate.start()
    assert gate.running and not gate.stopping
    # Repeated start is a no-op (no re-log, no state change).
    assert not gate.start()
    assert gate.running


def test_start_during_braking_cancels_braking_and_rearms():
    gate = make_gate()
    gate.start()
    gate.stop(now=0.0)
    assert gate.stopping
    assert gate.start()
    assert gate.running and not gate.stopping


def test_stop_while_idle_is_noop():
    gate = make_gate()
    assert not gate.stop(now=0.0)
    assert not gate.running and not gate.stopping


def test_stop_disarms_and_enters_braking():
    gate = make_gate()
    gate.start()
    assert gate.stop(now=0.0)
    assert not gate.running and gate.stopping


def test_repeated_stop_does_not_restart_brake_timer():
    gate = make_gate()
    gate.start()
    assert gate.stop(now=0.0)
    assert not gate.stop(now=1.9)
    # Past the timeout of the *first* stop: neutral even though the car still moves.
    command = gate.stop_step(now=2.1, speed=1.0)
    assert command.throttle == 0.0


def test_brakes_while_moving():
    gate = make_gate()
    gate.start()
    gate.stop(now=0.0)
    command = gate.stop_step(now=0.02, speed=1.5)
    assert command == (0.0, BRAKE_THROTTLE, False)
    assert gate.stopping


def test_neutral_when_standing_unknown_speed_or_timed_out():
    for now, speed in (
        (0.02, SPEED_THRESHOLD),  # standing (at/below threshold)
        (0.02, None),  # stale/missing odom: skip straight to neutral
        (STOP_TIMEOUT_S, 1.5),  # still moving but braking timed out
    ):
        gate = make_gate()
        gate.start()
        gate.stop(now=0.0)
        command = gate.stop_step(now=now, speed=speed)
        assert command == (0.0, 0.0, False)


def test_neutral_settle_window_then_release():
    gate = make_gate()
    gate.start()
    gate.stop(now=0.0)
    for _ in range(NEUTRAL_TICKS - 1):
        command = gate.stop_step(now=0.1, speed=0.0)
        assert command == (0.0, 0.0, False)
        assert gate.stopping
    command = gate.stop_step(now=0.1, speed=0.0)
    assert command == (0.0, 0.0, True)
    assert not gate.stopping and not gate.running


def test_resumed_motion_during_settle_window_brakes_again():
    gate = make_gate()
    gate.start()
    gate.stop(now=0.0)
    gate.stop_step(now=0.1, speed=0.0)
    command = gate.stop_step(now=0.2, speed=1.0)
    assert command.throttle == BRAKE_THROTTLE


def test_override_releases_immediately():
    gate = make_gate()
    gate.start()
    assert gate.override()
    assert not gate.running and not gate.stopping


def test_override_cancels_braking():
    gate = make_gate()
    gate.start()
    gate.stop(now=0.0)
    assert gate.override()
    assert not gate.stopping


def test_override_while_idle_is_noop():
    gate = make_gate()
    assert not gate.override()


def test_disarm_enters_the_same_braking_path_as_stop():
    gate = make_gate()
    gate.start()
    gate.disarm(now=0.0)
    assert not gate.running and gate.stopping
    assert gate.stop_step(now=0.02, speed=1.0).throttle == BRAKE_THROTTLE


def test_disarm_while_braking_does_not_restart_brake_timer():
    gate = make_gate()
    gate.start()
    gate.stop(now=0.0)
    gate.disarm(now=1.9)
    assert gate.stop_step(now=2.1, speed=1.0).throttle == 0.0


def test_classify_input_age_thresholds():
    gate = make_gate()
    assert gate.classify_input_age(0.0) is InputFreshness.FRESH
    assert gate.classify_input_age(INPUT_TIMEOUT_S) is InputFreshness.FRESH
    assert gate.classify_input_age(INPUT_TIMEOUT_S + 0.01) is InputFreshness.STALE
    assert gate.classify_input_age(DISARM_TIMEOUT_S) is InputFreshness.STALE
    assert gate.classify_input_age(DISARM_TIMEOUT_S + 0.01) is InputFreshness.DROPOUT
