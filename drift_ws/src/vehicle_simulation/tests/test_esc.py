"""
Unit tests for the ESC input stage (``EscStateMachine``).

Each test mirrors one behaviour measured on the real car (issue #72): the
reverse lockout is set by forward throttle and cleared only at
neutral-at-standstill; positive throttle always drives forward.
"""

from vehicle_simulation.models.esc import EscStateMachine, MotorCommand


def _lock(esc: EscStateMachine) -> None:
    """Give forward throttle once, setting the reverse lockout."""
    esc.resolve(1.0, speed=0.0)


def test_fresh_esc_reverses_immediately_even_while_rolling():
    """A hand-pushed car (never given gas) spins its wheels backwards at once."""
    esc = EscStateMachine()
    assert esc.resolve(-1.0, speed=1.0) == MotorCommand(drive=-1.0)


def test_after_forward_throttle_negative_pedal_brakes():
    esc = EscStateMachine()
    _lock(esc)
    assert esc.resolve(-0.6, speed=2.0) == MotorCommand(brake=0.6)


def test_brake_pumping_while_rolling_keeps_braking():
    """Releases while still moving do not unlock reverse: every press brakes."""
    esc = EscStateMachine()
    _lock(esc)
    for _ in range(3):
        assert esc.resolve(-1.0, speed=1.5) == MotorCommand(brake=1.0)
        assert esc.resolve(0.0, speed=1.2) == MotorCommand()
    assert esc.resolve(-1.0, speed=1.0) == MotorCommand(brake=1.0)


def test_holding_brake_at_standstill_stays_braking():
    """Standstill alone does not unlock — the pedal must also return to neutral."""
    esc = EscStateMachine()
    _lock(esc)
    esc.resolve(-1.0, speed=2.0)
    for _ in range(10):
        assert esc.resolve(-1.0, speed=0.0) == MotorCommand(brake=1.0)


def test_release_at_standstill_unlocks_reverse():
    """The classic double-tap: brake to a stop, release, press again -> reverse."""
    esc = EscStateMachine()
    _lock(esc)
    esc.resolve(-1.0, speed=2.0)
    esc.resolve(-1.0, speed=0.0)
    esc.resolve(0.0, speed=0.0)
    assert esc.resolve(-0.7, speed=0.0) == MotorCommand(drive=-0.7)


def test_coasting_to_standstill_unlocks_reverse():
    """A car that rolled out on its own reverses directly, no double-tap needed."""
    esc = EscStateMachine()
    _lock(esc)
    assert esc.resolve(0.0, speed=1.0) == MotorCommand()  # coasting: still locked
    assert esc.resolve(-0.5, speed=0.8) == MotorCommand(brake=0.5)
    esc.resolve(0.0, speed=0.8)
    esc.resolve(0.0, speed=0.0)  # stopped with the pedal neutral
    assert esc.resolve(-0.5, speed=0.0) == MotorCommand(drive=-0.5)


def test_positive_throttle_always_drives_forward():
    """No braking in the reverse direction: gas while reversing goes forward."""
    esc = EscStateMachine()
    assert esc.resolve(-1.0, speed=0.0) == MotorCommand(drive=-1.0)  # reversing
    assert esc.resolve(0.9, speed=-1.0) == MotorCommand(drive=0.9)
    # ...and that forward throttle set the lockout again.
    assert esc.resolve(-1.0, speed=1.0) == MotorCommand(brake=1.0)


def test_deadband_counts_as_neutral():
    esc = EscStateMachine()
    _lock(esc)
    assert esc.resolve(0.04, speed=0.0) == MotorCommand()
    assert esc.resolve(-0.04, speed=0.0) == MotorCommand()
    # The in-deadband samples at standstill cleared the lockout.
    assert esc.resolve(-0.5, speed=0.0) == MotorCommand(drive=-0.5)


def test_inputs_are_clamped():
    esc = EscStateMachine()
    assert esc.resolve(3.0, speed=0.0) == MotorCommand(drive=1.0)
    assert esc.resolve(-3.0, speed=2.0) == MotorCommand(brake=1.0)


def test_reset_returns_to_power_on_state():
    esc = EscStateMachine()
    _lock(esc)
    esc.reset()
    assert esc.resolve(-1.0, speed=1.0) == MotorCommand(drive=-1.0)
