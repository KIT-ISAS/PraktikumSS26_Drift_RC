"""
ESC input stage: resolves the throttle pedal into what the motor actually does.

The real car's ESC sits between the received throttle signal and the motor. Its
brake/reverse behaviour was measured on the vehicle (issue #72):

* **Positive throttle always drives forward, immediately** — even while rolling
  backwards. There is no braking in the reverse direction.
* **A negative pedal means brake or reverse, decided by a single lockout flag:**

  - the lockout is *set* by any positive (forward) throttle;
  - the lockout is *cleared* when the pedal is neutral while the car is
    stationary — however it stopped (braked to a stop and released, or coasted
    out on its own);
  - lockout set: negative pedal = brake. It holds the car at standstill, and
    pumping the brake while still rolling keeps braking;
  - lockout clear: negative pedal = reverse, at any speed — a car pushed
    forward by hand spins its wheels backwards immediately.

The familiar "double-tap to reverse" is not a mechanism of its own: it is just
the pedal gesture that happens to pass through neutral-at-standstill.

:class:`EscStateMachine` is that flag plus the deadband. It is deliberately
free of ROS/model/numpy dependencies so the *same* code runs everywhere the
pedal stream exists:

* in front of every simulation vehicle model (``VehicleModel.step`` resolves
  the pedal before the dynamics see it);
* when labeling recorded drives for dynamics training: replaying the logged
  commands (with the measured speed as the standstill input) reproduces the
  ``(drive, brake)`` the real ESC output, which keeps the learning problem
  Markov — raw negative throttle is ambiguous, the resolved command is not;
* in the deployment node, should the ESC state ever become a policy input.

The real ESC senses *motor* standstill (back-EMF, readable only while the
output is neutral); we proxy that with the forward speed and a small
threshold. If the real unit turns out to require neutral-at-standstill to
persist for some time before unlocking, add that debounce here — this is the
only place that would change.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MotorCommand:
    """What the motor actually does, after the ESC has interpreted the pedal."""

    drive: float = 0.0  # [-1, 1] — signed drive torque command (negative = reverse)
    brake: float = 0.0  # [0, 1] — brake strength (decelerates, never through zero)


@dataclass
class EscStateMachine:
    """The ESC's reverse lockout: one flag deciding what a negative pedal means."""

    neutral_deadband: float = 0.05  # |throttle| below this counts as neutral
    standstill_speed: float = 0.05  # m/s — below this the ESC senses standstill

    # Set by forward throttle, cleared at neutral + standstill. Starts clear: a
    # freshly powered car reverses on the first negative pedal.
    reverse_locked: bool = field(init=False, default=False)

    def resolve(self, throttle: float, speed: float) -> MotorCommand:
        """
        Interpret one pedal sample given the car's current signed forward speed.

        Call once per control tick, in command order — the lockout flag carries
        across calls, exactly like the real ESC.
        """
        throttle = max(-1.0, min(1.0, throttle))
        if throttle >= self.neutral_deadband:
            self.reverse_locked = True
            return MotorCommand(drive=throttle)
        if throttle <= -self.neutral_deadband:
            if self.reverse_locked:
                return MotorCommand(brake=-throttle)
            return MotorCommand(drive=throttle)
        if abs(speed) < self.standstill_speed:
            self.reverse_locked = False
        return MotorCommand()

    def reset(self) -> None:
        """Return to the power-on state (lockout clear)."""
        self.reverse_locked = False
