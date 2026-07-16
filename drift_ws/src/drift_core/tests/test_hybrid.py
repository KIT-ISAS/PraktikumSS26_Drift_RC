"""Neural-tire bicycle with latent wheel speed (torch-guarded)."""

import numpy as np
import pytest

pytest.importorskip("torch")

from drift_core.models import build_model  # noqa: E402
from drift_core.models.base import VehicleAction, VehicleState  # noqa: E402
from drift_core.models.registry import get_model_class  # noqa: E402

_NORM = {
    "input_mean": np.zeros(6),
    "input_std": np.ones(6),
    "output_mean": np.zeros(3),
    "output_std": np.ones(3),
    "dyn_std": np.ones(3),
}


def _new():
    from drift_core.models.hybrid import PhysicsInformedModel

    return PhysicsInformedModel.new(_NORM)


def test_registered_and_needs_a_checkpoint():
    from drift_core.models.hybrid import PhysicsInformedModel

    assert get_model_class("hybrid") is PhysicsInformedModel
    with pytest.raises(ValueError, match="needs a trained checkpoint"):
        build_model("hybrid")


def test_steering_lock_stays_near_the_measured_one():
    # The steering lock is anchored to the car's measured value and may only be
    # scaled inside a physical band: fitting it freely gave 30 deg against a
    # measured ~75, and the front tire curve absorbed the difference.
    import torch

    from drift_core.car_parameters import load_car_parameters

    model = _new()
    car = load_car_parameters("drift_rc")
    lo, hi = model._p._BOUNDED["steer_scale"]

    dyn = torch.tensor([[1.0, 0.0, 0.0]])
    latent = model.init_latent(dyn, torch.tensor([[1.0, 0.0, 0.0]]))
    lock = float(latent[0, 0])
    assert lo * car.max_steer_angle_rad <= lock <= hi * car.max_steer_angle_rad


def test_full_lock_washes_out_like_the_real_car():
    # The steering response is *not* monotone, and that is correct: this chassis
    # runs ~75 deg of lock, so full stick puts the front tire far past its force
    # peak and it slides. The recorded car does the same — at 1.1 m/s its mean yaw
    # rate peaks near half stick (1.64 rad/s) and drops at full lock (1.06). What
    # the model must not do is turn the washout into a plough.
    model = _new()

    def settled_yaw(steering):
        s = VehicleState(vx=1.2)
        for _ in range(75):
            s = model.step(s, VehicleAction(steering=steering, throttle=0.35), 0.02)
        return abs(s.yaw_rate)

    assert settled_yaw(1.0) < settled_yaw(0.5)  # washes out, as the real car does
    assert settled_yaw(1.0) > 0.5  # but still turns: the car is not a plough


def test_builds_and_steps():
    out = _new().step(VehicleState(vx=2.0, vy=0.2, yaw_rate=0.3), VehicleAction(0.4, 0.5), 0.03)
    assert np.isfinite([out.x, out.y, out.heading, out.vx, out.vy, out.yaw_rate]).all()


def test_launch_from_rest_is_stable_at_the_data_timestep():
    # The wheel-speed mode is the fastest in the model; at the 20-50 ms steps of the
    # recorded data it must still integrate without blowing up (this is what the iw
    # lower bound buys).
    model = _new()
    s = VehicleState()
    for _ in range(100):
        s = model.step(s, VehicleAction(1.0, 1.0), 0.05)
    assert np.isfinite([s.vx, s.vy, s.yaw_rate]).all()
    # Speed, not vx: full throttle at full lock spins the car up into a donut, so most
    # of its velocity is lateral. It must move off and stay bounded, not drive straight.
    assert 0.1 < s.speed < 20.0


def test_steering_at_standstill_does_not_turn():
    model = _new()
    s = VehicleState()
    for _ in range(100):
        s = model.step(s, VehicleAction(1.0, 0.0), 0.02)
    assert abs(s.yaw_rate) < 1e-3
    assert abs(s.vy) < 1e-3


def test_wheel_spins_up_when_the_rear_tire_saturates():
    # The point of the model: full throttle in a slide must drive the wheel faster
    # than the ground (positive slip ratio), which is what erodes rear lateral grip.
    # The drive-force surrogate in `hybrid` cannot represent this.
    import torch

    model = _new()
    # Hold the chassis sideways (a forced slide) and integrate only the latents, so
    # the wheel dynamics are isolated: an untrained model would otherwise straighten
    # the car out before the wheel had time to spin up.
    dyn = torch.tensor([[1.0, 0.8, 3.0]])
    action = torch.tensor([[0.0, 1.0, 0.0]])  # full throttle, no steering
    latent = model.init_latent(dyn, action)  # starts rolling: omega = vx / R
    for _ in range(60):
        state = torch.cat([dyn, latent], dim=-1)
        with torch.no_grad():
            latent = latent + model.derivative(state, action).mean[:, 3:] * 0.02

    omega, vx = float(latent[0, 2]), float(dyn[0, 0])
    sigma = (model._r_tire * omega - vx) / max(abs(vx), 0.4)
    assert sigma > 0.2, f"wheel did not spin up under full throttle in a slide (sigma={sigma:.2f})"


def test_releasing_the_controls_stops_the_car():
    # Coast down from a drift: with no throttle and no steering the car must settle,
    # not keep rotating. It used to, because the wheel had no motor droop — spun up
    # to ~200 rad/s in a donut it freewheeled for tens of seconds, holding the slip
    # ratio (and so the eroded rear grip) open the whole time. Only hard braking,
    # the one torque that could reach the wheel, would stop it.
    import torch

    model = _new()
    dyn = torch.tensor([[1.0, 0.0, 0.0]])
    drive = torch.tensor([[0.5, 0.6, 0.0]])
    state = torch.cat([dyn, model.init_latent(dyn, drive)], dim=-1)
    with torch.no_grad():
        for _ in range(250):  # into a donut
            state = state + model.derivative(state, drive).mean * 0.02
        coasting = torch.tensor([[0.0, 0.0, 0.0]])
        for _ in range(250):  # 5 s with everything released
            state = state + model.derivative(state, coasting).mean * 0.02

    speed = float(torch.hypot(state[0, 0], state[0, 1]))
    assert abs(float(state[0, 2])) < 0.5, "car is still rotating 5 s after release"
    assert speed < 0.3, f"car is still moving 5 s after release (speed={speed:.2f})"


def test_coasting_never_gains_energy():
    # With throttle and brake at zero the tires can only dissipate: any state whose
    # kinetic energy grows means the model is inventing energy. It did — the front
    # slip angle used the small-angle/forward-travel shortcut, so at this car's ~70 deg
    # of lock, rolling backwards, the front force pushed the axle the way it was already
    # sliding (+1 W out of nothing). Flick from full lock to full opposite lock, release
    # everything, and the car span about its rear axle forever.
    import torch

    model = _new()
    m, izz = model._m, float(model._p.izz)
    rng = np.random.default_rng(0)
    for _ in range(25):
        vx, vy, r = rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-6, 6)
        omega = rng.uniform(-80, 80)
        for steering in (-1.0, 0.0, 1.0):
            state = torch.tensor([[vx, vy, r, steering * model._steer_max, 0.0, omega]])
            action = torch.tensor([[steering, 0.0, 0.0]])  # coasting: no throttle, no brake

            def energy(s):
                return 0.5 * m * (s[0, 0] ** 2 + s[0, 1] ** 2) + 0.5 * izz * s[0, 2] ** 2

            with torch.no_grad():
                e0 = float(energy(state))
                for _ in range(50):  # 1 s
                    state = state + model.derivative(state, action).mean * 0.02
                e1 = float(energy(state))
            assert e1 <= e0 + 1e-6, (
                f"coasting gained energy: {e0:.3f} -> {e1:.3f} J "
                f"(vx={vx:.2f} vy={vy:.2f} r={r:.2f} omega={omega:.0f} steer={steering})"
            )


def test_flick_to_opposite_lock_and_release_settles():
    # The reported failure, end to end: donut, flick to full opposite lock, release
    # throttle and brake. The car must come to rest instead of spinning about its rear
    # axle forever.
    import torch

    model = _new()
    dyn = torch.tensor([[1.0, 0.0, 0.0]])
    drive = torch.tensor([[0.5, 0.6, 0.0]])
    state = torch.cat([dyn, model.init_latent(dyn, drive)], dim=-1)
    flick = torch.tensor([[-1.0, 0.0, 0.0]])
    with torch.no_grad():
        for _ in range(250):
            state = state + model.derivative(state, drive).mean * 0.02
        for _ in range(400):  # 8 s of full opposite lock, no throttle, no brake
            state = state + model.derivative(state, flick).mean * 0.02

    assert abs(float(state[0, 2])) < 0.5, "still spinning 8 s after release"
    assert float(torch.hypot(state[0, 0], state[0, 1])) < 0.3, "still moving 8 s after release"


def test_checkpoint_roundtrip_preserves_the_derivative():
    import torch

    from drift_core.models.hybrid import PhysicsInformedModel

    model = _new()
    ckpt = model.to_checkpoint()
    assert ckpt["model_type"] == "hybrid"

    reloaded = PhysicsInformedModel.from_checkpoint(ckpt)
    dyn = torch.tensor([[1.5, 0.4, 1.0, 0.1, 0.05, 60.0]])
    action = torch.tensor([[0.3, 0.8, 0.0]])
    with torch.no_grad():
        a = model.derivative(dyn, action).mean
        b = reloaded.derivative(dyn, action).mean
    assert torch.allclose(a, b)


def test_long_rollout_respects_the_tires_own_acceleration_limit():
    # A model cannot out-accelerate its own tires: with mu ~ 0.3 the whole car can pull
    # ~3 m/s^2, so 3 s of full throttle cannot reach 18 m/s. It did — the wheel-slip mode
    # (~2 ms) was being integrated with explicit Euler at 20-50 ms steps, so long rollouts
    # blew up numerically and "predicted" a recorded donut reaching 18 m/s.
    import torch

    model = _new()
    with torch.no_grad():
        mu = float(model._rear.mu)
    g = 9.81
    limit = mu * g * 3.0 + 2.0  # 3 s at the friction limit, plus slack

    for dt in (0.02, 0.05):
        s = VehicleState(vx=1.3, vy=0.5, yaw_rate=3.0)
        for _ in range(int(3.0 / dt)):
            s = model.step(s, VehicleAction(steering=0.4, throttle=1.0), dt)
        assert s.speed < limit, f"rollout diverged at dt={dt}: speed={s.speed:.1f} m/s"
