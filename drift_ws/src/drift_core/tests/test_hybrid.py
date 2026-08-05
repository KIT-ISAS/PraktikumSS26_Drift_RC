"""Neural-tire bicycle with latent wheel speed (torch-guarded)."""

import numpy as np
import pytest

pytest.importorskip("torch")

from drift_core.vehicle_model import build_model  # noqa: E402
from drift_core.vehicle_model.models.base import VehicleAction, VehicleState  # noqa: E402
from drift_core.vehicle_model.models.registry import get_model_class  # noqa: E402

_NORM = {
    "input_mean": np.zeros(6),
    "input_std": np.ones(6),
    "output_mean": np.zeros(3),
    "output_std": np.ones(3),
    "dyn_std": np.ones(3),
}


def _new(**kwargs):
    from drift_core.vehicle_model.models.hybrid import PhysicsInformedModel

    return PhysicsInformedModel.new(_NORM, **kwargs)


def test_registered_and_needs_a_checkpoint():
    from drift_core.vehicle_model.models.hybrid import PhysicsInformedModel

    assert get_model_class("hybrid") is PhysicsInformedModel
    with pytest.raises(ValueError, match="needs a trained checkpoint"):
        build_model("hybrid")


def test_steering_lock_stays_near_the_measured_one():
    # The steering lock is anchored to the car's measured value and may only be
    # scaled inside a physical band: fitting it freely gave 30 deg against a
    # measured ~75, and the front tire curve absorbed the difference.
    import torch

    from drift_core.vehicle_model.car_parameters import load_car_parameters

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

    model = _new(nn0_residual=True)  # untrained-physics check: needs the geometric NN0 init
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

    model = _new(nn0_residual=True)  # untrained-physics check: needs the geometric NN0 init
    m, izz = model._m, float(model._p.izz)
    rng = np.random.default_rng(0)
    for _ in range(25):
        vx, vy, r = rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-6, 6)
        omega = rng.uniform(-80, 80)
        for steering in (-1.0, 0.0, 1.0):
            state = torch.tensor([[vx, vy, r, steering * model._steer_max, 0.0, omega, 0.0]])
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

    model = _new(nn0_residual=True)  # untrained-physics check: needs the geometric NN0 init
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


def test_initial_direction_is_the_geometric_friction_circle():
    # NN0's residual is zero-initialized, so an untrained model must split the rear
    # force exactly along the slip vector (sqrt(f0)·sigma, -tan_alpha) — the proven
    # geometric friction circle. Training only ever learns a correction on top. This is
    # the residual form (no longer the default), constructed explicitly.
    import torch

    model = _new(nn0_residual=True)
    vx, vy, r = torch.tensor([1.5]), torch.tensor([0.6]), torch.tensor([2.0])
    omega = torch.tensor([80.0])
    with torch.no_grad():
        fxr, fyr, sigma = model._rear_slip(vx, vy, r, omega)
        f0 = model._p.f0
        tan_ar = (vy - model._b * r) / vx.abs().clamp(min=0.5)
        dx, dy = f0.sqrt() * sigma, -tan_ar
        norm = torch.hypot(dx, dy)
    assert torch.allclose(fxr / torch.hypot(fxr, fyr), dx / norm, atol=1e-5)
    assert torch.allclose(fyr / torch.hypot(fxr, fyr), dy / norm, atol=1e-5)


def test_rear_total_force_is_never_negative():
    # The rear ExpTanh runs on the squared combined slip (z >= 0) and its output is
    # clamped: a negative total force magnitude would flip the force *toward* the slip.
    import torch

    model = _new()
    with torch.no_grad():
        for omega in (-50.0, 0.0, 50.0, 200.0):
            fxr, fyr, _ = model._rear_slip(
                torch.linspace(-2, 2, 41),
                torch.linspace(-2, 2, 41),
                torch.zeros(41),
                torch.full((41,), omega),
            )
            f_tot = torch.hypot(fxr, fyr)
            assert torch.isfinite(f_tot).all()
            assert (f_tot >= 0).all()


def test_actuator_lag_is_stable_past_the_euler_limit():
    # delta and tq integrate exactly, so even dt >> tau must converge monotonically to
    # the commanded target instead of oscillating divergently (explicit Euler diverges
    # for dt > 2*tau, and tau_s may be as small as 0.02 s against 50 ms data steps).
    import torch

    model = _new()
    tau_s = float(model._p.tau_s)
    dt = 5.0 * tau_s  # far past the Euler stability limit
    dyn = torch.tensor([[1.0, 0.0, 0.0]])
    action = torch.tensor([[1.0, 0.0, 0.0]])
    target = float(model._p.steer_scale * model._steer_max)

    state = torch.cat([dyn, torch.zeros((1, model.latent_dim))], dim=-1)  # delta starts at 0
    prev_err = target
    with torch.no_grad():
        for _ in range(20):
            deriv = model.derivative(state, action).mean
            state = model.advance(state, action, deriv, dt)
            err = abs(float(state[0, 3]) - target)
            assert err <= prev_err + 1e-9, "servo angle overshot/oscillated at large dt"
            prev_err = err
    assert prev_err < 1e-3 * target


def _rolling_braked_model():
    """
    Fresh model with the ESC's reverse lockout armed, as after driving forward.

    A cold ESC treats a negative pedal as *reverse*, not brake (the lockout is only
    set by driving forward first — same convention as the real car). Tests that mean
    to exercise the brake must arm it, exactly like DriftEnv.reset does; and each
    rollout gets its own model so no latent/ESC state leaks between rollouts.

    Uses the residual (zero-init) direction net: these are structural-physics checks on
    an *untrained* model, which is only physically valid when the rear-force direction
    starts on the geometric friction circle. The default paper-literal NN0 starts random.
    """
    model = _new(nn0_residual=True)
    model.esc.reverse_locked = True
    return model


def test_brief_brake_tap_barely_slows_the_car():
    # The ESC's brake actuation is not instantaneous: a recorded 0.19 s brake pump
    # barely slowed the real car, while an instant-torque model braked at ~3 m/s^2.
    # The brake torque is a lagged latent (fixed ~50 ms), so a short tap must cost
    # far less speed than the same duration of held brake.
    def speed_after(brake_pattern):
        model = _rolling_braked_model()
        s = VehicleState(vx=1.5)
        for b in brake_pattern:
            s = model.step(s, VehicleAction(steering=0.0, throttle=-b), 0.02)
        return s.speed

    tap = speed_after([1.0] * 3 + [0.0] * 22)  # 60 ms tap, then released
    held = speed_after([1.0] * 25)  # brake held the whole 0.5 s
    assert tap > held + 0.05, (
        f"a 60 ms tap ({tap:.2f} m/s) should cost far less than held brake ({held:.2f})"
    )


def test_locked_wheel_braking_holds_to_low_speed():
    # With the slip denominator floored at |vx| alone (0.4 m/s), a locked wheel's
    # sigma faded as -vx/0.4 and the brake force died quadratically below 0.4 m/s —
    # the model crept at the end of every stop. The two-sided denominator keeps
    # sigma at -1 (full Coulomb sliding force) down to near-standstill, so a
    # braking car must actually stop, and quickly once it is slow.
    model = _rolling_braked_model()
    s = VehicleState(vx=1.5)
    t = 0.0
    while s.speed > 0.05 and t < 3.0:
        s = model.step(s, VehicleAction(steering=0.0, throttle=-1.0), 0.02)
        t += 0.02
    assert s.speed <= 0.05, f"still creeping at {s.speed:.2f} m/s after {t:.1f} s of full brake"
    assert t < 2.5, f"took {t:.2f} s to stop from 1.5 m/s — brake force is fading at low speed"


def test_flick_transition_cannot_outbrake_the_tires():
    # Counter-steer flick out of a fast drift: as sideslip sweeps past ~90 deg the
    # body-frame vx collapses while the car is still moving fast sideways. The
    # standstill gate used to key on |vx| alone, so it switched the tires off
    # mid-slide and let the 8/s scrub damper brake harder than any tire could —
    # the slide died in half a second instead of carrying through the transition.
    model = _new()
    a_tires = (model._front.peak_force() + model._rear.peak_force()) / model._m
    s = VehicleState(vx=1.5, vy=-1.3, yaw_rate=3.5)  # V = 2.0 m/s, beta = -41 deg
    prev_v, dt = s.speed, 0.02
    for _ in range(50):  # 1 s of full opposite lock, throttle released
        s = model.step(s, VehicleAction(steering=-1.0, throttle=0.0), dt)
        decel = (prev_v - s.speed) / dt
        drag_now = float(model._p.drag) * abs(s.vx) / model._m
        assert decel <= a_tires + drag_now + 0.5, (
            f"decelerated at {decel:.2f} m/s^2, more than tires+drag "
            f"({a_tires:.2f}+{drag_now:.2f}) can do — the standstill scrub is braking a moving car"
        )
        prev_v = s.speed


def test_stochastic_hybrid_returns_a_chassis_covariance():
    # With stochastic=True the model carries the paper's diffusion head Sigma(x, u):
    # derivative() returns a per-second (B, 3) variance for [vx, vy, r] (the latents
    # get no noise), which is what loss="nll" trains against. Deterministic default
    # stays cov=None.
    import torch

    from drift_core.vehicle_model.models.hybrid import PhysicsInformedModel

    det = PhysicsInformedModel.new(_NORM)
    sto = PhysicsInformedModel.new(_NORM, stochastic=True)
    dyn = torch.tensor([[1.5, 0.4, 1.0, 0.1, 0.05, 60.0, 0.02]])
    action = torch.tensor([[0.3, 0.8, 0.0]])
    with torch.no_grad():
        assert det.derivative(dyn, action).cov is None
        cov = sto.derivative(dyn, action).cov
    assert cov.shape == (1, 3)
    assert torch.isfinite(cov).all() and (cov > 0).all()


def test_stochastic_hybrid_checkpoint_roundtrip_and_sampling():
    import torch

    from drift_core.vehicle_model.models.hybrid import PhysicsInformedModel

    model = PhysicsInformedModel.new(_NORM, stochastic=True)
    ckpt = model.to_checkpoint()
    assert ckpt["stochastic"] is True

    # Reloaded without stochastic=True: the head survives (cov still returned, NLL
    # fine-tuning possible) but the sim does not sample from it.
    quiet = PhysicsInformedModel.from_checkpoint(ckpt)
    dyn = torch.tensor([[1.5, 0.4, 1.0, 0.1, 0.05, 60.0, 0.0]])
    action = torch.tensor([[0.3, 0.8, 0.0]])
    with torch.no_grad():
        a = model.derivative(dyn, action).cov
        b = quiet.derivative(dyn, action).cov
    assert torch.allclose(a, b)
    assert not quiet._can_sample

    # Reloaded with stochastic=True: step_stochastic must actually sample (and the
    # noise must broadcast correctly past the 4 latent states).
    noisy = PhysicsInformedModel.from_checkpoint(ckpt, stochastic=True)
    torch.manual_seed(0)
    s = VehicleState(vx=1.0)
    a1 = noisy.step_stochastic(s, VehicleAction(0.2, 0.5), 0.02)
    a2 = noisy.step_stochastic(s, VehicleAction(0.2, 0.5), 0.02)
    assert np.isfinite([a1.vx, a1.vy, a1.yaw_rate, a2.vx]).all()
    assert (a1.vx, a1.vy, a1.yaw_rate) != (a2.vx, a2.vy, a2.yaw_rate), "sampling had no effect"


@pytest.mark.xfail(
    reason="Sigma is ungated: the slip gate is deferred, see hybrid/model.py's "
    "'Deferred' note above the self._diff call. Drop this marker once it lands.",
)
def test_diffusion_is_gated_by_tire_slip():
    # Process noise originates in the tires, so Sigma fades with the combined slip:
    # parked and clean rolling are (near-)deterministic, wheelspin gets full noise —
    # even at zero ground speed, which is exactly the launch-instability case. A
    # speed gate failed here: microscopic kicks opened it, crawl-speed noise was
    # ungated, and a "parked" sampled sim bootstrapped itself to 1.5 m/s.
    import torch

    from drift_core.vehicle_model.models.hybrid import PhysicsInformedModel

    model = PhysicsInformedModel.new(_NORM, stochastic=True)
    action = torch.zeros((1, 3))
    parked = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    rolling = torch.tensor([[1.5, 0.0, 0.0, 0.0, 0.0, 1.5 / model._r_tire, 0.0]])
    spinning = torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.1, 100.0, 0.0]])  # launch wheelspin
    with torch.no_grad():
        cov_parked = model.derivative(parked, action).cov
        cov_rolling = model.derivative(rolling, action).cov
        cov_spinning = model.derivative(spinning, action).cov
    assert (cov_parked < 1e-3 * cov_spinning).all(), "parked noise not gated off"
    assert (cov_rolling < 0.05 * cov_spinning).all(), "clean rolling should be near-quiet"


def test_parked_wheel_settles_at_rest_even_with_a_trained_direction_net():
    # An unpinned NN0 residual shifted the tire's zero-force point to a nonzero wheel
    # slip, so init_latent settled a *parked* wheel at omega ~ -1.5 rad/s — phantom
    # slip that opened the slip-gated diffusion. The residual is scaled by the slip
    # magnitude now; perturb the net the way training would and the parked
    # equilibrium must stay at omega = 0.
    import torch

    model = _new()
    with torch.no_grad():  # emulate a trained residual: nonzero at any input
        model._nn0.net[2].bias.fill_(0.3)
        model._nn0.net[2].weight.normal_(0.0, 0.3)
    dyn = torch.tensor([[0.0, 0.0, 0.0]])
    latent = model.init_latent(dyn, torch.tensor([[0.0, 0.0, 0.0]]))
    assert abs(float(latent[0, 2])) < 1e-3, (
        f"parked wheel settled at omega={float(latent[0, 2]):.3f} instead of 0"
    )


@pytest.mark.xfail(
    reason="Same deferred slip gate as test_diffusion_is_gated_by_tire_slip: with "
    "Sigma ungated the parked car bootstraps itself off its own noise.",
)
def test_parked_car_does_not_wander_stochastically():
    # The user-reported failure, end to end: a stochastic sim with zero input must
    # leave the car parked. With slip-gated noise there is no bootstrap — no slip,
    # no kicks, so the speed never climbs into the noisy regime.
    import torch

    from drift_core.vehicle_model.models.hybrid import PhysicsInformedModel

    torch.manual_seed(0)
    model = PhysicsInformedModel.new(_NORM, stochastic=True)
    s = VehicleState()
    maxv = 0.0
    for _ in range(500):  # 10 s parked, sampling the SDE every step
        s = model.step_stochastic(s, VehicleAction(0.0, 0.0), 0.02)
        maxv = max(maxv, s.speed)
    assert maxv < 0.05, f"parked car reached {maxv:.3f} m/s from noise alone"
    assert np.hypot(s.x, s.y) < 0.05, f"parked car wandered {np.hypot(s.x, s.y):.3f} m"


def test_checkpoint_roundtrip_preserves_the_derivative():
    import torch

    from drift_core.vehicle_model.models.hybrid import PhysicsInformedModel

    model = _new()
    ckpt = model.to_checkpoint()
    assert ckpt["model_type"] == "hybrid"

    reloaded = PhysicsInformedModel.from_checkpoint(ckpt)
    dyn = torch.tensor([[1.5, 0.4, 1.0, 0.1, 0.05, 60.0, 0.02]])
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
    model = _new(nn0_residual=True)  # untrained-physics check: needs the geometric NN0 init
    # The rear's fitted peak force over the total mass is the hardest the car can
    # accelerate; anything far past it means the integrator, not the tire, is driving.
    a_max = model._rear.peak_force() / model._m
    limit = a_max * 3.0 + 3.0  # 3 s at the friction limit, plus slack

    for dt in (0.02, 0.05):
        s = VehicleState(vx=1.3, vy=0.5, yaw_rate=3.0)
        for _ in range(int(3.0 / dt)):
            s = model.step(s, VehicleAction(steering=0.4, throttle=1.0), dt)
        assert s.speed < limit, f"rollout diverged at dt={dt}: speed={s.speed:.1f} m/s"
