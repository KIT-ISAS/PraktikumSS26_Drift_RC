"""
The paper's neural-tire model (Djeumou et al., arXiv:2410.20990).

An ExpTanh force curve per axle plus a learned friction-circle direction net.
"""

import math

import torch
import torch.nn as nn

from ..dynamic import _inv_softplus
from ._nn import _mlp

# Tire-curve initialization. The curves are the paper's free ExpTanh fits (f1..f5), so
# only their *starting point* carries the physics we measured: peak ~ mu_init·Fz with
# mu_init from the car's acceleration envelope (p90 of smoothed CoG accel ~3 m/s², hard
# plastic drift tires on a smooth floor), near-flat falloff (their friction is close to
# Coulomb: kinetic ~ static, little peak-then-drop), and the force ramp saturating over
# ~0.1 of tan-slip. The peak is NOT bounded anymore — an earlier free fit chose mu ~ 0.49
# (a tire so grippy the wheel never spun and the model stopped drifting), so compare the
# fitted peak_force() against the measured envelope whenever the model is retrained.
_MU_INIT = 0.30
_TAN_SLIP_SAT = 0.1  # tan-slip over which the *front* force ramp saturates at init
_F3_INIT = 0.05  # near-zero falloff at init: a sliding tire keeps ~the peak force

# Rear ramp rate. The rear curve's argument is the *squared* combined slip, so its f4
# lives on a different scale than the front's — and initializing it as the front's rate
# squared (1111) put it where tanh(f4·(z - f5)) is saturated across the entire reachable
# domain z >= 0. sech²(20) is ~1e-18, so f4 and f5 received *exactly* zero gradient: the
# ramp stayed frozen at its initial value in every run ever trained (measured f4 = 1111.1
# to five digits across two seeds), leaving a rear tire that delivers 100% of its peak
# force at zero slip — a Coulomb friction, not the paper's shaped curve. Set it from the
# slip where the rear should saturate instead. Starting soft is recoverable (a fit that
# wants a sharper curve raises f4 from below); starting saturated is a one-way trap.
_REAR_SLIP_SAT = 0.3  # tan-slip (~17 deg) at which the rear reaches ~99% of peak at init
_TANH_99 = 2.65  # atanh(0.99): where a tanh ramp is done

# The rear's z is a sum of squares, so z >= 0 always and a negative f5 cannot represent
# anything physical — its only effect is to shift the curve so that it is *already*
# saturated at zero slip, which is precisely what the fits did (F(0) = f1 + f2, the whole
# peak). The rear's f5 is therefore constrained positive, making F(0) <= f1. The front's
# argument is signed, so its f5 is a legitimate left/right asymmetry and stays free.
_REAR_F5_INIT = 1e-3

# Network shapes, from the paper (Sec. IV-A): "Following the architecture of Sigma, NN0
# and NN in the tire force design only have 8 neurons each." Two-hidden-layer tanh MLPs.
_NN0_HIDDEN, _NN0_LAYERS = 8, 2
_TIRE_HIDDEN, _TIRE_LAYERS = 8, 2

# Nominal scales for the tire-curve features [r, V, beta(, omega_r)], from the car's own
# envelope: a hard spin, its top speed, and a quarter turn of sideslip. The wheel-speed
# scale is the motor's no-load speed and is filled in per car (see ``build_curves``).
_FEAT_SCALE_RV_BETA = (10.0, 3.0, math.pi / 2)


def _feat_scale_of(curve: "_ExpTanhCurve") -> list[float] | None:
    """A tire curve's feature scales as a plain list (``None`` if it is unconditioned)."""
    return None if curve.net is None else curve.feat_scale.tolist()


class _ExpTanhCurve(nn.Module):
    """
    The paper's ExpTanh tire curve (Djeumou et al., arXiv:2410.20990).

        F(z) = f1 + f2 · exp(-f3·|z|) · tanh(f4·(z - f5)),   f3, f4 >= 0.

    ``f2·tanh(f4·…)`` is the force ramp, ``exp(-f3·|z|)`` the past-peak falloff, and
    ``f1`` the fully-sliding asymptote (``F -> f1`` as ``z -> inf``).

    The five coefficients are *state-conditioned* like the paper's: ``(f1..f5) =
    NN(feat)``, with ``feat = [r, V, beta]`` for the front curve and ``[r, V, beta,
    omega_r]`` for the rear, so the curve reshapes with the operating point (load
    transfer, speed-dependent friction) instead of being one fixed curve per axle.
    ``feat_scale=None`` builds the earlier fixed-scalar curve instead — the shape every
    checkpoint before 2026-07-22 was trained in, and the fallback if conditioning turns
    out to overfit this dataset (it tops out near 3 m/s with little speed spread, so a
    speed dependence is only weakly identifiable, and a state-conditioned curve is free
    to absorb dynamics the fixed force balance is meant to explain).

    Transforms keep f3, f4 positive, which is the paper's only stated constraint, and f5
    positive where the curve's argument is a squared quantity (``positive_f5``, see
    _REAR_F5_INIT); f1 and f2 are free. The conditioned curve's output layer is
    zero-initialized with the bias holding the same coefficients the fixed curve starts
    at, so an untrained conditioned model is *identical* to the fixed one and training
    only learns the state dependence.
    """

    def __init__(
        self,
        peak: float,
        f4: float,
        f3: float = _F3_INIT,
        f5: float = 0.0,
        positive_f5: bool = False,
        feat_scale: "tuple[float, ...] | None" = None,
        hidden: int = _TIRE_HIDDEN,
        layers: int = _TIRE_LAYERS,
    ) -> None:
        super().__init__()
        # `self.f5` holds the raw pre-softplus value when positive_f5 is set; the
        # state_dict key stays "f5" either way, so the flag has to be checkpointed.
        self.positive_f5 = bool(positive_f5)
        raw_f5 = _inv_softplus(f5) if self.positive_f5 else f5
        init = (0.0, float(peak), _inv_softplus(f3), _inv_softplus(f4), raw_f5)
        if feat_scale is None:
            self.net = None
            self.f1 = nn.Parameter(torch.tensor(init[0]))
            self.f2 = nn.Parameter(torch.tensor(init[1]))
            self.raw_f3 = nn.Parameter(torch.tensor(init[2]))
            self.raw_f4 = nn.Parameter(torch.tensor(init[3]))
            self.f5 = nn.Parameter(torch.tensor(init[4]))
        else:
            self.net = _mlp(len(feat_scale), hidden, layers, 5)
            nn.init.zeros_(self.net[-1].weight)
            with torch.no_grad():
                self.net[-1].bias.copy_(torch.tensor(init))
            # Fixed physical scales, so every feature reaches the first tanh at O(1).
            # Raw units would saturate it immediately: omega_r runs to ~400 rad/s on
            # this car against a slip angle of ~1. Normalization, not a model change.
            self.register_buffer("feat_scale", torch.tensor(feat_scale, dtype=torch.float32))

    def coefficients(self, feat: torch.Tensor | None = None):
        """``(f1, f2, f3, f4, f5)`` — constants, or the network's output for ``feat``."""
        softplus = torch.nn.functional.softplus
        shift = softplus if self.positive_f5 else (lambda x: x)
        if self.net is None:
            return (
                self.f1,
                self.f2,
                softplus(self.raw_f3),
                softplus(self.raw_f4),
                shift(self.f5),
            )
        if feat is None:
            raise ValueError(
                "this tire curve is state-conditioned — pass its feature vector "
                "([r, V, beta] front, + omega_r rear); see PhysicsInformedModel._tire_feat"
            )
        c = self.net(feat / self.feat_scale)
        return (
            c[..., 0],
            c[..., 1],
            softplus(c[..., 2]),
            softplus(c[..., 3]),
            shift(c[..., 4]),
        )

    def forward(self, z: torch.Tensor, feat: torch.Tensor | None = None) -> torch.Tensor:
        # Paper-literal ExpTanh. The offset F(0) = f1 + f2·tanh(-f4·f5) is left free, as
        # in the paper: its always-sliding, wheel-speed-sensed regime never visits z~0.
        # Our parked/low-slip RC regime can expose a phantom force at zero slip; if that
        # bites, a zero-slip boundary condition (subtracting F(0)) is the fix.
        f1, f2, f3, f4, f5 = self.coefficients(feat)
        return f1 + f2 * torch.exp(-f3 * z.abs()) * torch.tanh(f4 * (z - f5))

    @torch.no_grad()
    def peak_force(self, feat: torch.Tensor | None = None, z_max: float = 30.0) -> float:
        """
        Max |F| over the slip range — the fitted 'mu·Fz' to check against the envelope.

        For a conditioned curve this is the peak *at one operating point*; ``feat``
        defaults to the zero state (parked, straight), which is the state the untrained
        bias describes.
        """
        z = torch.linspace(-z_max, z_max, 4001)
        if self.net is not None:
            if feat is None:
                feat = torch.zeros(len(self.feat_scale))
            feat = feat.reshape(-1, feat.shape[-1]).expand(z.shape[0], -1)
        return float(self.forward(z, feat).abs().max())


class _DirectionNet(nn.Module):
    """
    The paper's learned friction-circle direction NN0.

    The rear force points along ``d/||d||`` with ``d = NN0(alpha_r, sigma_r)``.

    ``residual=True`` (the default) instead adds the network to the geometric slip
    direction and zero-initializes its output layer, so an untrained model starts at
    exactly the friction circle the slip vector defines and training only learns the
    correction. ``residual=False`` is the paper's literal form — the direction *is* the
    network output — which needs a normal (non-zero) init, since a zero-initialized
    ``d`` has no direction to normalize.
    """

    def __init__(
        self,
        hidden: int = _NN0_HIDDEN,
        layers: int = _NN0_LAYERS,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = bool(residual)
        self.net = _mlp(2, hidden, layers, 2)
        if self.residual:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, alpha_r: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.net(torch.stack([alpha_r, sigma], dim=-1))


def build_curves(
    fzf: float, fzr: float, tire_features: bool, wheel_noload_speed: float
) -> tuple[_ExpTanhCurve, _ExpTanhCurve]:
    """
    Physically-initialized ``(front, rear)`` ExpTanh curves.

    Front input is tan(alpha_f), linear in the slip; the rear input is the paper's
    *squared* combined slip z = tan²(alpha_r) + f0·sigma². The two ramp rates are
    therefore set independently — see ``_REAR_SLIP_SAT`` for why the rear's cannot be
    the front's squared. With ``tire_features`` the coefficients are conditioned on the
    state ([r, V, beta] front, plus the rear wheel speed at the rear); both zero-init to
    the fixed curve so an untrained conditioned model is identical to the fixed one.
    """
    f4_front = 1.0 / (_TAN_SLIP_SAT * _MU_INIT)
    f4_rear = _TANH_99 / _REAR_SLIP_SAT**2
    front_feat, rear_feat = None, None
    if tire_features:
        front_feat = _FEAT_SCALE_RV_BETA
        rear_feat = (*_FEAT_SCALE_RV_BETA, float(wheel_noload_speed))
    front = _ExpTanhCurve(peak=_MU_INIT * fzf, f4=f4_front, feat_scale=front_feat)
    rear = _ExpTanhCurve(
        peak=_MU_INIT * fzr,
        f4=f4_rear,
        f5=_REAR_F5_INIT,
        positive_f5=True,
        feat_scale=rear_feat,
    )
    return front, rear
