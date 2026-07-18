"""The demo recorder's recovery gate must only re-open on track and in motion."""

import pytest

# Importing the node pulls in shared_interfaces (a colcon-built package); skip the
# whole module rather than error when the workspace has not been built yet.
try:
    from agent_pkg.demo_recorder_node import RECOVERY_LATERAL_FRACTION, is_recovered
    from drift_core.environments.rewards import TerminationConfig
except ImportError:
    pytest.skip("shared_interfaces not built", allow_module_level=True)

TERM = TerminationConfig(lateral_limit_m=0.4, stall_speed=0.1)


def test_recovered_on_track_and_moving():
    assert is_recovered(lateral_error=0.1, speed=1.0, term_cfg=TERM)


def test_not_recovered_while_standing():
    assert not is_recovered(lateral_error=0.0, speed=TERM.stall_speed, term_cfg=TERM)


def test_not_recovered_off_track():
    assert not is_recovered(lateral_error=0.45, speed=1.0, term_cfg=TERM)


def test_recovery_needs_margin_inside_the_limit():
    # Just inside the termination limit is not enough — the gate re-opens only
    # below RECOVERY_LATERAL_FRACTION of it, so the boundary cannot chatter.
    just_inside = 0.99 * TERM.lateral_limit_m
    assert not is_recovered(lateral_error=just_inside, speed=1.0, term_cfg=TERM)
    within_margin = 0.99 * RECOVERY_LATERAL_FRACTION * TERM.lateral_limit_m
    assert is_recovered(lateral_error=within_margin, speed=1.0, term_cfg=TERM)
    assert is_recovered(lateral_error=-within_margin, speed=1.0, term_cfg=TERM)
