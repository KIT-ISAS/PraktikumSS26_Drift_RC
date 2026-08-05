"""Argument handling of the evaluation CLI (bare ``--mcap-bag``), torch-guarded."""

import pytest

pytest.importorskip("torch")

from drift_core.vehicle_model.evaluation.evaluate import (  # noqa: E402
    _eval_spec,
    _parse_args,
    _resolve_mcap_bag,
)


def _resolve(argv):
    args = _parse_args(argv)
    return _resolve_mcap_bag(args, _eval_spec(args))


def test_no_mcap_bag_renders_nothing():
    assert _resolve(["--run-dir", "r", "--bags", "b1"]) is None


def test_bare_mcap_bag_reuses_the_lone_eval_bag():
    assert _resolve(["--run-dir", "r", "--bags", "b1", "--mcap-bag"]) == "b1"


def test_bare_mcap_bag_survives_a_following_option():
    # nargs="?" must not swallow the next flag's value as the bag path.
    argv = ["--run-dir", "r", "--bags", "b1", "--mcap-bag", "--mcap-start", "5"]
    assert _resolve(argv) == "b1"
    assert _parse_args(argv).mcap_start == [5]


def test_bare_mcap_bag_is_ambiguous_for_several_eval_bags():
    with pytest.raises(SystemExit):
        _resolve(["--run-dir", "r", "--bags", "b1", "b2", "--mcap-bag"])


def test_explicit_mcap_bag_wins():
    assert _resolve(["--run-dir", "r", "--bags", "b1", "b2", "--mcap-bag", "b2"]) == "b2"
