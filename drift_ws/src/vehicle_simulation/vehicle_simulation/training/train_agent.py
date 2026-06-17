"""
Train an RL agent in simulation (SCAFFOLD; not implemented yet).

Prepared structure for RL-agent training; not implemented — the current focus is
the vehicle models. Intended approach when RL work begins: build a vehicle model
via ``build_model`` (a trained run directory, or a baseline such as ``arcade`` /
``dynamic``), wrap it in :class:`~vehicle_simulation.environments.drift_env.DriftEnv`,
and train a policy (e.g. Stable-Baselines3) — reusing the shared
``environments/observation.py`` builder so simulation and real-car observations
match (sim-to-real). Requires the ``[rl]`` extra (gymnasium + stable-baselines3).
"""


def main(argv=None) -> None:
    """Entry point (not implemented yet)."""
    raise NotImplementedError(
        "RL-agent training is not implemented yet (scaffold). Current focus: the "
        "vehicle models. See environments/drift_env.py and the framework plan doc."
    )


if __name__ == "__main__":
    main()
