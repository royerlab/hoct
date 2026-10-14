"""Tests for the configurable ILP solver backend preference.

These tests mock tracksdata's solver so they do not depend on the installed
tracksdata version carrying the ``solver_preference`` parameter (PR #343).
"""

import hoct.tracking._solver as solver_module
from hoct.tracking import ILPSolverConfig, solve_tracking


class _FakeGraph:
    """Minimal graph stub for ``solve_tracking`` that touches no real solver."""

    def __init__(self):
        self.metadata = {}

    def node_attr_keys(self):
        return []


def _base_config_kwargs():
    return {
        "appearance_weight": 1.0,
        "disappearance_weight": 1.0,
        "division_weight": 1.0,
        "node_weight": 0.0,
        "delta_t_weight": 0.1,
        "edge_bias": 0.0,
        "timeout": 10.0,
    }


def test_config_accepts_solver_preference_str():
    config = ILPSolverConfig(**_base_config_kwargs(), tracklet_solver=False, solver_preference="gurobi")
    assert config.solver_preference == "gurobi"
    assert config.model_dump()["solver_preference"] == "gurobi"


def test_config_accepts_solver_preference_list():
    prefs = ["gurobi", "cuopt", "scip"]
    config = ILPSolverConfig(**_base_config_kwargs(), tracklet_solver=False, solver_preference=prefs)
    assert config.solver_preference == prefs
    assert config.model_dump()["solver_preference"] == prefs


def test_config_default_solver_preference_is_none():
    config = ILPSolverConfig(**_base_config_kwargs(), tracklet_solver=False)
    assert config.solver_preference is None
    # ``default()`` must not set a preference.
    assert ILPSolverConfig.default().solver_preference is None


def _install_recording_solvers(monkeypatch):
    """Replace the real ILPSolver/TrackletSolver with recording fakes.

    Returns a list that collects the constructor kwargs of every solver built
    by ``solve_tracking``.
    """
    records = []

    def make_fake(name):
        class _FakeSolver:
            def __init__(self, **kwargs):
                records.append((name, kwargs))

            def solve(self, graph):
                return graph

        return _FakeSolver

    monkeypatch.setattr(solver_module.td.solvers, "ILPSolver", make_fake("ILPSolver"))
    monkeypatch.setattr(solver_module, "TrackletSolver", make_fake("TrackletSolver"))
    return records


def test_forwards_solver_preference_single_pass(monkeypatch):
    records = _install_recording_solvers(monkeypatch)
    config = ILPSolverConfig(**_base_config_kwargs(), tracklet_solver=False, solver_preference="cuopt")

    solve_tracking(graph=_FakeGraph(), config=config)

    assert len(records) == 1
    name, kwargs = records[0]
    assert name == "ILPSolver"
    assert kwargs["solver_preference"] == "cuopt"


def test_forwards_solver_preference_two_pass(monkeypatch):
    records = _install_recording_solvers(monkeypatch)
    prefs = ["gurobi", "scip"]
    config = ILPSolverConfig(**_base_config_kwargs(), tracklet_solver=True, solver_preference=prefs)

    solve_tracking(graph=_FakeGraph(), config=config)

    # Two solvers are built: pass-1 ILPSolver and the TrackletSolver linkage pass.
    assert len(records) == 2
    built = {name for name, _ in records}
    assert built == {"ILPSolver", "TrackletSolver"}
    for _name, kwargs in records:
        assert kwargs["solver_preference"] == prefs


def test_omits_solver_preference_when_none(monkeypatch):
    records = _install_recording_solvers(monkeypatch)
    config = ILPSolverConfig(**_base_config_kwargs(), tracklet_solver=True, solver_preference=None)

    solve_tracking(graph=_FakeGraph(), config=config)

    assert len(records) == 2
    for _name, kwargs in records:
        assert "solver_preference" not in kwargs
