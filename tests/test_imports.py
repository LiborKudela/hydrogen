"""Smoke tests: package and submodule surfaces resolve without errors."""

from __future__ import annotations


def test_top_level_attributes():
    import hydrogen

    expected = {
        "Model",
        "Parameter",
        "Variable",
        "DifferentialVariable",
        "NewtonConvergenceFailure",
        "CoolPropMedium",
        "AmbientInlet",
        "AmbientOutlet",
        "TwoPortSegment",
        "AdiabaticPump",
        "StraightPipe",
        "PressureSource",
        "PressureOutlet",
        "PressureVessel",
        "LoopBuffer",
        "Splitter",
        "IntegrationTest",
        "SimpleODE",
        "InnerODE_1",
        "InnerODE_2",
        "plot_results",
        "local_results_path",
    }
    missing = expected - set(hydrogen.__all__)
    assert not missing, f"missing public exports: {missing}"
    for name in expected:
        assert hasattr(hydrogen, name), f"hydrogen.{name} not importable"


def test_submodule_imports():
    from hydrogen.caching import ModelCache, hash_args, numpy_cache  # noqa: F401
    from hydrogen.components import StraightPipe  # noqa: F401
    from hydrogen.medium import CoolPropMedium  # noqa: F401
    from hydrogen.model import Model, Variable  # noqa: F401
    from hydrogen.numerics import G_const, fast_error_norm, lambdify_compat  # noqa: F401
    from hydrogen.plotting import plot_results  # noqa: F401
    from hydrogen.test_models import IntegrationTest  # noqa: F401
