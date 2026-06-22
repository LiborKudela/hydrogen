"""Smoke tests: package and submodule surfaces resolve without errors."""

from __future__ import annotations


def test_top_level_attributes():
    import hydrogen

    expected = {
        "Model",
        "Parameter",
        "Variable",
        "DifferentialVariable",
        "Input",
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
        "MixingJunction",
        "Splitter",
        "ThermalPort_TQ",
        "FixedTemperature",
        "FixedHeatFlow",
        "ConvectiveBoundary",
        "ThermalConductor",
        "TwoNodeWall",
        "FlatWall",
        "CylindricalWall",
        "ConjugatePipe",
        "IntegrationTest",
        "SimpleODE",
        "InnerODE_1",
        "InnerODE_2",
        "Interpolation1D",
        "Interpolation2D",
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
    from hydrogen.components.power import ConjugatePipe  # noqa: F401
    from hydrogen.components.thermofluid import (  # noqa: F401
        CylindricalWall,
        FluidPort_phm,
        PermeationPort_pN,
        ThermalPort_TQ,
        TwoPortSegment,
    )
    from hydrogen.components.thermofluid.flow import StraightPipe as _SP  # noqa: F401
    from hydrogen.components.thermofluid.permeation import (  # noqa: F401
        SteadyRichardson,
        TransientDiffusion,
    )
    from hydrogen.components.thermofluid.walls import FlatWall  # noqa: F401
    from hydrogen.medium import CoolPropMedium  # noqa: F401
    from hydrogen.model import Model, Variable  # noqa: F401
    from hydrogen.numerics import G_const, fast_error_norm, lambdify_compat  # noqa: F401
    from hydrogen.plotting import plot_results  # noqa: F401
    from hydrogen.test_models import IntegrationTest  # noqa: F401
    from hydrogen.utilities import Interpolation1D, Interpolation2D  # noqa: F401
    from hydrogen.utilities.interpolation import Interpolation2D as _I2  # noqa: F401
