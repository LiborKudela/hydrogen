"""Hydrogen: symbolic DAE/ODE solver for fluid-system dynamics.

Public API:

    from hydrogen import (
        Model, Parameter, Variable, DifferentialVariable,        # framework
        CoolPropMedium,                                          # medium
        AmbientInlet, AmbientOutlet, TwoPortSegment,             # components
        AdiabaticPump, StraightPipe,
        IntegrationTest, SimpleODE, InnerODE_1, InnerODE_2,      # ODE test sub-models
        plot_results,                                            # plotting
    )

Lower-level helpers (`numpy_cache`, `lambdify_compat`, `fast_*`) live in their
respective submodules; import directly from there if you need them.
"""

from .components import (
    AdiabaticPump,
    AmbientInlet,
    AmbientOutlet,
    PressureOutlet,
    PressureSource,
    PressureVessel,
    Splitter,
    StraightPipe,
    TwoPortSegment,
)
from .medium import CoolPropMedium, get_symbolic_property_function
from .model import DifferentialVariable, Model, Parameter, Variable
from .plotting import plot_results
from .test_models import (
    InnerODE_1,
    InnerODE_2,
    IntegrationTest,
    SimpleODE,
)

__all__ = [
    # framework
    "Model",
    "Parameter",
    "Variable",
    "DifferentialVariable",
    # medium
    "CoolPropMedium",
    "get_symbolic_property_function",
    # components
    "AmbientInlet",
    "AmbientOutlet",
    "TwoPortSegment",
    "AdiabaticPump",
    "StraightPipe",
    "PressureSource",
    "PressureOutlet",
    "PressureVessel",
    "Splitter",
    # ODE test sub-models
    "IntegrationTest",
    "SimpleODE",
    "InnerODE_1",
    "InnerODE_2",
    # plotting
    "plot_results",
]

__version__ = "0.1.0"
