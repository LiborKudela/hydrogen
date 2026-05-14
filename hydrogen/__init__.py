"""Hydrogen: symbolic DAE/ODE solver for fluid-system dynamics.

Public API:

    from hydrogen import (
        Model, Parameter, Variable, DifferentialVariable,        # framework
        CoolPropMedium,                                          # medium
        AmbientInlet, AmbientOutlet, TwoPortSegment,             # components
        AdiabaticPump, StraightPipe, LoopBuffer, MixingJunction,
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
    LoopBuffer,
    MixingJunction,
    PressureOutlet,
    PressureSource,
    PressureVessel,
    Splitter,
    StraightPipe,
    TwoPortSegment,
)
from .medium import CoolPropMedium, get_symbolic_property_function
from .model import (
    DifferentialVariable,
    Model,
    NewtonConvergenceFailure,
    Parameter,
    Variable,
)
from .plotting import local_results_path, plot_results
from .ports import (
    ElectricalPort_VI,
    FluidPort_phm,
    Port,
    PortAlreadyConnectedError,
    PortChannelMissingError,
    PortError,
    PortKindMismatchError,
    PortMediumMismatchError,
    ThermalPort_TQ,
)
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
    "NewtonConvergenceFailure",
    # medium
    "CoolPropMedium",
    "get_symbolic_property_function",
    # ports
    "Port",
    "FluidPort_phm",
    "ThermalPort_TQ",
    "ElectricalPort_VI",
    "PortError",
    "PortAlreadyConnectedError",
    "PortKindMismatchError",
    "PortChannelMissingError",
    "PortMediumMismatchError",
    # components
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
    # ODE test sub-models
    "IntegrationTest",
    "SimpleODE",
    "InnerODE_1",
    "InnerODE_2",
    # plotting
    "plot_results",
    "local_results_path",
]

__version__ = "0.1.0"
