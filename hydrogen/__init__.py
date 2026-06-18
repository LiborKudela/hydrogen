"""Hydrogen: symbolic DAE/ODE solver for fluid-system dynamics.

Public API:

    from hydrogen import (
        Model, Parameter, Variable, DifferentialVariable, Input,  # framework
        CoolPropMedium,                                          # medium
        AmbientInlet, AmbientOutlet, TwoPortSegment,             # fluid components
        AdiabaticPump, StraightPipe, LoopBuffer, MixingJunction,
        FlatWall, CylindricalWall, FixedTemperature,             # thermal components
        FixedHeatFlow, ConvectiveBoundary, ThermalConductor,
        ConjugatePipe,                                           # power components
        Interpolation1D, Interpolation2D,                        # interpolation utilities
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
    ConjugatePipe,
    ConvectiveBoundary,
    CylindricalWall,
    FixedHeatFlow,
    FixedTemperature,
    FlatWall,
    FluidPort_phm,
    HeatedSegment,
    LoopBuffer,
    MixingJunction,
    PressureOutlet,
    PressureSource,
    PressureVessel,
    Splitter,
    StraightPipe,
    ThermalConductor,
    ThermalPort_TQ,
    TwoNodeWall,
    TwoPortSegment,
)
from .medium import CoolPropMedium, get_symbolic_property_function
from .model import (
    DifferentialVariable,
    EquationCacheValidationError,
    Input,
    Model,
    NewtonConvergenceFailure,
    Parameter,
    Variable,
    set_equation_cache_validation,
)
from .plotting import local_results_path, plot_results
from .ports import (
    Port,
    PortAlreadyConnectedError,
    PortChannelMissingError,
    PortError,
    PortKindMismatchError,
    PortMediumMismatchError,
)
from .test_models import (
    InnerODE_1,
    InnerODE_2,
    IntegrationTest,
    SimpleODE,
)
from .utilities import Interpolation1D, Interpolation2D

__all__ = [
    # framework
    "Model",
    "Parameter",
    "Variable",
    "DifferentialVariable",
    "Input",
    "NewtonConvergenceFailure",
    "EquationCacheValidationError",
    "set_equation_cache_validation",
    # medium
    "CoolPropMedium",
    "get_symbolic_property_function",
    # port machinery (generic)
    "Port",
    "PortError",
    "PortAlreadyConnectedError",
    "PortKindMismatchError",
    "PortChannelMissingError",
    "PortMediumMismatchError",
    # fluid-library ports
    "FluidPort_phm",
    # fluid components
    "AmbientInlet",
    "AmbientOutlet",
    "TwoPortSegment",
    "HeatedSegment",
    "AdiabaticPump",
    "StraightPipe",
    "PressureSource",
    "PressureOutlet",
    "PressureVessel",
    "LoopBuffer",
    "MixingJunction",
    "Splitter",
    # thermal-library ports
    "ThermalPort_TQ",
    # thermal components
    "FixedTemperature",
    "FixedHeatFlow",
    "ConvectiveBoundary",
    "ThermalConductor",
    "TwoNodeWall",
    "FlatWall",
    "CylindricalWall",
    # power components
    "ConjugatePipe",
    # ODE test sub-models
    "IntegrationTest",
    "SimpleODE",
    "InnerODE_1",
    "InnerODE_2",
    # utilities
    "Interpolation1D",
    "Interpolation2D",
    # plotting
    "plot_results",
    "local_results_path",
]

__version__ = "0.1.0"
