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
    Add,
    AdiabaticPump,
    AmbientInlet,
    AmbientOutlet,
    CompressibleValve,
    ConjugatePipe,
    Constant,
    ConvectiveBoundary,
    CylindricalWall,
    Feedback,
    FirstOrder,
    FixedHeatFlow,
    FixedTemperature,
    FlatWall,
    FluidPort_phm,
    Gain,
    HeatedSegment,
    IncompressibleValve,
    Integrator,
    Limiter,
    LoopBuffer,
    MixingJunction,
    PID,
    PressureOutlet,
    PressureSource,
    PressureVessel,
    Product,
    Ramp,
    RealSignal,
    Sine,
    Splitter,
    Step,
    StraightPipe,
    Sum,
    ThermalConductor,
    ThermalPort_TQ,
    TwoNodeWall,
    TwoPortSegment,
    Valve,
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
    "Valve",
    "IncompressibleValve",
    "CompressibleValve",
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
    # control-library port
    "RealSignal",
    # control components
    "Constant",
    "Step",
    "Ramp",
    "Sine",
    "Gain",
    "Add",
    "Feedback",
    "Sum",
    "Product",
    "Limiter",
    "Integrator",
    "FirstOrder",
    "PID",
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
    # serialization
    "to_dict",
    "from_dict",
    "to_json",
    "from_json",
]

__version__ = "0.1.0"

# Imported last: the serialization subpackage reads `__version__` and the
# fully-populated component registry, both of which must exist by import time.
from .serialization import from_dict, from_json, to_dict, to_json  # noqa: E402
