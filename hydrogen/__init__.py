"""Hydrogen: symbolic DAE/ODE solver for fluid-system dynamics.

Public API:

    from hydrogen import (
        Model, Parameter, Variable, DifferentialVariable, Input,  # framework
        CoolPropMedium,                                          # medium
        AmbientInlet, AmbientOutlet, TwoPortSegment,             # fluid components
        AdiabaticPump, StraightPipe, LoopBuffer, MixingJunction,
        FlatWall, CylindricalWall, SphericalWall, FixedTemperature,  # thermal components
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
    AISI_304,
    AISI_316,
    H2,
    H2_IN_AUSTENITIC,
    H2_IN_AISI_304,
    H2_IN_AISI_316,
    HELIUM,
    NITROGEN,
    Add,
    AdiabaticPump,
    AmbientInlet,
    AmbientOutlet,
    ClosedEnd,
    CompressibleValve,
    ConjugatePipe,
    Constant,
    ConvectiveBoundary,
    CylindricalWall,
    Feedback,
    FirstOrder,
    FixedHeatFlow,
    FixedPartialPressure,
    FixedTemperature,
    FlatWall,
    FluidPort_phm,
    Gain,
    IncompressibleValve,
    Integrator,
    Limiter,
    LoopBuffer,
    MixingJunction,
    PID,
    Permeant,
    PermeationFlux,
    PermeationPort_pN,
    Pipe,
    PressureOutlet,
    PressureSource,
    PressureVessel,
    Product,
    Ramp,
    RealSignal,
    Sine,
    Splitter,
    SphericalWall,
    SteadyRichardson,
    Step,
    StraightPipe,
    Sum,
    Tank,
    ThermalConductor,
    ThermalPort_TQ,
    TransientDiffusion,
    TransportFit,
    TwoNodeWall,
    TwoPortSegment,
    Valve,
    WallLayer,
    WallMaterial,
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
from .paramspec import ParamSpec
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
    "ParamSpec",
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
    "ClosedEnd",
    "TwoPortSegment",
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
    "SphericalWall",
    # composite assemblies
    "Pipe",
    "Tank",
    "WallLayer",
    # power components
    "ConjugatePipe",
    # permeation-library port
    "PermeationPort_pN",
    # permeation materials
    "Permeant",
    "TransportFit",
    "H2",
    "HELIUM",
    "NITROGEN",
    "H2_IN_AUSTENITIC",
    "H2_IN_AISI_304",
    "H2_IN_AISI_316",
    # permeation models / boundary
    "PermeationFlux",
    "SteadyRichardson",
    "TransientDiffusion",
    "FixedPartialPressure",
    # wall materials
    "WallMaterial",
    "AISI_304",
    "AISI_316",
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
    "component_catalog",
    "component_spec",
    "format_component_catalog",
    "available_domains",
    "value_object_catalog",
    "value_object_spec",
    "spec_template",
    "value_template",
    # out-of-process host / UI controller
    "start_host",
    "HostService",
    "SystemProxy",
]

__version__ = "0.1.0"

# Imported last: the serialization subpackage reads `__version__` and the
# fully-populated component registry, both of which must exist by import time.
from .serialization import (  # noqa: E402
    available_domains,
    component_catalog,
    component_spec,
    format_component_catalog,
    from_dict,
    from_json,
    spec_template,
    to_dict,
    to_json,
    value_object_catalog,
    value_object_spec,
    value_template,
)

# Client-side host launcher. `service.client` is stdlib-only (no heavy hydrogen
# imports), so this stays import-cheap; the host engine is loaded lazily only in
# the `python -m hydrogen.service` subprocess.
from .service import HostService, SystemProxy, start_host  # noqa: E402
