"""Hydrogen: symbolic DAE/ODE solver for fluid-system dynamics.

The top-level package exposes the modelling *framework* and tooling:

    from hydrogen import (
        Model, Parameter, Variable, DifferentialVariable, Input,  # framework
        CoolPropMedium, FeosMedium,                              # media
        Interpolation1D, Interpolation2D,                        # interpolation utilities
        IntegrationTest, SimpleODE, InnerODE_1, InnerODE_2,      # ODE test sub-models
        plot_results,                                            # plotting
        component_catalog, component_spec,                       # serialization tooling
    )

Components are **not** re-exported here.  Import each from the module where it
is defined, so the import path mirrors the package layout::

    from hydrogen.components.thermofluid.assemblies import Pipe, Tank
    from hydrogen.components.thermofluid.flow import StraightPipe, Valve
    from hydrogen.components.thermofluid.walls import CylindricalWall
    from hydrogen.components.thermofluid.ports import FluidPort_phm
    from hydrogen.components.power.power_components import ConjugatePipe
    from hydrogen.components.control.control_components import PID, Gain

Use ``hydrogen.component_catalog()`` to enumerate everything that ships.

Lower-level helpers (`numpy_cache`, `lambdify_compat`, `fast_*`) live in their
respective submodules; import directly from there if you need them.
"""

from .medium import CoolPropMedium, FeosMedium, get_symbolic_property_function
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
    "FeosMedium",
    "get_symbolic_property_function",
    # port machinery (generic)
    "Port",
    "PortError",
    "PortAlreadyConnectedError",
    "PortKindMismatchError",
    "PortChannelMissingError",
    "PortMediumMismatchError",
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
