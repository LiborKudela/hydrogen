# Fluid components

Compressible-fluid plumbing for the `hydrogen` solver. This domain
provides reusable components built on `hydrogen.model` and a single typed
connector that every component exposes.

## Layout

```
fluid/
├── __init__.py          # public API re-exports
├── fluid_components.py   # port + component implementations
└── README.md             # this file
```

The domain owns its own connector rather than putting it in the generic
`hydrogen.ports`. `hydrogen.ports` only defines the generic `Port` base
class and the shared error hierarchy; each physics domain declares its
own port kind(s) next to the components that use them.

## Connector

- **`FluidPort_phm`** — compressible-fluid interface carrying `(p, h, m_dot)`:
  - `p` — port pressure `[Pa]` (across)
  - `h` — port specific enthalpy `[J/kg]` (across)
  - `m_dot` — port mass flow rate `[kg/s]` (through; positive = "into me",
    Modelica "flow into me" convention)

  Connecting two ports with mismatched `medium` is refused at connect-time
  (`PortMediumMismatchError`) to catch cross-wiring early.

## Components

| Class | Purpose |
| --- | --- |
| `AmbientInlet` | Mass-flow-imposed inlet matched to ambient `(p, T)` conditions. |
| `AmbientOutlet` | Outlet that discharges to ambient conditions. |
| `TwoPortSegment` | Generic (adiabatic) inlet/outlet segment base for pipe-like elements. |
| `HeatedSegment` | `TwoPortSegment` with a `wall` `ThermalPort_TQ` for heat exchange. |
| `AdiabaticPump` | Pump with adiabatic compression. |
| `StraightPipe` | Straight-pipe wrapper over `TwoPortSegment` / `HeatedSegment`. |

## Heat transfer (`heat_port`)

`StraightPipe` is adiabatic by default. Build it with `heat_port=True` to make
every segment a `HeatedSegment` that exposes a `wall` `ThermalPort_TQ`
(`T`, `Q_dot`); the per-segment ports are available via
`pipe.segment_wall_ports`. Connect each to a thermal boundary or wall:

- a `thermal.FixedTemperature` reproduces a prescribed wall temperature
  (the replacement for the removed legacy fixed-`T_wall` parameter), or
- a `thermal.CylindricalWall` for full conjugate heat transfer — see the
  `power.ConjugatePipe` component, which wires this up automatically.

`T_wall` is always an algebraic `Variable`: adiabatically it is closed by the
identity `T_wall = T_avg` (so the heat term vanishes); with `heat_port=True`
it is set by whatever is connected to the `wall` port. A `wall` port left
unconnected raises a `PortNotConnectedWarning` at `instantiate()`. The
fluid energy balance adds the wall heat as `q / m_dot` `[J/kg]`, so the fluid
gains exactly `q` watts and energy is conserved across the interface.

The deprecated `adiabatic=` flag still constructs (it only ever toggled a
fixed-293.15 K wall, now removed); prefer `heat_port=`.
| `PressureSource` | Imposes a boundary pressure. |
| `PressureOutlet` | Pressure-imposed outlet boundary. |
| `PressureVessel` | Lumped pressure/enthalpy storage volume. |
| `Splitter` | One inlet to multiple outlets. |
| `MixingJunction` | Multiple inlets mixed into one outlet. |
| `LoopBuffer` | Loop-closing buffer for recirculating systems. |

## Usage

The public API is re-exported at the package level, so call sites stay
stable:

```python
from hydrogen.components import StraightPipe, FluidPort_phm
# equivalently:
from hydrogen.components.fluid import StraightPipe, FluidPort_phm
```

## Adding a new physics domain

Create a sibling subfolder (e.g. `thermal/`) containing the domain's
Python module(s) and a `README.md`, declare the domain's own `Port`
subclass(es) inside it, then re-export its public symbols from
`hydrogen/components/__init__.py`.
