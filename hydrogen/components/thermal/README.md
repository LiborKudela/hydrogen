# Thermal (heat-transfer) components

Lumped heat-transfer components for the `hydrogen` solver. This domain
provides a plane-wall conduction model, a small set of boundary
conditions to drive a thermal network, and a single typed connector that
every component exposes.

## Layout

```
thermal/
├── __init__.py            # public API re-exports
├── thermal_components.py   # port + boundary conditions + FlatWall
└── README.md               # this file
```

Like every physics domain, thermal owns its own connector rather than
putting it in the generic `hydrogen.ports`. `hydrogen.ports` defines only
the generic `Port` base class and the shared error hierarchy.

## Connector

- **`ThermalPort_TQ`** — heat-transfer interface carrying `(T, Q_dot)`:
  - `T` — port temperature `[K]` (across)
  - `Q_dot` — heat flow rate `[W]` (through; positive = "into me",
    Modelica "flow into me" convention)

  Every component uses `flow_orientation='in'`, so connecting two ports
  emits a sum-to-zero on `Q_dot` (`Q_dot_a + Q_dot_b == 0`) — the heat
  the source delivers is the heat the sink receives.

## Boundary conditions

| Class | Purpose | Pins |
| --- | --- | --- |
| `FixedTemperature` | Temperature reservoir at `T_set`. | `T_port = T_set`; `Q_dot` free. |
| `FixedHeatFlow` | Injects `Q_flow` W into the connected component (use `0` for an insulated/adiabatic face). | `Q_dot` so partner receives `Q_flow`; `T` free. |
| `ConvectiveBoundary` | Newton cooling to a far field: `Q_into_partner = h·A·(T_inf − T_surface)`. | `Q_dot` via the convective law; `T` follows the surface. |

> **Connecting a `FixedTemperature` to a capacitive node.** Wiring a
> prescribed temperature *directly* onto a `FlatWall` surface (a heat
> capacity) is a high-index DAE constraint and is singular at `t = 0`.
> Drive the capacity through a conductance instead: put a
> `ThermalConductor` between the `FixedTemperature` and the wall. A
> `ConvectiveBoundary` is already such a conductance-to-a-reservoir, so it
> may connect to a surface directly.

## Passive elements

| Class | Purpose |
| --- | --- |
| `ThermalConductor` | Massless conductance, `Q = G·(T_a − T_b)`. Use `G = k·A/L` for a slab, `G = 1/R` for a contact resistance. |

## Components

### `TwoNodeWall` (base class)

Both wall models are the same conceptually — a wall lumped into **two
surface nodes** with conduction between them — and differ only in the
per-node capacity and the node-to-node conductance. `TwoNodeWall` holds
everything shared (the two temperature states, the two heat ports, and the
first-law energy balance); a subclass supplies just three hooks:
`_declare_geometry()` (register its geometry parameters), `_node_capacity()`
(return `C_node`), and `_conductance()` (return `G`). Subclass it to add
other shapes (e.g. a spherical shell) without duplicating the wall
machinery.

#### `dynamic` flag (transient vs quasi-static)

Both `FlatWall` and `CylindricalWall` accept `dynamic` (default `True`):

- **`dynamic=True`** — each node carries thermal mass, so the surface
  temperatures are **differential** states and the wall stores energy
  (heat-up transient):
  `C_node · dT/dt = Q_dot − G·ΔT`.
- **`dynamic=False`** — quasi-static: the node capacities and their ODEs are
  removed, so the wall is a pure (massless) conductance and the surface
  temperatures are **algebraic** states (instantaneous steady conduction):
  `0 = Q_dot_a − G·(T_a − T_b)`, `0 = Q_dot_b − G·(T_b − T_a)` (hence
  `Q_dot_a + Q_dot_b = 0`). A `FixedTemperature` may be wired *straight*
  onto a quasi-static face — with no capacity it is not a high-index
  constraint, so no `ThermalConductor` is needed.

### `FlatWall`

A plane wall (slab) of area `A` and thickness `L` made of a material with
density `rho`, specific heat `cp`, and thermal conductivity `k`. It is
lumped into **two surface nodes** with conduction between them:

```
port_a  ->  [ C_node | T_a ]== G ==[ T_b | C_node ]  <-  port_b
                surface A                  surface B
```

- Each surface node owns half the slab's thermal mass:
  `C_node = rho · cp · A · L / 2` `[J/K]`
- The nodes are linked by the full-thickness conductance:
  `G = k · A / L` `[W/K]`

Differential states: the two surface temperatures `T_a`, `T_b`.
Algebraic states: the port heat flows `Q_dot_a`, `Q_dot_b`.

Energy balance (first law) at each surface node:

```
C_node · dT_a/dt = Q_dot_a − G·(T_a − T_b)
C_node · dT_b/dt = Q_dot_b − G·(T_b − T_a)
```

`rho`, `cp`, `k`, `A`, `L` are all parameters; pass plain scalars or an
existing `Parameter` to share a parent's symbol.

### `CylindricalWall`

The circular (hollow-tube) counterpart of `FlatWall`: an annular wall of
length `length` between inner radius `r_in` and outer radius `r_out`,
lumped into an inner-surface node (`port_a`) and an outer-surface node
(`port_b`) with **radial** conduction between them. The energy balance,
states, and ports are identical to `FlatWall`; only the geometry-derived
thermal mass and conductance change:

- Annular thermal mass split across the two nodes:
  `C_node = rho · cp · pi·(r_out² − r_in²)·length / 2` `[J/K]`
- Exact radial conductance of a cylindrical shell:
  `G = 2·pi·k·length / ln(r_out / r_in)` `[W/K]`

This `2·pi·k·length / ln(r_out/r_in)` is the cylindrical analogue of the
slab's `k·A/L`. Requires `r_out > r_in > 0`. Parameters: `rho`, `cp`, `k`,
`r_in`, `r_out`, `length`.

## Usage

The public API is re-exported at the package level:

```python
from hydrogen.components import FlatWall, FixedTemperature, FixedHeatFlow
# equivalently:
from hydrogen.components.thermal import FlatWall, ThermalPort_TQ
```

### Steady-state conduction (analytical check)

Drive the wall between two temperature reservoirs *through* conductors
(`FixedTemperature → ThermalConductor → FlatWall → ThermalConductor →
FixedTemperature`). The steady heat flow is the series-resistance result,
with the wall contributing its conduction resistance `L/(k·A)`:

```
Q = (T_hot − T_cold) / (1/G_a + L/(k·A) + 1/G_b)
```

### Transient (one face heated, the other insulated)

With `FixedHeatFlow(Q_in)` on `port_a` and `FixedHeatFlow(0)` on `port_b`,
the conduction terms cancel in the node-sum energy balance, so the wall's
mean temperature rises linearly:

```
d/dt (T_a + T_b)/2 = Q_in / (rho · cp · A · L)
```

while the surface-to-surface temperature difference relaxes toward
`Q_in · L / (2 · k · A)` with time constant `C_node · L / (2 · k · A)`.

See `examples/flat_wall.py` for a runnable demonstration and
`tests/test_flat_wall.py` for the analytical assertions.
