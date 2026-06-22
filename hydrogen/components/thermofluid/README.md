# Thermo-fluid components

A single domain for everything modelled around **pipes, vessels, and walls**:
compressible flow, wall heat conduction, and gas permeation. These compose
into the same physical objects — a heated, leaky pipe is one component, not
three — so they share one library and one set of connectors.

## Layout

```
thermofluid/
├── __init__.py     # public API re-exports
├── ports.py        # typed connectors (FluidPort_phm, ThermalPort_TQ, PermeationPort_pN)
├── flow.py         # compressible-flow components
├── walls.py        # lumped wall conduction + thermal boundaries
├── permeation.py   # permeation materials, flux models + partial-pressure boundary
├── assemblies.py   # composites (Pipe = flow pipe + per-segment wall stack)
└── README.md       # this file
```

`hydrogen.ports` only defines the generic `Port` base class and the shared
error hierarchy; this domain declares its own connector kinds in `ports.py`,
next to the components that use them.

## Connectors (`ports.py`)

- **`FluidPort_phm`** — compressible flow, `(p, h, m_dot)`. Mismatched `medium`
  is refused at connect-time (`PortMediumMismatchError`).
- **`ThermalPort_TQ`** — heat transfer, `(T, Q_dot)`. No `medium`.
- **`PermeationPort_pN`** — gas permeation leak, `(p_partial, m_dot_leak)`. No
  `medium`. Exposed by both a leaky `flow.TwoPortSegment` and any leaky
  `walls.TwoNodeWall` (`FlatWall` / `CylindricalWall` / `SphericalWall`).

All flow channels (`m_dot`, `Q_dot`, `m_dot_leak`) are positive **into** the
component ("flow into me"); `connect()` sums same-orientation flows to zero.

## Flow (`flow.py`)

`AmbientInlet`, `AmbientOutlet`, `ClosedEnd`, `TwoPortSegment`, `AdiabaticPump`,
`StraightPipe`, `Valve` / `IncompressibleValve` / `CompressibleValve`,
`PressureSource`, `PressureOutlet`, `PressureVessel`, `Splitter`,
`MixingJunction`, `LoopBuffer`.

`TwoPortSegment` (and the `StraightPipe` wrapper over it) carries three
orthogonal flags that toggle structure: `multiphase`, `heat_port` (exposes a
`wall` `ThermalPort_TQ`), and `leaky` (exposes a `leak` `PermeationPort_pN`).
Per-segment ports are collected via `pipe.segment_wall_ports` /
`pipe.segment_leak_ports`.

## Walls (`walls.py`)

Boundaries (`FixedTemperature`, `FixedHeatFlow`, `ConvectiveBoundary`), the
massless `ThermalConductor`, and the two-node conduction walls: `TwoNodeWall`
(base) with `FlatWall`, `CylindricalWall` and `SphericalWall` supplying the
geometry (Cartesian / radial-cylindrical / radial-spherical capacity and
conductance).

`TwoNodeWall` also owns the (optional) gas-permeation plumbing, so **any** wall
with `leaky=True, permeation_flux=...` additionally permeates a gas through its
thickness, exposing two `PermeationPort_pN` surfaces (`leak_a` inner, `leak_b`
outer) that mirror its two thermal ports. The wall stays
permeation-physics-agnostic — the pressure-gradient → mass-flow correlation is
the injected, geometry-agnostic flux model, which reads the shape back from the
wall via `_perm_geom_conductance()` / `_perm_shell_volumes(n)`. So a subclass
only overrides the shape physics (thermal `_node_capacity` / `_conductance`
plus the two permeation hooks); everything else is inherited.

A leaky **flow volume** is just a flag, not a subclass:
`PressureVessel(leaky=True)` and `StraightPipe(leaky=True)` /
`TwoPortSegment(leaky=True)` expose `leak` ports whose mass-flow enters the
continuity balance.

## Permeation (`permeation.py`)

The species-specific half of permeation (the wall and flow volumes above are
permeation-agnostic plumbing):

- **`Permeant`** — a permeating species: molar mass `M` and the surface-law
  exponent (`2.0` Sieverts / diatomic-in-metal, `1.0` Henry / non-dissociating).
  Presets `H2`, `HELIUM`, `NITROGEN`.
- **`TransportFit`** — Arrhenius transport (`Phi`, `D`, `S = Phi/D`) of one
  permeant through one wall material. Preset `H2_IN_AUSTENITIC`.
- **Flux models** injected into a leaky wall: `SteadyRichardson` (algebraic
  Richardson flux) and `TransientDiffusion` (finite-volume radial diffusion
  chain). Both take a `TransportFit` and register their own Arrhenius
  parameters / state variables. The surface concentration follows
  `C = S · p**(1/n)`, so the same models cover every permeant.
- **`FixedPartialPressure`** — pins `p_partial` on a leak port (e.g. the outer
  surface venting to the environment); the permeation analogue of
  `FixedTemperature`.

## Assemblies (`assemblies.py`)

- **`WallLayer`** — one radial wall layer: a thermal `WallMaterial`, a radial
  `thickness`, and an optional `permeation` flux model (leaky iff given).
- **`Pipe`** — the batteries-included pipe: a `StraightPipe` wrapped, per
  segment, in a radial stack of `WallLayer`s (conjugate heat transfer), with the
  outer surfaces terminated by internal thermal / partial-pressure boundaries.
  Only `inlet` / `outlet` are exposed, so a user model is just
  `boundary → Pipe → boundary`. Permeable layers must be contiguous from the
  bore; gas vents at the outermost leaky layer into an internal
  `FixedPartialPressure(p_ext)`. `power.ConjugatePipe` is a single-layer,
  non-permeable specialisation of `Pipe`.

## Usage

The public API is re-exported at the package level, so call sites stay stable:

```python
from hydrogen.components import StraightPipe, FluidPort_phm
# equivalently:
from hydrogen.components.thermofluid import StraightPipe, FluidPort_phm
```
