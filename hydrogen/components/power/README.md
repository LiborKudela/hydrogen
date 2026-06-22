# Power-engineering domain

Coupled (conjugate) components that compose the **thermofluid** domain into the
building blocks of power-plant / process plumbing. This domain introduces no new
primitive physics and no new connector kind — it wires together the existing
`FluidPort_phm` and `ThermalPort_TQ` ports.

## Components

### `ConjugatePipe`

A thin specialisation of `thermofluid.Pipe` with a single, non-permeable
`WallLayer`: a fluid `StraightPipe` (built with `heat_port=True`) whose every
segment is wrapped in a `CylindricalWall`, giving the metal wall real thermal
mass and letting the fluid exchange heat with it segment-by-segment. For
multi-layer walls or wall permeation, use `thermofluid.Pipe` directly.

```
fluid:   ===[ segment_0 ]===[ segment_1 ]=== ... ===[ segment_N-1 ]===
               |  wall          |  wall                 |  wall
            port_a           port_a                  port_a
metal:    [ wall_0_0 ]      [ wall_1_0 ]     ...   [ wall_{N-1}_0 ]
            |  port_b          |  port_b              |  port_b
            outer_0           outer_1               outer_N-1
```

Each segment's `wall` thermal port publishes the convective heat rate `q`
(W) and the inner wall-node temperature. The connection to
`CylindricalWall.port_a`:

- closes the segment's `T_wall` to the wall metal temperature (across
  channel `T`), and
- feeds `-q` into the wall node (same-orientation `Q_dot` sum-to-zero),

so energy is conserved across the fluid/metal interface.

#### Outer boundary (`outer=`)

- `"adiabatic"` (default) — `FixedHeatFlow(0)` per segment; the outer
  surface is perfectly insulated, so all heat the fluid gives up is stored in
  the metal. Self-contained and well-posed standalone.
- `"convective"` — `ConvectiveBoundary(h_ext, A_outer, T_ext)` per segment;
  Newton cooling to a far-field at `T_ext` through film coefficient `h_ext`
  over each segment's outer area `2·π·r_out·L_segment`.
- `"expose"` — no internal termination; each outer node is re-exposed as a
  `wall_outer_{i}` `ThermalPort_TQ` for the parent model to wire.

Fluid connectivity is re-exposed as `inlet` / `outlet` `FluidPort_phm`, so a
`ConjugatePipe` is a drop-in replacement for a `StraightPipe` in a fluid
network.

## Physics notes

The conjugate coupling relies on two corrections made to the fluid segment
energy balance (see `components/thermofluid/flow.py`):

1. **Specific heat input.** The wall heat `q` [W] enters the fluid energy
   balance as `q / ṁ` [J/kg], so the fluid gains exactly `q` watts — matching
   the heat the wall loses. (Adding the raw power `q` to a specific enthalpy
   is dimensionally wrong and breaks conservation.)
2. **Per-segment wetted area.** The convective area is the inner surface of
   one segment, `π·D·L_segment`, not the full pipe length.

A direct consequence, checkable at any converged step, is the telescoping
energy balance

```
ṁ · (h_out − h_in)  ≈  Σ_i q_i
```

i.e. the total enthalpy rise of the fluid equals the sum of the per-segment
heat inputs.
