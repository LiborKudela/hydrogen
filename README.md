# hydrogen

A small symbolic DAE/ODE solver for fluid-system dynamics. You compose components — inlets, pipes, vessels, sources — into a tree, declare conservation laws as SymPy expressions, and the framework symbolically reduces, lambdifies, and time-steps the resulting system with implicit Crank–Nicolson.

The library is research-friendly: every stage (symbol assignment, equation collection, trivial-equation reduction, lambdification, Newton inner loop, post-processing) is short, readable Python you can step through.

---

## Highlights

- **Declarative residuals.** Write conservation laws once as SymPy expressions; the runtime Jacobian + residual function are auto-generated via `sympy.lambdify` (with optional CSE).
- **Crank–Nicolson for free.** Declare a `DifferentialVariable` and the framework auto-creates a `der_x` companion variable plus the implicit time-step constraint that links them. You only declare the algebraic equation defining `der_x`.
- **Symbolic reduction before lambdify.** Linear identities (`x − y = 0`, `x + y = 0`, `x − const = 0` …) are detected and substituted away in multiple passes, so the runtime Newton vector contains only the strictly-needed unknowns.
- **Eliminated variables are still plotted.** Every original variable is reconstructed at record time from the substitution chain, surfaced under its full hierarchical name (`System.pipe.segment.p_in`).
- **CoolProp-backed media.** `(p, h)`-based property functions (`ρ`, `T`, `s`, `μ`, `k`) and their partial derivatives are wrapped as differentiable SymPy callables, so they participate in the symbolic Jacobian.
- **Numba-JITed kernels.** The Newton inner loop's linear solve and error-norm primitives are `@njit`-compiled.

## Install

```bash
git clone <repo-url>
cd hydrogen
pip install -e .
# or, with the test dependencies:
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Runtime deps: `numpy`, `sympy`, `numba`, `CoolProp`, `plotly`, `line_profiler`.

## Quickstart

A two-state harmonic oscillator (`y'' = −ω² y`), built from `DifferentialVariable`s. The framework auto-generates `der_y`, `der_z`, and the Crank–Nicolson constraints — you only write the ODE right-hand sides.

```python
import numpy as np
from hydrogen import DifferentialVariable, Model, Parameter

class Oscillator(Model):
    def declare_components(self):
        self.add_component('omega', Parameter(2 * np.pi))
        self.add_component('y', DifferentialVariable(1.0))   # y(0) = 1
        self.add_component('z', DifferentialVariable(0.0))   # z(0) = 0,  z = dy/dt

    def declare_equations(self):
        return [
            self['der_y'].symbol - self['z'].symbol,
            self['der_z'].symbol + self['omega'].symbol ** 2 * self['y'].symbol,
        ]

m = Oscillator()
m.instantiate()
m.initialise()
for _ in range(25):
    m.solve_dae_step(0.04)
    m.next_step()

t = np.array(m.record['time'])
y = np.array(m.record['state'])[:, m.record['vars_names'].index('Oscillator.y')]
print(f"max |y − cos(ωt)| = {np.max(np.abs(y - np.cos(2*np.pi*t))):.3e}")
```

## Examples

Three ready-to-run demos in `examples/`. Each writes an interactive Plotly HTML next to the script.

### `examples/run_system.py` — heated mass-flow loop

`AmbientInlet → StraightPipe → StraightPipe`, with a decoupled `IntegrationTest` sub-model whose ODEs are checked against analytical solutions to validate the integrator end-to-end.

```bash
python examples/run_system.py
```

### `examples/fill_vessel.py` — pressure-driven vessel charging

`PressureSource (2 bar)` → `StraightPipe (3 mm × 1 m, adiabatic)` → `PressureVessel (1 atm, 1 L)`. Textbook charging transient: as vessel pressure rises toward the source, the driving differential shrinks and the inflow velocity decays to zero.

```
=== Filling transient summary ===
Source:        p = 2.000 bar,  T = 293.15 K
Vessel start:  p = 1.013 bar,  T = 293.15 K,  m = 1.204 g
Vessel end:    p = 2.000 bar,  T = 339.44 K,  m = 2.053 g
Inlet w_in:    start = 142.670 m/s,  end = -0.000 m/s   (100.0% decay)
Vessel pressure has closed 100.0% of the gap to source pressure.
```

```bash
python examples/fill_vessel.py
```

### `examples/pipe_tree.py` — recursive K-ary tree of pipes

Build a balanced flow tree from four knobs at the top of the file: depth `N`, branching factor `K`, segments per pipe `M`, common pipe length `L`. The example wires:

```
PressureSource  →  root pipe  →  Splitter  →  K * (pipe → Splitter → ...)  →  pipe → PressureOutlet
```

so the whole system has `(K^(N+1) - 1) / (K - 1)` pipes, `(K^N - 1) / (K - 1)` splitters and `K^N` leaves. Defaults `N=2, K=2, M=2`, giving 7 pipes / 3 splitters / 4 leaves. Output prints the steady-state outlet conditions at every depth and confirms the symmetry of the solution:

```
=== Steady-state pipe-outlet conditions, by tree depth ===
  depth 0 ( 1 pipe ):  w_out min= 81.59 max= 81.59 m/s,  p_out min=1.0525 max=1.0525 bar,  spread(w)=0.00e+00
  depth 1 ( 2 pipes):  w_out min= 41.40 max= 41.40 m/s,  p_out min=1.0221 max=1.0221 bar,  spread(w)=0.00e+00
  depth 2 ( 4 pipes):  w_out min= 20.79 max= 20.79 m/s,  p_out min=1.0130 max=1.0130 bar,  spread(w)=0.00e+00
```

Velocity drops by ~`K` at each splitter (mass conservation with constant area), and `spread(w) = 0` confirms every pipe at the same depth carries the same flow.

```bash
python examples/pipe_tree.py
```

## Component catalogue (`hydrogen.components`)

| Class | Boundary type | Use it when… |
|---|---|---|
| `AmbientInlet` | Mass-flow-imposed inlet at ambient `(p, T)` | You know `ṁ` upstream |
| `AmbientOutlet` | Mass-flow-imposed outlet at ambient `(p, T)` | You know `ṁ` downstream |
| `PressureSource` | Stagnation reservoir at fixed `(p, T)` | You want flow driven by Δp; system finds `ṁ` |
| `PressureOutlet` | Fixed-pressure outlet (forces `p_in = p_ambient`) | Pressure-imposed termination; system finds `ṁ` |
| `PressureVessel` | Lumped rigid-volume vessel filling through one port | Charging / discharging dynamics |
| `Splitter` | Ideal `K`-way junction (no Δp, no Δh) | Building flow trees / manifolds |
| `TwoPortSegment` | One discrete CV of a 1D duct (continuity + momentum + energy) | Building block; rarely used directly |
| `StraightPipe` | Pipe split into N `TwoPortSegment`s with Churchill friction + smoothed Nusselt | Plumbing two boundaries together |
| `AdiabaticPump` | Adiabatic-pump segment with a custom `f = a_iz / (Re·Dh)` friction model | Idealised pump element |

`hydrogen.test_models` also exports a few synthetic ODE models (`IntegrationTest`, `SimpleODE`, …) used by the test suite — handy as templates when you want to test the integrator on something with a known closed form.

## Core concepts

- **`Model`** is the composition primitive. Override `declare_components()` to register sub-models / `Variable`s / `Parameter`s with `self.add_component(name, ...)`. Override `declare_equations()` to return a list of SymPy expressions, each implicitly equal to zero.
- **`Variable`** is an algebraic unknown — Newton adjusts it until residuals vanish.
- **`Parameter`** is a constant scalar lambdified into the residual function. You can mutate `param.value` between solves to sweep without re-compiling.
- **`DifferentialVariable`** is a `Variable` whose time evolution is governed by Crank–Nicolson:

    `x_{n+1} = x_n + ½·dt·(der_x_{n+1} + der_x_n)`

  Adding it to a model auto-creates a `der_x` companion. You declare only the algebraic equation that defines `der_x` (the ODE right-hand side); the framework emits the CN constraint.
- **`CoolPropMedium`** wraps a CoolProp `AbstractState` and exposes `(p, h)`-based property functions (`rho_ph`, `T_ph`, `s_ph`, `mu_ph`, `k_ph`, plus `h_pT` for initial-condition convenience) as differentiable SymPy callables. Their partial derivatives also come from CoolProp.

## Wiring components together

Every component exposes ports as named `Variable`s (typically `p_in/h_in/w_in` and `p_out/h_out/w_out`). Connections are just port-equality equations in the parent's `declare_equations`:

```python
class System(Model):
    def declare_components(self):
        self.medium = CoolPropMedium("Air", disable_warnings=True)
        self.add_component('source', PressureSource(self.medium, 2e5, 293.15))
        self.add_component('pipe',   StraightPipe(self.medium, 0.003, 1.0, 1e-6,
                                                  z_in=0, z_out=0, n_segments=2, adiabatic=True))
        self.add_component('vessel', PressureVessel(self.medium, 1e-3, A_port, 1.013e5, 293.15))

    def declare_equations(self):
        return [
            self['source']['p_out'].symbol - self['pipe']['p_in'].symbol,
            self['source']['h_out'].symbol - self['pipe']['h_in'].symbol,
            self['source']['w_out'].symbol - self['pipe']['w_in'].symbol,
            self['pipe']['p_out'].symbol   - self['vessel']['p_in'].symbol,
            self['pipe']['h_out'].symbol   - self['vessel']['h_in'].symbol,
            self['pipe']['w_out'].symbol   - self['vessel']['w_in'].symbol,
        ]
```

The trivial-equation pass collapses each `a − b = 0` connection into a single shared symbol before lambdification, so these "wires" are free at runtime.

## Project layout

```
hydrogen/
├── hydrogen/                # the package
│   ├── __init__.py          # public API
│   ├── model.py             # Model, Variable, DifferentialVariable, Newton, CN
│   ├── components.py        # AmbientInlet, PressureSource, PressureVessel, StraightPipe, …
│   ├── medium.py            # CoolPropMedium + sympy-able property functions
│   ├── numerics.py          # lambdify_compat + numba-JITed Newton primitives
│   ├── caching.py           # numpy_cache + ModelCache
│   ├── plotting.py          # plot_results (plotly)
│   └── test_models.py       # IntegrationTest + small ODE fixtures
├── examples/
│   ├── run_system.py        # heated-pipe flow demo
│   ├── fill_vessel.py       # pressure-driven charging demo
│   └── pipe_tree.py         # recursive K-ary pipe tree (Splitter + PressureOutlet)
├── tests/                   # pytest suite (caching, integrator, naming, vessel, …)
├── pyproject.toml
└── solver.py                # back-compat shim for the original single-file layout
```

## Tests

```bash
pytest                                  # full suite
pytest -v tests/test_integration.py     # just the time-integrator analytical checks
pytest -v tests/test_pressure_vessel.py # just the vessel mass/energy closures
```

The suite covers the cache primitives, `lambdify_compat` + numba kernels, hierarchical naming, the trivial-removal substitution chain (including a regression for non-`Symbol` RHSs), the CN integrator vs. analytical solutions for a decay + harmonic oscillator, and the `PressureVessel` mass / energy / closure laws driven by a constant-`ṁ` source.

## A few practical notes

- **Initial guesses matter.** Newton uses a full step by default. Systems with strong startup transients (e.g. a vessel suddenly exposed to a much higher upstream pressure — see `fill_vessel.py`) can need (a) warm-started velocities to keep the energy-equation row non-singular w.r.t. `w`, and (b) `initialise(relaxation=0.5)` to damp the `t=0` solve.
- **Trivial reduction protects parameters.** The reducer never eliminates a `Parameter` (or `t`/`dt`); only free `Variable`s can be substituted away. This avoids a class of bugs where a parameter would silently disappear from the lambdified signature.
- **Ill-conditioned Jacobians** (mixing Pa, J/kg, m/s, kg in one residual vector) can show `cond ≈ 10¹⁴` on small systems. Newton still converges in practice; row/column scaling is a worthwhile future improvement.
- **Single-phase only.** All built-in components assume a single fluid phase (whatever CoolProp returns from `(p, h)`). Two-phase support would need a different residual formulation.
