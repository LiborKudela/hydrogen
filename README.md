# hydrogen

A small symbolic DAE/ODE solver for fluid-system dynamics. You compose components — inlets, pipes, vessels, sources — into a tree, declare conservation laws as SymPy expressions, and the framework symbolically reduces, lambdifies, and time-steps the resulting system with implicit Crank–Nicolson.

The library is research-friendly: every stage (symbol assignment, equation collection, trivial-equation reduction, lambdification, Newton inner loop, post-processing) is short, readable Python you can step through.

---

## Highlights

- **Declarative residuals.** Write conservation laws once as SymPy expressions; the runtime Jacobian + residual function are auto-generated via `sympy.lambdify` (with optional CSE).
- **Crank–Nicolson for free.** Declare a `DifferentialVariable` and the framework auto-creates a `der_x` companion variable plus the implicit time-step constraint that links them. You only declare the algebraic equation defining `der_x`.
- **Symbolic reduction before lambdify.** Linear identities (`x − y = 0`, `x + y = 0`, `x − const = 0` …) are detected and substituted away in multiple passes, so the runtime Newton vector contains only the strictly-needed unknowns.
- **Eliminated variables are still plotted.** Every original variable is reconstructed at record time from the substitution chain, surfaced under its full hierarchical name (`System.pipe.segment.p_in`).
- **CoolProp-backed media.** `(p, h)`-based property functions (`ρ`, `T`, `s`, `μ`, `k`) and their partial derivatives are wrapped as differentiable SymPy callables, so they participate in the symbolic Jacobian. Pass `backend="BICUBIC&HEOS"` for a ~3× faster solve on large systems with engineering-grade accuracy (see [Performance & tuning](#performance--tuning)).
- **Numba-JITed kernels.** The Newton inner loop's linear solve and error-norm primitives are `@njit`-compiled.

## Install

```bash
git clone https://github.com/LiborKudela/hydrogen.git
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

## Performance & tuning

The default settings are tuned for accuracy and clean cold-cache numbers. The knobs below are documented in measured-impact order so you know which dial moves the needle on your problem.

### `CoolPropMedium(backend=...)` — biggest practical lever

`CoolPropMedium` always builds a `CoolProp.AbstractState` per medium; `backend` chooses which equation-of-state implementation backs it.

| `backend=` | What it does | Solve speedup | Init speedup | Accuracy vs HEOS |
|---|---|--:|--:|---|
| `"HEOS"` *(default)* | Full Helmholtz EOS Newton solver. Bit-exact reference quality. | 1.0× | 1.0× | reference |
| `"BICUBIC&HEOS"` | Bicubic spline interpolation table built lazily on first `update()`; falls back to HEOS outside the table. | **2.7×** | **3.6×** | max ~1e-4 on `ρ`, ~1e-5 on `T`, ~5e-4 on partial derivatives |

```python
air = CoolPropMedium("Air", disable_warnings=True, backend="BICUBIC&HEOS")
```

Numbers are from `examples/pipe_tree.py` (`N=4, K=2, M=3`, ~389 active variables) on `sympy 1.14` / `numpy 2.2`. The Newton iteration converges to the same fixed point (max relative diff ~1e-10 across 189 active variables) regardless of backend, because table-interpolation errors don't accumulate across iterations. Use BICUBIC for any workflow that doesn't demand reference-grade thermodynamic precision.

### `Model.instantiate(...)` — symbolic build phase

```python
def instantiate(self,
                cse=True,                       # SymPy common-subexpression elimination
                aditional_modules=None,         # extra namespaces for sympy.lambdify
                max_remove_trival_passes=1,     # iterations of trivial-equation reduction
                lambda_cache_dir=None):         # disk cache directory (None = $HOME/.cache/hydrogen)
```

- **`cse=True`** *(default)* runs SymPy's common-subexpression elimination on the residual + Jacobian before code generation. Cuts the generated lambda body 2-5× on systems with shared `(p,h)` boundary terms; turn off only when debugging.
- **`aditional_modules=medium.modules`** is the **scalar** evaluator namespace (default, recommended). Each `eval_*_ph` carries `lru_cache(maxsize=100)` that catches cross-template `(p,h)` reuse — splitter junctions whose state is shared across every connected pipe — and that sharing is the dominant CoolProp speedup in tree/network systems.
- **`aditional_modules=medium.batch_modules`** is the opt-in numpy-array-aware variant. It's **3× slower** than `modules` on tree systems (each template's batch of `(p,h)` pairs is opaque to neighbours, so shared boundary nodes get re-evaluated once per template) but useful for one-of-a-kind models with many instances of a single template and zero cross-template aliasing.
- **`max_remove_trival_passes=N`** controls how many sweeps of `x − y = 0` substitution run before lambdification. `N=1` (default) catches >95% of the wins on most systems; `N=4–5` shrinks the Newton vector a few % more on deeply nested compositions like `pipe_tree(N≥3)` at the cost of a longer instantiate phase. Pass `max_remove_trival_passes=4` for the largest pipe trees.
- **`lambda_cache_dir=Path(...)`** persists the lambdified source code (per-template) to disk so subsequent runs of the same model skip code generation entirely. Default location is `$HOME/.cache/hydrogen`; the cache key is content-addressed (`sha256` of args + expression + modules signature + `cse`), so identical templates across different scripts share cache entries.

### Environment variables

- **`HYDROGEN_LAMBDA_CACHE`** — path to the on-disk lambda cache directory. Set to `""` or `"0"` to disable caching entirely (useful for benchmarking a true cold build); set to a custom path to override the default `$HOME/.cache/hydrogen` location.
- **`HYDROGEN_PARALLEL_LAMBDIFY`** — number of worker processes used to lambdify cache-miss templates in parallel. Defaults to `min(n_cache_misses, n_cpus)`. Set `=0` or `=1` to disable parallel lambdification (helpful inside debuggers, or when CoolProp's static initialisers fight `fork()`).
- **`HYDROGEN_VECTORISE_MIN`** — minimum number of instances per template for the vectorised evaluator path to kick in. Default `8`. Below this cutoff the framework uses a per-instance Python loop, which is faster for small templates because the vectorised path's array-packing + medium-callback wrapping pay off only when amortised over many elements. Set `999` to disable vectorisation entirely (escape hatch for custom media that don't tolerate broadcasting).

### `CoolPropMedium` class-level knobs

- **`scalar_cache_maxsize = 100`** *(class attribute)* — `lru_cache` size on each `eval_X_ph`. Bigger = more cross-iteration caching when Newton settles into a tight neighbourhood, marginal RAM cost. Drop to `0` to disable per-property caching for diagnostic comparisons.
- **`batch_state_pool_size = 8`** *(class attribute)* — LRU pool size for the batch state cache used by `eval_*_batch`. Only matters if you opted into `batch_modules`. Default of 8 covers most pipe-tree templates (each typically references 2-4 distinct `(p,h)` boundary states).
- **`disable_warnings=True`** — silences "partial derivative … failed, using finite difference" warnings emitted when CoolProp can't evaluate `∂μ/∂p` analytically for some media. The finite-difference fallback is correct, just chatty.

### `Model.initialise(...)` and `Model.solve_dae_step(...)` — Newton tuning

Both share the same Newton-loop knobs:

| arg | default | meaning |
|---|---|---|
| `n` *(`initialise` only)* | `1` | Number of warm-start "iterations" of `solve_dae_step(dt=0)`-equivalents to run before integrating; bump to `3-5` if your initial state is far from the steady manifold. |
| `relaxation` | `1.0` | Newton damping factor (`0 < relaxation ≤ 1`). Drop to `0.5` when the `t=0` solve diverges or oscillates — typical for vessels suddenly exposed to a much higher upstream pressure (see `fill_vessel.py`). |
| `tol` | `1e-6` | Convergence threshold on the residual norm. Loosen to `1e-4` for fast prototyping, tighten only when post-processing depends on conservation closures below 1e-6. |
| `max_iter` | `100` | Newton iteration cap per timestep. Bump to `200-400` for stiff initialisations; if you regularly hit it during `solve_dae_step`, your `dt` is probably too large for the physics. |

### Recommended starter recipe for large pipe networks

```python
import os
os.environ.setdefault("HYDROGEN_PARALLEL_LAMBDIFY", str(os.cpu_count() or 4))

from hydrogen import CoolPropMedium

air = CoolPropMedium("Air", disable_warnings=True, backend="BICUBIC&HEOS")
system = MyTreeSystem(air, ...)
system.instantiate(
    aditional_modules=air.modules,    # keep the scalar lru_cache benefit
    max_remove_trival_passes=4,       # pays off on pipe trees with N >= 3
)
system.initialise(relaxation=0.5, max_iter=400)
for _ in range(N_STEPS):
    system.solve_dae_step(dt)
    system.next_step()
```

## A few practical notes

- **Initial guesses matter.** Newton uses a full step by default. Systems with strong startup transients (e.g. a vessel suddenly exposed to a much higher upstream pressure — see `fill_vessel.py`) can need (a) warm-started velocities to keep the energy-equation row non-singular w.r.t. `w`, and (b) `initialise(relaxation=0.5)` to damp the `t=0` solve.
- **Trivial reduction protects parameters.** The reducer never eliminates a `Parameter` (or `t`/`dt`); only free `Variable`s can be substituted away. This avoids a class of bugs where a parameter would silently disappear from the lambdified signature.
- **Ill-conditioned Jacobians** (mixing Pa, J/kg, m/s, kg in one residual vector) can show `cond ≈ 10¹⁴` on small systems. Newton still converges in practice; row/column scaling is a worthwhile future improvement.
- **Single-phase only.** All built-in components assume a single fluid phase (whatever CoolProp returns from `(p, h)`). Two-phase support would need a different residual formulation.

## Future roadmap

A non-binding wishlist of features that fit the framework's philosophy. Each bullet sketches the rough implementation path so a contributor can pick one up without a meeting.

- **Adaptive time-stepping.** Wrap `solve_dae_step(dt)` in an outer controller that watches Newton convergence (iter count, residual reduction ratio) and a cheap embedded error estimator — e.g. a single explicit-Euler "predictor" step compared against the implicit Crank–Nicolson solution at `t+dt`. Shrink `dt` (`× 0.5`) when Newton needs more than `max_iter/2` iterations or the predictor-corrector mismatch exceeds `tol_local`, grow it (`× 1.2`) on easy steps. The existing `next_step()` snapshotting already supports rejecting + retrying a step at smaller `dt`.

- **Reversible mixing junction.** Generalise `Splitter` (currently strictly 1-in / K-out) to a `MixingJunction` with M ports of unknown sign. For each port `i`, define an "upwind" weight `α_i = ½·(1 + tanh(w_i / w_smooth))` (smoothed Heaviside, keeps the Jacobian C¹ — see how `StraightPipe` already smooths Min/Max for friction), then write the mass + energy balances as `Σ ρ_i·A_i·w_i = 0` and `h_out = (Σ α_i·ṁ_i·h_i) / (Σ α_i·ṁ_i)`. The smoothing length `w_smooth` becomes a `Parameter` so users can trade Newton conditioning for sharper switching.

- **JSON system parser.** Add a `hydrogen.json_loader` module that takes a JSON document `{components: [{type, name, args}…], connections: [{from, to}…]}`, looks each `type` up in a `COMPONENT_REGISTRY` dict (populated by a `@register_component` decorator on each class in `hydrogen.components`), and emits a `Model` subclass on the fly via `type(name, (Model,), {…})`. Pair it with a `Model.to_json()` round-tripper and a tiny WebSocket/REST shim that pushes `record['state']` to a UI in real time and lets the UI mutate `Parameter.value`s mid-run — turns hydrogen into a backend for visual flow-sheet editors.

- **Conditional equations.** Support equations that switch form based on the current state (e.g. choked vs. subsonic flow, valve open/closed) using `sympy.Piecewise` for cases where the discontinuity is genuinely smooth (lambdify already handles `Piecewise` via `numpy.select`), and a sigmoid-blended `α(state)·eq_A + (1 − α(state))·eq_B` form for cases where the discontinuity must stay C¹ for Newton. For hard switches (compressible/incompressible regime change), a third route is to detect the active branch *between* timesteps in `next_step()`, swap the equation set, and rebuild the lambdified residual from cache — the existing per-template lambda cache (keyed by content hash) means each branch only pays its first-time lambdify cost once.
