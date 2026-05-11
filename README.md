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

Always install into a fresh virtual environment — the framework lambdifies `sp.Min` / `sp.Max` (used by the smoothed-Nusselt blending in `StraightPipe`) and that path only generates correct numpy code starting with **sympy ≥ 1.12**. Distro-managed Pythons (e.g. Ubuntu 22.04's `python3` ships sympy 1.5.1 from 2019) silently emit broken `np.amin` / `np.amax` calls that crash with `ValueError: setting an array element with a sequence ... inhomogeneous shape` at solve time.

```bash
git clone https://github.com/LiborKudela/hydrogen.git
cd hydrogen

python3 -m venv .venv               # create a venv (skip if you have your own)
source .venv/bin/activate           # activate it

pip install -e .                    # or pip install -e ".[dev]" for the test deps
```

Requires Python ≥ 3.10. Runtime deps (with the pinned minimums in `pyproject.toml`): `numpy ≥ 1.24`, `sympy ≥ 1.12`, `numba ≥ 0.58`, `CoolProp ≥ 6.5`, `plotly`, `line_profiler`.

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

Four ready-to-run demos in `examples/`. Each writes an interactive Plotly HTML into the git-ignored sandbox `local_results/examples/` (override the location with the `HYDROGEN_LOCAL_RESULTS` env var). Tests that opt in via the `local_results_path` fixture drop their artifacts under `local_results/tests/`.

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
Inlet m_dot:   start = 1.292 g/s,  end = 0.000 g/s   (100.0% decay)
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
=== Steady-state pipe-outlet conditions (Air) ===
  depth 0 ( 1 pipe):  m_dot min=   2.0449 max=   2.0449 g/s,  w min=  81.586 max=  81.586 m/s,  p_out min=1.0525 max=1.0525 bar
  depth 1 ( 2 pipes): m_dot min=   1.0224 max=   1.0224 g/s,  w min=  41.402 max=  41.402 m/s,  p_out min=1.0221 max=1.0221 bar
  depth 2 ( 4 pipes): m_dot min=   0.5112 max=   0.5112 g/s,  w min=  20.793 max=  20.793 m/s,  p_out min=1.0130 max=1.0130 bar

Mass conservation (Air):
  source m_dot                        =    2.0449 g/s
  depth 0 ( 1 pipes) m_dot   =    2.0449 g/s  (rel err vs source: 0.00e+00)
  depth 1 ( 2 pipes) m_dot   =    2.0449 g/s  (rel err vs source: 0.00e+00)
  depth 2 ( 4 pipes) m_dot   =    2.0449 g/s  (rel err vs source: 0.00e+00)
```

Mass flow is exactly conserved at every depth (rel err `0.00e+00`) because the `(p, h, m_dot)` port convention unifies `m_dot` across every joint — there is no `ρ·w·A` translation step that could introduce numerical drift. The per-leaf velocity drops by `~K` at each splitter (constant-area mass conservation), confirming the tree's symmetry.

```bash
python examples/pipe_tree.py
```

### `examples/loop_pump_pipe.py` — true closed pump-and-pipe loop with a `LoopBuffer`

`AdiabaticPump → StraightPipe → LoopBuffer → pump`, wired in a fully-closed loop via `add_connection`. A pure pump+pipe loop is **structurally rank-deficient by 2**: loop continuity (`m_dot` is conserved through every segment, so the loop-closing continuity equation is implied by the others) and adiabatic loop energy (`h + w²/2` is conserved through every adiabatic segment) are each one-equation tautologies of the per-segment equations. The framework's Newton solve needs a square non-singular Jacobian, so a fully-wired loop won't instantiate without mass and energy storage to absorb those redundancies. `LoopBuffer` (a two-port well-mixed lumped-volume vessel; see the component catalogue below) provides exactly that — its `m` and `U` are differential states whose own residuals replace the redundant loop-closure equations, so even at steady state the global Jacobian stays full rank.

The example also illustrates the three "anchors" you typically need to pin down a loop's operating point: pressure level (set by `LoopBuffer.p_init` via the EoS closure `m_init = ρ(p_init,h_init)·V`), enthalpy level (set by `T_init`, which determines `U_init = m_init·h_init - p_init·V`), and mass flow (one explicit equation `pump.m_dot_in == m_dot_target` returned from `declare_equations`, which fixes the pump's free `a_iz` strength). It then rides a sinusoidal `m_dot_target(t)` so the pump head continuously adjusts to track the prescribed flow.

```
=== Initial steady-state (t = 0, m_dot_target = m_dot_base) ===
Buffer state    : p = 2.0000 bar, h = 426074.48 J/kg, m = 2.324 g, U = 790.16 J
  (anchor target: p = 2.0000 bar, T = 300.00 K, V = 1.0 L)
Pump inlet p    :    2.0000 bar      (== buffer.p via wiring, residual 0.00e+00 Pa)
Pump rise       : dp =   4.677 kPa
Pipe drop       : dp =   4.677 kPa (must equal pump rise; residual 0.00e+00 Pa)
Pump strength   : a_iz = -1.4767e+02
Mass flow       :  20.000 g/s     (target 20.000 g/s)
```

```bash
python examples/loop_pump_pipe.py
```

## Component catalogue (`hydrogen.components`)

| Class | Boundary type | Use it when… |
|---|---|---|
| `AmbientInlet` | Mass-flow-imposed inlet at ambient `(p, T)` | You know `ṁ` upstream |
| `AmbientOutlet` | Mass-flow-imposed outlet at ambient `(p, T)` | You know `ṁ` downstream |
| `PressureSource` | Stagnation reservoir at fixed `(p, T)` | You want flow driven by Δp; system finds `ṁ` |
| `PressureOutlet` | Fixed-pressure outlet (forces `p_in = p_ambient`) | Pressure-imposed termination; system finds `ṁ` |
| `PressureVessel` | Lumped rigid-volume vessel filling through one port | Charging / discharging dynamics |
| `LoopBuffer` | Two-port well-mixed lumped-volume buffer (one inflow + one outflow) | Closed loops — provides the mass + energy storage that breaks loop continuity / loop energy rank-deficiency |
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

Every component exposes ports as named `Variable`s following the **`(p, h, m_dot)`** convention: `p_in/h_in/m_dot_in` and `p_out/h_out/m_dot_out`. Pressure and specific enthalpy are the standard `(p, h)` thermodynamic state, and **mass flow rate `m_dot` [kg/s]** is the conserved flow variable across every joint — using `m_dot` (rather than velocity `w`) means port connections are mass-conserving regardless of any cross-sectional-area mismatch on either side. Velocity is reconstructed internally where each component needs it (`w = m_dot / (ρ·A)`); to inspect it post-simulation, divide the recorded `m_dot` by `ρ(p, h)·A`.

Connections are just port-equality equations in the parent's `declare_equations`. Use `add_connection` (union-find) for pure variable equalities — it short-circuits these out of the symbolic Jacobian entirely instead of leaving them for the trivial-equation reducer:

```python
class System(Model):
    def declare_components(self):
        self.medium = CoolPropMedium("Air", disable_warnings=True)
        self.add_component('source', PressureSource(self.medium, 2e5, 293.15, A=A_port))
        self.add_component('pipe',   StraightPipe(self.medium, 0.003, 1.0, 1e-6,
                                                  z_in=0, z_out=0, n_segments=2, adiabatic=True))
        self.add_component('vessel', PressureVessel(self.medium, 1e-3, A_port, 1.013e5, 293.15))

    def declare_equations(self):
        for io in ('p', 'h', 'm_dot'):
            self.add_connection(self['source'][f'{io}_out'], self['pipe'][f'{io}_in'])
            self.add_connection(self['pipe'][f'{io}_out'],   self['vessel'][f'{io}_in'])
        return []
```

The union-find pass collapses each `add_connection(a, b)` into a single shared symbol before lambdification, so these "wires" are completely free at runtime — the unified `m_dot` symbol is exactly what flows through the whole chain.

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

### `CoolPropMedium` cache + warning knobs

- **`scalar_cache_maxsize = 100`** *(constructor kwarg, also a class attribute used as default)* — `lru_cache` size on each scalar `eval_X_ph`. **Critical for HEOS performance on systems with many segments.** When the working set of unique `(p, h)` states exceeds the cache size, the cache thrashes and HEOS scales super-linearly because every Newton iteration re-computes the same property at the same `(p, h)` it just evicted. Working set ≈ number of active variables for media with analytical `μ`/`k` partials, ≈ **5× that** for media that fall back to finite differences (Air, Hydrogen — each FD `dμ/d{p,h}` lookup adds 4 extra `(p±ε, h)` and `(p, h±ε)` keys). Measured impact on `run_system.py` with 2 pipes × N segments + IntegrationTest:

  | backend | N | active vars | `scalar_cache_maxsize` | ms/step | cache hit% |
  |---|--:|--:|--:|--:|--:|
  | HEOS | 50 | 408 | 100 *(default)* | 274 | 68% |
  | HEOS | 50 | 408 | **1000** | **96** | **89%** |
  | HEOS | 100 | 808 | 100 *(default)* | 2305 | 0.7% (catastrophic thrash) |
  | HEOS | 100 | 808 | **1000** | **211** | 88% |
  | BICUBIC&HEOS | 100 | 808 | 100 *(default)* | 116 | 0.6% |

  Rule of thumb: if your HEOS solve loop is 5× slower than expected at large N, set `scalar_cache_maxsize ≈ 5 × n_active_vars`. RAM cost is negligible (~1 KB per cache entry × number of properties × 1000 ≈ 16 MB max). For BICUBIC&HEOS the cache matters less (table lookup is ~5 µs, so a miss is cheap).

  ```python
  air = CoolPropMedium("Air", disable_warnings=True, scalar_cache_maxsize=1000)
  ```

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
| `raise_on_no_convergence` *(`solve_dae_step` only)* | `False` | When `True`, raises `NewtonConvergenceFailure` if Newton hits `max_iter` without reducing the residual below `tol`. The adaptive controller (below) sets this internally so it can catch + retry at smaller `dt`. |

### `Model.solve_adaptive_step(...)` — adaptive time-stepping

For transients with very different timescales (steep initial ramp, slow tail) a single fixed `dt` either over-resolves the late phase or under-resolves the early phase. `solve_adaptive_step` proposes a `dt`, asks a strategy whether the step is acceptable, and either commits or restores + retries at smaller `dt`. The caller still drives `next_step()` explicitly (so a rejected step never reaches `record`).

```python
dt_used, info = system.solve_adaptive_step(
    dt_target=0.025,                         # initial guess; gets scaled by the controller
    strategy="predictor_corrector",          # default; see strategies below
    dt_min=1e-5, dt_max=0.1,                 # hard floor / ceiling
    relaxation=1.0, tol=1e-6, max_iter=100,  # forwarded to every internal solve
)
system.next_step()                           # caller commits the accepted step
```

Four strategies are bundled (pass `strategy="name"` for defaults, or `strategy={"name": "...", **overrides}` to tune):

| `strategy` | What it measures | Extra solves / step | Best for |
|---|---|--:|---|
| `"fixed"` | nothing — short-circuit to `solve_dae_step(dt_target)` | 0 | reference / baseline |
| `"derivative_limit"` | per-variable relative state change `|x_new − x_pre|/max(|x_pre|, |x_new|, atol)` | 0 | monotonic transients (vessel charge/discharge); not reliable for variables that pass through zero |
| `"predictor_corrector"` *(default)* | mismatch between explicit-Euler predictor `x_pre + dt·der_pre` and the CN result, on differential variables only | 0 | most engineering transients; principled local-error estimate without paying for a second implicit solve |
| `"richardson"` | one full step at `dt` vs two half-steps at `dt/2`; commits the more-accurate half-step result | 2 | when you need a calibrated `O(dt^(p+1))` local-error bound and can afford 3× the per-step cost |

Per-strategy tuning lives in `_DEFAULT_STRATEGY_PARAMS` (`hydrogen/model.py`); the headline knob is `tol_local` (or `rel_tol` for `derivative_limit`). Variables with units that make the global `atol` inappropriate can override it per-variable: `Variable(value, unit, atol=tight_value)`.

Benchmark on `examples/fill_vessel.py` (3 s vessel-charging transient, run from `examples/bench_adaptive.py`):

| run | wall [s] | Newton iters | steps | dt range [s] | accuracy [Pa @ t=0.5s] |
|---|--:|--:|--:|--:|--:|
| `fixed dt=0.025`   | 0.79 |  423 | 120 | 0.025         |  4 |
| `predictor_corrector tol=1e-2` | **0.29** | **152** | **35** | 0.006 → 0.10 | 52 |

≈ **2.7× faster** in both wall time and Newton iters at ~0.03 % relative pressure error. The win comes entirely from `dt` growing 16× during the late equilibration phase.

### Recommended starter recipe for large pipe networks

```python
import os
os.environ.setdefault("HYDROGEN_PARALLEL_LAMBDIFY", str(os.cpu_count() or 4))

from hydrogen import CoolPropMedium

air = CoolPropMedium(
    "Air", disable_warnings=True,
    backend="BICUBIC&HEOS",          # ~10x faster than HEOS at large N, scales linearly
    scalar_cache_maxsize=1000,       # bump for systems with > ~50 active vars
)
system = MyTreeSystem(air, ...)
system.instantiate(
    aditional_modules=air.modules,    # keep the scalar lru_cache benefit
    max_remove_trival_passes=4,       # pays off on pipe trees with N >= 3
)
system.initialise(relaxation=0.5, max_iter=400)
for _ in range(N_STEPS):
    system.solve_dae_step(dt)
    system.next_step()

# Or, for stiff transients, swap the inner loop for adaptive time-stepping
# (typically 2-3x faster than the right fixed dt on uneven dynamics):
# while system.get_t_value() < t_end:
#     system.solve_adaptive_step(dt_target=dt, dt_max=10*dt)
#     system.next_step()
```

## A few practical notes

- **Initial guesses matter.** Newton uses a full step by default. Systems with strong startup transients (e.g. a vessel suddenly exposed to a much higher upstream pressure — see `fill_vessel.py`) can need (a) warm-started velocities to keep the energy-equation row non-singular w.r.t. `w`, and (b) `initialise(relaxation=0.5)` to damp the `t=0` solve.
- **Trivial reduction protects parameters.** The reducer never eliminates a `Parameter` (or `t`/`dt`); only free `Variable`s can be substituted away. This avoids a class of bugs where a parameter would silently disappear from the lambdified signature.
- **Ill-conditioned Jacobians** (mixing Pa, J/kg, m/s, kg in one residual vector) can show `cond ≈ 10¹⁴` on small systems. Newton still converges in practice; row/column scaling is a worthwhile future improvement.
- **Single-phase only.** All built-in components assume a single fluid phase (whatever CoolProp returns from `(p, h)`). Two-phase support would need a different residual formulation.

## Future roadmap

A non-binding wishlist of features that fit the framework's philosophy. Each bullet sketches the rough implementation path so a contributor can pick one up without a meeting.

- **Reversible mixing junction.** Generalise `Splitter` (currently strictly 1-in / K-out) to a `MixingJunction` with M ports of unknown sign. For each port `i`, define an "upwind" weight `α_i = ½·(1 + tanh(w_i / w_smooth))` (smoothed Heaviside, keeps the Jacobian C¹ — see how `StraightPipe` already smooths Min/Max for friction), then write the mass + energy balances as `Σ ρ_i·A_i·w_i = 0` and `h_out = (Σ α_i·ṁ_i·h_i) / (Σ α_i·ṁ_i)`. The smoothing length `w_smooth` becomes a `Parameter` so users can trade Newton conditioning for sharper switching.

- **JSON system parser.** Add a `hydrogen.json_loader` module that takes a JSON document `{components: [{type, name, args}…], connections: [{from, to}…]}`, looks each `type` up in a `COMPONENT_REGISTRY` dict (populated by a `@register_component` decorator on each class in `hydrogen.components`), and emits a `Model` subclass on the fly via `type(name, (Model,), {…})`. Pair it with a `Model.to_json()` round-tripper and a tiny WebSocket/REST shim that pushes `record['state']` to a UI in real time and lets the UI mutate `Parameter.value`s mid-run — turns hydrogen into a backend for visual flow-sheet editors.

- **Conditional equations.** Support equations that switch form based on the current state (e.g. choked vs. subsonic flow, valve open/closed) using `sympy.Piecewise` for cases where the discontinuity is genuinely smooth (lambdify already handles `Piecewise` via `numpy.select`), and a sigmoid-blended `α(state)·eq_A + (1 − α(state))·eq_B` form for cases where the discontinuity must stay C¹ for Newton. For hard switches (compressible/incompressible regime change), a third route is to detect the active branch *between* timesteps in `next_step()`, swap the equation set, and rebuild the lambdified residual from cache — the existing per-template lambda cache (keyed by content hash) means each branch only pays its first-time lambdify cost once.
