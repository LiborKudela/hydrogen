# hydrogen

A small, readable symbolic DAE/ODE framework. You compose **components** into a
tree, declare conservation/closure laws as SymPy expressions, and hydrogen
symbolically reduces them, generates the residual + Jacobian via `sympy.lambdify`,
and time-steps the system with implicit Crank–Nicolson.

It ships component libraries for several physics domains — **fluid**, **thermal**,
**power** (coupled fluid+thermal), and **control** — and they interoperate in one solve. Every stage (symbol assignment,
equation reduction, lambdify, Newton loop, recording) is short Python you can step
through.

## Key ideas

- **Declarative residuals.** Write each equation once as a SymPy expression
  (implicitly `= 0`); the runtime residual + Jacobian are auto-generated (with
  optional CSE).
- **Crank–Nicolson for free.** Declare a `DifferentialVariable` and hydrogen adds
  its `der_x` companion and the implicit time-step constraint. You only write the
  algebraic equation defining `der_x` (the ODE right-hand side).
- **Symbolic reduction before lambdify.** Linear identities (`x − y = 0`, …,
  connection "wires") are substituted away, so the Newton vector holds only the
  variables that actually need solving. Eliminated variables are still
  reconstructed and recorded under their full hierarchical name.
- **CoolProp-backed media.** `(p, h)`-based properties (`ρ, T, s, μ, k`) and their
  partial derivatives are wrapped as differentiable SymPy callables, so they enter
  the symbolic Jacobian.
- **Numba-JITed kernels** for the Newton inner loop; an **on-disk lambda cache** so
  repeated runs skip code generation.
- **Save / load as data.** Any system round-trips to a versioned `{components,
  connections}` spec (`to_dict`/`from_dict`, `to_json`/`from_json`).
- **Drive it from another process.** Launch a system in its own host process and
  control it live over a socket (`start_host`) — stream any variables on demand,
  pause/resume/stop mid-run, browse a variable tree — ideal for a separate UI.

## Install

Use a fresh virtual environment. hydrogen lambdifies `sp.Min`/`sp.Max` (smoothed
Nusselt blending in `StraightPipe`), and that path only generates correct numpy
code on **sympy ≥ 1.12** — older distro sympy silently emits broken `np.amin`/
`np.amax` calls that fail at solve time.

```bash
git clone https://github.com/LiborKudela/hydrogen.git
cd hydrogen
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -e ".[dev]"  (adds pytest)
```

Requires Python ≥ 3.10. Runtime deps (pinned in `pyproject.toml`): `numpy ≥ 1.24`,
`sympy ≥ 1.12`, `numba ≥ 0.58`, `CoolProp ≥ 6.5`, `plotly`, `line_profiler`. The
live host/client example also needs `matplotlib`.

## Quickstart

A harmonic oscillator `y'' = −ω²y`, built from `DifferentialVariable`s. hydrogen
generates `der_y`, `der_z` and the Crank–Nicolson constraints; you write only the
ODE right-hand sides.

```python
import numpy as np
from hydrogen import DifferentialVariable, Model, Parameter

class Oscillator(Model):
    def declare_components(self):
        self.add_component('omega', Parameter(2 * np.pi))
        self.add_component('y', DifferentialVariable(1.0))   # y(0) = 1
        self.add_component('z', DifferentialVariable(0.0))   # z(0) = 0, z = dy/dt

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

## Core concepts

- **`Model`** — the composition primitive. Override `declare_components()` to
  register sub-models / `Variable`s / `Parameter`s via `self.add_component(...)`,
  and `declare_equations()` to return a list of SymPy expressions (each `= 0`).
- **`Variable`** — an algebraic unknown Newton solves for.
- **`Parameter`** — a constant lambdified as a runtime argument; mutate
  `param.value` between solves to sweep without recompiling.
- **`DifferentialVariable`** — a state advanced by Crank–Nicolson
  (`x_{n+1} = x_n + ½·dt·(der_x_{n+1} + der_x_n)`); adding it auto-creates `der_x`.
- **`Input`** — a time-dependent signal sampled at `t_k`/`t_{k+1}` during the CN
  step, *not* part of the solved unknowns.
- **`Port`** — a typed bundle of variables. Connect ports with `self.connect(a, b)`
  / `add_connection(...)` in `declare_equations`; the union-find pass collapses
  each wire into a single shared symbol, so connections are free at runtime.
- **`CoolPropMedium`** — wraps a CoolProp `AbstractState` and exposes
  `(p, h)`-based property callables (`rho_ph`, `T_ph`, `s_ph`, `mu_ph`, `k_ph`,
  plus `h_pT`) with derivatives, usable directly inside equations.

## Components

`import` any class from `hydrogen` (or `hydrogen.components`). The libraries:

| Domain | Connector | Components |
|---|---|---|
| **fluid** | `FluidPort_phm` `(p, h, m_dot)` | `AmbientInlet`, `AmbientOutlet`, `PressureSource`, `PressureOutlet`, `PressureVessel`, `LoopBuffer`, `Splitter`, `MixingJunction`, `TwoPortSegment`, `StraightPipe`, `AdiabaticPump`, `Valve`, `IncompressibleValve`, `CompressibleValve` |
| **thermal** | `ThermalPort_TQ` `(T, Q_dot)` | `FixedTemperature`, `FixedHeatFlow`, `ConvectiveBoundary`, `ThermalConductor`, `FlatWall`, `CylindricalWall` (`TwoNodeWall` base) |
| **power** | (fluid + thermal) | `ConjugatePipe` — a fluid pipe wrapped segment-by-segment in a metal wall |
| **control** | `RealSignal` | `Constant`, `Step`, `Ramp`, `Sine`, `Gain`, `Add`, `Sum`, `Product`, `Feedback`, `Limiter`, `Integrator`, `FirstOrder`, `PID` |

The fluid library uses a **`(p, h, m_dot)`** port convention: mass flow `m_dot`
[kg/s] is the conserved variable across every joint (no `ρ·w·A` translation that
could drift), and velocity is reconstructed internally as `w = m_dot / (ρ·A)`.

For the authoritative, always-current list (constructor parameters, structural
flags, which need a medium), call the catalogue helpers:

```python
import hydrogen
print(hydrogen.format_component_catalog())            # human-readable table
print(hydrogen.format_component_catalog("thermal"))   # one domain
rows = hydrogen.component_catalog()                   # structured, for tooling/UIs
hydrogen.available_domains()                          # ['control','fluid','power','thermal']
```

`hydrogen.test_models` also exports synthetic ODE models (`IntegrationTest`,
`SimpleODE`, …) with known closed forms, used by the integrator tests.

## Saving & loading systems (`hydrogen.serialization`)

Any system round-trips to a portable spec stamped with the hydrogen + schema
version, and you can build one entirely from data — no Python subclass — including
nested inline composites (`"type": "Model"`). Builtin component types are named by
their fully-qualified path (`hydrogen.<domain>.<Class>`) so leaves in different
domains never collide.

```python
from hydrogen import to_json, from_json, from_dict

open("system.json", "w").write(to_json(system, indent=2))   # also: to_dict
system = from_json(open("system.json").read())
system.instantiate(); system.initialise(n=1)

spec = {
    "schema_version": 1,
    "components": {
        "hot":  {"type": "hydrogen.thermofluid.FixedTemperature", "params": {"T_set": 400.0}},
        "cold": {"type": "hydrogen.thermofluid.FixedTemperature", "params": {"T_set": 300.0}},
        "rod":  {"type": "hydrogen.thermofluid.ThermalConductor",  "params": {"G": 5.0}},
    },
    "connections": [
        {"from": "hot.heat",  "to": "rod.heat_a"},
        {"from": "cold.heat", "to": "rod.heat_b"},
    ],
}
model = from_dict(spec)
```

See `tutorials/serialize_system.py` for a round-trip plus a data-defined nested
network.

## Driving a system from another tool (host/client)

For a UI or any separate tool, run the system in its **own host process** and talk
to it over a socket. The client side is stdlib-only and non-blocking, so the UI
loop stays responsive. `start_host(workers=N)` launches the host under
`mpirun -n N` when `N > 1` (needs `mpi4py`); the default `workers=1` runs a plain
subprocess with no MPI dependency.

```python
import hydrogen

service = hydrogen.start_host(workers=1)                  # spawn + connect
system  = service.load_json(open("system.json").read())   # or service.load_dict(spec)

system.instantiate(max_remove_trival_passes=5)            # heavy compile step
system.initialise(n=1)

# Streaming run: returns immediately and advances in the host. `delay` paces it to
# ~real time; `every` throttles status events. Variable data is requested separately.
system.run(dt=0.05, steps=1200, stream=True, every=20, delay=0.05)
```

**Chart any variables, any time** with on-demand streams (open several, add/drop
mid-run). `time()` / `series()` / `series_values()` each return a **live handle**
whose `.array` is refreshed together by a single `stream.update()` call (which
also returns `True` on new rows) — so every handle stays the same length each frame with no
manual clipping, and only the watched columns are transferred (never the full
record). The stream watches nothing until you ask: `series` / `series_values`
register what they need on first use (expanding a suffix to every match, so
`series_values` aggregates per-instance quantities across segments) and backfill
the full history. `list_watched_names()` reports what was registered:

```python
stream = system.vars_stream()                 # nothing watched yet
p  = stream.series("p_in")                    # 1-D handle (registers on first use)
qa = stream.series_values("m_dot_a_leak")     # 2-D handle (cols = matches)
t  = stream.time()                            # 1-D handle
while running:
    if stream.update():                       # refills every .array; True on new rows
        redraw(t.array, p.array, qa.array.sum(axis=1))
    for ev in system.poll_events():           # status / log / done / error
        if ev["type"] in ("done", "closed", "error"):
            running = False
stream.list_watched_names()                   # ['…p_in', '…seg0.m_dot_a_leak', …]
```

**Control the run** at step boundaries (never mid-solve): `system.pause()`,
`system.resume()`, `system.stop()`.

**Discover variables for a picker** with `system.var_tree()` — a nested tree whose
nodes have a unique `path` (use as the UI key), a `leaf` flag, a `count` of
selectable descendants, and, on leaves, the exact `full` name to hand to
`stream.series` / `get_series`. Point queries (`get_state`), history slices
(`get_record` row-major, `get_series` column-major), `list_vars`, and `status()`
are available too. Finish with `service.shutdown()` (or use
`with hydrogen.start_host() as service:`).

A complete live example — a matplotlib window with Pause/Resume/Stop buttons fed by
a variable stream — is `tutorials/host_client/run_client.py`.

## Tutorials

Practical, self-validating worked guides in `tutorials/` (each writes artifacts
under the git-ignored `local_results/tutorials/`, overridable via
`HYDROGEN_LOCAL_RESULTS`). Run any with `python tutorials/<script>.py`:

| Script | What it shows |
|---|---|
| `run_system.py` | Heated mass-flow loop; integrator validated against analytics |
| `fill_vessel.py` | Pressure-driven vessel charging transient |
| `pipe_tree.py` | Recursive K-ary tree of pipes (`Splitter` + `PressureOutlet`) |
| `loop_pump_pipe.py` | Closed pump+pipe loop using a `LoopBuffer` |
| `mixing_junction_reversal.py` | Reversible-flow `MixingJunction` |
| `flat_wall.py`, `conjugate_pipe.py` | Thermal wall / coupled power pipe |
| `control_valve.py` | Control blocks driving a compressible valve |
| `two_phase_boiler.py` | Single-phase closures vs smooth HEM boiling |
| `serialize_system.py` | Save/load JSON + build a system from a dict |
| `saved_system_2.py` | Headless twin of a UI project (source → valve → pipe → tank) |
| `data_structures_info.py` | Inspect the component catalogue (UI tooling view) |
| `host_client/run_client.py` | Drive a host from a separate (UI) process, live chart |
| `h2_permeation_pressurize/` | Author a model, save JSON, run it locally and via a host |

## Benchmarks

Scaling / simulation-speed / analytical-correctness harnesses in `benchmarks/`
(less polished than tutorials; artifacts under `local_results/benchmarks/`). The
analytical ones also assert correctness against closed-form solutions, so they
double as regression checks. Run with `python benchmarks/<script>.py`:

| Script | What it measures |
|---|---|
| `analytical_integrator.py` | Integrator accuracy vs closed form (Crank-Nicolson order check) |
| `scaling_segmented_pipe.py` | `SegmentedChannel` scaling + straight-vs-segmented engine match |
| `bench_adaptive.py` | Adaptive vs fixed-step time stepping |
| `bench_blt.py` | Solve time with vs without BLT |
| `bench_pipe_tree.py` | Pipe-tree instantiate/solve scaling + correctness fingerprint |
| `bench_segmented.py` | `SegmentedChannel` vs `StraightPipe` (speed + match) |
| `bench_feos_vs_coolprop.py` | CoolProp vs feos medium A/B |
| `bench_line_search.py` | Damped Newton vs backtracking line search |

## API Documentation

This README is the narrative documentation. A full **API reference** is generated
directly from the package docstrings with [MkDocs](https://www.mkdocs.org/) +
[mkdocstrings](https://mkdocstrings.github.io/) (Material theme), so it never goes
stale — the reference pages and nav are produced by `docs/gen_ref_pages.py` at
build time.

```bash
pip install -e ".[docs]"     # mkdocs-material, mkdocstrings, gen-files, literate-nav
mkdocs serve                 # live preview at http://127.0.0.1:8000 (auto-reloads)
mkdocs build                 # render the static site into ./site
```

`mkdocs serve` is the easiest way to browse: the **Home** page is this README and
**API Reference** is the per-module docstring documentation. Without building the
site you can still read the same content directly as docstrings (`help(obj)` /
your IDE), and `hydrogen.format_component_catalog()` prints the live component
reference (parameters, flags, domains).

## Tests

```bash
pytest                                  # full suite (tutorial + benchmark scripts deselected)
pytest -m tutorials                     # also run the end-to-end tutorial scripts
pytest -m benchmarks                    # run the scaling / analytical benchmark scripts
pytest -v tests/test_integration.py     # CN integrator vs analytical solutions
```

## Performance & tuning

Defaults favour accuracy and clean cold-cache numbers. The levers that move the
needle, roughly in impact order:

- **`CoolPropMedium(backend="BICUBIC&HEOS")`** — bicubic interpolation table
  (lazy, HEOS fallback). ~2.7× faster solve / ~3.6× faster init vs the default
  `"HEOS"`, at ~1e-4 accuracy on `ρ`. Use it whenever you don't need
  reference-grade thermodynamics.
- **`CoolPropMedium(scalar_cache_maxsize=...)`** — `lru_cache` size on each scalar
  property eval (default 100). Critical for HEOS at scale: if the working set of
  unique `(p, h)` states exceeds it the cache thrashes and HEOS goes super-linear.
  Rule of thumb: set `≈ 5 × n_active_vars`. RAM cost is negligible.
- **`Model.instantiate(cse=True, max_remove_trival_passes=N, aditional_modules=...,
  lambda_cache_dir=...)`** — CSE before codegen (keep on); more trivial-removal
  passes (`4–5`) shrink the Newton vector a few % more on deep compositions; pass
  `aditional_modules=medium.modules` to keep the scalar cache benefit; the on-disk
  lambda cache (content-addressed) lets repeated runs skip codegen.
- **`Model.solve_adaptive_step(dt_target=..., strategy=...)`** — adaptive
  time-stepping for stiff/uneven transients (`"predictor_corrector"` default,
  also `"derivative_limit"`, `"richardson"`, `"fixed"`). Typically 2–3× faster
  than the right fixed `dt`; the caller still drives `next_step()`.
- **Newton knobs** (shared by `initialise` / `solve_dae_step`): `relaxation` (drop
  to `0.5` for hard `t=0` solves), `tol`, `max_iter`,
  `raise_on_no_convergence`.
- **Env vars**: `HYDROGEN_LAMBDA_CACHE` (cache dir; `""`/`"0"` disables),
  `HYDROGEN_PARALLEL_LAMBDIFY` (worker count for parallel codegen),
  `HYDROGEN_VECTORISE_MIN` (min instances/template before the vectorised evaluator
  engages, default 8).

## Notes & limitations

- **Two-phase via a smooth HEM flag.** Fluid segments default to `multiphase="single"`
  (single-phase property closures: correct and fast in one phase, but Newton goes
  singular at the saturation line because CoolProp's `(p, h)` partials collapse / are
  undefined in the dome). Pass `multiphase="HEM"` (`TwoPortSegment`/`StraightPipe`) to
  use the smooth homogeneous-equilibrium variants — same CoolProp values, but with
  consistent, continuous finite-difference partials — so Newton boils cleanly
  from liquid through the dome to superheated steam (no events). See
  `tutorials/two_phase_boiler.py`. Boiling density falls off a cliff (`vg/vf ~ 300`), so
  the HEM path either needs a damped step (`relaxation ~ 0.2`) with a fine continuation
  ramp, or — preferably — `line_search=True` (see below) to take full steps and cross
  the dome on a coarse ramp with no hand-tuned damping.
- **Globalization: backtracking line search.** Pass `line_search=True` to `initialise`,
  `solve_dae_step`, or `custom_solve` for a feasibility-guarded backtracking step: it
  takes the full Newton step (`relaxation=1.0`) wherever that lands in a valid state and
  automatically backs off (halving the step) only where the full step would overshoot
  into an infeasible region — e.g. a Newton step from the liquid side of a boiling
  density cliff that would predict negative density (the residual goes non-finite / a
  property call raises, so that step is rejected). This removes the need to hand-tune
  `relaxation` on stiff property cliffs. It costs `1 + n_backtracks` extra residual
  evaluations per iteration and is opt-in (default `line_search=False` = plain damped
  Newton). A pure Armijo decrease test is deliberately avoided because the mixed-unit
  residual (Pa `~1e5` dwarfs continuity `~1e-3`) makes the raw norm a poor merit.
- **Initial guesses matter.** Strong startup transients can need warm starts and
  `initialise(relaxation=0.5)` or `initialise(line_search=True)`.
- **Ill-conditioned Jacobians** (mixing Pa, J/kg, m/s, kg in one vector) can show
  high `cond`; Newton still converges in practice. Per-variable `atol` overrides
  (`Variable(value, unit, atol=...)`) help.

## Project layout

```
hydrogen/
├── hydrogen/
│   ├── model.py            # Model, Variable, DifferentialVariable, Input, Newton, CN
│   ├── ports.py            # Port base + connection machinery
│   ├── medium.py           # CoolPropMedium + sympy-able property functions
│   ├── numerics.py         # lambdify_compat + numba-JITed Newton primitives
│   ├── caching.py          # numpy_cache + on-disk lambda cache
│   ├── plotting.py         # plot_results (plotly) + local_results_path
│   ├── components/         # fluid / thermal / power / control libraries
│   ├── serialization/      # to_dict/from_dict/to_json/from_json + component registry
│   ├── service/            # out-of-process host + client (start_host, SystemProxy)
│   └── utilities/          # Interpolation1D / Interpolation2D
├── tutorials/              # practical, self-validating worked guides
├── benchmarks/             # scaling / speed / analytical-correctness harnesses
├── tests/
└── docs/                   # mkdocs sources
```
