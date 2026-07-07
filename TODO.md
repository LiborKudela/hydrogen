# TODO

Backlog of things to fix or improve later. Each entry has a title and enough
context to reconstruct what was meant without re-deriving it.

## Reducers: capacitive merge for diff-state identities written as plain equations

**Status:** open
**Area:** `hydrogen/model.py` (`remove_trivial_equations`, `_differential_state_symbols`)

Since the differential-state guard (added when the linear-block pass was caught
eliminating `p_pipe` and corrupting its TR-BDF2 integration), the reducers
refuse to pivot on any `DifferentialVariable` state. That is correct, but it
means an identity between two differential states written as a *plain equation*
(e.g. `T_a - T_b == 0` in `declare_equations`) is no longer reduced at all --
both states, both derivative companions, both closures, and the identity row
all survive. Only identities wired via `connect()` get the proper
*capacitive-node merge* (union-find in `instantiate`, which aliases the states
AND their `der_*` companions so the two `C*der = ...` ODE rows constrain the
same derivative and the capacities effectively add).

**Task:** teach `remove_trivial_equations` to recognise the 2-var linear case
`a*x + b*y + c == 0` where BOTH `x` and `y` are differential states and `a, b`
are plain Numbers (constant Parameters at most; never Input-dependent), and
perform the same merge the connection eliminator does instead of skipping:

1. substitute `x := k*y + c'` (`k = -b/a`, `c' = -c/a`) plus the `_prev` mirror;
2. substitute `der_x := k*der_y` (offset drops out of the derivative) plus its
   `_prev` mirror -- needs a state->der map in the reducer, i.e. extend
   `_differential_state_symbols` to also return the `diff_state_der` dict the
   connection eliminator already builds;
3. drop the now-redundant Crank-Nicolson closure of `x` explicitly (for
   `|k| != 1` it becomes a *scaled* duplicate that the dedup pass's
   `(coeff, rest)` signature will not catch); bookkeeping stays square:
   2 vars removed (`x`, `der_x`), 2 eqs removed (identity + closure).

Constraints: strictly linear-constant identities only (no `x = f(y)` -- that is
real index reduction requiring constraint differentiation); no time-varying
coefficients (the `der` alias would gain a `k_dot*y` term); deterministic
representative choice (name order, like `_rep_key`); log the merge. Chains
`x == y == z` fall out of the existing multi-pass loop. Optionally add the same
special case to `remove_linear_block_equations` for identities buried in >=3-var
linear rows (rarer, fiddlier pivot bookkeeping).

This is purely a reduction/conditioning optimisation -- correctness is already
guaranteed by the guard. Payoff: model-size parity between "wired via
`connect()`" and "wired via equation".

## Acoustic level: vapor-cavity / column-separation handling

**Status:** DONE (DVCM implemented); residual follow-ups below
**Area:** `hydrogen/components/thermofluid/flow.py` (`_declare_primitive`,
`dynamic="acoustic"`), validated via `benchmarks/bench_hammer_validation.py`
and `tests/test_dynamic_levels_reference.py` (column-separation tests)

Original problem: when a reflected rarefaction pulled a cell to the vapor
pressure, the HEM `(p, h)` state entered the two-phase dome, `drho/dp|h`
jumped ~4 orders of magnitude, the constant-reference acoustic row scaling
became meaningless and the step Jacobian went exactly singular -- the run
died at cavitation onset.

Implemented fix (`cavitation=True`, acoustic level only, wired through
`Pipe` which defaults `p_vap` to the saturation pressure at `T_wall_init`):
the implicit-FV Discrete Vapor Cavity Model.  Per cell: a cavity state
`V_cav_i >= 0` displacing `rho*dV_cav/dt` of mass storage and
`rho*h*dV_cav/dt` of energy storage (so `hc` is invariant while a cavity
grows -- omitting the energy twin lets `hc` drift into the dome, which is
exactly the failure it is meant to prevent), closed by the smoothed
Fischer-Burmeister complementarity `(pc - p_vap) >= 0 _|_ V_cav >= 0`
(solution manifold `a*b = cav_eps^2/2`), enforced in index-1 form
`Phi + tau*dPhi/dt = 0` so the row never degenerates at the dt=0
consistency solve.  All EoS lookups additionally see a smooth pressure
floor `p_eos = smoothmax(p, p_vap)` so no Newton ITERATE can drag a
property call below the saturation line (HEOS single-phase partials NaN
there).  Liquid stays single-phase throughout; the clamp, cavity growth,
collapse and the re-emitted shock are all resolved (water column-separation
tests match the rigid-column cavity-lifetime estimate; the DLR LN2 replay
now runs through the whole cavitation phase with sensible clamp/rebound
behaviour).

Remaining follow-ups:
1. `p_vap` is one constant per channel (fine for isothermal hammer events);
   a heated channel would want `p_sat(hc_i)` per cell -- needs a symbolic
   saturation-pressure function with derivatives from the medium.
2. `IncompressibleValve` at `theta -> 0` exactly (zero-flow enthalpy row
   degenerates; the DLR benchmark data keeps a 1e-3 leak floor as a
   workaround).
3. CoolProp tabular backends broken for subcooled LN2 (BICUBIC: invalid ph
   cells; TTSE: transport wants "four valid corners", pT flash picks the
   vapor branch).  MITIGATED by `TabulatedMedium` (`hydrogen/tabulated.py`):
   a dome-conforming spline surrogate of the HEOS backend (mapped-sigma
   liquid/vapor surfaces + saturation-line splines + exact HEM mixture rules
   inside the dome; analytic first AND second partials; vectorised; disk-
   cached).  `--tab` on the benchmark runs it: ~7x faster per step than
   HEOS with <~1e-4 surrogate error.  Remaining niggle: the per-call python
   overhead of the spline path (~0.1 ms scalar) means the speedup is
   template-vectorisation bound, not table bound; a C/numba kernel for
   `_Bicubic2D.ev_all` + `_Sat1D` would buy another ~2-3x.
   PARTIALLY DONE: `Model.instantiate(numba=True)` JIT-compiles whole
   vectorised equation templates in nopython mode against `@njit` twins of
   the tabulated evaluators (single-phase windows only; `TabulatedMedium`
   attaches them as `_hydrogen_numba`; `NumbaFriendlyPrinter` rewrites
   Min/Max/Piecewise as nested minimum/maximum/where).  Values match the
   numpy path to machine epsilon.  Measured on the ULg advection replay
   (`--numba`): 2.6x per step at N=10, 1.6x at N=100, ~1.1x at N=1000 --
   the win shrinks as the numpy path amortises and the sparse solve takes
   over.  Blockers for more: (a) ~45 s JIT compile per instantiate (numba
   `cache=True` needs file-backed sources -- write the numba variant of the
   template source to the lambdify disk cache as importable .py modules);
   (b) two-phase windows keep the numpy path (dome/blend logic is python);
   (c) the acoustic LN2 case is the real target once (b) lands.
4. DVCM clamps at `p_vap`; the DLR sensors read below it during the vapor
   phase (sensor response inside the cavity / distributed vaporous zone),
   so the RMSE during cavitation carries an irreducible offset.
5. DVCM collapse is too late and too strong on the full-trace replay
   (mild case: measured rebound ~12 bar at ~0.20 s; model ~27 bar at
   ~0.32 s, then ringing).  Known DVCM behaviour: a pure-vapor cavity
   clamped exactly at `p_vap` neither cushions the collapse nor accounts
   for free/desorbed gas.  Upgrade path: DGCM -- give each cell cavity a
   small isothermal free-gas mass (`p_gas*V_gas = const` added to the
   complementarity), which damps and advances the collapse; literature
   default alpha_gas ~ 1e-7 at reference pressure.

## Multiphase (two-phase dome) water benchmark + tab dome robustness

**Status:** DONE
**Area:** `benchmarks/bench_boiling_multiphase.py`, `hydrogen/numerics.py`,
`hydrogen/components/thermofluid/flow.py`

New boiling-water testbench: a heated 4-segment steady pipe ramps the wall heat
until the outlet crosses the WHOLE dome (subcooled liquid -> saturated mixture
-> superheat, x: -0.24 -> 1.37 at 5 bar) with an exact per-step energy-balance
reference.  Runs on both property paths (`--medium coolprop|tab|both`).  Both
now traverse the full dome with matching results: energy defect 2.2e-5 of h_fg,
max |dT_out| = 0.09 K, max |dh_out| = 0 J/kg (chart in
`benchmarks/plots/boiling_multiphase.png`).

Two fixes were needed to make the TabulatedMedium path cross the dome:
1. **Jacobian equilibration** (`numerics._equilibrated_splu_solve`): the steady
   momentum/continuity Jacobian in the dome is dominated by an all-pressures
   near-null mode; unscaled cond reaches ~1e22 (density drops ~100x across the
   cells) and SuperLU reports a spurious "Factor is exactly singular".  One pass
   of row+column max-norm equilibration before `splu` brings cond to ~1e10 at
   O(nnz) cost.  Applied to BOTH monolithic solve entry points; regression-safe
   (only improves conditioning, never changes the solution).  This is the fix
   that actually let the tab path cross the dome.
2. **Benchmark continuation**: fine fixed heat steps (>=150) + bounded adaptive
   bisection on failure + graceful ramp truncation; window sized to enclose the
   whole trajectory (deep superheat, so no log-linear T extrapolation).

(An earlier attempt flooring `Pr`/`fr` with `sp.Max` in `calculate_nu_smooth`
was REVERTED: it was bit-identical on the working trajectory yet its `Max`
derivatives exploded the heat template's Jacobian into `Piecewise`, ballooning
the cached lambda source and pushing `bench_segmented inst 100` from ~8 s to
~73 s.  Equilibration alone is sufficient.)

Note: the tab path is ~0.3x CoolProp here (BICUBIC&HEOS water flash is already
fast; tab's value is HEOS-quality accuracy + numba-compatibility, not raw speed
vs BICUBIC).  `--numba` runs and falls back to numpy for the dome templates
(no dome numba kernel yet -- see the numba blockers under item 3 above).

## LN2 hammer benchmark: S1 peak needs a real tank-inlet boundary

**Status:** open
**Area:** `benchmarks/bench_hammer_validation.py` (rig topology)

With the choked ball valve (ISA liquid choking + `trim_exp=2`) the S2/S3
peak rises match the measurement within ~3 %.  S1 (x/L = 0.065, 0.6 m from
the HP tank) still underpredicts its peak by ~40 %: the model terminates the
pipe in an ideal constant-pressure node (`PressureSource`) AT x = 0, so any
pressure rise at S1 is cancelled by the reflected wave after
2*x_S1/a ~ 1.6 ms -- yet the measured S1 holds 15-20 bar for ~5 ms.  The
real rig has feed hardware between the run tank and the pipe inlet (elbows /
manifold / hand valve); on millisecond timescales its liquid inertance makes
the inlet look partially CLOSED (positive reflection) before the tank
compliance takes over.  Grid refinement helps only marginally
(N=12 -> 24: -49 % -> -39 %).

**Task:** model the tank connection as a short feed element (acoustic stub of
the actual feed-line geometry, or a lumped inertance + the measured p_hp as
the source pressure -- `meta.json` already carries `p_hp_bar`).  Needs the
feed-line dimensions from the DLR rig description (Klein et al., Exp Fluids
2023); do not fit it to S1 itself.
