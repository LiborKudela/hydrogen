"""Dynamic `MixingJunction` driven by sinusoidal mass flow, solved adaptively.

System layout
-------------

    +------- bnd_0 (sinusoidal +M*sin(w*t),  warm h) -------+
    |                                                       |
    |                                                       v
    |                                            MixingJunction (dynamic, N=2)
    |                                                       ^
    |                                                       |
    +------- bnd_1 (sinusoidal -M*sin(w*t),  cool h) -------+

Two prescribed-mass-flow boundaries push fluid through a 2-port dynamic
`MixingJunction`.  Their `m_dot_out`s are 180 deg out of phase so the
algebraic mass conservation holds at every instant (the junction's `m`
storage stays flat).  Each cycle, port 0 reverses from "sourcing warm fluid"
to "sinking the junction's mix", and port 1 does the opposite.

The interesting bit is that the smooth-blend port-enthalpy closure has its
hardest moments at the zero-crossings of `sin(w*t)`, where `alpha_k`
transits through 1/2 and the sign of every donor-cell upwind flips.  We
solve this transient with the `predictor_corrector` adaptive controller
and plot the `dt` history -- expect dt to shrink near each reversal and
grow during the smooth peaks.

Why a *dynamic* junction here
-----------------------------
Quasi-static `MixingJunction(dynamic=False)` would over-determine this
topology: with every `m_dot` Parameter-pinned, the algebraic constraint
`sum_k m_dot_k = 0` collapses to `sum_of_two_sines = 0`, which is satisfied
only because the phases happen to be 180 deg apart.  The trivial reducer
sees that as a constant `0 = 0` and the Jacobian goes rank-deficient.
Dynamic mode side-steps the issue: `dm/dt = sum m_dot_k` is a real
differential equation, not a hard constraint -- it just sits at zero
because the inputs happen to balance.  See
`tests/test_mixing_junction.py::FourPortQuasiStaticSystem` for the
quasi-static topology that does work (one pressure-pin port + flow pins).

Run with `python examples/mixing_junction_reversal.py` from the project root.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

from hydrogen import (  # noqa: E402
    CoolPropMedium,
    Model,
    Parameter,
    Variable,
    local_results_path,
    plot_results,
)
from hydrogen.components.thermofluid.flow import MixingJunction  # noqa: E402

# --- Physical / numerical parameters ----------------------------------------

P_INIT      = 101325.0   # Pa, junction initial pressure (also boundary p)
T_JUNC_INIT = 293.15     # K, junction initial temperature
T_WARM      = 343.15     # K, port 0's supplied stream-in enthalpy
T_COOL      = 273.15     # K, port 1's supplied stream-in enthalpy
V_JUNC      = 5e-4       # m^3, junction control volume (0.5 L)

M_DOT_AMP   = 5e-3       # kg/s, peak prescribed mass flow magnitude
F_HZ        = 1.0        # Hz, sinusoid frequency (1 Hz = 1 reversal pair per second)
T_END       = 2.5        # s, total simulated time
DT_TARGET   = 0.05       # s, adaptive controller's dt hint (grows / shrinks from here)
DT_MIN      = 1e-5       # s, hard floor (must be well below the reversal sharpness)
TOL_LOCAL   = 1e-3       # local truncation tol used by the predictor-corrector
# Per-variable absolute tolerances -- the adaptive step-acceptance metric is
# scaled by these, so we want each variable's atol to be ~1% of its operating
# magnitude.  Without this, pressure (~1e5 Pa) dominates and the metric never
# notices that h is doing the interesting work.
ATOL = {
    "p":     100.0,      # Pa
    "h":     100.0,      # J/kg
    "h_k":   100.0,      # J/kg
    "m":     1e-6,       # kg  (junction mass barely moves)
    "U":     1.0,        # J
    "m_dot": 1e-5,       # kg/s
}


class _SinusoidalFlowBoundary(Model):
    """Boundary whose `m_dot_out` is `amplitude * sin(omega*t + phase)` and
    whose stream-in enthalpy `h_set_out` is pinned to a Parameter.

    The `m_dot` residual depends on the framework's time symbol `t`, so the
    trivial reducer can't collapse `m_dot_out` to a Parameter (its target is
    a function of `t`, not a constant).  That keeps `m_dot_out` a live
    Variable in every downstream expression -- crucial for the junction's
    smooth-blend port closure which has `m_dot_k` baked into `alpha_k`.
    """

    def __init__(self, medium: CoolPropMedium, amplitude: float,
                 omega: float, phase: float, h_set: float):
        self.medium = medium
        self._amplitude = amplitude
        self._omega = omega
        self._phase = phase
        self._h_set = h_set
        super().__init__()

    def declare_components(self):
        self.add_component('amplitude', Parameter(self._amplitude, "kg/s"))
        self.add_component('omega',     Parameter(self._omega, "rad/s"))
        self.add_component('phase',     Parameter(self._phase, "rad"))
        self.add_component('h_set',     Parameter(self._h_set, "J/kg"))
        # Initial-guess values are reasonable for t = 0 (sin(phase) ~ 0 if
        # phase = 0, else -amplitude * sin(phase) under "flow into me").
        self.add_component('p_out',     Variable(P_INIT, "Pa"))
        self.add_component('h_set_out', Variable(self._h_set, "J/kg"))
        self.add_component('m_dot_out',
                           Variable(-self._amplitude * np.sin(self._phase),
                                    "kg/s", atol=ATOL["m_dot"]))

    def declare_equations(self):
        # Under "flow into me", `m_dot_out` is positive when fluid enters
        # the boundary through its out-face.  The user-facing `amplitude`
        # is the boundary's physical *outflow* rate, so the residual reads
        # `m_dot_out = -amp*sin(omega*t + phase)`.
        t = self.t_symbols[0]
        amp = self['amplitude'].symbol
        omega = self['omega'].symbol
        phase = self['phase'].symbol
        return [
            self['m_dot_out'].symbol + amp * sp.sin(omega * t + phase),
            self['h_set_out'].symbol - self['h_set'].symbol,
        ]


class TwoPortMixingSystem(Model):
    """Two sinusoidal boundaries (180 deg out of phase) feeding a 2-port
    dynamic `MixingJunction`.  Demonstrates clean flow reversal under
    adaptive timestepping."""

    def declare_components(self):
        # BICUBIC&HEOS is plenty for this 2-port toy; we don't even hit the
        # CoolProp cache hard enough for the backend choice to matter.
        self.medium = CoolPropMedium("Air", disable_warnings=True,
                                     backend="BICUBIC&HEOS")

        self._h_warm = float(self.medium.eval_h_pT(P_INIT, T_WARM))
        self._h_cool = float(self.medium.eval_h_pT(P_INIT, T_COOL))

        omega = 2.0 * np.pi * F_HZ

        # Port 0: m_dot_0(t) = +A*sin(omega*t), stream-in = warm
        self.add_component(
            'bnd_0',
            _SinusoidalFlowBoundary(self.medium, +M_DOT_AMP, omega, 0.0, self._h_warm),
        )
        # Port 1: m_dot_1(t) = -A*sin(omega*t), stream-in = cool
        # (mass balance closes exactly: sum_k m_dot_k(t) = 0 for all t)
        self.add_component(
            'bnd_1',
            _SinusoidalFlowBoundary(self.medium, -M_DOT_AMP, omega, 0.0, self._h_cool),
        )

        self.add_component(
            'junction',
            MixingJunction(self.medium, N=2, V=V_JUNC,
                           p_init=P_INIT, T_init=T_JUNC_INIT,
                           m_dot_eps=1e-6, dynamic=True),
        )

    def declare_equations(self):
        # m_dot is a flow variable -- under "flow into me", boundary's
        # `m_dot_out` and junction's `m_dot_k` both measure fluid entering
        # their own component at the shared interface, so they sum to zero
        # (sign=-1 connection).  `p` and `h_set` are across variables and
        # stay direct equalities.
        for k in range(2):
            self.add_connection(
                self[f'bnd_{k}']['p_out'],
                self['junction'][f'p_{k}'],
            )
            self.add_connection(
                self[f'bnd_{k}']['h_set_out'],
                self['junction'][f'h_set_{k}'],
            )
            self.add_connection(
                self[f'bnd_{k}']['m_dot_out'],
                self['junction'][f'm_dot_{k}'],
                sign=-1,
            )
        return []

    def apply_atols(self):
        """Tag the junction's interesting Variables with per-variable atols.

        Called from `main` AFTER instantiate (so the active-var references
        exist).  This lets the predictor-corrector controller compare each
        variable to a sensible scale instead of the global default.
        """
        scale_for = {
            ".junction.p":  ATOL["p"],
            ".junction.h":  ATOL["h"],
            ".junction.h_0": ATOL["h_k"],
            ".junction.h_1": ATOL["h_k"],
            ".junction.m":  ATOL["m"],
            ".junction.U":  ATOL["U"],
        }
        for v in self.active_vars_references:
            full = getattr(v, "full_name", "")
            for suffix, atol in scale_for.items():
                if full.endswith(suffix):
                    v.atol = atol
                    break


def _trace(record, suffix: str) -> np.ndarray:
    names = list(record['vars_names'])
    state = np.asarray(record['state'])
    idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
    return state[:, idx]


def main():
    print("Building model...")
    system = TwoPortMixingSystem()

    print("Instantiating (symbolic Jacobian + lambdify)...")
    t0 = time.time()
    system.instantiate(
        aditional_modules=system.medium.modules,
        max_remove_trival_passes=3,
    )
    print(f"  instantiate: {time.time() - t0:.2f} s")

    system.apply_atols()

    print("Initialising at t = 0...")
    t0 = time.time()
    system.initialise(relaxation=0.5, max_iter=200)
    print(f"  initialise:  {time.time() - t0:.2f} s")

    # --- Transient solve with adaptive step --------------------------------
    print()
    print(f"Adaptive transient until t = {T_END} s "
          f"(strategy=predictor_corrector, dt_target={DT_TARGET}s, "
          f"dt_min={DT_MIN}s, tol_local={TOL_LOCAL})...")
    t0 = time.time()
    dt_history = []
    t_history = []
    n_rejections = 0
    n_iters_total = 0
    dt_try = DT_TARGET

    while system.get_t_value() < T_END - 1e-12:
        dt_try = min(dt_try, T_END - system.get_t_value())
        dt_used, info = system.solve_adaptive_step(
            dt_try,
            strategy={"name": "predictor_corrector", "tol_local": TOL_LOCAL},
            dt_min=DT_MIN, dt_max=4 * DT_TARGET,
            relaxation=1.0, tol=1e-6, max_iter=200,
        )
        system.next_step()
        dt_history.append(dt_used)
        t_history.append(system.get_t_value())
        n_rejections += info["rejections"]
        n_iters_total += info["n_iters"]
        # next call should try growing back toward DT_TARGET
        dt_try = min(DT_TARGET, dt_used * 1.5)

    elapsed = time.time() - t0
    print(f"  solve:       {elapsed:.2f} s   "
          f"({len(dt_history)} accepted steps, {n_rejections} rejections, "
          f"{n_iters_total} total Newton iters)")
    print(f"  mean dt:     {np.mean(dt_history) * 1000:.2f} ms")
    print(f"  min  dt:     {np.min(dt_history) * 1000:.2f} ms")
    print(f"  max  dt:     {np.max(dt_history) * 1000:.2f} ms")

    # --- Diagnostics ------------------------------------------------------
    rec = system.record
    t_arr = np.asarray(rec['time'])
    m_dot_0 = _trace(rec, ".junction.m_dot_0")
    m_dot_1 = _trace(rec, ".junction.m_dot_1")
    h_junc = _trace(rec, ".junction.h")
    h_0 = _trace(rec, ".junction.h_0")
    h_1 = _trace(rec, ".junction.h_1")
    m_junc = _trace(rec, ".junction.m")

    sign_flips_0 = int(np.sum(np.diff(np.sign(m_dot_0)) != 0))
    print()
    print("=== Reversal stats ===")
    print(f"port 0 sign flips : {sign_flips_0}  "
          f"(expected ~ {int(2 * F_HZ * T_END)} for {F_HZ} Hz over {T_END} s)")
    print(f"port 0 m_dot range: [{m_dot_0.min() * 1000:+.2f}, "
          f"{m_dot_0.max() * 1000:+.2f}] g/s")
    print(f"junction h range  : [{h_junc.min():.1f}, {h_junc.max():.1f}] J/kg")
    print(f"port 0 carrier h  : [{h_0.min():.1f}, {h_0.max():.1f}] J/kg")
    print(f"port 1 carrier h  : [{h_1.min():.1f}, {h_1.max():.1f}] J/kg")
    print(f"junction mass drift: {(m_junc.max() - m_junc.min()) * 1e9:.2f} ng "
          f"(should be ~ 0, sum m_dot == 0 by construction)")

    # Self-validation: the drive should actually reverse the flow at port 0,
    # and the well-mixed junction's stored mass must stay essentially constant
    # (sum of port mass flows is zero by construction).
    assert sign_flips_0 > 0, "port 0 flow should reverse under the sinusoidal drive"
    assert (m_junc.max() - m_junc.min()) < 1e-6, "junction mass drift should be ~0"

    # --- Plot ------------------------------------------------------------
    out_path = local_results_path("examples", "mixing_junction_reversal.html")
    plot_results(rec, out_path)
    print(f"\nPlot written to {out_path}")
    print("(open in a browser; m_dot_0 is the sinusoid; h_junction oscillates")
    print(" out of phase with it; dt squeezes around the zero-crossings.)")


if __name__ == "__main__":
    main()
