"""Closed-loop circulation: AdiabaticPump -> StraightPipe -> LoopBuffer -> back to pump.

System layout (everything wired closed via `add_connection`):

    +-->  AdiabaticPump  -->  StraightPipe  -->  LoopBuffer  --+
    |                                                          |
    +-----------------( buffer.out -> pump.in )----------------+

Why the loop needs a `LoopBuffer`
---------------------------------
A pure pump+pipe closed loop is **structurally rank-deficient by 2** in
steady state (the loop-closing continuity equation is implied by the
per-segment continuity equations -- `rho*w` is conserved everywhere -- and
the loop-closing energy equation is the same kind of tautology because
adiabatic flow conserves `h + w**2/2`).  The framework's Newton solve
needs a square non-singular Jacobian, so the loop refuses to instantiate
without something to break those redundancies.

`LoopBuffer` (in `hydrogen/components.py`) is a two-port well-mixed
lumped-volume vessel that introduces mass and energy storage into the
loop.  Its `m` and `U` are differential states whose own residuals --
`dm/dt = m_in_dot - m_out_dot` and `dU/dt = m_in_dot*h_in - m_out_dot*h`
-- replace the redundant loop-closure equations.  Even at steady state
where `dm/dt = dU/dt = 0`, the residuals constrain `(m, U)` (and hence
`(p, h)` via the EoS closure) to specific values rather than collapsing
to `0 = 0`, so the global Jacobian stays full rank.

Anchors
-------
The three classic closed-loop anchors land naturally on the buffer plus
one explicit equation:

  1. Pressure level   ->  buffer's initial state (`p_init`) via the EoS
                          closure `m_init = rho(p_init, h_init) * V`.
  2. Enthalpy level   ->  buffer's initial state (`T_init` -> `h_init`),
                          again via `U_init = m_init*h_init - p_init*V`.
  3. Mass flow        ->  one explicit equation `rho*w_in*A == m_dot_target`
                          returned from `declare_equations`, which fixes
                          the pump's free `a_iz` strength.

`m` and `U` are conserved in this purely-adiabatic closed loop, so they
stay at the initial values throughout the simulation -- the buffer
behaves as a steady operating point, NOT as a transient capacitor.  Its
job here is purely to supply the two missing equations that anchor the
loop.

Time-varying drive
------------------
Everything *outside* the buffer is algebraic, so the "transient" comes
from a time-varying mass-flow target: `m_dot_target(t)` is a sinusoid via
the framework's `t` symbol, and the loop tracks it quasi-statically with
`pump.a_iz` and the pump head responding step by step.

Run with `python examples/loop_pump_pipe.py` from the project root.
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
    AdiabaticPump,
    CoolPropMedium,
    LoopBuffer,
    Model,
    Parameter,
    StraightPipe,
    plot_results,
)


# Geometry ------------------------------------------------------------------------------
PIPE_D = 0.02            # m   (20 mm bore)
PIPE_L = 5.0             # m
PIPE_EPSILON = 1e-5      # m   (smooth wall)
N_SEGMENTS = 4           # pipe discretisation
A_PORT = np.pi * PIPE_D ** 2 / 4
P_PERIM = np.pi * PIPE_D
BUFFER_V = 1e-3          # m^3 (1 L lumped buffer; the actual value barely matters since
                         #      it just sets the loop's stored mass/energy operating point)

# Loop operating point (held by the buffer's initial state) ----------------------------
P_LOOP = 2.0e5           # Pa, anchored loop pressure (= buffer.p_init)
T_LOOP = 300.0           # K,  anchored loop temperature -> sets buffer.h_init

# Mass-flow drive: m_dot_target(t) = M_DOT_BASE + M_DOT_AMP * sin(2*pi*F_HZ*t)
M_DOT_BASE = 0.020       # kg/s ~ steady operating point
M_DOT_AMP = 0.010        # kg/s ~ +/- 50% modulation
F_HZ = 0.5               # Hz   (period 2 s)

# Time-stepping -------------------------------------------------------------------------
T_END = 4.0              # s   (two full sinusoid periods)
DT = 0.05                # s


class PumpedLoop(Model):
    """`AdiabaticPump -> StraightPipe -> LoopBuffer -> pump`, wired in a true closed loop.

    The `LoopBuffer` breaks the steady-state continuity / energy rank-deficiency
    by carrying `m` and `U` as differential states.  The mass-flow operating
    point is set by one anchor equation; pressure and enthalpy levels are
    anchored implicitly via the buffer's initial state.
    """

    def declare_components(self):
        # `BICUBIC&HEOS` keeps the inner-Newton CoolProp calls fast; HEOS works
        # fine too if you want bit-exact reference quality (see README perf section).
        self.medium = CoolPropMedium("Air", disable_warnings=True,
                                     backend="BICUBIC&HEOS",
                                     scalar_cache_maxsize=1000)

        self.add_component('m_dot_base', Parameter(M_DOT_BASE, "kg/s"))
        self.add_component('m_dot_amp', Parameter(M_DOT_AMP, "kg/s"))
        self.add_component('omega', Parameter(2 * np.pi * F_HZ, "rad/s"))

        self.add_component('pump', AdiabaticPump(
            self.medium,
            A_in=A_PORT, A_out=A_PORT,
            P_in=P_PERIM, P_out=P_PERIM,
            z_in=0.0, z_out=0.0,
        ))
        self.add_component('pipe', StraightPipe(
            self.medium,
            D=PIPE_D, L=PIPE_L, epsilon=PIPE_EPSILON,
            z_in=0.0, z_out=0.0, n_segments=N_SEGMENTS,
            adiabatic=True,
        ))
        self.add_component('buffer', LoopBuffer(
            self.medium,
            V=BUFFER_V, A_in=A_PORT, A_out=A_PORT,
            p_init=P_LOOP, T_init=T_LOOP,
        ))

    def declare_equations(self):
        # Closed-loop wiring (union-find short-circuits these out of the
        # symbolic Jacobian -- they never become equations).
        for io in ('p', 'h', 'w'):
            # pump  -> pipe
            self.add_connection(self['pump'][f'{io}_out'], self['pipe'][f'{io}_in'])
            # pipe  -> buffer
            self.add_connection(self['pipe'][f'{io}_out'], self['buffer'][f'{io}_in'])
            # buffer -> pump   (this is the wire that closes the loop)
            self.add_connection(self['buffer'][f'{io}_out'], self['pump'][f'{io}_in'])

        # Mass-flow operating point: prescribe m_dot, let the pump's free
        # `a_iz` (inside `f_pump = a_iz / (Re*Dh)`) adjust so that pump head
        # exactly balances pipe friction at the requested flow rate.  The
        # target oscillates sinusoidally in time so the loop has something
        # to do during the transient.
        #
        # `self.t_symbols[0]` is the framework's `t` symbol; the time stepper
        # injects the current `t` value before each Newton solve.
        t_sym = self.t_symbols[0]
        m_dot_target = (self['m_dot_base'].symbol
                        + self['m_dot_amp'].symbol * sp.sin(self['omega'].symbol * t_sym))
        rho_in = self.medium.rho_ph(self['pump']['p_in'].symbol,
                                    self['pump']['h_in'].symbol)
        m_dot_actual = rho_in * self['pump']['w_in'].symbol * self['pump']['A_in'].symbol
        eq_m_dot = m_dot_actual - m_dot_target

        return [eq_m_dot]


def _trace(record, suffix):
    names = list(record['vars_names'])
    state = np.asarray(record['state'])
    idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
    return state[:, idx]


def main():
    print("Building model...")
    system = PumpedLoop()

    print("Instantiating (symbolic Jacobian + lambdify)...")
    t0 = time.time()
    system.instantiate(
        aditional_modules=system.medium.modules,
        max_remove_trival_passes=3,
    )
    print(f"  instantiate: {time.time() - t0:.2f} s")

    # Warm-start the velocity unknowns near a physically plausible value.
    # The default w=0.1 m/s is far from a self-consistent operating point and
    # Newton can wander into negative-density territory; pinning w to the
    # ballpark consistent with `m_dot_base` keeps the first solve well-behaved.
    h_loop = float(system.medium.eval_h_pT(P_LOOP, T_LOOP))
    rho_loop = float(system.medium.eval_rho_ph(P_LOOP, h_loop))
    w0 = M_DOT_BASE / (rho_loop * A_PORT)
    print(f"Warm-starting velocities at w0 = {w0:.3f} m/s "
          f"(consistent with m_dot_base = {M_DOT_BASE * 1000:.1f} g/s, "
          f"rho_loop = {rho_loop:.3f} kg/m^3).")
    for var in system.active_vars_references:
        full = getattr(var, 'full_name', '')
        if full.endswith('.w_in') or full.endswith('.w_out'):
            var.value = w0

    print("Initialising (damped Newton at t = 0)...")
    t0 = time.time()
    system.initialise(relaxation=0.5, max_iter=400)
    print(f"  initialise:  {time.time() - t0:.2f} s")

    # --- Steady-state report at t = 0 ----------------------------------------------
    rec = system.record
    state0 = np.asarray(rec['state'])[0]
    names = list(rec['vars_names'])

    def get0(suffix):
        idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
        return state0[idx]

    p_pin = get0('.pump.p_in')
    p_pout = get0('.pump.p_out')
    h_pin = get0('.pump.h_in')
    w_pin = get0('.pump.w_in')
    a_iz0 = get0('.pump.a_iz')
    p_pipe_out = get0('.pipe.p_out')
    p_buffer = get0('.buffer.p')
    h_buffer = get0('.buffer.h')
    m_buffer = get0('.buffer.m')
    U_buffer = get0('.buffer.U')

    T_pin = float(system.medium.eval_T_ph(p_pin, h_pin))
    rho_pin = float(system.medium.eval_rho_ph(p_pin, h_pin))
    m_dot0 = rho_pin * w_pin * A_PORT

    # Loop pressure closure: pump head must exactly balance pipe friction loss
    # (the buffer is well-mixed with no throttling, so it neither adds nor
    # removes pressure).  Buffer.p == pump.p_in == pipe.p_out (all unioned).
    pump_rise = p_pout - p_pin
    pipe_drop = p_pout - p_pipe_out
    head_residual = pump_rise - pipe_drop

    print()
    print("=== Initial steady-state (t = 0, m_dot_target = m_dot_base) ===")
    print(f"Buffer state    : p = {p_buffer / 1e5:.4f} bar, h = {h_buffer:.2f} J/kg, "
          f"m = {m_buffer * 1000:.3f} g, U = {U_buffer:.2f} J")
    print(f"  (anchor target: p = {P_LOOP / 1e5:.4f} bar, T = {T_LOOP:.2f} K, "
          f"V = {BUFFER_V * 1000:.1f} L)")
    print(f"Pump inlet p    : {p_pin / 1e5:9.4f} bar      "
          f"(== buffer.p via wiring, residual {p_pin - p_buffer:.2e} Pa)")
    print(f"Pump inlet T    : {T_pin:9.2f} K        (loop temperature)")
    print(f"Pump rise       : dp = {pump_rise / 1e3:7.3f} kPa")
    print(f"Pipe drop       : dp = {pipe_drop / 1e3:7.3f} kPa "
          f"(must equal pump rise; residual {head_residual:.2e} Pa)")
    print(f"Pump strength   : a_iz = {a_iz0:.4e}")
    print(f"Mass flow       : {m_dot0 * 1000:7.3f} g/s     (target {M_DOT_BASE * 1000:.3f} g/s)")
    print(f"Loop velocity   : w = {w_pin:.3f} m/s     (rho_in = {rho_pin:.3f} kg/m^3)")

    assert abs(m_dot0 - M_DOT_BASE) < 1e-6, "m_dot anchor should hold to Newton tolerance"
    assert abs(head_residual) < 1.0, "Pump head must balance pipe friction loss in steady state"
    assert abs(p_buffer - P_LOOP) < 1.0, "Buffer pressure should sit at the anchored P_LOOP"

    # --- Transient: ride the m_dot_target sinusoid ---------------------------------
    print()
    print(f"Running {int(T_END / DT)} steps of dt = {DT:g} s "
          f"({T_END:g} s total, m_dot_target sinusoid at {F_HZ} Hz)...")
    t0 = time.time()
    while system.get_t_value() < T_END - 1e-12:
        dt = min(DT, T_END - system.get_t_value())
        system.solve_dae_step(dt)
        system.next_step()
    print(f"  solve loop:  {time.time() - t0:.2f} s")

    # --- Post-process: show the loop tracking the drive ----------------------------
    rec = system.record
    t = np.asarray(rec['time'])
    w_pin_t = _trace(rec, '.pump.w_in')
    a_iz_t = _trace(rec, '.pump.a_iz')
    p_pin_t = _trace(rec, '.pump.p_in')
    p_pout_t = _trace(rec, '.pump.p_out')
    h_pin_t = _trace(rec, '.pump.h_in')
    m_buffer_t = _trace(rec, '.buffer.m')
    U_buffer_t = _trace(rec, '.buffer.U')

    rho_t = np.array([float(system.medium.eval_rho_ph(p, h))
                      for p, h in zip(p_pin_t, h_pin_t)])
    m_dot_t = rho_t * w_pin_t * A_PORT
    m_dot_target_t = M_DOT_BASE + M_DOT_AMP * np.sin(2 * np.pi * F_HZ * t)
    dp_pump_t = p_pout_t - p_pin_t

    tracking_err = np.max(np.abs(m_dot_t - m_dot_target_t))
    p_drift = np.max(np.abs(p_pin_t - P_LOOP))
    m_drift = np.max(np.abs(m_buffer_t - m_buffer_t[0]))
    U_drift = np.max(np.abs(U_buffer_t - U_buffer_t[0]))

    print()
    print("=== Transient summary ===")
    print(f"|m_dot - m_dot_target|_max  : {tracking_err:.3e} kg/s   (loop tracks the drive)")
    print(f"|pump.p_in - P_LOOP|_max    : {p_drift:.3e} Pa     (loop pressure level holds)")
    print(f"|buffer.m - m_init|_max     : {m_drift:.3e} kg     (closed-loop mass conservation)")
    print(f"|buffer.U - U_init|_max     : {U_drift:.3e} J      (closed-loop energy conservation)")
    print(f"Pump head dp range          : {dp_pump_t.min() / 1e3:.3f} .. {dp_pump_t.max() / 1e3:.3f} kPa")
    print(f"Pump strength a_iz range    : {a_iz_t.min():.3e} .. {a_iz_t.max():.3e}")

    print()
    print(f"Sample trajectory (every {max(1, len(t) // 10)} steps):")
    print(f"{'t [s]':>6}  {'m_dot [g/s]':>11}  {'target [g/s]':>12}  "
          f"{'a_iz':>10}  {'dp_pump [kPa]':>13}")
    for i in range(0, len(t), max(1, len(t) // 10)):
        print(f"{t[i]:6.2f}  {m_dot_t[i] * 1000:11.4f}  {m_dot_target_t[i] * 1000:12.4f}  "
              f"{a_iz_t[i]:10.3e}  {dp_pump_t[i] / 1e3:13.4f}")

    out_path = plot_results(rec, "loop_pump_pipe.html",
                            show=False, subdir="examples")
    print(f"\nPlot written to {out_path}")


if __name__ == "__main__":
    main()
