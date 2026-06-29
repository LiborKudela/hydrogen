"""Signal-controlled valves: the control (blocks) domain driving fluid valves.

Two parts, both self-validating (they ``assert`` their invariants, so this
script doubles as an end-to-end test under ``pytest -m tutorials``):

  Part 1 - A liquid `IncompressibleValve` whose opening is commanded by a
      signal chain ``Ramp -> Limiter -> valve.opening``.  The ramp drives the
      command past 1.0; the `Limiter` clamps it to [0, 1]; the valve flow
      follows the metric-Kv law and rises, then plateaus once the valve is
      fully open.  Layout:

          Constant/Ramp -> Limiter ----(signal)----> opening
                                                        |
          PressureSource =====> [ IncompressibleValve ] =====> PressureOutlet

  Part 2 - A gas `CompressibleValve` at fixed full opening, swept over
      decreasing downstream pressure, showing IEC 60534 choking: the mass
      flow saturates once the pressure-drop ratio exceeds ``xT`` and stops
      depending on the downstream pressure.

Run with ``python tutorials/control_valve.py`` from the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

from hydrogen import CoolPropMedium, Model, plot_results  # noqa: E402
from hydrogen.components.control.control_components import (  # noqa: E402
    Constant,
    Limiter,
    Ramp,
)
from hydrogen.components.thermofluid.flow import (  # noqa: E402
    CompressibleValve,
    IncompressibleValve,
    PressureOutlet,
    PressureSource,
)

WATER = CoolPropMedium('water', disable_warnings=True)
AIR = CoolPropMedium('air', disable_warnings=True)


# ---------------------------------------------------------------------------
# Part 1 - liquid valve with a ramped, limited opening command
# ---------------------------------------------------------------------------

class ControlledLine(Model):
    Kv = 16.0
    D = 0.025
    P_IN = 3.0e5
    P_OUT = 1.0e5
    T = 320.0

    def declare_components(self):
        self.add_component('src', PressureSource(WATER, p_source=self.P_IN, T_source=self.T, A=1e-2))
        self.add_component('v', IncompressibleValve(WATER, Kv=self.Kv, D=self.D))
        self.add_component('out', PressureOutlet(WATER, p_ambient=self.P_OUT, T_ambient=self.T))
        # Command chain: ramp the opening from 0 up past 1, clamp to [0, 1].
        self.add_component('cmd', Ramp(height=1.3, duration=8.0, start_time=1.0))
        self.add_component('clamp', Limiter(lo=0.0, hi=1.0, eps=1e-3))

    def declare_equations(self):
        self.connect(self['src'].ports['outlet'], self['v'].ports['inlet'])
        self.connect(self['v'].ports['outlet'], self['out'].ports['inlet'])
        self.connect(self['cmd'].ports['y'], self['clamp'].ports['u'])
        self.connect(self['clamp'].ports['y'], self['v'].ports['opening'])
        return []


def part1_controlled_liquid_valve():
    print("=" * 70)
    print("Part 1 - liquid valve driven by Ramp -> Limiter -> opening")
    print("=" * 70)

    m = ControlledLine()
    m.instantiate(aditional_modules=WATER.modules, max_remove_trival_passes=5)
    m.initialise(n=1)

    DT, N = 0.5, 24
    for _ in range(N):
        m.solve_dae_step(DT)
        m.next_step()

    rec = m.record
    t = np.asarray(rec['time'])
    names = list(rec['vars_names'])
    state = np.asarray(rec['state'])

    def trace(suffix):
        return state[:, next(i for i, n in enumerate(names) if n.endswith(suffix))]

    opening = trace('.v.opening')
    m_dot = trace('.v.m_dot_in')
    # The valve is a single-cell SegmentedChannel: face 0 is the inlet, face 1
    # the outlet (p_0/p_1, rho_0/rho_1).
    p_in = trace('.v.p_0')
    p_out = trace('.v.p_1')
    rho = 0.5 * (trace('.v.rho_0') + trace('.v.rho_1'))

    print(f"\n{'t [s]':>6}  {'cmd opening':>11}  {'m_dot [kg/s]':>12}  {'dp [bar]':>9}")
    for i in range(0, len(t), max(1, N // 8)):
        print(f"{t[i]:6.1f}  {opening[i]:11.3f}  {m_dot[i]:12.4f}  {(p_in[i]-p_out[i])/1e5:9.3f}")

    # Validate the Kv law holds at the final (fully-open) step.
    dp = p_in[-1] - p_out[-1]
    g = dp / (dp ** 2 + 1.0 ** 2) ** 0.25
    expected = (ControlledLine.Kv / 36000.0) * opening[-1] * np.sqrt(rho[-1]) * g
    print(f"\nFinal: opening clamped to {opening[-1]:.4f} (command exceeded 1.0)")
    print(f"       m_dot = {m_dot[-1]:.4f} kg/s, Kv-law predicts {expected:.4f} kg/s")

    assert opening[-1] == _approx(1.0, 1e-3), "Limiter must clamp opening to 1.0"
    assert m_dot[-1] > m_dot[2], "flow must rise as the valve opens"
    assert m_dot[-1] == _approx(expected, rel=1e-6), "Kv law violated"

    out_path = plot_results(rec, "control_valve.html", show=False, subdir="tutorials")
    print(f"Plot written to {out_path}")


# ---------------------------------------------------------------------------
# Part 2 - compressible valve choking
# ---------------------------------------------------------------------------

class GasValve(Model):
    def __init__(self, p_out):
        self._p_out = p_out
        super().__init__()

    def declare_components(self):
        self.add_component('src', PressureSource(AIR, p_source=8e5, T_source=300.0, A=1e-2))
        self.add_component('v', CompressibleValve(AIR, Kv=5.0, D=0.02, xT=0.7, gamma=1.4, p_eps=200.0))
        self.add_component('out', PressureOutlet(AIR, p_ambient=self._p_out, T_ambient=300.0))
        self.add_component('cmd', Constant(k=1.0))

    def declare_equations(self):
        self.connect(self['src'].ports['outlet'], self['v'].ports['inlet'])
        self.connect(self['v'].ports['outlet'], self['out'].ports['inlet'])
        self.connect(self['cmd'].ports['y'], self['v'].ports['opening'])
        return []


def part2_compressible_choke():
    print()
    print("=" * 70)
    print("Part 2 - compressible valve choking (IEC 60534 expansion factor)")
    print("=" * 70)
    print(f"\n{'p_out [bar]':>11}  {'x = dp/p1':>10}  {'m_dot [kg/s]':>12}")

    flows = []
    for p_out in (7.5e5, 6e5, 5e5, 4e5, 3e5, 2e5, 1.2e5):
        g = GasValve(p_out)
        g.instantiate(aditional_modules=AIR.modules, max_remove_trival_passes=5)
        g.initialise(n=1)
        g.solve_dae_step(1.0)
        names = list(g.record['vars_names'])
        last = np.asarray(g.record['state'])[-1]

        def val(suffix):
            return last[next(i for i, n in enumerate(names) if n.endswith(suffix))]

        p_in = val('.v.p_0')          # single-cell valve: face 0 = inlet
        m_dot = val('.v.m_dot_in')
        flows.append(m_dot)
        print(f"{p_out/1e5:11.2f}  {(p_in-p_out)/p_in:10.3f}  {m_dot:12.4f}")

    # Monotone before choke, saturated after.
    assert flows[0] < flows[3], "flow must grow with dp below the choke point"
    assert flows[-1] == _approx(flows[-2], rel=1e-2), "flow must saturate when choked"
    print("\nFlow saturates past the critical pressure ratio (choked).")


# small local helpers so the script has no test-framework dependency
def _approx(target, rel=None, abs_=None):
    class _A:
        def __eq__(self, other):
            if rel is not None:
                return abs(other - target) <= rel * max(1.0, abs(target))
            return abs(other - target) <= (abs_ if abs_ is not None else 1e-9)
    return _A()


def main():
    part1_controlled_liquid_valve()
    part2_compressible_choke()
    print("\nAll control-valve demos passed.")


if __name__ == "__main__":
    main()
