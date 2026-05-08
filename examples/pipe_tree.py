"""K-ary tree of pipes with `Splitter`s at every node.

Build a balanced tree:

    PressureSource
        |
        v
    [root pipe]                                          <- depth 0
        |
        v
     Splitter --> [pipe] -> Splitter --> [pipe] -> ...   <- depth 1, 2, ...
                |                      |
                v                      v
              ...                    ...
                |                      |
                v                      v
       [leaf pipe] -> PressureOutlet  [leaf pipe] -> PressureOutlet   <- depth N

Tree topology knobs (top of file):

    N - depth: number of `Splitter` levels between source and leaves
    K - branching: number of children per `Splitter`
    M - segments per `StraightPipe`
    L - common pipe length [m]

Counts (closed-form):
    pipes      = (K^(N+1) - 1) / (K - 1)
    splitters  = (K^N - 1) / (K - 1)
    outlets    = K^N

In the symmetric case (every pipe geometrically identical, every leaf at the same
ambient pressure) the steady state has all pipes at the same depth carrying the
same velocity, with the velocity dropping by a factor `K` at each splitter (since
each branch has the same area as the inlet, total outlet area = K * A_in, so
mass conservation forces `w_out = w_in / K`). This script prints the velocity at
every pipe inlet so you can see the splits explicitly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

from hydrogen import (  # noqa: E402
    CoolPropMedium,
    Model,
    PressureOutlet,
    PressureSource,
    Splitter,
    StraightPipe,
    plot_results,
)

# --- tree geometry --------------------------------------------------------------------
N = 2          # depth (number of splitter levels)
K = 2          # branches per splitter
M = 2          # segments per pipe
L = 0.5        # m, common pipe length
D = 0.005      # m, common pipe diameter
EPSILON = 1e-6
A_PIPE = np.pi * D ** 2 / 4

# --- boundary conditions --------------------------------------------------------------
P_SOURCE = 1.2e5      # Pa  (1.2 bar -> ~20 kPa drop across the whole tree)
T_SOURCE = 293.15     # K
P_OUTLET = 1.013e5    # Pa  (atmospheric)
T_OUTLET = 293.15     # K


class BranchNode(Model):
    """One pipe followed by either:
      * a `K`-way splitter and `K` recursively-nested child `BranchNode`s
        (when `depth_remaining > 0`), or
      * a `PressureOutlet` (when `depth_remaining == 0`, i.e. this is a leaf).

    The pipe inlet ports `pipe.p_in/h_in/w_in` are exposed for the parent to wire to
    the upstream component (a `Splitter` outlet, or the root `PressureSource`).
    """

    def __init__(self, medium, depth_remaining, K, M, L, D, epsilon, p_outlet, T_outlet):
        self.medium = medium
        self.depth_remaining = depth_remaining
        self.K = K
        self.M = M
        self.L = L
        self.D = D
        self.epsilon = epsilon
        self.p_outlet = p_outlet
        self.T_outlet = T_outlet
        super().__init__()

    def declare_components(self):
        self.add_component('pipe', StraightPipe(
            self.medium, self.D, self.L, self.epsilon,
            z_in=0.0, z_out=0.0, n_segments=self.M, adiabatic=True,
        ))
        if self.depth_remaining > 0:
            self.add_component('splitter', Splitter(
                self.medium, self.K, A_in=A_PIPE, A_out=A_PIPE,
            ))
            for k in range(self.K):
                self.add_component(f'child_{k}', BranchNode(
                    self.medium, self.depth_remaining - 1,
                    self.K, self.M, self.L, self.D, self.epsilon,
                    self.p_outlet, self.T_outlet,
                ))
        else:
            self.add_component('outlet', PressureOutlet(self.medium, self.p_outlet, self.T_outlet))

    def declare_equations(self):
        eqs = []
        if self.depth_remaining > 0:
            # pipe -> splitter
            eqs.append(self['pipe']['p_out'].symbol - self['splitter']['p_in'].symbol)
            eqs.append(self['pipe']['h_out'].symbol - self['splitter']['h_in'].symbol)
            eqs.append(self['pipe']['w_out'].symbol - self['splitter']['w_in'].symbol)
            # splitter -> K children
            for k in range(self.K):
                eqs.append(self['splitter'][f'p_out_{k}'].symbol - self[f'child_{k}']['pipe']['p_in'].symbol)
                eqs.append(self['splitter'][f'h_out_{k}'].symbol - self[f'child_{k}']['pipe']['h_in'].symbol)
                eqs.append(self['splitter'][f'w_out_{k}'].symbol - self[f'child_{k}']['pipe']['w_in'].symbol)
        else:
            # leaf pipe -> outlet
            eqs.append(self['pipe']['p_out'].symbol - self['outlet']['p_in'].symbol)
            eqs.append(self['pipe']['h_out'].symbol - self['outlet']['h_in'].symbol)
            eqs.append(self['pipe']['w_out'].symbol - self['outlet']['w_in'].symbol)
        return eqs


class TreeSystem(Model):
    """`PressureSource -> BranchNode(depth=N)` rooted at the source."""

    def declare_components(self):
        self.medium = CoolPropMedium("Air", disable_warnings=True)
        self.add_component('source', PressureSource(self.medium, P_SOURCE, T_SOURCE))
        self.add_component('tree', BranchNode(
            self.medium, N, K, M, L, D, EPSILON, P_OUTLET, T_OUTLET,
        ))

    def declare_equations(self):
        return [
            self['source']['p_out'].symbol - self['tree']['pipe']['p_in'].symbol,
            self['source']['h_out'].symbol - self['tree']['pipe']['h_in'].symbol,
            self['source']['w_out'].symbol - self['tree']['pipe']['w_in'].symbol,
        ]


def _topology_summary():
    if K == 1:
        n_pipes, n_splitters = N + 1, N
    else:
        n_pipes = (K ** (N + 1) - 1) // (K - 1)
        n_splitters = (K ** N - 1) // (K - 1)
    n_outlets = K ** N
    return n_pipes, n_splitters, n_outlets


def main():
    n_pipes, n_splitters, n_outlets = _topology_summary()
    print(f"Tree topology: N={N}, K={K}, M={M}, L={L} m, D={D * 1000:.1f} mm")
    print(f"  -> {n_pipes} pipes, {n_splitters} splitters, {n_outlets} leaf outlets")
    print(f"  -> source = {P_SOURCE / 1e5:.3f} bar, leaf outlets = {P_OUTLET / 1e5:.3f} bar")
    print()

    print("Building model...")
    system = TreeSystem()

    print("Instantiating (symbolic Jacobian + lambdify can take a while)...")
    t0 = time.time()
    system.instantiate(
        aditional_modules=system.medium.modules,
        max_remove_trival_passes=4,
    )
    print(f"  instantiate: {time.time() - t0:.2f} s")

    # Warm-start velocities to keep Newton well-conditioned at t = 0. The default
    # initial guess is `w ~ 0.1 m/s`, but with a meaningful pressure differential
    # the energy equation `h_total = h_out + w**2/2` has a near-singular row at
    # small `w` (its w-derivative is `-w`), so Newton overshoots into negative
    # enthalpies and crashes CoolProp. A moderate uniform warm start works.
    WARM_W = 15.0
    for var in system.active_vars_references:
        full = getattr(var, 'full_name', '')
        if (
            full.endswith('.w_in')
            or full.endswith('.w_out')
            or '.w_out_' in full
        ):
            var.value = WARM_W

    print("Initialising (damped Newton)...")
    t0 = time.time()
    system.initialise(relaxation=0.5, max_iter=400)
    print(f"  initialise:  {time.time() - t0:.2f} s")

    # A handful of time steps verifies the solution actually is steady; with
    # adiabatic, no-storage components, no transient should remain after t = 0.
    print("Time-stepping (5 steps of 0.05 s) to verify steady state...")
    t0 = time.time()
    for _ in range(5):
        system.solve_dae_step(0.05)
        system.next_step()
    print(f"  solve loop:  {time.time() - t0:.2f} s")

    # --- post-process -----------------------------------------------------------------
    record = system.record
    state = np.asarray(record['state'])
    names = list(record['vars_names'])

    def trace_last(suffix):
        idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
        return state[-1, idx]

    # Group every pipe by depth (number of `.child_*.` segments in the dotted path).
    pipe_inlet_w_names = sorted(n for n in names if n.endswith('.pipe.w_in'))
    pipes_by_depth: dict[int, list[tuple[str, float]]] = {}
    for n in pipe_inlet_w_names:
        depth = n.count('.child_')
        w_val = state[-1, names.index(n)]
        pipes_by_depth.setdefault(depth, []).append((n, w_val))

    print()
    print("=== Steady-state velocities at every pipe inlet ===")
    for depth in sorted(pipes_by_depth):
        ws = [w for _, w in pipes_by_depth[depth]]
        print(f"  depth {depth} ({len(ws):2d} pipe{'s' if len(ws) != 1 else ''}): "
              f"w_in min={min(ws):.4f}, max={max(ws):.4f} m/s, "
              f"spread={(max(ws) - min(ws)):.2e}")
    print("(spread should be ~0 for a balanced symmetric tree -- rounding only.)")

    # Mass-conservation check at every level: sum of mass flows on a level = source mass flow.
    rho_source = float(system.medium.eval_rho_ph(
        float(trace_last('.source.p_out')),
        float(trace_last('.source.h_out')),
    ))
    m_dot_source = trace_last('.source.w_out') * A_PIPE * rho_source
    print()
    print(f"Source mass flow: {m_dot_source * 1000:.4f} g/s")
    print()
    print("Mass conservation across each level (sum_branches w * A * rho):")
    for depth in sorted(pipes_by_depth):
        n_pipes_d = K ** depth
        # Velocity is uniform per level by symmetry, so use the mean.
        w_mean = float(np.mean([w for _, w in pipes_by_depth[depth]]))
        # Density also depends on local p, h; for symmetry we sample one pipe at this depth.
        sample_name = pipes_by_depth[depth][0][0]
        prefix = sample_name[: -len('w_in')]
        p_sample = float(trace_last(prefix + 'p_in'))
        h_sample = float(trace_last(prefix + 'h_in'))
        rho_sample = float(system.medium.eval_rho_ph(p_sample, h_sample))
        m_dot_d = n_pipes_d * w_mean * A_PIPE * rho_sample
        print(f"  depth {depth}: {n_pipes_d:3d} x w*A*rho = {m_dot_d * 1000:.4f} g/s  "
              f"(deficit vs source: {(m_dot_d - m_dot_source) * 1000:+.2e} g/s)")

    plot_results(system.record, "pipe_tree.html", show=False)
    print()
    print("Plot written to pipe_tree.html")


if __name__ == "__main__":
    main()
