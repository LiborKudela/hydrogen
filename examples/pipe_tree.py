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

Tree topology knobs (passed to `TreeSystem.__init__`):

    medium    - a `CoolPropMedium` instance (Air, Hydrogen, Water, ...)
    N         - depth: number of `Splitter` levels between source and leaves
    K         - branching: number of children per `Splitter`
    M         - segments per `StraightPipe`
    L         - common pipe length [m]
    D         - pipe diameter [m]
    epsilon   - pipe roughness [m]
    P_source, T_source, P_outlet, T_outlet - boundary state

Counts (closed-form):
    pipes      = (K^(N+1) - 1) / (K - 1)
    splitters  = (K^N - 1) / (K - 1)
    outlets    = K^N

Multiple `TreeSystem` instances can co-exist in one Python process -- each gets
its own `CoolPropMedium` (so the symbolic property functions are *unique
sympy.Function classes* per instance, see `medium.py`) and its own lambdified
residual + Jacobian. `main()` below builds and solves an Air tree and a
Hydrogen tree side by side to demonstrate.
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
        A_pipe = np.pi * self.D ** 2 / 4
        self.add_component('pipe', StraightPipe(
            self.medium, self.D, self.L, self.epsilon,
            z_in=0.0, z_out=0.0, n_segments=self.M, adiabatic=True,
        ))
        if self.depth_remaining > 0:
            self.add_component('splitter', Splitter(
                self.medium, self.K, A_in=A_pipe, A_out=A_pipe,
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
        # All inter-component wiring here is variable-equality -- route via the
        # union-find `add_connection` API so the trivial-equation reducer never
        # sees these as sympy expressions.
        if self.depth_remaining > 0:
            for io in ('p', 'h', 'w'):
                self.add_connection(self['pipe'][f'{io}_out'], self['splitter'][f'{io}_in'])
            for k in range(self.K):
                for io in ('p', 'h', 'w'):
                    self.add_connection(
                        self['splitter'][f'{io}_out_{k}'],
                        self[f'child_{k}']['pipe'][f'{io}_in'],
                    )
        else:
            for io in ('p', 'h', 'w'):
                self.add_connection(self['pipe'][f'{io}_out'], self['outlet'][f'{io}_in'])
        return []


class TreeSystem(Model):
    """`PressureSource -> BranchNode(depth=N)` rooted at the source.

    All inputs (medium, geometry, boundary state) are constructor arguments, so
    multiple `TreeSystem`s with different fluids or sizes can co-exist in the same
    Python process and be `instantiate()`-d / `initialise()`-d independently.
    """

    def __init__(
        self,
        medium: CoolPropMedium,
        *,
        N: int = 2,
        K: int = 2,
        M: int = 2,
        L: float = 0.5,
        D: float = 0.005,
        epsilon: float = 1e-6,
        P_source: float = 1.2e5,
        T_source: float = 293.15,
        P_outlet: float = 1.013e5,
        T_outlet: float = 293.15,
    ):
        self.medium = medium
        self.N = N
        self.K = K
        self.M = M
        self.L = L
        self.D = D
        self.epsilon = epsilon
        self.P_source = P_source
        self.T_source = T_source
        self.P_outlet = P_outlet
        self.T_outlet = T_outlet
        super().__init__()

    @property
    def A_pipe(self):
        return np.pi * self.D ** 2 / 4

    def topology(self):
        if self.K == 1:
            n_pipes, n_splitters = self.N + 1, self.N
        else:
            n_pipes = (self.K ** (self.N + 1) - 1) // (self.K - 1)
            n_splitters = (self.K ** self.N - 1) // (self.K - 1)
        n_outlets = self.K ** self.N
        return n_pipes, n_splitters, n_outlets

    def declare_components(self):
        self.add_component('source', PressureSource(self.medium, self.P_source, self.T_source))
        self.add_component('tree', BranchNode(
            self.medium, self.N, self.K, self.M, self.L, self.D, self.epsilon,
            self.P_outlet, self.T_outlet,
        ))

    def declare_equations(self):
        for io in ('p', 'h', 'w'):
            self.add_connection(self['source'][f'{io}_out'], self['tree']['pipe'][f'{io}_in'])
        return []


def _bernoulli_warm_start(system: TreeSystem, fraction: float = 0.4) -> float:
    """Heuristic warm-start velocity that scales with the fluid -- the Newton solve
    needs `w` non-zero so the energy-equation row (`h_total = h + w**2/2`) is well
    conditioned, and the right magnitude depends on the medium's density at the
    source state. Lower density (e.g. hydrogen) -> much higher velocity for the
    same pressure drop. We pick a conservative fraction of the inviscid Bernoulli
    estimate so over-shoot doesn't push CoolProp into negative enthalpies on the
    first Newton step."""
    h_total = float(system.medium.eval_h_pT(system.P_source, system.T_source))
    rho = float(system.medium.eval_rho_ph(system.P_source, h_total))
    delta_p = max(system.P_source - system.P_outlet, 1.0)
    return float(fraction * np.sqrt(2.0 * delta_p / rho))


def run_tree(label: str, system: TreeSystem, *, warm_w: float | None = None,
             output_html: str | None = None) -> None:
    """Instantiate, initialise, time-step, and report on `system`. The label is
    used as a prefix for the printed report and the default plot filename."""

    n_pipes, n_splitters, n_outlets = system.topology()
    print(f"=========================  {label}  =========================")
    print(f"medium = {system.medium.medium}")
    print(f"Tree topology: N={system.N}, K={system.K}, M={system.M}, "
          f"L={system.L} m, D={system.D * 1000:.1f} mm")
    print(f"  -> {n_pipes} pipes, {n_splitters} splitters, {n_outlets} leaf outlets")
    print(f"  -> source = {system.P_source / 1e5:.3f} bar @ {system.T_source:.2f} K, "
          f"leaf outlets = {system.P_outlet / 1e5:.3f} bar")
    print()

    print("Instantiating (symbolic Jacobian + lambdify can take a while)...")
    t0 = time.time()
    system.instantiate(
        aditional_modules=system.medium.modules,
        max_remove_trival_passes=4,
    )
    print(f"  instantiate: {time.time() - t0:.2f} s")

    if warm_w is None:
        warm_w = _bernoulli_warm_start(system)
    print(f"Warm-starting velocity unknowns to {warm_w:.2f} m/s")
    for var in system.active_vars_references:
        full = getattr(var, 'full_name', '')
        if full.endswith('.w_in') or full.endswith('.w_out') or '.w_out_' in full:
            var.value = warm_w

    print("Initialising (damped Newton)...")
    t0 = time.time()
    system.initialise(relaxation=0.5, max_iter=400)
    print(f"  initialise:  {time.time() - t0:.2f} s")

    # Time-step with the default adaptive controller (`predictor_corrector`).
    # The system is initialised AT steady state, so PC's predictor matches the
    # CN result almost exactly -- the controller grows `dt` toward `DT_MAX`
    # within the first couple of steps and then breezes through the rest of
    # the window in a handful of cheap steps. A fixed `dt = 0.05 s` would
    # always take 5 steps regardless of how easy the problem is.
    #
    # Note the loop caps `dt_try` at `DT_MAX` (not `DT_TARGET`) and feeds the
    # controller's hint back in -- otherwise the hint could only ever shrink.
    T_END = 0.25
    DT_TARGET = 0.05
    DT_MAX = 4 * DT_TARGET
    print(f"Time-stepping adaptively until t = {T_END:g} s "
          f"(initial dt_target = {DT_TARGET:g} s, can grow to {DT_MAX:g} s)...")
    t0 = time.time()
    dt_history: list[float] = []
    n_rejections = 0
    while system.get_t_value() < T_END - 1e-12:
        dt_try = min(DT_MAX, T_END - system.get_t_value())
        if hasattr(system, "_dt_hint"):
            dt_try = min(dt_try, system._dt_hint)
        else:
            dt_try = min(dt_try, DT_TARGET)              # first step
        dt_used, info = system.solve_adaptive_step(
            dt_try, dt_max=DT_MAX,
            relaxation=0.5, max_iter=200,
        )
        dt_history.append(dt_used)
        n_rejections += info["rejections"]
        system.next_step()
    print(f"  solve loop:  {time.time() - t0:.2f} s "
          f"({len(dt_history)} accepted steps, {n_rejections} rejected, "
          f"dt range {min(dt_history):.4f} .. {max(dt_history):.4f} s)")

    # --- post-process -----------------------------------------------------------------
    record = system.record
    state = np.asarray(record['state'])
    names = list(record['vars_names'])

    def trace_last(suffix):
        idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
        return state[-1, idx]

    # Group every pipe by depth (number of `.child_*.` segments in the dotted path).
    pipe_outlet_names = sorted(n for n in names if n.endswith('.pipe.w_out'))
    pipes_by_depth: dict[int, list[tuple[str, float, float, float]]] = {}
    for n in pipe_outlet_names:
        depth = n.count('.child_')
        prefix = n[: -len('w_out')]
        w_val = state[-1, names.index(n)]
        p_val = state[-1, names.index(prefix + 'p_out')]
        h_val = state[-1, names.index(prefix + 'h_out')]
        pipes_by_depth.setdefault(depth, []).append((n, w_val, p_val, h_val))

    print()
    print(f"=== Steady-state pipe-outlet conditions ({label}) ===")
    for depth in sorted(pipes_by_depth):
        rows = pipes_by_depth[depth]
        ws = [w for _, w, _, _ in rows]
        ps = [p for _, _, p, _ in rows]
        print(
            f"  depth {depth} ({len(rows):2d} pipe{'s' if len(rows) != 1 else ''}):"
            f"  w_out min={min(ws):9.4f} max={max(ws):9.4f} m/s,"
            f"  p_out min={min(ps) / 1e5:6.4f} max={max(ps) / 1e5:6.4f} bar,"
            f"  spread(w)={(max(ws) - min(ws)):.2e}"
        )

    rho_source = float(system.medium.eval_rho_ph(
        float(trace_last('.source.p_out')),
        float(trace_last('.source.h_out')),
    ))
    m_dot_source = trace_last('.source.w_out') * system.A_pipe * rho_source
    print()
    print(f"Mass conservation ({label}):")
    print(f"  source m_dot                        = {m_dot_source * 1000:9.4f} g/s")
    for depth in sorted(pipes_by_depth):
        rows = pipes_by_depth[depth]
        m_dot_d = 0.0
        for _, w, p, h in rows:
            rho = float(system.medium.eval_rho_ph(float(p), float(h)))
            m_dot_d += rho * float(w) * system.A_pipe
        rel_err = abs(m_dot_d - m_dot_source) / max(abs(m_dot_source), 1e-30)
        print(
            f"  depth {depth} ({system.K ** depth:2d} pipes) m_dot   = {m_dot_d * 1000:9.4f} g/s"
            f"  (rel err vs source: {rel_err:.2e})"
        )

    if output_html is None:
        # Sanitise label into a usable filename.
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label.lower())
        output_html = f"pipe_tree_{safe}.html"
    plot_results(system.record, output_html, show=False)
    print()
    print(f"Plot written to {output_html}")
    print()


def main():
    # Two trees with different media -- same geometry, same boundary pressures, but
    # very different densities (hydrogen ~14x less dense than air at the same state),
    # which is what drives the large difference in steady-state velocities.
    N = 2
    K = 2
    M = 2
    L = 0.5
    D = 0.005
    air_tree = TreeSystem(
        CoolPropMedium("Air", disable_warnings=True),
        N=N, K=K, M=M, L=L, D=D,
    )
    hydrogen_tree = TreeSystem(
        CoolPropMedium("Hydrogen", disable_warnings=True),
        N=N, K=K, M=M, L=L, D=D,
    )

    carbon_dioxide_tree = TreeSystem(
        CoolPropMedium("CarbonDioxide", disable_warnings=True),
        N=N, K=K, M=M, L=L, D=D,
    )

    run_tree("Air", air_tree)
    run_tree("Hydrogen", hydrogen_tree)
    run_tree("CarbonDioxide", carbon_dioxide_tree)


if __name__ == "__main__":
    main()
