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

    The pipe inlet ports `pipe.p_in/h_in/m_dot_in` are exposed for the parent to wire to
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
            # With `m_dot` on ports the splitter is geometry-free -- it just
            # asserts `m_dot_in == sum_k m_dot_out_k`.  Areas, if any, live on
            # the adjacent pipes / outlets.
            self.add_component('splitter', Splitter(self.medium, self.K))
            for k in range(self.K):
                self.add_component(f'child_{k}', BranchNode(
                    self.medium, self.depth_remaining - 1,
                    self.K, self.M, self.L, self.D, self.epsilon,
                    self.p_outlet, self.T_outlet,
                ))
        else:
            self.add_component('outlet', PressureOutlet(self.medium, self.p_outlet, self.T_outlet))

    def declare_equations(self):
        # All inter-component wiring goes through the typed-port `connect()`
        # API.  Each call emits one signed `add_connection` per port channel
        # and rides the same union-find short-circuit as before, so the
        # trivial-equation reducer never sees these wires as sympy
        # residuals.  The m_dot union still enforces mass conservation
        # regardless of any port-area mismatch across the joint.
        if self.depth_remaining > 0:
            self.connect(self['pipe'].ports['outlet'],
                         self['splitter'].ports['inlet'])
            for k in range(self.K):
                self.connect(
                    self['splitter'].ports[f'outlet_{k}'],
                    self[f'child_{k}']['pipe'].ports['inlet'],
                )
        else:
            self.connect(self['pipe'].ports['outlet'],
                         self['outlet'].ports['inlet'])
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
        # PressureSource needs an outflow area to compute the kinetic-energy
        # correction in its isentropic balance.  Pass the pipe area so the
        # stagnation-to-static state at the boundary plane is consistent with
        # what the downstream pipe sees.
        self.add_component('source', PressureSource(
            self.medium, self.P_source, self.T_source, A=self.A_pipe,
        ))
        self.add_component('tree', BranchNode(
            self.medium, self.N, self.K, self.M, self.L, self.D, self.epsilon,
            self.P_outlet, self.T_outlet,
        ))

    def declare_equations(self):
        self.connect(self['source'].ports['outlet'],
                     self['tree']['pipe'].ports['inlet'])
        return []


def _bernoulli_warm_start(system: TreeSystem, fraction: float = 0.4) -> float:
    """Heuristic warm-start mass flow that scales with the fluid -- the Newton
    solve needs `m_dot` non-zero so the energy-equation row at the source
    (`h_total = h + (m_dot / (rho * A))**2 / 2`) is well conditioned, and the
    right magnitude depends on the medium's density at the source state.
    Lower density (e.g. hydrogen) -> much higher velocity (and hence m_dot)
    for the same pressure drop.  We pick a conservative fraction of the
    inviscid Bernoulli estimate so over-shoot doesn't push CoolProp into
    negative enthalpies on the first Newton step.

    Returns `m_dot` in kg/s; total per-tree mass flow (a leaf pipe carries
    `m_dot / K**N` of it).
    """
    h_total = float(system.medium.eval_h_pT(system.P_source, system.T_source))
    rho = float(system.medium.eval_rho_ph(system.P_source, h_total))
    delta_p = max(system.P_source - system.P_outlet, 1.0)
    w_bernoulli = float(fraction * np.sqrt(2.0 * delta_p / rho))
    return rho * w_bernoulli * system.A_pipe


def run_tree(label: str, system: TreeSystem, *, warm_m_dot: float | None = None,
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

    if warm_m_dot is None:
        warm_m_dot = _bernoulli_warm_start(system)
    # In a balanced K-way tree, the source carries the full m_dot, depth-d
    # pipes each carry m_dot/K**d, and leaves carry m_dot/K**N.  We seed every
    # unknown at its locally-appropriate value -- a single global guess works
    # poorly at large N because leaves are K**N times smaller than the source.
    print(f"Warm-starting m_dot unknowns at depth-scaled Bernoulli value "
          f"(source = {warm_m_dot * 1000:.3f} g/s, leaves = "
          f"{warm_m_dot / (system.K ** system.N) * 1000:.3f} g/s)")
    for var in system.active_vars_references:
        full = getattr(var, 'full_name', '')
        if (full.endswith('.m_dot_in') or full.endswith('.m_dot_out')
                or '.m_dot_out_' in full):
            depth = full.count('.child_')
            var.value = warm_m_dot / (system.K ** depth)

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
    pipe_outlet_names = sorted(n for n in names if n.endswith('.pipe.m_dot_out'))
    pipes_by_depth: dict[int, list[tuple[str, float, float, float]]] = {}
    for n in pipe_outlet_names:
        depth = n.count('.child_')
        prefix = n[: -len('m_dot_out')]
        m_dot_val = state[-1, names.index(n)]
        p_val = state[-1, names.index(prefix + 'p_out')]
        h_val = state[-1, names.index(prefix + 'h_out')]
        pipes_by_depth.setdefault(depth, []).append((n, m_dot_val, p_val, h_val))

    print()
    print(f"=== Steady-state pipe-outlet conditions ({label}) ===")
    for depth in sorted(pipes_by_depth):
        rows = pipes_by_depth[depth]
        m_dots = [m_dot for _, m_dot, _, _ in rows]
        ps = [p for _, _, p, _ in rows]
        # Derive velocity post-hoc for human-readable output (m_dot / (rho * A)).
        ws = []
        for _, m_dot, p, h in rows:
            rho = float(system.medium.eval_rho_ph(float(p), float(h)))
            ws.append(float(m_dot) / (rho * system.A_pipe))
        print(
            f"  depth {depth} ({len(rows):2d} pipe{'s' if len(rows) != 1 else ''}):"
            f"  m_dot min={min(m_dots) * 1000:9.4f} max={max(m_dots) * 1000:9.4f} g/s,"
            f"  w min={min(ws):8.3f} max={max(ws):8.3f} m/s,"
            f"  p_out min={min(ps) / 1e5:6.4f} max={max(ps) / 1e5:6.4f} bar"
        )

    m_dot_source = float(trace_last('.source.m_dot_out'))
    print()
    print(f"Mass conservation ({label}):")
    print(f"  source m_dot                        = {m_dot_source * 1000:9.4f} g/s")
    # Source must actually be flowing, otherwise the conservation check below
    # is vacuously satisfied at zero.
    assert abs(m_dot_source) > 1e-6, "source mass flow should be non-trivial"
    for depth in sorted(pipes_by_depth):
        rows = pipes_by_depth[depth]
        m_dot_d = sum(float(m_dot) for _, m_dot, _, _ in rows)
        rel_err = abs(m_dot_d - m_dot_source) / max(abs(m_dot_source), 1e-30)
        print(
            f"  depth {depth} ({system.K ** depth:2d} pipes) m_dot   = {m_dot_d * 1000:9.4f} g/s"
            f"  (rel err vs source: {rel_err:.2e})"
        )
        # Each depth must pass the full source mass flow (balanced K-way tree):
        # what the source emits is conserved across every cross-section.
        assert rel_err < 1e-6, (
            f"mass not conserved at depth {depth} ({label}): rel err {rel_err:.2e}"
        )

    if output_html is None:
        # Sanitise label into a usable filename.
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label.lower())
        output_html = f"pipe_tree_{safe}.html"
    out_path = plot_results(system.record, output_html, show=False, subdir="examples")
    print()
    print(f"Plot written to {out_path}")
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
