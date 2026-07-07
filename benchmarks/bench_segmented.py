"""Benchmark + correctness check for SegmentedChannel vs StraightPipe.

Builds the saved_system_2 network (PressureSource -> Valve -> Pipe -> Tank)
with a parametrised segment count and channel engine, then times instantiate
and a short run and compares the tank trajectories between engines.

    python benchmarks/bench_segmented.py smoke      # cheap symbolic build only
    python benchmarks/bench_segmented.py run 10      # full instantiate+run, N=10
    python benchmarks/bench_segmented.py run 10 --tab-speedup  # also run with a
                                                      # TabulatedMedium surrogate
                                                      # and report the speedup vs
                                                      # the CoolProp EoS backend
    python benchmarks/bench_segmented.py run 100 --tab-only --segmented
                                                      # TabulatedMedium only

Flags:
    --straight / --segmented   restrict to one channel engine
    --dynamic=LEVEL            static [default] / advective / compressible /
                               acoustic (transient levels -> segmented only)
    --tab-speedup              additionally run each case with a spline
                               `TabulatedMedium` surrogate wrapping the same
                               CoolProp fluid, and report the property-backend
                               speedup + trajectory match
    --tab-only                 run with `TabulatedMedium` only (skip CoolProp)
    --warm-cache               untimed pre-instantiate before timed runs (lambda
                               cache warm-up; off by default)
    --numba                    JIT-compile vectorised equation templates with
                               numba (requires TabulatedMedium; auto-enables
                               --tab-only if no other tab flag is set)
    --leaky                    enable wall permeation
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hydrogen import CoolPropMedium, Model, TabulatedMedium  # noqa: E402
from hydrogen.components.control.control_components import Ramp  # noqa: E402
from hydrogen.components.materials import AISI_316  # noqa: E402
from hydrogen.components.thermofluid.assemblies import Pipe, Tank, WallLayer  # noqa: E402
from hydrogen.components.thermofluid.flow import (  # noqa: E402
    CompressibleValve,
    PressureSource,
    SegmentedChannel,
    StraightPipe,
)
from hydrogen.components.thermofluid.permeation import (  # noqa: E402
    H2_IN_AISI_316,
    TransientDiffusion,
)

HYDROGEN = CoolPropMedium("Hydrogen", disable_warnings=True,
                          backend="BICUBIC&HEOS", scalar_cache_maxsize=1000)


PERMEATION = False  # toggled by the `--leaky` arg

_TAB_MEDIUM = None  # lazily-built TabulatedMedium surrogate (see `_tab_medium`)


def _tab_medium():
    """A spline `TabulatedMedium` wrapping `HYDROGEN` over the benchmark's
    operating window (~1..10 bar, ~280..330 K single-phase gas).

    Built once and cached.  The window is sized from the CoolProp source at the
    corner states so every state the run visits is interpolated (not
    extrapolated).  Hydrogen at these temperatures is far above its ~33 K
    critical point, so there is no two-phase dome (`two_phase=False`)."""
    global _TAB_MEDIUM
    if _TAB_MEDIUM is None:
        p_lo, p_hi = 0.9e5, 10.5e5
        T_lo, T_hi = 280.0, 330.0
        hs = [float(HYDROGEN.eval_h_pT(p, T))
              for p in (p_lo, p_hi) for T in (T_lo, T_hi)]
        h_lo, h_hi = min(hs), max(hs)
        pad = 0.05 * (h_hi - h_lo)
        t0 = time.perf_counter()
        _TAB_MEDIUM = TabulatedMedium(
            HYDROGEN, p_range=(p_lo, p_hi), h_range=(h_lo - pad, h_hi + pad),
            n_p=128, n_h=128, two_phase=False)
        print(f"TabulatedMedium build: {time.perf_counter() - t0:.2f}s "
              f"(p=[{p_lo/1e5:.1f}, {p_hi/1e5:.1f}] bar, grid 128x128)")
    return _TAB_MEDIUM


def _layers():
    perm = TransientDiffusion(H2_IN_AISI_316, n_nodes=5) if PERMEATION else None
    return [WallLayer(material=AISI_316, thickness=0.002, dynamic=True,
                      permeation=perm)]


def build_model(n_segments: int, engine: str, dynamic: str = "static",
                medium=None):
    med = medium if medium is not None else HYDROGEN
    layers = _layers()
    # `dispersion` only applies to the transient (cell-centred) levels; the
    # quasi-steady `static` level ignores it entirely (no axial-diffusion term).
    pipe_kwargs = {} if dynamic == "static" else {"dispersion": "general"}

    class BenchSystem(Model):
        def declare_components(self):
            self.add_component("pressuresource_5", PressureSource(
                med, p_source=1001325.0, T_source=293.15, A=0.001,
                p_control=False))
            self.add_component("compressiblevalve_4", CompressibleValve(
                med, Kv=1.0, D=0.01))
            self.add_component("pipe_2", Pipe(
                med, D=0.01, L=1.0, epsilon=1e-6, z_in=0.0, z_out=0.0,
                n_segments=n_segments, layers=layers, channel_engine=engine,
                dynamic=dynamic, p_ext=1.0, **pipe_kwargs))
            self.add_component("tank_3", Tank(
                med, volume=0.05, diameter=0.3, layers=layers,
                h_inner=50.0, p_ext=1.0))
            self.add_component("ramp_6", Ramp(
                height=1.0, duration=1.0, start_time=1.0, offset=0.0, unit="1"))

        def declare_equations(self):
            self.connect(self["pressuresource_5"].ports["outlet"],
                         self["compressiblevalve_4"].ports["inlet"])
            self.connect(self["compressiblevalve_4"].ports["outlet"],
                         self["pipe_2"].ports["inlet"])
            self.connect(self["pipe_2"].ports["outlet"],
                         self["tank_3"].ports["inlet"])
            self.connect(self["ramp_6"].ports["y"],
                         self["compressiblevalve_4"].ports["opening"])
            return []

    return BenchSystem()


def smoke(dynamic="static", engines=("straight", "segmented")):
    # Instantiate-only check of the full wired system (ports connected) at a
    # small N for both engines, reporting the reduced system size.
    for engine in engines:
        m = build_model(2, engine, dynamic)
        t0 = time.perf_counter()
        m.instantiate(
            aditional_modules=HYDROGEN.modules, cse=True, enable_blt=True,
            max_remove_trival_passes=1, max_remove_duplicate_passes=5,
            max_remove_linear_block_passes=3)
        dt = time.perf_counter() - t0
        n_states = len(getattr(m, "state_vars", []) or [])
        print(f"  engine={engine:9s} N=2  instantiate={dt:6.2f}s  "
              f"states={n_states}")
    print("smoke OK")


def _instantiate(m, medium, numba=False):
    m.instantiate(
        aditional_modules=medium.modules, cse=True, enable_blt=True,
        max_remove_trival_passes=1, max_remove_duplicate_passes=5,
        max_remove_linear_block_passes=3, numba=numba)


def _warm_lambda_cache(n_segments, engines, dynamic, media, numba=False):
    """Untimed instantiate pass so subsequent timed runs hit a warm lambda cache."""
    print("warming lambda cache ...")
    for engine in engines:
        for _label, med in media:
            _instantiate(build_model(n_segments, engine, dynamic, med), med,
                         numba=numba)


def _simulate(n_segments, engine, dynamic, medium, numba=False):
    """Build + instantiate + run one (engine, dynamic, medium) case; return the
    timings and tank end-state.  `enable_var_scaling` is left at its default
    (True)."""
    import numpy as np
    m = build_model(n_segments, engine, dynamic, medium)
    t0 = time.perf_counter()
    _instantiate(m, medium, numba=numba)
    t_inst = time.perf_counter() - t0
    m.initialise(n=1, relaxation=1.0, tol=1e-6, max_iter=200)
    t1 = time.perf_counter()
    summary = m.run(
        stop_time=100.0,
        strategy={"name": "tr_bdf2", "tol_local": 1e-3, "atol": 1.0},
        dt_start=1e-4, dt_min=1e-9, dt_max=50.0, grow=1.5, shrink=0.5,
        max_retries=30, relaxation=1.0, tol=1e-6, max_iter=200,
        raise_on_no_convergence=False)
    t_run = time.perf_counter() - t1
    p_tank = np.asarray(m.series("tank_3.gas.p"))
    T_tank = np.asarray(m.series("tank_3.gas.T"))
    return {
        "t_inst": t_inst, "t_run": t_run,
        "p_end": float(p_tank[-1]), "T_end": float(T_tank[-1]),
        "steps": summary["steps"], "stop": summary["stop_reason"],
    }


def _media_list(tab_speedup=False, tab_only=False):
    """Return ``[(label, medium), ...]`` for the requested property backend(s)."""
    if tab_only:
        return [("tab", _tab_medium())]
    media = [("coolprop", HYDROGEN)]
    if tab_speedup:
        media.append(("tab", _tab_medium()))
    return media


def run(n_segments: int, engines=("straight", "segmented"), dynamic="static",
        tab_speedup=False, tab_only=False, warm_cache=False, numba=False):
    media = _media_list(tab_speedup=tab_speedup, tab_only=tab_only)

    if warm_cache:
        _warm_lambda_cache(n_segments, engines, dynamic, media, numba=numba)

    results = {}  # (engine, medium_label) -> result dict
    for engine in engines:
        for med_label, med in media:
            print("=" * 60)
            print(f"engine={engine}  n_segments={n_segments}  "
                  f"dynamic={dynamic}  medium={med_label}  numba={numba}")
            r = _simulate(n_segments, engine, dynamic, med, numba=numba)
            results[(engine, med_label)] = r
            print(f"  instantiate={r['t_inst']:8.2f}s  run={r['t_run']:8.2f}s  "
                  f"steps={r['steps']}  stop={r['stop']}")
            print(f"  p_tank_end={r['p_end']/1e5:.5f} bar  "
                  f"T_tank_end={r['T_end']:.4f} K")

    # Engine cross-check (CoolProp): straight vs segmented should match.
    if ("straight", "coolprop") in results and ("segmented", "coolprop") in results:
        s, g = results[("straight", "coolprop")], results[("segmented", "coolprop")]
        dp = abs(s["p_end"] - g["p_end"]) / abs(s["p_end"])
        dT = abs(s["T_end"] - g["T_end"]) / abs(s["T_end"])
        print("=" * 60)
        print(f"ENGINE MATCH (coolprop): rel dp={dp:.3e}  rel dT={dT:.3e}")
        print(f"ENGINE SPEEDUP: instantiate {s['t_inst']/g['t_inst']:.2f}x  "
              f"run {s['t_run']/max(g['t_run'],1e-9):.2f}x")

    # Property-backend cross-check: tab vs coolprop, per engine.
    if tab_speedup and not tab_only:
        print("=" * 60)
        print("TABULATED vs COOLPROP (per engine):")
        for engine in engines:
            cp = results.get((engine, "coolprop"))
            tb = results.get((engine, "tab"))
            if cp is None or tb is None:
                continue
            dp = abs(cp["p_end"] - tb["p_end"]) / abs(cp["p_end"])
            dT = abs(cp["T_end"] - tb["T_end"]) / abs(cp["T_end"])
            print(f"  [{engine}] match: rel dp={dp:.3e}  rel dT={dT:.3e}  "
                  f"(tab steps={tb['steps']} vs cp steps={cp['steps']})")
            print(f"  [{engine}] SPEEDUP tab/coolprop: "
                  f"instantiate {cp['t_inst']/max(tb['t_inst'],1e-9):.2f}x  "
                  f"run {cp['t_run']/max(tb['t_run'],1e-9):.2f}x")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    if "--leaky" in args:
        PERMEATION = True
        args.remove("--leaky")
    tab_speedup = False
    tab_only = False
    if "--tab-only" in args:
        tab_only = True
        args.remove("--tab-only")
    if "--tab-speedup" in args:
        if not tab_only:
            tab_speedup = True
        args.remove("--tab-speedup")
    warm_cache = "--warm-cache" in args
    if warm_cache:
        args.remove("--warm-cache")
    numba = "--numba" in args
    if numba:
        args.remove("--numba")
    # numba JITs TabulatedMedium property twins inside compiled templates;
    # CoolProp callbacks cannot be njit-compiled.
    if numba and not tab_only and not tab_speedup:
        print("NOTE: --numba requires TabulatedMedium; enabling --tab-only")
        tab_only = True
    # Optional `--dynamic=LEVEL` (static [default] / advective / compressible /
    # acoustic).  The `straight` engine only implements `static`, so any
    # transient level is restricted to the `segmented` engine.  `--dynamics=`
    # is accepted as an alias for the common typo.
    dynamic = "static"
    for a in list(args):
        if a.startswith("--dynamic=") or a.startswith("--dynamics="):
            dynamic = a.split("=", 1)[1]
            args.remove(a)
    only = None
    for opt in ("--straight", "--segmented"):
        if opt in args:
            only = (opt[2:],)
            args.remove(opt)
    # Any remaining `--flag` is unrecognised: warn rather than silently ignore
    # (a mistyped flag would otherwise fall through to the defaults).
    unknown = [a for a in args if a.startswith("--")]
    if unknown:
        print(f"WARNING: ignoring unrecognised flag(s): {' '.join(unknown)}")
        for a in unknown:
            args.remove(a)
    engines = only or ("straight", "segmented")
    if dynamic != "static":
        engines = ("segmented",)
    if args and args[0] == "smoke":
        smoke(dynamic=dynamic, engines=engines)
    elif len(args) >= 2 and args[0] == "run":
        run(int(args[1]), engines=engines, dynamic=dynamic,
            tab_speedup=tab_speedup, tab_only=tab_only, warm_cache=warm_cache,
            numba=numba)
    elif len(args) >= 2 and args[0] == "inst":
        # Instantiate-only timing (no simulation); useful for the very large N
        # where the engine-independent solve would dominate wall time.
        media = _media_list(tab_speedup=tab_speedup, tab_only=tab_only)
        if warm_cache:
            _warm_lambda_cache(int(args[1]), engines, dynamic, media,
                               numba=numba)
        for engine in engines:
            for med_label, med in media:
                m = build_model(int(args[1]), engine, dynamic, med)
                t0 = time.perf_counter()
                _instantiate(m, med, numba=numba)
                print(f"INST engine={engine:9s} medium={med_label:8s} "
                      f"numba={numba}  N={args[1]}  "
                      f"instantiate={time.perf_counter() - t0:8.2f}s")
    else:
        print(__doc__)
