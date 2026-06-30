"""Benchmark + correctness check for SegmentedChannel vs StraightPipe.

Builds the saved_system_2 network (PressureSource -> Valve -> Pipe -> Tank)
with a parametrised segment count and channel engine, then times instantiate
and a short run and compares the tank trajectories between engines.

    python benchmarks/bench_segmented.py smoke      # cheap symbolic build only
    python benchmarks/bench_segmented.py run 10      # full instantiate+run, N=10
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hydrogen import CoolPropMedium, Model  # noqa: E402
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


def _layers():
    perm = TransientDiffusion(H2_IN_AISI_316, n_nodes=5) if PERMEATION else None
    return [WallLayer(material=AISI_316, thickness=0.002, dynamic=True,
                      permeation=perm)]


def build_model(n_segments: int, engine: str):
    layers = _layers()

    class BenchSystem(Model):
        def declare_components(self):
            self.add_component("pressuresource_5", PressureSource(
                HYDROGEN, p_source=1001325.0, T_source=293.15, A=0.001,
                p_control=False))
            self.add_component("compressiblevalve_4", CompressibleValve(
                HYDROGEN, Kv=1.0, D=0.01))
            self.add_component("pipe_2", Pipe(
                HYDROGEN, D=0.01, L=1.0, epsilon=1e-6, z_in=0.0, z_out=0.0,
                n_segments=n_segments, layers=layers, channel_engine=engine,
                dynamic="static",
                p_ext=1.0))
            self.add_component("tank_3", Tank(
                HYDROGEN, volume=0.05, diameter=0.3, layers=layers,
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


def smoke():
    # Instantiate-only check of the full wired system (ports connected) at a
    # small N for both engines, reporting the reduced system size.
    for engine in ("straight", "segmented"):
        m = build_model(2, engine)
        t0 = time.perf_counter()
        m.instantiate(
            aditional_modules=HYDROGEN.modules, cse=True,
            enable_blt=True, enable_var_scaling=False,
            max_remove_trival_passes=1, max_remove_duplicate_passes=5,
            max_remove_linear_block_passes=3)
        dt = time.perf_counter() - t0
        n_states = len(getattr(m, "state_vars", []) or [])
        print(f"  engine={engine:9s} N=2  instantiate={dt:6.2f}s  "
              f"states={n_states}")
    print("smoke OK")


def run(n_segments: int, engines=("straight", "segmented")):
    results = {}
    for engine in engines:
        print("=" * 60)
        print(f"engine={engine}  n_segments={n_segments}")
        m = build_model(n_segments, engine)
        t0 = time.perf_counter()
        m.instantiate(
            aditional_modules=HYDROGEN.modules, cse=True,
            enable_blt=True, enable_var_scaling=False,
            max_remove_trival_passes=1, max_remove_duplicate_passes=5,
            max_remove_linear_block_passes=3)
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
        import numpy as np
        p_tank = np.asarray(m.series("tank_3.gas.p"))
        T_tank = np.asarray(m.series("tank_3.gas.T"))
        results[engine] = {
            "t_inst": t_inst, "t_run": t_run,
            "p_end": float(p_tank[-1]), "T_end": float(T_tank[-1]),
            "steps": summary["steps"], "stop": summary["stop_reason"],
        }
        print(f"  instantiate={t_inst:8.2f}s  run={t_run:8.2f}s  "
              f"steps={summary['steps']}  stop={summary['stop_reason']}")
        print(f"  p_tank_end={p_tank[-1]/1e5:.5f} bar  T_tank_end={T_tank[-1]:.4f} K")

    if len(results) == 2:
        s, g = results["straight"], results["segmented"]
        dp = abs(s["p_end"] - g["p_end"]) / abs(s["p_end"])
        dT = abs(s["T_end"] - g["T_end"]) / abs(s["T_end"])
        print("=" * 60)
        print(f"RESULT MATCH: rel dp={dp:.3e}  rel dT={dT:.3e}")
        print(f"SPEEDUP: instantiate {s['t_inst']/g['t_inst']:.2f}x  "
              f"run {s['t_run']/max(g['t_run'],1e-9):.2f}x")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    if "--leaky" in args:
        PERMEATION = True
        args.remove("--leaky")
    only = None
    for opt in ("--straight", "--segmented"):
        if opt in args:
            only = (opt[2:],)
            args.remove(opt)
    if args and args[0] == "smoke":
        smoke()
    elif len(args) >= 2 and args[0] == "run":
        run(int(args[1]), engines=only or ("straight", "segmented"))
    elif len(args) >= 2 and args[0] == "inst":
        # Instantiate-only timing (no simulation); useful for the very large N
        # where the engine-independent solve would dominate wall time.
        for engine in (only or ("straight", "segmented")):
            m = build_model(int(args[1]), engine)
            t0 = time.perf_counter()
            m.instantiate(
                aditional_modules=HYDROGEN.modules, cse=True,
                enable_blt=True, enable_var_scaling=False,
                max_remove_trival_passes=1, max_remove_duplicate_passes=5,
                max_remove_linear_block_passes=3)
            print(f"INST engine={engine:9s} N={args[1]}  "
                  f"instantiate={time.perf_counter() - t0:8.2f}s")
    else:
        print(__doc__)
