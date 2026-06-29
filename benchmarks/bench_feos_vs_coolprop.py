"""Speed/accuracy A-B of the ``saved_system_2`` network: CoolProp vs feos media.

Same five-component network as ``saved_system_2.py`` (PressureSource -> Valve ->
Pipe(100 segments) -> Tank, ramp-commanded valve), run once with each
thermophysical backend at the SAME integrator (L-stable TR-BDF2) and tolerance,
so the only thing that changes is the medium:

    * coolprop : CoolPropMedium("Hydrogen", backend="BICUBIC&HEOS")  -- the tuned
                 tabular reference backend the example uses.
    * feos     : FeosMedium("Hydrogen")  -- feos Peng-Robinson EOS (built from
                 CoolProp critical constants) for thermodynamics, CoolProp for
                 transport, finite-difference partials.

Each backend runs in its OWN spawned process: the per-template lambda cache keys
on the medium *function names* (both media share the ``Hydrogen_`` prefix), so a
same-process A-B could otherwise reuse the first backend's compiled callables.
Separate processes give each its own clean in-memory cache.

Run from the project root::

    python benchmarks/bench_feos_vs_coolprop.py            # default stop_time
    python benchmarks/bench_feos_vs_coolprop.py 20         # custom stop_time
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# The integrator/controller is held fixed across backends so the comparison
# isolates the medium cost, not the solver.
STRATEGY = {"name": "tr_bdf2", "tol_local": 1e-4, "atol": 1.0}

# Sample grid for the tank-pressure trajectory agreement check.
_N_SAMPLE = 400


def _make_medium(provider: str):
    """Build the Hydrogen medium for the requested backend."""
    if provider == "feos":
        from hydrogen import FeosMedium
        return FeosMedium("Hydrogen", disable_warnings=True,
                          scalar_cache_maxsize=1000)
    from hydrogen import CoolPropMedium
    return CoolPropMedium("Hydrogen", disable_warnings=True,
                          backend="BICUBIC&HEOS", scalar_cache_maxsize=1000)


def _build_model(medium):
    """Instantiate + initialise the saved_system_2 network on `medium`."""
    from hydrogen import Model
    from hydrogen.components.control.control_components import Ramp
    from hydrogen.components.materials import AISI_316
    from hydrogen.components.thermofluid.assemblies import Pipe, Tank, WallLayer
    from hydrogen.components.thermofluid.flow import (
        CompressibleValve,
        PressureSource,
    )

    wall = [WallLayer(material=AISI_316, thickness=0.002, dynamic=True)]

    class Rig(Model):
        def declare_components(self):
            self.add_component("pressuresource_5", PressureSource(
                medium, p_source=1001325.0, T_source=293.15, A=0.001,
                p_control=False))
            self.add_component("compressiblevalve_4", CompressibleValve(
                medium, Kv=1.0, D=0.01))
            self.add_component("pipe_2", Pipe(
                medium, D=0.01, L=1.0, epsilon=1e-6, z_in=0.0, z_out=0.0,
                n_segments=100, layers=wall))
            self.add_component("tank_3", Tank(
                medium, volume=0.05, diameter=0.3, layers=wall, h_inner=50.0))
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

    m = Rig()
    m.instantiate(
        aditional_modules=medium.modules,
        cse=True, enable_blt=True, enable_var_scaling=False,
        max_remove_trival_passes=1, max_remove_duplicate_passes=5,
        max_remove_linear_block_passes=3,
    )
    m.initialise(n=1, relaxation=1.0, tol=1e-6, max_iter=200)
    return m


def _run_one(provider: str, stop_time: float) -> dict:
    """Build + run the network on one backend; return timings + trajectory."""
    import numpy as np

    medium = _make_medium(provider)

    t0 = time.perf_counter()
    m = _build_model(medium)
    t_instantiate = time.perf_counter() - t0

    t0 = time.perf_counter()
    summary = m.run(
        stop_time=stop_time, strategy=STRATEGY,
        dt_start=1e-4, dt_min=1e-9, dt_max=10.0,
        grow=1.5, shrink=0.5, max_retries=30,
        relaxation=1.0, tol=1e-6, max_iter=200,
        raise_on_no_convergence=False,
    )
    t_run = time.perf_counter() - t0

    tq = np.linspace(0.0, stop_time, _N_SAMPLE)
    p_tank = m.interp_series("tank_3.gas.p", tq)
    T_tank = m.interp_series("tank_3.gas.T", tq)
    return {
        "provider": provider,
        "t_instantiate": t_instantiate,
        "t_run": t_run,
        "steps": summary["steps"],
        "rejections": summary["rejections"],
        "t_end": summary["t_end"],
        "stop_reason": summary["stop_reason"],
        "tq": tq,
        "p_tank": p_tank,
        "T_tank": T_tank,
    }


def _worker(provider, stop_time, q):
    try:
        q.put(_run_one(provider, stop_time))
    except Exception as exc:  # surface the child's failure to the parent
        import traceback
        q.put({"provider": provider, "error": f"{exc}\n{traceback.format_exc()}"})


def _run_in_process(provider, stop_time):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(provider, stop_time, q))
    p.start()
    result = q.get()
    p.join()
    return result


def main():
    import numpy as np

    stop_time = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0

    print("=" * 84)
    print("saved_system_2 — CoolProp vs feos medium  (PressureSource -> Valve -> "
          "Pipe -> Tank)")
    print(f"stop_time={stop_time:g}s  integrator={STRATEGY['name']}"
          f"(tol_local={STRATEGY['tol_local']:g})  pipe n_segments=100")
    print("=" * 84)

    results = {}
    for provider in ("coolprop", "feos"):
        print(f"\n--- running '{provider}' (separate process) ...", flush=True)
        res = _run_in_process(provider, stop_time)
        if "error" in res:
            print(f"  {provider} FAILED:\n{res['error']}")
            return
        results[provider] = res
        print(f"  instantiate={res['t_instantiate']:.2f}s  run={res['t_run']:.2f}s  "
              f"steps={res['steps']}  rej={res['rejections']}  "
              f"t_end={res['t_end']:.3f}s  stop={res['stop_reason']}")

    cp, fe = results["coolprop"], results["feos"]

    print(f"\n{'backend':<10} {'instantiate':>12} {'run [s]':>10} {'steps':>7} "
          f"{'rej':>5} {'us/step':>9}")
    print("-" * 60)
    for r in (cp, fe):
        us_per_step = 1e6 * r["t_run"] / max(r["steps"], 1)
        print(f"{r['provider']:<10} {r['t_instantiate']:>11.2f}s {r['t_run']:>10.2f} "
              f"{r['steps']:>7d} {r['rejections']:>5d} {us_per_step:>9.0f}")
    print("-" * 60)

    run_ratio = fe["t_run"] / cp["t_run"] if cp["t_run"] else float("nan")
    # Normalise out the (different) step counts to isolate per-step medium cost.
    per_step_ratio = ((fe["t_run"] / max(fe["steps"], 1)) /
                      (cp["t_run"] / max(cp["steps"], 1)))
    print(f"feos run wall is {run_ratio:.2f}x CoolProp "
          f"({per_step_ratio:.2f}x per step).")

    # --- physical agreement of the tank-pressure / temperature trajectories --
    # (h has a different reference per backend, but p and T are physical).
    p_scale = max(float(np.nanmax(np.abs(cp["p_tank"]))), 1.0)
    p_rel = float(np.nanmax(np.abs(fe["p_tank"] - cp["p_tank"]))) / p_scale
    T_scale = max(float(np.nanmax(np.abs(cp["T_tank"]))), 1.0)
    T_rel = float(np.nanmax(np.abs(fe["T_tank"] - cp["T_tank"]))) / T_scale
    print(f"\ntank pressure: feos vs CoolProp max rel diff = {p_rel:.3e}  "
          f"(end p: CoolProp={cp['p_tank'][-1]/1e5:.3f} bar, "
          f"feos={fe['p_tank'][-1]/1e5:.3f} bar)")
    print(f"tank temperature: max rel diff = {T_rel:.3e}  "
          f"(end T: CoolProp={cp['T_tank'][-1]:.2f} K, feos={fe['T_tank'][-1]:.2f} K)")
    print("\nNote: feos here is by-name Peng-Robinson (cubic) + constant-cp ideal "
          "gas;\nexpect modest density differences vs CoolProp's reference EOS.")


if __name__ == "__main__":
    main()
