"""Run the hydrogen pipe from ``run_local.py`` -- but only from its JSON, live.

This is the *consumer* side of the tutorial.  It never imports the model class;
it launches a hydrogen **host** process, ships it the ``system.json`` that
``run_local.py`` saved, and runs it there.  Instead of waiting for the run to
finish, it opens a live **variable stream** and updates a matplotlib chart as
the data arrives -- the bore pressure and the wall permeation (inner uptake +
environment leak) over the year-long leak transient.

Run ``run_local.py`` first to produce ``system.json``, then::

    python examples/h2_permeation_pressurize/run_service.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import hydrogen as hd  # noqa: E402

SPEC_PATH = _HERE / "system.json"

# How long / how to integrate (a runtime choice, not part of the model).
DAY = 24 * 3600.0
T_END = 0.06 + 365.0 * DAY
STRATEGY = {"name": "richardson", "tol_local": 1e-3, "atol": 1.0}
ADAPTIVE = {"dt_start": 1e-4, "dt_min": 1e-7, "dt_max": 50.0 * DAY,
            "tol": 1e-6, "max_iter": 200}

# `delay` paces the host to ~this many wall-clock seconds per step so the chart
# visibly animates (the run itself is only ~80 adaptive steps).
STEP_DELAY = None

# A headless (Agg) backend can't show a live window -- we still build + save the
# figure, just skip the blocking plt.show().
_HEADLESS = matplotlib.get_backend().lower() == "agg"

P_VAR = "pipe_segment_0.p_in"


def main():
    service = hd.start_host(workers=1)
    try:
        # Load the saved model and compile it on the host.
        system = service.load_json(SPEC_PATH.read_text())
        system.instantiate()            # host auto-gathers the medium modules

        # drain compile logs
        for ev in system.poll_events():
            if ev["type"] == "log" or ev["type"] == "error" or ev["type"] == "warning ":
                print(f"  [host] {ev['message']}")

        system.initialise(n=1, tol=1e-6, max_iter=200)

        for ev in system.poll_events():
            if ev["type"] == "log":
                print(f"  [host] {ev['message']}")

        # --- live matplotlib chart ----------------------------------------
        plt.ion()
        fig, (ax_p, ax_q) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
        (line_p,) = ax_p.plot([], [], color="tab:blue")
        ax_p.set_ylabel("bore pressure [MPa]")
        ax_p.grid(True, alpha=0.3)
        ax_p.set_title("H2 pipe permeation (live, via hydrogen service)")
        (line_up,) = ax_q.plot([], [], color="tab:orange", label="inner uptake")
        (line_en,) = ax_q.plot([], [], color="tab:red", label="environment leak")
        ax_q.set_yscale("log")
        ax_q.set_xlabel("time [days]")
        ax_q.set_ylabel("permeation [kg/s]")
        ax_q.grid(True, alpha=0.3, which="both")
        ax_q.legend(loc="lower right")
        fig.tight_layout()

        # Kick off the run asynchronously (stream=True returns immediately) ...
        system.run(stop_time=T_END, strategy=STRATEGY, adaptive=ADAPTIVE, stream=True)
        
        # get live data from the simulation as updateble streams
        stream = system.vars_stream()
        p = stream.series(P_VAR)                    # 1-D bore pressure
        up = stream.series_values("m_dot_a_leak")   # 2-D, one col per segment
        en = stream.series_values("m_dot_b_leak")   # 2-D, one col per segment
        t = stream.time()                           # 1-D time
        # call stream.update() to update the data in the arrays above

        done = False
        while not done:
            for ev in system.poll_events():
                if ev["type"] == "error" or ev["type"] == "warning":
                    print(f"  [host error] {ev.get('kind')}: {ev.get('message')}")
                    done = True
                elif ev["type"] in ("done", "closed"):
                    done = True

             # try tio refresh and update chart if new data is available
            if stream.update():
                days = t.array / DAY
                line_p.set_data(days, p.array / 1e6)
                line_up.set_data(days, up.array.sum(axis=1))
                line_en.set_data(days, np.abs(en.array).sum(axis=1))
                for ax in (ax_p, ax_q):
                    ax.relim()
                    ax.autoscale_view()
                fig.canvas.draw_idle()

            plt.pause(0.03)

        stream.update()   # capture any final rows recorded after the last frame
        stream.close()

        # Final textual summary (matches run_local.py's tail).
        env = np.abs(en.array).sum(axis=1)
        _trapz = getattr(np, "trapezoid", np.trapz)   # numpy>=2 renamed trapz
        cum = float(_trapz(env, t.array)) if len(t) > 1 else 0.0
        print(f"\n  samples streamed     : {len(t)}")
        print(f"  watched (auto)       : {len(stream.list_watched_names())} vars")
        print(f"  final bore pressure  : {p.array[-1]/1e6:.3f} MPa")
        print(f"  env leak at end      : {env[-1]:.3e} kg/s")
        print(f"  cumulative H2 lost   : {cum*1e9:.3f} ug")

        out = Path(hd.local_results_path("examples", "h2_permeation_live.png"))
        fig.savefig(out, dpi=110)
        print(f"  chart saved to {out}")

        plt.ioff()
        if not _HEADLESS:
            print("Close the chart window to exit.")
            plt.show()

        system.close()
    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
