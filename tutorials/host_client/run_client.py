"""Drive a hydrogen system from a *separate tool* and chart it live.

This script plays the role of "the other tool" (a UI). It does NOT build a model
in-process: it launches a hydrogen **host** in its own process(es), ships it the
system described in the JSON file next to this script, then runs it in real time
while a matplotlib window updates **live** -- the UI loop stays responsive by
*polling* each frame instead of blocking.

It demonstrates the on-demand **variable stream** API: rather than telling
`system.run(...)` up front which variables to watch, the UI opens a
`system.vars_stream()` *after* the run has started, picks variables on demand
with `stream.series(name)` (live handles whose `.array` is refreshed together by
`stream.update()`), and redraws each frame.  A stream can be opened/closed at
any moment over any variables, so a UI can add or drop charts mid-run.

Run it::

    python tutorials/host_client/run_client.py

Under MPI with several workers (needs `mpi4py` + an MPI runtime on PATH)::

    HYDROGEN_WORKERS=5 python tutorials/host_client/run_client.py

Knobs (env vars):

    HYDROGEN_SIM_SECONDS   wall-clock duration of the live run    (default 60)
    HYDROGEN_DT            simulation step size in seconds        (default 0.05)

The system (`signal_dynamics.json`) is a 0.2 Hz sine source feeding a
first-order lag (T = 1.5 s): you watch the lag's output trail the source with
the classic attenuation + phase shift, drawn live.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Allow `python tutorials/host_client/run_client.py` without `pip install -e .`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import hydrogen
from hydrogen import local_results_path

# True when matplotlib has no interactive (GUI) backend -- e.g. a headless SSH
# session or CI where the backend falls back to 'Agg'. In that case the live
# window can't be shown, so we skip the blocking plt.show() at the end and rely
# on the saved PNG instead.
_HEADLESS = matplotlib.get_backend().lower() == "agg"

HERE = Path(__file__).resolve().parent
SYSTEM_JSON = HERE / "signal_dynamics.json"

# Signals to stream + plot (suffix-matched against full recorded names).
WATCH = ["src.y", "lag.y"]

SIM_SECONDS = float(os.environ.get("HYDROGEN_SIM_SECONDS", "60"))
DT = float(os.environ.get("HYDROGEN_DT", "0.05"))
STEPS = max(1, int(round(SIM_SECONDS / DT)))


def main():
    workers = 1
    print(f"Starting hydrogen host (workers={workers}) ...")
    service = hydrogen.start_host(workers=workers)

    try:
        # load from json file
        system = service.load_json(SYSTEM_JSON.read_text())
        print(f"Loaded {SYSTEM_JSON.name} -> {system.system_id}")

        # start instatniation and continualy read logs
        system.instantiate(max_remove_trival_passes=5)
        for ev in system.poll_events():           # drain compile logs
            if ev["type"] == "log":
                print(f"  [host] {ev['message']}")

        # initialise the system
        system.initialise(n=1)
        

        # --- THIS is UI: set up the live chart -------------------------------------
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.subplots_adjust(bottom=0.22)  # leave room for the control buttons
        lines = {name: ax.plot([], [], label=name)[0] for name in WATCH}
        ax.set_xlabel("time [s]")
        ax.set_ylabel("signal value")
        ax.set_title("hydrogen live (sine -> first-order lag) via poll_events")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        # --- UI controls: Pause / Resume / Stop the host computation ---------------
        # These call straight through to the SystemProxy; the host honours them at
        # the next step boundary (a pause freezes the model state until resume).
        paused = {"on": False}

        def on_pause(_event):
            paused["on"] = not paused["on"]
            if paused["on"]:
                system.pause()
                btn_pause.label.set_text("Resume")
                print("  [ui] pause requested")
            else:
                system.resume()
                btn_pause.label.set_text("Pause")
                print("  [ui] resume requested")
            fig.canvas.draw_idle()

        def on_stop(_event):
            system.stop()
            print("  [ui] stop requested")

        ax_pause = fig.add_axes([0.68, 0.04, 0.13, 0.075])
        ax_stop = fig.add_axes([0.82, 0.04, 0.13, 0.075])
        btn_pause = Button(ax_pause, "Pause")
        btn_stop = Button(ax_stop, "Stop")
        btn_pause.on_clicked(on_pause)
        btn_stop.on_clicked(on_stop)

        # start the simulation in the hydrogen service. Note: no `vars=` here --
        # the run just advances + records; we pick what to chart separately.
        print(f"Running {STEPS} steps at dt={DT}s, real-time (~{SIM_SECONDS:.0f}s)\n")
        system.run(dt=DT, steps=STEPS, stream=True, every=20, delay=DT)

        # Open an on-demand stream (watches nothing yet), then pick the signals
        # to chart as live handles -- each `.array` is refreshed together by
        # stream.update() and backfilled with whatever history already exists.
        # (We could open this before run(), or add more handles mid-run.)
        stream = system.vars_stream()
        sig = {name: stream.series(name) for name in WATCH}
        t = stream.time()

        # UI does stuff
        done = False
        next_report = time.monotonic() + 5.0
        while not done: # main ui loop
            # control/diagnostic feedback: status, logs, completion.
            for ev in system.poll_events(): 
                kind = ev["type"]
                if kind == "status":
                    phase = ev.get("phase", "running")
                    note = f" [{phase}]" if phase != "running" else ""
                    print(f"  [host] progress: {100*ev['step']/STEPS:.2f}%{note}")
                elif kind == "log":
                    print(f"  [host] {ev['message']}")
                elif kind == "error":
                    print(f"  [ERROR] {ev['kind']}: {ev['message']}")
                    done = True
                elif kind in ("done", "closed"):
                    done = True

            # chart data: refresh handles; True only when new rows arrived.
            if stream.update():
                for name in WATCH:
                    lines[name].set_data(t.array, sig[name].array)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw_idle()

            # Periodic console heartbeat so the headless run shows progress.
            if time.monotonic() >= next_report and len(t):
                print(f"  t={t.array[-1]:6.2f}s  " +
                      "  ".join(f"{n}={sig[n].array[-1]:+.3f}" for n in WATCH))
                next_report = time.monotonic() + 5.0

            # Yield to the GUI event loop and pace the UI at ~30 fps. This is
            # what keeps the window responsive while the sim streams.
            plt.pause(0.03)

        stream.update()    # capture any final rows
        stream.close()
        print(f"\nRun finished: {len(t)} samples plotted.")
        out = Path(local_results_path("tutorials", "signal_dynamics_live.png"))
        fig.savefig(out, dpi=110)
        print(f"Chart saved to {out}")

        plt.ioff()
        if not _HEADLESS:
            print("Close the chart window to exit.")
            plt.show()

        system.close()

    finally:
        service.shutdown()
        print("Host shut down.")


if __name__ == "__main__":
    main()
