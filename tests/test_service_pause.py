"""Pause / resume control over a live streaming run on the hydrogen host.

Spawns a real host subprocess (``workers=1``, no MPI) running the CoolProp-free
signal system from ``tutorials/host_client/signal_dynamics.json``, then exercises
the cooperative pause path end-to-end:

* ``pause()`` parks the run at a step boundary -> the host emits a
  ``status`` event with ``phase == "paused"`` and the streamed step stops
  advancing (model state frozen);
* ``status()`` keeps replying *while paused* (the run loop still services the
  command socket);
* ``resume()`` continues the run from exactly where it parked.
"""

from __future__ import annotations

import time
from pathlib import Path

import hydrogen

_SPEC = (
    Path(__file__).resolve().parent.parent
    / "tutorials" / "host_client" / "signal_dynamics.json"
).read_text()


def _drain_for(system, stream, events, samples, seconds):
    """Drain system events + stream chunks for ``seconds``.

    Appends events to ``events`` and counts streamed samples in ``samples``
    (a one-element list used as a mutable counter of recorded time points).
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        events.extend(system.poll_events())
        for ch in stream.poll():
            if ch.get("type") == "stream_data":
                samples[0] += len(ch["time"])
        time.sleep(0.02)


def test_pause_freezes_then_resume_continues():
    service = hydrogen.start_host(workers=1)
    try:
        system = service.load_json(_SPEC)
        system.instantiate(max_remove_trival_passes=2)
        system.poll_events()  # drop compile logs
        system.initialise(n=1)

        steps = 500
        # A small per-step delay makes the run last long enough to pause it
        # mid-flight deterministically (~10 s budget; we stop early below).
        system.run(dt=0.02, steps=steps, stream=True, every=5, delay=0.02)
        # Watch progress via an on-demand stream (the only live data channel).
        # We only count streamed time points, so no variables need registering;
        # an empty stream still emits time-stamped chunks each flush.
        stream = system.vars_stream()

        events: list[dict] = []
        samples = [0]  # mutable counter of streamed time points

        # Let a few steps stream, then request a pause.
        _drain_for(system, stream, events, samples, 0.3)
        ack = system.pause()
        assert ack.get("pausing") == system.system_id

        # Wait for the host to confirm it parked (status event, phase=paused).
        paused_seen = False
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not paused_seen:
            for ev in system.poll_events():
                events.append(ev)
                if ev.get("type") == "status" and ev.get("phase") == "paused":
                    paused_seen = True
            for ch in stream.poll():
                if ch.get("type") == "stream_data":
                    samples[0] += len(ch["time"])
            time.sleep(0.02)
        assert paused_seen, "host never reported phase == 'paused'"

        # Settle: drain anything still in flight, then sample the frozen count.
        _drain_for(system, stream, events, samples, 0.4)
        samples_frozen = samples[0]

        # status() must still answer while paused, and report the paused phase.
        st = system.status()
        assert st["phase"] == "paused"

        # While paused the stream must not advance.
        _drain_for(system, stream, events, samples, 0.6)
        assert samples[0] == samples_frozen, "run advanced while paused"

        # Resume and confirm progress picks back up past the frozen count.
        ack = system.resume()
        assert ack.get("resuming") == system.system_id

        resumed = False
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not resumed:
            for ev in system.poll_events():
                events.append(ev)
            for ch in stream.poll():
                if ch.get("type") == "stream_data":
                    samples[0] += len(ch["time"])
            if samples[0] > samples_frozen:
                resumed = True
            time.sleep(0.02)
        assert resumed, "run did not advance after resume"

        # Clean cooperative stop + terminal event.
        system.stop()
        done = False
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not done:
            for ev in system.poll_events():
                if ev.get("type") in ("done", "closed", "error"):
                    done = True
            time.sleep(0.02)
        assert done

        stream.close()
        system.close()
    finally:
        service.shutdown()
