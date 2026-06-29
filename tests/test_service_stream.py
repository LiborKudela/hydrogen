"""On-demand variable streams over a live host run (`SystemProxy.vars_stream`).

Spawns a real host (``workers=1``) running the CoolProp-free signal system from
``tutorials/host_client/signal_dynamics.json`` and checks the stream contract:

* a stream watches nothing until ``series`` / ``series_values`` register a name,
  which expands a suffix to every match and backfills the full history;
* the live handles (``time`` / ``series`` / ``series_values``) are refreshed
  together by ``update`` (which also flags new rows) and always share one length;
* streams are independent and can be closed, after which a ``stream_closed`` is
  delivered and no further chunks arrive.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import hydrogen

_SPEC = (
    Path(__file__).resolve().parent.parent
    / "tutorials" / "host_client" / "signal_dynamics.json"
).read_text()

# Full recorded names of the signal system (root composite is `_SpecComposite`).
Y_NAMES = {
    "_SpecComposite.src.y",
    "_SpecComposite.lag.der_y",
    "_SpecComposite.lag.y",
}


def _load_running(service, *, steps=300, delay=0.01):
    system = service.load_json(_SPEC)
    system.instantiate(max_remove_trival_passes=2)
    system.poll_events()  # drop compile logs
    system.initialise(n=1)
    system.run(dt=0.02, steps=steps, stream=True, every=5, delay=delay)
    return system


def _drain_to_end(system, stream, *, min_rows=20, timeout=8.0):
    """Pump update() until the run is done and enough rows have accumulated."""
    t = stream.time()  # handle to measure accumulated rows
    deadline = time.monotonic() + timeout
    done = False
    while time.monotonic() < deadline:
        stream.update()
        if any(ev.get("type") == "done" for ev in system.poll_events()):
            done = True
        if done and len(t) > min_rows:
            return len(t)
        time.sleep(0.02)
    stream.update()
    return len(t)


def test_vars_stream_expand_and_handles():
    """A suffix registered on demand expands to every match, and the live
    handles stay one consistent length (no manual clipping)."""
    service = hydrogen.start_host(workers=1)
    try:
        system = _load_running(service)

        # Watches nothing until we ask.
        stream = system.vars_stream()
        assert stream.list_watched_names() == []

        # Bare suffix "y" matches every *.y / *_y name (src.y, lag.der_y, lag.y).
        all_y = stream.series_values("y")     # one column per match
        one_y = stream.series("lag.y")         # single (dotted-suffix) match
        t = stream.time()

        # Registration reports the full names of every match.
        assert set(stream.vars) == Y_NAMES
        assert set(stream.list_watched_names()) == Y_NAMES

        # update() refreshes all handles and flags fresh rows.
        saw_new = False
        deadline = time.monotonic() + 8.0
        done = False
        while time.monotonic() < deadline:
            if stream.update():
                saw_new = True
            if any(ev.get("type") == "done" for ev in system.poll_events()):
                done = True
            if done and len(t) > 20:
                break
            time.sleep(0.02)
        assert saw_new

        assert t.array.ndim == 1 and len(t) > 20
        assert all_y.array.shape == (len(t), 3)    # rows = time, cols = matches
        assert one_y.array.shape == (len(t),)
        # All handles share one length -- the whole point of update().
        assert len(all_y) == len(t) == len(one_y)
        # series("lag.y") must equal the lag column of series_values("y").
        lag_col = stream.vars.index("_SpecComposite.lag.y")
        assert np.allclose(one_y.array, all_y.array[:, lag_col])

        stream.close()
        system.close()
    finally:
        service.shutdown()


def test_vars_stream_dynamic_watch_dedups_and_backfills():
    """Registering a narrower pattern that is already covered adds nothing new,
    and every stream backfills the full history from t=0."""
    service = hydrogen.start_host(workers=1)
    try:
        system = _load_running(service)
        # Let some history accumulate before the stream opens, to exercise the
        # backfill (the stream still starts from row 0).
        time.sleep(0.4)

        stream = system.vars_stream()
        all_y = stream.series_values("y")
        t = stream.time()

        _drain_to_end(system, stream)

        watched = stream.list_watched_names()
        assert set(watched) == Y_NAMES
        assert all_y.array.shape == (len(t), 3)
        assert t.array[0] == 0.0          # backfilled from the very first row

        # A narrower, already-covered pattern adds nothing new.
        one = stream.series("lag.y")
        stream.update()
        assert one.array.shape == (len(t),)
        assert set(stream.list_watched_names()) == set(watched)
        lag_col = stream.vars.index("_SpecComposite.lag.y")
        assert np.allclose(one.array, all_y.array[:, lag_col])

        stream.close()
        system.close()
    finally:
        service.shutdown()


def test_vars_stream_independence_and_close():
    """Two streams are independent; closing one delivers ``stream_closed`` and
    leaves the other producing data."""
    service = hydrogen.start_host(workers=1)
    try:
        system = _load_running(service, steps=600, delay=0.02)
        time.sleep(0.4)

        s1 = system.vars_stream()
        s2 = system.vars_stream()
        assert s1.stream_id != s2.stream_id

        a1, t1 = s1.series("src.y"), s1.time()
        a2, t2 = s2.series("src.y"), s2.time()

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            s1.update()
            s2.update()
            if len(t1) > 25 and len(t2) > 25:
                break
            time.sleep(0.02)

        # Both backfill the full history, so both start at the first recorded row
        # and their value columns track their own time vector.
        assert t1.array[0] == t2.array[0] == 0.0
        assert len(a1) == len(t1) and len(a2) == len(t2)

        # Closing s2 delivers a stream_closed and stops its chunks.
        s2.close()
        closed_seen = False
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not closed_seen:
            for ch in s2.poll():
                if ch.get("type") == "stream_closed":
                    closed_seen = True
            time.sleep(0.02)
        assert closed_seen, "no stream_closed after Stream.close()"

        # The surviving stream keeps producing data.
        before = len(t1)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            s1.update()
            if len(t1) > before:
                break
            time.sleep(0.02)
        assert len(t1) > before, "surviving stream stopped producing data"

        s1.close()
        system.stop()
        system.close()
    finally:
        service.shutdown()
