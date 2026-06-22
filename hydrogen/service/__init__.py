"""Out-of-process host for driving hydrogen systems from a separate tool/UI.

The client side (`start_host`, `HostService`, `SystemProxy`) lets another
Python process launch a hydrogen *host* -- optionally under ``mpirun -n N`` --
and interact with it over a socket without blocking::

    import hydrogen
    service = hydrogen.start_host(workers=1)
    system  = service.load_json(open("system.json").read())
    system.instantiate()
    system.initialise(n=1)
    system.run(dt=2.0, steps=40, stream=True)       # advance + record
    stream = system.vars_stream()                   # watches nothing yet
    Ta, t = stream.series("wall.T_a"), stream.time()
    while running:
        if stream.update():                         # redraw only on new rows
            chart(t.array, Ta.array)                # live handles
    service.shutdown()

The host itself runs as ``python -m hydrogen.service`` (see :mod:`.host`); it is
imported lazily so importing this subpackage from the client stays cheap.
"""

from __future__ import annotations

from .client import (
    HostConnectionError,
    HostError,
    HostService,
    Stream,
    SystemProxy,
    start_host,
)
from .protocol import PROTOCOL_VERSION, ProtocolError

__all__ = [
    "start_host",
    "HostService",
    "SystemProxy",
    "Stream",
    "HostError",
    "HostConnectionError",
    "ProtocolError",
    "PROTOCOL_VERSION",
]
