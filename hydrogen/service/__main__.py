"""``python -m hydrogen.service`` -- start a hydrogen host process.

Launched by :func:`hydrogen.start_host` (directly for a single worker, or under
``mpirun -n N`` for several).  Rank 0 binds the socket and serves the client;
any other ranks follow it.
"""

from __future__ import annotations

from .host import main

if __name__ == "__main__":
    main()
