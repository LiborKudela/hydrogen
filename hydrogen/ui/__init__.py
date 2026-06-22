"""Qt authoring UI for hydrogen: a drag-and-drop component palette -> canvas,
with per-node property editing and a "simulate via the hydrogen host" action.

Structured by functionality:

  * :mod:`~hydrogen.ui.qt`         -- Qt binding shim (PySide6 / PyQt5).
  * :mod:`~hydrogen.ui.style`      -- canvas colours + port-side conventions.
  * :mod:`~hydrogen.ui.forms`      -- ``component_spec``-driven parameter forms.
  * :mod:`~hydrogen.ui.introspect` -- read a component's ports by building it.
  * :mod:`~hydrogen.ui.catalog`    -- the catalogue tree (drag source).
  * :mod:`~hydrogen.ui.items`      -- canvas scene items (ports/nodes/wires).
  * :mod:`~hydrogen.ui.canvas`     -- the canvas view + interaction loop.
  * :mod:`~hydrogen.ui.properties` -- per-node parameter dialog.
  * :mod:`~hydrogen.ui.simulate`   -- run options + filterable host log dialog.
  * :mod:`~hydrogen.ui.project`    -- the editor's on-disk project format.
  * :mod:`~hydrogen.ui.app`        -- the main window glue + :func:`main`.

This subpackage needs a Qt binding (``pip install "hydrogen[qt]"``) and is *not*
imported by the top-level :mod:`hydrogen` package, so the core stays Qt-free.

Launch::

    python3 -m hydrogen.ui
"""

from __future__ import annotations

from .app import MainWindow, main

#: Alias so ``hydrogen.ui.run()`` reads naturally.
run = main

__all__ = ["MainWindow", "main", "run"]
