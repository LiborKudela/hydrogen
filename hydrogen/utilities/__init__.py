"""General-purpose utilities for hydrogen models.

Currently this package provides symbolic interpolation tables that plug into
the model layer the same way media properties do (a ``sympy.Function`` with
analytic derivatives, exposed to ``lambdify`` via a ``.modules`` list):

    from hydrogen.utilities import Interpolation1D, Interpolation2D
"""

from __future__ import annotations

from .interpolation import Interpolation1D, Interpolation2D

__all__ = [
    "Interpolation1D",
    "Interpolation2D",
]
