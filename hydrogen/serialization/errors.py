"""Error types raised by the system (de)serialization layer."""

from __future__ import annotations


class SerializationError(RuntimeError):
    """Base class for hydrogen system (de)serialization failures.

    Raised on the *dump* side (``to_dict`` / ``to_json``) when a live model
    cannot be represented in the spec format -- e.g. a constructor argument
    that is neither a JSON scalar nor stored as a recoverable attribute.
    """


class SystemSpecError(SerializationError):
    """A system spec (dict / JSON) is invalid on *load* (``from_dict`` /
    ``from_json``).

    Carries the full list of problems found so a hand-edited spec can be fixed
    in a single pass rather than one error at a time.  Access the individual
    messages via :attr:`errors`.
    """

    def __init__(self, errors):
        if isinstance(errors, str):
            errors = [errors]
        self.errors = list(errors)
        body = "\n  - ".join(self.errors)
        super().__init__(f"Invalid system spec:\n  - {body}")
