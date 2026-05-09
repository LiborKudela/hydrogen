"""Cache helpers shared across the package.

Exposes:
- `_CacheInfo`           : namedtuple compatible with `functools.lru_cache().cache_info()`
- `hash_array`           : stable hash of a numpy array (or any object) for cache keys
- `hash_args`            : md5 hash of an `*args` tuple, robust to numpy arrays
- `numpy_cache`          : descriptor-based, per-instance memoizing decorator that handles
                           numpy arrays and method binding (where `lru_cache` cannot)
- `ModelCache`           : tiny manual cache used by `Model` for ad-hoc memoization
- `lambda_cache_default_dir` / `save_lambdified_source` / `load_lambdified_source` :
                           on-disk cache for sympy-lambdified residual/Jacobian source
                           code, used by `Model.instantiate` to skip the multi-second
                           lambdify pass when the same geometry+medium has already
                           been compiled.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
from collections import OrderedDict, namedtuple
from pathlib import Path

import numpy as np
import sympy as _sp

_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


def hash_array(arr):
    """Hash that works for both numpy arrays and arbitrary hashable objects."""
    return hash(arr.tobytes()) if isinstance(arr, np.ndarray) else hash(arr)


def hash_args(*args):
    """Stable md5 hash of a positional-arg tuple, treating numpy arrays bytewise."""
    m = hashlib.md5()
    for arg in args:
        if isinstance(arg, np.ndarray):
            m.update(arg.tobytes())
        else:
            m.update(str(arg).encode())
    return m.hexdigest()


def numpy_cache(maxsize=128, include_data=None):
    """Per-instance method memoizer that hashes numpy-array arguments.

    Unlike `functools.lru_cache`, this works on instance methods (each instance gets
    its own cache, keyed by `id(self)`) and on numpy-array arguments (hashed bytewise
    via `hash_args`). Eviction is FIFO when `maxsize` is exceeded.
    """

    def decorator(func):
        # Use a descriptor to handle method binding
        class CacheDescriptor:
            def __init__(self):
                self.caches = {}  # Instance-specific caches
                self.include_data = include_data

            def __get__(self, obj, objtype=None):
                if obj is None:
                    return self
                # Get or create cache for this instance
                if id(obj) not in self.caches:
                    self.caches[id(obj)] = {
                        'cache': {},
                        'hits': 0,
                        'misses': 0,
                    }
                cache_state = self.caches[id(obj)]

                @functools.wraps(func)
                def wrapper(*args):
                    key = hash_args(*args)
                    cache = cache_state['cache']
                    if key not in cache:
                        cache[key] = func(obj, *args)  # Pass self (obj) to func
                        cache_state['misses'] += 1
                        # Handle maxsize (simple FIFO eviction)
                        if len(cache) > maxsize:
                            cache.pop(next(iter(cache)))
                    else:
                        cache_state['hits'] += 1
                    return cache[key]

                def cache_info():
                    return _CacheInfo(
                        cache_state['hits'],
                        cache_state['misses'],
                        maxsize,
                        len(cache_state['cache']),
                    )

                wrapper.cache_info = cache_info
                return wrapper

        return CacheDescriptor()

    return decorator


class ModelCache:
    """Light manual cache with hits/misses bookkeeping."""

    def __init__(self, name):
        self.name = name
        self.hits = 0
        self.misses = 0
        self.calls = 0
        self.cache = OrderedDict()

    def add_hit(self):
        self.hits += 1
        self.calls += 1

    def add_miss(self):
        self.misses += 1
        self.calls += 1

    def load_from_cache(self, key):
        if key in self.cache:
            self.add_hit()
            return self.cache[key]
        self.add_miss()
        return None

    def save_to_cache(self, key, value):
        self.cache[key] = value

    @property
    def cache_efficiency(self):
        return self.hits / (self.hits + self.misses) * 100 if (self.hits + self.misses) > 0 else 0

    def cache_info(self):
        return _CacheInfo(self.hits, self.misses, getattr(self, "maxsize", None), self.calls)

    def __repr__(self, title=None):
        return (
            f"{title}: ({self.calls} calls, {self.hits} hits, {self.misses} misses - "
            f"{self.cache_efficiency:.1f}% cache efficiency)"
        )


# --- lambdified-source cache --------------------------------------------------
# This is a thin disk cache around the source string produced by `sp.lambdify`.
# The expensive part of lambdify on big systems is *generating* and *parsing*
# the source (CSE analysis + Python parsing).  Once the source is on disk we
# can skip the generation step entirely on a cache hit.

def lambda_cache_default_dir() -> Path | None:
    """Returns the cache directory or `None` if caching is disabled.

    Set `HYDROGEN_LAMBDA_CACHE` to an empty string or `0` to disable.
    Set it to a path to override the location.
    """
    env = os.environ.get("HYDROGEN_LAMBDA_CACHE")
    if env == "" or env == "0":
        return None
    if env:
        return Path(env)
    return Path.home() / ".cache" / "hydrogen"


def lambda_cache_key(args, expr, modules_signature: list[str], cse: bool) -> str:
    """Stable hash key for `(args, expr, modules, cse)` of a future lambdify call.

    Uses `pickle.dumps` (which has a C implementation) to serialise sympy
    objects rather than `srepr` -- on big systems pickle is roughly 50x
    faster, which matters because this function runs in the cache-miss path
    too and we don't want to inflate cold-start time.
    """
    import pickle
    h = hashlib.sha256()
    h.update(_sp.__version__.encode())
    h.update(b"|cse=")
    h.update(b"1" if cse else b"0")
    h.update(b"|args=")
    h.update(pickle.dumps(_pickle_safe(args), protocol=4))
    h.update(b"|modules=")
    for m in sorted(modules_signature):
        h.update(m.encode())
        h.update(b",")
    h.update(b"|expr=")
    h.update(pickle.dumps(_pickle_safe(expr), protocol=4))
    return h.hexdigest()[:32]


def _pickle_safe(obj):
    """Pickle works on sympy `Matrix`/`Symbol`/`Expr` directly, but for the
    custom `Symbolic_property` subclasses (per-medium dynamic Function classes)
    pickle fails with `PicklingError` because the class name is generated at
    import time and isn't reachable by qualified name.  Substitute a stable
    placeholder for those nodes -- their identity is captured by their class
    name (`Air_rho_ph`, `Hydrogen_h_pT`, ...) which is included via the
    modules signature in `lambda_cache_key`.
    """
    if isinstance(obj, _sp.Matrix):
        return ("MAT", obj.shape, [_pickle_safe(x) for x in obj])
    if isinstance(obj, (list, tuple)):
        return [_pickle_safe(x) for x in obj]
    if isinstance(obj, _sp.Function):
        # encode as (class_name, args)
        return (type(obj).__name__, [_pickle_safe(a) for a in obj.args])
    if isinstance(obj, _sp.Basic):
        # General sympy node -- recurse on args, tag with class name.
        if obj.is_Symbol or obj.is_Number:
            return str(obj)
        return (type(obj).__name__, [_pickle_safe(a) for a in obj.args])
    return obj


def save_lambdified_source(cache_dir: Path, key: str, func, modules_signature: list[str]):
    """Persist the source of a sympy-lambdified function to disk."""
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return  # source not retrievable -- silently skip
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "modules_signature": list(modules_signature),
        "func_name": func.__name__,
        "source": source,
    }
    (cache_dir / f"{key}.json").write_text(json.dumps(payload))


def load_lambdified_source(cache_dir: Path, key: str, namespace: dict):
    """Re-exec a previously-saved lambdified function inside `namespace`.

    Returns the resulting callable, or `None` on cache miss.
    """
    path = cache_dir / f"{key}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    source = payload["source"]
    func_name = payload["func_name"]
    try:
        exec(compile(source, str(path), "exec"), namespace)
    except Exception:
        return None
    return namespace.get(func_name)
