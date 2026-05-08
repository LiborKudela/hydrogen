"""Cache helpers shared across the package.

Exposes:
- `_CacheInfo`           : namedtuple compatible with `functools.lru_cache().cache_info()`
- `hash_array`           : stable hash of a numpy array (or any object) for cache keys
- `hash_args`            : md5 hash of an `*args` tuple, robust to numpy arrays
- `numpy_cache`          : descriptor-based, per-instance memoizing decorator that handles
                           numpy arrays and method binding (where `lru_cache` cannot)
- `ModelCache`           : tiny manual cache used by `Model` for ad-hoc memoization
"""

from __future__ import annotations

import functools
import hashlib
from collections import OrderedDict, namedtuple

import numpy as np

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
