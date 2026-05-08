"""Unit tests for `hydrogen.caching`."""

from __future__ import annotations

import numpy as np

from hydrogen.caching import ModelCache, hash_args, hash_array, numpy_cache


def test_hash_array_stable_across_copies():
    a = np.array([1.0, 2.0, 3.0])
    b = a.copy()
    assert hash_array(a) == hash_array(b)


def test_hash_array_distinguishes_payload():
    a = np.array([1.0, 2.0, 3.0])
    c = np.array([1.0, 2.0, 4.0])
    assert hash_array(a) != hash_array(c)


def test_hash_args_combines_mixed_types():
    a = np.array([1.0, 2.0])
    h1 = hash_args(a, "key", 7)
    h2 = hash_args(a.copy(), "key", 7)
    h_diff = hash_args(a, "key", 8)
    assert h1 == h2
    assert h1 != h_diff


def test_numpy_cache_hits_and_misses():
    class Dummy:
        @numpy_cache(maxsize=4)
        def square(self, x):
            return x * x

    d = Dummy()
    assert d.square(3) == 9
    assert d.square(3) == 9  # cache hit
    info = d.square.cache_info()
    assert info.hits == 1
    assert info.misses == 1
    assert info.currsize == 1


def test_numpy_cache_is_per_instance():
    class Dummy:
        @numpy_cache(maxsize=4)
        def f(self, x):
            return x

    a = Dummy()
    b = Dummy()
    a.f(1)
    b.f(1)
    # Each instance has its own cache; both calls are misses.
    assert a.f.cache_info().misses == 1
    assert b.f.cache_info().misses == 1
    # And separate cache contents.
    assert a.f.cache_info().currsize == 1
    assert b.f.cache_info().currsize == 1


def test_numpy_cache_fifo_eviction():
    class Dummy:
        @numpy_cache(maxsize=2)
        def f(self, x):
            return x

    d = Dummy()
    d.f(1)
    d.f(2)
    d.f(3)  # evicts 1
    d.f(1)  # miss again
    info = d.f.cache_info()
    assert info.misses == 4
    assert info.hits == 0


def test_numpy_cache_handles_array_argument():
    class Dummy:
        @numpy_cache(maxsize=2)
        def norm_sq(self, arr):
            return float(np.dot(arr, arr))

    d = Dummy()
    a = np.array([3.0, 4.0])
    assert d.norm_sq(a) == 25.0
    assert d.norm_sq(a.copy()) == 25.0  # bytewise equal -> cache hit
    info = d.norm_sq.cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_model_cache_efficiency():
    cache = ModelCache("test")
    assert cache.cache_efficiency == 0
    cache.add_miss()
    cache.add_miss()
    cache.add_hit()
    assert abs(cache.cache_efficiency - 100 / 3) < 1e-9
    assert cache.calls == 3
