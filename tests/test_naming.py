"""Verify each leaf gets a hierarchical dotted `full_name` that flows into `record['vars_names']`."""

from __future__ import annotations

import pytest

from hydrogen.model import Model, Variable


class _Inner(Model):
    def declare_components(self):
        self.add_component('a', Variable(1.0))
        self.add_component('b', Variable(2.0))


class _Outer(Model):
    def declare_components(self):
        self.add_component('inner', _Inner())

    def declare_equations(self):
        # 2 vars, 2 eqs -> well-posed.
        return [
            self['inner']['a'].symbol - self['inner']['b'].symbol,
            self['inner']['a'].symbol - 1,
        ]


@pytest.fixture(scope="module")
def instantiated_outer():
    model = _Outer()
    model.instantiate(max_remove_trival_passes=0)
    return model


def test_dotted_names_include_top_class_root(instantiated_outer):
    names = list(instantiated_outer.record['vars_names'])
    assert "_Outer.inner.a" in names
    assert "_Outer.inner.b" in names


def test_full_name_set_on_leaves(instantiated_outer):
    leaves = instantiated_outer.raw_vars_references
    full_names = [v.full_name for v in leaves]
    assert "_Outer.inner.a" in full_names
    assert "_Outer.inner.b" in full_names


def test_dotted_root_uses_set_name_when_provided():
    model = _Outer()
    model.set_name("my_system")
    model.instantiate(max_remove_trival_passes=0)
    names = list(model.record['vars_names'])
    assert any(n.startswith("my_system.inner.") for n in names)
    assert all(not n.startswith("_Outer.") for n in names)
