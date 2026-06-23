"""Save / load a system as a spec (dict + JSON) -- `hydrogen.serialization`.

Two complementary demos:

  Part 1 - Round-trip a Python-built system.
      Build a fluid line in code, dump it to JSON (written to disk, stamped
      with the hydrogen version + schema), load it straight back, and prove
      the reloaded system solves to the *identical* state.  This is the
      "save my model to a file and reopen it later" workflow.

  Part 2 - Build a system from pure data (no Python class at all).
      A thermal network is described entirely as a dict -- including a
      *nested* inline composite (`"type": "Model"`) with re-exposed ports --
      then loaded and solved.  This is the "ship a system as config / let a
      GUI emit it" workflow, and it exercises validation + nesting.

Both parts self-validate (they `assert` their invariants), so this script
doubles as an end-to-end serialization test under `pytest -m examples`.

Run with `python examples/serialize_system.py` from the project root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

from hydrogen import (  # noqa: E402
    CoolPropMedium,
    Model,
    from_dict,
    from_json,
    local_results_path,
    to_dict,
    to_json,
)
from hydrogen.components.thermofluid.flow import AmbientInlet, StraightPipe  # noqa: E402

# Default HEOS backend keeps this demo fast (no BICUBIC table build).
AIR = CoolPropMedium('air', disable_warnings=True)

# Time-stepping shared by both runs in Part 1.
DT = 0.5
N_STEPS = 6


# ---------------------------------------------------------------------------
# Part 1 - round-trip a Python-built fluid line
# ---------------------------------------------------------------------------

class FluidLine(Model):
    """Hot air pushed at fixed mass flow through a short adiabatic pipe."""

    def declare_components(self):
        self.add_component('src', AmbientInlet(
            AIR, p_ambient=101325, T_ambient=273.15 + 80, m_flow=0.05, D=0.05))
        self.add_component('pipe', StraightPipe(
            AIR, D=0.05, L=2.0, epsilon=1e-4, z_in=0, z_out=0, n_segments=4))

    def declare_equations(self):
        self.connect(self['src'].ports['outlet'], self['pipe'].ports['inlet'])
        return []


def _final_state(model):
    """Map every solved variable -> its last recorded value, keyed by the
    name *without* the root prefix (the root class name differs between a
    hand-written model and a loaded `_SpecComposite`)."""
    rec = model.record
    names = rec['vars_names']
    last = np.asarray(rec['state'])[-1]
    return {n.split('.', 1)[1]: last[i] for i, n in enumerate(names)}


def _solve_line(model):
    medium = model['pipe'].medium
    model.instantiate(aditional_modules=medium.modules, max_remove_trival_passes=5)
    model.initialise(n=1)
    for _ in range(N_STEPS):
        model.solve_dae_step(DT)
        model.next_step()
    return _final_state(model)


def part1_roundtrip():
    print("=" * 70)
    print("Part 1 - round-trip a Python-built fluid line")
    print("=" * 70)

    # Build + wire the original (instantiate applies the port connections,
    # which the reflective dumper then reads back out).
    original = FluidLine()
    state_original = _solve_line(original)

    # Dump to a spec dict and to JSON text...
    spec = to_dict(original)
    text = to_json(original, indent=2)

    # ...and persist the JSON to disk (git-ignored sandbox).
    out_path = Path(local_results_path("examples", "fluid_line.json"))
    out_path.write_text(text)
    print(f"\nSpec written to {out_path}")
    print(f"  hydrogen_version : {spec['hydrogen_version']}")
    print(f"  schema_version   : {spec['schema_version']}")
    print(f"  media            : {list(spec['media'])}")
    print(f"  components        : {list(spec['components'])}")
    print(f"  connections      : {len(spec['connections'])}")
    print("\n--- fluid_line.json ---")
    print(text)

    # Load it back from disk and solve again.
    reloaded = from_json(out_path.read_text())
    state_reloaded = _solve_line(reloaded)

    # The reloaded system must reproduce the original to round-off.
    assert state_original.keys() == state_reloaded.keys(), \
        "reloaded system has a different set of variables"
    max_diff = max(
        abs(state_original[k] - state_reloaded[k]) for k in state_original
    )
    rel = max_diff / max(1.0, max(abs(v) for v in state_original.values()))
    print(f"\nReloaded vs original final state: max |diff| = {max_diff:.3e} "
          f"(rel {rel:.2e}) over {len(state_original)} variables")
    assert rel < 1e-9, "round-trip changed the solved state"

    # Dumping the loaded model echoes its canonical spec exactly.
    assert to_dict(reloaded) == spec, "to_dict(from_dict(spec)) is not stable"
    print("Round-trip is exact: solved state identical and to_dict is stable.")


# ---------------------------------------------------------------------------
# Part 2 - build a thermal network from pure data (no Python class)
# ---------------------------------------------------------------------------

def part2_from_data():
    print()
    print("=" * 70)
    print("Part 2 - build a system entirely from a dict (nested composite)")
    print("=" * 70)

    # Two FixedTemperature reservoirs (400 K / 300 K) drive heat through a
    # `stack` made of two conductors in series (G = 10 W/K each -> G_eq = 5).
    # The `stack` is an INLINE composite (`"type": "Model"`): it owns its two
    # conductors + their internal wiring and re-exposes the outer faces as
    # ports `a` / `b`.  No Python subclass is involved -- it's all data.
    spec = {
        "schema_version": 1,
        "components": {
            "hot": {"type": "hydrogen.thermofluid.FixedTemperature", "params": {"T_set": 400.0}},
            "cold": {"type": "hydrogen.thermofluid.FixedTemperature", "params": {"T_set": 300.0}},
            "stack": {
                "type": "Model",
                "components": {
                    "c1": {"type": "hydrogen.thermofluid.ThermalConductor", "params": {"G": 10.0}},
                    "c2": {"type": "hydrogen.thermofluid.ThermalConductor", "params": {"G": 10.0}},
                },
                "connections": [
                    {"from": "c1.heat_b", "to": "c2.heat_a"},
                ],
                "exposed_ports": {"a": "c1.heat_a", "b": "c2.heat_b"},
            },
        },
        "connections": [
            {"from": "hot.heat", "to": "stack.a"},
            {"from": "cold.heat", "to": "stack.b"},
        ],
    }

    print("\n--- input spec (dict) ---")
    print(json.dumps(spec, indent=2))

    model = from_dict(spec)
    model.instantiate(max_remove_trival_passes=5)
    model.initialise(n=1)
    model.solve_dae_step(1.0)
    model.next_step()

    # Read the heat flow through the first conductor: Q = G_eq * dT.
    rec = model.record
    names = list(rec['vars_names'])
    state = np.asarray(rec['state'])[-1]

    def trace(suffix):
        idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
        return state[idx]

    q = trace('.stack.c1.Q_dot_a')
    t_mid = trace('.stack.c1.T_b')
    g_eq = 1.0 / (1.0 / 10.0 + 1.0 / 10.0)  # series conductances
    q_expected = g_eq * (400.0 - 300.0)

    print(f"\nMid-stack temperature : {t_mid:.2f} K   (expected 350.00 K)")
    print(f"Series heat flow      : {q:.2f} W    (expected {q_expected:.2f} W)")
    assert abs(t_mid - 350.0) < 1e-6, "series node temperature wrong"
    assert abs(q - q_expected) < 1e-6, "series conductance heat flow wrong"
    print("Data-defined nested system built, solved, and validated.")


def main():
    part1_roundtrip()
    part2_from_data()
    print()
    print("All serialization demos passed.")


if __name__ == "__main__":
    main()
