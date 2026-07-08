"""Run checkpoint: resume/continue only while the canvas model is unchanged."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from hydrogen.ui.session import SimulationSession, structural_param_names

_SPEC = json.loads(
    (Path(__file__).resolve().parent.parent
     / "tutorials" / "host_client" / "signal_dynamics.json").read_text()
)
_INST = {"max_remove_trival_passes": 2}


def test_pipe_geometry_params_are_structural():
    """L/D/elevation bake derived segment geometry at compile time."""
    sp = structural_param_names("hydrogen.thermofluid.Pipe")
    assert {"L", "D", "z_in", "z_out"}.issubset(sp)


def test_steering_resume_while_checkpoint_matches():
    session = SimulationSession()
    session.set_run_checkpoint(_SPEC, _INST)
    session._run_phase = "paused"
    assert session.can_steering_resume(_SPEC, _INST)


def test_steering_resume_false_after_pure_param_change():
    session = SimulationSession()
    session.set_run_checkpoint(_SPEC, _INST)
    session._run_phase = "paused"
    changed = copy.deepcopy(_SPEC)
    changed["components"]["src"]["params"]["amplitude"] = 2.0
    assert not session.can_steering_resume(changed, _INST)


def test_mark_model_stale_blocks_resume_even_if_params_reverted():
    session = SimulationSession()
    session.set_run_checkpoint(_SPEC, _INST)
    session._run_phase = "paused"
    session.mark_model_stale()
    assert session._run_stale
    assert not session.can_steering_resume(_SPEC, _INST)


def test_mark_model_stale_is_no_op_before_any_run():
    session = SimulationSession()
    session.mark_model_stale()
    assert not session._run_stale
    assert session._run_checkpoint is None
