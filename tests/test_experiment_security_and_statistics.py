from pathlib import Path

import pytest

from autoresearch.experiments.policy import ActionType, ExecutionAction, resolve_action
from autoresearch.experiments.result_analysis import paired_effects


def test_typed_action_resolves_without_shell(tmp_path: Path):
    script = tmp_path / "train.py"
    script.write_text("print('ok')")
    command = resolve_action(ExecutionAction(ActionType.RUN_PYTHON_SCRIPT, "train.py", ("--seed", "1")), tmp_path, "/usr/bin/python3")
    assert command == ("/usr/bin/python3", str(script), "--seed", "1")
    assert all("shell" not in item for item in command)


@pytest.mark.parametrize("target", ["../escape.py", "/tmp/escape.py"])
def test_typed_action_rejects_path_escape(tmp_path: Path, target: str):
    with pytest.raises(ValueError, match="unsafe_execution_request"):
        resolve_action(ExecutionAction(ActionType.RUN_PYTHON_SCRIPT, target), tmp_path, "/usr/bin/python3")


def test_typed_action_rejects_symlink_escape(tmp_path: Path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("print('outside')")
    (tmp_path / "link.py").symlink_to(outside)
    with pytest.raises(ValueError, match="unsafe_execution_request"):
        resolve_action(ExecutionAction(ActionType.RUN_PYTHON_SCRIPT, "link.py"), tmp_path, "/usr/bin/python3")


def test_single_seed_is_insufficient_evidence():
    result = paired_effects([0.8], [0.9])
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["mean_difference"] == pytest.approx(0.1)


def test_paired_effects_are_deterministic():
    result = paired_effects([1.0, 2.0, 3.0], [1.5, 2.25, 3.75])
    assert result["sample_count"] == 3
    assert result["mean_difference"] == pytest.approx(0.5)
    assert result["status"] == "ESTIMATE"
