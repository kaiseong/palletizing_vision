from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_live_view_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "live_view.py"
    spec = importlib.util.spec_from_file_location("parcel_pose_live_view_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_short_live_view_command_injects_current_nominal_calibration() -> None:
    module = _load_live_view_script()

    args = module.build_live_view_args(["--fullscreen"])

    assert args[:2] == ["live-view", "--calibration"]
    assert (
        Path(args[2])
        .as_posix()
        .endswith("/Palletizing/Codex/configs/rby1m_v1_2_fixed_table_nominal.json")
    )
    assert args[3:] == ["--fullscreen"]


def test_wrapper_default_calibration_is_tracked_and_loadable() -> None:
    from parcel_pose.calibration import load_calibration

    module = _load_live_view_script()
    calibration = load_calibration(module.DEFAULT_CALIBRATION)

    assert module.DEFAULT_CALIBRATION.is_file()
    assert calibration.T_base_from_depth is not None
    assert calibration.diagnostics["camera_profile"]["serial"] == "250122079439"


def test_explicit_calibration_replaces_wrapper_default() -> None:
    module = _load_live_view_script()

    args = module.build_live_view_args(["--calibration", "validated.json"])

    assert args == ["live-view", "--calibration", "validated.json"]


def test_live_view_main_forwards_short_user_arguments(monkeypatch) -> None:
    module = _load_live_view_script()
    received: list[list[str]] = []

    def fake_run(argv) -> int:
        received.append(list(argv))
        return 19

    monkeypatch.setattr(module, "_run_parcel_pose", fake_run)

    assert module.main(["--max-frames", "2"]) == 19
    assert received[0][0] == "live-view"
    assert received[0][-2:] == ["--max-frames", "2"]
