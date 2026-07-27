from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_record_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "record.py"
    spec = importlib.util.spec_from_file_location("parcel_pose_record_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_short_record_command_injects_default_output() -> None:
    module = _load_record_script()

    args = module.build_record_args(
        ["--session-name", "box_0", "--duration-sec", "10"]
    )

    assert args[:2] == ["record", "--output"]
    assert Path(args[2]).as_posix().endswith(
        "/Palletizing/recordings/codex_640x480"
    )
    assert args[3:] == ["--session-name", "box_0", "--duration-sec", "10"]


def test_record_main_forwards_short_user_arguments(monkeypatch) -> None:
    module = _load_record_script()
    received: list[list[str]] = []

    def fake_run(argv) -> int:
        received.append(list(argv))
        return 17

    monkeypatch.setattr(module, "_run_parcel_pose", fake_run)

    assert module.main(["--session-name", "empty_table"]) == 17
    assert received[0][0] == "record"
    assert received[0][-2:] == ["--session-name", "empty_table"]
