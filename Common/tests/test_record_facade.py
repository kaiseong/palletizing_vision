from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FACADE = REPO_ROOT / "Common" / "record.py"


def _load_record_facade():
    spec = importlib.util.spec_from_file_location("common_record_facade_under_test", FACADE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_record_facade_uses_box_picking_config_by_default(monkeypatch) -> None:
    module = _load_record_facade()
    calls: list[tuple[list[str], Path]] = []

    def fake_record_main(argv, *, default_config):
        calls.append((list(argv), default_config))
        return 0

    monkeypatch.setattr("parcel_pose_common.record_cli.main", fake_record_main)

    assert module.main(["--duration-sec", "0"]) == 0

    assert calls == [
        (
            [
                "--output",
                str(REPO_ROOT / "recordings" / "codex_640x480"),
                "--duration-sec",
                "0",
            ],
            REPO_ROOT / "Box_picking" / "configs" / "d435_rby1_nominal.json",
        )
    ]
