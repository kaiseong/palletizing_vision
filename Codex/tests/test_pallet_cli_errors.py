"""Control faults must not be reported as CLI usage errors."""

from __future__ import annotations

import pytest

from parcel_pose import pallet_cli
from parcel_pose.pallet_control import CombinedStreamError


ARGV = ["replay", "--session", "nonexistent-session", "--no-default-artifacts"]


def test_control_fault_exits_with_its_own_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def explode(*args, **kwargs):
        raise CombinedStreamError("This command stream is expired")

    monkeypatch.setattr(
        "parcel_pose.pallet_evaluation.evaluate_pallet_session", explode
    )
    assert pallet_cli.main(ARGV) == 3
    captured = capsys.readouterr()
    assert "pallet control fault: CombinedStreamError" in captured.err
    # A usage error would print the argparse banner instead of a traceback.
    assert "usage: pallet" not in captured.err
    assert "Traceback" in captured.err


def test_plain_value_errors_remain_usage_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args, **kwargs):
        raise ValueError("bad session layout")

    monkeypatch.setattr(
        "parcel_pose.pallet_evaluation.evaluate_pallet_session", explode
    )
    with pytest.raises(SystemExit) as excinfo:
        pallet_cli.main(ARGV)
    assert excinfo.value.code == 2


def test_keyboard_interrupt_keeps_its_conventional_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "parcel_pose.pallet_evaluation.evaluate_pallet_session", explode
    )
    assert pallet_cli.main(ARGV) == 130
