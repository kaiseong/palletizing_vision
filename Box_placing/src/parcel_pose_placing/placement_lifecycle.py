"""Runtime-bound evidence for the pallet placement lifecycle.

This module owns no SDK objects and creates no robot, camera, stream, or
command.  It sequences an already-authorized controller supplied by the
placing entrypoint and emits a phase fact only after every lower-service
acknowledgement for that phase has succeeded.  Downstream manipulation accepts
only the exact fact instance emitted by the same runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


_WHEEL_STOP_TIMEOUT_S = 2.0


class PlacementLifecycleError(RuntimeError):
    """Raised when a placement phase cannot prove its required evidence."""


@dataclass(frozen=True, slots=True)
class PlaceAlignmentStoppedAndReleased:
    """Proof that base alignment stopped and released its ownership."""

    place_alignment_stopped_and_released: Literal[True] = field(
        default=True,
        init=False,
    )
    exact_zero_latched: Literal[True] = field(default=True, init=False)
    exact_zero_acknowledged: Literal[True] = field(default=True, init=False)
    measured_wheel_stop: Literal[True] = field(default=True, init=False)
    alignment_released: Literal[True] = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class PlaceAcknowledgedAndReleased:
    """Proof that place was acknowledged and release was authorized."""

    place_acknowledged_and_released: Literal[True] = field(
        default=True,
        init=False,
    )
    place_command_acknowledged: Literal[True] = field(default=True, init=False)
    release_authorized: Literal[True] = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class RetreatCompleted:
    """Proof that the demonstrated retreat command was acknowledged."""

    retreat_completed: Literal[True] = field(default=True, init=False)
    retreat_command_acknowledged: Literal[True] = field(default=True, init=False)


class PlacementLifecycleRuntime:
    """Sequence one authorized place/retreat lifecycle over an owned controller."""

    def __init__(
        self,
        *,
        controller: Any,
        release_alignment: Callable[[], Any],
        prepare: Callable[[], Any] | None = None,
    ) -> None:
        if controller is None:
            raise TypeError("controller is required")
        if not callable(release_alignment):
            raise TypeError("release_alignment must be callable")
        if prepare is not None and not callable(prepare):
            raise TypeError("prepare must be callable or None")
        self._controller = controller
        self._release_alignment = release_alignment
        self._prepare = prepare
        self._alignment_evidence: PlaceAlignmentStoppedAndReleased | None = None
        self._place_evidence: PlaceAcknowledgedAndReleased | None = None
        self._retreat_evidence: RetreatCompleted | None = None
        self._start_attempted = False
        self._started = False
        self._place_attempted = False
        self._retreat_attempted = False
        self._closing = False
        self._closed = False

    def start(self) -> None:
        """Run the optional ready/stream preparation at most once.

        ``start`` is intentionally not a prerequisite for the evidence methods:
        existing lower-service callers already own preparation and may begin at
        alignment handoff.  New staged entrypoints can inject that preparation
        here without duplicating it on repeated calls.
        """

        self._require_open()
        if self._started:
            return
        if self._start_attempted:
            raise PlacementLifecycleError(
                "placement lifecycle prepare was already attempted"
            )
        self._start_attempted = True
        if self._prepare is not None:
            try:
                result = self._prepare()
            except Exception as exc:
                raise PlacementLifecycleError(
                    f"placement lifecycle prepare failed: {exc}"
                ) from exc
            if result is False:
                raise PlacementLifecycleError(
                    "placement lifecycle prepare reported failure"
                )
        self._started = True

    def stop_alignment_for_place(self) -> PlaceAlignmentStoppedAndReleased:
        """Latch exact zero, prove stop, release alignment, and emit its fact."""

        self._require_open()
        if self._alignment_evidence is not None:
            return self._alignment_evidence
        if self._place_attempted:
            raise PlacementLifecycleError("place manipulation has already been attempted")

        try:
            self._controller.send_zero_mobility_hold(latch=True)
        except Exception as exc:
            raise PlacementLifecycleError(
                f"cannot latch exact-zero mobility hold: {exc}"
            ) from exc

        try:
            zero_telemetry = self._controller.placement_telemetry()
        except Exception as exc:
            raise PlacementLifecycleError(
                f"cannot verify exact-zero acknowledgement: {exc}"
            ) from exc
        if not bool(getattr(zero_telemetry, "zero_latched", False)):
            raise PlacementLifecycleError("exact-zero mobility hold was not latched")
        if not bool(getattr(zero_telemetry, "target_acknowledged", False)):
            raise PlacementLifecycleError("exact-zero command was not acknowledged")

        try:
            wheel_stop = self._controller.wait_for_wheel_stop(
                timeout_s=_WHEEL_STOP_TIMEOUT_S
            )
        except Exception as exc:
            raise PlacementLifecycleError(
                f"cannot prove measured wheel stop: {exc}"
            ) from exc
        if not bool(getattr(wheel_stop, "stopped", False)):
            raise PlacementLifecycleError("measured wheel stop was not confirmed")

        try:
            release_result = self._release_alignment()
        except Exception as exc:
            raise PlacementLifecycleError(
                f"cannot release stopped placement alignment: {exc}"
            ) from exc
        if release_result is False:
            raise PlacementLifecycleError(
                "stopped placement alignment release reported failure"
            )

        evidence = PlaceAlignmentStoppedAndReleased()
        self._alignment_evidence = evidence
        return evidence

    def execute_place(
        self,
        evidence: PlaceAlignmentStoppedAndReleased,
        *,
        descent_plan: Any,
        await_release_authorization: Callable[[], bool],
    ) -> PlaceAcknowledgedAndReleased:
        """Execute place once after this runtime's alignment handoff evidence."""

        self._require_open()
        if not isinstance(evidence, PlaceAlignmentStoppedAndReleased):
            raise PlacementLifecycleError(
                "place requires PlaceAlignmentStoppedAndReleased evidence"
            )
        if evidence is not self._alignment_evidence:
            raise PlacementLifecycleError(
                "place alignment evidence was not emitted by this runtime"
            )
        if self._place_attempted:
            raise PlacementLifecycleError("place manipulation was already attempted")
        if not callable(await_release_authorization):
            raise TypeError("await_release_authorization must be callable")

        # From this point onward a physical manipulation may have started.  A
        # retry could duplicate an uncertain one-shot command, so every failure
        # below remains terminal for this runtime.
        self._place_attempted = True
        try:
            self._controller.start_cartesian_lowering_hold(
                descent_plan=descent_plan
            )
        except Exception as exc:
            raise PlacementLifecycleError(f"place command failed: {exc}") from exc

        try:
            place_telemetry = self._controller.placement_telemetry()
        except Exception as exc:
            raise PlacementLifecycleError(
                f"cannot verify place acknowledgement: {exc}"
            ) from exc
        if not bool(getattr(place_telemetry, "target_acknowledged", False)):
            raise PlacementLifecycleError("place command was not acknowledged")

        try:
            release_authorized = bool(await_release_authorization())
        except Exception as exc:
            raise PlacementLifecycleError(
                f"place release authorization failed: {exc}"
            ) from exc
        if not release_authorized:
            raise PlacementLifecycleError("place release was not authorized")

        placed = PlaceAcknowledgedAndReleased()
        self._place_evidence = placed
        return placed

    def execute_retreat(
        self,
        evidence: PlaceAcknowledgedAndReleased,
    ) -> RetreatCompleted:
        """Execute the demonstrated retreat once after successful place evidence."""

        self._require_open()
        if not isinstance(evidence, PlaceAcknowledgedAndReleased):
            raise PlacementLifecycleError(
                "retreat requires PlaceAcknowledgedAndReleased evidence"
            )
        if evidence is not self._place_evidence:
            raise PlacementLifecycleError(
                "place evidence was not emitted by this runtime"
            )
        if self._retreat_evidence is not None:
            return self._retreat_evidence
        if self._retreat_attempted:
            raise PlacementLifecycleError("retreat manipulation was already attempted")

        self._retreat_attempted = True
        try:
            self._controller.start_cartesian_release_hold()
        except Exception as exc:
            raise PlacementLifecycleError(f"retreat command failed: {exc}") from exc

        try:
            retreat_telemetry = self._controller.placement_telemetry()
        except Exception as exc:
            raise PlacementLifecycleError(
                f"cannot verify retreat acknowledgement: {exc}"
            ) from exc
        if not bool(getattr(retreat_telemetry, "target_acknowledged", False)):
            raise PlacementLifecycleError("retreat command was not acknowledged")

        completed = RetreatCompleted()
        self._retreat_evidence = completed
        return completed

    def close(self) -> None:
        """Close the controller once; retry only an unresolved close failure."""

        if self._closed:
            return
        self._closing = True
        try:
            result = self._controller.close()
        except Exception as exc:
            raise PlacementLifecycleError(
                f"placement lifecycle close failed: {exc}"
            ) from exc
        if result is False:
            raise PlacementLifecycleError(
                "placement lifecycle controller close reported failure"
            )
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise PlacementLifecycleError("placement lifecycle is already closed")
        if self._closing:
            raise PlacementLifecycleError(
                "placement lifecycle close is pending after a cleanup attempt"
            )


__all__ = [
    "PlaceAcknowledgedAndReleased",
    "PlaceAlignmentStoppedAndReleased",
    "PlacementLifecycleError",
    "PlacementLifecycleRuntime",
    "RetreatCompleted",
]
