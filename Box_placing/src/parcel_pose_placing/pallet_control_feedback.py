"""Pure RB-Y1 combined-stream feedback parsing.

The controller owns stream lifetime and policy.  This module only turns the
SDK's nested feedback tree into a small immutable acknowledgement and interprets
terminal status codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ComponentFeedbackAck:
    root: bool
    component: bool
    mobility: bool
    torso: bool
    head: bool
    right_arm: bool
    left_arm: bool
    status_code: int | None
    finish_code: int | None

    @property
    def all_components(self) -> bool:
        return all(
            (
                self.root,
                self.component,
                self.mobility,
                self.torso,
                self.head,
                self.right_arm,
                self.left_arm,
            )
        )

    @property
    def running(self) -> bool:
        return self.all_components and self.status_code == 2 and self.finish_code == 0


def _wire_enum_code(value: Any) -> int:
    return int(getattr(value, "value", value))


def _node_is_valid(node: Any) -> bool:
    return node is not None and bool(getattr(node, "valid", False))


def parse_component_feedback(feedback: Any) -> ComponentFeedbackAck:
    """Parse one SDK feedback tree without controller or SDK dependencies."""

    if not _node_is_valid(feedback):
        return ComponentFeedbackAck(
            root=False,
            component=False,
            mobility=False,
            torso=False,
            head=False,
            right_arm=False,
            left_arm=False,
            status_code=None,
            finish_code=None,
        )
    try:
        status = _wire_enum_code(feedback.status)
        finish = _wire_enum_code(feedback.finish_code)
    except (AttributeError, TypeError, ValueError):
        status = None
        finish = None

    component = getattr(feedback, "component_based_command", None)
    mobility = getattr(component, "mobility_command", None)
    se2 = getattr(mobility, "se2_velocity_command", None)
    body = getattr(component, "body_command", None)
    body_components = getattr(body, "body_component_based_command", None)
    torso = getattr(body_components, "torso_command", None)
    torso_position = getattr(torso, "joint_position_command", None)
    right = getattr(body_components, "right_arm_command", None)
    right_impedance = getattr(right, "joint_impedance_control_command", None)
    right_cartesian = getattr(
        right,
        "cartesian_impedance_control_command",
        None,
    )
    left = getattr(body_components, "left_arm_command", None)
    left_impedance = getattr(left, "joint_impedance_control_command", None)
    left_cartesian = getattr(
        left,
        "cartesian_impedance_control_command",
        None,
    )
    head = getattr(component, "head_command", None)
    head_position = getattr(head, "joint_position_command", None)
    right_command_valid = _node_is_valid(right_impedance) or _node_is_valid(
        right_cartesian
    )
    left_command_valid = _node_is_valid(left_impedance) or _node_is_valid(
        left_cartesian
    )

    return ComponentFeedbackAck(
        root=True,
        component=_node_is_valid(component),
        mobility=all(_node_is_valid(node) for node in (mobility, se2)),
        torso=all(
            _node_is_valid(node)
            for node in (body, body_components, torso, torso_position)
        ),
        head=all(_node_is_valid(node) for node in (head, head_position)),
        right_arm=all(_node_is_valid(node) for node in (body, body_components, right))
        and right_command_valid,
        left_arm=all(_node_is_valid(node) for node in (body, body_components, left))
        and left_command_valid,
        status_code=status,
        finish_code=finish,
    )


def raise_for_terminal_feedback(
    feedback: ComponentFeedbackAck,
    operation: str,
    *,
    error_type: type[Exception] = RuntimeError,
) -> None:
    """Raise the caller's domain error for non-success terminal feedback."""

    if feedback.status_code == 3:
        if feedback.finish_code == 1:
            return
        raise error_type(f"{operation} terminated: finish_code={feedback.finish_code}")
    if feedback.status_code == 0:
        raise error_type(f"{operation} was not activated")
    if feedback.status_code not in (1, 2, 3):
        raise error_type(
            f"{operation} returned unexpected status={feedback.status_code}"
        )


__all__ = [
    "ComponentFeedbackAck",
    "parse_component_feedback",
    "raise_for_terminal_feedback",
]
