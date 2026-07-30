"""A recording stand-in for ``rby1_sdk`` so the whole pallet motion path runs offline.

The fakes are deliberately dumb: every builder records what the controller asked
for and returns itself, so a test can read back the exact commanded motion of
each packet.  Nothing here models robot dynamics; the arms are assumed to hold
whatever pose the test injects, which is what a compliant Cartesian impedance
hold does while a carton resists the squeeze.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any

import numpy as np


JOINT_NAMES = (
    tuple(f"mobility_{i}" for i in range(4))
    + tuple(f"torso_{i}" for i in range(6))
    + tuple(f"right_arm_{i}" for i in range(7))
    + tuple(f"left_arm_{i}" for i in range(7))
    + tuple(f"head_{i}" for i in range(2))
)
MOBILITY_IDX = list(range(0, 4))
TORSO_IDX = list(range(4, 10))
RIGHT_ARM_IDX = list(range(10, 17))
LEFT_ARM_IDX = list(range(17, 24))
HEAD_IDX = list(range(24, 26))


# --------------------------------------------------------------------------- #
# recorded command packets
# --------------------------------------------------------------------------- #
@dataclass
class ArmTargetRecord:
    reference_link: str
    link: str
    transform: np.ndarray
    linear_velocity_limit: float
    angular_velocity_limit: float
    linear_acceleration_limit: float
    angular_acceleration_limit: float


@dataclass
class Packet:
    """One combined command as the controller handed it to the SDK."""

    torso_position: np.ndarray | None = None
    head_position: np.ndarray | None = None
    minimum_time_s: float | None = None
    control_hold_time_s: float | None = None
    right_arm: ArmTargetRecord | None = None
    left_arm: ArmTargetRecord | None = None
    right_arm_joint_position: np.ndarray | None = None
    left_arm_joint_position: np.ndarray | None = None
    mobility_velocity: tuple[float, float, float] | None = None
    joint_stiffness: np.ndarray | None = None
    nullspace_right: np.ndarray | None = None
    nullspace_left: np.ndarray | None = None

    @property
    def arm_mode(self) -> str:
        if self.right_arm is not None:
            return "CARTESIAN"
        if self.right_arm_joint_position is not None:
            return "JOINT"
        return "NONE"

    def eef_separation_m(self) -> float | None:
        if self.right_arm is None or self.left_arm is None:
            return None
        return float(
            np.linalg.norm(
                self.right_arm.transform[:3, 3] - self.left_arm.transform[:3, 3]
            )
        )


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
class _Header:
    def __init__(self) -> None:
        self.hold_time_s: float | None = None

    def set_control_hold_time(self, value: float) -> "_Header":
        self.hold_time_s = float(value)
        return self


class _JointPosition:
    def __init__(self) -> None:
        self.header: _Header | None = None
        self.minimum_time_s: float | None = None
        self.position: np.ndarray | None = None
        self.stiffness: np.ndarray | None = None

    def set_command_header(self, header: _Header) -> "_JointPosition":
        self.header = header
        return self

    def set_minimum_time(self, value: float) -> "_JointPosition":
        self.minimum_time_s = float(value)
        return self

    def set_position(self, value: Any) -> "_JointPosition":
        self.position = np.asarray(value, dtype=np.float64).copy()
        return self

    def set_stiffness(self, value: Any) -> "_JointPosition":
        self.stiffness = np.asarray(value, dtype=np.float64).copy()
        return self


class _CartesianImpedance:
    def __init__(self) -> None:
        self.header: _Header | None = None
        self.target: ArmTargetRecord | None = None
        self.minimum_time_s: float | None = None
        self.joint_stiffness: np.ndarray | None = None
        self.damping_ratio: float | None = None
        self.nullspace: np.ndarray | None = None
        self.reset_reference: bool | None = None

    def set_command_header(self, header: _Header) -> "_CartesianImpedance":
        self.header = header
        return self

    def add_target(self, reference_link, link, transform, lv, av, la, aa):
        self.target = ArmTargetRecord(
            str(reference_link),
            str(link),
            np.asarray(transform, dtype=np.float64).copy(),
            float(lv),
            float(av),
            float(la),
            float(aa),
        )
        return self

    def set_joint_stiffness(self, value: Any) -> "_CartesianImpedance":
        self.joint_stiffness = np.asarray(value, dtype=np.float64).copy()
        return self

    def set_joint_damping_ratio(self, value: float) -> "_CartesianImpedance":
        self.damping_ratio = float(value)
        return self

    def set_minimum_time(self, value: float) -> "_CartesianImpedance":
        self.minimum_time_s = float(value)
        return self

    def set_nullspace_joint_target(self, joints, weight, kp, kd, cost):
        self.nullspace = np.asarray(joints, dtype=np.float64).copy()
        return self

    def set_reset_reference(self, value: bool) -> "_CartesianImpedance":
        self.reset_reference = bool(value)
        return self


class _SE2Velocity:
    def __init__(self) -> None:
        self.header: _Header | None = None
        self.minimum_time_s: float | None = None
        self.linear: np.ndarray | None = None
        self.angular: float | None = None

    def set_command_header(self, header: _Header) -> "_SE2Velocity":
        self.header = header
        return self

    def set_minimum_time(self, value: float) -> "_SE2Velocity":
        self.minimum_time_s = float(value)
        return self

    def set_velocity(self, linear: Any, angular: float) -> "_SE2Velocity":
        self.linear = np.asarray(linear, dtype=np.float64).copy()
        self.angular = float(angular)
        return self

    def set_acceleration_limit(self, linear: Any, angular: float) -> "_SE2Velocity":
        return self


class _Body:
    def __init__(self) -> None:
        self.torso = None
        self.right_arm = None
        self.left_arm = None

    def set_torso_command(self, command):
        self.torso = command
        return self

    def set_right_arm_command(self, command):
        self.right_arm = command
        return self

    def set_left_arm_command(self, command):
        self.left_arm = command
        return self


class _Component:
    def __init__(self) -> None:
        self.mobility = None
        self.body = None
        self.head = None

    def set_mobility_command(self, command):
        self.mobility = command
        return self

    def set_body_command(self, command):
        self.body = command
        return self

    def set_head_command(self, command):
        self.head = command
        return self


class _RobotCommand:
    def __init__(self) -> None:
        self.component: _Component | None = None

    def set_command(self, component: _Component) -> "_RobotCommand":
        self.component = component
        return self

    def to_packet(self) -> Packet:
        c = self.component
        assert c is not None
        packet = Packet()
        body = c.body
        if body is not None and body.torso is not None:
            packet.torso_position = body.torso.position
            packet.minimum_time_s = body.torso.minimum_time_s
            if body.torso.header is not None:
                packet.control_hold_time_s = body.torso.header.hold_time_s
        if c.head is not None:
            packet.head_position = c.head.position
        if body is not None:
            for side in ("right", "left"):
                command = getattr(body, f"{side}_arm")
                if isinstance(command, _CartesianImpedance):
                    setattr(packet, f"{side}_arm", command.target)
                    packet.joint_stiffness = command.joint_stiffness
                    setattr(packet, f"nullspace_{side}", command.nullspace)
                elif isinstance(command, _JointPosition):
                    setattr(packet, f"{side}_arm_joint_position", command.position)
        if c.mobility is not None and c.mobility.linear is not None:
            packet.mobility_velocity = (
                float(c.mobility.linear[0]),
                float(c.mobility.linear[1]),
                float(c.mobility.angular),
            )
        return packet


# --------------------------------------------------------------------------- #
# feedback tree
# --------------------------------------------------------------------------- #
class _Node:
    def __init__(self, valid: bool = True, **children: Any) -> None:
        self.valid = valid
        for name, value in children.items():
            setattr(self, name, value)


def running_feedback(*, status: int = 2, finish_code: int = 0) -> _Node:
    arm = _Node(
        joint_impedance_control_command=_Node(),
        cartesian_impedance_control_command=_Node(),
    )
    body_components = _Node(
        torso_command=_Node(joint_position_command=_Node()),
        right_arm_command=arm,
        left_arm_command=arm,
    )
    component = _Node(
        mobility_command=_Node(se2_velocity_command=_Node()),
        body_command=_Node(body_component_based_command=body_components),
        head_command=_Node(joint_position_command=_Node()),
    )
    feedback = _Node(component_based_command=component)
    feedback.status = status
    feedback.finish_code = finish_code
    return feedback


# --------------------------------------------------------------------------- #
# stream and robot
# --------------------------------------------------------------------------- #
class FakeStream:
    def __init__(self, recorder: list[Packet]) -> None:
        self._recorder = recorder
        self.cancelled = False
        self.send_count = 0

    def send_command(self, command: _RobotCommand, timeout_ms: int | None = None):
        self.send_count += 1
        self._recorder.append(command.to_packet())
        return running_feedback()

    def request_feedback(self, timeout_ms: int | None = None):
        return running_feedback()

    def cancel(self) -> None:
        self.cancelled = True

    def wait_for(self, timeout_ms: int) -> bool:
        return True


@dataclass
class FakeRobot:
    """Publishes measured states at the requested rate from a mutable pose."""

    right_eef_xyz: tuple[float, float, float] = (0.450, -0.170, 0.732)
    left_eef_xyz: tuple[float, float, float] = (0.450, 0.170, 0.732)
    torso_xyz: tuple[float, float, float] = (0.0, 0.0, 0.90)
    ready_pose: Any = None
    packets: list[Packet] = field(default_factory=list)
    streams: list[FakeStream] = field(default_factory=list)
    _connected: bool = False
    _callback: Any = None
    _thread: Any = None
    _stop: Any = field(default_factory=threading.Event)

    # --- connection -------------------------------------------------------- #
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        self._connected = True
        return True

    def get_robot_info(self) -> Any:
        return _Node(robot_model_name="M", robot_model_version="1.2")

    def model(self) -> Any:
        return _Node(
            model_name="M",
            robot_joint_names=list(JOINT_NAMES),
            mobility_idx=MOBILITY_IDX,
            torso_idx=TORSO_IDX,
            right_arm_idx=RIGHT_ARM_IDX,
            left_arm_idx=LEFT_ARM_IDX,
            head_idx=HEAD_IDX,
        )

    def get_dynamics(self) -> Any:  # the controller prefers an injected fk_provider
        raise RuntimeError("fake robot has no dynamics; inject fk_provider")

    # --- measured state ---------------------------------------------------- #
    def start_state_update(self, callback: Any, rate_hz: float) -> None:
        self._callback = callback
        self._stop.clear()
        period = 1.0 / max(float(rate_hz), 1.0)

        def loop() -> None:
            while not self._stop.is_set():
                callback(self.state())
                self._stop.wait(period)

        callback(self.state())
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_state_update(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def state(self) -> Any:
        position = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        pose = self.ready_pose
        if pose is not None:
            position[TORSO_IDX] = np.asarray(pose.torso_rad)
            position[RIGHT_ARM_IDX] = np.asarray(pose.right_arm_rad)
            position[LEFT_ARM_IDX] = np.asarray(pose.left_arm_rad)
            position[HEAD_IDX] = np.asarray(pose.head_rad)
        se2 = np.eye(3, dtype=np.float64)
        return _Node(
            position=position,
            velocity=np.zeros(len(JOINT_NAMES), dtype=np.float64),
            is_ready=np.ones(len(JOINT_NAMES), dtype=np.bool_),
            odometry=se2,
            timestamp=time.monotonic(),
        )

    # --- fk provider ------------------------------------------------------- #
    def fk_provider(self, position: np.ndarray, velocity: np.ndarray) -> Any:
        def transform(xyz):
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, 3] = np.asarray(xyz, dtype=np.float64)
            return matrix

        return {
            "T_base_torso": transform(self.torso_xyz),
            "T_base_head": transform((0.10, 0.0, 1.20)),
            "T_base_right_eef": transform(self.right_eef_xyz),
            "T_base_left_eef": transform(self.left_eef_xyz),
            "base_twist_w_vx_vy": (0.0, 0.0, 0.0),
        }

    # --- stream ------------------------------------------------------------ #
    def create_command_stream(self, priority: int = 0) -> FakeStream:
        stream = FakeStream(self.packets)
        self.streams.append(stream)
        return stream


class FakeSdk:
    """Module-like object exposing the builders the controller reaches for."""

    def __init__(self, robot: FakeRobot) -> None:
        self._robot = robot

    def create_robot(self, address: str, model: str) -> FakeRobot:
        return self._robot

    CommandHeaderBuilder = _Header
    JointPositionCommandBuilder = _JointPosition
    JointImpedanceControlCommandBuilder = _JointPosition
    CartesianImpedanceControlCommandBuilder = _CartesianImpedance
    SE2VelocityCommandBuilder = _SE2Velocity
    BodyComponentBasedCommandBuilder = _Body
    ComponentBasedCommandBuilder = _Component
    RobotCommandBuilder = _RobotCommand
