# Grabbing Box Demo (based on picking_box.py)
# This example moves the robot to a recorded START pose (joint position control)
# and then makes both end-effectors (ee_right / ee_left) grab a box along the
# y-axis using a Cartesian IMPEDANCE controller (ImpedanceControlCommand).
#
# Motion sequence:
#   1. start_pose -> move to the recorded starting posture (joint position control)
#   2. grab_box   -> push both hands inward along y (Cartesian impedance control)
#                    right hand -> +y, left hand -> -y (both toward the center)
#   3. lift_box   -> raise both hands along EACH EEF's OWN +z axis while
#                    KEEPING the inward y squeeze (Cartesian impedance control,
#                    timed trajectory)
#
# The impedance translation weight is set WEAK ONLY ON THE Y-AXIS, so the hands
# approach the box gently and compliantly along the grabbing direction (y),
# while x/z stay stiff to keep the hands from drifting off the grab plane.
#
# Step 3 uses CartesianImpedanceControlCommand: each hand target is the CURRENT
# EEF pose offset by LIFT_DISTANCE along the EEF's LOCAL +z axis (ee_right /
# ee_left frame, NOT base z) PLUS an inward y offset (LIFT_SQUEEZE_DISTANCE).
# The inward offset is essential: it keeps a permanent position error toward
# the box center, so the impedance controller keeps PRESSING the box while it
# rises. Without it the targets sit exactly at the current hand positions,
# nothing squeezes, and the hands spread apart and drop the box.
# The move follows a TIMED trajectory (set_minimum_time + velocity limits) for
# a smooth rise, while joint-space impedance (stiffness / damping / torque
# limit) tracks it compliantly.
#
# Force/torque (FT) sensor data on both arms is monitored in real time through
# a state-update callback (robot.start_state_update) while the motion executes.
#
# Usage example after installing the Codex package:
#     python -m parcel_pose.grabbing --address 192.168.30.1:50051 --model m
#
# Copyright (c) 2025 Rainbow Robotics. All rights reserved.
#
# DISCLAIMER:
# This is a sample code provided for educational and reference purposes only.
# Rainbow Robotics shall not be held liable for any damages or malfunctions resulting from
# the use or misuse of this demo code. Please use with caution and at your own discretion.

import argparse

import numpy as np
import rby1_sdk as rby

# Time (seconds) the robot takes to reach the start pose. Larger = slower/safer.
MINIMUM_TIME = 2.0

# Arm-only posture used while the mobile base approaches the box.  These are
# measured RB-Y1 M v1.2 joint angles supplied in degrees; the SDK receives
# radians.  Torso/head are intentionally absent so the fixed camera
# calibration posture cannot be changed by this preparatory command.
MOBILE_READY_RIGHT_ARM_DEG = np.asarray(
    [6.644, -21.489, -17.252, -129.031, -83.302, 53.394, 37.071],
    dtype=np.float64,
)
MOBILE_READY_LEFT_ARM_DEG = np.asarray(
    [6.644, 21.488, 17.245, -129.036, 83.304, 53.392, -37.070],
    dtype=np.float64,
)
MOBILE_READY_RIGHT_ARM_RAD = np.deg2rad(MOBILE_READY_RIGHT_ARM_DEG)
MOBILE_READY_LEFT_ARM_RAD = np.deg2rad(MOBILE_READY_LEFT_ARM_DEG)
MOBILE_READY_MINIMUM_TIME = 0.5
MOBILE_READY_HOLD_TIME = 0.5
MOBILE_READY_COMMAND_TIMEOUT_MS = 10_000

# Rate (Hz) at which the FT-sensor monitoring callback is invoked.
FT_MONITOR_RATE = 10.0

# ---- Cartesian impedance "grab along y" parameters ----
# Link indices into the dynamics state (order MUST match DYN_LINK_NAMES below).
DYN_LINK_NAMES = ["base", "link_torso_5", "ee_right", "ee_left"]
TORSO_INDEX, EE_RIGHT_INDEX, EE_LEFT_INDEX = 1, 2, 3
# Reference link the EEF target pose is expressed in (torso tip = arm root).
TORSO_REFERENCE_LINK = "link_torso_5"
# Distance [m] each hand moves inward along the reference-frame y-axis.
# Right hand -> +y, left hand -> -y (both toward the center of the box).
GRAB_DISTANCE = 0.3
# Task-space (Cartesian) stiffness diag(Kx, Ky, Kz) [N/m].
# Y is intentionally WEAK: the hands press the box gently/compliantly along the
# grabbing axis. X/Z stay stiff so the hands hold their position in the other
# directions. TUNE Ky for your object (higher = firmer grip force).
IMPEDANCE_TRANSLATION_WEIGHT = [1000.0, 200.0, 1000.0]
# Rotational stiffness diag [Nm/rad].
IMPEDANCE_ROTATION_WEIGHT = [50.0, 50.0, 50.0]
# control_hold_time [s] for the grab: short settle time only, because the lift
# command that follows keeps holding the box (see LIFT_HOLD_TIME).
GRAB_HOLD_TIME = 1.0

# ---- Cartesian IMPEDANCE lift: raise along each EEF's LOCAL +z axis ----
# The lift target is computed per hand: current EEF pose offset by
# LIFT_DISTANCE along ITS OWN +z axis (ee_right / ee_left frame) plus
# LIFT_SQUEEZE_DISTANCE inward along the reference-frame y-axis. Targets are
# expressed in the torso-tip frame (same as the grab) so the inward y
# direction is straightforward.
# Distance [m] to move each hand along its LOCAL +z axis.
LIFT_DISTANCE = 0.15
# Extra INWARD offset [m] along the reference-frame y-axis baked into the lift
# targets (right +y / left -y). This keeps a permanent inward position error,
# i.e. a continuous squeezing force, while the box rises. Set it deep enough
# that the error never reaches zero during the lift. Larger = firmer squeeze.
LIFT_SQUEEZE_DISTANCE = 0.15
# Trajectory time [s] for the lift. Larger = slower / smoother rise.
LIFT_MINIMUM_TIME = 1.0
# Cartesian velocity / acceleration caps for the lift trajectory (safety bounds;
# with LIFT_MINIMUM_TIME large, the time governs and these rarely bind).
LIFT_LINEAR_VELOCITY_LIMIT = 0.1                # m/s
LIFT_ANGULAR_VELOCITY_LIMIT = float(np.pi / 4)  # rad/s
LIFT_LINEAR_ACCELERATION_LIMIT = 0.5            # m/s^2
LIFT_ANGULAR_ACCELERATION_LIMIT = float(np.pi)  # rad/s^2
# Per-arm (7-joint) joint-space impedance that holds the box while lifting.
# TUNE for your robot/payload: stiffness firm enough to hold, torque above load.
LIFT_JOINT_STIFFNESS = [150.0] * 7              # Nm/rad
LIFT_JOINT_DAMPING_RATIO = 1.0                  # critically damped -> smooth, no overshoot
LIFT_JOINT_TORQUE_LIMIT = [100.0] * 7            # Nm (must exceed the holding torque)
# control_hold_time [s]: keep the box raised/held after the lift finishes.
LIFT_HOLD_TIME = 100.0


# ========================================================================================
# Recorded joint set [rad], grouped by component (from a measured robot state).
# Order per component follows model.torso_idx / right_arm_idx / left_arm_idx / head_idx.
# ========================================================================================

# --- Start pose: posture right before grabbing the box ---
START_POSE = {
    "torso":     [0.000,  0.960, -1.047,  0.114,  0.000,  0.000],
    "right_arm": [-0.375, -0.391, -0.238, -1.576, -1.013,  1.467, -0.003],
    "left_arm":  [-0.375,  0.391,  0.238, -1.576,  1.013,  1.467,  0.003],
    "head":      [0.000,  0.870],
}


def build_pose_command(pose, minimum_time):
    """Build a RobotCommandBuilder that drives every component to `pose` using
    Joint Position Command builders (torso / right arm / left arm / head)."""
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder()
        .set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_torso_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(pose["torso"])
            )
            .set_right_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(pose["right_arm"])
            )
            .set_left_arm_command(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(pose["left_arm"])
            )
        )
        .set_head_command(
            rby.HeadCommandBuilder(
                rby.JointPositionCommandBuilder()
                .set_minimum_time(minimum_time)
                .set_position(pose["head"])
            )
        )
    )


def build_mobile_ready_command(
    minimum_time=MOBILE_READY_MINIMUM_TIME,
    hold_time=MOBILE_READY_HOLD_TIME,
):
    """Build the arm-only Joint Position command used before base motion."""

    minimum_time = float(minimum_time)
    hold_time = float(hold_time)
    if not np.isfinite(minimum_time) or minimum_time <= 0.0:
        raise ValueError("mobile-ready minimum_time must be positive and finite")
    if not np.isfinite(hold_time) or hold_time <= 0.0:
        raise ValueError("mobile-ready hold_time must be positive and finite")

    def arm_command(position):
        return (
            rby.JointPositionCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
            )
            .set_position(position)
            .set_minimum_time(minimum_time)
        )

    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_right_arm_command(arm_command(MOBILE_READY_RIGHT_ARM_RAD))
            .set_left_arm_command(arm_command(MOBILE_READY_LEFT_ARM_RAD))
        )
    )


def _cancel_active_command(handler) -> None:
    """Best-effort cancellation that never masks the command's root failure."""
    try:
        handler.cancel()
    except Exception as exc:
        print(f"[grabbing] command cancellation failed: {exc}")
        return

    try:
        if handler.wait_for(2_000) is False:
            print("[grabbing] command did not finish within cancellation timeout")
    except Exception as exc:
        print(f"[grabbing] command cancellation wait failed: {exc}")


def send_once(robot, builder, *, timeout_ms=None):
    """Send a single (one-shot) command and block until it finishes.

    This is the "send once" pattern: hand one command to the robot, wait for the
    handler to complete, and check the finish code (no persistent stream)."""
    if timeout_ms is not None:
        timeout_ms = int(timeout_ms)
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")

    handler = robot.send_command(builder)
    try:
        if timeout_ms is not None:
            if handler.wait_for(timeout_ms) is False:
                _cancel_active_command(handler)
                return False
        feedback = handler.get()
    except KeyboardInterrupt:
        _cancel_active_command(handler)
        raise
    except Exception:
        # SDK/gRPC feedback failures do not prove that the controller stopped
        # executing the command. Cancel the active handler before propagating
        # the original error so long-hold arm commands fail closed.
        _cancel_active_command(handler)
        raise
    return feedback.finish_code == rby.RobotCommandFeedback.FinishCode.Ok


def move_arms_to_mobile_ready_pose(robot):
    """Complete one arm-only position command before creating a base stream."""

    if not robot.is_connected():
        raise ConnectionError("Robot is not connected")

    print("[grabbing] moving both arms to mobile-ready pose (joint position) ...")
    succeeded = send_once(
        robot,
        build_mobile_ready_command(),
        timeout_ms=MOBILE_READY_COMMAND_TIMEOUT_MS,
    )
    if not succeeded:
        print("[grabbing] FAILED while moving arms to mobile-ready pose.")
        return False

    print("[grabbing] mobile-ready Joint Position command completed with OK feedback.")
    return True


def offset_translation_y(T, dy):
    """Return a copy of the 4x4 transform T translated by `dy` along y
    (in the reference frame; orientation unchanged)."""
    T_offset = T.copy()
    T_offset[1, 3] += dy
    return T_offset


def offset_translation_local_z(T, dz):
    """Return a copy of the 4x4 transform T translated by `dz` along ITS OWN
    z-axis (the link's local frame, i.e. the 3rd column of the rotation part;
    orientation unchanged)."""
    T_offset = T.copy()
    T_offset[:3, 3] += T[:3, :3] @ np.array([0.0, 0.0, dz])
    return T_offset


def build_impedance_grab_command(dyn_model, dyn_state, q):
    """grab_box: push both hands inward along y to grab the box.

    Targets are expressed in TORSO_REFERENCE_LINK (torso tip). Each hand
    target = its CURRENT EEF pose offset along y by GRAB_DISTANCE:
      - right hand: +y (toward the center)
      - left hand:  -y (toward the center)

    Cartesian impedance provides compliance: the hands press toward the target
    with a bounded, spring-like force (F = K * pose_error) instead of rigidly
    tracking a position. The y stiffness is WEAK on purpose so the box is
    grabbed gently along the y-axis."""
    # Current EEF poses expressed in the reference link, via FK.
    dyn_state.set_q(q)
    dyn_model.compute_forward_kinematics(dyn_state)
    T_ref2right = dyn_model.compute_transformation(dyn_state, TORSO_INDEX, EE_RIGHT_INDEX)
    T_ref2left = dyn_model.compute_transformation(dyn_state, TORSO_INDEX, EE_LEFT_INDEX)

    # Offset targets: inward along y (right +y / left -y).
    T_right_target = offset_translation_y(T_ref2right, +GRAB_DISTANCE)
    T_left_target = offset_translation_y(T_ref2left, -GRAB_DISTANCE)

    print(
        f"[grab] ref='{TORSO_REFERENCE_LINK}' right EEF y: "
        f"{T_ref2right[1, 3]:+.3f} -> {T_right_target[1, 3]:+.3f} m"
        f"  |  left EEF y: {T_ref2left[1, 3]:+.3f} -> {T_left_target[1, 3]:+.3f} m"
    )

    def arm_impedance(link_name, T_target):
        return (
            rby.ImpedanceControlCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(GRAB_HOLD_TIME)
            )
            .set_reference_link_name(TORSO_REFERENCE_LINK)
            .set_link_name(link_name)
            .set_translation_weight(IMPEDANCE_TRANSLATION_WEIGHT)
            .set_rotation_weight(IMPEDANCE_ROTATION_WEIGHT)
            .set_transformation(T_target)
        )

    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_right_arm_command(arm_impedance("ee_right", T_right_target))
            .set_left_arm_command(arm_impedance("ee_left", T_left_target))
        )
    )


def build_impedance_lift_command(dyn_model, dyn_state, q):
    """lift_box: raise both hands by LIFT_DISTANCE along EACH EEF's OWN +z axis
    while KEEPING the inward squeeze, SMOOTHLY.

    Each hand target = CURRENT EEF pose
      + LIFT_DISTANCE along the EEF's LOCAL +z axis        (raise the box)
      + LIFT_SQUEEZE_DISTANCE inward along reference y     (keep pressing)
        (right hand +y, left hand -y, torso-tip frame)

    The inward offset keeps a permanent position error toward the box center,
    so the impedance controller exerts a continuous squeezing force during and
    after the rise -- without it the hands spread apart and drop the box.

    Uses a Cartesian IMPEDANCE controller that follows a timed trajectory
    (set_minimum_time + velocity limits) so the box rises gently over
    LIFT_MINIMUM_TIME, while joint-space impedance (stiffness / damping /
    torque limit) tracks it compliantly."""
    # Current EEF poses in the torso-tip frame (command reference), via FK.
    dyn_state.set_q(q)
    dyn_model.compute_forward_kinematics(dyn_state)
    T_ref2right = dyn_model.compute_transformation(dyn_state, TORSO_INDEX, EE_RIGHT_INDEX)
    T_ref2left = dyn_model.compute_transformation(dyn_state, TORSO_INDEX, EE_LEFT_INDEX)

    # Targets: raise along each EEF's LOCAL +z axis, then push the target
    # further INWARD along the reference-frame y-axis to keep the squeeze.
    T_right_target = offset_translation_local_z(T_ref2right, +LIFT_DISTANCE)
    T_right_target = offset_translation_y(T_right_target, +LIFT_SQUEEZE_DISTANCE)
    T_left_target = offset_translation_local_z(T_ref2left, +LIFT_DISTANCE)
    T_left_target = offset_translation_y(T_left_target, -LIFT_SQUEEZE_DISTANCE)

    print(
        f"[lift] ref='{TORSO_REFERENCE_LINK}' right EEF pos: "
        f"{T_ref2right[:3, 3]} -> {T_right_target[:3, 3]} m"
        f"  (local +z {T_ref2right[:3, 2]})"
    )
    print(
        f"[lift] ref='{TORSO_REFERENCE_LINK}' left  EEF pos: "
        f"{T_ref2left[:3, 3]} -> {T_left_target[:3, 3]} m"
        f"  (local +z {T_ref2left[:3, 2]})  (over {LIFT_MINIMUM_TIME:.0f}s)"
    )

    def arm_cartesian_impedance(link_name, T_target):
        return (
            rby.CartesianImpedanceControlCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(LIFT_HOLD_TIME)
            )
            .add_target(
                TORSO_REFERENCE_LINK, link_name, T_target,
                LIFT_LINEAR_VELOCITY_LIMIT, LIFT_ANGULAR_VELOCITY_LIMIT,
                LIFT_LINEAR_ACCELERATION_LIMIT, LIFT_ANGULAR_ACCELERATION_LIMIT,
            )
            .set_joint_stiffness(LIFT_JOINT_STIFFNESS)
            .set_joint_damping_ratio(LIFT_JOINT_DAMPING_RATIO)
            .set_joint_torque_limit(LIFT_JOINT_TORQUE_LIMIT)
            .set_minimum_time(LIFT_MINIMUM_TIME)
        )

    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_right_arm_command(arm_cartesian_impedance("ee_right", T_right_target))
            .set_left_arm_command(arm_cartesian_impedance("ee_left", T_left_target))
        )
    )


class FTMonitor:
    """Monitors force/torque sensor data through a state-update callback.

    `callback` is passed to robot.start_state_update() and is invoked in a
    background thread at FT_MONITOR_RATE Hz. It prints the latest right/left
    FT readings and keeps track of the peak force magnitude seen on each arm."""

    def __init__(self):
        self.samples = 0
        self.peak_force_right = 0.0
        self.peak_force_left = 0.0

    def callback(self, robot_state):
        ft_right = robot_state.ft_sensor_right
        ft_left = robot_state.ft_sensor_left

        # Force magnitude (Euclidean norm of the 3-axis force vector) in Newtons.
        force_right = float(np.linalg.norm(ft_right.force))
        force_left = float(np.linalg.norm(ft_left.force))

        self.samples += 1
        self.peak_force_right = max(self.peak_force_right, force_right)
        self.peak_force_left = max(self.peak_force_left, force_left)

        print(
            f"[FT] right | force {ft_right.force} |F|={force_right:6.2f}N "
            f"  ||  left | force {ft_left.force} |F|={force_left:6.2f}N "
        )


def prepare_robot(robot, *, power=".*") -> None:
    """Prepare an already connected robot for command execution.

    Connection ownership stays with the caller so a mobile-base command stream
    can hand the same robot instance directly to ``run_grabbing_sequence``.
    """
    if not robot.is_connected():
        raise ConnectionError("Robot is not connected")

    # Bring the robot up: power, servo, clear faults, enable the control manager.
    if not robot.is_power_on(power):
        if not robot.power_on(power):
            raise RuntimeError(f"Failed to power on devices matching {power!r}")
    if not robot.is_servo_on(".*"):
        if not robot.servo_on(".*"):
            raise RuntimeError("Failed to servo on")
    robot.reset_fault_control_manager()
    if not robot.enable_control_manager():
        raise RuntimeError("Failed to enable control manager")


def run_grabbing_sequence(robot, *, monitor_ft=True) -> bool:
    """Run start pose -> impedance grab -> impedance lift exactly once.

    ``robot`` must already be connected and prepared. This function deliberately
    does not connect, reconnect, disconnect, or alter the mobile-base lifecycle.
    """
    if not robot.is_connected():
        raise ConnectionError("Robot is not connected")

    # Dynamics model used to compute current EEF poses (forward kinematics) for
    # the Cartesian impedance grab. Joint order follows model.robot_joint_names.
    robot_model = robot.model()
    model_name = str(robot_model.model_name).lower()
    if model_name != "m":
        raise ValueError(
            f"packaged grasp supports RB-Y1 Model M only; connected model is {model_name!r}"
        )
    dyn_model = robot.get_dynamics()
    dyn_state = dyn_model.make_state(DYN_LINK_NAMES, robot_model.robot_joint_names)

    # Optionally monitor FT data without taking ownership of the robot connection.
    ft_monitor = FTMonitor() if monitor_ft else None
    monitoring_started = False
    if ft_monitor is not None:
        robot.start_state_update(ft_monitor.callback, FT_MONITOR_RATE)
        monitoring_started = True

    try:
        # 1) Joint position control: move to the start pose.
        print("[grabbing] moving to 'start_pose' (joint position) ...")
        if not send_once(robot, build_pose_command(START_POSE, MINIMUM_TIME)):
            print("[grabbing] FAILED while moving to 'start_pose'. Aborting.")
            return False
        print("[grabbing] reached 'start_pose'.")

        # 2) grab_box: Cartesian impedance grab (both hands inward on y,
        #    weak y stiffness -> gentle, compliant grip).
        print("[grabbing] grabbing the box along y with impedance control ...")
        q = robot.get_state().position
        if not send_once(robot, build_impedance_grab_command(dyn_model, dyn_state, q)):
            print("[grabbing] FAILED during Cartesian impedance grab. Aborting.")
            return False
        print("[grabbing] Cartesian impedance grab done.")

        # 3) lift_box: keep the grip and raise both hands by LIFT_DISTANCE
        #    along each EEF's OWN +z axis.
        print(
            f"[grabbing] lifting the box ~{LIFT_DISTANCE * 100:.0f} cm "
            "along each EEF's local +z ..."
        )
        q = robot.get_state().position
        if not send_once(robot, build_impedance_lift_command(dyn_model, dyn_state, q)):
            print("[grabbing] FAILED during lift. Aborting.")
            return False
        print("[grabbing] box lifted.")

        # All steps completed successfully.
        print("=" * 60)
        print("[grabbing] grabbing motion COMPLETED. done = True")
        print("=" * 60)
    finally:
        # Stop only the monitoring lifecycle started by this invocation.
        if monitoring_started:
            robot.stop_state_update()
            print(
                f"[FT] monitoring stopped. samples={ft_monitor.samples}, "
                f"peak |F| right={ft_monitor.peak_force_right:.2f}N, "
                f"left={ft_monitor.peak_force_left:.2f}N"
            )

    return True


def main(address, model="m", power=".*") -> bool:
    """Connect a Model M robot, prepare it, run one grasp, and disconnect."""
    normalized_model = model.lower()
    if normalized_model != "m":
        print(f"Unsupported robot model {model!r}; packaged grasp requires Model M ('m').")
        return False

    robot = rby.create_robot(address, normalized_model)
    try:
        connected = robot.connect()
        if connected is False or not robot.is_connected():
            print(f"Failed to connect to RB-Y1 at {address}")
            return False

        prepare_robot(robot, power=power)
        return run_grabbing_sequence(robot)
    except Exception as exc:
        print(f"[grabbing] ERROR: {exc}")
        return False
    finally:
        disconnect = getattr(robot, "disconnect", None)
        if callable(disconnect):
            disconnect()


def _parse_args():
    parser = argparse.ArgumentParser(description="RB-Y1 Model M box-grabbing sequence")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument(
        "--model",
        type=str.lower,
        choices=("m",),
        default="m",
        help="Robot Model Name (Model M only; default: 'm')",
    )
    parser.add_argument(
        "--power",
        type=str,
        default=".*",
        help="Power device name regex pattern (default: '.*')",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        success = main(address=args.address, model=args.model, power=args.power)
    except KeyboardInterrupt:
        print("\n[grabbing] interrupted by user")
        raise SystemExit(130) from None
    raise SystemExit(0 if success else 1)
