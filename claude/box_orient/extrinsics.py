"""Camera -> robot (link_torso_5 / "T5") extrinsic transforms.

Adapted from KETI_DEMO/box_codex/camera_extrinsics.py. Convention is
target_from_source:  p_target = T_TARGET_FROM_SOURCE @ [x, y, z, 1].

The camera is mounted on link_head_2 (after the head_1 pitch joint). On the
robot, feed the live head joints through FK for an exact transform; offline we
use the static transform for the recorded posture.

D435 SWAP NOTE: only the *mount* measurement (camera optical origin relative to
link_head_2) changes when moving from the D405 to the D435 -- re-measure
``HEAD2_TO_D435_XYZ_RPY_ZYX_DEG`` and the head pitch used for observation. The
rest of the chain is identical.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# --- rotation / transform primitives (ZYX euler, degrees) ------------------ #
def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_from_euler_zyx_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def make_transform(translation: Any, rotation: Any = None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if rotation is not None:
        T[:3, :3] = np.asarray(rotation, dtype=np.float64)
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def transform_from_xyz_rpy_zyx_deg(xyz_rpy_deg: Any) -> np.ndarray:
    x, y, z, roll, pitch, yaw = np.asarray(xyz_rpy_deg, dtype=np.float64)
    return make_transform([x, y, z], rotation_from_euler_zyx_deg(roll, pitch, yaw))


def invert_transform(T: Any) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


# --- mount measurements ---------------------------------------------------- #
# D405 optical origin in link_head_2 (x,y,z metres; roll,pitch,yaw deg, ZYX).
HEAD2_TO_D405_XYZ_RPY_ZYX_DEG = np.array([0.023, 0.0, 0.066, 0.0, 90.0, 0.0], dtype=np.float64)

# D435 mount, measured link_head_2 -> camera (camera-plane referenced),
# xyz [m] + roll,pitch,yaw [deg] ZYX.
HEAD2_TO_D435_XYZ_RPY_ZYX_DEG = np.array([0.049, -0.0115, 0.057, -90.0, 0.0, -90.0], dtype=np.float64)

# Depth "ground zero" offset from the camera front glass along the optical axis
# (datasheet "Depth Start Point"). Deprojected points are referenced to the
# depth optical origin, but the mounts above were measured to the front plane,
# so shift by this along optical +z. Negligible for yaw; matters at the mm level
# for the grasp / placement position.
DEPTH_START_POINT_M = {"d405": 0.0, "d435": -0.0042}

# link_torso_5 -> link_head_1 with head_0 = 0 (static URDF chain; rby1a/A-type
# values -- verify against the M-type V1.2 URDF for exact position. Yaw uses only
# rotation, so this translation does not affect the orientation estimate.)
T5_TO_HEAD1_ZERO_HEAD0_XYZ_M = np.array([0.022, 0.0, 0.200073451525], dtype=np.float64)
# Recorded observation posture, head_1 pitch [rad] (override via head1_pitch_rad).
#   D405 clips: 0.436 (25 deg).
#   D435IF clips (RB-Y1 V1.2, M-type): head = [0, 49.846] deg;
#       torso = [0, 55, -59.988, 6.532, 0, 0] deg (torso only positions T5).
HEAD1_PITCH_RAD_D405_RECORDED = 0.436
HEAD1_PITCH_RAD_D435_RECORDED = float(np.deg2rad(49.846))
_RECORDED_PITCH = {"d405": HEAD1_PITCH_RAD_D405_RECORDED, "d435": HEAD1_PITCH_RAD_D435_RECORDED}

_MOUNTS = {"d405": HEAD2_TO_D405_XYZ_RPY_ZYX_DEG, "d435": HEAD2_TO_D435_XYZ_RPY_ZYX_DEG}
_CALIBRATED = {"d405": True, "d435": True}

# base <- link_torso_5 for the FIXED recorded torso (RB-Y1 V1.2, M-type):
#   torso = [0, 55, -59.988, 6.532, 0, 0] deg,  head = [0, 49.846] deg.
# Computed offline with rby1_sdk.dynamics FK on models/rby1m/urdf/model_v1.2.urdf:
#   robot = Robot(load_robot_from_urdf(urdf, "base")); set torso/head q;
#   compute_transformation(state, base_idx, link_torso_5_idx).
# Torso is fixed, so this is a constant -- recompute only if the torso posture
# changes (no rby1_sdk needed at runtime).
BASE_FROM_T5 = np.array(
    [
        [0.99963693, 0.0, 0.02694462, 0.26460911],
        [0.0, 1.0, 0.0, 0.0],
        [-0.02694462, 0.0, 0.99963693, 1.15524048],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _head2_from_camera(key: str) -> np.ndarray:
    """Pose of the depth optical frame in link_head_2, including the datasheet
    depth-start-point offset along the optical axis."""
    T = transform_from_xyz_rpy_zyx_deg(_MOUNTS[key])
    dz = DEPTH_START_POINT_M.get(key, 0.0)
    if dz:
        T = T @ make_transform([0.0, 0.0, dz])
    return T


def camera_to_t5_static(camera: str = "d405", *, head1_pitch_rad: float | None = None) -> np.ndarray:
    """Static camera->T5 transform for a fixed observation posture (head_0 = 0)."""
    key = str(camera).lower()
    if key not in _MOUNTS:
        raise ValueError(f"unknown camera {camera!r}; use 'd405' or 'd435'")
    pitch = (
        _RECORDED_PITCH.get(key, HEAD1_PITCH_RAD_D405_RECORDED)
        if head1_pitch_rad is None
        else float(head1_pitch_rad)
    )
    t_head2_from_camera = _head2_from_camera(key)
    t_t5_from_head2 = make_transform(T5_TO_HEAD1_ZERO_HEAD0_XYZ_M) @ make_transform(
        [0.0, 0.0, 0.0], _rot_y(pitch)
    )
    return t_t5_from_head2 @ t_head2_from_camera


def camera_to_base_static(camera: str = "d435", *, head1_pitch_rad: float | None = None) -> np.ndarray:
    """Static camera->base transform (p_base = T @ p_camera) for the fixed torso.

    Composes the constant base<-T5 (torso FK) with camera<-T5.
    """
    return BASE_FROM_T5 @ camera_to_t5_static(camera, head1_pitch_rad=head1_pitch_rad)


def camera_to_t5_from_fk(
    camera: str,
    t5_from_head2: np.ndarray,
) -> np.ndarray:
    """Exact camera->T5 given a live link_torso_5<-link_head_2 transform (FK).

    On the robot, obtain ``t5_from_head2`` from the RB-Y1 dynamics/FK for the
    current head joints, then compose with the calibrated camera mount.
    """
    key = str(camera).lower()
    if key not in _MOUNTS:
        raise ValueError(f"unknown camera {camera!r}; use 'd405' or 'd435'")
    return np.asarray(t5_from_head2, dtype=np.float64) @ _head2_from_camera(key)


def is_calibrated(camera: str) -> bool:
    return bool(_CALIBRATED.get(str(camera).lower(), False))
