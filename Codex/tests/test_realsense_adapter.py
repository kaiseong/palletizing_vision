from types import SimpleNamespace

import numpy as np
import pytest

from parcel_pose.models import BoxModel
from parcel_pose.realsense_adapter import (
    D435StreamConfig,
    RealSenseAdapter,
    RealSenseUnavailableError,
    load_realsense_sdk,
)


class FakeVideoProfile:
    def __init__(self, stream_name, width=640, height=480):
        self.stream_name = stream_name
        self.intrinsics = SimpleNamespace(
            width=width,
            height=height,
            fx=500.0,
            fy=501.0,
            ppx=320.0,
            ppy=240.0,
            model="brown_conrady",
            coeffs=[0.0] * 5,
        )

    def as_video_stream_profile(self):
        return self

    def get_intrinsics(self):
        return self.intrinsics

    def get_extrinsics_to(self, target):
        translation = [0.015, 0.0, 0.0] if self.stream_name == "depth" else [-0.015, 0.0, 0.0]
        return SimpleNamespace(
            rotation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            translation=translation,
        )


class FakeSensor:
    def __init__(self, option_value=1.0):
        self.option_value = option_value

    def get_depth_scale(self):
        return 0.001

    def supports(self, option):
        return True

    def get_option(self, option):
        return self.option_value


class FakeDevice:
    def first_depth_sensor(self):
        return FakeSensor(1.0)

    def first_color_sensor(self):
        return FakeSensor(2.0)

    def supports(self, info):
        return True

    def get_info(self, info):
        return {
            "serial_number": "fake-D435",
            "firmware_version": "5.fake",
            "usb_type_descriptor": "3.2",
        }[info]


class FakePipelineProfile:
    def __init__(self):
        self.profiles = {
            "depth": FakeVideoProfile("depth"),
            "color": FakeVideoProfile("color"),
        }

    def get_device(self):
        return FakeDevice()

    def get_stream(self, stream):
        return self.profiles[stream]


class FakeFrame:
    def __init__(self, data, number, timestamp):
        self._data = data
        self._number = number
        self._timestamp = timestamp

    def __bool__(self):
        return True

    def get_data(self):
        return self._data

    def get_frame_number(self):
        return self._number

    def get_timestamp(self):
        return self._timestamp

    def get_frame_timestamp_domain(self):
        return "hardware_clock"


class FakeFrames:
    def __init__(self):
        self.depth = FakeFrame(np.full((480, 640), 700, dtype=np.uint16), 11, 100.0)
        self.color = FakeFrame(np.full((480, 640, 3), 9, dtype=np.uint8), 12, 100.5)

    def get_depth_frame(self):
        return self.depth

    def get_color_frame(self):
        return self.color


class FakePipeline:
    def __init__(self, sdk):
        self.sdk = sdk
        self.stopped = False

    def start(self, config):
        self.sdk.enabled_streams = list(config.enabled)
        return FakePipelineProfile()

    def wait_for_frames(self):
        return FakeFrames()

    def stop(self):
        self.stopped = True


class FakeConfig:
    def __init__(self):
        self.enabled = []

    def enable_stream(self, *args):
        self.enabled.append(args)


class FakeAlign:
    def __init__(self, target):
        self.target = target

    def process(self, frames):
        aligned = FakeFrames()
        aligned.color = FakeFrame(np.full((480, 640, 3), 17, dtype=np.uint8), 12, 100.5)
        return aligned


class FakeSdk:
    stream = SimpleNamespace(depth="depth", color="color")
    format = SimpleNamespace(z16="z16", bgr8="bgr8")
    camera_info = SimpleNamespace(
        serial_number="serial_number",
        firmware_version="firmware_version",
        usb_type_descriptor="usb_type_descriptor",
    )
    option = SimpleNamespace(
        exposure="exposure",
        gain="gain",
        emitter_enabled="emitter_enabled",
        laser_power="laser_power",
        visual_preset="visual_preset",
        enable_auto_exposure="enable_auto_exposure",
    )

    def __init__(self):
        self.enabled_streams = []

    def pipeline(self):
        return FakePipeline(self)

    def config(self):
        return FakeConfig()

    def align(self, target):
        return FakeAlign(target)


def test_sdk_is_loaded_lazily_and_missing_message_is_actionable(monkeypatch):
    import parcel_pose.realsense_adapter as adapter_module

    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(adapter_module.importlib, "import_module", missing)
    with pytest.raises(RealSenseUnavailableError, match="Python 3.12.*USB 3"):
        load_realsense_sdk()


def test_sdk_import_error_preserves_binary_compatibility_detail(monkeypatch):
    import parcel_pose.realsense_adapter as adapter_module

    def incompatible(name):
        raise ImportError("GLIBC_2.38 not found")

    monkeypatch.setattr(adapter_module.importlib, "import_module", incompatible)
    with pytest.raises(
        RealSenseUnavailableError,
        match="build_jetson_pyrealsense2.*GLIBC_2.38",
    ):
        load_realsense_sdk()


def test_fake_sdk_requests_d435_profiles_and_preserves_raw_frames():
    sdk = FakeSdk()
    adapter = RealSenseAdapter(D435StreamConfig(warmup_frames=0), sdk=sdk).start()
    try:
        assert sdk.enabled_streams == [
            ("depth", 640, 480, "z16", 30),
            ("color", 640, 480, "bgr8", 30),
        ]
        frame = adapter.capture()
        assert frame.raw_depth_z16.dtype == np.uint16
        assert frame.raw_color_bgr.dtype == np.uint8
        assert frame.raw_depth_z16[0, 0] == 700
        assert frame.raw_color_bgr[0, 0, 0] == 9
        assert frame.color_on_depth_bgr[0, 0, 0] == 17
        metadata = adapter.session_metadata(
            box_model=BoxModel(
                long_m=0.410,
                short_m=0.260,
                height_m=0.170,
                model_id="custom_recording_model",
            ),
            robot_state={
                "head_joints": None,
                "torso_joints": None,
                "base_state": None,
                "T_base_from_head": None,
            },
            nominal_transform={
                "target_frame": "head",
                "source_frame": "color",
                "translation_m": [0, 0, 0],
                "euler_zyx_deg": [0, 0, 0],
                "euler_input_order": ["roll", "pitch", "yaw"],
                "rotation_formula": "Rz(yaw) @ Ry(pitch) @ Rx(roll)",
            },
            table={"plane": None, "config_schema_version": 1},
        )
        assert metadata.camera_serial == "fake-D435"
        assert metadata.depth_to_color.source_stream == "depth"
        assert metadata.depth_to_color.target_stream == "color"
        assert metadata.color_to_depth.source_stream == "color"
        assert metadata.depth_scale_m == pytest.approx(0.001)
        assert metadata.capture_options["exposure"] == pytest.approx(1.0)
        assert metadata.capture_options["color_exposure"] == pytest.approx(2.0)
        assert metadata.capture_options["color_gain"] == pytest.approx(2.0)
        assert metadata.box_model.model_id == "custom_recording_model"
        assert metadata.box_model.height_m == pytest.approx(0.170)
    finally:
        adapter.stop()
