from parcel_pose.recording import recording_summary, replay_session, write_session

from tests.test_recording import make_frame, make_metadata


def test_two_replays_are_identical(tmp_path):
    path = tmp_path / "session"
    write_session(path, make_metadata(), [make_frame(0), make_frame(1)])
    assert replay_session(path) == replay_session(path)
    assert recording_summary(path) == recording_summary(path)


def test_processor_replay_order_is_deterministic(tmp_path):
    path = tmp_path / "session"
    write_session(path, make_metadata(), [make_frame(0), make_frame(1)])

    def processor(frame, metadata):
        return {
            "serial": metadata.camera_serial,
            "frame": frame.depth_frame_number,
            "first_depth": frame.raw_depth_z16[0, 0],
        }

    assert replay_session(path, processor) == [
        {"serial": "D435-test-001", "frame": 10, "first_depth": 0},
        {"serial": "D435-test-001", "frame": 11, "first_depth": 1},
    ]
