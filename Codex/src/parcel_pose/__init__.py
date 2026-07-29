"""D435 parcel-pose perception package.

The package core deliberately has no import-time dependency on pyrealsense2 or
rby1_sdk.  Import the submodule you need (``parcel_pose.estimator``,
``parcel_pose.pallet_runtime``, ...) rather than package-level re-exports, so a
perception-only caller never pays for the robot-control modules.
"""
