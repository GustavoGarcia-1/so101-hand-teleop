"""Read the SO-101 gripper model height without commanding motor movement."""

import argparse

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from teleoperate import JOINTS, SO101_URDF_PATH, ensure_so101_urdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Follower motor bus serial port")
    parser.add_argument("--robot-id", default="triton_follower", help="Saved follower calibration ID")
    args = parser.parse_args()

    ensure_so101_urdf(SO101_URDF_PATH)
    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    if not robot.calibration:
        raise RuntimeError(f"No saved calibration for robot ID {args.robot_id!r}")
    try:
        robot.bus.connect(handshake=False)
        positions = robot.bus.sync_read("Present_Position", num_retry=2)
        kin = RobotKinematics(str(SO101_URDF_PATH), joint_names=list(JOINTS[:-1]))
        pose = kin.forward_kinematics([positions[name] for name in JOINTS[:-1]])
        print(f"Gripper model height above base: {pose[2, 3]:.3f} m")
        print("Joint positions (degrees):")
        for name in JOINTS[:-1]:
            print(f"  {name}: {positions[name]:.1f}")
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
