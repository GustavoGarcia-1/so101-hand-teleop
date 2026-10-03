"""Read the SO-101 follower motor bus voltage without commanding movement."""

import argparse

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Follower motor bus serial port")
    parser.add_argument("--robot-id", default="triton_follower", help="Saved follower calibration ID")
    args = parser.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    try:
        robot.bus.connect(handshake=False)
        voltage = robot.bus.sync_read("Present_Voltage", num_retry=2)
        for motor, raw in voltage.items():
            print(f"{motor}: {raw / 10:.1f} V")
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
