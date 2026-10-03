"""Control an SO-101/100 follower with one hand and a webcam or RealSense D435.

Install into the existing LeRobot environment as described in README.md.
Run preview: python teleoperate.py --control-mode gesture-depth
Run robot: python teleoperate.py --port /dev/tty.usbmodem... --robot-id my_arm

The camera window must have focus. Press E to engage/disengage, Q to quit. Every
engagement captures both the current hand pose and the current robot pose.
"""

from __future__ import annotations

import argparse
import shutil
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
START_BODY_JOINTS = np.array([0.0, -40.0, 60.0, -20.0])
MODEL_PATH = Path.home() / ".cache" / "lerobot" / "hand_landmarker.task"
GESTURE_MODEL_PATH = Path.home() / ".cache" / "lerobot" / "gesture_recognizer.task"
SO101_URDF_URL = (
    "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/"
    "Simulation/SO101/so101_new_calib.urdf"
)
SO101_URDF_PATH = Path.home() / ".cache" / "lerobot" / "so101_kinematics.urdf"


def bounded_target(start: np.ndarray, hand_delta: np.ndarray, scale: float, limit: float) -> np.ndarray:
    """Map camera right/up/toward-camera onto robot left/up/forward, within a local box."""
    offset = np.array([-hand_delta[2], -hand_delta[0], -hand_delta[1]]) * scale
    return start + np.clip(offset, -limit, limit)


def gesture_depth_target(
    start: np.ndarray, hand_delta: np.ndarray, scale: float, lift_scale: float, limit: float
) -> np.ndarray:
    """Screen up reaches forward; camera toward lowers; camera away raises."""
    offset = np.array(
        [-hand_delta[1] * scale, -hand_delta[0] * scale, hand_delta[2] * lift_scale]
    )
    return start + np.clip(offset, -limit, limit)


def pinch_fraction(landmarks) -> float:
    """0 means fingertips together; 1 means well separated, normalized to palm size."""
    thumb, index, wrist, middle = (landmarks[i] for i in (4, 8, 0, 9))
    tip_distance = np.hypot(thumb.x - index.x, thumb.y - index.y)
    palm_size = max(np.hypot(wrist.x - middle.x, wrist.y - middle.y), 1e-3)
    return float(np.clip((tip_distance / palm_size - 0.3) / 0.9, 0, 1))


def hand_position(landmarks, depth_m: float | None) -> np.ndarray:
    """Normalized image x/y and camera depth; webcam z is inverse apparent palm size."""
    wrist, index, pinky = (landmarks[i] for i in (0, 5, 17))
    x = (index.x + pinky.x + wrist.x) / 3
    y = (index.y + pinky.y + wrist.y) / 3
    palm_width = max(np.hypot(index.x - pinky.x, index.y - pinky.y), 0.02)
    z = depth_m if depth_m is not None else 0.08 / palm_width
    return np.array([x, y, z], dtype=float)


def smooth_value(previous, current, elapsed: float, time_constant: float):
    """Low-pass filter that behaves consistently across different camera frame rates."""
    if previous is None or time_constant == 0:
        return np.array(current, copy=True)
    alpha = 1.0 - np.exp(-max(elapsed, 0.0) / time_constant)
    return previous + alpha * (current - previous)


def apply_deadband(delta: np.ndarray, width: float) -> np.ndarray:
    """Ignore small camera movements without a discontinuity at the threshold."""
    return np.sign(delta) * np.maximum(np.abs(delta) - width, 0.0)


def ensure_model(path: Path, url: str = MODEL_URL) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe model to {path}...")
    temporary = path.with_suffix(".download")
    try:
        with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_so101_urdf(path: Path) -> None:
    """Cache the official kinematic model without mesh references Placo cannot resolve."""
    if path.exists():
        return
    print(f"Downloading SO-101 kinematic model to {path}...")
    with urllib.request.urlopen(SO101_URDF_URL, timeout=30) as response:
        root = ET.fromstring(response.read())
    for link in root.findall("link"):
        for child in list(link):
            if child.tag in ("visual", "collision"):
                link.remove(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    try:
        ET.ElementTree(root).write(temporary, encoding="utf-8", xml_declaration=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def joint_limits(urdf: Path) -> dict[str, tuple[float, float]]:
    """Read revolute joint limits from the matching URDF, returned in degrees."""
    root = ET.parse(urdf).getroot()
    limits = {}
    for joint in root.findall("joint"):
        if joint.get("name") not in JOINTS[:-1]:
            continue
        limit = joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"URDF has no bounds for {joint.get('name')}")
        limits[joint.get("name")] = (
            np.rad2deg(float(limit.get("lower"))),
            np.rad2deg(float(limit.get("upper"))),
        )
    if set(limits) != set(JOINTS[:-1]):
        raise ValueError("URDF is missing one or more SO arm joints")
    return limits


def start_pose_problem(
    joints: np.ndarray, pose: np.ndarray, limits: dict[str, tuple[float, float]], min_height: float | None
) -> str | None:
    """Give a clear reason to avoid arming from a cramped or model-invalid pose."""
    if min_height is not None and pose[2, 3] < min_height - 0.05:
        return f"gripper height {pose[2, 3]:.3f} m is too far below {min_height:.3f} m"
    outside = [
        name
        for i, name in enumerate(JOINTS[:4])
        if not limits[name][0] - 2 <= joints[i] <= limits[name][1] + 2
    ]
    if outside:
        return "joints outside model limits: " + ", ".join(outside)
    return None


def move_to_start(robot, kinematics, min_height: float | None) -> None:
    """Slowly move the arm body to a central pose without rotating the wrist roll."""
    obs = robot.get_observation()
    initial = np.array([obs[f"{name}.pos"] for name in JOINTS], dtype=float)
    stages = (np.r_[START_BODY_JOINTS, initial[4]],)
    for stage_number, goal in enumerate(stages, start=1):
        stage_obs = robot.get_observation()
        stage_start = np.array([stage_obs[f"{name}.pos"] for name in JOINTS[:-1]])
        # A user-supplied floor is checked along the proposed joint-space path.
        if min_height is not None:
            starting_height = float(kinematics.forward_kinematics(stage_start)[2, 3])
            previous_height = starting_height
            for fraction in np.linspace(0, 1, 41):
                sample = stage_start + fraction * (goal - stage_start)
                pose = kinematics.forward_kinematics(sample)
                height = float(pose[2, 3])
                if starting_height >= min_height and height < min_height:
                    raise RuntimeError(
                        f"Start-pose path crosses below {min_height:.2f} m at stage {stage_number}"
                    )
                if starting_height < min_height and height < previous_height - 0.005:
                    raise RuntimeError("Start-pose path descends while the gripper is near the base")
                previous_height = height
        print("Moving to start pose with wrist roll held. Press Ctrl+C to stop.")
        deadline = time.monotonic() + 60
        last_progress = time.monotonic()
        last_report = 0.0
        progress_anchor = stage_start[:4].copy()
        path_fraction = 0.0
        largest_joint_move = float(np.max(np.abs(goal[:4] - stage_start[:4])))
        while True:
            tick = time.monotonic()
            obs = robot.get_observation()
            measured = np.array([obs[f"{name}.pos"] for name in JOINTS], dtype=float)
            error = goal[:4] - measured[:4]
            largest_error = float(np.max(np.abs(error)))
            # These servos may settle a little short of the commanded position
            # under the arm's weight. Two degrees is adequate for a start pose.
            if largest_error <= 2.0:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Could not reach start pose within 60 seconds")
            if np.max(np.abs(measured[:4] - progress_anchor)) >= 0.25:
                progress_anchor = measured[:4].copy()
                last_progress = time.monotonic()
            if time.monotonic() - last_report > 2.0:
                print(f"  Largest joint error: {largest_error:.1f} deg")
                last_report = time.monotonic()
            if time.monotonic() - last_progress > 4.0:
                torque = robot.bus.sync_read("Torque_Enable", num_retry=2)
                voltage = robot.bus.sync_read("Present_Voltage", num_retry=2)
                status = robot.bus.sync_read("Status", num_retry=2)
                current = robot.bus.sync_read("Present_Current", num_retry=2)
                load = robot.bus.sync_read("Present_Load", num_retry=2)
                temperature = robot.bus.sync_read("Present_Temperature", num_retry=2)
                joint_errors = {
                    name: round(float(error[i]), 1) for i, name in enumerate(JOINTS[:4])
                }
                raise RuntimeError(
                    f"Joints have stopped progressing. Joint errors={joint_errors}; "
                    f"torque={torque}; bus voltage raw={voltage}; status={status}; "
                    f"current raw={current}; load raw={load}; temperature C={temperature}. "
                    "Check for a blocked joint or motor fault."
                )
            # Advance the commanded pose independently of the measured pose.
            # A measured+small-step target lets a position servo creep without
            # ever accumulating enough position error to drive the arm.
            if largest_joint_move > 0:
                candidate_fraction = min(1.0, path_fraction + 0.6 / largest_joint_move)
                candidate = stage_start + candidate_fraction * (goal - stage_start)
                if np.max(np.abs(candidate[:4] - measured[:4])) <= 3.0:
                    path_fraction = candidate_fraction
            commanded = stage_start + path_fraction * (goal - stage_start)
            action = {
                f"{name}.pos": float(commanded[i])
                for i, name in enumerate(JOINTS[:4])
            }
            action["wrist_roll.pos"] = float(measured[4])
            action["gripper.pos"] = float(measured[5])
            robot.send_action(action)
            time.sleep(max(0.05 - (time.monotonic() - tick), 0))
    print("Start pose reached. Hand tracking is DISENGAGED until you press E.")


def solve_bounded_step(kinematics, joints, desired, reference_rotation, limits):
    """Find a nearby reachable target, backing off instead of jumping to an IK edge."""
    current = kinematics.forward_kinematics(joints)
    for fraction in (1.0, 0.5, 0.25, 0.125):
        candidate = current[:3, 3] + fraction * (desired - current[:3, 3])
        pose = np.eye(4)
        pose[:3, :3] = reference_rotation
        pose[:3, 3] = candidate
        # RobotKinematics sets the joint guess but does not refresh Placo's
        # kinematic state until solving; refresh it for every retry here.
        kinematics.forward_kinematics(joints)
        solved = kinematics.inverse_kinematics(joints, pose)
        if not np.all(np.isfinite(solved)):
            continue
        # The physical wrist-roll stop may be narrower than the URDF range.
        # Keep this joint at its measured angle during hand translation.
        solved[4] = joints[4]
        if any(
            not limits[name][0] <= solved[i] <= limits[name][1]
            for i, name in enumerate(JOINTS[:4])
        ):
            continue
        if np.max(np.abs(solved - joints)) > 15.0:
            continue
        achieved = kinematics.forward_kinematics(solved)
        if np.linalg.norm(achieved[:3, 3] - candidate) > 0.01:
            continue
        return solved, candidate
    return None


class VideoSource:
    def __init__(self, source: str, camera_index: int):
        import cv2

        self.source = source
        self.cv2 = cv2
        self.pipeline = None
        self.capture = None
        if source == "webcam":
            self.capture = cv2.VideoCapture(camera_index)
            if not self.capture.isOpened():
                raise RuntimeError(f"Cannot open webcam index {camera_index}; check camera permission")
        else:
            try:
                import pyrealsense2 as rs
            except ImportError as exc:
                raise RuntimeError(
                    "Install the LeRobot RealSense extra: pip install -e '.[intelrealsense]'"
                ) from exc
            self.rs = rs
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)

    def read(self):
        if self.capture is not None:
            ok, image = self.capture.read()
            if not ok:
                raise RuntimeError("Webcam stopped providing frames")
            return image, None
        frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms=1000))
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("RealSense stopped providing color or depth frames")
        return np.asanyarray(color.get_data()), depth

    def close(self):
        if self.capture is not None:
            self.capture.release()
        if self.pipeline is not None:
            self.pipeline.stop()


def palm_depth(depth_frame, landmarks, width: int, height: int) -> float | None:
    """Median of valid aligned depth readings near the palm center, in meters."""
    if depth_frame is None:
        return None
    palm = landmarks[9]
    px = int(np.clip(palm.x * width, 0, width - 1))
    py = int(np.clip(palm.y * height, 0, height - 1))
    values = [
        depth_frame.get_distance(x, y)
        for y in range(max(0, py - 3), min(height, py + 4), 2)
        for x in range(max(0, px - 3), min(width, px + 4), 2)
    ]
    valid = [v for v in values if 0.2 < v < 2.0]
    return float(np.median(valid)) if len(valid) >= 3 else None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("webcam", "realsense"), default="webcam")
    parser.add_argument(
        "--control-mode",
        choices=("follow-hand", "gesture-depth"),
        default="follow-hand",
        help="Use fist/open-palm gripper gestures and camera depth for vertical movement",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--port", help="Serial port of calibrated follower; omitted means preview only")
    parser.add_argument(
        "--robot-id", default="my_awesome_follower_arm", help="Existing LeRobot calibration ID"
    )
    parser.add_argument("--robot-type", choices=("so101", "so100"), default="so101")
    parser.add_argument("--urdf", type=Path, help="SO-101 or SO-100 URDF for inverse kinematics")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--gesture-model", type=Path, default=GESTURE_MODEL_PATH)
    parser.add_argument("--scale", type=float, default=0.35, help="Robot meters per camera-space unit")
    parser.add_argument(
        "--lift-scale",
        type=float,
        help="Robot vertical meters per camera-depth unit in gesture-depth mode; default 0.3 for webcam or 0.7 for RealSense",
    )
    parser.add_argument("--workspace", type=float, default=0.12, help="Max offset in meters from engage pose")
    parser.add_argument(
        "--min-height", type=float, help="Optional minimum EE height above robot base, meters"
    )
    parser.add_argument(
        "--hand-smoothing", type=float, default=0.12, help="Hand x/y and pinch filter time constant, seconds"
    )
    parser.add_argument(
        "--depth-smoothing", type=float, default=0.25, help="Hand depth filter time constant, seconds"
    )
    parser.add_argument(
        "--deadband", type=float, default=0.01, help="Ignore tiny hand-position changes near the engage pose"
    )
    parser.add_argument(
        "--max-step", type=float, default=0.008, help="Max end-effector move per frame, meters"
    )
    parser.add_argument(
        "--max-joint-step", type=float, default=3.0, help="Max per-joint move per frame, degrees"
    )
    parser.add_argument("--gripper-open", type=float, default=100.0)
    parser.add_argument("--gripper-closed", type=float, default=0.0)
    parser.add_argument(
        "--expected-bus-voltage",
        type=float,
        help="Check the motor bus voltage before sending any motion command (for example, 12)",
    )
    parser.add_argument(
        "--move-to-start",
        action="store_true",
        help="Slowly raise/center the arm to a known pose before hand tracking",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        args.scale <= 0
        or args.workspace <= 0
        or args.max_step <= 0
        or args.max_joint_step <= 0
        or (args.min_height is not None and args.min_height < 0)
        or args.hand_smoothing < 0
        or args.depth_smoothing < 0
        or args.deadband < 0
    ):
        raise ValueError("scale, workspace, max-step, and max-joint-step must be positive")
    if args.port and args.urdf is None:
        if args.robot_type == "so100":
            raise ValueError("SO-100 robot mode needs --urdf pointing to a matching SO-100 URDF")
        args.urdf = SO101_URDF_PATH
        ensure_so101_urdf(args.urdf)
    if args.port and not args.urdf.is_file():
        raise ValueError("--urdf must point to the matching SO arm URDF")
    if args.move_to_start and not args.port:
        raise ValueError("--move-to-start requires --port")
    if args.move_to_start and args.robot_type != "so101":
        raise ValueError("--move-to-start has only been checked for the SO-101 model")
    if args.expected_bus_voltage is not None and args.expected_bus_voltage <= 0:
        raise ValueError("--expected-bus-voltage must be positive")
    if args.expected_bus_voltage is not None and not args.port:
        raise ValueError("--expected-bus-voltage requires --port")
    if args.lift_scale is not None and args.lift_scale <= 0:
        raise ValueError("--lift-scale must be positive")
    if args.lift_scale is None:
        args.lift_scale = 0.3 if args.source == "webcam" else 0.7

    import cv2
    import mediapipe as mp

    if args.control_mode == "gesture-depth":
        ensure_model(args.gesture_model, GESTURE_MODEL_URL)
    else:
        ensure_model(args.model)
    video = VideoSource(args.source, args.camera_index)
    robot = None
    armed = False
    hand_origin = None
    ee_origin = None
    last_target = None
    last_commanded_joints = None
    last_timestamp = -1
    filtered_position = None
    filtered_pinch = None
    last_hand_time = None
    gripper_gesture = None
    gesture_candidate = None
    gesture_frames = 0
    try:
        if args.port:
            from lerobot.model.kinematics import RobotKinematics
            from lerobot.robots.so_follower import (
                SO100Follower,
                SO100FollowerConfig,
                SO101Follower,
                SO101FollowerConfig,
            )

            config_cls, robot_cls = (
                (SO101FollowerConfig, SO101Follower)
                if args.robot_type == "so101"
                else (SO100FollowerConfig, SO100Follower)
            )
            robot = robot_cls(config_cls(port=args.port, id=args.robot_id, max_relative_target=3.0))
            if not robot.calibration:
                raise RuntimeError(f"No saved calibration for robot ID {args.robot_id!r}; calibrate first")
            robot.connect(calibrate=False)
            if args.expected_bus_voltage is not None:
                raw_voltage = robot.bus.sync_read("Present_Voltage", num_retry=2)
                bus_voltage = {name: value / 10.0 for name, value in raw_voltage.items()}
                tolerance = max(1.0, args.expected_bus_voltage * 0.2)
                if any(
                    abs(value - args.expected_bus_voltage) > tolerance
                    for value in bus_voltage.values()
                ):
                    raise RuntimeError(
                        f"Motor bus voltage {bus_voltage} V is outside the expected "
                        f"{args.expected_bus_voltage:g} ± {tolerance:g} V. "
                        "Check the adapter and its connection before moving the arm."
                    )
                print(f"Motor bus voltage: {bus_voltage} V")
            # SOFollower.configure() sets every joint's position P gain to 16
            # to reduce shakiness. On this follower, the loaded lift and elbow
            # need the motor's factory-default gain of 32 to move reliably.
            lift_elbow = ("shoulder_lift", "elbow_flex")
            with robot.bus.torque_disabled(list(lift_elbow)):
                present = robot.bus.sync_read("Present_Position", num_retry=2)
                robot.bus.sync_write(
                    "Goal_Position", {joint: present[joint] for joint in lift_elbow}
                )
                for joint in lift_elbow:
                    robot.bus.write("P_Coefficient", joint, 32)
            print("Shoulder lift and elbow position gain: 32")
            kin = RobotKinematics(str(args.urdf), joint_names=list(JOINTS[:-1]))
            limits = joint_limits(args.urdf)
            if args.move_to_start:
                move_to_start(robot, kin, args.min_height)

        if args.control_mode == "gesture-depth":
            options = mp.tasks.vision.GestureRecognizerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(args.gesture_model)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.65,
                min_hand_presence_confidence=0.65,
                min_tracking_confidence=0.65,
            )
            tracker_cls = mp.tasks.vision.GestureRecognizer
        else:
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.65,
                min_hand_presence_confidence=0.65,
                min_tracking_confidence=0.65,
            )
            tracker_cls = mp.tasks.vision.HandLandmarker
        print("Camera window: E engages/disengages or recenters, Q quits. Tracking loss disengages automatically.")
        if args.control_mode == "gesture-depth":
            print("Gesture mode: engage with a fist; open palm opens gripper; toward camera lowers, away raises.")
        print("PREVIEW ONLY" if robot is None else "Robot connected; starting DISENGAGED")
        with tracker_cls.create_from_options(options) as tracker:
            while True:
                frame, depth_frame = video.read()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = max(time.monotonic_ns() // 1_000_000, last_timestamp + 1)
                last_timestamp = timestamp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = (
                    tracker.recognize_for_video(mp_image, timestamp)
                    if args.control_mode == "gesture-depth"
                    else tracker.detect_for_video(mp_image, timestamp)
                )
                detected = bool(result.hand_landmarks)
                depth = None
                position = None
                pinch = None
                blocked = False
                floor_limited = False
                arm_height = None
                gesture_changed = False
                if detected:
                    landmarks = result.hand_landmarks[0]
                    if args.control_mode == "gesture-depth":
                        category = result.gestures[0][0] if result.gestures and result.gestures[0] else None
                        label = (
                            category.category_name
                            if category is not None
                            and category.score >= 0.65
                            and category.category_name in ("Closed_Fist", "Open_Palm")
                            else None
                        )
                        if label is None:
                            gesture_candidate = None
                            gesture_frames = 0
                        elif label == gesture_candidate:
                            gesture_frames += 1
                        else:
                            gesture_candidate = label
                            gesture_frames = 1
                        if gesture_frames >= 3 and gripper_gesture != label:
                            gripper_gesture = label
                            gesture_changed = True
                    depth = palm_depth(depth_frame, landmarks, frame.shape[1], frame.shape[0])
                    if args.source == "webcam" or depth is not None:
                        position = hand_position(landmarks, depth)
                        if args.control_mode == "follow-hand":
                            pinch = pinch_fraction(landmarks)
                        hand_time = time.monotonic()
                        elapsed = 0.0 if last_hand_time is None else hand_time - last_hand_time
                        if filtered_position is None:
                            filtered_position = position.copy()
                        else:
                            filtered_position[:2] = smooth_value(
                                filtered_position[:2], position[:2], elapsed, args.hand_smoothing
                            )
                            filtered_position[2] = smooth_value(
                                filtered_position[2], position[2], elapsed, args.depth_smoothing
                            )
                        if pinch is not None:
                            filtered_pinch = smooth_value(
                                filtered_pinch, pinch, elapsed, args.hand_smoothing
                            )
                        position = filtered_position.copy()
                        if filtered_pinch is not None:
                            pinch = float(filtered_pinch)
                        last_hand_time = hand_time
                    for landmark in landmarks:
                        cv2.circle(
                            frame,
                            (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])),
                            2,
                            (0, 255, 0),
                            -1,
                        )

                if position is None:
                    filtered_position = None
                    filtered_pinch = None
                    last_hand_time = None
                    gesture_candidate = None
                    gesture_frames = 0
                    gripper_gesture = None
                    if armed:
                        armed = False
                        last_commanded_joints = None
                        print("Tracking/depth lost: disengaged. Press E to recenter and re-engage.")

                if armed and position is not None and hand_origin is not None:
                    # Reject implausible tracking jumps instead of commanding a sudden move.
                    delta = position - hand_origin
                    if np.any(np.abs(delta[:2]) > 0.8) or abs(delta[2]) > 0.8:
                        armed = False
                        last_commanded_joints = None
                        print("Tracking jump: disengaged")
                    elif robot is not None:
                        delta = apply_deadband(delta, args.deadband)
                        obs = robot.get_observation()
                        measured = np.array([obs[f"{name}.pos"] for name in JOINTS], dtype=float)
                        current_pose = kin.forward_kinematics(measured[:5])
                        arm_height = float(current_pose[2, 3])
                        if gesture_changed:
                            # Opening/closing a fist changes apparent hand size
                            # on a webcam; keep it from also moving the arm up/down.
                            hand_origin[2] = position[2]
                            ee_origin[2, 3] = arm_height
                            last_target[2] = arm_height
                            delta[2] = 0.0
                        desired = (
                            gesture_depth_target(
                                ee_origin[:3, 3], delta, args.scale, args.lift_scale, args.workspace
                            )
                            if args.control_mode == "gesture-depth"
                            else bounded_target(ee_origin[:3, 3], delta, args.scale, args.workspace)
                        )
                        if args.min_height is not None:
                            floor_limited = desired[2] < args.min_height
                            desired[2] = max(desired[2], args.min_height)
                        step = desired - last_target
                        norm = np.linalg.norm(step)
                        if norm > args.max_step:
                            desired = last_target + step * args.max_step / norm
                        if args.control_mode == "gesture-depth":
                            grip = args.gripper_open if gripper_gesture == "Open_Palm" else args.gripper_closed
                        else:
                            grip = args.gripper_closed + pinch * (args.gripper_open - args.gripper_closed)
                        grip = float(np.clip(grip, 0, 100))
                        grip_goal = float(np.clip(grip, measured[5] - 3, measured[5] + 3))
                        result = solve_bounded_step(
                            kin, measured[:5], desired, ee_origin[:3, :3], limits
                        )
                        if result is None:
                            # Hold position at a workspace boundary. Moving the hand
                            # back toward the reachable region resumes motion.
                            blocked = True
                            last_target = kin.forward_kinematics(measured[:5])[:3, 3].copy()
                            last_commanded_joints = measured[:5].copy()
                            robot.bus.sync_write("Goal_Position", {"gripper": grip_goal})
                        else:
                            solved, accepted = result
                            if last_commanded_joints is None:
                                last_commanded_joints = measured[:5].copy()
                            proposed = last_commanded_joints + np.clip(
                                solved - last_commanded_joints,
                                -args.max_joint_step,
                                args.max_joint_step,
                            )
                            commanded = np.clip(proposed, measured[:5] - 3.0, measured[:5] + 3.0)
                            last_commanded_joints = commanded.copy()
                            action = {
                                f"{name}.pos": float(commanded[i])
                                for i, name in enumerate(JOINTS[:-1])
                            }
                            action["wrist_roll.pos"] = float(measured[4])
                            action["gripper.pos"] = grip_goal
                            robot.send_action(action)
                            last_target = accepted

                if floor_limited:
                    status = "FLOOR LIMIT"
                elif blocked:
                    status = "LIMIT: MOVE HAND BACK"
                elif armed:
                    status = "ENGAGED"
                elif position is not None:
                    status = "HAND FOUND"
                else:
                    status = "NO HAND / DEPTH"
                cv2.putText(
                    frame,
                    f"{status} | E engage/disengage | Q quit",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0) if armed else (0, 200, 255),
                    2,
                )
                if position is not None:
                    depth_label = "m" if depth is not None else "(webcam estimate)"
                    detail = (
                        f"gesture: {gripper_gesture or 'show fist'}"
                        if args.control_mode == "gesture-depth"
                        else f"pinch: {pinch:.2f}"
                    )
                    cv2.putText(
                        frame,
                        f"depth: {position[2]:.2f} {depth_label}  {detail}",
                        (12, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )
                if arm_height is not None:
                    floor_label = (
                        f"minimum: {args.min_height:.3f} m"
                        if args.min_height is not None
                        else "floor limit: off"
                    )
                    cv2.putText(
                        frame,
                        f"arm height: {arm_height:.3f} m | {floor_label} | E clutch",
                        (12, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )
                cv2.imshow("SO hand teleoperation", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("e"):
                    if armed:
                        armed = False
                        last_commanded_joints = None
                        print("Disengaged")
                    elif position is not None:
                        if args.control_mode == "gesture-depth" and gripper_gesture != "Closed_Fist":
                            print("Show a closed fist for a few frames, then press E to engage")
                            continue
                        hand_origin = position.copy()
                        if robot is not None:
                            obs = robot.get_observation()
                            measured = np.array([obs[f"{name}.pos"] for name in JOINTS], dtype=float)
                            last_commanded_joints = measured[:5].copy()
                            ee_origin = kin.forward_kinematics(measured[:5]).copy()
                            last_target = ee_origin[:3, 3].copy()
                            problem = start_pose_problem(
                                measured, ee_origin, limits, args.min_height
                            )
                            if problem:
                                print(f"Cannot engage: {problem}")
                                continue
                        armed = True
                        print("Engaged from current hand/robot pose" if robot else "Preview engaged")
                    else:
                        print("Show one hand clearly before engaging")
    finally:
        if robot is not None and robot.is_connected:
            robot.disconnect()
        video.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
