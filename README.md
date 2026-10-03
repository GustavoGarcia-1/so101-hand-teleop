# SO-101 webcam hand teleoperation

Control a calibrated SO-101 follower from a laptop webcam. In `gesture-depth`
mode, a closed fist closes the gripper, an open palm opens it, moving your hand
toward the camera lowers the arm, and moving away raises it. Hand movement in
the image moves the arm sideways and forward/back. Wrist roll stays at its
measured angle because this arm has a physical stop. The original pinch mode
is available with `--control-mode follow-hand`.

Tested on macOS Apple Silicon, Python 3.10, LeRobot 0.4.4 at commit
`6600b60e7f5cc7476ddc34beaaf0e0692f82e4b6`, MediaPipe 0.10.35,
OpenCV 4.12.0.88, and Placo 0.9.26. The local serial retry patch is included
so a fresh checkout matches the working setup. The robot must have six
configured motors and a saved LeRobot follower calibration.

## Install on a new Mac

Clone this repository into your home directory, then install the pinned
LeRobot checkout beside it:

```bash
cd ~
git clone https://github.com/GustavoGarcia-1/so101-hand-teleop.git
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 6600b60e7f5cc7476ddc34beaaf0e0692f82e4b6
git apply ../so101-hand-teleop/lerobot_serial_retries.patch
conda create -n lerobot python=3.10 -y
conda activate lerobot
python -m pip install -e '.[feetech,placo-dep]'
cd ../so101-hand-teleop
python -m pip install -r requirements.txt
python -m pip check
```

If you already have the working `lerobot` Conda environment, use it directly:

```bash
conda activate lerobot
cd ~/so101-hand-teleop
python -m pip install -r requirements.txt
```

On macOS, grant Camera access to your terminal in **System Settings → Privacy
& Security → Camera**. The hand/gesture models and the SO-101 kinematics-only
URDF download into `~/.cache/lerobot` on first use. They are not committed to
this repository.

## Preview without the robot

```bash
conda activate lerobot
cd ~/so101-hand-teleop
python teleoperate.py --control-mode gesture-depth
```

Show a closed fist and an open palm; the camera window should label each.
If the preview is black, close other camera apps and try `--camera-index 1`.
Press `Q` in the focused camera window to quit.

## Connect and run the robot

The examples below use the tested **12 V Pro follower**, calibration ID
`triton_follower`, and its current macOS serial port. Confirm your own motor
and adapter ratings before using that voltage. The port may change after a
reconnect; `lerobot-find-port` identifies it.

```bash
conda activate lerobot
cd ~/so101-hand-teleop
lerobot-find-port
python check_power.py \
  --port /dev/tty.usbmodem5A7A0570191 \
  --robot-id triton_follower
```

All six readings should be near 12 V for this Pro follower. Then run:

```bash
python teleoperate.py \
  --control-mode gesture-depth --source webcam \
  --port /dev/tty.usbmodem5A7A0570191 \
  --robot-id triton_follower \
  --expected-bus-voltage 12 --move-to-start \
  --scale 0.2 --lift-scale 0.3 --workspace 0.12 \
  --max-step 0.002 --max-joint-step 1
```

The arm first moves to a central body pose while holding wrist roll. It then
starts **disengaged**. Hold a fist in view for a few frames and press `E` in the
camera window. Open palm opens the gripper; fist closes it. Move toward the
camera to lower and away to raise. Press `E` to disengage, bring your hand back
to the middle of the frame, and press `E` again to continue from the arm's
current pose. Loss of hand tracking also disengages; re-engage with `E` after
your hand returns. Press `Q` to quit and release motor torque. Keep the arm's
path clear while testing.

There is **no default floor-height lock**. The gripper can contact the table.
The per-engagement `--workspace 0.12` still limits travel to 12 cm from the
pose where `E` was pressed, so use the clutch to continue beyond it. If you
want a height floor, add `--min-height HEIGHT_IN_METERS`. `check_pose.py`
reads the current gripper model height without commanding movement:

```bash
python check_pose.py \
  --port /dev/tty.usbmodem5A7A0570191 \
  --robot-id triton_follower
```

The model height is relative to the robot base; it is not a table distance
sensor. The script still enforces joint limits, per-frame movement limits, and
the fixed wrist-roll angle.

## Tuning and alternatives

- `--lift-scale 0.3`: how much hand toward/away changes robot height. Start
  lower for finer table moves; increase gradually if motion is too small.
- `--workspace 0.12`: maximum offset from each engage pose, in meters. Use `E`
  to clutch and continue.
- `--max-step 0.002 --max-joint-step 1`: Cartesian and joint movement limits
  per frame. Smaller values slow the arm.
- `--hand-smoothing 0.12 --depth-smoothing 0.25 --deadband 0.01`: webcam input
  filtering. Larger smoothing values reduce jitter but add lag.
- `--control-mode follow-hand`: original mode with pinching for the gripper.
  In this mode, hand toward/away controls reach rather than height.
- `--source realsense`: optional Intel RealSense D435 input. Its measured hand
  depth should be steadier than the webcam's apparent-size estimate, but the
  `pyrealsense2` driver must be installed separately. D435 support has not
  been tested on this Mac.

If the follower has no saved calibration, first use LeRobot's calibration
command with your actual port and ID:

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemYOURPORT \
  --robot.id=YOUR_FOLLOWER_ID
```

## Publish your copy to GitHub

From the directory containing this README and the three Python scripts:

```bash
git init -b main
git add README.md requirements.txt .gitignore teleoperate.py check_power.py check_pose.py lerobot_serial_retries.patch
git commit -m "Initial SO-101 hand teleoperation"
gh auth login
gh repo create so101-hand-teleop --public --source=. --remote=origin --push
```

To publish later edits:

```bash
git add README.md requirements.txt teleoperate.py check_power.py check_pose.py
git commit -m "Update hand teleoperation"
git push
```
