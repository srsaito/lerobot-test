import time
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.scripts.control_robot import busy_wait

from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

leader_config = DynamixelMotorsBusConfig(
    port="/dev/tty.usbmodem101",
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl330-m077"),
        "shoulder_lift": (2, "xl330-m077"),
        "elbow_flex": (3, "xl330-m077"),
        "wrist_flex": (4, "xl330-m077"),
        "wrist_roll": (5, "xl330-m077"),
        "gripper": (6, "xl330-m077"),
    },
)

follower_config = DynamixelMotorsBusConfig(
    port="/dev/tty.usbmodem1101",
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl430-w250"),
        "shoulder_lift": (2, "xl430-w250"),
        "elbow_flex": (3, "xl330-m288"),
        "wrist_flex": (4, "xl330-m288"),
        "wrist_roll": (5, "xl330-m288"),
        "gripper": (6, "xl330-m288"),
    },
)

leader_arm = DynamixelMotorsBus(leader_config)
follower_arm = DynamixelMotorsBus(follower_config)

from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera

robot_config = KochRobotConfig(
    leader_arms={"main": leader_config},
    follower_arms={"main": follower_config},
    max_relative_target=0.1,  # Limit movement to 0.1 degrees per step
    cameras={
        "overhead_cam": OpenCVCameraConfig(2, fps=30, width=640, height=480),
        "side_cam": OpenCVCameraConfig(0, fps=30, width=640, height=480)
    },
)
robot = ManipulatorRobot(robot_config)

robot.connect()


inference_time_s = 60
fps = 30
device = "mps"  # TODO: On Mac, use "mps" or "cpu"

ckpt_path = "outputs/train/act_koch_test/checkpoints/last/pretrained_model"
policy = ACTPolicy.from_pretrained(ckpt_path)
policy.to(device)

for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)

    # Compute the next action with the policy
    # based on the current observation
    action = policy.select_action(observation)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    action = action.to("cpu")
    # Order the robot to move
    robot.send_action(action)

    dt_s = time.perf_counter() - start_time
    # busy_wait(1 / fps - dt_s)
    busy_wait(.5)
