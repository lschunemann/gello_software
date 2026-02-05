#!/usr/bin/env python3
# Data collection script for GELLO teleoperation with ROS2
# Saves data in RLinf training format with normalized actions

import os
import pickle as pkl
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
import message_filters
from message_filters import ApproximateTimeSynchronizer
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import WrenchStamped, TwistStamped


@dataclass
class DataCollectionConfig:
    """Configuration for data collection."""
    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.51121, -0.00682, 0.275706, -3.12343, 0.02347, -0.01645])
    )
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.5, 0.5, 0.5])
    )
    camera_topic: str = "/zed/zed_node/left/image_rect_color"
    robot_joint_states_topic: str = "/joint_states"
    franka_wrench_topic: str = "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame"
    franka_twist_topic: str = "/franka_robot_state_broadcaster/desired_end_effector_twist"
    franka_gripper_topic: str = "/panda_gripper/joint_states"
    base_frame: str = "panda_link0"
    ee_frame: str = "panda_hand_tcp"
    control_rate: float = 1.0
    max_episode_steps: int = 100
    image_size: tuple = (128, 128)
    output_dir: str = "./collected_data"
    num_episodes: int = 20


class GelloDataCollector(Node):
    """ROS2 node for collecting teleoperation data with GELLO."""

    def __init__(self, config: DataCollectionConfig):
        super().__init__("gello_data_collector")
        self.config = config
        self.bridge = CvBridge()

        # Data storage
        self.data_list = []
        self.episode_data = []

        # State variables
        self._lock = threading.Lock()
        self._latest_image: Optional[np.ndarray] = None
        self._latest_robot_joints: Optional[np.ndarray] = None
        self._latest_franka_gripper: float = 0.0
        self._latest_ee_pose: Optional[np.ndarray] = None
        self._latest_tcp_force: np.ndarray = np.zeros(3)
        self._latest_tcp_torque: np.ndarray = np.zeros(3)
        self._latest_tcp_vel: np.ndarray = np.zeros(6)

        # Topic reception tracking
        self._received_image = False
        self._received_robot_joints = False
        self._received_franka_gripper = False
        self._received_tf = False

        # Episode state
        self._episode_active = False
        self._episode_step = 0
        self._episode_count = 0
        self._success_count = 0
        self._episode_success = False

        # TF2 for forward kinematics
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Synchronized subscribers
        self.image_sub = message_filters.Subscriber(self, Image, config.camera_topic)
        self.robot_joint_sub = message_filters.Subscriber(self, JointState, config.robot_joint_states_topic)
        self._sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.robot_joint_sub], queue_size=40, slop=0.1
        )
        self._sync.registerCallback(self._synced_callback)

        # Other subscribers
        self.franka_gripper_sub = self.create_subscription(
            JointState, config.franka_gripper_topic, self._franka_gripper_callback, 40
        )
        self.franka_wrench_sub = self.create_subscription(
            WrenchStamped, config.franka_wrench_topic, self._franka_wrench_callback, 40
        )
        self.franka_twist_sub = self.create_subscription(
            TwistStamped, config.franka_twist_topic, self._franka_twist_callback, 10
        )

        # Control timer
        self.control_timer = self.create_timer(1.0 / config.control_rate, self._control_loop)

        # Keyboard input thread
        self._running = True
        self._keyboard_thread = threading.Thread(target=self._keyboard_input_loop, daemon=True)
        self._keyboard_thread.start()

        self.get_logger().info("GelloDataCollector initialized")
        self.get_logger().info("Controls: SPACE=start/stop, S=success, Q=quit")

    def _is_ready(self) -> bool:
        return (self._received_image and self._received_robot_joints and
                self._received_tf and self._received_franka_gripper)

    def _get_ee_pose_from_tf(self) -> Optional[np.ndarray]:
        try:
            if not self._tf_buffer.can_transform(
                self.config.base_frame, self.config.ee_frame,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.0)
            ):
                return None
            transform = self._tf_buffer.lookup_transform(
                self.config.base_frame, self.config.ee_frame,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.0)
            )
            position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            quat = [
                transform.transform.rotation.x, transform.transform.rotation.y,
                transform.transform.rotation.z, transform.transform.rotation.w
            ]
            euler = R.from_quat(quat).as_euler('xyz')
            return np.concatenate([position, euler])
        except Exception:
            return None

    def _synced_callback(self, image_msg: Image, joint_msg: JointState):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            h, w = cv_image.shape[:2]
            crop_size = min(h, w)
            start_x, start_y = (w - crop_size) // 2, (h - crop_size) // 2
            cropped = cv_image[start_y:start_y+crop_size, start_x:start_x+crop_size]
            resized = cv2.resize(cropped, self.config.image_size)

            joint_data = list(zip(joint_msg.name, joint_msg.position))
            joint_data_sorted = sorted([jd for jd in joint_data if 'joint' in jd[0]], key=lambda x: x[0])
            joint_positions = [pos for _, pos in joint_data_sorted[:7]]

            ee_pose = self._get_ee_pose_from_tf()

            if len(joint_positions) == 7:
                with self._lock:
                    self._latest_image = resized
                    self._latest_robot_joints = np.array(joint_positions)
                    if ee_pose is not None:
                        self._latest_ee_pose = ee_pose
                    if not self._received_image:
                        self._received_image = True
                        self.get_logger().info(f"[OK] Camera")
                    if not self._received_robot_joints:
                        self._received_robot_joints = True
                        self.get_logger().info(f"[OK] Robot joints")
                    if ee_pose is not None and not self._received_tf:
                        self._received_tf = True
                        self.get_logger().info(f"[OK] TF")
        except Exception as e:
            self.get_logger().warning(f"Callback error: {e}")

    def _franka_gripper_callback(self, msg: JointState):
        try:
            gripper_pos = sum(msg.position[:2]) if len(msg.position) >= 2 else msg.position[0] * 2
            with self._lock:
                self._latest_franka_gripper = gripper_pos
                if not self._received_franka_gripper:
                    self._received_franka_gripper = True
                    self.get_logger().info(f"[OK] Gripper")
        except Exception:
            pass

    def _franka_wrench_callback(self, msg: WrenchStamped):
        with self._lock:
            self._latest_tcp_force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
            self._latest_tcp_torque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

    def _franka_twist_callback(self, msg: TwistStamped):
        with self._lock:
            self._latest_tcp_vel = np.array([
                msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
                msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z
            ])

    def _get_observation(self) -> Optional[dict]:
        """Get observation in RLinf training format."""
        with self._lock:
            if self._latest_image is None or self._latest_ee_pose is None:
                return None

            # States: gripper(1), tcp_force(3), tcp_pose(6), tcp_torque(3), tcp_vel(6) = 19D
            states = np.concatenate([
                np.array([self._latest_franka_gripper]),
                self._latest_tcp_force.copy(),
                self._latest_ee_pose.copy(),
                self._latest_tcp_torque.copy(),
                self._latest_tcp_vel.copy()
            ]).astype(np.float32)

            # Image: BGR to RGB, uint8 [0, 255]
            image = self._latest_image[..., ::-1].copy()

            return {
                "states": torch.tensor(states, dtype=torch.float32),
                "main_images": torch.tensor(image, dtype=torch.uint8),
            }

    def _compute_action(self, obs: dict, next_obs: dict) -> np.ndarray:
        """Compute 7D action: EE delta (6D) + gripper (1D)."""
        current_ee = obs["states"][4:10].numpy()
        next_ee = next_obs["states"][4:10].numpy()
        ee_delta = next_ee - current_ee

        # Wrap rotation deltas to [-pi, pi]
        ee_delta[3:6] = (ee_delta[3:6] + np.pi) % (2 * np.pi) - np.pi

        # Binary gripper action
        next_gripper = next_obs["states"][0].item()
        gripper_action = 1.0 if next_gripper > 0.04 else -1.0

        return np.concatenate([ee_delta, [gripper_action]]).astype(np.float32)

    def _check_success(self) -> bool:
        with self._lock:
            if self._latest_ee_pose is None:
                return False
            delta = np.abs(self._latest_ee_pose - self.config.target_ee_pose)
            return np.all(delta[:3] <= self.config.reward_threshold[:3])

    def _control_loop(self):
        if not self._episode_active:
            return

        obs = self._get_observation()
        if obs is None:
            return

        if self._episode_step > 0 and self._check_success() and not self._episode_success:
            self._episode_success = True
            self.get_logger().info(f"Success at step {self._episode_step}!")

        reward = 1.0 if self._episode_success else 0.0

        # Update previous transition
        if self.episode_data:
            prev = self.episode_data[-1]
            prev["next_obs"] = {k: v.clone() for k, v in obs.items()}
            prev["action"] = self._compute_action(prev["obs"], prev["next_obs"])

        self.episode_data.append({
            "obs": {k: v.clone() for k, v in obs.items()},
            "next_obs": None,
            "action": None,
            "reward": reward,
            "done": self._episode_success,
        })

        self._episode_step += 1
        if self._episode_step >= self.config.max_episode_steps:
            self._end_episode(self._episode_success)

    def _start_episode(self):
        if self._episode_active or not self._is_ready():
            return
        self._episode_active = True
        self._episode_step = 0
        self._episode_success = False
        self.episode_data = []
        self._episode_count += 1
        with self._lock:
            self._latest_ee_pose = None
        self.get_logger().info(f"Started episode {self._episode_count}")

    def _end_episode(self, is_success: bool):
        if not self._episode_active:
            return
        self._episode_active = False
        if is_success:
            self._success_count += 1

        # Convert episode data to RLinf format
        for trans in self.episode_data:
            if trans["next_obs"] is None:
                trans["next_obs"] = trans["obs"]
            if trans["action"] is None:
                trans["action"] = self._compute_action(trans["obs"], trans["next_obs"])

            self.data_list.append({
                "transitions": {"obs": trans["obs"], "next_obs": trans["next_obs"]},
                "action": torch.tensor(trans["action"], dtype=torch.float32),
                "rewards": torch.tensor([trans["reward"]], dtype=torch.float32),
                "dones": torch.tensor([trans["done"]], dtype=torch.float32),
                "terminations": torch.tensor([trans["done"]], dtype=torch.float32),
                "truncations": torch.tensor([0.0], dtype=torch.float32),
            })

        self.episode_data = []
        self.get_logger().info(f"Episode {self._episode_count}: {'SUCCESS' if is_success else 'TIMEOUT'} ({self._success_count}/{self._episode_count})")

        if self._success_count >= self.config.num_episodes:
            self._save_data()
            self._running = False

    def _save_data(self):
        """Save data with minmax normalized actions."""
        if not self.data_list:
            return

        # Collect all actions for normalization
        all_actions = np.array([d["action"].numpy() for d in self.data_list])

        # Compute minmax for 6D EE delta (exclude gripper)
        ee_actions = all_actions[:, :6]
        action_min = ee_actions.min(axis=0)
        action_max = ee_actions.max(axis=0)
        action_range = action_max - action_min
        action_range = np.where(action_range < 1e-8, 1.0, action_range)

        # Normalize and save
        normalized_data = []
        for d in self.data_list:
            d_new = d.copy()
            action = d["action"].numpy()

            # Normalize 6D EE to [-1, 1]
            ee_norm = 2 * (action[:6] - action_min) / action_range - 1
            ee_norm = np.clip(ee_norm, -1, 1)

            # Keep gripper as-is (already -1 or 1)
            normalized_action = np.concatenate([ee_norm, action[6:7]])
            d_new["action"] = torch.tensor(normalized_action, dtype=torch.float32)
            normalized_data.append(d_new)

        # Save data
        os.makedirs(self.config.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.config.output_dir, f"gello_data_{timestamp}.pkl")

        with open(filepath, "wb") as f:
            pkl.dump(normalized_data, f)

        # Save normalization params
        norm_params = {"min": action_min, "max": action_max, "range": action_range}
        with open(filepath.replace(".pkl", "_norm_params.pkl"), "wb") as f:
            pkl.dump(norm_params, f)

        self.get_logger().info(f"Saved {len(normalized_data)} transitions to {filepath}")

    def _keyboard_input_loop(self):
        import sys, select, termios, tty
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self._running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    if key == ' ':
                        if self._episode_active:
                            self._end_episode(self._episode_success)
                        else:
                            self._start_episode()
                    elif key == 'q':
                        if self._episode_active:
                            self._end_episode(False)
                        if self.data_list:
                            self._save_data()
                        self._running = False
                    elif key == 's' and self._episode_active:
                        self._end_episode(True)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def is_running(self) -> bool:
        return self._running


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./collected_data")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--control-rate", type=float, default=1.0)
    parser.add_argument("--camera-topic", default="/zed/zed_node/left/image_rect_color")
    parser.add_argument("--target-pose", type=float, nargs=6, default=[0.5, 0.0, 0.3, -3.14, 0.0, 0.0])
    parser.add_argument("--threshold", type=float, nargs=6, default=[0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    args = parser.parse_args()

    config = DataCollectionConfig(
        target_ee_pose=np.array(args.target_pose),
        reward_threshold=np.array(args.threshold),
        camera_topic=args.camera_topic,
        control_rate=args.control_rate,
        max_episode_steps=args.max_steps,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
    )

    rclpy.init()
    collector = GelloDataCollector(config)
    executor = SingleThreadedExecutor()
    executor.add_node(collector)

    print("\n" + "="*50)
    print("GELLO Data Collection")
    print("SPACE=start/stop  S=success  Q=quit")
    print("="*50 + "\n")

    try:
        while collector.is_running() and rclpy.ok():
            executor.spin_once()
    except KeyboardInterrupt:
        pass
    finally:
        collector.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
