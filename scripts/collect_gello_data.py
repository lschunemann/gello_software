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
from sensor_msgs.msg import JointState, Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
from message_filters import ApproximateTimeSynchronizer
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import WrenchStamped, TwistStamped


@dataclass
class DataCollectionConfig:
    """Configuration for data collection."""
    task_type: str = "reach"  # "reach", "pick", or add new ones

    # Reach task config
    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.51121, -0.00682, 0.275706, -3.12343, 0.02347, -0.01645])
    )
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.5, 0.5, 0.5])
    )
    check_rotation_z: bool = False  # Also require rotation-z within reward_threshold[5]
    require_gripper_open: bool = False  # Also require gripper to be open for reach success

    # Pick task config
    object_z: float = 0.03
    lift_height: float = 0.10
    gripper_closed_min: float = 0.002
    gripper_closed_max: float = 0.035

    # Slide task config (AprilTag goal + HSV object)
    tag_family: str = "tagStandard41h12"
    goal_tag_id: int = 5     # Tag on the goal area
    green_hsv_low: list = field(default_factory=lambda: [35, 50, 50])
    green_hsv_high: list = field(default_factory=lambda: [85, 255, 255])
    min_contour_area: int = 50
    success_pixel_distance: float = 15.0
    _tag_detector: object = None  # lazy-initialized
    _goal_centroid: Optional[np.ndarray] = None  # cached, detected once

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
    extra_image_size: tuple = None  # If set (e.g. (224, 224)), saves a second pkl with this image size
    output_dir: str = "./collected_data"
    num_episodes: int = 20

    # Rich observation pkl (separate file with full-res rgb, depth, joints, gripper_pose)
    save_obs_pkl: bool = False
    depth_topic: str = "/zed/zed_node/depth/depth_registered"
    camera_info_topic: str = "/zed/zed_node/left/camera_info"
    # 4x4 camera-to-robot extrinsics, row-major flat list of 16 values (optional)
    camera_extrinsics: Optional[list] = None


# --- Success checkers: each takes (config, ee_pose, gripper_pos, image=None) -> bool ---

def _check_success_reach(config, ee_pose, gripper_pos, image=None, image_full=None):
    """Success = EE pose within threshold of target, optionally with gripper open."""
    delta = np.abs(ee_pose - config.target_ee_pose)
    success = bool(np.all(delta[:3] <= config.reward_threshold[:3]))
    if success and config.check_rotation_z:
        success = bool(delta[5] <= config.reward_threshold[5])
    if success and config.require_gripper_open:
        success = gripper_pos > 0.04  # sum of both finger positions
    return success


def _check_success_pick(config, ee_pose, gripper_pos, image=None, image_full=None):
    """Success = tcp_z >= lift_height (absolute z above table) while gripper is holding object."""
    tcp_z = ee_pose[2]
    is_lifted = tcp_z >= config.lift_height
    is_holding = config.gripper_closed_min <= gripper_pos <= config.gripper_closed_max
    return bool(is_lifted and is_holding)


def _detect_goal_tag(image_bgr, tag_detector, tag_id):
    """Detect goal AprilTag centroid in the image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detections = tag_detector.detect(gray)
    centers = [np.array(d.center) for d in detections if d.tag_id == tag_id]
    if not centers:
        return None
    return np.mean(centers, axis=0)


def _detect_green_object(image_bgr, hsv_low, hsv_high, min_area):
    """Detect green object centroid via HSV color detection."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return None
    largest = max(valid, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])


def _check_success_slide(config, ee_pose, gripper_pos, image=None, image_full=None):
    """Success = green object centroid is close to goal AprilTag centroid.

    Uses full-res image for detection when available, falls back to 128x128.
    """
    detect_image = image_full if image_full is not None else image
    if detect_image is None:
        return False

    # Lazy-init detector
    if not hasattr(config, '_tag_detector') or config._tag_detector is None:
        from pupil_apriltags import Detector
        config._tag_detector = Detector(families=config.tag_family, quad_decimate=1.0)

    # Detect goal centroid once and cache it (on full-res for reliable AprilTag detection)
    if config._goal_centroid is None:
        config._goal_centroid = _detect_goal_tag(
            detect_image, config._tag_detector, config.goal_tag_id
        )
        if config._goal_centroid is not None:
            print(f"[Slide] Detected goal tag (id={config.goal_tag_id}) at pixel "
                  f"({config._goal_centroid[0]:.1f}, {config._goal_centroid[1]:.1f})")
        else:
            return False

    object_centroid = _detect_green_object(
        detect_image, config.green_hsv_low, config.green_hsv_high, config.min_contour_area
    )
    if object_centroid is None:
        return False

    distance = np.linalg.norm(object_centroid - config._goal_centroid)
    # Scale success_pixel_distance if using full-res crop (threshold is tuned for 128x128)
    threshold = config.success_pixel_distance
    if image_full is not None:
        threshold *= image_full.shape[0] / 128.0
    return bool(distance < threshold)


SUCCESS_CHECKERS = {
    "reach": _check_success_reach,
    "pick": _check_success_pick,
    "slide": _check_success_slide,
}


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
        self._latest_image_extra: Optional[np.ndarray] = None  # Extra resolution image
        self._latest_image_full: Optional[np.ndarray] = None
        self._latest_robot_joints: Optional[np.ndarray] = None
        self._latest_franka_gripper: float = 0.0
        self._latest_gripper_joints: np.ndarray = np.zeros(2)
        self._latest_image_native: Optional[np.ndarray] = None  # full native res, no crop
        self._latest_depth: Optional[np.ndarray] = None
        self._camera_intrinsics: Optional[np.ndarray] = None
        self._latest_ee_pose: Optional[np.ndarray] = None
        self._latest_tcp_force: np.ndarray = np.zeros(3)
        self._latest_tcp_torque: np.ndarray = np.zeros(3)
        self._latest_tcp_vel: np.ndarray = np.zeros(6)

        # Topic reception tracking
        self._received_image = False
        self._received_robot_joints = False
        self._received_franka_gripper = False
        self._received_tf = False

        # Rich obs pkl state
        self._obs_episodes: list = []
        self._current_obs_episode: list = []

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

        # Optional: depth + camera_info for obs pkl
        if config.save_obs_pkl:
            if config.depth_topic:
                self.depth_sub = self.create_subscription(
                    Image, config.depth_topic, self._depth_callback, 10
                )
            if config.camera_info_topic:
                self.camera_info_sub = self.create_subscription(
                    CameraInfo, config.camera_info_topic, self._camera_info_callback, 1
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

            resized_full = cropped  # Native crop resolution for detection
            if self.config.save_obs_pkl:
                with self._lock:
                    self._latest_image_native = cv_image.copy()  # full native res, no crop
            resized_extra = None
            if self.config.extra_image_size is not None:
                resized_extra = cv2.resize(cropped, self.config.extra_image_size)

            if len(joint_positions) == 7:
                with self._lock:
                    self._latest_image = resized
                    self._latest_image_extra = resized_extra
                    self._latest_image_full = resized_full  # 512x512 for detection
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
            if len(msg.position) >= 2:
                fingers = np.array([msg.position[0], msg.position[1]])
            else:
                fingers = np.array([msg.position[0], msg.position[0]])
            with self._lock:
                self._latest_franka_gripper = float(fingers.sum())
                self._latest_gripper_joints = fingers
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

    def _depth_callback(self, msg: Image):
        try:
            if msg.encoding == "32FC1":
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            elif msg.encoding == "16UC1":
                depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
                depth = depth_mm.astype(np.float32) / 1000.0
            else:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)
            with self._lock:
                self._latest_depth = depth.copy()  # full native resolution, no crop
        except Exception as e:
            self.get_logger().warning(f"Depth callback error: {e}")

    def _camera_info_callback(self, msg: CameraInfo):
        if self._camera_intrinsics is not None:
            return  # Already captured once
        # Save raw K for the native resolution — no crop adjustment needed since
        # rgb and depth are saved at full native resolution.
        K = np.array(msg.k).reshape(3, 3).copy()
        with self._lock:
            self._camera_intrinsics = K
        self.get_logger().info(
            f"[OK] Intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
            f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}"
        )

    def _get_obs_step(self) -> Optional[dict]:
        """Build a rich obs dict for the separate obs pkl (called from _control_loop)."""
        with self._lock:
            if self._latest_image_native is None or self._latest_ee_pose is None:
                return None
            rgb = self._latest_image_native.copy()  # full native resolution BGR, no crop
            depth = self._latest_depth.copy() if self._latest_depth is not None else None
            intrinsics = self._camera_intrinsics.copy() if self._camera_intrinsics is not None else None
            gripper_joints = self._latest_gripper_joints.copy()
            gripper_open = bool(gripper_joints.sum() > 0.04)
            joints = self._latest_robot_joints.copy() if self._latest_robot_joints is not None else np.zeros(7)
            ee = self._latest_ee_pose.copy()

        T = np.eye(4)
        T[:3, :3] = R.from_euler('xyz', ee[3:]).as_matrix()
        T[:3, 3] = ee[:3]

        extrinsics = None
        if self.config.camera_extrinsics is not None:
            extrinsics = np.array(self.config.camera_extrinsics).reshape(4, 4)

        return {
            "rgb": rgb,                    # (H, W, 3) uint8 BGR, full native resolution
            "depth": depth,                # (H, W) float32 meters, full native resolution, or None
            "intrinsics": intrinsics,      # (3, 3) K matrix for native resolution, or None
            "extrinsics": extrinsics,      # (4, 4) T_cam_to_robot, or None
            "gripper_open": gripper_open,  # bool
            "gripper_joints": gripper_joints,  # (2,) float64, [left, right] finger pos in m
            "joints": joints,              # (7,) float64, joint angles in rad
            "gripper_pose": T,             # (4, 4) SE(3) base->EE
        }

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

            obs = {
                "states": torch.tensor(states, dtype=torch.float32),
                "main_images": torch.tensor(image, dtype=torch.uint8),
            }

            if self._latest_image_extra is not None:
                image_extra = self._latest_image_extra[..., ::-1].copy()
                obs["_extra_image"] = torch.tensor(image_extra, dtype=torch.uint8)

            return obs

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
            checker = SUCCESS_CHECKERS.get(self.config.task_type)
            if checker is None:
                raise ValueError(f"Unknown task_type '{self.config.task_type}'. "
                                 f"Available: {list(SUCCESS_CHECKERS.keys())}")
            return checker(self.config, self._latest_ee_pose, self._latest_franka_gripper,
                           image=self._latest_image, image_full=self._latest_image_full)

    def _control_loop(self):
        if not self._episode_active:
            return

        obs = self._get_observation()
        if obs is None:
            return

        if self.config.save_obs_pkl:
            obs_step = self._get_obs_step()
            if obs_step is not None:
                self._current_obs_episode.append(obs_step)

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
        self._current_obs_episode = []
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
            # Mark success on episode data (needed when 's' is pressed manually,
            # since _episode_success may not have been set by the auto-checker)
            if self.episode_data:
                self.episode_data[-1]["reward"] = 1.0
                self.episode_data[-1]["done"] = True

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
        if self.config.save_obs_pkl and self._current_obs_episode:
            self._obs_episodes.append(self._current_obs_episode)
        self._current_obs_episode = []
        self.get_logger().info(f"Episode {self._episode_count}: {'SUCCESS' if is_success else 'TIMEOUT'} ({self._success_count}/{self._episode_count})")

        if self._success_count >= self.config.num_episodes:
            self._save_data()
            self._running = False

    def _save_data(self):
        """Save data with raw (unnormalized) actions in absolute frame."""
        if not self.data_list:
            return

        os.makedirs(self.config.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.config.output_dir, f"gello_data_{timestamp}.pkl")

        # Save main pkl (128x128 images, without _extra_image key)
        main_data = []
        extra_data = []
        has_extra = self.config.extra_image_size is not None

        for sample in self.data_list:
            # Split out _extra_image from obs and next_obs
            main_sample = dict(sample)
            main_sample["transitions"] = {}
            for tkey in ["obs", "next_obs"]:
                if tkey in sample["transitions"]:
                    obs = dict(sample["transitions"][tkey])
                    extra_img = obs.pop("_extra_image", None)
                    main_sample["transitions"][tkey] = obs

                    if has_extra and extra_img is not None:
                        # Build extra sample lazily below
                        pass

            main_data.append(main_sample)

            if has_extra:
                # Build extra sample: same structure but main_images replaced with extra size
                extra_sample = dict(sample)
                extra_sample["transitions"] = {}
                for tkey in ["obs", "next_obs"]:
                    if tkey in sample["transitions"]:
                        obs = dict(sample["transitions"][tkey])
                        extra_img = obs.pop("_extra_image", None)
                        if extra_img is not None:
                            obs["main_images"] = extra_img
                        extra_sample["transitions"][tkey] = obs
                extra_data.append(extra_sample)

        with open(filepath, "wb") as f:
            pkl.dump(main_data, f)
        self.get_logger().info(f"Saved {len(main_data)} transitions to {filepath}")

        if self.config.save_obs_pkl and self._obs_episodes:
            obs_path = os.path.join(self.config.output_dir, f"drema_obs_{timestamp}.pkl")
            with open(obs_path, "wb") as f:
                pkl.dump(self._obs_episodes, f)
            self.get_logger().info(
                f"Saved {len(self._obs_episodes)} obs episodes to {obs_path}"
            )

            img_dir = os.path.join(self.config.output_dir, f"drema_obs_{timestamp}_images")
            os.makedirs(img_dir, exist_ok=True)
            for ep_idx, episode in enumerate(self._obs_episodes):
                ep_dir = os.path.join(img_dir, f"episode_{ep_idx:03d}")
                os.makedirs(ep_dir, exist_ok=True)
                for step_idx, obs_step in enumerate(episode):
                    cv2.imwrite(
                        os.path.join(ep_dir, f"step_{step_idx:04d}.png"),
                        obs_step["rgb"]  # already BGR, cv2.imwrite expects BGR
                    )
            total_imgs = sum(len(ep) for ep in self._obs_episodes)
            self.get_logger().info(f"Saved {total_imgs} images to {img_dir}/")

        if has_extra and extra_data:
            w, h = self.config.extra_image_size
            extra_path = os.path.join(self.config.output_dir, f"gello_data_{timestamp}_{w}x{h}.pkl")
            with open(extra_path, "wb") as f:
                pkl.dump(extra_data, f)
            self.get_logger().info(f"Saved {len(extra_data)} transitions ({w}x{h}) to {extra_path}")

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
    parser.add_argument("--task", default="reach", choices=list(SUCCESS_CHECKERS.keys()),
                        help="Task type for success condition")
    parser.add_argument("--output-dir", default="./collected_data")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--control-rate", type=float, default=1.0)
    parser.add_argument("--camera-topic", default="/zed/zed_node/left/image_rect_color")
    # Reach task args
    parser.add_argument("--target-pose", type=float, nargs=6, default=[0.5, 0.0, 0.3, -3.14, 0.0, 0.0])
    parser.add_argument("--threshold", type=float, nargs=6, default=[0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    # Pick task args
    parser.add_argument("--object-z", type=float, default=0.03)
    parser.add_argument("--lift-height", type=float, default=0.10)
    parser.add_argument("--gripper-closed-min", type=float, default=0.002)
    parser.add_argument("--gripper-closed-max", type=float, default=0.035)
    # Slide task args
    # Slide task args
    parser.add_argument("--success-pixel-distance", type=float, default=7.0,
                        help="Pixel distance threshold for slide task success (in 128x128 px)")
    parser.add_argument("--goal-tag-id", type=int, default=5, help="AprilTag ID for slide goal")
    parser.add_argument("--green-hsv-low", type=int, nargs=3, default=[35, 50, 50])
    parser.add_argument("--green-hsv-high", type=int, nargs=3, default=[85, 255, 255])
    parser.add_argument("--min-contour-area", type=int, default=50)
    parser.add_argument("--extra-image-size", type=int, default=None,
                        help="If set, also save a second pkl with images at this resolution (e.g. 224)")
    parser.add_argument("--check-rotation-z", action="store_true", default=False,
                        help="Also require rotation-z within threshold[5] for reach success")
    parser.add_argument("--require-gripper-open", action="store_true", default=False,
                        help="Also require gripper to be open for reach success")
    parser.add_argument("--save-obs-pkl", action="store_true", default=False,
                        help="Save a separate drema_obs_*.pkl with full-res rgb, depth, intrinsics, joints, gripper_pose")
    parser.add_argument("--depth-topic", default="/zed/zed_node/depth/depth_registered")
    parser.add_argument("--camera-info-topic", default="/zed/zed_node/left/camera_info")
    parser.add_argument("--camera-extrinsics", type=float, nargs=16, default=None,
                        metavar="V", help="4x4 T_cam_to_robot row-major (16 values)")
    args = parser.parse_args()

    extra_image_size = None
    if args.extra_image_size is not None:
        extra_image_size = (args.extra_image_size, args.extra_image_size)

    config = DataCollectionConfig(
        task_type=args.task,
        target_ee_pose=np.array(args.target_pose),
        reward_threshold=np.array(args.threshold),
        check_rotation_z=args.check_rotation_z,
        require_gripper_open=args.require_gripper_open,
        object_z=args.object_z,
        lift_height=args.lift_height,
        gripper_closed_min=args.gripper_closed_min,
        gripper_closed_max=args.gripper_closed_max,
        success_pixel_distance=args.success_pixel_distance,
        goal_tag_id=args.goal_tag_id,
        green_hsv_low=args.green_hsv_low,
        green_hsv_high=args.green_hsv_high,
        min_contour_area=args.min_contour_area,
        camera_topic=args.camera_topic,
        control_rate=args.control_rate,
        max_episode_steps=args.max_steps,
        extra_image_size=extra_image_size,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        save_obs_pkl=args.save_obs_pkl,
        depth_topic=args.depth_topic,
        camera_info_topic=args.camera_info_topic,
        camera_extrinsics=args.camera_extrinsics,
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
