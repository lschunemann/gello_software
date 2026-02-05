#!/usr/bin/env python3
"""Visualize collected GELLO data - images, poses, and actions."""

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import torch
import argparse


def load_data(filepath):
    with open(filepath, "rb") as f:
        data = pkl.load(f)
    print(f"Loaded {len(data)} samples from {filepath}")
    return data


def extract_obs(sample):
    """Extract observation from sample (handles both formats)."""
    if "transitions" in sample:
        obs = sample["transitions"]["obs"]
    else:
        obs = sample.get("obs", sample)
    return obs


def get_image(obs):
    """Extract image from observation."""
    if "main_images" in obs:
        img = obs["main_images"]
    elif "frames" in obs:
        img = obs["frames"].get("wrist_1", list(obs["frames"].values())[0])
    else:
        return None

    if isinstance(img, torch.Tensor):
        img = img.numpy()

    # Squeeze batch dim if present (e.g. rollout data has shape (1, 128, 128, 3))
    if img.ndim > 3:
        img = img.squeeze(0)

    # Handle different formats
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

    return img


def get_states(obs):
    """Extract states from observation."""
    if "states" in obs:
        states = obs["states"]
    elif "state" in obs:
        # Nested format - reconstruct flat states
        state = obs["state"]
        parts = []
        for key in ["gripper_position", "tcp_force", "tcp_pose", "tcp_torque", "tcp_vel"]:
            if key in state:
                val = state[key]
                if isinstance(val, torch.Tensor):
                    val = val.numpy()
                parts.append(val.flatten())
        states = np.concatenate(parts)
    else:
        return None

    if isinstance(states, torch.Tensor):
        states = states.numpy()
    # Squeeze batch dim if present (e.g. rollout data has shape (1, 19))
    if states.ndim > 1:
        states = states.squeeze(0)
    return states


def get_action(sample):
    """Extract action from sample."""
    action = sample.get("action")
    if action is None:
        return None
    if isinstance(action, torch.Tensor):
        action = action.numpy()
    return action


def get_reward(sample):
    """Extract reward from sample."""
    reward = sample.get("rewards", sample.get("reward"))
    if reward is None:
        return 0.0
    if isinstance(reward, torch.Tensor):
        return reward.item() if reward.numel() == 1 else reward.sum().item()
    return float(reward)


class DataVisualizer:
    def __init__(self, data):
        self.data = data
        self.idx = 0

        # Create figure
        self.fig = plt.figure(figsize=(14, 8))

        # Image subplot
        self.ax_img = self.fig.add_subplot(2, 2, 1)
        self.ax_img.set_title("Camera Image")

        # Pose subplot
        self.ax_pose = self.fig.add_subplot(2, 2, 2)
        self.ax_pose.set_title("TCP Pose (position + euler)")

        # Action subplot
        self.ax_action = self.fig.add_subplot(2, 2, 3)
        self.ax_action.set_title("Action (EE delta + gripper)")

        # Info text
        self.ax_info = self.fig.add_subplot(2, 2, 4)
        self.ax_info.axis('off')

        # Slider
        ax_slider = self.fig.add_axes([0.2, 0.02, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Sample', 0, len(data)-1, valinit=0, valstep=1)
        self.slider.on_changed(self.update)

        # Buttons
        ax_prev = self.fig.add_axes([0.05, 0.02, 0.05, 0.03])
        ax_next = self.fig.add_axes([0.9, 0.02, 0.05, 0.03])
        self.btn_prev = Button(ax_prev, '<')
        self.btn_next = Button(ax_next, '>')
        self.btn_prev.on_clicked(self.prev)
        self.btn_next.on_clicked(self.next)

        # Keyboard navigation
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update(0)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

    def update(self, val):
        self.idx = int(val)
        sample = self.data[self.idx]
        obs = extract_obs(sample)

        # Update image
        self.ax_img.clear()
        img = get_image(obs)
        if img is not None:
            self.ax_img.imshow(img)
            self.ax_img.set_title(f"Camera Image ({img.shape}, {img.dtype})")
        else:
            self.ax_img.set_title("No image")
        self.ax_img.axis('off')

        # Update pose
        self.ax_pose.clear()
        states = get_states(obs)
        if states is not None:
            # States: gripper(1), tcp_force(3), tcp_pose(6), tcp_torque(3), tcp_vel(6)
            gripper = states[0]
            tcp_pose = states[4:10]

            labels = ['x', 'y', 'z', 'rx', 'ry', 'rz']
            colors = ['r', 'g', 'b', 'orange', 'purple', 'cyan']
            bars = self.ax_pose.bar(labels, tcp_pose, color=colors)
            self.ax_pose.set_ylabel('Value')
            self.ax_pose.set_title(f'TCP Pose | Gripper: {gripper:.4f}')
            self.ax_pose.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

            # Add value labels
            for bar, val in zip(bars, tcp_pose):
                self.ax_pose.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                  f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        # Update action
        self.ax_action.clear()
        action = get_action(sample)
        if action is not None:
            if len(action) == 7:
                labels = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'grip']
            elif len(action) == 6:
                labels = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz']
            else:
                labels = [f'a{i}' for i in range(len(action))]
            colors = ['r', 'g', 'b', 'orange', 'purple', 'cyan', 'gray'][:len(action)]
            bars = self.ax_action.bar(labels, action, color=colors)
            self.ax_action.set_ylabel('Value')
            self.ax_action.set_title(f'Action ({len(action)}D)')
            self.ax_action.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            self.ax_action.set_ylim(-1., 1.)

            for bar, val in zip(bars, action):
                self.ax_action.text(bar.get_x() + bar.get_width()/2,
                                    bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.15,
                                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        # Update info
        self.ax_info.clear()
        self.ax_info.axis('off')
        reward = get_reward(sample)
        done = sample.get("dones", sample.get("done", 0))
        if isinstance(done, torch.Tensor):
            done = done.item()

        info_text = f"Sample: {self.idx}/{len(self.data)-1}\n"
        info_text += f"Reward: {reward}\n"
        info_text += f"Done: {bool(done)}\n"

        if states is not None:
            info_text += f"\nStates ({len(states)}D):\n"
            info_text += f"  Gripper: {states[0]:.4f}\n"
            info_text += f"  Force: [{states[1]:.2f}, {states[2]:.2f}, {states[3]:.2f}]\n"
            info_text += f"  Pose: [{states[4]:.3f}, {states[5]:.3f}, {states[6]:.3f}]\n"
            info_text += f"        [{states[7]:.3f}, {states[8]:.3f}, {states[9]:.3f}]\n"
            info_text += f"  Torque: [{states[10]:.2f}, {states[11]:.2f}, {states[12]:.2f}]\n"

        self.ax_info.text(0.1, 0.9, info_text, transform=self.ax_info.transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace')

        self.fig.canvas.draw_idle()

    def prev(self, event):
        if self.idx > 0:
            self.slider.set_val(self.idx - 1)

    def next(self, event):
        if self.idx < len(self.data) - 1:
            self.slider.set_val(self.idx + 1)

    def on_key(self, event):
        if event.key == 'left':
            self.prev(None)
        elif event.key == 'right':
            self.next(None)

    def show(self):
        plt.show()


def print_summary(data):
    """Print summary statistics of the data."""
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)

    rewards = [get_reward(d) for d in data]
    print(f"Total samples: {len(data)}")
    print(f"Rewards: min={min(rewards):.2f}, max={max(rewards):.2f}, sum={sum(rewards):.2f}")
    print(f"Successes (reward>0): {sum(1 for r in rewards if r > 0)}")

    # Check first sample structure
    sample = data[0]
    print(f"\nSample keys: {list(sample.keys())}")

    obs = extract_obs(sample)
    print(f"Obs keys: {list(obs.keys())}")

    img = get_image(obs)
    if img is not None:
        print(f"Image: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")

    states = get_states(obs)
    if states is not None:
        print(f"States: shape={states.shape}, dtype={states.dtype}")

    action = get_action(sample)
    if action is not None:
        print(f"Action: shape={action.shape}, dtype={action.dtype}, range=[{action.min():.3f}, {action.max():.3f}]")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize GELLO collected data")
    parser.add_argument("filepath", help="Path to pkl file")
    parser.add_argument("--summary", action="store_true", help="Only print summary, no GUI")
    args = parser.parse_args()

    data = load_data(args.filepath)
    print_summary(data)

    if not args.summary:
        viz = DataVisualizer(data)
        print("\nControls: Left/Right arrows or < > buttons to navigate")
        viz.show()


if __name__ == "__main__":
    main()
