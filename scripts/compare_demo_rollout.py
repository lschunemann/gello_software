#!/usr/bin/env python3
"""Compare demonstration data with live RL rollout data.

This script helps verify:
1. Action scale/range is consistent between demo and RL
2. State formats match
3. Image processing is the same
"""

import pickle as pkl
import numpy as np
import torch
import argparse
from pathlib import Path


def load_pkl(filepath):
    with open(filepath, "rb") as f:
        data = pkl.load(f)
    return data


def extract_obs(sample):
    """Extract observation from sample (handles both formats)."""
    if "transitions" in sample:
        obs = sample["transitions"]["obs"]
    else:
        obs = sample.get("obs", sample)
    return obs


def get_states(obs):
    """Extract states from observation."""
    if "states" in obs:
        states = obs["states"]
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


def get_image(obs):
    """Extract image from observation."""
    if "main_images" in obs:
        img = obs["main_images"]
    else:
        return None
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    # Squeeze batch dim if present (e.g. rollout data has shape (1, 128, 128, 3))
    if img.ndim > 3:
        img = img.squeeze(0)
    return img


def compute_stats(values, name):
    """Compute and print statistics for a set of values."""
    values = np.array(values)
    print(f"\n{name}:")
    print(f"  Shape: {values.shape}")
    print(f"  Mean:  {np.mean(values, axis=0)}")
    print(f"  Std:   {np.std(values, axis=0)}")
    print(f"  Min:   {np.min(values, axis=0)}")
    print(f"  Max:   {np.max(values, axis=0)}")
    return values


def analyze_dataset(data, name):
    """Analyze a dataset and return statistics."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {name}")
    print(f"{'='*60}")
    print(f"Total samples: {len(data)}")

    # Sample structure
    sample = data[0]
    print(f"\nSample keys: {list(sample.keys())}")

    obs = extract_obs(sample)
    print(f"Obs keys: {list(obs.keys())}")

    # Collect all actions and states
    all_actions = []
    all_states = []
    all_rewards = []
    all_tcp_poses = []
    all_grippers = []

    for d in data:
        action = get_action(d)
        if action is not None:
            all_actions.append(action)

        obs = extract_obs(d)
        states = get_states(obs)
        if states is not None:
            all_states.append(states)
            # Extract specific fields: gripper(1), force(3), pose(6), torque(3), vel(6)
            all_grippers.append(states[0])
            all_tcp_poses.append(states[4:10])

        reward = d.get("rewards", d.get("reward", 0))
        if isinstance(reward, torch.Tensor):
            reward = reward.item() if reward.numel() == 1 else reward.sum().item()
        all_rewards.append(float(reward))

    # Compute statistics
    stats = {}

    if all_actions:
        actions = compute_stats(all_actions, "Actions")
        stats["actions"] = {
            "mean": np.mean(actions, axis=0),
            "std": np.std(actions, axis=0),
            "min": np.min(actions, axis=0),
            "max": np.max(actions, axis=0),
        }

    if all_states:
        states = compute_stats(all_states, "Full States (19D)")
        stats["states"] = {
            "mean": np.mean(states, axis=0),
            "std": np.std(states, axis=0),
        }

    if all_tcp_poses:
        poses = compute_stats(all_tcp_poses, "TCP Pose (6D: xyz + euler)")
        stats["tcp_pose"] = {
            "mean": np.mean(poses, axis=0),
            "std": np.std(poses, axis=0),
        }

    if all_grippers:
        grippers = np.array(all_grippers)
        print(f"\nGripper Position:")
        print(f"  Mean: {np.mean(grippers):.4f}")
        print(f"  Min:  {np.min(grippers):.4f}")
        print(f"  Max:  {np.max(grippers):.4f}")
        stats["gripper"] = {"mean": np.mean(grippers), "min": np.min(grippers), "max": np.max(grippers)}

    # Rewards
    rewards = np.array(all_rewards)
    print(f"\nRewards:")
    print(f"  Total: {np.sum(rewards):.2f}")
    print(f"  Successes (>0): {np.sum(rewards > 0)}")
    stats["rewards"] = {"total": np.sum(rewards), "successes": np.sum(rewards > 0)}

    # Image info
    img = get_image(extract_obs(data[0]))
    if img is not None:
        print(f"\nImage:")
        print(f"  Shape: {img.shape}")
        print(f"  Dtype: {img.dtype}")
        print(f"  Range: [{img.min()}, {img.max()}]")
        stats["image"] = {"shape": img.shape, "dtype": str(img.dtype), "range": (img.min(), img.max())}

    return stats


def compare_stats(demo_stats, rollout_stats):
    """Compare statistics between demo and rollout data."""
    print(f"\n{'='*60}")
    print("COMPARISON: Demo vs Rollout")
    print(f"{'='*60}")

    # Compare actions
    if "actions" in demo_stats and "actions" in rollout_stats:
        print("\nAction Comparison:")
        demo_act = demo_stats["actions"]
        roll_act = rollout_stats["actions"]

        print(f"  {'Dim':<6} {'Demo Mean':>12} {'Roll Mean':>12} {'Demo Range':>20} {'Roll Range':>20}")
        print(f"  {'-'*70}")

        labels = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
        for i in range(min(len(demo_act["mean"]), len(roll_act["mean"]))):
            label = labels[i] if i < len(labels) else f"a{i}"
            demo_range = f"[{demo_act['min'][i]:.3f}, {demo_act['max'][i]:.3f}]"
            roll_range = f"[{roll_act['min'][i]:.3f}, {roll_act['max'][i]:.3f}]"
            print(f"  {label:<6} {demo_act['mean'][i]:>12.4f} {roll_act['mean'][i]:>12.4f} {demo_range:>20} {roll_range:>20}")

    # Compare TCP poses
    if "tcp_pose" in demo_stats and "tcp_pose" in rollout_stats:
        print("\nTCP Pose Comparison (6D: xyz + euler):")
        demo_pose = demo_stats["tcp_pose"]
        roll_pose = rollout_stats["tcp_pose"]

        labels = ["x", "y", "z", "rx", "ry", "rz"]
        print(f"  {'Dim':<6} {'Demo Mean':>12} {'Roll Mean':>12} {'Demo Std':>12} {'Roll Std':>12}")
        print(f"  {'-'*54}")
        for i in range(6):
            print(f"  {labels[i]:<6} {demo_pose['mean'][i]:>12.4f} {roll_pose['mean'][i]:>12.4f} {demo_pose['std'][i]:>12.4f} {roll_pose['std'][i]:>12.4f}")

    # Compare gripper
    if "gripper" in demo_stats and "gripper" in rollout_stats:
        print("\nGripper Comparison:")
        demo_grip = demo_stats["gripper"]
        roll_grip = rollout_stats["gripper"]
        print(f"  Demo: mean={demo_grip['mean']:.4f}, range=[{demo_grip['min']:.4f}, {demo_grip['max']:.4f}]")
        print(f"  Roll: mean={roll_grip['mean']:.4f}, range=[{roll_grip['min']:.4f}, {roll_grip['max']:.4f}]")

    # Compare images
    if "image" in demo_stats and "image" in rollout_stats:
        print("\nImage Comparison:")
        demo_img = demo_stats["image"]
        roll_img = rollout_stats["image"]
        shape_match = demo_img["shape"] == roll_img["shape"]
        dtype_match = demo_img["dtype"] == roll_img["dtype"]
        range_match = demo_img["range"] == roll_img["range"]

        print(f"  Shape: Demo={demo_img['shape']}, Roll={roll_img['shape']} {'✓' if shape_match else '✗ MISMATCH'}")
        print(f"  Dtype: Demo={demo_img['dtype']}, Roll={roll_img['dtype']} {'✓' if dtype_match else '✗ MISMATCH'}")
        print(f"  Range: Demo={demo_img['range']}, Roll={roll_img['range']} {'✓' if range_match else '✗ MISMATCH'}")


def main():
    parser = argparse.ArgumentParser(description="Compare demonstration data with rollout data")
    parser.add_argument("demo_file", help="Path to demonstration pkl file")
    parser.add_argument("--rollout", "-r", help="Path to rollout pkl file (optional)")
    args = parser.parse_args()

    # Analyze demo data
    demo_data = load_pkl(args.demo_file)
    demo_stats = analyze_dataset(demo_data, f"Demo: {args.demo_file}")

    # Analyze rollout data if provided
    if args.rollout:
        rollout_data = load_pkl(args.rollout)
        rollout_stats = analyze_dataset(rollout_data, f"Rollout: {args.rollout}")
        compare_stats(demo_stats, rollout_stats)
    else:
        print("\n" + "="*60)
        print("TIP: To compare with rollout data, use --rollout option")
        print("To save rollout data, add logging to RLinf's realworld_env.py")
        print("="*60)


if __name__ == "__main__":
    main()
