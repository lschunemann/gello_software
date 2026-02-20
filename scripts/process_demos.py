#!/usr/bin/env python3
"""Process demo data: filter + transform to relative frame in one pipeline.

Pipeline order:
  1. Extract first pose per episode (from raw data, before any filtering)
  2. Done-fix / --keep-done (move done to last step of done-sequence)
  3. Episode length check (--expected-episode-length)
  4. Safety box filter (--safety-box-min/max)
  5. Transform to relative frame (using saved first poses)
  6. Near-zero action filter
  7. Sparse reward (--sparse-reward)

Usage:
    python process_demos.py input.pkl output.pkl \\
        --safety-box-min 0.3 -0.3 0.01 --safety-box-max 0.7 0.3 0.3 \\
        --action-scale 0.02 0.1 1.0 --action-dim 6 \\
        --keep-done --expected-episode-length 30 --sparse-reward
"""

import argparse
import pickle as pkl
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_states(sample):
    """Get numpy states from a sample."""
    obs = sample["transitions"]["obs"] if "transitions" in sample else sample["obs"]
    states = obs["states"]
    if isinstance(states, torch.Tensor):
        states = states.numpy()
    if states.ndim > 1:
        states = states.squeeze(0)
    return states


def get_tcp_xyz(sample):
    return get_states(sample)[4:7]


def get_tcp_pose_6d(sample):
    return get_states(sample)[4:10]


def is_inside_safety_box(xyz, box_min, box_max):
    return np.all(xyz >= box_min) and np.all(xyz <= box_max)


def is_done(sample):
    done = sample.get("dones", sample.get("done", False))
    if isinstance(done, torch.Tensor):
        done = done.item()
    return bool(done)


def has_positive_reward(sample):
    r = sample.get("rewards", 0)
    if isinstance(r, torch.Tensor):
        r = r.item()
    return float(r) > 0


def construct_homogeneous_matrix(pose_7d):
    T = np.eye(4)
    T[:3, 3] = pose_7d[:3]
    T[:3, :3] = R.from_quat(pose_7d[3:7]).as_matrix()
    return T


def euler_to_quat(euler_xyz):
    return R.from_euler("xyz", euler_xyz).as_quat()


def transform_tcp_pose_to_relative(tcp_pose_6d, reset_pose_6d):
    reset_7d = np.concatenate([reset_pose_6d[:3], euler_to_quat(reset_pose_6d[3:6])])
    current_7d = np.concatenate([tcp_pose_6d[:3], euler_to_quat(tcp_pose_6d[3:6])])
    T_b_r_inv = np.linalg.inv(construct_homogeneous_matrix(reset_7d))
    T_r_o = T_b_r_inv @ construct_homogeneous_matrix(current_7d)
    p_r_o = T_r_o[:3, 3]
    euler_r_o = R.from_matrix(T_r_o[:3, :3]).as_euler("xyz")
    return np.concatenate([p_r_o, euler_r_o]).astype(np.float32)


def transform_velocity_to_ee_frame(tcp_vel_6d, tcp_pose_6d):
    quat = euler_to_quat(tcp_pose_6d[3:6])
    pose_7d = np.concatenate([tcp_pose_6d[:3], quat])
    rot = R.from_quat(pose_7d[3:7]).as_matrix()
    p = pose_7d[:3]
    p_skew = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])
    adjoint = np.zeros((6, 6))
    adjoint[:3, :3] = rot
    adjoint[3:, 3:] = rot
    adjoint[3:, :3] = p_skew @ rot
    return (np.linalg.inv(adjoint) @ tcp_vel_6d).astype(np.float32)


def get_states_fields(states):
    if isinstance(states, torch.Tensor):
        states = states.numpy()
    return {
        "gripper_position": states[0:1],
        "tcp_force": states[1:4],
        "tcp_pose": states[4:10],
        "tcp_torque": states[10:13],
        "tcp_vel": states[13:19],
    }


def rebuild_states(fields):
    return np.concatenate([
        fields["gripper_position"],
        fields["tcp_force"],
        fields["tcp_pose"],
        fields["tcp_torque"],
        fields["tcp_vel"],
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process demo data: filter + transform to relative frame")
    parser.add_argument("input", help="Input pkl file (absolute poses, ending in _raw.pkl)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output pkl file. Auto-generated from input if not provided.")

    # Filter args
    parser.add_argument("--safety-box-min", type=float, nargs=3, required=True,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--safety-box-max", type=float, nargs=3, required=True,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--keep-done", action="store_true",
                        help="Keep done=1 transitions (fix done flags instead of removing)")
    parser.add_argument("--expected-episode-length", type=int, default=None,
                        help="Remove episodes that don't have exactly this many steps")

    # Transform args
    parser.add_argument("--reset-pose", type=float, nargs=6, default=None,
                        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                        help="Fixed reset pose. If not provided, uses first obs of each episode.")
    parser.add_argument("--action-scale", type=float, nargs=3, default=None,
                        metavar=("POS", "ROT", "GRIP"),
                        help="Action scale [pos, rot, grip]")
    parser.add_argument("--action-dim", type=int, default=6, choices=[6, 7])
    parser.add_argument("--sparse-reward", action="store_true",
                        help="Only last step of each trajectory gets reward=1.0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Auto-generate output path if not provided
    if args.output is None:
        base = args.input
        # Strip _raw.pkl suffix
        if base.endswith("_raw.pkl"):
            base = base[:-len("_raw.pkl")]
        elif base.endswith(".pkl"):
            base = base[:-len(".pkl")]
        # Build suffix
        suffix = "_filtered_relative"
        if not args.keep_done:
            suffix += "_truncated"
        if args.sparse_reward:
            suffix += "_sparse"
        args.output = base + suffix + ".pkl"

    SAFETY_BOX_MIN = np.array(args.safety_box_min)
    SAFETY_BOX_MAX = np.array(args.safety_box_max)
    use_per_episode_reset = args.reset_pose is None
    fixed_reset_pose = np.array(args.reset_pose) if args.reset_pose is not None else None

    with open(args.input, "rb") as f:
        data = pkl.load(f)

    print(f"Loaded {len(data)} samples from {args.input}")
    print(f"Output: {args.output}")

    # ==================================================================
    # Phase 0: Extract per-episode first pose BEFORE any filtering
    # ==================================================================
    # Map: sample index in raw data -> reset pose for its episode
    episode_reset_poses_raw = {}
    if use_per_episode_reset:
        current_reset_pose = None
        episode_start = True
        for idx, sample in enumerate(data):
            if episode_start:
                current_reset_pose = get_tcp_pose_6d(sample).copy()
                episode_start = False
            episode_reset_poses_raw[idx] = current_reset_pose
            if is_done(sample):
                episode_start = True
        n_episodes_raw = sum(1 for i in range(len(data)) if i == 0 or
                             not np.array_equal(episode_reset_poses_raw[i],
                                                episode_reset_poses_raw.get(i-1, None)))
        print(f"  Extracted reset poses from {n_episodes_raw} episodes (from raw data)")
    else:
        print(f"  Using fixed reset_pose=[{', '.join(f'{v:.3f}' for v in fixed_reset_pose)}]")

    # ==================================================================
    # Phase 1: Done padding removal / --keep-done fix
    # ==================================================================
    # Track original indices so we can look up reset poses later
    filtered_with_idx = []  # list of (original_idx, sample)
    n_done_padding = 0
    seen_first_done = False

    for idx, sample in enumerate(data):
        done = is_done(sample)
        if done and not seen_first_done:
            seen_first_done = True
            is_padding = False
        elif done and seen_first_done:
            is_padding = True
        else:
            seen_first_done = False
            is_padding = False

        if is_padding and not args.keep_done:
            n_done_padding += 1
        else:
            filtered_with_idx.append((idx, sample))

    if n_done_padding > 0:
        print(f"\n  Done padding removed: {n_done_padding}")

    # --keep-done: fix done flags
    if args.keep_done:
        n_done_fixed = 0
        for i in range(len(filtered_with_idx)):
            _, sample = filtered_with_idx[i]
            if not is_done(sample):
                continue
            if i + 1 < len(filtered_with_idx) and is_done(filtered_with_idx[i + 1][1]):
                for key in ["dones", "done", "terminations"]:
                    if key in sample:
                        sample[key] = torch.tensor([0.0], dtype=torch.float32)
                n_done_fixed += 1
        print(f"  --keep-done: {n_done_fixed} intermediate dones cleared")

    print(f"  After done processing: {len(filtered_with_idx)} samples")

    # ==================================================================
    # Phase 2: Episode length check
    # ==================================================================
    if args.expected_episode_length is not None:
        exp_len = args.expected_episode_length
        episodes = []
        current_ep = []
        for item in filtered_with_idx:
            current_ep.append(item)
            if is_done(item[1]):
                episodes.append(current_ep)
                current_ep = []
        if current_ep:
            episodes.append(current_ep)

        kept = []
        n_removed_eps = 0
        for i, ep in enumerate(episodes):
            if len(ep) == exp_len:
                kept.extend(ep)
            elif len(ep) > exp_len:
                n_corrupt = len(ep) - exp_len
                print(f"  Episode {i+1}: {len(ep)} steps — dropping first {n_corrupt}, keeping last {exp_len}")
                kept.extend(ep[n_corrupt:])
                n_removed_eps += 1
            else:
                print(f"  Episode {i+1}: {len(ep)} steps — dropping (expected {exp_len})")
                n_removed_eps += 1

        print(f"\n  --expected-episode-length: {len(episodes)} episodes, removed {n_removed_eps}, kept {len(kept)} steps")
        filtered_with_idx = kept

    # ==================================================================
    # Phase 3: Safety box filter
    # ==================================================================
    print(f"\n  Safety box min: [{SAFETY_BOX_MIN[0]:.3f}, {SAFETY_BOX_MIN[1]:.3f}, {SAFETY_BOX_MIN[2]:.3f}]")
    print(f"  Safety box max: [{SAFETY_BOX_MAX[0]:.3f}, {SAFETY_BOX_MAX[1]:.3f}, {SAFETY_BOX_MAX[2]:.3f}]")

    safe_filtered = []
    n_outside_box = 0
    outside_xyzs = []
    for orig_idx, sample in filtered_with_idx:
        xyz = get_tcp_xyz(sample)
        if is_inside_safety_box(xyz, SAFETY_BOX_MIN, SAFETY_BOX_MAX):
            safe_filtered.append((orig_idx, sample))
        else:
            n_outside_box += 1
            outside_xyzs.append(xyz)

    if n_outside_box > 0:
        outside_xyzs = np.array(outside_xyzs)
        print(f"  Outside safety box: {n_outside_box}")
        for i, label in enumerate(["x", "y", "z"]):
            print(f"    {label}: [{outside_xyzs[:, i].min():.4f}, {outside_xyzs[:, i].max():.4f}]  "
                  f"(box: [{SAFETY_BOX_MIN[i]:.4f}, {SAFETY_BOX_MAX[i]:.4f}])")

    filtered_with_idx = safe_filtered
    print(f"  After safety box: {len(filtered_with_idx)} samples")

    # ==================================================================
    # Phase 4: Transform to relative frame
    # ==================================================================
    print(f"\n--- Transform to relative frame ---")
    print(f"  Action dim: {args.action_dim}")

    per_dim_scale = None
    if args.action_scale is not None:
        per_dim_scale = np.array([
            args.action_scale[0], args.action_scale[0], args.action_scale[0],
            args.action_scale[1], args.action_scale[1], args.action_scale[1],
        ])
        print(f"  Action scale: pos={args.action_scale[0]}, rot={args.action_scale[1]}, grip={args.action_scale[2]}")

    rot_clip = np.array([0.01, 0.01, np.pi / 6])

    transformed_data = []
    n_skipped_action = 0

    for orig_idx, sample in filtered_with_idx:
        states = get_states(sample)
        fields = get_states_fields(states)

        # Get reset pose: from raw first-obs or fixed
        if use_per_episode_reset:
            reset_pose = episode_reset_poses_raw[orig_idx]
        else:
            reset_pose = fixed_reset_pose

        # Transform TCP pose
        original_pose = fields["tcp_pose"].copy()
        fields["tcp_pose"] = transform_tcp_pose_to_relative(original_pose, reset_pose)
        fields["tcp_pose"][3:6] = np.clip(fields["tcp_pose"][3:6], -rot_clip, rot_clip)

        # Transform velocity
        fields["tcp_vel"] = transform_velocity_to_ee_frame(fields["tcp_vel"], original_pose)

        # Transform action
        action = sample["action"]
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        if action.ndim > 1:
            action = action.squeeze(0)
        action_base = action.copy().astype(np.float32)

        if per_dim_scale is not None:
            action_base[:6] = action_base[:6] / per_dim_scale

        new_action_ee = transform_velocity_to_ee_frame(action_base[:6], original_pose)

        # Filter near-zero actions (max < 0.05), but never skip reward/done steps
        is_important = has_positive_reward(sample) or is_done(sample)
        if not is_important and np.max(np.abs(new_action_ee[:6])) < 0.05:
            n_skipped_action += 1
            continue

        # Clip rotation actions
        if per_dim_scale is not None:
            rot_action_scale = per_dim_scale[3]
            action_rot_clip = rot_clip / rot_action_scale
            new_action_ee[3:6] = np.clip(new_action_ee[3:6], -action_rot_clip, action_rot_clip)

        # Assemble final action
        if args.action_dim == 7 and len(action) >= 7:
            new_action = np.concatenate([new_action_ee, action[6:7]]).astype(np.float32)
        else:
            new_action = new_action_ee.astype(np.float32)
        new_action = np.clip(new_action, -1.0, 1.0)

        # Rebuild states
        new_states = rebuild_states(fields)

        # Transform next_obs
        next_obs = sample["transitions"]["next_obs"] if "transitions" in sample else sample.get("next_obs")
        new_next_states = None
        if next_obs is not None:
            next_states = next_obs["states"]
            if isinstance(next_states, torch.Tensor):
                next_states = next_states.numpy()
            if next_states.ndim > 1:
                next_states = next_states.squeeze(0)
            next_fields = get_states_fields(next_states)
            original_next_pose = next_fields["tcp_pose"].copy()
            next_fields["tcp_pose"] = transform_tcp_pose_to_relative(original_next_pose, reset_pose)
            next_fields["tcp_pose"][3:6] = np.clip(next_fields["tcp_pose"][3:6], -rot_clip, rot_clip)
            next_fields["tcp_vel"] = transform_velocity_to_ee_frame(next_fields["tcp_vel"], original_next_pose)
            new_next_states = rebuild_states(next_fields)

        # Build output sample
        obs = sample["transitions"]["obs"] if "transitions" in sample else sample["obs"]
        new_obs = {}
        for ok, ov in obs.items():
            if ok == "states":
                new_obs[ok] = torch.tensor(new_states, dtype=torch.float32)
            elif ok == "main_images":
                if isinstance(ov, torch.Tensor):
                    img = ov.squeeze(0) if ov.dim() > 3 else ov
                else:
                    img = torch.tensor(ov)
                    if img.dim() > 3:
                        img = img.squeeze(0)
                new_obs[ok] = img
            else:
                new_obs[ok] = ov

        new_next_obs = None
        if next_obs is not None:
            new_next_obs = {}
            for ok, ov in next_obs.items():
                if ok == "states":
                    new_next_obs[ok] = torch.tensor(new_next_states, dtype=torch.float32)
                elif ok == "main_images":
                    if isinstance(ov, torch.Tensor):
                        img = ov.squeeze(0) if ov.dim() > 3 else ov
                    else:
                        img = torch.tensor(ov)
                        if img.dim() > 3:
                            img = img.squeeze(0)
                    new_next_obs[ok] = img
                else:
                    new_next_obs[ok] = ov

        done = sample.get("dones", sample.get("done", False))
        if isinstance(done, torch.Tensor):
            done = done.view(-1)
        else:
            done = torch.tensor(bool(done), dtype=torch.bool)

        rewards = sample.get("rewards", torch.tensor(0.0, dtype=torch.float32))
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.view(-1)
        else:
            rewards = torch.tensor(float(rewards), dtype=torch.float32)

        terminations = done.clone() if isinstance(done, torch.Tensor) else torch.tensor(bool(done), dtype=torch.bool)
        truncations = torch.tensor(False, dtype=torch.bool).view(-1)

        new_sample = {
            "transitions": {"obs": new_obs},
            "action": torch.tensor(new_action, dtype=torch.float32),
            "rewards": rewards,
            "dones": done,
            "terminations": terminations,
            "truncations": truncations,
        }
        if new_next_obs is not None:
            new_sample["transitions"]["next_obs"] = new_next_obs

        transformed_data.append(new_sample)

    if n_skipped_action > 0:
        print(f"  Skipped {n_skipped_action} near-zero action samples (max < 0.05)")

    # ==================================================================
    # Phase 5: Sparse reward
    # ==================================================================
    if args.sparse_reward:
        traj_boundaries = []
        traj_start = 0
        for i, sample in enumerate(transformed_data):
            d = sample.get("dones", torch.tensor(False))
            if isinstance(d, torch.Tensor):
                d = d.item()
            if bool(d) or i == len(transformed_data) - 1:
                traj_boundaries.append((traj_start, i))
                traj_start = i + 1

        num_positive_before = sum(1 for s in transformed_data if has_positive_reward(s))

        num_positive_after = 0
        for start, end in traj_boundaries:
            had_reward = any(has_positive_reward(transformed_data[i]) for i in range(start, end + 1))
            for i in range(start, end + 1):
                transformed_data[i]["rewards"] = torch.tensor(0.0, dtype=torch.float32).view(-1)
            if had_reward:
                transformed_data[end]["rewards"] = torch.tensor(1.0, dtype=torch.float32).view(-1)
                num_positive_after += 1

        print(f"\n  Sparse reward: {len(traj_boundaries)} trajectories, "
              f"{num_positive_before} -> {num_positive_after} reward steps")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n--- Summary ---")
    print(f"  Input:  {len(data)} samples")
    print(f"  Output: {len(transformed_data)} samples")

    if transformed_data:
        new_poses = []
        for s in transformed_data:
            n_s = s["transitions"]["obs"]["states"]
            if isinstance(n_s, torch.Tensor):
                n_s = n_s.numpy()
            if n_s.ndim > 1:
                n_s = n_s.squeeze(0)
            new_poses.append(n_s[4:10])
        new_poses = np.array(new_poses)

        labels = ["x", "y", "z", "rx", "ry", "rz"]
        print(f"\n{'':8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        for i, label in enumerate(labels):
            print(f"  {label:6} {new_poses[:, i].mean():>10.4f} {new_poses[:, i].std():>10.4f} "
                  f"{new_poses[:, i].min():>10.4f} {new_poses[:, i].max():>10.4f}")

    if args.dry_run:
        print("\nDry run - not saving")
        return

    with open(args.output, "wb") as f:
        pkl.dump(transformed_data, f)
    print(f"\nSaved {len(transformed_data)} samples to {args.output}")


if __name__ == "__main__":
    main()
