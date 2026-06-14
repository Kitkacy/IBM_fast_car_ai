import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fast_teacher_agent import FastTeacherAgent
from teacher_agent import Agent
from torcs_rl.env import TorcsRLEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Collect TORCS behavior-cloning data from the rule-based teacher.")
    parser.add_argument("--output", type=Path, default=Path("data/corkscrew_teacher.npz"))
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means stop only by --steps.")
    parser.add_argument(
        "--teacher",
        choices=("fast", "stable"),
        default="fast",
        help="Teacher controller to collect from.",
    )
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--torcs-exe", type=Path, default=None)
    parser.add_argument("--race-config", type=Path, default=None)
    parser.add_argument("--reward-mode", choices=("blog", "shaped", "speed"), default="speed")
    parser.add_argument(
        "--off-track-limit",
        type=float,
        default=1.35,
        help="Reset only when abs(trackPos) exceeds this value. 1.0 is strict TORCS off-track.",
    )
    parser.add_argument(
        "--off-track-mode",
        choices=("terminate", "penalty", "ignore"),
        default="penalty",
        help="How to handle abs(trackPos) over --off-track-limit.",
    )
    return parser.parse_args()


def raw_to_teacher_obs(raw_obs):
    return SimpleNamespace(
        track=raw_obs["track"],
        speedX=raw_obs["speedX"],
        angle=raw_obs["angle"],
        trackPos=raw_obs["trackPos"],
    )


def build_teacher(name):
    if name == "fast":
        return FastTeacherAgent()
    return Agent(dim_action=3)


def teacher_action(teacher, raw_obs, reward, done):
    if isinstance(teacher, FastTeacherAgent):
        action = teacher.act(raw_obs)
        return np.array([action["steer"], action["accel"], action["brake"]], dtype=np.float32)
    return teacher.act(raw_to_teacher_obs(raw_obs), reward, done, False).astype(np.float32)


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    env = TorcsRLEnv(
        torcs_port=args.port,
        torcs_exe=str(args.torcs_exe) if args.torcs_exe else None,
        race_config=str(args.race_config) if args.race_config else None,
        control_mode="full",
        reward_mode=args.reward_mode,
        obs_norm=False,
        off_track_track_pos_limit=args.off_track_limit,
        off_track_mode=args.off_track_mode,
        terminate_low_progress=False,
        terminate_on_damage=False,
        terminate_on_backward=False,
        terminate_on_fully_off_track=False,
    )

    observations = []
    actions = []
    rewards = []
    dones = []
    episode_returns = []
    episode_lengths = []

    try:
        obs, _ = env.reset()
        teacher = build_teacher(args.teacher)
        last_reward = 0.0
        done = False
        episode_return = 0.0
        episode_length = 0
        episodes = 0

        while len(observations) < args.steps:
            raw_obs = env.client.S.d
            action = teacher_action(teacher, raw_obs, last_reward, done)

            observations.append(np.asarray(obs, dtype=np.float32))
            actions.append(np.asarray(action, dtype=np.float32))

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            last_reward = float(reward)
            episode_return += last_reward
            episode_length += 1
            rewards.append(last_reward)
            dones.append(done)

            if len(observations) % 1000 == 0:
                print(
                    f"collected={len(observations)} episodes={episodes} "
                    f"last_reward={last_reward:.3f} reason={info.get('termination_reason')}"
                )

            if done:
                episodes += 1
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
                print(
                    f"episode={episodes} length={episode_length} "
                    f"return={episode_return:.3f} reason={info.get('termination_reason')}"
                )
                if args.max_episodes and episodes >= args.max_episodes:
                    break
                obs, _ = env.reset()
                teacher = build_teacher(args.teacher)
                last_reward = 0.0
                done = False
                episode_return = 0.0
                episode_length = 0

        np.savez_compressed(
            args.output,
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.bool_),
            episode_returns=np.asarray(episode_returns, dtype=np.float32),
            episode_lengths=np.asarray(episode_lengths, dtype=np.int32),
        )
        print(f"saved {len(observations)} samples to {args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
