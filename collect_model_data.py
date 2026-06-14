import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC, TD3

from torcs_rl.env import TorcsRLEnv


ALGORITHMS = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Collect behavior-cloning data from a trained SB3 TORCS model.")
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="sac")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/model_teacher.npz"))
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means stop only by --steps.")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--torcs-exe", type=Path, default=None)
    parser.add_argument("--race-config", type=Path, default=None)
    parser.add_argument("--reward-mode", choices=("blog", "shaped", "speed"), default="speed")
    parser.add_argument("--control-mode", choices=("steering", "full"), default="full")
    parser.add_argument("--obs-norm", action="store_true", help="Enable env observation normalization.")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions.")
    parser.add_argument(
        "--off-track-limit",
        type=float,
        default=1.35,
        help="Reset only when abs(trackPos) exceeds this value if off-track mode terminates.",
    )
    parser.add_argument(
        "--off-track-mode",
        choices=("terminate", "penalty", "ignore"),
        default="ignore",
        help="How to handle abs(trackPos) over --off-track-limit.",
    )
    return parser.parse_args()


def resolve_model_path(path):
    model_path = Path(path)
    if model_path.exists():
        return model_path
    if model_path.suffix != ".zip":
        zipped_path = model_path.with_suffix(".zip")
        if zipped_path.exists():
            return zipped_path
    raise FileNotFoundError(f"Model not found: {model_path}")


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model_cls = ALGORITHMS[args.algorithm]
    model = model_cls.load(str(resolve_model_path(args.model_path)))

    env = TorcsRLEnv(
        torcs_port=args.port,
        torcs_exe=str(args.torcs_exe) if args.torcs_exe else None,
        race_config=str(args.race_config) if args.race_config else None,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        obs_norm=args.obs_norm,
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
        deterministic = not args.stochastic
        episode_return = 0.0
        episode_length = 0
        episodes = 0

        while len(observations) < args.steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float32).reshape(env.action_space.shape)

            observations.append(np.asarray(obs, dtype=np.float32))
            actions.append(np.asarray(action, dtype=np.float32))

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            reward = float(reward)
            episode_return += reward
            episode_length += 1
            rewards.append(reward)
            dones.append(done)

            if len(observations) % 1000 == 0:
                raw_obs = info.get("raw_obs", {})
                speed = float(raw_obs.get("speedX", 0.0)) if isinstance(raw_obs, dict) else 0.0
                print(
                    f"collected={len(observations)} episodes={episodes} "
                    f"speed={speed:.1f} reward={reward:.3f} reason={info.get('termination_reason')}"
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
            source_algorithm=args.algorithm,
            source_model=str(args.model_path),
        )
        print(f"saved {len(observations)} samples to {args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
