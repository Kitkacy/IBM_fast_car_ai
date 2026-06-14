import argparse
from pathlib import Path

from stable_baselines3 import PPO, SAC, TD3

from torcs_rl.env import TorcsRLEnv


ALGORITHMS = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained TORCS model visually.")
    parser.add_argument(
        "--algorithm",
        choices=sorted(ALGORITHMS),
        default="sac",
        help="Algorithm class used to load the model.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("runs/smoke-sac/models/sac_torcs_latest.zip"),
        help="Path to the saved Stable Baselines3 model zip.",
    )
    parser.add_argument("--target-speed", type=float, default=60.0, help="Target speed for the built-in throttle controller.")
    parser.add_argument("--port", type=int, default=3001, help="TORCS SCR UDP port.")
    parser.add_argument(
        "--control-mode",
        choices=("steering", "full"),
        default="steering",
        help="Action space used by the saved model.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=("blog", "shaped", "speed"),
        default="blog",
        help="Reward function used by the environment while replaying.",
    )
    obs_norm_group = parser.add_mutually_exclusive_group()
    obs_norm_group.add_argument("--obs-norm", dest="obs_norm", action="store_true", help="Enable running observation normalization.")
    obs_norm_group.add_argument("--no-obs-norm", dest="obs_norm", action="store_false", help="Disable running observation normalization.")
    parser.set_defaults(obs_norm=True)
    parser.add_argument(
        "--torcs-exe",
        type=Path,
        default=None,
        help="Optional TORCS executable. Omit to attach to an already-running TORCS instance.",
    )
    parser.add_argument(
        "--race-config",
        type=Path,
        default=None,
        help="Optional race config. Omit when attaching to an already-running TORCS instance.",
    )
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions instead of deterministic.")
    parser.add_argument(
        "--off-track-limit",
        type=float,
        default=1.0,
        help="Terminate only when abs(trackPos) exceeds this value. 1.0 is strict TORCS off-track.",
    )
    parser.add_argument(
        "--off-track-mode",
        choices=("terminate", "penalty", "ignore"),
        default="terminate",
        help="How to handle abs(trackPos) over --off-track-limit.",
    )
    parser.add_argument(
        "--no-low-progress-termination",
        action="store_true",
        help="Disable shaped-reward low-progress resets.",
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

    available = sorted(Path("runs").glob("**/*.zip"))
    print(f"Model not found: {model_path}")
    if available:
        print("Available models:")
        for available_path in available:
            print(f"  {available_path}")
    raise SystemExit(2)


def main():
    args = parse_args()
    model_cls = ALGORITHMS[args.algorithm]
    model_path = resolve_model_path(args.model_path)
    model = model_cls.load(str(model_path))

    env = None
    try:
        env = TorcsRLEnv(
            target_speed=args.target_speed,
            torcs_port=args.port,
            torcs_exe=str(args.torcs_exe) if args.torcs_exe else None,
            race_config=str(args.race_config) if args.race_config else None,
            control_mode=args.control_mode,
            reward_mode=args.reward_mode,
            obs_norm=args.obs_norm,
            off_track_track_pos_limit=args.off_track_limit,
            off_track_mode=args.off_track_mode,
            terminate_low_progress=not args.no_low_progress_termination,
        )

        obs, _ = env.reset()

        deterministic = not args.stochastic
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"reset: {info.get('termination_reason')}")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
