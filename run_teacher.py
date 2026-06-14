import argparse
from pathlib import Path
from types import SimpleNamespace

from teacher_agent import Agent
from torcs_rl.env import TorcsRLEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Run the rule-based teacher visually in TORCS.")
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


def main():
    args = parse_args()
    env = None
    try:
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
        obs, _ = env.reset()
        teacher = Agent(dim_action=3)
        last_reward = 0.0
        done = False
        steps = 0

        while True:
            raw_obs = env.client.S.d
            action = teacher.act(raw_to_teacher_obs(raw_obs), last_reward, done, False)
            obs, reward, terminated, truncated, info = env.step(action)
            last_reward = float(reward)
            done = bool(terminated or truncated)
            steps += 1

            if steps % 500 == 0:
                print(
                    f"steps={steps} speed={float(raw_obs['speedX']):.1f} "
                    f"trackPos={float(raw_obs['trackPos']):.3f} reward={last_reward:.3f}"
                )

            if done:
                print(f"reset after {steps} steps: {info.get('termination_reason')}")
                obs, _ = env.reset()
                teacher = Agent(dim_action=3)
                last_reward = 0.0
                done = False
                steps = 0
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
