import argparse

from torcs_rl import parse_args, train_and_evaluate


def main() -> None:
    args = parse_args()

    if args.sweep:
        from torcs_rl.sweep import launch_sweep
        launch_sweep(args)
    elif args.launch:
        # Force W&B on and allow all config overrides so W&B Launch can parameterise the run.
        args.wandb = True
        args.wandb_allow_config = True
        train_and_evaluate(args)
    else:
        train_and_evaluate(args)


if __name__ == "__main__":
    main()

