from torcs_rl import train_and_evaluate
from torcs_rl.config import load_config


def main() -> None:
    config = load_config()
    if config.mode == "sweep":
        from torcs_rl.sweep import launch_sweep
        launch_sweep(config)
    else:
        train_and_evaluate(config)


if __name__ == "__main__":
    main()

