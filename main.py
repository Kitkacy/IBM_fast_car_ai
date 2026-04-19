from torcs_rl import parse_args, train_and_evaluate


def main() -> None:
    train_and_evaluate(parse_args())


if __name__ == "__main__":
    main()
