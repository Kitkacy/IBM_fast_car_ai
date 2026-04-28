import argparse
import csv
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a two-week user activity grid from an activity tracker CSV export."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="activity_tracker_report.csv",
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--today",
        help="Override the current date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output-dir",
        default="activity_reports",
        help="Directory where the generated report files will be saved.",
    )
    return parser.parse_args()


def resolve_today(today_arg: str | None) -> date:
    if today_arg:
        return datetime.strptime(today_arg, "%Y-%m-%d").date()
    return date.today()


def load_activity(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        return list(reader)


def parse_timestamp(row: dict[str, str]) -> datetime:
    date_text = row["Date"].strip()
    time_text = row["Time"].strip()
    return datetime.strptime(f"{date_text} {time_text}", DATE_FORMAT)


def build_grid(
    rows: list[dict[str, str]], today: date
) -> tuple[list[str], list[date], np.ndarray, dict[str, int]]:
    start_date = today - timedelta(days=13)
    last_8_days_start = today - timedelta(days=7)
    day_range = [start_date + timedelta(days=offset) for offset in range(14)]
    day_to_index = {day: index for index, day in enumerate(day_range)}

    counts_by_user = defaultdict(lambda: [0] * len(day_range))
    last_8_days_activity = defaultdict(int)
    users_seen = {row["User"].strip() for row in rows if row.get("User", "").strip()}

    for row in rows:
        timestamp = parse_timestamp(row)
        activity_date = timestamp.date()
        user = row["User"].strip()
        if not user:
            continue

        if activity_date < start_date or activity_date > today:
            continue

        counts_by_user[user][day_to_index[activity_date]] += 1
        if activity_date >= last_8_days_start:
            last_8_days_activity[user] += 1

    users = sorted(users_seen, key=str.casefold)
    grid = np.array([counts_by_user[user] for user in users], dtype=int)
    return users, day_range, grid, dict(last_8_days_activity)


def build_table(
    users: list[str], day_range: list[date], grid: np.ndarray, last_8_days_activity: dict[str, int]
) -> str:
    headers = ["User", "Last 8d", *[day.strftime("%m-%d") for day in day_range], "2w Total"]
    rows = []
    for index, user in enumerate(users):
        values = [str(value) if value else "." for value in grid[index]]
        last_8d_total = str(last_8_days_activity.get(user, 0))
        total = str(int(grid[index].sum()))
        rows.append([user, last_8d_total, *values, total])

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def save_table_csv(
    output_path: Path, users: list[str], day_range: list[date], grid: np.ndarray, last_8_days_activity: dict[str, int]
) -> None:
    headers = ["User", "last_8_days_total", *[day.isoformat() for day in day_range], "two_week_total"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for index, user in enumerate(users):
            writer.writerow(
                [
                    user,
                    last_8_days_activity.get(user, 0),
                    *grid[index].tolist(),
                    int(grid[index].sum()),
                ]
            )


def save_heatmap(
    output_path: Path, users: list[str], day_range: list[date], grid: np.ndarray, last_8_days_activity: dict[str, int]
) -> None:
    if not users:
        raise ValueError("No activity found in the last two weeks.")

    last_8d_values = np.array([last_8_days_activity.get(user, 0) for user in users], dtype=int).reshape(-1, 1)
    masked_grid = np.ma.masked_where(grid == 0, grid)
    activity_cmap = matplotlib.colormaps["RdYlGn"].copy()
    activity_cmap.set_bad("#bdbdbd")

    summary_max = max(1, int(last_8d_values.max()) if last_8d_values.size else 1)
    summary_norm = plt.Normalize(vmin=0, vmax=summary_max)
    summary_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "last8d", ["#ff2b2b", "#fff176", "#2e7d32"]
    )

    figure_width = max(13, len(day_range) * 0.75 + 2.5)
    figure_height = max(4, len(users) * 0.6)
    fig = plt.figure(figsize=(figure_width, figure_height), constrained_layout=True)
    grid_spec = fig.add_gridspec(1, 3, width_ratios=[2.8, 1.2, len(day_range)], wspace=0.05)

    member_ax = fig.add_subplot(grid_spec[0, 0])
    summary_ax = fig.add_subplot(grid_spec[0, 1])
    activity_ax = fig.add_subplot(grid_spec[0, 2])

    member_ax.set_xlim(0, 1)
    member_ax.set_ylim(len(users) - 0.5, -0.5)
    member_ax.axis("off")
    member_ax.set_title("Member", pad=12)
    for row_index, user in enumerate(users):
        member_ax.text(0.02, row_index, user, va="center", ha="left", fontsize=10)

    summary_image = summary_ax.imshow(
        last_8d_values,
        aspect="auto",
        cmap=summary_cmap,
        norm=summary_norm,
        interpolation="nearest",
    )
    summary_ax.set_title("Last 8d", pad=12)
    summary_ax.set_xticks([0])
    summary_ax.set_xticklabels(["Total"])
    summary_ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    summary_ax.set_yticks(np.arange(len(users)))
    summary_ax.set_yticklabels([])
    summary_ax.tick_params(left=False)
    for row_index in range(len(users)):
        value = int(last_8d_values[row_index, 0])
        text_color = "white" if value == 0 or value > summary_max / 2 else "black"
        summary_ax.text(0, row_index, str(value), ha="center", va="center", fontsize=9, color=text_color)

    activity_image = activity_ax.imshow(
        masked_grid, aspect="auto", cmap=activity_cmap, interpolation="nearest"
    )
    activity_ax.set_xticks(np.arange(len(day_range)))
    activity_ax.set_xticklabels([day.strftime("%m-%d") for day in day_range], rotation=45, ha="right")
    activity_ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    activity_ax.set_yticks(np.arange(len(users)))
    activity_ax.set_yticklabels([])
    activity_ax.tick_params(left=False)

    max_value = int(grid.max()) if grid.size else 0
    text_threshold = max(1, max_value / 2)
    for row_index in range(grid.shape[0]):
        for column_index in range(grid.shape[1]):
            value = int(grid[row_index, column_index])
            text_color = "white" if value > text_threshold else "black"
            activity_ax.text(
                column_index,
                row_index,
                str(value) if value else "0",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    activity_ax.set_title("Activity By Day", pad=12)
    colorbar = fig.colorbar(activity_image, ax=activity_ax)
    colorbar.set_label("Actions per day")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    today = resolve_today(args.today)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_activity(csv_path)
    users, day_range, grid, last_8_days_activity = build_grid(rows, today)

    if not users:
        raise ValueError(
            f"No activity found between {(today - timedelta(days=13)).isoformat()} and {today.isoformat()}."
        )

    table_output = build_table(users, day_range, grid, last_8_days_activity)
    print(table_output)

    table_csv_path = output_dir / "activity_grid_last_14_days.csv"
    heatmap_path = output_dir / "activity_grid_last_14_days.png"
    save_table_csv(table_csv_path, users, day_range, grid, last_8_days_activity)
    save_heatmap(heatmap_path, users, day_range, grid, last_8_days_activity)

    print()
    print(f"Saved table CSV: {table_csv_path}")
    print(f"Saved heatmap PNG: {heatmap_path}")


if __name__ == "__main__":
    main()
