import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CartPole learning curves from HW2 experiment logs."
    )
    parser.add_argument(
        "--exp_dir",
        type=Path,
        default=Path("exp"),
        help="Directory containing experiment subdirectories with flags.json and log.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("exp/plots_part1"),
        help="Directory where plot images will be written.",
    )
    return parser.parse_args()


def load_flags(flags_path: Path) -> dict:
    with flags_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_log(log_path: Path) -> tuple[list[float], list[float]]:
    envsteps = []
    returns = []
    with log_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            envsteps.append(float(row["Train_EnvstepsSoFar"]))
            returns.append(float(row["Eval_AverageReturn"]))
    return envsteps, returns


def variant_label(exp_name: str) -> str:
    if exp_name.startswith("cartpole_lb_"):
        suffix = exp_name[len("cartpole_lb_") :]
    elif exp_name == "cartpole_lb":
        suffix = ""
    elif exp_name.startswith("cartpole_"):
        suffix = exp_name[len("cartpole_") :]
    elif exp_name == "cartpole":
        suffix = ""
    else:
        suffix = exp_name

    labels = {
        "": "Vanilla PG",
        "rtg": "Reward-to-go",
        "na": "Normalize advantages",
        "rtg_na": "Reward-to-go + normalize advantages",
    }
    return labels.get(suffix, suffix.replace("_", " + "))


def collect_runs(exp_dir: Path) -> tuple[list[dict], list[dict]]:
    small_batch_runs = []
    large_batch_runs = []

    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        flags_path = run_dir / "flags.json"
        log_path = run_dir / "log.csv"
        if not flags_path.exists() or not log_path.exists():
            continue

        flags = load_flags(flags_path)
        exp_name = flags.get("exp_name", "")
        env_name = flags.get("env_name", "")
        if env_name != "CartPole-v0":
            continue

        envsteps, returns = load_log(log_path)
        run = {
            "exp_name": exp_name,
            "label": variant_label(exp_name),
            "envsteps": envsteps,
            "returns": returns,
            "run_dir": run_dir,
        }

        if exp_name.startswith("cartpole_lb"):
            large_batch_runs.append(run)
        elif exp_name.startswith("cartpole"):
            small_batch_runs.append(run)

    return small_batch_runs, large_batch_runs


def sort_runs(runs: list[dict]) -> list[dict]:
    order = {
        "Vanilla PG": 0,
        "Reward-to-go": 1,
        "Normalize advantages": 2,
        "Reward-to-go + normalize advantages": 3,
    }
    return sorted(runs, key=lambda run: order.get(run["label"], 99))


def plot_group(ax: plt.Axes, runs: list[dict], title: str) -> None:
    for run in sort_runs(runs):
        ax.plot(run["envsteps"], run["returns"], linewidth=2, label=run["label"])

    ax.set_title(title)
    ax.set_xlabel("Environment steps (Train_EnvstepsSoFar)")
    ax.set_ylabel("Average return (Eval_AverageReturn)")
    ax.grid(True, alpha=0.3)
    ax.legend()


def main() -> None:
    args = parse_args()
    small_batch_runs, large_batch_runs = collect_runs(args.exp_dir)

    if not small_batch_runs:
        raise ValueError("No small-batch CartPole runs found in the experiment directory.")
    if not large_batch_runs:
        raise ValueError("No large-batch CartPole runs found in the experiment directory.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    small_fig, small_ax = plt.subplots(figsize=(8, 5))
    plot_group(
        small_ax,
        small_batch_runs,
        "CartPole small batch experiments",
    )
    small_fig.tight_layout()
    small_fig.savefig(args.output_dir / "cartpole_small_batch.png", dpi=200)
    plt.close(small_fig)

    large_fig, large_ax = plt.subplots(figsize=(8, 5))
    plot_group(
        large_ax,
        large_batch_runs,
        "CartPole large batch experiments",
    )
    large_fig.tight_layout()
    large_fig.savefig(args.output_dir / "cartpole_large_batch.png", dpi=200)
    plt.close(large_fig)

    combined_fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    plot_group(axes[0], small_batch_runs, "CartPole small batch experiments")
    plot_group(axes[1], large_batch_runs, "CartPole large batch experiments")
    combined_fig.tight_layout()
    combined_fig.savefig(args.output_dir / "cartpole_learning_curves.png", dpi=200)
    plt.close(combined_fig)

    print(f"Saved plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()