import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


HALFCHEETAH_BGS_RUNS = [
    ("cheetah_baseline", "baseline (bgs=5)"),
    ("cheetah_baseline_bgs2", "bgs=2"),
    ("cheetah_baseline_bgs10", "bgs=10"),
    ("cheetah_baseline_bgs15", "bgs=15"),
]

HALFCHEETAH_BLR_RUNS = [
    ("cheetah_baseline", "baseline (blr=0.01)"),
    ("cheetah_baseline_blr0.001", "blr=0.001"),
    ("cheetah_baseline_blr0.1", "blr=0.1"),
]

LUNARLANDER_RUNS = [
    ("lunar_lander_lambda0", "lambda=0"),
    ("lunar_lander_lambda0.95", "lambda=0.95"),
    ("lunar_lander_lambda0.98", "lambda=0.98"),
    ("lunar_lander_lambda0.99", "lambda=0.99"),
    ("lunar_lander_lambda1", "lambda=1"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot HalfCheetah and LunarLander-v2 curves from HW2 experiment logs."
    )
    parser.add_argument(
        "--exp_dir",
        type=Path,
        default=Path("exp"),
        help="Directory containing experiment subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("exp_plots"),
        help="Directory where output figures will be saved.",
    )
    return parser.parse_args()


def load_flags(flags_path: Path) -> dict:
    with flags_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_latest_run(exp_dir: Path, env_name: str, exp_name: str) -> Path:
    matches = []

    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        flags_path = run_dir / "flags.json"
        log_path = run_dir / "log.csv"
        if not flags_path.exists() or not log_path.exists():
            continue

        flags = load_flags(flags_path)
        if flags.get("env_name") != env_name:
            continue
        if flags.get("exp_name") != exp_name:
            continue

        matches.append(run_dir)

    if not matches:
        raise FileNotFoundError(
            f"Could not find a run for env={env_name!r}, exp_name={exp_name!r} in {exp_dir}."
        )

    return sorted(matches, key=lambda path: path.name)[-1]


def load_metric_series(log_path: Path, metric_name: str) -> tuple[list[float], list[float]]:
    x_values = []
    y_values = []

    with log_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or metric_name not in reader.fieldnames:
            raise KeyError(f"Metric {metric_name!r} not found in {log_path}.")

        x_key = "Train_EnvstepsSoFar" if "Train_EnvstepsSoFar" in reader.fieldnames else "step"
        for row in reader:
            x_values.append(float(row[x_key]))
            y_values.append(float(row[metric_name]))

    return x_values, y_values


def collect_runs(exp_dir: Path, env_name: str, run_specs: list[tuple[str, str]]) -> list[dict]:
    runs = []

    for exp_name, label in run_specs:
        run_dir = find_latest_run(exp_dir, env_name, exp_name)
        runs.append(
            {
                "exp_name": exp_name,
                "label": label,
                "run_dir": run_dir,
                "log_path": run_dir / "log.csv",
            }
        )

    return runs


def plot_runs(runs: list[dict], metric_name: str, title: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for run in runs:
        x_values, y_values = load_metric_series(run["log_path"], metric_name)
        ax.plot(x_values, y_values, linewidth=2, label=run["label"])

    ax.set_title(title)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_run_summary(header: str, runs: list[dict]) -> None:
    print(header)
    for run in runs:
        print(f"  {run['label']}: {run['run_dir']}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    halfcheetah_bgs_runs = collect_runs(
        args.exp_dir,
        env_name="HalfCheetah-v4",
        run_specs=HALFCHEETAH_BGS_RUNS,
    )
    halfcheetah_blr_runs = collect_runs(
        args.exp_dir,
        env_name="HalfCheetah-v4",
        run_specs=HALFCHEETAH_BLR_RUNS,
    )
    lunarlander_runs = collect_runs(
        args.exp_dir,
        env_name="LunarLander-v2",
        run_specs=LUNARLANDER_RUNS,
    )

    plot_runs(
        halfcheetah_bgs_runs,
        metric_name="Baseline Loss",
        title="HalfCheetah-v4 baseline loss: bgs sweep",
        ylabel="Baseline Loss",
        output_path=args.output_dir / "halfcheetah_bgs_baseline_loss.png",
    )
    plot_runs(
        halfcheetah_bgs_runs,
        metric_name="Eval_AverageReturn",
        title="HalfCheetah-v4 eval return: bgs sweep",
        ylabel="Eval Average Return",
        output_path=args.output_dir / "halfcheetah_bgs_eval_return.png",
    )
    plot_runs(
        halfcheetah_blr_runs,
        metric_name="Baseline Loss",
        title="HalfCheetah-v4 baseline loss: blr sweep",
        ylabel="Baseline Loss",
        output_path=args.output_dir / "halfcheetah_blr_baseline_loss.png",
    )
    plot_runs(
        halfcheetah_blr_runs,
        metric_name="Eval_AverageReturn",
        title="HalfCheetah-v4 eval return: blr sweep",
        ylabel="Eval Average Return",
        output_path=args.output_dir / "halfcheetah_blr_eval_return.png",
    )
    plot_runs(
        lunarlander_runs,
        metric_name="Eval_AverageReturn",
        title="LunarLander-v2 eval return: lambda sweep",
        ylabel="Eval Average Return",
        output_path=args.output_dir / "lunarlander_lambda_eval_return.png",
    )

    print_run_summary("HalfCheetah bgs runs:", halfcheetah_bgs_runs)
    print_run_summary("HalfCheetah blr runs:", halfcheetah_blr_runs)
    print_run_summary("LunarLander runs:", lunarlander_runs)
    print(f"Saved plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()