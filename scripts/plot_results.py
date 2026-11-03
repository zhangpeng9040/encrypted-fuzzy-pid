"""Create readable figures from the retained full-course trajectories."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main(args):
    names = {
        "PID": ROOT / "data/pid_plain/trajectory.csv",
        "CKKS-PID": ROOT / "data/ckks_pid/trajectory.csv",
        "Fuzzy-PID": ROOT / "data/fuzzy_comparison/fuzzy_trajectory.csv",
        "Chen-based Fuzzy-PID": ROOT / "data/fuzzy_comparison/chen_trajectory.csv",
        "Poly-Fuzzy-PID": ROOT / "data/fuzzy_comparison/poly_trajectory.csv",
        "CKKS-Fuzzy-PID": ROOT / "data/fuzzy_comparison/ckks_poly_trajectory.csv"}
    traces = {name: read(path) for name, path in names.items()}
    course = np.loadtxt(ROOT / "data/racetrack_waypoints.txt", delimiter=",")
    groups = {"figure03_pid": ["PID", "CKKS-PID"],
              "figure04_fuzzy": ["Fuzzy-PID", "Chen-based Fuzzy-PID"],
              "figure05_encrypted_fuzzy": ["Fuzzy-PID", "Poly-Fuzzy-PID", "CKKS-Fuzzy-PID"],
              "figure06_encrypted_comparison": ["CKKS-PID", "CKKS-Fuzzy-PID"]}
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    for stem, controllers in groups.items():
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes[0, 0].plot(course[:, 0], course[:, 1], "k--", lw=1., label="Reference path")
        for name in controllers:
            trace = traces[name];time = np.arange(len(trace))*.1
            axes[0, 0].plot([float(r["x"]) for r in trace], [float(r["y"]) for r in trace], label=name)
            axes[0, 1].plot(time, [float(r["speed"]) for r in trace], label=name)
            axes[1, 0].plot(time, [float(r["error"]) for r in trace], label=name, lw=.8)
            error = np.array([float(r["error"]) for r in trace])
            metrics = [np.sqrt(np.mean(error**2)), np.mean(abs(error)), np.max(abs(error)), max(0., -min(error))]
            x = np.arange(4)+(controllers.index(name)-(len(controllers)-1)/2)*.22
            axes[1, 1].bar(x, metrics, width=.22, label=name)
        first = traces[controllers[0]]
        axes[0, 1].plot(np.arange(len(first))*.1, [float(r["ref"]) for r in first], "k--", lw=1., label="Reference speed")
        axes[0, 0].set(xlabel="x (m)", ylabel="y (m)")
        axes[0, 1].set(xlabel="Time (s)", ylabel="Speed (m/s)")
        axes[1, 0].set(xlabel="Time (s)", ylabel="Speed error (m/s)")
        axes[1, 1].set(xticks=np.arange(4), xticklabels=["RMSE", "MAE", "Max error", "Overshoot"], ylabel="m/s")
        for ax in axes.ravel():
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        figure.savefig(args.output / (stem + ".png"), dpi=180)
        figure.savefig(args.output / (stem + ".pdf"))
        plt.close(figure)
    print("Created four figure groups from the recorded trajectories.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "reproduced_results/figures")
    main(p.parse_args())
