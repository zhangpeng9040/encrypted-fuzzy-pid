"""Re-evaluate domain-wide fit errors, optionally refitting the quadratics."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from offline_helpers import fuzzy_engine, features, target, region_indices


def refit(edges, engine):
    coefficient = []
    for indices in np.ndindex(tuple(len(a)-1 for a in edges)):
        grid = [np.linspace(axis[i], axis[i+1], n) for axis, i, n in zip(edges, indices, [10, 8, 7])]
        ee, dd = np.meshgrid(grid[0], grid[1], indexing="ij")
        pairs = np.column_stack([ee.ravel(), dd.ravel()])
        s = np.array([engine._plain_throttle(float(e), float(d)) for e, d in pairs])
        kp = np.clip(1*(1+.5*s), .5, 1.5)
        ki = np.clip(.3*(1-.3*s), .5*.3, 1.5*.3)
        kd = np.clip(.001*(1+.2*abs(pairs[:, 1])), .5*.001, 1.5*.001)
        z = np.array(list(itertools.product(*grid)))
        y = np.clip((kp*pairs[:, 0]+kd*pairs[:, 1])[:, None]+ki[:, None]*grid[2], -1, 1).ravel()
        coefficient.append(np.linalg.lstsq(features(z), y, rcond=None)[0])
    return np.asarray(coefficient)


def main(args):
    protocol = json.loads((ROOT / "configs/partitions.json").read_text())
    engine = fuzzy_engine()
    rng = np.random.default_rng(20261020)
    z = rng.uniform([-15, -3, -20], [15, 3, 20], (12000, 3))
    truth = target(z, engine)
    expected = json.loads((ROOT / "configs/expected_results.json").read_text())["partitions"]
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for name, axes in protocol["partitions"].items():
        if args.grid and name != args.grid:
            continue
        edges = [np.asarray(a, dtype=float) for a in axes]
        recorded = np.load(ROOT / "coefficients/partitions" / (name + "_coeff.npy"))
        coefficient = recorded
        row = dict(partition=name, cells=len(recorded), test_inputs=len(z))
        if args.refit:
            began = time.perf_counter()
            coefficient = refit(edges, engine)
            row["refit_seconds"] = time.perf_counter()-began
            row["max_coefficient_difference"] = float(np.max(abs(recorded-coefficient)))
            np.save(args.output / (name + "_refitted.npy"), coefficient)
        prediction = np.clip(np.sum(features(z)*coefficient[region_indices(z, edges)], axis=1), -1, 1)
        row["output_rmse"] = float(np.sqrt(np.mean((prediction-truth)**2)))
        row["max_abs_residual"] = float(np.max(abs(prediction-truth)))
        row["coefficient_kib"] = coefficient.nbytes/1024
        assert abs(row["output_rmse"]-expected[name]["output_rmse"]) < 1e-10, row
        rows.append(row)
        print(json.dumps(row), flush=True)
    assert rows, "Unknown partition name"
    with (args.output / "approximation_checks.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "reproduced_results")
    p.add_argument("--refit", action="store_true")
    p.add_argument("--grid", help="Optional partition label, for example base_8x8x6")
    main(p.parse_args())
