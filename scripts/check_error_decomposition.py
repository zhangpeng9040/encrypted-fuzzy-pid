"""Reconstruct the matched-input error analysis from the full CKKS trajectory."""
import argparse
from collections import deque
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from offline_helpers import fuzzy_engine


def main(args):
    with (ROOT / "data/fuzzy_comparison/ckks_poly_trajectory.csv").open(encoding="utf-8-sig", newline="") as f:
        trace = list(csv.DictReader(f))
    engine = fuzzy_engine()
    coefficients = np.load(ROOT / "coefficients/piecewise_coefficients.npy")
    applied = {}
    for row in trace:
        sequence = int(row["applied_seq"])
        if sequence >= 0:
            assert int(row["step"])-sequence == 1
            assert sequence not in applied
            applied[sequence] = float(row["command"])
    previous = 0.
    buffer = deque(maxlen=20)
    rows = []
    for row in trace:
        step = int(row["step"])
        e = float(row["error"])
        d = (e-previous)/.1 if step else 0.
        previous = e
        buffer.append(e)
        z = np.clip([e, d, sum(buffer)*.1], [-15, -3, -20], [15, 3, 20])
        if step not in applied:
            continue
        index = engine._region_index(*z)
        raw = float(engine.polynomial_features(*z) @ coefficients[index])
        polynomial = float(np.clip(raw, -1, 1))
        fuzzy = float(engine._pid_throttle(z[0], z[1], ie_sum=z[2]))
        encrypted = applied[step]
        assert abs(encrypted) < 1. and abs(raw) < 1.
        fit = polynomial-fuzzy
        eta = encrypted-raw
        assert abs(encrypted-polynomial) <= abs(eta)+1e-14
        assert abs(encrypted-fuzzy) <= abs(fit)+abs(eta)+1e-14
        rows.append(dict(input_step=step, actuation_step=step+1, e=z[0], de=z[1], integral=z[2],
                         region=index, fuzzy_output=fuzzy, polynomial_raw=raw, recorded_output=encrypted,
                         fitting_error=fit, ckks_error=eta))
    assert len(rows) == 1228
    fit_rmse = float(np.sqrt(np.mean([r["fitting_error"]**2 for r in rows])))
    eta_rmse = float(np.sqrt(np.mean([r["ckks_error"]**2 for r in rows])))
    expected = json.loads((ROOT / "configs/expected_results.json").read_text())["control_errors"]
    assert abs(fit_rmse-expected["fitting_rmse"]) < 1e-12
    assert abs(eta_rmse-expected["ckks_rmse"]) < 1e-12
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "reconstructed_control_errors.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report = dict(status="passed", matched_updates=1228, fitting_rmse=fit_rmse, ckks_rmse=eta_rmse,
                  ratio=fit_rmse/eta_rmse, units="normalized control-command units",
                  both_pointwise_inequalities_hold=True, final_request_excluded=1228)
    (args.output / "error_decomposition.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "reproduced_results")
    main(p.parse_args())
