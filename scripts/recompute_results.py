"""Recompute the paper's main result tables from retained CSV records.

Only the Python standard library is needed. All outputs are derived and are
written to reproduced_results, leaving the recorded data unchanged.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(trace, requests):
    error = [float(r["error"]) for r in trace]
    # Independently check that the error is reference minus same-sample speed.
    assert all(abs(float(r["ref"])-float(r["speed"])-float(r["error"])) < 1e-11 for r in trace)
    latency = [float(r["total_ms"]) for r in requests]
    return dict(steps=len(trace), requests=len(requests),
                RMSE=math.sqrt(statistics.fmean(x*x for x in error)),
                MAE=statistics.fmean(abs(x) for x in error),
                Max_Error=max(abs(x) for x in error), Overshoot=max(0., -min(error)),
                mean_ms=statistics.fmean(latency), max_ms=max(latency),
                over100=sum(x > 100 for x in latency))


def verify_manifest():
    path = ROOT / "MANIFEST.sha256"
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        file = ROOT / name
        assert file.is_file(), "Missing file: " + name
        assert hashlib.sha256(file.read_bytes()).hexdigest() == digest, "Checksum mismatch: " + name
        count += 1
    return count


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checked = verify_manifest() if not args.skip_manifest else 0
    fuzzy_requests = read(ROOT / "data/fuzzy_comparison/requests.csv")
    result = {}
    for name, directory, mode in [
        ("PID", "pid_plain", "pid_plain"), ("CKKS-PID", "ckks_pid", "pid_packed")]:
        folder = ROOT / "data" / directory
        result[name] = metrics(read(folder / "trajectory.csv"), read(folder / "requests.csv"))
    for name, mode in [("Chen-based Fuzzy-PID", "chen"), ("Fuzzy-PID", "fuzzy"),
                       ("Poly-Fuzzy-PID", "poly"), ("CKKS-Fuzzy-PID", "ckks_poly")]:
        result[name] = metrics(read(ROOT / "data/fuzzy_comparison" / (mode + "_trajectory.csv")),
                               [r for r in fuzzy_requests if r["mode"] == mode])
    expected = json.loads((ROOT / "configs/expected_results.json").read_text(encoding="utf-8"))
    for name, actual in result.items():
        for key in ["steps", "requests", "RMSE", "MAE", "Max_Error", "Overshoot", "mean_ms", "max_ms", "over100"]:
            reference = expected["main"][name][key]
            assert abs(actual[key] - reference) < 2e-10, (name, key, actual[key], reference)
    rows = [dict(controller=name, **values) for name, values in result.items()]
    write(out / "main_metrics.csv", rows)
    for number, names in [(7, ["PID", "CKKS-PID"]),
                          (8, ["Chen-based Fuzzy-PID", "Fuzzy-PID"]),
                          (9, ["Fuzzy-PID", "Poly-Fuzzy-PID", "CKKS-Fuzzy-PID"]),
                          (10, ["CKKS-PID", "CKKS-Fuzzy-PID"])]:
        write(out / f"table_{number:02}.csv", [dict(controller=name, **result[name]) for name in names])
    fit = {r["partition"]: r for r in read(ROOT / "data/partition_comparison/fit_summary.csv")}
    q = read(ROOT / "data/partition_comparison/requests.csv")
    partition = []
    for mode in ["fuzzy", "global_1x1x1", "coarse_4x4x3", "base_8x8x6", "fine_16x16x12"]:
        trace = read(ROOT / "data/partition_comparison" / (mode + "_trajectory.csv"))
        m = metrics(trace, [r for r in q if r["mode"] == mode])
        assert abs(m["RMSE"] - expected["partitions"][mode]["tracking_rmse"]) < 1e-12
        f = fit.get(mode)
        partition.append(dict(grid=mode, cells=int(f["regions"]) if f else "",
                              output_rmse=float(f["rmse"]) if f else "", tracking_rmse=m["RMSE"],
                              coefficient_kib=int(f["coefficient_bytes"])/1024 if f else ""))
    write(out / "table_06.csv", partition)
    condition_rows = []
    for condition in ["nominal", "noise", "actuator", "disturbance"]:
        rmse, mae = [], []
        for repetition in range(3):
            trace = read(ROOT / "data/operating_conditions" / f"{condition}_baseline_r{repetition}.csv")
            assert len(trace) == 1200
            errors = [float(r["ref"])-float(r["speed"]) for r in trace]
            rmse.append(math.sqrt(statistics.fmean(x*x for x in errors)))
            mae.append(statistics.fmean(abs(x) for x in errors))
        row = dict(condition=condition, runs=3, samples_per_run=1200,
                   RMSE=statistics.fmean(rmse), MAE=statistics.fmean(mae))
        for key in ["RMSE", "MAE"]:
            assert abs(row[key] - expected["conditions"][condition][key]) < 1e-12
        condition_rows.append(row)
    write(out / "table_12.csv", condition_rows)
    matched = read(ROOT / "data/control_output_errors/matched_errors.csv")
    assert len(matched) == 1228
    rmse = lambda key: math.sqrt(statistics.fmean(float(r[key])**2 for r in matched))
    errors = dict(updates=len(matched), fitting_rmse=rmse("fitting_error"),
                  ckks_rmse=rmse("ckks_raw_error"), units="normalized control-command units")
    assert abs(errors["fitting_rmse"]-expected["control_errors"]["fitting_rmse"]) < 1e-12
    assert abs(errors["ckks_rmse"]-expected["control_errors"]["ckks_rmse"]) < 1e-12
    ratios = dict(poly_vs_fuzzy_rmse_reduction_percent=(1-result["CKKS-Fuzzy-PID"]["RMSE"]/result["Fuzzy-PID"]["RMSE"])*100,
                  poly_vs_fuzzy_mae_reduction_percent=(1-result["CKKS-Fuzzy-PID"]["MAE"]/result["Fuzzy-PID"]["MAE"])*100,
                  fuzzy_vs_chen_rmse_reduction_percent=(1-result["Fuzzy-PID"]["RMSE"]/result["Chen-based Fuzzy-PID"]["RMSE"])*100,
                  poly_vs_pid_rmse_reduction_percent=(1-result["CKKS-Fuzzy-PID"]["RMSE"]/result["CKKS-PID"]["RMSE"])*100,
                  fit_to_ckks_rmse_ratio=errors["fitting_rmse"]/errors["ckks_rmse"])
    report = dict(status="passed", files_verified=checked, controllers=len(result),
                  formal_requests=sum(x["requests"] for x in result.values()),
                  partition_rows=len(partition), condition_runs=12,
                  control_errors=errors, derived_comparisons=ratios,
                  data_scope="Frozen recorded runs; analysis does not rerun simulation or encryption.")
    (out / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced_results")
    parser.add_argument("--skip-manifest", action="store_true")
    main(parser.parse_args())
