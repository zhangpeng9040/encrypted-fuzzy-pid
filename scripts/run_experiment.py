"""Prepare isolated working directories and run the released experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "pid_plain": ("run_pid_twohost_fullcourse.py", "run_pid_twohost_fullcourse.py", 47857),
    "ckks_pid": ("run_packed_pid_twohost.py", "run_packed_pid_twohost.py", 47862),
    "fuzzy": ("run_original_four_twohost.py", "run_original_four_twohost.py", 47856),
    "partitions": ("run_original_partition_twohost.py", "run_original_partition_twohost.py", 47858),
    "conditions": ("final_evidence_twohost.py", "final_evidence_server.py", 47843),
}


def main(args):
    work = args.workdir.resolve()
    if work == ROOT or ROOT in work.parents and work.parts[len(ROOT.parts)] in ["data", "coefficients", "code", "configs"]:
        raise ValueError("Choose a separate directory under runs, or outside this package")
    work.mkdir(parents=True, exist_ok=False)
    for source in (ROOT / "code").glob("*.py"):
        shutil.copy2(source, work / source.name)
    shutil.copy2(ROOT / "data/racetrack_waypoints.txt", work / "racetrack_waypoints.txt")
    (work / "revision_results").mkdir()
    shutil.copy2(ROOT / "coefficients/piecewise_coefficients.npy", work / "revision_results/piecewise_coefficients.npy")
    (work / "original_partition_ablation").mkdir()
    shutil.copy2(ROOT / "configs/partitions.json", work / "original_partition_ablation/ablation_protocol.json")
    for path in (ROOT / "coefficients/partitions").glob("*.npy"):
        shutil.copy2(path, work / "original_partition_ablation" / path.name)
    (work / "pid_twohost_fullcourse").mkdir()
    route_hash = hashlib.sha256((work / "racetrack_waypoints.txt").read_bytes()).hexdigest()
    (work / "pid_twohost_fullcourse/protocol.json").write_text(json.dumps(dict(route_sha256=route_hash)), encoding="utf-8")
    client, server, port = CASES[args.case]
    plan = dict(case=args.case, role=args.role, server_host=args.host, port=port,
                entry=server if args.role == "server" else client,
                source_data="Frozen data remain under the package data directory")
    (work / "run_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    if args.role == "prepare":
        return
    env = dict(os.environ, FUZZY_SERVER_HOST=args.host, PYTHONIOENCODING="utf-8",
               PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    command = [sys.executable, "-u", server if args.role == "server" else client]
    if args.role == "server":
        command.append("--server")
    if args.case == "ckks_pid" and args.role == "client":
        command += ["--host", args.host]
    subprocess.run(command, cwd=work, env=env, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--role", choices=["prepare", "server", "client"], required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--workdir", type=Path, required=True)
    main(p.parse_args())
