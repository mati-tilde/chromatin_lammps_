#!/usr/bin/env python3
"""Controlled LAMMPS sweep for the single-chromosome model.

From the 19Oct_sweep directory:
    python run_sweep.py --dry-run
    python run_sweep.py --lmp /path/to/lmp

The initial grid is 4 timesteps x 4 common B-matrix values = 16 runs.
This script does not change the LJ pair_coeff values in the template.
It disables trajectory dumps and restart output for this first thermodynamic sweep.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import subprocess
import sys
from pathlib import Path

TIMESTEPS = [0.0025, 0.005, 0.01, 0.02]
B_VALUES = [0.00, 0.10, 0.25, 0.50]
B_PAIRS = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
THERMO_COLUMNS = ["Step", "Temp", "PotEng", "TotEng", "E_vdwl"]


def replace_single(pattern: str, replacement: str, text: str, description: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(
            f"Expected exactly one {description} line matching {pattern!r}; found {count}."
        )
    return updated


def make_run_input(template: str, run_id: str, timestep: float, b_value: float) -> str:
    text = replace_single(
        r"^\s*log\s+\S+.*$",
        f"log results/logs/{run_id}.lammps.log",
        template,
        "log",
    )
    text = replace_single(
        r"^\s*timestep\s+\S+.*$",
        f"timestep {timestep:g}",
        text,
        "timestep",
    )

    for i, j in B_PAIRS:
        pattern = rf"^\s*pair_coeff\s+B\s+{i}\s+{j}\s+\S+.*$"
        replacement = f"pair_coeff B {i} {j} {b_value:g}"
        text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(
                f"Expected one B-matrix entry for pair ({i},{j}); found {count}."
            )

    text, dump_count = re.subn(
        r"^\s*dump\s+sim\b.*$",
        lambda match: "# disabled for initial sweep: " + match.group(0).lstrip(),
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if dump_count != 1:
        raise ValueError("Expected exactly one 'dump sim ...' line to disable.")

    text, restart_count = re.subn(
        r"^\s*restart\b.*$",
        "# disabled for initial sweep: restart output not needed",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if restart_count != 1:
        raise ValueError("Expected exactly one restart line to disable.")

    return text


def find_data_file(template: str, root: Path) -> Path:
    match = re.search(r"^\s*read_data\s+(\S+)", template, flags=re.MULTILINE)
    if not match:
        raise ValueError("Could not find a read_data command in the input template.")
    data_path = Path(match.group(1))
    if not data_path.is_absolute():
        data_path = root / data_path
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Data file referenced by read_data was not found: {data_path}\n"
            "Put the data file beside the input template or update the read_data path."
        )
    return data_path


def parse_thermo(log_path: Path) -> list[dict[str, str]]:
    """Extract rows printed under the production thermo_style header."""
    rows: list[dict[str, str]] = []
    reading = False

    if not log_path.exists():
        return rows

    with log_path.open("r", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()

            if fields == THERMO_COLUMNS:
                reading = True
                continue

            if reading:
                if line.startswith(("Loop time", "ERROR:")):
                    reading = False
                    continue
                if len(fields) == len(THERMO_COLUMNS):
                    try:
                        float(fields[0])
                        for value in fields[1:]:
                            float(value)
                    except ValueError:
                        continue
                    rows.append(dict(zip(THERMO_COLUMNS, fields)))

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 19Oct single-chromosome sweep.")
    parser.add_argument(
        "--input",
        default="informed_lammps.in",
        help="Input template filename (default: informed_lammps.in)",
    )
    parser.add_argument(
        "--lmp",
        default="lmp",
        help="LAMMPS executable or path to it (default: lmp on PATH)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate run-specific inputs without launching LAMMPS.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of combinations for a small pilot.",
    )
    args = parser.parse_args()

    root = Path.cwd().resolve()
    template_path = (root / args.input).resolve()
    if not template_path.is_file():
        print(
            f"ERROR: input template not found: {template_path}\n"
            "Run this script from 19Oct_sweep, or pass --input.",
            file=sys.stderr,
        )
        return 2

    template = template_path.read_text()
    try:
        find_data_file(template, root)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    inputs_dir = root / "run_inputs"
    results_dir = root / "results"
    logs_dir = results_dir / "logs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    combinations = list(itertools.product(TIMESTEPS, B_VALUES))
    if args.limit is not None:
        if args.limit < 1:
            print("ERROR: --limit must be at least 1.", file=sys.stderr)
            return 2
        combinations = combinations[: args.limit]

    print(f"Working directory: {root}")
    print(f"Input template:    {template_path.name}")
    print(f"Combinations:      {len(combinations)}")
    print(f"Timesteps:         {TIMESTEPS}")
    print(f"Common B values:   {B_VALUES}")

    planned_runs = []
    for index, (dt, b_value) in enumerate(combinations, start=1):
        run_id = f"run_{index:02d}_dt{dt:g}_B{b_value:g}".replace(".", "p")
        run_text = make_run_input(template, run_id, dt, b_value)
        run_input_path = inputs_dir / f"{run_id}.in"
        run_input_path.write_text(run_text)
        planned_runs.append((run_id, dt, b_value, run_input_path))

    if args.dry_run:
        print("\nDRY RUN ONLY: generated inputs; LAMMPS was not launched.")
        print(f"Generated inputs: {inputs_dir}")
        print("Inspect a generated input before running the sweep.")
        return 0

    csv_path = results_dir / "sweep_results.csv"
    summary_path = results_dir / "sweep_status.csv"
    all_rows: list[dict[str, str]] = []
    status_rows: list[dict[str, str]] = []

    for run_id, dt, b_value, run_input_path in planned_runs:
        stdout_path = logs_dir / f"{run_id}.stdout.log"
        lammps_log_path = logs_dir / f"{run_id}.lammps.log"
        print(f"[{run_id}] starting", flush=True)

        with stdout_path.open("w") as stdout_file:
            try:
                completed = subprocess.run(
                    [args.lmp, "-in", str(run_input_path)],
                    cwd=root,
                    stdout=stdout_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                return_code = completed.returncode
                error_text = ""
            except OSError as exc:
                return_code = 127
                error_text = str(exc)

        thermo_rows = parse_thermo(lammps_log_path)
        for row in thermo_rows:
            step = float(row["Step"])
            row.update(
                {
                    "run_id": run_id,
                    "timestep": f"{dt:g}",
                    "B_common": f"{b_value:g}",
                    "simulation_time": f"{step * dt:g}",
                }
            )
            all_rows.append(row)

        status = "success" if return_code == 0 else "failed"
        status_rows.append(
            {
                "run_id": run_id,
                "timestep": f"{dt:g}",
                "B_common": f"{b_value:g}",
                "return_code": str(return_code),
                "status": status,
                "thermo_rows": str(len(thermo_rows)),
                "error": error_text,
                "stdout_log": str(stdout_path.relative_to(root)),
                "lammps_log": str(lammps_log_path.relative_to(root)),
            }
        )
        print(
            f"[{run_id}] {status}; thermo rows collected: {len(thermo_rows)}",
            flush=True,
        )

    thermo_fields = [
        "run_id", "timestep", "B_common", "Step", "simulation_time",
        "Temp", "PotEng", "TotEng", "E_vdwl",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=thermo_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    status_fields = [
        "run_id", "timestep", "B_common", "return_code", "status",
        "thermo_rows", "error", "stdout_log", "lammps_log",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=status_fields)
        writer.writeheader()
        writer.writerows(status_rows)

    failed = sum(row["status"] != "success" for row in status_rows)
    print(f"\nThermo results: {csv_path}")
    print(f"Run status:     {summary_path}")
    print(f"Runs failed:    {failed}/{len(status_rows)}")
    print(f"Detailed logs:  {logs_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
