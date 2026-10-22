#!/usr/bin/env python3
"""Run a controlled LAMMPS thermostat comparison at dt=0.005.

For each common B-matrix value (0, 0.10, 0.25, 0.30, 0.40, 0.50),
run one Langevin simulation and one Nosé-Hoover NVT simulation.

The script reads informed_lammps.in as a template, changes only the timestep,
the six B coefficients, and thermostat fixes, and gives each run its own
folder, input, log, dump, and output files.

Usage from the directory containing informed_lammps.in:
    python run_thermostat_sweep.py
    python run_thermostat_sweep.py --lmp /path/to/lmp
    LAMMPS_EXECUTABLE=/path/to/lmp python run_thermostat_sweep.py
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TIMESTEP = 0.005
B_VALUES = [0.00, 0.10, 0.25, 0.30, 0.40, 0.50]
THERMOSTATS = ["langevin", "nvt"]
TARGET_TEMPERATURE = 1.0
NVT_TDAMP = 1.0
OUTPUT_DIR_NAME = "results/thermostat_sweep_dt0p005"


def replace_one(pattern: str, replacement: str, text: str, description: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one match for {description}, found {count}. "
            "The template may have changed; inspect informed_lammps.in."
        )
    return updated


def fmt_b(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def make_input(template: str, thermostat: str, b_value: float) -> str:
    text = replace_one(
        r"^\s*timestep\s+\S+.*$", f"timestep {TIMESTEP:g}", template, "timestep line"
    )

    # Let the command-line -log argument control the per-run logfile.
    text = replace_one(
        r"^\s*log\s+\S+.*$",
        "# Logfile is set by the command-line -log argument",
        text,
        "log command",
    )

    # Replace all six entries in the symmetric common-B matrix.
    for i, j in [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]:
        text = replace_one(
            rf"^\s*pair_coeff\s+B\s+{i}\s+{j}\s+\S+.*$",
            f"pair_coeff B {i} {j} {b_value:g}",
            text,
            f"B coefficient ({i}, {j})",
        )

    if thermostat == "langevin":
        # Keep the template's NVE + Langevin fixes and Langevin random seed.
        if not re.search(r"^\s*fix\s+\S+\s+rest\s+nve\b", text, re.MULTILINE):
            raise RuntimeError("Could not find the existing NVE fix for Langevin.")
        if not re.search(r"^\s*fix\s+\S+\s+rest\s+langevin\b", text, re.MULTILINE):
            raise RuntimeError("Could not find the existing Langevin fix.")
    elif thermostat == "nvt":
        # NVT integrates motion itself, so it replaces both NVE and Langevin.
        text = replace_one(
            r"^\s*fix\s+\S+\s+rest\s+nve\b.*$",
            f"fix 1 rest nvt temp {TARGET_TEMPERATURE:g} {TARGET_TEMPERATURE:g} {NVT_TDAMP:g}",
            text,
            "NVE fix (to replace with NVT)",
        )
        text = replace_one(
            r"^\s*fix\s+\S+\s+rest\s+langevin\b.*$",
            "# Langevin fix removed for NVT comparison",
            text,
            "Langevin fix (to remove for NVT)",
        )
    else:
        raise ValueError(f"Unknown thermostat: {thermostat}")

    # Give each run its own trajectory file if the template contains a standard dump.
    text, _ = re.subn(
        r"(?m)^(\s*dump\s+\S+\s+\S+\s+custom\s+\S+\s+)\S+",
        r"\1trajectory.lammpstrj",
        text,
        count=1,
    )
    return text


def get_read_data_file(template: str) -> Path:
    match = re.search(r"(?m)^\s*read_data\s+(\S+)", template)
    if not match:
        raise RuntimeError("Could not find a read_data command in informed_lammps.in.")
    return Path(match.group(1))


def parse_thermo(log_path: Path, run_id: str, b_value: float, thermostat: str) -> list[dict]:
    """Read numeric rows from LAMMPS thermo blocks in a log file."""
    if not log_path.exists():
        return []

    rows: list[dict] = []
    header: list[str] | None = None
    with log_path.open("r", errors="replace") as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if not parts:
                continue
            if parts[0] == "Step" and "Temp" in parts and (
                "PotEng" in parts or "E_pair" in parts
            ):
                header = parts
                continue
            if header is None or len(parts) != len(header):
                continue
            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue
            record = dict(zip(header, values))
            if "Step" not in record or "Temp" not in record:
                continue
            rows.append({
                "run_id": run_id,
                "timestep": TIMESTEP,
                "B_common": b_value,
                "thermostat": thermostat,
                "Step": int(record["Step"]),
                "simulation_time": record["Step"] * TIMESTEP,
                "Temp": record.get("Temp", ""),
                "PotEng": record.get("PotEng", record.get("Pe", "")),
                "TotEng": record.get("TotEng", ""),
                "E_vdwl": record.get("E_vdwl", ""),
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lmp", default=os.environ.get("LAMMPS_EXECUTABLE", "lmp"),
        help="LAMMPS executable/path (default: $LAMMPS_EXECUTABLE or 'lmp')",
    )
    parser.add_argument("--template", default="informed_lammps.in", help="Input template")
    parser.add_argument("--output", default=OUTPUT_DIR_NAME, help="Output directory")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate run inputs and directories without launching LAMMPS",
    )
    args = parser.parse_args()

    project_dir = Path.cwd().resolve()
    template_path = (project_dir / args.template).resolve()
    output_root = (project_dir / args.output).resolve()
    if not template_path.is_file():
        print(f"ERROR: template not found: {template_path}", file=sys.stderr)
        return 2

    lmp_arg = Path(args.lmp)
    executable = str(lmp_arg.resolve()) if lmp_arg.is_file() else shutil.which(args.lmp)
    if executable is None:
        print(
            f"ERROR: cannot find LAMMPS executable '{args.lmp}'. "
            "Pass --lmp /path/to/lmp or set LAMMPS_EXECUTABLE.", file=sys.stderr,
        )
        return 2

    template = template_path.read_text()
    data_file = get_read_data_file(template)
    source_data = (project_dir / data_file).resolve() if not data_file.is_absolute() else data_file
    if not source_data.is_file():
        print(f"ERROR: read_data file not found: {source_data}", file=sys.stderr)
        return 2

    output_root.mkdir(parents=True, exist_ok=True)
    thermo_rows: list[dict] = []
    status_rows: list[dict] = []
    combinations = [(b, thermostat) for b in B_VALUES for thermostat in THERMOSTATS]

    print(f"Project:     {project_dir}")
    print(f"LAMMPS:      {executable}")
    print(f"Template:    {template_path}")
    print(f"Output:      {output_root}")
    print(f"Timestep:    {TIMESTEP}")
    print(f"B values:    {B_VALUES}")
    print(f"Thermostats: {THERMOSTATS}")
    print(f"Total runs:  {len(combinations)}\n")

    for index, (b_value, thermostat) in enumerate(combinations, start=1):
        run_id = f"run_{index:02d}_B{fmt_b(b_value)}_{thermostat}"
        run_dir = output_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        input_path = run_dir / "in.lammps"
        input_path.write_text(make_input(template, thermostat, b_value))

        # Copy the data file into the run folder so relative read_data paths still work.
        relative_data = data_file if not data_file.is_absolute() else Path(data_file.name)
        destination_data = run_dir / relative_data
        destination_data.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_data, destination_data)

        log_path = run_dir / "log.lammps"
        status, return_code, error = "prepared", "", ""
        print(f"[{index:02d}/{len(combinations)}] {run_id}", flush=True)
        if not args.dry_run:
            try:
                with (run_dir / "stdout.txt").open("w") as stdout_handle:
                    completed = subprocess.run(
                        [executable, "-in", str(input_path), "-log", str(log_path)],
                        cwd=run_dir,
                        stdout=stdout_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                return_code = completed.returncode
                status = "success" if completed.returncode == 0 else "failed"
                if completed.returncode != 0:
                    error = f"LAMMPS exited with return code {completed.returncode}"
            except Exception as exc:
                status, error = "failed", repr(exc)

            run_rows = parse_thermo(log_path, run_id, b_value, thermostat)
            thermo_rows.extend(run_rows)
            if status == "success" and not run_rows:
                status = "warning_no_thermo_rows"
                error = "LAMMPS returned 0, but no thermo rows were parsed from log.lammps"

        status_rows.append({
            "run_id": run_id,
            "timestep": TIMESTEP,
            "B_common": b_value,
            "thermostat": thermostat,
            "return_code": return_code,
            "status": status,
            "thermo_rows": 0 if args.dry_run else len(parse_thermo(log_path, run_id, b_value, thermostat)),
            "run_dir": str(run_dir),
            "error": error,
        })

        # Save after each run, so a later failure does not lose previous results.
        with (output_root / "sweep_status.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(status_rows[0].keys()))
            writer.writeheader()
            writer.writerows(status_rows)
        with (output_root / "sweep_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "run_id", "timestep", "B_common", "thermostat", "Step",
                "simulation_time", "Temp", "PotEng", "TotEng", "E_vdwl",
            ])
            writer.writeheader()
            writer.writerows(thermo_rows)

    failed = [r for r in status_rows if r["status"] not in ("success", "prepared")]
    print(f"\nStatus CSV:  {output_root / 'sweep_status.csv'}")
    print(f"Thermo CSV:  {output_root / 'sweep_results.csv'}")
    print(f"Run folders: {output_root / 'runs'}")
    print(f"Successful:  {sum(r['status'] == 'success' for r in status_rows)}/{len(status_rows)}")
    if failed:
        print("Runs needing inspection:")
        for row in failed:
            print(f"  {row['run_id']}: {row['status']} {row['error']}")
        return 1
    if args.dry_run:
        print("Dry run only: inputs generated, LAMMPS not launched.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
