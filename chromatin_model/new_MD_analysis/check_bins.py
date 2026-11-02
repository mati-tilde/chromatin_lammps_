#!/usr/bin/env python3
"""
check_bins.py -- H0 check: the bead -> genomic bin table must match the LAMMPS data file.

Usage:  python3 check_bins.py DATAFILE BINS.tsv [--bed BEDFILE]
Checks: same atoms (ids, molecule ids, frozen = type 2); each copy is chr2L/3L then R (or X)
with bins 0, 1, 2, ... without gaps; arm lengths; with --bed, in_bed and compartment agree with the .bed file.
"""
import argparse, sys
from collections import Counter, defaultdict

p = argparse.ArgumentParser()
p.add_argument("data"); p.add_argument("bins"); p.add_argument("--bed")
a = p.parse_args()

ARM_LENGTHS = {"chr2L": 2352, "chr2R": 2529, "chr3L": 2812, "chr3R": 3208, "chrX": 2355}
RES = 10_000
errors = []

# ---- atoms from the data file: id -> (mol, type)
atoms, sec = {}, None
for line in open(a.data):
    s = line.split("#")[0].strip()
    if not s: continue
    w = s.split()
    if w[0][0].isalpha(): sec = w[0]; continue
    if sec == "Atoms": atoms[int(w[0])] = (int(w[1]), int(w[2]))

# ---- bin table
rows = []
with open(a.bins) as fh:
    header = None
    for line in fh:
        if line.startswith("#") or not line.strip(): continue
        w = line.rstrip("\n").split("\t")
        if header is None: header = w; continue
        rows.append(dict(zip(header, w)))

print(f"{len(atoms)} atoms in {a.data}, {len(rows)} rows in {a.bins}")
if len(rows) != len(atoms): errors.append("different number of atoms and rows")

for r in rows:
    aid, mol, frozen = int(r["atom_id"]), int(r["mol_id"]), int(r["frozen"])
    if aid not in atoms: errors.append(f"atom {aid} not in data file"); continue
    if atoms[aid][0] != mol: errors.append(f"atom {aid}: mol {mol} vs {atoms[aid][0]} in data file")
    if (atoms[aid][1] == 2) != (frozen == 1): errors.append(f"atom {aid}: frozen={frozen} but type {atoms[aid][1]}")
    s, e, b = int(r["start"]), int(r["end"]), int(r["bin"])
    if s // RES != b: errors.append(f"atom {aid}: start {s} not in bin {b}")

# ---- order within each copy: arms in order, bins 0..L-1 without gaps
by_copy = defaultdict(list)
for r in rows: by_copy[r["copy"]].append(r)
for copy, rs in by_copy.items():
    ids = [int(r["atom_id"]) for r in rs]
    if ids != list(range(ids[0], ids[0] + len(ids))): errors.append(f"{copy}: atom ids not contiguous")
    arms = list(dict.fromkeys(r["arm"] for r in rs))
    for arm in arms:
        bins = [int(r["bin"]) for r in rs if r["arm"] == arm]
        if bins != list(range(ARM_LENGTHS[arm])): errors.append(f"{copy} {arm}: bins are not 0..{ARM_LENGTHS[arm]-1}")
    print(f"  {copy}: atoms {ids[0]}-{ids[-1]}, arms {arms}, in_bed {sum(int(r['in_bed']) for r in rs)}/{len(rs)}, "
          f"frozen ids {[int(r['atom_id']) for r in rs if r['frozen'] == '1']}")

# ---- comparison with the .bed file
if a.bed:
    bed = {}
    for line in open(a.bed):
        w = line.split()
        if not w or w[0] == "chrom" or w[0] not in ARM_LENGTHS: continue
        bed[(w[0], int(w[1]) // RES)] = w[3] if len(w) > 3 else "NA"
    for r in rows:
        key = (r["arm"], int(r["bin"]))
        if (key in bed) != (r["in_bed"] == "1"): errors.append(f"atom {r['atom_id']}: in_bed disagrees with the .bed file")
        elif key in bed and bed[key] != r["compartment"]: errors.append(f"atom {r['atom_id']}: compartment differs")

if errors:
    print(f"\nFAILED: {len(errors)} problems, first ones:"); [print("  ", e) for e in errors[:10]]; sys.exit(1)
print("\nAll checks passed.")
