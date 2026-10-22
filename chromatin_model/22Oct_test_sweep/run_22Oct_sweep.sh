
#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Four-case LAMMPS comparison: 22 October
# ============================================================

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

DATA_FILE="22Oct_chr2_rw_without_H3K27ac.data"
REFERENCE_EXE="$HOME/lammps-22Jul2025/build/lmp_ref"
NEW_EXE="$HOME/lammps-22Jul2025/build_experiment/lmp"
PROGRESS_LOG="22Oct_sweep_progress.log"

# ============================================================
# Preflight checks: do not overwrite previous results
# ============================================================

[[ -f "$DATA_FILE" ]] || {
    echo "ERROR: Missing data file: $DATA_FILE"
    exit 1
}

for exe in "$REFERENCE_EXE" "$NEW_EXE"; do
    [[ -x "$exe" ]] || {
        echo "ERROR: Executable missing: $exe"
        exit 1
    }
done

OUTPUTS=(
    22Oct_test_serial_reference.txt
    22Oct_test_serial_new.txt
    22Oct_test_omp_reference.txt
    22Oct_test_omp_new.txt
    22Oct_test_serial_reference.in
    22Oct_test_serial_new.in
    22Oct_test_omp_reference.in
    22Oct_test_omp_new.in
    22Oct_traj_serial_reference.lammpstrj
    22Oct_traj_serial_new.lammpstrj
    22Oct_traj_omp_reference.lammpstrj
    22Oct_traj_omp_new.lammpstrj
    22Oct_sweep_progress.log
)

for file in "${OUTPUTS[@]}"; do
    if [[ -e "$file" ]]; then
        echo "ERROR: Output already exists: $file"
        echo "Remove or rename old sweep outputs before restarting."
        exit 1
    fi
done

# ============================================================
# Run one test case
# ============================================================

run_case() {
    local name="$1"
    local exe="$2"
    local style="$3"

    local logfile="22Oct_test_${name}.txt"
    local inputfile="22Oct_test_${name}.in"
    local trajectory="22Oct_traj_${name}.lammpstrj"

    echo
    echo "============================================================"
    echo "STARTING: $name"
    echo "Executable : $exe"
    echo "Pair style : $style"
    echo "OMP threads: $OMP_NUM_THREADS"
    echo "Started    : $(date)"
    echo "============================================================"

    # Header: package omp must be configured before read_data.
    cat > "$inputfile" <<EOF
# Sweep case: ${name}

# Initialization
units lj
boundary f f f
atom_style bond
EOF

    if [[ "$style" == */omp ]]; then
        echo "package omp ${OMP_NUM_THREADS}" >> "$inputfile"
    fi

    cat >> "$inputfile" <<EOF
log ${logfile}
variable seed equal 42

# Read topology, coordinates and signals
fix signals all property/atom d2_signals 3 ghost yes
read_data ${DATA_FILE} fix signals NULL Signals

# Neighbor list and bonds
neighbor 0.2 bin
neigh_modify delay 0 every 1 check yes

bond_style harmonic
bond_coeff 1 40.0 1.0

# Pair interactions
pair_style ${style} 2.5

# Groups
group for_freeze type 2
group rest subtract all for_freeze

# Confinement and frozen atoms
region Cyl cylinder z 0 0 28 -1 69 side in
fix nucleus all wall/region Cyl lj126 0.15 1.0 2.5
fix freeze for_freeze setforce 0.0 0.0 0.0

# Ordinary Lennard-Jones coefficients
pair_coeff 1 1 0.415 1.0 2.5
pair_coeff 1 2 0.425 1.0 2.5
pair_coeff 2 2 0.55  1.0 2.5

# Epigenetic B matrix
pair_coeff B 1 1 0.25
pair_coeff B 1 2 0.25
pair_coeff B 1 3 0.25
pair_coeff B 2 2 0.25
pair_coeff B 2 3 0.25
pair_coeff B 3 3 0.25

# Thermodynamic output, saved in this case's log
thermo 100
thermo_style custom step temp press pe etotal evdwl lx ly lz pxx pyy pzz
thermo_modify flush yes

# 1. Initial force and energy evaluation
run 0

# 2. Short minimization
min_style cg
minimize 0 1.0e-6 1000 10000

# 3. 1000-step molecular dynamics run
velocity rest create 1.0 \${seed} rot yes dist gaussian
timestep 0.01

fix integrate rest nve
fix thermostat rest langevin 1.0 1.0 1.0 928374

dump trajectory all custom 1000 ${trajectory} id type x y z
dump_modify trajectory sort id

run 1000

# Cleanup
unfix thermostat
unfix integrate
undump trajectory

EOF

    echo "Generated input: $inputfile"

    if "$exe" -in "$inputfile" 2>&1 | tee -a "$PROGRESS_LOG"; then
        echo "COMPLETED: $name at $(date)" | tee -a "$PROGRESS_LOG"
    else
        echo "FAILED: $name at $(date)" | tee -a "$PROGRESS_LOG"
        echo "Stopping sweep to avoid misleading comparisons." \
            | tee -a "$PROGRESS_LOG"
        return 1
    fi
}

# ============================================================
# Run all four cases
# ============================================================

{
    echo "22 October LAMMPS sweep"
    echo "Started: $(date)"
    echo "Data: $DATA_FILE"
    echo "Reference: $REFERENCE_EXE"
    echo "New: $NEW_EXE"
    echo "OpenMP threads: $OMP_NUM_THREADS"
    echo "Sequence: run 0 -> minimize -> run 1000"
} | tee "$PROGRESS_LOG"

run_case serial_reference "$REFERENCE_EXE" "lj/episig/cut"
run_case serial_new       "$NEW_EXE"       "lj/episig/cut"
run_case omp_reference    "$REFERENCE_EXE" "lj/episig/cut/omp"
run_case omp_new          "$NEW_EXE"       "lj/episig/cut/omp"

echo
echo "============================================================"
echo "ALL FOUR CASES COMPLETED"
echo "Finished: $(date)"
echo "Thermo logs:"
printf '  %s\n' \
    22Oct_test_serial_reference.txt \
    22Oct_test_serial_new.txt \
    22Oct_test_omp_reference.txt \
    22Oct_test_omp_new.txt
echo "============================================================"
