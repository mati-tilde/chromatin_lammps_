
#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Two-case LAMMPS comparison: OLD OMP vs NEW OMP
# ============================================================

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

DATA_FILE="22Oct_chr2_rw_without_H3K27ac.data"
REFERENCE_EXE="$HOME/lammps-22Jul2025/build/lmp_ref"
NEW_EXE="$HOME/lammps-22Jul2025/build_experiment/lmp"
PREFIX="new_omp_ab"

# Fast diagnostic settings
MIN_TOL="1.0e-6"
MIN_ITER=100
MIN_EVAL=1000
MD_STEPS=20

# ============================================================
# Preflight checks
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
    "${PREFIX}_reference.out"
    "${PREFIX}_new.out"
    "${PREFIX}_reference.log"
    "${PREFIX}_new.log"
    "${PREFIX}_reference.in"
    "${PREFIX}_new.in"
    "${PREFIX}_reference.lammpstrj"
    "${PREFIX}_new.lammpstrj"
    "${PREFIX}_progress.log"
)

for file in "${OUTPUTS[@]}"; do
    if [[ -e "$file" ]]; then
        echo "ERROR: Output already exists: $file"
        echo "Rename or remove previous test outputs before restarting."
        exit 1
    fi
done

# ============================================================
# Run one OpenMP test
# ============================================================

run_case() {
    local name="$1"
    local exe="$2"

    local inputfile="${PREFIX}_${name}.in"
    local outputfile="${PREFIX}_${name}.out"
    local logfile="${PREFIX}_${name}.log"
    local trajectory="${PREFIX}_${name}.lammpstrj"

    echo
    echo "============================================================"
    echo "STARTING: $name"
    echo "Executable : $exe"
    echo "Pair style : lj/episig/cut/omp"
    echo "OMP threads: $OMP_NUM_THREADS"
    echo "Started    : $(date)"
    echo "============================================================"

    cat > "$inputfile" <<EOF_INPUT
# OpenMP A/B case: ${name}

units lj
boundary f f f
atom_style bond

# Bash substitutes the thread count here before LAMMPS reads this file.
package omp ${OMP_NUM_THREADS}

log ${logfile}
variable seed equal 42

# Three signals
fix signals all property/atom d2_signals 3 ghost yes
read_data ${DATA_FILE} fix signals NULL Signals

# Neighbor list and bonds
neighbor 0.2 bin
neigh_modify delay 0 every 1 check yes

bond_style harmonic
bond_coeff 1 40.0 1.0

# OpenMP epigenetic pair style
pair_style lj/episig/cut/omp 2.5

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

# Three-signal B matrix
pair_coeff B 1 1 0.25
pair_coeff B 1 2 0.25
pair_coeff B 1 3 0.25
pair_coeff B 2 2 0.25
pair_coeff B 2 3 0.25
pair_coeff B 3 3 0.25

# Thermodynamics
thermo 1
thermo_style custom step temp press pe etotal evdwl
thermo_modify flush yes

# Initial force and energy evaluation
run 0

# Short minimization
min_style cg
minimize ${MIN_TOL} 1.0e-9 ${MIN_ITER} ${MIN_EVAL}

# Short molecular dynamics run
velocity rest create 1.0 \${seed} rot yes dist gaussian
timestep 0.01

fix integrate rest nve
fix thermostat rest langevin 1.0 1.0 1.0 928374

dump trajectory all custom 10 ${trajectory} id type x y z
dump_modify trajectory sort id

run ${MD_STEPS}

unfix thermostat
unfix integrate
undump trajectory
EOF_INPUT

    echo "Generated input: $inputfile"

    if "$exe" -in "$inputfile" > "$outputfile" 2>&1; then
        echo "COMPLETED: $name at $(date)" | tee -a "${PREFIX}_progress.log"
    else
        status=$?
        echo "FAILED: $name (exit code $status) at $(date)" \
            | tee -a "${PREFIX}_progress.log"
        echo "Last output lines:"
        tail -40 "$outputfile"
        return "$status"
    fi
}

# ============================================================
# Exactly two runs
# ============================================================

{
    echo "OpenMP A/B diagnostic"
    echo "Started: $(date)"
    echo "Data: $DATA_FILE"
    echo "Reference executable: $REFERENCE_EXE"
    echo "New executable: $NEW_EXE"
    echo "OMP threads: $OMP_NUM_THREADS"
    echo "Sequence: run 0 -> short minimize -> run ${MD_STEPS}"
} | tee "${PREFIX}_progress.log"

run_case reference "$REFERENCE_EXE"
run_case new "$NEW_EXE"

echo
echo "============================================================"
echo "BOTH OPENMP TESTS COMPLETED"
echo "Finished: $(date)"
echo "Logs: ${PREFIX}_reference.log, ${PREFIX}_new.log"
echo "Console output: ${PREFIX}_reference.out, ${PREFIX}_new.out"
echo "============================================================"
