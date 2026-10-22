
#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Four-signal diagnostic sweep: new serial vs new OpenMP
# Cases:
#   1. serial_new: lj/episig/cut
#   2. omp_new:    lj/episig/cut/omp
#
# Each case performs:
#   run 0 -> short minimization -> run 1000
# ============================================================

DATA_FILE="22Oct_chr2_rw.data"
NEW_EXE="$HOME/lammps-22Jul2025/build_experiment/lmp"
OMP_THREADS="${OMP_NUM_THREADS:-2}"
PROGRESS_LOG="22Oct_4sig_new_sweep_progress.log"

# ------------------------------------------------------------
# Preflight checks
# ------------------------------------------------------------

[[ -f "$DATA_FILE" ]] || {
    echo "ERROR: Missing data file: $DATA_FILE"
    exit 1
}

[[ -x "$NEW_EXE" ]] || {
    echo "ERROR: Missing executable: $NEW_EXE"
    exit 1
}

OUTPUTS=(
    22Oct_4sig_new_serial.txt
    22Oct_4sig_new_omp.txt
    22Oct_4sig_new_serial.in
    22Oct_4sig_new_omp.in
    22Oct_4sig_new_serial.lammpstrj
    22Oct_4sig_new_omp.lammpstrj
    22Oct_4sig_new_sweep_progress.log
)

for file in "${OUTPUTS[@]}"; do
    if [[ -e "$file" ]]; then
        echo "ERROR: Output already exists: $file"
        echo "Rename or remove old outputs before restarting."
        exit 1
    fi
done

# ------------------------------------------------------------
# Run one case
# ------------------------------------------------------------

run_case() {
    local mode="$1"
    local style="$2"
    local input="22Oct_4sig_new_${mode}.in"
    local logfile="22Oct_4sig_new_${mode}.txt"
    local trajectory="22Oct_4sig_new_${mode}.lammpstrj"

    echo "============================================================"
    echo "STARTING: ${mode}"
    echo "Executable : ${NEW_EXE}"
    echo "Pair style : ${style}"
    echo "OMP threads: ${OMP_THREADS}"
    echo "Started    : $(date)"
    echo "============================================================"

    local omp_package=""
    local launch_omp_threads=1

    if [[ "$mode" == "omp" ]]; then
        omp_package="package omp ${OMP_THREADS}"
        launch_omp_threads="$OMP_THREADS"
    fi

    cat > "$input" <<LMP
# Four-signal diagnostic sweep: ${mode}

units lj
boundary f f f
atom_style bond
log ${logfile}

variable seed equal 42

# Configure OpenMP before read_data.
${omp_package}

# Four per-atom signals.
fix signals all property/atom d2_signals 4 ghost yes
read_data ${DATA_FILE} fix signals NULL Signals

# Neighbor settings.
neighbor 0.2 bin
neigh_modify delay 0 every 1 check yes

# Bonds.
bond_style harmonic
bond_coeff 1 40.0 1.0

# Pair style.
pair_style ${style} 2.5

# Groups.
group for_freeze type 2
group rest subtract all for_freeze

# Confinement.
region Cyl cylinder z 0 0 28 -1 69 side in
fix nucleus all wall/region Cyl lj126 0.15 1.0 2.5
fix freeze for_freeze setforce 0.0 0.0 0.0

# Lennard-Jones coefficients.
pair_coeff 1 1 0.415 1.0 2.5
pair_coeff 1 2 0.425 1.0 2.5
pair_coeff 2 2 0.55  1.0 2.5

# Four-signal symmetric B matrix: 10 independent elements.
pair_coeff B 1 1 0.25
pair_coeff B 1 2 0.25
pair_coeff B 1 3 0.25
pair_coeff B 1 4 0.25
pair_coeff B 2 2 0.25
pair_coeff B 2 3 0.25
pair_coeff B 2 4 0.25
pair_coeff B 3 3 0.25
pair_coeff B 3 4 0.25
pair_coeff B 4 4 0.25

# Thermodynamic output.
thermo 100
thermo_style custom step temp press pe etotal evdwl lx ly lz pxx pyy pzz
thermo_modify flush yes

# ------------------------------------------------------------
# Stage 1: initial configuration, no integration.
# ------------------------------------------------------------
run 0

# ------------------------------------------------------------
# Stage 2: short minimization.
# ------------------------------------------------------------
min_style cg
minimize 0 1.0e-9 5000 50000

# ------------------------------------------------------------
# Stage 3: short molecular dynamics run.
# ------------------------------------------------------------
velocity rest create 1.0 \${seed} rot yes dist gaussian
timestep 0.01

fix 1 rest nve
fix 2 rest langevin 1.0 1.0 1.0 928374

dump sim all custom 100 ${trajectory} id mol type x y z d2_signals[1] d2_signals[2] d2_signals[3] d2_signals[4]

run 1000
LMP

    echo "[$(date)] START ${mode}" >> "$PROGRESS_LOG"

    if [[ "$mode" == "omp" ]]; then
        OMP_NUM_THREADS="$OMP_THREADS" "$NEW_EXE" -sf omp -in "$input" \
            >> "$PROGRESS_LOG" 2>&1
    else
        OMP_NUM_THREADS=1 "$NEW_EXE" -in "$input" \
            >> "$PROGRESS_LOG" 2>&1
    fi

    echo "[$(date)] DONE ${mode}" | tee -a "$PROGRESS_LOG"
}

# ------------------------------------------------------------
# Execute both cases, using only the new implementation.
# ------------------------------------------------------------

run_case serial "lj/episig/cut"
run_case omp "lj/episig/cut/omp"

echo "============================================================"
echo "Both four-signal diagnostic cases completed."
echo "Finished: $(date)"
echo "============================================================"
