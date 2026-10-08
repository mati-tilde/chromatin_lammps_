#!/bin/bash

LAMMPS=~/lammps-22Jul2025/build/lmp
INPUT=informed_lammps.in

for N in 1 2 4 8 16
do
    echo "========================================"
    echo "Running with $N OpenMP threads"
    echo "========================================"

    OMP_NUM_THREADS=$N \
    $LAMMPS -in $INPUT -log log_omp_${N}threads.lammps
done