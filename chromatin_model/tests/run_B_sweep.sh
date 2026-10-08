#!/bin/bash

for B in 0.15 0.20 0.25 0.30
do
    echo "======================================"
    echo "Running simulation with B = $B"
    echo "======================================"

    ~/lammps-22Jul2025/build/lmp \
        -var B "$B" \
        -in informed_lammps.in \
        > "lammps_B${B}.out" 2>&1

    echo "Finished B = $B"
    echo
done