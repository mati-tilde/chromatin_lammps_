import numpy as np
import time

from constrained_builder import build_single_chrom
from genome_constants import (
    ARM_LENGTHS,
    NUCLEAR_HEIGHT,
    NUCLEAR_RADIUS,
)   


# ------------------------------------------------------------
# Small artificial two-arm chromosome
# ------------------------------------------------------------

arms = {
    "chrL": [f"L{i}" for i in range(ARM_LENGTHS["chr2L"])],
    "chrR": [f"R{i}" for i in range(ARM_LENGTHS["chr2R"])],
}

chromosome_arms = ("chrL", "chrR")


# ------------------------------------------------------------
# Generate coordinates
# ------------------------------------------------------------
start_time = time.time()

coords = build_single_chrom(
    arms,
    chromosome_arms,
    method="random_walk",
    step_size=1.0,
    rng=np.random.default_rng(42),
    boundary="cylinder",
    boundary_kwargs={
        "cyl_radius": 8.0,
        "cyl_height": 68.0,
    },
)

elapsed = time.time() - start_time
print(f"Generation time: {elapsed:.2f} s")

r = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)

assert np.all(r <= 8.0)
assert np.all(coords[:, 2] >= 0.0)
assert np.all(coords[:, 2] <= 68.0)

print("All beads inside the cylinder.")

# ------------------------------------------------------------
# Basic information
# ------------------------------------------------------------

print("Number of beads:", len(coords))
print("First bead:", coords[0])
print("Junction:", coords[len(arms["chrL"]) - 1])
print("Last bead:", coords[-1])


# ------------------------------------------------------------
# z positions
# ------------------------------------------------------------

junction_index =  ARM_LENGTHS["chr2L"] - 1

print("\nZ positions:")
print("first bead:   ", coords[0, 2])
print("junction:     ", coords[junction_index, 2])
print("last bead:    ", coords[-1, 2])


# ------------------------------------------------------------
# Bond lengths
# ------------------------------------------------------------

bond_lengths = np.linalg.norm(
    coords[1:] - coords[:-1],
    axis=1
)

print("\nBond lengths:")
print("min:", bond_lengths.min())
print("max:", bond_lengths.max())


# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

assert len(coords) == (
    ARM_LENGTHS["chr2L"] + ARM_LENGTHS["chr2R"]
)

assert np.isclose(
    coords[junction_index, 2],
    NUCLEAR_HEIGHT,
)

assert np.isclose(
    coords[-1, 2],
    0.0,
)


assert np.allclose(
    bond_lengths,
    1.0
)

print("\nAll checks passed.")