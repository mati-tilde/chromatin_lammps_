from genome_parser import read_bed
from constrained_builder import split_by_arms, build_chroms, build_genome
from writer_funcs import write_data, write_signal
from signal_annotator import annotate_genome
from epigenetic_track import EpigeneticTrack

from genome_constants import NUCLEAR_HEIGHT, NUCLEAR_RADIUS, SIGNAL_NAMES

import numpy as np

#------------------------------------------------------
# Parameters.
#------------------------------------------------------

# import choices:
BED_FILE = "comp_HiTC_10000_WT_ST10_refined_with_chromwide.bed"

GEOMETRY = "random_walk"

# for helix:
HELIX_RADIUS = 5.0
HELIX_HEIGHT = 68
HELIX_TURNS = 50

# for random walk:
STEP_SIZE = 1.0
SEED = 42

# cylinder boundary (for random walk only):
CYLINDER_RADIUS = 8.0
CYLINDER_HEIGHT = 68.0


# types (for lammps):
N_ATOM_TYPES = 2
N_BOND_TYPES = 1

# epigenetic tracks:
H3K27AC = EpigeneticTrack(
    "H3K27ac",
    "H3K27ac_WT_st10.pooled.bw"
)

H3K9AC = EpigeneticTrack(
    "H3K9ac",
    "H3K9ac_WT_st10.pooled.bw"
)

H3K27ME3 = EpigeneticTrack(
    "H3K27me3",
    "H3K27me3_WT_st10.pooled.bw"
)

H3K9ME3 = EpigeneticTrack(
    "H3K9me3",
    "H3K9me3_WT_st10.pooled.bw"
)


#------------------------------------------------------
# Building genome.
#------------------------------------------------------

# random generatore for random walk:
# ------------------------------------------------------
# Independent random-number generators for each
# chromosome copy.
# ------------------------------------------------------

MASTER_SEED = np.random.SeedSequence(SEED)

COPY_NAMES = [
    "chr2A",
    "chr2B",
    "chr3A",
    "chr3B",
    "chrXA",
    "chrXB",
]

child_sequences = MASTER_SEED.spawn(len(COPY_NAMES))


rngs = {
    name: np.random.default_rng(seed_sequence)
    for name, seed_sequence in zip(COPY_NAMES, child_sequences)
}


print("Available RNGs:")
for name in rngs:
    print(f"  {name}")

beads = read_bed(BED_FILE)

annotate_genome(beads, H3K27AC)
annotate_genome(beads, H3K9AC)
annotate_genome(beads, H3K27ME3)
annotate_genome(beads, H3K9ME3)

#------------------------------------------------------
# sanity check for annotation:
print("\nExample bead signals:")
print(beads[0].signals)
#------------------------------------------------------

arms = split_by_arms(beads)

# generating genome accordingly to method:
if GEOMETRY == "random_walk":

    chroms = build_chroms(arms, method = GEOMETRY, step_size = STEP_SIZE, rngs = rngs, boundary = "cylinder", cyl_radius = CYLINDER_RADIUS, cyl_height = CYLINDER_HEIGHT)

    genome = build_genome(chroms, arms, method = GEOMETRY, step_size = STEP_SIZE, rngs = rngs, boundary = "cylinder", cyl_radius = CYLINDER_RADIUS, cyl_height = CYLINDER_HEIGHT, nuclear_radius = NUCLEAR_RADIUS, nuclear_height = NUCLEAR_HEIGHT)


elif GEOMETRY == "helix":

    chroms = build_chroms(arms, method = GEOMETRY, radius = HELIX_RADIUS, height = HELIX_HEIGHT, turns = HELIX_TURNS)

    genome = build_genome(chroms, arms, method = GEOMETRY, radius = HELIX_RADIUS, height = HELIX_HEIGHT, turns = HELIX_TURNS, nuclear_radius = NUCLEAR_RADIUS, nuclear_height = NUCLEAR_HEIGHT)

#------------------------------------------------------
# Writing .data file.
#------------------------------------------------------
if GEOMETRY == "random_walk":
    write_data(filename = "5Oct_layout_test_rw.data", genome = genome, beads = beads, n_atom_types = N_ATOM_TYPES, n_bond_types = N_BOND_TYPES, signal_names = SIGNAL_NAMES)
elif GEOMETRY == "helix":
    write_data(filename = "5Oct_topology_test_helix.data", genome = genome, beads = beads, n_atom_types = N_ATOM_TYPES, n_bond_types = N_BOND_TYPES, signal_names = SIGNAL_NAMES)

