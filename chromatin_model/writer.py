from genome_parser import read_bed
from builder import split_by_arms, build_chroms, build_genome
from writer_funcs import write_data

from genome_constants import NUCLEAR_HEIGHT, NUCLEAR_RADIUS

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

#------------------------------------------------------
# Building genome.
#------------------------------------------------------

# random generatore for random walk:
rng = np.random.default_rng(SEED)

beads = read_bed(BED_FILE)

arms = split_by_arms(beads)

# generating genome accordingly to method:
if GEOMETRY == "random_walk":

    chroms = build_chroms(arms, method = GEOMETRY, step_size = STEP_SIZE, rng = rng, boundary = "cylinder", cyl_radius = CYLINDER_RADIUS, cyl_height = CYLINDER_HEIGHT)

    genome = build_genome(chroms, arms, method = GEOMETRY, step_size = STEP_SIZE, rng = rng, boundary = "cylinder", cyl_radius = CYLINDER_RADIUS, cyl_height = CYLINDER_HEIGHT, nuclear_radius = NUCLEAR_RADIUS, nuclear_height = NUCLEAR_HEIGHT)


elif GEOMETRY == "helix":

    chroms = build_chroms(arms, method = GEOMETRY, radius = HELIX_RADIUS, height = HELIX_HEIGHT, turns = HELIX_TURNS)

    genome = build_genome(chroms, arms, method = GEOMETRY, radius = HELIX_RADIUS, height = HELIX_HEIGHT, turns = HELIX_TURNS, nuclear_radius = NUCLEAR_RADIUS, nuclear_height = NUCLEAR_HEIGHT)

#------------------------------------------------------
# Writing .data file.
#------------------------------------------------------
if GEOMETRY == "random_walk":
    write_data(filename = "rw_genome.data", genome = genome, beads = beads, n_atom_types = N_ATOM_TYPES, n_bond_types = N_BOND_TYPES)
elif GEOMETRY == "helix":
    write_data(filename = "helix_genome.data", genome = genome, beads = beads, n_atom_types = N_ATOM_TYPES, n_bond_types = N_BOND_TYPES)

