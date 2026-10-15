import numpy as np
import argparse

from genome_parser import read_bed
from constrained_builder import split_by_arms, build_chroms, build_genome, build_single_chrom
from writer_funcs import write_data
from signal_annotator import annotate_genome
from epigenetic_track import EpigeneticTrack

from genome_constants import (CHROMOSOME_ARMS, INITIAL_LAYOUT, INITIAL_LAYOUT_RW, NUCLEAR_HEIGHT, NUCLEAR_RADIUS, SIGNAL_NAMES)



#-------------------------------------------------------------
# Command-line arguments.
#-------------------------------------------------------------

AVAILABLE_SIGNALS = [
    "H3K27ac",
    "H3K9ac",
    "H3K27me3",
    "H3K9me3",
]

parser = argparse.ArgumentParser(description = "Build a coarse-grained Drosophila chromosome/genome geometry.")

parser.add_argument("--geometry", required = True, choices = ["random_walk", "helix"], default = "random_walk", help = "Geometry generation method.")

parser.add_argument("--chromosomes", required = True, choices = ["all", "chr2", "chr3", "chrX"], default = "all", help = "Chromosome to generate, or 'all' for the full genome.")

parser.add_argument("--bed", default = "comp_HiTC_10000_WT_ST10_refined_with_chromwide.bed", help = "BED file containing the chromosome segmentation.")

parser.add_argument("--remove-signals", nargs = "+", choices = AVAILABLE_SIGNALS, default = [], help = "Epigenetic marks to exclude from the output.")

args = parser.parse_args()

BED_FILE = args.bed
GEOMETRY = args.geometry
CHROMOSOMES = args.chromosomes
SIGNAL_NAMES = [
    signal
    for signal in AVAILABLE_SIGNALS
    if signal not in args.remove_signals
]

VALID_CHROMOSOMES = {"chr2", "chr3", "chrX"}

#-------------------------------------------------------------
# Parameters for the chosen geometry.
#-------------------------------------------------------------

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

SIGNAL_TRACKS = {
    "H3K27ac": H3K27AC,
    "H3K9ac": H3K9AC,
    "H3K27me3": H3K27ME3,
    "H3K9me3": H3K9ME3,
}

#---------------------------------------------------------------
# Rng setup for random walk:

# Independent random-number generators for each chromosome copy.
#---------------------------------------------------------------

MASTER_SEED = np.random.SeedSequence(SEED)

if CHROMOSOMES == "all":
    COPY_NAMES = ["chr2A", "chr2B", "chr3A", "chr3B", "chrXA", "chrXB"]
else:
    COPY_NAMES = [CHROMOSOMES]

child_sequences = MASTER_SEED.spawn(len(COPY_NAMES))

rngs = {
    name: np.random.default_rng(seed_sequence)
    for name, seed_sequence in zip(COPY_NAMES, child_sequences)
}


print("Available RNGs:")
for name in rngs:
    print(f"  {name}")

#------------------------------------------------------
# Building annotated genome.
#------------------------------------------------------

beads = read_bed(BED_FILE)

for signal_name in SIGNAL_NAMES:
    annotate_genome(beads, SIGNAL_TRACKS[signal_name])

#------------------------------------------------------
# sanity check for annotation:
# print("\nExample bead signals:")
# print(beads[0].signals)
#------------------------------------------------------

arms = split_by_arms(beads)

#-------------------------------------------------------------------------
# Building genome accordingly to chosen geometry and chromosome selection.
#-------------------------------------------------------------------------

# ****************************************************************************************
if CHROMOSOMES == "all":

    if GEOMETRY == "random_walk":

        chroms = build_chroms(arms, method = GEOMETRY, step_size = STEP_SIZE, rngs = rngs, boundary = "cylinder", cyl_radius = CYLINDER_RADIUS, cyl_height = CYLINDER_HEIGHT)

        genome = build_genome(chroms, arms, method = GEOMETRY, step_size = STEP_SIZE, rngs = rngs, boundary = "cylinder", cyl_radius = CYLINDER_RADIUS, cyl_height = CYLINDER_HEIGHT, nuclear_radius = NUCLEAR_RADIUS, nuclear_height = NUCLEAR_HEIGHT)

    elif GEOMETRY == "helix":

        chroms = build_chroms(arms, method = GEOMETRY, radius = HELIX_RADIUS, height = HELIX_HEIGHT, turns = HELIX_TURNS)

        genome = build_genome(chroms, arms, method = GEOMETRY, radius = HELIX_RADIUS, height = HELIX_HEIGHT, turns = HELIX_TURNS, nuclear_radius = NUCLEAR_RADIUS, nuclear_height = NUCLEAR_HEIGHT)

# ************************************************************************

elif CHROMOSOMES in VALID_CHROMOSOMES:

    chromosome = CHROMOSOMES
    chromosome_arms = CHROMOSOME_ARMS[chromosome]

    if GEOMETRY == "random_walk":

        coords = build_single_chrom(arms, chromosome_arms, method="random_walk", rng=rngs[chromosome], step_size=STEP_SIZE, boundary="cylinder", cyl_radius=CYLINDER_RADIUS, cyl_height=CYLINDER_HEIGHT)

        genome = {chromosome: coords}

    elif GEOMETRY == "helix":

        coords = build_single_chrom(arms, chromosome_arms, method="helix", radius=HELIX_RADIUS, height=HELIX_HEIGHT, turns=HELIX_TURNS)

        genome = {chromosome: coords}

# ************************************************************************

    else:
        raise ValueError(f"Unknown geometry method: {GEOMETRY}")

# ************************************************************************


#------------------------------------------------------
# Writing .data file.
#------------------------------------------------------

# Filename encoding the geometry and chromosome selection, as well as any removed signals.
if args.remove_signals:
    removed_suffix = "_without_" + "_".join(args.remove_signals)
else:
    removed_suffix = ""

if GEOMETRY == "random_walk":
    filename = f"15Oct_{CHROMOSOMES}_rw{removed_suffix}.data"
    topology_filename = f"15Oct_{CHROMOSOMES}_rw_topology{removed_suffix}.data"

elif GEOMETRY == "helix":
    filename = f"15Oct_{CHROMOSOMES}_helix{removed_suffix}.data"
    topology_filename = f"15Oct_{CHROMOSOMES}_helix_topology{removed_suffix}.data"

else:
    raise ValueError(f"Unknown geometry method: {GEOMETRY}")

# Writing .data file.
write_data(filename = filename, genome = genome, beads = beads, n_atom_types = N_ATOM_TYPES, n_bond_types = N_BOND_TYPES, signal_names = SIGNAL_NAMES, include_signals = True)

write_data(filename = topology_filename, genome = genome, beads = beads, n_atom_types = N_ATOM_TYPES, n_bond_types = N_BOND_TYPES, signal_names = SIGNAL_NAMES, include_signals = False)
