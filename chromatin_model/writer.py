import numpy as np
import argparse
import signal
import sys
import time
from datetime import date
from pathlib import Path

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

parser.add_argument("--seed", default = "42", help = "Master seed for the random-walk geometry (default: 42). If it gets stuck, seed+1, seed+2, ... are tried.")

parser.add_argument("--bed", default = "comp_HiTC_10000_WT_ST10_refined_with_chromwide.bed", help = "BED file containing the chromosome segmentation.")

parser.add_argument("--remove-signals", nargs = "+", choices = AVAILABLE_SIGNALS, default = [], help = "Epigenetic marks to exclude from the output.")

parser.add_argument("--output", default = None, help = "Output .data file with signals. Default: <date>_<chromosomes>_<geometry>_seed<seed>[_without_...].data")

parser.add_argument("--topology-output", default = None, help = "Output .data file without signals. Default: the --output name with '_topology' added before '.data'.")

parser.add_argument("--attempt-timeout", type = float, default = 60.0, help = "Maximum time in seconds for one geometry attempt (default: 60).")

parser.add_argument("--total-timeout", type = float, default = 300.0, help = "Maximum total time in seconds over all attempts; after that, abort without writing files (default: 300).")

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

# Timeout for the generation of the geometry (in seconds):
GENERATION_TIMEOUT = 180 

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

if CHROMOSOMES == "all":
    COPY_NAMES = ["chr2A", "chr2B", "chr3A", "chr3B", "chrXA", "chrXB"]
else:
    COPY_NAMES = [CHROMOSOMES]

# One independent generator per chromosome copy, all derived from a single master seed.
# Subsequent seeds give completely unrelated random streams, so if one seed gets stuck, we can try the next one.
def make_rngs(seed, copy_names):
    master = np.random.SeedSequence(seed)
    children = master.spawn(len(copy_names))
    return {name: np.random.default_rng(s) for name, s in zip(copy_names, children)}

#------------------------------------------------------
# Building annotated genome.
#------------------------------------------------------

beads = read_bed(BED_FILE)

for signal_name in SIGNAL_NAMES:
    annotate_genome(beads, SIGNAL_TRACKS[signal_name])

arms = split_by_arms(beads)

#-------------------------------------------------------------------------
# Building genome accordingly to chosen geometry and chromosome selection.
#-------------------------------------------------------------------------

# Build the coordinates for the selected chromosomes; returns the genome dictionary.
def build_geometry(rngs):

    if CHROMOSOMES == "all":

        if GEOMETRY == "random_walk":

            chroms = build_chroms(arms, method = GEOMETRY, step_size = STEP_SIZE, rngs = rngs, boundary = "cylinder", cyl_radius = CYLINDER_RADIUS, cyl_height = CYLINDER_HEIGHT)

            return build_genome(chroms, arms, method = GEOMETRY, step_size = STEP_SIZE, rngs = rngs, boundary = "cylinder", cyl_radius = CYLINDER_RADIUS, cyl_height = CYLINDER_HEIGHT, nuclear_radius = NUCLEAR_RADIUS, nuclear_height = NUCLEAR_HEIGHT)

        elif GEOMETRY == "helix":

            chroms = build_chroms(arms, method = GEOMETRY, radius = HELIX_RADIUS, height = HELIX_HEIGHT, turns = HELIX_TURNS)

            return build_genome(chroms, arms, method = GEOMETRY, radius = HELIX_RADIUS, height = HELIX_HEIGHT, turns = HELIX_TURNS, nuclear_radius = NUCLEAR_RADIUS, nuclear_height = NUCLEAR_HEIGHT)

    elif CHROMOSOMES in VALID_CHROMOSOMES:

        chromosome_arms = CHROMOSOME_ARMS[CHROMOSOMES]

        if GEOMETRY == "random_walk":

            coords = build_single_chrom(arms, chromosome_arms, method = "random_walk", rng = rngs[CHROMOSOMES], step_size = STEP_SIZE, boundary = "cylinder", cyl_radius = CYLINDER_RADIUS, cyl_height = CYLINDER_HEIGHT)

        elif GEOMETRY == "helix":

            coords = build_single_chrom(arms, chromosome_arms, method = "helix", radius = HELIX_RADIUS, height = HELIX_HEIGHT, turns = HELIX_TURNS)

        return {CHROMOSOMES: coords}

    raise ValueError(f"Unknown geometry/chromosome selection: {GEOMETRY}, {CHROMOSOMES}")


#--------------------------------------------------------------------------------------
# Retry loop: a stuck or failed seed is replaced by seed+1, within a total time budget.
#--------------------------------------------------------------------------------------

class AttemptTimeout(Exception):
    """Raised by the alarm handler when one geometry attempt exceeds its time limit."""

def _on_alarm(signum, frame):
    raise AttemptTimeout()

signal.signal(signal.SIGALRM, _on_alarm)

SEED = int(args.seed)
genome = None
attempt = 0
t_start = time.monotonic()

while genome is None:

    remaining = args.total_timeout - (time.monotonic() - t_start)
    if remaining <= 0:
        break

    attempt += 1
    budget = min(args.attempt_timeout, remaining)
    print(f"\n[writer] attempt {attempt}: seed = {SEED}, time limit = {budget:.0f} s", flush = True)

    rngs = make_rngs(SEED, COPY_NAMES)

    # start the alarm
    signal.setitimer(signal.ITIMER_REAL, budget)      
    try:
        genome = build_geometry(rngs)
    except AttemptTimeout:
        print(f"[writer] seed {SEED} got stuck (no result after {budget:.0f} s): changing seed.", flush = True)
    except RuntimeError as err:
        print(f"[writer] seed {SEED} failed ({err}): changing seed.", flush = True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)       # always cancel the alarm

    # helix is deterministic: another seed would not change anything.
    if genome is None and GEOMETRY == "helix":
        break                                          

    if genome is None:
        SEED += 1

elapsed = time.monotonic() - t_start

if genome is None:
    print(f"\n[writer] ABORTED: no valid geometry after {attempt} attempt(s) in {elapsed:.0f} s. No files were written.", file = sys.stderr, flush = True)
    sys.exit(1)

print(f"\n[writer] geometry built with seed {SEED} (attempt {attempt}, {elapsed:.1f} s).", flush = True)


#------------------------------------------------------
# Writing .data file.
#------------------------------------------------------

# Filename encoding the geometry and chromosome selection, as well as any removed signals.

removed_suffix = "_without_" + "_".join(args.remove_signals) if args.remove_signals else ""
geometry_tag = {"random_walk": "rw", "helix": "helix"}[GEOMETRY]
date_tag = date.today().strftime("%d%b")                    
seed_tag = f"_seed{SEED}" if GEOMETRY == "random_walk" else ""

if args.output is not None:
    filename = args.output
else:
    filename = f"{date_tag}_{CHROMOSOMES}_{geometry_tag}{seed_tag}{removed_suffix}.data"

if args.topology_output is not None:
    topology_filename = args.topology_output
else:
    path = Path(filename)
    topology_filename = str(path.with_name(path.stem + "_topology" + path.suffix))

if args.output is not None and SEED != args.seed:
    print(f"[writer] NOTE: requested seed {args.seed} was replaced by seed {SEED}; "
          f"the file name '{filename}' does not show this.", flush = True)

print(f"[writer] writing {filename} and {topology_filename}", flush = True)

# Writing .data file.
write_data(filename = filename, genome = genome, beads = beads, n_atom_types = N_ATOM_TYPES, n_bond_types = N_BOND_TYPES, signal_names = SIGNAL_NAMES, include_signals = True)

write_data(filename = topology_filename, genome = genome, beads = beads, n_atom_types = N_ATOM_TYPES, n_bond_types = N_BOND_TYPES, signal_names = SIGNAL_NAMES, include_signals = False)
