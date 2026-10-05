import numpy as np
from scipy.spatial import cKDTree

from geometry import generate_coords
from genome_constants import ARM_LENGTHS, CHROMOSOME_ARMS, INITIAL_LAYOUT,INITIAL_LAYOUT_RW, NUCLEAR_HEIGHT, NUCLEAR_RADIUS

#---------------------------------------------------------------------------
# BUILDER: the one that will (hopefully) keep everything together.

# From the parser information, contained in the Bead objects, the builder
# splites chromosomes and assigns coordinates coherently.
#---------------------------------------------------------------------------




#===========================================================================================
# HELPERS -- that helped only very slightly.

#---------------------------------------------------------------------------------------------
# Check whether an entire chromosome fits inside the nuclear cylinder.
def _inside_nucleus(coords, radius, height):

    radial_squared = coords[:, 0]**2 + coords[:, 1]**2

    inside_radial = np.all(radial_squared <= radius**2)
    inside_height = np.all(
        (coords[:, 2] >= 0.0) &
        (coords[:, 2] <= height)
    )

    return inside_radial and inside_height
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# Minimum distance between two chromosomes.
def _min_interchrom_distance(coords_a, coords_b):

    tree = cKDTree(coords_a)

    distances, _ = tree.query(
        coords_b,
        k=1
    )

    return np.min(distances)
#---------------------------------------------------------------------------------------------

#===========================================================================================




#--------------------------------------------------------------------------------------------
# Splitting beads according to arms.
# output: returns a dictionary whose keys are arms' names (chr2L, chr2R, chr3L, chr3R, chrX),
#         and whose values are the corresponding Bead objects.

def split_by_arms(beads):

    # create the dictionary for arms:
    arms = {
        arm: []
        for arm in ARM_LENGTHS
    }

    # filling the dictionary:
    for bead in beads:
        arms[bead.arm].append(bead)

    #------------------------------------------------
    # sanity check
    for arm, arm_beads in arms.items():

        expected = ARM_LENGTHS[arm]

        if len(arm_beads) != expected:
            raise ValueError(
                f"{arm}: expected {expected} beads, "
                f"found {len(arm_beads)}."
            )
    #------------------------------------------------

    return arms
#---------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------
# Helper to build a single chromosome.
def build_single_chrom(arms, chromosome_arms, method, **kwargs):

    #------------------------------------------------------------------------
    # if method is random walk -- building arm by arm, starting from the junction point:

    if method == "random_walk":

        # first arm: 
        n_beads = len(arms[chromosome_arms[0]])

        print(
            f"  Generating {chromosome_arms[0]} "
            f"({len(arms[chromosome_arms[0]])} beads)",
            flush=True
        )

        coords = generate_coords(
            method=method,
            n_beads=n_beads,
            target_z=NUCLEAR_HEIGHT,
            lambda_drift=0.0,
            drift_angle=0.0,
            reverse=False,
            **kwargs
        )

        # second arm
        for arm in chromosome_arms[1:]:

            # start from the junction
            start = coords[-1]

            print(
                f"  Generating {arm} "
                f"({len(arms[arm])} beads)",
                flush=True
            )

            new_coords = generate_coords(
                method=method,
                n_beads=len(arms[arm]),
                start=start,
                avoid_coords=coords,
                target_z=0.0,
                lambda_drift=0.0,
                drift_angle=np.pi,
                reverse=True,
                **kwargs
            )

            coords = np.concatenate((coords, new_coords))

        return coords
    #------------------------------------------------------------------------


    #------------------------------------------------------------------------
    # if method is helix:

    if method == "helix":

        # first arm:
        n_beads = len(arms[chromosome_arms[0]])

        coords = generate_coords(method = method, n_beads = n_beads, **kwargs)

        # second arm:
        for arm in chromosome_arms[1:]:

            # end of previous arm for continuity:
            start = coords[-1, :2]

            new_coords = generate_coords(method = method, n_beads = len(arms[arm]) + 1, start = start, reverse = True, **kwargs)

            # discard duplicated bead:
            new_coords = new_coords[1:]

            coords = np.concatenate((coords, new_coords))

        return coords
    #------------------------------------------------------------------------

    # for unknown geometry (have not implemented persistent random walk thoroughlly for now):
    raise ValueError(f"Unknown geometry method: {method}")
#---------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------
# Building coordinates for each arm, and binding chromosome together.
def build_chroms(arms, method, rngs=None, **kwargs):

    chrom_coords = {}

    for chromosome, chromosome_arms in CHROMOSOME_ARMS.items():

        if method == "helix":

            if chromosome in ("chr2", "chr3"):
                drift = 5.0

            elif chromosome == "chrX":
                drift = 0.0

            else:
                raise ValueError(f"Unknown chromosome: {chromosome}")

            chrom_coords[chromosome] = build_single_chrom(arms, chromosome_arms, method, drift = drift, **kwargs)

        else:

            chrom_coords[chromosome] = build_single_chrom(arms, chromosome_arms, method, rng=rngs[f"{chromosome}A"], **kwargs)

    return chrom_coords
#---------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------
# Build the diploid genome and apply the initial layout.
def build_genome(chrom_coords, arms, method, nuclear_radius = NUCLEAR_RADIUS, nuclear_height = NUCLEAR_HEIGHT, min_interchrom_dist = 0.5, max_layout_attempts = 1000, rngs=None, **kwargs):

    genome = {}

    for chromosome, coords in chrom_coords.items():

        # --------------------------------------------------------------
        # Select the appropriate layout
        # --------------------------------------------------------------

        if method == "random_walk":
            layout = INITIAL_LAYOUT_RW

        elif method == "helix":
            layout = INITIAL_LAYOUT

        else:
            raise ValueError(f"Unknown geometry: {method}")


        # --------------------------------------------------------------
        # homolog A
        # --------------------------------------------------------------

        chromA = f"{chromosome}A"

        genome[chromA] = coords.copy() + layout[chromA]


        # --------------------------------------------------------------
        # homolog B
        # --------------------------------------------------------------

        chromB = f"{chromosome}B"

        if method == "helix":

            # Same deterministic chromosome geometry
            coordsB = coords.copy()

        elif method == "random_walk":

            print(
                f"\nGenerating homolog B: {chromosome}B",
                flush=True
            )

            coordsB = build_single_chrom(
                arms,
                CHROMOSOME_ARMS[chromosome],
                method,
                rng=rngs[chromB],
                **kwargs
            )

        else:
            raise ValueError(f"Unknown geometry: {method}")


        genome[chromB] = coordsB.copy() + layout[chromB]

    return genome
#---------------------------------------------------------------------------------------------

