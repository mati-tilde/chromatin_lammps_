import numpy as np

from geometry import generate_coords
from genome_constants import ARM_LENGTHS, CHROMOSOME_ARMS, INITIAL_LAYOUT

#---------------------------------------------------------------------------
# BUILDER: the one that will (hopefully) keep everything together.

# From the parser information, contained in the Bead objects, the builder
# splites chromosomes and assigns coordinates coherently.
#---------------------------------------------------------------------------


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
# Building coordinates for each arm, and binding chromosome together.
def build_chroms(arms, method, **kwargs):

    chrom_coords = {}

    for chromosome, chromosome_arms in CHROMOSOME_ARMS.items():

        # first arm:
        n_beads = len(arms[chromosome_arms[0]])

        coords = generate_coords(method = method, n_beads = n_beads, **kwargs)


        for arm in chromosome_arms[1:]:

            # Geometry-dependent definition of "start"
            if method == "helix":
                start = coords[-1, :2]
            else:
                start = coords[-1]

            new_coords = generate_coords(
                method=method,
                n_beads=len(arms[arm]),
                start=start,
                reverse=True,
                **kwargs
            )

            coords = np.concatenate((coords, new_coords))

        # concatenated chromosome coordinates:
        chrom_coords[chromosome] = coords

    return chrom_coords
#---------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------
# Copying the complete genome to work with tw ocopie of each chromosome.
def build_genome(chrom_coords):

    genome = {}

    for chromosome, coords in chrom_coords.items():
        chromA = f"{chromosome}A"
        chromB = f"{chromosome}B"

        # first homolog:
        genome[chromA] = coords.copy() + INITIAL_LAYOUT[chromA]
        genome[chromB] = coords.copy() + INITIAL_LAYOUT[chromB]


    return genome

#---------------------------------------------------------------------------------------------
