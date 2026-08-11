from beads_class import Bead
from genome_constants import (RESOLUTION, ARM_LENGTHS, ARM_TO_CHROMOSOME, CHROMOSOME_TO_MOLECULE, EXCLUDED_CHROMOSOMES)


##########################################################################################
# GENOME PARSER: includes functions needed to read a .bed file and create a list of beads.

# List of functions:
# get_chromosome: retrieves chromosome according to arm of it;
# read_bed: reads the .bed file containing the genomic information;
##########################################################################################


#-----------------------------------------------------------------------------------------------
# Returning the chromosome corresponding to a chromosome arm.
def get_chromosome(arm):

    return ARM_TO_CHROMOSOME[arm]
#-----------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------
# Reading the .bed file for genomic information and creating a Bead object for each genomic bin.
def read_bed(filename):

    # variables:
    bed_lookup = {}

    # opening the .bed file:
    with open(filename, "r") as file:

        for line in file:

            # Phase 1: .bed lookup (the file may not contain info for every genome fragment!).
            # avoiding blank lines:
            if not line.strip():
                continue

            if line.startswith("chrom"):
                continue

            # splitting the lines in columns:
            columns = line.split()

            # extracting relevant information (ignoring columns[3] = compartment information):
            arm = columns[0]
            if arm in EXCLUDED_CHROMOSOMES:
                continue

            start = int(columns[1])
            end = int(columns[2])

            # assign a bin_id to each line of info in the file (zero based indexing):
            bin_id = start // RESOLUTION

            # .bed file lookup:
            bed_lookup[(arm, bin_id)] = {"start" : start, "end" : end}

    print(f"Experimental bins found: {len(bed_lookup)}.")


    # Phase 2: creating the beads with correct genome length.
    # variables:
    beads = []

    bead_id = 1
    arm_id = 1

    for arm, n_bins in ARM_LENGTHS.items():
        chromosome = ARM_TO_CHROMOSOME[arm]

        for bin_id in range(n_bins):

            # lookup key to look for experimental information in the file:
            key = (arm, bin_id)

            if key in bed_lookup:
                start = bed_lookup[key]["start"]
                end = bed_lookup[key]["end"]
            else:
                start = bin_id * RESOLUTION + 1
                end = (bin_id + 1) * RESOLUTION

            # creating the bead:
            bead = Bead(
                bead_id = bead_id,
                arm = arm,
                chromosome = chromosome,
                arm_id = arm_id,
                start = start,
                end = end,
            )

            beads.append(bead)

            bead_id += 1

        arm_id += 1

    return beads
#-----------------------------------------------------------------------------------------------






















