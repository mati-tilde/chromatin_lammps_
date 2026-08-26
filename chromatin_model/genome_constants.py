import numpy as np

##################################################
# DEFINING CONSTANTS THAT OTHER .py FILE MAY NEED.
##################################################

#-------------------------------------------------
# Resolution of our analysis.

RESOLUTION = 10_000
#-------------------------------------------------


#-------------------------------------------------
# Defines the length of each chromosome arm.

ARM_LENGTHS = {
    "chr2L": 2352,
    "chr2R": 2529,
    "chr3L": 2812,
    "chr3R": 3208,
    "chrX": 2355,
}
#-------------------------------------------------


#-------------------------------------------------
# Defines which chromosome the arms belong to.

ARM_TO_CHROMOSOME = {
    "chr2L": "chr2",
    "chr2R": "chr2",
    "chr3L": "chr3",
    "chr3R": "chr3",
    "chrX": "chrX",
    "chr4": "chr4",
    "chrY": "chrY",
}
#-------------------------------------------------


#-------------------------------------------------
# Assigns molecule_id to a single chromosome.

CHROMOSOME_TO_MOLECULE = {
    "chr2": 1,
    "chr3": 2,
    "chrX": 3,
}
#-------------------------------------------------


#-------------------------------------------------
# Names used in the BigWig files.
ARM_TO_BIGWIG = {
    "chr2L": "2L",
    "chr2R": "2R",
    "chr3L": "3L",
    "chr3R": "3R",
    "chrX": "X",
}
#-------------------------------------------------


#-------------------------------------------------
CHROMOSOME_ARMS = {
    "chr2": ("chr2L", "chr2R"),
    "chr3": ("chr3L", "chr3R"),
    "chrX": ("chrX",),
}
#-------------------------------------------------


#-------------------------------------------------
# Defines chromosomes to be excluded.

EXCLUDED_CHROMOSOMES = {
    "chr4",
    "chrY",
}
#-------------------------------------------------


#-------------------------------------------------
# Initial layout of the diploid genome.
'''
X_SPACING = 40.0
Y_SPACING = 40.0
'''

# for the random walk:
INITIAL_LAYOUT_RW = {

    "chr2A": np.array([ 18.0,   0.0,   0.0]),
    "chr3A": np.array([  9.0,  15.588, 0.0]),
    "chrXA": np.array([ -9.0,  15.588, 0.0]),

    "chr2B": np.array([-18.0,   0.0,   0.0]),
    "chr3B": np.array([ -9.0, -15.588, 0.0]),
    "chrXB": np.array([  9.0, -15.588, 0.0]),
}

#for the helix:
INITIAL_LAYOUT = {

    "chr2A": np.array([-3.0,  3.0, 0.0]),
    "chr3A": np.array([ 3.0,  3.0, 0.0]),
    "chrXA": np.array([ 0.0,  3.0, 0.0]),

    "chr2B": np.array([-3.0, -3.0, 0.0]),
    "chr3B": np.array([ 3.0, -3.0, 0.0]),
    "chrXB": np.array([ 0.0, -3.0, 0.0]),
}
#-------------------------------------------------


#-------------------------------------------------
# IDs for each copy of the chromosome.
MOLECULE_IDS = {
    "chr2A": 1,
    "chr2B": 2,
    "chr3A": 3,
    "chr3B": 4,
    "chrXA": 5,
    "chrXB": 6,
}
#-------------------------------------------------


#-------------------------------------------------
NUCLEAR_RADIUS = 28.0
NUCLEAR_HEIGHT = 68.0
#-------------------------------------------------
