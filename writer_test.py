from genome_parser import read_bed
from builder import split_by_arms, build_chroms, build_genome
from lammps_writer import write_header, write_masses, write_atoms, write_bonds, write_data

# ------------------------------------------------------
# Parameters
# ------------------------------------------------------

BED_FILE = "comp_HiTC_10000_WT_ST10_refined_with_chromwide.bed"

GEOMETRY = "helix"

HELIX_RADIUS = 17
HELIX_HEIGHT = 68
HELIX_TURNS = 10

N_ATOM_TYPES = 4
N_BOND_TYPES = 1

# ------------------------------------------------------
# Build genome
# ------------------------------------------------------

beads = read_bed(BED_FILE)

arms = split_by_arms(beads)

chroms = build_chroms(
    arms,
    method=GEOMETRY,
    radius=HELIX_RADIUS,
    height=HELIX_HEIGHT,
    turns=HELIX_TURNS
)

genome = build_genome(chroms)

write_data(
    filename="genome.data",
    genome=genome,
    beads=beads,
    n_atom_types=4,
    n_bond_types=1
)

'''
# ------------------------------------------------------
# Header
# ------------------------------------------------------

header = write_header(
    genome,
    n_atom_types=N_ATOM_TYPES,
    n_bond_types=N_BOND_TYPES
)

print("\n========== HEADER ==========\n")

for line in header:
    print(line)

# ------------------------------------------------------
# Masses
# ------------------------------------------------------

masses = write_masses(
    n_atom_types=N_ATOM_TYPES
)

print("\n========== MASSES ==========\n")

for line in masses:
    print(line)


# ------------------------------------------------------
# Atoms
# ------------------------------------------------------

atoms = write_atoms(genome, beads)

print("\n========== ATOMS ==========\n")

for line in atoms[:15]:
    print(line)



with open("writer_test.data", "w") as file:

    file.write("\n".join(header))
    file.write("\n")

    file.write("\n".join(masses))
    file.write("\n")

    file.write("\n".join(atoms))
'''
