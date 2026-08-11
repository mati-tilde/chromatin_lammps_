import numpy as np

#########################################################
#
# LAMMPS WRITER: allows to write a .data file for lammps
#
#########################################################


#--------------------------------------------------------------------------
# Writing the header for the file.
# input: 1) genome: genome dictionary built before;
#  	 2) n_atom_types: number of atom types;
#  	 3) n_bond_types: number of bond types;
#   	 4) margin: to construct length of the box from coordinates.

def write_header(genome, n_atom_types, n_bond_types, margin = 2.0):

    #---------------------------------------------------
    # sanity check:
    if not genome:
        raise ValueError("Genome dictionary is empty.")
    #---------------------------------------------------

    # global quantities:
    n_atoms = sum(len(coords) for coords in genome.values())
    n_bonds = sum(len(coords) - 1 for coords in genome.values())

    # stacking coordinates together (gives array (n_atoms, 3)):
    all_coords = np.vstack(list(genome.values()))

    # computing box limitations (adapts to all type of geometry at once):
    xmin = np.min(all_coords[:, 0]) - margin
    xmax = np.max(all_coords[:, 0]) + margin

    ymin = np.min(all_coords[:, 1]) - margin
    ymax = np.max(all_coords[:, 1]) + margin

    zmin = np.min(all_coords[:, 2]) - margin
    zmax = np.max(all_coords[:, 2]) + margin

    # header writing:
    header = []

    header.append("LAMMPS data file")
    header.append("")

    header.append(f"{n_atoms} atoms")
    header.append(f"{n_bonds} bonds")
    header.append("")

    header.append(f"{n_atom_types} atom types")
    header.append(f"{n_bond_types} bond types")
    header.append("")

    header.append(f"{xmin:.6f} {xmax:.6f} xlo xhi")
    header.append(f"{ymin:.6f} {ymax:.6f} ylo yhi")
    header.append(f"{zmin:.6f} {zmax:.6f} zlo zhi")
    header.append("")


    return header
#--------------------------------------------------------------------------




#--------------------------------------------------------------------------
# Writing the masses section in lammps file.
def write_masses(n_atom_types, mass = 1.0):

    #---------------------------------------------------------------
    # sanity check
    if n_atom_types < 1:
        raise ValueError("Number of atom types must be at least 1.")
    #---------------------------------------------------------------

    # masses section:
    masses = []

    masses.append("Masses")
    masses.append("")

    for atom_type in range(1, n_atom_types + 1):

        masses.append(f"{atom_type} {mass:.1f}")

    masses.append("")

    return masses
#--------------------------------------------------------------------------




#--------------------------------------------------------------------------
# Associates Beads with correct coordinates from genome info.
# Works UNAWARE OF HOMOLOGS.
def get_beads_for_chrom(beads, chromosome):

    #---------------------------------------------------------------
    # sanity check:
    valid_chroms = {"chr2", "chr3", "chrX"}

    if chromosome not in valid_chroms:
        raise ValueError(f"Unknown chromosome: {chromosome}.")
    #---------------------------------------------------------------

    chromosome_beads = []

    for bead in beads:
        if bead.chromosome == chromosome:
            chromosome_beads.append(bead)


    return chromosome_beads
#--------------------------------------------------------------------------




#--------------------------------------------------------------------------
#  Writing atoms section in lammps file.
def write_atoms(genome, beads):

    # NOTE:
    # setting it to 1 cause it doesn't bring relevant information now.
    atom_type = 1

    atoms = []

    atoms.append("Atoms")
    atoms.append("")

    atom_id = 1
    molecule_id = 1

    # looping over the genome:
    for chromosome, coords in genome.items():

        # recover the original chromosome name:
        base_chrom = chromosome[:-1]

        # retrieve the corresponding beads:
        chrom_beads = get_beads_for_chrom(beads, base_chrom)

        #-----------------------------------------------------------------------------------------------
        # sanity check:
        if len(coords) != len(chrom_beads):
            raise RuntimeError(f"{chromosome}: {len(coords)} coordinates but {len(chrom_beads)} beads.")
        #-----------------------------------------------------------------------------------------------

        # looping over beads and coordinates simultaneously:
        for bead, coord in zip(chrom_beads, coords):

            x, y, z = coord

            atoms.append(
                f"{atom_id} "
                f"{molecule_id} "
                f"{atom_type} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{z:.6f}"
            )

            atom_id += 1

        molecule_id += 1

    atoms.append("")

    return atoms
#--------------------------------------------------------------------------




#--------------------------------------------------------------------------
# Writing the bonds section.
def write_bonds(genome):

    # one bond type:
    bond_type = 1

    bonds = []

    bonds.append("Bonds")
    bonds.append("")

    bond_id = 1

    # first atom of the current chromosome:
    atom_offset = 1

    # looping over the chromosomes:
    for chromosome, coords in genome.items():

        n_atoms = len(coords)

        # actual bonds:
        for i in range(n_atoms - 1):

            atom1 = atom_offset + i
            atom2 = atom_offset + i + 1

            bonds.append(
                f"{bond_id} "
                f"{bond_type} "
                f"{atom1} "
                f"{atom2}"
            )

            bond_id += 1

        # shifting th eoffset to first atom of next chromosome:
        atom_offset += n_atoms

    bonds.append("")

    return bonds
#--------------------------------------------------------------------------




#--------------------------------------------------------------------------
# Writing the actual .data file.
def write_data(filename, genome, beads, n_atom_types, n_bond_types):

    # header section:
    header = write_header(genome, n_atom_types = n_atom_types, n_bond_types = n_bond_types)

    # masses section:
    masses = write_masses(n_atom_types = n_atom_types)

    # atoms section:
    atoms = write_atoms(genome, beads)

    # bonds section:
    bonds = write_bonds(genome)


    # assemble:
    lines = []

    lines.extend(header)
    lines.extend(masses)
    lines.extend(atoms)
    lines.extend(bonds)

    # writing a file:
    with open(filename, "w") as file:

        file.write("\n".join(lines))
#--------------------------------------------------------------------------









