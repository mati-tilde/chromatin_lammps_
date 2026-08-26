import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import cKDTree
from datetime import datetime

from genome_parser import read_bed
from builder import (split_by_arms, build_single_chrom, build_chroms, build_genome)

from genome_constants import (ARM_LENGTHS, INITIAL_LAYOUT, INITIAL_LAYOUT_RW)


# ==========================================================
# PARAMETERS
# ==========================================================

BED_FILE = "comp_HiTC_10000_WT_ST10_refined_with_chromwide.bed"

GEOMETRY = "helix"

# for random walk:
STEP_SIZE = 1.0
SEED = 42

# for helix:
HELIX_RADIUS = 5
HELIX_HEIGHT = 68
HELIX_TURNS = 50
HELIX_DRIFT = 5

# for cylindrical boundary (needed only for random walks):
CYL_RADIUS = 8
CYL_HEIGHT = 68

# global nuclear confinement:
NUCLEAR_RADIUS = 28.0
NUCLEAR_HEIGHT = 68.0


REPORT_FILE = "validation_report.txt"

HAPLOID_FIG = "genome_3chroms.png"
DIPLOID_FIG = "genome_6chroms.png"


# ==========================================================
# REPORT UTILITIES
# ==========================================================

def write_line(report, text=""):
    report.write(text + "\n")


def write_separator(report):

    write_line(report, "=" * 70)


def write_section(report, title):

    write_line(report)
    write_line(report, "-" * 70)
    write_line(report, title)
    write_line(report, "-" * 70)

# ==========================================================
# HEADER
# ==========================================================

def write_header(report):

    write_separator(report)

    write_line(report, "GENOME BUILDER VALIDATION REPORT")

    write_separator(report)

    write_line(
        report,
        f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
    )

    write_line(report)

    write_line(report, "Geometry parameters")

    write_line(report, f"Method : {GEOMETRY}")

    if GEOMETRY == "helix":

        write_line(report, f"Radius : {HELIX_RADIUS}")
        write_line(report, f"Height : {HELIX_HEIGHT}")
        write_line(report, f"Turns  : {HELIX_TURNS}")
        write_line(report, f"Drift : 5 for chr2/chr3, 0 for chrX")

    elif GEOMETRY == "random_walk":

        write_line(report, f"Step size : {STEP_SIZE}")
        write_line(report, f"Seed      : {SEED}")


    write_line(report)

    write_line(report, "Boundary")

    if GEOMETRY == "helix":

        write_line(report, "Helix geometry")
        write_line(report, "No intrinsic generation boundary -- section 5 missing.")

    elif GEOMETRY == "random_walk":

        write_line(report, "Random-walk generation boundary")
        write_line(report, f"Cylinder radius : {CYL_RADIUS}")
        write_line(report, f"Cylinder height : {CYL_HEIGHT}")



    write_line(report)

    write_line(report, "Genome layout")


    write_separator(report)



# ==========================================================
# PARSER VALIDATION
# ==========================================================

def validate_parser(report, beads, arms):

    write_section(report, "1. Genome parsing")

    write_line(report, f"Experimental bins : 11377")
    write_line(report, f"Final beads       : {len(beads)}")

    write_line(report)
    write_line(report, "Chromosome arms")

    total = 0

    for arm, expected in ARM_LENGTHS.items():

        found = len(arms[arm])

        total += found

        status = "PASS" if found == expected else "FAIL"

        write_line(
            report,
            f"{arm:5} "
            f"{found:5d} beads "
            f"(expected {expected:5d}) "
            f"{status}"
        )

    write_line(report)

    write_line(report, f"Total beads : {total}")

    if total == len(beads):
        write_line(report, "Genome parsing : PASS")
    else:
        write_line(report, "Genome parsing : FAIL")

# ==========================================================
# CHROMOSOME VALIDATION
# ==========================================================

def validate_chromosomes(report, chroms, arms):

    write_section(report, "2. Chromosome construction")

    expected_sizes = {
        "chr2": len(arms["chr2L"]) + len(arms["chr2R"]),
        "chr3": len(arms["chr3L"]) + len(arms["chr3R"]),
        "chrX": len(arms["chrX"])
    }

    for chrom, coords in chroms.items():

        expected = expected_sizes[chrom]
        found = len(coords)

        status = "PASS" if found == expected else "FAIL"

        write_line(
            report,
            f"{chrom:4} : {found:5d} beads "
            f"(expected {expected:5d}) {status}"
        )

# ==========================================================
# BOND STATISTICS
# ==========================================================

def validate_bonds(report, chroms):

    write_section(report, "3. Bond statistics")

    for chrom, coords in chroms.items():

        bonds = np.linalg.norm(
            coords[1:] - coords[:-1],
            axis=1
        )

        write_line(report, chrom)

        write_line(report,
                   f"    minimum : {bonds.min():.6f}")

        write_line(report,
                   f"    maximum : {bonds.max():.6f}")

        write_line(report,
                   f"    mean    : {bonds.mean():.6f}")

        write_line(report)

# ==========================================================
# ARM JUNCTIONS
# ==========================================================

def validate_junctions(report, chroms, arms):

    write_section(report, "4. Arm junctions")

    junctions = {
        "chr2": len(arms["chr2L"]),
        "chr3": len(arms["chr3L"])
    }

    for chrom, j in junctions.items():

        d = np.linalg.norm(
            chroms[chrom][j] -
            chroms[chrom][j-1]
        )

        write_line(
            report,
            f"{chrom}: {d:.6f}"
        )

# ==========================================================
# BOUNDARY CHECK
# ==========================================================

def validate_boundary(report, chroms):

    write_section(report, "5. Geometry extension")

    for chrom, coords in chroms.items():

        r = np.sqrt(
            coords[:,0]**2 +
            coords[:,1]**2
        )

        z = coords[:,2]

        write_line(report, chrom)

        write_line(report,
                   f"    max radius : {r.max():.6f}")

        write_line(report,
                   f"    min z      : {z.min():.6f}")

        write_line(report,
                   f"    max z      : {z.max():.6f}")

        inside = (
            (r <= CYL_RADIUS).all()
            and
            (z >= 0).all()
            and
            (z <= CYL_HEIGHT).all()
        )


        
        write_line(
            report,
            f"    inside cylinder : {inside}"
        )
        

        write_line(report)

#==========================================================
# SELF OVERLAP CHECK
#==========================================================
def validate_self_overlap(report, chroms):

    write_section(report, "6. Intrachromosomal proximity")

    for chrom, coords in chroms.items():

        min_distance = np.inf
        closest_pair = None

        n = len(coords)

        for i in range(n):

            # Ignore nearby bonded beads.
            # We are looking for non-consecutive overlap.
            start = i + 2

            if start >= n:
                continue

            distances = np.linalg.norm(
                coords[start:] - coords[i],
                axis=1
            )

            j_rel = np.argmin(distances)
            distance = distances[j_rel]
            j = start + j_rel

            if distance < min_distance:
                min_distance = distance
                closest_pair = (i, j)

        write_line(report, chrom)

        write_line(
            report,
            f"    minimum non-consecutive distance : "
            f"{min_distance:.6e}"
        )

        write_line(
            report,
            f"closest bead pair                : "
            f"{closest_pair[0]} - {closest_pair[1]}"
        )

        write_line(report)

# ==========================================================
# NUCLEAR CONFINEMENT
# ==========================================================

def validate_nuclear_extent(report, genome):

    write_section(report, "7. Diploid genome extent")

    all_coords = np.vstack(list(genome.values()))

    r = np.sqrt(
        all_coords[:, 0] ** 2 +
        all_coords[:, 1] ** 2
    )

    z = all_coords[:, 2]

    max_radius = r.max()
    min_z = z.min()
    max_z = z.max()

    radial_clearance = NUCLEAR_RADIUS - max_radius

    write_line(report, f"Nuclear radius : {NUCLEAR_RADIUS:.3f}")
    write_line(report, f"Nuclear height : {NUCLEAR_HEIGHT:.3f}")
    write_line(report)

    write_line(
        report,
        f"Genome radial extent : "
        f"0.000000 - {max_radius:.6f}"
    )

    write_line(
        report,
        f"Genome z extent      : "
        f"{min_z:.6f} - {max_z:.6f}"
    )

    write_line(report)

    write_line(
        report,
        f"Radial clearance to nucleus : "
        f"{radial_clearance:.6f}"
    )

# ==========================================================
# INTER-CHROMOSOME DISTANCES
# ==========================================================

def validate_interchromosome_distances(
    report,
    genome
):

    write_section(
        report,
        "8. Initial inter-chromosome separation"
    )

    chromosome_names = list(genome.keys())

    global_min_distance = np.inf
    global_pair = None

    for a in range(len(chromosome_names)):

        chrom_a = chromosome_names[a]
        coords_a = genome[chrom_a]

        tree = cKDTree(coords_a)

        for b in range(a + 1, len(chromosome_names)):

            chrom_b = chromosome_names[b]
            coords_b = genome[chrom_b]

            distances, indices = tree.query(
                coords_b,
                k=1
            )

            min_distance = distances.min()

            bead_b = np.argmin(distances)
            bead_a = indices[bead_b]

            write_line(
                report,
                f"{chrom_a} - {chrom_b}"
            )

            write_line(
                report,
                f"    minimum distance : "
                f"{min_distance:.6f}"
            )

            write_line(
                report,
                f"    closest beads    : "
                f"{bead_a} - {bead_b}"
            )


            write_line(report)

            if min_distance < global_min_distance:

                global_min_distance = min_distance
                global_pair = (
                    chrom_a,
                    chrom_b,
                    bead_a,
                    bead_b
                )

    write_line(report)

    write_line(
        report,
        "Global closest inter-chromosome pair"
    )

    write_line(
        report,
        f"    chromosomes : "
        f"{global_pair[0]} - {global_pair[1]}"
    )

    write_line(
        report,
        f"    beads       : "
        f"{global_pair[2]} - {global_pair[3]}"
    )

    write_line(
        report,
        f"    distance    : "
        f"{global_min_distance:.6f}"
    )


# ==========================================================
# GENOME VALIDATION
# ==========================================================

def validate_genome(report, genome):

    write_section(report, "9. Diploid genome")

    for chrom in ["chr2", "chr3", "chrX"]:

        equal = np.array_equal(
            genome[f"{chrom}A"],
            genome[f"{chrom}B"]
        )

        independent = (
            genome[f"{chrom}A"]
            is not
            genome[f"{chrom}B"]
        )

        write_line(report, chrom)

        write_line(report,
                   f"    identical coordinates : {equal}")

        write_line(report,
                   f"    independent arrays    : {independent}")

        write_line(report)
# ==========================================================
# CHROMOSOME CENTRES
# ==========================================================

def validate_centres(report, genome):

    write_section(report, "10. Chromosome centres")

    for chrom, coords in genome.items():

        centre = coords.mean(axis=0)

        write_line(
            report,
            f"{chrom:5}: "
            f"({centre[0]:8.3f}, "
            f"{centre[1]:8.3f}, "
            f"{centre[2]:8.3f})"
        )



# ==========================================================
# HAPLOID GENOME
# ==========================================================

def plot_haploid(chroms):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for chrom, coords in chroms.items():

        ax.plot(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            linewidth=1.5,
            label=chrom
        )

    ax.set_title("Initial haploid genome")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=35)

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        HAPLOID_FIG,
        dpi=300
    )

    plt.close()


# ==========================================================
# DIPLOID GENOME
# ==========================================================

def plot_diploid(genome):

    plt.figure(figsize=(10,8))

    for chrom, coords in genome.items():

        plt.plot(
            coords[:,0],
            coords[:,1],
            linewidth=1.5,
            label=chrom
        )

    plt.title("Initial diploid genome")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.axis("equal")
    plt.grid(True)

    plt.legend(
        ncol=2
    )

    plt.tight_layout()

    plt.savefig(
        DIPLOID_FIG,
        dpi=300
    )

    plt.close()


def report_figures(report):

    write_section(report, "11. Figures")

    write_line(report, f"Haploid genome : {HAPLOID_FIG}")
    write_line(report, f"Diploid genome : {DIPLOID_FIG}")

def final_summary(report, genome):

    all_coords = np.vstack(list(genome.values()))

    r = np.sqrt(
        all_coords[:, 0] ** 2 +
        all_coords[:, 1] ** 2
    )

    max_radius = r.max()

    write_separator(report)
    write_line(report)
    write_line(report, "SUMMARY")
    write_line(report)

    write_line(report, f"Geometry : {GEOMETRY}")
    write_line(report, f"Helix radius : {HELIX_RADIUS}")
    write_line(report, f"Helix height : {HELIX_HEIGHT}")
    write_line(report, f"Helix turns : {HELIX_TURNS}")
    write_line(report, f"Drift : chromosome-dependent")
    write_line(report)

    write_line(report, f"Total beads : {len(all_coords)}")
    write_line(report, f"Maximum genome radius : {max_radius:.6f}")
    write_line(report, f"Nuclear radius : {NUCLEAR_RADIUS:.6f}")
    write_line(
        report,
        f"Radial clearance : "
        f"{NUCLEAR_RADIUS - max_radius:.6f}"
    )

    write_line(report)
    write_line(
        report,
        "Initial configuration prepared for "
        "LAMMPS equilibration."
    )

    write_line(report)
    write_separator(report)

def final_summary_rw(report):

    write_separator(report)

    write_line(report)

    write_line(report,
               "Validation completed successfully.")

    write_line(report)

    write_line(report,
               "Genome parser ........ PASS")

    write_line(report,
               "Chromosome builder ... PASS")

    write_line(report,
               "Genome builder ....... PASS")

    write_line(report)

    write_separator(report)

#---------------------------------------------------------------------------------------------------MAIN-----------------------------------------------------------------------------------
# ==========================================================
# MAIN
# ==========================================================

def main():

    report = open(REPORT_FILE, "w")

    write_header(report)

    # ------------------------------------------------------
    # Parse genome
    # ------------------------------------------------------

    beads = read_bed(BED_FILE)

    arms = split_by_arms(beads)

    validate_parser(report, beads, arms)

    # ------------------------------------------------------
    # Build chromosomes
    # ------------------------------------------------------
    rng = np.random.default_rng(SEED)

    if GEOMETRY == "helix":

        chroms = build_chroms(
            arms,
            method="helix",
            radius=HELIX_RADIUS,
            height=HELIX_HEIGHT,
            turns=HELIX_TURNS,
        )

        genome = build_genome(
            chroms,
            arms,
            method="helix",
            radius=HELIX_RADIUS,
            height=HELIX_HEIGHT,
            turns=HELIX_TURNS,
            nuclear_radius=NUCLEAR_RADIUS,
            nuclear_height=NUCLEAR_HEIGHT
        )



    elif GEOMETRY == "random_walk":

        chroms = build_chroms(
            arms,
            method="random_walk",
            step_size=STEP_SIZE,
            rng = rng,
            boundary="cylinder",
            cyl_radius=CYL_RADIUS,
            cyl_height=CYL_HEIGHT
        )

        genome = build_genome(
            chroms,
            arms,
            method="random_walk",
            step_size=STEP_SIZE,
            rng=rng,
            boundary="cylinder",
            cyl_radius=CYL_RADIUS,
            cyl_height=CYL_HEIGHT
        )


    validate_chromosomes(report, chroms, arms)

    validate_bonds(report, chroms)

    validate_junctions(report, chroms, arms)

    if GEOMETRY == "random_walk":
        validate_boundary(report, chroms)

    validate_self_overlap(report, chroms)

    validate_nuclear_extent(
        report,
        genome
    )

    validate_interchromosome_distances(
        report,
        genome
    )

    validate_genome(report, genome)
    validate_centres(report, genome)

    # ------------------------------------------------------
    # Figures
    # ------------------------------------------------------

    plot_haploid(chroms)

    plot_diploid(genome)

    report_figures(report)

    final_summary(report, genome)

    report.close()




if __name__ == "__main__":

    main()
