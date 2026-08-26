import time
import numpy as np

from scipy.spatial import cKDTree

from genome_parser import read_bed
from builder import split_by_arms, build_chroms, build_genome
from genome_constants import ARM_LENGTHS


# --------------------------------------------------------------------------
# PARAMETERS
# --------------------------------------------------------------------------

BED_FILE = "comp_HiTC_10000_WT_ST10_refined_with_chromwide.bed"

STEP_SIZE = 1.0
SEED = 42

CYL_RADIUS = 8.0
CYL_HEIGHT = 68.0


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

def min_nonconsecutive_distance(coords):
    """
    Find the minimum distance between beads that are not directly bonded.
    """

    min_distance = np.inf
    closest_pair = None

    n = len(coords)

    for i in range(n - 2):

        candidates = coords[i + 2:]

        distances = np.linalg.norm(
            candidates - coords[i],
            axis=1
        )

        j_rel = np.argmin(distances)
        distance = distances[j_rel]
        j = i + 2 + j_rel

        if distance < min_distance:
            min_distance = distance
            closest_pair = (i, j)

    return min_distance, closest_pair


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():

    print("=" * 70)
    print("RANDOM-WALK SMOKE TEST")
    print("=" * 70)

    print()
    print(f"Step size      : {STEP_SIZE}")
    print(f"Seed           : {SEED}")
    print(f"Cylinder R     : {CYL_RADIUS}")
    print(f"Cylinder height: {CYL_HEIGHT}")

    # ----------------------------------------------------------------------
    # Read experimental data
    # ----------------------------------------------------------------------

    print()
    print("Reading experimental bins...")

    beads = read_bed(BED_FILE)
    arms = split_by_arms(beads)

    print(f"Experimental bins found: {len(beads)}")

    # ----------------------------------------------------------------------
    # Expected chromosome sizes
    # ----------------------------------------------------------------------

    expected_sizes = {
        "chr2": ARM_LENGTHS["chr2L"] + ARM_LENGTHS["chr2R"],
        "chr3": ARM_LENGTHS["chr3L"] + ARM_LENGTHS["chr3R"],
        "chrX": ARM_LENGTHS["chrX"],
    }

    # ----------------------------------------------------------------------
    # Random generator
    # ----------------------------------------------------------------------

    rng = np.random.default_rng(SEED)

    # ----------------------------------------------------------------------
    # Build haploid chromosomes
    # ----------------------------------------------------------------------

    print()
    print("Building chromosomes...")
    print()

    t0 = time.perf_counter()

    chroms = build_chroms(
        arms,
        method="random_walk",
        step_size=STEP_SIZE,
        rng=rng,
        boundary="cylinder",
        cyl_radius=CYL_RADIUS,
        cyl_height=CYL_HEIGHT,
    )

    elapsed = time.perf_counter() - t0

    print(f"Generation time: {elapsed:.2f} s")

    # ----------------------------------------------------------------------
    # Haploid chromosome diagnostics
    # ----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("CHROMOSOME RESULTS")
    print("=" * 70)

    for chrom, coords in chroms.items():

        print()
        print(chrom)
        print("-" * 70)

        # --------------------------------------------------------------
        # Bead count
        # --------------------------------------------------------------

        expected = expected_sizes[chrom]

        print(f"Beads    : {len(coords)}")
        print(f"Expected : {expected}")

        # --------------------------------------------------------------
        # Bond lengths
        # --------------------------------------------------------------

        bonds = np.linalg.norm(
            coords[1:] - coords[:-1],
            axis=1
        )

        print()
        print("Bond lengths")
        print(f"    min  : {bonds.min():.6f}")
        print(f"    max  : {bonds.max():.6f}")
        print(f"    mean : {bonds.mean():.6f}")

        # --------------------------------------------------------------
        # Cylinder
        # --------------------------------------------------------------

        r = np.sqrt(
            coords[:, 0] ** 2 +
            coords[:, 1] ** 2
        )

        inside = (
            np.all(r <= CYL_RADIUS)
            and
            np.all(coords[:, 2] >= 0.0)
            and
            np.all(coords[:, 2] <= CYL_HEIGHT)
        )

        print()
        print("Cylinder")
        print(f"    max radius : {r.max():.6f}")
        print(f"    min z      : {coords[:, 2].min():.6f}")
        print(f"    max z      : {coords[:, 2].max():.6f}")
        print(f"    inside     : {inside}")

        # --------------------------------------------------------------
        # Self-avoidance
        # --------------------------------------------------------------

        dmin, pair = min_nonconsecutive_distance(coords)

        print()
        print("Self-avoidance")
        print(
            f"    minimum non-consecutive distance : "
            f"{dmin:.6f}"
        )
        print(f"    closest pair : {pair}")

    # ----------------------------------------------------------------------
    # Build diploid genome
    # ----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING DIPLOID GENOME")
    print("=" * 70)

    t1 = time.perf_counter()

    genome = build_genome(
        chroms,
        arms,
        method="random_walk",
        step_size=STEP_SIZE,
        rng=rng,
        boundary="cylinder",
        cyl_radius=CYL_RADIUS,
        cyl_height=CYL_HEIGHT,
    )

    elapsed_genome = time.perf_counter() - t1

    print(f"Genome build time: {elapsed_genome:.2f} s")
    print(f"Chromosomes generated: {len(genome)}")

    # ----------------------------------------------------------------------
    # Diploid genome diagnostics
    # ----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("DIPLOID GENOME RESULTS")
    print("=" * 70)

    for name, coords in genome.items():

        r = np.sqrt(
            coords[:, 0] ** 2 +
            coords[:, 1] ** 2
        )

        print()
        print(name)
        print("-" * 70)

        print(f"Beads : {len(coords)}")

        print()
        print("Nuclear extent")
        print(f"    max radius : {r.max():.6f}")
        print(f"    min z      : {coords[:, 2].min():.6f}")
        print(f"    max z      : {coords[:, 2].max():.6f}")

    # ----------------------------------------------------------------------
    # Inter-chromosome proximity
    # ----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("INTER-CHROMOSOME PROXIMITY")
    print("=" * 70)

    names = list(genome.keys())

    global_min = np.inf
    global_pair = None

    for i in range(len(names)):

        chrom_a = names[i]
        coords_a = genome[chrom_a]

        tree = cKDTree(coords_a)

        for j in range(i + 1, len(names)):

            chrom_b = names[j]
            coords_b = genome[chrom_b]

            distances, indices = tree.query(
                coords_b,
                k=1
            )

            min_distance = distances.min()

            bead_b = np.argmin(distances)
            bead_a = indices[bead_b]

            print(
                f"{chrom_a:5} - {chrom_b:5} : "
                f"{min_distance:.6f} "
                f"(beads {bead_a} - {bead_b})"
            )

            if min_distance < global_min:

                global_min = min_distance

                global_pair = (
                    chrom_a,
                    chrom_b,
                    int(bead_a),
                    int(bead_b),
                )

    print()
    print(f"Global minimum distance : {global_min:.6f}")

    print(
        f"Closest pair            : "
        f"{global_pair[0]} - {global_pair[1]}"
    )

    print(
        f"Closest beads            : "
        f"{global_pair[2]} - {global_pair[3]}"
    )

    # ----------------------------------------------------------------------
    # End
    # ----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
