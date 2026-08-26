import numpy as np

from genome_parser import read_bed
from builder import split_by_arms, build_chroms, build_genome


# ------------------------------------------------------
# Parameters
# ------------------------------------------------------

BED_FILE = "comp_HiTC_10000_WT_ST10_refined_with_chromwide.bed"

STEP_SIZE = 1.0
SEED = 42

CYLINDER_RADIUS = 8.0
CYLINDER_HEIGHT = 68.0

# ------------------------------------------------------
# Minimum non-consecutive distance
# ------------------------------------------------------

def check_self_overlap(coords):

    n = len(coords)

    min_distance = np.inf
    min_pair = None

    counts = {
        0.2: 0,
        0.5: 0,
        1.0: 0
    }

    # Compare only non-consecutive beads.
    # j = i+1 is the bonded neighbor, so skip it.
    for i in range(n):

        for j in range(i + 2, n):

            distance = np.linalg.norm(coords[i] - coords[j])

            if distance < min_distance:
                min_distance = distance
                min_pair = (i, j)

            for cutoff in counts:
                if distance < cutoff:
                    counts[cutoff] += 1

    return min_distance, min_pair, counts


# ------------------------------------------------------
# Build genome
# ------------------------------------------------------

beads = read_bed(BED_FILE)
arms = split_by_arms(beads)

rng = np.random.default_rng(SEED)

chroms = build_chroms(
    arms,
    method="random_walk",
    step_size=STEP_SIZE,
    rng=rng,
    boundary="cylinder",
    cyl_radius=CYLINDER_RADIUS,
    cyl_height=CYLINDER_HEIGHT
)

genome = build_genome(
    chroms,
    arms,
    method="random_walk",
    step_size=STEP_SIZE,
    rng=rng,
    boundary="cylinder",
    cyl_radius=CYLINDER_RADIUS,
    cyl_height=CYLINDER_HEIGHT
)


# ------------------------------------------------------
# Check each chromosome
# ------------------------------------------------------

print("\nSelf-overlap check")
print("=" * 60)

for chromosome, coords in genome.items():

    min_distance, min_pair, counts = check_self_overlap(coords)

    print(f"\n{chromosome}")

    print(f"Minimum non-consecutive distance : {min_distance:.6f}")

    if min_pair is not None:
        print(
            f"Closest pair                     : "
            f"{min_pair[0]} - {min_pair[1]}"
        )

    print(f"Pairs with r < 0.2                : {counts[0.2]}")
    print(f"Pairs with r < 0.5                : {counts[0.5]}")
    print(f"Pairs with r < 1.0                : {counts[1.0]}")


# ------------------------------------------------------
# Inter chromosomes check
# ------------------------------------------------------


chromosome_names = list(genome.keys())

print("\nInter-chromosome distance check")
print("=" * 70)

for a in range(len(chromosome_names)):

    chrom_a = chromosome_names[a]
    coords_a = genome[chrom_a]

    for b in range(a + 1, len(chromosome_names)):

        chrom_b = chromosome_names[b]
        coords_b = genome[chrom_b]

        differences = (
            coords_a[:, None, :]
            - coords_b[None, :, :]
        )

        squared_distances = np.sum(
            differences ** 2,
            axis=2
        )

        closest_pair = np.unravel_index(
            np.argmin(squared_distances),
            squared_distances.shape
        )

        min_distance = np.sqrt(
            squared_distances[closest_pair]
        )

        print(f"\n{chrom_a} - {chrom_b}")
        print(f"Minimum distance : {min_distance:.6f}")

        for cutoff in [0.1, 0.2, 0.5, 0.7, 1.0]:

            count = np.count_nonzero(
                squared_distances < cutoff ** 2
            )

            print(
                f"Pairs with r < {cutoff:<3} : {count}"
            )
