import numpy as np
from scipy.spatial import cKDTree

from genome_parser import read_bed
from builder import split_by_arms, build_chroms


# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

BED_FILE = "comp_HiTC_10000_WT_ST10_refined_with_chromwide.bed"

HELIX_RADIUS = 5.0
HELIX_HEIGHT = 68.0
HELIX_TURNS = 50.0


# ------------------------------------------------------------
# Helper: self-overlap
# ------------------------------------------------------------

def minimum_nonconsecutive_distance(coords):
    """
    Find the minimum distance between non-consecutive beads.

    Directly bonded neighbours are excluded.
    """

    min_distance = np.inf
    closest_pair = None

    n = len(coords)

    for i in range(n - 2):

        # Exclude:
        #   i itself
        #   i+1, the directly bonded neighbour
        candidates = np.arange(i + 2, n)

        distances = np.linalg.norm(
            coords[candidates] - coords[i],
            axis=1
        )

        j_local = np.argmin(distances)
        distance = distances[j_local]
        j = candidates[j_local]

        if distance < min_distance:
            min_distance = distance
            closest_pair = (i, int(j))

    return min_distance, closest_pair


# ------------------------------------------------------------
# Generate all chromosomes using the actual builder
# ------------------------------------------------------------

beads = read_bed(BED_FILE)
arms = split_by_arms(beads)

chroms = build_chroms(
    arms,
    method="helix",
    radius=HELIX_RADIUS,
    height=HELIX_HEIGHT,
    turns=HELIX_TURNS
)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print("=" * 70)
print("HELIX FOOTPRINT TEST")
print("=" * 70)

print()
print(f"Radius : {HELIX_RADIUS}")
print(f"Height : {HELIX_HEIGHT}")
print(f"Turns  : {HELIX_TURNS}")

print()
print("=" * 70)
print("CHROMOSOMES")
print("=" * 70)


for chrom, coords in chroms.items():

    # --------------------------------------------------------
    # Radial extent
    # --------------------------------------------------------

    r = np.sqrt(
        coords[:, 0] ** 2 +
        coords[:, 1] ** 2
    )

    # --------------------------------------------------------
    # Cartesian extent
    # --------------------------------------------------------

    xmin = coords[:, 0].min()
    xmax = coords[:, 0].max()

    ymin = coords[:, 1].min()
    ymax = coords[:, 1].max()

    zmin = coords[:, 2].min()
    zmax = coords[:, 2].max()

    # --------------------------------------------------------
    # Bonds
    # --------------------------------------------------------

    bonds = np.linalg.norm(
        coords[1:] - coords[:-1],
        axis=1
    )

    # --------------------------------------------------------
    # Self overlap
    # --------------------------------------------------------

    min_distance, closest_pair = (
        minimum_nonconsecutive_distance(coords)
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print(chrom)
    print("-" * 70)

    print(f"Number of beads : {len(coords)}")

    print()
    print("Radial extent")
    print(
        f"    min radius : {r.min():.6f}"
    )
    print(
        f"    max radius : {r.max():.6f}"
    )

    print()
    print("Cartesian extent")
    print(
        f"    x : [{xmin:.6f}, {xmax:.6f}]"
    )
    print(
        f"    y : [{ymin:.6f}, {ymax:.6f}]"
    )
    print(
        f"    z : [{zmin:.6f}, {zmax:.6f}]"
    )

    print()
    print("Bond statistics")
    print(
        f"    minimum : {bonds.min():.6f}"
    )
    print(
        f"    maximum : {bonds.max():.6f}"
    )
    print(
        f"    mean    : {bonds.mean():.6f}"
    )

    print()
    print("Self-overlap")
    print(
        f"    minimum non-consecutive distance : "
        f"{min_distance:.6e}"
    )
    print(
        f"    closest pair : {closest_pair}"
    )


print()
print("=" * 70)
print("END")
print("=" * 70)
