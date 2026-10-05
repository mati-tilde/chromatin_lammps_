import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# PARAMETERS
# ============================================================

DATA_FILE = "5Oct_layout_test_rw.data"

NUCLEAR_RADIUS = 28.0
NUCLEAR_HEIGHT = 68.0

MOLECULE_NAMES = {
    1: "chr2A",
    2: "chr2B",
    3: "chr3A",
    4: "chr3B",
    5: "chrXA",
    6: "chrXB",
}


# ============================================================
# READ ATOMS FROM LAMMPS DATA FILE
# ============================================================

def read_atoms(data_file):

    with open(data_file, "r") as f:
        lines = f.readlines()

    # Find the Atoms section
    atoms_start = None

    for i, line in enumerate(lines):
        if line.strip() == "Atoms":
            atoms_start = i + 1
            break

    if atoms_start is None:
        raise ValueError("Could not find 'Atoms' section.")

    # Skip blank lines immediately after "Atoms"
    while atoms_start < len(lines) and not lines[atoms_start].strip():
        atoms_start += 1

    atoms = []

    for line in lines[atoms_start:]:

        line = line.strip()

        # Stop when another section begins
        if not line:
            continue

        if line in {
            "Velocities",
            "Bonds",
            "Angles",
            "Dihedrals",
            "Impropers",
            "Masses",
            "PairIJ Coeffs",
        }:
            break

        # Ignore comments
        if line.startswith("#"):
            continue

        fields = line.split()

        # Expected atom_style bond:
        #
        # atom-ID molecule-ID atom-type x y z
        #
        # e.g.
        # 1 1 4 -2.955425 3.666153 0.000000

        if len(fields) < 6:
            continue

        atom_id = int(fields[0])
        molecule_id = int(fields[1])
        atom_type = int(fields[2])

        x = float(fields[3])
        y = float(fields[4])
        z = float(fields[5])

        atoms.append(
            [atom_id, molecule_id, atom_type, x, y, z]
        )

    atoms = np.asarray(atoms)

    if len(atoms) == 0:
        raise ValueError("No atoms were found.")

    return atoms


# ============================================================
# SPLIT INTO CHROMOSOMES
# ============================================================

def split_chromosomes(atoms):

    chromosomes = {}

    for mol_id, name in MOLECULE_NAMES.items():

        mask = atoms[:, 1] == mol_id

        coords = atoms[mask, 3:6]

        if len(coords) == 0:
            print(f"WARNING: no atoms found for {name}")
        else:
            chromosomes[name] = coords

    return chromosomes


# ============================================================
# CHROMOSOME CENTRES
# ============================================================

def chromosome_centres(chromosomes):

    centres = {}

    for name, coords in chromosomes.items():
        centres[name] = np.mean(coords, axis=0)

    return centres


# ============================================================
# MINIMUM DISTANCE BETWEEN TWO CHROMOSOMES
# ============================================================

def minimum_distance(coords_a, coords_b):

    # Simple version using pairwise distances.
    #
    # Fine for this diagnostic script.
    #
    # If this becomes slow, we can replace it with cKDTree.

    diff = coords_a[:, None, :] - coords_b[None, :, :]

    distances_squared = np.sum(diff**2, axis=2)

    return np.sqrt(np.min(distances_squared))


# ============================================================
# BASIC GEOMETRY SUMMARY
# ============================================================

def print_summary(chromosomes):

    print("\n" + "=" * 70)
    print("CHROMOSOME GEOMETRY SUMMARY")
    print("=" * 70)

    centres = chromosome_centres(chromosomes)

    for name, coords in chromosomes.items():

        centre = centres[name]

        radial_distance = np.sqrt(
            coords[:, 0]**2 +
            coords[:, 1]**2
        )

        print(f"\n{name}")
        print(f"  beads       : {len(coords)}")
        print(
            f"  centre      : "
            f"({centre[0]:8.3f}, "
            f"{centre[1]:8.3f}, "
            f"{centre[2]:8.3f})"
        )
        print(
            f"  x range     : "
            f"[{coords[:,0].min():8.3f}, "
            f"{coords[:,0].max():8.3f}]"
        )
        print(
            f"  y range     : "
            f"[{coords[:,1].min():8.3f}, "
            f"{coords[:,1].max():8.3f}]"
        )
        print(
            f"  z range     : "
            f"[{coords[:,2].min():8.3f}, "
            f"{coords[:,2].max():8.3f}]"
        )
        print(
            f"  max radius  : "
            f"{radial_distance.max():8.3f}"
        )

    # --------------------------------------------------------
    # Pairwise chromosome distances
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("MINIMUM INTER-CHROMOSOME DISTANCES")
    print("-" * 70)

    names = list(chromosomes.keys())

    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            name_a = names[i]
            name_b = names[j]

            d = minimum_distance(
                chromosomes[name_a],
                chromosomes[name_b]
            )

            print(
                f"{name_a:6s} - {name_b:6s} : "
                f"{d:8.3f}"
            )

    print()


# ============================================================
# NUCLEAR CYLINDER
# ============================================================

def plot_nuclear_boundary_xy(ax):

    theta = np.linspace(0, 2*np.pi, 300)

    x = NUCLEAR_RADIUS * np.cos(theta)
    y = NUCLEAR_RADIUS * np.sin(theta)

    ax.plot(
        x,
        y,
        linestyle="--",
        linewidth=1.5,
        label="nuclear boundary"
    )


def plot_nuclear_boundary_xz(ax):

    z = np.linspace(0, NUCLEAR_HEIGHT, 300)

    x_right = np.full_like(z, NUCLEAR_RADIUS)
    x_left = np.full_like(z, -NUCLEAR_RADIUS)

    ax.plot(
        x_right,
        z,
        linestyle="--",
        linewidth=1.2
    )

    ax.plot(
        x_left,
        z,
        linestyle="--",
        linewidth=1.2
    )

    # top and bottom
    ax.plot(
        [-NUCLEAR_RADIUS, NUCLEAR_RADIUS],
        [0, 0],
        linestyle="--",
        linewidth=1.2
    )

    ax.plot(
        [-NUCLEAR_RADIUS, NUCLEAR_RADIUS],
        [NUCLEAR_HEIGHT, NUCLEAR_HEIGHT],
        linestyle="--",
        linewidth=1.2
    )


# ============================================================
# XY PLOT
# ============================================================

def plot_xy(chromosomes):

    fig, ax = plt.subplots(figsize=(9, 9))

    for name, coords in chromosomes.items():

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=1,
            alpha=0.35,
            label=name
        )

    # chromosome centres
    centres = chromosome_centres(chromosomes)

    for name, centre in centres.items():

        ax.scatter(
            centre[0],
            centre[1],
            s=100,
            marker="x",
            linewidth=2
        )

        ax.text(
            centre[0],
            centre[1],
            f"  {name}",
            fontsize=10
        )

    plot_nuclear_boundary_xy(ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_title("Six-chromosome genome: XY projection")

    ax.set_aspect("equal")

    ax.set_xlim(
        -NUCLEAR_RADIUS - 2,
        NUCLEAR_RADIUS + 2
    )

    ax.set_ylim(
        -NUCLEAR_RADIUS - 2,
        NUCLEAR_RADIUS + 2
    )

    ax.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# XZ PLOT
# ============================================================

def plot_xz(chromosomes):

    fig, ax = plt.subplots(figsize=(10, 7))

    for name, coords in chromosomes.items():

        ax.scatter(
            coords[:, 0],
            coords[:, 2],
            s=1,
            alpha=0.35,
            label=name
        )

    centres = chromosome_centres(chromosomes)

    for name, centre in centres.items():

        ax.scatter(
            centre[0],
            centre[2],
            s=100,
            marker="x",
            linewidth=2
        )

        ax.text(
            centre[0],
            centre[2],
            f"  {name}",
            fontsize=10
        )

    plot_nuclear_boundary_xz(ax)

    ax.set_xlabel("x")
    ax.set_ylabel("z")

    ax.set_title("Six-chromosome genome: XZ projection")

    ax.set_xlim(
        -NUCLEAR_RADIUS - 2,
        NUCLEAR_RADIUS + 2
    )

    ax.set_ylim(
        -2,
        NUCLEAR_HEIGHT + 2
    )

    ax.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# YZ PLOT
# ============================================================

def plot_yz(chromosomes):

    fig, ax = plt.subplots(figsize=(10, 7))

    for name, coords in chromosomes.items():

        ax.scatter(
            coords[:, 1],
            coords[:, 2],
            s=1,
            alpha=0.35,
            label=name
        )

    centres = chromosome_centres(chromosomes)

    for name, centre in centres.items():

        ax.scatter(
            centre[1],
            centre[2],
            s=100,
            marker="x",
            linewidth=2
        )

        ax.text(
            centre[1],
            centre[2],
            f"  {name}",
            fontsize=10
        )

    plot_nuclear_boundary_xz(ax)

    ax.set_xlabel("y")
    ax.set_ylabel("z")

    ax.set_title("Six-chromosome genome: YZ projection")

    ax.set_xlim(
        -NUCLEAR_RADIUS - 2,
        NUCLEAR_RADIUS + 2
    )

    ax.set_ylim(
        -2,
        NUCLEAR_HEIGHT + 2
    )

    ax.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# 3D PLOT
# ============================================================

def plot_3d(chromosomes):

    fig = plt.figure(figsize=(11, 9))

    ax = fig.add_subplot(111, projection="3d")

    for name, coords in chromosomes.items():

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            s=1,
            alpha=0.25,
            label=name
        )

    centres = chromosome_centres(chromosomes)

    for name, centre in centres.items():

        ax.scatter(
            centre[0],
            centre[1],
            centre[2],
            s=80,
            marker="x",
            linewidth=2
        )

        ax.text(
            centre[0],
            centre[1],
            centre[2],
            name
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_title("Six-chromosome genome: 3D")

    ax.set_xlim(
        -NUCLEAR_RADIUS,
        NUCLEAR_RADIUS
    )

    ax.set_ylim(
        -NUCLEAR_RADIUS,
        NUCLEAR_RADIUS
    )

    ax.set_zlim(
        0,
        NUCLEAR_HEIGHT
    )

    ax.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    atoms = read_atoms(DATA_FILE)

    print(f"\nRead {len(atoms)} atoms from:")
    print(f"  {DATA_FILE}")

    chromosomes = split_chromosomes(atoms)

    print_summary(chromosomes)

    plot_xy(chromosomes)
    plot_xz(chromosomes)
    plot_yz(chromosomes)
    plot_3d(chromosomes)


if __name__ == "__main__":
    main()