import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


DATA_FILE = "1Sep_informed_rw_genome.data"

# Same cutoff as the LJ pair style.
CUTOFF = 2.5

# Current test:
# B_11 = B_12 = ... = B_44 = 0.0013
B_VALUE = 0.0013

SIGNAL_NAMES = [
    "H3K27ac",
    "H3K9ac",
    "H3K27me3",
    "H3K9me3",
]


def read_data_file(filename):
    """
    Read the Atoms and Signals sections from a LAMMPS data file.

    Atoms format:
        atom_id  molecule_id  atom_type  x  y  z

    Signals format:
        atom_id  signal_1  signal_2  signal_3  signal_4
    """

    atoms = {}
    signals = {}

    section = None

    with open(filename) as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            # Identify the relevant sections.
            if line == "Atoms":
                section = "atoms"
                continue

            if line == "Signals":
                section = "signals"
                continue

            # Stop reading when another section begins.
            if section in {"atoms", "signals"}:
                if line in {
                    "Bonds",
                    "Angles",
                    "Dihedrals",
                    "Impropers",
                    "Velocities",
                    "Masses",
                    "Pair Coeffs",
                    "Bond Coeffs",
                    "Angle Coeffs",
                    "Dihedral Coeffs",
                    "Improper Coeffs",
                }:
                    section = None
                    continue

            values = line.split()

            if section == "atoms":

                # atom_id mol_id atom_type x y z
                if len(values) < 6:
                    continue

                try:
                    atom_id = int(values[0])
                    mol_id = int(values[1])
                    atom_type = int(values[2])

                    x = float(values[3])
                    y = float(values[4])
                    z = float(values[5])

                except ValueError:
                    # Skip comments or header-like lines.
                    continue

                atoms[atom_id] = {
                    "mol_id": mol_id,
                    "type": atom_type,
                    "coord": [x, y, z],
                }

            elif section == "signals":

                # atom_id signal1 signal2 signal3 signal4
                if len(values) < 5:
                    continue

                try:
                    atom_id = int(values[0])
                    signal_values = [float(x) for x in values[1:5]]

                except ValueError:
                    continue

                signals[atom_id] = signal_values

    return atoms, signals


def print_signal_statistics(signal_array):
    """Print statistics for the four signal channels."""

    print("\n=== RAW SIGNAL STATISTICS ===")
    print(f"Number of signal vectors: {len(signal_array)}")

    for i, name in enumerate(SIGNAL_NAMES):
        values = signal_array[:, i]

        print(f"\n{name}")
        print(f"  min    = {np.min(values):g}")
        print(f"  median = {np.median(values):g}")
        print(f"  mean   = {np.mean(values):g}")
        print(f"  std    = {np.std(values):g}")
        print(f"  p95    = {np.percentile(values, 95):g}")
        print(f"  p99    = {np.percentile(values, 99):g}")
        print(f"  max    = {np.max(values):g}")


def epsilon0_for_types(type_i, type_j):
    """Return the standard LJ epsilon for an atom-type pair."""

    pair = tuple(sorted((type_i, type_j)))

    epsilon_map = {
        (1, 1): 0.415,
        (1, 2): 0.425,
        (2, 2): 0.550,
    }

    try:
        return epsilon_map[pair]
    except KeyError:
        raise ValueError(
            f"Unexpected atom-type pair: ({type_i}, {type_j})"
        )


def print_interaction_statistics(atoms, signals, common_ids):
    """
    Find all pairs with r < CUTOFF and calculate interaction statistics
    for the special case B_{alpha beta} = B_VALUE.
    """

    coordinates = np.asarray(
        [atoms[atom_id]["coord"] for atom_id in common_ids],
        dtype=float,
    )

    atom_types = np.asarray(
        [atoms[atom_id]["type"] for atom_id in common_ids],
        dtype=int,
    )

    signal_array = np.asarray(
        [signals[atom_id] for atom_id in common_ids],
        dtype=float,
    )

    # For B_{alpha beta} = b:
    #
    # S_ij = b * sum_alpha(s_i,alpha) * sum_beta(s_j,beta)
    signal_sum = np.sum(signal_array, axis=1)

    tree = cKDTree(coordinates)

    # Each pair appears once.
    pairs = tree.query_pairs(
        CUTOFF,
        output_type="ndarray",
    )

    print("\n=== INTERACTION STATISTICS ===")
    print(f"Cutoff = {CUTOFF}")
    print(f"B_ij  = {B_VALUE}")
    print(f"Number of spatial pairs = {len(pairs)}")

    if len(pairs) == 0:
        print("No interacting pairs found.")
        return

    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]

    # Calculate S_ij.
    Sij = (
        B_VALUE
        * signal_sum[i_idx]
        * signal_sum[j_idx]
    )

    # Calculate epsilon0 according to atom types.
    epsilon0 = np.empty(len(pairs), dtype=float)

    for n, (i, j) in enumerate(zip(i_idx, j_idx)):
        epsilon0[n] = epsilon0_for_types(
            atom_types[i],
            atom_types[j],
        )

    epsilon_eff = np.maximum(epsilon0, Sij)

    scale = epsilon_eff / epsilon0

    print("\n--- S_ij ---")
    print(f"  min    = {np.min(Sij):g}")
    print(f"  median = {np.median(Sij):g}")
    print(f"  mean   = {np.mean(Sij):g}")
    print(f"  std    = {np.std(Sij):g}")
    print(f"  p95    = {np.percentile(Sij, 95):g}")
    print(f"  p99    = {np.percentile(Sij, 99):g}")
    print(f"  max    = {np.max(Sij):g}")

    print("\n--- epsilon_eff ---")
    print(f"  min    = {np.min(epsilon_eff):g}")
    print(f"  median = {np.median(epsilon_eff):g}")
    print(f"  mean   = {np.mean(epsilon_eff):g}")
    print(f"  p95    = {np.percentile(epsilon_eff, 95):g}")
    print(f"  p99    = {np.percentile(epsilon_eff, 99):g}")
    print(f"  max    = {np.max(epsilon_eff):g}")

    print("\n--- epsilon_eff / epsilon0 ---")
    print(f"  min    = {np.min(scale):g}")
    print(f"  median = {np.median(scale):g}")
    print(f"  mean   = {np.mean(scale):g}")
    print(f"  std    = {np.std(scale):g}")
    print(f"  p95    = {np.percentile(scale, 95):g}")
    print(f"  p99    = {np.percentile(scale, 99):g}")
    print(f"  max    = {np.max(scale):g}")

    modified = Sij > epsilon0

    print("\n--- Pairs affected by B ---")
    print(
        f"  modified pairs = {np.sum(modified)} "
        f"({100.0 * np.mean(modified):.3f}%)"
    )
    print(
        f"  standard LJ pairs = {np.sum(~modified)} "
        f"({100.0 * np.mean(~modified):.3f}%)"
    )

    return Sij, scale

## PLOTS.

import numpy as np
import matplotlib.pyplot as plt


def log_bins(values, n_bins=60):
    """
    Build logarithmically spaced histogram bins from positive values.
    """

    positive = values[np.isfinite(values) & (values > 0)]

    if len(positive) == 0:
        raise ValueError("No positive finite values available for log bins.")

    vmin = np.min(positive)
    vmax = np.max(positive)

    return np.logspace(
        np.log10(vmin),
        np.log10(vmax),
        n_bins + 1,
    )


def plot_signal_distributions(signal_array):
    """
    Plot the distributions of the four raw epigenetic signals.
    """

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for i, (ax, name) in enumerate(zip(axes.flat, SIGNAL_NAMES)):

        values = signal_array[:, i]
        positive = values[np.isfinite(values) & (values > 0)]

        bins = log_bins(positive)

        ax.hist(
            positive,
            bins=bins,
        )

        ax.set_xscale("log")

        ax.set_xlabel("Signal")
        ax.set_ylabel("Count")
        ax.set_title(name)

    fig.suptitle("Raw epigenetic signal distributions")

    fig.tight_layout()
    fig.savefig(
        "raw_signal_distributions.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def plot_signal_boxplot(signal_array):
    """
    Compare the scales of the four raw signal channels.
    """

    data = []

    for i in range(4):
        values = signal_array[:, i]
        positive = values[np.isfinite(values) & (values > 0)]
        data.append(positive)

    plt.figure(figsize=(9, 6))

    plt.boxplot(
        data,
        tick_labels=SIGNAL_NAMES,
        showfliers=False,
    )

    plt.yscale("log")

    plt.ylabel("Signal")
    plt.title("Scale of the four epigenetic signals")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        "signal_scale_boxplot.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def plot_Sij_distribution(Sij):
    """
    Plot the distribution of S_ij using logarithmic bins.
    """

    positive = Sij[np.isfinite(Sij) & (Sij > 0)]

    bins = log_bins(positive)

    plt.figure(figsize=(8, 5))

    plt.hist(
        positive,
        bins=bins,
    )

    plt.xscale("log")

    plt.xlabel(r"$S_{ij}$")
    plt.ylabel("Count")
    plt.title(r"Distribution of $S_{ij}$")

    plt.tight_layout()

    plt.savefig(
        "Sij_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def plot_interaction_scale(scale):
    """
    Plot the distribution of epsilon_eff / epsilon0.
    """

    positive = scale[np.isfinite(scale) & (scale > 0)]

    bins = log_bins(positive)

    plt.figure(figsize=(8, 5))

    plt.hist(
        positive,
        bins=bins,
    )

    plt.xscale("log")

    # Standard LJ interaction:
    plt.axvline(
        1.0,
        linestyle="--",
        label=r"$\epsilon_{ij}^{\mathrm{eff}}/\epsilon_{ij}^{0}=1$",
    )

    # Edoardo reference value:
    plt.axvline(
        16.5,
        linestyle="--",
        label=r"Edoardo reference $\approx 16.5$",
    )

    plt.xlabel(
        r"$\epsilon_{ij}^{\mathrm{eff}}/\epsilon_{ij}^{0}$"
    )
    plt.ylabel("Count")
    plt.title("Interaction-strength multiplier")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "interaction_scale_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

def plot_interaction_scale_cumulative(scale):
    """
    Plot the cumulative distribution of epsilon_eff / epsilon0.
    """

    values = scale[np.isfinite(scale) & (scale > 0)]

    values = np.sort(values)

    cumulative = np.arange(1, len(values) + 1) / len(values)

    plt.figure(figsize=(8, 5))

    plt.plot(
        values,
        cumulative,
    )

    plt.xscale("log")

    plt.axvline(
        1.0,
        linestyle="--",
        label=r"$\epsilon_{ij}^{\mathrm{eff}}/\epsilon_{ij}^{0}=1$",
    )

    plt.axvline(
        16.5,
        linestyle="--",
        label=r"Edoardo reference $\approx 16.5$",
    )

    plt.xlabel(
        r"$\epsilon_{ij}^{\mathrm{eff}}/\epsilon_{ij}^{0}$"
    )
    plt.ylabel("Cumulative fraction")
    plt.title("Cumulative interaction-strength distribution")

    plt.ylim(0, 1)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "interaction_scale_cumulative.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

def main():

    atoms, signals = read_data_file(DATA_FILE)

    print("=== DATA CONSISTENCY ===")
    print(f"Number of atoms          = {len(atoms)}")
    print(f"Number of signal entries = {len(signals)}")

    atom_ids = set(atoms)
    signal_ids = set(signals)

    common_ids = sorted(atom_ids & signal_ids)

    atoms_without_signals = atom_ids - signal_ids
    signals_without_atoms = signal_ids - atom_ids

    print(f"Atoms with signals       = {len(common_ids)}")
    print(f"Atoms without signals    = {len(atoms_without_signals)}")
    print(f"Signals without atom     = {len(signals_without_atoms)}")

    if atoms_without_signals:
        print("\nWARNING: some atoms have no signal entry.")

    if signals_without_atoms:
        print("\nWARNING: some signal entries have no atom.")

    if not common_ids:
        raise RuntimeError("No common atom/signal IDs found.")

    # Build signal array in atom-ID order.
    signal_array = np.asarray(
        [signals[atom_id] for atom_id in common_ids],
        dtype=float,
    )

    plot_signal_distributions(signal_array)

    plot_signal_boxplot(signal_array)

    Sij, scale = print_interaction_statistics(
        atoms,
        signals,
        common_ids,
    )

    plot_Sij_distribution(Sij)

    plot_interaction_scale(scale)

    plot_interaction_scale_cumulative(scale)


if __name__ == "__main__":
    main()
