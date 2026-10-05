import numpy as np
import matplotlib

# Avoid the WSL graphical-backend problem
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


SIGNAL_NAMES = [
    "H3K27ac",
    "H3K9ac",
    "H3K27me3",
    "H3K9me3",
]


# ============================================================
# Read coordinates and signals
# ============================================================

def read_data_file(filename):

    coordinates = {}
    signals = {}

    section = None

    with open(filename, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # --------------------------------------------
            # Section detection
            # --------------------------------------------

            if line == "Atoms":
                section = "atoms"
                continue

            if line == "Signals":
                section = "signals"
                continue

            if line in {
                "Bonds",
                "Angles",
                "Dihedrals",
                "Impropers",
                "Velocities",
                "Masses",
                "Pair Coeffs",
                "Bond Coeffs",
            }:
                section = None
                continue

            # --------------------------------------------
            # Atoms
            # --------------------------------------------

            if section == "atoms":

                parts = line.split()

                if len(parts) < 6:
                    continue

                atom_id = int(parts[0])

                coordinates[atom_id] = [
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                ]

            # --------------------------------------------
            # Signals
            # --------------------------------------------

            elif section == "signals":

                parts = line.split()

                if len(parts) != 5:
                    break

                atom_id = int(parts[0])

                signals[atom_id] = [
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                ]

    # --------------------------------------------------------
    # Keep atoms for which both coordinates and signals exist
    # --------------------------------------------------------

    atom_ids = sorted(
        set(coordinates) & set(signals)
    )

    coords = np.array(
        [coordinates[i] for i in atom_ids],
        dtype=float
    )

    signal_array = np.array(
        [signals[i] for i in atom_ids],
        dtype=float
    )

    return atom_ids, coords, signal_array


# ============================================================
# Normalization 1
# Log transform with median scale
# ============================================================

def log_normalization(signals):

    s0 = np.median(
        signals,
        axis=0
    )

    normalized = np.log1p(
        signals / s0
    )

    return normalized, s0


# ============================================================
# Normalization 2
# Supervisor's Version A
# ============================================================

def log_clip_normalization(signals):

    epsilon = 1e-6

    values_log = np.log(
        signals + epsilon
    )

    p_high = np.percentile(
        values_log,
        99,
        axis=0
    )

    clipped = np.clip(
        values_log,
        0,
        p_high
    )

    normalized = clipped / p_high

    return normalized, p_high


# ============================================================
# Spatial pairs
# ============================================================

def find_pairs(coordinates, cutoff):

    tree = cKDTree(coordinates)

    pairs = tree.query_pairs(
        cutoff,
        output_type="ndarray"
    )

    if len(pairs) == 0:
        return np.empty(
            (0, 2),
            dtype=int
        )

    return pairs


# ============================================================
# B interaction matrix
# ============================================================

def make_B(value):

    """
    Symmetric 4x4 interaction matrix.
    Every element is set to the same value.
    """

    return np.full(
        (4, 4),
        value,
        dtype=float
    )


# ============================================================
# Calculate S_ij
# ============================================================

def calculate_Sij(
    signals,
    pairs,
    B
):

    if len(pairs) == 0:
        return np.array([])

    si = signals[pairs[:, 0]]
    sj = signals[pairs[:, 1]]

    # Sij = si^T B sj

    Sij = np.einsum(
        "ia,ab,ib->i",
        si,
        B,
        sj
    )

    return Sij


# ============================================================
# Calculate lambda
# ============================================================

def calculate_lambda(
    Sij,
    epsilon0
):

    epsilon_eff = np.maximum(
        epsilon0,
        Sij
    )

    return epsilon_eff / epsilon0


# ============================================================
# Print signal statistics
# ============================================================

def print_signal_statistics(
    name,
    signals
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    for i, signal_name in enumerate(SIGNAL_NAMES):

        values = signals[:, i]

        print(
            f"{signal_name:10s} "
            f"median={np.median(values):8.4f}   "
            f"p75={np.percentile(values, 75):8.4f}   "
            f"p95={np.percentile(values, 95):8.4f}   "
            f"p99={np.percentile(values, 99):8.4f}   "
            f"max={np.max(values):8.4f}"
        )


# ============================================================
# Print fraction of zero signals
# ============================================================

def print_zero_fraction(
    name,
    signals
):

    print()
    print(f"Zero fraction for {name}:")

    for i, signal_name in enumerate(SIGNAL_NAMES):

        fraction = np.mean(
            signals[:, i] == 0
        )

        print(
            f"  {signal_name:10s}: "
            f"{100 * fraction:.4f}%"
        )


# ============================================================
# Print interaction statistics
# ============================================================

def print_interaction_statistics(
    name,
    Sij,
    lambdas,
    epsilon0
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Number of spatial pairs: "
        f"{len(Sij):,}"
    )

    print()
    print("Sij:")

    print(
        f"  min    = {np.min(Sij):.6g}"
    )

    print(
        f"  median = {np.median(Sij):.6g}"
    )

    print(
        f"  p75    = {np.percentile(Sij, 75):.6g}"
    )

    print(
        f"  p95    = {np.percentile(Sij, 95):.6g}"
    )

    print(
        f"  p99    = {np.percentile(Sij, 99):.6g}"
    )

    print(
        f"  max    = {np.max(Sij):.6g}"
    )

    # --------------------------------------------------------
    # Number of affected pairs
    # --------------------------------------------------------

    affected = Sij > epsilon0

    n_affected = np.count_nonzero(
        affected
    )

    fraction_affected = (
        n_affected / len(Sij)
    )

    print()
    print(
        f"Pairs with Sij > epsilon0 "
        f"(epsilon0 = {epsilon0}):"
    )

    print(
        f"  number   = {n_affected:,}"
    )

    print(
        f"  fraction = "
        f"{100 * fraction_affected:.4f}%"
    )

    # --------------------------------------------------------
    # Lambda statistics
    # --------------------------------------------------------

    print()
    print("lambda = epsilon_eff / epsilon0:")

    print(
        f"  min    = {np.min(lambdas):.6g}"
    )

    print(
        f"  median = {np.median(lambdas):.6g}"
    )

    print(
        f"  p75    = {np.percentile(lambdas, 75):.6g}"
    )

    print(
        f"  p95    = {np.percentile(lambdas, 95):.6g}"
    )

    print(
        f"  p99    = {np.percentile(lambdas, 99):.6g}"
    )

    print(
        f"  max    = {np.max(lambdas):.6g}"
    )


# ============================================================
# Plot Sij distributions
# ============================================================

def plot_Sij(
    results,
    output="Sij_normalization_comparison.png"
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    line_styles = [
        "-",
        "--",
    ]

    grayscale = [
        "0.15",
        "0.60",
    ]

    for (
        name,
        Sij,
        lambdas
    ), linestyle, gray in zip(
        results,
        line_styles,
        grayscale
    ):

        positive = Sij[Sij > 0]

        ax.hist(
            positive,
            bins=150,
            density=True,
            histtype="step",
            linewidth=1.6,
            linestyle=linestyle,
            color=gray,
            label=name
        )

    ax.axvline(
        0.415,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=r"$\epsilon_0=0.415$"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$S_{ij}$")
    ax.set_ylabel("Density")

    ax.set_title(
        r"Interaction strength $S_{ij}$"
    )

    ax.legend()

    ax.grid(
        True,
        which="both",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Plot lambda distributions
# ============================================================

def plot_lambda(
    results,
    output="lambda_normalization_comparison.png"
):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    line_styles = [
        "-",
        "--",
    ]

    grayscale = [
        "0.15",
        "0.60",
    ]

    for (
        name,
        Sij,
        lambdas
    ), linestyle, gray in zip(
        results,
        line_styles,
        grayscale
    ):

        ax.hist(
            lambdas,
            bins=150,
            density=True,
            histtype="step",
            linewidth=1.6,
            linestyle=linestyle,
            color=gray,
            label=name
        )

    ax.axvline(
        1.0,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=r"$\lambda=1$"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        r"$\lambda_{ij}=\epsilon_{\mathrm{eff}}/\epsilon_0$"
    )

    ax.set_ylabel("Density")

    ax.set_title(
        r"Effective interaction scale $\lambda_{ij}$"
    )

    ax.legend()

    ax.grid(
        True,
        which="both",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def B_sweep(
    signals,
    pairs,
    epsilon0,
    B_values
):

    # Calculate Sij for B = 1
    B = np.ones((4, 4))

    Sij_B1 = calculate_Sij(
        signals,
        pairs,
        B
    )

    print()
    print("=" * 70)
    print("B SWEEP")
    print("=" * 70)

    print(
        f"{'B':>8s}"
        f"{'N affected':>15s}"
        f"{'Fraction (%)':>15s}"
        f"{'Sij median':>15s}"
        f"{'Sij p95':>15s}"
        f"{'Sij max':>15s}"
    )

    for B_value in B_values:

        Sij = B_value * Sij_B1

        affected = Sij > epsilon0

        n_affected = np.count_nonzero(
            affected
        )

        fraction = (
            100.0
            * n_affected
            / len(Sij)
        )

        print(
            f"{B_value:8.4f}"
            f"{n_affected:15,d}"
            f"{fraction:15.4f}"
            f"{np.median(Sij):15.6f}"
            f"{np.percentile(Sij, 95):15.6f}"
            f"{np.max(Sij):15.6f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    data_file = "1Sep_informed_rw_genome.data"

    cutoff = 2.5

    epsilon0 = 0.415

    B_value = 0.5

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    atom_ids, coordinates, signals = read_data_file(
        data_file
    )

    print()
    print("=" * 70)
    print("DATA")
    print("=" * 70)

    print(
        f"Number of beads: {len(atom_ids):,}"
    )

    # --------------------------------------------------------
    # Raw statistics
    # --------------------------------------------------------

    print_signal_statistics(
        "RAW SIGNALS",
        signals
    )

    # --------------------------------------------------------
    # Normalization 1
    # --------------------------------------------------------

    log_signals, log_scales = log_normalization(
        signals
    )

    # --------------------------------------------------------
    # Normalization 2
    # --------------------------------------------------------

    clipped_signals, clip_scales = (
        log_clip_normalization(signals)
    )

    # --------------------------------------------------------
    # Print normalization information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("NORMALIZATION PARAMETERS")
    print("=" * 70)

    print()
    print("Median values used for log normalization:")

    for name, value in zip(
        SIGNAL_NAMES,
        log_scales
    ):

        print(
            f"  {name:10s}: {value:.6g}"
        )

    print()
    print(
        "99th percentile of log(signal) "
        "used for Version A:"
    )

    for name, value in zip(
        SIGNAL_NAMES,
        clip_scales
    ):

        print(
            f"  {name:10s}: {value:.6g}"
        )

    # --------------------------------------------------------
    # Statistics after normalization
    # --------------------------------------------------------

    print_signal_statistics(
        "LOG NORMALIZATION",
        log_signals
    )

    print_signal_statistics(
        "LOG + 99% CLIPPING (VERSION A)",
        clipped_signals
    )

    # --------------------------------------------------------
    # Zero fractions
    # --------------------------------------------------------

    print_zero_fraction(
        "LOG NORMALIZATION",
        log_signals
    )

    print_zero_fraction(
        "VERSION A",
        clipped_signals
    )

    # --------------------------------------------------------
    # Find spatial pairs
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SPATIAL PAIRS")
    print("=" * 70)

    pairs = find_pairs(
        coordinates,
        cutoff
    )

    print(
        f"Pairs with r < {cutoff}: "
        f"{len(pairs):,}"
    )

    # --------------------------------------------------------
    # B sweep for Version A
    # --------------------------------------------------------

    B_values = np.arange(
        0.03,
        0.301,
        0.005
    )

    B_sweep(
        clipped_signals,
        pairs,
        epsilon0,
        B_values
    )

    # --------------------------------------------------------
    # Interaction matrix
    # --------------------------------------------------------

    B = make_B(B_value)

    print()
    print("B matrix:")

    print(B)

    print(
        f"\nB value = {B_value}"
    )

    print(
        f"epsilon0 = {epsilon0}"
    )

    # --------------------------------------------------------
    # Calculate interactions
    # --------------------------------------------------------

    results = []

    cases = [
        (
            "Log normalization",
            log_signals
        ),
        (
            "Version A",
            clipped_signals
        ),
    ]

    for name, normalized_signals in cases:

        Sij = calculate_Sij(
            normalized_signals,
            pairs,
            B
        )

        lambdas = calculate_lambda(
            Sij,
            epsilon0
        )

        print_interaction_statistics(
            name,
            Sij,
            lambdas,
            epsilon0
        )

        results.append(
            (
                name,
                Sij,
                lambdas
            )
        )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_Sij(results)

    plot_lambda(results)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print("Created:")
    print("  Sij_normalization_comparison.png")
    print("  lambda_normalization_comparison.png")


if __name__ == "__main__":
    main()