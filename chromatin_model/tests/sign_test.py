import sys
import numpy as np


SIGNAL_NAMES = [
    "H3K27ac",
    "H3K9ac",
    "H3K27me3",
    "H3K9me3",
]


def read_signals(filename):

    signals = []
    atom_ids = []

    reading = False

    with open(filename, "r") as file:

        for line in file:

            line = line.strip()

            if line == "Signals":
                reading = True
                continue

            if not reading:
                continue

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                break

            atom_ids.append(int(parts[0]))

            signals.append([
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            ])

    return np.asarray(atom_ids), np.asarray(signals)


def main():

    if len(sys.argv) != 2:
        print("Usage:")
        print("python validate_signals.py <data_file>")
        sys.exit(1)

    filename = sys.argv[1]

    atom_ids, signals = read_signals(filename)

    print()
    print("=" * 70)
    print("SIGNAL VALIDATION")
    print("=" * 70)

    print(f"Number of signal entries: {len(signals):,}")

    # --------------------------------------------------------
    # Atom ID check
    # --------------------------------------------------------

    expected_ids = np.arange(
        1,
        len(atom_ids) + 1
    )

    if np.array_equal(atom_ids, expected_ids):
        print("Atom IDs: OK")
    else:
        print("Atom IDs: ERROR")

    # --------------------------------------------------------
    # Signal statistics
    # --------------------------------------------------------

    print()

    for i, name in enumerate(SIGNAL_NAMES):

        values = signals[:, i]

        print(
            f"{name:10s} "
            f"min={np.min(values):.6f}   "
            f"median={np.median(values):.6f}   "
            f"p75={np.percentile(values, 75):.6f}   "
            f"p95={np.percentile(values, 95):.6f}   "
            f"p99={np.percentile(values, 99):.6f}   "
            f"max={np.max(values):.6f}"
        )

    # --------------------------------------------------------
    # Bounds check
    # --------------------------------------------------------

    print()

    if np.all(signals >= 0.0) and np.all(signals <= 1.0):
        print("Signal bounds [0, 1]: OK")
    else:
        print("Signal bounds [0, 1]: ERROR")

    # --------------------------------------------------------
    # Zero fraction
    # --------------------------------------------------------

    print()
    print("Zero fractions:")

    for i, name in enumerate(SIGNAL_NAMES):

        zero_fraction = np.mean(
            signals[:, i] == 0.0
        )

        print(
            f"{name:10s}: "
            f"{100 * zero_fraction:.4f}%"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()