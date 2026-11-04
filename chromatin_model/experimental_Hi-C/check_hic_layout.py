#!/usr/bin/env python3
"""
check_hic_layout.py -- aggregate diagnostics of an experimental genome-wide Hi-C .npz (scipy sparse), to confirm
the bin layout and find out how the matrix is normalized. Prints only AGGREGATE numbers (sums, means, ratios),
never individual contact values.

Usage:  python3 check_hic_layout.py WT_ST10.10000_exp.npz [replicate1.npz ...]
(13758 bins: all chromosomes; 13391 bins: without chrY, as in the replicates.)

Assumed layout (dm6, 10 kb, order of the BED file): chr2L 2352, chr2R 2529, chr3L 2812, chr3R 3208, chr4 135,
chrX 2355, chrY 367 bins = 13758.
"""
import sys
import numpy as np
from scipy import sparse

LAYOUT = [("chr2L", 2352), ("chr2R", 2529), ("chr3L", 2812), ("chr3R", 3208), ("chr4", 135), ("chrX", 2355), ("chrY", 367)]
LAYOUTS = {13758: LAYOUT, 13391: LAYOUT[:-1]}          # replicates have no chrY


def cis_fraction(M, sizes):
    """Fraction of the total contact weight that falls inside the diagonal blocks defined by `sizes`."""
    edges = np.concatenate([[0], np.cumsum(sizes)])
    coo = M.tocoo()
    block_r = np.searchsorted(edges, coo.row, side="right")
    block_c = np.searchsorted(edges, coo.col, side="right")
    return coo.data[block_r == block_c].sum() / coo.data.sum()


def main(path):
    M = sparse.load_npz(path).tocsr()
    n = M.shape[0]
    print(f"\n=== {path}: {n} x {n}")
    if n not in LAYOUTS:
        print(f"  size matches no known layout ({sorted(LAYOUTS)}): layout check skipped")
        return
    layout = LAYOUTS[n]
    print(f"  layout: {', '.join(name for name, _ in layout)}")
    sizes = [s for _, s in layout]

    # 1. layout: cis fraction with the assumed boundaries vs boundaries shifted by +/- 100 bins
    edges = np.cumsum(sizes)[:-1]
    print("  1. layout (cis fraction = share of contacts inside chromosome blocks):")
    print(f"     assumed boundaries: {cis_fraction(M, sizes):.4f}")
    for shift in (-100, +100):
        shifted = np.diff(np.concatenate([[0], np.clip(edges + shift, 1, n - 1), [n]]))
        print(f"     all boundaries shifted by {shift:+d} bins: {cis_fraction(M, shifted):.4f}")

    # 2. empty bins per chromosome
    rowsum = np.asarray(M.sum(axis=1)).ravel()
    start = 0
    print("  2. empty bins (no contacts) per chromosome:")
    for name, size in layout:
        print(f"     {name:6s} {int(np.sum(rowsum[start:start + size] == 0)):5d} of {size}")
        start += size

    # 3. normalization: spread of row sums (balanced -> nearly equal), decay with genomic distance
    nz = rowsum[rowsum > 0]
    print(f"  3. row sums of non-empty bins: coefficient of variation {nz.std() / nz.mean():.3f} "
          f"(near 0 -> balanced, e.g. ICE/KR)")
    coo = sparse.triu(M).tocoo()
    s = coo.col - coo.row
    print("     mean value per diagonal, relative to s = 1 (flat ~1 everywhere -> observed/expected;"
          " steady decay -> contact frequencies):")
    diag_mean = lambda d: coo.data[s == d].sum() / (n - d)      # mean over the whole diagonal, zeros included
    ref = diag_mean(1)
    for d in (0, 1, 2, 3, 4, 5, 10, 30, 100, 300, 1000):
        if not np.any(s == d):
            print(f"     s = {d:4d}: no entries")
        else:
            print(f"     s = {d:4d}: {diag_mean(d) / ref:.4g}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        main(p)
