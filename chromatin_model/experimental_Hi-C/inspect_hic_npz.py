#!/usr/bin/env python3
"""
inspect_hic_npz.py -- print the STRUCTURE of an experimental Hi-C .npz file, without printing any data values.

Usage:  python3 inspect_hic_npz.py file.npz [file2.npz ...]

For each array in the file: name, shape, dtype. For square matrices also: sparse or dense, whether entries are
whole numbers (raw counts) or not (normalized), whether the matrix is symmetric / upper triangular, whether
there are NaN values, and how many rows are entirely empty. Name-like arrays (strings) show their length and
the distinct values only if they look like chromosome names (e.g. chr2L), since those are needed to align bins.
"""
import sys
import numpy as np

CHROM_LIKE = ("chr", "2L", "2R", "3L", "3R", "X", "4", "Y")


def describe_matrix(M, label):
    from scipy import sparse
    is_sparse = sparse.issparse(M)
    n, m = M.shape
    print(f"    {label}: {'sparse' if is_sparse else 'dense'} {n} x {m}")
    if n != m:
        return
    vals = M.data if is_sparse else M[np.isfinite(M)] if np.issubdtype(M.dtype, np.floating) else M.ravel()
    vals = np.asarray(vals)
    print(f"      whole numbers (raw counts?): {bool(np.all(np.mod(vals, 1) == 0)) if vals.size else 'n/a'}")
    if np.issubdtype(np.asarray(vals).dtype, np.floating):
        nan = (np.isnan(M.data).sum() if is_sparse else np.isnan(M).sum())
        print(f"      NaN entries: {int(nan)}")
    if is_sparse:
        coo = M.tocoo()
        upper = bool(np.all(coo.row <= coo.col)); lower = bool(np.all(coo.row >= coo.col))
        sym = (abs(M - M.T) > 1e-9 * (abs(M).max() + 1e-300)).nnz == 0
        empty = int(np.sum(np.diff(M.tocsr().indptr) == 0))
    else:
        A = np.nan_to_num(M)
        upper = bool(np.allclose(np.tril(A, -1), 0)); lower = bool(np.allclose(np.triu(A, 1), 0))
        sym = bool(np.allclose(A, A.T))
        empty = int(np.sum(~np.any(A != 0, axis=1)))
    print(f"      symmetric: {sym}, upper triangular only: {upper and not lower}, lower only: {lower and not upper}")
    print(f"      rows with no contacts at all: {empty}")
    if n in (2352, 2529, 2812, 3208, 2355, 4881, 6020, 13256):
        print(f"      note: {n} bins matches the model's dm6 bead counts (chr2L 2352, chr2R 2529, chr2 4881, ...)")


def main(paths):
    from scipy import sparse
    for path in paths:
        print(f"\n=== {path}")
        z = np.load(path, allow_pickle=False)
        keys = list(z.keys())
        print(f"  keys: {keys}")
        if {"data", "indices", "indptr", "shape"} <= set(keys) or {"row", "col", "data", "shape"} <= set(keys):
            M = sparse.load_npz(path)
            describe_matrix(M.tocsr(), "scipy sparse matrix (whole file)")
            continue
        for k in keys:
            a = z[k]
            if a.dtype.kind in "US":
                distinct = np.unique(a)
                show = [str(v) for v in distinct if any(c in str(v) for c in CHROM_LIKE)]
                print(f"  {k}: {a.dtype} array of length {len(a)}"
                      + (f", chromosome-like values: {show[:20]}" if show and len(distinct) <= 50 else ""))
            elif a.ndim == 2:
                describe_matrix(a, f"{k}")
            else:
                print(f"  {k}: shape {a.shape}, dtype {a.dtype}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
