#-------------------------------------------------------------------------------------------------------------------------
# contact_maps.py -- functions forsimulated contact maps from LAMMPS trajectories.
#
# A contact between beads i and j is weighted with the SAME function g(r) used in the energy by pair sigmoid/episig:
#     g(r) = f(r) - f(r_cut) for r < r_cut (pair_modify shift yes), 0 otherwise,
#     f(r) = 1 / (1 + exp(-2 mu (r_c - r)))       form "exp" or "tanh" (the same function),
#     f(r) = 1 / (1 + (r / r_c)^(2n))             form "rational" (n given in the mu slot, as in the pair style),
#     f(r) = theta(r_c - r)                        form "step" (hard cutoff at r_c; NOT the energy's function).
# Pairs closer than d0 along the same chain are excluded (d0 = 4 <-> |i-j| <= 3 excluded, as with
# "pair_modify pair sigmoid/episig special lj 0 0 0").
#
# The map is accumulated per ATOM (index = atom_id - 1), upper triangle i < j, as a sum over frames.
# Contact probabilities are W / n_frames. Mapping atoms to genomic bins is done separately with the
# bead -> bin table written by writer.py.
#-------------------------------------------------------------------------------------------------------------------------

import hashlib          # functions to calculate hash values of data.
import json
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz
from scipy.spatial import cKDTree


#------------------------
# 1 -- CONTACT FUNCTION.
#------------------------

#----------------------------------------------------------------------------------------------
# Switching function f(r), identical to pair sigmoid/episig.
def switch_f(r, form="exp", r_c=1.78, mu=3.22):

    r = np.asarray(r, dtype=float)

    if form in ("exp", "tanh"):
        return 1.0 / (1.0 + np.exp(-2.0 * mu * (r_c - r)))

    # hard contact: 1 if r < r_c, else 0 (mu not used);
    # not the same energy function as the simulation: cannot be checked against LAMMPS.
    if form == "step":
        return (r < r_c).astype(float)                  

    if form == "rational":
        n = int(round(mu))
        if n < 1 or abs(mu - n) > 1e-12:
            raise ValueError("form rational: mu must be a positive integer n")
        return 1.0 / (1.0 + (r / r_c) ** (2 * n))

    raise ValueError(f"Unknown form {form}")
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Contact weight g(r) = f(r) - f(r_cut) inside the cutoff (shift = True), as in the energy.
def contact_weight(r, form="exp", r_c=1.78, mu=3.22, r_cut=2.5, shift=True):

    g = switch_f(r, form, r_c, mu)
    if shift:
        g = g - switch_f(r_cut, form, r_c, mu)

    return np.where(np.asarray(r) < r_cut, g, 0.0)
#----------------------------------------------------------------------------------------------


#------------------------
# 2 -- INPUT FILES.
#------------------------

#----------------------------------------------------------------------------------------------
# read_dump is a generator: reads one frame at a time from a LAMMPS text dump file, yielding (step, ids, positions),
# with positions an (N,3) array. It is memory-efficient: it never loads the whole trajectory.
def read_dump(path):

    with open(path) as fh:                                  # fh is the file handle, an iterator over the lines of the file
        while True:
            line = fh.readline()                            # fh.readline() reads the next line from the file
            if not line:
                return
            # finding the beginning of a frame:
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            # reading timestep and number of atoms:
            step = int(fh.readline())
            fh.readline()                                   # ITEM: NUMBER OF ATOMS
            n = int(fh.readline())
            # skipping the box bounds:
            fh.readline()                                   # ITEM: BOX BOUNDS ...
            for _ in range(3):                              # _ when the variable is not used
                fh.readline()
            # reading the atom data: columns are given in the header line, but we only need id, x, y, z.
            cols = fh.readline().split()[2:]                # ITEM: ATOMS id mol type x y z ...; splitting line into a list, starting from the 3rd element (0-based indexing).
            data = np.array([fh.readline().split() for _ in range(n)], dtype=float)
            ids = data[:, cols.index("id")].astype(int)     # data[:, j] -- every row, column j. 
            order = np.argsort(ids)                         # returns the indices that would sort the array.
            xyz = data[order][:, [cols.index("x"), cols.index("y"), cols.index("z")]]
            yield step, ids[order], xyz
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Reading the Atoms and Signals sections of a LAMMPS data file.
# Returns ids (sorted), molecule ids, types, positions and signals (None if absent), all sorted by id.
def read_data(path):

    atoms, signals, sec = {}, {}, None
    with open(path) as fh:
        for line in fh:
            s = line.split("#")[0].strip()          # remove comments and whitespace from the line
            if not s:                               # if the line is empty or contains only a comment, s becomes an empty string.
                continue
            w = s.split()                           # splitting line into fields (words) based on whitespace.
            if w[0][0].isalpha():                   # if first character == letter --> section header.
                sec = w[0]
                continue
            if sec == "Atoms":
                atoms[int(w[0])] = (int(w[1]), int(w[2]), float(w[3]), float(w[4]), float(w[5]))
            elif sec == "Signals":
                signals[int(w[0])] = [float(v) for v in w[1:]]

    ids = np.array(sorted(atoms))
    mol = np.array([atoms[i][0] for i in ids])
    typ = np.array([atoms[i][1] for i in ids])
    xyz = np.array([atoms[i][2:] for i in ids])
    sig = np.array([signals[i] for i in ids]) if signals else None

    return ids, mol, typ, xyz, sig
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Reading the _phi.dat file written by fix ave/time: {timestep: [E_sig, Phi_11, ..., N_c]}.
def read_phi(path):

    rows = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            w = [float(v) for v in line.split()]
            rows[int(w[0])] = w[1:]                         # w[0] is the timestep, w[1:] is the list of values (E_sig, Phi_11, ..., N_c) for that timestep.

    return rows
#-----------------------------------------------------------------------------------------------


#------------------------
# 3 -- CONTACTS.
#------------------------

#----------------------------------------------------------------------------------------------
# Contacts of one frame: pairs (i < j) within r_cut, excluding pairs closer than d0 along the same chain.
# positions and mol are sorted by atom id; atoms of a chain are consecutive (checked by check_bins.py),
# so the genomic separation along a chain is |i - j|.
def frame_contacts(positions, mol, d0=4, r_cut=2.5, **weight_kw):

    # using a spatial search (cKDTree - from scipy.spatial) to find pairs within r_cut, 
    # instead of calculating the distance between all pairs (O(N^2)).
    pairs = cKDTree(positions).query_pairs(r_cut, output_type="ndarray")    # method .query_pairs: returns all unique pairs (i<j) of points whose distance is less than or equal to r_cut
    i, j = pairs[:, 0], pairs[:, 1]                                         # extracts the two columns of pairs array.
                                                                            # so, i[k] and j[k] identify the two beads of contact pair k.  

    # a pair is kept if at least one option is satisfied: in different molecules OR at least d0 beads apart.
    # NOTE: this assumes that the beads of a chain are consecutive in the atom list (checked by check_bins.py)!
    keep = (mol[i] != mol[j]) | (j - i >= d0)
    i, j = i[keep], j[keep]

    r = np.linalg.norm(positions[i] - positions[j], axis=1)                 # calculating the euclidean distance for each kept pair.
    w = contact_weight(r, r_cut=r_cut, **weight_kw)                         # then assign a weight to each contact.

    return i, j, w
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Statistics of one frame from its contacts: N_c and Phi_ab (a <= b, row order), as printed by compute pair.
def frame_statistics(i, j, w, signals):

    n = signals.shape[1]                    # number of signals (columns) in the signals array.
    phi = []
    # explore all pairs of signals with a <= b (upper triangle).
    for a in range(n):
        for b in range(a, n):
            # same signal: Phi_aa = sum(w * s_i * s_j) for all contacts (i,j) in the frame.
            if a == b:
                c = signals[i, a] * signals[j, a]
            else:
                c = signals[i, a] * signals[j, b] + signals[i, b] * signals[j, a]
            phi.append(np.sum(c * w))

    return np.array(phi), np.sum(w)
#----------------------------------------------------------------------------------------------


#------------------------
# 4 -- ACCUMULATION.
#------------------------

#----------------------------------------------------------------------------------------------
# Accumulates the weighted contact map over the frames of one trajectory.
# frames: iterable of (step, ids, positions); returns W (csr, upper triangle), the steps used, and,
# if check is given ({step: [E, Phi..., N_c]} and signals), the largest relative deviation between
# the statistics of the map and those printed by LAMMPS (test H4).
def accumulate(frames, mol, n_atoms, d0=4, r_cut=2.5, steps=None, check=None, signals=None, **weight_kw):

    W = csr_matrix((n_atoms, n_atoms), dtype=float)
    used, worst = [], 0.0

    for step, ids, pos in frames:

        if steps is not None and step not in steps:
            continue
        if len(ids) != n_atoms or ids[0] != 1 or ids[-1] != n_atoms:
            raise ValueError(f"step {step}: atom ids do not match the data file")

        # storing only the upper triangle - contact map is symmetric by definition!
        # contacts in the current frame:
        i, j, w = frame_contacts(pos, mol, d0=d0, r_cut=r_cut, **weight_kw)
        # adding the contacts of the current frame to the accumulated contact map W (sparse matrix in COO format):
        W = W + coo_matrix((w, (i, j)), shape=(n_atoms, n_atoms)).tocsr()       # converting COO to CSR matrix 
        # recording the timestep of the current frame in the list of used steps:
        used.append(step)

        # comparison with lammps to obtain largest relative deviation (if check is given):
        if check is not None:
            if step not in check:
                print(f"  step {step}: no line in the phi file, not checked")
            else:
                phi, nc = frame_statistics(i, j, w, signals)
                ref = np.array(check[step][1:])                 # Phi_ab ..., N_c
                mine = np.append(phi, nc)
                dev = np.max(np.abs(mine - ref) / np.abs(ref))
                worst = max(worst, dev)
                print(f"  step {step}: N_c = {nc:.10g} (LAMMPS {ref[-1]:.10g}), max rel. deviation {dev:.2e}")

    W.sum_duplicates()
    return W, used, worst
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Saving / loading a map with its metadata (contact function, exclusion, frames, files, ...).
def save_map(path, W, n_frames, meta):
    # <path>.npz: compressed sparse matrix in CSR format; <path>.json: metadata + n_frames.
    save_npz(path if path.endswith(".npz") else path + ".npz", W.tocsr(), compressed=True)
    meta = dict(meta, n_frames=int(n_frames))
    with open((path[:-4] if path.endswith(".npz") else path) + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)


def load_map(path):
    base = path[:-4] if path.endswith(".npz") else path
    W = load_npz(base + ".npz").tocsr()
    with open(base + ".json") as fh:
        meta = json.load(fh)
    return W, meta
#----------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------
# 5 -- COMBINING MAPS (H3).
#-----------------------------------------------------------------------------------------------------------------------

# Settings that must be identical for two maps to be combined.
MUST_MATCH = ("form", "r_c", "mu", "r_cut", "d0", "shift", "n_atoms", "topology")

#----------------------------------------------------------------------------------------------
# Fingerprint of the topology (number of atoms and molecule of each atom): replicas from different
# data files (different seeds) have the same topology and the same fingerprint.
def topology_fingerprint(mol):

    mol = np.asarray(mol, dtype=np.int64)
    return hashlib.sha1(mol.tobytes()).hexdigest()[:16]
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Combining maps of replicas (or of pieces of one run): sums of W AND of frame counts.
# The combined probability is sum(W) / sum(n_frames), i.e. every frame has the same weight.
def combine_maps(paths):

    W_tot, n_tot, ref, sources = None, 0, None, []

    for path in paths:
        W, meta = load_map(path)

        if ref is None:
            ref = meta
            W_tot = W.copy()
        else:
            bad = [k for k in MUST_MATCH if meta.get(k) != ref.get(k)]
            # not allowing sum of maps with different characteristics:
            if bad or W.shape != W_tot.shape:
                raise ValueError(f"{path}: cannot be combined with {paths[0]} (different {bad or 'shape'})")
            W_tot = W_tot + W

        n_tot += meta["n_frames"]
        sources.append({"map": path, "traj": meta.get("traj"), "data": meta.get("data"), "n_frames": meta["n_frames"]})

    meta = {k: ref.get(k) for k in MUST_MATCH}
    meta.update(sources=sources, normalization="sum over all frames of all sources; probability = W / n_frames",
                storage=ref.get("storage"))
    W_tot.sum_duplicates()

    return W_tot, n_tot, meta
#----------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------
# 6 -- ATOMS -> GENOMIC BINS (H3).
#-----------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Reading the bead -> bin table written by writer.py; rows are returned in atom-id order.
def read_bin_table(path):

    rows, header = [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            w = line.rstrip("\n").split("\t")       # splitting line into fields (words) based on tab character.
            if header is None:
                header = w
                continue
            rows.append(dict(zip(header, w)))       # converting rows into dictionary.

    rows.sort(key=lambda r: int(r["atom_id"]))      # sorting the rows by atom_id (as integer).
    if [int(r["atom_id"]) for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"{path}: atom ids are not 1..N")

    return rows
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Converting a per-atom map (upper triangle) into maps on genomic bins.
# Each genomic bin (arm, bin) gets one index, in the order of the first copy of each chromosome.
# Every atom pair falls in exactly one class:
#   "cis"     -- both atoms in the same copy (summed over copies; per copy: divide by n_copies),
#   "homolog" -- atoms in the two copies of the same chromosome (A-B pairs; also same bin = pairing),
#   "inter"   -- atoms in different chromosomes.
# The binned matrices are upper triangular (bin_i <= bin_j); the total weight is conserved.
def atoms_to_bins(W, table):

    # extracting information from the table: copy number, chromosome name, and genomic bin for each atom:
    copies = np.array([r["copy"] for r in table])
    chroms = np.array([r["chromosome"] for r in table])
    keys = [(r["arm"], int(r["bin"])) for r in table]

    # one index per genomic bin, in order of first appearance:
    bin_index, bins = {}, []
    for r, key in zip(table, keys):
        if key not in bin_index:
            bin_index[key] = len(bins)
            bins.append({"arm": r["arm"], "bin": int(r["bin"]), "start": int(r["start"]), "end": int(r["end"]),
                         "in_bed": int(r["in_bed"]), "compartment": r["compartment"]})
    # mapping every atom to its genomic bin index:
    atom_bin = np.array([bin_index[k] for k in keys])
    n_bins = len(bins)

    # extracting non-zero contacts from the sparse matrix W (upper triangle):
    coo = W.tocoo()
    a, b, w = coo.row, coo.col, coo.data
    # convert atom indices to bin indices, and sort each pair so that the lower index is first:
    ba, bb = atom_bin[a], atom_bin[b]
    lo, hi = np.minimum(ba, bb), np.maximum(ba, bb)

    # classifying each contact pair:
    same_copy = copies[a] == copies[b]          # are the two atoms from the same chromosome copy?
    same_chrom = chroms[a] == chroms[b]         # are the two atoms from the same chromosome (but possibly different copies)?
    classes = {"cis": same_copy, "homolog": (~same_copy) & same_chrom, "inter": ~same_chrom}

    # actual aggregation: for each class, sparse matrix of n_bins x n_bins, with weights w for the corresponding pairs (lo, hi).
    maps = {}
    # classes are kept separate so that the treatment of homolog pairing in the comparison with Hi-C stays an explicit choice.
    for name, sel in classes.items():
        M = coo_matrix((w[sel], (lo[sel], hi[sel])), shape=(n_bins, n_bins)).tocsr()
        M.sum_duplicates()
        maps[name] = M

    # how many copies for each chromosome type:
    n_copies = {str(c): len(set(copies[chroms == c])) for c in sorted(set(chroms))}

    # returning class maps, bin list, and copies for chromosome:
    return maps, bins, n_copies
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Saving binned maps: <out>_<class>.npz for each class, <out>_bins.tsv (bin list, with in_bed for masking),
# <out>_binned.json (metadata). Probabilities: cis / (n_frames * copies of that chromosome); homolog and
# inter / n_frames.
def save_binned(out, maps, bins, n_frames, meta, n_copies):

    for name, M in maps.items():
        save_npz(f"{out}_{name}.npz", M, compressed=True)

    with open(f"{out}_bins.tsv", "w") as fh:
        fh.write("index\tarm\tbin\tstart\tend\tin_bed\tcompartment\n")
        for k, b in enumerate(bins):
            fh.write(f"{k}\t{b['arm']}\t{b['bin']}\t{b['start']}\t{b['end']}\t{b['in_bed']}\t{b['compartment']}\n")

    meta = dict(meta, n_frames=int(n_frames), n_bins=len(bins), copies_per_chromosome=n_copies,
                classes={k: float(M.sum()) for k, M in maps.items()},
                probability="cis: W / (n_frames * copies); homolog, inter: W / n_frames",
                storage_binned="upper triangle on genomic bins (index as in _bins.tsv)")
    with open(f"{out}_binned.json", "w") as fh:
        json.dump(meta, fh, indent=2)
#----------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------
# 7 -- EXPERIMENTAL MAPS (H6).
#-----------------------------------------------------------------------------------------------------------------------

# Bin layouts of the genome-wide experimental matrices (dm6, 10 kb, order of the BED file).
# The merged map has all chromosomes (13758 bins); the replicates have no chrY (13391 bins).
EXP_LAYOUT = (("chr2L", 2352), ("chr2R", 2529), ("chr3L", 2812), ("chr3R", 3208), ("chr4", 135), ("chrX", 2355), ("chrY", 367))
EXP_LAYOUT_NO_Y = EXP_LAYOUT[:-1]
KNOWN_LAYOUTS = {sum(s for _, s in L): L for L in (EXP_LAYOUT, EXP_LAYOUT_NO_Y)}

#----------------------------------------------------------------------------------------------
# Experimental layout matching a matrix of size n x n.
def layout_for(n):
    if n not in KNOWN_LAYOUTS:
        raise ValueError(f"matrix size {n} matches no known layout ({sorted(KNOWN_LAYOUTS)})")
    return KNOWN_LAYOUTS[n]
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Genomic bins of the model, in the order used by atoms_to_bins (first copy of each chromosome).
def genomic_bins(table):

    seen, bins = set(), []
    for r in table:
        key = (r["arm"], int(r["bin"]))
        if key not in seen:
            seen.add(key)
            bins.append({"arm": r["arm"], "bin": int(r["bin"]), "start": int(r["start"]), "end": int(r["end"]),
                         "in_bed": int(r["in_bed"]), "compartment": r["compartment"]})
    return bins
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Index of each model bin in the genome-wide experimental matrix.
def exp_indices(bins, layout=EXP_LAYOUT):

    offset, start = {}, 0
    for name, size in layout:
        offset[name] = start
        start += size
    sizes = dict(layout)
    for b in bins:
        if b["bin"] >= sizes[b["arm"]]:
            raise ValueError(f"bin {b['arm']}:{b['bin']} outside the experimental layout")
    return np.array([offset[b["arm"]] + b["bin"] for b in bins])
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Experimental map restricted to the model bins (symmetric, model bin order).
# The layout (with or without chrY) is chosen from the size of the matrix.
def extract_region(path, bins):

    M = load_npz(path).tocsr()
    if M.shape[0] != M.shape[1]:
        raise ValueError(f"{path}: matrix is not square {M.shape}")
    idx = exp_indices(bins, layout_for(M.shape[0]))
    R = M[idx][:, idx].tocsr()
    R.eliminate_zeros()
    return R
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Keeping only pixels with |i - j| >= d0 between two valid bins, as an upper triangle (i < j).
def valid_upper(R, valid, d0=4):

    coo = R.tocoo()
    keep = (coo.col - coo.row >= d0) & valid[coo.row] & valid[coo.col]
    return coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=R.shape).tocsr()
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Mean of an upper-triangular map along diagonal s, over the pixels between valid bins (zeros included).
def diagonal_means(U, valid, s_max=None):

    n = U.shape[0]
    s_max = n - 1 if s_max is None else s_max
    coo = U.tocoo()
    s = coo.col - coo.row
    sums = np.bincount(s, weights=coo.data, minlength=n)[: s_max + 1]
    v = valid.astype(np.int64)
    counts = np.array([np.sum(v[: n - d] * v[d:]) for d in range(s_max + 1)])    # valid pixel pairs per diagonal
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan), counts
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Scale factor that sets the mean contact probability of adjacent bins (s = 1, valid pairs) to 1.
def scale_s1(R, valid):

    coo = R.tocoo()
    # selecting adjacent valid bins (boolean mask):
    sel = (coo.col - coo.row == 1) & valid[coo.row] & valid[coo.col]
    # counting valid consecutive pairs:
    n_pairs = np.sum(valid[:-1] & valid[1:])
    return n_pairs / coo.data[sel].sum()
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Experimental target map: region, masks, excluded diagonals, scale.
# Masked bins: empty in the target map or in any of the maps in mask_paths (the replicates: a bin filtered out
# in one replicate would otherwise count as zeros and inflate the variance), and optionally absent from the BED.
# Returns P (upper triangle, |i-j| >= d0, valid bins only), the boolean valid vector and a summary dict.
def experimental_map(path, bins, d0=4, scale="s1", mask_not_in_bed=False, mask_paths=()):

    R = extract_region(path, bins)
    nonempty_target = np.asarray(R.sum(axis=1)).ravel() > 0         # whether bin has any contact weight in the target experimental map
    nonempty = nonempty_target.copy()
    # bin is retained only if it has contacts in the target map and in every additional map:
    for other in mask_paths:
        nonempty &= np.asarray(extract_region(other, bins).sum(axis=1)).ravel() > 0
    in_bed = np.array([b["in_bed"] == 1 for b in bins])
    valid = nonempty & (in_bed if mask_not_in_bed else True)

    factor = scale_s1(R, valid) if scale == "s1" else 1.0
    # applying mask and scaling to the upper triangle of the experimental map:
    P = valid_upper(R, valid, d0) * factor

    # summary and diagnostic information:
    summary = dict(n_bins=len(bins), empty_in_target=int(np.sum(~nonempty_target)),
                   empty_in_experiment=int(np.sum(~nonempty)),
                   not_in_bed=int(np.sum(~in_bed)), empty_and_not_in_bed=int(np.sum(~nonempty & ~in_bed)),
                   valid_bins=int(np.sum(valid)), scale=scale, scale_factor=float(factor), d0=d0,
                   mask_not_in_bed=bool(mask_not_in_bed))
    return P, valid, summary
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Variance across replicates and sigma^2(s).
# Each replicate is extracted and scaled exactly like the target (same bins, same valid mask, s = 1 scaling).
# var_ij is the unbiased variance across the R replicates (missing pixels count as zeros);
# sigma2_s[s] is its mean over the valid pixels of diagonal s.
# The uncertainty used in the chi^2 is sigma2_ij = lam * var_ij + (1 - lam) * sigma2_s[|i-j|]  (lam = 0: sigma2(s) only).
def replicate_variance(paths, bins, valid, d0=4, scale="s1"):

    S1 = S2 = None
    # processing each replicate: extracting the region, scaling, masking, and accumulating the sum and sum of squares.
    for path in paths:
        R = extract_region(path, bins)
        factor = scale_s1(R, valid) if scale == "s1" else 1.0
        X = valid_upper(R, valid, d0) * factor
        # sum of X over replicates, and sum of X^2 (elementwise) over replicates:
        S1 = X if S1 is None else S1 + X
        S2 = X.multiply(X) if S2 is None else S2 + X.multiply(X)

    # checking the number of replicates:
    n_rep = len(paths)
    if n_rep < 2:
        raise ValueError("at least two replicates are needed")

    # calculating mean contact map, sample variance map,
    mean = S1 / n_rep
    var = (S2 - S1.multiply(S1) / n_rep) / (n_rep - 1)
    # numerical rounding: 
    var = var.tocsr()
    var.data = np.maximum(var.data, 0.0)                      # remove -0 from rounding
    var.eliminate_zeros()

    sigma2_s, counts = diagonal_means(var, valid)
    return var, sigma2_s, mean.tocsr(), counts
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# sigma^2 for given pixels (arrays i < j), with shrinkage parameter lam.
def sigma2_pixels(i, j, var, sigma2_s, lam=0.0):

    s = j - i
    local = np.asarray(var[i, j]).ravel() if lam > 0 else 0.0
    return lam * local + (1.0 - lam) * sigma2_s[s]
#----------------------------------------------------------------------------------------------
