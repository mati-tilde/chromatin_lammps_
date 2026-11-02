#-------------------------------------------------------------------------------------------------------------------------
# contact_maps.py -- simulated contact maps from LAMMPS trajectories (step H2 of the Hi-C plan).
#
# A contact between beads i and j is weighted with the SAME function g(r) used in the energy by pair sigmoid/episig:
#     g(r) = f(r) - f(r_cut) for r < r_cut (pair_modify shift yes), 0 otherwise,
#     f(r) = 1 / (1 + exp(-2 mu (r_c - r)))       form "exp" or "tanh" (the same function),
#     f(r) = 1 / (1 + (r / r_c)^(2n))             form "rational" (n given in the mu slot, as in the pair style).
# Pairs closer than d0 along the same chain are excluded (d0 = 4 <-> |i-j| <= 3 excluded, as with
# "pair_modify pair sigmoid/episig special lj 0 0 0").
#
# The map is accumulated per ATOM (index = atom_id - 1), upper triangle i < j, as a sum over frames.
# Contact probabilities are W / n_frames. Mapping atoms to genomic bins is done separately with the
# bead -> bin table written by writer.py (H0).
#-------------------------------------------------------------------------------------------------------------------------

import hashlib
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
# Reading a LAMMPS text dump frame by frame (any column order; atoms are sorted by id).
# Yields (timestep, ids, positions) with positions an (N,3) array.
def read_dump(path):

    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline())
            fh.readline()                                   # ITEM: NUMBER OF ATOMS
            n = int(fh.readline())
            fh.readline()                                   # ITEM: BOX BOUNDS ...
            for _ in range(3):
                fh.readline()
            cols = fh.readline().split()[2:]                # ITEM: ATOMS id mol type x y z ...
            data = np.array([fh.readline().split() for _ in range(n)], dtype=float)
            ids = data[:, cols.index("id")].astype(int)
            order = np.argsort(ids)
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
            s = line.split("#")[0].strip()
            if not s:
                continue
            w = s.split()
            if w[0][0].isalpha():
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
            rows[int(w[0])] = w[1:]                         # later rows with the same step overwrite earlier ones

    return rows
#----------------------------------------------------------------------------------------------


#------------------------
# 3 -- CONTACTS.
#------------------------

#----------------------------------------------------------------------------------------------
# Contacts of one frame: pairs (i < j) within r_cut, excluding pairs closer than d0 along the same chain.
# positions and mol are sorted by atom id; atoms of a chain are consecutive (checked by check_bins.py),
# so the genomic separation along a chain is |i - j|.
def frame_contacts(positions, mol, d0=4, r_cut=2.5, **weight_kw):

    pairs = cKDTree(positions).query_pairs(r_cut, output_type="ndarray")     # unique pairs, i < j
    i, j = pairs[:, 0], pairs[:, 1]

    keep = (mol[i] != mol[j]) | (j - i >= d0)
    i, j = i[keep], j[keep]

    r = np.linalg.norm(positions[i] - positions[j], axis=1)
    w = contact_weight(r, r_cut=r_cut, **weight_kw)

    return i, j, w
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Statistics of one frame from its contacts: N_c and Phi_ab (a <= b, row order), as printed by compute pair.
def frame_statistics(i, j, w, signals):

    n = signals.shape[1]
    phi = []
    for a in range(n):
        for b in range(a, n):
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

        i, j, w = frame_contacts(pos, mol, d0=d0, r_cut=r_cut, **weight_kw)
        W = W + coo_matrix((w, (i, j)), shape=(n_atoms, n_atoms)).tocsr()
        used.append(step)

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
# 5 -- COMBINING MAPS.
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
# It refuses to combine maps made differently -- compares the .json files.
def combine_maps(paths):

    W_tot, n_tot, ref, sources = None, 0, None, []

    for path in paths:
        W, meta = load_map(path)

        if ref is None:
            ref = meta
            W_tot = W.copy()
        else:
            bad = [k for k in MUST_MATCH if meta.get(k) != ref.get(k)]
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
# 6 -- ATOMS -> GENOMIC BINS.
# Each genomic bin (arm, bin) gets one index.
#-----------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Reading the bead -> bin table written by writer.py (H0); rows are returned in atom-id order.
def read_bin_table(path):

    rows, header = [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            w = line.rstrip("\n").split("\t")
            if header is None:
                header = w
                continue
            rows.append(dict(zip(header, w)))

    rows.sort(key=lambda r: int(r["atom_id"]))
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
    atom_bin = np.array([bin_index[k] for k in keys])
    n_bins = len(bins)

    coo = W.tocoo()
    a, b, w = coo.row, coo.col, coo.data
    ba, bb = atom_bin[a], atom_bin[b]
    lo, hi = np.minimum(ba, bb), np.maximum(ba, bb)

    same_copy = copies[a] == copies[b]
    same_chrom = chroms[a] == chroms[b]
    classes = {"cis": same_copy, "homolog": (~same_copy) & same_chrom, "inter": ~same_chrom}

    maps = {}
    for name, sel in classes.items():
        M = coo_matrix((w[sel], (lo[sel], hi[sel])), shape=(n_bins, n_bins)).tocsr()
        M.sum_duplicates()
        maps[name] = M

    n_copies = {str(c): len(set(copies[chroms == c])) for c in sorted(set(chroms))}

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
