#!/usr/bin/env python3
"""
check_phi.py -- test A5: independent computation of Phi_ab and N_c (sigmoid/episig) from a data file.

    Phi_ab = sum_{pairs} c_ij^ab g(r_ij),  N_c = sum_{pairs} g(r_ij)
    g(r)   = f(r) - f(rcut) for r < rcut,  f(r) = 1/2 [1 + tanh(mu (rc - r))]
    c_ij^aa = s_ia s_ja ;  c_ij^ab = s_ia s_jb + s_ib s_ja  (a < b)
    pairs  = all i<j with r < rcut, except pairs within --exclude bonds of each other along the chain
             (--exclude 3 matches "special_bonds lj 0.0 0.0 0.0": 1-2, 1-3 and 1-4 neighbours excluded;
              --exclude 1 matches "special_bonds lj 0.0 1.0 1.0": only directly bonded pairs excluded)

Usage:
    python3 check_phi.py DATAFILE [--phi PREFIX_phi.dat] [--C 0.1] [--B 0.25 | --Bmat B11 B12 B13 B22 B23 B33]
If --phi is given, the step-0 line of the LAMMPS output is compared with the Python values.
"""
import argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("data")
p.add_argument("--phi", help="LAMMPS <prefix>_phi.dat to compare with (first line = step 0)")
p.add_argument("--rc", type=float, default=1.78)
p.add_argument("--mu", type=float, default=3.22)
p.add_argument("--rcut", type=float, default=3.0)
p.add_argument("--exclude", type=int, default=3, help="exclude pairs up to this many bonds apart (default 3)")
p.add_argument("--C", type=float, default=0.0)
p.add_argument("--B", type=float, default=None, help="uniform B entry")
p.add_argument("--Bmat", type=float, nargs="+", default=None, help="upper triangle of B, row order")
args = p.parse_args()

# ---- read the data file: Atoms (id mol type x y z ...), Bonds (id type i j), Signals (id s1 ... sn)
sections, current = {"Atoms": [], "Bonds": [], "Signals": []}, None
with open(args.data) as fh:
    for line in fh:
        s = line.split("#")[0].strip()
        if not s:
            continue
        word = s.split()[0]
        if word in sections:
            current = word
            continue
        if word[0].isalpha():          # another section header (Masses, ...)
            current = None
            continue
        if current:
            sections[current].append(s.split())

atoms = np.array(sections["Atoms"], dtype=float)
ids = atoms[:, 0].astype(int)
order = np.argsort(ids)
ids, X = ids[order], atoms[order, 3:6]
index = {aid: k for k, aid in enumerate(ids)}
sig = {int(r[0]): [float(v) for v in r[1:]] for r in sections["Signals"]}
S = np.array([sig[aid] for aid in ids])
n = S.shape[1]
adj = {k: set() for k in range(len(ids))}
for bnd in sections["Bonds"]:
    u, v = index[int(bnd[2])], index[int(bnd[3])]
    adj[u].add(v); adj[v].add(u)
print(f"{len(ids)} atoms, {len(sections['Bonds'])} bonds, {n} signal channels")

def within(i, j, maxd):
    """True if j is at most maxd bonds away from i along the bond graph."""
    front, seen = {i}, {i}
    for _ in range(maxd):
        front = {m for x in front for m in adj[x]} - seen
        if j in front:
            return True
        seen |= front
    return False

f = lambda r: 0.5 * (1.0 + np.tanh(args.mu * (args.rc - r)))
fcut = f(args.rcut)

# ---- all pairs within rcut (chunked to limit memory)
I, J, R = [], [], []
N = len(X)
for a in range(0, N, 500):
    d = np.linalg.norm(X[a:a + 500, None, :] - X[None, :, :], axis=2)
    ii, jj = np.nonzero(d < args.rcut)
    ii = ii + a
    keep = jj > ii
    I.append(ii[keep]); J.append(jj[keep]); R.append(d[ii[keep] - a, jj[keep]])
I, J, R = np.concatenate(I), np.concatenate(J), np.concatenate(R)
keep = np.array([not within(i, j, args.exclude) for i, j in zip(I, J)])
I, J, R = I[keep], J[keep], R[keep]
g = f(R) - fcut
print(f"{len(I)} pairs within rcut = {args.rcut} (pairs up to {args.exclude} bonds apart excluded)")

phi, labels = [], []
for a in range(n):
    for b in range(a, n):
        c = S[I, a] * S[J, a] if a == b else S[I, a] * S[J, b] + S[I, b] * S[J, a]
        phi.append(np.sum(c * g)); labels.append(f"Phi_{a+1}{b+1}")
Nc = np.sum(g)
values = phi + [Nc]
labels.append("N_c")

# ---- energy from the linear form, if B is given
E = None
if args.B is not None or args.Bmat is not None:
    Bup = [args.B] * len(phi) if args.Bmat is None else args.Bmat
    if len(Bup) != len(phi):
        raise SystemExit(f"--Bmat needs {len(phi)} values")
    E = -args.C * Nc - np.dot(Bup, phi)

# ---- compare with LAMMPS
lmp = None
if args.phi:
    with open(args.phi) as fh:
        rows = [l.split() for l in fh if l.strip() and not l.startswith("#")]
    first = [float(v) for v in rows[0]]           # TimeStep c_episig c_episig[1] ... c_episig[k]
    print(f"LAMMPS line: step {int(first[0])}")
    lmp_E, lmp = first[1], first[2:]

print(f"\n{'quantity':>10} {'python':>22} {'lammps':>22} {'rel. diff':>11}")
for k, (lab, v) in enumerate(zip(labels, values)):
    if lmp is None:
        print(f"{lab:>10} {v:22.12g}")
    else:
        rd = abs(v - lmp[k]) / max(abs(v), 1e-300)
        print(f"{lab:>10} {v:22.12g} {lmp[k]:22.12g} {rd:11.2e}")
if E is not None:
    if lmp is None:
        print(f"{'E_sig':>10} {E:22.12g}")
    else:
        print(f"{'E_sig':>10} {E:22.12g} {lmp_E:22.12g} {abs(E - lmp_E) / max(abs(E), 1e-300):11.2e}")
