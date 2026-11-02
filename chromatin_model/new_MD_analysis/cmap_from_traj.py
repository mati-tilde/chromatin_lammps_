#!/usr/bin/env python3
#-----------------------------------------------------------------------------------------------------------------
# cmap_from_traj.py -- weighted contact map of one LAMMPS trajectory (H2), with the exact check against
# the statistics printed by pair sigmoid/episig (H4).
#
# Example:
#   python3 cmap_from_traj.py --data 02Nov_chr2_rw_seed42_without_H3K27ac.data --traj H2_test.lammpstrj \
#       --phi H2_test_phi.dat --out H2_test_cmap
# The contact-function options must be the ones of the LAMMPS run (form, r_c, mu, r_cut, d0).
#-----------------------------------------------------------------------------------------------------------------
import argparse
import contact_maps as cm

p = argparse.ArgumentParser(description="Weighted contact map from a LAMMPS trajectory.")
p.add_argument("--data", required=True, help="LAMMPS data file used for the run (molecules and signals).")
p.add_argument("--traj", required=True, help="LAMMPS text dump with id and x y z columns.")
p.add_argument("--phi", help="<prefix>_phi.dat of the same run: enables the exact check (H4).")
p.add_argument("--out", default="contact_map", help="Output name (writes <out>.npz and <out>.json).")
p.add_argument("--form", default="exp", choices=["exp", "tanh", "rational"])
p.add_argument("--r_c", type=float, default=1.78)
p.add_argument("--mu", type=float, default=3.22)
p.add_argument("--r_cut", type=float, default=2.5)
p.add_argument("--d0", type=int, default=4, help="Minimum separation along a chain (4: |i-j| <= 3 excluded).")
p.add_argument("--first-step", type=int, default=0, help="Ignore frames before this timestep (burn-in).")
p.add_argument("--last-step", type=int, default=None, help="Ignore frames after this timestep.")
p.add_argument("--tol", type=float, default=1e-10,
               help="Tolerance of the H4 check. 1e-10 needs a full-precision dump (dump_modify ... format float %%.15g); "
                    "with the default 6-digit dump the agreement is ~1e-6.")
a = p.parse_args()

ids, mol, typ, xyz, sig = cm.read_data(a.data)
weight = dict(form=a.form, r_c=a.r_c, mu=a.mu)
check = cm.read_phi(a.phi) if a.phi else None
if check is not None and sig is None:
    raise SystemExit("The check needs the signals: use the data file with the Signals section.")

frames = ((s, i, x) for s, i, x in cm.read_dump(a.traj)
          if s >= a.first_step and (a.last_step is None or s <= a.last_step))
W, used, worst = cm.accumulate(frames, mol, len(ids), d0=a.d0, r_cut=a.r_cut, check=check, signals=sig, **weight)

meta = dict(data=a.data, traj=a.traj, steps=used, form=a.form, r_c=a.r_c, mu=a.mu, r_cut=a.r_cut, d0=a.d0,
            shift=True, n_atoms=len(ids), topology=cm.topology_fingerprint(mol),
            normalization="sum over frames; probability = W / n_frames", storage="upper triangle, atom index = id - 1")
cm.save_map(a.out, W, len(used), meta)

print(f"\n{len(used)} frames, {W.nnz} pairs with nonzero weight; saved {a.out}.npz and {a.out}.json")
if check is not None:
    print(f"H4 check: largest relative deviation from LAMMPS = {worst:.2e}  ->  {'PASSED' if worst < a.tol else 'FAILED'} (tol {a.tol:g})")
