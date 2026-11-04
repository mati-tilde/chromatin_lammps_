#-------------------------------------------------------------------------------------------------------------------------
# reweighting.py -- Boltzmann reweighting of a sigmoid/episig ensemble to new parameters (B, C).
#
# The snapshots were simulated with (B_ref, C_ref). The energy is linear in B and C:
#     E[s] = - C * N_c[s] - sum_{a<=b} B_ab * Phi_ab[s]
# so, for new parameters (B, C), the energy change of snapshot s is
#     dE[s] = - (C - C_ref) * N_c[s] - sum_{a<=b} (B_ab - B_ref_ab) * Phi_ab[s]
# and the snapshot weights are
#     w[s] = exp(-dE[s] / kT) / sum_t exp(-dE[t] / kT).
# The reweighted contact map is
#     P_ij = sum_s w[s] * g(r_ij[s]),
# where g(r) is the contact weight of contact_maps.py: the same sigmoid g(r) used in the energy.
# With w[s] = 1 / n_frames (B = B_ref, C = C_ref), P is exactly W / n_frames of cmap_from_traj.py.
#
# All file reading, contact detection and the per-frame check use the functions of contact_maps.py.
#-------------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy.sparse import coo_matrix
from scipy.special import logsumexp
import contact_maps as cm

#----------------------------------------------------------------------------------------------
# Symmetric B -> vector in the order of the pair style (tri_index) and of the _phi.dat columns:
# 11 12 13 22 23 33 (row-major upper triangle).
def B_to_vector(B):

    B = np.asarray(B, dtype=float)
    if not np.array_equal(B, B.T):
        raise ValueError("B must be symmetric")

    n = B.shape[0]
    return np.array([B[a, b] for a in range(n) for b in range(a, n)])
#----------------------------------------------------------------------------------------------

class Reweighter:
    """Snapshots simulated at (B_ref, C_ref), reweighted to any (B, C).

    data_file   : LAMMPS data file of the run (molecule ids and signals).
    traj_files  : list of LAMMPS dump files, one per replica.
    phi_files   : list of _phi.dat files, same order as traj_files.
    B_ref, C_ref: parameters used in the simulations.
    form, r_c, mu, r_cut : contact function, the SAME as in the LAMMPS run.
    d0          : pairs closer than d0 along a chain are excluded (4 <-> |i-j| <= 3).
    kT          : thermal energy (LJ units: the temperature).
    first_step, last_step : frames outside this range are ignored.
    check_tol   : the statistics of each frame (Phi_ab, N_c), recomputed from the contacts, must
                  match the _phi.dat file within this relative tolerance. This catches
                  contact-function parameters that differ from the run. 
    """

    def __init__(self, data_file, traj_files, phi_files, B_ref, C_ref,
                 form, r_c, mu, r_cut, d0=4, kT=1.0, first_step=0, last_step=None, check_tol=1e-4):

        # Topology and signals from the data file (sorted by atom id).
        ids, self.mol, _, _, self.signals = cm.read_data(data_file)
        self.n_atoms = len(ids)

        # Settings, kept for reference.
        self.b_ref = B_to_vector(B_ref)
        self.C_ref = float(C_ref)
        self.kT = float(kT)
        self.weight_kw = dict(form=form, r_c=r_c, mu=mu)
        self.r_cut, self.d0 = r_cut, d0

        self.contacts = []          # one sparse map per snapshot: g(r_ij) for i < j
        self.steps = []             # (replica index, timestep) of each snapshot
        E, Phi, Nc = [], [], []
        self.worst_deviation = 0.0  # largest relative deviation of the H4 check

        for rep, (traj_file, phi_file) in enumerate(zip(traj_files, phi_files)):

            phi_rows = cm.read_phi(phi_file)
            seen = set()

            for step, ids_frame, positions in cm.read_dump(traj_file):

                # Keep only the requested time window.
                if step < first_step or (last_step is not None and step > last_step):
                    continue

                # A repeated timestep would count the same configuration twice.
                if step in seen:
                    raise ValueError(f"{traj_file}: timestep {step} appears twice")
                seen.add(step)

                # The dump must contain the atoms of the data file, ids 1..N.
                if len(ids_frame) != self.n_atoms or ids_frame[0] != 1 or ids_frame[-1] != self.n_atoms:
                    raise ValueError(f"{traj_file}, step {step}: atom ids do not match {data_file}")

                if step not in phi_rows:
                    raise ValueError(f"{traj_file}, step {step}: no line in {phi_file}")
                row = phi_rows[step]                  # [E_sig, Phi_11, ..., Phi_33, N_c]

                # Contacts of this frame, with the energy's weight g(r).
                i, j, w = cm.frame_contacts(positions, self.mol, d0=d0, r_cut=r_cut, **self.weight_kw)
                self.contacts.append(coo_matrix((w, (i, j)), shape=(self.n_atoms, self.n_atoms)).tocsr())

                # H4 check: statistics from the contacts vs. statistics printed by LAMMPS.
                phi_mine, nc_mine = cm.frame_statistics(i, j, w, self.signals)
                mine = np.append(phi_mine, nc_mine)
                ref = np.array(row[1:])
                self.worst_deviation = max(self.worst_deviation, np.max(np.abs(mine - ref) / np.abs(ref)))

                self.steps.append((rep, step))
                E.append(row[0])
                Phi.append(row[1:-1])
                Nc.append(row[-1])

        if not self.steps:
            raise ValueError("no frames selected")

        self.E_ref = np.array(E)          # E_sig of each snapshot
        self.Phi = np.array(Phi)          # shape (n_snapshots, n_upper_triangle)
        self.Nc = np.array(Nc)            # N_c of each snapshot

        # The contact function must be the one of the run.
        if self.worst_deviation > check_tol:
            raise ValueError(f"contacts do not reproduce the _phi.dat statistics (max rel. deviation "
                             f"{self.worst_deviation:.2e} > {check_tol:g}): check form, r_c, mu, r_cut, d0")

        # The reference parameters must reproduce the energies written by LAMMPS.
        E_check = -self.C_ref * self.Nc - self.Phi @ self.b_ref
        if not np.allclose(E_check, self.E_ref, rtol=1e-9, atol=1e-9):
            raise ValueError("B_ref, C_ref do not reproduce E_sig in the _phi.dat files")

        print(f"{len(self.steps)} snapshots from {len(traj_files)} replica(s), {self.n_atoms} atoms; "
              f"H4 check: max rel. deviation {self.worst_deviation:.1e}")

    #------------------------------------------------------------------------------------------
    # Energy change of each snapshot: dE[s] = E_new[s] - E_ref[s].
    def delta_energy(self, B, C):

        db = B_to_vector(B) - self.b_ref
        dC = float(C) - self.C_ref
        return -dC * self.Nc - self.Phi @ db

    #------------------------------------------------------------------------------------------
    # Normalized Boltzmann weights (sum to 1). logsumexp avoids overflow.
    def weights(self, B, C):

        log_w = -self.delta_energy(B, C) / self.kT
        return np.exp(log_w - logsumexp(log_w))

    #------------------------------------------------------------------------------------------
    # Effective sample size (Kish), in number of snapshots. Equal to n_snapshots for B = B_ref, C = C_ref.
    def ess(self, B, C):

        w = self.weights(B, C)
        return 1.0 / np.sum(w ** 2)

    #------------------------------------------------------------------------------------------
    # Reweighted per-atom map: P = sum_s w[s] * contacts[s]. Upper triangle, csr.
    def contact_map(self, B, C):

        w = self.weights(B, C)
        P = sum(w_s * c_s for w_s, c_s in zip(w, self.contacts))
        P.sum_duplicates()
        return P

    #------------------------------------------------------------------------------------------
    # Reweighted maps on genomic bins, with the bead -> bin table of writer.py (contact_maps.atoms_to_bins).
    # Returns ({"cis": ..., "homolog": ..., "inter": ...}, bins, n_copies).
    # "cis" is summed over the copies of each chromosome (see contact_maps.save_binned).
    def binned_maps(self, B, C, table):

        if len(table) != self.n_atoms:
            raise ValueError(f"the bin table has {len(table)} atoms, the data file {self.n_atoms}")
        return cm.atoms_to_bins(self.contact_map(B, C), table)
