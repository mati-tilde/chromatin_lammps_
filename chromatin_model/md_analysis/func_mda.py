#-------------------------------------------------------------------------------------------------------------------------
# RELEVANT ANNOTATIONS:
# Here are all the functions needed in script_mda.py and in ensemble_mda.py for simulation and operations on contact maps.
# 1) UTILITY FUNCTIONS.
# 2) FUNCTIONS FOR CONTACT MAP.
# 3) FUNCTIONS FOR ENSEMBLE.
#-------------------------------------------------------------------------------------------------------------------------

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

from MDAnalysis.analysis import distances as mda_dist
from scipy.sparse import csr_matrix, save_npz, load_npz


#------------------------
# 1 -- UTILITY FUNCTIONS.
#------------------------

#----------------------------------------------------------
# Saving a contact map created through accumulate_contacts.
def save_cmap(cmap: csr_matrix, f_out: str):

    save_npz(f'{f_out}.npz', cmap.tocsr(), compressed=True)
#----------------------------------------------------------

#----------------------------------------------------------
# Loading a previously saved contact map.
def load_cmap(f_out: str):

    return load_npz(f_out).tocsr()
#----------------------------------------------------------



#--------------------------------
# 2 -- FUNCTIONS FOR CONTACT MAP.
#--------------------------------

#---------------------------------------------------------------------------------------------------------
# Loading MDA universe (topology file, trajectory file).
def load_universe(topology_file: str, traj_file: str):

    return mda.Universe(topology_file,traj_file,format="LAMMPSDUMP")			# format="LAMMPSDUMP" because .lammpstrj is not supported otherwise.
#---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
# Taking a generic matrix and return it symmetric (for physical relevance).
def symmetric_form(mat: csr_matrix):

    sym_mat = ((mat + mat.T) > 0)
    mat.sum_duplicates()
    sym_mat.setdiag(0)
    sym_mat.eliminate_zeros()

    return sym_mat
#---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
# Accumulating contacts from different frames (given from terminal) to create a single contact map.
def accumulate_contacts(universe, atom_range, frame_range, genome_length, cut):

    # selecting the requested atoms, for both copies in the file:
    start, stop = atom_range
    A_atoms = universe.select_atoms(f'index {start}:{stop - 1}')
    B_atoms = universe.select_atoms(f'index {start + genome_length}:{stop + genome_length -1}')

    # selecting the requested frames:
    if frame_range is None:
        f_start = 0
        f_stop = len(universe.trajectory)
        f_stride = 1
    else:
        f_start, f_stop, f_stride = frame_range

    f_stop = min(f_stop, len(universe.trajectory))		# safety check

    # initializing variables
    acc = None
    num_frames = 0

    # looping over the frames indicated from terminal
    for traj in universe.trajectory[f_start:f_stop:f_stride]:
        if traj.frame % 100 == 0:
            print(f"Frame: {traj.frame}")

        # first copy (A_atoms)
        A_coords = A_atoms.positions
        mat = mda_dist.contact_matrix(A_coords, cutoff=cut, returntype="sparse", box=None)
        mat = mat.tocsr().astype(np.float32, copy=False)

        if acc is None:
            acc = mat.copy()				# use mat.copy() and not just mat to ensure acc and mat are independent (avoid bugs)
        else:
            acc += mat

        # second copy (B_atoms)
        B_coords = B_atoms.positions
        mat = mda_dist.contact_matrix(B_coords, cutoff=cut, returntype="sparse", box=None)
        mat = mat.tocsr().astype(np.float32, copy=False)

        acc += mat

        num_frames += 1

    acc = symmetric_form(acc)				# computing it just once at the end because symmetrization and sum are commutative; gives a more efficient code.
    acc.sum_suplicates()
    acc.eliminate_zeros()

    return acc, num_frames
#---------------------------------------------------------------------------------------------------------



#-----------------------------
# 3 -- FUNCTIONS FOR ENSEMBLE.
#-----------------------------

#------------------------------------------------------------------------------
# Summing contact maps together (assuming they deal with same number of atoms).
def sum_cmaps(f_list):

    # Initializing variables
    acc = None
    expected_shape = None
    used_files = 0
    skipped = []

    # Looping over the file list
    for file in f_list:

        # Opening one matrix at a time, checking correct shape and summing
        try:
            mat = load_cmap(file)
        except FileNotFoundError:
            print(f"File {file} not found -- skipping.")
            continue

        if expected_shape is None:
            expected_shape = mat.shape
        elif mat.shape != expected_shape:
            print(f"File {file} has cmap with shape {mat.shape}, but expected shape is {expected_shape}! Discarding matrix.")
            skipped.append(file)
            continue

        if acc is None:
            acc = mat.copy()
        else:
            acc += mat

        used_files += 1

    # Safety check for loaded matrices
    if acc is None:
        raise RuntimeError("No valid contact maps were loaded.")

    acc.sum_duplicates()
    acc.eliminate_zeros()

    return acc, used_files, skipped
#------------------------------------------------------------------------------
























