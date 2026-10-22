import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import argparse
import func_mda as f

from MDAnalysis.analysis import distances as mda_dist
from scipy.sparse import csr_matrix, save_npz

#-----------------------------------
# MAIN SCRIPT TO CREATE CONTACT MAPS.
#-----------------------------------

#--------------------------------------------------------------------------------------------------------------
# Main function.

def main():
    #---------------------------------------------------------------
    # Introducing parser to read required information from terminal.
    #---------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Compute contact maps for a given trajectory.")

    parser.add_argument("--data", required=True, help="Topology file (.data, .psf).")
    parser.add_argument("--traj", required=True, help="Trajectory file (.lammpstrj).")
    parser.add_argument("--range", nargs=2, type=int, required=True, help="Atom range: [start, stop)")
    parser.add_argument("--frames", nargs=3, type=int, help="Trajectory frames: [start, stop), stride")
    parser.add_argument("--glen", type=int, default=13256, help="Genome length. Default: 13256.")
    parser.add_argument("--cut", type=float, default=1.5, help="Pairwise distance limit for contact.")
    parser.add_argument("--out", type=str, default="contact_map", help="Output filename (w/out extension).")

    args = parser.parse_args()

    #---------------------------------------------------------------
    # Loading topology and trajectory.
    #---------------------------------------------------------------
    universe = f.load_universe(args.data, args.traj)

    print("Trajectory loaded successfully - yuhu!\n")

    #---------------------------------------------------------------
    # Evaluating contact maps.
    #---------------------------------------------------------------
    c_map, n_frames = f.accumulate_contacts(universe, args.range, args.frames, args.glen, args.cut)

    print(f"Frames analyzed: {n_frames}")
    print(f"Contacts accumulated: {c_map.nnz}")

    #---------------------------------------------------------------
    # Saving and creating output file.
    #---------------------------------------------------------------
    f.save_cmap(acc, args.out)
#--------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
































































