import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import argparse
import func_mda as f

from MDAnalysis.analysis import distances as mda_dist
from scipy.sparse import csr_matrix, save_npz

#--------------------------------------------------------------------
# ENSEMBLE SCRIPT FOR STATISTICS ON CMAPS FROM DIFFERENT SIMULATIONS.
#--------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------
# Main function.
def main():

    # Introducing a parser to read from terminal.
    parser = argparse.ArgumentParser(description="Operations on previously computed contact maps.")

    parser.add_argument("--directory", required=True, help="Directory containing simulation files.")
    parser.add_argument("--sim", nargs=2, type=int, required=True, help="Simulations range: [start, stop].")
    parser.add_argument("--filename", type=str, default="contact_map", help="File with contact map (without extension).")
    parser.add_argument("--action", type=str, choices=["sum", "mean"], required=True, help="Operation on the contact maps (sum or mean).")
    parser.add_argument("--out", type=str, default="ensemble", help="Output filen (without extension).")

    args = parser.parse_args()

    # Creating a file list to pass to operation functions.
    file_list = []

    path = args.directory
    start, stop = args.sim

    for i in range(start, stop + 1):
        file_list.append(f"{path}/{i}/{args.filename}.npz")

    # Implementing requested operations.
    acc, used_files, skipped = f.sum_cmaps(file_list)

    if args.action == "sum":
        total = acc
    elif args.action == "mean":
        total = acc / used_files

    # Saving result.
    f.save_cmap(total, args.out)
    print(f"Result has been saved in {args.out}.npz.")

    # Retrieving possibly damaged files.
    if skipped:
        print(f"Skipped {len(skipped)} files.")
        for file in skipped:
            print(f"  -  {file}")

    else:
        print("No files skipped.")

#--------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()




