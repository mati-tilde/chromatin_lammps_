#!/usr/bin/env python3
#-----------------------------------------------------------------------------------------------------------------
# combine_maps.py -- H3: combine per-atom maps of several runs (replicas) and convert them to genomic bins.
#
# Example:
#   python3 combine_maps.py --maps rep01_cmap rep02_cmap rep03_cmap --bins 02Nov_chr2_..._bins.tsv --out pilot_C01
# Writes:
#   <out>.npz / <out>.json          combined per-atom map (sum of W, sum of frames), reusable for further combining
#   <out>_cis.npz, _homolog.npz, _inter.npz, _bins.tsv, _binned.json   maps on genomic bins (see contact_maps.py)
# All maps must have been made with the same contact function, d0 and topology (checked).
#-----------------------------------------------------------------------------------------------------------------
import argparse
import contact_maps as cm

p = argparse.ArgumentParser(description="Combine contact maps of replicas and convert them to genomic bins.")
p.add_argument("--maps", nargs="+", required=True, help="Maps written by cmap_from_traj.py (with or without .npz).")
p.add_argument("--bins", required=True, help="Bead -> bin table written by writer.py (same topology as the runs).")
p.add_argument("--out", required=True, help="Output name.")
a = p.parse_args()

W, n_frames, meta = cm.combine_maps(a.maps)
cm.save_map(a.out, W, n_frames, meta)

table = cm.read_bin_table(a.bins)
if len(table) != W.shape[0]:
    raise SystemExit(f"{a.bins} has {len(table)} atoms, the maps have {W.shape[0]}")
maps, bins, n_copies = cm.atoms_to_bins(W, table)
cm.save_binned(a.out, maps, bins, n_frames, meta, n_copies)

total = W.sum()
print(f"{len(a.maps)} maps, {n_frames} frames in total, {W.shape[0]} atoms -> {len(bins)} genomic bins")
for name, M in maps.items():
    print(f"  {name:8s} total weight {M.sum():14.6f}  ({100 * M.sum() / total:.2f} %)")
print(f"  weight conserved: {abs(sum(M.sum() for M in maps.values()) - total) <= 1e-9 * total}")
print(f"saved {a.out}.npz/.json and {a.out}_[cis|homolog|inter].npz, {a.out}_bins.tsv, {a.out}_binned.json")
