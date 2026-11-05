#!/usr/bin/env python3
#-----------------------------------------------------------------------------------------------------------------
# exp_map.py -- experimental target map on the model bins, and its uncertainties from replicates.
#
# Example:
#   python3 exp_map.py --exp WT_ST10.10000_exp.npz --bins 02Nov_chr2_rw_seed42_without_H3K27ac_bins.tsv \
#           --replicates rep1.npz rep2.npz rep3.npz rep4.npz rep5.npz --out exp/WT_st10_chr2
# Writes:
#   <out>_exp.npz        P_exp: upper triangle, |i-j| >= d0, masked bins removed, scaled so that mean p(s=1) = 1
#   <out>_var.npz        variance across replicates per pixel (same pixels), if replicates are given
#   <out>_sigma2s.npy    sigma^2(s): mean variance over the valid pixels of each diagonal
#   <out>_bins.tsv       model bins with their experimental index and the 'valid' mask
#   <out>.json           settings and summary
#-----------------------------------------------------------------------------------------------------------------
import argparse
import json
import numpy as np
from scipy.sparse import save_npz
import contact_maps as cm

p = argparse.ArgumentParser(description="Experimental target map on the model bins (H6).")
p.add_argument("--exp", required=True, help="Genome-wide experimental map (scipy sparse .npz, dm6 10 kb).")
p.add_argument("--bins", required=True, help="Bead -> bin table of the model (H0) or a _bins.tsv from combine_maps.")
p.add_argument("--replicates", nargs="*", default=[], help="Replicate maps, same format (for sigma).")
p.add_argument("--d0", type=int, default=4, help="Exclude |i-j| < d0 (as in the energy).")
p.add_argument("--scale", choices=["s1", "none"], default="s1", help="s1: mean p of adjacent valid bins = 1.")
p.add_argument("--mask-not-in-bed", action="store_true", help="Also mask bins absent from the BED file.")
p.add_argument("--out", required=True)
a = p.parse_args()

# bins: either an original table (atom_id column) or a _bins.tsv written by combine_maps (index column)
with open(a.bins) as fh:
    header = next(l for l in fh if not l.startswith("#")).split()
if "atom_id" in header:
    bins = cm.genomic_bins(cm.read_bin_table(a.bins))
else:
    rows = [l.rstrip("\n").split("\t") for l in open(a.bins) if not l.startswith("#")]
    bins = [dict(zip(rows[0], r)) for r in rows[1:]]
    bins = [dict(b, bin=int(b["bin"]), start=int(b["start"]), end=int(b["end"]), in_bed=int(b["in_bed"])) for b in bins]

P, valid, summary = cm.experimental_map(a.exp, bins, d0=a.d0, scale=a.scale, mask_not_in_bed=a.mask_not_in_bed,
                                       mask_paths=a.replicates)
save_npz(f"{a.out}_exp.npz", P, compressed=True)

from scipy.sparse import load_npz
idx = cm.exp_indices(bins, cm.layout_for(load_npz(a.exp).shape[0]))
with open(f"{a.out}_bins.tsv", "w") as fh:
    fh.write("index\tarm\tbin\tstart\tend\tin_bed\tcompartment\texp_index\tvalid\n")
    for k, (b, e, v) in enumerate(zip(bins, idx, valid)):
        fh.write(f"{k}\t{b['arm']}\t{b['bin']}\t{b['start']}\t{b['end']}\t{b['in_bed']}\t{b['compartment']}\t{e}\t{int(v)}\n")

print(f"{summary['n_bins']} model bins; masked: {summary['empty_in_experiment']} empty in the experiment "
      f"({summary['empty_in_target']} in the target, the rest only in some replicate), "
      f"{summary['not_in_bed']} not in the BED ({summary['empty_and_not_in_bed']} both); valid: {summary['valid_bins']}")
print(f"scale ({a.scale}): factor {summary['scale_factor']:.6g}; excluded |i-j| < {a.d0}")

ps, counts = cm.diagonal_means(P, valid, s_max=1000)
print("mean P_exp along the diagonal: " + ", ".join(f"s={s}: {ps[s]:.3g}" for s in (4, 10, 30, 100, 300, 1000)))

if a.replicates:
    var, sigma2_s, mean, _ = cm.replicate_variance(a.replicates, bins, valid, d0=a.d0, scale=a.scale)
    save_npz(f"{a.out}_var.npz", var, compressed=True)
    np.save(f"{a.out}_sigma2s.npy", sigma2_s)
    pm, _ = cm.diagonal_means(mean, valid, s_max=1000)
    print(f"{len(a.replicates)} replicates; relative noise sigma(s)/<p>(s): "
          + ", ".join(f"s={s}: {np.sqrt(sigma2_s[s]) / pm[s]:.3g}" for s in (4, 10, 30, 100, 300, 1000)))
    print(f"mean of replicates vs target, ratio of diagonal means: "
          + ", ".join(f"s={s}: {pm[s] / ps[s]:.3g}" for s in (4, 100, 1000)))
    summary["replicates"] = a.replicates

summary.update(exp=a.exp, bins=a.bins, storage="upper triangle, |i-j| >= d0, model bin index (see _bins.tsv)")
with open(f"{a.out}.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"saved {a.out}_exp.npz" + (f", {a.out}_var.npz, {a.out}_sigma2s.npy" if a.replicates else "")
      + f", {a.out}_bins.tsv, {a.out}.json")
