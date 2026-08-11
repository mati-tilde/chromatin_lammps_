import matplotlib.pyplot as plt
import numpy as np

from epigenetic_track import EpigeneticTrack
import functions 

#------------------------------------------------------------------------
# AVAILABLE TRACKS: trakcs container: implementing the epigenetic tracks
#------------------------------------------------------------------------

tracks = {
	"H3K27me3" : EpigeneticTrack(
		"H3K27me3",
		"H3K27me3_WT_st10.pooled.bw",
	),

	"H3K9me3" : EpigeneticTrack(
                "H3K9me3",
                "H3K9me3_WT_st10.pooled.bw",
        ),

	"H3K27ac" : EpigeneticTrack(
                "H3K27ac",
                "H3K27ac_WT_st10.pooled.bw",
        ),

	"H3K9ac" : EpigeneticTrack(
                "H3K9ac",
                "H3K9ac_WT_st10.pooled.bw",
        )
}

#-----------------------------------------
# SELECT TRACK AND CHROMOSOME TO ANALYSE
#-----------------------------------------

selected_track = tracks["H3K9ac"]

n_beads = functions.chrom_length('init_drosophila.data', 1)

selected_chroms = ["2L","2R"]

'''
signal = selected_track.multi_bead_signal(
    selected_chroms,
    n_beads
)

'''

#----------------------
# RETRIEVING FILE INFO
#----------------------

functions.write_track_info(selected_track)


#-------------------------------------------------------------
# STATISTICAL ANALYSIS WITH RESOLUTION ACCORDING TO .data FILE
#-------------------------------------------------------------

# reading the signal for the chromosome
signal = selected_track.multi_bead_signal(
    selected_chroms,
    n_beads
)


#----------------------------------
# sanity check for length of signal
assert len(signal) == n_beads
#----------------------------------


# writing beads stats
functions.write_bead_stats(selected_track, selected_chroms, signal)

# reading the file to outline chrom properties and define theta function
atom_ids, atom_types = functions.read_beads('init_drosophila.data', 1)
theta = functions.comparts_theta(atom_types)


#-----------------------------------------------
# sanity check for number of beads
print("\n")
print("Sanity check:")
print(f"Number of beads in the file: {n_beads}")
#-----------------------------------------------


# preparing ranges for plotting acetylation/metilation signal and theta function
# all atoms: 
beads = range(len(signal))
start_bead = 0
end_bead = n_beads

signal_region = signal[start_bead:end_bead]

theta_region = theta[start_bead:end_bead]

beads_region = np.arange(
    start_bead,
    end_bead
)

# groups:
start_group = 3000
end_group = 3500

g_signal_region = signal[start_group:end_group]

g_theta_region = theta[start_group:end_group]

group_region = np.arange(
    start_group,
    end_group
)



#--------------------------------------
# sanity check
print("Length of signal:", len(signal))
print("Length of theta:", len(theta))
print("Length of beads:", len(beads))
print("\n")
#--------------------------------------


# graph one (acetylation/metilation profile only)
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    beads_region,
    signal_region,
    color="black",
    linestyle="-",
    label="H3K9ac"
)

ax.set_xlabel("Bead number")
ax.set_ylabel("H3K9ac signal")

plt.title("H3K9ac")

plt.tight_layout()

plt.savefig(
    "H3K9ac_2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# graph 2 (ac/met profile and theta compartments)
fig, ax1 = plt.subplots(figsize=(12, 5))

ax1.plot(
    group_region,
    g_signal_region,
    color="black",
    linestyle="-",
    label="H3K9ac"
)

ax1.set_xlabel("Bead number")
ax1.set_ylabel("H3K9ac signal")

max_signal = max(signal)
ax1.set_ylim(- 1.2 * max_signal, 1.2 * max_signal)

ax2 = ax1.twinx()

ax2.step(
    group_region,
    g_theta_region,
    color="blue",
    #linestyle=":",
    where="mid",
    label="θ = 1 --> type A \n θ = - 1 --> type B \n θ = 0 --> centromere"
)

ax2.set_ylabel("Compartment θ")
ax2.set_ylim(-1.5, 1.5)
#ax2.set_yticks([0, 0.5])

ax2.legend(
    title="Compartment θ",
    loc="upper left"
)

plt.title("H3K9ac and chromosome compartments")

plt.tight_layout()

plt.savefig(
    "H3K9ac_2_theta.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


#-------------
# CLOSING FILE
#-------------

selected_track.close()











