from genome_constants import ARM_LENGTHS
from beads_class import Bead
from epigenetic_track import EpigeneticTrack
import func_epigen
from genome_constants import ARM_LENGTHS, ARM_TO_BIGWIG

#---------------------------------------------------------------------------------------
# FUNCTIONS TO ASSIGN CORRECT SIGNAL TO BEADS.

# RELEVANT ANNOTATIONS:
# Currently epigenetic signals are obtained using pyBigWig.stats(..., nBins=n_beads).
# Investigate whether annotation should instead use the exact genomic intervals
# stored in each Bead (start/end). The difference is small but affects the exact
# correspondence between polymer beads and genomic coordinates.
#---------------------------------------------------------------------------------------


#------------------------------------------------------------------
# Assignning signal to the arm of a chromosome.
def annotate_arm(beads, track, arm):

    # extract the beads belonging to that arm:
    arm_beads = [
        bead
        for bead in beads
        if bead.arm == arm
    ]

    #----------------------------------------
    # sanity check for arm length:
    expected = ARM_LENGTHS[arm]
    assert len(arm_beads) == expected
    #----------------------------------------

    # read signal:
    bw_arm = ARM_TO_BIGWIG[arm]

    signal = track.bead_signal(
        bw_arm,
        expected
    )

    assert len(signal) == len(arm_beads)

    signal = [
        0 if value is None else value
        for value in signal
    ]

    # store values (zip() to iterate over two lists):
    for bead, value in zip(arm_beads, signal):
        bead.add_signal(track.name, value)

# no return, because it changes the already given beads.
#------------------------------------------------------------------



#------------------------------------------------------------------
# Annotating all arms.
def annotate_genome(beads, track):

    for arm in ARM_LENGTHS:
        annotate_arm(beads, track, arm)
#------------------------------------------------------------------



