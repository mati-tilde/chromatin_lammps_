from dataclasses import dataclass, field

##############################################################################################
#
# BEAD CLASS: contains all the information to store for a single bead from .bed and .bw files.
#
##############################################################################################

@dataclass
class Bead:

    # method to add signal
    def add_signal(self, name, value):
        self.signals[name] = value

    bead_id: int

    arm: str

    chromosome: str

    arm_id: int


    start: int

    end: int

    # storing the signals
    signals: dict[str, float] = field(default_factory=dict)					# every bead gets its own empty dictionary.

    # storing the coordinates
    x: float | None = None
    y: float | None = None
    z: float | None = None

    # genomic bookkeeping (for the bead -> Hi-C bin table):
    bin_id: int = -1                # 0-based bin index within the arm
    in_bed: bool = False            # True if the bin is listed in the .bed file, False if it was filled in
    compartment: str = "NA"         # 4th column of the .bed file (compartment score), "NA" if absent