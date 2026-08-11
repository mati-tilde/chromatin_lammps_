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
