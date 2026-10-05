import numpy as np
import time

from geometry import generate_random_walk
from genome_constants import NUCLEAR_HEIGHT


N_BEADS = 2352

start_time = time.time()

coords = generate_random_walk(
    n_beads=N_BEADS,
    step_size=1.0,
    target_z=NUCLEAR_HEIGHT,
    reverse=False,
    rng=np.random.default_rng(42),
    boundary="cylinder",
    cyl_radius=8.0,
    cyl_height=68.0,
)

elapsed = time.time() - start_time

print(f"Generation time: {elapsed:.2f} s")
print("Number of beads:", len(coords))
print("Final z:", coords[-1, 2])