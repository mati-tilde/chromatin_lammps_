import numpy as np

from geometry import generate_random_walk


# ------------------------------------------------------------
# Generate first arm
# ------------------------------------------------------------

rng = np.random.default_rng(42)

arm1 = generate_random_walk(
    n_beads=10,
    step_size=1.0,
    target_z=5.0,
    reverse=False,
    rng=rng,
    boundary="cylinder",
    cyl_radius=8.0,
    cyl_height=20.0,
)

print("First arm generated.")
print("Arm 1 endpoint:", arm1[-1])


# ------------------------------------------------------------
# Generate second arm
# ------------------------------------------------------------

arm2 = generate_random_walk(
    n_beads=10,
    step_size=1.0,
    start=arm1[-1],
    avoid_coords=arm1,
    target_z=0.0,
    reverse=True,
    rng=rng,
    boundary="cylinder",
    cyl_radius=8.0,
    cyl_height=20.0,
)

print("Second arm generated.")
print("Arm 2 endpoint:", arm2[-1])
print("Arm 2 final z:", arm2[-1, 2])