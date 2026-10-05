import numpy as np
import time

from geometry import generate_coords


def run_two_arm_test(n_arm, lambda_drift, seed=42):

    radius = 10.0
    height = 68.0
    step_size = 1.0

    rng_seed = seed

    # --------------------------------------------------
    # Arm 1: start near z = 0, finish at z = height
    # --------------------------------------------------

    t0 = time.perf_counter()

    print("Generating arm 1...", flush=True)

    arm1 = generate_coords(
        method="random_walk",
        n_beads=n_arm,
        step_size=step_size,
        start=np.array([0.0, 0.0, 0.0]),
        target_z=height,
        reverse=False,
        boundary="cylinder",
        cyl_radius=radius,
        cyl_height=height,
        rng=np.random.default_rng(rng_seed),
        lambda_drift=lambda_drift,
        drift_angle=0.0,
    )

    t1 = time.perf_counter()

    print(
        f"Arm 1 completed in {t1 - t0:.4f} s",
        flush=True
    )

    print("Generating arm 2...", flush=True)

    arm2 = generate_coords(
        method="random_walk",
        n_beads=n_arm,
        step_size=step_size,
        start=arm1[-1],
        target_z=0.0,
        reverse=True,
        avoid_coords=arm1,
        boundary="cylinder",
        cyl_radius=radius,
        cyl_height=height,
        rng=np.random.default_rng(rng_seed + 1),
        lambda_drift=lambda_drift,
        drift_angle=np.pi,
    )

    t2 = time.perf_counter()

    print(
        f"Arm 2 completed in {t2 - t1:.4f} s",
        flush=True
    )

    coords = np.concatenate((arm1, arm2))

    elapsed = t2 - t0
    # --------------------------------------------------
    # Checks
    # --------------------------------------------------

    bond_lengths = np.linalg.norm(
        np.diff(coords, axis=0),
        axis=1
    )

    rho2 = coords[:, 0]**2 + coords[:, 1]**2

    inside_cylinder = (
        np.all(rho2 <= radius**2 + 1e-10)
        and
        np.all(coords[:, 2] >= -1e-10)
        and
        np.all(coords[:, 2] <= height + 1e-10)
    )

    return {
        "lambda": lambda_drift,
        "time": elapsed,
        "z_junction": arm1[-1, 2],
        "z_final": arm2[-1, 2],
        "bond_min": bond_lengths.min(),
        "bond_max": bond_lengths.max(),
        "inside": inside_cylinder,
        "n_beads": len(coords),
    }


lambdas = [0.1]

for lam in lambdas:

    print(f"\n=== lambda = {lam} ===")

    try:
        result = run_two_arm_test(
            n_arm=2200,
            lambda_drift=lam,
            seed=42,
        )

        print(f"time       = {result['time']:.4f} s")
        print(f"junction z = {result['z_junction']:.6f}")
        print(f"final z    = {result['z_final']:.6f}")
        print(f"bond range = "
              f"{result['bond_min']:.12f} - "
              f"{result['bond_max']:.12f}")
        print(f"inside     = {result['inside']}")

    except Exception as e:
        print(f"FAILED: {e}")