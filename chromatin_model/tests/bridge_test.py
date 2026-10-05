import numpy as np
import time
import multiprocessing as mp

from geometry import generate_coords


TIMEOUT = 120.0


def generate_chr3(radius):

    n_left = 2352
    n_right = 2529
    height = 68.0
    step_size = 1.0

    print("Generating chr3L...", flush=True)

    t0 = time.perf_counter()

    arm1 = generate_coords(
        method="random_walk",
        n_beads=n_left,
        step_size=step_size,
        start=np.array([0.0, 0.0, 0.0]),
        target_z=height,
        reverse=False,
        boundary="cylinder",
        cyl_radius=radius,
        cyl_height=height,
        rng=np.random.default_rng(42),
        lambda_drift=0.0,
        drift_angle=0.0,
    )

    t1 = time.perf_counter()

    print(
        f"chr3L completed in {t1 - t0:.4f} s",
        flush=True
    )

    print("Generating chr3R...", flush=True)

    arm2 = generate_coords(
        method="random_walk",
        n_beads=n_right,
        step_size=step_size,
        start=arm1[-1],
        target_z=0.0,
        reverse=True,
        avoid_coords=arm1,
        boundary="cylinder",
        cyl_radius=radius,
        cyl_height=height,
        rng=np.random.default_rng(43),
        lambda_drift=0.0,
        drift_angle=np.pi,
    )

    t2 = time.perf_counter()

    print(
        f"chr3R completed in {t2 - t1:.4f} s",
        flush=True
    )

    coords = np.concatenate((arm1, arm2))

    # --------------------------------------------------
    # Checks
    # --------------------------------------------------

    bond_lengths = np.linalg.norm(
        np.diff(coords, axis=0),
        axis=1
    )

    rho2 = coords[:, 0]**2 + coords[:, 1]**2

    inside = (
        np.all(rho2 <= radius**2 + 1e-10)
        and
        np.all(coords[:, 2] >= -1e-10)
        and
        np.all(coords[:, 2] <= height + 1e-10)
    )

    return {
        "radius": radius,
        "time": t2 - t0,
        "junction_z": arm1[-1, 2],
        "final_z": arm2[-1, 2],
        "bond_min": bond_lengths.min(),
        "bond_max": bond_lengths.max(),
        "inside": inside,
        "n_beads": len(coords),
    }


def worker(radius, conn):

    try:
        result = generate_chr3(radius)
        conn.send(("completed", result))

    except Exception as e:
        conn.send(("failed", str(e)))

    finally:
        conn.close()


def run_with_timeout(radius):

    parent_conn, child_conn = mp.Pipe()

    process = mp.Process(
        target=worker,
        args=(radius, child_conn),
    )

    process.start()
    child_conn.close()

    process.join(TIMEOUT)

    if process.is_alive():

        print(
            f"\nTIMEOUT: R = {radius}",
            flush=True
        )

        process.terminate()
        process.join()

        return None

    if parent_conn.poll():

        status, result = parent_conn.recv()

        if status == "failed":

            print(f"FAILED: {result}")

            return None

        return result

    print("Process ended without returning a result.")

    return None


if __name__ == "__main__":

    mp.freeze_support()

    radius = 8.0

    print("=" * 60)
    print(f"Testing Brownian-bridge proposal")
    print(f"chr3: 2812 + 3208 beads")
    print(f"R = {radius}")
    print("=" * 60)

    result = run_with_timeout(radius)

    if result is not None:

        print("\nRESULT")
        print("=" * 60)

        print(f"time       = {result['time']:.4f} s")
        print(f"junction z = {result['junction_z']:.6f}")
        print(f"final z    = {result['final_z']:.6f}")
        print(
            f"bond range = "
            f"{result['bond_min']:.12f} - "
            f"{result['bond_max']:.12f}"
        )
        print(f"inside     = {result['inside']}")
        print(f"n_beads    = {result['n_beads']}")