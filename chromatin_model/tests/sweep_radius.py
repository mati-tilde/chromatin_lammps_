import numpy as np
import time
import multiprocessing as mp

from geometry import generate_coords


TIMEOUT = 120.0


def run_two_arm_test(radius, lambda_drift=0.0, seed=42):

    n_arm = 2200
    height = 68.0
    step_size = 1.0

    # --------------------------------------------------
    # Arm 1
    # --------------------------------------------------

    t0 = time.perf_counter()

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
        rng=np.random.default_rng(seed),
        lambda_drift=lambda_drift,
        drift_angle=0.0,
    )

    t1 = time.perf_counter()

    # --------------------------------------------------
    # Arm 2
    # --------------------------------------------------

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
        rng=np.random.default_rng(seed + 1),
        lambda_drift=lambda_drift,
        drift_angle=np.pi,
    )

    t2 = time.perf_counter()

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
        and np.all(coords[:, 2] >= -1e-10)
        and np.all(coords[:, 2] <= height + 1e-10)
    )

    return {
        "radius": radius,
        "status": "completed",
        "arm1_time": t1 - t0,
        "arm2_time": t2 - t1,
        "time": t2 - t0,
        "z_junction": arm1[-1, 2],
        "z_final": arm2[-1, 2],
        "bond_min": bond_lengths.min(),
        "bond_max": bond_lengths.max(),
        "inside": inside,
        "n_beads": len(coords),
    }


def worker(radius, conn):

    try:
        result = run_two_arm_test(
            radius=radius,
            lambda_drift=0.0,
            seed=42,
        )

        conn.send(result)

    except Exception as e:

        conn.send({
            "radius": radius,
            "status": "failed",
            "error": str(e),
        })

    finally:
        conn.close()


def run_with_timeout(radius, timeout=TIMEOUT):

    parent_conn, child_conn = mp.Pipe()

    process = mp.Process(
        target=worker,
        args=(radius, child_conn),
    )

    process.start()
    child_conn.close()

    process.join(timeout)

    # --------------------------------------------------
    # Timeout
    # --------------------------------------------------

    if process.is_alive():

        print(
            f"TIMEOUT after {timeout:.0f} s -> "
            f"interrupting R = {radius}",
            flush=True
        )

        process.terminate()
        process.join()

        return {
            "radius": radius,
            "status": "timeout",
        }

    # --------------------------------------------------
    # Process finished
    # --------------------------------------------------

    if parent_conn.poll():

        return parent_conn.recv()

    return {
        "radius": radius,
        "status": "failed",
        "error": "Process ended without returning a result.",
    }


if __name__ == "__main__":

    mp.freeze_support()

    radii = [
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
    ]

    print("=" * 60)
    print("RADIUS SWEEP")
    print(f"timeout per test = {TIMEOUT:.0f} s")
    print("=" * 60)

    results = []

    for radius in radii:

        print(
            f"\n=== R = {radius:.1f} ===",
            flush=True
        )

        t0 = time.perf_counter()

        result = run_with_timeout(radius)

        elapsed = time.perf_counter() - t0

        results.append(result)

        # --------------------------------------------------
        # Print result
        # --------------------------------------------------

        if result["status"] == "completed":

            print(
                f"completed in {result['time']:.4f} s "
                f"(wall time {elapsed:.4f} s)",
                flush=True
            )

            print(
                f"  arm 1: {result['arm1_time']:.4f} s",
                flush=True
            )

            print(
                f"  arm 2: {result['arm2_time']:.4f} s",
                flush=True
            )

            print(
                f"  junction z: {result['z_junction']:.6f}",
                flush=True
            )

            print(
                f"  final z:    {result['z_final']:.6f}",
                flush=True
            )

            print(
                f"  bond range: "
                f"{result['bond_min']:.12f} - "
                f"{result['bond_max']:.12f}",
                flush=True
            )

            print(
                f"  inside: {result['inside']}",
                flush=True
            )

        elif result["status"] == "timeout":

            print(
                f"  R = {radius:.1f} timed out.",
                flush=True
            )

        else:

            print(
                f"  FAILED: {result.get('error', 'unknown error')}",
                flush=True
            )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for result in results:

        radius = result["radius"]

        if result["status"] == "completed":

            print(
                f"R = {radius:4.1f}   "
                f"COMPLETED   "
                f"time = {result['time']:.4f} s"
            )

        elif result["status"] == "timeout":

            print(
                f"R = {radius:4.1f}   "
                f"TIMEOUT"
            )

        else:

            print(
                f"R = {radius:4.1f}   "
                f"FAILED"
            )