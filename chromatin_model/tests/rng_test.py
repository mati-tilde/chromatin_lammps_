import numpy as np
import time
import multiprocessing as mp

from genome_parser import read_bed
from constrained_builder import (
    split_by_arms,
    build_chroms,
    build_single_chrom,
)
from genome_constants import CHROMOSOME_ARMS


# ==========================================================
# PARAMETERS
# ==========================================================

BED_FILE = "comp_HiTC_10000_WT_ST10_refined_with_chromwide.bed"

STEP_SIZE = 1.0
RADIUS = 8.0
HEIGHT = 68.0

SEED = 42
TIMEOUT = 120.0


# ==========================================================
# TEST FUNCTION
# ==========================================================

def generate_chr2B(rng):

    beads = read_bed(BED_FILE)
    arms = split_by_arms(beads)

    print(
        "Generating chr2B...",
        flush=True
    )

    t0 = time.perf_counter()

    coords = build_single_chrom(
        arms,
        CHROMOSOME_ARMS["chr2"],
        method="random_walk",
        step_size=STEP_SIZE,
        rng=rng,
        boundary="cylinder",
        cyl_radius=RADIUS,
        cyl_height=HEIGHT,
    )

    elapsed = time.perf_counter() - t0

    print(
        f"chr2B completed in {elapsed:.4f} s",
        flush=True
    )

    return {
        "time": elapsed,
        "n_beads": len(coords),
        "final_z": coords[-1, 2],
    }


# ==========================================================
# TEST 1 WORKER
# ==========================================================

def writer_style_worker(conn):

    try:

        print(
            "Creating writer-style RNG...",
            flush=True
        )

        rng = np.random.default_rng(SEED)

        print(
            "Generating initial chr2, chr3 and chrX...",
            flush=True
        )

        # This reproduces the random-number consumption
        # that occurs in the writer before the homolog-B loop.

        beads = read_bed(BED_FILE)
        arms = split_by_arms(beads)

        build_chroms(
            arms,
            method="random_walk",
            step_size=STEP_SIZE,
            rng=rng,
            boundary="cylinder",
            cyl_radius=RADIUS,
            cyl_height=HEIGHT,
        )

        print(
            "Initial chromosomes completed.",
            flush=True
        )

        result = generate_chr2B(rng)

        conn.send(
            ("completed", result)
        )

    except Exception as e:

        conn.send(
            ("failed", str(e))
        )

    finally:

        conn.close()


# ==========================================================
# TEST 2 WORKER
# ==========================================================

def fresh_rng_worker(conn):

    try:

        print(
            "Creating fresh RNG...",
            flush=True
        )

        rng = np.random.default_rng(SEED)

        result = generate_chr2B(rng)

        conn.send(
            ("completed", result)
        )

    except Exception as e:

        conn.send(
            ("failed", str(e))
        )

    finally:

        conn.close()


# ==========================================================
# RUN ONE TEST WITH TIMEOUT
# ==========================================================

def run_with_timeout(worker, label):

    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)

    parent_conn, child_conn = mp.Pipe()

    process = mp.Process(
        target=worker,
        args=(child_conn,),
    )

    process.start()
    child_conn.close()

    process.join(TIMEOUT)

    # ------------------------------------------------------
    # Timeout
    # ------------------------------------------------------

    if process.is_alive():

        print(
            f"\nTIMEOUT after {TIMEOUT:.0f} s",
            flush=True
        )

        process.terminate()
        process.join()

        return {
            "status": "timeout"
        }

    # ------------------------------------------------------
    # Process finished
    # ------------------------------------------------------

    if parent_conn.poll():

        status, result = parent_conn.recv()

        if status == "completed":

            return {
                "status": "completed",
                **result,
            }

        else:

            return {
                "status": "failed",
                "error": result,
            }

    return {
        "status": "failed",
        "error": "Process ended without returning a result.",
    }


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    mp.freeze_support()

    # ------------------------------------------------------
    # Test 1
    # ------------------------------------------------------

    writer_style_result = run_with_timeout(
        writer_style_worker,
        "TEST 1: WRITER-STYLE RNG STATE",
    )

    # ------------------------------------------------------
    # Test 2
    # ------------------------------------------------------

    fresh_result = run_with_timeout(
        fresh_rng_worker,
        "TEST 2: FRESH RNG",
    )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nWriter-style RNG:")

    if writer_style_result["status"] == "completed":

        print(
            f"  COMPLETED in "
            f"{writer_style_result['time']:.4f} s"
        )

        print(
            f"  n_beads = "
            f"{writer_style_result['n_beads']}"
        )

        print(
            f"  final z = "
            f"{writer_style_result['final_z']:.6f}"
        )

    elif writer_style_result["status"] == "timeout":

        print("  TIMEOUT")

    else:

        print(
            f"  FAILED: "
            f"{writer_style_result['error']}"
        )


    print("\nFresh RNG:")

    if fresh_result["status"] == "completed":

        print(
            f"  COMPLETED in "
            f"{fresh_result['time']:.4f} s"
        )

        print(
            f"  n_beads = "
            f"{fresh_result['n_beads']}"
        )

        print(
            f"  final z = "
            f"{fresh_result['final_z']:.6f}"
        )

    elif fresh_result["status"] == "timeout":

        print("  TIMEOUT")

    else:

        print(
            f"  FAILED: "
            f"{fresh_result['error']}"
        )