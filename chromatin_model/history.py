######################################################################################################
#
# OLD VERSIONS OF CODES.
#
######################################################################################################


#=====================================================================================================
# FROM: builder.py (up to 26th August)

#-----------------------------------------------------------------------------------------------------
# Helper to build a single chromosome.

# THE OG.
def build_single_chrom(arms, chromosome_arms, method, **kwargs):

    # first arm
    n_beads = len(arms[chromosome_arms[0]])

    coords = generate_coords(method = method, n_beads = n_beads, **kwargs)


    for arm in chromosome_arms[1:]:

        # geometry-dependent definition of "start":
        if method == "helix":
            start = coords[-1, :2]
        else:
            start = coords[-1]

        new_coords = generate_coords(method = method, n_beads = len(arms[arm]), start = start, reverse = True, **kwargs) #avoid_coords=coords if method == "random_walk" else None, **kwargs)

        coords = np.concatenate((coords, new_coords))


    return coords


# VERSION TWO -- WORKED FOR SOMETHING I DON'T REMEMBER.
def build_single_chrom(arms, chromosome_arms, method, **kwargs):

    # --------------------------------------------------------------
    # Random walk:
    # generate the complete chromosome as ONE continuous walk.
    # --------------------------------------------------------------

    if method == "random_walk":

        n_beads = sum(
            len(arms[arm])
            for arm in chromosome_arms
        )

        coords = generate_coords(
            method=method,
            n_beads=n_beads,
            **kwargs
        )

        return coords

    # --------------------------------------------------------------
    # Helix:
    # preserve the arm-by-arm construction.
    # --------------------------------------------------------------

    if method == "helix":

        # First arm
        n_beads = len(
            arms[chromosome_arms[0]]
        )

        coords = generate_coords(
            method=method,
            n_beads=n_beads,
            **kwargs
        )

        # Remaining arms
        for arm in chromosome_arms[1:]:

            # The helix uses the end of the previous arm
            # as the x-y starting point of the next one.
            start = coords[-1, :2]

            new_coords = generate_coords(
                method=method,
                n_beads=len(arms[arm]),
                start=start,
                reverse=True,
                **kwargs
            )

            coords = np.concatenate(
                (coords, new_coords)
            )

        return coords

    # --------------------------------------------------------------
    # Unknown geometry
    # --------------------------------------------------------------

    raise ValueError(
        f"Unknown geometry method: {method}"
    )


#-----------------------------------------------------------------------------------------------------
# Building coordinates for each chromosome.

# THE OG.
def build_chroms(arms, method, **kwargs):

    chrom_coords = {}

    for chromosome, chromosome_arms in CHROMOSOME_ARMS.items():

        chrom_coords[chromosome] = build_single_chrom(arms, chromosome_arms, method, **kwargs)

    return chrom_coords


# VERSION TWO
def build_chroms(arms, method, **kwargs):

    chrom_coords = {}

    for chromosome, chromosome_arms in CHROMOSOME_ARMS.items():

        # ----------------------------------------------------------
        # Helix-specific parameters.
        #
        # Edoardo used:
        #   drift = 5 for chr2 and chr3
        #   drift = 0 for chrX
        # ----------------------------------------------------------

        if method == "helix":

            if chromosome in ("chr2", "chr3"):
                drift = 5.0

            elif chromosome == "chrX":
                drift = 0.0

            else:
                raise ValueError(
                    f"Unknown chromosome for helix: {chromosome}"
                )

            chrom_coords[chromosome] = build_single_chrom(
                arms,
                chromosome_arms,
                method,
                drift=drift,
                **kwargs
            )

        # ----------------------------------------------------------
        # Random walk:
        # build_single_chrom() generates the whole chromosome
        # as one continuous random walk.
        # ----------------------------------------------------------

        elif method == "random_walk":

            chrom_coords[chromosome] = build_single_chrom(
                arms,
                chromosome_arms,
                method,
                **kwargs
            )

        # ----------------------------------------------------------
        # Other methods.
        # ----------------------------------------------------------

        else:

            chrom_coords[chromosome] = build_single_chrom(
                arms,
                chromosome_arms,
                method,
                **kwargs
            )

    return chrom_coords



#-----------------------------------------------------------------------------------------------------
# Build the diploid genome and apply the initial layout.


# THE OG.
def build_genome(chrom_coords, arms, method, **kwargs):

    genome = {}

    for chromosome, coords in chrom_coords.items():

        # first homolog:
        genome[f"{chromosome}A"] = coords.copy() + INITIAL_LAYOUT[f"{chromosome}A"]

        # second homolog:
        if method == "helix":
            coordsB = coords.copy()

        elif method == "random_walk":
            coordsB = build_single_chrom(arms, CHROMOSOME_ARMS[chromosome], method, **kwargs)

        elif method == "persistent_random_walk":
            coordsB = build_single_chrom(arms, CHROMOSOME_ARMS[chromosome], method, **kwargs)

        else:
            raise ValueError(f"Unknown geometry requested: {method}.")


        genome[f"{chromosome}B"] = coordsB + INITIAL_LAYOUT[f"{chromosome}B"]


    return genome



# VERSION TWO -- WORKED FOR RANDOM WALK.
def build_genome(
    chrom_coords,
    arms,
    method,
    nuclear_radius=16.0,
    nuclear_height=68.0,
    min_interchrom_dist=0.5,
    max_layout_attempts=1000,
    **kwargs
):

    genome = {}

    for chromosome, coords in chrom_coords.items():

        # --------------------------------------------------------------
        # Homolog A
        # --------------------------------------------------------------

        chromA = f"{chromosome}A"

        coordsA = (
            coords.copy()
            + INITIAL_LAYOUT[chromA]
        )

        if not _inside_nucleus(
            coordsA,
            nuclear_radius,
            nuclear_height
        ):
            raise RuntimeError(
                f"{chromA} does not fit inside the nuclear cylinder."
            )

        genome[chromA] = coordsA

        # --------------------------------------------------------------
        # Homolog B
        # --------------------------------------------------------------

        chromB = f"{chromosome}B"

        if method == "helix":

            coordsB = coords.copy()

        elif method == "random_walk":

            coordsB = build_single_chrom(
                arms,
                CHROMOSOME_ARMS[chromosome],
                method,
                **kwargs
            )

        else:

            raise ValueError(
                f"Unknown geometry '{method}'."
            )

        # --------------------------------------------------------------
        # Try the requested layout first.
        # --------------------------------------------------------------

        target = INITIAL_LAYOUT[chromB].copy()

        placed = False

        for attempt in range(max_layout_attempts):

            if attempt == 0:

                translation = target.copy()

            else:

                # Small random displacement around requested location.
                jitter = np.random.normal(
                    loc=0.0,
                    scale=1.0,
                    size=3
                )

                # Keep z fixed for now.
                jitter[2] = 0.0

                translation = target + jitter

            candidate = coordsB.copy() + translation

            # ----------------------------------------------------------
            # Must fit inside nucleus.
            # ----------------------------------------------------------

            if not _inside_nucleus(
                candidate,
                nuclear_radius,
                nuclear_height
            ):
                continue

            # ----------------------------------------------------------
            # Check against already placed chromosomes.
            # ----------------------------------------------------------

            valid = True
            for other_name, other_coords in genome.items():

                distance = _min_interchrom_distance(
                    candidate,
                    other_coords
                )

                if distance < min_interchrom_dist:
                    valid = False
                    break

            if valid:
                genome[chromB] = candidate
                placed = True
                break

        if not placed:

            raise RuntimeError(
                f"Unable to place {chromB} inside the nucleus "
                f"without pathological overlap."
            )

    return genome



#========================================================================================================
# FROM: geometry.py (up to 26th August)

#--------------------------------------------------------------------------------------------------------
# generate self avoiding random walk

# VERSION ONE -- KINDA WORKED.
def generate_random_walk(n_beads: int, step_size: float, start = None, avoid_coords = None, rng = None, reverse: bool = False, boundary = None, **boundary_kwargs):

    # sanity check:
    if n_beads < 2:
        raise ValueError(f"Number of beads has to be at least 2 (first and last)! {n_beads} is invalid.")

    if step_size <= 0:
        raise ValueError(f"Step length must be a positive value. {step_size} is invalid.")

    # introducing a random number generator for seed:
    if rng is None:
        rng = np.random.default_rng()

    # introducing a starting point for each bead:
    if start is None:
        start = np.array([0.0, 0.0, 0.0])
    else:
        start = np.asarray(start)

        if start.shape != (3,):
            raise ValueError("Start must be a 3-element coordinate.")

    # check for the start to be inside the boundary:
    if not check_constraint(boundary, start, **boundary_kwargs):
        raise ValueError("The starting point does not satisfy the selected boundary.")

    # allocating coordinates and imposing first bead:
    coords = np.zeros((n_beads + 1,3))

    coords[0] = start

    # checking presence of previous coordinates:
    if avoid_coords is not None:
        avoid_coords = np.asarray(avoid_coords)

        if avoid_coords.ndim != 2 or avoid_coords.shape[1] != 3:
            raise ValueError("avoid_coords must be an array of shape (N,3).")

    # generating the chain:
    i = 1

    while i < n_beads + 1:

        accepted = False
        max_attempts = 10000
        attempts = 0

        min_dist = 0.7
        #accepted_dist = False

        max_backtracks = 100
        backtracks = 0

        while not accepted: #and accepted_dist):
            # counting number of attempts to avoid infinite loops:
            attempts += 1

            if attempts >= max_attempts:
                backtracks += 1
                if backtracks >= max_backtracks:
                    raise RuntimeError("Unable to generate a point inside the requested boundary or self-avoiding.")
                # backtracking one bead:
                i -= 1
                coords[i:] = np.nan
                continue

            # generating a random direction:
            direction = rng.normal(size = 3)

            # normalizing direction to get random unit vector (see *2 below) and sign for reverse:
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue

            direction /= norm

            direction_sign = -1 if reverse else 1

            # candidate point from random walk definition:
            new_point = (coords[i-1] + direction_sign * step_size * direction)

            # boundary check:
            accepted_boundary = check_constraint(boundary, new_point, **boundary_kwargs)

            if not accepted_boundary:
                continue

            # self-avoidance check:
            # exclude the directly bonded previous bead:
            previous_points = coords[:i - 1]

            # add points of previous arm (if present):
            if avoid_coords is not None:
                avoid_points = avoid_coords[:-1]
            else:
                avoid_points = np.empty((0,3))

            # check distances of candidate from every other point:
            if len(avoid_points) == 0:
                points_to_avoid = previous_points
            elif len(previous_points) == 0:
                points_to_avoid = avoid_points
            else:
                points_to_avoid = np.vstack((avoid_points, previous_points))

            if len(points_to_avoid) > 0:
                squared_distances = np.sum((points_to_avoid - new_point)**2, axis = 1)
                if not np.all(squared_distances >= min_dist**2):
                    continue


        # store only accepted points:
        coords[i] = new_point
        accepted = True

    # output (discrading the first point - that is a continuation point, not to be returned):
    return coords[1:]



# VERSION 2: worked, but no endpoints were constrained.
def generate_random_walk(n_beads: int, step_size: float, start = None, avoid_coords = None, rng = None, reverse: bool = False, boundary = None, **boundary_kwargs):

    #------------------------------------------------------------------
    # sanity check:
    #------------------------------------------------------------------

    if n_beads < 2:
        raise ValueError("Number of beads has to be at least 2 (first and last)!")

    if step_size <= 0:
        raise ValueError("Step length must be a positive value.")

    # random generator:
    if rng is None:
        rng = np.random.default_rng()

    # starting point:
    if start is None:
        start = np.array([0.0, 0.0, 0.0])

    else:
        start = np.asarray(start)

        if start.shape != (3,):
            raise ValueError("Start must be a 3-element coordinate.")

    # check starting point against boundary:
    if not check_constraint(boundary, start, **boundary_kwargs):
        raise ValueError("The starting point does not satisfy the selected boundary.")

    #------------------------------------------------------------------
    # allocate coordinates:
    #------------------------------------------------------------------

    # extra first point is used as the continuation point:
    coords = np.zeros((n_beads + 1, 3))

    coords[0] = start

    #------------------------------------------------------------------
    # previous coordinates to avoid:
    #------------------------------------------------------------------

    if avoid_coords is not None:

        avoid_coords = np.asarray(avoid_coords)

        if avoid_coords.ndim != 2 or avoid_coords.shape[1] != 3:
            raise ValueError("avoid_coords must be an array of shape (N,3).")

    #------------------------------------------------------------------
    # parameters for self-avoidance and backtracking:
    #------------------------------------------------------------------

    min_dist = 0.7
    max_attempts = 10000
    max_backtracks = 100

    backtracks = 0

    #------------------------------------------------------------------
    # generate chain:
    #------------------------------------------------------------------

    i = 1

    while i < n_beads + 1:

        accepted = False
        attempts = 0

        #--------------------------------------------------------------
        # try to generate bead i:
        #--------------------------------------------------------------

        while not accepted and attempts < max_attempts:

            attempts += 1

            #----------------------------------------------------------
            # random direction:
            #----------------------------------------------------------

            direction = rng.normal(size=3)

            norm = np.linalg.norm(direction)

            if norm == 0:
                continue

            direction /= norm

            direction_sign = -1 if reverse else 1

            # candidate point:
            new_point = (coords[i - 1] + direction_sign * step_size * direction)

            #----------------------------------------------------------
            # boundary check:
            #----------------------------------------------------------

            accepted_boundary = check_constraint(boundary, new_point, **boundary_kwargs)

            if not accepted_boundary:
                continue

            #----------------------------------------------------------
            # self-avoidance check:
            #----------------------------------------------------------

            # previous beads in the current arm.
            # the directly bonded previous bead coords[i-1] is excluded automatically.

            previous_points = coords[:i - 1]

            # points from a previously generated arm.

            # NOTE:
            # the last point of avoid_coords is the arm-junction bead, which is bonded to the first bead of this arm, so it must NOT be subject to the minimum-distance rule.

            if avoid_coords is not None:
                avoid_points = avoid_coords[:-1]
            else:
                avoid_points = np.empty((0, 3))

            # combine all points that the new point must avoid:
            if len(avoid_points) == 0:

                points_to_avoid = previous_points

            elif len(previous_points) == 0:

                points_to_avoid = avoid_points

            else:

                points_to_avoid = np.vstack((avoid_points, previous_points))

            # check distances:
            if len(points_to_avoid) == 0:

                accepted_dist = True

            else:

                squared_distances = np.sum((points_to_avoid - new_point) ** 2, axis = 1)

                accepted_dist = np.all(squared_distances >= min_dist ** 2)

                if not accepted_dist:
                    continue

            # if we reached this point, the candidate is valid:
            accepted = True

        #--------------------------------------------------------------
        # no valid candidate found :/
        #--------------------------------------------------------------

        if not accepted:

            # cannot backtrack beyond the initial point:
            if i <= 1:
                raise RuntimeError("Unable to generate a self-avoiding random walk.")

            backtracks += 1

            if backtracks > max_backtracks:
                raise RuntimeError("Unable to generate a self-avoiding random walk after repeated backtracking.")

            # move back one bead:
            i -= 1

            # remove the abandoned part of the chain:
            coords[i + 1:] = np.nan

            # try generating this bead again:
            continue

        #--------------------------------------------------------------
        # store accepted point:
        #--------------------------------------------------------------

        coords[i] = new_point

        # restore backtracks:
        backtracks = 0

        # move to next bead:
        i += 1

    #------------------------------------------------------------------
    # output:
    #------------------------------------------------------------------

    # discard the artificial continuation point:
    return coords[1:]




#========================================================================================================
# FROM: writer_funcs.py (up to 1st October)


#--------------------------------------------------------------------------
# Writing a signal section in the file to give lammps.
def write_signal(genome, beads, signal_names):

    signals = []
    
    # header:
    signals.append("Signals")
    signals.append(" ")
    
    atom_id = 1

    # looping over diploid genome:
    for chromosome, coords in genome.items():

        base_chrom = chromosome[:-1]

        chrom_beads = get_beads_for_chrom(beads, base_chrom)

        #------------------------------------------------
        # sanity check:
        if len(coords) != len(chrom_beads):
            raise RuntimeError(f"{chromosome}: {len(coords)} coordinates but {len(chrom_beads)} beads.")
        #------------------------------------------------

        for bead in chrom_beads:

            values = []

            for signal_name in signal_names:

                #---------------------------------------
                # sanity check:
                if signal_name not in bead.signals:
                    raise RuntimeError(f"Bead {bead.bead_id} is missing signal {signal_name}.")
                #---------------------------------------

                values.append(bead.signals[signal_name])

            signals.append(f"{atom_id} " + " ".join(f"{value:.10g}" for value in values))

            atom_id += 1

    return signals
#--------------------------------------------------------------------------








































