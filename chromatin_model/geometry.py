import numpy as np


#----------------------------------------------------------------------------------------------------------
# GEOMETRY MODULE (for file.data generation).

# Purpose:
# - create initial coordinates, dealing only with general beads objects with no biological background.
# - gives a public interface to choose the method with which generate beads.
# - allows to choose border conditions independently.

# Additional informations:
# see "NOTES ON THE CODE FOR USER: REASONING OF CHOICES." at the end of code.
#----------------------------------------------------------------------------------------------------------




# METHOD DISPATCHER.
#----------------------------------------------------------------------------------------------------------
# Defining the public interface:

# generate_coords generates coordinates according to a general chosen method;
# input parameters: "method" --> method of choice; "**kwargs" --> several parameters related to the method.
# return output: "coords" --> np.ndarray, as an array of shape (N,3), giving generated coordinates.

def generate_coords(method: str, boundary: str | None = None, boundary_kwargs: dict | None = None, **kwargs):

    # forwarding boundaries:
    if boundary_kwargs is None:
        boundary_kwargs = {}

    if method == "helix":
        return generate_helix(**kwargs)

    elif method == "random_walk":
        return generate_random_walk(boundary = boundary, **boundary_kwargs, **kwargs)

    elif method == "persistent_walk":
        return generate_persistent_walk(boundary = boundary, **boundary_kwargs, **kwargs)

    else:
        raise ValueError(f"Unknown coordinate generation method: {method}.")
#----------------------------------------------------------------------------------------------------------




# METHODS.
#----------------------------------------------------------------------------------------------------------
# Defining the methods to use in the public interface (helix, random walk, persistent random walk):

#**********************************************************************************************************

# 1) helix method (deterministic): allows to generate coordinates using a helix method.
# input parameters:
# 	n_beads: number of beads to use in the simulation;
# 	radius: radius of the helix;
# 	height: extension along z of the helix;
# 	start: point where the helix starts (necessary for initial coordinates);
# 	turns: number of full 2π revolutions occurring;
# 	phase: initial angular offset.
# return output: "coords" --> np.ndarray, as an array of shape (n_beads,3), giving [x,y,z] for each bead.

def generate_helix(n_beads: int, radius: float, height: float, turns: float, start = None, phase: float = 0.0, reverse: bool = False, drift: float = 0.0):

    # sanity check:
    if n_beads < 2:
        raise ValueError(f"Number of beads has to be at least 2 (first and last)! {n_beads} is invalid.")

    # defining variables:
    s = np.linspace(0.0, 1.0, n_beads)

    theta = 2 * np.pi * turns * s

    # introducing a starting point for each bead:
    if start is None:
        start = np.array([0.0, 0.0])
    else:
        start = np.asarray(start)

        if start.shape != (2,):
            raise ValueError("Start must be a 3-element coordinate.")

    # coordinates:
    x = start[0] - radius * np.cos(theta[-2]) + radius * np.cos(theta) + s * drift
    y = start[1] - radius * np.sin(theta[-2]) + radius * np.sin(theta) + s * drift

    if reverse:
        z = height * (1.0 - s)
    else:
        z = height * s


    coords = np.column_stack([x,y,z])

    # output:
    return coords

# NOTE: the helix method does not have a boundary check implemented directly. this choice is related to
#       its deterministic nature, for which a point could only slightly be outside the walls but would
#       however stop the simulation. this will be taken care of in LAMMPS. the random walks, however,
#       need the check to be intrinsic since they could end up extremely far away, and so i believe it
#       may be best for those to be implemented already inside the boundary.


#**********************************************************************************************************

# 2) random walk method (stochastic)
# input parameters:
# 	n_beads: number of beads to use in the simulation;
# 	step_length: length of the random walk step;
# 	start: point where the helix starts (necessary for initial coordinates);
# 	avoid_coords: contains coords of previous arm to check self avoidance with;
# 	rng: random generator for reproducibility;
# 	reverse: possibility to build the arm lopsided;
# 	boundary: imposing boundary on building the random walk;
# return output: "coords" --> np.ndarray, as an array of shape (n_beads,3), giving [x,y,z] for each bead.

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

#**********************************************************************************************************

# 3) persistent random walk method (stochastic)
# input parameters:
#       n_beads: number of beads to use in the simulation;
#       step_length: length of the random walk step;
# 	persistence: memory of previous direction (0 --> no memory,1 --> straight chain);
#       start: point where the helix starts (necessary for initial coordinates);
#       seed: random seed for reproducibility.
# return output: "coords" --> np.ndarray, as an array of shape (n_beads,3), giving [x,y,z] for each bead.

def generate_persistent_walk(n_beads: int, step_size: float, persistence: float, start = None, rng = None, reverse: bool = False, boundary = None, **boundary_kwargs):

    if n_beads < 2:
        raise ValueError(f"Number of beads has to be at least 2 (first and last)! {n_beads} is invalid.")

    if step_size 	<= 0:
        raise ValueError(f"Step length must be a positive value. {step_size} is invalid.")

    if not 0 <= persistence <= 1:
        raise ValueError(f"Persistence must be in [0,1]. {persistence} is invalid")

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

    # initial random direction:
    direction = rng.normal(size = 3)

    norm = np.linalg.norm(direction)

    direction /= norm

    # generating chain:
    for i in range(1, n_beads + 1):

        accepted = False
        max_attempts = 10000
        attempts = 0

        direction_sign = -1 if reverse else 1

        while not accepted:
            # generating a random perturbation:
            random_direction = rng.normal(size = 3)

            # normalizing direction to get random unit vector (see *2 below):
            r_norm = np.linalg.norm(random_direction)
            if r_norm == 0:
                continue

            random_direction /= r_norm

            # imposing definition of persistent random walk:
            try_direction = (persistence * direction) + (1 - persistence) * random_direction

            norm = np.linalg.norm(try_direction)

            try_direction /= norm

            # counting attempts to avoid infinite loops:
            attempts += 1

            # candidate point from random walk definition:
            new_point = (coords[i-1] + direction_sign * step_size * try_direction)

            # boundary check:
            accepted = check_constraint(boundary, new_point, **boundary_kwargs)

            #---------- DEBUG ----------
            if attempts % 1000 == 0:
                r = np.sqrt(new_point[0]**2 + new_point[1]**2)

                print(f"Bead {i}, attempt {attempts}")
                print(f"r = {r:.3f}, "f"z = {new_point[2]:.3f}, "f"accepted = {accepted}")
            #----------------------------


            if attempts >= max_attempts:
                raise RuntimeError("Unable to generate a point inside the requested boundary.")

            if accepted:
                direction = try_direction

        # store only accepted points:
        coords[i] = new_point


    # output:
    return coords[1:]

#----------------------------------------------------------------------------------------------------------




# BOUNDARY DISPATCHER.
#----------------------------------------------------------------------------------------------------------
def check_constraint(boundary, point, **kwargs):

    if boundary is None:
        return True

    elif boundary == "cylinder":
        return _is_inside_cyl(point, **kwargs)

    else:
        raise ValueError("Invalid boundary condition! Choices: none, cylinder.")
#----------------------------------------------------------------------------------------------------------




# BOUNDARY CONDITIONS.
#----------------------------------------------------------------------------------------------------------
# Defining functions to apply boundary conditions to the coordinates. Method dependent applications.

#**********************************************************************************************************

# CYLINDER BOUNDARY CONDITIONS.

# 1) _is_inside_cyl: evaluating if a generated point is inside the cylinder.
# input parameters:
# 	point: array_like of shape (3,), cartesian coords of a point;
# 	radius: cylinder radius;
# 	height: cylinder height.
# output: bool return (True = point inside the cylinder).

def _is_inside_cyl(point, cyl_radius: float, cyl_height: float):

    point = np.asarray(point)

    # sanity check:
    if point.shape != (3,):
        raise ValueError(f"Invalid point shape, {point}. Correct shape is (3,).")

    if cyl_radius <= 0:
        raise ValueError("Radius must be higher than zero.")

    if cyl_height <= 0:
        raise ValueError("Height must be higher than zero.")

    # variables:
    squared_dist = (point[0]**2) + (point[1]**2)
    radial_in = squared_dist <= cyl_radius**2
    height_in = 0 <= point[2] <= cyl_height


    return radial_in and height_in
#**********************************************************************************************************
#----------------------------------------------------------------------------------------------------------




# NOTES ON THE CODE FOR USER: REASONING OF CHOICES.
#----------------------------------------------------------------------------------------------------------

# *1: translating ONLY x and y coordinates, not the z one.
# *2: normalizing because we request a unit normal vector in the definition of random walk; sanity check
#     to have norm non-null otherwise would create problems in the normalization of direction.
# *3: the helix is a stochastic mechanism: there is no meaningful way to regenerate a point if it lies
#     outside the boundary; so, consider the whole helix and then eventually change the parameters.
#----------------------------------------------------------------------------------------------------------






















