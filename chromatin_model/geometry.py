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



# HELPERS.
#----------------------------------------------------------------------------------------------------------

#**********************************************************************************************************
# Return the integer 3D cell index containing a point.
def _cell_index(point, cell_size):

    point = np.asarray(point)

    if point.shape != (3,):
        raise ValueError("Point must be a 3-element coordinate.")

    return tuple(np.floor(point / cell_size).astype(int))


#**********************************************************************************************************


# Return the cell itself and its 26 neighboring cells.
def _neighboring_cells(cell):
    cx, cy, cz = cell

    neighbors = []

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                neighbors.append((cx + dx, cy + dy, cz + dz))

    return neighbors


#**********************************************************************************************************


# Assign each point to its spatial cell; returns a dictionary (cell index -> list of point indeces).
def _build_cell_list(points, cell_size):
    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must be an array of shape (N,3).")

    if cell_size <= 0:
        raise ValueError("Cell size must be positive.")

    cell_list = {}

    for i, point in enumerate(points):

        cell = _cell_index(point, cell_size)

        if cell not in cell_list:
            cell_list[cell] = []

        cell_list[cell].append(i)

    return cell_list


#**********************************************************************************************************


# Return the indices of points lying in the cell of `point` or any of its 26 neighboring cells.
def _get_neighbor_indices(cell_list, point, cell_size):

    cell = _cell_index(point, cell_size)

    neighbor_indices = []

    for neighbor in _neighboring_cells(cell):

        if neighbor in cell_list:
            neighbor_indices.extend(cell_list[neighbor])


    return neighbor_indices


#**********************************************************************************************************


# Check whether `point` is closer than `min_dist` to any point stored in `cell_list`.
def _violates_min_distance(points, point, cell_list, cell_size, min_dist):

    neighbor_indices = _get_neighbor_indices(cell_list, point, cell_size)

    if len(neighbor_indices) == 0:
        return False

    neighbor_points = points[neighbor_indices]

    squared_distances = np.sum((neighbor_points - point) ** 2, axis=1)

    return np.any(squared_distances < min_dist ** 2)


#**********************************************************************************************************


# Add one point index to its corresponding spatial cell.
def _add_to_cell_list(cell_list, point_index, point, cell_size):
    cell = _cell_index(point, cell_size)

    if cell not in cell_list:
        cell_list[cell] = []

    cell_list[cell].append(point_index)


#**********************************************************************************************************


# Remove one point index from its spatial cell.
def _remove_from_cell_list(cell_list, point_index, point, cell_size):
    cell = _cell_index(point, cell_size)

    if cell not in cell_list:
        raise KeyError(f"Cell {cell} does not exist in the cell list.")

    cell_list[cell].remove(point_index)

    if len(cell_list[cell]) == 0:
        del cell_list[cell]
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
# 	phase: initial angular offset;
#   reverse: allows the helix to grow the other way around;
#   drift: angular distance to the z axis.
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
# 	step_size: length of the random walk step;
# 	start: point where the helix starts (necessary for initial coordinates);
# 	avoid_coords: contains coords of previous arm to check self avoidance with;
# 	rng: random generator for reproducibility;
# 	reverse: possibility to build the arm lopsided;
#   target_z: endpoint constraints for beginning/end of each arm and junction;
# 	boundary: imposing boundary on building the random walk;
# return output: "coords" --> np.ndarray, as an array of shape (n_beads,3), giving [x,y,z] for each bead.

def generate_random_walk(
    n_beads: int, step_size: float, start = None, avoid_coords = None, rng = None, reverse: bool = False, 
    target_z = None, lambda_drift = 0.0, drift_angle = 0.0, boundary = None, **boundary_kwargs):

    #------------------------------------------------------------------
    # sanity check:
    if n_beads < 2:
        raise ValueError("Number of beads has to be at least 2 (first and last)!")

    if step_size <= 0:
        raise ValueError("Step length must be a positive value.")

    if target_z is not None and not np.isfinite(target_z):
        raise ValueError("target_z must be finite.")

    if target_z is not None and not (0.0 <= target_z <= boundary_kwargs.get("cyl_height", np.inf)):
        raise ValueError("target_z must lie inside the selected boundary.")

    if lambda_drift < 0: 
        raise ValueError("Preference strength 'lambda_drift' must be non-negative.")
    #------------------------------------------------------------------

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

    # check whether the target is reachable at all
    if target_z is not None:

        max_dz = (n_beads - 1) * step_size

        if abs(target_z - start[2]) > max_dz:
            raise ValueError("The requested target_z cannot be reached with the given number of beads and step size.")

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

    # Cell list for the previously generated arm.
    if avoid_coords is not None:
        avoid_cell_list = _build_cell_list(
            avoid_coords[:-1],
            min_dist
        )
    else:
        avoid_cell_list = {}

    # Cell list for the current arm.
    current_cell_list = {}

    backtracks = 0

    #------------------------------------------------------------------
    # generate chain:
    #------------------------------------------------------------------
    i = 1

    while i < n_beads + 1:

        accepted = False
        attempts = 0

        # try to generate bead i:
        while not accepted and attempts < max_attempts:

            attempts += 1

            # candidate point -- target aware:
            if target_z is not None and i == n_beads:

                # final step: force the last bead to target_z
                dz = target_z - coords[i - 1, 2]

                if abs(dz) > step_size:
                    continue

                xy_step = np.sqrt(step_size**2 - dz**2)

                phi = rng.uniform(0.0, 2.0 * np.pi)

                new_point = coords[i - 1] + np.array([xy_step * np.cos(phi), xy_step * np.sin(phi), dz])

            else:
                # ordinary random-walk step
                direction_sign = -1 if reverse else 1

                if target_z is not None:

                    # ----------------------------------------------------------
                    # Target-aware random step.
                    #
                    # We choose the z-component so that the target remains
                    # reachable with the steps that are still available.
                    # ----------------------------------------------------------

                    current_z = coords[i - 1, 2]

                    dz_target = target_z - current_z

                    remaining_steps = n_beads - i

                    # Allowed range for the ACTUAL z displacement.
                    dz_min = max(-step_size, dz_target - remaining_steps * step_size)

                    dz_max = min(step_size, dz_target + remaining_steps * step_size)

                    if dz_min > dz_max:
                        continue

                    # Because reverse changes the sign of the random direction,
                    # transform the allowed interval accordingly.
                    raw_dz_min = min(direction_sign * dz_min, direction_sign * dz_max)

                    raw_dz_max = max(direction_sign * dz_min, direction_sign * dz_max)

                    # ----------------------------------------------------------
                    # Brownian-bridge-like proposal for the axial step.
                    # ----------------------------------------------------------

                    remaining_steps = n_beads - i + 1

                    delta_target = target_z - coords[i - 1, 2]

                    bridge_mean = delta_target / remaining_steps

                    bridge_var = (
                        step_size**2 / 3.0
                        * (remaining_steps - 1)
                        / remaining_steps
                    )

                    bridge_std = np.sqrt(bridge_var)

                    # Express the bridge mean in the raw-dz coordinate system.
                    bridge_mean_raw = direction_sign * bridge_mean

                    raw_dz = rng.normal(
                        loc=bridge_mean_raw,
                        scale=bridge_std
                    )

                    if not (
                        raw_dz_min <= raw_dz <= raw_dz_max
                    ):
                        continue

                    dz = direction_sign * raw_dz

                    # Remaining displacement in x-y.
                    xy_step = np.sqrt(step_size**2 - raw_dz**2)

                    # Random azimuth in the x-y plane.
                    if lambda_drift == 0.0:
                        phi = rng.uniform(0.0, 2.0 * np.pi)
                    else:
                        phi = rng.vonmises(drift_angle, lambda_drift)

                    dx = xy_step * np.cos(phi)
                    dy = xy_step * np.sin(phi)

                    new_point = coords[i - 1] + np.array([dx, dy, dz])

                else:
                    # Original unconstrained random-walk step.

                    direction = rng.normal(size=3)

                    norm = np.linalg.norm(direction)

                    if norm == 0:
                        continue

                    direction /= norm

                    new_point = (coords[i - 1] + direction_sign * step_size * direction)

            # target reachability check:
            if target_z is not None:

                remaining_steps = n_beads - i

                max_remaining_dz = remaining_steps * step_size

                if abs(target_z - new_point[2]) > max_remaining_dz:
                    continue

            #----------------------------------------------------------
            # boundary check:
            #----------------------------------------------------------

            accepted_boundary = check_constraint(boundary, new_point, **boundary_kwargs)

            if not accepted_boundary:
                continue

            # ----------------------------------------------------------
            # self-avoidance check:
            # ----------------------------------------------------------
            # check against the previously generated arm using the spatial cell list:
            if avoid_coords is not None:
                if _violates_min_distance(avoid_coords[:-1], new_point, avoid_cell_list, min_dist, min_dist):
                    continue

            # Check against the previous beads of the current arm using the dynamic cell list.

            if _violates_min_distance(coords, new_point, current_cell_list, min_dist, min_dist):
                continue
            
            # if we reached this point, the candidate is valid:
            accepted = True

        #--------------------------------------------------------------
        # no valid candidate found (backtracking section) :/
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

            _remove_from_cell_list(current_cell_list, i, coords[i], min_dist)

            # remove the abandoned part of the chain:
            coords[i + 1:] = np.nan

            # try generating this bead again:
            continue

        #--------------------------------------------------------------
        # store accepted point:
        #--------------------------------------------------------------

        coords[i] = new_point

        _add_to_cell_list(current_cell_list, i, new_point, min_dist)

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



if __name__ == "__main__":

    points = np.array([
        [1.2, -0.3, 4.8],
        [1.4, -0.1, 4.9],
        [-2.0, 1.0, 3.0],
    ])

    cell_size = 0.7

    cell_list = {}

    # Add points one at a time.
    for i, point in enumerate(points):

        _add_to_cell_list(
            cell_list,
            i,
            point,
            cell_size
        )

    print("After adding:")
    print(cell_list)

    # Remove point 1.
    _remove_from_cell_list(
        cell_list,
        1,
        points[1],
        cell_size
    )

    print("After removing point 1:")
    print(cell_list)