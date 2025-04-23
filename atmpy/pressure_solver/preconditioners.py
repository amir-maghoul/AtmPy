""" This module contains the function for different preconditioning functions"""

import numpy as np
import scipy as sp

from atmpy.infrastructure.utility import one_element_inner_slice, directional_indices
from typing import Callable, TYPE_CHECKING, Dict, Tuple, Any
if TYPE_CHECKING:
    from atmpy.pressure_solver.classical_pressure_solvers import ClassicalPressureSolver


if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
    from atmpy.variables.multiple_pressure_variables import MPV
    from atmpy.pressure_solver.classical_pressure_solvers import ClassicalPressureSolver
    # Assuming utility function is accessible
    from atmpy.pressure_solver.utility import one_element_inner_slice

# =============================================================================
# Diagonal Preconditioner Components & Application
# =============================================================================

def compute_inverse_diagonal_components(
    pressure_solver: "ClassicalPressureSolver",
    dt: float,
    is_nongeostrophic: bool,
    is_nonhydrostatic: bool,
    is_compressible: bool,
) -> Dict[str, Any]:
    """
    Computes the inverse diagonal components for the Helmholtz operator.
    Returns data needed by apply_inverse_diagonal.

    Notes
    -----
    This function uses the matrix probing on colored grid to compute the diagonal components of the Helmholtz operator:
    Idea: The basic classical idea is to apply the basis vectors on the operator:
          A[k, k] = Σ_j A[k, j] * e_k[j] = A @ e_k
          This requires to apply the Helmholtz operator on basis vectors N times
          Instead , here the function applies operator on the test vectors that are specially designed as follows:
    1. Define a set S = grid points that are not neighbor to each other
        a. if i ∈ S and j ∈ S then j ∉ Neigh(i)
    2. Define the test vector x_S as characteristic (indicator) vectors of the set S:
        x_S[k] = 1 if k ∈ S
        x_S[k] = 0 if k ∉ S
    3. To find the k-th diagonal, assume k ∈ S and apply the operator on x_S:
        A @ x_S = Σ_j A[k, j] * e_k[j] = A[k, k]*x_S[k] + Σ_{j≠k} A[k, j] * x_S[j]
        a. notice k ∈ S therefore x_S[k] = 1
        b. if j ∈ S, since also k ∈ S, we know j ∉ Neigh(k), therefore A[k, j] = 0 (due to sparsity of FDM differentiation)
        c. if j ∉ S, then x_S[j] = 0
        Therefore A @ x_S = A[k, k].

    In order to create this algorithm, first we create S as the staggered grid of combinations of even/odd nodes in each direction
    *   In 1D: Two sets are needed (even indices `0::2`, odd indices `1::2`).
    *   In 2D: Four sets are needed (even-even `(0::2, 0::2)`, even-odd `(0::2, 1::2)`, odd-even `(1::2, 0::2)`, odd-odd `(1::2, 1::2)`).
    *   In 3D: Eight sets are needed (combinations like even-even-even, even-even-odd, etc.).

    """
    grid = pressure_solver.grid
    mpv = pressure_solver.mpv

    ################## Create shape and slice of one element inner ((nx-1, ny-1, nz-1)) ################################
    inner_slice = one_element_inner_slice(grid.ndim, full=False)
    inner_shape = mpv.wcenter[inner_slice].shape

    ################# Placeholder for inner points of the diagonal values ##############################################
    diag_inner = np.zeros(inner_shape, dtype=np.float64)

    full_nodal_shape = grid.nshape

    ################## Create the staggered grid and the test characteristic vector x_s (See notes) ####################
    # create list of tuple of ndim elements less than 2: For example for ndim=3 (0,0,0), (0,0,1), (0,1,0), etc
    x_s = np.zeros(full_nodal_shape, dtype=np.float64)
    indices = list(np.ndindex(*([2] * grid.ndim)))

    # Create different sets of S (staggered grid of point that are not neighbors)
    for index_tuple in indices:
        # Refresh x_S (characteristic vector) for each new S
        x_s.fill(0.0)

        # Indices of nodes in the staggered grid S
        inner_slice = tuple(slice(start, None, 2) for start in index_tuple)

        #### Create the test vector (characteristic vector x_S)
        # Create a mask for the nodes in the staggered grid and set them to 1
        inner_nodes_mask = np.zeros(inner_shape, dtype=bool)
        inner_nodes_mask[inner_slice] = True
        x_s[inner_slice][inner_nodes_mask] = 1.0

        # Apply helmholtz operator on the characteristic vector of current S
        helmholtz_op = pressure_solver.helmholtz_operator(
            x_s, dt, is_nongeostrophic, is_nonhydrostatic, is_compressible
        )
        # Since every diagonal entry represent a vector, all those vectors are linearly independent. The += ensures
        # that all values for all diagonals are stored.
        diag_inner[inner_slice] += helmholtz_op[inner_slice]

    diag_inv_inner = np.zeros_like(diag_inner)
    non_zero_mask = np.abs(diag_inner) > 1e-15
    diag_inv_inner[non_zero_mask] = 1.0 / diag_inner[non_zero_mask]

    # Return data as a dictionary
    return {'diag_inv': diag_inv_inner}


def apply_inverse_diagonal(r_flat: np.ndarray, *, diag_inv: np.ndarray) -> np.ndarray:
    """
    Applies the inverse diagonal preconditioner.
    Accepts keyword arguments matching the output of compute_inverse_diagonal_components.
    """
    original_shape = diag_inv.shape
    if r_flat.shape[0] != np.prod(original_shape):
         raise ValueError(f"Shape mismatch: r flat {r_flat.shape} vs diag_inv {original_shape}")

    r_unflat = r_flat.reshape(original_shape)
    z_unflat = diag_inv * r_unflat
    return z_unflat.flatten()

# =================================================================================
# Column (Tridiagonal) Preconditioner Components & Application (Example Structure)
# =================================================================================

def compute_tridiagonal_components(
    pressure_solver: "ClassicalPressureSolver",
    dt: float,
    is_nongeostrophic: bool,
    is_nonhydrostatic: bool,
    is_compressible: bool,
) -> Dict[str, Any]:
    """
    Computes the components (lower, diag, upper bands) for the
    tridiagonal column preconditioner by numerically probing the Helmholtz
    operator, focusing on vertical coupling. Mimics the mathematical
    approach of BK19's precon_column_prepare.

    Returns data needed by apply_inverse_tridiagonal.
    """
    grid = pressure_solver.grid
    mpv = pressure_solver.mpv

    # Identify vertical axis (assuming 1 = 'y' is vertical)
    gravity_axis = 1 # TODO: Make dynamic if needed
    if grid.ndim <= gravity_axis:
        raise ValueError("Tridiagonal preconditioner requires at least 2 dimensions with a defined vertical.")

    # --- Determine shapes and slices ---
    inner_slice_nodes = one_element_inner_slice(grid.ndim, full=False)
    inner_shape = mpv.wcenter[inner_slice_nodes].shape
    full_nodal_shape = grid.nshape
    num_inner_vert = inner_shape[gravity_axis]

    # Initialize bands (matching inner grid shape)
    lower_band = np.zeros(inner_shape, dtype=np.float64)
    diag_band = np.zeros(inner_shape, dtype=np.float64)
    upper_band = np.zeros(inner_shape, dtype=np.float64)
    p_test = np.zeros(full_nodal_shape, dtype=np.float64)

    # --- Vertical Probing (jj loop in BK19) ---
    # Probe operator response to inputs that are 1 only on specific vertical levels
    for jj in range(3): # Corresponds to j = start + jj :: 3 within inner grid
        p_test.fill(0.0)
        # Create slice for vertical staggering within the *inner* grid
        slicing_inner = list(slice(None) for _ in range(grid.ndim))
        slicing_inner[gravity_axis] = slice(jj, None, 3)
        slicing_inner = tuple(slicing_inner)

        # Create mask for inner nodes and set p_test to 1 at staggered locations
        inner_nodes_mask = np.zeros(inner_shape, dtype=bool)
        inner_nodes_mask[slicing_inner] = True
        p_test[inner_slice_nodes][inner_nodes_mask] = 1.0

        # Apply Helmholtz operator (result corresponds to inner grid)
        helmholtz_op_unflat = pressure_solver.helmholtz_operator(
            p_test, dt, is_nongeostrophic, is_nonhydrostatic, is_compressible
        )

        # --- Extract tridiagonal components ---
        # Where p_test was 1 (mask_j = inner_nodes_mask), the result gives contributions.
        mask_j = inner_nodes_mask

        # Diagonal: Result at j where input was at j
        diag_band[mask_j] += helmholtz_op_unflat[mask_j]

        # Lower: Result at j+1 where input was at j
        # Need to select results at indices corresponding to mask_j shifted +1 vertically
        if jj < num_inner_vert - 1: # Ensure j+1 is within bounds
            mask_jp1 = np.roll(mask_j, shift=1, axis=gravity_axis)
            # Only add contributions where both the input (j) and output (j+1) are valid inner points
            valid_points_lower = mask_j & mask_jp1
            lower_band[valid_points_lower] += helmholtz_op_unflat[valid_points_lower]

        # Upper: Result at j-1 where input was at j
        # Need to select results at indices corresponding to mask_j shifted -1 vertically
        if jj > 0: # Ensure j-1 is within bounds
            mask_jm1 = np.roll(mask_j, shift=-1, axis=gravity_axis)
            # Only add contributions where both input (j) and output (j-1) are valid
            valid_points_upper = mask_j & mask_jm1
            upper_band[valid_points_upper] += helmholtz_op_unflat[valid_points_upper]

    # --- Add contribution from wcenter ONLY to the diagonal ---
    # The probing above already included the effect of wcenter[j]*p[j]
    # from the input p_test[j]=1. We *don't* add wcenter again here,
    # UNLESS helmholtz_operator used for probing was *only* the Laplacian part.
    # Assuming helmholtz_operator is the *full* operator:
    # No need to add wcenter separately here. The probing captured its diagonal effect.

    # BK19's code adds horizontal probing results and then maybe wcenter to diag[1].
    # Replicating that might be needed if vertical probing alone isn't sufficient.
    # For now, let's stick to vertical probing as the primary source for tridiagonal approx.
    # If performance is poor, reconsider adding horizontal diag contributions.

    # --- Return components ---
    return {'lower': lower_band, 'diag': diag_band, 'upper': upper_band, 'grid': grid, 'gravity_axis': gravity_axis}


def apply_inverse_tridiagonal(r_flat: np.ndarray, *,
                              lower: np.ndarray,
                              diag: np.ndarray,
                              upper: np.ndarray,
                              grid: "Grid",
                              gravity_axis: int) -> np.ndarray:
    """
    Applies the inverse tridiagonal preconditioner by solving column-wise systems
    using scipy.linalg.solve_banded.

    Accepts keyword arguments matching the output of compute_tridiagonal_components.
    """
    inner_slice = one_element_inner_slice(grid.ndim, full=False)
    inner_shape = tuple(len(range(*s.indices(dim))) for s, dim in zip(inner_slice, grid.nshape))
    num_inner_vert = inner_shape[gravity_axis]

    # Check shapes
    if not (lower.shape == diag.shape == upper.shape == inner_shape):
         raise ValueError("Shape mismatch between bands and inner grid shape.")
    if r_flat.shape[0] != np.prod(inner_shape):
         raise ValueError(f"Shape mismatch: r_flat {r_flat.shape} vs inner_shape {inner_shape}")

    r_unflat = r_flat.reshape(inner_shape)
    z_unflat = np.zeros_like(r_unflat)

    iter_dims = [grid.icshape[d] for d in range(grid.ndim) if d != gravity_axis]
    iter_indices = np.ndindex(*iter_dims)

    # --- Loop over all columns ---
    for index_tuple in iter_indices:
        col_slice = list(slice(None) for _ in range(grid.ndim))
        idx_counter = 0
        for d in range(grid.ndim):
            if d != gravity_axis:
                col_slice[d] = index_tuple[idx_counter]
                idx_counter += 1
        col_slice = tuple(col_slice)

        r_col = r_unflat[col_slice]
        l_col = lower[col_slice]
        d_col = diag[col_slice]
        u_col = upper[col_slice]

        # Assemble banded matrix for scipy.linalg.solve_banded
        ab = np.zeros((3, num_inner_vert), dtype=r_col.dtype)
        ab[0, 1:] = u_col[:-1]  # Upper diagonal u_j -> row j-1
        ab[1, :]  = d_col       # Main diagonal d_j -> row j
        ab[2, :-1] = l_col[1:]  # Lower diagonal l_j -> row j+1

        # Solve
        try:
            z_col = sp.linalg.solve_banded((1, 1), ab, r_col, check_finite=False)
            z_unflat[col_slice] = z_col
        except np.linalg.LinAlgError:
            # Handle singularity - check if diagonal is near zero
            if np.any(np.abs(d_col) < 1e-15):
                 print(f"Warning: Near-zero diagonal found in tridiagonal solve for column {col_slice}. Setting result to zero.")
                 z_unflat[col_slice] = 0.0
            else:
                 print(f"Warning: LinAlgError (possibly singular) in tridiagonal solve for column {col_slice}. Setting result to zero.")
                 z_unflat[col_slice] = 0.0 # Fallback

    return z_unflat.flatten()

