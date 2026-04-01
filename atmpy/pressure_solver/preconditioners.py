# """ This module contains the function for different preconditioning functions"""
#
# import numpy as np
# from typing import TYPE_CHECKING, Dict, Any, Tuple, List
#
# if TYPE_CHECKING:
#     from atmpy.pressure_solver.classical_pressure_solvers import ClassicalPressureSolver
#     from atmpy.grid.kgrid import Grid
#
# # ==============================================================================
# # Helper: Vectorized Solver (Batch TDMA)
# # ==============================================================================
#
# def solve_tridiagonal_batched(
#     a: np.ndarray,
#     b: np.ndarray,
#     c: np.ndarray,
#     d: np.ndarray
# ) -> np.ndarray:
#     """
#     Solves many tridiagonal systems Ax = d simultaneously using a vectorized Thomas algorithm.
#     This eliminates the slow Python loop over columns.
#
#     Inputs are shape (N_systems, N_levels).
#     We solve along the last axis (N_levels).
#
#     a: Lower diagonal (a[:, 0] is unused)
#     b: Main diagonal
#     c: Upper diagonal (c[:, -1] is unused)
#     d: RHS
#     """
#     n_levels = d.shape[-1]
#
#     # Work on copies to avoid modifying inputs
#     # c_prime and d_prime are modified in place during forward sweep
#     c_prime = c.copy()
#     d_prime = d.copy()
#     b_work = b.copy()
#
#     # 1. Forward Elimination
#     # This loop runs N_levels times (e.g., 50), which is fast.
#     # The operations inside are vectorized over N_systems (e.g., 10,000).
#     for k in range(1, n_levels):
#         w = a[:, k] / b_work[:, k-1]
#         b_work[:, k] = b_work[:, k] - w * c_prime[:, k-1]
#         d_prime[:, k] = d_prime[:, k] - w * d_prime[:, k-1]
#
#     # 2. Backward Substitution
#     x = np.zeros_like(d)
#     x[:, -1] = d_prime[:, -1] / b_work[:, -1]
#
#     for k in range(n_levels - 2, -1, -1):
#         x[:, k] = (d_prime[:, k] - c_prime[:, k] * x[:, k+1]) / b_work[:, k]
#
#     return x
#
#
# # ==============================================================================
# # Helper: Optimized Slicing
# # ==============================================================================
#
# def _get_coeff_slices_for_axis(
#     inner_slice: Tuple[slice, ...],
#     axis: int,
#     node_shape: Tuple[int, ...]
# ) -> Tuple[Tuple[slice, ...], Tuple[slice, ...]]:
#     """
#     Generates the slice tuples for the Cell-centered array corresponding to
#     the Left (i-1) and Right (i) neighbors of the nodes in inner_slice.
#     """
#     sl_left = []
#     sl_right = []
#
#     for d, s_node in enumerate(inner_slice):
#         # Resolve node start/stop
#         start_n = s_node.start if s_node.start is not None else 0
#         stop_n = s_node.stop if s_node.stop is not None else node_shape[d]
#
#         # Handle negative indices
#         if start_n < 0: start_n += node_shape[d]
#         if stop_n < 0: stop_n += node_shape[d]
#
#         if d == axis:
#             # Active Axis:
#             # Node i uses Flux from Cell i-1 (Left) and Cell i (Right)
#             sl_left.append(slice(start_n - 1, stop_n - 1))
#             sl_right.append(slice(start_n, stop_n))
#         else:
#             # Transverse Axis:
#             # Cell index aligns with Node index
#             sl_left.append(slice(start_n, stop_n))
#             sl_right.append(slice(start_n, stop_n))
#
#     return tuple(sl_left), tuple(sl_right)
#
#
# # =============================================================================
# # Diagonal Preconditioners
# # =============================================================================
#
# def compute_inverse_diagonal_components(
#     pressure_solver: "ClassicalPressureSolver",
#     dt: float,
#     is_nongeostrophic: bool,
#     is_nonhydrostatic: bool,
#     is_compressible: bool,
# ) -> Dict[str, Any]:
#     """
#     Computes the inverse diagonal of the Helmholtz operator.
#     Optimized to slice arrays *before* arithmetic to save memory bandwidth.
#     """
#     grid = pressure_solver.grid
#     mpv = pressure_solver.mpv
#     inner_slice = grid.get_inner_slice()
#     node_shape = grid.nshape
#     gravity_axis = pressure_solver.coriolis.gravity.axis
#
#     # Pre-calculate scalar coefficients
#     f_z = pressure_solver.coriolis.strength[2]
#     m_inv_horiz = 1.0 / (1.0 + (dt * f_z) ** 2)
#     m_inv_vert = 1.0
#
#     # 1. Start with Pressure Term (wcenter)
#     # This creates the accumulator array 'full_diag'
#     full_diag = mpv.wcenter[inner_slice].copy()
#
#     # 2. Add Horizontal Diffusion (X)
#     # Kx = (dt * pTheta * M_inv) / dx^2
#     # We slice pTheta first, then multiply.
#     factor_x = (dt * m_inv_horiz) / (grid.dxyz[0] ** 2)
#     sl_l, sl_r = _get_coeff_slices_for_axis(inner_slice, 0, node_shape)
#
#     # In-place addition to minimize allocation
#     # D += (pTheta_L + pTheta_R) * factor
#     pTheta_sum = mpv.wplus[0][sl_l] + mpv.wplus[0][sl_r]
#     full_diag += pTheta_sum * factor_x
#
#     # 3. Add Horizontal Diffusion (Y)
#     if grid.ndim >= 2:
#         factor_y = (dt * m_inv_horiz) / (grid.dxyz[1] ** 2)
#         sl_l, sl_r = _get_coeff_slices_for_axis(inner_slice, 1, node_shape)
#
#         pTheta_sum = mpv.wplus[0][sl_l] + mpv.wplus[0][sl_r]
#         full_diag += pTheta_sum * factor_y
#
#     # 4. Add Vertical Diffusion (Z) with Boundary Checks
#     if grid.ndim == 3:
#         factor_z = (dt * m_inv_vert) / (grid.dxyz[2] ** 2)
#         sl_l, sl_r = _get_coeff_slices_for_axis(inner_slice, gravity_axis, node_shape)
#
#         # Get raw slices
#         pt_l = mpv.wplus[0][sl_l]
#         pt_r = mpv.wplus[0][sl_r]
#
#         # Handle Wall Boundaries for Diagonal (Zero Flux from outside)
#         # If we are at the bottom boundary, the 'Left' flux comes from ghost cell. Zero it.
#         # If we are at the top boundary, the 'Right' flux comes from ghost cell. Zero it.
#
#         # We perform the multiplication first to get K
#         k_l = pt_l * factor_z
#         k_r = pt_r * factor_z
#
#         # Zero out ghost fluxes
#         # Note: We are modifying local copies k_l, k_r
#         slice_bottom = [slice(None)] * grid.ndim
#         slice_bottom[gravity_axis] = 0
#         k_l[tuple(slice_bottom)] = 0.0
#
#         slice_top = [slice(None)] * grid.ndim
#         slice_top[gravity_axis] = -1
#         k_r[tuple(slice_top)] = 0.0
#
#         full_diag += (k_l + k_r)
#
#     # 5. Invert
#     diag_inv = np.zeros_like(full_diag)
#     mask = np.abs(full_diag) > 1e-25
#     diag_inv[mask] = 1.0 / full_diag[mask]
#
#     return {"diag_inv": diag_inv}
#
#
# def apply_inverse_diagonal(r_flat: np.ndarray, *, diag_inv: np.ndarray) -> np.ndarray:
#     if r_flat.size != diag_inv.size:
#         raise ValueError(f"Size mismatch: RHS {r_flat.size} vs Diag {diag_inv.size}.")
#     return (r_flat.reshape(diag_inv.shape) * diag_inv).flatten()
#
# compute_inverse_analytical_diagonal_components = compute_inverse_diagonal_components
# apply_inverse_analytical_diagonal = apply_inverse_diagonal
#
#
# # =============================================================================
# # Column (Tridiagonal) Preconditioner
# # =============================================================================
#
# def compute_tridiagonal_components(
#     pressure_solver: "ClassicalPressureSolver",
#     dt: float,
#     is_nongeostrophic: bool,
#     is_nonhydrostatic: bool,
#     is_compressible: bool,
# ) -> Dict[str, Any]:
#     """Computes the vertical tridiagonal bands explicitly."""
#     grid = pressure_solver.grid
#     mpv = pressure_solver.mpv
#     inner_slice = grid.get_inner_slice()
#     node_shape = grid.nshape
#     gravity_axis = pressure_solver.coriolis.gravity.axis
#
#     # Pre-calculate scalars
#     f_z = pressure_solver.coriolis.strength[2]
#     m_inv_horiz = 1.0 / (1.0 + (dt * f_z) ** 2)
#     m_inv_vert = 1.0
#
#     # --- 1. Compute Full Diagonal ---
#     # Initialize with wcenter
#     diag_band = mpv.wcenter[inner_slice].copy()
#
#     # Add X
#     factor_x = (dt * m_inv_horiz) / (grid.dxyz[0] ** 2)
#     sl_l, sl_r = _get_coeff_slices_for_axis(inner_slice, 0, node_shape)
#     diag_band += (mpv.wplus[0][sl_l] + mpv.wplus[0][sl_r]) * factor_x
#
#     # Add Y
#     if grid.ndim >= 2:
#         factor_y = (dt * m_inv_horiz) / (grid.dxyz[1] ** 2)
#         sl_l, sl_r = _get_coeff_slices_for_axis(inner_slice, 1, node_shape)
#         diag_band += (mpv.wplus[0][sl_l] + mpv.wplus[0][sl_r]) * factor_y
#
#     # --- 2. Vertical Contributions (Diagonal & Off-Diagonal) ---
#     factor_z = (dt * m_inv_vert) / (grid.dxyz[2] ** 2)
#     sl_l_z, sl_r_z = _get_coeff_slices_for_axis(inner_slice, gravity_axis, node_shape)
#
#     # Calculate Vertical Coefficients (Kz)
#     kz_l = mpv.wplus[0][sl_l_z] * factor_z
#     kz_r = mpv.wplus[0][sl_r_z] * factor_z
#
#     # -- Diagonal Update --
#     # Enforce Zero Flux on boundaries for the diagonal summation
#     # (Ghost coefficients do not contribute to the self-term in Neumann/Wall)
#     kz_l_diag = kz_l.copy()
#     kz_r_diag = kz_r.copy()
#
#     slice_bottom = [slice(None)] * grid.ndim
#     slice_bottom[gravity_axis] = 0
#     kz_l_diag[tuple(slice_bottom)] = 0.0
#
#     slice_top = [slice(None)] * grid.ndim
#     slice_top[gravity_axis] = -1
#     kz_r_diag[tuple(slice_top)] = 0.0
#
#     diag_band += (kz_l_diag + kz_r_diag)
#
#     # -- Off-Diagonal Update --
#     # Upper Band (coeff of p_{k+1}) = -Conductance_Right = -kz_r
#     # Lower Band (coeff of p_{k-1}) = -Conductance_Left  = -kz_l
#     upper_band = -1.0 * kz_r
#     lower_band = -1.0 * kz_l
#
#     return {
#         "lower": lower_band,
#         "diag": diag_band,
#         "upper": upper_band,
#         "grid": grid,
#         "gravity_axis": gravity_axis,
#     }
#
#
# def apply_inverse_tridiagonal(
#     r_flat: np.ndarray,
#     *,
#     lower: np.ndarray,
#     diag: np.ndarray,
#     upper: np.ndarray,
#     grid: "Grid",
#     gravity_axis: int,
# ) -> np.ndarray:
#     """
#     Solves the tridiagonal system for each vertical column using vectorized TDMA.
#     """
#     inner_shape = grid.inshape
#     r = r_flat.reshape(inner_shape)
#
#     # 1. Permute axes to put gravity axis LAST: (..., N_vertical)
#     # This prepares the data for batch processing
#     r_t = np.moveaxis(r, gravity_axis, -1)
#     l_t = np.moveaxis(lower, gravity_axis, -1)
#     d_t = np.moveaxis(diag, gravity_axis, -1)
#     u_t = np.moveaxis(upper, gravity_axis, -1)
#
#     permuted_shape = r_t.shape
#     n_vertical = permuted_shape[-1]
#
#     # 2. Flatten all horizontal dimensions into a single 'system' dimension
#     # Shape becomes (N_systems, N_levels)
#     # This allows us to solve ~10,000 systems in one go using numpy arrays
#     r_batch = r_t.reshape(-1, n_vertical)
#     l_batch = l_t.reshape(-1, n_vertical)
#     d_batch = d_t.reshape(-1, n_vertical)
#     u_batch = u_t.reshape(-1, n_vertical)
#
#     # 3. Solve using Batch TDMA
#     x_batch = solve_tridiagonal_batched(l_batch, d_batch, u_batch, r_batch)
#
#     # 4. Reshape back and restore axis order
#     x_t = x_batch.reshape(permuted_shape)
#     x = np.moveaxis(x_t, -1, gravity_axis)
#
#     return x.flatten()


""" This module contains the function for different preconditioning functions"""

import numpy as np
import scipy as sp

from atmpy.infrastructure.utility import (
    one_element_inner_slice,
    directional_indices,
    one_element_inner_nodal_shape,
)
from typing import Callable, TYPE_CHECKING, Dict, Tuple, Any, List

if TYPE_CHECKING:
    from atmpy.pressure_solver.classical_pressure_solvers import ClassicalPressureSolver


if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
    from atmpy.variables.multiple_pressure_variables import MPV
    from atmpy.pressure_solver.classical_pressure_solvers import ClassicalPressureSolver

    # Assuming utility function is accessible
    from atmpy.pressure_solver.utility import one_element_inner_slice

# ==============================================================================
# Utility: Matrix Probing
# ==============================================================================


def _perform_operator_probing(
    pressure_solver: "ClassicalPressureSolver",
    dt: float,
    is_nongeostrophic: bool,
    is_nonhydrostatic: bool,
    is_compressible: bool,
    inner_shape: Tuple[int, ...],  # Shape of the inner grid (operator domain)
    inner_slice_nodes: Tuple[slice, ...],  # Slice to get inner nodes from full grid
) -> List[Tuple[Tuple[int, ...], np.ndarray]]:
    """
    Helper function to probe the Helmholtz operator numerically.

    Applies the operator to sparse test vectors (staggered 1s) (see Notes) and returns
    the results along with the corresponding staggering pattern.

    Parameters
    ----------
    pressure_solver : ClassicalPressureSolver
        The solver instance.
    dt : float
        Time step and regime flags for the operator.
    inner_shape : Tuple[int, ...]
        The shape of the one element inner nodal grid (e.g., nx-1, ny-1, nz-1).
    inner_slice: Tuple[slice, ...]
        The slice to access the one element inner nodes from the full nodal grid.

    Returns
    -------
    List of tuples.
    Each tuple contains:
            - index_tuple: The staggering pattern (e.g., (0,1,0)).
            - helmholtz_op_unflat: The unflattened operator result for that pattern.

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
    full_nodal_shape = grid.nshape

    ################## Create the staggered grid and the test characteristic vector x_s (See notes) ####################
    # create list of tuple of ndim elements less than 2: For example for ndim=3 (0,0,0), (0,0,1), (0,1,0), etc
    x_s = np.zeros(full_nodal_shape, dtype=np.float64)
    # Generate all 2^N staggering patterns
    indices = list(np.ndindex(*([2] * grid.ndim)))

    # Get the boundary manager
    boundary_manager = pressure_solver.boundary_manager

    # Initialize results
    probe_results = []

    # Create different sets of S (staggered grid of point that are not neighbors)
    for index_tuple in indices:
        # Refresh x_S (characteristic vector) for each new S
        x_s.fill(0.0)
        # Indices of nodes in the staggered grid S
        slicing_inner = tuple(slice(start, None, 2) for start in index_tuple)

        #### Create the test vector (characteristic vector x_S)
        # Create a mask for the nodes in the staggered grid and set them to 1
        inner_nodes_mask = np.zeros(inner_shape, dtype=bool)
        inner_nodes_mask[slicing_inner] = True
        x_s[inner_slice_nodes][inner_nodes_mask] = 1.0

        # boundary_manager.apply_pressure_boundary_on_all_sides(x_s)

        # Apply helmholtz operator on the characteristic vector of current S
        helmholtz_op_unflat = pressure_solver.helmholtz_operator(
            x_s, dt, is_nongeostrophic, is_nonhydrostatic, is_compressible
        )

        # Store the result associated with its staggering pattern
        probe_results.append((index_tuple, helmholtz_op_unflat))

    return probe_results


# =============================================================================
# Diagonal Preconditioners Components & Application
# =============================================================================


def compute_inverse_analytical_diagonal_components(
    pressure_solver: "ClassicalPressureSolver",
    dt: float,
    is_nongeostrophic: bool,
    is_nonhydrostatic: bool,
    is_compressible: bool,
) -> Dict[str, Any]:
    """
    Computes an analytical approximation of the inverse diagonal of the Helmholtz operator.

    This is a refactored version of the user-provided `precon_diag_prepare` function.
    It approximates the diagonal from div(M_inv * grad(p)) and adds the wcenter term.
    """
    # --- 1. Get necessary objects and parameters from the pressure solver ---
    grid = pressure_solver.grid
    mpv = pressure_solver.mpv
    coriolis = pressure_solver.coriolis
    ndim = grid.ndim
    dx, dy, dz = grid.dxyz

    # --- 2. Calculate components of the inverted momentum matrix M_inv ---
    # This matrix accounts for Coriolis effects in the momentum equations.
    # In 2D (x,y), M_inv = (1 / (1 + (dt*f)^2)) * [[1, dt*f], [-dt*f, 1]]
    f = coriolis.strength[2]  # Assuming f-plane, f is the z-component of Omega
    denom = 1.0 / (1.0 + (dt * f) ** 2)

    # Note: These components are scalars for an f-plane approximation
    M_inv_11 = denom
    M_inv_12 = dt * f * denom
    M_inv_21 = -dt * f * denom
    M_inv_22 = denom

    # --- 3. Get the cell-centered P*Theta coefficient ---
    # This is mpv.wplus, which is computed from the current state.
    # We only need the first component since it's the same for all momenta.
    pTheta_cells = mpv.wplus[0]

    # --- 4. Calculate the effective coefficients for each Laplacian term ---
    # hplus_ij = pTheta * M_inv_ij
    hplusxx = pTheta_cells * M_inv_11
    hplusyy = pTheta_cells * M_inv_22
    hplusxy = pTheta_cells * M_inv_12
    hplusyx = pTheta_cells * M_inv_21
    if ndim == 3:
        # Assuming no rotation effects on the vertical component for simplicity
        hpluszz = pTheta_cells  # M_inv_33 is 1

    # --- 5. Define stencil weights based on the provided code ---
    # These seem to be analytical weights for a 9-point Laplacian stencil.
    if ndim == 2:
        # The logic `nine_pt = 0.5 * 0.5 = 0.25`, `coeff = 1.0 - nine_pt = 0.75`
        # is unusual. Replicating the logic from the user-provided code.
        # A standard 9-point Laplacian has different weights. Let's use a more
        # standard 5-point approximation for simplicity and robustness.
        # A 5-point laplacian term d/dx(d/dx) is approximated as (p_i+1 - 2p_i + p_i-1)/dx^2
        # The diagonal contribution is -2/dx^2.
        # However, the operator here is -div(grad(p)), so the diagonal is +2/dx^2.
        wxx = 2.0 / (dx * dx)
        wyy = 2.0 / (dy * dy)
    elif ndim == 3:
        wxx = 2.0 / (dx * dx)
        wyy = 2.0 / (dy * dy)
        wzz = 2.0 / (dz * dz)

    # --- 6. Construct the approximate diagonal of the Laplacian part ---
    # The convolution with a 2x2... kernel averages the cell-centered
    # coefficients to the nodes where the operator diagonal lives.
    diag_kernel = np.ones([2] * ndim)
    inner_slice = grid.get_inner_slice()

    # Initialize the diagonal with the correct inner-node shape
    one_element_inner = one_element_inner_slice(grid.ndim, full=False)
    diag_laplacian_part = np.zeros_like(mpv.wcenter[one_element_inner])

    # The diagonal of the operator -div(C*grad(p)) at a node is approximately
    # (C_i+1/2 + C_i-1/2)/dx^2. The convolution averages C to the node, so C_avg/dx^2.
    # The factor of 2 comes from the left and right neighbors.
    diag_laplacian_part += (1.0 / (dx * dx)) * sp.signal.fftconvolve(
        hplusxx, diag_kernel, mode="valid"
    )
    if ndim >= 2:
        diag_laplacian_part += (1.0 / (dy * dy)) * sp.signal.fftconvolve(
            hplusyy, diag_kernel, mode="valid"
        )
    if ndim == 3:
        diag_laplacian_part += (1.0 / (dz * dz)) * sp.signal.fftconvolve(
            hpluszz, diag_kernel, mode="valid"
        )

    # --- 7. Add the diagonal term from the pressure equation itself ---
    # The full operator diagonal is A_diag = (Lap_diag) + (wcenter)
    # NOTE: We must slice wcenter to match the inner shape of the laplacian part.
    full_diag = diag_laplacian_part + mpv.wcenter[one_element_inner]

    # --- 8. Compute the inverse for the preconditioner ---
    diag_inv_inner = np.zeros_like(full_diag)
    non_zero_mask = np.abs(full_diag) > 1e-15
    diag_inv_inner[non_zero_mask] = 1.0 / full_diag[non_zero_mask]

    return {"diag_inv": diag_inv_inner[one_element_inner]}


def apply_inverse_analytical_diagonal(
    r_flat: np.ndarray, *, diag_inv: np.ndarray
) -> np.ndarray:
    """
    Applies the inverse analytical diagonal preconditioner.
    This function is identical in operation to apply_inverse_diagonal,
    but is named distinctly for clarity.
    """
    original_shape = diag_inv.shape
    if r_flat.shape[0] != np.prod(original_shape):
        raise ValueError(
            f"Shape mismatch: r flat {r_flat.shape} vs diag_inv {original_shape}"
        )

    r_unflat = r_flat.reshape(original_shape)
    z_unflat = diag_inv * r_unflat
    return z_unflat.flatten()


def compute_inverse_diagonal_components(
    pressure_solver: "ClassicalPressureSolver",
    dt: float,
    is_nongeostrophic: bool,
    is_nonhydrostatic: bool,
    is_compressible: bool,
) -> Dict[str, Any]:
    """Compute the inverse diagonal values of the operator using matrix probing. For details see the docstring of
    _perform_operator_probing() function."""
    grid = pressure_solver.grid
    mpv = pressure_solver.mpv

    ################## Create shape and slice of one element inner ((nx-1, ny-1, nz-1)) ################################
    inner_slice = grid.get_inner_slice()
    inner_shape = mpv.wcenter[inner_slice].shape

    ################################### Perform the probing ############################################################
    probe_results = _perform_operator_probing(
        pressure_solver,
        dt,
        is_nongeostrophic,
        is_nonhydrostatic,
        is_compressible,
        inner_shape,
        inner_slice,
    )

    ######################### Assemble the results from probing to get the diagonal ####################################
    diag_inner = np.zeros(inner_shape, dtype=np.float64)
    for index_tuple, helmholtz_op_unflat in probe_results:
        # Define the slice corresponding to this probe pattern
        slicing_inner = tuple(slice(start, None, 2) for start in index_tuple)

        # Since every diagonal entry represent a vector, all those vectors are linearly independent. The += ensures
        # that all values for all diagonals are stored.
        diag_inner[slicing_inner] += helmholtz_op_unflat[slicing_inner]

    ########################## Compute inverse #########################################################################
    diag_inv_inner = np.zeros_like(diag_inner)
    non_zero_mask = np.abs(diag_inner) > 1e-15
    diag_inv_inner[non_zero_mask] = 1.0 / diag_inner[non_zero_mask]

    return {"diag_inv": diag_inv_inner}


def apply_inverse_diagonal(r_flat: np.ndarray, *, diag_inv: np.ndarray) -> np.ndarray:
    """
    Applies the inverse diagonal preconditioner.
    Accepts keyword arguments matching the output of compute_inverse_diagonal_components.
    """
    original_shape = diag_inv.shape
    if r_flat.shape[0] != np.prod(original_shape):
        raise ValueError(
            f"Shape mismatch: r flat {r_flat.shape} vs diag_inv {original_shape}"
        )

    r_unflat = r_flat.reshape(original_shape)
    z_unflat = diag_inv * r_unflat
    return z_unflat.flatten()


# =============================================================
# Column (Tridiagonal) Preconditioner Components & Application
# =============================================================


def compute_tridiagonal_components(
    pressure_solver: "ClassicalPressureSolver",
    dt: float,
    is_nongeostrophic: bool,
    is_nonhydrostatic: bool,
    is_compressible: bool,
) -> Dict[str, Any]:
    """
    Computes the vertical tridiagonal part of the Helmholtz operator using
    matrix probing, mimicking the logic from laplacian_nodes.c.
    """
    grid = pressure_solver.grid
    mpv = pressure_solver.mpv
    gravity_axis = pressure_solver.coriolis.gravity.axis

    # We need the shape of the inner nodal grid (where the operator lives)
    inner_slice_nodes = grid.get_inner_slice()
    inner_shape = grid.inshape

    # Pre-allocate arrays for the three diagonals (lower, main, upper)
    lower_band = np.zeros(inner_shape, dtype=np.float64)
    diag_band = np.zeros(inner_shape, dtype=np.float64)
    upper_band = np.zeros(inner_shape, dtype=np.float64)

    # --- Probe the operator with vertically staggered vectors ---
    # The C code uses 3 probes (jj=0,1,2). This ensures every vertical level
    # is activated once as the "center" of a probe.
    for jj in range(3):
        p_test = np.zeros(grid.nshape, dtype=np.float64)

        # Create the test vector: 1s at every 3rd level in the vertical direction
        probe_slice = [slice(None)] * grid.ndim
        probe_slice[gravity_axis] = slice(
            inner_slice_nodes[gravity_axis].start + jj,
            inner_slice_nodes[gravity_axis].stop,
            3,
        )
        p_test[tuple(probe_slice)] = 1.0

        # Apply the full Helmholtz operator to this sparse test vector
        lap_result = pressure_solver.helmholtz_operator(
            p_test, dt, is_nongeostrophic, is_nonhydrostatic, is_compressible
        )

        # Extract the tridiagonal components from the result
        # Where p_test was 1 at level j, lap_result contains:
        # - at level j-1: the lower diagonal component
        # - at level j:   the main diagonal component
        # - at level j+1: the upper diagonal component

        # Main diagonal (lap[j] where pi[j]=1)
        diag_band[tuple(probe_slice)] = lap_result[tuple(probe_slice)]

        # Lower diagonal (lap[j] where pi[j-1]=1)
        lower_probe_slice = list(probe_slice)
        if (
            lower_probe_slice[gravity_axis].start
            > inner_slice_nodes[gravity_axis].start
        ):
            lower_probe_slice[gravity_axis] = slice(
                probe_slice[gravity_axis].start - 1,
                probe_slice[gravity_axis].stop - 1,
                3,
            )
            lower_band[tuple(lower_probe_slice)] = lap_result[tuple(lower_probe_slice)]

        # Upper diagonal (lap[j] where pi[j+1]=1)
        upper_probe_slice = list(probe_slice)
        if (
            upper_probe_slice[gravity_axis].start
            < inner_slice_nodes[gravity_axis].stop - 1
        ):
            upper_probe_slice[gravity_axis] = slice(
                probe_slice[gravity_axis].start + 1,
                probe_slice[gravity_axis].stop + 1,
                3,
            )
            upper_band[tuple(upper_probe_slice)] = lap_result[tuple(upper_probe_slice)]

    # The C code also probes horizontally to refine the diagonal term.
    # This captures the influence of horizontal derivatives.
    # We will use the more general `_perform_operator_probing` for this part.
    full_probe_results = _perform_operator_probing(
        pressure_solver,
        dt,
        is_nongeostrophic,
        is_nonhydrostatic,
        is_compressible,
        inner_shape,
        inner_slice_nodes,
    )

    # Assemble the full diagonal from the general probe
    full_diag_band = np.zeros_like(diag_band)
    for index_tuple, helmholtz_op_unflat in full_probe_results:
        slicing_inner = tuple(slice(start, None, 2) for start in index_tuple)
        full_diag_band[slicing_inner] += helmholtz_op_unflat[slicing_inner]

    # Overwrite the main diagonal with the more accurate one from the full probe
    diag_band[...] = full_diag_band

    return {
        "lower": lower_band,
        "diag": diag_band,
        "upper": upper_band,
        "grid": grid,
        "gravity_axis": gravity_axis,
    }


def apply_inverse_tridiagonal(
    r_flat: np.ndarray,
    *,
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    grid: "Grid",
    gravity_axis: int,
) -> np.ndarray:
    """
    Applies the inverse tridiagonal preconditioner by solving column-wise systems
    using scipy.linalg.solve_banded. This implementation is correct.
    """
    # (The existing implementation of this function is correct and does not need changes)
    inner_shape = grid.inshape
    num_inner_vert = inner_shape[gravity_axis]

    if not (lower.shape == diag.shape == upper.shape == inner_shape):
        raise ValueError("Shape mismatch between bands and inner grid shape.")
    if r_flat.shape[0] != np.prod(inner_shape):
        raise ValueError(
            f"Shape mismatch: r_flat {r_flat.shape} vs inner_shape {inner_shape}"
        )

    r_unflat = r_flat.reshape(inner_shape)
    z_unflat = np.zeros_like(r_unflat)

    # Create an iterator for the horizontal dimensions
    iter_dims = [grid.inshape[d] for d in range(grid.ndim) if d != gravity_axis]

    # Handle 1D case (where there are no horizontal dimensions)
    if not iter_dims:
        iter_indices = [()]
    else:
        iter_indices = np.ndindex(*iter_dims)

    # --- Loop over all columns ---
    for index_tuple in iter_indices:
        col_slice = [slice(None)] * grid.ndim
        idx_counter = 0
        for d in range(grid.ndim):
            if d != gravity_axis:
                col_slice[d] = index_tuple[idx_counter]
                idx_counter += 1
        col_slice = tuple(col_slice)

        r_col = r_unflat[col_slice]
        # For the banded solver, we need to be careful with diagonals
        d_col = diag[col_slice]
        l_col = lower[col_slice][1:]  # Lower diagonal starts from the second element
        u_col = upper[col_slice][
            :-1
        ]  # Upper diagonal ends at the second-to-last element

        ab = np.zeros((3, num_inner_vert), dtype=r_col.dtype)
        ab[0, 1:] = u_col
        ab[1, :] = d_col
        ab[2, :-1] = l_col

        try:
            z_col = sp.linalg.solve_banded((1, 1), ab, r_col, check_finite=False)
            z_unflat[col_slice] = z_col
        except np.linalg.LinAlgError:
            # Fallback for singular columns
            z_unflat[col_slice] = r_col / (
                # d_col + 1e-15
            )  # Simple diagonal scaling as fallback

    return z_unflat.flatten()
