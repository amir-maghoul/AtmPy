""" This module contains the function for different preconditioning functions"""
import numpy as np

from atmpy.infrastructure.utility import one_element_inner_slice
from typing import Callable, TYPE_CHECKING

def diagonal_preconditioner():
    pass

import numpy as np
from typing import TYPE_CHECKING, Dict, Tuple, Callable
import scipy as sp

if TYPE_CHECKING:
    from atmpy.pressure_solver.classical_pressure_solvers import ClassicalPressureSolver

def compute_inverse_diagonal(
    pressure_solver: "ClassicalPressureSolver",
    dt: float,
    is_nongeostrophic: bool,
    is_nonhydrostatic: bool,
    is_compressible: bool,
) -> np.ndarray:
    """
    Computes the inverse diagonal of the Helmholtz operator numerically,
    using the pressure_solver's methods.
    """
    grid = pressure_solver.grid
    mpv = pressure_solver.mpv

    # --- Determine the shape of the inner grid being solved ---
    inner_slice_nodes = one_element_inner_slice(grid.ndim, full=False)
    # Use wcenter's inner shape as the reference for the operator's domain/codomain shape
    inner_shape = mpv.wcenter[inner_slice_nodes].shape

    full_nodal_shape = grid.nshape
    diag_inner = np.zeros(inner_shape, dtype=np.float64) # Store diagonal for inner grid directly
    p_test = np.zeros(full_nodal_shape, dtype=np.float64)

    # --- Loop inspired by BK19's ii, jj, kk loops ---
    indices = list(np.ndindex(*([2] * grid.ndim)))

    for index_tuple in indices:
        p_test.fill(0.0)
        # Create slicing like [ii::2, jj::2, kk::2] relative to the *inner* grid
        slicing_inner = tuple(slice(start, None, 2) for start in index_tuple)

        # Map inner slicing to full nodal grid slicing
        # We need to place 1s only at the inner nodes corresponding to slicing_inner
        inner_nodes_mask = np.zeros(inner_shape, dtype=bool)
        inner_nodes_mask[slicing_inner] = True
        p_test[inner_slice_nodes][inner_nodes_mask] = 1.0 # Place 1s in p_test

        # --- Compute Action of Helmholtz Operator (A * p_test) ---
        # Call the existing helmholtz_operator, which returns the unflattened result
        # corresponding to the inner grid points.
        helmholtz_op_unflat = pressure_solver.helmholtz_operator(
            p_test, dt, is_nongeostrophic, is_nonhydrostatic, is_compressible
        )

        # --- Extract diagonal entries ---
        # Where p_test was 1 (within the inner domain), store the operator result
        # The result helmholtz_op_unflat already corresponds to the inner domain.
        diag_inner[slicing_inner] += helmholtz_op_unflat[slicing_inner]
        # We use += because different iterations (index_tuple) fill different parts
        # of the diagonal. (If ii=0,jj=0 hits index 0, ii=1,jj=0 hits index 1, etc.)
        # BK19 effectively does this by storing into a larger diag array and then selecting.
        # Here we build the inner diagonal directly.

    # --- Compute inverse, handle potential zeros ---
    diag_inv_inner = np.zeros_like(diag_inner)
    non_zero_mask = np.abs(diag_inner) > 1e-15
    diag_inv_inner[non_zero_mask] = 1.0 / diag_inner[non_zero_mask]

    return diag_inv_inner

def apply_inverse_diagonal(diag_inv: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Applies the inverse diagonal preconditioner.
    """
    original_shape = diag_inv.shape
    if r.shape[0] != np.prod(original_shape):
         raise ValueError(f"Shape mismatch: r flat {r.shape} vs diag_inv {original_shape}")

    r_unflat = r.reshape(original_shape)
    z_unflat = diag_inv * r_unflat
    return z_unflat.flatten()