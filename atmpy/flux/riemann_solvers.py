from typing import Tuple
import numpy as np
from numba import njit
from atmpy.variables.variables import Variables
from atmpy.data.enums import VariableIndices as VI, PrimitiveVariableIndices as PVI
from atmpy.flux.utility import directional_indices


def modified_hll(
    left_state: np.ndarray,
    right_state: np.ndarray,
    *args,
    **kwargs,
):
    """Computes the flux using a modified HLL solver from the left and right states.

    Parameters
    ----------
    left_state: np.ndarray
        The array of primitive variables for the left state
    right_state: np.ndarray
        The array of primitive variables for the right state.

    Notes
    -----
    To have a uniform signature for all the riemann solvers, only the left and right states are passed as
    positional arguments. In modified HLL, the order of other arguments are as follows:

    iflux = arg[0]
    direction = arg[1]
    ndim = arg[2]
    """

    iflux = args[0]
    direction = args[1]
    ndim = args[2]

    left_idx, right_idx, directional_inner_idx, _ = directional_indices(
        ndim, direction, full=False
    )

    upwind = 0.5 * (1.0 + np.sign(iflux))
    upl = upwind[right_idx]
    upr = 1.0 - upwind[left_idx]

    # Compute U = sigma U- + (1-sigma) U+:
    # -------------------------------------
    # -------------------------------------

    # Initialize factors
    # left_factor = np.zeros_like(iflux)
    # right_factor = np.zeros_like(iflux)

    # Compute the ADVECTING flux: Pu/Theta = rho*Theta*u/Theta = rho*u
    left_factor = iflux[left_idx] * upl / left_state[..., PVI.Y][left_idx]
    right_factor = iflux[right_idx] * upr / right_state[..., PVI.Y][right_idx]

    # Compute the ADVECTED values
    flux_variables = np.zeros_like(left_state)
    print("shape of left factor = ", left_factor.shape)
    print("shape of flux_variables = ", flux_variables.shape)
    print("shape of left state = ", left_state.shape)
    flux_variables[..., VI.RHO] = left_factor + right_factor
    flux_variables[..., 1:] = (
        left_factor[..., np.newaxis] * left_state[..., 1:][left_idx]
        + right_factor[..., np.newaxis] * right_state[..., 1:][right_idx]
    )

    # TODO: SHAPE MISMATCH ABOVE

    return flux_variables


def roe(left_state: Variables, right_state: Variables, direction: str, *args, **kwargs):
    raise NotImplementedError(f"Roe solver for {direction}-direction not implemented.")


def hll(left_state: Variables, right_state: Variables, direction: str, *args, **kwargs):
    pass


def hllc(
    left_state: Variables, right_state: Variables, direction: str, *args, **kwargs
):
    raise NotImplementedError(f"HLLC solver for {direction}-direction not implemented.")


def rusanov(
    left_state: Variables, right_state: Variables, direction: str, *args, **kwargs
):
    raise NotImplementedError(
        f"Rusanov solver for {direction}-direction not implemented."
    )
