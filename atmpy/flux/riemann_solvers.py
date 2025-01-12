import numpy as np
from typing import Tuple
from numba import njit, prange
from atmpy.variables.variables import Variables
from atmpy.data.enums import VariableIndices as VI, PrimitiveVariableIndices as PVI
from atmpy.physics.eos import *
from atmpy.data.enums import *


def roe(left_state: Variables, right_state: Variables, direction: str,  *args, **kwargs):
    raise NotImplementedError(f"Roe solver for {direction}-direction not implemented.")


def hll(left_state: Variables, right_state: Variables, direction: str, *args, **kwargs):
    """
    HLL Riemann solver. Updates the flux container in place.

    Parameters:
    left_state : np.ndarray
        State variables on the left side of the interface.
    right_state : np.ndarray
        State variables on the right side of the interface.
    direction : str
        Spatial direction ('x', 'y', or 'z').
    gamma : float
        Ratio of specific heats (adiabatic index).

    Returns:
    np.ndarray
        Numerical flux array across the interface.
    """

    left_state.to_primitive(eos)
    right_state.to_primitive(eos)

    left_prim = left_state.primitives
    right_prim = right_state.primitives

    shape = left_state.cell_vars.shape

    if direction == "x":
        v_l = left_prim[..., U]
        v_r = right_prim[..., U]
    elif direction == "y":
        v_l = left_prim[..., V]
        v_r = right_prim[..., V]
    elif direction == "z":
        v_l = left_prim[..., W]
        v_r = right_prim[..., W]
    else:
        raise ValueError(f"Unsupported direction '{direction}'")

    rho_l = left_state.cell_vars[..., RHO]
    rho_r = right_state.cell_vars[..., RHO]

    # Compute sound speeds
    cs_l = eos.sound_speed(rho_l, left_prim[..., P])
    cs_r = eos.sound_speed(rho_r, right_prim[..., P])

    return _hll_numba(rho_l, rho_r, v_r, cs_l, cs_r, shape)


from numba import njit
import numpy as np


@njit(fastmath=True)
def _hll_numba(
    rho_l: np.ndarray,
    rho_r: np.ndarray,
    v_l: np.ndarray,
    v_r: np.ndarray,
    cs_l: np.ndarray,
    cs_r: np.ndarray,
    shape: tuple,
) -> np.ndarray:
    flux = np.zeros(shape)
    lambda_hll = (
        np.minimum(v_l - cs_l, v_r - cs_r) + np.maximum(v_l + cs_l, v_r + cs_r)
    ) / 2.0

    for i in range(shape[0]):
        if lambda_hll[i] >= 0:
            flux[i] = rho_l[i] * v_l[i]
        else:
            flux[i] = rho_r[i] * v_r[i]

    return flux


def hllc(left_state: Variables, right_state: Variables, direction: str, *args, **kwargs):
    raise NotImplementedError(f"HLLC solver for {direction}-direction not implemented.")


def rusanov(left_state: Variables, right_state: Variables, direction: str, *args, **kwargs):
    raise NotImplementedError(
        f"Rusanov solver for {direction}-direction not implemented."
    )
