"""This module contains functions that do specific discrete jobs on the nodes or cells."""

import numpy as np
from typing import TYPE_CHECKING, List, Tuple
import itertools as it

if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
    from atmpy.variables.variables import Variables
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager
from atmpy.infrastructure.enums import VariableIndices as VI
from atmpy.grid.utility import nodal_derivative


def momenta_divergence(grid: "Grid", variables: "Variables") -> np.ndarray:
    """Calculates the divergence of the pressured-momenta vector (Pu, Pv, Pw). This works as the right hand side
    of the pressure equation in the euler steps.

    Parameters
    ----------
    grid : Grid
        The grid on which the problem is defined.
    variables : Variables
        The container for the variables.

    Returns
    -------
    np.ndarray of shape (nx-1, [ny-1], [nz-1])
        The divergence of the velocity vector on the nodes.

    Notes
    -----
    This function calculates the divergence of the pressured-momenta vector on the nodes. The result can fill
    the inner nodes of a nodal variable as the result is of shape (nx-1, [ny-1], [nz-1]).
    """

    ndim = grid.ndim

    # Compute the weighting factor Y = (rhoY / rho)
    Y = variables.cell_vars[..., VI.RHOY] / variables.cell_vars[..., VI.RHO]

    # Derivative of u in x.
    Ux = nodal_derivative(
        variables.cell_vars[..., VI.RHOU] * Y, ndim, axis=0, ds=grid.dx
    )

    # For one dimensions, we are done.
    if ndim == 1:
        return Ux

    # Derivative of v in y.
    Vy = nodal_derivative(
        variables.cell_vars[..., VI.RHOV] * Y, ndim, axis=1, ds=grid.dy
    )
    if ndim == 2:
        return Ux + Vy

    # Derivative of w in z.
    Wz = nodal_derivative(
        variables.cell_vars[..., VI.RHOW] * Y, ndim, axis=2, ds=grid.dz
    )
    if ndim == 3:
        return Ux + Vy + Wz
    else:
        raise ValueError("Unsupported dimension {}".format(ndim))


def nodal_gradient(p: np.ndarray, ndim: int, dxyz: List[float]):
    """
    Taken from reference. See Notes.
    Calculate the discrete gradient of a given scalar field.

    Parameters
    ----------
    p : np.ndarray
        The scalar field on which the gradient is applied.
    ndim : int
        The number of dimensions of the scalar field.
    dxyz : List[float]
        The list of discretization fineness

    Returns
    -------
    np.ndarray

    Notes
    -----
    Taken from https://github.com/ray-chew/pyBELLA/blob/develop/src/dycore/physics/low_mach/second_projection.py
    """
    dx, dy, dz = dxyz

    # Compute the slices for differencing (for example p[:-1] - p[1:])
    indices = [idx for idx in it.product([slice(0, -1), slice(1, None)], repeat=ndim)]
    if ndim == 2:
        # Compute the sign factors of each neighboring cell to the center of calculation
        # Basically in 2D we have for example in x-direction:
        # Dpx = (-p00 - p01 + p10 + p11) * 0.5 / dx
        signs_x: Tuple[float, ...] = (-1.0, -1.0, +1.0, +1.0)
        signs_y: Tuple[float, ...] = (-1.0, +1.0, -1.0, +1.0)
        signs_z: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    elif ndim == 3:
        # Compute the sign factors of each neighboring cell to the center of calculation
        # Basically in 3D we have for example in x-direction:
        # Dpx = (-p000 - p001 - p010 - p011 + p100 + p101 + p110 + p111) * 0.25 / dx
        signs_x: Tuple[float, ...] = (-1.0, -1.0, -1.0, -1.0, +1.0, +1.0, +1.0, +1.0)
        signs_y: Tuple[float, ...] = (-1.0, -1.0, +1.0, +1.0, -1.0, -1.0, +1.0, +1.0)
        signs_z: Tuple[float, ...] = (-1.0, +1.0, -1.0, +1.0, -1.0, +1.0, -1.0, +1.0)

    Dpx, Dpy, Dpz = 0.0, 0.0, 0.0
    cnt = 0

    # Compute the unfactored gradient
    for index in indices:
        Dpx += signs_x[cnt] * p[index]
        Dpy += signs_y[cnt] * p[index]
        Dpz += signs_z[cnt] * p[index]
        cnt += 1

    Dpx *= 0.5 ** (ndim - 1) / dx
    Dpy *= 0.5 ** (ndim - 1) / dy
    Dpz *= 0.5 ** (ndim - 1) / dz

    return Dpx, Dpy, Dpz
