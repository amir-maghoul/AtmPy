"""This module contains functions that do specific discrete jobs on the nodes or cells."""

import numpy as np
from typing import TYPE_CHECKING

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
