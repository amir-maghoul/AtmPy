""" This module contains functions that do specific discrete jobs on the nodes or cells."""
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
    from atmpy.variables.variables import Variables
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager
from atmpy.infrastructure.enums import VariableIndices as VI

def velocity_divergence(grid:"Grid", variables:"Variables", boundary_manager: "BoundaryManager") -> np.ndarray:
    """ Calculates the divergence of the velocity vector. This works as the right hand side of the pressure equation in
    the euler steps.

    Parameters
    ----------
    grid : Grid
        The grid on which the problem is defined.
    variables : Variables
        The container for the variables.
    boundary_manager : BoundaryManager
        The boundary manager object.

    """

    ndim = grid.ndim
    ng = grid.ng

    # # Enforce boundary conditions (for wall or Rayleigh) in the y–direction if needed.
    # if not hasattr(ud, 'LAMB_BDRY'):
    #     if ud.bdry_type[1] in (BdryType.WALL, BdryType.RAYLEIGH):
    #         # Assuming a 2D or 3D variable cell_vars shape: (ncx_total, ncy_total, [ncz_total,] num_vars).
    #         # Set the first and last two cells in the y–direction to zero for the momentum variables.
    #         variables.cell_vars[..., VI.RHOU][:, :2] = 0.0
    #         variables.cell_vars[..., VI.RHOU][:, -2:] = 0.0
    #         variables.cell_vars[..., VI.RHOV][:, :2] = 0.0
    #         variables.cell_vars[..., VI.RHOV][:, -2:] = 0.0
    #         if ndim == 3:
    #             variables.cell_vars[..., VI.RHOW][:, :2] = 0.0
    #             variables.cell_vars[..., VI.RHOW][:, -2:] = 0.0

    # Compute the weighting factor Y = (rhoY / rho)
    # (Make sure to avoid division by zero in a real implementation)
    Y = variables.cell_vars[..., VI.RHOY] / variables.cell_vars[..., VI.RHO]

    # Compute finite differences along the x-direction (axis 0)
    Ux = np.diff(variables.cell_vars[..., VI.RHOU] * Y, axis=0) / grid.dx
    # Average to center the differences on the interfaces (discard one extra slice)
    Ux = 0.5 * (Ux[:, :-1] + Ux[:, 1:])

    # For one dimensions, we are done.
    if ndim == 1:
        return Ux

    # Compute finite differences along the y-direction (axis 1)
    Vy = np.diff(variables.cell_vars[..., VI.RHOV] * Y, axis=1) / grid.dy
    Vy = 0.5 * (Vy[:-1, :] + Vy[1:, :])

    if ndim == 2:
        rhs = Ux + Vy
        return rhs

    if ndim == 3:
        # For three dimensions, adjust further along the third dimension.
        Ux = -0.5 * (Ux[..., :-1] + Ux[..., 1:])
        Vy = 0.5 * (Vy[..., :-1] + Vy[..., 1:])
        Wz = np.diff(variables.cell_vars[..., VI.RHOW] * Y, axis=2) / grid.dz
        Wz = 0.5 * (Wz[:-1, ...] + Wz[1:, ...])
        # A second average in the y-direction (or x–direction) for proper centering.
        Wz = 0.5 * (Wz[:, :-1, ...] + Wz[:, 1:, ...])
        # Fill the interior nodes of rhs with the divergence.
        rhs[1:-1, 1:-1, 1:-1] = Ux + Vy + Wz
    return rhs