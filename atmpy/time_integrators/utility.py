"""This module contains functions that do specific discrete jobs on the nodes or cells."""

import numpy as np
from typing import TYPE_CHECKING, List, Tuple
import itertools as it

import scipy as sp

from atmpy.physics.thermodynamics import Thermodynamics

if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
    from atmpy.variables.variables import Variables
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager
from atmpy.infrastructure.enums import VariableIndices as VI
from atmpy.grid.utility import nodal_derivative


def pressured_momenta_divergence(grid: "Grid", variables: "Variables") -> np.ndarray:
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


def nodal_variable_gradient(p: np.ndarray, ndim: int, dxyz: List[float]):
    """
    Taken from reference. See Notes.
    Calculate the discrete gradient of a given scalar field in 1D, 2D, or 3D. The algorithm mimics the calculation of
    nodal pressure gradient specified in eq. (30a) in BK19 paper.


    Parameters
    ----------
    p : np.ndarray of shape (nx+1, [ny+1], [nz+1])
        The nodal scalar field on which the gradient is applied.
    ndim : int
        The number of dimensions of the scalar field (1, 2, or 3).
    dxyz : List[float]
        The list of discretization fineness [dx, (dy), (dz)]

    Returns
    -------
    Tuple[np.ndarray, ...] of shape (nx, ny, nz)
        The gradient components (Dpx, Dpy, Dpz). For ndim < 3, unused components are zero.
        The gradient is defined on cells.

    Notes
    -----
    Taken from https://github.com/ray-chew/pyBELLA/blob/develop/src/dycore/physics/low_mach/second_projection.py
    """
    dx, dy, dz = dxyz

    # Compute the slices for differencing (for example p[:-1] - p[1:])
    indices = [idx for idx in it.product([slice(0, -1), slice(1, None)], repeat=ndim)]
    if ndim == 1:
        # In 1D, gradient is (p[1:] - p[:-1]) / dx (centered difference)
        signs_x: Tuple[float, ...] = (-1.0, +1.0)
        signs_y: Tuple[float, ...] = (0.0, 0.0)
        signs_z: Tuple[float, ...] = (0.0, 0.0)
        scale: float = 1.0  # No averaging needed in 1D
    if ndim == 2:
        # Compute the sign factors of each neighboring cell to the center of calculation
        # Basically in 2D we have for example in x-direction:
        # Dpx = (-p00 - p01 + p10 + p11) * 0.5 / dx
        signs_x: Tuple[float, ...] = (-1.0, -1.0, +1.0, +1.0)
        signs_y: Tuple[float, ...] = (-1.0, +1.0, -1.0, +1.0)
        signs_z: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
        scale: float = 0.5
    elif ndim == 3:
        # Compute the sign factors of each neighboring cell to the center of calculation
        # Basically in 3D we have for example in x-direction:
        # Dpx = (-p000 - p001 - p010 - p011 + p100 + p101 + p110 + p111) * 0.25 / dx
        signs_x: Tuple[float, ...] = (-1.0, -1.0, -1.0, -1.0, +1.0, +1.0, +1.0, +1.0)
        signs_y: Tuple[float, ...] = (-1.0, -1.0, +1.0, +1.0, -1.0, -1.0, +1.0, +1.0)
        signs_z: Tuple[float, ...] = (-1.0, +1.0, -1.0, +1.0, -1.0, +1.0, -1.0, +1.0)
        scale: float = 0.25

    Dpx, Dpy, Dpz = 0.0, 0.0, 0.0
    cnt = 0

    # Compute the unfactored gradient
    for index in indices:
        Dpx += signs_x[cnt] * p[index]
        Dpy += signs_y[cnt] * p[index]
        Dpz += signs_z[cnt] * p[index]
        cnt += 1

    Dpx *= scale / dx
    Dpy *= scale / dy
    Dpz *= scale / dz

    return Dpx, Dpy, Dpz


def calculate_dpi_dp(P: np.ndarray, Msq: float) -> float:
    """Calculate the derivative of the Exner pressure (Pi) with respect to the P = rho Theta. This is the left hand side of
    the pressure equation.

    Parameters
    ----------
    P : np.ndarray
        The unphysical pressure variables. P = rho Theta.
    Msq : float
        The mach number squared

    Notes
    -----
    Pi and P are connected via the formula (in the non-dimensionalized equation) Pi = (1/Msq) * P^(gamma - 1) we can
    analytically calculate the derivative of Pi with respect to the P: dpidP = (gamma - 1) * (1/Msq) * P^(gamma - 2).
    This analytical derivative is then passed to the convolution function for averaging.
    """
    th: Thermodynamics = Thermodynamics()
    ndim = P.ndim
    dpi_temp = (1 / Msq) * th.gm1 * (P ** (th.gamma - 2.0))
    averaging_kernel = np.ones([2] * ndim)
    return (
        sp.signal.fftconvolve(dpi_temp, averaging_kernel, mode="valid") / dpi_temp.sum()
    )


if __name__ == "__main__":
    x = np.arange(30).reshape((5, 6))
    print(x)
    dx, dy, dz = nodal_variable_gradient(x, ndim=2, dxyz=[0.1] * 3)
    print(dx.shape, dy.shape, dz.shape)
