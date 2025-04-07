"""This module contains functions that do specific discrete jobs on the nodes or cells."""

import numpy as np
from typing import TYPE_CHECKING, List, Tuple
import itertools as it

import scipy as sp

from atmpy.physics.thermodynamics import Thermodynamics

if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
    from atmpy.variables.variables import Variables
    from atmpy.variables.multiple_pressure_variables import MPV
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager
from atmpy.infrastructure.enums import VariableIndices as VI

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
