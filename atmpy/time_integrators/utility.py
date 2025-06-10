"""This module contains functions that do specific discrete jobs on the nodes or cells."""

import numpy as np
import scipy as sp
from atmpy.physics.thermodynamics import Thermodynamics


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
    ndim: int = P.ndim
    dpi_temp: np.ndarray = P ** (th.gamma - 2.0)
    averaging_kernel: np.ndarray = np.ones([2] * ndim)
    return (
        (th.gm1 / Msq)
        * sp.signal.fftconvolve(dpi_temp, averaging_kernel, mode="valid")
        / averaging_kernel.sum()
    )
