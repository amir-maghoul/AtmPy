import numpy as np
from numba import njit, prange

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def exner_to_pressure_numba(
    pi: np.ndarray, p_ref: float, cp: float, R: float
) -> np.ndarray:
    """
    Convert Exner pressure (pi) to real pressure using Numba.

    Parameters
    ----------
    pi : np.ndarray
        The array containing the Exner pressure.
    p_ref : float
        The reference pressure.
    cp : float
        The heat capacity at constant pressure.
    R : float
        The gas constant, defined as cp - cv.

    Returns
    -------
    np.ndarray
        The array containing the real pressure.
    """
    p_out = np.empty_like(pi, dtype=np.float64)
    total_elements = pi.size

    for idx in prange(total_elements):
        # Access the element using flat indexing to support multi-dimensional arrays
        pi_val = pi.flat[idx]

        # Optional: Handle non-positive Exner pressure values to avoid invalid computations
        # TODO: Uncomment the following code to avoid negative pressure
        # if pi_val <= 0.0:
        #     p_out.flat[idx] = 0.0  # or some default value or raise an error
        # else:
        p_out.flat[idx] = p_ref * (pi_val ** (cp / R))

    return p_out


@njit(parallel=True)
def P_to_pressure_numba(
    P: np.ndarray, p_ref: float, cp: float, cv: float
) -> np.ndarray:
    """
    Convert unphysical pressure (P) to real pressure (p).

    P = (p_ref / R) * (p / p_ref)^(cv/cp)
    Solving for p => p = p_ref * [ (P * R)/p_ref ]^(cp/cv).

    Parameters
    ----------
    P : np.ndarray
        The array containing the Exner pressure.
    p_ref : float
        The reference pressure.
    cp : float
        The heat capacity at constant pressure.
    cv : float
        The heat capacity at constant volume.

    Returns
    -------
    np.ndarray
        The array containing the real pressure.
    """
    R = cp - cv
    p_out = np.empty_like(P, dtype=np.float64)
    total_elements = P.size

    for idx in prange(total_elements):
        fac = P.flat[idx] * R / p_ref
        p_out.flat[idx] = p_ref * (fac ** (cp / cv))

    return p_out


import numpy as np
from numba import njit, prange


@njit(parallel=True)
def exner_sound_speed_numba(
    rho_arr: np.ndarray, p_arr: np.ndarray, gamma: float
) -> np.ndarray:
    """
    Calculate sound speed using Numba for the Exner-Based Equation of State (EOS).

    The sound speed is calculated as:
        a = sqrt(gamma * p / rho)

    If 'rho' is zero or negative, the sound speed is set to NaN to avoid division by zero or invalid values.
    Alternatively, you can set it to zero based on your physical or solver constraints.

    Parameters
    ----------
    rho_arr : np.ndarray
        The array containing the density (units: kg/m³).
    p_arr : np.ndarray
        The array containing the pressure (units: Pa).
    gamma : float
        The adiabatic index (dimensionless), typically the ratio of specific heats (cp/cv).

    Returns
    -------
    np.ndarray
        The array containing the calculated sound speed (units: m/s). The shape matches `rho_arr` and `p_arr`.

    Notes
    -----
    - Both `rho_arr` and `p_arr` must have the same shape.
    - The function handles multi-dimensional arrays (1D, 2D, 3D, etc.) seamlessly.
    - Ensure that `rho_arr` and `p_arr` are of floating-point types for accurate computations.
    """
    # Verify that rho_arr and p_arr have the same total number of elements
    # Note: Numba does not support explicit shape checks, so ensure this externally if necessary

    a_out = np.empty_like(rho_arr, dtype=np.float64)
    total_elements = rho_arr.size

    for idx in prange(total_elements):
        rho_val = rho_arr.flat[idx]
        p_val = p_arr.flat[idx]

        if rho_val <= 0.0:
            # TODO: Handle division by zero
            # Option 1: Set to NaN (mathematically consistent but might break a solver)
            a_out.flat[idx] = np.nan

            # TODO: Handle division by zero
            # Option 2: Set to 0.0 (uncomment the following line if preferred)
            # a_out.flat[idx] = 0.0
        else:
            a_out.flat[idx] = np.sqrt(gamma * p_val / rho_val)

    return a_out
