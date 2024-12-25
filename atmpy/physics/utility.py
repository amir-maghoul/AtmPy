import numpy as np
from numba import njit, prange


@njit(parallel=True)
def exner_to_pressure_numba(pi: np.ndarray, p_ref: float, cp: float, R: float):
    """
    Convert Exner pi to real pressure using numba.

    Parameters
    ----------
    pi : np.ndarray
        the array containing the Exner pressure
    p_ref : float
        the reference pressure
    cp : float
        the heat capacity in constant pressure
    R : float
        cp - cv

    Return
    ------
    np.ndarray
        the array containing the real pressure
    """
    n = len(pi)
    p_out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        p_out[i] = p_ref * pi[i] ** (cp / R)
    return p_out


@njit(parallel=True)
def P_to_pressure_numba(P: np.ndarray, p_ref: float, cp: float, cv: float):
    """
    convert P (= rho*Theta) to pressure.
    P = (p_ref / R)*(p / p_ref)^(cv/cp).
    Solving for p => p = p_ref * [ (P*R)/p_ref ]^(cp/cv).

    Parameters
    ----------
    pi : np.ndarray
        the array containing the Exner pressure
    p_ref : float
        the reference pressure
    cp : float
        the heat capacity in constant pressure
    cv : float
        the heat capacity in constant volume

    Returns
    -------
    np.ndarray
        the array containing the real pressure


    """
    n = len(P)
    R = cp - cv
    p_out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        fac = P[i] * R / p_ref
        p_out[i] = p_ref * (fac ** (cp / cv))
    return p_out


@njit(parallel=True)
def exner_sound_speed_numba(rho_arr, p_arr, gamma):
    """
    Calculate sound speed using numba for the ExnerBasedEOS.
    If 'rho' is zero, returning NaN is mathematically consistent (division by zero).
    Whether you treat sound speed as zero or NaN depends on your physical / solver constraints.

    Parameters
    ----------
    rho_arr : np.ndarray
        the array containing the density
    p_arr : np.ndarray
        the array containing the pressure
    gamma : float
        the heat capacity in constant pressure
    """
    n = len(rho_arr)
    a_out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        if rho_arr[i] <= 0.0:
            # Option 1: set to 0.0
            # a_out[i] = 0.0
            # Option 2: set to NaN (mathematically consistent but might break a solver)
            a_out[i] = np.nan
        else:
            a_out[i] = np.sqrt(gamma * p_arr[i] / rho_arr[i])
    return a_out
