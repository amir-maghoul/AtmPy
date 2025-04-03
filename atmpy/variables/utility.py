"""Utility module for variables"""

import numpy as np
from typing import Union, TYPE_CHECKING
from atmpy.physics.thermodynamics import Thermodynamics
if TYPE_CHECKING:
    from atmpy.variables.variables import Variables
from atmpy.infrastructure.enums import HydrostateIndices as HI


def get_left_index_in_all_directions(ndim):
    """ Return the slices for left indices in all dimensions."""
    return [slice(0, -1)] * ndim


def _cumulative_integral(arr: np.ndarray, ghost_idx: int) -> np.ndarray:
    """Helper: cumulative integration that applies a reverse cumsum for indices below ghost."""
    result = np.empty_like(arr)
    result[:ghost_idx] = np.flip(
        np.cumsum(np.flip(arr[:ghost_idx], axis=0), axis=0), axis=0
    )
    result[ghost_idx:] = np.cumsum(arr[ghost_idx:], axis=0)
    return result


def compute_stratification(
    hydrostate: "Variables",
    Y: np.ndarray,
    Y_n: np.ndarray,
    axis: int,
    gravity_strength: Union[np.ndarray, list],
    Msq: float,
) -> None:
    """Compute the stratification of pressure variables for constructing the hydrostatic state.
    This function computes the stratification of the hydrostatic background by integrating an inverse variable (S = 1/Y)
    along the specified (gravity) axis. The integration is performed after moving the chosen axis to the front (axis 0)
    and then the results are moved back. The cell‐ and node–based hydro-state variables are updated in-place.

    Parameters
    ----------
    hydrostate : Variables
        The container for hydrostatic cell and node variables.
    Y : np.ndarray
        Cell-centered stratification variable (e.g. potential temperature).
    Y_n : np.ndarray
        Node-centered stratification variable (e.g. pressure variable).
    axis : int
        The axis along which stratification (integration) is performed.
    gravity_strength : Union[np.ndarray, list]
        Gravity strength vector—for the integration only the component along the chosen axis is used.
    Msq : float
        Mach number squared.
    """
    # Instantiate thermodynamics (assumed fast)
    th = Thermodynamics()
    grid = hydrostate.grid
    dr = grid.dxyz[axis]

    # 'ghost' is the number of ghost cells on the bottom side; this will be used to split integration
    ghost = grid.ng[axis][0]

    # Move the integration axis to the front for easier vectorized operations.
    Y_axis0 = np.moveaxis(Y, axis, 0)
    Y_n_axis0 = np.moveaxis(Y_n, axis, 0)
    hydrostate_cells_vars_axis0 = np.moveaxis(hydrostate.cell_vars, axis, 0)
    hydrostate_nodes_vars_axis0 = np.moveaxis(hydrostate.node_vars, axis, 0)

    # Computing reference values
    rhoY0 = 1.0
    p0 = rhoY0**th.gamma
    pi0 = rhoY0**th.gm1

    # Compute the primary integrand arrays:
    # S_p is cell based (1/Y)
    S_p = 1.0 / Y_axis0

    # S_m represents mid‐point/interpolated values; note the indices [ghost-1:ghost+1] use the node–centered value.
    S_m = np.empty_like(S_p)
    S_m[ghost - 1 : ghost + 1] = 1.0 / np.expand_dims(Y_n_axis0[ghost], axis=0)
    S_m[0] = 1.0 / Y_axis0[ghost - 1]
    S_m[ghost + 1 :] = 1.0 / Y_axis0[ghost:-1]

    # Precompute the integration spacing vector 'd' along the moved axis.
    n_str = Y_axis0.shape[0]
    if n_str < 3:
        raise ValueError(
            "Insufficient points along the integration axis (need at least 3)."
        )
    # Instead of concatenating arrays, we preallocate and then set special values.
    d_values = np.full(n_str, dr, dtype=Y_axis0.dtype)
    d_values[0] = -dr
    d_values[1] = -dr / 2
    d_values[2] = dr / 2
    # Reshape for broadcasting across the remaining dimensions.
    d = d_values.reshape([n_str] + [1] * (Y_axis0.ndim - 1))

    # Compute the local integrand
    integrand = d * 0.5 * (S_p + S_m)

    # Integrate the primary integrand vertically.
    integrated_val = _cumulative_integral(integrand, ghost)

    rhoY0 = 1.0
    gravity_val = gravity_strength[axis]
    pi0 = rhoY0**th.gm1

    # Compute hydrostatic profiles (cell‐based)
    pi_hydro = pi0 - th.Gamma * gravity_val * integrated_val
    p_hydro = pi_hydro**th.Gammainv
    rhoY_hydro = pi_hydro**th.gm1inv

    # Move results back to original axis
    p_hydro_final = np.moveaxis(p_hydro, 0, axis)
    rhoY_hydro_final = np.moveaxis(rhoY_hydro, 0, axis)
    # S_p_final will be used to compute rho on cells.
    S_p_final = np.moveaxis(S_p, 0, axis)

    # Build index tuple for inner (non-ghost) cells (the same shape is used for node_vars here).
    inner_idx = tuple(slice(0, s) for s in grid.cshape)
    # Update cell-based variables (assumed channel ordering:
    # 0: rho0, 1: p0, 2: p2_0, 3: S0, 4: Y0, 5: rhoY0)
    hydrostate.cell_vars[inner_idx + (HI.RHO0,)] = rhoY_hydro_final * S_p_final
    hydrostate.cell_vars[inner_idx + (HI.P0,)] = p_hydro_final
    # For p2_0, move pi_hydro on the fly and scale by Msq.
    hydrostate.cell_vars[inner_idx + (HI.P2_0,)] = (
        np.moveaxis(pi_hydro, 0, axis)[inner_idx] / Msq
    )
    hydrostate.cell_vars[inner_idx + (HI.S0,)] = S_p_final
    hydrostate.cell_vars[inner_idx + (HI.Y0,)] = 1.0 / S_p_final
    hydrostate.cell_vars[inner_idx + (HI.RHOY0,)] = rhoY_hydro_final

    # ----- Node-based integration -----
    # For the node–based computation, define an analogous integration for Sn, using the same Y_axis0.
    Sn_p = 1.0 / Y_axis0
    # Build the spacing for nodes; for indices below ghost, the spacing is negative.
    d_node = np.full(n_str, dr, dtype=Y_axis0.dtype)
    d_node[:ghost] = -dr
    d_node = d_node.reshape([n_str] + [1] * (Y_axis0.ndim - 1))

    Sn_int = d_node * Sn_p
    Sn_int = _cumulative_integral(Sn_int, ghost)

    # Compute node hydrostatic pressures.
    pi_hydro_n = pi0 - th.Gamma * gravity_val * Sn_int
    rhoY_hydro_n = pi_hydro_n**th.gm1inv

    # Move back to original axis.
    pi_hydro_n_final = np.moveaxis(pi_hydro_n, 0, axis)
    rhoY_hydro_n_final = np.moveaxis(rhoY_hydro_n, 0, axis)

    # Update node-based variables.
    hydrostate.node_vars[inner_idx + (HI.RHO0,)] = rhoY_hydro_n_final
    hydrostate.node_vars[inner_idx + (HI.P0,)] = rhoY_hydro_n_final**th.gamma
    hydrostate.node_vars[inner_idx + (HI.P2_0,)] = pi_hydro_n_final / Msq
    # Here, we use the provided node-based stratification variable Y_n directly.
    hydrostate.node_vars[inner_idx + (HI.S0,)] = 1.0 / Y_n
    hydrostate.node_vars[inner_idx + (HI.Y0,)] = 1.0 / Y_n
    hydrostate.node_vars[inner_idx + (HI.RHOY0,)] = rhoY_hydro_n_final
