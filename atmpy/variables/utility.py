"""Utility module for variables"""

import numpy as np
from typing import Union, List
from atmpy.physics.thermodynamics import Thermodynamics
from atmpy.grid.kgrid import Grid
from atmpy.variables.variables import Variables
from atmpy.infrastructure.enums import HydrostateIndices as HI


def get_left_index_in_all_directions(ndim):
    return [slice(0, -1)] * ndim


# def compute_stratification(
#     hydrostate: Variables,
#     Y: np.ndarray,
#     Y_n: np.ndarray,
#     grid: Grid,
#     axis: int,
#     gravity_strength: Union[np.ndarray, list],
#     Msq: float,
# ):
#     """Compute the stratification of pressure variables
#
#     Parameters
#     ----------
#     hydrostate: Variables
#         The hydrostatic pressure container as Variables which contains both nodal and cellular pressures.
#     Y: np.ndarray
#         The cell centered stratification variable (potential temperature)
#     Y_n: np.ndarray
#         The node centered stratification variable (pressure variable)
#     grid: Grid
#         The grid on which to compute the stratification
#     axis: int
#         The axis along which to compute the stratification
#     gravity_strength: np.ndarray or List of shape (3,)
#         the array of gravity strengths in each direction
#     Msq: float
#         Mach number squared
#
#     """
#     th: Thermodynamics = Thermodynamics()
#     dr: float = grid.dxyz[axis]
#     ig: int = grid.ng[axis][0]  # The bottom side of the gravity axis is considered
#
#     # Move the stratification axis to axis 0.
#     Y_moved: np.ndarray = np.moveaxis(Y, axis, 0)
#     Y_n_moved: np.ndarray = np.moveaxis(Y_n, axis, 0)
#
#     # Primary integrand.
#     S_p: np.ndarray = 1.0 / Y_moved
#     S_m: np.ndarray = np.empty_like(S_p)
#     S_m[ig - 1 : ig + 1] = 1.0 / np.expand_dims(Y_n_moved[ig], axis=0)
#     S_m[0] = 1.0 / Y_moved[ig - 1]
#     S_m[ig + 1 :] = 1.0 / Y_moved[ig:-1]
#
#     n_str: int = Y_moved.shape[0]
#     d: np.ndarray = np.concatenate(
#         (np.array([-dr, -dr / 2, dr / 2]), np.ones(n_str - 3) * dr)
#     )
#     newshape: List[int] = [n_str] + [1] * (Y_moved.ndim - 1)
#     d = d.reshape(newshape)
#     integrand = d * 0.5 * (S_p + S_m)
#     integrand[:ig] = np.flip(np.cumsum(np.flip(integrand[:ig], axis=0), axis=0), axis=0)
#     integrand[ig:] = np.cumsum(integrand[ig:], axis=0)
#
#     Gamma = th.gm1 / th.gamma
#     Gamma_inv = 1.0 / Gamma
#     gamm = th.gamma
#     gm1 = th.gm1
#     gm1_inv = 1.0 / gm1
#
#     rhoY0 = 1.0
#     gravity = gravity_strength[axis]
#     p0_ref = rhoY0**gamm
#     pi0 = rhoY0**gm1
#
#     pi_hydro = pi0 - Gamma * gravity * integrand
#     p_hydro = pi_hydro**Gamma_inv
#     rhoY_hydro = pi_hydro**gm1_inv
#
#     p_hydro_final = np.moveaxis(p_hydro, 0, axis)
#     del p_hydro
#     rhoY_hydro_final = np.moveaxis(rhoY_hydro, 0, axis)
#     del rhoY_hydro
#     S_p_final = np.moveaxis(S_p, 0, axis)
#     del S_p
#
#     # Update cell-based variables.
#     inner_cells = tuple(slice(0, s) for s in grid.cshape)
#     # Assuming channel ordering:
#     # channel 0: rho0, 1: p0, 2: p20, 3: S0, 4: Y0, 5: rhoY0.
#     hydrostate.cell_vars[inner_cells + (HI.RHO0,)] = rhoY_hydro_final * S_p_final
#     hydrostate.cell_vars[inner_cells + (HI.P0,)] = p_hydro_final
#     hydrostate.cell_vars[inner_cells + (HI.P2_0,)] = (
#         np.moveaxis(pi_hydro, 0, axis)[inner_cells] / Msq
#     )
#     hydrostate.cell_vars[inner_cells + (HI.S0,)] = S_p_final
#     hydrostate.cell_vars[inner_cells + (HI.Y0,)] = 1.0 / S_p_final
#     hydrostate.cell_vars[inner_cells + (HI.RHOY0,)] = rhoY_hydro_final
#
#     # Node-based integration.
#     Sn_p = 1.0 / Y_moved
#     n_node = Y_moved.shape[0]
#     d_node = np.ones(n_node) * dr
#     d_node[:ig] *= -1
#     newshape_node = [n_node] + [1] * (Y_moved.ndim - 1)
#     d_node = d_node.reshape(newshape_node)
#     Sn_int = d_node * Sn_p
#     Sn_int[:ig] = np.flip(np.cumsum(np.flip(Sn_int[:ig], axis=0), axis=0), axis=0)
#     Sn_int[ig:] = np.cumsum(Sn_int[ig:], axis=0)
#
#     pi_hydro_n = pi0 - Gamma * gravity * Sn_int
#     rhoY_hydro_n = pi_hydro_n**gm1_inv
#
#     pi_hydro_n_final = np.moveaxis(pi_hydro_n, 0, axis)
#     del pi_hydro_n
#     rhoY_hydro_n_final = np.moveaxis(rhoY_hydro_n, 0, axis)
#     del rhoY_hydro_n
#
#     inner_nodes = tuple(slice(0, s) for s in grid.cshape)
#     # For node variables, assume similar channel ordering as cell_vars.
#     hydrostate.node_vars[inner_nodes + (HI.RHO0,)] = rhoY_hydro_n_final
#     hydrostate.node_vars[inner_nodes + (HI.P0,)] = rhoY_hydro_n_final**gamm
#     hydrostate.node_vars[inner_nodes + (HI.P2_0,)] = pi_hydro_n_final / Msq
#     hydrostate.node_vars[inner_nodes + (HI.S0,)] = 1.0 / Y_n
#     hydrostate.node_vars[inner_nodes + (HI.Y0,)] = 1.0 / Y_n
#     hydrostate.node_vars[inner_nodes + (HI.RHOY0,)] = rhoY_hydro_n_final


def _cumulative_integral(arr: np.ndarray, ghost_idx: int) -> np.ndarray:
    """Helper: cumulative integration that applies a reverse cumsum for indices below ghost."""
    result = np.empty_like(arr)
    result[:ghost_idx] = np.flip(
        np.cumsum(np.flip(arr[:ghost_idx], axis=0), axis=0), axis=0
    )
    result[ghost_idx:] = np.cumsum(arr[ghost_idx:], axis=0)
    return result


def compute_stratification(
    hydrostate: Variables,
    Y: np.ndarray,
    Y_n: np.ndarray,
    grid: Grid,
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
    grid : Grid
        spatial data
    axis : int
        The axis along which stratification (integration) is performed.
    gravity_strength : Union[np.ndarray, list]
        Gravity strength vector—for the integration only the component along the chosen axis is used.
    Msq : float
        Mach number squared.
    """
    # Instantiate thermodynamics (assumed fast)
    th = Thermodynamics()
    dr = grid.dxyz[axis]

    # 'ghost' is the number of ghost cells on the bottom side; this will be used to split integration
    ghost = grid.ng[axis][0]

    # Move the integration axis to the front for easier vectorized operations.
    Y_axis0 = np.moveaxis(Y, axis, 0)
    Y_n_axis0 = np.moveaxis(Y_n, axis, 0)

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
