"""Utility module for variables"""

import numpy as np
from typing import Union, TYPE_CHECKING, Optional
from atmpy.physics.thermodynamics import Thermodynamics

if TYPE_CHECKING:
    from atmpy.variables.variables import Variables
from atmpy.infrastructure.enums import HydrostateIndices as HI


def get_left_index_in_all_directions(ndim):
    """Return the slices for left indices in all dimensions."""
    return [slice(0, -1)] * ndim


def _cumulative_integral(arr: np.ndarray, ghost_idx: int) -> np.ndarray:
    """Helper: cumulative integration that applies a reverse cumsum for indices below ghost."""
    result = np.empty_like(arr)
    result[:ghost_idx] = np.flip(
        np.cumsum(np.flip(arr[:ghost_idx], axis=0), axis=0), axis=0
    )
    result[ghost_idx:] = np.cumsum(arr[ghost_idx:], axis=0)
    return result


def column_hydrostatics(
    mpv: "MPV",
    Y_cells: np.ndarray,
    Y_nodes: np.ndarray,
    gravity_strength: Union[np.ndarray, list],
    Msq: float,
    ref_node_idx: Optional[int] = None,
    ref_rhoY: float = 1.0,
) -> None:
    """
    Computes the hydrostatic state by vertically integrating a given
    potential temperature profile (Y).

    This function mimics the logic of PyBella's `hydrostatics.column` but is
    adapted for the Atmpy framework. It calculates the hydrostatic Exner
    pressure, pressure, density, etc., on both cell centers and nodes based
    on the provided potential temperature profiles Y_cells and Y_nodes.

    The integration assumes d(pi)/dr = -Gamma * g * S, where S = 1/Y and
    r is the coordinate in the direction of gravity. A trapezoidal rule is
    used for integration node-to-node. Cell-centered values are derived
    from the nodal values.

    Parameters
    ----------
    mpv : MPV
        The Multiple Pressure Variables object whose `hydrostate` variable
        will be populated. The `mpv.direction` determines the hydrostatic
        direction, and `mpv.grid1D` provides the 1D grid details.
    Y_cells : np.ndarray
        1D array of potential temperature (or equivalent) values at the
        cell centers of the 1D hydrostatic grid (`mpv.grid1D`).
        Shape should be `(mpv.grid1D.ncx_total,)`.
    Y_nodes : np.ndarray
        1D array of potential temperature (or equivalent) values at the
        nodes of the 1D hydrostatic grid (`mpv.grid1D`).
        Shape should be `(mpv.grid1D.nnx_total,)`.
    gravity_strength : np.ndarray or list
        Array or list of gravity strengths in [x, y, z] directions.
    Msq : float
        Mach number squared (used for scaling p2).
    ref_node_idx : Optional[int], optional
        The index of the node in `Y_nodes` where the reference pressure
        is defined. If None (default), uses the index of the first inner
        node (boundary between first ghost layer and first inner cell).
    ref_rhoY : float, optional
        The reference value for rho * Y (density * potential temperature)
        at the `ref_node_idx`. Defaults to 1.0.

    Returns
    -------
    None
        Modifies `mpv.hydrostate.cell_vars` and `mpv.hydrostate.node_vars`
        in place.

    Raises
    ------
    ValueError
        If input array shapes are incorrect or gravity is zero in the
        specified direction but `ref_rhoY` leads to non-uniform state.
    """
    thermo = Thermodynamics()
    Gamma = thermo.Gamma  # Specific heat ratio / (Specific heat ratio - 1)
    gamm = thermo.gamma  # Specific heat ratio
    gm1 = thermo.gamma - 1.0  # Specific heat ratio - 1
    Gamma_inv = 1.0 / Gamma
    gm1_inv = 1.0 / gm1

    grid1D = mpv.grid1D
    direction = mpv.direction
    g = gravity_strength[direction]

    # Check for zero gravity case
    if g == 0.0:
        print(
            "Warning: Gravity is zero in the hydrostatic direction. Setting uniform state."
        )
        # Set uniform state based on reference values if possible
        # Note: This assumes Y=1 if rhoY=1 and rho=1 initially.
        # A more robust approach might require specifying reference rho OR Y.
        p0 = ref_rhoY**gamm
        pi0 = ref_rhoY**gm1
        Y0 = 1.0  # Assuming ref_rhoY=1 implies rho=1, Y=1
        rho0 = ref_rhoY / Y0 if Y0 != 0 else 1.0  # Avoid division by zero
        S0 = 1.0 / Y0 if Y0 != 0 else 1.0

        mpv.hydrostate.cell_vars[..., HI.P0] = p0
        mpv.hydrostate.cell_vars[..., HI.P2_0] = pi0 / Msq
        mpv.hydrostate.cell_vars[..., HI.RHO0] = rho0
        mpv.hydrostate.cell_vars[..., HI.RHOY0] = ref_rhoY
        mpv.hydrostate.cell_vars[..., HI.Y0] = Y0
        mpv.hydrostate.cell_vars[..., HI.S0] = S0

        mpv.hydrostate.node_vars[..., HI.P0] = p0
        mpv.hydrostate.node_vars[..., HI.P2_0] = pi0 / Msq
        mpv.hydrostate.node_vars[..., HI.RHO0] = rho0
        mpv.hydrostate.node_vars[..., HI.RHOY0] = ref_rhoY
        mpv.hydrostate.node_vars[..., HI.Y0] = Y0
        mpv.hydrostate.node_vars[..., HI.S0] = S0
        return

    # --- Grid and Input Validation ---
    # Since grid1D is always 1D, its properties are in the first dimension
    dr = grid1D.dx  # Equivalent to dx, dy, or dz of original grid
    n_cells_total = grid1D.ncx_total
    n_nodes_total = grid1D.nnx_total
    ng = grid1D.ngx  # Number of ghost cells on one side

    if Y_cells.shape != (n_cells_total,):
        raise ValueError(
            f"Shape of Y_cells {Y_cells.shape} does not match grid1D cell shape ({n_cells_total},)"
        )
    if Y_nodes.shape != (n_nodes_total,):
        raise ValueError(
            f"Shape of Y_nodes {Y_nodes.shape} does not match grid1D node shape ({n_nodes_total},)"
        )

    # --- Determine Reference Point ---
    idx_ref = ref_node_idx if ref_node_idx is not None else ng
    if not (0 <= idx_ref < n_nodes_total):
        raise ValueError(
            f"ref_node_idx {idx_ref} is out of bounds for nodes [0, {n_nodes_total-1}]"
        )

    # --- Calculate Reference State Values ---
    p0_ref = ref_rhoY**gamm
    pi0_ref = ref_rhoY**gm1
    Y0_ref = Y_nodes[idx_ref]
    if Y0_ref == 0:
        raise ValueError(
            f"Potential temperature Y is zero at reference node {idx_ref}, cannot compute density."
        )
    rho0_ref = ref_rhoY / Y0_ref
    S0_ref = 1.0 / Y0_ref

    # --- Calculate Nodal Hydrostatic Profile ---
    S_nodes = 1.0 / Y_nodes
    pi_nodes = np.zeros(n_nodes_total)
    pi_nodes[idx_ref] = pi0_ref

    # Integrate upwards (increasing index) from reference node
    for i in range(idx_ref + 1, n_nodes_total):
        # d(pi) = -Gamma * g * S * dr
        # Integrate from i-1 to i using trapezoidal rule for S
        delta_pi = -Gamma * g * 0.5 * (S_nodes[i] + S_nodes[i - 1]) * dr
        pi_nodes[i] = pi_nodes[i - 1] + delta_pi

    # Integrate downwards (decreasing index) from reference node
    for i in range(idx_ref - 1, -1, -1):
        # Integrate from i+1 to i using trapezoidal rule for S
        delta_pi = -Gamma * g * 0.5 * (S_nodes[i] + S_nodes[i + 1]) * dr
        # Since we integrate downwards, dr is negative, so we add (-delta_pi)
        pi_nodes[i] = pi_nodes[i + 1] - delta_pi  # pi[i] - pi[i+1] = delta_pi

    # --- Calculate Node Variables ---
    # Avoid division by zero or invalid ops if pi becomes negative
    pi_nodes_safe = np.maximum(pi_nodes, 1e-15)  # Prevent log/pow issues
    p_nodes = pi_nodes_safe**Gamma_inv
    rhoY_nodes = pi_nodes_safe**gm1_inv
    rho_nodes = rhoY_nodes * S_nodes  # rho = rhoY * S = rhoY / Y
    p2_nodes = pi_nodes / Msq  # Use original pi for p2

    # Store Node Variables
    mpv.hydrostate.node_vars[..., HI.P0] = p_nodes
    mpv.hydrostate.node_vars[..., HI.P2_0] = p2_nodes
    mpv.hydrostate.node_vars[..., HI.RHO0] = rho_nodes
    mpv.hydrostate.node_vars[..., HI.RHOY0] = rhoY_nodes
    mpv.hydrostate.node_vars[..., HI.Y0] = Y_nodes  # Given input
    mpv.hydrostate.node_vars[..., HI.S0] = S_nodes

    # --- Calculate Cell Variables ---
    # Average nodal Exner pressure to cell centers
    pi_cells = 0.5 * (pi_nodes[:-1] + pi_nodes[1:])

    # Calculate cell variables from cell-centered Exner pressure and Y_cells
    pi_cells_safe = np.maximum(pi_cells, 1e-15)  # Prevent log/pow issues
    p_cells = pi_cells_safe**Gamma_inv
    rhoY_cells = pi_cells_safe**gm1_inv
    S_cells = 1.0 / Y_cells
    rho_cells = rhoY_cells * S_cells  # rho = rhoY * S = rhoY / Y
    p2_cells = pi_cells / Msq  # Use original pi_cells for p2

    # Store Cell Variables
    mpv.hydrostate.cell_vars[..., HI.P0] = p_cells
    mpv.hydrostate.cell_vars[..., HI.P2_0] = p2_cells
    mpv.hydrostate.cell_vars[..., HI.RHO0] = rho_cells
    mpv.hydrostate.cell_vars[..., HI.RHOY0] = rhoY_cells
    mpv.hydrostate.cell_vars[..., HI.Y0] = Y_cells  # Given input
    mpv.hydrostate.cell_vars[..., HI.S0] = S_cells


if __name__ == "__main__":
    from atmpy.physics.thermodynamics import Thermodynamics

    th = Thermodynamics()
    grav = 9.81
    t_ref = 100.0
    T_ref = 300.0
    R_gas = 287.4
    h_ref = 10_000
    cp = th.gamma * R_gas / (th.gm1)
    N_ref = 9.81 / np.sqrt(cp * T_ref)

    g = grav * h_ref / (R_gas * T_ref)
    Nsq_ref = N_ref * N_ref

    Msq = 0.115
    gravity_vec = [0.0, g, 0.0]
