"""This module handles multiple pressure variables. It is used to calculate the hydrostate pressure.
The code is the modified version of what appears in PyBella project.
"""

import scipy as sp
import numpy as np
from typing import List, Union, TYPE_CHECKING, Callable

# Import numba for JIT compilation, as used in the reference code.
import numba as nb

if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
    from atmpy.variables.variables import Variables
from atmpy.infrastructure.utility import direction_axis
from atmpy.grid.utility import DimensionSpec, create_grid
from atmpy.infrastructure.enums import HydrostateIndices as HI, VariableIndices as VI
from atmpy.variables.variables import Variables
from atmpy.physics.thermodynamics import Thermodynamics


# JIT-compiled helper function. This is a direct translation.
@nb.jit(nopython=True)
def _loop_over_array(igx, igy, icxn, icyn, p, dp):
    """
    Recursively populates the pressure array 'p' based on the values in 'dp'.
    """
    for j in range(igy, icyn - igy):
        for i in range(igx + 1, icxn - igx):
            p[i, j] = 2.0 * dp[i - 1, j] - p[i - 1, j]
    return p


class HydrostaticStateND:
    """A simple container for N-Dimensional hydrostatic state variables."""

    def __init__(self, shape_cells, shape_nodes):
        self.p0 = np.zeros(shape_cells)
        self.p20 = np.zeros(shape_cells)
        self.rhoY0 = np.zeros(shape_cells)
        self.S0 = np.zeros(shape_cells)
        self.p20_nodes = np.zeros(shape_nodes)


class MPV:
    """Container for 'Multiple Pressure Variables'

    Attributes
    ----------
    grid : Grid
        the Grid object
    grid1D : Grid
        the 1D version of the self.grid in the direction of hydrostacity
    """

    def __init__(self, grid: "Grid", num_vars: int = 6, direction: str = "y"):
        """Initialize the container for pressure variables.

        Parameters
        ----------
        grid : Grid
            the Grid object
        num_vars : int (default=6)
            the number of pressure variables for the hydrostate container
        direction : str (default="y")
            the direction of hydrostacity. Values can be "x", "y" or "z".
        """
        self.grid: "Grid" = grid
        self.direction_str: str = direction
        self.direction: int = direction_axis(direction)
        self.grid1D: Grid = self._create_1D_grid_in_direction()
        self.p0: float = 1.0
        self.p00: float = 1.0

        self.p2_cells: np.ndarray = np.zeros(grid.cshape)
        self.dp2_cells: np.ndarray = np.zeros(grid.cshape)
        self.p2_nodes: np.ndarray = np.zeros(grid.nshape)
        self.p2_nodes0: np.ndarray = np.zeros(grid.nshape)
        self.dp2_nodes: np.ndarray = np.zeros(grid.nshape)

        self.Pu: np.ndarray = np.zeros(grid.cshape)
        self.Pv: np.ndarray = np.zeros(grid.cshape)
        self.Pw: np.ndarray = np.zeros(grid.cshape)

        # Containers for the pressure equation
        self.rhs: np.ndarray = np.zeros(
            grid.nshape
        )  # Container for the divergence of momenta
        self.wcenter: np.ndarray = np.zeros(
            grid.nshape
        )  # Container for the dP/dpi in the Helmholtz eq.
        self.wplus: np.ndarray = np.zeros(
            (grid.ndim,) + grid.cshape
        )  # Container for the (P*Theta) in momenta eq.

        # The variable container for hydrostate
        self.hydrostate: Variables = Variables(
            self.grid1D, num_vars_cell=num_vars, num_vars_node=num_vars
        )

    def _create_1D_grid_in_direction(self):
        """Reduce the dimension of the Grid object and create a new one in the given direction"""
        if self.grid.ndim == 1:
            return self.grid
        start: float = getattr(self.grid, self.direction_str + "_start")
        end: float = getattr(self.grid, self.direction_str + "_end")
        ncells: int = getattr(self.grid, "n" + self.direction_str)
        nghosts: int = getattr(self.grid, "ng" + self.direction_str)

        dims: List[DimensionSpec] = [DimensionSpec(ncells, start, end, nghosts)]
        grid: "Grid" = create_grid(dims)

        return grid

    def compute_dS_on_nodes(self):
        """
        Computes the derivative of the nodal stratification (S) along the
        gravity axis and broadcasts the result to the full cell grid.

        The derivative is computed between nodes, resulting in a cell-centered
        quantity.

        Returns
        -------
        ndarray of shape grid.cshape
            The derivative of S with respect to the gravity direction, defined
            on the cell centers.
        """
        # 1. Get grid spacing and 1D nodal stratification array.
        dr: float = self.grid.dxyz[self.direction]
        S0: np.ndarray = self.hydrostate.node_vars[..., HI.S0]

        # 2. Compute the derivative. The result `dS` is 1D and has a length
        #    equal to the number of cells in the gravity direction.
        dS: np.ndarray = np.diff(S0) / dr

        # 3. Handle the 1D base case.
        if self.grid.ndim == 1:
            return dS

        # 4. Reshape the 1D result to be broadcastable.
        #    For a 3D grid with gravity in y-dir, this changes the shape
        #    from (ny,) to (1, ny, 1).
        new_shape = [1] * self.grid.ndim
        new_shape[self.direction] = dS.shape[0]
        dS_reshaped = dS.reshape(tuple(new_shape))

        # 5. Broadcast the reshaped array to the full cell grid shape.
        #    This is memory-efficient as it creates a view, not a copy.
        dS_full = np.broadcast_to(dS_reshaped, self.grid.cshape)

        return dS_full

    def get_S0c_on_cells(self):
        """
        Gets the 1D stratification array S0c and broadcasts it to the full
        grid shape (e.g., 2D or 3D) using NumPy broadcasting.

        This is the recommended, most efficient method.
        """
        # 1. Get the 1D stratification array defined along the gravity axis.
        S0c: np.ndarray = self.hydrostate.cell_vars[..., HI.S0]

        # If the grid is already 1D, no change is needed.
        if self.grid.ndim == 1:
            return S0c

        # 2. Reshape the 1D array to be broadcastable.
        #    Create a new shape tuple like [1, 1, ..., 1].
        new_shape = [1] * self.grid.ndim

        #    Set the dimension along the gravity axis to its actual size.
        #    This transforms S0c from shape (N,) to, for example, (1, N, 1) for g_axis = 1
        new_shape[self.direction] = S0c.shape[0]
        S0c_reshaped = S0c.reshape(tuple(new_shape))

        # 3. Use np.broadcast_to to create a read-only view of the array
        return np.broadcast_to(S0c_reshaped, self.grid.cshape)

    def set_rhs_to_zero(self):
        self.rhs[...] = 0.0

    def state(self, gravity_strength: Union[np.ndarray, list, tuple], Msq: float):
        """Computes the initial values for the multiple pressure variables

        Parameters
        ----------
        gravity_strength : np.ndarray, List or tuple of shape (3,)
            the array of gravity strengths in each direction

        Msq : float
            Mach number squared
        """
        thermo: Thermodynamics = Thermodynamics()
        g: float = gravity_strength[self.direction]

        if g != 0.0:
            Gamma: float = thermo.Gamma
            Hex: float = 1.0 / (Gamma * g)
            dr: float = self.grid.dxyz[self.direction]
            nodes: np.ndarray = self.grid.get_node_coordinates(self.direction)
            cells: np.ndarray = self.grid.get_cell_coordinates(self.direction)

            pi_np: np.ndarray = np.exp(-(nodes + 0.5 * dr) / Hex)
            pi_nm: np.ndarray = np.exp(-(nodes - 0.5 * dr) / Hex)
            pi_n: np.ndarray = np.exp(-(nodes) / Hex)

            Y_n: np.ndarray = -Gamma * g * dr / (pi_np - pi_nm)
            P_n: np.ndarray = pi_n**thermo.gm1inv
            p_n: np.ndarray = pi_n**thermo.Gammainv
            rho_n: np.ndarray = P_n / Y_n

            self.hydrostate.node_vars[..., HI.P2_0] = pi_n / Msq
            self.hydrostate.node_vars[..., HI.P0] = p_n
            self.hydrostate.node_vars[..., HI.RHO0] = rho_n
            self.hydrostate.node_vars[..., HI.RHOY0] = P_n
            self.hydrostate.node_vars[..., HI.Y0] = Y_n
            self.hydrostate.node_vars[..., HI.S0] = 1.0 / Y_n

            pi_cp: np.ndarray = np.exp(-(cells + 0.5 * dr) / Hex)
            pi_cm: np.ndarray = np.exp(-(cells - 0.5 * dr) / Hex)
            pi_c: np.ndarray = np.exp(-(cells) / Hex)

            # Y_c: np.ndarray = -Gamma * g * dr / (pi_c - pi_cm)
            # P_c: np.ndarray = (1.0 / Y_c)**thermo.gm1inv
            # p_c: np.ndarray = pi_c**thermo.Gammainv
            # rho_c: np.ndarray = P_c / Y_c
            #
            # self.hydrostate.cell_vars[..., HI.P2_0] = 1.0 / Y_c / Msq
            Y_c: np.ndarray = -Gamma * g * dr / (pi_cp - pi_cm)
            P_c: np.ndarray = pi_c**thermo.gm1inv
            p_c: np.ndarray = pi_c**thermo.Gammainv
            rho_c: np.ndarray = P_c / Y_c

            self.hydrostate.cell_vars[..., HI.P2_0] = pi_c / Msq
            self.hydrostate.cell_vars[..., HI.P0] = p_c
            self.hydrostate.cell_vars[..., HI.RHO0] = rho_c
            self.hydrostate.cell_vars[..., HI.RHOY0] = P_c
            self.hydrostate.cell_vars[..., HI.Y0] = Y_c
            self.hydrostate.cell_vars[..., HI.S0] = 1.0 / Y_c

        else:
            # In the absence of gravity, we set every pressure to 1.
            self.hydrostate.cell_vars[...] = 1.0
            self.hydrostate.node_vars[...] = 1.0

    # def integrate_density_hydrostatically(self, rho_field_3d: np.ndarray, gravity_strength: np.ndarray) -> np.ndarray:
    #     """
    #     Computes the hydrostatic pressure perturbation field corresponding to a
    #     given 3D density perturbation field.
    #
    #     The integration is performed column-by-column in the direction of gravity.
    #     The pressure perturbation is assumed to be zero at the bottom boundary of each column.
    #
    #     Parameters
    #     ----------
    #     rho_field_3d : np.ndarray
    #         A 3D array (matching the grid's cell shape) containing the density perturbation.
    #     gravity_strength : np.ndarray
    #         The array of gravity strengths in each direction.
    #
    #     Returns
    #     -------
    #     np.ndarray
    #         A 3D array of the same shape containing the hydrostatic pressure perturbation.
    #     """
    #     g_eff = gravity_strength[self.direction]
    #     dy = self.grid.dxyz[self.direction]
    #     p_hydro_pert = np.zeros_like(rho_field_3d)
    #
    #     # Get the number of cells in the vertical direction
    #     num_vertical_cells = rho_field_3d.shape[self.direction]
    #
    #     # Define slices for accessing adjacent vertical layers
    #     # This is a robust way to write the integration loop without hardcoding the axis index
    #     prev_slice = [slice(None)] * self.grid.ndim
    #     curr_slice = [slice(None)] * self.grid.ndim
    #
    #     # Integrate from the "bottom" (index 0) to the "top"
    #     for j in range(1, num_vertical_cells):
    #         prev_slice[self.direction] = j - 1
    #         curr_slice[self.direction] = j
    #
    #         # Use trapezoidal rule for the integration step
    #         p_hydro_pert[tuple(curr_slice)] = p_hydro_pert[tuple(prev_slice)] - 0.5 * g_eff * \
    #                                           (rho_field_3d[tuple(curr_slice)] + rho_field_3d[tuple(prev_slice)]) * dy
    #
    #     return p_hydro_pert

    def integrate_density_hydrostatically(self, rho_field_3d: np.ndarray, gravity_strength: np.ndarray) -> np.ndarray:
        """
        Computes the hydrostatic pressure perturbation field corresponding to a
        given 3D density perturbation field.

        The integration is performed column-by-column in the direction of gravity.
        The pressure perturbation is assumed to be zero at the top boundary of each column.
        """
        g_eff = gravity_strength[self.direction]
        dy = self.grid.dxyz[self.direction]
        p_hydro_pert = np.zeros_like(rho_field_3d)

        # Get the number of cells in the vertical direction
        num_vertical_cells = rho_field_3d.shape[self.direction]

        # Set p=0 at the top (last index)
        top_slice = [slice(None)] * self.grid.ndim
        top_slice[self.direction] = num_vertical_cells - 1
        p_hydro_pert[tuple(top_slice)] = 0.0

        # Define slices for accessing adjacent vertical layers
        next_slice = [slice(None)] * self.grid.ndim
        curr_slice = [slice(None)] * self.grid.ndim

        # Integrate downward from the top to the bottom
        for j in range(num_vertical_cells - 2, -1, -1):
            next_slice[self.direction] = j + 1  # Higher level
            curr_slice[self.direction] = j  # Current (lower) level

            # Use trapezoidal rule: p_lower = p_higher + avg_rho * g * dy (pressure increases downward)
            p_hydro_pert[tuple(curr_slice)] = p_hydro_pert[tuple(next_slice)] + 0.5 * g_eff * \
                                              (rho_field_3d[tuple(curr_slice)] + rho_field_3d[tuple(next_slice)]) * dy

        return p_hydro_pert

    def compute_hydrostate_from_profile(
        self,
        Y_cells_nd: np.ndarray,
        Y_nodes_nd: np.ndarray,
        gravity_strength: Union[np.ndarray, list, tuple],
        Msq: float,
    ):
        """
        Computes an N-D hydrostatically balanced state from a given N-D potential temperature profile.
        This is a direct, multi-dimensional translation of the 'column' function.
        """
        thermo = Thermodynamics()
        Gamma, gamm, gm1 = thermo.Gamma, thermo.gamma, thermo.gm1
        Gamma_inv, gm1_inv = thermo.Gammainv, thermo.gm1inv

        g_axis = self.direction
        g = gravity_strength[g_axis]

        # Get 1D vertical grid parameters
        icy_1d = self.grid1D.ncx_total
        igy_1d = self.grid1D.ng[0][0]
        dy_1d = self.grid1D.dx

        # Initialize the N-D state container
        HySt = HydrostaticStateND(self.grid.cshape, self.grid.nshape)

        rhoY0 = 1.0
        pi0 = rhoY0**gm1

        # --- 1. Calculate Hydrostatic State at Cell Centers ---
        dys = np.ones(icy_1d) * dy_1d
        dys[0] = -dy_1d
        dys[1] = -dy_1d / 2
        dys[2] = dy_1d / 2

        # Reshape for broadcasting over the N-D grid
        broadcast_shape = [1] * self.grid.ndim
        broadcast_shape[g_axis] = icy_1d
        dys_nd = dys.reshape(broadcast_shape)

        # Create slice objects for indexing the reference level
        slice_ref_node = tuple(
            igy_1d if i == g_axis else slice(None) for i in range(self.grid.ndim)
        )

        S_p = 1.0 / Y_cells_nd
        S_m = np.zeros_like(S_p)

        # Build slices for S_m assignment
        s_m_ref1 = tuple(
            slice(igy_1d - 1, igy_1d + 1) if i == g_axis else slice(None)
            for i in range(self.grid.ndim)
        )
        s_m_ref2 = tuple(
            0 if i == g_axis else slice(None) for i in range(self.grid.ndim)
        )
        s_m_ref3 = tuple(
            slice(igy_1d + 1, None) if i == g_axis else slice(None)
            for i in range(self.grid.ndim)
        )
        s_m_src1 = tuple(
            igy_1d - 1 if i == g_axis else slice(None) for i in range(self.grid.ndim)
        )
        s_m_src2 = tuple(
            slice(igy_1d, -1) if i == g_axis else slice(None)
            for i in range(self.grid.ndim)
        )

        # Use broadcasting with an added axis for the reference slice
        S_m[s_m_ref1] = 1.0 / np.expand_dims(Y_nodes_nd[slice_ref_node], axis=g_axis)
        S_m[s_m_ref2] = 1.0 / Y_cells_nd[s_m_src1]
        S_m[s_m_ref3] = 1.0 / Y_cells_nd[s_m_src2]

        S_integral_p = 0.5 * dys_nd * (S_p + S_m)

        # Perform cumulative sum along the gravity axis
        # We need to slice, reverse, cumsum, and reverse back for the part below the reference level
        s_below = tuple(
            slice(0, igy_1d) if i == g_axis else slice(None)
            for i in range(self.grid.ndim)
        )
        s_above = tuple(
            slice(igy_1d, None) if i == g_axis else slice(None)
            for i in range(self.grid.ndim)
        )
        S_integral_p[s_below] = np.flip(
            np.cumsum(np.flip(S_integral_p[s_below], axis=g_axis), axis=g_axis),
            axis=g_axis,
        )
        S_integral_p[s_above] = np.cumsum(S_integral_p[s_above], axis=g_axis)

        pi_hydro = pi0 - Gamma * g * S_integral_p

        HySt.rhoY0 = pi_hydro**gm1_inv
        HySt.S0 = S_p
        HySt.p0 = pi_hydro**Gamma_inv
        HySt.p20 = pi_hydro / Msq

        # --- 2. Calculate Hydrostatic State at Node Locations ---
        dys_n = np.ones(icy_1d) * dy_1d
        dys_n[:igy_1d] *= -1
        dys_n_nd = dys_n.reshape(broadcast_shape)

        Sn_integral_p = dys_n_nd / Y_cells_nd

        Sn_integral_p[s_below] = np.flip(
            np.cumsum(np.flip(Sn_integral_p[s_below], axis=g_axis), axis=g_axis),
            axis=g_axis,
        )
        Sn_integral_p[s_above] = np.cumsum(Sn_integral_p[s_above], axis=g_axis)

        pi_hydro_n = pi0 - Gamma * g * Sn_integral_p

        slice_ref_node_perp = tuple(
            slice(-1, None) if i != g_axis else slice(None)
            for i in range(self.grid.ndim)
        )
        mask = np.ones_like(HySt.p20_nodes, dtype=bool)
        mask[slice_ref_node] = False
        mask[slice_ref_node_perp] = False

        p20n_full_domain = np.zeros_like(HySt.p20_nodes)
        p20n_full_domain[mask] = pi_hydro_n.flatten() / Msq
        p20n_full_domain[slice_ref_node] = pi0 / Msq

        HySt.p20_nodes[...] = p20n_full_domain

        return HySt

    def correct_initial_pressure(self, main_vars: "Variables", Msq: float):
        """
        Adjusts the initial pressure field to ensure hydrostatic balance.
        This is a direct translation of 'initial_pressure' from hydrostatics.py,
        specifically for a 2D grid with gravity in 'y' and periodicity in 'x'.
        """
        if self.direction_str != "y" or self.grid.ndim != 2:
            raise NotImplementedError(
                "This translated correct_initial_pressure is specific to 2D grids with 'y' direction gravity."
            )

        thermo = Thermodynamics()
        Gammainv = thermo.Gammainv

        # Grid parameters (mapping elem/node to self.grid)
        ngx, ngy = self.grid.ngx, self.grid.ngy
        icx, icy = self.grid.cshape
        icxn, icyn = self.grid.nshape
        dx, dy = self.grid.dx, self.grid.dy
        height = self.grid.y_nodes[-ngy - 1]

        # --- BLOCK 1: Cell-based pressure correction (p2_cells) ---
        x_idx_m_c = slice(0, -1)
        x_idx_c_c = slice(1, None)
        y_idx_c = slice(ngy, -ngy)
        xn_idx_c = slice(
            1, -1
        )  # Original code had this slice, seems off but we translate exactly

        rho_y = main_vars.cell_vars[..., VI.RHOY]
        rho = main_vars.cell_vars[..., VI.RHO]

        Pc = rho_y[x_idx_c_c, y_idx_c]
        Pm = rho_y[x_idx_m_c, y_idx_c]
        thc = Pc / rho[x_idx_c_c, y_idx_c]
        thm = Pm / rho[x_idx_m_c, y_idx_c]

        beta_c = np.zeros(icxn)  # Use nodal x-dimension for beta array
        bdpdx_c = np.zeros(icxn)

        beta_c[xn_idx_c] = np.sum(0.5 * (Pm * thm + Pc * thc) * dy, axis=1)
        bdpdx_c[xn_idx_c] = np.sum(
            0.5
            * (Pm * thm + Pc * thc)
            * (self.p2_cells[x_idx_c_c, y_idx_c] - self.p2_cells[x_idx_m_c, y_idx_c])
            * dy,
            axis=1,
        )

        beta_c = np.where(beta_c == 0, 1e-12, beta_c)
        beta_c *= Gammainv / height
        bdpdx_c *= Gammainv / height / dx

        coeff_c = np.zeros(icx)
        pibot_c = np.zeros(icx)

        active_x_c = slice(ngx, icx - ngx)
        active_x_c_plus1 = slice(ngx + 1, icx - ngx + 1)
        # Note: beta array is nodal-sized, so we slice beta[ngx+1:-ngx]
        coeff_c[active_x_c_plus1] = np.cumsum(
            coeff_c[active_x_c] + dx / beta_c[ngx + 1 : -ngx]
        )
        pibot_c[active_x_c_plus1] = np.cumsum(
            pibot_c[active_x_c] - dx * bdpdx_c[ngx + 1 : -ngx] / beta_c[ngx + 1 : -ngx]
        )

        dotPU_c = (
            pibot_c[icx - ngx] / coeff_c[icx - ngx] if coeff_c[icx - ngx] != 0 else 0.0
        )
        pibot_c[active_x_c] -= dotPU_c * coeff_c[active_x_c]

        y_idx = slice(ngy, -ngy + 1)
        x_idx = slice(ngx, -ngx + 1)

        p20_hydro_c = self.hydrostate.cell_vars[y_idx, HI.P2_0].reshape(1, -1)
        self.p2_cells[x_idx, y_idx] += pibot_c[x_idx].reshape(-1, 1) - p20_hydro_c

        # --- BLOCK 2: Node-based pressure correction (p2_nodes) ---
        y_idx_n = slice(ngy, -ngy + 1)
        height = self.grid.y_nodes[-ngy]

        beta_n = np.zeros(icx)
        bdpdx_n = np.zeros(icx)

        Pc_n = rho_y[1:, y_idx_n]
        thc_n = Pc_n / rho[1:, y_idx_n]

        beta_n[1:] = np.sum(Pc_n * thc_n * dy, axis=1)
        beta_n = np.where(beta_n == 0, 1e-12, beta_n)
        beta_n *= Gammainv / height

        bdpdx_n[1:] = np.sum(
            Pc_n
            * thc_n
            * (self.p2_nodes[1:-1, y_idx_c] - self.p2_nodes[:-2, y_idx_c])
            * dy,
            axis=1,
        )
        bdpdx_n *= Gammainv / height / dx

        coeff_n = np.zeros(icxn)
        pibot_n = np.zeros(icxn)

        active_x_n = slice(ngx, -ngx)
        active_x_n_plus1 = slice(ngx + 1, -ngx + 1)

        coeff_n[active_x_n_plus1] = np.cumsum(
            coeff_n[active_x_n] + dx / beta_n[ngx + 1 :]
        )
        pibot_n[active_x_n_plus1] = np.cumsum(
            pibot_n[active_x_n] - dx * bdpdx_n[ngx + 1 :] / beta_n[ngx + 1 :]
        )

        dotPU_n = (
            pibot_n[icxn - ngx] / coeff_n[icxn - ngx]
            if coeff_n[icxn - ngx] != 0
            else 0.0
        )
        pibot_n[active_x_n] -= dotPU_n * coeff_n[active_x_n]

        y_idx_p2 = slice(ngy, -ngy + 1)
        x_idx_p2 = slice(ngx, -ngx + 1)
        p20_hydro_n = self.hydrostate.node_vars[y_idx_p2, HI.P2_0].reshape(1, -1)
        self.p2_nodes[x_idx_p2, y_idx_p2] += (
            pibot_n[x_idx_p2].reshape(-1, 1) - p20_hydro_n
        )

        # --- BLOCK 3: Recursive and Periodic Correction for p2_nodes ---
        self.dp2_nodes[...] = self.p2_nodes.copy()

        self.p2_nodes = _loop_over_array(
            ngx, ngy, icxn, icyn, self.p2_nodes, self.dp2_nodes
        )

        delp2 = 0.5 * (self.p2_nodes[-ngx - 1, y_idx_c] - self.p2_nodes[ngx, y_idx_c])
        delp2 = delp2.reshape(1, -1)

        num_x_inner_nodes = icxn - 2 * ngx
        sgn = np.ones((num_x_inner_nodes, 1))
        sgn[1::2] *= -1

        self.p2_nodes[ngx:-ngx, y_idx_c] += sgn * delp2

        # --- BLOCK 4: Final State Update (based on corrected CELL pressure) ---
        self.dp2_nodes[...] = 0.0

        inner_domain_c = self.grid.get_inner_slice()
        p20_hydro_nd_c = self.get_S0c_on_cells().copy()
        p20_hydro_nd_c[...] = self.hydrostate.cell_vars[:, HI.P2_0]

        pi = Msq * (self.p2_cells[inner_domain_c] + p20_hydro_nd_c[inner_domain_c])

        rhoold = main_vars.cell_vars[inner_domain_c][..., VI.RHO].copy()
        Y = main_vars.cell_vars[inner_domain_c][..., VI.RHOY] / rhoold

        main_vars.cell_vars[inner_domain_c][..., VI.RHOY] = pi**thermo.gm1inv
        main_vars.cell_vars[inner_domain_c][..., VI.RHO] = (
            main_vars.cell_vars[inner_domain_c][..., VI.RHOY] / Y
        )

        rho_new = main_vars.cell_vars[inner_domain_c][..., VI.RHO]
        rho_ratio = rho_new / rhoold

        for var_index in [VI.RHOU, VI.RHOV, VI.RHOW, VI.RHOX]:
            if var_index < main_vars.num_vars_cell:
                main_vars.cell_vars[inner_domain_c][..., var_index] *= rho_ratio

    def compute_hydrostate_from_profile3D(
        self,
        Y_cells: np.ndarray,
        Y_nodes: np.ndarray,
        gravity_strength: tuple,
        Msq: float,
    ) -> HydrostaticStateND:
        """
        Compute exact hydrostatic balance for an arbitrary 3D potential temperature field.
        This is the 3D generalization of the 2D version used in InertialGravityLongWaves.

        Parameters
        ----------
        Y_cells : np.ndarray
            Full-grid cell-centered potential temperature θ (shape = grid.cshape_total)
        Y_nodes : np.ndarray
            Full-grid node-centered potential temperature θ (shape = grid.nshape_total)
        gravity_strength : tuple[float, float, float]
        Msq : float

        Returns
        -------
        HydrostaticStateND containing rhoY0, p20 on cells and p20_nodes
        """
        grid = self.grid
        thermo = Thermodynamics()
        Gamma = thermo.Gamma
        gm1inv = 1.0 / (thermo.gamma - 1.0)

        g_axis = self.direction
        g = gravity_strength[g_axis]
        dy = grid.dxyz[g_axis]

        # Containers (only inner parts will be used later)
        HySt = HydrostaticStateND(grid.cshape, grid.nshape)

        if abs(g) < 1e-12:
            # No gravity → uniform reference state
            HySt.rhoY0[...] = 1.0
            HySt.p20[...] = 1.0 / Msq
            HySt.p20_nodes[...] = 1.0 / Msq
            return HySt

        # Reference Exner pressure at the very first inner cell/node in gravity direction
        pi_ref = 1.0

        # ------------------------------------------------------------------
        # 1. Iterate over every horizontal column (x-z plane when g_axis=1)
        # ------------------------------------------------------------------
        # Shape of one horizontal slice
        horiz_shape = list(grid.cshape)
        horiz_shape[g_axis] = 1

        # Indices for horizontal directions
        horiz_axes = [0, 1, 2]
        horiz_axes.remove(g_axis)

        for idx_x in range(grid.ncx_total if 0 != g_axis else 1):
            for idx_z in range(grid.ncz_total if 2 != g_axis else 1):
                # Build full column index tuple
                col_idx = [slice(None)] * 3
                if 0 != g_axis:
                    col_idx[0] = idx_x
                if 2 != g_axis:
                    col_idx[2] = idx_z

                col_tuple = tuple(col_idx)

                # Extract 1D profiles for this column
                Y_col_cells = Y_cells[col_tuple].flatten()      # length = ny_total
                Y_col_nodes = Y_nodes[col_tuple].flatten()      # length = ny_total + 1

                ny = len(Y_col_cells)

                # ------------------- Cell-centered integration -------------------
                pi_col = np.zeros(ny)
                pi_col[grid.ng[g_axis][0]] = pi_ref

                # Upward
                for j in range(grid.ng[g_axis][0], ny - 1):
                    Y_avg = 0.5 * (Y_col_cells[j] + Y_col_cells[j + 1])
                    dpi = -Gamma * g * dy / Y_avg
                    pi_col[j + 1] = pi_col[j] + dpi

                # Downward (into ghosts)
                for j in range(grid.ng[g_axis][0] - 1, -1, -1):
                    Y_avg = 0.5 * (Y_col_cells[j] + Y_col_cells[j + 1])
                    dpi = Gamma * g * dy / Y_avg
                    pi_col[j] = pi_col[j + 1] + dpi

                rhoY_col = pi_col ** gm1inv
                p2_col = pi_col / Msq

                # Write back into 3D arrays
                HySt.rhoY0[col_tuple] = rhoY_col
                HySt.p20[col_tuple] = p2_col

                # ------------------- Node-centered integration -------------------
                pi_node_col = np.zeros(len(Y_col_nodes))
                pi_node_col[grid.ng[g_axis][0]] = pi_ref

                # Upward (use cell value to the "left")
                for j in range(grid.ng[g_axis][0], len(Y_col_nodes) - 1):
                    dpi = -Gamma * g * dy / Y_col_cells[j]
                    pi_node_col[j + 1] = pi_node_col[j] + dpi

                # Downward
                for j in range(grid.ng[g_axis][0] - 1, -1, -1):
                    dpi = Gamma * g * dy / Y_col_cells[j]
                    pi_node_col[j] = pi_node_col[j + 1] + dpi

                HySt.p20_nodes[col_tuple] = pi_node_col / Msq

        return HySt

    def correct_initial_pressure3D(self, variables: "Variables", Msq: float, g):
        """
        Removes the tiny hydrostatic imbalance that appears from round-off
        when the initial condition is built from an analytic potential-temperature field.
        This is the exact 3D generalization of the 2D version used in all PyBella/atmpy tests.

        Works for any gravity direction (x, y, or z) and for 2D or 3D grids.
        """
        grid = self.grid
        thermo = Thermodynamics()
        Gamma = thermo.Gamma
        gm1inv = 1.0 / (thermo.gamma - 1.0)

        g_axis = self.direction                # 0=x, 1=y, 2=z

        dy = grid.dxyz[g_axis]
        inner = grid.get_inner_slice()

        # ------------------------------------------------------------------
        # 1. Reconstruct total Exner pressure π from current prognostic variables
        # ------------------------------------------------------------------
        rho = variables.cell_vars[inner + (VI.RHO,)]
        rhoY = variables.cell_vars[inner + (VI.RHOY,)]
        Y = rhoY / rho                              # potential temperature θ
        pi_total = (rhoY ** gm1inv)                 # π = (ρ θ)^{γ/(γ-1)}  (non-dimensional)

        # ------------------------------------------------------------------
        # 2. Average π to nodes (simple arithmetic average in gravity direction)
        # ------------------------------------------------------------------
        # Shape of inner node array in gravity direction is ny+1 (or nx+1, nz+1)
        pi_node = np.zeros(grid.nshape)
        s = [slice(None)] * 3

        # Lower part (from cell j to node j)
        s[g_axis] = slice(None, -1)
        pi_node = pi_total

        # Upper part (from cell j to node j+1)
        s[g_axis] = slice(1, None)
        pi_node += pi_total

        # Average (2 contributions except at domain edges → exactly what we want)
        s[g_axis] = slice(1, -1)
        pi_node *= 0.5

        # ------------------------------------------------------------------
        # 3. Compute hydrostatic correction by integrating dπ/dn = -Γ g / Y
        # ------------------------------------------------------------------
        # Reference π at first inner node (same as in compute_hydrostate_from_profile)
        pi_ref = pi_node
        pi_ref = pi_ref.flatten()[grid.ng[g_axis][0]]   # scalar

        # Integrate upward
        for j in range(grid.ng[g_axis][0] + 1, grid.nshape[g_axis]):
            idx = [slice(None)] * 3
            idx[g_axis] = j - 1
            Y_cell = Y[tuple(idx)] if g_axis == 1 else np.take(Y, j - 1, axis=g_axis)
            dpi = -Gamma * g * dy / Y_cell
            idx[g_axis] = j
            pi_node[tuple(idx)] = pi_node[tuple(idx[:-1] + [j-1])] + dpi

        # Integrate downward (ghosts + lower domain)
        for j in range(grid.ng[g_axis][0] - 1, -1, -1):
            idx = [slice(None)] * 3
            idx[g_axis] = j + 1
            Y_cell = Y[tuple(idx)] if g_axis == 1 else np.take(Y, j, axis=g_axis)
            dpi = Gamma * g * dy / Y_cell
            idx[g_axis] = j
            pi_node[tuple(idx)] = pi_node[tuple(idx[:-1] + [j+1])] + dpi

        # ------------------------------------------------------------------
        # 4. Overwrite p2 with the corrected hydrostatic part
        # ------------------------------------------------------------------
        # p2 = (π_hydro - π_ref) / Msq   (the reference π_ref is already in hydrostate)
        self.p2_nodes[:] = (pi_node - pi_ref) / Msq

        # Also correct cell-centered p2 (average from nodes)
        s = [slice(None)] * 3
        s[g_axis] = slice(None, -1)
        p2_cells = self.p2_nodes[tuple(s)]
        s[g_axis] = slice(1, None)
        p2_cells += self.p2_nodes[tuple(s)]
        s[g_axis] = slice(1, -1)
        self.p2_cells[tuple(s)] = p2_cells[tuple(s)] * 0.5 / Msq

        print("correct_initial_pressure applied (3D) – hydrostatic imbalance removed.")
