"""This module handles multiple pressure variables. It is used to calculate the hydrostate pressure.
The code is the modified version of what appears in PyBella project.
"""

import scipy as sp
import numpy as np
from typing import List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
from atmpy.infrastructure.utility import direction_axis
from atmpy.grid.utility import DimensionSpec, create_grid
from atmpy.infrastructure.enums import HydrostateIndices as HI
from atmpy.variables.variables import Variables
from atmpy.physics.thermodynamics import Thermodynamics


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
        """Compute the derivative of S with respect to the direction of gravity. The S variable is assumed to be
        defined on the nodes.

        Returns
        -------
        ndarray of shape grid.cshape
            The derivative of S with respect to the direction of gravity.

        Notes
        -----
        This function computes the nodal derivative in the direction of gravity. Number of nodes in each direction are
        equal to the number of cells plus one. The derivative reduces this shape in the direction of gravity to equal to
        the number of cells. After that the resulting 1D array gets tiled to be of the same shape as the grid.cshape.
        """

        dr: float = self.grid.dxyz[self.direction]

        # Since variables in self.hydrostate are 1D, it suffices to calculate the convolution in their only direction
        # and then divide the result be the correct dx, dy or dz to get the derivative.
        S0: np.ndarray = self.hydrostate.node_vars[..., HI.S0]
        dS: np.ndarray = np.diff(S0) / dr

        if self.grid.ndim == 1:
            return dS

        # Expand and repeat along even axes so that the result matches the cell array shape.
        tile_shape: list = self._get_tile_shape()
        dS = np.tile(dS, tile_shape)
        return dS

    def get_S0c_on_cells(self):
        """Get method to get the S0c attribute. The S0 variable is assumed to be defined on the cells."""
        S0c: np.ndarray = self.hydrostate.cell_vars[..., HI.S0]

        if self.grid.ndim == 1:
            return S0c

        # Expand and repeat along even axes so that the result matches the cell array shape.
        tile_shape: list = self._get_tile_shape()
        S0c = np.tile(S0c, tile_shape)
        return S0c

    def _get_tile_shape(self):
        tile_shape = list(self.grid.cshape)
        tile_shape[self.direction] = 1
        return tile_shape

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

            # Y_c: np.ndarray = -Gamma * g * dr / (pi_cp - pi_cm)
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