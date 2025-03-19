"""This module handles multiple pressure variables. It is used to calculate the hydrostate pressure.
The code is the modified version of what appears in PyBella project.
"""

import numpy as np
from atmpy.flux.utility import direction_mapping
from atmpy.grid.kgrid import Grid
from atmpy.grid.utility import DimensionSpec, create_grid
from atmpy.infrastructure.enums import HydrostateIndices as HI
from atmpy.variables.variables import Variables
from atmpy.physics.thermodynamics import Thermodynamics
from atmpy.variables.utility import get_left_index_in_all_directions
from typing import List, Union


class MPV:
    """Container for 'Multiple Pressure Variables'

    Attributes
    ----------
    grid : Grid
        the Grid object
    grid1D : Grid
        the 1D version of the self.grid in the direction of hydrostacity
    """

    def __init__(self, grid, num_vars=6, direction="y"):
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
        # Instead of using elem.sc and node.sc, get the shapes from the grid,
        # where grid.cshape (cell shape) and grid.nshape (node shape) are defined in kgrid.py.
        self.grid: Grid = grid
        self.direction_str: str = direction
        self.direction: int = direction_mapping(direction)
        self.grid1D: Grid = self._create_1D_grid_in_direction()
        self.p0: float = 1.0
        self.p00: float = 1.0

        self.p2_cells: np.ndarray = np.zeros(grid.cshape)
        self.dp2_cells: np.ndarray = np.zeros(grid.cshape)
        self.p2_nodes: np.ndarray = np.zeros(grid.nshape)
        self.p2_nodes0: np.ndarray = np.zeros(grid.nshape)
        self.dp2_nodes: np.ndarray = np.zeros(grid.nshape)

        self.u: np.ndarray = np.zeros(grid.cshape)
        self.v: np.ndarray = np.zeros(grid.cshape)
        self.w: np.ndarray = np.zeros(grid.cshape)

        self.rhs: np.ndarray = np.zeros(grid.nshape)
        self.wcenter: np.ndarray = np.zeros(grid.nshape)
        self.wplus: np.ndarray = np.zeros((grid.ndim,) + grid.cshape)

        # The variable container for hydrostate
        self.hydrostate: Variables = Variables(
            self.grid1D, num_vars_cell=num_vars, num_vars_node=num_vars
        )

    def _create_1D_grid_in_direction(self):
        """Reduce the dimension of the Grid object and create a new one in the given direction"""

        start: float = getattr(self.grid, self.direction_str + "_start")
        end: float = getattr(self.grid, self.direction_str + "_end")
        ncells: int = getattr(self.grid, "n" + self.direction_str)
        nghosts: int = getattr(self.grid, "ng" + self.direction_str)

        dims: List[DimensionSpec] = [DimensionSpec(ncells, start, end, nghosts)]
        grid: Grid = create_grid(dims)

        return grid

    def compute_dS_on_nodes(self, direction):
        """Compute the derivative of S with respect to the direction of gravity. The S variable is assumed to be
        defined on the nodes.

        Parameters
        ----------
        direction : str
            direction of gravity. Values must be "x", "y" or "z".

        Returns
        -------
        ndarray
            The derivative of S with respect to the direction of gravity
        """
        ndim: int = self.grid.ndim  # Use the vertical spacing defined by the grid.
        dr: float = self.grid.dxyz[self.direction]

        # Since variables in self.hydrostate are 1D, it suffices to calulate the convolotion in their only direction
        # and then divide the result be the correct dx, dy or dz to get the derivative.
        S0: np.ndarray = self.hydrostate.node_vars[..., HI.S0]
        dS: np.ndarray = np.diff(S0) / dr

        # Expand and repeat along even axes so that the result matches the cell array shape.
        tile_shape: list = self._get_tile_shape()
        dS = np.tile(dS, tile_shape)
        return dS

    def get_S0c_on_cells(self):
        """Get method to get the S0c attribute. The S0 variable is assumed to be defined on the cells."""
        S0c: np.ndarray = self.hydrostate.cell_vars[..., HI.S0]

        # Expand and repeat along even axes so that the result matches the cell array shape.
        tile_shape: list = self._get_tile_shape()
        S0c = np.tile(S0c, tile_shape)
        return S0c

    def _get_tile_shape(self):
        tile_shape = list(self.grid.cshape)
        tile_shape[self.direction] = 1
        return tile_shape

    def state(self, gravity_strength: Union[np.ndarray, list], Msq: float):
        """Computes the initial values for the multiple pressure variables

        Parameters
        ----------
        gravity_strength : np.ndarray or List of shape (3,)
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
            # In absence of gravity, we set every pressure to 1.
            self.hydrostate.cell_vars[...] = 1.0
            self.hydrostate.node_vars[...] = 1.0


def simple_test():
    from atmpy.variables.utility import compute_stratification

    R_gas = 287.4
    R_vap = 461.0
    Q_vap = 2.53e06
    gamma = 1.4

    h_ref = 10000.0
    t_ref = 100.0
    T_ref = 300.00
    p_ref = 1e5
    u_ref = h_ref / t_ref
    scale_factor = 20.0
    delth = 0.01 / T_ref
    xc = 0.0
    gravity = [0, 1.0, 0.0]
    Nsq_ref = 1.0e-4

    Msq = u_ref * u_ref / (R_gas * T_ref)
    a = scale_factor * 5.0e3 / h_ref
    axis = 1

    xmin = -15.0 * scale_factor
    xmax = 15.0 * scale_factor
    ymin = 0.0
    ymax = 1.0
    zmin = -1.0
    zmax = 1.0

    def stratification(y):
        Nsq = Nsq_ref * t_ref * t_ref
        g = gravity[1] / Msq

        return np.exp(Nsq * y / g)

    def molly(x):
        del0 = 0.25
        L = xmax - xmin
        xi_l = np.minimum(1.0, (x - xmin) / (del0 * L))
        xi_r = np.minimum(1.0, (xmax - x) / (del0 * L))
        return 0.5 * np.minimum(1.0 - np.cos(np.pi * xi_l), 1.0 - np.cos(np.pi * xi_r))

    dims = [DimensionSpec(301 + 1, xmin, xmax, 2), DimensionSpec(10 + 1, ymin, ymax, 2)]
    grid = create_grid(dims)

    x = grid.x_cells.reshape(-1, 1)
    y = grid.y_cells.reshape(1, -1)

    xn = grid.x_nodes[:-1].reshape(-1, 1)
    yn = grid.y_nodes[:-1].reshape(1, -1)

    Y = stratification(y) + delth * molly(x) * np.sin(np.pi * y) / (
        1.0 + (x - xc) ** 2 / (a**2)
    )

    Yn = stratification(yn) + delth * molly(xn) * np.sin(np.pi * yn) / (
        1.0 + (xn - xc) ** 2 / (a**2)
    )

    mpv = MPV(grid)
    mpv.state(gravity, Msq)
    hydrostatics = Variables(grid, num_vars_cell=7, num_vars_node=7)
    hydrostatics2 = Variables(grid, num_vars_cell=7, num_vars_node=7)
    compute_stratification(hydrostatics, Y, Yn, grid, axis, gravity, Msq)
    column(hydrostatics, Y, Yn, grid, gravity, Msq)
    print(hydrostatics.cell_vars[4, :, HI.P0])


    # End of compute_stratification method


if __name__ == "__main__":
    simple_test()
    # dt = 0.1
    #
    # nx = 1
    # ngx = 2
    # nnx = nx + 2 * ngx
    # ny = 2
    # ngy = 2
    # nny = ny + 2 * ngy
    #
    # dim = [DimensionSpec(nx, 0, 2, ngx), DimensionSpec(ny, 0, 2, ngy)]
    # grid = create_grid(dim)
    # rng = np.random.default_rng()
    # arr = np.arange(nnx * nny)
    # rng.shuffle(arr)
    #
    # mvp = MPV(grid)
    #
    # mvp.state([0, 1, 0], 1.)
    # print(mvp.hydrostate.cell_vars[...])
