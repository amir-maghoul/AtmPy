"""This module handles multiple pressure variables. It is used to calculate the hydrostate pressure.
    The code is the modified version of what appears in PyBella project.
"""
import numpy as np
from atmpy.flux.utility import direction_mapping
from atmpy.grid.kgrid import Grid
from atmpy.grid.utility import DimensionSpec, create_grid
from atmpy.infrastructure.enums import HydrostateIndices as HI
from atmpy.variables.variables import Variables
import scipy as sp

class MPV:
    """ Container for 'Multiple Pressure Variables'

    Attributes
    ----------
    grid : Grid
        the Grid object
    grid1D : Grid
        the 1D version of the self.grid in the direction of hydrostacity
    """
    def __init__(self, grid, num_vars=6, direction="y"):
        """ Initialize the container for pressure variables.

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
        self.p2_nodes0: np.ndarray  = np.zeros(grid.nshape)
        self.dp2_nodes: np.ndarray = np.zeros(grid.nshape)

        self.u: np.ndarray = np.zeros(grid.cshape)
        self.v: np.ndarray = np.zeros(grid.cshape)
        self.w: np.ndarray = np.zeros(grid.cshape)

        self.rhs: np.ndarray = np.zeros(grid.nshape)
        self.wcenter: np.ndarray = np.zeros(grid.nshape)
        self.wplus: np.ndarray = np.zeros((grid.ndim,) + grid.cshape)

        # The variable container for hydrostate
        self.hydrostate: Variables= Variables(self.grid1D, num_vars_cell=num_vars, num_vars_node=num_vars)

    def _create_1D_grid_in_direction(self):
        """ Reduce the dimension of the Grid object and create a new one in the given direction """

        start = getattr(self.grid, self.direction_str +"_start")
        end = getattr(self.grid, self.direction_str + "_end")
        ncells = getattr(self.grid, "n" + self.direction_str)
        nghosts = getattr(self.grid, "ng" + self.direction_str)

        dims = [DimensionSpec(ncells, start, end, nghosts)]
        grid = create_grid(dims)

        return grid

    def compute_dS_on_nodes(self, direction):
        """ Compute the derivative of S with respect to the direction of gravity. The S variable is assumed to be
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
        ndim = self.grid.ndim  # Use the vertical spacing defined by the grid.
        dr = self.grid.dxyz[self.direction]

        # Since variables in self.hydrostate are 1D, it suffices to calulate the convolotion in their only direction
        # and then divide the result be the correct dx, dy or dz to get the derivative.
        S0 = self.hydrostate.node_vars[..., HI.S0]
        dS = np.diff(S0) / dr

        # Expand and repeat along even axes so that the result matches the cell array shape.
        tile_shape = self._get_tile_shape()
        dS = np.tile(dS, tile_shape)
        return dS

    def get_S0c_on_cells(self):
        """ Get method to get the S0c attribute. The S0 variable is assumed to be defined on the cells.
        """
        S0c = self.hydrostate.cell_vars[..., HI.S0]

        # Expand and repeat along even axes so that the result matches the cell array shape.
        tile_shape = self._get_tile_shape()
        S0c = np.tile(S0c, tile_shape)
        return S0c

    def _get_tile_shape(self):
        tile_shape = list(self.grid.cshape)
        tile_shape[self.direction] = 1
        return tile_shape
