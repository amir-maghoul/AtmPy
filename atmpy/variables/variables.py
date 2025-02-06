from abc import abstractmethod
import numpy as np
from atmpy.data.enums import VariableIndices as VI, PrimitiveVariableIndices as PVI
from atmpy.grid.utility import DimensionSpec, create_grid
from atmpy.physics import eos


class Variables:
    """
    A unified class for storing both cell-centered and node-based conservative variables.

    Attributes
    ----------
    cell_vars : np.ndarray
        Array storing cell-centered conservative variables.
    node_vars : np.ndarray
        Array storing node-based conservative variables.
    primitives : np.ndarray of shape = (cell_vars.shape)
        Array of primitive variables. Since everything is divided by rho, so rho is dropped but pressure is added
        so the number of variables remain the same as cell_vars.
    grid : atmpy.grid.kgrid.Grid
        The computational grid.
    num_vars_cell : int
        Number of cell-centered variables.
    num_vars_node : int
        Number of node-based variables.
    ndim : int
        Number of spatial dimensions.
    """

    def __init__(self, grid, num_vars_cell: int, num_vars_node: int):
        """
        Initializes the VariableContainer with cell-centered and node-based variables.

        Parameters
        ----------
        grid : atmpy.grid.kgrid.Grid
            The computational grid.
        num_vars_cell : int
            Number of cell-centered variables.
        num_vars_node : int
            Number of node-based variables.
        """
        self.grid = grid
        self.num_vars_cell = num_vars_cell
        self.num_vars_node = num_vars_node
        self.ndim = grid.dimensions

        if self.ndim not in [1, 2, 3]:
            raise ValueError("Number of dimensions not supported.")

        # Initialize cell-centered variables
        if self.num_vars_cell is not None and self.num_vars_cell >= 4:
            if self.ndim == 1:
                self.cell_vars = np.zeros((self.grid.ncx_total, self.num_vars_cell))
            elif self.ndim == 2:
                self.cell_vars = np.zeros(
                    (self.grid.ncx_total, self.grid.ncy_total, self.num_vars_cell)
                )
            elif self.ndim == 3:
                self.cell_vars = np.zeros(
                    (
                        self.grid.ncx_total,
                        self.grid.ncy_total,
                        self.grid.ncz_total,
                        self.num_vars_cell,
                    )
                )
        else:
            raise ValueError(
                "Number of cell-based variables should not be None or less than 4."
            )

        self.primitives = np.zeros(self.cell_vars.shape)

        # Initialize node-based variables
        if self.num_vars_node is not None and self.num_vars_node > 0:
            nnx = self.grid.nnx_total
            if self.ndim == 1:
                self.node_vars = np.zeros((nnx, self.num_vars_node))
            elif self.ndim == 2:
                nny = self.grid.nny_total
                self.node_vars = np.zeros((nnx, nny, self.num_vars_node))
            elif self.ndim == 3:
                nny = self.grid.nny_total
                nnz = self.grid.nnz_total
                self.node_vars = np.zeros((nnx, nny, nnz, self.num_vars_node))
        else:
            raise ValueError(
                "Number of node-based variables should not be None, zero or negative."
            )

    def print_debug_info(self):
        """
        Prints debugging information about the VariableContainer.
        """
        print(f"{self.__class__.__name__} Info:")
        print("Dimensions:", self.ndim)
        print("Number of cell vars:", self.num_vars_cell)
        print("Number of node vars:", self.num_vars_node)
        if hasattr(self, "cell_vars"):
            print("Shape of cell_vars:", self.cell_vars.shape)
        if hasattr(self, "node_vars"):
            print("Shape of node_vars:", self.node_vars.shape)
        if hasattr(self, "primitives"):
            print("Shape of primitives:", self.primitives.shape)

    # --------------------
    # Cell-Centered Methods
    # --------------------

    def get_cell_vars(self):
        """
        Retrieves the cell-centered conservative variables.

        Returns
        -------
        np.ndarray
            The cell-centered variable array.
        """
        return self.cell_vars

    def update_cell_vars(self, new_values):
        """
        Updates the cell-centered conservative variables.

        Parameters
        ----------
        new_values : np.ndarray
            New values for the cell-centered variables.
        """
        if new_values.shape != self.cell_vars.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.cell_vars.shape}, got {new_values.shape}"
            )
        self.cell_vars = new_values

    def to_primitive(self, eos):
        """
        Converts cell-centered conservative variables to primitive variables.

        The conservative variables are assumed to be ordered as follows:
        - 1D: [rho, rhoX, rhoY, rho*u]
        - 2D: [rho, rhoX, rhoY, rho*u, rho*v]
        - 3D: [rho, rhoX, rhoY, rho*u, rho*v, rho*w]

        Parameters
        ----------
        eos : :py:class:`atmpy.physics.eos.EOS`
            The equation of state.

        Returns
        -------
        np.ndarray
            Array of primitive variables with one additional dimension.
        """
        ndim = self.ndim

        rho = self.cell_vars[..., VI.RHO]
        self.primitives[..., PVI.P] = eos.pressure(self.cell_vars[..., VI.RHOY])
        self.primitives[..., PVI.U] = self.cell_vars[..., VI.RHOU] / rho
        self.primitives[..., PVI.X] = self.cell_vars[..., VI.RHOX] / rho
        self.primitives[..., PVI.Y] = self.cell_vars[..., VI.RHOY] / rho

        if ndim == 2:
            self.primitives[..., PVI.V] = self.cell_vars[..., VI.RHOV] / rho
        elif ndim == 3:
            self.primitives[..., PVI.V] = self.cell_vars[..., VI.RHOV] / rho
            self.primitives[..., PVI.W] = self.cell_vars[..., VI.RHOW] / rho

        elif ndim > 3 or ndim < 1:
            raise ValueError("Unsupported number of dimensions.")

    # -------------------
    # Node-Based Methods
    # -------------------

    def to_conservative(self):
        ndim = self.ndim
        rho = self.primitives[..., PVI.P] / self.primitives[..., PVI.Y]
        self.cell_vars[..., VI.RHO] = rho
        self.cell_vars[..., VI.RHOX] = rho * self.primitives[..., PVI.X]
        self.cell_vars[..., VI.RHOY] = self.primitives[
            ..., PVI.P
        ]  # Remember P = rho*Theta = rho*Y
        self.cell_vars[..., VI.RHOU] = rho * self.primitives[..., PVI.U]

        if ndim == 2:
            self.cell_vars[..., VI.RHOV] = rho * self.primitives[..., PVI.V]
        elif ndim == 3:
            self.cell_vars[..., VI.RHOV] = rho * self.primitives[..., PVI.V]
            self.cell_vars[..., VI.RHOW] = rho * self.primitives[..., PVI.W]
        elif ndim > 3 or ndim < 1:
            raise ValueError("Unsupported number of dimensions.")

    def get_node_vars(self):
        """
        Retrieves the node-based variables.

        Returns
        -------
        np.ndarray
            The node-based variable array.
        """
        return self.node_vars

    def update_node_vars(self, new_values):
        """
        Updates the node-based variables.

        Parameters
        ----------
        new_values : np.ndarray
            New values for the node-based variables.
        """
        if new_values.shape != self.node_vars.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.node_vars.shape}, got {new_values.shape}"
            )
        self.node_vars = new_values
