from abc import abstractmethod
import numpy as np
from atmpy.data.constants import VariableIndices
from atmpy.grid.utility import DimensionSpec, create_grid


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
        if self.num_vars_cell is not None and self.num_vars_cell > 0:
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
            raise ValueError("Number of cell-based variables should not be None, zero or negative.")

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
            raise ValueError("Number of node-based variables should not be None, zero or negative.")

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

    def to_primitive(self, gamma):
        """
        Converts cell-centered conservative variables to primitive variables.

        The conservative variables are assumed to be ordered as follows:
        - 1D: [rho, rhoX, rhoY, rho*u]
        - 2D: [rho, rhoX, rhoY, rho*u, rho*v]
        - 3D: [rho, rhoX, rhoY, rho*u, rho*v, rho*w]

        Parameters
        ----------
        gamma : float
            The constant adiabatic ration (ration of specific heats)

        Returns
        -------
        np.ndarray
            Array of primitive variables with one additional dimension.
        """
        ndim = self.ndim
        RHO, RHOX, RHOY, RHOU, RHOV, RHOW = VariableIndices.values()

        rho = self.cell_vars[..., RHO]
        self.primitives[..., 0] = self.cell_vars[..., RHOY] ** gamma        # Pressure p
        self.primitives[..., RHOU] = self.cell_vars[..., RHOU] / rho        #
        self.primitives[..., RHOX] = self.cell_vars[..., RHOX] / rho
        self.primitives[..., RHOY] = self.cell_vars[..., RHOY] / rho

        if ndim == 2:
            self.primitives[..., RHOV] = self.cell_vars[..., RHOV] / rho
        elif ndim == 3:
            self.primitives[..., RHOV] = self.cell_vars[..., RHOV] / rho
            self.primitives[..., RHOW] = self.cell_vars[..., RHOW] / rho


        elif ndim > 3 or ndim < 1:
            raise ValueError("Unsupported number of dimensions.")



    # -------------------
    # Node-Based Methods
    # -------------------

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

