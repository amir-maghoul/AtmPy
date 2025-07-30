from abc import abstractmethod
import numpy as np
from typing import Union, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
from atmpy.infrastructure.enums import (
    VariableIndices as VI,
    PrimitiveVariableIndices as PVI,
)
from atmpy.physics.eos import IdealGasEOS, BarotropicEOS, ExnerBasedEOS
from atmpy.infrastructure.utility import momentum_index
import warnings


class Variables:
    """
    A unified class for storing both cell-centered and node-based conservative variables.

    Notes
    -----
    The assumption is that the number of conservative variables and the number of primitive variables are the same. So
    that if future places such as reconstruction or riemann solvers, one can use the numpy vectorization to do
    calculations on a combination of both.

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
        Number of spatial ndim.
    """

    def __init__(self, grid: "Grid", num_vars_cell: int, num_vars_node: int = 1):
        """
        Initializes the VariableContainer with cell-centered and node-based variables.

        Parameters
        ----------
        grid : atmpy.grid.kgrid.Grid
            The computational grid.
        num_vars_cell : int
            Number of cell-centered variables.
        num_vars_node : int (default = 1)
            Number of node-based variables.
        """
        self.grid: "Grid" = grid
        self.num_vars_cell: int = num_vars_cell
        self.num_vars_node: int = num_vars_node
        self.ndim: int = grid.ndim

        if self.ndim not in [1, 2, 3]:
            raise ValueError("Number of ndim not supported.")

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

        # Notice the hard assumption: The number of primitive variables are the same as
        # the number of conservative variables
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
        - [rho, rhoX, rhoY, rho*u, [rho*v], [rho*w]]

        Parameters
        ----------
        eos : :py:class:`atmpy.physics.eos.EOS`
            The equation of state.

        Returns
        -------
        np.ndarray
            Array of primitive variables with one additional dimension.
        """
        if isinstance(eos, IdealGasEOS):
            raise NotImplementedError(
                "IdealGasEOS not yet supported for primitives: No way to calculate the energy"
            )
        elif isinstance(eos, BarotropicEOS):
            args = self.cell_vars[..., VI.RHO]
        elif isinstance(eos, ExnerBasedEOS):
            args = (self.cell_vars[..., VI.RHOY], True)

        rho = self.cell_vars[..., VI.RHO]
        nonzero_idx = np.nonzero(rho)
        self.primitives[..., PVI.RHO] = rho
        self.primitives[*nonzero_idx, PVI.U] = (
            self.cell_vars[*nonzero_idx, VI.RHOU] / rho[nonzero_idx]
        )
        self.primitives[*nonzero_idx, PVI.X] = (
            self.cell_vars[*nonzero_idx, VI.RHOX] / rho[nonzero_idx]
        )
        self.primitives[*nonzero_idx, PVI.Y] = (
            self.cell_vars[*nonzero_idx, VI.RHOY] / rho[nonzero_idx]
        )

        warnings.warn(
            """For a better performance and a good vectorization in calculations, the current assumption of 
        the project is that the number of primitive variables and the number of conservative variables are the same."""
        )
        if PVI.V < self.num_vars_cell:
            self.primitives[*nonzero_idx, PVI.V] = (
                self.cell_vars[*nonzero_idx, VI.RHOV] / rho[nonzero_idx]
            )
        if PVI.W < self.num_vars_cell:
            self.primitives[*nonzero_idx, PVI.W] = (
                self.cell_vars[*nonzero_idx, VI.RHOW] / rho[nonzero_idx]
            )

    def to_conservative(self, rho: np.ndarray) -> None:
        """Converts the primitive variables to conservative variables using the given rho."""
        self.cell_vars[..., VI.RHO] = rho
        self.cell_vars[..., VI.RHOX] = rho * self.primitives[..., PVI.X]
        self.cell_vars[..., VI.RHOY] = rho * self.primitives[..., PVI.Y]
        self.cell_vars[..., VI.RHOU] = rho * self.primitives[..., PVI.U]

        if VI.RHOV < self.num_vars_cell:
            self.cell_vars[..., VI.RHOV] = rho * self.primitives[..., PVI.V]
        if VI.RHOW < self.num_vars_cell:
            self.cell_vars[..., VI.RHOW] = rho * self.primitives[..., PVI.W]

    def adjust_background_wind(
        self, wind_speeds: Union[np.ndarray, list], scale: float, in_place: bool = True
    ) -> np.ndarray:
        """Modify the momenta using the background wind and the given factor

        Parameters
        ----------
        wind_speeds : Union[np.ndarray, list] of shape (3, 1)
            The list or numpy array containing the wind velocities in each direction.
        scale : float
            The scaling factor

        Returns
        -------
        np.ndarray
        """
        indices = [VI.RHOU, VI.RHOV, VI.RHOW]
        rho = self.cell_vars[..., VI.RHO, np.newaxis]
        adjustment = (np.array(wind_speeds) * scale) * rho

        if not in_place:
            # The non-in-place logic remains the same
            adjusted_momenta = self.cell_vars[..., indices].copy()
            adjusted_momenta += adjustment
            return adjusted_momenta

        rho = self.cell_vars[..., VI.RHO, np.newaxis]
        adjustments = wind_speeds * scale * rho
        self.cell_vars[..., indices] += adjustments

        return self.cell_vars[..., indices]

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
