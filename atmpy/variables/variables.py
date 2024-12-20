from abc import abstractmethod

import numpy as np
from atmpy.data.constants import VariableIndices1D, VariableIndices2D, VariableIndices3D
from atmpy.grid.utility import DimensionSpec, create_grid


class BaseVariableContainer:
    """Base class for all conservative variables"""

    def __init__(self, grid, num_vars):
        """Constructor of the base class

        Parameters
        ----------
        grid : :py:class:`atmpy.grid.kgrid.Grid`
            basis grid of the problem

        num_vars : int
            number of variables in the system of equation to solve

        """
        self.grid = grid
        self.num_vars = num_vars
        self.ndim = grid.dimensions

        if self.ndim not in [1, 2, 3]:
            raise ValueError("Number of dimensions not supported.")

    def print_debug_info(self):
        print(f"{self.__class__.__name__} Info:")
        print("Dimensions:", self.ndim)
        print("Number of vars:", self.num_vars)
        if hasattr(self, "vars"):
            print("Shape of vars:", self.vars.shape)


class CellVariable(BaseVariableContainer):
    """
    A class for storing cell-centered state of the conservative variables.

    This class stores conservative variables at the cell centers
    of the computational grid.

    Examples of variables:
      - For Euler equations: [rho, rho*u, rho*v, rho*w, P, PX]
      - The shape of the storage depends on the number of dimensions (ndim):
         - 1D: (num_cells_x, num_vars)
         - 2D: (num_cells_x, num_cells_y, num_vars)
         - 3D: (num_cells_x, num_cells_y, num_cells_z, num_vars)
    """

    def __init__(self, grid, num_vars):
        super().__init__(grid, num_vars)
        # Allocate cell-based conservative variables
        if self.ndim == 1:
            self.vars = np.zeros((self.grid.ncx_total, self.num_vars))
        elif self.ndim == 2:
            self.vars = np.zeros(
                (self.grid.ncx_total, self.grid.ncy_total, self.num_vars)
            )
        elif self.ndim == 3:
            self.vars = np.zeros(
                (
                    self.grid.ncx_total,
                    self.grid.ncy_total,
                    self.grid.ncz_total,
                    self.num_vars,
                )
            )

    def get_conservative_vars(self):
        """
        Return the array of cell-centered variables.

        Returns
        -------
        np.ndarray
            The cell-centered variable array.
        """
        return self.vars

    def update_vars(self, new_values):
        """
        Update the cell-centered variables with new values.

        Parameters
        ----------
        new_values : np.ndarray
            The new variable values to be assigned, matching the shape of self.vars.
        """
        self.vars = new_values


    def to_primitive(self):
        """
        Convert conservative variables to primitive variables.

        The conservative variables are:
        - 1D: [ρ, ρu, P, Pχ]
        - 2D: [ρ, ρu, ρv, P, Pχ]
        - 3D: [ρ, ρu, ρv, ρw, P, Pχ]

        Here:
        P = ρ * theta
        χ = 1 / theta
        and Pχ = ρ

        From Pχ, we get ρ = Pχ directly.
        Then, theta = P / ρ and χ = 1 / theta.

        The primitive variables become:
        - 1D: [ρ, u, theta, χ]
        - 2D: [ρ, u, v, theta, χ]
        - 3D: [ρ, u, v, w, theta, χ]

        Returns
        -------
        primitive_vars : np.ndarray
            Array of primitive variables with shape matching `self.vars` in all but the last dimension.
        """
        variables = self.vars
        ndim = self.grid.dimensions

        # Extract ρ directly from Pχ = ρ
        # Indices differ with dimension:
        # ndim=1: variables[..., RHO, RHOU, P, PX]
        # ndim=2: variables[..., RHO, RHOU, RHOV, P, PX]
        # ndim=3: variables[..., RHO, RHOU, RHOV, RHOW, P, PX]

        if ndim == 1:
            RHO, RHOU, P, PX = VariableIndices1D.values()
            rho = variables[..., RHO]
            u = variables[..., RHOU] / rho
            P = variables[..., P]
            PX = variables[..., PX]
            # No v, w in 1D
            X = PX / P
            # Primitive: [u, P, X]
            primitive_vars = np.stack((u, P, X), axis=-1)

        elif ndim == 2:
            RHO, RHOU, RHOV, P, PX = VariableIndices2D.values()
            rho = variables[..., RHO]
            u = variables[..., RHOU] / rho
            v = variables[..., RHOV] / rho
            P = variables[..., P]
            PX = variables[..., PX]
            X = PX / P
            # Primitive: [ρ, u, v, theta, χ]
            primitive_vars = np.stack((u, v, P, X), axis=-1)

        elif ndim == 3:
            RHO, RHOU, RHOV, RHOW, P, PX = VariableIndices3D.values()
            rho = variables[..., RHO]
            u = variables[..., RHOU] / rho
            v = variables[..., RHOV] / rho
            w = variables[..., RHOW] / rho
            P = variables[..., P]
            PX = variables[..., PX]
            X = PX / P
            # Primitive: [ρ, u, v, w, P, χ]
            primitive_vars = np.stack((u, v, w, P, X), axis=-1)

        else:
            raise ValueError("Unsupported number of dimensions.")

        return primitive_vars


class NodeVariable(BaseVariableContainer):
    """
    A class for storing node-based conservative variables.

    This class stores variables at the nodes (grid vertices) of the computational domain.
    It is especially useful when you need a variable defined at nodes rather than cells.

    For example, if you have a variable like the Exner pressure (π) that must reside at nodes:
      - 1D: (num_cells_x + 1, num_vars)
      - 2D: (num_cells_x + 1, num_cells_y + 1, num_vars)
      - 3D: (num_cells_x + 1, num_cells_y + 1, num_cells_z + 1, num_vars)
    """

    def __init__(self, grid, num_vars):
        super().__init__(grid, num_vars)
        # Allocate node-based variables
        # Assuming a structured Cartesian grid, nodes are typically one more
        # than the number of cells in each dimension.
        nnx = self.grid.nnx_total
        if self.ndim == 1:
            self.vars = np.zeros((nnx, self.num_vars))
        elif self.ndim == 2:
            nny = self.grid.nny_total
            self.vars = np.zeros((nnx, nny, self.num_vars))
        elif self.ndim == 3:
            nny = self.grid.nny_total
            nnz = self.grid.nnz_total
            self.vars = np.zeros((nnx, nny, nnz, self.num_vars))

    def get_node_vars(self):
        """
        Return the array of node-based variables.

        Returns
        -------
        np.ndarray
            The node-based variable array.
        """
        return self.vars

    def update_node_vars(self, new_values):
        """
        Update the node-based variables with new values.

        Parameters
        ----------
        new_values : np.ndarray
            The new variable values to be assigned, matching the shape of self.vars.
        """
        self.vars = new_values


