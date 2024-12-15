import numpy as np


class Variable:
    def __init__(self, grid, num_vars):
        """
        Variable container for conservative state variables.

        Parameters
        ----------
        grid : :py:class:`atmpy.grid.kgrid.Grid`
            The grid object of the problem
        num_vars : int
            Number of variables (e.g. for Euler: density, momentum components, energy)
        """
        self.grid = grid
        self.num_vars = num_vars

        # Allocate conservative variable array based on dimension
        if self.grid.dimensions == 1:
            self.cons_vars = np.zeros((self.grid.nx, num_vars))
        elif self.grid.dimensions == 2:
            self.cons_vars = np.zeros((self.grid.nx, self.grid.ny, num_vars))
        elif self.grid.dimensions == 3:
            self.cons_vars = np.zeros(
                (self.grid.nx, self.grid.ny, self.grid.nz, num_vars)
            )
        else:
            raise ValueError("Number of dimensions not supported.")

    def get_conservative_vars(self):
        """
        Return the array of conservative variables.
        """
        return self.cons_vars

    def update_vars(self, new_values):
        """
        Update the state variables with new values (e.g. after a time step).

        Parameters
        ----------
        new_values : np.ndarray
            New values to assign to self.cons_vars (matching dimension and shape).
        """
        self.cons_vars = new_values
