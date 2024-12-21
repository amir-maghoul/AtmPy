import numpy as np
from atmpy.variables.variables import CellVariable, NodeVariable

def create_variables(grid, num_cell_vars, num_node_vars):
    """ Creates both cell and node variable containers

        Parameters
        ----------
        grid : :py:class:`atmpy.grid.kgrid.Grid`
            basis grid of the problem
        num_cell_vars : int
            number of cell-based variables in the system of equation to solve
        num_node_vars : int
            number of node-based variables in the system of equation to solve

        Returns
        -------
        tuple
            of form (cell_vars, node_vars)
        """
    cell_vars = CellVariable(grid, num_cell_vars)
    node_vars = NodeVariable(grid, num_node_vars)
    return cell_vars, node_vars
