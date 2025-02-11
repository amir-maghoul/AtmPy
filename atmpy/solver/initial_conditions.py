""" This module containes initial conditions for the problem. The structure is such that the functions
should have grid and variable container as the input and update variables in-place."""

import numpy as np


def sod_shock_tube_initial_conditions(grid, cell_vars, node_vars):
    """
    Initialize variable container in-place.

    Parameters:
        grid : :py:class:`atmpy.grid.kgrid.Grid`
            The grid object of the problem

        cell_vars : :py:class:`atmpy.variables.variables.CellVariable`
            The variable container for cell-based variables.

        node_vars : :py:class:`atmpy.variables.variables.NodeVariable`
            The variable container for node-based variables.

    Returns:
        None
    """
    cell_vars.vars[..., 1] += 1
