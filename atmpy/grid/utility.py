from atmpy.grid.kgrid import Grid
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class DimensionSpec:
    """The data class for creating the dimensions of the problem"""

    n: int
    start: float
    end: float
    ng: int


def to_grid_args(dimensions: List[DimensionSpec]):
    """Convert list of dimensions to grid arguments

    Parameters
    ----------
    dimensions : List[DimensionSpec]
        List of dimensions in forms of objects of the Dimension class

    Returns
    -------
    dict
        Dictionary of grid arguments

    Examples
    --------
    >>> dimensions = [DimensionSpec(5, 0, 3, 2), DimensionSpec(6, 1, 4, 3)]
    >>> to_grid_args(dimensions) # doctest: +NORMALIZE_WHITESPACE
    {'nx': 5, 'x_start': 0, 'x_end': 3, 'ngx': 2, 'ny': 6, 'y_start': 1, 'y_end': 4, 'ngy': 3}

    """
    # We'll build a dictionary of arguments for the Grid constructor
    # Dimension order: x=0, y=1, z=2
    dim_letters = ["x", "y", "z"]
    args = {}
    for i, dim in enumerate(dimensions):
        letter = dim_letters[i]
        args[f"n{letter}"] = dim.n
        args[f"{letter}_start"] = dim.start
        args[f"{letter}_end"] = dim.end
        args[f"ng{letter}"] = dim.ng
    return args


def create_grid(dimensions: List[DimensionSpec]):
    """Unpacks the dimensions parameter (which is a list of DimensionSpec objects)
    into a dictionary and pass it to create a Grid object using them

    Parameters
    ----------
    dimensions : List[DimensionSpec]
        List of dimensions in forms of objects of the Dimension class

    Returns
    -------
    atmpy.grid.kgrid.Grid
    """
    # Grid(nx, x_start, x_end, ngx, ny=None, y_start=None, y_end=None, ngy=None, nz=None, z_start=None, z_end=None, ngz=None)
    args = to_grid_args(dimensions)
    return Grid(**args)


def cell_to_node_average(grid: Grid, var_cells: np.ndarray) -> np.ndarray:
    """Averages the values of the primary/secondary variables from cells onto nodes

    Parameters
    ----------
    grid : Grid
        grid object on which the averaging takes place

    var_cells : np.ndarray
        the discrete function values (defined on the cells) from which the averaging takes place

    Returns
    -------
    np.ndarray
        The values of the var_cells averages on nodes
    """
    if grid.dimensions == 1:
        return cell_to_node_average_1d(grid, var_cells)
    elif grid.dimensions == 2:
        return cell_to_node_average_2d(grid, var_cells)
    elif grid.dimensions == 3:
        return cell_to_node_average_3d(grid, var_cells)
    else:
        raise ValueError("Invalid grid dimension")


def node_to_cell_average(grid: Grid, var_nodes: np.ndarray) -> np.ndarray:
    """Averages the values of the primary/secondary variables from nodes onto cells

       Parameters
    ----------
    grid : Grid
        grid object on which the averaging takes place

    var_nodes : np.ndarray
        the discrete function values (defined on the nodes) from which the averaging takes place

    Returns
    -------
    np.ndarray
        The values of the var_cells averages on nodes
    """
    if grid.dimensions == 1:
        return node_to_cell_average_1d(grid, var_nodes)
    elif grid.dimensions == 2:
        return node_to_cell_average_2d(grid, var_nodes)
    elif grid.dimensions == 3:
        return node_to_cell_average_3d(grid, var_nodes)
    else:
        raise ValueError("Invalid grid dimension")


def cell_to_node_average_1d(grid, var_cells) -> np.ndarray:
    """
    Compute the 1D cell-to-node averaging for a given grid and cell-centered variable array.

    In 1D, each inner node value is computed as the average of the two adjacent cells.
    Ghost nodes remain unchanged, as we never overwrite them.

    Parameters:
        grid: A Grid object that includes nx_total, ngx attributes.
        var_cells (np.ndarray): A 1D array of cell-centered values of shape (nx_total,).

    Returns:
        np.ndarray: A 1D array of node-centered values with shape (nx_total + 1,).
    """
    ngx = grid.ngx
    nx_total = grid.nx_total

    var_nodes = np.zeros(nx_total + 1)

    # Compute inner nodes:
    # var_nodes[i] = 0.5 * (var_cells[i-1] + var_cells[i]) for i in inner region
    var_nodes[ngx:-ngx] = 0.5 * (
        var_cells[ngx - 1:-ngx] +
        var_cells[ngx:-ngx + 1]
    )

    return var_nodes


def node_to_cell_average_1d(grid, var_nodes):
    """
    Compute the 1D node-to-cell averaging for a given grid and node-centered variable array.

    In 1D, each inner cell is computed as the average of the two adjacent nodes.
    Ghost cells remain unchanged, as we never overwrite them.

    Parameters:
        grid: A Grid object that includes nx_total, ngx attributes.
        var_nodes (np.ndarray): A 1D array of node-centered values of shape (nx_total + 1,).

    Returns:
        np.ndarray: A 1D array of cell-centered values with shape (nx_total,).
    """
    ngx = grid.ngx
    nx_total = grid.nx_total

    var_cells = np.zeros(nx_total)

    # var_cells[i] = 0.5 * (var_nodes[i] + var_nodes[i+1]) for inner region
    var_cells[ngx:-ngx] = 0.5 * (
            var_nodes[ngx:-ngx] +
            var_nodes[ngx + 1:-ngx + 1]
    )

    return var_cells


def cell_to_node_average_2d(grid: Grid, var_cells: np.ndarray) -> np.ndarray:
    pass


def node_to_cell_average_2d(grid: Grid, var_nodes: np.ndarray) -> np.ndarray:
    pass


def cell_to_node_average_3d(grid, var_cells):
    """
    Compute the 3D cell-to-node averaging for a given grid and cell-centered variable array.

    Parameters:
        grid: Grid object
        var_cells (np.ndarray): A 3D array of cell-centered values of shape (nx_total, ny_total, nz_total).

    Returns:
        np.ndarray: A 3D array of node-centered values with shape (nx_total + 1, ny_total + 1, nz_total + 1).
    """
    ngx, ngy, ngz = grid.ngx, grid.ngy, grid.ngz
    nx_total, ny_total, nz_total = grid.nx_total, grid.ny_total, grid.nz_total

    var_nodes = np.zeros((nx_total + 1, ny_total + 1, nz_total + 1))

    # Perform averaging for inner nodes only,
    # leaving ghost nodes unchanged.
    var_nodes[ngx:-ngx, ngy:-ngy, ngz:-ngz] = (
        var_cells[ngx - 1 : -ngx, ngy - 1 : -ngy, ngz - 1 : -ngz]
        + var_cells[ngx : -ngx + 1, ngy - 1 : -ngy, ngz - 1 : -ngz]
        + var_cells[ngx - 1 : -ngx, ngy : -ngy + 1, ngz - 1 : -ngz]
        + var_cells[ngx : -ngx + 1, ngy : -ngy + 1, ngz - 1 : -ngz]
        + var_cells[ngx - 1 : -ngx, ngy - 1 : -ngy, ngz : -ngz + 1]
        + var_cells[ngx : -ngx + 1, ngy - 1 : -ngy, ngz : -ngz + 1]
        + var_cells[ngx - 1 : -ngx, ngy : -ngy + 1, ngz : -ngz + 1]
        + var_cells[ngx : -ngx + 1, ngy : -ngy + 1, ngz : -ngz + 1]
    ) / 8.0

    return var_nodes


def node_to_cell_average_3d(grid: Grid, var_nodes: np.ndarray) -> np.ndarray:
    pass
