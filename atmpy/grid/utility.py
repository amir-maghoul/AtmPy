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
    >>> dims = [DimensionSpec(5, 0, 3, 2), DimensionSpec(6, 1, 4, 3)]
    >>> to_grid_args(dims) # doctest: +NORMALIZE_WHITESPACE
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


def cell_to_node_average(
    grid: Grid, var_cells: np.ndarray, var_nodes: np.ndarray = None
) -> np.ndarray:
    """Averages the values of the primary/secondary variables from cells onto nodes

    Parameters
    ----------
    grid : :py:class:`~atmpy.grid.kgrid.Grid`
        grid object on which the averaging takes place

    var_cells : np.ndarray
        the discrete function values (defined on the cells) from which the averaging takes place

    Returns
    -------
    np.ndarray
        The values of the var_cells averages on nodes
    """
    if var_nodes is None:
        var_nodes = np.zeros(grid.nshape)
    else:
        if var_nodes.shape != grid.nshape:
            raise ValueError(
                "Not an expected shape for the given variable evaluated on nodes"
            )

    if grid.dimensions == 1:
        return _cell_to_node_average_1d(grid, var_cells, var_nodes)
    elif grid.dimensions == 2:
        return _cell_to_node_average_2d(grid, var_cells, var_nodes)
    elif grid.dimensions == 3:
        return _cell_to_node_average_3d(grid, var_cells, var_nodes)
    else:
        raise ValueError("Invalid grid dimension")


def node_to_cell_average(
    grid: Grid, var_nodes: np.ndarray, var_cells: np.ndarray = None
) -> np.ndarray:
    """Averages the values of the primary/secondary variables from nodes onto cells

       Parameters
    ----------
    grid : :py:class:`~atmpy.grid.kgrid.Grid`
        grid object on which the averaging takes place

    var_nodes : np.ndarray
        the discrete function values (defined on the nodes) from which the averaging takes place

    Returns
    -------
    np.ndarray
        The values of the var_cells averages on nodes
    """

    if var_cells is None:
        var_cells = np.zeros(grid.cshape)
    else:
        if var_cells.shape != grid.cshape:
            raise ValueError(
                "Not an expected shape for the given variable evaluated on cells"
            )

    if grid.dimensions == 1:
        return _node_to_cell_average_1d(grid, var_nodes, var_cells)
    elif grid.dimensions == 2:
        return _node_to_cell_average_2d(grid, var_nodes, var_cells)
    elif grid.dimensions == 3:
        return _node_to_cell_average_3d(grid, var_nodes, var_cells)
    else:
        raise ValueError("Invalid grid dimension")


def _cell_to_node_average_1d(
    grid: Grid, var_cells: np.ndarray, var_nodes: np.ndarray = None
) -> np.ndarray:
    """
    Compute the 1D cell-to-node averaging for a given grid and cell-centered variable array.

    In 1D, each inner node value is computed as the average of the two adjacent cells.
    Ghost nodes remain unchanged, as we never overwrite them.

    Parameters:
        grid: :py:class:`~atmpy.grid.kgrid.Grid`
        var_cells (np.ndarray): A 1D array of cell-centered values of shape (ncx_total,).
        var_nodes : np.ndarray, default=None
            A 1D array of node-centered values of shape (nx,).
            If it is None, an array of zeros is created.

    Returns:
        np.ndarray: A 1D array of node-centered values with shape (nnx_total,).

    """
    i_slice_node = slice(grid.ngx, grid.ngx + grid.nx + 1)

    # Pad the var_cells by one cell in each direction (to create an array with the same shape as var_nodes)
    # This makes indexing easier: now padded[i] corresponds to var_cells[i-1],
    # and we can form averages with consistent slicing.
    padded = np.pad(var_cells, pad_width=1, mode="constant", constant_values=0.0)
    # var_nodes[i] = 0.5 * (var_cells[i+1] + var_cells[i]) for i in inner region
    temp = 0.5 * (padded[:-1] + padded[1:])
    var_nodes[i_slice_node] = temp[i_slice_node]
    return var_nodes


def _node_to_cell_average_1d(
    grid: Grid, var_nodes: np.ndarray, var_cells: np.ndarray = None
) -> np.ndarray:
    """
    Compute the 1D node-to-cell averaging for a given grid and node-centered variable array.

    In 1D, each inner cell is computed as the average of the two adjacent nodes.
    Ghost cells remain unchanged, as we never overwrite them.

    Parameters:
        grid: :py:class:`~atmpy.grid.kgrid.Grid`
        var_nodes (np.ndarray): A 1D array of node-centered values of shape (nnx_total,).

    Returns:
        np.ndarray: A 1D array of cell-centered values with shape (ncx_total,).

    """
    i_slice_node = slice(grid.ngx, grid.ngx + grid.nx)

    # var_cells[i] = 0.5 * (var_nodes[i] + var_nodes[i+1]) for inner region
    # Temp will have the same shape as nodes
    temp = 0.5 * (var_nodes[:-1] + var_nodes[1:])
    var_cells[i_slice_node] = temp[i_slice_node]
    return var_cells


def _cell_to_node_average_2d(
    grid: Grid, var_cells: np.ndarray, var_nodes: np.ndarray = None
) -> np.ndarray:
    """
    Evaluate a variable on nodes of the grid using averaging the already evaluated values on cells.

    Parameters
    ----------
    grid : :py:class:`~atmpy.grid.kgrid.Grid`
        grid object on which the variable is evaluated.
    var_cells : np.ndarray
    A 1D array of cell-centered values of shape (ncx_total,ncx_total).
    var_nodes : np.ndarray
    A 1D array of node-centered values of shape (nnx_total,nny_total).

    Returns
    -------
    np.ndarray of shape (nnx_total,nny_total)
    """
    i_slice_node = slice(grid.ngx, grid.ngx + grid.nx + 1)
    j_slice_node = slice(grid.ngy, grid.ngy + grid.ny + 1)

    # Pad the var_cells by one cell in each direction (to create an array with the same shape as var_nodes)
    # This makes indexing easier: now padded[i,j] corresponds to var_cells[i-1,j-1],
    # and we can form averages with consistent slicing.
    padded = np.pad(var_cells, pad_width=1, mode="constant", constant_values=0.0)
    # padded shape: (nnx_total+1, nny_total+1)

    # Note: The indexing here corresponds to shifted indices after padding.
    # Temp will have the same shape as nodes
    temp = 0.25 * (
        padded[:-1, :-1]  # top-left corner
        + padded[:-1, 1:]  # top-right corner
        + padded[1:, :-1]  # bottom-left corner
        + padded[1:, 1:]  # bottom-right corner
    )
    # temp shape: (nnx_total, nny_total)
    # fill out the inner nodes of the var_nodes
    var_nodes[i_slice_node, j_slice_node] = temp[i_slice_node, j_slice_node]
    return var_nodes


def _node_to_cell_average_2d(
    grid: Grid, var_nodes: np.ndarray, var_cells: np.ndarray = None
) -> np.ndarray:
    """
    Evaluate a variable on cells of the grid using averaging the already evaluated values on nodes.

    Parameters
    ----------
    grid : :py:class:`~atmpy.grid.kgrid.Grid`
        grid object on which the variable is evaluated.
    var_nodes : np.ndarray
        A 1D array of node values of shape `grid.nshape`.
    var_cells : np.ndarray
        A 1D array of cell values of shape `grid.cshape`.

    Returns
    -------
    ndarray of shape `grid.cshape`
    """

    i_slice_node = slice(grid.ngx, grid.ngx + grid.nx)
    j_slice_node = slice(grid.ngy, grid.ngy + grid.ny)

    # Compute cell_data by averaging the four corner nodes.
    temp = 0.25 * (
        var_nodes[:-1, :-1]  # top-left nodes
        + var_nodes[1:, :-1]  # bottom-left nodes
        + var_nodes[:-1, 1:]  # top-right nodes
        + var_nodes[1:, 1:]  # bottom-right nodes
    )
    var_cells[i_slice_node, j_slice_node] = temp[i_slice_node, j_slice_node]
    return var_cells


def _cell_to_node_average_3d(grid, var_cells, var_nodes):
    """
    Evaluate a variable on nodes of the grid using averaging the already evaluated values on cells.

    Parameters
    ----------
    grid : :py:class:`~atmpy.grid.kgrid.Grid`
        grid object on which the variable is evaluated.
    var_cells : np.ndarray
    A 1D array of cell-centered values of shape (ncx_total,ncx_total).
    var_nodes : np.ndarray
    A 1D array of node-centered values of shape (nnx_total,nny_total).

    Returns
    -------
    np.ndarray of shape (nnx_total,nny_total)
    """
    i_slice_node = slice(grid.ngx, grid.ngx + grid.nx + 1)
    j_slice_node = slice(grid.ngy, grid.ngy + grid.ny + 1)
    k_slice_node = slice(grid.ngz, grid.ngz + grid.nz + 1)

    # Pad cell_data by one cell in each dimension for easier indexing
    padded = np.pad(var_cells, pad_width=1, mode="constant", constant_values=0.0)
    # padded shape: (nx_full+2, ny_full+2, nz_full+2)

    # Each node is the average of 8 surrounding cells.
    # node_data[i,j,k] = average of padded[i:i+2, j:j+2, k:k+2]
    # Temp will have the same shape as nodes
    temp = 0.125 * (
        padded[:-1, :-1, :-1]
        + padded[:-1, :-1, 1:]
        + padded[:-1, 1:, :-1]
        + padded[:-1, 1:, 1:]
        + padded[1:, :-1, :-1]
        + padded[1:, :-1, 1:]
        + padded[1:, 1:, :-1]
        + padded[1:, 1:, 1:]
    )
    var_nodes[i_slice_node, j_slice_node, k_slice_node] = temp[
        i_slice_node, j_slice_node, k_slice_node
    ]
    return var_nodes


def _node_to_cell_average_3d(
    grid: Grid, var_nodes: np.ndarray, var_cells: np.ndarray = None
) -> np.ndarray:
    """
    Evaluate a variable on cells of the grid using averaging the already evaluated values on nodes.

    Parameters
    ----------
    grid : :py:class:`~atmpy.grid.kgrid.Grid`
        grid object on which the variable is evaluated.
    var_nodes : np.ndarray
        A 1D array of node values of shape `grid.nshape`.
    var_cells : np.ndarray
        A 1D array of cell values of shape `grid.cshape`.

    Returns
    -------
    ndarray of shape `grid.cshape`
    """

    i_slice_node = slice(grid.ngx, grid.ngx + grid.nx)
    j_slice_node = slice(grid.ngy, grid.ngy + grid.ny)
    k_slice_node = slice(grid.ngz, grid.ngz + grid.nz)

    # cell_data[i,j,k] = average of node_data[i:i+2, j:j+2, k:k+2]
    temp = 0.125 * (
        var_nodes[:-1, :-1, :-1]
        + var_nodes[:-1, :-1, 1:]
        + var_nodes[:-1, 1:, :-1]
        + var_nodes[:-1, 1:, 1:]
        + var_nodes[1:, :-1, :-1]
        + var_nodes[1:, :-1, 1:]
        + var_nodes[1:, 1:, :-1]
        + var_nodes[1:, 1:, 1:]
    )
    var_cells[i_slice_node, j_slice_node, k_slice_node] = temp[
        i_slice_node, j_slice_node, k_slice_node
    ]
    return var_cells


if __name__ == "__main__":
    nx, ny, nz = 2, 3, 4  # number of cells
    ngx, ngy, ngz = 1, 2, 3  # number of ghost cells

    dimensions = [
        DimensionSpec(n=nx, start=0, end=3, ng=ngx),
        DimensionSpec(n=ny, start=0, end=3, ng=ngy),
        DimensionSpec(n=nz, start=0, end=3, ng=ngz),
    ]
    grid = create_grid(dimensions)
