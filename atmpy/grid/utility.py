from numba import stencil, njit

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


def cell_to_node_average(grid: Grid, var_cells: np.ndarray) -> np.ndarray:
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
    grid : :py:class:`~atmpy.grid.kgrid.Grid`
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


@stencil
def average_kernel_1d(x):
    return (x[-1] + x[0]) // 2


@njit
def cell_to_node_average_1d(
    ngx: int, var_cells: np.ndarray, var_nodes: np.ndarray = None
) -> np.ndarray:
    temp = average_kernel_1d(var_cells)
    # var_nodes[ngx:-ngx] = temp[ngx:-ngx]
    return temp


# def cell_to_node_average_1d(
#     grid: Grid, var_cells: np.ndarray, var_nodes: np.ndarray = None
# ) -> np.ndarray:
#     """
#     Compute the 1D cell-to-node averaging for a given grid and cell-centered variable array.
#
#     In 1D, each inner node value is computed as the average of the two adjacent cells.
#     Ghost nodes remain unchanged, as we never overwrite them.
#
#     Parameters:
#         grid: :py:class:`~atmpy.grid.kgrid.Grid`
#         var_cells (np.ndarray): A 1D array of cell-centered values of shape (nx_total,).
#         var_nodes : np.ndarray, default=None
#             A 1D array of node-centered values of shape (nx,).
#             If it is None, an array of zeros is created.
#
#     Returns:
#         np.ndarray: A 1D array of node-centered values with shape (nnx_total,).
#
#     Notes
#     -----
#         The calculation is done using
#         `var_nodes[ngx:-ngx] = (0.5 * (var_cells + np.roll(var_cells, -1)))[ngx:]`
#         Notice:
#         len(var_cells) == len(var_nodes) - 1 == len(var_nodes[ngx:-ngx]) + 1
#         Therefore after averaging the var_cells, the first entry of the array should be
#         dropped for the shapes to match.
#     """
#     ngx = grid.ngx
#     nnx_total = grid.nnx_total
#
#     if var_nodes is None:
#         var_nodes = np.zeros(nnx_total)
#     else:
#         if len(var_nodes) != nnx_total:
#             raise ValueError(
#                 "Not an expected shape for the given variable evaluated on nodes"
#             )
#
#     # Compute inner nodes:
#     # var_nodes[i] = 0.5 * (var_cells[i-1] + var_cells[i]) for i in inner region
#     # See Notes for details
#     var_nodes[ngx:-ngx] = (0.5 * (var_cells + np.roll(var_cells, -1)))[ngx:]
#     return var_nodes


def node_to_cell_average_1d(
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
        np.ndarray: A 1D array of cell-centered values with shape (nnx_total,).

    Notes
    -----
        The calculation is done using
        `var_cells[ngx:-ngx] = (0.5 * (var_nodes + np.roll(var_nodes, -1)))[ngx:-ngx-1]`
        Notice:
        len(var_nodes) == len(var_cells) + 1 == len(var_cells[ngx:-ngx]) + 2
        Therefore after averaging the var_cells, the first two entries of the array should be
        dropped for the shapes to match.
        Assume:
        ngx = 1
        var_nodes = [-1, 0, 1, 2, 3, 4]
        avg = 0.5*(x + np.roll(x, -1)) = 0.5*[-1, 1, 3, 5, 7, 3]
        len(avg) = 6
        len(inner_cells(var_cells)) = 3
        inner_cell[0] = 0.5 = avg[ngx]
        inner_cell[-1] = 2.5 = avg[-ngx-1]
    """
    ngx = grid.ngx
    ncx_total = grid.ncx_total

    if var_cells is None:
        var_cells = np.zeros(ncx_total)
    else:
        if len(var_cells) != ncx_total:
            raise ValueError(
                "Not an expected shape for the given variable evaluated on cells"
            )

    # var_cells[i] = 0.5 * (var_nodes[i] + var_nodes[i+1]) for inner region
    var_cells[ngx:-ngx] = 0.5 * (var_nodes + np.roll(var_nodes, -1))[ngx : -ngx - 1]

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
        var_cells (np.ndarray): A 3D array of cell-centered values of shape (ncx_total, ncy_total, ncz_total).

    Returns:
        np.ndarray: A 3D array of node-centered values with shape (nnx_total, nny_total, nnz_total).
    """
    ngx, ngy, ngz = grid.ngx, grid.ngy, grid.ngz
    nnx_total, nny_total, nnz_total = grid.nnx_total, grid.nny_total, grid.nnz_total

    var_nodes = np.zeros((nnx_total, nny_total, nnz_total))

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


if __name__ == "__main__":
    dimensions = [DimensionSpec(n=3, start=0, end=3, ng=2)]
    grid = create_grid(dimensions)
    var_nodes = np.zeros(grid.nnx_total)
    var_cells = grid.x_cells

    print(grid.x_cells)
    print(grid.x_nodes)
    print(var_nodes)
    ngx = grid.ngx
    print(grid.ncx_total)
    # print(node_to_cell_average_1d(grid, grid.x_nodes))
    print(cell_to_node_average_1d(grid.ngx, var_cells))
    print((0.5 * (var_cells + np.roll(var_cells, -1))))
