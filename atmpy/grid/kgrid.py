import numpy as np
from typing import Optional, Tuple
from numba import njit, stencil


class Grid:
    """
    Basic grid data structure for the FVM solver.

    Attributes
    ----------
    nx : int
        Number of cells in the x-direction.
    x_start : float
        Starting coordinate in the x-direction.
    x_end : float
        Ending coordinate in the x-direction.
    ngx : int, optional
        Number of ghost cells in the x-direction. Default is 2.
    ny : int, optional
        Number of cells in the y-direction (for 2D/3D grids).
    y_start : float, optional
        Starting coordinate in the y-direction.
    y_end : float, optional
        Ending coordinate in the y-direction.
    ngy : int, optional
        Number of ghost cells in the y-direction.
    nz : int, optional
        Number of cells in the z-direction (for 3D grids).
    z_start : float, optional
        Starting coordinate in the z-direction.
    z_end : float, optional
        Ending coordinate in the z-direction.
    ngz : int, optional
        Number of ghost cells in the z-direction.
    dimensions : int
        dimension of the grid
    inner_slice_x, inner_slice_y, inner_slice_z : slice
        the index slices of inner cells/nodes in each direction
    nx_total : int
        Total number of cells in the x-direction (inner cells + ghost cells)
    ny_total : int
        Total number of cells in the y-direction (inner cells + ghost cells)
    nz_total : int
        Total number of cells in the z-direction (inner cells + ghost cells)
    cell_mesh : Tuple[np.ndarray, ...]
        the meshgrid created from the cell centers according to the dimension
    node_mesh : Tuple[np.ndarray, ...]
        the meshgrid created from the nodes according to the dimension

    """

    def __init__(
        self,
        nx: int,
        x_start: float,
        x_end: float,
        ngx: int = 2,
        ny: Optional[int] = None,
        y_start: Optional[float] = None,
        y_end: Optional[float] = None,
        ngy: Optional[int] = None,
        nz: Optional[int] = None,
        z_start: Optional[float] = None,
        z_end: Optional[float] = None,
        ngz: Optional[int] = None,
    ) -> None:
        """
        Initialize the Grid object for FVM simulations. Notice the number of cells are given as paramters.

        Parameters
        ----------
            nx : int
                Number of cells in the x-direction.
            x_start : float
                Starting coordinate in the x-direction.
            x_end : float
                Ending coordinate in the x-direction.
            ngx : int, optional
                Number of ghost cells in the x-direction. Default is 2.
            ny : int, optional
                Number of cells in the y-direction (for 2D/3D grids).
            y_start : float, optional
                Starting coordinate in the y-direction.
            y_end : float, optional
                Ending coordinate in the y-direction.
            ngy : int, optional
                Number of ghost cells in the y-direction.
            nz : int, optional
                Number of cells in the z-direction (for 3D grids).
            z_start : float, optional
                Starting coordinate in the z-direction.
            z_end : float, optional
                Ending coordinate in the z-direction.
            ngz : int, optional
                Number of ghost cells in the z-direction.
        """
        self.dimensions: int = 1  # Default to 1D

        # Grid parameters in x-direction
        self.nx: int = nx
        self.x_start: float = x_start
        self.x_end: float = x_end
        self.dx: float = (x_end - x_start) / nx
        self.ngx: int = ngx  # Number of ghost cells in x-direction

        # Total number of cells including ghost cells in x-direction
        self.nx_total: int = nx + 2 * ngx

        # Generate x-coordinate arrays
        self.x_cell_centers: np.ndarray = np.linspace(
            x_start - ngx * self.dx + 0.5 * self.dx,
            x_end + ngx * self.dx - 0.5 * self.dx,
            self.nx_total,
        )
        self.x_nodes: np.ndarray = np.linspace(
            x_start - ngx * self.dx,
            x_end + ngx * self.dx,
            self.nx_total + 1,
        )

        # Inner cell indices in x-direction
        self.inner_slice_x = slice(ngx, -ngx)

        # Check for 2D grid
        if ny is not None and y_start is not None and y_end is not None:
            self.dimensions = 2
            self.ny: int = ny
            self.y_start: float = y_start
            self.y_end: float = y_end
            self.dy: float = (y_end - y_start) / ny
            self.ngy: int = (
                ngy if ngy is not None else ngx
            )  # Use ngx if ngy not specified

            # Total number of cells including ghost cells in y-direction
            self.ny_total: int = ny + 2 * self.ngy

            # Generate y-coordinate arrays
            self.y_cell_centers: np.ndarray = np.linspace(
                y_start - self.ngy * self.dy + 0.5 * self.dy,
                y_end + self.ngy * self.dy - 0.5 * self.dy,
                self.ny_total,
            )
            self.y_nodes: np.ndarray = np.linspace(
                y_start - self.ngy * self.dy,
                y_end + self.ngy * self.dy,
                self.ny_total + 1,
            )

            # Inner cell indices in y-direction
            self.inner_slice_y = slice(self.ngy, -self.ngy)

        # Check for 3D grid
        if nz is not None and z_start is not None and z_end is not None:
            self.dimensions = 3
            self.nz: int = nz
            self.z_start: float = z_start
            self.z_end: float = z_end
            self.dz: float = (z_end - z_start) / nz
            self.ngz: int = (
                ngz if ngz is not None else ngx
            )  # Use ngx if ngz not specified

            # Total number of cells including ghost cells in z-direction
            self.nz_total: int = nz + 2 * self.ngz

            # Generate z-coordinate arrays
            self.z_cell_centers: np.ndarray = np.linspace(
                z_start - self.ngz * self.dz + 0.5 * self.dz,
                z_end + self.ngz * self.dz - 0.5 * self.dz,
                self.nz_total,
            )
            self.z_nodes: np.ndarray = np.linspace(
                z_start - self.ngz * self.dz,
                z_end + self.ngz * self.dz,
                self.nz_total + 1,
            )

            # Inner cell indices in z-direction
            self.inner_slice_z = slice(self.ngz, -self.ngz)

    @property
    def cell_mesh(self):
        """Create a mesh using the dimension and cell coordinates"""

        if self.dimensions == 1:
            return (self.x_cell_centers,)
        elif self.dimensions == 2:
            return np.meshgrid(self.x_cell_centers, self.y_cell_centers, indexing="ij")
        elif self.dimensions == 3:
            return np.meshgrid(
                self.x_cell_centers,
                self.y_cell_centers,
                self.z_cell_centers,
                indexing="ij",
            )

    @property
    def node_mesh(self):
        """Create a mesh using the dimension and node coordinates"""
        if self.dimensions == 1:
            return (self.x_nodes,)
        elif self.dimensions == 2:
            return np.meshgrid(self.x_nodes, self.y_nodes, indexing="ij")
        elif self.dimensions == 3:
            return np.meshgrid(self.x_nodes, self.y_nodes, self.z_nodes, indexing="ij")

    def get_inner_cells(self) -> Tuple[slice, ...]:
        """
        Get slices corresponding to the inner cells (excluding ghost cells).

        Returns:
            Tuple[slice, ...]: Slices for indexing inner cells.
        """
        if self.dimensions == 1:
            return (self.inner_slice_x,)
        elif self.dimensions == 2:
            return self.inner_slice_x, self.inner_slice_y
        elif self.dimensions == 3:
            return self.inner_slice_x, self.inner_slice_y, self.inner_slice_z
        else:
            raise ValueError("Invalid grid dimension")

    def get_boundary_cells(self) -> Tuple[slice, ...]:
        """
        Get slices corresponding to the boundary cells (ghost cells).

        Returns:
            Tuple[slice, ...]: Slices for indexing boundary cells.
        """
        if self.dimensions == 1:
            left = slice(0, self.ngx)
            right = slice(-self.ngx, None)
            return left, right
        elif self.dimensions == 2:
            left = slice(0, self.ngx)
            right = slice(-self.ngx, None)
            bottom = slice(0, self.ngy)
            top = slice(-self.ngy, None)
            return left, right, bottom, top
        elif self.dimensions == 3:
            left = slice(0, self.ngx)
            right = slice(-self.ngx, None)
            front = slice(0, self.ngy)
            back = slice(-self.ngy, None)
            bottom = slice(0, self.ngz)
            top = slice(-self.ngz, None)
            return left, right, front, back, bottom, top
        else:
            raise ValueError("Invalid grid dimension")

    def get_inner_nodes(self) -> Tuple[slice, ...]:
        """
        Get slices corresponding to the inner nodes (excluding ghost nodes).

        Returns:
            Tuple[slice, ...]: Slices for indexing inner nodes.
        """
        if self.dimensions == 1:
            return (self.inner_slice_x,)
        elif self.dimensions == 2:
            return self.inner_slice_x, self.inner_slice_y
        elif self.dimensions == 3:
            return self.inner_slice_x, self.inner_slice_y, self.inner_slice_z
        else:
            raise ValueError("Invalid grid dimension")

    def get_boundary_nodes(self) -> Tuple[slice, ...]:
        """
        Get slices corresponding to the boundary nodes (ghost nodes).

        Returns:
            Tuple[slice, ...]: Slices for indexing boundary nodes.
        """
        if self.dimensions == 1:
            left = slice(0, self.ngx)
            right = slice(-self.ngx + 1, None)
            return left, right
        elif self.dimensions == 2:
            left = slice(0, self.ngx)
            right = slice(-self.ngx + 1, None)
            bottom = slice(0, self.ngy)
            top = slice(-self.ngy + 1, None)
            return left, right, bottom, top
        elif self.dimensions == 3:
            left = slice(0, self.ngx)
            right = slice(-self.ngx + 1, None)
            front = slice(0, self.ngy)
            back = slice(-self.ngy + 1, None)
            bottom = slice(0, self.ngz)
            top = slice(-self.ngz + 1, None)
            return left, right, front, back, bottom, top
        else:
            raise ValueError("Invalid grid dimension")

    def evaluate_function_on_cells(self, func):
        """
        Evaluate a given function on the cell-centered coordinates of the grid.
        The function signature depends on the grid dimension:
        - 1D: func(x)
        - 2D: func(x, y)
        - 3D: func(x, y, z)

        Returns:
            np.ndarray: Array of function values at each cell center.
        """
        if self.dimensions == 1:
            return func(self.x_cell_centers)
        elif self.dimensions == 2:
            xc, yc = self.cell_mesh  # self.cell_mesh is a meshgrid (Xc, Yc)
            return func(xc, yc)
        elif self.dimensions == 3:
            xc, yc, zc = self.cell_mesh  # self.cell_mesh is a meshgrid (xc, yc, zc)
            return func(xc, yc, zc)
        else:
            raise ValueError("Invalid grid dimension.")

    def evaluate_function_on_nodes(self, func):
        """
        Evaluate a given function on the node-centered coordinates of the grid.
        The function signature depends on the grid dimension:
        - 1D: func(x)
        - 2D: func(x, y)
        - 3D: func(x, y, z)

        Returns:
            np.ndarray: Array of function values at each node.
        """
        if self.dimensions == 1:
            return func(self.x_nodes)
        elif self.dimensions == 2:
            Xn, Yn = self.node_mesh  # self.nodes is a meshgrid (Xn, Yn)
            return func(Xn, Yn)
        elif self.dimensions == 3:
            Xn, Yn, Zn = self.node_mesh  # self.nodes is a meshgrid (Xn, Yn, Zn)
            return func(Xn, Yn, Zn)
        else:
            raise ValueError("Invalid grid dimension.")

    @staticmethod
    @njit
    def cell_to_node_average_1d(var_cells: np.ndarray, ngx: int) -> np.ndarray:
        pass

    @staticmethod
    @njit
    def node_to_cell_average_1d(var_nodes: np.ndarray, ngx: int) -> np.ndarray:
        pass

    @staticmethod
    @njit
    def cell_to_node_average_2d(
        var_cells: np.ndarray, ngx: int, ngy: int
    ) -> np.ndarray:
        pass

    @staticmethod
    @njit
    def node_to_cell_average_2d(
        var_nodes: np.ndarray, ngx: int, ngy: int
    ) -> np.ndarray:
        pass

    @staticmethod
    @stencil
    def node_to_cell_2d_kernel(var_nodes: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @njit
    def cell_to_node_average_3d(
        var_cells: np.ndarray, ngx: int, ngy: int, ngz: int
    ) -> np.ndarray:
        pass

    @staticmethod
    @stencil
    def node_to_cell_3d_kernel(var_nodes: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def node_to_cell_average_3d(
        var_nodes: np.ndarray, ngx: int, ngy: int, ngz: int
    ) -> np.ndarray:
        return Grid.node_to_cell_3d_kernel(var_nodes)

    def cell_to_node_average(self, var_cells: np.ndarray) -> np.ndarray:
        if self.dimensions == 1:
            return self.cell_to_node_average_1d(var_cells, self.ngx)
        elif self.dimensions == 2:
            return self.cell_to_node_average_2d(var_cells, self.ngx, self.ngy)
        elif self.dimensions == 3:
            return self.cell_to_node_average_3d(var_cells, self.ngx, self.ngy, self.ngz)
        else:
            raise ValueError("Invalid grid dimension")

    def node_to_cell_average(self, var_nodes: np.ndarray) -> np.ndarray:
        if self.dimensions == 1:
            return self.node_to_cell_average_1d(var_nodes, self.ngx)
        elif self.dimensions == 2:
            return self.node_to_cell_average_2d(var_nodes, self.ngx, self.ngy)
        elif self.dimensions == 3:
            return self.node_to_cell_average_3d(var_nodes, self.ngx, self.ngy, self.ngz)
        else:
            raise ValueError("Invalid grid dimension")

    def apply_boundary_conditions_cells(self, var_cells: np.ndarray) -> None:
        """
        Apply boundary conditions to cell-centered variables.

        Parameters:
            var_cells (np.ndarray): Array of cell-centered variable values.
        """
        pass

    def apply_boundary_conditions_nodes(self, var_nodes: np.ndarray) -> None:
        """
        Apply boundary conditions to node-centered variables.

        Parameters:
            var_nodes (np.ndarray): Array of node-centered variable values.
        """
        pass


if __name__ == "__main__":

    from atmpy.grid.utility import *

    dimensions = [DimensionSpec(n=3, start=0, end=3, ng=1)]
    grid = create_grid(dimensions)
    print(grid.x_cell_centers)
    print(grid.x_nodes)
    print(grid.inner_slice_x)

    @stencil
    def kernel(variable):
        return (variable[0] + variable[1] + variable[-1]) / 3

    x = slice(1, -1)
    print(grid.x_cell_centers[(x,)])
