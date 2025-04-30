import numpy as np
from typing import Optional, Tuple, List, Dict


class Grid:
    """
    Basic grid infrastructure structure for the FVM solver.

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
    ncx_total : int
        Total number of cells in the x-direction (inner cells + ghost cells)
    ncy_total : int
        Total number of cells in the y-direction (inner cells + ghost cells)
    ncz_total : int
        Total number of cells in the z-direction (inner cells + ghost cells)
    nnx_total : int
        Total number of nodes in the x-direction (inner nodes + ghost nodes)
    nny_total : int
        Total number of nodes in the y-direction (inner nodes + ghost nodes)
    nnz_total : int
        Total number of nodes in the z-direction (inner nodes + ghost nodes)
    cshape : Tuple[int, ...]
        The total shape of the cell grid
    nshape : Tuple[int, ...]
        The total shape of the node grid
    icshape : Tuple[int, ...]
        The shape of inner cells
    inshape : Tuple[int, ...]
        The shape of inner nodes
    x_cells : ndarray
        Coordinate array of cell centers in x-direction
    y_cells : ndarray, optional
        Coordinate array of cell centers in y-direction
    z_cells : ndarray, optional
        Coordinate array of cell centers in z-direction
    x_nodes : np.ndarray
        Coordinate array of nodes in x-direction
    y_nodes : np.ndarray, optional
        Coordinate array of nodes in y-direction
    z_nodes : np.ndarray, optional
        Coordinate array of nodes in z-direction
    grid_type : str, default="cartesian"
        the grid type
    ng : List[Tuple[int, int]]
        The list of the number of ghost cells in at each side of each direction as a tuple
    inner_slice : List[slice]
        The list of the index slices of inner cells in each direction
    dxyz : List[int, int, int]
        The discretization fineness of all directions as a list. If a direction does not exist, the value is None.
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
        grid_type: str = "cartesian",
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
            grid_type : str, default="cartesian"
                the grid type
        """
        self.ndim: int = 1  # Default to 1D
        self.grid_type: str = grid_type

        if nx is None or x_start is None or x_end is None or ngx is None:
            raise ValueError(
                "Cannot have None as given parameters for the first dimension."
            )
        if nx < 0:
            raise ValueError("Cannot have negative values for number of cells.")
        if ngx < 2:
            raise ValueError("Number of ghost cells should at least be 2.")

        # Unifying lists of important properties for all dimensions
        self.dxyz: List[float] = np.ones(3)  # Discretization fineness
        self.nc: List[int, ...] = [None] * 3  # Number of inner cells in all direction

        ################################
        # Grid parameters in x-direction
        ################################
        self.nx: int = nx
        self.x_start: float = x_start
        self.x_end: float = x_end
        self.dx: float = (x_end - x_start) / nx
        self.ngx: int = ngx  # Number of ghost cells in x-direction
        self.dxyz[0] = self.dx
        self.nc[0] = self.nx

        # Total number of cells and nodes including ghost cells in x-direction

        self.ncx_total: int = nx + 2 * ngx
        self.nnx_total: int = self.ncx_total + 1
        self.cshape: Tuple[int, ...] = (self.ncx_total,)
        self.nshape: Tuple[int, ...] = (self.nnx_total,)
        self.icshape: Tuple[int, ...] = (self.nx,)
        self.inshape: Tuple[int, ...] = (self.nx + 1,)
        self.ng: List[Tuple[int, int]] = [None]
        self.inner_slice = [None]

        # Generate x-coordinate arrays
        self.x_cells: np.ndarray = np.linspace(
            x_start - ngx * self.dx + 0.5 * self.dx,
            x_end + ngx * self.dx - 0.5 * self.dx,
            self.ncx_total,
        )

        self.x_nodes: np.ndarray = np.linspace(
            x_start - ngx * self.dx,
            x_end + ngx * self.dx,
            self.ncx_total + 1,
        )

        # Inner cell indices in x-direction
        self.inner_slice_x = slice(ngx, -ngx)
        self.ng[0] = (self.ngx, self.ngx)
        self.inner_slice[0] = self.inner_slice_x

        ################################
        # Grid parameters in y direction
        ################################
        if (
            ny is not None
            and y_start is not None
            and y_end is not None
            and ngy is not None
        ):
            if ny < 0:
                raise ValueError("Cannot have negative values for number of cells.")
            if ngy < 2:
                raise ValueError("Number of ghost cells should at least be 2.")
            self.ndim = 2
            self.ny: int = ny
            self.y_start: float = y_start
            self.y_end: float = y_end
            self.dy: float = (y_end - y_start) / ny
            self.nc[1] = self.ny
            self.dxyz[1] = self.dy
            self.ngy: int = (
                ngy if ngy is not None else ngx
            )  # Use ngx if ngy not specified

            # Total number of cells and nodes including ghost cells in y-direction
            self.ncy_total: int = ny + 2 * self.ngy
            self.nny_total: int = self.ncy_total + 1
            self.cshape = (self.ncx_total, self.ncy_total)
            self.nshape = (self.nnx_total, self.nny_total)
            self.icshape = (self.nx, self.ny)
            self.inshape = (self.nx + 1, self.ny + 1)

            # Generate y-coordinate arrays
            self.y_cells: np.ndarray = np.linspace(
                y_start - self.ngy * self.dy + 0.5 * self.dy,
                y_end + self.ngy * self.dy - 0.5 * self.dy,
                self.ncy_total,
            )
            self.y_nodes: np.ndarray = np.linspace(
                y_start - self.ngy * self.dy,
                y_end + self.ngy * self.dy,
                self.ncy_total + 1,
            )

            # Inner cell indices in y-direction
            self.inner_slice_y = slice(self.ngy, -self.ngy)
            self.ng.append((self.ngy, self.ngy))
            self.inner_slice.append(self.inner_slice_y)

        elif not (ny is None and y_start is None and y_end is None and ngy is None):
            raise ValueError(
                "Cannot have mixed of None and not None values for parameters of the ndim."
            )

        ################################
        # Grid parameters in z direction
        ################################
        if (
            nz is not None
            and z_start is not None
            and z_end is not None
            and ngz is not None
        ):
            if nz < 0:
                raise ValueError("Cannot have negative values for number of cells.")
            if ngz < 2:
                raise ValueError("Number of ghost cells should at least be 2.")
            self.ndim = 3
            self.nz: int = nz
            self.z_start: float = z_start
            self.z_end: float = z_end
            self.dz: float = (z_end - z_start) / nz
            self.dxyz[2] = self.dz
            self.nc[2] = self.nz
            self.ngz: int = (
                ngz if ngz is not None else ngx
            )  # Use ngx if ngz not specified

            # Total number of cells and nodes including ghost cells in z-direction
            self.ncz_total: int = nz + 2 * self.ngz
            self.nnz_total: int = self.ncz_total + 1

            self.cshape = (self.ncx_total, self.ncy_total, self.ncz_total)
            self.nshape = (self.nnx_total, self.nny_total, self.nnz_total)
            self.icshape = (self.nx, self.ny, self.nz)
            self.inshape = (self.nx + 1, self.ny + 1, self.nz + 1)

            # Generate z-coordinate arrays
            self.z_cells: np.ndarray = np.linspace(
                z_start - self.ngz * self.dz + 0.5 * self.dz,
                z_end + self.ngz * self.dz - 0.5 * self.dz,
                self.ncz_total,
            )
            self.z_nodes: np.ndarray = np.linspace(
                z_start - self.ngz * self.dz,
                z_end + self.ngz * self.dz,
                self.ncz_total + 1,
            )

            # Inner cell indices in z-direction
            self.inner_slice_z = slice(self.ngz, -self.ngz)
            self.ng.append((self.ngz, self.ngz))
            self.inner_slice.append(self.inner_slice_z)

        elif not (nz is None and z_start is None and z_end is None and ngz is None):
            raise ValueError(
                "Cannot have mixed of None and not None values for parameters of the ndim."
            )

    def get_cell_coordinates(self, axis):
        """Get method for cell coordinate values in each direction"""
        if axis == 0:
            return self.x_cells
        elif axis == 1:
            return self.y_cells
        elif axis == 2:
            return self.z_cells
        else:
            raise ValueError("Invalid value for 'axis'.")

    def get_node_coordinates(self, axis):
        """Get method for node coordinate values in each direction"""
        if axis == 0:
            return self.x_nodes
        elif axis == 1:
            return self.y_nodes
        elif axis == 2:
            return self.z_nodes
        else:
            raise ValueError("Invalid value for 'axis'.")

    def get_inner_slice(self) -> Tuple[slice, ...]:
        """
        Get slices corresponding to the inner cells/nodes.

        Returns:
            Tuple[slice, ...]: Slices for indexing inner cells.
        """
        if self.ndim == 1:
            return (self.inner_slice_x,)
        elif self.ndim == 2:
            return self.inner_slice_x, self.inner_slice_y
        elif self.ndim == 3:
            return self.inner_slice_x, self.inner_slice_y, self.inner_slice_z
        else:
            raise ValueError("Invalid grid dimension")

    def get_boundary_cells(self) -> Tuple[slice, ...]:
        """
        Get slices corresponding to the boundary cells (ghost cells).

        Returns:
            Tuple[slice, ...]: Slices for indexing boundary cells.
        """
        if self.ndim == 1:
            left = slice(0, self.ngx)
            right = slice(-self.ngx, None)
            return left, right
        elif self.ndim == 2:
            left = slice(0, self.ngx)
            right = slice(-self.ngx, None)
            bottom = slice(0, self.ngy)
            top = slice(-self.ngy, None)
            return left, right, bottom, top
        elif self.ndim == 3:
            left = slice(0, self.ngx)
            right = slice(-self.ngx, None)
            front = slice(0, self.ngy)
            back = slice(-self.ngy, None)
            bottom = slice(0, self.ngz)
            top = slice(-self.ngz, None)
            return left, right, front, back, bottom, top
        else:
            raise ValueError("Invalid grid dimension")

    def compute_normals(self):
        """Computes the normals based on the grid type"""
        if self.grid_type == "cartesian":
            self._cartesian_normals()
        else:
            raise NotImplementedError("Currently, only cartesian grids are supported.")

    def _cartesian_normals(self):
        """
        Set up normal vectors for a structured Cartesian grid.
        Since the grid is structured, all interfaces in a given direction have the same normal.
        """
        if self.ndim == 1:
            # Only one direction: x
            self.normal_x = np.array([1.0])
        elif self.ndim == 2:
            # Two directions: x and y
            self.normal_x = np.array([1.0, 0.0])
            self.normal_y = np.array([0.0, 1.0])
        elif self.ndim == 3:
            # Three directions: x, y, and z
            self.normal_x = np.array([1.0, 0.0, 0.0])
            self.normal_y = np.array([0.0, 1.0, 0.0])
            self.normal_z = np.array([0.0, 0.0, 1.0])
        else:
            raise ValueError("Invalid grid dimension. Must be 1, 2, or 3.")

    def get_inner_nodes(self) -> Tuple[slice, ...]:
        """
        Get slices corresponding to the inner nodes (excluding ghost nodes).

        Returns:
            Tuple[slice, ...]: Slices for indexing inner nodes.
        """
        if self.ndim == 1:
            return (self.inner_slice_x,)
        elif self.ndim == 2:
            return self.inner_slice_x, self.inner_slice_y
        elif self.ndim == 3:
            return self.inner_slice_x, self.inner_slice_y, self.inner_slice_z
        else:
            raise ValueError("Invalid grid dimension")

    def get_boundary_nodes(self) -> Tuple[slice, ...]:
        """
        Get slices corresponding to the boundary nodes (ghost nodes).

        Returns:
            Tuple[slice, ...]: Slices for indexing boundary nodes.
        """
        if self.ndim == 1:
            left = slice(0, self.ngx)
            right = slice(-self.ngx + 1, None)
            return left, right
        elif self.ndim == 2:
            left = slice(0, self.ngx)
            right = slice(-self.ngx + 1, None)
            bottom = slice(0, self.ngy)
            top = slice(-self.ngy + 1, None)
            return left, right, bottom, top
        elif self.ndim == 3:
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
        if self.ndim == 1:
            return func(self.x_cells)
        elif self.ndim == 2:
            xc, yc = self.cell_mesh  # self.cell_mesh is a meshgrid (Xc, Yc)
            return func(xc, yc)
        elif self.ndim == 3:
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
        if self.ndim == 1:
            return func(self.x_nodes)
        elif self.ndim == 2:
            Xn, Yn = self.node_mesh  # self.mesh is a meshgrid (Xn, Yn)
            return func(Xn, Yn)
        elif self.ndim == 3:
            Xn, Yn, Zn = self.node_mesh  # self.mesh is a meshgrid (Xn, Yn, Zn)
            return func(Xn, Yn, Zn)
        else:
            raise ValueError("Invalid grid dimension.")
