from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Tuple, Optional, List, Iterable


class AbstractGrid(ABC):
    """
    Abstract base class for discretization points (DP).

    Attributes
    ----------
    ncells : np.ndarray
        The number of cells in each coordinate axis.
    lims : np.ndarray
        The limits of the space in each coordinate direction.
    nghosts : np.ndarray
        The number of ghost cells in each direction.
    ishape : tuple
        The shape of the inner point grid
    oshape : tuple
        The shape of the outer points of the grid = number of points + number of ghosts
    coords : dict
        Dictionary of coordinates points. Keys are "x", "y" and "z" and values are coordinate point for the whole interval + ghosts
    inner_coords : dict
        Dictionary of the inner coordinate points. See description for self.coords.
    ghost_coords : dict
        Dictionary of the ghost coordinate points. See description for self.coords.
    inner_idx : dict
        Dictionary of the inner indices. See description for self.coords.
    inner_slice : dict
        Dictionary of the inner point slices
    gidx_f : dict
        "Ghost indices front". Slice of ghost cells for front of the array in each axis direction
    gidx_b : dict
        "Ghost indices back". Slice of ghost cells for back of the array in each axis direction
    dim : int
        The dimension of the space.
    L : np.ndarray
        Length of the spatial intervals in each direction.
    dX : np.ndarray
        Array of spatial step size.
    """

    def __init__(
        self,
        ncells: Union[list, np.ndarray],
        lims: Union[np.ndarray, list],
        nghosts: Union[list, np.ndarray],
    ):
        """
        Initializes the DP with a given number of cells, limits, and ghost cells.

        Args:
            ncells (list or np.ndarray): Number of cells in each coordinate axis.
            lims (list or np.ndarray): Limits of the space in each coordinate direction.
            nghosts (list or np.ndarray): Number of ghost cells in each direction.
        """
        lims = np.array(lims)
        if lims.ndim != 2 or lims.shape[1] != 2:
            raise ValueError("The shape of the limits array must be (n, 2)")
        if lims.shape[0] != len(ncells):
            raise ValueError("The spatial dimensions in the input do not match")
        if len(ncells) != len(nghosts):
            raise ValueError("Dimension of ncells and nghost should match")
        if 0 in ncells or 0 in nghosts:
            raise ValueError("Number of ghost cells and cells should be nonzero")

        self.dim: int = len(ncells)
        self.ncells: np.ndarray = np.array(ncells)
        self.nghosts: np.ndarray = np.array(nghosts)
        self.ishape: Optional[Tuple[int, ...]] = ()
        self.oshape: Optional[Tuple[int, ...]] = ()
        self.L: np.ndarray = lims[:, 1] - lims[:, 0]
        self.dX: np.ndarray = self.L / self.ncells

        self.coords: dict = {"x": None, "y": None, "z": None}
        self.inner_coords: dict = {"x": None, "y": None, "z": None}
        self.ghost_coords: dict = {"x": None, "y": None, "z": None}
        self.inner_idx: dict = {"x": None, "y": None, "z": None}
        self.inner_slice: List[slice] = [slice(None)] * self.dim
        self.gidx_b: dict = {"x": None, "y": None, "z": None}
        self.gidx_f: dict = {"x": None, "y": None, "z": None}

        self._initialize_limits(lims)

    def _initialize_limits(self, lims):
        """
        Initialize grid limits for the coordinate directions.

        Args:
            lims (np.ndarray): Limits of the space in each coordinate direction.
        """
        self.lims = {}
        for i, (min_val, max_val) in enumerate(lims):
            self.lims[f"{self.axis(i)}"] = (min_val, max_val)

    def axis(self, i):
        """Return the axis label (x, y, z) for a given index."""
        return ["x", "y", "z"][i]

    def _get_front_ghost_slices(self, axis: int, ngh: int) -> list:
        """Get the index slice for front ghost cells."""
        front_ghost_idx: list = [slice(None)] * self.dim
        if ngh != 0:
            front_ghost_idx[axis] = slice(None, ngh)
        return front_ghost_idx

    def _get_back_ghost_slices(self, axis_index: int, ngh: int) -> list:
        """Get the index slice for back ghost cells."""
        back_ghost_idx: list = [slice(None)] * self.dim
        if ngh != 0:
            back_ghost_idx[axis_index] = slice(-ngh, None)
        return back_ghost_idx

    def _get_inner_slice(self, ngh: int) -> slice:
        """Get the inner slice excluding ghost cells."""
        if ngh == 0:
            return slice(None)
        return slice(ngh, -ngh)

    def _create_indices(self) -> None:
        """Create the indices for the grid."""
        if self.oshape is None:
            raise ValueError(
                "The calculate method cannot be called for the Grid parent class"
            )
        for i, n in enumerate(self.oshape):
            axis: str = self.axis(i)
            ngh: int = self.nghosts[i]
            temp: np.ndarray = np.arange(n)

            self.inner_slice[i] = self._get_inner_slice(ngh)
            self.inner_idx[axis] = temp[self.inner_slice[i]]
            self.gidx_b[axis] = self._get_back_ghost_slices(i, ngh)
            self.gidx_f[axis] = self._get_front_ghost_slices(i, ngh)

    def calculate_limits(self, axis: str, dx: float, ng: int) -> tuple:
        """Calculate the min and max limits for the axis based on the dx, number of ghost cells ng."""
        minval, maxval = self.lims[axis]
        minval -= ng * dx
        maxval += ng * dx
        return minval, maxval

    @abstractmethod
    def _initialize_coords(self):
        """Abstract method to initialize the coordinates for the grid."""
        pass


class CellGrid(AbstractGrid):
    """
    Class for cell-centered grid.
    """

    def __init__(
        self,
        ncells: Union[list, np.ndarray],
        lims: Union[np.ndarray, list],
        nghosts: Union[list, np.ndarray],
    ):
        """
        Initializes the cell-centered grid with a given number of cells, limits, and ghost cells.

        Args:
            ncells (list or np.ndarray): Number of cells in each coordinate axis.
            lims (list or np.ndarray): Limits of the space in each coordinate direction.
            nghosts (list or np.ndarray): Number of ghost cells in each direction.
        """
        super().__init__(ncells, lims, nghosts)
        self.ishape: Tuple[int, ...] = tuple(ncells)
        self.oshape: Tuple[int, ...] = tuple(
            [ncells[i] + 2 * nghosts[i] for i in range(len(ncells))]
        )
        super()._create_indices()
        self._initialize_coords()

    @staticmethod
    def _calculate_cells(minval: float, maxval: float, n: int, dx: float) -> np.ndarray:
        """Calculate the cell-centered coordinates."""
        return np.linspace(minval, maxval, n, endpoint=False, dtype=np.float64) + dx / 2

    @staticmethod
    def _calculate_inner_cells(cells: np.ndarray, ng: int) -> np.ndarray:
        """Calculate the inner cell coordinates, excluding ghost cells."""
        return cells[ng:-ng] if ng != 0 else cells

    @staticmethod
    def _calculate_ghost_cells(cells: np.ndarray, ng: int) -> np.ndarray:
        """Calculate the ghost cell coordinates."""
        return np.hstack((cells[:ng], cells[-ng:])) if ng != 0 else []

    def _initialize_coords(self) -> None:
        """
        Initialize cell-centered coordinates for the grid.
        """

        for i in range(self.dim):
            axis: str = self.axis(i)
            dx: float = self.dX[i]
            ng: int = self.nghosts[i]
            minval, maxval = self.calculate_limits(axis, dx, ng)
            n: int = self.oshape[i]

            cells: np.ndarray = CellGrid._calculate_cells(minval, maxval, n, dx)
            inner_cells: np.ndarray = CellGrid._calculate_inner_cells(cells, ng)

            self.coords[axis] = cells
            self.inner_coords[axis] = inner_cells
            self.ghost_coords[axis] = CellGrid._calculate_ghost_cells(cells, ng)


class NodeGrid(AbstractGrid):
    """
    Class for node-centered grid.
    """

    def __init__(
        self,
        ncells: Union[list, np.ndarray],
        lims: Union[np.ndarray, list],
        nghosts: Union[list, np.ndarray],
    ):
        """
        Initializes the node-centered grid with a given number of cells, limits, and ghost cells.

        Args:
            ncells (list or np.ndarray): Number of cells in each coordinate axis.
            lims (list or np.ndarray): Limits of the space in each coordinate direction.
            nghosts (list or np.ndarray): Number of ghost cells in each direction.

            Note
            ----
            The given ncells must be the number of cells as given to the parent class not the number of nodes
        """
        nnodes = tuple(ncell + 1 for ncell in ncells)
        super().__init__(ncells, lims, nghosts)
        self.ishape: Tuple[int, ...] = tuple(
            [ncells[i] + 1 for i in range(len(ncells))]
        )
        self.oshape: Tuple[int, ...] = tuple(
            [ncells[i] + 2 * nghosts[i] + 1 for i in range(len(ncells))]
        )
        super()._create_indices()
        self._initialize_coords()

    @staticmethod
    def _calculate_nodes(minval: float, maxval: float, n: int, dx: float) -> np.ndarray:
        """Calculate the cell-centered coordinates."""
        return np.linspace(minval, maxval, n, endpoint=True, dtype=np.float64)

    @staticmethod
    def _calculate_inner_nodes(nodes: np.ndarray, ng: int) -> np.ndarray:
        """Calculate the inner cell coordinates, excluding ghost cells."""
        return nodes[ng:-ng] if ng != 0 else nodes

    @staticmethod
    def _calculate_ghost_nodes(nodes: np.ndarray, ng: int) -> np.ndarray:
        """Calculate the ghost cell coordinates."""
        return np.hstack((nodes[:ng], nodes[-ng:])) if ng != 0 else []

    def _initialize_coords(self) -> None:
        """
        Initialize node-centered coordinates for the grid.
        """
        for i in range(self.dim):
            axis: int = self.axis(i)
            dx: float = self.dX[i]
            ng: int = self.nghosts[i]
            minval, maxval = self.calculate_limits(axis, dx, ng)
            n = self.oshape[i]

            nodes: np.ndarray = NodeGrid._calculate_nodes(minval, maxval, n, dx)
            inner_nodes: np.ndarray = NodeGrid._calculate_inner_nodes(nodes, ng)

            self.coords[axis] = nodes
            self.inner_coords[axis] = inner_nodes
            self.ghost_coords[axis] = NodeGrid._calculate_ghost_nodes(nodes, ng)


class Grid:
    """Grid factory class. Creates both node grid and cell grid with a given cell size."""

    def __init__(
        self,
        ncells: Union[list, np.ndarray],
        lims: Union[np.ndarray, list],
        nghosts: Union[list, np.ndarray],
    ):
        self.cell_grid = CellGrid(ncells, lims, nghosts)
        self.node_grid = NodeGrid(ncells, lims, nghosts)


if __name__ == "__main__":
    grid = Grid([4], [[-1, 1]], [2])
    print(grid.cell_grid.oshape)
    print(grid.node_grid.oshape)
    # print(grid.inner_coords)
    # print(grid.ghost_coords)

    # cell = CellGrid([4], [[-1, 1]], [2])
    # print(grid.coords)
    # print(grid.ghost_coords)
    # print(cell.coords)
    # print(cell.ghost_coords)
    # print(cell.dX)
    #
    # print(grid.inner_slice)

    # x = np.random.rand(4, 5)
    # print(x)
    # print(x[grid.inner_slice.values()])
