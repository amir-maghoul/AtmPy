import numpy as np
from typing import Callable, Union
from atmpy.grid.grid import Grid, NodeGrid, CellGrid


class Flux:
    """Create and manage the flux of the problem.

    Attributes
    ----------
    cflux : Callable
        The continuous flux function as a function of u.
    flux : np.ndarray
        The discretized flux.
    cell : :py:class:`~atmpy.grid.grid.CellGrid`
        The grid of cell-centered points.
    node : :py:class:`~atmpy.grid.grid.CellGrid`
        The grid of node-centered points.
    grid_type : str
        The type of grid on which the problem is based on. FVM or FDM are the choices
    iu : np.ndarray
        The solution on inner nodes.
    u : np.ndarray
        The discretized function on ghost nodes as well as inner nodes. The solution container.
    uint : np.ndarray
        The array of u values on interfaces (cell centers)
    u0 : Union[Callable, np.ndarray]
        The initial function.
    method : str
        The method to calculate the flux.
    boundary : str
        The boundary condition of the problem. Choices are:
        "zero" : zero padding of the ghost cells
        "periodic" : periodic boundary condition
        "linear" : linear extrapolation on the ghost cells
    """

    def __init__(
        self,
        cflux: Callable[[np.ndarray], np.ndarray],
        u0: Union[Callable[[np.ndarray], np.ndarray], np.ndarray],
        grid: Grid,
        discretization: str = "FDM",
        method: str = "upwind",
        boundary: str = "zero",
    ):
        """
        Initialize the flux object
        Parameters
        ----------
        cflux : Callable
            The continuous flux function as a function of u.
        u0 : Union[Callable, np.ndarray]
            The initial value of the flux.
        grid : :py:class:`~atmpy.grid.grid.Grid`
            The grid on which the problem is based
        discretization : str
            The method of discretization. Choices are "FVM" or "FDM".
        method : str, default = "upwind
            The method to calculate the flux.
        boundary : str
            The boundary condition of the problem. Choices are:
            "zero" : zero padding of the ghost cells
            "periodic" : periodic boundary condition
            "linear" : linear extrapolation on the ghost cells
        """
        self.cflux: Callable[[np.ndarray], np.ndarray] = cflux
        self.u0: Union[Callable[[np.ndarray], np.ndarray], np.ndarray] = u0
        self.grid: Grid = grid
        self.cell: CellGrid = grid.cell
        self.node: NodeGrid = grid.node
        self.discretization: str = discretization
        self.u: np.ndarray = np.zeros(self.node.oshape)
        self.cellu: np.ndarray = np.zeros(self.cell.oshape)
        self.uint: np.ndarray = np.zeros(self.cell.oshape)
        self.method: str = method
        self.boundary: str = boundary
        self.flux: np.ndarray = np.zeros(self.cell_grid.oshape)
        self.inner_u: np.ndarray = np.zeros(self.node.ishape)
        self._node_eval_u()
        self.update_solution(self.inner_u)
        self.update_flux()
