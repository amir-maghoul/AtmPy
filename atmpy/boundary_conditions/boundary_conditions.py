"""Module for boundary conditions handling"""

import numpy as np
from abc import ABC, abstractmethod
from typing import (
    List,
    Any,
    Tuple,
    Callable,
    TypeGuard,
    TypedDict,
    Unpack,
    cast,
    Union,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
    from atmpy.physics.thermodynamics import Thermodynamics

from atmpy.infrastructure.utility import direction_axis
from atmpy.infrastructure.enums import (
    BoundaryConditions as BdryType,
    BoundarySide,
    VariableIndices as VI,
)


class KwargsTypes(TypedDict):
    """Constructor class for dictionary typing of kwargs"""

    direction: str
    grid: "Grid"
    side: str


class BaseBoundaryCondition(ABC):
    """Abstract base class for all boundary conditions.

    Attributes
    ----------
    params : dict
        The dictionary of all parameters. It must at least contain the key "direction", "inner_slice", "side" and "ng".
    direction : int
        The direction of the boundary condition.
    side : BoundarySide
        The side in the direction for the boundary condition.
    ng : tuple
        The tuple of the number of ghost cells in each side of all directions.
    pad_width : List[Tuple[int, int]]
        list of tuples of the number of ghost cells in a single direction.
        Example: [(ngx, ngx), (0, 0), (0, 0)]. For more details see the docstring for
        py:meth:`atmpy.boundary_conditions.BoundaryCondition._create_pad_width_for_single_direction`.
    directional_inner_slice : Tuple[slice,...]
        Create the tuple of inner slices for a single direction.


    Notes
    -----
    the "inner_slice" key of the params is the tuple of all inner slices in all directions.
    """

    def __init__(self, **kwargs: Unpack[KwargsTypes]) -> None:

        self.direction: int = direction_axis(kwargs["direction"])
        self.grid: "Grid" = kwargs["grid"]
        self.ndim: int = self.grid.ndim
        self.ng: List[Tuple[Any, Any]] = self.grid.ng[self.direction]
        self.pad_width: List[Tuple[int, int]] = (
            self._create_pad_width_for_single_direction()
        )
        self.directional_inner_slice: Tuple[slice, ...] = (
            self._create_directional_inner_slice()
        )
        self.type: BdryType = BdryType.ABSTRACT
        self.side = kwargs["side"]

    @abstractmethod
    def apply(self, cell_vars, *args, **kwargs):
        pass

    def _create_directional_inner_slice(self) -> Tuple[slice, ...]:
        """Create the directional inner slice for a single direction.For more details see the docstring for
        py:meth:`atmpy.boundary_conditions.boundary_conditions.BaseBoundaryCondition._create_pad_width_for_single_direction`.
        """

        directional_inner_slice = [slice(None)] * self.ndim
        directional_inner_slice[self.direction] = self.grid.inner_slice[self.direction]
        return tuple(directional_inner_slice)

    def _create_pad_width_for_single_direction(self) -> List[Tuple[int, int]]:
        """Create the pad width for a single direction. The "ng" attribute contains a
        list of size 3 of the tuples containing the number of ghost cells in each side of each direction.
        Assuming we are working in 3D,"ng"= [(ngx, ngx), (ngy, ngy), (ngz, ngz)]. In order to
        avoid padding the boundary in all direction with the same number of ghost cells, we need to create a
        pad width tuple containing zero tuples in the undesired directions, i.e.  for x direction padding
        [(ngx, ngx), (0, 0), (0, 0)]."""

        pad_width = [(0, 0)] * self.ndim
        pad_width[self.direction] = tuple(self.ng)
        return pad_width


class PeriodicBoundary(BaseBoundaryCondition):
    def __init__(self, **kwargs: Unpack[KwargsTypes]):
        super().__init__(**kwargs)
        self.type = BdryType.PERIODIC

    def apply(self, cell_vars: np.ndarray, *args, **kwargs):
        """Apply periodic boundary condition to the cell_vars.

        Parameters
        ----------
        cell_vars : ndarray of shape (nx, [ny], [nz], num_vars)
            The array container for all the variables.
        """
        cell_vars[..., VI.RHO] = np.pad(
            cell_vars[self.directional_inner_slice + (VI.RHO,)],
            self.pad_width,
            mode="wrap",
        )


class ReflectiveGravityBoundary(BaseBoundaryCondition):
    """Compute the boundary condition for the side affected by the gravity. The gravity axis must be specified
    in the kwargs as an int.

    Attributes
    ----------
    gravity : List[float, float, float]
        The list of the gravity strength in each axis direction. For example gravity = [0.0, 0.0, 3.0] means the
        gravity exist in the third direction with the strength of 3.0
    gravity_axis: int
        The axis in which the gravity affects the boundary.
    stratification: Callable
        The given stratification function
    facet: str
        The signifier of which side of the gravity axis is considered. The valid values are "bottom" and "top".
        They don't mean anything in themselves. "bottom" is a signifier for the beginning of the array
        in the given axis. "top" is a signifier for the end of the array in the given axis.
    is_lamb: bool
        Determines if the lamb boundary condition is applied.
    is_compressible: bool
        Determines if we are in the compressible regime.
    """

    class KwargsType(TypedDict):
        """Constructor class for typing of kwargs dictionary"""

        thermodynamics: "Thermodynamics"
        gravity: Union[List[float], np.ndarray]
        stratification: Callable[[Any], Any]
        side: str
        is_lamb: bool
        is_compressible: bool

    def __init__(self, **kwargs: Unpack[KwargsType]) -> None:
        super().__init__(**kwargs)
        self.type = BdryType.REFLECTIVE_GRAVITY
        self.th: "Thermodynamics" = kwargs["thermodynamics"]
        self.gravity: Union[List[float], np.ndarray] = kwargs["gravity"]
        if len(np.nonzero(self.gravity)[0]) > 1:
            raise ValueError("Only one axis can have gravity strength.")
        if np.isclose(self.gravity[self.direction], 0):
            raise ValueError(
                "There is no gravity strength in the specified direction. Wrong boundary conditions."
            )
        self.gravity_axis: int = cast(int, np.nonzero(self.gravity)[0][0])
        if self.gravity_axis >= self.ndim:
            raise ValueError(f"An {self.ndim}-dimensional problem cannot have gravity on axis {self.gravity_axis}.")

        if self.gravity_axis == 0:
            raise ValueError(
                "In reflective gravity boundary condition, the first axis is reserved for horizontal velocity. It cannot have gravity."
            )
        self.stratification: Callable[[Any], Any] = kwargs["stratification"]
        self.facet: str = self._find_facet()
        if self.facet not in ["top", "bottom"]:
            raise ValueError("Facet must be either 'top' or 'bottom'.")
        self.is_lamb: bool = kwargs["is_lamb"]
        self.is_compressible: bool = kwargs["is_compressible"]

    def apply(self, cell_vars: np.ndarray, *args, **kwargs):
        """Apply the reflective boundary condition for the given side of the gravity axis. If self.side is top, this means
        that the boundary condition for the top side of the vertical axis is the 'Lid' boundary. The sponge BC should be
        implemented separately in another class."""

        # calculate the boundary indices
        nsource, nlast, nimage = self._create_boundary_indices()

        # evaluate the Theta on nlast indices
        Y_last = cell_vars[nlast + (VI.RHOY,)] / cell_vars[nlast + (VI.RHO,)]

        # get the coordinate values of the grid in the direction of gravity
        axis_coordinate_cells = self.grid.get_cell_coordinates(self.gravity_axis)

        # calculate the stratification function on the ghost cells
        strat = 1.0 / self.stratification(
            axis_coordinate_cells[nimage[self.gravity_axis]]
        )

        # sign to calculate the derivative of Pi. -1 if on the topside boundary and 1 if on the downside
        sign = -1 if self.facet == "top" else 1
        dr = self.grid.dxyz[
            self.gravity_axis
        ]  # discretization fineness in the gravity direction

        # Calculate the derivative of Pi (Exner pressure) in existence/nonexistence of lamb boundary
        if not self.is_lamb:
            dpi = sign * self.th.Gamma * 0.5 * dr * (1.0 / Y_last + strat)
        else:
            raise NotImplementedError(
                "The lamb boundary condition is not implemented yet."
            )

        # Calculate the P = rho Theta on the nlast indices in compressible or pseudo-incompressible regimes
        if self.is_compressible:
            rhoY = (
                (cell_vars[nlast + (VI.RHOY,)] ** self.th.gm1) + dpi
            ) ** self.th.gm1inv
        else:
            raise NotImplementedError(
                "The compressible boundary condition is not implemented yet."
            )

        # Get the index of the velocities in cell_vars for the gravity and nongravity directions
        momentum_index = self._get_gravity_momentum_index()
        # calculate the Pv for the ghost cells. "v" here is a placeholder for the velocity in the direction of gravity
        Pv = (
            -cell_vars[nsource + (momentum_index[0],)]
            * cell_vars[nsource + (VI.RHOY,)]
            / cell_vars[nsource + (VI.RHO,)]
        )

        if not self.is_lamb:
            v = Pv / rhoY
            Th_slc = 1.0
        else:
            raise ValueError("The lamb boundary condition is not implemented yet.")

        # This is the actual horizontal velocity, since the gravity axis can never be the first axis.
        u = cell_vars[nsource + (VI.RHOU,)] / cell_vars[nsource + (VI.RHO,)]

        # w is a placeholder for the velocity in the direction of non-gravity.
        w = cell_vars[nsource + (momentum_index[1],)] / cell_vars[nsource + (VI.RHO,)]

        # Actual X
        X = cell_vars[nsource + (VI.RHOX,)] / cell_vars[nsource + (VI.RHO,)]

        # Calculate intermediate rho
        rho = rhoY * strat

        # Evaluate variables (all) at the ghost cells
        cell_vars[nimage + (VI.RHO,)] = rho
        cell_vars[nimage + (VI.RHOU,)] = rho * u * Th_slc
        if not self.is_lamb:
            cell_vars[nimage + (momentum_index[0],)] = rho * v
        else:
            raise NotImplementedError(
                "The lamb boundary condition is not implemented yet."
            )
        cell_vars[nimage + (momentum_index[1],)] = rho * w * Th_slc
        cell_vars[nimage + (VI.RHOY,)] = rhoY
        cell_vars[nimage + (VI.RHOX,)] = rho * X

    def _get_gravity_momentum_index(self) -> Tuple[int, int]:
        """Helper method to get the momentum variable index in the direction of gravity as the first output and the
        momentum in the nongravity direction as the second output. Notice since RHOU can never be the momentum in the
        gravity axis (or more clearly, first axis can never be the gravity axis), it is not included in the array
        """

        if self.gravity_axis == 1:
            gravity_index = VI.RHOV
            non_gravity_index = VI.RHOW
        elif self.gravity_axis == 2:
            gravity_index = VI.RHOW
            non_gravity_index = VI.RHOV
        return gravity_index, non_gravity_index

    def _create_boundary_indices(
        self,
    ) -> tuple[tuple[slice, ...], tuple[slice, ...], tuple[slice, ...]]:
        """Create three type of indices:
        nsource: the combination of first two inner cells (for bottom side) and last two inner cells (for top side)
        nlast: the combination of the two set of neighboring inner cell and ghost cells, i.e. for the bottom side,
                it is the last ghost cell of the side and first inner cells.
        nimage: the combination of the set of two ghost cells at both sides

        For the lower boundary (ghost cells at beginning along the axis):
          For i=0,...,ghost-1:
             nimage = ghost - 1 - i
             nlast  = ghost - i
             nsource = ghost + i

        For the upper boundary (ghost cells at end along the axis; let N be size along axis):
          For i=0,...,ghost-1:
             nimage = N - ghost + i
             nsource = N - ghost - 1 - i
             nlast  = N - ghost - 1 + i

        Returns
        -------
        Tuple
            Return the three set of indices.

        """
        ng_tuple: Tuple[int, ...] = self.grid.ng[self.direction]
        N: int = self.grid.cshape[self.direction]

        if self.facet == "bottom":
            ng = ng_tuple[0]
            ng_arange = np.arange(ng)
            image = ng - 1 - ng_arange  # ghost cells indices (will be updated)
            last = ng - ng_arange  # adjacent inner/ghost cell index
            source = ng + ng_arange  # inner cells moving inward
        elif self.facet == "top":
            ng = ng_tuple[1]
            ng_arange = np.arange(ng)
            image = N - ng + ng_arange  # ghost cell indices at upper boundary
            last = N - ng + ng_arange - 1  # cell adjacent to inner cell on upper side
            source = N - ng - ng_arange - 1  # inner cells near the upper boundary
        else:
            raise ValueError("side must be 'bottom' or 'top'.")

        # Make the indices valid for multi-dimensional array
        nimage: list = [slice(None)] * self.ndim
        nlast = [slice(None)] * self.ndim
        nsource = [slice(None)] * self.ndim

        nimage[self.gravity_axis] = image
        nlast[self.gravity_axis] = last
        nsource[self.gravity_axis] = source

        return tuple(nsource), tuple(nlast), tuple(nimage)

    def _find_facet(self):
        """Find which end of the array is the gravity being applied on"""
        if self.side == BoundarySide.TOP or self.side == BoundarySide.BACK:
            return "top"
        elif self.side == BoundarySide.BOTTOM or self.side == BoundarySide.FRONT:
            return "bottom"
        else:
            raise ValueError("'Left' and 'Right' are not valid side for gravity axis.")


class Wall(BaseBoundaryCondition):
    def apply(self, cell_vars, *args, **kwargs):
        # Zero normal velocity, reflect tangential components
        pass


class NonReflectiveOutlet(BaseBoundaryCondition):
    def apply(self, cell_vars, *args, **kwargs):
        # Characteristic-based extrapolation
        pass


class WallBoundary(BaseBoundaryCondition):
    def apply(self, cell_vars, *args, **kwargs):
        # Implement wall BC logic (e.g., reflect normal velocity)
        pass


class InflowBoundary(BaseBoundaryCondition):
    def apply(self, cell_vars, *args, **kwargs):
        # Implement inflow BC logic (e.g., fixed pressure, temperature, velocity)
        pass


class OutflowBoundary(BaseBoundaryCondition):
    def apply(self, cell_vars, *args, **kwargs):
        # Implement outflow BC logic (e.g., zero gradient)
        pass


def example_usage():
    from atmpy.grid.utility import create_grid, DimensionSpec
    from atmpy.grid.utility import DimensionSpec, create_grid
    from atmpy.variables.variables import Variables
    from atmpy.physics.thermodynamics import Thermodynamics

    np.set_printoptions(linewidth=100)

    dt = 0.1

    nx = 1
    ngx = 2
    nnx = nx + 2 * ngx
    ny = 2
    ngy = 2
    nny = ny + 2 * ngy

    dim = [DimensionSpec(nx, 0, 2, ngx), DimensionSpec(ny, 0, 2, ngy)]
    grid = create_grid(dim)
    rng = np.random.default_rng()
    arr = np.arange(nnx * nny)
    rng.shuffle(arr)
    array = arr.reshape(nnx, nny)

    variables = Variables(grid, 6, 1)
    variables.cell_vars[..., VI.RHO] = 1
    variables.cell_vars[..., VI.RHOU] = array
    variables.cell_vars[..., VI.RHOY] = 2
    variables.cell_vars[..., VI.RHOW] = 3

    rng.shuffle(arr)
    array = arr.reshape(nnx, nny)
    variables.cell_vars[..., VI.RHOV] = array
    gravity = np.array([0.0, 1.0, 0.0])
    th = Thermodynamics()
    params = {
        "direction": "y",
        "grid": grid,
        "side": BoundarySide.TOP,
        "gravity": gravity,
        "stratification": lambda x: x**2,
        "thermodynamics": th,
        "is_lamb": False,
        "is_compressible": True,
    }
    x = ReflectiveGravityBoundary(**params)
    print(variables.cell_vars[..., VI.RHO])
    x.apply(variables.cell_vars)
    print(variables.cell_vars[..., VI.RHO])


if __name__ == "__main__":
    example_usage()
