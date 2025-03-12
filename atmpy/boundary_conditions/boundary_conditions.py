"""Module for boundary conditions handling"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Callable, cast

from atmpy.grid.utility import DimensionSpec, create_grid
from atmpy.infrastructure.enums import VariableIndices as VI
from atmpy.infrastructure.enums import BoundarySide
from atmpy.flux.utility import direction_mapping
from atmpy.physics.thermodynamics import Thermodynamics


class BaseBoundaryCondition(ABC):
    """Abstract base class for all boundary conditions.

    Attributes
    ----------
    params : dict
        The dictionary of all parameters. It must at least contain the key "direction", "inner_slice", side and ng.
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

    def __init__(self, **kwargs: dict[str, Any]) -> None:

        self.params: dict = kwargs
        self.direction: int = direction_mapping(self.params["direction"])
        self.grid = self.params["grid"]
        self.ndim: int = self.grid.ndim
        self.ng: List[Tuple[int, int]] = self.grid.ng[self.direction]
        self.pad_width: List[Tuple[int, int]] = (
            self._create_pad_width_for_single_direction()
        )
        self.directional_inner_slice: Tuple[slice, ...] = (
            self._create_directional_inner_slice()
        )

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
        """Create the pad width for a single direction. The self.params["ng"] attribute contains a
        list of size 3 of the tuples containing the number of ghost cells in each side of each direction.
        Assuming we are working in 3D, self.params["ng"]= [(ngx, ngx), (ngy, ngy), (ngz, ngz)]. In order to
        avoid padding the boundary in all direction with the same number of ghost cells, we need to create a
        pad width tuple containing zero tuples in the undesired directions, i.e.  for x direction padding
        [(ngx, ngx), (0, 0), (0, 0)]."""

        pad_width = [(0, 0)] * self.ndim
        pad_width[self.direction] = tuple(self.ng)
        return pad_width


class PeriodicBoundary(BaseBoundaryCondition):
    def apply(self, cell_vars: np.ndarray, *args, **kwargs):
        """Apply periodic boundary condition to the cell_vars.

        Parameters
        ----------
        cell_vars : ndarray of shape (nx, [ny], [nz], num_vars)
            The array container for all the variables.
        """
        cell_vars[..., VI.RHO] = np.pad(
            cell_vars[..., VI.RHO][self.directional_inner_slice],
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
    side: str
        The signifier of which side we are working on. The valid values are "bottom" and "top".
        They don't mean anything in themselves. "bottom" is a signifier for the beginning of the array
        in the given axis. "top" is a signifier for the end of the array in the given axis.
    is_lamb: bool
        Determines if the lamb boundary condition is applied.
    is_compressible: bool
        Determines if we are in the compressible regime.
    """

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.th: Thermodynamics = kwargs["thermodynamics"]
        self.gravity: List[int, ...] = kwargs.pop("gravity")
        if len(np.nonzero(self.gravity)[0]) > 1:
            raise ValueError("Only one axis can have gravity strength.")
        if np.isclose(self.gravity[self.direction], 0):
            raise ValueError(
                "There is no gravity strength in the specified direction. Wrong boundary conditions."
            )
        self.gravity_axis: int = np.nonzero(self.gravity)[0][0]
        self.stratification: Callable[[Any], Any] = kwargs.pop("stratification")
        self.side: str = kwargs.pop("side")
        if self.side not in ["top", "bottom"]:
            raise ValueError("Side must be either 'top' or 'bottom'.")
        self.is_lamb: bool = kwargs.pop("is_lamb")
        self.is_compressible: bool = kwargs.pop("is_compressible")

    def apply(self, cell_vars: np.ndarray, *args, **kwargs):
        """Apply the reflective boundary condition for the given side of the gravity axis. If self.side is top, this means
        that the boundary condition for the top side of the vertical axis is the 'Lid' boundary. The sponge BC should be
        implemented separately in another class."""
        nsource, nlast, nimage = self._create_boundary_indices(side=self.side)

        # bring the variables in gravity axis to the first axis to avoid index confusion
        shift_vars = np.moveaxis(cell_vars, self.gravity_axis, 0)
        Y_last = cell_vars[..., VI.RHOY][nlast] / cell_vars[..., VI.RHO][nlast]
        axis_coordinate_cells = self.grid.get_coordinates(self.gravity_axis)
        strat = 1.0 / self.stratification(axis_coordinate_cells[nimage])
        sign = -1 if self.side == "top" else 1
        dr = self.grid.dxyz[
            self.gravity_axis
        ]  # discretization fineness in the gravity direction
        if not self.is_lamb:
            # The derivative of Pi (Exner pressure)
            dpi = sign * (self.th.Gamma) * 0.5 * dr * (1.0 / Y_last + strat)
        else:
            raise NotImplementedError(
                "The lamb boundary condition is not implemented yet."
            )
        if self.is_compressible:
            rhoY = (
                (cell_vars[..., VI.RHOY][nlast] ** self.th.gm1) + dpi
            ) ** self.th.gm1inv
        else:
            raise NotImplementedError(
                "The compressible boundary condition is not implemented yet."
            )

        momentum_index = self._get_gravity_momentum_index()
        # calculate the Pv for the ghost cells. v here is a placeholder for the velocity in the direction of gravity
        Pv = (
            -cell_vars[..., momentum_index][nsource]
            * cell_vars[..., VI.RHOY][nsource]
            / cell_vars[..., VI.RHO][nsource]
        )

        if not self.is_lamb:
            v = Pv / rhoY
            Th_slc = 1.0
        else:
            raise ValueError("The lamb boundary condition is not implemented yet.")

        u = cell_vars[..., VI.RHOU][nsource] / cell_vars[..., VI.RHO][nsource]

        # TODO: 1. this is completely wrong. The nsource and nlast and nimage indexing should be applied after moved axis
        #       2. Use inplace indexing instead of making copies
        return Y_last

    def _get_gravity_momentum_index(self):
        """Helper method to get the momentum variable index in the direction of gravity as one output and the index of
        the rest of the momentums as the second output."""
        velocity_indices = [VI.RHOU, VI.RHOV, VI.RHOW]
        return velocity_indices[self.gravity_axis]

    def _create_boundary_indices(self):
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
        ng_tuple: Tuple[int, int] = self.grid.ng[self.direction]
        N: int = self.grid.nc_total[self.direction]

        # Bring the BC axis to the front.
        # u_shift = np.moveaxis(u, axis, 0)
        # N = u_shift.shape[0]

        if self.side == "bottom":
            ng = ng_tuple[0]
            ng_arange = np.arange(ng)
            nimage = ng - 1 - ng_arange  # ghost cells indices (will be updated)
            nlast = ng - ng_arange  # adjacent inner/ghost cell index
            nsource = ng + ng_arange  # inner cells moving inward
        elif self.side == "top":
            ng = ng_tuple[1]
            ng_arange = np.arange(ng)
            nimage = N - ng + ng_arange  # ghost cell indices at upper boundary
            nlast = N - ng + ng_arange - 1  # cell adjacent to inner cell on upper side
            nsource = N - ng - ng_arange - 1  # inner cells near the upper boundary
        else:
            raise ValueError("side must be 'bottom' or 'top'.")

        # # Evaluate the operation first into temporary arrays to avoid dependency issues.
        # tmp_lower = f(u_shift[lower_nsource, ...], u_shift[lower_nlast, ...])
        # tmp_upper = f(u_shift[top_nsource, ...], u_shift[top_nlast, ...])
        #
        # # Assign the computed values to the ghost cells.
        # u_shift[lower_nimage, ...] = tmp_lower
        # u_shift[top_nimage, ...]   = tmp_upper

        # Return the array with the original axis ordering.
        # return np.moveaxis(u_shift, 0, axis)
        return nsource, nlast, nimage


class SlipWall(BaseBoundaryCondition):
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


if __name__ == "__main__":
    from atmpy.grid.utility import create_grid, DimensionSpec

    y = np.arange(7 * 8 * 5).reshape(7, 8, 5)
    dims = [DimensionSpec(3, 0, 1, 2), DimensionSpec(3, 0, 1, 2)]
    grid = create_grid(dims)
    gravity = np.array([0.0, 1.0, 0.0])
    th = Thermodynamics()
    params = {
        "direction": "y",
        "grid": grid,
        "gravity": gravity,
        "stratification": lambda x: x**2,
        "side": "bottom",
        "thermodynamics": th,
        "is_lamb": False,
        "is_compressible": True,
    }
    x = ReflectiveGravityBoundary(**params)
    idx = x.gravity_axis
    nsource, nlast, nimage = x._create_boundary_indices()
    print(nsource, nlast, nimage)
    print("--------------------------")
    print(y[..., 0])
    print(y[nimage, ..., 0])
    y_shift = np.moveaxis(y, 1, 0)
    print(y_shift[nimage, ..., 0])
    print("---------------------------")
    y_shift[nimage, ..., 0] = 0
    print(y_shift[nimage, ..., 0])
    print(y[..., 0])
