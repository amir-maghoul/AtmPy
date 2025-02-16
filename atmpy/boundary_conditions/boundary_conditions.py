""" Module for boundary conditions handling"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Any, Tuple

from atmpy.grid.utility import DimensionSpec, create_grid
from atmpy.infrastructure.enums import VariableIndices as VI
from atmpy.infrastructure.enums import BoundarySide
from atmpy.flux.utility import direction_mapping


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

    Notes
    -----
    the "inner_slice" key of the params is the tuple of all inner slices in all directions.
    """

    def __init__(self, **kwargs: dict[str, Any]) -> None:

        self.params: dict = kwargs
        self.direction: int = direction_mapping(self.params["direction"])
        self.grid = self.params["grid"]
        self.ndim: int = self.grid.ndim
        self.ng: tuple = self.grid.ng[self.direction]
        self.side: BoundarySide = self.params["side"]
        self.pad_width: List[Tuple[int, int]] = self._create_pad_width_for_single_direction()
        self.directional_inner_slice = self._create_directional_inner_slice()


    @abstractmethod
    def apply(self, cell_vars, *args, **kwargs):
        pass

    def _create_directional_inner_slice(self) -> Tuple[slice, ...]:
        """Create the directional inner slice for a single direction.For more details see the docstring for
        py:meth:`atmpy.boundary_conditions.boundary_conditions.BaseBoundaryCondition._create_pad_width_for_single_direction`."""

        directional_inner_slice = [slice(None)] * self.ndim
        directional_inner_slice[self.direction] = self.grid.inner_slice[self.direction]
        return tuple(directional_inner_slice)

    def _create_pad_width_for_single_direction(self) -> List[Tuple[int, int]]:
        """ Create the pad width for a single direction. The self.params["ng"] attribute contains a
        tuple of size 3 of the tuples containing the number of ghost cells in each side of each direction.
        Assuming we are working in 3D, self.params["ng"]= ((ngx, ngx), (ngy, ngy), (ngz, ngz)). In order to
         avoid padding the boundary in all direction with the same number of ghost cells, we need to create a
         pad width tuple containing zero tuples in the undesired directions, i.e.  for x direction padding
         ((ngx, ngx), (0, 0), (0, 0))."""

        pad_width = [(0, 0)]*self.ndim
        pad_width[self.direction] = tuple(self.ng)
        return pad_width


class PeriodicBoundary(BaseBoundaryCondition):
    def apply(self, cell_vars, *args, **kwargs):
        cell_vars[..., VI.RHO] = np.pad(cell_vars[..., VI.RHO][self.directional_inner_slice], self.pad_width, mode="wrap")

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
    y = np.arange(7*8*2).reshape(7, 8, 2)
    dims = [DimensionSpec(3, 0, 1, 2), DimensionSpec(4, 0, 1, 2)]
    grid = create_grid(dims)
    params = {"direction":"x", "grid":grid, "side":BoundarySide.LEFT}
    x = PeriodicBoundary(**params)
    print(y[..., 0])
    x.apply(y)
    print(y[..., 0])