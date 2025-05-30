"""Module for boundary conditions handling"""

import numpy as np
import copy
from abc import ABC, abstractmethod
from typing import (
    List,
    Any,
    Tuple,
    Callable,
    TypedDict,
    Unpack,
    cast,
    Union,
    Literal,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
    from atmpy.physics.thermodynamics import Thermodynamics
    from atmpy.boundary_conditions.contexts import (
        BCApplicationContext,
        BCInstantiationOptions,
        ReflectiveGravityBCInstantiationOptions as RFBCInstantiationOptions,
    )
    from atmpy.boundary_conditions.bc_extra_operations import ExtraBCOperation
    from atmpy.variables.multiple_pressure_variables import MPV

from atmpy.boundary_conditions.bc_extra_operations import WallAdjustment, WallFluxCorrection
from atmpy.infrastructure.utility import (
    direction_axis,
    momentum_index,
    side_direction_mapping,
)
from atmpy.infrastructure.enums import (
    BoundaryConditions as BdryType,
    BoundarySide,
    VariableIndices as VI,
    HydrostateIndices as HI,
)
from atmpy.physics.gravity import Gravity


class BaseBoundaryCondition(ABC):
    """Abstract base class for all boundary conditions.

    Attributes
    ----------
    direction : int
        The direction of the boundary condition.
    side : BoundarySide
        The side in the direction for the boundary condition.
    direction_str: str
        The indicator that which axis is the boundary condition being applied on.
    direction: int
        The direction axis in integer
    full_inner_slice: List[Tuple[Any, Any]]
        The slice of full inner domain.
    ng : tuple
        The tuple of the number of ghost cells in each side of all directions.


    Notes
    -----
    the "inner_slice" key of the params is the tuple of all inner slices in all directions.
    """

    def __init__(self, inst_opts: "BCInstantiationOptions") -> None:
        """Constructor for boundary conditions.

        Parameters:
        -----------
        inst_opts : BCInstantiationOptions
            The context object that contains information about the instantiation of the boundary conditions.
        """

        self.direction_str = inst_opts.direction
        self.direction: int = direction_axis(self.direction_str)
        self.grid: "Grid" = inst_opts.grid
        self.ndim: int = self.grid.ndim
        self.ng: Tuple[int, int] = self.grid.ng[self.direction]
        self.full_inner_slice: List[Tuple[int, int]] = self.grid.inner_slice
        self.type: BdryType = inst_opts.type
        self.side = inst_opts.side
        self.side_axis: int = self._find_side_axis()

    @abstractmethod
    def apply(self, cell_vars):
        """Apply the boundary condition on the cell variables"""
        pass

    @abstractmethod
    def apply_single_variable(
        self, variable: np.ndarray, context: "BCApplicationContext"
    ):
        """Apply the boundary condition on a variable. Original motivation is to apply the correction on the
        right-hand side of the pressure equation.
        """
        pass

    @abstractmethod
    def apply_extra(self, variable: np.ndarray, operation: "ExtraBCOperation") -> None:
        """Optional extra update on variable, like scaling wall nodes. This function should be considered as a
        refinement of the boundary condition. It should be called in the euler steps (explicit and implicit) to
        refine the already applied boundary conditions."""
        pass

    @property
    def pad_width(self):
        """Create the pad width for a single direction. The "ng" attribute contains a
        list of size 3 of the tuples containing the number of ghost cells in each side of each direction.
        Assuming we are working in 3D,"ng"= [(ngx, ngx), (ngy, ngy), (ngz, ngz)]. In order to
        avoid padding the boundary in all direction with the same number of ghost cells, we need to create a
        pad width tuple containing zero tuples in the undesired directions, i.e.  for x direction padding
        [(ngx, ngx), (0, 0), (0, 0)]. For wall boundary, this is reduced to one-sided padding: for example for x direction
        and the left side: [(ngx, 0), (0, 0), (0, 0)]"""

        side: int = self.side_axis
        # create (ng, 0) or (0, ng)
        ng: List[int] = [0 if i != side else self.ng[side] for i in range(2)]
        # create the pad width
        pad_width = [(0, 0)] * (self.ndim + 1)
        pad_width[self.direction] = ng
        return pad_width

    def _find_side_axis(self):
        """Find the integer axis of the side on which the BC is applied in the given direction."""
        sides: Tuple[BoundarySide, BoundarySide] = side_direction_mapping(
            self.direction_str
        )
        return sides.index(
            self.side
        )  # Index of the side in the tuple. Used to index the sided number of ghost cells


class PeriodicBoundary(BaseBoundaryCondition):
    """Implements the periodic boundary condition. See the parent class for the documentation of the shared attributes.

    Attributes
    ----------
    pad_width : List[Tuple[int, int]]
        list of tuples of the number of ghost cells in a single direction.
        Example: [(ngx, ngx), (0, 0), (0, 0)]. For more details see the docstring for
        py:meth:`atmpy.boundary_conditions.BoundaryCondition._create_pad_width_for_single_direction`.
    directional_inner_slice : Tuple[slice,...]
        Create the tuple of inner slices for a single direction."""

    def __init__(self, inst_opts: "BCInstantiationOptions") -> None:
        super().__init__(inst_opts)

    def apply(self, cell_vars: np.ndarray):
        """Apply periodic boundary condition to the cell_vars.

        Parameters
        ----------
        cell_vars : ndarray of shape (nx, [ny], [nz], num_vars)
            The array container for all the variables.
        context: BCApplicationContext
            The context object that contains information about the boundary conditions apply method.
        """
        inner_padded_slice = self.inner_and_padded_slices()
        # Apply np.pad with mode "wrap" on ALL variables for periodic BC.
        cell_vars[inner_padded_slice] = np.pad(
            cell_vars[self.inner_slice],
            self.pad_width,
            mode="wrap",
        )

    def apply_single_variable(
        self, variable: np.ndarray, context: "BCApplicationContext"
    ):
        """Apply the periodic boundary condition on the pressure variable."""
        inner_padded_slice = self.inner_and_padded_slices()

        variable[inner_padded_slice[:-1]] = np.pad(
            variable[self.inner_slice[:-1]],
            self.pad_width[:-1],
            mode="wrap",
        )

    def apply_extra(self, variable: np.ndarray, operation: "ExtraBCOperation") -> None:
        pass

    @property
    def inner_slice(self) -> Tuple[slice, ...]:
        """Create the directional inner slice for a single direction."""
        directional_inner_slice = [slice(None)] * (self.ndim + 1)
        directional_inner_slice[self.direction] = self.grid.inner_slice[self.direction]
        return tuple(directional_inner_slice)

    def inner_and_padded_slices(self) -> Tuple[slice, ...]:
        """Returns the slice of the padded part."""
        side = self._find_side_axis()
        slc = slice(0, -self.ng[side]) if side == 0 else slice(self.ng[side], None)
        pad_slice = [slice(None)] * (self.ndim + 1)
        pad_slice[self.direction] = slc
        return tuple(pad_slice)


class ReflectiveGravityBoundary(BaseBoundaryCondition):
    """Compute the boundary condition for the side affected by the gravity. The gravity axis must be specified
    in the kwargs as an int. For shared attributes, see the documentation of the parent class.

    Attributes
    ----------
    gravity : List[float, float, float]
        The list of the gravity strength in each axis direction. For example gravity = [0.0, 0.0, 3.0] means the
        gravity exist in the third direction with the strength of 3.0
    stratification: Callable
        The given stratification function
    facet: str
        The signifier of which side of the gravity axis is considered. The valid values are "bottom" and "top".
        "begin" is a signifier for the beginning of the array in the given axis. "end" is a signifier for the end
        of the array in the given axis.
        "end" = BdrySide.TOP or BdrySide.BACK
        "begin" = BdrySide.BOTTOM or BdrySide.FRONT
    is_compressible: bool
        Determines if we are in the compressible regime.
    """

    def __init__(self, inst_opts: "RFBCInstantiationOptions") -> None:
        super().__init__(inst_opts)
        if self.ndim == 1:
            raise ValueError(
                "Reflective gravity is not implemented for 1 dimensional problem."
            )
        self.th: "Thermodynamics" = inst_opts.thermodynamics
        self.mpv: "MPV" = inst_opts.mpv
        self.gravity: Gravity = Gravity(inst_opts.gravity, self.ndim)
        if self.gravity.strength == 0.0:
            raise ValueError(
                "The gravity strength cannot be zero for ReflectiveGravityBoundary."
            )
        if np.isclose(self.gravity.vector[self.direction], 0):
            raise ValueError(
                "There is no gravity strength in the specified direction. Wrong boundary conditions."
            )
        self.stratification: Callable[[Any], Any] = inst_opts.stratification
        self.facet: str = self._find_facet()
        if self.facet not in ["begin", "end"]:
            raise ValueError("Facet must be either 'begin' or 'end'.")
        self.is_compressible: bool = inst_opts.is_compressible

    def apply(self, cell_vars: np.ndarray):
        """Apply the reflective boundary condition for the given side of the gravity axis. If self.side is top, this means
        that the boundary condition for the top side of the vertical axis is the 'Lid' boundary.
        """

        # calculate the boundary indices
        ng = self._get_ng_for_side()

        # Layer by layer filling the boundary:
        for i in range(ng):
            nimage, nlast, nsource = self._create_boundary_slice(i)

            # evaluate the Theta on nlast indices
            Y_last = cell_vars[nlast + (VI.RHOY,)] / cell_vars[nlast + (VI.RHO,)]

            # get the coordinate values of the grid in the direction of gravity
            axis_coordinate_cells = self.gravity.get_coordinate_cells(self.grid)

            # calculate the stratification function on the ghost cells
            strat = 1.0 / self.stratification(
                axis_coordinate_cells[nimage[self.gravity.axis]]
            )

            # sign to calculate the derivative of Pi. -1 if on the topside boundary and 1 if on the downside
            sign = -1 if self.facet == "end" else 1
            dr = self.grid.dxyz[
                self.gravity.axis
            ]  # discretization fineness in the gravity direction

            # Calculate the derivative of Pi (Exner pressure) in existence/nonexistence of lamb boundary
            dpi = (
                sign
                * self.th.Gamma
                * 0.5
                * self.gravity.strength
                * dr
                * (1.0 / Y_last + strat)
            )

            # Calculate the P = rho Theta on the nlast indices in compressible or pseudo-incompressible regimes
            if self.is_compressible:
                rhoY = (
                    (cell_vars[nlast + (VI.RHOY,)] ** self.th.gm1) + dpi
                ) ** self.th.gm1inv
            else:
                rhoY = self.mpv.hydrostate.cell_vars[
                    nimage[self.gravity.axis], HI.RHOY0
                ]

            # Get the index of the velocities in cell_vars for the gravity and nongravity directions
            gravity_momentum_index = self.gravity.vertical_momentum_index  # VI.RHOV
            # calculate the Pv for the ghost cells. "v" here is the velocity in the direction of gravity
            Pv = (
                -cell_vars[nsource + (gravity_momentum_index,)]
                * cell_vars[nsource + (VI.RHOY,)]
                / cell_vars[nsource + (VI.RHO,)]
            )

            # Calculate intermediate rho and evaluate the corresponding ghost cell.
            rho = rhoY * strat
            cell_vars[nimage + (VI.RHO,)] = rho

            # Find velocity in the direction of gravity and update ghost cells.
            v = Pv / rhoY
            Th_slc = 1.0
            cell_vars[nimage + (gravity_momentum_index,)] = rho * v

            # This is the actual horizontal velocity, since the gravity axis can never be the first axis.
            u = cell_vars[nsource + (VI.RHOU,)] / cell_vars[nsource + (VI.RHO,)]
            cell_vars[nimage + (VI.RHOU,)] = rho * u * Th_slc

            # w is a placeholder for the velocity in the direction of non-gravity. First, check whether the variable container
            # can be indexed that far.
            perpendicular_momentum_index = self.gravity.perpendicular_momentum_index
            if perpendicular_momentum_index < cell_vars.shape[-1]:
                w = (
                    cell_vars[nsource + (perpendicular_momentum_index,)]
                    / cell_vars[nsource + (VI.RHO,)]
                )
                cell_vars[nimage + (perpendicular_momentum_index,)] = rho * w * Th_slc

            # Compute the actual X and evaluate the ghost cells.
            X = cell_vars[nsource + (VI.RHOX,)] / cell_vars[nsource + (VI.RHO,)]
            cell_vars[nimage + (VI.RHOY,)] = rhoY
            cell_vars[nimage + (VI.RHOX,)] = rho * X

    def apply_single_variable(
        self, variable: np.ndarray, context: "BCApplicationContext"
    ):
        pass

    def apply_extra(
        self, variable: np.ndarray, context: "BCApplicationContext"
    ) -> None:
        pass

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

        if self.facet == "begin":
            ng = ng_tuple[0]
            ng_arange = np.arange(ng)
            image = ng - 1 - ng_arange  # ghost cells indices (will be updated)
            last = ng - ng_arange  # adjacent inner/ghost cell index
            source = ng + ng_arange  # inner cells moving inward
        elif self.facet == "end":
            ng = ng_tuple[1]
            ng_arange = np.arange(ng)
            image = N - ng + ng_arange  # ghost cell indices at upper boundary
            last = N - ng + ng_arange - 1  # cell adjacent to inner cell on upper side
            source = N - ng - ng_arange - 1  # inner cells near the upper boundary
        else:
            raise ValueError("Facet must be 'bottom' or 'top'.")

        return image, last, source

    def _get_ng_for_side(self):
        """Get the number of ghost cells for the side"""
        ng_tuple: Tuple[int, ...] = self.grid.ng[self.direction]
        if self.facet == "begin":
            ng = ng_tuple[0]
        elif self.facet == "end":
            ng = ng_tuple[1]
        else:
            raise ValueError("Facet must be 'bottom' or 'top'.")
        return ng

    def _create_boundary_slice(self, layer: int):
        """Choose which one of the boundary should be worked with.
        Source, last and image are all lists of size equal to the number of ghost cells. In order to fill the
        boundary layer-by-layer, we choose the corresponding values of source, last and image one by one

        Parameters
        ----------
        layer : int
            The layer to choose from the source, image and last indices
        """
        image, last, source = self._create_boundary_indices()
        # Make the indices valid for multi-dimensional array
        nimage: list = [slice(None)] * self.ndim
        nlast: list = [slice(None)] * self.ndim
        nsource: list = [slice(None)] * self.ndim

        nimage[self.gravity.axis] = image[layer]
        nlast[self.gravity.axis] = last[layer]
        nsource[self.gravity.axis] = source[layer]

        return tuple(nimage), tuple(nlast), tuple(nsource)

    def _find_facet(self):
        """Find which end of the array is the gravity being applied on"""
        if self.side == BoundarySide.TOP or self.side == BoundarySide.BACK:
            return "end"
        elif self.side == BoundarySide.BOTTOM or self.side == BoundarySide.FRONT:
            return "begin"
        else:
            raise ValueError(f"{self.side} is not valid side for gravity axis.")


class Wall(BaseBoundaryCondition):
    def __init__(self, inst_opts: "BCInstantiationOptions"):
        super().__init__(inst_opts)

    def apply(self, cell_vars: np.ndarray):
        """Apply the wall boundary condition. All the variables get reflected by the boundary, the normal velocity gets
        reflected and negated.

        Notes
        -----
        in np.pad, mode="symmetric" is used instead of mode="reflect". See the documentation for np.pad.
        """

        normal_momentum_index = momentum_index(self.direction)
        pad_slice = self.padded_slices()

        # Apply np.pad with mode "symmetric" on ALL variables for wall BC.
        mode: Literal["symmetric", "edge"] = "symmetric"
        if self.grid.nc[self.direction] == 1:
            # Single cell case
            # Apply np.pad with mode "edge" to copy the single cell to ghost cells
            mode = "edge"

        cell_vars[...] = np.pad(
            cell_vars[self.inner_slice],
            self.pad_width,
            mode=mode,
        )

        cell_vars[pad_slice + (normal_momentum_index,)] *= -1

    def apply_single_variable(
        self, variable: np.ndarray, context: "BCApplicationContext"
    ):
        """Apply the wall boundary condition on a single variable. Used predominantly to apply BC on the second
        asymptotics of the pressure variable.

        Parameters
        ----------
        variable : np.ndarray
            The node-centered or cell-centered variable.
        context: BCApplicationContext
            The context object of the method. It contains the flag 'is_nodal' to determine if the variable is defined
            on nodes or cells.

        Notes
        -----
        If the flag is_nodal is on, i.e. if the variable
        is defined on nodes, the boundary node is not copied on ghost nodes. Therefore, the 'mode' will be set to
        'reflect'. Otherwise, it will be 'symmetric' as usual for wall boundary conditions for cells.
        """

        mode: Literal["symmetric", "edge", "reflect"] = "symmetric"
        is_nodal = context.is_nodal
        if is_nodal:
            mode = "reflect"

        if self.grid.nc[self.direction] == 1:
            # Single cell case
            # Apply np.pad with mode "edge" to copy the single cell to ghost cells
            mode = "edge"

        variable[...] = np.pad(
            variable[self.inner_slice[:-1]],
            self.pad_width[:-1],
            mode=mode,
        )

    def apply_extra(self, variables: np.ndarray, operation: "ExtraBCOperation") -> None:
        """This is applied on a nodal variable. Rescale nodes at the boundary by a factor.

        Parameters
        ----------
        variable : np.ndarray
            The variables to apply the extra boundary condition on. The compatibility of variables with the
            extra BC is on the user.
        operation: ExtraBCOperation
            The ExtraBCOperation object to be applied.
        """

        if isinstance(operation, WallAdjustment):
            # Assumption: Variables is a single nodal variable
            factor = operation.factor
            boundary_nodes_slice = self._boundary_slice()
            variables[boundary_nodes_slice] *= factor
        elif isinstance(operation, WallFluxCorrection):
            # Assumption: Variables is the momenta stacked on the last axis.
            factor = operation.factor
            pad_slice = self.padded_slices()
            for i in range(variables.shape[-1]):
                variables[pad_slice + (i,)] *= factor
        else:
            pass

    @property
    def inner_slice(self) -> Tuple[slice, ...]:
        """Create the inner slice of the array from one side, i.e. either (0, -ng) or (ng, None) in the corresponding
        direction."""

        side: int = self.side_axis
        # create slice(ng, None) or slice(0, ng)
        slc = slice(self.ng[side], None) if side == 0 else slice(0, -self.ng[side])
        # Create the slice object
        inner_slice = [slice(None)] * (self.ndim + 1)
        inner_slice[self.direction] = slc
        return tuple(inner_slice)

    def _boundary_slice(self) -> Tuple[slice, ...]:
        """Create the slice for the nodes/cells at the boundary."""
        # Notice in backward indexing, the third to last element in the last inner node (boundary node).
        # Therefore, in backward indexing -ig - 1 is the correct index of the last inner node.
        idx = (
            self.ng[self.side_axis]
            if self.side_axis == 0
            else -self.ng[self.side_axis] - 1
        )
        boundary_nodes_slice = copy.deepcopy(self.full_inner_slice)
        boundary_nodes_slice[self.direction] = idx
        return tuple(boundary_nodes_slice)

    def padded_slices(self) -> Tuple[slice, ...]:
        """Returns the slice of the padded part."""
        side = self._find_side_axis()
        slc = slice(0, self.ng[side]) if side == 0 else slice(-self.ng[side], None)
        pad_slice = [slice(None)] * self.ndim
        pad_slice[self.direction] = slc
        return tuple(pad_slice)


class NonReflectiveOutlet(BaseBoundaryCondition):
    def apply(self, cell_vars, *args, **kwargs):
        # Characteristic-based extrapolation
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
    x = Wall(**params)
    print(variables.cell_vars[..., VI.RHO])
    x.apply(variables.cell_vars)
    print(variables.cell_vars[..., VI.RHO])


if __name__ == "__main__":
    example_usage()
