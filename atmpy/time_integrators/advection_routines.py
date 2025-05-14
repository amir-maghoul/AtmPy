"""This module contains the different advection routines to be passed to the solver class. The signature of the following
function are the same."""

import numpy as np
from typing import TYPE_CHECKING

from atmpy.infrastructure.utility import (
    dimension_directions,
    directional_indices,
    direction_axis,
)

if TYPE_CHECKING:
    from atmpy.flux.flux import Flux
    from atmpy.variables.variables import Variables
    from atmpy.grid.kgrid import Grid
from atmpy.infrastructure.enums import (
    VariableIndices as VI,
    PrimitiveVariableIndices as PI,
)


def upwind_strang_split_advection(
    grid: "Grid", variables: "Variables", flux: "Flux", dt: float, *args, **kwargs
) -> None:
    """Compute the Strang-split upwind advection in all directions. This function applies the boundary conditions on
    updated values of the variables in each iteration. The boundary manager object should be passed as a keyword argument.
    It should not be an empty object. It should have been completely set up using its setup_boundary_conditions method
    prior to passing it to this function.

    Parameters
    ----------
    grid: Grid
        The spatial grid object
    variables: Variables
        The variables container.
    flux: Flux
        The flux object.
    dt: float
        The time step.
    """
    boundary_manager: BoundaryManager = kwargs.pop("boundary_manager")
    # Check whether the boundary manager has already been set up or not
    if not boundary_manager.boundary_conditions:
        raise ValueError(
            "The setup of the boundary manager object should have been done before passing it to the advection function."
        )

    ndim: int = grid.ndim
    directions = dimension_directions(ndim)

    # First round of strang splitting. Forward: x-y-z
    for direction in directions:
        boundary_manager.apply_boundary_on_direction(variables.cell_vars, direction)
        first_order_directional_rk(grid, variables, flux, direction, dt / 2)
        # boundary_manager.apply_single_boundary_condition(variables.cell_vars, direction)

    # Second round of strang splitting. Backward: z-y-x
    for direction in directions[::-1]:
        boundary_manager.apply_boundary_on_direction(variables.cell_vars, direction)
        first_order_directional_rk(grid, variables, flux, direction, dt / 2)
        # boundary_manager.apply_single_boundary_condition(variables.cell_vars, direction)


def first_order_directional_rk(
    grid: "Grid", variables: "Variables", flux: "Flux", direction: str, dt: float
) -> None:
    """Computes the upwind first-order Runge-Kutta integration in the given direction. This is the core function to be
    applied to multiple directions to complete the advection routine in strang splitting.

    Parameters
    ----------
    grid: Grid
        The spatial grid object
    variables: Variables
        The variables container.
    flux: Flux
        The flux object.
    direction: str
        The direction of advection.
    dt: float
        The time step.
    """

    ndim: int = grid.ndim
    direction_int: int = direction_axis(direction)

    lmbda: float = dt / grid.dxyz[direction_int]
    # Find the left and right indices
    left_idx, right_idx, _ = directional_indices(ndim, direction, full=True)
    flux.apply_riemann_solver(lmbda, direction)
    variables.cell_vars[...] += lmbda * (
        flux.flux[direction][left_idx] - flux.flux[direction][right_idx]
    )


if __name__ == "__main__":
    from atmpy.physics.eos import ExnerBasedEOS
    from atmpy.grid.utility import DimensionSpec, create_grid
    from atmpy.variables.variables import Variables
    from atmpy.flux.flux import Flux
    from atmpy.boundary_conditions.utility import create_params
    from atmpy.infrastructure.enums import (
        BoundarySide as BdrySide,
        BoundaryConditions as BdryType,
    )
    from atmpy.boundary_conditions.boundary_manager import BoundaryManager

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

    variables = Variables(grid, 5, 1)
    variables.cell_vars[..., VI.RHO] = 1
    variables.cell_vars[..., VI.RHO][1:-1, 1:-1] = 4
    variables.cell_vars[..., VI.RHOU] = array
    variables.cell_vars[..., VI.RHOY] = 2

    rng.shuffle(arr)
    array = arr.reshape(nnx, nny)
    variables.cell_vars[..., VI.RHOV] = array
    eos = ExnerBasedEOS()
    flux = Flux(grid, variables, eos, dt)

    bc_data = {}
    create_params(bc_data, BdrySide.LEFT, BdryType.PERIODIC, direction="x", grid=grid)
    create_params(bc_data, BdrySide.RIGHT, BdryType.PERIODIC, direction="x", grid=grid)
    create_params(bc_data, BdrySide.BOTTOM, BdryType.PERIODIC, direction="y", grid=grid)
    create_params(bc_data, BdrySide.TOP, BdryType.PERIODIC, direction="y", grid=grid)

    manager = BoundaryManager()
    manager.setup_conditions(bc_data)

    upwind_strang_split_advection(grid, variables, flux, dt, boundary_manager=manager)
    print(variables.cell_vars[..., VI.RHOU])
