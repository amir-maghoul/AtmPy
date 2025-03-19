"""This module contains the different advection routines to be passed to the solver class. The signature of the following
function are the same."""

import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from atmpy.flux.flux import Flux
    from atmpy.variables.variables import Variables
    from atmpy.grid.kgrid import Grid
from atmpy.infrastructure.enums import VariableIndices as VI, PrimitiveVariableIndices as PI
from atmpy.flux.utility import directional_indices
from atmpy.solver.utility import zipped_direction

def upwind_strang_split_advection(grid: "Grid", variables: "Variables", flux_obj: "Flux", direction: str, dt: float):
    """Compute advection routine per Strang splitting.

    It delegates the computation of interface fluxes by calling the flux_obj's
    apply_riemann_solver() method. The 'lmbda' parameter is computed (for example)
    as dt divided by the grid spacing in the specified direction.
    Finally, it applies the flux differences to update the variables.

    Parameters:
      flux_obj   - The flux object that holds apply_riemann_solver().
      variables  - Variable container (cell-centered data).
      grid       - Grid container (to extract spacing dx, dy, etc.).
      direction  - A string specifying the update direction, e.g. 'x' or 'y'.
      dt         - The time step for this update.
    """
    # For this example, we assume uniform spacing; in practice use grid.dx, grid.dy, etc.
    spacing = getattr(grid, f"{direction}d") if hasattr(grid, f"{direction}d") else 1.0
    # Compute the lambda parameter for the Riemann solver
    lmbda = dt / spacing

    # Compute the flux for this direction;
    # your flux object's method will compute left/right states and call the Riemann solver accordingly.
    flux_obj.apply_riemann_solver(lmbda, direction)

    # Now apply the flux update. In a full FVM implementation, you would compute the divergence
    # of the flux and update the cell-centered variables, e.g.
    #   U_new = U_old - dt/dx * (flux_right - flux_left)
    # Here we only simulate this process with a print statement.
    print(f"Advection update in {direction} direction with dt = {dt}")
    # For illustration purposes, you might imagine
    # variables.cell_vars = update(variables.cell_vars, flux_obj.flux_data[direction])

def first_order_rk(grid: "Grid", variables: "Variables", flux: "Flux", dt: float, *args, **kwargs):
    """ Computes the first order Runge-Kutta integration.

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
    # direction of calculation of the flux
    direction: str = kwargs.pop("direction")
    ndim: int = grid.ndim
    directions = zipped_direction(ndim)



    for direction_str, direction_int in directions:
        lmbda = dt / grid.dxyz[direction_int]
        flux.apply_riemann_solver(lmbda, direction_str)

if  __name__ == "__main__":
    from atmpy.physics.eos import ExnerBasedEOS
    from atmpy.grid.utility import DimensionSpec, create_grid
    from atmpy.variables.variables import Variables
    from atmpy.flux.flux import Flux

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
    variables.cell_vars[..., VI.RHOU] = array
    variables.cell_vars[..., VI.RHOY] = 2

    rng.shuffle(arr)
    array = arr.reshape(nnx, nny)
    variables.cell_vars[..., VI.RHOV] = array
    eos = ExnerBasedEOS()
    flux = Flux(grid, variables, eos, dt)

    kwargs = {"direction": "x"}
    first_order_rk(grid, variables, flux, dt, **kwargs)
    print(flux.flux["x"][..., VI.RHOU])


