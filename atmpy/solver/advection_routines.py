"""This module contains the different advection routines to be passed to the solver class."""


def upwind_strang_split_advection(flux_obj, variables, grid, direction, dt):
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
