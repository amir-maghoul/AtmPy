import numpy as np
# from numba import njit

def apply_boundary_conditions(variables, grid, bc_type="periodic"):
    """
    Applies the chosen boundary condition in-place on 'variables'.

    Parameters
    ----------
    variables : Variables
        The object holding cell_vars, node_vars, and so on.
    grid : Grid
        The computational grid (has info like number of ghost cells, etc.)
    bc_type : str, optional
        Type of boundary condition to apply (e.g., 'periodic', 'dirichlet', etc.)

    Returns
    -------
    None
        The function modifies 'variables' in-place.
    """
    if bc_type == "periodic":
        # A simple 1D example for demonstration
        # (Extend similarly for multi-d, or place 2D/3D code in separate sections.)
        if grid.dim == 1:
            # For example, in 1D: cell_vars is shape (nx, num_vars)
            nx = grid.cell.nnx  # or grid.node.nnx depending on your usage
            for var_idx in range(variables.num_vars_cell):
                # left ghost cells = last cells
                variables.cell_vars[0, var_idx] = variables.cell_vars[nx-2, var_idx]
                # right ghost cells = first cells
                variables.cell_vars[nx-1, var_idx] = variables.cell_vars[1, var_idx]
        # Extend to 2D, 3D, or other BC types as needed
    else:
        # Implement other BC types similarly
        pass
