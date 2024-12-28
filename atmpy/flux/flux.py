import numpy as np
from typing import Callable, Union
from atmpy.grid.kgrid import Grid
from atmpy.flux.riemann_solvers import *

import numpy as np
from numba import njit

from atmpy.flux.boundary_conditions import apply_boundary_conditions
from .reconstruction import muscl_reconstruct_1d
# from .slope_limiters import ...
# from atmpy.physics import eos (if needed or pass in the constructor)

class Flux:
    """
    The Flux class encapsulates all the steps to compute fluxes for the FVM.
    Emphasizes in-place updates and optional numba for performance.
    """

    def __init__(self, eos, bc=apply_boundary_conditions, recon_fn=muscl_reconstruct_1d):
        """
        Parameters
        ----------
        eos : EOS
            An equation-of-state object (IdealGasEOS, BarotropicEOS, etc.).
        bc : callable
            Boundary condition function with signature bc(variables, grid, bc_type).
        recon_fn : callable
            Reconstruction function, e.g., muscl_reconstruct_1d(cell_values, dx).
        """
        self.eos = eos
        self.bc = bc
        self.recon_fn = recon_fn

    def compute_fluxes_1d(self, variables, grid, dt, bc_type="periodic"):
        """
        Example 1D flux computation with MUSCL reconstruction + Rusanov flux
        (just as an example Riemann solver).

        Parameters
        ----------
        variables : Variables
            The container holding cell_vars, etc.
        grid : Grid
            The 1D grid object with e.g. dx, nx, ...
        dt : float
            Time step
        bc_type : str
            Type of boundary condition
        """
        # 1) Apply boundary conditions (in-place)
        self.bc(variables, grid, bc_type)

        # 2) Reconstruction (in-place or returns arrays)
        cell_vals = variables.cell_vars  # shape (nx, num_vars) in 1D
        dx = grid.cell.dx  # or grid.node.dx, whichever is correct
        left_states, right_states = self.recon_fn(cell_vals, dx)

        # 3) Riemann solver for each interface. Example: Rusanov or any other.
        nx = cell_vals.shape[0]
        fluxes = np.zeros_like(cell_vals)  # shape (nx, num_vars)
        for i in range(1, nx):
            # left state from right_states[i-1], right state from left_states[i]
            ul = right_states[i-1]
            ur = left_states[i]

            # Example: Rusanov flux in 1D
            # approximate wave speed
            # (Just a demonstration; real Riemann solver can be more advanced.)
            rho_l = ul[0]
            rho_r = ur[0]
            # Pressure from EOS if needed:
            # p_l = self.eos.pressure(...)
            # p_r = self.eos.pressure(...)

            # Some wave speed estimation (or real formula)
            # s_max = max(|u_l| + a_l, |u_r| + a_r) etc.
            s_max = 1.0  # placeholder

            # simple flux
            f_left = self._physical_flux(ul)
            f_right = self._physical_flux(ur)
            flux_face = 0.5*(f_left + f_right) - 0.5*s_max*(ur - ul)

            # store in flux array
            fluxes[i] = flux_face

        # 4) Update the cell averages in place (Godunov's method or similar)
        for i in range(1, nx-1):
            cell_vals[i] -= dt/dx * (fluxes[i] - fluxes[i-1])

    @staticmethod
    @njit
    def _physical_flux(u):
        """
        Compute the physical flux f(u) for each cell state u.
        For example, in 1D Euler:
            u = [ rho, rhoU, rhoE ],
            f(u) = [ rhoU, rhoU^2 + p, (rhoE + p)U ]
        Here we demonstrate a placeholder for a simpler scalar PDE or something.
        """
        # For demonstration: let's say u is shape (1,) for a simple advective PDE
        return u  # Identity flux if PDE is just "du/dt + u*du/dx=0"

        # Real 1D Euler example (requires more details: gamma, etc.).
        # We'll keep it short to highlight structure.




# class Flux:
#     def __init__(
#         self,
#         grid: Grid,
#         flux_type: str,
#         use_limiter: bool = False,
#         limiter_type: str = None,
#         **kwargs,
#     ):
#         """
#         Initialize the Flux class with parameters and references.
#
#         Parameters
#         ----------
#         grid : :py:class:`atmpy.grid.kgrid.Grid`
#             A reference to the Grid object that holds geometry and indexing info.
#             The grid object should provide:
#               - grid.ndim: number of spatial dimensions (1, 2, or 3)
#               - grid.num_cells_x, grid.num_cells_y, (grid.num_cells_z if 3D): number of cells
#         flux_type : str
#             Indicates which flux scheme to use (e.g. "Roe", "HLLC", "Rusanov")
#         use_limiter : bool, optional
#             Whether to apply a slope/flux limiter
#         limiter_type : str, optional
#             Type of limiter (e.g. "vanleer", "minmod")
#         kwargs : dict
#             Additional parameters for flux computation (e.g. epsilon for Roe fix)
#         """
#         self.flux_type = flux_type
#         self.grid = grid
#         self.use_limiter = use_limiter
#         self.limiter_type = limiter_type
#
#         # Additional solver-specific parameters can be passed in kwargs
#         for key, value in kwargs.items():
#             setattr(self, key, value)
#
#         self.ndim = self.grid.dimensions
#         # Initialize flux storage arrays after we know how many vars and cells we have.
#         # This will be done in initialize_flux_storage
#
#         # We'll store interface fluxes in a dictionary keyed by dimension:
#         # 'x' -> fluxes in x-direction
#         # 'y' -> fluxes in y-direction (if ndim >= 2)
#         # 'z' -> fluxes in z-direction (if ndim == 3)
#         self.interface_fluxes = {}
#
#     def set_flux_type(self, flux_type: str):
#         """Set the flux type to a new scheme
#
#         Parameters
#         ----------
#         flux_type : str
#             New flux scheme identifier."""
#         self.flux_type = flux_type
#
#     def initialize_flux_storage(self, num_vars):
#         """
#         Allocate memory for flux storage arrays.
#         Shape of flux arrays depends on the dimension:
#           - 1D: (num_cells_x + 1, num_vars)
#           - 2D: (num_cells_x + 1, num_cells_y, num_vars) for x-direction
#                  (num_cells_x, num_cells_y + 1, num_vars) for y-direction
#           - 3D: Similarly extended to z-direction
#         """
#         if self.ndim == 1:
#             self.interface_fluxes["x"] = np.zeros((self.grid.nx + 1, num_vars))
#         elif self.ndim == 2:
#             self.interface_fluxes["x"] = np.zeros(
#                 (self.grid.nx + 1, self.grid.ny, num_vars)
#             )
#             self.interface_fluxes["y"] = np.zeros(
#                 (self.grid.nx, self.grid.ny + 1, num_vars)
#             )
#         elif self.ndim == 3:
#             self.interface_fluxes["x"] = np.zeros(
#                 (self.grid.nx + 1, self.grid.ny, self.grid.nz, num_vars)
#             )
#             self.interface_fluxes["y"] = np.zeros(
#                 (self.grid.nx, self.grid.ny + 1, self.grid.nz, num_vars)
#             )
#             self.interface_fluxes["z"] = np.zeros(
#                 (self.grid.nx, self.grid.ny, self.grid.nz + 1, num_vars)
#             )
#         else:
#             raise ValueError("Number of dimensions not supported.")
#
#     def compute_interface_fluxes(self, variables):
#         """
#         Given the current state variables, compute the fluxes at each interface in all directions.
#
#         Parameters
#         ----------
#         variables : Variable
#             The current state of the variables.
#         """
#         cons_states = variables.get_conservative_vars()
#
#         # Depending on the dimension, perform reconstruction and Riemann solves along each direction.
#         # For example, in 1D:
#         #   - Reconstruct left/right states in x-direction, call Riemann solver.
#         # In 2D:
#         #   - For x-direction interfaces: reconstruct states in x-direction slices, solve Riemann problem.
#         #   - For y-direction interfaces: similarly for y-direction.
#         # In 3D:
#         #   - Do the same for x, y, and z directions.
#
#         # The actual indexing and the way you slice cons_states depends heavily on your data layout.
#         # Below we show a conceptual approach:
#
#         if self.ndim >= 1:
#             # Compute fluxes in x-direction
#             left_states_x, right_states_x = self._reconstruct_states(
#                 cons_states, direction="x"
#             )
#             self._solve_all_interfaces(left_states_x, right_states_x, direction="x")
#
#         if self.ndim >= 2:
#             # Compute fluxes in y-direction
#             left_states_y, right_states_y = self._reconstruct_states(
#                 cons_states, direction="y"
#             )
#             self._solve_all_interfaces(left_states_y, right_states_y, direction="y")
#
#         if self.ndim == 3:
#             # Compute fluxes in z-direction
#             left_states_z, right_states_z = self._reconstruct_states(
#                 cons_states, direction="z"
#             )
#             self._solve_all_interfaces(left_states_z, right_states_z, direction="z")
#
#     def _reconstruct_states(self, cons_states, direction="x"):
#         """
#         Reconstruct states at cell interfaces in the given direction.
#
#         Parameters
#         ----------
#         cons_states : np.ndarray
#             Array of conservative variables per cell. Shape typically depends on dimension:
#               1D: (num_cells_x, num_vars)
#               2D: (num_cells_x, num_cells_y, num_vars)
#               3D: (num_cells_x, num_cells_y, num_cells_z, num_vars)
#         direction : str
#             'x', 'y', or 'z' for which direction to reconstruct.
#
#         Returns
#         -------
#         left_states, right_states : np.ndarray
#             Arrays containing reconstructed left and right states at each interface.
#             Their shapes depend on dimension and direction. For example, in 2D:
#               - For direction='x': left_states and right_states might have shape
#                 (num_cells_x+1, num_cells_y, num_vars).
#               - For direction='y': (num_cells_x, num_cells_y+1, num_vars).
#         """
#         # Placeholder - actual implementation needed.
#         raise NotImplementedError(
#             f"State reconstruction for {direction}-direction not implemented."
#         )
#
#     def _solve_all_interfaces(self, left_states, right_states, direction="x"):
#         """
#         Apply the chosen Riemann solver scheme to every interface in the specified direction.
#
#         Updates self.interface_fluxes[direction] in place.
#
#         Parameters
#         ----------
#         left_states : np.ndarray
#             Left-side reconstructed states at each interface.
#         right_states : np.ndarray
#             Right-side reconstructed states at each interface.
#         direction : str
#             'x', 'y', or 'z'
#         """
#         # Example pseudo-code (1D-like indexing shown, adapt for multiple dimensions):
#         # for i in all interfaces in direction:
#         #    flux = self.riemann_solver(left_states[i], right_states[i], direction)
#         #    self.interface_fluxes[direction][i] = flux
#         raise NotImplementedError(
#             f"Riemann solver loop for {direction}-direction not implemented."
#         )
#
#     def riemann_solver(self, left_state, right_state, direction="x"):
#         """
#         Solve the Riemann problem for the given left and right states in the specified direction.
#
#         Parameters
#         ----------
#         left_state, right_state : np.ndarray
#             State vectors (usually conservative or primitive) on each side of an interface.
#         direction : str
#             'x', 'y', or 'z' for the corresponding direction
#
#         Returns
#         -------
#         flux : np.ndarray
#             The flux across this interface in the given direction.
#         """
#         if self.flux_type == "Roe":
#             flux = roe_solver(left_state, right_state, direction)
#         elif self.flux_type == "HLLC":
#             flux = hllc_solver(left_state, right_state, direction)
#         elif self.flux_type == "Rusanov":
#             flux = rusanov_solver(left_state, right_state, direction)
#         else:
#             raise ValueError(f"Unsupported flux_type: {self.flux_type}")
#         return flux
#
#     def apply_flux_limiters(self, *args, **kwargs):
#         """
#         Apply limiters to slopes or fluxes if necessary.
#         """
#         raise NotImplementedError("Flux limiter application not implemented.")
#
#     def get_interface_fluxes(self, direction=None):
#         """
#         Return the computed fluxes for external use, e.g. in the time management step.
#
#         Parameters
#         ----------
#         direction : str, optional
#             If specified, return fluxes only for that direction ('x', 'y', or 'z').
#             If None, return a dict of all directions.
#         """
#         if direction is not None:
#             return self.interface_fluxes[direction]
#         return self.interface_fluxes
#
#     def print_debug_info(self):
#         """
#         Print out debug information for inspection.
#         """
#         print("Flux Type:", self.flux_type)
#         # print("Gamma:", self.gamma)
#         print("Use Limiter:", self.use_limiter)
#         print("Limiter Type:", self.limiter_type)
#         print("Number of dimensions:", self.ndim)
#         for d in self.interface_fluxes:
#             print(
#                 f"Interface fluxes [{d}]-direction shape:",
#                 self.interface_fluxes[d].shape,
#             )
