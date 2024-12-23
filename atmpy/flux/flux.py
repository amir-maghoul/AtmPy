import numpy as np
from typing import Callable, Union
from atmpy.grid.kgrid import Grid
from atmpy.flux.riemann_solvers import *


class Flux:
    def __init__(
        self,
        grid: Grid,
        flux_type: str,
        use_limiter: bool = False,
        limiter_type: str = None,
        **kwargs,
    ):
        """
        Initialize the Flux class with parameters and references.

        Parameters
        ----------
        grid : :py:class:`atmpy.grid.kgrid.Grid`
            A reference to the Grid object that holds geometry and indexing info.
            The grid object should provide:
              - grid.ndim: number of spatial dimensions (1, 2, or 3)
              - grid.num_cells_x, grid.num_cells_y, (grid.num_cells_z if 3D): number of cells
        flux_type : str
            Indicates which flux scheme to use (e.g. "Roe", "HLLC", "Rusanov")
        use_limiter : bool, optional
            Whether to apply a slope/flux limiter
        limiter_type : str, optional
            Type of limiter (e.g. "vanleer", "minmod")
        kwargs : dict
            Additional parameters for flux computation (e.g. epsilon for Roe fix)
        """
        self.flux_type = flux_type
        self.grid = grid
        self.use_limiter = use_limiter
        self.limiter_type = limiter_type

        # Additional solver-specific parameters can be passed in kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.ndim = self.grid.dimensions
        # Initialize flux storage arrays after we know how many vars and cells we have.
        # This will be done in initialize_flux_storage

        # We'll store interface fluxes in a dictionary keyed by dimension:
        # 'x' -> fluxes in x-direction
        # 'y' -> fluxes in y-direction (if ndim >= 2)
        # 'z' -> fluxes in z-direction (if ndim == 3)
        self.interface_fluxes = {}

    def set_flux_type(self, flux_type: str):
        """Set the flux type to a new scheme

        Parameters
        ----------
        flux_type : str
            New flux scheme identifier."""
        self.flux_type = flux_type

    def initialize_flux_storage(self, num_vars):
        """
        Allocate memory for flux storage arrays.
        Shape of flux arrays depends on the dimension:
          - 1D: (num_cells_x + 1, num_vars)
          - 2D: (num_cells_x + 1, num_cells_y, num_vars) for x-direction
                 (num_cells_x, num_cells_y + 1, num_vars) for y-direction
          - 3D: Similarly extended to z-direction
        """
        if self.ndim == 1:
            self.interface_fluxes["x"] = np.zeros((self.grid.nx + 1, num_vars))
        elif self.ndim == 2:
            self.interface_fluxes["x"] = np.zeros(
                (self.grid.nx + 1, self.grid.ny, num_vars)
            )
            self.interface_fluxes["y"] = np.zeros(
                (self.grid.nx, self.grid.ny + 1, num_vars)
            )
        elif self.ndim == 3:
            self.interface_fluxes["x"] = np.zeros(
                (self.grid.nx + 1, self.grid.ny, self.grid.nz, num_vars)
            )
            self.interface_fluxes["y"] = np.zeros(
                (self.grid.nx, self.grid.ny + 1, self.grid.nz, num_vars)
            )
            self.interface_fluxes["z"] = np.zeros(
                (self.grid.nx, self.grid.ny, self.grid.nz + 1, num_vars)
            )
        else:
            raise ValueError("Number of dimensions not supported.")

    def compute_interface_fluxes(self, variables):
        """
        Given the current state variables, compute the fluxes at each interface in all directions.

        Parameters
        ----------
        variables : Variable
            The current state of the variables.
        """
        cons_states = variables.get_conservative_vars()

        # Depending on the dimension, perform reconstruction and Riemann solves along each direction.
        # For example, in 1D:
        #   - Reconstruct left/right states in x-direction, call Riemann solver.
        # In 2D:
        #   - For x-direction interfaces: reconstruct states in x-direction slices, solve Riemann problem.
        #   - For y-direction interfaces: similarly for y-direction.
        # In 3D:
        #   - Do the same for x, y, and z directions.

        # The actual indexing and the way you slice cons_states depends heavily on your data layout.
        # Below we show a conceptual approach:

        if self.ndim >= 1:
            # Compute fluxes in x-direction
            left_states_x, right_states_x = self._reconstruct_states(
                cons_states, direction="x"
            )
            self._solve_all_interfaces(left_states_x, right_states_x, direction="x")

        if self.ndim >= 2:
            # Compute fluxes in y-direction
            left_states_y, right_states_y = self._reconstruct_states(
                cons_states, direction="y"
            )
            self._solve_all_interfaces(left_states_y, right_states_y, direction="y")

        if self.ndim == 3:
            # Compute fluxes in z-direction
            left_states_z, right_states_z = self._reconstruct_states(
                cons_states, direction="z"
            )
            self._solve_all_interfaces(left_states_z, right_states_z, direction="z")

    def _reconstruct_states(self, cons_states, direction="x"):
        """
        Reconstruct states at cell interfaces in the given direction.

        Parameters
        ----------
        cons_states : np.ndarray
            Array of conservative variables per cell. Shape typically depends on dimension:
              1D: (num_cells_x, num_vars)
              2D: (num_cells_x, num_cells_y, num_vars)
              3D: (num_cells_x, num_cells_y, num_cells_z, num_vars)
        direction : str
            'x', 'y', or 'z' for which direction to reconstruct.

        Returns
        -------
        left_states, right_states : np.ndarray
            Arrays containing reconstructed left and right states at each interface.
            Their shapes depend on dimension and direction. For example, in 2D:
              - For direction='x': left_states and right_states might have shape
                (num_cells_x+1, num_cells_y, num_vars).
              - For direction='y': (num_cells_x, num_cells_y+1, num_vars).
        """
        # Placeholder - actual implementation needed.
        raise NotImplementedError(
            f"State reconstruction for {direction}-direction not implemented."
        )

    def _solve_all_interfaces(self, left_states, right_states, direction="x"):
        """
        Apply the chosen Riemann solver scheme to every interface in the specified direction.

        Updates self.interface_fluxes[direction] in place.

        Parameters
        ----------
        left_states : np.ndarray
            Left-side reconstructed states at each interface.
        right_states : np.ndarray
            Right-side reconstructed states at each interface.
        direction : str
            'x', 'y', or 'z'
        """
        # Example pseudo-code (1D-like indexing shown, adapt for multiple dimensions):
        # for i in all interfaces in direction:
        #    flux = self.riemann_solver(left_states[i], right_states[i], direction)
        #    self.interface_fluxes[direction][i] = flux
        raise NotImplementedError(
            f"Riemann solver loop for {direction}-direction not implemented."
        )

    def riemann_solver(self, left_state, right_state, direction="x"):
        """
        Solve the Riemann problem for the given left and right states in the specified direction.

        Parameters
        ----------
        left_state, right_state : np.ndarray
            State vectors (usually conservative or primitive) on each side of an interface.
        direction : str
            'x', 'y', or 'z' for the corresponding direction

        Returns
        -------
        flux : np.ndarray
            The flux across this interface in the given direction.
        """
        if self.flux_type == "Roe":
            flux = roe_solver(left_state, right_state, direction)
        elif self.flux_type == "HLLC":
            flux = hllc_solver(left_state, right_state, direction)
        elif self.flux_type == "Rusanov":
            flux = rusanov_solver(left_state, right_state, direction)
        else:
            raise ValueError(f"Unsupported flux_type: {self.flux_type}")
        return flux

    def apply_flux_limiters(self, *args, **kwargs):
        """
        Apply limiters to slopes or fluxes if necessary.
        """
        raise NotImplementedError("Flux limiter application not implemented.")

    def get_interface_fluxes(self, direction=None):
        """
        Return the computed fluxes for external use, e.g. in the time management step.

        Parameters
        ----------
        direction : str, optional
            If specified, return fluxes only for that direction ('x', 'y', or 'z').
            If None, return a dict of all directions.
        """
        if direction is not None:
            return self.interface_fluxes[direction]
        return self.interface_fluxes

    def print_debug_info(self):
        """
        Print out debug information for inspection.
        """
        print("Flux Type:", self.flux_type)
        # print("Gamma:", self.gamma)
        print("Use Limiter:", self.use_limiter)
        print("Limiter Type:", self.limiter_type)
        print("Number of dimensions:", self.ndim)
        for d in self.interface_fluxes:
            print(
                f"Interface fluxes [{d}]-direction shape:",
                self.interface_fluxes[d].shape,
            )
