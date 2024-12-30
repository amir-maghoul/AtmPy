import numpy as np
from typing import Callable, Union, Tuple
from ..grid.kgrid import Grid
from ..variables.variables import Variables
from .registries import *
from ..data.enums import SlopeLimiters, RiemannSolvers, FluxReconstructions
from ._factory import get_slope_limiter, get_riemann_solver, get_reconstruction_method


class Flux:
    def __init__(
        self,
        grid: Grid,
        vars: Variables,
        solver: RiemannSolvers = RIEMANN_SOLVERS.HLL,
        limiter: SlopeLimiters = SLOPE_LIMITERS.MINMOD,
        reconstruction: FluxReconstructions = FLUX_RECONSTRUCTION.MUSCL,
    ):
        """
        Parameters
        ----------
        grid : :py:class:`atmpy.grid.kgrid.Grid`
            The computational grid.
        vars : :py:class:`atmpy.variables.variables.Variables`
            The variable container
        solver : RiemannSolvers (Enum)
            An enum name from the implemented RiemannSolvers Enum,
            e.g. RiemannSolvers.HLL, RiemannSolvers.RUSANOV, etc.
        limiter : SlopeLimiters
            An enum name from the implemented SlopeLimiters Enum,
            e.g. SlopeLimiters.MINMOD, SlopeLimiters.MC_LIMITER etc.
        reconstruction : FluxReconstructions
            An enum name from the implemented FluxReconstructions Enum,
            e.g. FluxReconstructions.MUSCL, etc.
        """
        self.grid = grid
        self.vars = vars
        self.limiter = get_slope_limiter(limiter)
        self.riemann_solver = get_riemann_solver(solver)
        self.reconstruction_function = get_reconstruction_method(reconstruction)
        self.ndim = self.grid.dimensions

        self.flux = {
            "x": np.empty(
                (
                    grid.ncx_total + 1,
                    grid.ncy_total,
                    grid.ncz_total,
                    vars.num_vars_cell,
                ),
                dtype=np.float64,
            )
        }
        if self.ndim > 1:
            self.flux["y"] = np.empty(
                (
                    grid.ncx_total,
                    grid.ncy_total + 1,
                    grid.ncz_total,
                    vars.num_vars_cell,
                ),
                dtype=np.float64,
            )
        if self.ndim > 2:
            self.flux["z"] = np.empty(
                (
                    grid.ncx_total,
                    grid.ncy_total,
                    grid.ncz_total + 1,
                    vars.num_vars_cell,
                ),
                dtype=np.float64,
            )

    def compute_fluxes(self) -> None:
        """Compute fluxes in the x, y, and z directions. Update flux attribute in-place."""
        if self.ndim == 1:
            self.flux["x"] = self._compute_directional_flux(
                self.vars.cell_vars, direction="x"
            )
        if self.ndim == 2:
            self.flux["x"] = self._compute_directional_flux(
                self.vars.cell_vars, direction="x"
            )
            self.flux["y"] = self._compute_directional_flux(
                self.vars.cell_vars, direction="y"
            )
        if self.ndim == 3:
            self.flux["x"] = self._compute_directional_flux(
                self.vars.cell_vars, direction="x"
            )
            self.flux["y"] = self._compute_directional_flux(
                self.vars.cell_vars, direction="y"
            )
            self.flux["z"] = self._compute_directional_flux(
                self.vars.cell_vars, direction="z"
            )

    def _compute_directional_flux(
        self, cell_vars: np.ndarray, direction: str
    ) -> np.ndarray:
        """
        Compute the numerical flux in a specific direction.

        Parameters:
        cell_vars : np.ndarray
            Conservative variables with shape (nx, [ny], [nz], n_vars).
        direction : str
            Spatial direction of flux calculation.
        ndim : int:
            Number of spatial dimensions.

        Returns:
        - np.ndarray: Numerical flux array in the specified direction.
        """
        left_state, right_state = self.reconstruction_function(
            cell_vars, direction, self.ndim
        )
        flux = self.riemann_solver(left_state, left_state, direction)

        return flux

    def compute_physical_flux(self):
        pass
