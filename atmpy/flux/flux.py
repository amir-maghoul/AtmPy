import numpy as np
from typing import Callable, Union, Tuple, Sequence
from atmpy.grid.kgrid import Grid
from atmpy.variables.variables import Variables
from atmpy.flux.registries import minmod, van_leer, superbee, mc_limiter
from atmpy.data.enums import SlopeLimiters, RiemannSolvers, FluxReconstructions
from atmpy.flux._factory import (
    get_slope_limiter,
    get_riemann_solver,
    get_reconstruction_method,
)
from numba import njit, prange


@njit(parallel=True)
def compute_physical_flux_numba(
    s: np.ndarray, rho: np.ndarray, velocities: np.ndarray
) -> np.ndarray:
    """
    Calculate the physical flux vector for a given flow direction using Numba for optimization.
    """
    num_velocities = velocities.shape[0]
    N, M = s.shape
    physical_flux = np.empty((1 + num_velocities, N, M), dtype=s.dtype)

    # Compute mass flux
    for i in prange(N):
        for j in prange(M):
            physical_flux[0, i, j] = s[i, j] * rho[i, j]
            for k in range(num_velocities):
                physical_flux[1 + k, i, j] = (
                    physical_flux[0, i, j] * velocities[k, i, j]
                )

    return physical_flux


class Flux:
    def __init__(
        self,
        grid: Grid,
        variables: Variables,
        solver: RiemannSolvers = RiemannSolvers.HLL,
        limiter: SlopeLimiters = SlopeLimiters.MINMOD,
        reconstruction: FluxReconstructions = FluxReconstructions.MUSCL,
    ):
        """
        Parameters
        ----------
        grid : :py:class:`atmpy.grid.kgrid.Grid`
            The computational grid.
        variables : :py:class:`atmpy.variables.variables.Variables`
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
        self.variables = variables
        self.limiter = get_slope_limiter(limiter)
        self.riemann_solver = get_riemann_solver(solver)
        self.reconstruction_function = get_reconstruction_method(reconstruction)
        self.ndim = self.grid.dimensions

        if self.ndim == 1:
            self.flux = {
                "x": np.empty(
                    (grid.ncx_total + 1, variables.num_vars_cell), dtype=np.float16
                )
            }
        elif self.ndim == 2:
            self.flux = {
                "x": np.empty(
                    (grid.ncx_total + 1, grid.ncy_total, variables.num_vars_cell),
                    dtype=np.float16,
                ),
                "y": np.empty(
                    (grid.ncx_total, grid.ncy_total + 1, variables.num_vars_cell),
                    dtype=np.float16,
                ),
            }
        elif self.ndim == 3:
            self.flux = {
                "x": np.empty(
                    (
                        grid.ncx_total + 1,
                        grid.ncy_total,
                        grid.ncz_total,
                        variables.num_vars_cell,
                    ),
                    dtype=np.float16,
                ),
                "y": np.empty(
                    (
                        grid.ncx_total,
                        grid.ncy_total + 1,
                        grid.ncz_total,
                        variables.num_vars_cell,
                    ),
                    dtype=np.float16,
                ),
                "z": np.empty(
                    (
                        grid.ncx_total,
                        grid.ncy_total,
                        grid.ncz_total + 1,
                        variables.num_vars_cell,
                    ),
                    dtype=np.float16,
                ),
            }

    def compute_fluxes(self) -> None:
        """Compute fluxes in the x, y, and z directions."""
        if self.ndim == 1:
            self.flux["x"] = self._compute_directional_flux(
                self.variables, direction="x"
            )
        if self.ndim == 2:
            self.flux["x"] = self._compute_directional_flux(
                self.variables, direction="x"
            )
            self.flux["y"] = self._compute_directional_flux(
                self.variables, direction="y"
            )
        if self.ndim == 3:
            self.flux["x"] = self._compute_directional_flux(
                self.variables, direction="x"
            )
            self.flux["y"] = self._compute_directional_flux(
                self.variables, direction="y"
            )
            self.flux["z"] = self._compute_directional_flux(
                self.variables, direction="z"
            )

    def _compute_directional_flux(
        self, variables: Variables, direction: str
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

        # TODO: Either preallocate left_state and right_state, or do an inplace calculation in reconstruction
        #     : Question to answer: Whether the reconstruction function should return two Variable objects or
        #     : Simply two arrays.
        left_state, right_state = self.reconstruction_function(
            variables, direction, self.ndim
        )
        flux = self.riemann_solver(self.flux, left_state, right_state, direction)

        return flux

    @staticmethod
    def compute_physical_flux(
        s: np.ndarray, rho: np.ndarray, velocities: Sequence[np.ndarray]
    ) -> np.ndarray:
        """
        Calculate the physical flux using NumPy's einsum for improved performance.

        Parameters
        ----------
        s : np.ndarray
            The velocity of the flow in the current direction.
        rho : np.ndarray
            The density of the flow.
        velocities : Sequence[np.ndarray]
            Other velocity components.

        Returns
        -------
        np.ndarray
            The physical flux as a 2D NumPy array.
        """
        # Convert velocities to a 2D array: (num_velocities, N, M)
        velocities_array = np.stack(velocities, axis=0)
        mass_flux = s * rho
        momentum_flux = mass_flux * velocities_array
        physical_flux = np.concatenate(
            [mass_flux[np.newaxis, ...], momentum_flux], axis=0
        )
        return physical_flux


def main():
    pass


if __name__ == "__main__":
    main()
