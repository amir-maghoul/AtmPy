import numpy as np
from typing import Callable, Union, Tuple, Sequence, List

from atmpy.flux.utility import create_averaging_kernels
from atmpy.grid.kgrid import Grid
from atmpy.grid.utility import cell_to_node_average
from atmpy.variables.variables import Variables
from atmpy.physics.eos import EOS
from atmpy.data.enums import VariableIndices as VI, PrimitiveVariableIndices as PVI
from atmpy.data.enums import SlopeLimiters, RiemannSolvers, FluxReconstructions
from atmpy.flux._factory import (
    get_slope_limiter,
    get_riemann_solver,
    get_reconstruction_method,
)
import scipy as sp


class Flux:
    def __init__(
        self,
        grid: Grid,
        variables: Variables,
        eos: EOS,
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
        self.eos = eos
        self.limiter = get_slope_limiter(limiter)
        self.riemann_solver = get_riemann_solver(solver)
        self.reconstruction_function = get_reconstruction_method(reconstruction)
        self.ndim = self.grid.dimensions

        if self.ndim == 1:
            self.flux = {
                "x": np.empty((grid.nx + 1, variables.num_vars_cell), dtype=np.float16)
            }
        elif self.ndim == 2:
            self.flux = {
                "x": np.empty(
                    (grid.nx + 1, grid.ny, variables.num_vars_cell),
                    dtype=np.float16,
                ),
                "y": np.empty(
                    (grid.nx, grid.ny + 1, variables.num_vars_cell),
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

        self.compute_averaging_fluxes()

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

    def compute_unphysical_fluxes(self) -> None:
        """
        Compute unphysical fluxes in the x, y, and z directions. The unphysical fluxes are Pu, Pv and Pw in BK19 paper.
        Reminder: P = rho*Theta.

        Returns
        -------
        List[np.ndarray]
        List of unphysical fluxes in the x, y, and z directions e.g. [Pu, Pv, Pw].
        """

        cell_vars = self.variables.cell_vars
        Pu = cell_vars[..., VI.RHOU] * cell_vars[..., VI.RHOY] / cell_vars[..., VI.RHO]
        fluxes = [Pu]  # container for unphysical fluxes Pu, Pv and Pw

        if self.ndim >= 2:
            Pv = (
                cell_vars[..., VI.RHOV]
                * cell_vars[..., VI.RHOY]
                / cell_vars[..., VI.RHO]
            )
            fluxes.append(Pv)
        if self.ndim == 3:
            Pw = (
                cell_vars[..., VI.RHOW]
                * cell_vars[..., VI.RHOY]
                / cell_vars[..., VI.RHO]
            )
            fluxes = [Pu, Pv, Pw]

        return fluxes

    def compute_averaging_fluxes(self, mode: str = "constant") -> None:
        """
        Compute the physical flux in x, y, and z directions. The physical flux in the BK19 algorithm is
        calculated starting with averaging of the Pu variable from cell to nodes and then averaging nodes to get
        the value on the corresponding interface. But in the end, it is a weighted average of the immediate cells.
        This algorithm updates the flux attribute in-place.

        Parameters
        ----------
        mode : str
            mode of boundary handling for the sp.ndimage.convolve.
        """

        unphysical_fluxes = self.compute_unphysical_fluxes()  # [Pu, Pv, ...]
        kernels = create_averaging_kernels(self.ndim)  # [kernel_x, kernel_y, ...]
        directions = ["x", "y", "z"]

        for flux, kernel, direction in zip(unphysical_fluxes, kernels, directions):
            self.flux[direction] = sp.ndimage.convolve(flux, kernel, mode=mode)
            # self.flux[direction] = sp.signal.fftconvolve(flux, kernel, mode="valid")

def main():
    from atmpy.grid.utility import DimensionSpec, create_grid
    from atmpy.physics.eos import ExnerBasedEOS

    dim = [DimensionSpec(1, 0, 2, 2), DimensionSpec(2, 0, 2, 2)]
    grid = create_grid(dim)
    rng = np.random.default_rng()
    arr = np.arange(30)
    rng.shuffle(arr)
    arr = arr.reshape(5, 6)

    variables = Variables(grid, 5, 1)
    variables.cell_vars[..., VI.RHO] = 1
    variables.cell_vars[..., VI.RHOU] = arr
    variables.cell_vars[..., VI.RHOY] = 2
    variables.cell_vars[..., VI.RHOV] = np.ones((5, 6)) * 2
    eos = ExnerBasedEOS()
    flux = Flux(grid, variables, eos)
    print(flux.flux["x"])
    print(flux.flux["y"].shape)
    # print(variables.cell_vars[..., VI.RHOV])
    # print(variables.cell_vars[..., VI.RHOU])
    Pu, Pv = flux.compute_unphysical_fluxes()
    print(Pu)
    print(arr)

    # TODO: The current implementation evaluates the flux on the whole grid including ghost cells.
    #       This is correct. What remains is choosing indices from the flux the make it have one element
    #       more that the number of INNER cells of variables IN THE CORRESPONDING DIRECTION.


if __name__ == "__main__":
    main()
