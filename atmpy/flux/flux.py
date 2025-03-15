import numpy as np
from typing import List, Tuple
from atmpy.flux.utility import create_averaging_kernels
from atmpy.grid.kgrid import Grid
from atmpy.variables.variables import Variables
from atmpy.physics.eos import EOS
from atmpy.infrastructure.enums import (
    VariableIndices as VI,
    PrimitiveVariableIndices as PVI,
)
from atmpy.infrastructure.enums import (
    SlopeLimiters,
    RiemannSolvers,
    FluxReconstructions,
)
from atmpy.infrastructure.factory import (
    get_riemann_solver,
    get_reconstruction_method,
    get_slope_limiter,
)
import scipy as sp
from atmpy.flux.utility import directional_indices, direction_mapping


class Flux:
    """Flux container. The attributes shared with the constructor parameters have docstrings there.

    Attributes
    ----------
    ndim : int
        The dimension of the problem
    flux : dict[str, np.ndarray]
        The full flux container. This contains the main numerical flux of the problem. The keys are the direction of
         the flux. There is a full variable ndarray for each direction in this container.
    iflux : dict[str, np.ndarray]
        The intermediate flux container. This is basically an array of [Pu, Pv, Pw] of the unphysical flux. The main
        flux in each direction would be for example Flux["x"] = Pu*Psi where Psi is the variable container.
    kernels : List[np.ndarray]
        The averaging kernels for the flux calculation in each direction.
    riemann_solver : Callable
        The given riemann solver function. Default is modified HLL.
    reconstruction : Callable
        The given reconstruction function. Default is modiefid MUSCL.
    limiter : Callable
        The given limiter function. Default is van Leer.
    """

    def __init__(
        self,
        grid: Grid,
        variables: Variables,
        eos: EOS,
        dt: float,
        solver: RiemannSolvers = RiemannSolvers.MODIFIED_HLL,
        reconstruction: FluxReconstructions = FluxReconstructions.MODIFIED_MUSCL,
        limiter: SlopeLimiters = SlopeLimiters.VAN_LEER,
    ):
        """
        Parameters
        ----------
        grid : :py:class:`atmpy.grid.kgrid.Grid`
            The computational grid.
        variables : :py:class:`atmpy.variables.variables.Variables`
            The variable container
        eos : :py:class:`atmpy.physics.eos.EOS`
            The equation of state
        dt : float
            The time step of the problem.
        solver : RiemannSolvers (Enum)
            An enum name from the implemented RiemannSolvers Enum,
            e.g. RiemannSolvers.HLL, RiemannSolvers.RUSANOV, etc.
            The attribute created from this enum is a callable function.
        reconstruction : FluxReconstructions (Enum)
            An enum name from the implemented FluxReconstructions Enum,
            e.g. FluxReconstructions.MUSCL. The attribute created from this
            enum is a callable function.
        limiter : SlopeLimiters (Enum)
            An enum name from the implemented SlopeLimiters Enum,
            e.g. SlopeLimiters.MINMOD. The attribute created from this
            enum is a callable function.

        Notes
        -----
        Note that the timestep of the problem should be an attribute of the flux class since the CFL condition
        restricts the physical fineness and the reconstruction schemes like MUSCL depend on the time step and the
        slope of the state.
        """
        self.grid = grid
        self.variables = variables
        self.eos = eos
        self.dt = dt
        self.riemann_solver = get_riemann_solver(solver)
        self.reconstruction = get_reconstruction_method(reconstruction)
        self.limiter = get_slope_limiter(limiter)
        self.ndim = self.grid.ndim
        self.kernels = create_averaging_kernels(self.ndim)  # [kernel_x, kernel_y, ...]

        self.iflux = {"x": None, "y": None, "z": None}

        if self.ndim == 1:
            self.flux = {
                "x": np.zeros(
                    (grid.ncx_total + 1, variables.num_vars_cell), dtype=np.float32
                )
            }
        elif self.ndim == 2:
            self.flux = {
                "x": np.zeros(
                    (grid.ncx_total + 1, grid.ncy_total, variables.num_vars_cell),
                    dtype=np.float32,
                ),
                "y": np.zeros(
                    (grid.ncx_total, grid.ncy_total + 1, variables.num_vars_cell),
                    dtype=np.float32,
                ),
            }

        elif self.ndim == 3:
            self.flux = {
                "x": np.zeros(
                    (
                        grid.ncx_total + 1,
                        grid.ncy_total,
                        grid.ncz_total,
                        variables.num_vars_cell,
                    ),
                    dtype=np.float32,
                ),
                "y": np.zeros(
                    (
                        grid.ncx_total,
                        grid.ncy_total + 1,
                        grid.ncz_total,
                        variables.num_vars_cell,
                    ),
                    dtype=np.float32,
                ),
                "z": np.zeros(
                    (
                        grid.ncx_total,
                        grid.ncy_total,
                        grid.ncz_total + 1,
                        variables.num_vars_cell,
                    ),
                    dtype=np.float32,
                ),
            }

        # Initialize iflux
        self.compute_averaging_fluxes()

    def _compute_unphysical_fluxes(self) -> dict[str, np.ndarray]:
        """
        Compute unphysical fluxes in the x, y, and z directions. The unphysical fluxes are Pu, Pv and Pw in BK19 paper.
        Reminder: P = rho*Theta.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of unphysical fluxes in the x, y, and z directions e.g. [Pu, Pv, Pw].
        """

        cell_vars = self.variables.cell_vars
        Pu = cell_vars[..., VI.RHOU] * cell_vars[..., VI.RHOY] / cell_vars[..., VI.RHO]
        fluxes = {
            "Pu": Pu,
            "Pv": None,
            "Pw": None,
        }  # container for unphysical fluxes Pu, Pv and Pw

        if self.ndim >= 2:
            Pv = (
                cell_vars[..., VI.RHOV]
                * cell_vars[..., VI.RHOY]
                / cell_vars[..., VI.RHO]
            )
            fluxes["Pv"] = Pv
        if self.ndim == 3:
            Pw = (
                cell_vars[..., VI.RHOW]
                * cell_vars[..., VI.RHOY]
                / cell_vars[..., VI.RHO]
            )
            fluxes["Pw"] = Pw

        return fluxes

    def compute_averaging_fluxes(self, mode: str = "valid") -> None:
        """
        Compute the physical flux in x, y, and z directions. The physical flux in the BK19 algorithm is
        calculated starting with averaging of the Pu variable from cell to nodes and then averaging nodes to get
        the value on the corresponding interface. But in the end, it is a weighted average of the immediate cells.
        This algorithm updates the iflux attribute in-place.

        Parameters
        ----------
        mode : str
            mode of boundary handling for the sp.ndimage.convolve.

        Notes
        -----
        For the sake of matching sizes to avoid confusion, the sizes are the same as the sizes of the variables.
        But in reality the first and last entries in the direction of the flux calculation is not needed.
        """

        unphysical_fluxes = self._compute_unphysical_fluxes()  # [Pu, Pv, [Pw]]
        directions = [
            "x",
            "y",
            "z",
        ]  # direction of the flux calculation: x: 0, y: 1 and z: 2

        # Compute the averaging fluxes and place them in the flux container
        for flux, kernel, direction in zip(
            unphysical_fluxes.values(), self.kernels, directions
        ):
            self.iflux[direction] = sp.signal.fftconvolve(flux, kernel, mode=mode)
            _, _, _, inner_index = directional_indices(self.ndim, direction, full=False)
            self.flux[direction][inner_index + (VI.RHOY,)] = self.iflux[direction]

    def apply_reconstruction(
        self, lmbda: float, direction: str
    ) -> Tuple[Variables, Variables]:
        """Calculate the left and right states for the flux calculation by applying a reconstruction scheme to the
            variables and flux.

        Parameters
        ----------
        direction : str
            The direction of the flux calculation.
        lmbda : float
            The ratio of delta_t to delta_x.
        """
        # Initialize variable objects
        lefts = Variables(
            self.grid, self.variables.num_vars_cell, self.variables.num_vars_node
        )
        rights = Variables(
            self.grid, self.variables.num_vars_cell, self.variables.num_vars_node
        )

        # Use reconstruction scheme (MUSCL) to create left and right primitive variables
        lefts.primitives, rights.primitives = self.reconstruction(
            self.variables, self.flux, self.eos, self.limiter, lmbda, direction
        )

        # left and right indices
        lefts_idx, rights_idx, directional_inner_idx, inner_idx = directional_indices(
            self.ndim, direction, full=False
        )

        # Index mapping
        velocity_indices = [PVI.U, PVI.V, PVI.W]
        direction_int = direction_mapping(direction)

        # Find velocity in the direction
        cell_vars = self.variables.cell_vars
        velocity = cell_vars[..., velocity_indices[direction_int]]

        # Compute the unphysical flux Pu
        Pu = velocity * cell_vars[..., VI.RHOY]

        # Calculate the P = rho*Theta by averaging and advecting
        lefts.cell_vars[lefts_idx + (VI.RHOY,)] = rights.cell_vars[
            rights_idx + (VI.RHOY,)
        ] = 0.5 * (
            (cell_vars[lefts_idx + (VI.RHOY,)] + cell_vars[rights_idx + (VI.RHOY,)])
            - lmbda * (Pu[lefts_idx] + Pu[rights_idx])
        )

        # Find the rho conservative variable from the new updated primitive variables
        left_rho = lefts.cell_vars[..., VI.RHO] / lefts.primitives[..., PVI.Y]
        right_rho = rights.cell_vars[..., VI.RHO] / rights.primitives[..., PVI.Y]

        # Compute the conservative variables for left and right states
        lefts.to_conservative(left_rho)
        rights.to_conservative(right_rho)

        return lefts, rights

    def apply_riemann_solver(self, lmbda: float, direction: str) -> None:
        """Apply the Riemann solver to the left and right states to obtain the flux. It updates the flux attribute in-place.

        Parameters
        ----------
        lmbda : float
            The ratio of delta_t to delta_x.
        direction : str
            The direction of the flux calculation.
        """
        lefts, rights = self.apply_reconstruction(lmbda, direction)
        self.riemann_solver(
            lefts.primitives, rights.primitives, self.flux, direction, self.ndim
        )


from line_profiler import profile


@profile
def main():
    from atmpy.grid.utility import DimensionSpec, create_grid
    from atmpy.physics.eos import ExnerBasedEOS

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
    variables.cell_vars[..., VI.RHOU] = array
    variables.cell_vars[..., VI.RHOY] = 2

    rng.shuffle(arr)
    array = arr.reshape(nnx, nny)
    variables.cell_vars[..., VI.RHOV] = array
    eos = ExnerBasedEOS()
    flux = Flux(grid, variables, eos, dt)
    variables.to_primitive(eos)
    primitives = variables.primitives

    from atmpy.flux.reconstruction import (
        calculate_variable_differences,
        calculate_amplitudes,
        calculate_slopes,
        modified_muscl,
    )
    from atmpy.flux.utility import directional_indices, direction_mapping

    direction = "x"
    diffs = calculate_variable_differences(primitives, 2, direction_str=direction)
    slopes = calculate_slopes(diffs, direction, flux.limiter, 2)
    amplitudes = calculate_amplitudes(
        slopes, np.arange(nnx * nny).reshape(nnx, nny), 1, True
    )

    lefts_idx, rights_idx, directional_inner_idx, inner_idx = directional_indices(
        2, direction
    )

    cell_vars = variables.cell_vars
    iflux = flux.iflux

    left, right = modified_muscl(
        variables, flux.flux, eos, flux.limiter, 0.5, direction
    )

    print(flux.flux[direction][..., VI.RHOU])
    print(flux.variables.cell_vars[..., VI.RHOU])
    flux.apply_riemann_solver(1, direction)
    print(flux.flux[direction][..., VI.RHOU])
    print(flux.variables.cell_vars[..., VI.RHOU])

    # TODO: SHAPE MISMATCH IN HLL


if __name__ == "__main__":
    main()
